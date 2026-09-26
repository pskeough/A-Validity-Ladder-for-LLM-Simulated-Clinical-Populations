"""Level 1 rebuild, step d: the random-routing null for the gateway rate, old and corrected.

Audit item B5. The paper says the null "redraws each case's items from its own cell's item marginals
at the same total". Script 22 instead pools elevated cases per (model, profile_id), so both framings
share one pool, draws item marginals from elevated cases only, and conditions on the total by
rejection with at most 60 retries (cases that never hit their total are dropped).

This script reports four versions on corpus v3 (invalid row dropped):
  old_as_coded      script 22's construction re-run on v3 (Monte Carlo, seed 20260726)
  old_pool_exact    script 22's pools (model x profile, both framings, elevated marginals), with the
                    conditional probability computed exactly instead of by rejection
  cell_elev_exact   own cell (model x profile x framing), elevated-case marginals, exact
  cell_all_exact    own cell, marginals from all 30 draws of the cell, exact. This is the null as the
                    paper describes it, and the corrected null.
Exact: for a case with total T in a cell with item pmfs p_1..p_8, the null violation probability is
P(X1 < 2, X2 < 2 | sum = T) under independent items, computed by convolution. The null rate of a
group is the mean over its elevated cases.

It also copies the published values from analysis/gateway_null.csv so the report can cite them from
one file, and applies the paper's pass rule (observed under half of the null) to each version.
Ratio intervals: persona-clustered bootstrap (B = 2000), 95%.

Emits analysis/brm/l1_gateway_null.csv. Seeded.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("l1", os.path.join(os.path.dirname(__file__), "79_l1_lib.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

B = 2000


def pmfs(block):
    """(8, 4) item pmfs from the rows of block."""
    X = block[L.ITEMS].to_numpy()
    return np.stack([np.bincount(X[:, j], minlength=4)[:4] / len(X) for j in range(8)])


def exact_null(p, totals):
    """P(X1<2, X2<2 | sum = T) for each T in totals under independent items with pmfs p."""
    d = np.array([1.0])
    for j in range(2, 8):
        d = np.convolve(d, p[j])
    d = np.concatenate([d, np.zeros(30)])
    out = []
    for T in totals:
        num = den = 0.0
        for a in range(4):
            for b in range(4):
                r = T - a - b
                if 0 <= r <= 18:
                    pr = p[0, a] * p[1, b] * d[r]
                    den += pr
                    if a < 2 and b < 2:
                        num += pr
        out.append(num / den if den > 0 else np.nan)
    return np.array(out)


def old_as_coded(elev, rng):
    """Script 22's null_conditional_marginal, verbatim logic."""
    viol = kept = 0
    violates = lambda a: (a[:, 0] < 2) & (a[:, 1] < 2)
    for _, grp in elev.groupby(["model", "profile_id"]):
        pools = [grp[c].to_numpy() for c in L.ITEMS]
        for _ in range(200 // 20):
            drawn = np.stack([rng.choice(p, len(grp)) for p in pools], axis=1)
            for _ in range(60):
                bad = drawn.sum(axis=1) != grp.total.to_numpy()
                if not bad.any():
                    break
                drawn[bad] = np.stack([rng.choice(p, bad.sum()) for p in pools], axis=1)
            keep = drawn.sum(axis=1) == grp.total.to_numpy()
            if keep.sum():
                viol += int(violates(drawn[keep][:, :2]).sum())
                kept += int(keep.sum())
    return viol / kept


def main():
    rng = np.random.default_rng(20260726)
    m = L.load_corpus()
    m["viol"] = ((m.phq8_1 < 2) & (m.phq8_2 < 2) & (m.total >= 10)).astype(float)
    elev_mask = m.total >= 10

    # per-case exact null probabilities under three constructions
    m["p_old_pool_exact"] = np.nan
    m["p_cell_elev_exact"] = np.nan
    m["p_cell_all_exact"] = np.nan
    for _, g in m[elev_mask].groupby(["model", "profile_id"]):
        m.loc[g.index, "p_old_pool_exact"] = exact_null(pmfs(g), g.total.to_numpy())
    for _, g in m.groupby(["model", "profile_id", "framing"]):
        e = g[g.total >= 10]
        if len(e):
            m.loc[e.index, "p_cell_all_exact"] = exact_null(pmfs(g), e.total.to_numpy())
            m.loc[e.index, "p_cell_elev_exact"] = exact_null(pmfs(e), e.total.to_numpy())

    pub = pd.read_csv(os.path.join(L.BASE, "analysis", "gateway_null.csv"))
    pub["short"] = pub.scope.map(lambda s: L.SHORT.get(s, s))
    rows = []
    boot_rng = np.random.default_rng(20260928)
    for mdl in ["pooled"] + L.MODELS:
        mm = m if mdl == "pooled" else m[m.model == mdl]
        e_all = mm[mm.total >= 10]
        old = old_as_coded(e_all, rng)
        pr = pub[pub.short == L.SHORT.get(mdl, mdl)].iloc[0]
        for fr in ["both", "clinical", "narrative"]:
            e = e_all if fr == "both" else e_all[e_all.framing == fr]
            obs = e.viol.mean()
            r = dict(model=L.SHORT.get(mdl, mdl), framing=fr, n_elevated=len(e), observed=obs,
                     published_observed=pr.observed_pct / 100 if fr == "both" else np.nan,
                     published_null=pr.null_conditional_marginal_pct / 100 if fr == "both" else np.nan,
                     null_old_as_coded=old if fr == "both" else np.nan)
            pers = (e.model + "|" + e.profile_id).to_numpy()
            pu, inv = np.unique(pers, return_inverse=True)
            strata = np.array([x.split("|")[0] for x in pu])
            idx = np.concatenate([np.where(strata == s)[0][boot_rng.integers(0, (strata == s).sum(),
                                                                             size=(B, (strata == s).sum()))]
                                  for s in np.unique(strata)], axis=1)
            vs = np.bincount(inv, weights=e.viol.to_numpy(), minlength=len(pu))
            for col in ["p_old_pool_exact", "p_cell_elev_exact", "p_cell_all_exact"]:
                nul = e[col].mean()
                ps = np.bincount(inv, weights=e[col].to_numpy(), minlength=len(pu))
                rb = vs[idx].sum(axis=1) / ps[idx].sum(axis=1)
                key = col[2:]
                r[f"null_{key}"] = nul
                r[f"ratio_obs_to_{key}"] = obs / nul
                r[f"ratio_{key}_ci95_lo"] = np.quantile(rb, 0.025)
                r[f"ratio_{key}_ci95_hi"] = np.quantile(rb, 0.975)
                r[f"paper_rule_pass_{key}"] = bool(obs < 0.5 * nul)
            if fr == "both":
                r["ratio_obs_to_old_as_coded"] = obs / old
                r["paper_rule_pass_old_as_coded"] = bool(obs < 0.5 * old)
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(L.OUT, "l1_gateway_null.csv"), index=False)
    cols = ["model", "framing", "n_elevated", "observed", "published_null", "null_old_as_coded",
            "null_old_pool_exact", "null_cell_elev_exact", "null_cell_all_exact", "ratio_obs_to_cell_all_exact",
            "ratio_cell_all_exact_ci95_lo", "ratio_cell_all_exact_ci95_hi", "paper_rule_pass_cell_all_exact",
            "paper_rule_pass_old_as_coded"]
    print(res[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
