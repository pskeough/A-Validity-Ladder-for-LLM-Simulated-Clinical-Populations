"""Validity ladder (frozen rules, paper_brm/LADDER_SPEC.md v1.0) on Bisbee et al. (2024),
"Synthetic Replacements for Human Survey Data? The Perils of Large Language Models", Political
Analysis. Replication archive Harvard Dataverse doi:10.7910/DVN/VPN481 (CC0), parsed by
bisbee_parse.py.

Design. Each persona is one real ANES 2016 or 2020 respondent (7,530 in the authors' anes_simp.csv),
described by ten attributes. ChatGPT (gpt-3.5-turbo, June 2023, temperature 1) gave a 0-100 feeling
thermometer toward 11 groups, about 30 draws per persona, under three prompts: "full"
(demographics and politics), "demogs" (demographics only) and "pol" (politics only). Every persona
therefore has a human twin: the same respondent's own thermometer. The mapping follows the
paper's other same-respondent demonstration (Argyle et al., argyle_ladder.py and 78e):
reference = the same respondents, unweighted; inference by respondent bootstrap that keeps each
persona with its human twin.

Rungs (per model and framing; framing = prompt):
  Gate     91_ladder_core.gate on the draws of each thermometer; reference SD = SD of the same
           respondents' thermometer; k = 30; B = 1,000 persona bootstrap for k intervals.
  Level 1  not applicable: each thermometer is a single item, and person fit needs a scale.
  Level 2  seven contrasts per thermometer (female-male, Black-white, Hispanic-white, high school-
           bachelor's or more, 18-29 vs 65+, Republican-Democrat, $30,000 vs more than $150,000);
           persona-mean gap g against the same respondents' gap p; 78e.r3cov with df infinite;
           family = one framing, Bonferroni over its 77 contrasts.
  Level 3  residual of persona means against the same respondents' answers, per group (all and
           the 13 contrast groups); paired SE; TOST at 90% against 0.5 reference SD (default) and
           0.2 SD.
  Level 4  not applicable: the 11 thermometers are separate targets, not items of one instrument.

Receipts against the authors' logs:
  2_DATA_detailed_prompt_june_prep_LOG.txt: 248,490 respondent x prompt x group cells, minimum
    10 rows, and the ten least-populated cells it prints;
  SI_Section8_LOG.txt: feols(|ANES - LLM| ~ year * group), full prompt, 2,235,705 observations;
    the saturated model's coefficients are differences of cell means and are recomputed here.

Outputs: ../results/bisbee/{gate,level2_contrasts,level2_model,level3_groups,level3_outcome,
receipts}.csv
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UPD = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RAW = os.path.join(HERE, "..", "raw", "bisbee2024")
OUT = os.path.join(HERE, "..", "results", "bisbee")
os.makedirs(OUT, exist_ok=True)
B_GATE, B_L2 = 1000, 2000
K = 30
Z90 = 1.6448536269514722
GROUPS = ["Democratic Party", "Republican Party", "Black Americans", "White Americans", "Asian Americans",
          "Gays and Lesbians", "Muslims", "Jews", "Liberals", "Conservatives", "Christians"]
CONTRASTS = [("gender", "female", "male"),
             ("raceth", "non-Hispanic black", "non-Hispanic white"),
             ("raceth", "Hispanic", "non-Hispanic white"),
             ("education", "a high school diploma", "a bachelor's degree or more"),
             ("agegrp", "18-29", "65+"),
             ("PID", "Republican", "Democrat"),
             ("income", "$30,000", "more than $150,000")]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(m)
    return m


CORE = load_module("ladder_core", os.path.join(UPD, "scripts", "91_ladder_core.py"))
E78 = load_module("l2_external", os.path.join(UPD, "scripts", "78e_l2_external_r3.py"))


def human():
    h = pd.read_csv(os.path.join(RAW, "PA_replication", "data", "raw", "anes_simp.csv"))
    h["respID"] = h.respID.astype(str)
    h["agegrp"] = np.select([h.age.between(18, 29), h.age >= 65], ["18-29", "65+"], None)
    return h


def receipts(long, h, rec):
    """Authors' printed numbers, recomputed from the parsed draws."""
    pr = pd.read_csv(os.path.join(OUT, "parse_receipts_rr1.csv")).set_index("quantity").value
    rows = [dict(check="cells (respondent x prompt x group)", got=pr["cells"], want="248490"),
            dict(check="minimum rows in a cell", got=pr["cell_min_draws"], want="10")]
    log = os.path.join(RAW, "PA_replication", "code", "LOG", "2_DATA_detailed_prompt_june_prep_LOG.txt")
    named = []
    with open(log, encoding="utf-8") as f:
        for line in f.read().splitlines()[808:818]:
            p = line.split()
            named.append((p[1], p[2], " ".join(p[3:-1]), int(p[-1])))
    cnt = pd.read_parquet(os.path.join(RAW, "derived", "rr1_cellcounts.parquet")).set_index(
        ["respID", "framing", "group"]).n
    for rid, fr, grp, n in named:
        rows.append(dict(check=f"rows in cell {rid} {fr} {grp}", got=str(int(cnt.loc[(rid, fr, grp)])), want=str(n)))
    # SI Section 8: MAE by year x group, full prompt
    hl = h.melt(id_vars=["respID", "year"], value_vars=[f"therm_{g}" for g in GROUPS],
                var_name="group", value_name="y")
    hl["group"] = hl.group.str.replace("therm_", "", regex=False)
    d = long[long.framing == "full"].merge(hl, on=["respID", "group"])
    d = d[d.y.notna()]
    d["ae"] = (d.y - d.therm).abs()
    rows.append(dict(check="SI Section 8 observations", got=str(len(d)), want="2235705"))
    m = d.groupby(["year", "group"]).ae.mean()
    want = {"(Intercept)": 16.616461, "year2020": 0.026391, "groupDemocratic Party": 1.412408,
            "groupGays and Lesbians": 3.652706, "groupMuslims": 2.371490, "groupRepublican Party": 1.652356,
            "year2020:groupChristians": 3.484029, "year2020:groupGays and Lesbians": -0.979443}
    base, y20 = m.loc[(2016, "Asian Americans")], m.loc[(2020, "Asian Americans")] - m.loc[(2016, "Asian Americans")]
    for name, w in want.items():
        if name == "(Intercept)":
            got = base
        elif name == "year2020":
            got = y20
        elif name.startswith("year2020:group"):
            g = name.split("group", 1)[1]
            got = m.loc[(2020, g)] - m.loc[(2016, g)] - y20
        else:
            g = name.split("group", 1)[1]
            got = m.loc[(2016, g)] - base
        rows.append(dict(check=f"SI Section 8 coefficient {name}", got=f"{got:.6f}", want=f"{w:.6f}"))
    r = pd.DataFrame(rows)
    r["match"] = r.got.astype(str) == r.want.astype(str)
    rec.extend(r.to_dict("records"))
    return r


def persona_stats(long):
    """Per (framing, group, respID): n, mean, within SS of the draws."""
    g = long.groupby(["framing", "group", "respID"]).therm
    s = pd.DataFrame(dict(n=g.size(), mean=g.mean(), var=g.var(ddof=1))).reset_index()
    s["ss"] = s["var"].fillna(0) * (s.n - 1)
    return s


def gate_rows(ps, h, rng, model):
    rows = []
    for (fr, grp), s in ps.groupby(["framing", "group"]):
        y = h.set_index("respID")[f"therm_{grp}"]
        sd_ref = float(y.std(ddof=1))
        mean, n, ss = s["mean"].values, s.n.values.astype(float), s.ss.values
        c = CORE.GL.one_facet(mean, n, ss)
        t_min, t_rec = CORE.gate_tolerances(sd_ref)
        se_k = float(np.sqrt(CORE.GL.pos(c["e"]) / K))
        bk = []
        for _ in range(B_GATE):
            i = rng.integers(0, len(n), len(n))
            cb = CORE.GL.one_facet(mean[i], n[i], ss[i])
            bk.append((CORE.GL.k_for_se(cb["e"], t_min), CORE.GL.k_for_se(cb["e"], t_rec),
                       CORE.GL.phi_k(cb["p"], cb["e"], K)))
        bk = np.array(bk, float)
        rows.append(dict(model=model, framing=fr, outcome=grp, n_personas=c["n_cells"], n_draws=c["n_draws"],
                         draws_per_persona_median=float(np.median(n)), sd_ref=sd_ref, s2_p=c["p"], s2_r=c["e"],
                         within_sd=float(np.sqrt(CORE.GL.pos(c["e"]))), tol_min=t_min, tol_rec=t_rec,
                         k=K, se_k=se_k, phi_k=CORE.GL.phi_k(c["p"], c["e"], K),
                         phi_k_lo=np.quantile(bk[:, 2], .025), phi_k_hi=np.quantile(bk[:, 2], .975),
                         k_min=CORE.GL.k_for_se(c["e"], t_min), k_min_lo=np.quantile(bk[:, 0], .025),
                         k_min_hi=np.quantile(bk[:, 0], .975),
                         k_rec=CORE.GL.k_for_se(c["e"], t_rec), k_rec_lo=np.quantile(bk[:, 1], .025),
                         k_rec_hi=np.quantile(bk[:, 1], .975),
                         pass_min=se_k <= t_min, pass_rec=se_k <= t_rec,
                         pass_single=np.sqrt(CORE.GL.pos(c["e"])) <= t_min,
                         k_sep80=CORE.GL.k_for_phi(c["p"], c["e"], .80), k_sep90=CORE.GL.k_for_phi(c["p"], c["e"], .90)))
    return pd.DataFrame(rows)


def level2_rows(ps, h, rng, model):
    hs = h.set_index("respID")
    rows = []
    for fr, sf in ps.groupby("framing"):
        for grp in GROUPS:
            sm = sf[sf.group == grp].set_index("respID")["mean"]
            y = hs[f"therm_{grp}"]
            for attr, a, b in CONTRASTS:
                d = pd.DataFrame(dict(s=sm, y=y, attr=hs[attr])).dropna(subset=["s", "y"])
                A, Bb = d[d.attr == a], d[d.attr == b]
                g = A.s.mean() - Bb.s.mean()
                p = A.y.mean() - Bb.y.mean()
                ia = rng.integers(0, len(A), (B_L2, len(A)))
                ib = rng.integers(0, len(Bb), (B_L2, len(Bb)))
                bg = A.s.values[ia].mean(1) - Bb.s.values[ib].mean(1)
                bp = A.y.values[ia].mean(1) - Bb.y.values[ib].mean(1)
                rows.append(dict(model=model, framing=fr, outcome=grp, contrast=f"{a} - {b}", n_a=len(A),
                                 n_b=len(Bb), g=g, se_g=bg.std(ddof=1), p=p, se_p=bp.std(ddof=1),
                                 cov=np.cov(bg, bp, ddof=1)[0, 1]))
    d = pd.DataFrame(rows)
    d["m_family"] = d.groupby(["model", "framing"]).g.transform("size")
    res = [E78.r3cov(r.g, r.se_g, r.p, r.se_p, r.cov, np.inf, E78.ALPHA / r.m_family, E78.BANDS["asym"])
           for r in d.itertuples()]
    return pd.concat([d, pd.DataFrame(res)], axis=1)


def l2_model(l2):
    rows = []
    for (m, fr), g in l2.groupby(["model", "framing"]):
        stopped = g.verdict.isin(["no population gap", "reference too imprecise"])
        live = g[~stopped].verdict
        excl = live.map(lambda v: "kept" not in v.split(" or ") and v != "undetermined")
        verdict = "pass" if (live == "kept").all() and len(live) else ("fail" if excl.any() else "unresolved")
        rows.append(dict(model=m, framing=fr, contrasts=len(g), stopped=int(stopped.sum()),
                         kept=int((live == "kept").sum()), excludes_kept=int(excl.sum()), verdict=verdict,
                         **{f"n_{k}": int(v) for k, v in g.verdict.value_counts().items()}))
    return pd.DataFrame(rows)


def level3_rows(ps, h, model):
    hs = h.set_index("respID")
    groups = [(None, None)] + sorted({(a, v) for a, x, y in CONTRASTS for v in (x, y)})
    rows = []
    for fr, sf in ps.groupby("framing"):
        for grp in GROUPS:
            sm = sf[sf.group == grp].set_index("respID")["mean"]
            y = hs[f"therm_{grp}"]
            sd_ref = float(y.std(ddof=1))
            attrs = sorted({a for a, _ in groups[1:]})
            d = pd.DataFrame(dict(s=sm, y=y)).join(hs[attrs]).dropna(subset=["s", "y"])
            for attr, val in groups:
                sub = d if attr is None else d[d[attr] == val]
                res = sub.s - sub.y
                r, se = res.mean(), res.std(ddof=1) / np.sqrt(len(sub))
                lo, hi = r - Z90 * se, r + Z90 * se
                rows.append(dict(model=model, framing=fr, outcome=grp, group="all" if attr is None else f"{attr}:{val}",
                                 n=len(sub), human_mean=sub.y.mean(), sim_mean=sub.s.mean(), residual=r,
                                 ci90_lo=lo, ci90_hi=hi, sd_ref=sd_ref,
                                 pass_05sd=(lo > -.5 * sd_ref) and (hi < .5 * sd_ref),
                                 pass_02sd=(lo > -.2 * sd_ref) and (hi < .2 * sd_ref),
                                 fail_05sd=(lo > .5 * sd_ref) or (hi < -.5 * sd_ref)))
    return pd.DataFrame(rows)


def main(model="rr1", with_fix2=False):
    rng = np.random.default_rng(20261002)
    long = pd.read_parquet(os.path.join(RAW, "derived", f"{model}_long.parquet"))
    if not with_fix2:
        # The authors' analysis file drops the rows their regex repair recovered (fix2): their
        # fixed2 table has no respID, so the left_join on (uniqueID, respID) returns NA for them.
        # The primary run scores the data as the authors analysed them; "_withfix2" keeps them.
        long = long[long.fix != 2]
    else:
        model = model + "_withfix2"
    h = human()
    rec = []
    if model == "rr1":
        r = receipts(long, h, rec)
        print(r.to_string(index=False))
        if not r.match.all():
            pd.DataFrame(rec).to_csv(os.path.join(OUT, f"receipts_{model}.csv"), index=False)
            sys.exit("receipt mismatch")
    ps = persona_stats(long)
    gt = gate_rows(ps, h, rng, model)
    l2 = level2_rows(ps, h, rng, model)
    l2m = l2_model(l2)
    l3 = level3_rows(ps, h, model)
    l3o = (l3.groupby(["model", "framing", "outcome"])
           .agg(groups=("group", "size"), pass_05sd=("pass_05sd", "all"), pass_02sd=("pass_02sd", "all"),
                groups_failing_05sd=("fail_05sd", "sum"), worst_abs_residual=("residual", lambda v: v.abs().max()))
           .reset_index())
    for name, df in [("gate", gt), ("level2_contrasts", l2), ("level2_model", l2m), ("level3_groups", l3),
                     ("level3_outcome", l3o), ("receipts", pd.DataFrame(rec))]:
        df.to_csv(os.path.join(OUT, f"{name}_{model}.csv"), index=False)
    pd.set_option("display.width", 220)
    print(gt[["framing", "outcome", "sd_ref", "within_sd", "se_k", "tol_min", "k_min", "k_rec", "phi_k",
              "pass_min", "pass_rec", "pass_single"]].round(3).to_string(index=False))
    print(l2m.to_string(index=False))
    print(l2.groupby("framing").verdict.value_counts().unstack(0).fillna(0).astype(int).to_string())
    print(l3o.groupby("framing")[["pass_05sd", "pass_02sd"]].sum().to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rr1", with_fix2="--with-fix2" in sys.argv)
