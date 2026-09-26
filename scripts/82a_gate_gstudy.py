"""Gate as a generalizability study: variance components and D-study on the v3 corpus.

G-study. Outcome is the PHQ-8 total (primary) or the indicator PHQ-8 >= 10 (secondary). Facets:
persona p (120), model m (4), framing f (2), draw r nested in the p x m x f cell (30). Components are
estimated jointly (p x m x f, all random), per model (p x f) and per model and framing (p only).
The draw index is also fitted as a crossed facet (p x i) within each model and framing as a check on
whether draw number carries a main effect; if it does not, absolute and relative error coincide.

Intervals are percentile bootstrap over personas (B = 1000, seeded): a resampled persona carries all
eight of its cells. They cover persona sampling only, not the choice of the four models or two
framings.

D-study (per model and framing). For a k-draw persona mean,
    phi(k)     = s2_p / (s2_p + (s2_i + s2_pi,e) / k)     absolute
    Erho2(k)   = s2_p / (s2_p + s2_pi,e / k)              relative
    SE(k)      = sqrt(s2_e / k)                           s2_e = pooled within-cell variance
k needed for phi >= .80 and .90, and for SE <= 0.5 and 1.0 point (indicator: .05 and .10).
The across-framing coefficients treat framing as a random facet, one framing used:
    phi_F(k)   = s2_p / (s2_p + s2_f + s2_pf + s2_e / k),   Erho2_F(k) = s2_p / (s2_p + s2_pf + s2_e / k)

Also runs a synthetic self-test of the estimators against known components.

Emits (analysis/brm/): gate_selftest.csv, gate_components.csv, gate_dstudy.csv,
gate_dstudy_curve.csv, gate_dstudy_across_framing.csv, gate_iteration_check.csv,
gate_withincell_spread.csv.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "gl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "82_gate_lib.py"))
gl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gl)

B = 1000
SEED = 20260925
KS = [1, 2, 3, 5, 10, 20, 30, 60]
TOL = {"y": (0.5, 1.0), "elev": (0.05, 0.10)}
OUTCOME_LABEL = {"y": "PHQ-8 total", "elev": "PHQ-8 >= 10"}


# ----------------------------------------------------------------------------------------------
def selftest():
    """Simulate a balanced p x m x f design with known components and check recovery."""
    rng = np.random.default_rng(SEED + 1)
    true = dict(p=4.0, m=2.0, f=0.5, pm=1.5, pf=0.3, mf=0.4, pmf=0.8, e=3.0)
    P, M, F, R, REPS = 120, 4, 2, 30, 200
    est = {k: [] for k in true}
    for _ in range(REPS):
        mu = (rng.normal(0, np.sqrt(true["p"]), (P, 1, 1)) + rng.normal(0, np.sqrt(true["m"]), (1, M, 1))
              + rng.normal(0, np.sqrt(true["f"]), (1, 1, F)) + rng.normal(0, np.sqrt(true["pm"]), (P, M, 1))
              + rng.normal(0, np.sqrt(true["pf"]), (P, 1, F)) + rng.normal(0, np.sqrt(true["mf"]), (1, M, F))
              + rng.normal(0, np.sqrt(true["pmf"]), (P, M, F)))
        y = mu[..., None] + rng.normal(0, np.sqrt(true["e"]), (P, M, F, R))
        mean = y.mean(-1)
        ss = ((y - mean[..., None]) ** 2).sum(-1)
        n = np.full((P, M, F), R, float)
        s = gl.three_facet(mean, n, ss)
        for k in true:
            est[k].append(s[k])
    rows = []
    for k, t in true.items():
        e = np.array(est[k])
        rows.append(dict(component=k, true=t, mean_estimate=round(e.mean(), 4),
                         sd_estimate=round(e.std(ddof=1), 4),
                         bias_in_mc_se=round((e.mean() - t) / (e.std(ddof=1) / np.sqrt(REPS)), 2)))
    out = pd.DataFrame(rows)
    # one-facet check on the first model/framing slice of a fresh draw
    return out


# ----------------------------------------------------------------------------------------------
def components_for(arrs, personas_idx):
    """All components for one resample (personas_idx indexes the persona axis)."""
    mean, n, ss = (a[personas_idx] for a in arrs)
    res = {"joint": gl.three_facet(mean, n, ss)}
    for mi, m in enumerate(gl.MODELS):
        res[("model", m)] = gl.two_facet(mean[:, mi, :], n[:, mi, :], ss[:, mi, :])
        for fi, f in enumerate(gl.FRAMES):
            res[("mf", m, f)] = gl.one_facet(mean[:, mi, fi], n[:, mi, fi], ss[:, mi, fi])
    return res


def iteration_check(v, outcome):
    """p x i crossed, one observation per (persona, iteration), per model and framing."""
    rows = []
    for m in gl.MODELS:
        for f in gl.FRAMES:
            d = v[(v.model == m) & (v.framing == f)]
            w = d.pivot_table(index="profile_id", columns="iteration", values=outcome, aggfunc="first")
            w = w.dropna(axis=0)          # drops the one persona with 29 valid draws
            Y = w.to_numpy()
            P, I = Y.shape
            g = Y.mean()
            ms_p = I * ((Y.mean(1) - g) ** 2).sum() / (P - 1)
            ms_i = P * ((Y.mean(0) - g) ** 2).sum() / (I - 1)
            r = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + g
            ms_r = (r ** 2).sum() / ((P - 1) * (I - 1))
            rows.append(dict(outcome=OUTCOME_LABEL[outcome], model=gl.SHORT[m], framing=f,
                             n_persona=P, n_iter=I, s2_p=(ms_p - ms_r) / I, s2_iter=(ms_i - ms_r) / P,
                             s2_pi_e=ms_r, F_iter=ms_i / ms_r))
    out = pd.DataFrame(rows)
    from scipy import stats
    out["p_iter"] = [stats.f.sf(r.F_iter, r.n_iter - 1, (r.n_persona - 1) * (r.n_iter - 1))
                     for r in out.itertuples()]
    return out


def q(x, lo=2.5, hi=97.5, method="linear"):
    x = np.asarray(x, float)
    return np.percentile(x, lo, method=method), np.percentile(x, hi, method=method)


def main(december_only=False):
    """december_only=True is the sensitivity run of 82e: outputs carry the suffix _december."""
    os.makedirs(gl.OUTD, exist_ok=True)
    sfx = "_december" if december_only else ""

    if not december_only:
        st = selftest()
        st.to_csv(os.path.join(gl.OUTD, "gate_selftest.csv"), index=False)
        print("self-test (true vs mean estimate over 200 simulated designs):")
        print(st.to_string(index=False))
        assert (st.bias_in_mc_se.abs() < 4).all(), "estimator self-test failed"

    v = gl.load_corpus(december_only=december_only)
    personas = sorted(v.profile_id.unique())
    assert len(personas) == 120

    comp_rows, d_rows, curve_rows, af_rows, spread_rows, it_all = [], [], [], [], [], []
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, 120, 120) for _ in range(B)]

    for outcome in ("y", "elev"):
        lab = OUTCOME_LABEL[outcome]
        s = gl.cell_summaries(v, outcome)
        arrs = gl.to_arrays(s, personas, gl.MODELS, gl.FRAMES)
        point = components_for(arrs, np.arange(120))
        boots = [components_for(arrs, ix) for ix in boot_idx]

        it = iteration_check(v, outcome)
        it_all.append(it)

        # ---- components table
        def add(scope, model, framing, key, names):
            tot = sum(gl.pos(point[key][c]) for c in names)
            for c in names:
                bs = [b[key][c] for b in boots]
                lo, hi = q(bs)
                comp_rows.append(dict(outcome=lab, scope=scope, model=model, framing=framing,
                                      component=c, estimate_raw=point[key][c],
                                      estimate=gl.pos(point[key][c]),
                                      share_pct=100 * gl.pos(point[key][c]) / tot,
                                      boot_lo=lo, boot_hi=hi,
                                      n_persona=point[key].get("n_persona", point[key].get("n_cells"))))
        add("joint", "all", "both", "joint", ["p", "m", "f", "pm", "pf", "mf", "pmf", "e"])
        for m in gl.MODELS:
            add("model", gl.SHORT[m], "both", ("model", m), ["p", "f", "pf", "e"])
            for f in gl.FRAMES:
                add("model_x_framing", gl.SHORT[m], f, ("mf", m, f), ["p", "e"])

        # ---- D-study per model and framing
        t1, t2 = TOL[outcome]
        for m in gl.MODELS:
            for f in gl.FRAMES:
                key = ("mf", m, f)
                c = point[key]
                itr = it[(it.model == gl.SHORT[m]) & (it.framing == f)].iloc[0]
                s2i, s2pie = gl.pos(itr.s2_iter), gl.pos(itr.s2_pi_e)
                s2p_x = itr.s2_p
                bp = np.array([b[key]["p"] for b in boots])
                be = np.array([b[key]["e"] for b in boots])
                rec = dict(outcome=lab, model=gl.SHORT[m], framing=f, n_persona=c["n_cells"],
                           grand_mean=c["grand"], s2_p=c["p"], s2_p_lo=q(bp)[0], s2_p_hi=q(bp)[1],
                           s2_e=c["e"], s2_e_lo=q(be)[0], s2_e_hi=q(be)[1],
                           within_sd=np.sqrt(c["e"]), s2_iter=itr.s2_iter, s2_pi_e=itr.s2_pi_e)
                for k in (1, 30):
                    rec[f"phi_{k}"] = gl.phi_k(s2p_x, s2i + s2pie, k)
                    rec[f"erho2_{k}"] = gl.phi_k(s2p_x, s2pie, k)
                    rec[f"phi_nested_{k}"] = gl.phi_k(c["p"], c["e"], k)
                    bphi = [gl.phi_k(a, e_, k) for a, e_ in zip(bp, be)]
                    rec[f"phi_nested_{k}_lo"], rec[f"phi_nested_{k}_hi"] = q(bphi)
                for tgt in (0.80, 0.90):
                    tag = int(tgt * 100)
                    rec[f"k_phi{tag}"] = gl.k_for_phi(s2p_x, s2i + s2pie, tgt)
                    rec[f"k_phi{tag}_nested"] = gl.k_for_phi(c["p"], c["e"], tgt)
                    bk = [gl.k_for_phi(a, e_, tgt) for a, e_ in zip(bp, be)]
                    rec[f"k_phi{tag}_lo"], rec[f"k_phi{tag}_hi"] = q(bk, method="inverted_cdf")
                cellvar = s[(s.model == m) & (s.framing == f)]["var"].to_numpy()
                for tol, nm in ((t1, "tol1"), (t2, "tol2")):
                    rec[f"{nm}"] = tol
                    rec[f"k_se_{nm}"] = gl.k_for_se(c["e"], tol)
                    bk = [gl.k_for_se(e_, tol) for e_ in be]
                    rec[f"k_se_{nm}_lo"], rec[f"k_se_{nm}_hi"] = q(bk, method="inverted_cdf")
                    rec[f"k_se_{nm}_q90cell"] = gl.k_for_se(np.quantile(cellvar, 0.9), tol)
                    rec[f"k_se_{nm}_maxcell"] = gl.k_for_se(cellvar.max(), tol)
                rec["se_at_30"] = np.sqrt(gl.pos(c["e"]) / 30)
                d_rows.append(rec)
                for k in range(1, 61):
                    curve_rows.append(dict(outcome=lab, model=gl.SHORT[m], framing=f, k=k,
                                           phi=gl.phi_k(s2p_x, s2i + s2pie, k),
                                           erho2=gl.phi_k(s2p_x, s2pie, k),
                                           se=np.sqrt(gl.pos(c["e"]) / k)))
                spread_rows.append(dict(outcome=lab, model=gl.SHORT[m], framing=f,
                                        cell_sd_min=np.sqrt(cellvar.min()),
                                        cell_sd_median=np.sqrt(np.median(cellvar)),
                                        cell_sd_q90=np.sqrt(np.quantile(cellvar, 0.9)),
                                        cell_sd_max=np.sqrt(cellvar.max()),
                                        pooled_sd=np.sqrt(c["e"]),
                                        pct_cells_one_valued=100 * np.mean(cellvar == 0),
                                        sd_of_cell_means=np.std(s[(s.model == m) & (s.framing == f)]["mean"], ddof=1)))

        # ---- across-framing coefficients per model
        for m in gl.MODELS:
            key = ("model", m)
            c = point[key]
            sp, sf, spf, se = (gl.pos(c[x]) for x in ("p", "f", "pf", "e"))
            rec = dict(outcome=lab, model=gl.SHORT[m], s2_p=c["p"], s2_f=c["f"], s2_pf=c["pf"], s2_e=c["e"])
            for k in (1, 30, 60):
                rec[f"phiF_{k}"] = sp / (sp + sf + spf + se / k) if sp > 0 else 0.0
                rec[f"erho2F_{k}"] = sp / (sp + spf + se / k) if sp > 0 else 0.0
            rec["phiF_ceiling"] = sp / (sp + sf + spf) if sp > 0 else 0.0
            rec["erho2F_ceiling"] = sp / (sp + spf) if sp > 0 else 0.0
            bps = []
            for b in boots:
                bb = b[key]
                a1, a2, a3 = gl.pos(bb["p"]), gl.pos(bb["f"]), gl.pos(bb["pf"])
                bps.append(a1 / (a1 + a2 + a3) if a1 > 0 else 0.0)
            rec["phiF_ceiling_lo"], rec["phiF_ceiling_hi"] = q(bps)
            for tgt in (0.80, 0.90):
                ceil_ = rec["phiF_ceiling"]
                if sp == 0 or ceil_ <= tgt:
                    kk = np.inf
                else:
                    kk = float(max(1, int(np.ceil(se / (sp / tgt - sp - sf - spf)))))
                rec[f"k_phiF{int(tgt * 100)}"] = kk
            af_rows.append(rec)

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(os.path.join(gl.OUTD, f"gate_components{sfx}.csv"), index=False, float_format="%.5g")
    ds = pd.DataFrame(d_rows)
    ds.to_csv(os.path.join(gl.OUTD, f"gate_dstudy{sfx}.csv"), index=False, float_format="%.5g")
    pd.DataFrame(curve_rows).to_csv(os.path.join(gl.OUTD, f"gate_dstudy_curve{sfx}.csv"), index=False,
                                    float_format="%.5g")
    af = pd.DataFrame(af_rows)
    af.to_csv(os.path.join(gl.OUTD, f"gate_dstudy_across_framing{sfx}.csv"), index=False, float_format="%.5g")
    it = pd.concat(it_all)
    it.to_csv(os.path.join(gl.OUTD, f"gate_iteration_check{sfx}.csv"), index=False, float_format="%.5g")
    sp_ = pd.DataFrame(spread_rows)
    sp_.to_csv(os.path.join(gl.OUTD, f"gate_withincell_spread{sfx}.csv"), index=False, float_format="%.5g")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\nCOMPONENTS")
    print(comp.round(4).to_string(index=False))
    print("\nD-STUDY")
    cols = ["outcome", "model", "framing", "grand_mean", "s2_p", "s2_e", "s2_iter", "phi_1", "erho2_1",
            "phi_30", "k_phi80", "k_phi80_lo", "k_phi80_hi", "k_phi90", "k_phi90_lo", "k_phi90_hi",
            "k_se_tol1", "k_se_tol2", "k_se_tol1_q90cell", "se_at_30"]
    print(ds[cols].round(3).to_string(index=False))
    print("\nACROSS FRAMING")
    print(af.round(3).to_string(index=False))
    print("\nITERATION CHECK")
    print(it.round(4).to_string(index=False))
    print("\nWITHIN-CELL SPREAD")
    print(sp_.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
