"""Positive control and planted failures: summary of the 83b replicates.

For each rung, dose and scope: the share of replicates giving each reading, with a Wilson 95%
interval for the rate that matters (pass for the positive control, detection for a planted
failure). Readings:
  L1  pass | fail: excess misfit | fail: compressed | other fail | undetermined. For "shuffle" the
      expected reading is any fail naming excess misfit; for "typical", any fail naming a deficit.
  L2  correct (the true region alone) | compatible (a two-region verdict containing it) | wrong (a
      determinate verdict excluding it, alone or in a pair) | undetermined | stopped (reference too
      imprecise, no population gap). Coverage: the share of Fieller intervals containing g, the
      true ratio (nominal 98.6%).
  L3  share passing every group at 0.2 SD, 1 and 2 points; null bias and 90% coverage per group.
  L4  share holding R1, R2, R3 and passing (all three, both framings).

Emits analysis/brm/83c_l1.csv, 83c_l2.csv, 83c_l2_coverage.csv, 83c_l3.csv, 83c_l3_null.csv,
83c_l4.csv, and prints the headline tables.
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "analysis", "brm")
L2_LABELS = ["reversed", "missing", "attenuated", "kept", "steepened"]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def rate(x, name):
    k, n = int(x.sum()), int(len(x))
    lo, hi = wilson(k, n)
    return {name: k / n if n else np.nan, f"{name}_lo95": lo, f"{name}_hi95": hi}


def l1_class(v):
    if v == "pass":
        return "pass"
    if v == "undetermined":
        return "undetermined"
    if "compressed" in v:
        return "fail: compressed"
    if "excess misfit" in v:
        return "fail: excess misfit"
    return "fail: other (" + v.replace("fail: ", "") + ")"


def l2_class(v, true):
    if v in ("reference too imprecise", "no population gap"):
        return "stopped"
    if v == "undetermined":
        return "undetermined"
    names = v.split(" or ")
    if names == [true]:
        return "correct"
    if true in names:
        return "compatible"
    return "wrong"


def main():
    files = sorted(glob.glob(os.path.join(OUT, "83b_reps", "rep_*.csv")))
    df = pd.concat([pd.read_csv(f, low_memory=False, keep_default_na=False, na_values=[""]) for f in files],
                   ignore_index=True)          # "null" is a condition label, not a missing value
    tf = {"True": True, "False": False, True: True, False: False}
    for c in ["passed", "R1", "R2", "R3", "stop_conditional", "no_population_gap", "pass_0.2SD", "pass_1pt",
              "pass_2pt"]:
        df[c] = df[c].map(tf)                  # blanks -> NaN; never astype(bool) on the strings
    for c in ["dose", "ratio", "ci_lo", "ci_hi", "resid", "se", "df", "misfit_ratio", "overfit_ratio"]:
        df[c] = pd.to_numeric(df[c])
    df["model"] = df.model.astype(str)
    R = df.rep.nunique()
    print(f"{R} replicates")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    # --------------------------------------------------------------------------------------- L1
    l1 = df[df.rung == "L1"].copy()
    l1["cls"] = l1.verdict.map(l1_class)
    rows = []
    for (cond, dose, unit), g in l1.groupby(["condition", "dose", "unit"]):
        exp = {"null": g.cls == "pass", "L1 shuffle": g.verdict.str.contains("excess misfit"),
               "L1 typical": g.verdict.str.contains("deficit")}[cond]
        r = dict(condition=cond, dose=dose, unit=unit, n=len(g), **rate(exp, "expected_reading"),
                 misfit_ratio_mean=g.misfit_ratio.mean(), misfit_ratio_p05=g.misfit_ratio.quantile(.05),
                 misfit_ratio_p95=g.misfit_ratio.quantile(.95), overfit_ratio_mean=g.overfit_ratio.mean(),
                 overfit_ratio_p05=g.overfit_ratio.quantile(.05), overfit_ratio_p95=g.overfit_ratio.quantile(.95))
        r.update({f"share_{c}": float((g.cls == c).mean()) for c in sorted(l1.cls.unique())})
        rows.append(r)
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(OUT, "83c_l1.csv"), index=False)
    print("\nL1 (null rows pool all four pseudo-models; planted arms are model 0)")
    print(t1[t1.unit == "both"][["condition", "dose", "n", "expected_reading", "expected_reading_lo95",
                                 "expected_reading_hi95", "misfit_ratio_mean", "overfit_ratio_mean"] +
                                [c for c in t1.columns if c.startswith("share_")]].round(3).to_string(index=False))

    # --------------------------------------------------------------------------------------- L2
    l2 = df[df.rung == "L2"].copy()
    l2["scope"] = np.where(l2.model == "pooled", "pooled", "per model")
    rows, cov = [], []
    for vcol in ("verdict", "verdict_auditprec"):
        l2["cls"] = [l2_class(v, t) for v, t in zip(l2[vcol], l2.true_region)]
        for (scope, unit, dose), g in l2.groupby(["scope", "unit", "dose"]):
            r = dict(reference=("half sample" if vcol == "verdict" else "audit precision (approx.)"),
                     scope=scope, contrast=unit, dose=dose, true_region=g.true_region.iloc[0], n=len(g))
            r.update(rate(g.cls == "correct", "correct"))
            r.update(rate(g.cls.isin(["correct", "compatible"]), "correct_or_compatible"))
            r.update(rate(g.cls == "wrong", "wrong"))
            r.update({f"share_{c}": float((g.cls == c).mean()) for c in
                      ["undetermined", "stopped"]})
            r["share_no_population_gap"] = float((g[vcol] == "no population gap").mean())
            rows.append(r)
    for (scope, unit, dose), g in l2.groupby(["scope", "unit", "dose"]):
        ok = g.ci_lo.notna()
        inside = (g.ci_lo <= dose) & (dose <= g.ci_hi)
        cov.append(dict(scope=scope, contrast=unit, dose=dose, n=len(g), n_interval=int(ok.sum()),
                        coverage=float(inside[ok].mean()) if ok.any() else np.nan,
                        ratio_mean=g.ratio.mean(), ratio_sd=g.ratio.std(),
                        ref_se_mean=g.ref_se.mean(), sim_se_mean=g.sim_se.mean()))
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(OUT, "83c_l2.csv"), index=False)
    pd.DataFrame(cov).to_csv(os.path.join(OUT, "83c_l2_coverage.csv"), index=False)
    for ref in ("half sample", "audit precision (approx.)"):
        print(f"\nL2, reference {ref}")
        x = t2[t2.reference == ref]
        print(x[["scope", "contrast", "dose", "true_region", "n", "correct", "correct_or_compatible", "wrong",
                 "share_undetermined", "share_stopped"]].round(3).to_string(index=False))
    print("\nL2 interval coverage of the true ratio (nominal .986)")
    print(pd.DataFrame(cov).round(3).to_string(index=False))

    # --------------------------------------------------------------------------------------- L3
    l3 = df[df.rung == "L3"].copy()
    l3["scope"] = np.where(l3.model == "pooled", "pooled", "per model")
    s = l3[l3.condition == "L3 shift"]
    rows = []
    for (scope, dose), g in s.groupby(["scope", "dose"]):
        r = dict(scope=scope, dose=dose, n=len(g), cell_min=g.cell_min.min(), cell_max=g.cell_max.max())
        for t in ("0.2SD", "1pt", "2pt"):
            r.update(rate(g[f"pass_{t}"].astype(bool), f"pass_{t}"))
        rows.append(r)
    t3 = pd.DataFrame(rows)
    t3.to_csv(os.path.join(OUT, "83c_l3.csv"), index=False)
    nul = l3[l3.condition == "null"].copy()
    from scipy import stats
    nul["cover90"] = np.abs(nul.resid) <= stats.t.ppf(0.95, nul.df) * nul.se
    t3n = nul.groupby(["scope", "unit"]).agg(n=("resid", "size"), bias=("resid", "mean"),
                                             sd_resid=("resid", "std"), mean_se=("se", "mean"),
                                             coverage90=("cover90", "mean")).reset_index()
    t3n.to_csv(os.path.join(OUT, "83c_l3_null.csv"), index=False)
    print("\nL3 pass rates (all ten groups)")
    print(t3.round(3).to_string(index=False))
    print("\nL3 null: bias, SD of residual vs mean SE, 90% coverage")
    print(t3n.round(3).to_string(index=False))

    # --------------------------------------------------------------------------------------- L4
    l4 = df[df.rung == "L4"].copy()
    rows = []
    for (cond, dose, unit), g in l4.groupby(["condition", "dose", "unit"]):
        r = dict(condition=cond, dose=dose, unit=unit, n=len(g))
        for k in ("R1", "R2", "R3", "passed"):
            if g[k].notna().any():
                r.update(rate(g[k].dropna().astype(bool), k))
        for k in ("ev_ratio", "load_min", "phi", "phi_lo90", "between_share", "within_ev_ratio"):
            if k in g and g[k].notna().any():
                r[f"{k}_mean"] = g[k].mean()
        rows.append(r)
    t4 = pd.DataFrame(rows)
    t4.to_csv(os.path.join(OUT, "83c_l4.csv"), index=False)
    print("\nL4")
    cols = [c for c in ["condition", "dose", "unit", "n", "R1", "R2", "R3", "passed", "ev_ratio_mean",
                        "load_min_mean", "phi_mean", "phi_lo90_mean", "within_ev_ratio_mean"] if c in t4]
    print(t4[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
