"""Check that the instrument-general ladder (91_ladder_core) returns the audit library's numbers on
the PHQ-8 corpus and NHANES 2005-2018.

Checks
  1. GRM fitted to NHANES with the MEC weights vs the published weighted fit (l1_grm_params.csv).
  2. theta, lz, lz* for every corpus draw vs 79_l1_lib.score_vectors on the same parameters.
  3. polychoric matrix of each model x framing vs 81_l4_lib.poly_from.
  4. level-1 point ratios per model x framing vs l1_personfit_main.csv.
  5. gate phi(30) and SE(30) per model x framing vs 82_gate_lib on the same draws, and the
     SE verdicts with the SD-unit tolerances (0.25, 0.125 SD) against 1.0 and 0.5 points.
Output: analysis/brm/91a_core_regression.csv
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(BASE, "analysis", "brm")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = [old[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


V = _load("core91", "91_ladder_core.py")
C = _load("c83", "83_controls_lib.py")
L1, L4 = C.L1, C.L4


def main():
    rows = []

    def rec(check, unit, value, reference):
        diff = float(np.max(np.abs(np.asarray(value, float) - np.asarray(reference, float))))
        rows.append(dict(check=check, unit=unit, max_abs_diff=diff))
        print(f"{check:28s} {unit:32s} max|diff| {diff:.2e}", flush=True)

    d = C.load_frame()
    sd = float(C.SD0)
    g = V.GRM(8, 4)
    fit = g.fit(d[C.DPQ].to_numpy(int), d.w7.to_numpy())
    a0, b0 = C.grm_params()
    rec("GRM fit vs published", "a", fit["a"], a0)
    rec("GRM fit vs published", "b", fit["b"], b0)

    m = L1.load_corpus()
    X = m[L1.ITEMS].to_numpy(int)
    th, lz, lzs = g.score(a0, b0, X)
    th1, lz1, lzs1 = L1.score_vectors(a0, b0, X)
    rec("scoring vs 79_l1_lib", "theta", th, th1)
    rec("scoring vs 79_l1_lib", "lz*", lzs, lzs1)

    pub = pd.read_csv(os.path.join(OUT, "l1_personfit_main.csv"))
    pub = pub[(pub.reference == "2005_2018 weighted GRM") & (pub["sample"] == "full") & (pub.subset == "all")]
    rng = np.random.default_rng(1)
    ref = V.L1Ref(d.total.to_numpy(), d.lz.to_numpy(), d.w7.to_numpy(), 24, 2, rng)
    for (mod, fr), sub in m.assign(lzs=lzs).groupby(["short", "framing"]):
        r = V.l1_eval(ref, sub.total.to_numpy(), sub.lzs.to_numpy(), sub.profile_id.to_numpy(), rng, B=2)
        p = pub[(pub.model == mod) & (pub.framing == fr)].iloc[0]
        rec("L1 point ratios vs published", f"{mod} {fr}", [r["misfit_ratio"], r["overfit_ratio"]],
            [p.misfit_ratio, p.overfit_ratio])

        Xs = sub[L1.ITEMS].to_numpy(int)
        T = V.Tables(Xs, np.ones(len(Xs)), sub.profile_id.to_numpy(), 4)
        Tl = L4.Tables(Xs, np.ones(len(Xs)), sub.profile_id.to_numpy())
        rec("polychorics vs 81_l4_lib", f"{mod} {fr}", V.poly_from(T.agg(), 8, 4), L4.poly_from(Tl.agg()))

        gt = V.gate(sub.total.to_numpy(), sub.profile_id.to_numpy(), 30, sd)
        s = sub.groupby("profile_id").total.agg(["size", "mean", "var"])
        c = V.GL.one_facet(s["mean"].to_numpy(), s["size"].to_numpy(float),
                           (s["var"].fillna(0) * (s["size"] - 1)).to_numpy())
        rec("gate phi(30) vs 82_gate_lib", f"{mod} {fr}", gt["phi_k"], V.GL.phi_k(c["p"], c["e"], 30))
        old_min = gt["se_k"] <= 1.0
        old_rec = gt["se_k"] <= 0.5
        rows.append(dict(check="gate verdict, SD tolerances vs points", unit=f"{mod} {fr}",
                         max_abs_diff=float((gt["pass_min"] != old_min) or (gt["pass_rec"] != old_rec)),
                         se30=gt["se_k"], tol_min=gt["tol_min"], tol_rec=gt["tol_rec"]))
        print(f"{'gate verdict SD vs points':28s} {mod + ' ' + fr:32s} SE(30) {gt['se_k']:.3f} "
              f"min {gt['pass_min']}/{old_min} rec {gt['pass_rec']}/{old_rec}", flush=True)
    tol = pd.read_csv(os.path.join(OUT, "l1_personfit_tolerance.csv"))
    res = [max(V._resolved(r.misfit_ci90_lo, r.misfit_ci90_hi), V._resolved(r.overfit_ci90_lo, r.overfit_ci90_hi))
           for r in tol.itertuples()]
    worst = max(res)
    tau_new = next(t for t in V.TAUS if worst <= t and t >= V.TAU_FLOOR)
    rows.append(dict(check="tau, resolved vs point departure", unit="NHANES subgroups",
                     max_abs_diff=abs(tau_new - tol.tau_chosen.iloc[0]), worst_resolved=worst,
                     worst_point=tol.max_fold_departure.iloc[0]))
    print(f"{'tau resolved vs point':28s} {'NHANES subgroups':32s} worst resolved {worst:.3f} -> tau {tau_new}; "
          f"point {tol.max_fold_departure.iloc[0]:.3f} -> tau {tol.tau_chosen.iloc[0]}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "91a_core_regression.csv"), index=False)
    print("worst:", out.groupby("check").max_abs_diff.max().to_string())


if __name__ == "__main__":
    main()
