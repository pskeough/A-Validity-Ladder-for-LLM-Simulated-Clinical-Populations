"""Level 4 of the intermediate panel (88) re-read with the R2 size condition.

Each replicate's simulated populations are rebuilt exactly as in 88 (same seed, same split, same
pseudo-models and planted types; the generation consumes the replicate's generator before any
statistic is computed). The reference loadings and the R2 bootstrap then use a separate generator
(default_rng([20261088, r, 4])), so R1 and the point loadings equal 88's and the R2 intervals are a
fresh bootstrap of the same data. R2 is recorded in both forms: shape only (congruence, as in 88)
and the frozen rule (91_ladder_core.r2_verdict: pass, fail or unresolved on congruence .95 and
loading RMSD .10). Level 4 is recombined with 88's own
R4 verdicts for the replicates where R4 ran.

usage: python 88d_panel_r2_size.py R [workers] [B_L4]      |  python 88d_panel_r2_size.py summary
Writes paper_brm/analysis_brm/intermediate_panel/r2_size_reps/rep_NNN.csv and 88d_r2_size.csv.
"""
import glob
import importlib.util
import os
import sys
import time

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "..", "paper_brm", "analysis_brm", "intermediate_panel")
P = None


def _load():
    spec = importlib.util.spec_from_file_location("p88", os.path.join(HERE, "88_intermediate_panel.py"))
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = [old[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


def run_rep(rep, B4):
    C, d, pers = P.C, P.STATE["d"], P.STATE["pers"]
    rng = np.random.default_rng([P.SEED, rep])
    donor = C.split(d, rng)
    don, ref = d[donor].reset_index(drop=True), d[~donor].reset_index(drop=True)
    gens = [C.generate(don, pers, rng) for _ in range(P.K)]
    types, _ = P.make_types(gens, don, pers, rng)
    rng4 = np.random.default_rng([P.SEED, rep, 4])
    l4ref = C.L4Ref(ref, B4, rng4)
    out = []
    for t in P.TYPES:
        g = types[t][0]
        for f, fn in ((0, "clinical"), (1, "narrative")):
            m = g["framing"] == f
            r = C.l4_eval(g["X"][m], g["persona"][m], l4ref, rng4)
            out.append(dict(rep=rep, type=t, unit=fn, **r))
    return pd.DataFrame(out)


def worker(args):
    global P
    rep, B4, path = args
    if P is None:
        P = _load()
        P.init(30, 100)
    t0 = time.time()
    df = run_rep(rep, B4)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return rep, time.time() - t0


def summarise():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUTD, "r2_size_reps", "rep_*.csv")))])
    for k in ("R1", "R2", "R2_size", "R2_both"):
        d[k] = d[k].astype(str) == "True"
    m = d.groupby(["rep", "type"])[["R1", "R2", "R2_size", "R2_both"]].all().reset_index()
    v = d.groupby(["rep", "type"]).R2_verdict.agg(
        lambda x: "fail" if (x == "fail").any() else ("pass" if (x == "pass").all() else "unresolved"))
    m = m.merge(v.rename("R2_new").reset_index(), on=["rep", "type"])
    reps = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(os.path.join(OUTD, "reps", "rep_*.csv")))])
    r4 = reps[(reps.rung == "L4-R4") & (reps.rule == "both framings, margin .08")][["rep", "type", "verdict"]]
    r4 = r4.rename(columns={"verdict": "R4_e08"})
    old = reps[(reps.rung == "L4-R2") & (reps.unit == "model")][["rep", "type", "verdict"]].rename(columns={"verdict": "R2_in_88"})
    m = m.merge(r4, on=["rep", "type"], how="left").merge(old, on=["rep", "type"], how="left")

    def l4(r, r2):
        if pd.isna(r.R4_e08):
            return np.nan
        if r.R1 and r2 == "pass" and r.R4_e08 == "pass":
            return "pass"
        if (not r.R1) or r2 == "fail" or r.R4_e08 == "fail":
            return "fail"
        return "unresolved"

    m["L4_old"] = m.apply(lambda r: l4(r, "pass" if r.R2 else "fail"), axis=1)
    m["L4_new"] = m.apply(lambda r: l4(r, r.R2_new), axis=1)
    m.to_csv(os.path.join(OUTD, "88d_r2_size_by_rep.csv"), index=False)
    g = m.groupby("type")
    s = pd.DataFrame(dict(
        reps=g.size(), R1=g.R1.mean(), R2_shape=g.R2.mean(), R2_shape_and_size=g.R2_both.mean(),
        R2_new_pass=g.R2_new.apply(lambda x: float(np.mean(x == "pass"))),
        R2_new_unresolved=g.R2_new.apply(lambda x: float(np.mean(x == "unresolved"))),
        R2_new_fail=g.R2_new.apply(lambda x: float(np.mean(x == "fail"))),
        R2_agrees_with_88=g.apply(lambda x: float(np.mean((x.R2_in_88 == "pass") == x.R2))),
        reps_with_R4=g.L4_old.count(),
        L4_pass_old=g.L4_old.apply(lambda x: float(np.mean(x.dropna() == "pass")) if x.notna().any() else np.nan),
        L4_pass_new=g.L4_new.apply(lambda x: float(np.mean(x.dropna() == "pass")) if x.notna().any() else np.nan),
        L4_fail_old=g.L4_old.apply(lambda x: float(np.mean(x.dropna() == "fail")) if x.notna().any() else np.nan),
        L4_fail_new=g.L4_new.apply(lambda x: float(np.mean(x.dropna() == "fail")) if x.notna().any() else np.nan)))
    f = d.groupby("type")[["load_min", "phi_lo90", "loading_rmsd", "loading_rmsd_hi90"]].mean()
    s = s.join(f).reindex([t for t in ["REAL", "ORACLE-INDEP", "SHIFTED-2.1", "SHIFTED-4.7", "STEEPENED-2x",
                                       "COMPRESSED", "NONINVARIANT"] if t in s.index]).reset_index()
    s.to_csv(os.path.join(OUTD, "88d_r2_size.csv"), index=False)
    pd.set_option("display.width", 250)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    if sys.argv[1:2] == ["summary"]:
        summarise()
        return
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    B4 = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    outdir = os.path.join(OUTD, "r2_size_reps")
    os.makedirs(outdir, exist_ok=True)
    todo = [(r, B4, os.path.join(outdir, f"rep_{r:03d}.csv")) for r in range(R)]
    todo = [t for t in todo if not os.path.exists(t[2])]
    print(f"{len(todo)} replicates, {W} workers, B_L4 {B4}", flush=True)
    if W == 1:
        for t in todo:
            print("rep %d: %.0fs" % worker(t), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(W) as ex:
            for f in as_completed([ex.submit(worker, t) for t in todo]):
                print("rep %d: %.0fs" % f.result(), flush=True)
    summarise()


if __name__ == "__main__":
    main()
