"""Calibration of the R2 size condition (loading RMSD) on the controls design.

R2 as first written reads Tucker's congruence only, which ignores the scale of the loadings: with
half the draws decorrelated the loadings fall from about .70 to .52 and congruence stays at .999
(83c_l4.csv). The size condition adds that the upper 90% limit of the root mean square difference
between the model's loadings and the reference loadings is at most R2_RMSD (.10). This script runs
83b's level-4 block (null and the four decorrelation doses, pseudo-model 0, both framings) and
records both conditions, so the pass rate of real people (null) and the detection rate of each
dose can be read for the old and the new R2.

usage: python 83h_r2_size_calibration.py R [workers] [B_L4] [first_rep]
Writes analysis/brm/83h_reps/rep_NNN.csv and analysis/brm/83h_r2_size.csv.
Seed: (20261083, replicate).
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
OUTD = os.path.join(HERE, "..", "analysis", "brm")
SEED = 20261083
L4_DOSES = [0.25, 0.5, 0.75, 1.0]

C = None
STATE = {}


def init():
    global C
    spec = importlib.util.spec_from_file_location("c83", os.path.join(HERE, "83_controls_lib.py"))
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    STATE.update(d=C.load_frame(), pers=C.load_personas())


def combine(vs):
    """Model verdict over framings: fail if any fails, pass if all pass, otherwise unresolved."""
    return "fail" if "fail" in vs else ("pass" if all(v == "pass" for v in vs) else "unresolved")


def run_rep(rep, B4):
    d, pers = STATE["d"], STATE["pers"]
    rng = np.random.default_rng([SEED, rep])
    donor = C.split(d, rng)
    don, ref = d[donor].reset_index(drop=True), d[~donor].reset_index(drop=True)
    gen = C.generate(don, pers, rng)
    n = len(gen["total"])
    l4ref = C.L4Ref(ref, B4, rng)
    out = []
    v4 = rng.random(n)
    grp = gen["persona"] * 2 + gen["framing"]
    order = np.argsort(grp, kind="stable")
    start = np.searchsorted(grp[order], np.arange(grp.max() + 1))
    size = np.bincount(grp)
    pick = order[start[grp][:, None] + (rng.random((n, 8)) * size[grp][:, None]).astype(int)]
    Xd = gen["X"][pick, np.arange(8)[None, :]]
    for cond, p in [("null", 0.0)] + [("L4 decorrelate", p) for p in L4_DOSES]:
        X = np.where((v4 < p)[:, None], Xd, gen["X"])
        res = {}
        for f in (0, 1):
            m = gen["framing"] == f
            res[f] = C.l4_eval(X[m], gen["persona"][m], l4ref, rng)
            out.append(dict(rep=rep, condition=cond, dose=p, unit=("clinical", "narrative")[f], **res[f]))
        out.append(dict(rep=rep, condition=cond, dose=p, unit="model",
                        **{k: bool(res[0][k] and res[1][k]) for k in ("R1", "R2", "R2_size", "R2_both", "R3")},
                        R2_verdict=combine([res[f]["R2_verdict"] for f in (0, 1)])))
    return pd.DataFrame(out)


def worker(args):
    rep, B4, path = args
    if C is None:
        init()
    t0 = time.time()
    df = run_rep(rep, B4)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return rep, time.time() - t0


def summarise():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUTD, "83h_reps", "rep_*.csv")))])
    m = d[d.unit == "model"].copy()
    for k in ("R1", "R2", "R2_size", "R2_both", "R3"):
        m[k] = m[k].astype(str) == "True"
    m["L4_old_R1R2"] = m.R1 & m.R2
    m["L4_new_R1R2"] = m.R1 & m.R2_both
    s = m.groupby(["condition", "dose"], dropna=False)[["R1", "R2", "R2_size", "R2_both", "L4_old_R1R2", "L4_new_R1R2"]].mean()
    s.insert(0, "reps", m.groupby(["condition", "dose"], dropna=False).size())
    for v in ("pass", "unresolved", "fail"):
        s[f"R2_{v}"] = m.assign(x=m.R2_verdict == v).groupby(["condition", "dose"], dropna=False).x.mean()
    f = d[d.unit != "model"].groupby(["condition", "dose"], dropna=False)[
        ["load_min", "phi", "phi_lo90", "loading_rmsd", "loading_rmsd_lo90", "loading_rmsd_hi90"]].mean()
    s = s.join(f).reset_index()
    s.to_csv(os.path.join(OUTD, "83h_r2_size.csv"), index=False)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    if sys.argv[1:2] == ["summary"]:
        summarise()
        return
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    B4 = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    outdir = os.path.join(OUTD, "83h_reps")
    os.makedirs(outdir, exist_ok=True)
    todo = [(r, B4, os.path.join(outdir, f"rep_{r:03d}.csv")) for r in range(first, first + R)]
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
