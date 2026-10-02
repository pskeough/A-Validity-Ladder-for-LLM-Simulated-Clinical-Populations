"""Level 2 of the controls (83b) recomputed under the frozen level-2 rule.

83b's level-2 block reads only the four pseudo-models, which each replicate draws immediately after
the donor/reference split, so it is rebuilt exactly here without rerunning level 1 (the replicate's
generator is consumed in the same order up to that point). The rows have 83b's columns; `verdict`
and `verdict_auditprec` now follow 78c_l2_r3_verdicts.r3 as frozen (LADDER_SPEC.md). Each
replicate is checked against 83b's stored ratio and interval.

usage: python 83i_controls_l2_frozen.py R [workers]
Writes analysis/brm/83i_l2_reps/rep_NNN.csv; 83c_controls_summary.py reads level 2 from there.
"""
import importlib.util
import os
import sys

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "..", "analysis", "brm")
B = None


def _load():
    spec = importlib.util.spec_from_file_location("b83", os.path.join(HERE, "83b_controls_simulate.py"))
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = [old[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


def run_rep(rep):
    C, d, pers = B.C, B.STATE["d"], B.STATE["pers"]
    rng = np.random.default_rng([B.SEED, rep])
    donor = C.split(d, rng)
    don, ref = d[donor].reset_index(drop=True), d[~donor].reset_index(drop=True)
    gens = [C.generate(don, pers, rng) for _ in range(B.K)]
    out = []
    cref = C.CellRef(ref)
    l2ref = C.l2_reference(cref)
    cs = [(B.cell_stats(g, pers, 0), B.cell_stats(g, pers, 1)) for g in gens]

    def l2_rows(cond, dose, name, shift, true_reg):
        sims = [C.l2_sim(sc + shift, sn + shift) for (sc, _), (sn, _) in cs]
        for scope, sim in [(str(m), s) for m, s in enumerate(sims)] + [("pooled", C.l2_pool(sims))]:
            o = C.l2_verdict(sim, l2ref, name)
            o2 = C.l2_verdict(sim, l2ref, name, 1 / np.sqrt(2))
            out.append(dict(rep=rep, rung="L2", condition=cond, dose=dose, model=scope, unit=name,
                            true_region=true_reg, g_true=B.STATE["G"][name][0], ref_est=l2ref[name][0],
                            ref_se=l2ref[name][1], sim_gap=sim[name][0], sim_se=sim[name][1],
                            sim_df=sim[name][2], verdict_auditprec=o2["verdict"],
                            k_rel=o["k_rel_halfwidth"], k_rel_auditprec=o2["k_rel_halfwidth"],
                            verdict_conditional=o["verdict_conditional"],
                            **{k: o[k] for k in B.L2_KEEP}))

    for name, dim, hi, lo in C.L2_CONTRASTS:
        l2_rows("null", 1.0, name, 0.0, "kept")
        if name not in B.L2_TARGETS:
            continue
        mask = C._attr(dim) == hi
        G = B.STATE["G"][name][0]
        for g in B.L2_DOSES:
            l2_rows("L2 gap", g, name, np.where(mask, (g - 1.0) * G, 0.0), C.true_region(g))
    new = pd.DataFrame(out)
    old = pd.read_csv(os.path.join(OUTD, "83b_reps", f"rep_{rep:03d}.csv"), low_memory=False,
                      keep_default_na=False, na_values=[""])
    old = old[old.rung == "L2"].reset_index(drop=True)
    for c in ("ratio", "ci_lo", "ci_hi", "sim_gap", "ref_est"):
        a, b = new[c].to_numpy(float), pd.to_numeric(old[c]).to_numpy(float)
        assert np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-9), (rep, c)
    return new


def worker(args):
    global B
    rep, path = args
    if B is None:
        B = _load()
        B.init()
    df = run_rep(rep)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return rep


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    outdir = os.path.join(OUTD, "83i_l2_reps")
    os.makedirs(outdir, exist_ok=True)
    todo = [(r, os.path.join(outdir, f"rep_{r:03d}.csv")) for r in range(R)]
    todo = [t for t in todo if not os.path.exists(t[1])]
    print(f"{len(todo)} replicates, {W} workers", flush=True)
    if W == 1:
        for t in todo:
            print("rep", worker(t), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(W) as ex:
            for f in as_completed([ex.submit(worker, t) for t in todo]):
                print("rep", f.result(), flush=True)


if __name__ == "__main__":
    main()
