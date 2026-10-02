"""Level 2 of the intermediate panel (88) recomputed under the frozen level-2 rule.

88's level-2 block reads only the planted types, which each replicate builds right after the
donor/reference split and before any statistic draws from the generator, so the block is rebuilt
exactly here (same seed, split, pseudo-models and plants). The rows have 88's level-2 columns plus
k_rel and the pre-freeze conditional verdict; `verdict` follows 78c_l2_r3_verdicts.r3 as frozen
(LADDER_SPEC.md). Each replicate is checked against 88's stored ratios.

usage: python 88e_panel_l2_frozen.py R [workers]
Writes paper_brm/analysis_brm/intermediate_panel/l2_frozen_reps/rep_NNN.csv; 88b reads level 2
from there.
"""
import importlib.util
import os
import sys

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


def run_rep(rep):
    C, d, pers = P.C, P.STATE["d"], P.STATE["pers"]
    rng = np.random.default_rng([P.SEED, rep])
    donor = C.split(d, rng)
    don, ref = d[donor].reset_index(drop=True), d[~donor].reset_index(drop=True)
    gens = [C.generate(don, pers, rng) for _ in range(P.K)]
    types, _ = P.make_types(gens, don, pers, rng)
    out = []
    cref = C.CellRef(ref)
    l2ref = C.l2_reference(cref)
    for t in P.TYPES:
        cs = [(P.cell_stats(g, pers, 0), P.cell_stats(g, pers, 1)) for g in types[t]]
        sims = [C.l2_sim(sc, sn, vc, vn) for (sc, vc), (sn, vn) in cs]
        scopes = [(str(m), s) for m, s in enumerate(sims)] + [("pooled", C.l2_pool(sims))]
        for scope, sim in scopes:
            variants = {"pairs_half": (sim, 1.0), "pairs_auditprec": (sim, 1 / np.sqrt(2))}
            if scope != "pooled":
                variants["cond_half"] = (C.l2_conditional(sim), 1.0)
                variants["cond_auditprec"] = (C.l2_conditional(sim), 1 / np.sqrt(2))
            for vname, (sm, sc_) in variants.items():
                vv = []
                for name, *_ in C.L2_CONTRASTS:
                    o = C.l2_verdict(sm, l2ref, name, sc_)
                    vv.append(o["verdict"])
                    out.append(dict(rep=rep, type=t, rung="L2", rule=f"contrast [{vname}]", model=scope, unit=name,
                                    verdict=o["verdict"], ratio=o["ratio"], ci_lo=o["ci_lo"], ci_hi=o["ci_hi"],
                                    sim_gap=sm[name][0], sim_se=sm[name][1], ref_est=l2ref[name][0],
                                    ref_se=l2ref[name][1] * sc_, g_true=P.STATE["G"][name][0],
                                    k_rel=o["k_rel_halfwidth"], verdict_conditional=o["verdict_conditional"],
                                    r3_unstopped=o["r3_unstopped"]))
                out.append(dict(rep=rep, type=t, rung="L2", rule=f"model [{vname}]", model=scope, unit="model",
                                verdict=P.l2_model_verdict(vv), detail=" | ".join(vv)))
    new = pd.DataFrame(out)
    old = pd.read_csv(os.path.join(OUTD, "reps", f"rep_{rep:03d}.csv"), low_memory=False,
                      keep_default_na=False, na_values=[""])
    old = old[old.rung == "L2"].reset_index(drop=True)
    assert len(old) == len(new), (rep, len(old), len(new))
    for c in ("ratio", "ci_lo", "ci_hi", "sim_gap"):
        a, b = pd.to_numeric(new[c]).to_numpy(float), pd.to_numeric(old[c]).to_numpy(float)
        assert np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-9), (rep, c)
    return new


def worker(args):
    global P
    rep, path = args
    if P is None:
        P = _load()
        P.init(30, 100)
    df = run_rep(rep)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return rep


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    outdir = os.path.join(OUTD, "l2_frozen_reps")
    os.makedirs(outdir, exist_ok=True)
    todo = [(r, os.path.join(outdir, f"rep_{r:03d}.csv")) for r in range(R)]
    todo = [t for t in todo if not os.path.exists(t[1])]
    print(f"{len(todo)} replicates, {W} workers", flush=True)
    for t in todo:
        print("rep", worker(t), flush=True)


if __name__ == "__main__":
    main()
