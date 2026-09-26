"""Positive control and planted failures: replicates.

One replicate = one random donor/reference split of NHANES 2005-2018 and a panel of four
pseudo-models (the audit's panel size), each drawn independently from the donor half
(83_controls_lib). The unaltered panel is the positive control. Each planted failure is applied at
a grid of doses to that same panel, and every rung reads it against the reference half.

Scopes. L2 and L3 are read per model (four independent pseudo-models per replicate) and pooled
over the panel (78b and 80c conventions). L1 is read for all four models at dose 0 and for model 0
under the planted failures; L4 is read for model 0 only (its bootstrap is the cost of the run).

Planted failures (nested: a draw altered at dose p is altered at every larger dose):
  L1 shuffle    a share p of draws has its eight item values permuted at random (total kept): an
                incoherent routing. Expected reading: excess misfit.
  L1 typical    a share p of draws is replaced by the most typical human vector at the same total
                (the donor pattern nearest the weighted median lz* at that total). Expected
                reading: both tails in deficit, "compressed", the pattern all four models show.
  L2 gap        the persona cells of the first-named group are shifted so that the simulated gap
                is g times the population gap (NHANES full-sample standardised estimate), both
                framings, every model, for Women-Men, Low-High, Middle-High and Black-White.
                g = -0.5, 0, 0.5, 1.5, 2, 3 sit inside reversed, missing, attenuated, steepened,
                steepened, steepened (dose 1 is the positive control). Only the target contrast is
                read; the Bonferroni family stays seven contrasts.
  L3 shift      every persona cell mean moves by delta points. The PS residual moves by exactly
                delta and its SE does not change, so this is computed from the dose-0 rows.
                Integer clipping at 0 and 24 is ignored (the cell_min / cell_max columns show the
                cell means stay inside 0..24).
  L4 decorrelate  a share p of draws is rebuilt item by item, each item taken from a random draw
                of the same persona and framing: persona means kept, within-persona covariance
                removed. Expected reading: no general factor within personas (R3), then none at all
                (R1).

Reference precision. The reference half has about twice the sampling variance of the full NHANES
sample the audit used. L2 is therefore also read with the reference SE divided by sqrt(2)
(`verdict_auditprec`), an approximation to the audit's precision (the half-sample estimate still
carries the half's error, so this reading is slightly optimistic).

usage: python 83b_controls_simulate.py R [workers] [B_L1] [B_L4] [first_rep]
Writes analysis/brm/83b_reps/rep_NNN.csv (one file per replicate; existing files are skipped, so
a stopped run resumes). Seed: (20261001, replicate).
"""
import importlib.util
import os
import sys
import time

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261001
K = 4
L1_DOSES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
L2_DOSES = [-0.5, 0.0, 0.5, 1.5, 2.0, 3.0]
L2_TARGETS = ["Women minus Men", "Low minus High SES", "Middle minus High SES", "Black minus White"]
L3_DOSES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
L4_DOSES = [0.25, 0.5, 0.75, 1.0]
L2_KEEP = ("ratio", "ci_lo", "ci_hi", "verdict", "verdict_popstop", "r3_unstopped", "stop_conditional",
           "no_population_gap")

C = None
STATE = {}


def init():
    global C
    spec = importlib.util.spec_from_file_location("c83", os.path.join(HERE, "83_controls_lib.py"))
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    d = C.load_frame()
    STATE.update(d=d, pers=C.load_personas(), ab=C.grm_params(),
                 G=C.l2_reference(C.CellRef(d)))


def cell_stats(gen, pers, f):
    """48 anchored persona cells: mean total and its variance over draws, framing f."""
    s = np.full(48, np.nan)
    v = np.full(48, np.nan)
    for k, c in enumerate(pers.c48):
        if c < 0:
            continue
        t = gen["total"][(gen["persona"] == k) & (gen["framing"] == f)]
        s[c], v[c] = t.mean(), t.var(ddof=1) / len(t)
    return s, v


def run_rep(rep, B1, B4):
    t0 = time.time()
    d, pers, (a, b) = STATE["d"], STATE["pers"], STATE["ab"]
    rng = np.random.default_rng([SEED, rep])
    donor = C.split(d, rng)
    don, ref = d[donor].reset_index(drop=True), d[~donor].reset_index(drop=True)
    gens = [C.generate(don, pers, rng) for _ in range(K)]
    gen = gens[0]
    n = len(gen["total"])
    out = []

    def rec(**kw):
        out.append(dict(rep=rep, **kw))

    # ------------------------------------------------------------------------------------- L1
    ref1 = C.L1Ref(ref, B1, rng)
    frames = (("both", np.ones(n, bool)), ("clinical", gen["framing"] == 0), ("narrative", gen["framing"] == 1))
    for m in range(1, K):
        g = gens[m]
        for fr, msk in frames:
            r = C.l1_eval(ref1, g["total"][msk], g["lz"][msk], g["persona"][msk], rng)
            rec(rung="L1", condition="null", dose=0.0, model=m, unit=fr, **r)
    u = rng.random(n)
    Xs = rng.permuted(gen["X"], axis=1)
    _, _, lz_shuf = C.L1.score_vectors(a, b, Xs)
    bank = C.typical_bank(don)
    lz_typ = np.array([bank[int(t)][1] for t in gen["total"]])
    arms = [("null", 0.0, gen["lz"])]
    arms += [("L1 shuffle", p, np.where(u < p, lz_shuf, gen["lz"])) for p in L1_DOSES]
    arms += [("L1 typical", p, np.where(u < p, lz_typ, gen["lz"])) for p in L1_DOSES]
    for cond, p, lz in arms:
        for fr, msk in frames:
            r = C.l1_eval(ref1, gen["total"][msk], lz[msk], gen["persona"][msk], rng)
            rec(rung="L1", condition=cond, dose=p, model=0, unit=fr, **r)
    t1 = time.time()

    # ------------------------------------------------------------------------------------- L2
    cref = C.CellRef(ref)
    l2ref = C.l2_reference(cref)
    cs = [(cell_stats(g, pers, 0), cell_stats(g, pers, 1)) for g in gens]

    def l2_rows(cond, dose, name, shift, true_reg):
        sims = [C.l2_sim(sc + shift, sn + shift) for (sc, _), (sn, _) in cs]
        for scope, sim in [(str(m), s) for m, s in enumerate(sims)] + [("pooled", C.l2_pool(sims))]:
            o = C.l2_verdict(sim, l2ref, name)
            o2 = C.l2_verdict(sim, l2ref, name, 1 / np.sqrt(2))
            rec(rung="L2", condition=cond, dose=dose, model=scope, unit=name, true_region=true_reg,
                g_true=STATE["G"][name][0], ref_est=l2ref[name][0], ref_se=l2ref[name][1],
                sim_gap=sim[name][0], sim_se=sim[name][1], sim_df=sim[name][2],
                verdict_auditprec=o2["verdict"], **{k: o[k] for k in L2_KEEP})

    for name, dim, hi, lo in C.L2_CONTRASTS:
        l2_rows("null", 1.0, name, 0.0, "kept")
        if name not in L2_TARGETS:
            continue
        mask = C._attr(dim) == hi
        G = STATE["G"][name][0]
        for g in L2_DOSES:
            l2_rows("L2 gap", g, name, np.where(mask, (g - 1.0) * G, 0.0), C.true_region(g))

    # ------------------------------------------------------------------------------------- L3
    S = [((sc + sn) / 2, (vc + vn) / 4) for (sc, vc), (sn, vn) in cs]
    pooled = (np.mean([s for s, _ in S], axis=0), np.sum([v for _, v in S], axis=0) / K ** 2)
    for scope, (s, v) in [(str(m), x) for m, x in enumerate(S)] + [("pooled", pooled)]:
        rows = C.l3_eval(cref, s, v)
        for r in rows:
            rec(rung="L3", condition="null", dose=0.0, model=scope, unit=r["group"], resid=r["resid"],
                se=r["se"], df=r["df"])
        for delta in L3_DOSES:
            rec(rung="L3", condition="L3 shift", dose=delta, model=scope, unit="all ten groups",
                cell_min=float(np.min(s + delta)), cell_max=float(np.max(s + delta)),
                **{f"pass_{k}": C.l3_pass(rows, delta, tol) for k, tol in C.L3_TOLS.items()})
    t2 = time.time()

    # ------------------------------------------------------------------------------------- L4
    l4ref = C.L4Ref(ref, B4, rng)
    rec(rung="L4", condition="reference half", dose=0.0, model="reference", unit="reference", R1=l4ref.gf,
        within_ev_ratio=l4ref.within_ratio, R3=l4ref.within_gf, between_share=l4ref.between_share)
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
            rec(rung="L4", condition=cond, dose=p, model=0, unit=("clinical", "narrative")[f], **res[f])
        rec(rung="L4", condition=cond, dose=p, model=0, unit="model",
            **{k: bool(res[0][k] and res[1][k]) for k in ("R1", "R2", "R3")},
            passed=bool(all(res[f][k] for f in (0, 1) for k in ("R1", "R2", "R3"))))
    t3 = time.time()
    return pd.DataFrame(out), dict(rep=rep, l1=t1 - t0, l2l3=t2 - t1, l4=t3 - t2)


def worker(args):
    rep, B1, B4, path = args
    if C is None:
        init()
    df, tm = run_rep(rep, B1, B4)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return tm


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    B1 = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    B4 = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    outdir = os.path.join(HERE, "..", "analysis", "brm", "83b_reps")
    os.makedirs(outdir, exist_ok=True)
    todo = [(r, B1, B4, os.path.join(outdir, f"rep_{r:03d}.csv")) for r in range(first, first + R)]
    todo = [t for t in todo if not os.path.exists(t[3])]
    print(f"{len(todo)} replicates to run, {W} workers, B_L1 {B1}, B_L4 {B4}", flush=True)
    t0 = time.time()
    if W == 1:
        for t in todo:
            tm = worker(t)
            print(f"rep {tm['rep']}: L1 {tm['l1']:.0f}s  L2+L3 {tm['l2l3']:.0f}s  L4 {tm['l4']:.0f}s", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(W) as ex:
            futs = [ex.submit(worker, t) for t in todo]
            for k, f in enumerate(as_completed(futs), 1):
                tm = f.result()
                print(f"[{k}/{len(todo)} {time.time() - t0:.0f}s] rep {tm['rep']}: L1 {tm['l1']:.0f}s  "
                      f"L2+L3 {tm['l2l3']:.0f}s  L4 {tm['l4']:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
