"""Level 2 power under two SE conventions, on the same splits and pseudo-model panels as 83b.

83b's replicate r draws its split and its four pseudo-models first from default_rng((20261001, r));
this script repeats exactly those calls, so replicates 0..199 here are 83b's panels. Level 2 only
(closed form, seconds per replicate), so it runs more replicates than 83b.

SE conventions for the simulated gap:
  pairs        78b: SD of persona-pair differences / sqrt(pairs), df pairs - 1 (the audit's rule)
  conditional  draw variance only, conditional on the fixed persona set (level 3's convention);
               df infinite
Reference precision: the reference half's jackknife SE, and that SE / sqrt(2) (approximately the
audit's full-sample precision).
Scopes: each of the four pseudo-models, and the pooled panel. Doses as in 83b.

Donor term: a pseudo-model reproduces the donor half, whose own contrast differs from the population
by that half's design error. That error is independent of the reference half's and is added to the
simulated side's SE (Satterthwaite df), so coverage is judged fairly. A real model has no donor
term, so these power figures are conservative; 83f gives power at the audit's own precision.
Level 3 null calibration is repeated with the same term (83e_l3_null.csv).

usage: python 83e_l2_power.py R [workers]
Emits analysis/brm/83e_l2_power_reps.csv and 83e_l2_power.csv.
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
L2_DOSES = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
C = None
ST = {}


def init():
    global C
    spec = importlib.util.spec_from_file_location("c83", os.path.join(HERE, "83_controls_lib.py"))
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    d = C.load_frame()
    ST.update(d=d, pers=C.load_personas(), G=C.l2_reference(C.CellRef(d)))


def cell_stats(gen, pers, f):
    s, v = np.full(48, np.nan), np.full(48, np.nan)
    for k, c in enumerate(pers.c48):
        if c >= 0:
            t = gen["total"][(gen["persona"] == k) & (gen["framing"] == f)]
            s[c], v[c] = t.mean(), t.var(ddof=1) / len(t)
    return s, v


def one(rep):
    if C is None:
        init()
    d, pers = ST["d"], ST["pers"]
    rng = np.random.default_rng([SEED, rep])
    donor = C.split(d, rng)
    don, ref = d[donor].reset_index(drop=True), d[~donor].reset_index(drop=True)
    gens = [C.generate(don, pers, rng) for _ in range(K)]
    cref = C.CellRef(ref)
    l2ref = C.l2_reference(cref)
    # the pseudo-models reproduce the donor half, not the population: that half's own design SE of
    # each contrast is added to the simulated side (a real model has no such term)
    dref = C.CellRef(don)
    l2don = C.l2_reference(dref)
    cs = [(cell_stats(g, pers, 0), cell_stats(g, pers, 1)) for g in gens]
    rows = []
    # level 3 with the same correction: PS residual, donor design variance of sum_c p_c y_c(donor)
    S = [((sc + sn) / 2, (vc + vn) / 4) for (sc, vc), (sn, vn) in cs]
    pooled = (np.mean([s for s, _ in S], axis=0), np.sum([v for _, v in S], axis=0) / K ** 2)
    for scope, (s, v) in [(str(m), x) for m, x in enumerate(S)] + [("pooled", pooled)]:
        for r in C.l3_eval(cref, s, v):
            gname = r["group"]
            dim = next(dd for nm, dd, _ in C.L3_GROUPS if nm == gname)
            val = next(vv for nm, _, vv in C.L3_GROUPS if nm == gname)
            gm = np.ones(48, bool) if dim is None else C._attr(dim) == val
            p = cref.Nf[gm] / cref.Nf[gm].sum()
            vdon = float(dref.des.jk_var(p @ dref.mf[gm], dref.mr[:, gm] @ p))
            se2 = np.sqrt(r["se"] ** 2 + vdon)
            rows.append(dict(rep=rep, rung="L3", contrast=gname, scope=scope, resid=r["resid"], se=r["se"],
                             se_with_donor=se2, tcrit=r["tcrit"]))
    for name, dim, hi, lo in C.L2_CONTRASTS:
        mask = C._attr(dim) == hi
        G = ST["G"][name][0]
        for g in (L2_DOSES if name in ("Women minus Men", "Low minus High SES", "Middle minus High SES",
                                       "Black minus White") else [1.0]):
            sh = np.where(mask, (g - 1.0) * G, 0.0)
            sims = [C.l2_sim(sc + sh, sn + sh, vc, vn) for (sc, vc), (sn, vn) in cs]
            scopes = [(str(m), s) for m, s in enumerate(sims)] + [("pooled", C.l2_pool(sims))]
            for scope, sim in scopes:
                if scope == "pooled":
                    cond = {k: (np.mean([s[k][0] for s in sims]),
                                float(np.sqrt(np.sum([s[k][3] ** 2 for s in sims]))) / K, np.inf) for k in sim}
                else:
                    cond = C.l2_conditional(sim)
                for se_conv, s_ in (("pairs", sim), ("conditional", cond)):
                    s_ = {k: (v[0], float(np.sqrt(v[1] ** 2 + l2don[k][1] ** 2)),
                              (v[1] ** 2 + l2don[k][1] ** 2) ** 2 /
                              (v[1] ** 4 / v[2] + l2don[k][1] ** 4 / l2don[k][2])) for k, v in s_.items()}
                    for refp, scale in (("half", 1.0), ("audit", 1 / np.sqrt(2))):
                        o = C.l2_verdict(s_, l2ref, name, scale)
                        rows.append(dict(rep=rep, rung="L2", contrast=name, dose=g, true_region=C.true_region(g),
                                         scope=scope, se_convention=se_conv, reference=refp,
                                         sim_gap=s_[name][0], sim_se=s_[name][1], ref_se=l2ref[name][1] * scale,
                                         ratio=o["ratio"], ci_lo=o["ci_lo"], ci_hi=o["ci_hi"],
                                         verdict=o["verdict"]))
    return rows


def l2_class(v, true):
    if v in ("reference too imprecise", "no population gap"):
        return "stopped"
    if v == "undetermined":
        return "undetermined"
    names = v.split(" or ")
    return "correct" if names == [true] else ("compatible" if true in names else "wrong")


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    t0 = time.time()
    rows = []
    if W == 1:
        for r in range(R):
            rows += one(r)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(W) as ex:
            for rr in ex.map(one, range(R), chunksize=4):
                rows += rr
    out = pd.DataFrame(rows)
    OUT = os.path.join(HERE, "..", "analysis", "brm")
    out.to_csv(os.path.join(OUT, "83e_l2_power_reps.csv"), index=False)
    pd.set_option("display.width", 220)
    l2 = out[out.rung == "L2"].copy()
    l2["cls"] = [l2_class(v, t) for v, t in zip(l2.verdict, l2.true_region)]
    l2["scope2"] = np.where(l2.scope == "pooled", "pooled", "per model")
    l2["covered"] = (l2.ci_lo <= l2.dose) & (l2.dose <= l2.ci_hi)
    summ = (l2.groupby(["se_convention", "reference", "scope2", "contrast", "dose", "true_region"])
            .agg(n=("cls", "size"), correct=("cls", lambda x: (x == "correct").mean()),
                 correct_or_compatible=("cls", lambda x: x.isin(["correct", "compatible"]).mean()),
                 wrong=("cls", lambda x: (x == "wrong").mean()),
                 undetermined=("cls", lambda x: (x == "undetermined").mean()),
                 stopped=("cls", lambda x: (x == "stopped").mean()),
                 coverage=("covered", "mean"), sim_se=("sim_se", "mean"), ref_se=("ref_se", "mean"))
            .reset_index())
    summ.to_csv(os.path.join(OUT, "83e_l2_power.csv"), index=False)
    print(summ.round(3).to_string(index=False))
    # level 3 null calibration, with and without the donor-half term
    from scipy import stats
    l3 = out[out.rung == "L3"].copy()
    l3["scope2"] = np.where(l3.scope == "pooled", "pooled", "per model")
    rows3 = []
    for (sc, gname), g in l3.groupby(["scope2", "contrast"]):
        for lab, col in (("draws + reference", "se"), ("draws + reference + donor", "se_with_donor")):
            cover = np.abs(g.resid) <= g.tcrit * g[col]
            rows3.append(dict(scope=sc, group=gname, se_terms=lab, n=len(g), bias=g.resid.mean(),
                              sd_resid=g.resid.std(), mean_se=g[col].mean(), coverage90=cover.mean()))
    t3 = pd.DataFrame(rows3)
    t3.to_csv(os.path.join(OUT, "83e_l3_null.csv"), index=False)
    print(t3.round(3).to_string(index=False))
    print(f"{R} replicates in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
