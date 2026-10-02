"""Panel power: how many personas per group does a faithful simulator need to pass levels 2 and 4?

Question. In the intermediate panel (88) a simulator that returns real NHANES respondents never
passes level 2 or level 4 at the corpus design (12 to 24 anchored personas per group). This script
asks whether that is a property of the design's size or of the rungs. It enlarges the persona grid
by crossing the 48 anchored cells with age and education bands, which NHANES records, and reruns
the rungs that did not pass on the same kind of faithful simulator.

Grids (one persona per usable cell):
  G48       race x sex x income x marital status (the corpus's anchored personas)
  G144_age  G48 x age band (18-39, 40-59, 60+)
  G144_edu  G48 x education band (high school or less, some college, college graduate)
  G432      G48 x age band x education band
A cell is usable when the full NHANES 2005-2018 sample has at least MIN_N respondents in it from
at least MIN_PSU primary sampling units. Education is DMDEDUC2 (adults 20+) and DMDEDUC3 for
18- and 19-year-olds (grade 12 or below, high school graduate and GED -> high school or less;
more than high school -> some college).

Replicate. One random donor/reference split of the NHANES frame, stratified by the finest cell
(83_controls_lib.split), shared by every grid. A pseudo-model draws 30 donor respondents per
persona per framing, weighted by the MEC weight (83_controls_lib.generate, on the grid's cells).
K = 4 pseudo-models per grid.

Types:
  REAL          the faithful simulator above.
  STEEPENED-2x  REAL with every Low-income persona's mean raised by the grid's full-sample
                Low-minus-High gap, so the true Low-High ratio is 2 (level 2 only; level 2 reads
                cell means, so the plant is applied to the means).
  NONINVARIANT  REAL with items 3 to 5 of the Low-income personas resampled independently from
                the cell's item marginals (88's plant; level 4 only, on the first R4_NONINV reps).

Rungs (rules unchanged from 88):
  gate  82_gate_lib one-facet G-study, minimum rule phi(30) >= .80 and SE(30) <= 1.0, both framings.
  L2    78c r3 on the grid's standardised estimand: the equal-weight average over strata of the
        other attributes of within-stratum differences of weighted cell means (reference half,
        stratified delete-one-PSU jackknife), and 78b's pairing on the simulated side. A stratum
        enters a contrast when both its cells have donors and a positive reference total in the
        full half and in every jackknife replicate. Four variants as in 88 (pairs or draw-only SE,
        half-sample or audit-precision reference). One further reading, "limit", sets the
        simulated gap equal to the reference estimate with zero error: the best verdict the
        reference's own precision permits.
  L4    R1 and R2 (83_controls_lib.l4_eval) per framing on pseudo-model 0; R4 (88.r4_eval, 81c)
        for sex, race and income, compared with the NHANES full-sample verdicts at .08 and .05.

usage: python 90_panel_power.py cells
       python 90_panel_power.py R [workers] [R4_REPS] [R4_NONINV] [first_rep] [K_PERM] [B_R4]
Writes paper_brm/analysis_brm/panel_power/reps/rep_NNN.csv (resumable).
Seeds: replicate r uses default_rng([20261090, r]); R4 uses default_rng([20261090, r, grid, type,
framing, attribute]).
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
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUTD = os.path.join(BASE, "paper_brm", "analysis_brm", "panel_power")
RAW = os.path.join(BASE, "data", "nhanes_raw")
SEED = 20261090
K = 4
N_DRAWS = 30
B4 = 100
MIN_N, MIN_PSU = 4, 2
AGES = ["18-39", "40-59", "60+"]
EDUS = ["HS or less", "Some college", "College graduate"]
DIMS = ["race", "sex", "pir_band", "mar"]
GRIDS = {"G48": [], "G144_age": ["age_band"], "G144_edu": ["edu_band"], "G432": ["age_band", "edu_band"]}
CONTRAST_DIM = {0: "race", 1: "sex", 2: "pir_band"}
NONINV_ITEMS = [2, 3, 4]

P = None
STATE = {}


def _load(name, fname, argv=None):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    if argv is not None:
        sys.argv = argv
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


# ------------------------------------------------------------------------------------------ data
def load_edu():
    out = []
    for c in "DEFGHIJ":
        x = pd.read_sas(os.path.join(RAW, f"DEMO_{c}.xpt"), format="xport")
        out.append(x.reindex(columns=["SEQN", "DMDEDUC2", "DMDEDUC3"]))
    e = pd.concat(out)
    e2, e3 = e.DMDEDUC2, e.DMDEDUC3
    band = pd.Series(np.nan, index=e.index, dtype=object)
    band[e2.isin([1, 2, 3])] = "HS or less"
    band[e2 == 4] = "Some college"
    band[e2 == 5] = "College graduate"
    young = e2.isna()
    band[young & (e3.between(0, 14) | e3.isin([55, 66]))] = "HS or less"
    band[young & (e3 == 15)] = "Some college"
    return pd.DataFrame(dict(SEQN=e.SEQN.astype(int), edu_band=band.values))


def frame(C):
    d = C.load_frame()
    d["SEQN"] = d.SEQN.astype(int)
    n0 = len(d)
    d = d.merge(load_edu(), on="SEQN", how="left")
    assert len(d) == n0
    d["age_band"] = pd.cut(d.age, [17, 39, 59, 200], labels=AGES).astype(object)
    d.loc[~d.race.isin(C.RACES4), "cell"] = "unclassified"     # the Other group has no persona cell
    ok = (d.cell != "unclassified") & d.edu_band.notna()
    for g, extra in GRIDS.items():
        cols = DIMS + extra
        key = d[cols[0]].fillna("").astype(str)
        for c in cols[1:]:
            key = key + "|" + d[c].fillna("").astype(str)
        d[f"key_{g}"] = np.where((d.cell != "unclassified") & d[extra].notna().all(axis=1), key, "")
    d["fine"] = np.where(ok, d["key_G432"], "unclassified")
    return d


def usable_cells(d):
    """Cells with at least MIN_N respondents from at least MIN_PSU PSUs in the full sample."""
    out = {}
    for g in GRIDS:
        k = d[f"key_{g}"]
        s = d[k != ""].groupby(k[k != ""]).agg(n=("SEQN", "size"), psus=("psu", "nunique"))
        out[g] = sorted(s.index[(s.n >= MIN_N) & (s.psus >= MIN_PSU)])
    return out


def personas(cells, extra):
    rows = [dict(zip(DIMS + extra, c.split("|"))) for c in cells]
    p = pd.DataFrame(rows)
    p["sex_n"], p["ses_normalized"], p["cis"], p["cell"] = p.sex, p.pir_band, True, cells
    return p


def init(k_perm, b_r4):
    global P
    P = _load("p88", "88_intermediate_panel.py")
    P.init(k_perm, b_r4)
    C = P.C
    d = frame(C)
    cells = usable_cells(d)
    STATE.update(C=C, cells=cells)
    G = {}
    for g, extra in GRIDS.items():
        gr = GridRef(d, d[f"key_{g}"].to_numpy(), cells[g], design=False)
        G[g] = {name: l2_ref_est(gr, g, extra, name, dim, hi, lo, np.ones(len(cells[g]), bool))[0]
                for name, dim, hi, lo in C.L2_CONTRASTS}
    STATE.update(C=C, d=d, cells=cells, G=G,
                 pers={g: personas(cells[g], extra) for g, extra in GRIDS.items()})


# ------------------------------------------------------------------------------- reference side
class GridRef:
    """Weighted cell totals of one grid on a (half) sample, with JKn replicates (80_l3_lib.Design)."""

    def __init__(self, ref, key, cells, design=True):
        C = STATE.get("C") or P.C
        idx = {c: i for i, c in enumerate(cells)}
        ci = np.array([idx.get(k, -1) for k in key])
        ok = ci >= 0
        w = ref.w.to_numpy(float)[ok]
        y = ref.total.to_numpy(float)[ok]
        n = len(cells)
        if not design:
            self.Nf = np.bincount(ci[ok], weights=w, minlength=n)
            self.Yf = np.bincount(ci[ok], weights=w * y, minlength=n)
            self.Nr = self.Yr = None
        else:
            self.des = C.L3.Design(ref)
            oh = np.zeros((len(ref), n))
            oh[np.flatnonzero(ok), ci[ok]] = 1.0
            wa, ya = ref.w.to_numpy(float), ref.total.to_numpy(float)
            self.Nf, self.Nr = self.des.replicate(self.des.psu_totals(oh * wa[:, None]))
            self.Yf, self.Yr = self.des.replicate(self.des.psu_totals(oh * (wa * ya)[:, None]))
            self.df = self.des.dfree
        with np.errstate(divide="ignore", invalid="ignore"):
            self.mf = self.Yf / self.Nf
            self.mr = None if self.Nr is None else self.Yr / self.Nr
        self.ok = self.Nf > 0
        if self.Nr is not None:
            self.ok &= (self.Nr > 0).all(axis=0)


def strata_pairs(cells, extra, dim, hi, lo, live):
    """(hi cell, lo cell) per stratum of the other attributes, both cells live."""
    dims = DIMS + extra
    j = dims.index(CONTRAST_DIM[dim])
    parts = [c.split("|") for c in cells]
    by = {}
    for i, p in enumerate(parts):
        if not live[i] or p[j] not in (hi, lo):
            continue
        by.setdefault("|".join(p[:j] + p[j + 1:]), {})[p[j]] = i
    return [(v[hi], v[lo]) for v in by.values() if hi in v and lo in v]


def l2_ref_est(gr, g, extra, name, dim, hi, lo, live):
    pr = strata_pairs(STATE["cells"][g], extra, dim, hi, lo, live & gr.ok)
    h, l = np.array([a for a, _ in pr]), np.array([b for _, b in pr])
    est = float(np.mean(gr.mf[h] - gr.mf[l]))
    if gr.mr is None:
        return est, np.nan, np.nan, pr
    reps = (gr.mr[:, h] - gr.mr[:, l]).mean(axis=1)
    return est, float(np.sqrt(gr.des.jk_var(est, reps))), gr.df, pr


# ------------------------------------------------------------------------------ simulated side
def generate(don, key, cells, rng):
    w = don.w7.to_numpy()
    pools = {}
    for i, c in enumerate(cells):
        pool = np.flatnonzero(key == c)
        pools[i] = (pool, w[pool] / w[pool].sum()) if len(pool) else None
    rows, pers, frm = [], [], []
    for i in range(len(cells)):
        if pools[i] is None:
            continue
        pool, pr = pools[i]
        for f in (0, 1):
            rows.append(rng.choice(pool, N_DRAWS, replace=True, p=pr))
            pers.append(np.full(N_DRAWS, i))
            frm.append(np.full(N_DRAWS, f))
    r = np.concatenate(rows)
    C = P.C
    return dict(X=don[C.DPQ].to_numpy(int)[r], lz=don.lz.to_numpy()[r], total=don.total.to_numpy()[r],
                persona=np.concatenate(pers), framing=np.concatenate(frm)), pools


def cell_means(gen, n):
    s, v = np.full((2, n), np.nan), np.full((2, n), np.nan)
    for f in (0, 1):
        m = gen["framing"] == f
        p, t = gen["persona"][m], gen["total"][m].astype(float)
        cnt = np.bincount(p, minlength=n)
        sm = np.bincount(p, weights=t, minlength=n)
        ss = np.bincount(p, weights=t * t, minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            s[f] = sm / cnt
            v[f] = (ss - cnt * s[f] ** 2) / (cnt - 1) / cnt
    return s, v


def sim_gap(s, v, pr):
    h, l = np.array([a for a, _ in pr]), np.array([b for _, b in pr])
    diff = ((s[0, h] - s[0, l]) + (s[1, h] - s[1, l])) / 2
    vc = (v[0] + v[1]) / 4
    se_c = float(np.sqrt(np.sum((vc[h] + vc[l]) / len(pr) ** 2)))
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff))), len(diff) - 1, se_c


def resample_items(gen, rows_mask, pools, donX, rng):
    X = gen["X"].copy()
    items = np.asarray(NONINV_ITEMS)
    for i, pl in pools.items():
        if pl is None:
            continue
        rows = np.flatnonzero((gen["persona"] == i) & rows_mask)
        if len(rows) == 0:
            continue
        pick = rng.choice(pl[0], size=(len(rows), len(items)), replace=True, p=pl[1])
        X[rows[:, None], items[None, :]] = donX[pick, items[None, :]]
    return X


# ------------------------------------------------------------------------------------ replicate
def run_rep(rep, r4_reps, r4_noninv):
    t0 = time.time()
    C, d = P.C, STATE["d"]
    rng = np.random.default_rng([SEED, rep])
    dd = d.assign(cell=d.fine)
    donor = C.split(dd, rng)
    don, ref = d[donor].reset_index(drop=True), d[~donor].reset_index(drop=True)
    l4ref = C.L4Ref(ref, B4, rng)
    out, tm = [], {}

    def rec(**kw):
        out.append(dict(rep=rep, **kw))

    for gi, (g, extra) in enumerate(GRIDS.items()):
        tg = time.time()
        cells, pers = STATE["cells"][g], STATE["pers"][g]
        n = len(cells)
        dkey = don[f"key_{g}"].to_numpy()
        gens, pools = [], None
        for m in range(K):
            gen, pools = generate(don, dkey, cells, rng)
            gens.append(gen)
        live = np.array([pools[i] is not None for i in range(n)])
        gr = GridRef(ref, ref[f"key_{g}"].to_numpy(), cells)
        per_group = {CONTRAST_DIM[0]: pers.race.value_counts().min(), "sex": pers.sex.value_counts().min(),
                     "pir_band": pers.pir_band.value_counts().min()}
        rec(grid=g, type="design", rung="check", model="all", unit="grid", n_cells=n, n_live=int(live.sum()),
            n_ref_ok=int(gr.ok.sum()), min_race=per_group["race"], min_sex=per_group["sex"],
            min_income=per_group["pir_band"])

        # gate
        for m, gen in enumerate(gens):
            r = P.gate_eval(gen)
            rec(grid=g, type="REAL", rung="gate", model=str(m), unit="model",
                verdict="pass" if r["pass_min"] else "fail", **r)

        # L2
        means = [cell_means(gen, n) for gen in gens]
        refd = {}
        for name, dim, hi, lo in C.L2_CONTRASTS:
            est, se, df, pr = l2_ref_est(gr, g, extra, name, dim, hi, lo, live)
            refd[name] = (est, se, df, pr)
        low = (pers.pir_band == "Low").to_numpy()
        for t in ("REAL", "STEEPENED-2x"):
            for m, (s, v) in enumerate(means):
                s = s.copy()
                if t == "STEEPENED-2x":
                    s[:, low] += STATE["G"][g]["Low minus High SES"]
                sims = {name: sim_gap(s, v, refd[name][3]) for name, *_ in C.L2_CONTRASTS}
                variants = {"pairs_half": (0, 1.0), "pairs_auditprec": (0, 1 / np.sqrt(2)),
                            "cond_half": (1, 1.0), "cond_auditprec": (1, 1 / np.sqrt(2))}
                for vname, (cond, sc) in variants.items():
                    vv = []
                    for name, *_ in C.L2_CONTRASTS:
                        gg, se_p, df_p, se_c = sims[name]
                        sm = {name: (gg, se_c, np.inf) if cond else (gg, se_p, df_p)}
                        o = C.l2_verdict(sm, {name: refd[name][:3]}, name, sc)
                        vv.append(o["verdict"])
                        rec(grid=g, type=t, rung="L2", rule=f"contrast [{vname}]", model=str(m), unit=name,
                            verdict=o["verdict"], ratio=o["ratio"], ci_lo=o["ci_lo"], ci_hi=o["ci_hi"], k_rel=o["k_rel_halfwidth"], verdict_unstopped=o["r3_unstopped"], verdict_popstop=o["verdict_popstop"],
                            sim_gap=gg, sim_se=se_c if cond else se_p, ref_est=refd[name][0],
                            ref_se=refd[name][1] * sc, n_pairs=len(refd[name][3]))
                    rec(grid=g, type=t, rung="L2", rule=f"model [{vname}]", model=str(m), unit="model",
                        verdict=P.l2_model_verdict(vv), detail=" | ".join(vv))
        for vname, sc in (("limit_half", 1.0), ("limit_auditprec", 1 / np.sqrt(2))):
            vv = []
            for name, *_ in C.L2_CONTRASTS:
                est = refd[name][0]
                o = C.l2_verdict({name: (est, 1e-9, np.inf)}, {name: refd[name][:3]}, name, sc)
                vv.append(o["verdict"])
                rec(grid=g, type="REAL", rung="L2", rule=f"contrast [{vname}]", model="limit", unit=name,
                    verdict=o["verdict"], ratio=o["ratio"], ci_lo=o["ci_lo"], ci_hi=o["ci_hi"], k_rel=o["k_rel_halfwidth"], verdict_unstopped=o["r3_unstopped"], verdict_popstop=o["verdict_popstop"],
                    ref_est=est, ref_se=refd[name][1] * sc, n_pairs=len(refd[name][3]))
            rec(grid=g, type="REAL", rung="L2", rule=f"model [{vname}]", model="limit", unit="model",
                verdict=P.l2_model_verdict(vv), detail=" | ".join(vv))
        tm[f"{g}_l2"] = time.time() - tg

        # L4 on pseudo-model 0
        tl = time.time()
        gen = gens[0]
        donX = don[C.DPQ].to_numpy(int)
        types = {"REAL": gen["X"]}
        if rep < r4_noninv:
            types["NONINVARIANT"] = resample_items(gen, low[gen["persona"]], pools, donX, rng)
        for ti, (t, X) in enumerate(types.items()):
            res = {}
            for f, fn in ((0, "clinical"), (1, "narrative")):
                msk = gen["framing"] == f
                r = C.l4_eval(X[msk], gen["persona"][msk], l4ref, rng)
                r4 = {}
                if rep < r4_reps or t == "NONINVARIANT":
                    if r["R1"]:
                        tr = time.time()
                        r4, steps = P.r4_eval(X[msk], gen["persona"][msk], pers, [SEED, rep, gi, ti, f])
                        r4["r4_seconds"] = time.time() - tr
                        for s_ in steps:
                            rec(grid=g, type=t, rung="L4-R4 step", rule=f"{s_['attribute']} {s_['step']}",
                                model="0", unit=fn, **{k: v for k, v in s_.items() if k not in ("attribute", "step")})
                    else:
                        r4 = {"R4_e05": "no verdict (no general factor)", "R4_e08": "no verdict (no general factor)"}
                res[f] = dict(r, **r4)
                rec(grid=g, type=t, rung="L4", rule="framing", model="0", unit=fn, **res[f])
            both = lambda k: bool(res[0][k] and res[1][k])  # noqa: E731
            for k in ("R1", "R2"):
                rec(grid=g, type=t, rung=f"L4-{k}", rule="both framings", model="0", unit="model",
                    verdict="pass" if both(k) else "fail")
            if "R4_e08" in res[0]:
                for e in ("08", "05"):
                    vs = [res[f][f"R4_e{e}"] for f in (0, 1)]
                    v = "fail" if any(x == "fail" or x.startswith("no verdict") for x in vs) else \
                        ("pass" if all(x == "pass" for x in vs) else "unresolved")
                    rec(grid=g, type=t, rung="L4-R4", rule=f"both framings, margin .{e}", model="0",
                        unit="model", verdict=v, detail=" | ".join(vs))
                    l4 = "pass" if (both("R1") and both("R2") and v == "pass") else \
                        ("fail" if (not both("R1") or not both("R2") or v == "fail") else "unresolved")
                    rec(grid=g, type=t, rung="L4", rule=f"level 4 (R1, R2, R4 at .{e})", model="0",
                        unit="model", verdict=l4)
        tm[f"{g}_l4"] = time.time() - tl
    tm["total"] = time.time() - t0
    return pd.DataFrame(out), dict(rep=rep, **tm)


def worker(args):
    rep, r4_reps, r4_noninv, k_perm, b_r4, path = args
    if P is None:
        init(k_perm, b_r4)
    df, tm = run_rep(rep, r4_reps, r4_noninv)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return tm


def cells_report():
    init(30, 100)
    d, cells = STATE["d"], STATE["cells"]
    os.makedirs(OUTD, exist_ok=True)
    rows = []
    for g, extra in GRIDS.items():
        pers = STATE["pers"][g]
        k = d[f"key_{g}"]
        n_all = len(pers.cell) if not extra else int(48 * np.prod([3] * len(extra)))
        cov = float(d.w[k.isin(cells[g])].sum() / d.w[d.cell != "unclassified"].sum())
        rows.append(dict(grid=g, cells_possible=n_all, cells_usable=len(cells[g]), weight_share=round(cov, 4),
                         personas_per_race_min=int(pers.race.value_counts().min()),
                         personas_per_sex_min=int(pers.sex.value_counts().min()),
                         personas_per_income_min=int(pers.pir_band.value_counts().min()),
                         **{f"G_{n}": round(v, 3) for n, v in STATE["G"][g].items()}))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTD, "90_cells.csv"), index=False)
    print(out.T.to_string())
    print("education missing in frame:", int(d.edu_band.isna().sum()), "of", len(d))


def main():
    a = sys.argv[1:]
    if a and a[0] == "cells":
        return cells_report()
    R = int(a[0]) if len(a) > 0 else 1
    W = int(a[1]) if len(a) > 1 else 1
    r4_reps = int(a[2]) if len(a) > 2 else R
    r4_noninv = int(a[3]) if len(a) > 3 else 0
    first = int(a[4]) if len(a) > 4 else 0
    k_perm = int(a[5]) if len(a) > 5 else 30
    b_r4 = int(a[6]) if len(a) > 6 else 100
    repdir = os.path.join(OUTD, "reps")
    os.makedirs(repdir, exist_ok=True)
    todo = [(r, r4_reps, r4_noninv, k_perm, b_r4, os.path.join(repdir, f"rep_{r:03d}.csv"))
            for r in range(first, first + R)]
    todo = [t for t in todo if not os.path.exists(t[-1])]
    print(f"{len(todo)} replicates, {W} workers, R4 on reps < {r4_reps}, NONINVARIANT on reps < {r4_noninv}, "
          f"K_PERM {k_perm}, B_R4 {b_r4}", flush=True)
    t0 = time.time()
    log = []
    if W == 1:
        for t in todo:
            tm = worker(t)
            log.append(tm)
            print({k: round(v, 1) if isinstance(v, float) else v for k, v in tm.items()}, flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(W) as ex:
            futs = [ex.submit(worker, t) for t in todo]
            for k, f in enumerate(as_completed(futs), 1):
                tm = f.result()
                log.append(tm)
                print(f"[{k}/{len(todo)} {time.time() - t0:.0f}s] " +
                      str({kk: round(v, 1) if isinstance(v, float) else v for kk, v in tm.items()}), flush=True)
    pd.DataFrame(log).to_csv(os.path.join(OUTD, f"90_timing_{first:03d}_{first + R - 1:03d}.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
