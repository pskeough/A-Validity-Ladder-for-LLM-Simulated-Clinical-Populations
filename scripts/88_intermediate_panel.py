"""Intermediate simulator panel: does each rung discriminate, and which rung does each failure trip?

Question. The positive control (83b) shows that real people pass and that four raw LLMs fail. It
does not show which rung each kind of failure trips, or whether a non-human simulator that gets
part of the answer right can pass the rungs it should pass. This script builds simulators that are
right in some respects and wrong in one, and runs every one through every rung with that rung's
own pass rule.

Design. The corpus design of 83b (83_controls_lib): one replicate is one random donor/reference
split of NHANES 2005-2018 adults, stratified by the 48 persona cells, and a panel of four
pseudo-models (120 personas x 2 framings x 30 draws), each drawn independently from the donor half.
Every simulator type is derived from that same panel, so the types are paired within a replicate.

Simulator types (applied to every draw of every pseudo-model in the panel):
  REAL            the 83b positive control: a draw is a donor respondent from the persona's cell,
                  sampled with probability proportional to the MEC weight.
  ORACLE-INDEP    cell-mean oracle with independent item noise: each of a draw's eight item scores
                  is sampled independently from the donor cell's weighted marginal of that item.
                  Cell means and gaps are right in expectation; covariance within a draw is gone.
  SHIFTED-2.1     real people from the persona's cell, but more depressed ones: each persona's draws
  SHIFTED-4.7     are resampled from its donor cell under an exponential tilt of the MEC weights,
                  p_i ~ w_i exp(beta_c t_i), with beta_c solved per cell so the cell's expected total
                  rises by d points (d = 2.1 and 4.7, the range of the real models' overall L3
                  residuals). Every draw is still a whole human answer vector, so the level moves and
                  the answer patterns stay human. (A first version that moved each draw by d points
                  through a donor vector at total t + d removed the all-zero mass that carries much of
                  the PHQ-8's polychoric correlation and failed R1 at d = 4.7; it was replaced before
                  the run.)
  STEEPENED-2x    REAL, except that every Low-income persona is resampled under the same tilt with
                  d = G, G = the full-sample standardised Low-minus-High gap (1.30 points), so the
                  simulated Low-High gap is 2 times the population gap (83b's L2 dose g = 2, applied
                  to the first-named group as 83b does). Low-Middle steepens as a consequence.
  COMPRESSED      every draw replaced by the most typical donor vector at its total (83b's "L1
                  typical" plant at dose 1.0; 83_controls_lib.typical_bank).
  NONINVARIANT    REAL, except that within the Low-income personas items 3, 4 and 5 (sleep, fatigue,
                  appetite: DPQ030-050) are replaced by values sampled independently from the donor
                  cell's weighted marginal of each item. Item means are unchanged in expectation;
                  the three items lose their loading in the Low-income group only.

Rungs, each with its own rule (functions reused from 82_gate_lib, 83_controls_lib, 81c):
  gate  one-facet G-study per pseudo-model and framing (82_gate_lib.one_facet); minimum rule at
        k = 30: phi(30) >= .80 and SE(30) <= 1.0 in both framings; recommended: .90 and 0.5.
  L1    83_controls_lib.l1_eval per framing (tau = 1.5); a model passes when it passes in both.
  L2    83_controls_lib.l2_sim / l2_verdict (78c r3, Bonferroni 98.6%, asymmetric bands, conditional
        stop), seven contrasts; model rule: pass when every non-stopped contrast is "kept" alone,
        fail when any verdict excludes "kept", otherwise unresolved. Read per model and pooled.
  L3    83_controls_lib.l3_eval / l3_pass, PS residual, ten groups, TOST; primary tolerance 2 points.
  L4    83_controls_lib.l4_eval (R1, R2, R3) per framing on model 0; R4 by 81c.analyse (multi-group
        CFA, permutation nu over personas, persona bootstrap), run only where R1 holds (the rung's
        rule: a population without a general factor gets no R4 verdict). R4 compares each of six
        steps with the NHANES full-sample verdict (l4_invariance.csv) at margin .08 (primary) and .05.
        Level 4 passes when R1, R2 and R4 pass in both framings; R3 is a diagnostic.

usage: python 88_intermediate_panel.py R [workers] [B_L1] [B_L4] [R4_REPS] [K_PERM] [B_R4] [R4_REPS_OTHER]
       [first_rep]
Writes paper_brm/analysis_brm/intermediate_panel/reps/rep_NNN.csv (resumable).
Seeds: replicate r uses default_rng([20261088, r]); R4 uses default_rng([20261088, r, type, framing,
attribute]).
"""
import contextlib
import importlib.util
import io
import os
import sys
import time

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUTD = os.path.join(BASE, "paper_brm", "analysis_brm", "intermediate_panel")
SEED = 20261088
K = 4
TYPES = ["REAL", "ORACLE-INDEP", "SHIFTED-2.1", "SHIFTED-4.7", "STEEPENED-2x", "COMPRESSED", "NONINVARIANT"]
R4_PRIMARY = ["REAL", "NONINVARIANT"]   # R4 calibration and power: replicates < R4_REPS
# every other type gets R4 on replicates < R4_REPS_OTHER (budget); R4 runs only where R1 holds
NONINV_ITEMS = [2, 3, 4]  # DPQ030 sleep, DPQ040 fatigue, DPQ050 appetite
STOPS = ("reference too imprecise", "no population gap")
R4_ATTRS = [("sex", ["F", "M"]), ("race", ["White", "Black", "Asian", "Hispanic"]),
            ("income", ["Low", "Middle", "High"])]

C = GL = R4M = None
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


def init(k_perm, b_r4):
    global C, GL, R4M
    C = _load("c83", "83_controls_lib.py")
    GL = _load("gl82", "82_gate_lib.py")
    R4M = _load("r4_81c", "81c_l4_invariance.py", argv=["81c", "all", str(b_r4), str(k_perm), "2"])
    R4M.K_PERM, R4M.B, R4M.B_POLY = k_perm, b_r4, 2      # polychoric refit is not part of R4's rule
    d = C.load_frame()
    pers = C.load_personas()
    inv = pd.read_csv(os.path.join(C.OUTD, "l4_invariance.csv"))
    nh = inv[inv.population.str.startswith("NHANES") & inv.step.isin(["metric", "scalar"])]
    STATE.update(d=d, pers=pers, ab=C.grm_params(), G=C.l2_reference(C.CellRef(d)),
                 nh_r4={(r.attribute, r.step): (r.verdict_e05, r.verdict_e08) for r in nh.itertuples()})


# ------------------------------------------------------------------------------ simulator types
def cell_pools(don, pers):
    w = don.w7.to_numpy()
    out = {}
    for c in pers.cell.unique():
        pool = np.flatnonzero(don.cell.to_numpy() == c)
        out[c] = (pool, w[pool] / w[pool].sum())
    return out


def rescore(g, X, ab):
    a, b = ab
    _, _, lz = C.L1.score_vectors(a, b, X)
    return dict(X=X, total=X.sum(1), lz=lz, persona=g["persona"], framing=g["framing"])


def resample_items(g, rows_mask, items, pers, pools, donX, rng):
    """Replace the given items of the masked rows by independent draws from the cell marginals."""
    X = g["X"].copy()
    items = np.asarray(items)
    for k, c in enumerate(pers.cell):
        rows = np.flatnonzero((g["persona"] == k) & rows_mask)
        if len(rows) == 0:
            continue
        pool, pr = pools[c]
        pick = rng.choice(pool, size=(len(rows), len(items)), replace=True, p=pr)
        X[rows[:, None], items[None, :]] = donX[pick, items[None, :]]
    return X


def tilt(pr, t, d):
    """Exponential tilt of a cell's donor weights, p_i proportional to pr_i exp(beta t_i), with beta
    chosen so the tilted mean total is the cell mean + d. Returns (probabilities, realised shift, ESS)."""
    m0 = float(pr @ t)

    def mean_at(beta):
        z = beta * (t - t.max())
        q = pr * np.exp(z)
        return float(q @ t / q.sum())

    target = m0 + d
    if target >= t.max() - 1e-9:
        beta = 50.0
    else:
        from scipy.optimize import brentq
        beta = brentq(lambda b: mean_at(b) - target, 0.0, 50.0) if mean_at(50.0) > target else 50.0
    q = pr * np.exp(beta * (t - t.max()))
    q /= q.sum()
    return q, float(q @ t) - m0, float(1.0 / np.sum(q ** 2))


def tilted_draws(g, delta_p, pers, pools, donX, dontot, donlz, rng, diag):
    """A simulator that returns real people from the persona's cell, but more depressed ones: persona
    k's draws are resampled from its donor cell under an exponential tilt that raises the cell's
    expected total by delta_p[k] (personas with delta 0 keep their REAL draws)."""
    X, lz = g["X"].copy(), g["lz"].copy()
    for k, c in enumerate(pers.cell):
        if delta_p[k] == 0:
            continue
        rows = np.flatnonzero(g["persona"] == k)
        pool, pr = pools[c]
        key = (c, float(delta_p[k]))
        if key not in diag:
            diag[key] = tilt(pr, dontot[pool].astype(float), float(delta_p[k]))
        q = diag[key][0]
        pick = rng.choice(pool, len(rows), replace=True, p=q)
        X[rows], lz[rows] = donX[pick], donlz[pick]
    return dict(X=X, total=X.sum(1), lz=lz, persona=g["persona"], framing=g["framing"])


def make_types(gens, don, pers, rng):
    ab = STATE["ab"]
    pools = cell_pools(don, pers)
    donX, donlz, dontot = don[C.DPQ].to_numpy(int), don.lz.to_numpy(), don.total.to_numpy()
    bank = C.typical_bank(don)
    G_lh = STATE["G"]["Low minus High SES"][0]
    low_p = (pers.ses_normalized == "Low").to_numpy()
    diag = {}
    out = {t: [] for t in TYPES}
    for g in gens:
        n = len(g["total"])
        out["REAL"].append(g)
        out["ORACLE-INDEP"].append(rescore(g, resample_items(g, np.ones(n, bool), range(8), pers, pools,
                                                             donX, rng), ab))
        for d, tn in ((2.1, "SHIFTED-2.1"), (4.7, "SHIFTED-4.7")):
            out[tn].append(tilted_draws(g, np.full(len(pers), d), pers, pools, donX, dontot, donlz, rng, diag))
        out["STEEPENED-2x"].append(tilted_draws(g, np.where(low_p, G_lh, 0.0), pers, pools, donX, dontot,
                                                donlz, rng, diag))
        Xc = np.array([bank[int(t)][0] for t in g["total"]])
        out["COMPRESSED"].append(dict(X=Xc, total=Xc.sum(1), lz=np.array([bank[int(t)][1] for t in g["total"]]),
                                      persona=g["persona"], framing=g["framing"]))
        out["NONINVARIANT"].append(rescore(g, resample_items(g, low_p[g["persona"]], NONINV_ITEMS, pers,
                                                             pools, donX, rng), ab))
    return out, diag


# ------------------------------------------------------------------------------------ rungs
def cell_stats(gen, pers, f):
    s, v = np.full(48, np.nan), np.full(48, np.nan)
    for k, c in enumerate(pers.c48):
        if c < 0:
            continue
        t = gen["total"][(gen["persona"] == k) & (gen["framing"] == f)]
        s[c], v[c] = t.mean(), t.var(ddof=1) / len(t)
    return s, v


def gate_eval(g):
    """One-facet G-study per framing on the 120 personas; minimum and recommended rules at k = 30."""
    r = {}
    for f, fn in ((0, "clinical"), (1, "narrative")):
        m = g["framing"] == f
        df = pd.DataFrame(dict(p=g["persona"][m], y=g["total"][m].astype(float)))
        s = df.groupby("p").y.agg(["size", "mean", "var"])
        c = GL.one_facet(s["mean"].to_numpy(), s["size"].to_numpy(float),
                         (s["var"].fillna(0) * (s["size"] - 1)).to_numpy())
        r[f"phi30_{fn}"] = GL.phi_k(c["p"], c["e"], 30)
        r[f"phi1_{fn}"] = GL.phi_k(c["p"], c["e"], 1)
        r[f"se30_{fn}"] = float(np.sqrt(GL.pos(c["e"]) / 30))
        r[f"s2p_{fn}"], r[f"s2e_{fn}"] = c["p"], c["e"]
        r[f"k_min_{fn}"] = max(GL.k_for_phi(c["p"], c["e"], .80), GL.k_for_se(c["e"], 1.0))
        r[f"k_rec_{fn}"] = max(GL.k_for_phi(c["p"], c["e"], .90), GL.k_for_se(c["e"], 0.5))
    fr = ("clinical", "narrative")
    r["pass_min"] = all(r[f"phi30_{f}"] >= .80 and r[f"se30_{f}"] <= 1.0 for f in fr)
    r["pass_rec"] = all(r[f"phi30_{f}"] >= .90 and r[f"se30_{f}"] <= 0.5 for f in fr)
    r["pass_single"] = all(r[f"phi1_{f}"] >= .80 and np.sqrt(GL.pos(r[f"s2e_{f}"])) <= 1.0 for f in fr)
    return r


def l2_model_verdict(verdicts):
    """pass: every non-stopped contrast 'kept' alone; fail: any verdict excludes 'kept'."""
    live = [v for v in verdicts if v not in STOPS]
    if any(v != "undetermined" and "kept" not in v.split(" or ") for v in live):
        return "fail"
    if live and all(v == "kept" for v in live):
        return "pass"
    return "unresolved"


def r4_eval(X, persona, pers, rng_seed):
    """Six invariance steps on one framing population, compared with NHANES at .05 and .08."""
    rows = []
    race = pers.race.to_numpy()
    sexv = np.where(pers.cis, np.where(pers.sex_n == "Women", "F", "M"), "")
    inc = pers.ses_normalized.to_numpy()
    col = {"sex": sexv, "race": race, "income": inc}
    for ai, (attr, levels) in enumerate(R4_ATTRS):
        lab = col[attr][persona]
        keep = np.isin(lab, levels)
        gidx = np.array([levels.index(v) for v in lab[keep]])
        rng = np.random.default_rng(rng_seed + [ai])
        with contextlib.redirect_stdout(io.StringIO()):
            res, fitrow = R4M.analyse("sim", X[keep].astype(float), np.ones(keep.sum()), gidx, len(levels),
                                      persona[keep], None, persona[keep], None, rng, (attr, levels))
        for r in res:
            if r["step"] not in ("metric", "scalar"):
                continue
            ref05, ref08 = STATE["nh_r4"][(attr, r["step"])]
            rows.append(dict(attribute=attr, step=r["step"], rmsea_d=r["rmsea_d"], lo=r["rmsea_d_lo90"],
                             hi=r["rmsea_d_hi90"], p_perm=r["p_perm"], v05=r["verdict_e05"],
                             v08=r["verdict_e08"], ref05=ref05, ref08=ref08,
                             converged=fitrow["converged"]))
    out = {}
    for e in ("05", "08"):
        st = []
        for r in rows:
            vm, vr = r[f"v{e}"], r[f"ref{e}"]
            det = vm in ("holds", "fails") and vr in ("holds", "fails")
            st.append("match" if det and vm == vr else ("mismatch" if det else "unresolved"))
        out[f"R4_e{e}"] = "fail" if "mismatch" in st else ("pass" if all(s == "match" for s in st) else "unresolved")
        out[f"steps_e{e}"] = ";".join(f"{r['attribute']}-{r['step']}:{s}" for r, s in zip(rows, st))
    return out, rows


def run_rep(rep, B1, B4, r4_reps, r4_other):
    t0 = time.time()
    d, pers = STATE["d"], STATE["pers"]
    rng = np.random.default_rng([SEED, rep])
    donor = C.split(d, rng)
    don, ref = d[donor].reset_index(drop=True), d[~donor].reset_index(drop=True)
    gens = [C.generate(don, pers, rng) for _ in range(K)]
    types, diag = make_types(gens, don, pers, rng)
    out = []

    def rec(**kw):
        out.append(dict(rep=rep, **kw))

    for dose in sorted({k[1] for k in diag}):
        v = np.array([x[1:] for k, x in diag.items() if k[1] == dose])
        rec(type="tilt", rung="check", rule=f"tilt dose {dose:.3f}", model="all", unit="cells",
            n_cells=len(v), realised_min=v[:, 0].min(), realised_mean=v[:, 0].mean(), ess_min=v[:, 1].min(),
            ess_median=float(np.median(v[:, 1])))

    tm = {}
    # ------------------------------------------------------------------- realised shifts (checks)
    for t in TYPES:
        g, g0 = types[t][0], types["REAL"][0]
        low = (pers.ses_normalized == "Low").to_numpy()[g["persona"]]
        rec(type=t, rung="check", rule="realised", model="0", unit="all",
            mean_total=float(g["total"].mean()), shift_total=float(g["total"].mean() - g0["total"].mean()),
            shift_low=float(g["total"][low].mean() - g0["total"][low].mean()),
            shift_notlow=float(g["total"][~low].mean() - g0["total"][~low].mean()))

    # --------------------------------------------------------------------------------- gate
    for t in TYPES:
        for m, g in enumerate(types[t]):
            r = gate_eval(g)
            rec(type=t, rung="gate", rule="minimum k=30", model=str(m), unit="model",
                verdict="pass" if r["pass_min"] else "fail", **r)
    t1 = time.time(); tm["gate"] = t1 - t0

    # ----------------------------------------------------------------------------------- L1
    ref1 = C.L1Ref(ref, B1, rng)
    for t in TYPES:
        g = types[t][0]
        vs = []
        for f, fn in ((0, "clinical"), (1, "narrative")):
            msk = g["framing"] == f
            r = C.l1_eval(ref1, g["total"][msk], g["lz"][msk], g["persona"][msk], rng)
            vs.append(r["verdict"])
            rec(type=t, rung="L1", rule="framing", model="0", unit=fn, **r)
        rec(type=t, rung="L1", rule="model (both framings pass)", model="0", unit="model",
            verdict="pass" if all(v == "pass" for v in vs) else
            ("fail" if any(v.startswith("fail") for v in vs) else "unresolved"), detail=" | ".join(vs))
    t2 = time.time(); tm["l1"] = t2 - t1

    # ------------------------------------------------------------------------------ L2 and L3
    cref = C.CellRef(ref)
    l2ref = C.l2_reference(cref)
    for t in TYPES:
        cs = [(cell_stats(g, pers, 0), cell_stats(g, pers, 1)) for g in types[t]]
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
                    rec(type=t, rung="L2", rule=f"contrast [{vname}]", model=scope, unit=name,
                        verdict=o["verdict"], ratio=o["ratio"], ci_lo=o["ci_lo"], ci_hi=o["ci_hi"],
                        sim_gap=sm[name][0], sim_se=sm[name][1], ref_est=l2ref[name][0],
                        ref_se=l2ref[name][1] * sc_, g_true=STATE["G"][name][0])
                rec(type=t, rung="L2", rule=f"model [{vname}]", model=scope, unit="model",
                    verdict=l2_model_verdict(vv), detail=" | ".join(vv))
        S = [((sc + sn) / 2, (vc + vn) / 4) for (sc, vc), (sn, vn) in cs]
        pooled = (np.mean([s for s, _ in S], axis=0), np.sum([v for _, v in S], axis=0) / K ** 2)
        for scope, (s, v) in [(str(m), x) for m, x in enumerate(S)] + [("pooled", pooled)]:
            rows = C.l3_eval(cref, s, v)
            for r in rows:
                rec(type=t, rung="L3", rule="group", model=scope, unit=r["group"], resid=r["resid"],
                    se=r["se"], df=r["df"],
                    verdict="pass" if abs(r["resid"]) + r["tcrit"] * r["se"] < 2.0 else "fail")
            p = {k: C.l3_pass(rows, 0.0, tol) for k, tol in C.L3_TOLS.items()}
            rec(type=t, rung="L3", rule="model (all ten groups, 2 pt)", model=scope, unit="model",
                verdict="pass" if p["2pt"] else "fail", **{f"pass_{k}": x for k, x in p.items()},
                max_abs_resid=max(abs(r["resid"]) for r in rows))
    t3 = time.time(); tm["l2l3"] = t3 - t2

    # ----------------------------------------------------------------------------------- L4
    l4ref = C.L4Ref(ref, B4, rng)
    for ti, t in enumerate(TYPES):
        g = types[t][0]
        res = {}
        for f, fn in ((0, "clinical"), (1, "narrative")):
            m = g["framing"] == f
            r = C.l4_eval(g["X"][m], g["persona"][m], l4ref, rng)
            r4 = {}
            if rep < (r4_reps if t in R4_PRIMARY else r4_other):
                if r["R1"]:
                    tr = time.time()
                    r4, steps = r4_eval(g["X"][m], g["persona"][m], pers, [SEED, rep, ti, f])
                    r4["r4_seconds"] = time.time() - tr
                    for s in steps:
                        rec(type=t, rung="L4-R4 step", rule=f"{s['attribute']} {s['step']}", model="0", unit=fn,
                            **{k: v for k, v in s.items() if k not in ("attribute", "step")})
                else:
                    r4 = {"R4_e05": "no verdict (no general factor)", "R4_e08": "no verdict (no general factor)"}
            res[f] = dict(r, **r4)
            rec(type=t, rung="L4", rule="framing", model="0", unit=fn, **res[f])
        both = lambda k: bool(res[0][k] and res[1][k])  # noqa: E731
        for k in ("R1", "R2", "R3"):
            rec(type=t, rung=f"L4-{k}", rule="both framings", model="0", unit="model",
                verdict="pass" if both(k) else "fail")
        if "R4_e08" in res[0]:
            for e in ("08", "05"):
                vs = [res[f][f"R4_e{e}"] for f in (0, 1)]
                v = "fail" if any(x == "fail" or x.startswith("no verdict") for x in vs) else \
                    ("pass" if all(x == "pass" for x in vs) else "unresolved")
                rec(type=t, rung="L4-R4", rule=f"both framings, margin .{e}", model="0", unit="model",
                    verdict=v, detail=" | ".join(vs))
                l4 = "pass" if (both("R1") and both("R2") and v == "pass") else \
                    ("fail" if (not both("R1") or not both("R2") or v == "fail") else "unresolved")
                rec(type=t, rung="L4", rule=f"level 4 (R1, R2, R4 at .{e})", model="0", unit="model",
                    verdict=l4)
    t4 = time.time(); tm["l4"] = t4 - t3
    tm["total"] = t4 - t0
    return pd.DataFrame(out), dict(rep=rep, **tm)


def worker(args):
    rep, B1, B4, r4_reps, r4_other, k_perm, b_r4, path = args
    if C is None:
        init(k_perm, b_r4)
    df, tm = run_rep(rep, B1, B4, r4_reps, r4_other)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return tm


def main():
    a = sys.argv[1:]
    R = int(a[0]) if len(a) > 0 else 1
    W = int(a[1]) if len(a) > 1 else 1
    B1 = int(a[2]) if len(a) > 2 else 400
    B4 = int(a[3]) if len(a) > 3 else 100
    r4_reps = int(a[4]) if len(a) > 4 else R
    k_perm = int(a[5]) if len(a) > 5 else 200
    b_r4 = int(a[6]) if len(a) > 6 else 200
    r4_other = int(a[7]) if len(a) > 7 else r4_reps
    first = int(a[8]) if len(a) > 8 else 0
    repdir = os.path.join(OUTD, "reps")
    os.makedirs(repdir, exist_ok=True)
    todo = [(r, B1, B4, r4_reps, r4_other, k_perm, b_r4, os.path.join(repdir, f"rep_{r:03d}.csv"))
            for r in range(first, first + R)]
    todo = [t for t in todo if not os.path.exists(t[-1])]
    print(f"{len(todo)} replicates, {W} workers, B_L1 {B1}, B_L4 {B4}, R4 on reps < {r4_reps} (REAL, NONINVARIANT) and < {r4_other} (others), "
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
    pd.DataFrame(log).to_csv(os.path.join(OUTD, f"88_timing_{first:03d}_{first + R - 1:03d}.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
