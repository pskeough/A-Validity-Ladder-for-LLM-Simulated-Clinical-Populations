"""Level 4, part 3: is the floor-multiple reading of scripts 68 and 73 still usable, per model?

The published statistic is the severity-matched Frobenius distance between two groups' Pearson
item-correlation matrices, read as a multiple of a split-half floor, with a pass band of 0.5 to 1.4
taken from NHANES. Script 73 moved the corpus floor from draws to personas. This script asks two
questions of that reading, per model and framing on corpus v3:

  1. Where does a no-difference contrast land? Group labels are permuted among personas (the
     independent units), within persona-severity tertiles so the permuted groups keep comparable
     severity, and the matched distance is recomputed. The null's 95th percentile is expressed as a
     multiple of the same persona floor. If it falls inside 0.5 to 1.4, the band cannot tell a
     reproduced divergence from no divergence.
  2. A size-free replacement. The floor multiple grows with sample size under a real difference, so
     it cannot be compared across sources of different size. The squared Frobenius distance
     between two population matrices can be estimated without that dependence by cross-fitting:
     split each group's clusters in half, and take <R_a1 - R_b1, R_a2 - R_b2>_F, whose expectation
     is ||R_a - R_b||^2 because the halves are independent. Mean over 50 splits; percentile
     interval over 200 cluster-bootstrap replicates (personas; NHANES PSUs within strata, Rao-Wu).
     Reported as the root, sqrt(max(0, D2)), on the correlation scale; no severity matching.

NHANES is run the same way with respondents as the unit for the floor and the null (script 68's
procedure) and PSUs for the cross-fit bootstrap.

usage: python 81d_l4_floor_null.py [all|dec]
Emits analysis/brm/l4_floor_null{_dec}.csv.
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l4lib", os.path.join(HERE, "81_l4_lib.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

SUBSET = sys.argv[1] if len(sys.argv) > 1 else "all"
SUF = "" if SUBSET == "all" else "_" + SUBSET
IT = L.ITEMS
BANDS = [-1, 4, 9, 14, 19, 24]
N_MATCH, N_SPLIT, N_PERM, N_XFIT, B_XFIT = 20, 200, 200, 50, 200
SEED = 8104
BAND_LO, BAND_HI = 0.5, 1.4
CONTRASTS = [("Women vs Men", "sex", "F", "M"), ("Black vs White", "race", "Black", "White"),
             ("Asian vs White", "race", "Asian", "White"),
             ("Hispanic vs White", "race", "Hispanic", "White"),
             ("Low vs High income", "ses", "Low", "High"),
             ("Middle vs High income", "ses", "Middle", "High"),
             ("Low vs Middle income", "ses", "Low", "Middle")]


def cmat(a):
    return np.corrcoef(a, rowvar=False)


def frob(a, b):
    return float(np.linalg.norm(a - b))


def matched(Xa, ba, Xb, bb, rng, n_match=N_MATCH):
    ds = []
    for _ in range(n_match):
        ia, ib = [], []
        for k in range(5):
            xa = np.flatnonzero(ba == k); xb = np.flatnonzero(bb == k)
            m = min(len(xa), len(xb))
            if m < 2:
                continue
            ia.append(rng.choice(xa, m, replace=False)); ib.append(rng.choice(xb, m, replace=False))
        ds.append(frob(cmat(Xa[np.concatenate(ia)]), cmat(Xb[np.concatenate(ib)])))
    return float(np.mean(ds))


def floor_p95(X, unit, rng):
    """Split-half floor over units (personas, or respondents in NHANES)."""
    u = np.unique(unit)
    ds = []
    for _ in range(N_SPLIT):
        p = rng.permutation(u); h = len(p) // 2
        left = np.isin(unit, p[:h]); right = np.isin(unit, p[h:2 * h])
        ds.append(frob(cmat(X[left]), cmat(X[right])))
    return float(np.quantile(ds, 0.95))


def sums(X, w, cl):
    ids, ci = np.unique(cl, return_inverse=True)
    C = len(ids)
    s0 = np.bincount(ci, weights=w, minlength=C)
    s1 = np.stack([np.bincount(ci, weights=w * X[:, i], minlength=C) for i in range(L.P)], 1)
    s2 = np.zeros((C, L.P, L.P))
    for i in range(L.P):
        for j in range(i, L.P):
            v = np.bincount(ci, weights=w * X[:, i] * X[:, j], minlength=C)
            s2[:, i, j] = v; s2[:, j, i] = v
    return ids, s0, s1, s2


def corr_from(s0, s1, s2, m):
    S0 = m @ s0; S1 = m @ s1; S2 = np.tensordot(m, s2, 1)
    mu = S1 / S0
    S = S2 / S0 - np.outer(mu, mu)
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def crossfit(A, Bb, rng, ma=None, mb=None):
    """Mean over splits of <R_a1 - R_b1, R_a2 - R_b2>; A, Bb = (ids, s0, s1, s2); ma, mb are
    bootstrap multiplicities per cluster (None = 1)."""
    ca, cb = len(A[0]), len(Bb[0])
    ma = np.ones(ca) if ma is None else ma
    mb = np.ones(cb) if mb is None else mb
    ia = np.flatnonzero(ma > 0); ib = np.flatnonzero(mb > 0)
    vals = []
    for _ in range(N_XFIT):
        pa = rng.permutation(ia); pb = rng.permutation(ib)
        ha, hb = len(pa) // 2, len(pb) // 2
        out = []
        for sa, sb in ((pa[:ha], pb[:hb]), (pa[ha:], pb[hb:])):
            m1 = np.zeros(ca); m1[sa] = ma[sa]
            m2 = np.zeros(cb); m2[sb] = mb[sb]
            out.append(corr_from(*A[1:], m1) - corr_from(*Bb[1:], m2))
        vals.append(float((out[0] * out[1]).sum()))
    # a half with a constant item has an undefined correlation; such splits are skipped
    vals = np.asarray(vals)
    return float(np.mean(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else float("nan")


def boot_mult(C, rng, strata):
    return L.cluster_boot_mult(C, rng, strata)


def run_contrast(label, df, cname, col, ga, gb, is_nh, rng):
    sa, sb = df[df[col] == ga], df[df[col] == gb]
    Xa, Xb = sa[IT].to_numpy(float), sb[IT].to_numpy(float)
    ba = pd.cut(sa[IT].sum(axis=1), BANDS, labels=False).to_numpy()
    bb = pd.cut(sb[IT].sum(axis=1), BANDS, labels=False).to_numpy()
    obs = matched(Xa, ba, Xb, bb, rng)
    unit_a = (sa.SEQN if is_nh else sa.profile_id).to_numpy()
    unit_b = (sb.SEQN if is_nh else sb.profile_id).to_numpy()
    fl = max(floor_p95(Xa, unit_a, rng), floor_p95(Xb, unit_b, rng))

    # null: permute group labels among units within severity strata
    both = pd.concat([sa.assign(_g=0), sb.assign(_g=1)])
    Xab = both[IT].to_numpy(float)
    bab = pd.cut(both[IT].sum(axis=1), BANDS, labels=False).to_numpy()
    unit = (both.SEQN if is_nh else both.profile_id).to_numpy()
    if is_nh:
        # script 68's null: rows are the units; permute labels within severity band
        strat_of_row = bab
        uid, uinv = np.unique(unit, return_inverse=True)
        ug = both._g.to_numpy()
        ustrat = bab
    else:
        tot = both[IT].sum(axis=1)
        pm = tot.groupby(both.profile_id).mean()
        terc = pd.qcut(pm, 3, labels=False)
        uid, uinv = np.unique(unit, return_inverse=True)
        ug = both.groupby("profile_id")._g.first().loc[uid].to_numpy()
        ustrat = terc.loc[uid].to_numpy()
    null = []
    for _ in range(N_PERM):
        ugp = ug.copy()
        for s in np.unique(ustrat):
            ix = np.flatnonzero(ustrat == s)
            ugp[ix] = rng.permutation(ug[ix])
        g = ugp if is_nh else ugp[uinv]
        ma_, mb_ = g == 0, g == 1
        null.append(matched(Xab[ma_], bab[ma_], Xab[mb_], bab[mb_], rng, n_match=1))
    null95 = float(np.quantile(null, 0.95))

    # cross-fitted size-free distance with a cluster bootstrap
    cl_a = sa.cluster.to_numpy(); cl_b = sb.cluster.to_numpy()
    A = sums(Xa, sa.w.to_numpy(float), cl_a)
    Bq = sums(Xb, sb.w.to_numpy(float), cl_b)
    d2 = crossfit(A, Bq, rng)
    bd = []
    if is_nh:
        s_of = df.groupby("cluster").stratum.first()
        st_a, st_b = s_of.loc[A[0]].to_numpy(), s_of.loc[Bq[0]].to_numpy()
        allc = np.union1d(A[0], Bq[0])
        st_all = s_of.loc[allc].to_numpy()
    for _ in range(B_XFIT):
        if is_nh:
            m = boot_mult(len(allc), rng, st_all)
            mm = pd.Series(m, index=allc)
            ma, mb = mm.loc[A[0]].to_numpy(), mm.loc[Bq[0]].to_numpy()
        else:
            ma = np.bincount(rng.integers(0, len(A[0]), len(A[0])), minlength=len(A[0])).astype(float)
            mb = np.bincount(rng.integers(0, len(Bq[0]), len(Bq[0])), minlength=len(Bq[0])).astype(float)
        bd.append(crossfit(A, Bq, rng, ma, mb, ))
    root = lambda v: float(np.sign(v) * np.sqrt(abs(v)))
    return dict(population=label, contrast=cname, n_a=len(sa), n_b=len(sb),
                units_a=len(np.unique(unit_a)), units_b=len(np.unique(unit_b)),
                matched_distance=obs, floor_p95=fl, floor_multiple=obs / fl,
                null_p95=null95, null_multiple=null95 / fl,
                null_inside_band=bool(BAND_LO <= null95 / fl <= BAND_HI),
                obs_inside_band=bool(BAND_LO <= obs / fl <= BAND_HI),
                above_null=bool(obs > null95),
                xfit_d2=d2, xfit_dist=root(d2), xfit_lo95=root(np.nanquantile(bd, 0.025)),
                xfit_hi95=root(np.nanquantile(bd, 0.975)),
                xfit_boot_valid=int(np.isfinite(bd).sum()))


def main():
    t0 = time.time()
    sim = L.load_sim(SUBSET)
    nh = L.load_nhanes()
    pops = [("NHANES 2005-2018", nh, True)] if SUBSET == "all" else []
    for mdl in L.MODELS:
        for fr in ("clinical", "narrative"):
            pops.append((f"{mdl} {fr}", sim[(sim.model_s == mdl) & (sim.prompt_condition == fr)], False))
    rows = []
    for pi, (label, df, is_nh) in enumerate(pops):
        for ci, (cname, col, ga, gb) in enumerate(CONTRASTS):
            rng = np.random.default_rng([SEED, pi, ci])
            r = run_contrast(label, df, cname, col, ga, gb, is_nh, rng)
            rows.append(r)
            print(f"[{time.time() - t0:5.0f}s] {label:28s} {cname:22s} mult {r['floor_multiple']:.2f}  "
                  f"null mult {r['null_multiple']:.2f}  above null {r['above_null']!s:5s}  "
                  f"xfit {r['xfit_dist']:.3f} [{r['xfit_lo95']:.3f}, {r['xfit_hi95']:.3f}]", flush=True)
    out = pd.DataFrame(rows)
    if SUBSET != "all":
        ref = pd.read_csv(os.path.join(L.OUT, "l4_floor_null.csv"))
        out = pd.concat([ref[ref.population.str.startswith("NHANES")], out], ignore_index=True)
    out.to_csv(os.path.join(L.OUT, f"l4_floor_null{SUF}.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
