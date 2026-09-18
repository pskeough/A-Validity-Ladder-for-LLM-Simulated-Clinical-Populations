"""The within-cohort factor fit again, on the estimator a psychometrician would have used.

Section 9 fits one maximum-likelihood factor to Pearson correlations of four-category items and
concedes in the text that this is not the right estimator and that a weighted least squares variant
would fit everything better. A concession is not a result. Reviewers asked three times for the
comparison to be run, and the objection is a real one: Pearson correlations between coarse ordinal
items attenuate, attenuation inflates residuals, and SRMR is computed from residuals. If the
simulated cohorts fail 0.08 only because of attenuation, the section's relative claim dissolves.

So this refits everything on polychoric correlations by diagonally weighted least squares, which is
the estimation half of WLSMV. The other half, the mean- and variance-adjusted chi-square, corrects a
test statistic; nothing here reports one, since SRMR is a descriptive residual, so it does not enter
and we do not claim to have run WLSMV's inferential machinery. Two honest limits on the weights:
they are the two-stage asymptotic variances of each polychoric coefficient with thresholds held at
their marginal estimates, so uncertainty in the thresholds is not propagated into them, and they
assume the items are independent draws, which for the simulated cohorts they are not (30 iterations
per design cell). Both affect the weighting of the fit and neither affects SRMR much, because SRMR
is a function of the residual correlations rather than of the weights.

Everything is run on both sides, NHANES and simulated, under one specification, exactly as the ML
version was.

Emits analysis/ordinal_factor.csv.
"""
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(BASE, "data", "nhanes_raw")
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
NH_ITEMS = [f"DPQ0{i}0" for i in range(1, 9)]
CYCLES = list("DEFGHIJ")

# Gauss-Legendre nodes for the bivariate normal integral below. Forty nodes puts the quadrature
# error well under the sampling error of any correlation estimated here.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)


def bvn(h, k, r):
    """P(X <= h, Y <= k) for standard bivariate normal with correlation r, vectorised over h, k.

    Uses the integral of the density's derivative with respect to the correlation, which turns a
    two-dimensional orthant probability into a one-dimensional quadrature.
    """
    h = np.asarray(h, float)[:, None]
    k = np.asarray(k, float)[None, :]
    if abs(r) < 1e-12:
        return stats.norm.cdf(h) * stats.norm.cdf(k)
    t = (_GL_X * r / 2 + r / 2)[:, None, None]
    w = (_GL_W * r / 2)[:, None, None]
    q = 1 - t ** 2
    f = np.exp(-(h ** 2 - 2 * t * h * k + k ** 2) / (2 * q)) / np.sqrt(q)
    return stats.norm.cdf(h) * stats.norm.cdf(k) + (f * w).sum(axis=0) / (2 * np.pi)


def _cells(ta, tb, r):
    """Model cell probabilities of the contingency table implied by thresholds ta, tb and r."""
    # +-8 rather than +-inf: the quadrand carries h^2 - 2thk + k^2, which is inf - inf at an
    # infinite threshold and returns nan. Phi(8) differs from 1 by 6e-16, well below anything here.
    a = np.concatenate([[-8.0], ta, [8.0]])
    b = np.concatenate([[-8.0], tb, [8.0]])
    P = bvn(a, b, r)
    return np.clip(P[1:, 1:] - P[:-1, 1:] - P[1:, :-1] + P[:-1, :-1], 1e-12, 1.0)


def polychoric(x, y):
    """Two-stage polychoric correlation with its asymptotic variance (thresholds held fixed)."""
    cx, cy = np.unique(x), np.unique(y)
    n = len(x)
    tab = np.zeros((len(cx), len(cy)))
    for i, u in enumerate(cx):
        for j, v in enumerate(cy):
            tab[i, j] = np.sum((x == u) & (y == v))
    ta = stats.norm.ppf(np.clip(np.cumsum(tab.sum(1))[:-1] / n, 1e-8, 1 - 1e-8))
    tb = stats.norm.ppf(np.clip(np.cumsum(tab.sum(0))[:-1] / n, 1e-8, 1 - 1e-8))

    def nll(r):
        return -float((tab * np.log(_cells(ta, tb, r))).sum())

    res = optimize.minimize_scalar(nll, bounds=(-0.95, 0.95), method="bounded",
                                   options={"xatol": 1e-6})
    r = float(res.x)
    d = 1e-4
    dp = (_cells(ta, tb, min(r + d, 0.9999)) - _cells(ta, tb, max(r - d, -0.9999))) / (2 * d)
    info = n * float((dp ** 2 / _cells(ta, tb, r)).sum())
    return r, 1.0 / max(info, 1e-8)


def poly_matrix(X):
    p = X.shape[1]
    R, V = np.eye(p), np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            r, v = polychoric(X[:, i], X[:, j])
            R[i, j] = R[j, i] = r
            V[i, j] = V[j, i] = v
    return R, V


def dwls_one_factor(R, V):
    """One common factor fitted to R by DWLS on the off-diagonal residuals."""
    iu = np.triu_indices(len(R), k=1)
    w = 1.0 / np.clip(V[iu], 1e-10, None)

    def obj(lam):
        return float((w * (R[iu] - np.outer(lam, lam)[iu]) ** 2).sum())

    best = None
    for start in (0.6, 0.4, 0.8):
        res = optimize.minimize(obj, np.full(len(R), start), method="L-BFGS-B",
                                bounds=[(-0.99, 0.99)] * len(R))
        if best is None or res.fun < best.fun:
            best = res
    lam = best.x
    if lam.sum() < 0:
        lam = -lam
    srmr = float(np.sqrt(np.mean((R[iu] - np.outer(lam, lam)[iu]) ** 2)))
    return lam, srmr, float(np.mean(lam ** 2))


def ml_pearson(X):
    """The Section 9 fit, reproduced here so both estimators are read off the same subsample."""
    from sklearn.decomposition import FactorAnalysis
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    lam = FactorAnalysis(n_components=1, random_state=0).fit(Z).components_[0]
    if lam.sum() < 0:
        lam = -lam
    R = np.corrcoef(Z, rowvar=False)
    iu = np.triu_indices(len(R), k=1)
    return float(np.sqrt(np.mean((R[iu] - np.outer(lam, lam)[iu]) ** 2)))


def fit(X, source, dimension, cohort, model="POOLED"):
    R, V = poly_matrix(X)
    lam, srmr, var = dwls_one_factor(R, V)
    row = dict(source=source, model=model, dimension=dimension, cohort=cohort, n=len(X),
               srmr_dwls_polychoric=round(srmr, 4), srmr_ml_pearson=round(ml_pearson(X), 4),
               good_fit=bool(srmr < 0.08), mean_loading=round(float(lam.mean()), 4),
               pct_variance=round(var * 100, 2))
    print(f"  {model[:22]:23s} {cohort[:26]:27s} n={len(X):6d}  "
          f"ML/Pearson {row['srmr_ml_pearson']:.4f}  DWLS/polychoric {srmr:.4f}"
          f"{'  <- clears .08' if srmr < 0.08 else ''}")
    return row


# ---- NHANES reference, same rows as script 31 -------------------------------------------------
frames = []
for cy in CYCLES:
    demo = pd.read_sas(os.path.join(RAW, f"DEMO_{cy}.xpt"))[["SEQN", "RIDAGEYR", "WTMEC2YR"]]
    frames.append(demo.merge(pd.read_sas(os.path.join(RAW, f"DPQ_{cy}.xpt")), on="SEQN"))
d = pd.concat(frames, ignore_index=True)
d[NH_ITEMS] = d[NH_ITEMS].where(d[NH_ITEMS] <= 3)
d = d[(d.RIDAGEYR >= 18) & d[NH_ITEMS].notna().all(axis=1) & (d.WTMEC2YR > 0)]

print("population reference:")
rows = [fit(np.rint(d[NH_ITEMS].to_numpy(float)).astype(int), "NHANES adults", "population",
            "All adults 18+")]

# ---- simulated cohorts, pooled and within each model ------------------------------------------
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
for col in ["race", "gender", "ses", "model"]:
    m[col] = m[col].astype(str).str.strip().str.strip('"')
m = m[m[ITEMS].notna().all(axis=1)]
COHORTS = ([("race", r) for r in sorted(m.race.unique())]
           + [("gender", g) for g in sorted(m.gender.unique())]
           + [("ses", s) for s in sorted(m.ses.unique())])

print("\nsimulated cohorts, four models pooled:")
for dim, lev in COHORTS:
    g = m[m[dim] == lev]
    rows.append(fit(g[ITEMS].to_numpy(int), "simulated", dim, lev))

print("\nsimulated cohorts, within model:")
for mod, gm in m.groupby("model"):
    for dim, lev in COHORTS:
        g = gm[gm[dim] == lev]
        if len(g) > 400:
            rows.append(fit(g[ITEMS].to_numpy(int), "simulated", dim, lev, model=mod))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "ordinal_factor.csv"), index=False)

ref = res.iloc[0]
pool = res[(res.source == "simulated") & (res.model == "POOLED")]
print(f"\npopulation: ML/Pearson {ref.srmr_ml_pearson:.4f} -> DWLS/polychoric "
      f"{ref.srmr_dwls_polychoric:.4f}")
print(f"pooled simulated cohorts: ML/Pearson {pool.srmr_ml_pearson.min():.4f}-"
      f"{pool.srmr_ml_pearson.max():.4f} -> DWLS/polychoric "
      f"{pool.srmr_dwls_polychoric.min():.4f}-{pool.srmr_dwls_polychoric.max():.4f}; "
      f"{int(pool.good_fit.sum())} of {len(pool)} below .08")
for mod, g in res[(res.source == "simulated") & (res.model != "POOLED")].groupby("model"):
    print(f"  {mod:34s} {int(g.good_fit.sum()):2d}/{len(g)} clear .08   "
          f"DWLS {g.srmr_dwls_polychoric.min():.3f}-{g.srmr_dwls_polychoric.max():.3f} "
          f"(ML {g.srmr_ml_pearson.min():.3f}-{g.srmr_ml_pearson.max():.3f})")
