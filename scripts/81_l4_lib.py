"""Shared machinery for the level-4 rebuild (scripts 81a onward). Not run on its own.

Unit of analysis. One simulated draw is one simulated respondent. A model's simulated population is
every valid draw it produced in one framing (120 personas x 30 draws). Its item correlations are
computed across all of those draws, which is the counterpart of NHANES correlations computed across
all respondents. Both matrices mix a between-cell part (demographic cell or persona) with a
within-cell part; 81b reports the split in each source so the reader can see whether the two
populations get their structure from the same place. Inference respects the clustering: personas
are resampled for the corpus and PSUs within strata for NHANES.

Everything item-level is built from cluster-level pairwise contingency tables (28 item pairs x 16
cells per cluster). A bootstrap replicate is a weighted sum of cluster tables, so polychoric and
Pearson matrices for thousands of replicates cost only the correlation step.

NHANES coding follows script 78a (the level-2 rebuild) so the rungs share one population frame:
2005-2018 (DEMO_D..J, DPQ_D..J), age 18+, all eight PHQ items in 0..3, WTMEC2YR > 0, weight / 7.
Race: White RIDRETH1 3, Black 4, Hispanic 1 or 2, Asian RIDRETH3 6 (2011-2018 only), everything
else Other. Sex RIAGENDR. Income INDFMPIR < 1.3 Low, 1.3-3.5 Middle, > 3.5 High. Relationship:
Married DMDMARTL 1; Single DMDMARTL 2-5 or an unasked 18-19-year-old; cohabiting left out.
"""
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(BASE, "analysis", "brm")
RAW = os.path.join(BASE, "data", "nhanes_raw")
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
NH_ITEMS = [f"DPQ0{i}0" for i in range(1, 9)]
CYCLES = list("DEFGHIJ")
P = 8
PAIRS = [(i, j) for i in range(P) for j in range(i + 1, P)]
SHORT = {"deepseek/deepseek-chat-v3": "DeepSeek-V3", "google/gemini-3-flash-preview": "Gemini-3-Flash",
         "openai/gpt-4o-mini": "GPT-4o-mini", "z-ai/glm-4.7": "GLM-4.7"}
MODELS = ["DeepSeek-V3", "Gemini-3-Flash", "GLM-4.7", "GPT-4o-mini"]
DEC_SOURCES = {"dec28_main", "dec28_restored"}


# ------------------------------------------------------------------------------------------ data
def load_sim(subset="all"):
    """Valid rows of corpus v3. subset='dec' keeps clinical rows from the December run only."""
    m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v3.csv"), low_memory=False)
    m = m[m.phq8_valid == True].copy()  # noqa: E712  (drops the one invalid DeepSeek row)
    if subset == "dec":
        keep = (m.prompt_condition == "narrative") | m.row_source.isin(DEC_SOURCES)
        m = m[keep].copy()
    m["model_s"] = m.model.map(SHORT)
    m["sex"] = m.gender.map({"Cisgender Woman": "F", "Cisgender Man": "M"})  # cis frame only
    m["ses"] = m.ses_normalized
    m["rel"] = m.relationship
    m["cluster"] = m.profile_id
    m[ITEMS] = m[ITEMS].astype(int)
    m["w"] = 1.0
    return m.reset_index(drop=True)


def load_nhanes():
    fr = []
    for c in CYCLES:
        demo = pd.read_sas(os.path.join(RAW, f"DEMO_{c}.xpt"), format="xport")
        keep = [k for k in ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3", "INDFMPIR",
                            "WTMEC2YR", "SDMVSTRA", "SDMVPSU", "DMDMARTL"] if k in demo.columns]
        dpq = pd.read_sas(os.path.join(RAW, f"DPQ_{c}.xpt"), format="xport")
        x = demo[keep].merge(dpq[["SEQN"] + NH_ITEMS], on="SEQN")
        x["cycle"] = c
        fr.append(x)
    d = pd.concat(fr, ignore_index=True)
    if "RIDRETH3" not in d:
        d["RIDRETH3"] = np.nan
    d[NH_ITEMS] = d[NH_ITEMS].where(d[NH_ITEMS] <= 3)
    d = d[(d.RIDAGEYR >= 18) & d[NH_ITEMS].notna().all(axis=1) & (d.WTMEC2YR > 0)].copy()
    d = d.rename(columns=dict(zip(NH_ITEMS, ITEMS)))
    d[ITEMS] = d[ITEMS].round().astype(int)
    d["w"] = d.WTMEC2YR / len(CYCLES)
    d["race"] = np.select([d.RIDRETH3 == 6, d.RIDRETH1 == 3, d.RIDRETH1 == 4, d.RIDRETH1.isin([1, 2])],
                          ["Asian", "White", "Black", "Hispanic"], "Other")
    d["sex"] = d.RIAGENDR.map({1: "M", 2: "F"})
    d["ses"] = np.select([d.INDFMPIR < 1.3, d.INDFMPIR <= 3.5, d.INDFMPIR > 3.5],
                         ["Low", "Middle", "High"], "")
    d.loc[d.INDFMPIR.isna(), "ses"] = ""
    mm = d.DMDMARTL
    young_na = mm.isna() & (d.RIDAGEYR < 20)
    d["rel"] = np.select([mm == 1, mm.isin([2, 3, 4, 5]) | young_na], ["Married", "Single"], "")
    d["stratum"] = d.SDMVSTRA.astype(int)
    d["cluster"] = d.SDMVSTRA.astype(int) * 10 + d.SDMVPSU.astype(int)
    return d.reset_index(drop=True)


# ------------------------------------------------------------------------- cluster-level tables
class Tables:
    """Weighted pairwise 4x4 tables per cluster, plus first and second moments per cluster.

    tab[c, k, a*4+b]  weighted count of (item i = a, item j = b) for pair k = (i, j) in cluster c
    s0[c]             weighted n;  s1[c, i] weighted sum;  s2[c, i, j] weighted cross-product sum
    n0[c]             unweighted n
    """

    def __init__(self, X, w, cluster):
        X = np.asarray(X, int)
        w = np.asarray(w, float)
        self.ids, cidx = np.unique(np.asarray(cluster), return_inverse=True)
        C = len(self.ids)
        self.C = C
        tab = np.zeros((C, len(PAIRS), 16))
        for k, (i, j) in enumerate(PAIRS):
            code = X[:, i] * 4 + X[:, j]
            tab[:, k, :] = np.bincount(cidx * 16 + code, weights=w, minlength=C * 16).reshape(C, 16)
        self.tab = tab
        self.s0 = np.bincount(cidx, weights=w, minlength=C)
        self.n0 = np.bincount(cidx, minlength=C).astype(float)
        self.s1 = np.stack([np.bincount(cidx, weights=w * X[:, i], minlength=C) for i in range(P)], 1)
        s2 = np.zeros((C, P, P))
        for i in range(P):
            for j in range(i, P):
                v = np.bincount(cidx, weights=w * X[:, i] * X[:, j], minlength=C)
                s2[:, i, j] = v
                s2[:, j, i] = v
        self.s2 = s2

    def agg(self, mult=None):
        """Aggregate over clusters with multiplicities (bootstrap counts) or all ones."""
        if mult is None:
            mult = np.ones(self.C)
        return dict(tab=np.tensordot(mult, self.tab, 1), s0=mult @ self.s0, n0=mult @ self.n0,
                    s1=mult @ self.s1, s2=np.tensordot(mult, self.s2, 1))


def moments(a):
    """Weighted mean vector and ML covariance matrix from aggregated sums."""
    mu = a["s1"] / a["s0"]
    S = a["s2"] / a["s0"] - np.outer(mu, mu)
    return mu, S


def pearson_from(a):
    mu, S = moments(a)
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    return S / np.outer(d, d)


# -------------------------------------------------------------------------------- polychorics
_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)


def _bvn(h, k, r):
    h = np.asarray(h, float)[:, None]
    k = np.asarray(k, float)[None, :]
    if abs(r) < 1e-12:
        return stats.norm.cdf(h) * stats.norm.cdf(k)
    t = (_GL_X * r / 2 + r / 2)[:, None, None]
    ww = (_GL_W * r / 2)[:, None, None]
    q = 1 - t ** 2
    f = np.exp(-(h ** 2 - 2 * t * h * k + k ** 2) / (2 * q)) / np.sqrt(q)
    return stats.norm.cdf(h) * stats.norm.cdf(k) + (f * ww).sum(axis=0) / (2 * np.pi)


def _cellp(ta, tb, r):
    a = np.concatenate([[-8.0], ta, [8.0]])
    b = np.concatenate([[-8.0], tb, [8.0]])
    Pm = _bvn(a, b, r)
    return np.clip(Pm[1:, 1:] - Pm[:-1, 1:] - Pm[1:, :-1] + Pm[:-1, :-1], 1e-12, 1.0)


def polychoric_table(t16):
    """Two-step ML polychoric correlation from a (weighted) 4x4 table; empty categories dropped."""
    tab = np.asarray(t16, float).reshape(4, 4)
    tab = tab[tab.sum(1) > 0][:, tab.sum(0) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    n = tab.sum()
    ta = stats.norm.ppf(np.clip(np.cumsum(tab.sum(1))[:-1] / n, 1e-8, 1 - 1e-8))
    tb = stats.norm.ppf(np.clip(np.cumsum(tab.sum(0))[:-1] / n, 1e-8, 1 - 1e-8))

    def nll(r):
        return -float((tab * np.log(_cellp(ta, tb, r))).sum())

    res = optimize.minimize_scalar(nll, bounds=(-0.995, 0.995), method="bounded",
                                   options={"xatol": 1e-5})
    return float(res.x)


def poly_from_slow(a):
    """Reference implementation, one pair at a time with Brent's method (used to check poly_from)."""
    R = np.eye(P)
    for k, (i, j) in enumerate(PAIRS):
        R[i, j] = R[j, i] = polychoric_table(a["tab"][k])
    return R


def _thresholds(tabs):
    """Row and column thresholds for K tables (K x 4 x 4), padded at +-8; empty categories give
    coincident thresholds, whose zero-probability cells carry zero counts and drop out."""
    n = tabs.sum(axis=(1, 2))
    cr = np.cumsum(tabs.sum(2), 1)[:, :-1] / n[:, None]
    cc = np.cumsum(tabs.sum(1), 1)[:, :-1] / n[:, None]
    ta = stats.norm.ppf(np.clip(cr, 1e-8, 1 - 1e-8))
    tb = stats.norm.ppf(np.clip(cc, 1e-8, 1 - 1e-8))
    pad = np.full((len(tabs), 1), 8.0)
    return np.hstack([-pad, ta, pad]), np.hstack([-pad, tb, pad])


def _loglik_grid(tabs, A, B, rg):
    """Log-likelihood of each table (K) at each candidate r (K x G) by 40-node quadrature."""
    K, G = rg.shape
    h = A[:, None, :, None]                       # K,1,5,1
    k = B[:, None, None, :]                       # K,1,1,5
    base = stats.norm.cdf(h) * stats.norm.cdf(k)  # K,1,5,5
    out = np.zeros((K, G, 5, 5))
    for x, wt in zip(_GL_X, _GL_W):
        t = (x * rg / 2 + rg / 2)[:, :, None, None]
        q = 1 - t ** 2
        out += (wt * rg / 2)[:, :, None, None] * np.exp(
            -(h ** 2 - 2 * t * h * k + k ** 2) / (2 * q)) / np.sqrt(q)
    Pm = base + out / (2 * np.pi)
    cell = Pm[:, :, 1:, 1:] - Pm[:, :, :-1, 1:] - Pm[:, :, 1:, :-1] + Pm[:, :, :-1, :-1]
    cell = np.clip(cell, 1e-12, 1.0)
    return (tabs[:, None, :, :] * np.log(cell)).sum(axis=(2, 3))


def poly_from(a):
    """All 28 polychorics at once: coarse grid, fine grid, then a parabolic step (|error| < 2e-4
    against poly_from_slow on the corpus and NHANES; checked in 81b)."""
    tabs = a["tab"].reshape(len(PAIRS), 4, 4)
    A, B = _thresholds(tabs)
    K = len(PAIRS)
    g1 = np.tile(np.linspace(-0.98, 0.98, 50), (K, 1))
    ll = _loglik_grid(tabs, A, B, g1)
    r0 = g1[np.arange(K), ll.argmax(1)]
    g2 = np.clip(r0[:, None] + np.linspace(-0.04, 0.04, 41)[None, :], -0.995, 0.995)
    ll2 = _loglik_grid(tabs, A, B, g2)
    j = np.clip(ll2.argmax(1), 1, 39)
    idx = np.arange(K)
    y0, y1, y2 = ll2[idx, j - 1], ll2[idx, j], ll2[idx, j + 1]
    den = y0 - 2 * y1 + y2
    step = g2[idx, 1] - g2[idx, 0]
    off = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0)
    r = g2[idx, j] + np.clip(off, -1, 1) * step
    R = np.eye(P)
    for kk, (i, jj) in enumerate(PAIRS):
        R[i, jj] = R[jj, i] = r[kk]
    return R


# ----------------------------------------------------------------------------- factor summaries
def one_factor_uls(R):
    """One common factor by unweighted least squares on the off-diagonal of R (minres)."""
    iu = np.triu_indices(P, 1)
    r = R[iu]

    def f(lam):
        e = r - np.outer(lam, lam)[iu]
        g = np.zeros(P)
        E = np.zeros((P, P))
        E[iu] = e
        E = E + E.T
        g = -2 * E @ lam
        return float((e ** 2).sum()), g

    vals, vecs = np.linalg.eigh(R)
    start = vecs[:, -1] * np.sqrt(max(vals[-1] - 1, 0.1) / 1.0)
    best = None
    for s0 in (start, np.full(P, 0.6), -start):
        res = optimize.minimize(f, s0, jac=True, method="L-BFGS-B", bounds=[(-0.999, 0.999)] * P)
        if best is None or res.fun < best.fun:
            best = res
    lam = best.x
    if lam.sum() < 0:
        lam = -lam
    resid = r - np.outer(lam, lam)[iu]
    srmr = float(np.sqrt(np.mean(resid ** 2)))
    sl = lam.sum()
    omega = float(sl ** 2 / (sl ** 2 + np.sum(1 - lam ** 2)))
    return lam, srmr, omega


def congruence(a, b):
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def eig_desc(R):
    return np.sort(np.linalg.eigvalsh(R))[::-1]


def general_factor(lam, ev, min_load=0.30, min_ratio=3.0):
    """Pre-stated rule: every one-factor loading >= .30 in one direction and lambda1/lambda2 >= 3."""
    return bool((lam >= min_load).all() and ev[0] / ev[1] >= min_ratio)


# -------------------------------------------------------------------------------- bootstraps
def cluster_boot_mult(C, rng, strata=None):
    """Multiplicity vector for one cluster bootstrap replicate.

    Without strata: resample C clusters with replacement (personas). With strata: the Rao-Wu
    rescaling bootstrap used for NHANES, which draws n_h - 1 PSUs with replacement in each stratum
    and scales them by n_h / (n_h - 1); with two PSUs per stratum, one PSU carries the stratum.
    """
    if strata is None:
        return np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
    mult = np.zeros(C)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        nh = len(idx)
        if nh < 2:
            mult[idx] += 1.0
            continue
        pick = rng.choice(idx, nh - 1, replace=True)
        np.add.at(mult, pick, nh / (nh - 1))
    return mult


def cluster_strata(T, stratum_of_cluster):
    """Array of stratum labels aligned with T.ids."""
    return np.array([stratum_of_cluster[c] for c in T.ids])
