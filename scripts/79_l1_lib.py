"""Shared code for the level-1 rebuild (scripts 79a to 79e). Not run on its own.

Contents
  load_corpus()          data/model_outputs_v3.csv, invalid PHQ-8 row dropped, December flag
  load_nhanes(which)     NHANES adults with all eight DPQ items in 0..3, MEC weight > 0, design vars
  GRM                    graded response model (Samejima 1969) with a N(0,1) latent trait,
                         fitted by marginal maximum likelihood on Gauss-Hermite style quadrature,
                         survey weights entering as a pseudo-likelihood (weights rescaled to sum n)
  wle()                  Warm (1989) weighted likelihood estimate of theta for each response vector
  personfit()            polytomous lz (Drasgow, Levine & Williams 1985) and lz* with the
                         Snijders (2001) correction for estimated theta, extended to polytomous
                         items by Sinharay (2016)
  raowu_weights()        Rao-Wu rescaling bootstrap over PSUs within strata
  lin_se_ratio()         Taylor-linearised design SE of a domain ratio
  cond_tail()            share of cases below the reference 5th / above the 95th percentile of lz*
                         within strata of total score, with ties split by mass (exact expectation of
                         the randomised PIT indicator), so the reference itself gives exactly 5%

Seeds are set by the calling script.
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(BASE, "analysis", "brm")
RAW = os.path.join(BASE, "data", "nhanes_raw")
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
DPQ = [f"DPQ0{i}0" for i in range(1, 9)]
MODELS = ["deepseek/deepseek-chat-v3", "google/gemini-3-flash-preview", "openai/gpt-4o-mini",
          "z-ai/glm-4.7"]
SHORT = {"deepseek/deepseek-chat-v3": "DeepSeek-V3", "google/gemini-3-flash-preview": "Gemini-3-Flash",
         "openai/gpt-4o-mini": "GPT-4o-mini", "z-ai/glm-4.7": "GLM-4.7"}
DEC_SOURCES = {"dec28_main", "dec28_restored"}
CYCLES_0518 = list("DEFGHIJ")


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load_corpus():
    m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v3.csv"), low_memory=False)
    assert m.shape[0] == 28800
    m = m[m.phq8_valid == True].copy()  # noqa: E712
    assert m.shape[0] == 28799
    m[ITEMS] = m[ITEMS].astype(int)
    assert m[ITEMS].isin([0, 1, 2, 3]).all().all()
    m["total"] = m[ITEMS].sum(axis=1)
    m["short"] = m.model.map(SHORT)
    m["framing"] = m.prompt_condition
    m["gender_group"] = np.where(m.gender.str.startswith("Transgender"), "trans", "cis")
    m["dec_only"] = (m.framing == "narrative") | m.row_source.isin(DEC_SOURCES)
    # persona id is profile_id; a cell is (model, profile_id, framing)
    m["cell"] = m.model + "|" + m.profile_id + "|" + m.framing
    return m.reset_index(drop=True)


def _read(name, cols):
    d = pd.read_sas(os.path.join(RAW, f"{name}.xpt"), format="xport")
    return d[[c for c in cols if c in d.columns]]


def load_nhanes(which="2005_2018", flow=None):
    """Adults 18+, all eight DPQ items answered 0..3, WTMEC2YR > 0.

    which = "2005_2018" pools cycles D..J with weight WTMEC2YR / 7 (NCHS guidance for pooling
    seven two-year cycles). which = "2021_2023" is cycle L alone with WTMEC2YR as released.
    flow, if a list, receives sample-flow rows.
    """
    cyc = CYCLES_0518 if which == "2005_2018" else ["L"]
    fr = []
    for c in cyc:
        demo = _read(f"DEMO_{c}", ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3",
                                   "INDFMPIR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"])
        dpq = _read(f"DPQ_{c}", ["SEQN"] + DPQ)
        d = demo.merge(dpq, on="SEQN")
        d["cycle"] = c
        n0 = len(d)
        d = d[d.RIDAGEYR >= 18]
        n1 = len(d)
        d[DPQ] = d[DPQ].where(d[DPQ] <= 3)
        d = d[d[DPQ].notna().all(axis=1)]
        n2 = len(d)
        d = d[d.WTMEC2YR > 0]
        n3 = len(d)
        if flow is not None:
            flow.append(dict(reference=which, cycle=c, dpq_records_with_demo=n0, age18plus=n1,
                             all_items_0to3=n2, mec_weight_pos=n3))
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d[DPQ] = d[DPQ].astype(int)
    d["w"] = d.WTMEC2YR / len(cyc)
    d["total"] = d[DPQ].sum(axis=1)
    d["stratum"] = d.SDMVSTRA.astype(int)
    d["psu"] = d.SDMVSTRA.astype(int) * 10 + d.SDMVPSU.astype(int)
    d["sex"] = np.where(d.RIAGENDR == 1, "men", "women")
    eth = {1: "Hispanic", 2: "Hispanic", 3: "NH White", 4: "NH Black", 5: "Other"}
    d["race_eth"] = d.RIDRETH1.map(eth)
    if "RIDRETH3" in d.columns:
        d.loc[d.RIDRETH3 == 6, "race_eth"] = "NH Asian"
    d["pir_band"] = pd.cut(d.INDFMPIR, [-1, 1.3, 3.5, 99], labels=["PIR<1.3", "PIR1.3-3.5", "PIR>3.5"],
                           right=True).astype(str)
    d.loc[d.INDFMPIR.isna(), "pir_band"] = "PIR missing"
    return d.reset_index(drop=True)


def nhanes_items(d):
    return d[DPQ].to_numpy(int)


# ----------------------------------------------------------------------------------------------
# graded response model
# ----------------------------------------------------------------------------------------------
QN = 81
QT = np.linspace(-6, 6, QN)
QW = np.exp(-0.5 * QT ** 2)
QW = QW / QW.sum()


def unpack(par):
    """par per item: log a, b1, log(b2 - b1), log(b3 - b2)."""
    p = par.reshape(8, 4)
    a = np.exp(p[:, 0])
    b1 = p[:, 1]
    b2 = b1 + np.exp(p[:, 2])
    b3 = b2 + np.exp(p[:, 3])
    return a, np.stack([b1, b2, b3], axis=1)


def pack(a, b):
    return np.column_stack([np.log(a), b[:, 0], np.log(b[:, 1] - b[:, 0]),
                            np.log(b[:, 2] - b[:, 1])]).ravel()


def cat_probs(a, b, theta):
    """Category probabilities and first two theta derivatives.

    Returns P, P1, P2 with shape (len(theta), 8, 4)."""
    th = np.asarray(theta, float)[:, None, None]
    s = expit(a[None, :, None] * (th - b[None, :, :]))          # (T, 8, 3) cumulative P(X >= k)
    ds = a[None, :, None] * s * (1 - s)
    d2s = a[None, :, None] ** 2 * s * (1 - s) * (1 - 2 * s)
    one = np.ones(s.shape[:2] + (1,))
    zero = np.zeros_like(one)
    S = np.concatenate([one, s, zero], axis=2)
    D = np.concatenate([zero, ds, zero], axis=2)
    D2 = np.concatenate([zero, d2s, zero], axis=2)
    P = S[:, :, :4] - S[:, :, 1:]
    P1 = D[:, :, :4] - D[:, :, 1:]
    P2 = D2[:, :, :4] - D2[:, :, 1:]
    return np.clip(P, 1e-300, 1.0), P1, P2


def patterns(X, w=None):
    """Collapse response vectors to unique patterns with summed weights."""
    X = np.asarray(X, int)
    key = (X * (4 ** np.arange(8))).sum(axis=1)
    uk, inv = np.unique(key, return_inverse=True)
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    f = np.bincount(inv, weights=w)
    U = np.stack([(uk // 4 ** j) % 4 for j in range(8)], axis=1)
    return U, f, inv


def _negll_grad(par, U, f):
    a, b = unpack(par)
    s = expit(a[None, :, None] * (QT[:, None, None] - b[None, :, :]))   # (Q, 8, 3)
    one = np.ones((QN, 8, 1))
    zero = np.zeros((QN, 8, 1))
    S = np.concatenate([one, s, zero], axis=2)
    P = np.clip(S[:, :, :4] - S[:, :, 1:], 1e-300, 1.0)                 # (Q, 8, 4)
    logP = np.log(P)
    LL = np.zeros((len(U), QN))
    for j in range(8):
        LL += logP[:, j, :][:, U[:, j]].T
    mx = LL.max(axis=1, keepdims=True)
    L = np.exp(LL - mx) * QW[None, :]
    marg = L.sum(axis=1)
    ll = float((f * (np.log(marg) + mx[:, 0])).sum())
    post = L / marg[:, None] * f[:, None]                                  # (npat, Q)
    grad = np.zeros((8, 4))
    for j in range(8):
        r = np.zeros((QN, 4))
        for k in range(4):
            r[:, k] = post[U[:, j] == k].sum(axis=0)
        rp = r / P[:, j, :]                                                # dO/dP_k
        gS = rp[:, 1:4] - rp[:, 0:3]                                       # dO/dS_k, k = 1..3
        sk = s[:, j, :]
        dsk = sk * (1 - sk)
        g_a = (gS * dsk * (QT[:, None] - b[j][None, :])).sum()
        g_b = -(gS * dsk).sum(axis=0) * a[j]
        e2, e3 = b[j, 1] - b[j, 0], b[j, 2] - b[j, 1]
        grad[j] = [a[j] * g_a, g_b.sum(), e2 * (g_b[1] + g_b[2]), e3 * g_b[2]]
    return -ll, -grad.ravel()


def fit_grm(X, w=None):
    """Marginal ML fit. w, if given, is rescaled to sum to n (pseudo-likelihood)."""
    X = np.asarray(X, int)
    if w is not None:
        w = np.asarray(w, float) * len(X) / np.sum(w)
    U, f, _ = patterns(X, w)
    a0 = np.full(8, 1.5)
    b0 = np.tile([0.5, 1.5, 2.5], (8, 1))
    res = minimize(_negll_grad, pack(a0, b0), args=(U, f), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=5000, gtol=1e-7, ftol=1e-13))
    a, b = unpack(res.x)
    return dict(a=a, b=b, loglik=-res.fun, converged=bool(res.success), nit=res.nit,
                message=str(res.message), n_patterns=len(U))


def expected_total_dist(a, b):
    """Model-implied distribution of the sum score under N(0,1), by recursion over items."""
    P, _, _ = cat_probs(a, b, QT)
    dist = np.zeros((QN, 25))
    dist[:, 0] = 1.0
    for j in range(8):
        new = np.zeros_like(dist)
        for k in range(4):
            new[:, k:] += dist[:, :25 - k] * P[:, j, k][:, None]
        dist = new
    return (dist * QW[:, None]).sum(axis=0)


# ----------------------------------------------------------------------------------------------
# theta estimation and person fit
# ----------------------------------------------------------------------------------------------
GRID = np.linspace(-10, 10, 2001)


def _wle_score(a, b, U, theta):
    """WLE estimating function at a per-pattern theta vector."""
    P, P1, P2 = cat_probs(a, b, theta)                   # (n, 8, 4)
    idx = np.arange(len(U))[:, None]
    jj = np.arange(8)[None, :]
    r = (P1 / P)[idx, jj, U].sum(axis=1)
    I = (P1 ** 2 / P).sum(axis=(1, 2))
    J = (P1 * P2 / P).sum(axis=(1, 2))
    return r + J / (2 * I)


def wle(a, b, U):
    """Warm WLE for each row of U (unique patterns), grid bracket then bisection."""
    U = np.asarray(U, int)
    P, P1, P2 = cat_probs(a, b, GRID)
    R = P1 / P                                              # (G, 8, 4)
    I = (P1 ** 2 / P).sum(axis=(1, 2))
    J = (P1 * P2 / P).sum(axis=(1, 2))
    corr = J / (2 * I)
    theta = np.empty(len(U))
    for s0 in range(0, len(U), 4000):
        u = U[s0:s0 + 4000]
        S = corr[None, :].repeat(len(u), 0)
        for j in range(8):
            S += R[:, j, :][:, u[:, j]].T
        sgn = S > 0
        # last grid point with S > 0 followed by S <= 0
        chg = sgn[:, :-1] & ~sgn[:, 1:]
        has = chg.any(axis=1)
        k = np.where(has, chg.shape[1] - 1 - np.argmax(chg[:, ::-1], axis=1), 0)
        lo, hi = GRID[k], GRID[k + 1]
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            sm = _wle_score(a, b, u, mid)
            lo = np.where(sm > 0, mid, lo)
            hi = np.where(sm > 0, hi, mid)
        t = 0.5 * (lo + hi)
        t = np.where(has, t, np.where(sgn[:, -1], GRID[-1], GRID[0]))
        theta[s0:s0 + 4000] = t
    return theta


def personfit(a, b, U, theta):
    """Polytomous lz and lz* (Snijders 2001 / Sinharay 2016 correction, WLE theta)."""
    U = np.asarray(U, int)
    P, P1, P2 = cat_probs(a, b, theta)
    logP = np.log(P)
    idx = np.arange(len(U))[:, None]
    jj = np.arange(8)[None, :]
    obs = logP[idx, jj, U]                                  # (n, 8)
    Ew = (P * logP).sum(axis=2)                             # (n, 8)
    Vw = (P * logP ** 2).sum(axis=2) - Ew ** 2
    num = (obs - Ew).sum(axis=1)
    lz = num / np.sqrt(Vw.sum(axis=1))
    r = P1 / P
    I = (P1 * r).sum(axis=(1, 2))
    J = (P1 * P2 / P).sum(axis=(1, 2))
    r0 = J / (2 * I)
    cn = (P1 * logP).sum(axis=(1, 2)) / I
    wt = logP - cn[:, None, None] * r
    Ewt = (P * wt).sum(axis=2)
    Vwt = (P * wt ** 2).sum(axis=2) - Ewt ** 2
    lzs = (num + cn * r0) / np.sqrt(Vwt.sum(axis=1))
    return lz, lzs


def score_vectors(a, b, X):
    """theta (WLE), lz and lz* for every row of X, computed once per unique pattern."""
    U, _, inv = patterns(X)
    th = wle(a, b, U)
    lz, lzs = personfit(a, b, U, th)
    return th[inv], lz[inv], lzs[inv]


# ----------------------------------------------------------------------------------------------
# design-based variance
# ----------------------------------------------------------------------------------------------
def raowu_mult(d, B, rng):
    """Rao-Wu rescaling bootstrap: in each stratum draw n_h - 1 PSUs with replacement from n_h and
    scale by n_h / (n_h - 1). Returns (B, n_psu) multipliers and the PSU index of every row of d.
    The PSU structure always comes from the full frame d, so subsets keep the full design."""
    psu_codes, psu_idx = np.unique(d.psu.to_numpy(), return_inverse=True)
    strat_of_psu = (psu_codes // 10).astype(int)
    mult = np.zeros((B, len(psu_codes)))
    for h in np.unique(strat_of_psu):
        members = np.where(strat_of_psu == h)[0]
        nh = len(members)
        draws = rng.integers(0, nh, size=(B, nh - 1))
        for jpos, m in enumerate(members):
            mult[:, m] = (draws == jpos).sum(axis=1) * nh / (nh - 1)
    return mult, psu_idx


def raowu_weights(d, B, rng, rows=None):
    """(B, n_rows) replicate weights for the rows selected by boolean mask `rows` (all if None)."""
    mult, psu_idx = raowu_mult(d, B, rng)
    rows = np.ones(len(d), bool) if rows is None else np.asarray(rows, bool)
    return mult[:, psu_idx[rows]] * d.w.to_numpy()[rows][None, :]


def lin_se_ratio(d, y, x, dom=None):
    """Linearised SE of R = sum(w*y*dom) / sum(w*x*dom), strata SDMVSTRA, PSUs SDMVPSU,
    domain handled by zeroing (all PSUs retained)."""
    w = d.w.to_numpy()
    dom = np.ones(len(d)) if dom is None else np.asarray(dom, float)
    y = np.asarray(y, float) * dom
    x = np.asarray(x, float) * dom
    X = (w * x).sum()
    R = (w * y).sum() / X
    z = w * (y - R * x) / X
    t = pd.DataFrame(dict(z=z, psu=d.psu.to_numpy(), h=d.stratum.to_numpy()))
    pt = t.groupby(["h", "psu"]).z.sum().reset_index()
    v = 0.0
    for _, g in pt.groupby("h"):
        nh = len(g)
        if nh > 1:
            v += nh / (nh - 1) * ((g.z - g.z.mean()) ** 2).sum()
    return R, np.sqrt(v)


# ----------------------------------------------------------------------------------------------
# conditional tail shares
# ----------------------------------------------------------------------------------------------
def total_strata(nh_total, min_n=100):
    """Map totals 0..24 to strata: exact totals, adjacent totals merged upward from the top until
    each stratum holds at least min_n unweighted NHANES respondents."""
    cnt = np.bincount(nh_total, minlength=25)
    lab = np.arange(25)
    # merge from the top down
    t = 24
    groups = []
    cur = []
    ccount = 0
    while t >= 0:
        cur.append(t)
        ccount += cnt[t]
        if ccount >= min_n:
            groups.append(cur)
            cur, ccount = [], 0
        t -= 1
    if cur:
        groups[-1].extend(cur)
    for gi, g in enumerate(groups):
        lab[g] = min(g)
    return lab


class TailRef:
    """Reference CDF of lz* within strata (e.g. total-score strata), built from NHANES values v and
    stratum labels s. For a query (stratum, value) returns F(value-) and F(value) under any set of
    reference weights, so bootstrap replicates reuse the sort."""

    def __init__(self, s_ref, v_ref):
        self.s = np.asarray(s_ref)
        self.v = np.round(np.asarray(v_ref, float), 10)
        self.order = np.lexsort((self.v, self.s))
        self.ss = self.s[self.order]
        self.vs = self.v[self.order]
        self.strata = np.unique(self.ss)
        self.start = {h: np.searchsorted(self.ss, h, "left") for h in self.strata}
        self.stop = {h: np.searchsorted(self.ss, h, "right") for h in self.strata}

    def locate(self, s_q, v_q):
        """Positions (lo, hi, st, en) in the sorted reference for each query."""
        s_q = np.asarray(s_q)
        v_q = np.round(np.asarray(v_q, float), 10)
        lo = np.empty(len(s_q), int)
        hi = np.empty(len(s_q), int)
        st = np.empty(len(s_q), int)
        en = np.empty(len(s_q), int)
        for h in np.unique(s_q):
            m = s_q == h
            a, e = self.start[h], self.stop[h]
            seg = self.vs[a:e]
            lo[m] = a + np.searchsorted(seg, v_q[m], "left")
            hi[m] = a + np.searchsorted(seg, v_q[m], "right")
            st[m] = a
            en[m] = e
        return lo, hi, st, en

    def cdf(self, loc, w_ref):
        """F(v-) and F(v) within stratum under reference weights w_ref (length of reference)."""
        lo, hi, st, en = loc
        cw = np.concatenate([[0.0], np.cumsum(np.asarray(w_ref, float)[self.order])])
        tot = cw[en] - cw[st]
        return (cw[lo] - cw[st]) / tot, (cw[hi] - cw[st]) / tot


def tail_probs(Fm, F, q=0.05):
    """Expected indicator that the randomised PIT falls below q (misfit) and above 1 - q (overfit),
    plus the mid-PIT."""
    mass = np.maximum(F - Fm, 1e-15)
    low = np.clip((q - Fm) / mass, 0, 1)
    high = np.clip((F - (1 - q)) / mass, 0, 1)
    return low, high, 0.5 * (Fm + F)


def compositions_uniform(total, n, rng, k=8, maxv=3):
    """n uniform draws from the compositions of `total` into k parts each in 0..maxv (rejection)."""
    out = np.empty((n, k), dtype=int)
    filled = 0
    while filled < n:
        cand = rng.integers(0, maxv + 1, size=(max(1000, (n - filled) * 20), k))
        ok = cand[cand.sum(axis=1) == total]
        take = min(len(ok), n - filled)
        out[filled:filled + take] = ok[:take]
        filled += take
    return out


def wquantile(v, w, q):
    o = np.argsort(v)
    cw = np.cumsum(np.asarray(w, float)[o])
    cw = cw / cw[-1]
    return np.asarray(v)[o][np.searchsorted(cw, q)]


# ----------------------------------------------------------------------------------------------
# persona-clustered bootstrap for the corpus
# ----------------------------------------------------------------------------------------------
def cluster_boot_means(values, cluster, groups, B, rng):
    """For each group label, B bootstrap means of `values` (n, k) resampling clusters within group.

    values: (n,) or (n, k) array of per-row quantities (sums are formed, then ratio of sums over
    rows). Returns dict group -> (B, k) array of means, and the point means."""
    values = np.atleast_2d(np.asarray(values, float).T).T
    out, pt = {}, {}
    for g in np.unique(groups):
        m = groups == g
        cl, inv = np.unique(cluster[m], return_inverse=True)
        sums = np.zeros((len(cl), values.shape[1]))
        np.add.at(sums, inv, values[m])
        cnt = np.bincount(inv).astype(float)
        idx = rng.integers(0, len(cl), size=(B, len(cl)))
        bs = sums[idx].sum(axis=1)
        bc = cnt[idx].sum(axis=1)
        out[g] = bs / bc[:, None]
        pt[g] = sums.sum(axis=0) / cnt.sum()
    return out, pt


def cluster_resample_index(cluster, B, rng):
    """(B, n_clusters) resample of cluster positions and the row->cluster map."""
    cl, inv = np.unique(cluster, return_inverse=True)
    return rng.integers(0, len(cl), size=(B, len(cl))), inv, len(cl)


def verdict(lo, hi, tol):
    """R3-style reading of a ratio interval (lo, hi) against [1/tol, tol].

    Regions: 'too prototypical' below 1/tol, 'equivalent' inside, 'less coherent' above tol. The
    caller flips the labels where a high ratio means more prototypical."""
    L, U = 1.0 / tol, tol
    if lo >= L and hi <= U:
        return "equivalent"
    if hi < L:
        return "below"
    if lo > U:
        return "above"
    if lo < L <= hi <= U:
        return "below or equivalent"
    if L <= lo <= U < hi:
        return "equivalent or above"
    return "undetermined"
