"""Instrument-general ladder: the gate, level 1 and level 4 (R1, R2, R3) for any scale of J items
with M ordered response categories, scored 0..M-1 after reverse-keying.

Why. The audit code (79_l1_lib, 81_l4_lib, 82_gate_lib, 83_controls_lib) is written for the PHQ-8:
eight items, four categories, NHANES design weights. Running the ladder on another group's release
(the PersonaLLM shakedown, script 92) needs the same statistics with J, M and the reference's
sampling scheme as inputs. This file restates them in that form. Script 91a checks that, on the
PHQ-8 corpus and NHANES, every function here returns the audit library's numbers.

Contents
  gate     one-facet G-study (82_gate_lib.one_facet, phi_k, k_for_phi, k_for_se, reused);
           tolerances in reference SD units (gate_tolerances).
  L1       graded response model (marginal ML, Gauss-Hermite on a 81-point grid as 79_l1_lib),
           WLE theta, lz and lz* (Snijders / Sinharay), total-matched tail ratios with a persona
           bootstrap of the simulation jointly with a bootstrap of the reference (iid persons, or a
           caller-supplied replicate-weight function for a survey), verdict by 79c.read_verdict.
           tau_from_subgroups: the paper's rule for tau from reference subgroups.
  L4       weighted pairwise MxM tables per cluster, two-step polychorics on a grid with a
           parabolic step (81_l4_lib.poly_from), one-factor ULS, R1 rule, Tucker congruence, R2 with
           persona-bootstrap x reference-bootstrap pairing, R3 within-cluster factor.
"""
import importlib.util
import os
import sys

import numpy as np
from scipy import optimize, stats
from scipy.optimize import minimize
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = [old[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


GL = _load("gl82", "82_gate_lib.py")
PF = _load("l1pf", "79c_l1_personfit.py")

# ============================================================================================ gate
# The gate reads precision: SE(k) <= 0.25 reference SD (minimum) or 0.125 SD (recommended). phi(k)
# is persona separation; it is required (>= .80 / .90) only for uses that compare individual
# personas with each other, and is reported beside every gate verdict.
GATE_MIN = (0.80, 0.25)      # (separation phi for persona-comparison uses, SE tolerance in reference SD)
GATE_REC = (0.90, 0.125)


def gate_tolerances(sd_ref):
    """SE tolerances in score points. For the PHQ-8 (NHANES SD 3.94) these are 0.98 and 0.49,
    the 1.0 and 0.5 points the paper states."""
    return GATE_MIN[1] * sd_ref, GATE_REC[1] * sd_ref


def gate(score, persona, k, sd_ref):
    """One-facet G-study over personas for one model x framing. score: per-draw scale score."""
    persona = np.asarray(persona)
    score = np.asarray(score, float)
    _, inv = np.unique(persona, return_inverse=True)
    n = np.bincount(inv).astype(float)
    s1 = np.bincount(inv, weights=score)
    s2 = np.bincount(inv, weights=score ** 2)
    mean = s1 / n
    ss = s2 - n * mean ** 2
    c = GL.one_facet(mean, n, ss)
    t_min, t_rec = gate_tolerances(sd_ref)
    se_k = float(np.sqrt(GL.pos(c["e"]) / k))
    phi = GL.phi_k(c["p"], c["e"], k)
    return dict(s2p=c["p"], s2e=c["e"], n_personas=c["n_cells"], k=k, phi_k=phi, se_k=se_k,
                phi_1=GL.phi_k(c["p"], c["e"], 1), sd_ref=sd_ref, tol_min=t_min, tol_rec=t_rec,
                k_min=GL.k_for_se(c["e"], t_min), k_rec=GL.k_for_se(c["e"], t_rec),
                pass_min=bool(se_k <= t_min), pass_rec=bool(se_k <= t_rec),
                pass_single=bool(np.sqrt(GL.pos(c["e"])) <= t_min),
                k_sep_min=GL.k_for_phi(c["p"], c["e"], GATE_MIN[0]), k_sep_rec=GL.k_for_phi(c["p"], c["e"], GATE_REC[0]),
                sep_min=bool(phi >= GATE_MIN[0]), sep_rec=bool(phi >= GATE_REC[0]))


# ============================================================================================== L1
QN = 81
QT = np.linspace(-6, 6, QN)
QW = np.exp(-0.5 * QT ** 2)
QW = QW / QW.sum()
GRID = np.linspace(-10, 10, 2001)


class GRM:
    """Graded response model for J items with M categories."""

    def __init__(self, J, M):
        self.J, self.M = J, M

    # parameters per item: log a, b1, log(b2 - b1), ..., log(b_{M-1} - b_{M-2})
    def unpack(self, par):
        p = par.reshape(self.J, self.M)
        a = np.exp(p[:, 0])
        b = np.cumsum(np.column_stack([p[:, 1], np.exp(p[:, 2:])]), axis=1)
        return a, b

    def pack(self, a, b):
        return np.column_stack([np.log(a), b[:, 0], np.log(np.diff(b, axis=1))]).ravel()

    def cat_probs(self, a, b, theta):
        """P, P', P'' with shape (len(theta), J, M)."""
        th = np.asarray(theta, float)[:, None, None]
        s = expit(a[None, :, None] * (th - b[None, :, :]))
        ds = a[None, :, None] * s * (1 - s)
        d2s = a[None, :, None] ** 2 * s * (1 - s) * (1 - 2 * s)
        one = np.ones(s.shape[:2] + (1,))
        zero = np.zeros_like(one)
        S = np.concatenate([one, s, zero], axis=2)
        D = np.concatenate([zero, ds, zero], axis=2)
        D2 = np.concatenate([zero, d2s, zero], axis=2)
        M = self.M
        return np.clip(S[:, :, :M] - S[:, :, 1:], 1e-300, 1.0), D[:, :, :M] - D[:, :, 1:], D2[:, :, :M] - D2[:, :, 1:]

    def patterns(self, X, w=None):
        X = np.asarray(X, np.int64)
        key = (X * (self.M ** np.arange(self.J, dtype=np.int64))).sum(axis=1)
        uk, inv = np.unique(key, return_inverse=True)
        w = np.ones(len(X)) if w is None else np.asarray(w, float)
        f = np.bincount(inv, weights=w)
        U = np.stack([(uk // self.M ** j) % self.M for j in range(self.J)], axis=1)
        return U, f, inv

    def _negll_grad(self, par, U, f):
        J, M = self.J, self.M
        a, b = self.unpack(par)
        s = expit(a[None, :, None] * (QT[:, None, None] - b[None, :, :]))     # (Q, J, M-1)
        S = np.concatenate([np.ones((QN, J, 1)), s, np.zeros((QN, J, 1))], axis=2)
        P = np.clip(S[:, :, :M] - S[:, :, 1:], 1e-300, 1.0)
        logP = np.log(P)
        LL = np.zeros((len(U), QN))
        for j in range(J):
            LL += logP[:, j, :][:, U[:, j]].T
        mx = LL.max(axis=1, keepdims=True)
        L = np.exp(LL - mx) * QW[None, :]
        marg = L.sum(axis=1)
        ll = float((f * (np.log(marg) + mx[:, 0])).sum())
        post = L / marg[:, None] * f[:, None]
        grad = np.zeros((J, M))
        for j in range(J):
            r = np.zeros((QN, M))
            for k in range(M):
                r[:, k] = post[U[:, j] == k].sum(axis=0)
            rp = r / P[:, j, :]
            gS = rp[:, 1:M] - rp[:, 0:M - 1]
            sk = s[:, j, :]
            dsk = sk * (1 - sk)
            g_a = (gS * dsk * (QT[:, None] - b[j][None, :])).sum()
            g_b = -(gS * dsk).sum(axis=0) * a[j]                               # d/d b_k, k = 1..M-1
            e = np.diff(b[j])
            tail = np.cumsum(g_b[::-1])[::-1]                                  # sum_{k >= m} g_b[k]
            grad[j] = np.concatenate([[a[j] * g_a, g_b.sum()], e * tail[1:]])
        return -ll, -grad.ravel()

    def fit(self, X, w=None):
        X = np.asarray(X, int)
        if w is not None:
            w = np.asarray(w, float) * len(X) / np.sum(w)
        U, f, _ = self.patterns(X, w)
        a0 = np.full(self.J, 1.5)
        b0 = np.tile(np.linspace(-1.5, 1.5, self.M - 1) if self.M > 4 else np.array([0.5, 1.5, 2.5])[:self.M - 1],
                     (self.J, 1))
        res = minimize(self._negll_grad, self.pack(a0, b0), args=(U, f), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=5000, gtol=1e-7, ftol=1e-13))
        a, b = self.unpack(res.x)
        return dict(a=a, b=b, loglik=-res.fun, converged=bool(res.success), nit=res.nit, n_patterns=len(U))

    def _wle_score(self, a, b, U, theta):
        P, P1, P2 = self.cat_probs(a, b, theta)
        idx = np.arange(len(U))[:, None]
        jj = np.arange(self.J)[None, :]
        r = (P1 / P)[idx, jj, U].sum(axis=1)
        I = (P1 ** 2 / P).sum(axis=(1, 2))
        Jt = (P1 * P2 / P).sum(axis=(1, 2))
        return r + Jt / (2 * I)

    def wle(self, a, b, U):
        U = np.asarray(U, int)
        P, P1, P2 = self.cat_probs(a, b, GRID)
        R = P1 / P
        I = (P1 ** 2 / P).sum(axis=(1, 2))
        Jt = (P1 * P2 / P).sum(axis=(1, 2))
        corr = Jt / (2 * I)
        theta = np.empty(len(U))
        for s0 in range(0, len(U), 4000):
            u = U[s0:s0 + 4000]
            S = corr[None, :].repeat(len(u), 0)
            for j in range(self.J):
                S += R[:, j, :][:, u[:, j]].T
            sgn = S > 0
            chg = sgn[:, :-1] & ~sgn[:, 1:]
            has = chg.any(axis=1)
            k = np.where(has, chg.shape[1] - 1 - np.argmax(chg[:, ::-1], axis=1), 0)
            lo, hi = GRID[k], GRID[k + 1]
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                sm = self._wle_score(a, b, u, mid)
                lo = np.where(sm > 0, mid, lo)
                hi = np.where(sm > 0, hi, mid)
            t = 0.5 * (lo + hi)
            theta[s0:s0 + 4000] = np.where(has, t, np.where(sgn[:, -1], GRID[-1], GRID[0]))
        return theta

    def personfit(self, a, b, U, theta):
        U = np.asarray(U, int)
        P, P1, P2 = self.cat_probs(a, b, theta)
        logP = np.log(P)
        idx = np.arange(len(U))[:, None]
        jj = np.arange(self.J)[None, :]
        obs = logP[idx, jj, U]
        Ew = (P * logP).sum(axis=2)
        Vw = (P * logP ** 2).sum(axis=2) - Ew ** 2
        num = (obs - Ew).sum(axis=1)
        lz = num / np.sqrt(Vw.sum(axis=1))
        r = P1 / P
        I = (P1 * r).sum(axis=(1, 2))
        Jt = (P1 * P2 / P).sum(axis=(1, 2))
        r0 = Jt / (2 * I)
        cn = (P1 * logP).sum(axis=(1, 2)) / I
        wt = logP - cn[:, None, None] * r
        Ewt = (P * wt).sum(axis=2)
        Vwt = (P * wt ** 2).sum(axis=2) - Ewt ** 2
        lzs = (num + cn * r0) / np.sqrt(Vwt.sum(axis=1))
        return lz, lzs

    def score(self, a, b, X):
        """theta (WLE), lz, lz* for every row of X."""
        U, _, inv = self.patterns(X)
        th = self.wle(a, b, U)
        lz, lzs = self.personfit(a, b, U, th)
        return th[inv], lz[inv], lzs[inv]


def total_strata(ref_total, max_total, min_n=100):
    """79_l1_lib.total_strata for totals 0..max_total: exact totals, merged downward from the top
    until each stratum holds min_n unweighted reference respondents."""
    cnt = np.bincount(ref_total, minlength=max_total + 1)
    lab = np.arange(max_total + 1)
    groups, cur, cc = [], [], 0
    for t in range(max_total, -1, -1):
        cur.append(t)
        cc += cnt[t]
        if cc >= min_n:
            groups.append(cur)
            cur, cc = [], 0
    if cur:
        groups[-1].extend(cur)
    for g in groups:
        lab[g] = min(g)
    return lab


class L1Ref:
    """Within-total reference CDF of lz* with B bootstrap reweightings of the reference.

    wrep: None for an iid sample (multinomial person bootstrap), or a function b -> weights."""

    def __init__(self, total, lz, w, max_total, B, rng, wrep=None, min_n=100):
        L1 = _load("l1lib", "79_l1_lib.py")
        self.L1 = L1
        self.smap = total_strata(np.asarray(total, int), max_total, min_n)
        self.cond = L1.TailRef(self.smap[np.asarray(total, int)], lz)
        self.w = np.ones(len(total)) if w is None else np.asarray(w, float)
        self.B = B
        if wrep is None:
            n = len(total)
            self.mult = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)
            self._wrep = lambda b: self.mult[b] * self.w
        else:
            self._wrep = wrep

    def wrep(self, b):
        return self.w if b is None else self._wrep(b)


def l1_eval(ref, total, lz, persona, rng, tau=1.5, B=None):
    """83_controls_lib.l1_eval with the reference passed in: matched misfit / overfit ratios with
    90% intervals (persona bootstrap paired with reference replicates) and the verdict."""
    L1 = ref.L1
    B = ref.B if B is None else B
    _, persona = np.unique(np.asarray(persona), return_inverse=True)
    loc = ref.cond.locate(ref.smap[np.asarray(total, int)], lz)
    Fm, F = ref.cond.cdf(loc, ref.w)
    cl, ch, _ = L1.tail_probs(Fm, F)
    npers = persona.max() + 1
    cnt = np.bincount(persona, minlength=npers).astype(float)
    ridx = rng.integers(0, npers, size=(B, npers))
    bl, bh = np.empty(B), np.empty(B)
    for b in range(B):
        Fm, F = ref.cond.cdf(loc, ref.wrep(b))
        l, h, _ = L1.tail_probs(Fm, F)
        sl = np.bincount(persona, weights=l, minlength=npers)
        sh = np.bincount(persona, weights=h, minlength=npers)
        den = cnt[ridx[b]].sum()
        bl[b], bh[b] = sl[ridx[b]].sum() / den, sh[ridx[b]].sum() / den
    r = dict(misfit_ratio=cl.mean() / .05, misfit_ratio_ci90_lo=np.quantile(bl, .05) / .05,
             misfit_ratio_ci90_hi=np.quantile(bl, .95) / .05, overfit_ratio=ch.mean() / .05,
             overfit_ratio_ci90_lo=np.quantile(bh, .05) / .05, overfit_ratio_ci90_hi=np.quantile(bh, .95) / .05)
    vm, vo, overall = PF.read_verdict(r, tau)
    r.update(misfit_verdict=vm, overfit_verdict=vo, verdict=overall, passed=overall == "pass", tau=tau)
    return r


TAUS = (1.10, 1.25, 1.50, 2.00, 3.00)
TAU_FLOOR = 1.50   # real respondents drawn through the corpus design pass at 1.5 and not at 1.25 (83c_l1.csv)


def _resolved(lo, hi):
    """Fold departure from 1 that a 90% interval [lo, hi] of a ratio establishes: lo when the
    interval lies above 1, 1/hi when it lies below 1, and 1 when it covers 1."""
    return lo if lo > 1 else (1 / hi if hi < 1 else 1.0)


def tau_from_subgroups(ref, total, lz, groups, B=None):
    """Smallest tau in TAUS that is at least TAU_FLOOR and contains the resolved departure of every
    reference subgroup scored against the pooled reference (paper Section 2, level 1). A subgroup's
    departure is read on the 90% interval of each ratio (_resolved), from the reference's own B
    replicates, which resample the subgroup's members together with the pooled reference. On a
    large reference the interval ends sit near the point ratios; on a small one the point ratios
    carry sampling noise that would otherwise set tau, and the floor keeps a reference too small to
    resolve its subgroups from setting a tolerance that real subpopulations exceed.
    groups: dict name -> boolean mask."""
    L1 = ref.L1
    B = ref.B if B is None else B
    loc = ref.cond.locate(ref.smap[np.asarray(total, int)], lz)

    def ratios(w):
        Fm, F = ref.cond.cdf(loc, w)
        cl, ch, _ = L1.tail_probs(Fm, F)
        return {k: (np.average(cl[m], weights=w[m]) / .05, np.average(ch[m], weights=w[m]) / .05)
                for k, m in groups.items()}

    pt = ratios(ref.w)
    reps = [ratios(ref.wrep(b)) for b in range(B)]
    rows, worst = [], 1.0
    for name, m in groups.items():
        rm, ro = pt[name]
        bm = np.array([r[name][0] for r in reps])
        bo = np.array([r[name][1] for r in reps])
        mlo, mhi, olo, ohi = (np.quantile(bm, .05), np.quantile(bm, .95), np.quantile(bo, .05), np.quantile(bo, .95))
        dev = max(_resolved(mlo, mhi), _resolved(olo, ohi))
        worst = max(worst, dev)
        rows.append(dict(group=name, n=int(m.sum()), misfit_ratio=rm, misfit_ci90_lo=mlo, misfit_ci90_hi=mhi,
                         overfit_ratio=ro, overfit_ci90_lo=olo, overfit_ci90_hi=ohi,
                         point_departure=max(rm, 1 / rm, ro, 1 / ro), resolved_departure=dev))
    tau = next((t for t in TAUS if worst <= t and t >= TAU_FLOOR), np.inf)
    return tau, worst, rows


# ============================================================================================== L4
class Tables:
    """81_l4_lib.Tables for J items with M categories."""

    def __init__(self, X, w, cluster, M):
        X = np.asarray(X, int)
        w = np.asarray(w, float)
        self.M, self.J = M, X.shape[1]
        self.pairs = [(i, j) for i in range(self.J) for j in range(i + 1, self.J)]
        self.ids, cidx = np.unique(np.asarray(cluster), return_inverse=True)
        C = len(self.ids)
        self.C = C
        MM = M * M
        tab = np.zeros((C, len(self.pairs), MM))
        for k, (i, j) in enumerate(self.pairs):
            code = X[:, i] * M + X[:, j]
            tab[:, k, :] = np.bincount(cidx * MM + code, weights=w, minlength=C * MM).reshape(C, MM)
        self.tab = tab

    def agg(self, mult=None):
        mult = np.ones(self.C) if mult is None else mult
        return np.tensordot(mult, self.tab, 1)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)


def _thresholds(tabs):
    n = tabs.sum(axis=(1, 2))
    cr = np.cumsum(tabs.sum(2), 1)[:, :-1] / n[:, None]
    cc = np.cumsum(tabs.sum(1), 1)[:, :-1] / n[:, None]
    ta = stats.norm.ppf(np.clip(cr, 1e-8, 1 - 1e-8))
    tb = stats.norm.ppf(np.clip(cc, 1e-8, 1 - 1e-8))
    pad = np.full((len(tabs), 1), 8.0)
    return np.hstack([-pad, ta, pad]), np.hstack([-pad, tb, pad])


def _loglik_grid(tabs, A, B, rg):
    K, G = rg.shape
    M1 = A.shape[1]
    h = A[:, None, :, None]
    k = B[:, None, None, :]
    base = stats.norm.cdf(h) * stats.norm.cdf(k)
    out = np.zeros((K, G, M1, M1))
    for x, wt in zip(_GL_X, _GL_W):
        t = (x * rg / 2 + rg / 2)[:, :, None, None]
        q = 1 - t ** 2
        out += (wt * rg / 2)[:, :, None, None] * np.exp(-(h ** 2 - 2 * t * h * k + k ** 2) / (2 * q)) / np.sqrt(q)
    Pm = base + out / (2 * np.pi)
    cell = Pm[:, :, 1:, 1:] - Pm[:, :, :-1, 1:] - Pm[:, :, 1:, :-1] + Pm[:, :, :-1, :-1]
    cell = np.clip(cell, 1e-12, 1.0)
    return (tabs[:, None, :, :] * np.log(cell)).sum(axis=(2, 3))


def poly_from(tab, J, M, chunk=60):
    """All J(J-1)/2 polychorics: coarse grid, fine grid, parabolic step (81_l4_lib.poly_from)."""
    pairs = [(i, j) for i in range(J) for j in range(i + 1, J)]
    tabs_all = tab.reshape(len(pairs), M, M)
    r = np.empty(len(pairs))
    for s0 in range(0, len(pairs), chunk):
        tabs = tabs_all[s0:s0 + chunk]
        K = len(tabs)
        A, B = _thresholds(tabs)
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
        r[s0:s0 + K] = g2[idx, j] + np.clip(off, -1, 1) * step
    R = np.eye(J)
    for kk, (i, jj) in enumerate(pairs):
        R[i, jj] = R[jj, i] = r[kk]
    return R


def one_factor_uls(R):
    """81_l4_lib.one_factor_uls for any J."""
    P = R.shape[0]
    iu = np.triu_indices(P, 1)
    r = R[iu]

    def f(lam):
        e = r - np.outer(lam, lam)[iu]
        E = np.zeros((P, P))
        E[iu] = e
        E = E + E.T
        return float((e ** 2).sum()), -2 * E @ lam

    vals, vecs = np.linalg.eigh(R)
    start = vecs[:, -1] * np.sqrt(max(vals[-1] - 1, 0.1))
    best = None
    for s0 in (start, np.full(P, 0.6), -start):
        res = optimize.minimize(f, s0, jac=True, method="L-BFGS-B", bounds=[(-0.999, 0.999)] * P)
        if best is None or res.fun < best.fun:
            best = res
    lam = best.x
    if lam.sum() < 0:
        lam = -lam
    return lam


def eig_desc(R):
    return np.sort(np.linalg.eigvalsh(R))[::-1]


R2_PHI = 0.95    # R2 shape: lower 90% limit of Tucker's congruence with the reference loadings
R2_RMSD = 0.10   # R2 size: upper 90% limit of the RMS loading difference from the reference


def r2_verdict(phi_lo, phi_hi, rmsd_lo, rmsd_hi):
    """pass: congruence lower limit >= R2_PHI and RMSD upper limit <= R2_RMSD; fail: congruence
    upper limit < R2_PHI or RMSD lower limit > R2_RMSD; otherwise unresolved."""
    if phi_lo >= R2_PHI and rmsd_hi <= R2_RMSD:
        return "pass"
    if phi_hi < R2_PHI or rmsd_lo > R2_RMSD:
        return "fail"
    return "unresolved"


R1_LOAD, R1_RATIO = 0.30, 3.0


def general_factor(lam, ev, min_load=R1_LOAD, min_ratio=R1_RATIO):
    """Absolute R1: every loading >= min_load and lambda1/lambda2 >= min_ratio."""
    return bool((np.asarray(lam) >= min_load).all() and ev[0] / ev[1] >= min_ratio)


def general_factor_vs_ref(lam, ev, ref_lam, ref_ev):
    """Frozen R1: the absolute thresholds, lowered to the reference's own value where the reference
    falls below them (item by item for the loadings), so a model needs a general factor at least
    as strong as people's and the reference passes itself."""
    floor = np.minimum(R1_LOAD, np.asarray(ref_lam))
    return bool((np.asarray(lam) >= floor).all() and ev[0] / ev[1] >= min(R1_RATIO, ref_ev[0] / ref_ev[1]))


def congruence(a, b):
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def boot_mult(C, rng):
    return np.bincount(rng.integers(0, C, C), minlength=C).astype(float)


def within_R(X, w, cell):
    """Pooled within-cell correlation matrix (81b CellUnits.split, within part)."""
    X = np.asarray(X, float)
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    _, ci = np.unique(np.asarray(cell), return_inverse=True)
    S0 = np.bincount(ci, weights=w)
    Mn = np.stack([np.bincount(ci, weights=w * X[:, i]) for i in range(X.shape[1])], 1) / S0[:, None]
    D = X - Mn[ci]
    Sw = (D * w[:, None]).T @ D / w.sum()
    d = np.sqrt(np.clip(np.diag(Sw), 1e-12, None))
    return Sw / np.outer(d, d)


class L4Ref:
    """Reference loadings, point and B bootstrap replicates (iid persons unless mult_fn given)."""

    def __init__(self, X, w, M, B, rng, cluster=None, mult_fn=None):
        X = np.asarray(X, int)
        n, J = X.shape
        self.J, self.M = J, M
        cl = np.arange(n) if cluster is None else cluster
        T = Tables(X, np.ones(n) if w is None else w, cl, M)
        R = poly_from(T.agg(), J, M)
        self.R = R
        self.lam = one_factor_uls(R)
        self.ev = eig_desc(R)
        self.gf = general_factor(self.lam, self.ev)
        mult_fn = mult_fn or (lambda: boot_mult(T.C, rng))
        self.lam_b = np.array([one_factor_uls(poly_from(T.agg(mult_fn()), J, M)) for _ in range(B)])


def l4_eval(X, persona, ref, rng, B=None):
    """R1, R2, R3 for one simulated population (unweighted draws, persona clusters)."""
    B = ref.lam_b.shape[0] if B is None else B
    J, M = ref.J, ref.M
    T = Tables(X, np.ones(len(X)), persona, M)
    R = poly_from(T.agg(), J, M)
    lam = one_factor_uls(R)
    ev = eig_desc(R)
    lbs = [one_factor_uls(poly_from(T.agg(boot_mult(T.C, rng)), J, M)) for _ in range(B)]
    phis = np.array([congruence(lb, ref.lam_b[b]) for b, lb in enumerate(lbs)])
    rmsds = np.array([np.sqrt(np.mean((lb - ref.lam_b[b]) ** 2)) for b, lb in enumerate(lbs)])
    phi_lo, phi_hi = float(np.quantile(phis, 0.05)), float(np.quantile(phis, 0.95))
    rmsd_lo, rmsd_hi = float(np.quantile(rmsds, 0.05)), float(np.quantile(rmsds, 0.95))
    Rw = within_R(X, None, persona)
    lw, evw = one_factor_uls(Rw), eig_desc(Rw)
    eq = np.full(J, 1.0)
    r2 = r2_verdict(phi_lo, phi_hi, rmsd_lo, rmsd_hi)
    return dict(ev_ratio=ev[0] / ev[1], load_min=lam.min(), R1=general_factor_vs_ref(lam, ev, ref.lam, ref.ev),
                R1_absolute=general_factor(lam, ev),
                phi=congruence(lam, ref.lam), phi_lo90=phi_lo, phi_hi90=phi_hi,
                loading_rmsd=float(np.sqrt(np.mean((lam - ref.lam) ** 2))), loading_rmsd_lo90=rmsd_lo,
                loading_rmsd_hi90=rmsd_hi, R2=r2,
                phi_equal_loadings=congruence(eq, ref.lam), within_ev_ratio=evw[0] / evw[1],
                within_load_min=lw.min(), R3=general_factor(lw, evw), lam=lam)
