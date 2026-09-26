"""Level 4, part 2: measurement invariance across sex, race and income, fitted the same way in
NHANES and in each model's simulated population, and read as "does the model reproduce the
population's invariance pattern".

Model. One common factor for the eight PHQ-8 items, multi-group, normal-theory ML with a mean
structure on the 0-3 item scores (semopy 2.3 has no ordinal multi-group estimator and its
multigroup() fits each group separately with no cross-group constraints, so the constrained models
are fitted here by direct minimisation with an analytic gradient; semopy is used as a gate on the
configural fit, which is the sum of separate per-group fits).
  configural  loadings, residual variances and intercepts free in every group; factor variance 1,
              factor mean 0 (the mean structure is saturated, so it drops out)
  metric      loadings equal; factor variance free in groups 2..G
  scalar      loadings and intercepts equal; factor means free in groups 2..G
Groups: sex F/M (corpus: cisgender personas only), race White/Black/Asian/Hispanic, income
Low/Middle/High. NHANES moments are MEC-weighted; N is the respondent count.

Fit function F = sum_g (n_g/N) [ln|Sigma_g| + tr(S_g Sigma_g^-1) - ln|S_g| - p
                                 + (m_g - mu_g)' Sigma_g^-1 (m_g - mu_g)],  T = N F.

Change in fit. For a step with df difference d and fit-function difference DF = F_constrained -
F_free, the RMSEA of the difference (Savalei et al.'s RMSEA_D, multi-group form) is
    RMSEA_D = sqrt(G) * sqrt(max(0, DF - nu) / d),
where nu is what DF would be under exact invariance at this sample size and clustering. Normal
theory gives nu = d/N. Draws are clustered in personas (30 per persona) and the items are coarse and
skewed, so nu is estimated instead by permuting group membership at the cluster level (personas;
for NHANES, respondents within PSU), the permutation approach of Jorgensen et al. (2018), and taking
the mean DF over permutations. The permutation p is reported beside it.
Interval: B cluster-bootstrap replicates (personas resampled within group; NHANES Rao-Wu PSU within
stratum); the replicate distribution of DF is recentred on the corrected estimate and mapped to the
RMSEA_D scale. The 90% interval gives the one-sided 5% equivalence test of Yuan & Chan (2016).
Verdict at margin e0: holds if the upper 90% limit < e0; fails if the lower 90% limit > e0;
otherwise indeterminate. e0 = .05 (primary, "close" in MacCallum, Browne & Sugawara 1996) and .08.

Ordinal robustness. The metric step is refitted on each group's polychoric matrix (the same
constrained model on correlation structure, a delta-parameterisation analogue), with the same
permutation correction and bootstrap. The scalar step has no polychoric analogue without thresholds
and is not refitted.

usage: python 81c_l4_invariance.py [all|dec] [B] [K_PERM] [B_POLY]
Emits analysis/brm/l4_invariance{_dec}.csv, l4_invariance_fits{_dec}.csv, l4_semopy_gate.csv.
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l4lib", os.path.join(HERE, "81_l4_lib.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

SUBSET = sys.argv[1] if len(sys.argv) > 1 else "all"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 200
K_PERM = int(sys.argv[3]) if len(sys.argv) > 3 else 200
B_POLY = int(sys.argv[4]) if len(sys.argv) > 4 else 100
SUF = "" if SUBSET == "all" else "_" + SUBSET
SEED = 8103
IT = L.ITEMS
P = L.P
E0 = (0.05, 0.08)
ATTRS = [("sex", ["F", "M"]), ("race", ["White", "Black", "Asian", "Hispanic"]),
         ("income", ["Low", "Middle", "High"])]


# ------------------------------------------------------------------------------ group moments
def group_moments(X, w, gidx, G, n_mult=None):
    """Weighted means, ML covariances and unweighted n per group from row weights."""
    out = []
    for k in range(G):
        mk = gidx == k
        wk = w[mk]
        Xk = X[mk]
        s0 = wk.sum()
        m = wk @ Xk / s0
        D = Xk - m
        S = (D * wk[:, None]).T @ D / s0
        n = mk.sum() if n_mult is None else n_mult[mk].sum()
        out.append((m, S, float(n)))
    return out


def group_polys(X, w, gidx, G):
    Rs = []
    for k in range(G):
        mk = gidx == k
        T = L.Tables(X[mk], w[mk], np.zeros(mk.sum()))
        Rs.append(L.poly_from(T.agg()))
    return Rs


# ---------------------------------------------------------------------------------- MG-CFA ML
class Spec:
    """Parameter layout. kind in {configural, metric, scalar}."""

    def __init__(self, kind, G):
        self.kind, self.G = kind, G
        k = 0
        self.lam, self.th, self.phi, self.nu, self.kap = [], [], [], [], []
        shared_lam = shared_nu = None
        if kind in ("metric", "scalar"):
            shared_lam = np.arange(k, k + P); k += P
        if kind == "scalar":
            shared_nu = np.arange(k, k + P); k += P
        for g in range(G):
            if kind == "configural":
                self.lam.append(np.arange(k, k + P)); k += P
            else:
                self.lam.append(shared_lam)
            self.th.append(np.arange(k, k + P)); k += P
            if kind != "configural" and g > 0:
                self.phi.append(k); k += 1
            else:
                self.phi.append(-1)
            if kind == "scalar":
                self.nu.append(shared_nu)
                if g > 0:
                    self.kap.append(k); k += 1
                else:
                    self.kap.append(-1)
            else:
                self.nu.append(None); self.kap.append(-1)
        self.n = k

    def df(self, p=P):
        cov = self.G * p * (p + 1) // 2
        mom = cov + (self.G * p if self.kind == "scalar" else 0)
        par = self.n - (0 if self.kind == "scalar" else 0)
        # saturated means in configural and metric are neither moments nor parameters here
        return mom - par

    def bounds(self, varS):
        b = [(None, None)] * self.n
        for g in range(self.G):
            for i, ix in enumerate(self.th[g]):
                b[ix] = (1e-4 * varS[g][i], None)
            if self.phi[g] >= 0:
                b[self.phi[g]] = (1e-3, None)
        return b


def objective(theta, spec, mom, N, with_means):
    F = 0.0
    grad = np.zeros_like(theta)
    for g, (m, S, n) in enumerate(mom):
        lam = theta[spec.lam[g]]
        th = theta[spec.th[g]]
        phi = 1.0 if spec.phi[g] < 0 else theta[spec.phi[g]]
        Sig = phi * np.outer(lam, lam) + np.diag(th)
        try:
            Lc = np.linalg.cholesky(Sig)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(theta)
        Si = np.linalg.inv(Sig)
        logdet = 2 * np.log(np.diag(Lc)).sum()
        _, logdetS = np.linalg.slogdet(S)
        Ssum = S.copy()
        wg = n / N
        Fg = logdet + np.trace(S @ Si) - logdetS - P
        h = None
        if with_means:
            nu = theta[spec.nu[g]]
            kap = 0.0 if spec.kap[g] < 0 else theta[spec.kap[g]]
            d = m - (nu + kap * lam)
            Fg += d @ Si @ d
            Ssum = S + np.outer(d, d)
            h = -2 * Si @ d
        Gm = Si - Si @ Ssum @ Si
        F += wg * Fg
        glam = 2 * phi * Gm @ lam
        if h is not None:
            glam = glam + kap * h
            np.add.at(grad, spec.nu[g], wg * h)
            if spec.kap[g] >= 0:
                grad[spec.kap[g]] += wg * float(lam @ h)
        np.add.at(grad, spec.lam[g], wg * glam)
        np.add.at(grad, spec.th[g], wg * np.diag(Gm))
        if spec.phi[g] >= 0:
            grad[spec.phi[g]] += wg * float(lam @ Gm @ lam)
    return F, grad


def start_values(spec, mom, prev=None):
    th0 = np.zeros(spec.n)
    lam_g = []
    for (m, S, n) in mom:
        sd = np.sqrt(np.diag(S))
        lam_c, _, _ = L.one_factor_uls(S / np.outer(sd, sd))
        lam_g.append(lam_c * sd)
    for g, (m, S, n) in enumerate(mom):
        th0[spec.lam[g]] = lam_g[g] if spec.kind == "configural" else np.mean(lam_g, 0)
        th0[spec.th[g]] = np.clip(np.diag(S) - th0[spec.lam[g]] ** 2, 0.05 * np.diag(S), None)
        if spec.phi[g] >= 0:
            th0[spec.phi[g]] = 1.0
        if spec.kind == "scalar":
            th0[spec.nu[g]] = mom[0][0]
            if spec.kap[g] >= 0:
                lam = th0[spec.lam[g]]
                th0[spec.kap[g]] = float(lam @ (m - mom[0][0]) / (lam @ lam))
    if prev is not None:
        th0 = prev.copy()
    return th0


def fit(kind, mom, prev=None):
    G = len(mom)
    spec = Spec(kind, G)
    N = sum(n for _, _, n in mom)
    varS = [np.diag(S) for _, S, _ in mom]
    x0 = start_values(spec, mom, prev)
    res = optimize.minimize(objective, x0, args=(spec, mom, N, kind == "scalar"), jac=True,
                            method="L-BFGS-B", bounds=spec.bounds(varS),
                            options=dict(maxiter=5000, ftol=1e-13, gtol=1e-9))
    return res.fun, spec, res.x, N, res.success


def metric_to_configural(xm, G):
    sm, sc = Spec("metric", G), Spec("configural", G)
    x = np.zeros(sc.n)
    for g in range(G):
        phi = 1.0 if sm.phi[g] < 0 else xm[sm.phi[g]]
        x[sc.lam[g]] = xm[sm.lam[g]] * np.sqrt(phi)
        x[sc.th[g]] = xm[sm.th[g]]
    return x


def scalar_to_metric(xs, G):
    ss, sm = Spec("scalar", G), Spec("metric", G)
    x = np.zeros(sm.n)
    for g in range(G):
        x[sm.lam[g]] = xs[ss.lam[g]]
        x[sm.th[g]] = xs[ss.th[g]]
        if sm.phi[g] >= 0:
            x[sm.phi[g]] = xs[ss.phi[g]]
    return x


def fit_all(mom, prev=None):
    """Fit the three nested models. Nesting guarantees F_configural <= F_metric <= F_scalar at the
    global minima; where an optimiser stops short of that, the less constrained model is refitted
    from the more constrained solution, which is a feasible point for it."""
    G = len(mom)
    out = {}
    for kind in ("configural", "metric", "scalar"):
        pv = None if prev is None else prev.get(kind)
        out[kind] = fit(kind, mom, pv)
    if out["scalar"][0] < out["metric"][0] - 1e-10:
        alt = fit("metric", mom, scalar_to_metric(out["scalar"][2], G))
        if alt[0] < out["metric"][0]:
            out["metric"] = alt
    if out["metric"][0] < out["configural"][0] - 1e-10:
        alt = fit("configural", mom, metric_to_configural(out["metric"][2], G))
        if alt[0] < out["configural"][0]:
            out["configural"] = alt
    return out


def fit_cov_only(kind, Rs, ns):
    mom = [(np.zeros(P), R, n) for R, n in zip(Rs, ns)]
    return fit(kind, mom)


def poly_dF(Rs, ns):
    """Metric-minus-configural fit-function difference on polychoric matrices, nesting enforced."""
    G = len(Rs)
    mom = [(np.zeros(P), R, n) for R, n in zip(Rs, ns)]
    c = fit("configural", mom)
    m = fit("metric", mom)
    if m[0] < c[0] - 1e-10:
        alt = fit("configural", mom, metric_to_configural(m[2], G))
        if alt[0] < c[0]:
            c = alt
    return m[0] - c[0]


def baseline_F(mom):
    N = sum(n for _, _, n in mom)
    return sum((n / N) * (np.log(np.diag(S)).sum() - np.linalg.slogdet(S)[1]) for _, S, n in mom)


def rmsea_from_T(T, df, N, G):
    return float(np.sqrt(G) * np.sqrt(max(0.0, (T - df) / (df * N))))


def rmsea_d(DF, nu, d, G):
    return float(np.sqrt(G) * np.sqrt(max(0.0, DF - nu) / d))


def verdict(lo, hi, e0):
    if hi < e0:
        return "holds"
    if lo > e0:
        return "fails"
    return "indeterminate"


# ------------------------------------------------------------------------------------ analysis
def analyse(label, X, w, gidx, G, clusters, strata, perm_units, perm_blocks, rng, levels,
            resample=True):
    """clusters: cluster id per row (bootstrap unit); strata: stratum per cluster or None, in which
    case clusters are resampled within group; perm_units: the unit whose group label is permuted
    (persona for the corpus, respondent for NHANES); perm_blocks: block per perm unit (permute within
    block; NHANES PSU) or None. resample=False (a population with no general factor, 81b) gives
    point estimates only: the one-factor model does not describe that population, so its
    invariance steps have no reading, and the permutation and bootstrap refits are unstable."""
    t0 = time.time()
    mom = group_moments(X, w, gidx, G)
    N = sum(n for _, _, n in mom)
    fits = fit_all(mom)
    Fc, Fm, Fs = fits["configural"][0], fits["metric"][0], fits["scalar"][0]
    dfc = Spec("configural", G).df(); dfm = Spec("metric", G).df(); dfs = Spec("scalar", G).df()
    Fb = baseline_F(mom); dfb = G * P * (P - 1) // 2
    Tc, Tb = N * Fc, N * Fb
    cfi_c = 1 - max(Tc - dfc, 0) / max(Tb - dfb, Tc - dfc, 1e-12)
    prev = {k: v[2] for k, v in fits.items()}
    Rs = group_polys(X, w, gidx, G)
    ns = [n for _, _, n in mom]
    DPo = poly_dF(Rs, ns)
    DMo, DSo = Fm - Fc, Fs - Fm
    d_m, d_s = dfm - dfc, dfs - dfm

    # ---- permutation null at the cluster level
    uid, uinv = np.unique(perm_units, return_inverse=True)
    ug = np.zeros(len(uid), int); ug[uinv] = gidx
    ub = None
    if perm_blocks is not None:
        ub = np.zeros(len(uid), int); ub[uinv] = perm_blocks
    pDM, pDS, pDP = [], [], []
    for kperm in range(K_PERM if resample else 0):
        if ub is None:
            ugp = rng.permutation(ug)
        else:
            ugp = ug.copy()
            for bk in np.unique(ub):
                ix = np.flatnonzero(ub == bk)
                ugp[ix] = rng.permutation(ug[ix])
        gp = ugp[uinv]
        momp = group_moments(X, w, gp, G)
        fp = fit_all(momp, prev)
        pDM.append(fp["metric"][0] - fp["configural"][0])
        pDS.append(fp["scalar"][0] - fp["metric"][0])
        if kperm < B_POLY:
            Rp = group_polys(X, w, gp, G)
            nsp = [n for _, _, n in momp]
            pDP.append(poly_dF(Rp, nsp))
    if resample:
        nuM, nuS, nuP = np.mean(pDM), np.mean(pDS), np.mean(pDP)
    else:
        nuM, nuS, nuP = d_m / N, d_s / N, d_m / N

    # ---- cluster bootstrap
    cid, cinv = np.unique(clusters, return_inverse=True)
    cg = np.zeros(len(cid), int); cg[cinv] = gidx
    bDM, bDS, bDP = [], [], []
    for b in range(B if resample else 0):
        if strata is None:
            mult = np.zeros(len(cid))
            for k in range(G):
                ix = np.flatnonzero(cg == k)
                np.add.at(mult, rng.choice(ix, len(ix), replace=True), 1.0)
        else:
            mult = L.cluster_boot_mult(len(cid), rng, strata)
        rw = mult[cinv]
        keep = rw > 0
        momb = group_moments(X[keep], (w * rw)[keep], gidx[keep], G, n_mult=rw[keep])
        fb = fit_all(momb, prev)
        bDM.append(fb["metric"][0] - fb["configural"][0])
        bDS.append(fb["scalar"][0] - fb["metric"][0])
        if b < B_POLY:
            Rb = group_polys(X[keep], (w * rw)[keep], gidx[keep], G)
            nsb = [n for _, _, n in momb]
            bDP.append(poly_dF(Rb, nsb))

    rows = []
    for step, DFo, nu, perm, boot, d in (("metric", DMo, nuM, pDM, bDM, d_m),
                                          ("scalar", DSo, nuS, pDS, bDS, d_s),
                                          ("metric (polychoric)", DPo, nuP, pDP, bDP, d_m)):
        boot = np.asarray(boot)
        est = DFo - nu
        if resample:
            cen = boot - boot.mean() + est
            lo_df, hi_df = np.quantile(cen, 0.05), np.quantile(cen, 0.95)
            lo, hi = rmsea_d(lo_df + nu, nu, d, G), rmsea_d(hi_df + nu, nu, d, G)
            pp = (1 + np.sum(np.asarray(perm) >= DFo)) / (len(perm) + 1)
        else:
            lo = hi = pp = np.nan
        r = dict(population=label, attribute=levels[0], groups="/".join(levels[1]), G=G, N=N,
                 step=step, df_diff=d, dF=DFo, dT_normal=N * DFo, nu_perm=nu if resample else np.nan,
                 nu_normal=d / N, c_hat=nu / (d / N) if resample else np.nan, p_perm=pp,
                 rmsea_d_naive=rmsea_d(DFo, d / N, d, G), rmsea_d=rmsea_d(DFo, nu, d, G),
                 rmsea_d_lo90=lo, rmsea_d_hi90=hi, n_perm=len(perm), n_boot=len(boot))
        for e0 in E0:
            r[f"verdict_e{int(e0 * 100):02d}"] = (verdict(lo, hi, e0) if resample
                                                  else "not interpretable (no general factor)")
        rows.append(r)
    fitrow = dict(population=label, attribute=levels[0], G=G, N=N, F_configural=Fc, F_metric=Fm,
                  F_scalar=Fs, df_configural=dfc, df_metric=dfm, df_scalar=dfs,
                  rmsea_configural_normal=rmsea_from_T(Tc, dfc, N, G), cfi_configural_normal=cfi_c,
                  converged=all(v[4] for v in fits.values()))
    # standardised loadings per group from the configural fit, for the table
    spec = fits["configural"][1]; th = fits["configural"][2]
    for g in range(G):
        S = mom[g][1]
        lam = th[spec.lam[g]] / np.sqrt(np.diag(S))
        fitrow[f"std_load_min_{levels[1][g]}"] = float(lam.min())
    print(f"  {label:32s} {levels[0]:7s} N={N:6.0f}  config RMSEA {fitrow['rmsea_configural_normal']:.3f} "
          f"CFI {cfi_c:.3f} | " + "  ".join(f"{r['step']}: {r['rmsea_d']:.3f} "
                                             f"[{r['rmsea_d_lo90']:.3f},{r['rmsea_d_hi90']:.3f}] "
                                             f"c={r['c_hat']:.1f}" for r in rows)
          + f"  ({time.time() - t0:.0f}s)", flush=True)
    return rows, fitrow


def semopy_gate(nh):
    """Configural T from separate per-group semopy fits against the direct fit, unweighted NHANES."""
    import semopy
    desc = "F =~ " + " + ".join(IT)
    rows = []
    for attr, col, levels in (("sex", "sex", ["F", "M"]), ("income", "ses", ["Low", "Middle", "High"])):
        sub = nh[nh[col].isin(levels)]
        X = sub[IT].to_numpy(float)
        gidx = sub[col].map({v: i for i, v in enumerate(levels)}).to_numpy()
        mom = group_moments(X, np.ones(len(X)), gidx, len(levels))
        Fc = fit("configural", mom)[0]
        own = sum(n for _, _, n in mom) * Fc
        tot = 0.0
        for k, lev in enumerate(levels):
            m = semopy.Model(desc)
            d = sub[sub[col] == lev][IT].astype(float)
            m.fit(d)
            st = semopy.calc_stats(m)
            tot += float(st["chi2"].iloc[0])
        rows.append(dict(attribute=attr, T_direct=own, T_semopy_sum=tot, rel_diff=(own - tot) / tot))
        print(f"  semopy gate {attr}: direct T {own:.2f}  semopy sum {tot:.2f}  rel diff {(own - tot) / tot:+.5f}")
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    sim = L.load_sim(SUBSET)
    nh = L.load_nhanes()
    print(f"subset={SUBSET} B={B} K_PERM={K_PERM} B_POLY={B_POLY}")
    if SUBSET == "all":
        gate = semopy_gate(nh)
        gate.to_csv(os.path.join(L.OUT, "l4_semopy_gate.csv"), index=False)
        if (gate.rel_diff.abs() > 0.005).any():
            sys.exit("GATE FAILED: direct configural fit does not reproduce semopy")

    # general-factor flags from 81b decide which populations get the resampling
    stc = pd.read_csv(os.path.join(L.OUT, f"l4_structure{SUF}.csv")).set_index("population")
    # progress cache, so an interrupted run resumes; removed at the end
    pr_path = os.path.join(L.OUT, f"l4_invariance{SUF}.partial.csv")
    pf_path = os.path.join(L.OUT, f"l4_invariance_fits{SUF}.partial.csv")
    rows, fitrows, done = [], [], set()
    if os.path.exists(pr_path) and os.path.exists(pf_path):
        rows = pd.read_csv(pr_path).to_dict("records")
        fitrows = pd.read_csv(pf_path).to_dict("records")
        done = {(f["population"], f["attribute"]) for f in fitrows}
        print(f"resuming: {len(done)} population x attribute fits already cached")
    pops = [] if SUBSET != "all" else [("NHANES 2005-2018 (weighted)", nh, True)]
    for mdl in L.MODELS:
        for fr in ("clinical", "narrative"):
            pops.append((f"{mdl} {fr}", sim[(sim.model_s == mdl) & (sim.prompt_condition == fr)], False))
    for pi, (label, df, is_nh) in enumerate(pops):
        resample = True if is_nh else bool(stc.loc[label, "general_factor"])
        for ai, (attr, levels) in enumerate(ATTRS):
            if (label, attr) in done:
                continue
            col = "ses" if attr == "income" else attr
            sub = df[df[col].isin(levels)]
            X = sub[IT].to_numpy(float)
            w = sub.w.to_numpy(float)
            gidx = sub[col].map({v: i for i, v in enumerate(levels)}).to_numpy()
            rng = np.random.default_rng([SEED, pi, ai])
            if is_nh:
                cl = sub.cluster.to_numpy()
                cid = np.unique(cl)
                s_of = sub.groupby("cluster").stratum.first()
                strata = s_of.loc[cid].to_numpy()
                r, f = analyse(label, X, w, gidx, len(levels), cl, strata,
                               sub.SEQN.to_numpy(), cl, rng, (attr, levels))
            else:
                cl = sub.profile_id.to_numpy()
                r, f = analyse(label, X, w, gidx, len(levels), cl, None, cl, None, rng, (attr, levels),
                               resample=resample)
            f["resampled"] = resample
            rows += r
            fitrows.append(f)
            pd.DataFrame(rows).to_csv(pr_path, index=False)
            pd.DataFrame(fitrows).to_csv(pf_path, index=False)
    inv = pd.DataFrame(rows)
    fits = pd.DataFrame(fitrows)
    os.remove(pr_path)
    os.remove(pf_path)
    if SUBSET != "all":
        ref = pd.read_csv(os.path.join(L.OUT, "l4_invariance.csv"))
        inv = pd.concat([ref[ref.population.str.startswith("NHANES")], inv], ignore_index=True)
    inv.to_csv(os.path.join(L.OUT, f"l4_invariance{SUF}.csv"), index=False)
    fits.to_csv(os.path.join(L.OUT, f"l4_invariance_fits{SUF}.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
