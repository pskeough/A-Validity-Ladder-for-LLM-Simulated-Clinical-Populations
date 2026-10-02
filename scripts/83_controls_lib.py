"""Shared code for the positive control and planted-failure simulation (scripts 83a to 83c).

Question. Every model fails nearly every rung. Does a simulator that returns real people pass the
same tests, and how large must a failure be before each rung sees it? This file builds a
pseudo-model out of NHANES respondents and runs each rung's pass rule on it, with the design,
reference and inference of the real audit.

Split. NHANES 2005-2018 adults (the 80a frame joined to the 79a item scores) are split 50/50 at
random, stratified by demographic cell (race x sex x income band x marital status, unclassified
rows as their own stratum). The donor half supplies the pseudo-model's draws; the reference half is
the benchmark every rung reads against, with its own design-based variance (every PSU survives in
both halves, so the design is kept). No donor is in the reference.

Pseudo-model. The corpus design: 120 personas x 2 framings x 30 draws. A draw is a donor respondent
from the persona's cell, drawn with replacement with probability proportional to the MEC weight, so
a persona's expected cell mean is the population cell mean. Personas map to cells as the rungs map
them: race White/Black/Hispanic/Asian, Multiracial -> the NHANES "Other" group (placeholder, used
only at L1 and L4, which read every draw; L2 and L3 read the 48 anchored cisgender personas);
cisgender and transgender men -> Men, women -> Women (placeholder for the transgender half, same
reason); income Low/Middle/High -> INDFMPIR bands; Married/Single -> the level-2 marital coding.
The two framings are independent donor samples.

Rung rules reproduced (each checked against the published rung outputs by 83a):
  L1  total-matched lz* tail ratios, reference = the reference half, persona bootstrap jointly with
      Rao-Wu PSU replicates on the reference half; pass when both 90% ratio intervals lie in
      [1/tau, tau], tau = 1.5 (read_verdict of 79c). GRM item parameters are the full-sample fit
      (l1_grm_params.csv), which has seen the reference half; the leak is one set of 32 parameters
      fitted on 36,274 people and is stated with the result.
  L2  R3 (78c r3) on the standardised estimand, one model scope, Bonferroni over the seven
      contrasts (98.6% interval), asymmetric band, conditional stop. Simulated gap: 78b's pairing
      (persona pairs differing in one attribute, framings averaged within a pair, SE sd / sqrt(n)).
      Reference: the equal-weight average of within-stratum differences of weighted cell means on
      the reference half, stratified delete-one-PSU jackknife (78a used Taylor linearisation; 83a
      compares the two on the full sample).
  L3  post-stratified residual (80c "PS"), mean PHQ-8, framings combined, ten groups; TOST at
      0.2 SD, 1 and 2 points; a model passes a tolerance only when all ten groups pass.
  L4  R1 general factor, R2 phi with the reference-half loadings (5th percentile of B bootstrap
      replicates >= .95, personas for the model, Rao-Wu PSUs for the reference, paired by
      replicate), R3 general factor within personas; a model passes when all three hold in both
      framings. R4 (invariance) is not simulated: it needs a semopy fit per replicate and dose.

Seeds are set by the calling script.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUTD = os.path.join(BASE, "analysis", "brm")
DPQ = [f"DPQ0{i}0" for i in range(1, 9)]


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L1 = _load("l1lib", "79_l1_lib.py")
PF = _load("l1pf", "79c_l1_personfit.py")
L3 = _load("l3lib", "80_l3_lib.py")
L4 = _load("l4lib", "81_l4_lib.py")
R2M = _load("l2r3", "78c_l2_r3_verdicts.py")
_argv = sys.argv
sys.argv = [_argv[0]]                       # 81b reads its options from argv at import
L4S = _load("l4s", "81b_l4_structure.py")
sys.argv = _argv

RACES5 = ["White", "Black", "Asian", "Hispanic", "Other"]
RACES4 = L3.RACES
SEXES, INCS, MARS = L3.SEXES, L3.INCS, L3.MARS
CELLS48 = [(r, s, i, m) for r in RACES4 for s in SEXES for i in INCS for m in MARS]
C48 = {c: k for k, c in enumerate(CELLS48)}
TAU = 1.5
R2_RMSD = 0.10   # R2 size condition: upper 90% limit of the loading RMSD against the reference
_tb = pd.read_csv(os.path.join(OUTD, "80a_tolerance_basis.csv"))
SD0 = float(_tb[(_tb.window == "2005-2018") & (_tb.population == "all adults 18+")].sd.iloc[0])
L3_TOLS = {"0.2SD": 0.2 * SD0, "1pt": 1.0, "2pt": 2.0}
L3_GROUPS = [("Overall", None, None)] + [(r, 0, r) for r in RACES4] + \
            [(s, 1, s) for s in SEXES] + [(i, 2, i) for i in INCS]
L2_CONTRASTS = [("Black minus White", 0, "Black", "White"), ("Hispanic minus White", 0, "Hispanic", "White"),
                ("Asian minus White", 0, "Asian", "White"), ("Women minus Men", 1, "Women", "Men"),
                ("Low minus High SES", 2, "Low", "High"), ("Middle minus High SES", 2, "Middle", "High"),
                ("Low minus Middle SES", 2, "Low", "Middle")]
L2_ALPHA = R2M.ALPHA / R2M.FAMILY
L2_BANDS = R2M.BANDS["asym"]


# ------------------------------------------------------------------------------------------ data
def load_frame():
    """NHANES 2005-2018 adults: design, cell attributes, items, lz* (full-sample GRM)."""
    fr = pd.read_csv(os.path.join(OUTD, "80a_nhanes_frame.csv"), low_memory=False)
    d = L3.window_frame(fr, "2005-2018", asian_adjust=True)          # w: L3's weight
    sc = pd.read_csv(os.path.join(OUTD, "l1_scores_nhanes.csv"))
    sc = sc[sc.reference == "2005_2018"][["SEQN", "w", "total", "lzstar_weighted"] + DPQ]
    sc = sc.rename(columns={"w": "w7", "lzstar_weighted": "lz"})
    n0 = len(d)
    d = d.merge(sc, on="SEQN", how="inner")
    assert len(d) == n0 == len(sc), (n0, len(d), len(sc))
    assert np.allclose(d.phq8, d.total)
    d["psu"] = d.stratum.astype(int) * 10 + d.psu.astype(int)       # 79 and 81 code PSUs this way
    d["cluster"] = d.psu
    ok = d.pir_band.isin(INCS) & d.mar.isin(MARS) & d.race.isin(RACES5)
    d["cell"] = np.where(ok, d.race + "|" + d.sex + "|" + d.pir_band.astype(str) + "|" + d.mar.astype(str),
                         "unclassified")
    d["c48"] = [C48.get(tuple(c.split("|")), -1) if c != "unclassified" else -1 for c in d.cell]
    return d.reset_index(drop=True)


def load_personas():
    v = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v3.csv"), low_memory=False,
                    usecols=["profile_id", "race", "gender", "ses_normalized", "relationship"])
    p = v.drop_duplicates().sort_values("profile_id").reset_index(drop=True)
    assert len(p) == 120
    p["race_n"] = p.race.replace({"Multiracial": "Other"})
    p["sex_n"] = np.where(p.gender.str.endswith("Man"), "Men", "Women")
    p["cis"] = p.gender.str.startswith("Cisgender")
    p["cell"] = p.race_n + "|" + p.sex_n + "|" + p.ses_normalized + "|" + p.relationship
    p["c48"] = [C48.get((r, s, i, m), -1) if c else -1 for r, s, i, m, c in
                zip(p.race_n, p.sex_n, p.ses_normalized, p.relationship, p.cis)]
    assert sorted(p.c48[p.c48 >= 0]) == list(range(48))
    return p


def split(d, rng):
    """Boolean donor mask, 50/50 within each cell (odd cells: the extra row goes either way)."""
    perm = rng.permutation(len(d))
    cell = d.cell.to_numpy()[perm]
    rank = pd.Series(cell).groupby(cell).cumcount().to_numpy()
    off = {c: int(rng.integers(0, 2)) for c in np.unique(cell)}
    donor = np.zeros(len(d), bool)
    donor[perm] = ((rank + np.array([off[c] for c in cell])) % 2) == 0
    return donor


def generate(don, personas, rng, n_draws=30):
    """Draws of the pseudo-model. Returns dict of arrays over 120 x 2 x n_draws rows."""
    idx_by_cell = {c: np.flatnonzero(don.cell.to_numpy() == c) for c in personas.cell.unique()}
    w = don.w7.to_numpy()
    rows, pers, frm = [], [], []
    for k, c in enumerate(personas.cell):
        pool = idx_by_cell[c]
        assert len(pool) > 0, c
        pr = w[pool] / w[pool].sum()
        for f in (0, 1):
            rows.append(rng.choice(pool, n_draws, replace=True, p=pr))
            pers.append(np.full(n_draws, k))
            frm.append(np.full(n_draws, f))
    r = np.concatenate(rows)
    return dict(X=don[DPQ].to_numpy(int)[r], lz=don.lz.to_numpy()[r], total=don.total.to_numpy()[r],
                persona=np.concatenate(pers), framing=np.concatenate(frm))


def grm_params():
    par = pd.read_csv(os.path.join(OUTD, "l1_grm_params.csv"))
    par = par[par.fit == "weighted"].set_index("item").loc[DPQ]
    return par.a.to_numpy(float), par[["b1", "b2", "b3"]].to_numpy(float)


# -------------------------------------------------------------------------------------------- L1
class L1Ref:
    def __init__(self, ref, B, rng):
        self.smap = L1.total_strata(ref.total.to_numpy(), 100)
        self.cond = L1.TailRef(self.smap[ref.total.to_numpy()], ref.lz.to_numpy())
        self.w = ref.w7.to_numpy()
        self.mult, self.pidx = L1.raowu_mult(ref, B, rng)
        self.B = B

    def wrep(self, b):
        return self.w if b is None else self.mult[b, self.pidx] * self.w


def l1_eval(ref, total, lz, persona, rng, B=None):
    """Matched misfit / overfit ratios with 90% intervals (persona bootstrap x Rao-Wu, paired)."""
    B = ref.B if B is None else B
    loc = ref.cond.locate(ref.smap[total], lz)
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
    vm, vo, overall = PF.read_verdict(r, TAU)
    r.update(misfit_verdict=vm, overfit_verdict=vo, verdict=overall, passed=overall == "pass")
    return r


def typical_bank(don):
    """For each total, the donor pattern whose lz* is nearest the weighted median lz* at that total."""
    bank = {}
    for t, g in don.groupby("total"):
        med = L1.wquantile(g.lz.to_numpy(), g.w7.to_numpy(), 0.5)
        j = int(np.argmin(np.abs(g.lz.to_numpy() - med)))
        bank[int(t)] = (g[DPQ].to_numpy(int)[j], float(g.lz.to_numpy()[j]))
    return bank


# -------------------------------------------------------------------------------------- L2 / L3
class CellRef:
    """48-cell weighted totals on a (half) sample with JKn replicates (80_l3_lib.Design)."""

    def __init__(self, ref, wcol="w"):
        self.des = L3.Design(ref)
        ci = ref.c48.to_numpy()
        ok = ci >= 0
        oh = np.zeros((len(ref), 48))
        oh[np.flatnonzero(ok), ci[ok]] = 1.0
        w = ref[wcol].to_numpy(float)
        y = ref.total.to_numpy(float)
        self.Nf, self.Nr = self.des.replicate(self.des.psu_totals(oh * w[:, None]))
        self.Yf, self.Yr = self.des.replicate(self.des.psu_totals(oh * (w * y)[:, None]))
        self.mf, self.mr = self.Yf / self.Nf, self.Yr / self.Nr
        self.df = self.des.dfree


def _attr(k):
    return np.array([c[k] for c in CELLS48])


def l2_coef(dim, hi, lo):
    """Equal-weight standardisation coefficients over the 48 cells, and the stratum key per cell."""
    a = _attr(dim)
    others = [j for j in range(4) if j != dim]
    key = np.array(["|".join(CELLS48[c][j] for j in others) for c in range(48)])
    strata = np.unique(key[(a == hi) | (a == lo)])
    coef = np.zeros(48)
    coef[a == hi] = 1.0 / len(strata)
    coef[a == lo] = -1.0 / len(strata)
    return coef, key


def l2_reference(cref):
    out = {}
    for name, dim, hi, lo in L2_CONTRASTS:
        coef, _ = l2_coef(dim, hi, lo)
        est = float(cref.mf @ coef)
        reps = cref.mr @ coef
        out[name] = (est, float(np.sqrt(cref.des.jk_var(est, reps))), cref.df)
    return out


def l2_sim(s_clin, s_narr, v_clin=None, v_narr=None):
    """78b pairing for one model: per contrast, (gap, se, df, se_cond). s_*: 48 persona cell means.

    se is 78b's (SD of persona-pair differences / sqrt(pairs), df pairs - 1). se_cond is conditional
    on the fixed persona set, as level 3 is: sqrt(sum coef^2 var(cell mean)), with var from the
    within-cell draw variances v_* (variance of each framing's cell mean); NaN when v_* is absent."""
    out = {}
    vc = None if v_clin is None else (v_clin + v_narr) / 4
    for name, dim, hi, lo in L2_CONTRASTS:
        a = _attr(dim)
        coef, key = l2_coef(dim, hi, lo)
        ih, il = np.flatnonzero(a == hi), np.flatnonzero(a == lo)
        kh = {key[i]: i for i in ih}
        pairs = [(kh[key[j]], j) for j in il]
        diff = np.array([((s_clin[h] - s_clin[l]) + (s_narr[h] - s_narr[l])) / 2 for h, l in pairs])
        se_c = np.nan if vc is None else float(np.sqrt(np.sum(coef ** 2 * vc)))
        out[name] = (float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff))), len(diff) - 1, se_c)
    return out


def l2_conditional(sim):
    """Swap in the conditional SE (df infinite: the draw variances rest on 60 draws per cell)."""
    return {k: (v[0], v[3], np.inf) for k, v in sim.items()}


def l2_pool(sims):
    """Pooled scope of 78b: equal model weights, SE sqrt(sum se_m^2) / K, Satterthwaite df."""
    out = {}
    K = len(sims)
    for name in sims[0]:
        g = np.array([s[name][0] for s in sims])
        v = np.array([s[name][1] ** 2 for s in sims]) / K ** 2
        dfs = np.array([s[name][2] for s in sims], float)
        out[name] = (float(g.mean()), float(np.sqrt(v.sum())), float(v.sum() ** 2 / np.sum(v ** 2 / dfs)))
    return out


def l2_verdict(sim, ref, name, ref_se_scale=1.0):
    """ref_se_scale 1/sqrt(2) approximates the audit's full-sample reference precision."""
    g, se_g, df_g = sim[name][:3]
    p, se_p, df_p = ref[name]
    return R2M.r3(g, se_g, df_g, p, se_p * ref_se_scale, df_p, L2_ALPHA, L2_BANDS)


def true_region(g):
    """R3 region of a true ratio g (strictly inside a region by construction of the doses)."""
    return R2M.LABELS[int(np.sum(g > L2_BANDS))]


def l3_eval(cref, s, v):
    """PS residual per group with TOST. s, v: 48 cell values and their variances (combined framing)."""
    rows = []
    for gname, dim, val in L3_GROUPS:
        g = np.ones(48, bool) if dim is None else _attr(dim) == val
        Ng, Ngr = cref.Nf[g].sum(), cref.Nr[:, g].sum(1)
        p, pr = cref.Nf[g] / Ng, cref.Nr[:, g] / Ngr[:, None]
        sim, simr = float(p @ s[g]), pr @ s[g]
        ref, refr = cref.Yf[g].sum() / Ng, cref.Yr[:, g].sum(1) / Ngr
        res, resr = sim - ref, simr - refr
        vref = float(cref.des.jk_var(res, resr))
        vsim = float(np.sum(p ** 2 * v[g]))
        var = vsim + vref
        df = var ** 2 / (vref ** 2 / cref.df) if vref > 0 else 1e9
        rows.append(dict(group=gname, resid=res, se=np.sqrt(var), df=df,
                         tcrit=float(stats.t.ppf(0.95, df)), ref=ref))
    return rows


def l3_pass(rows, shift, tol):
    """All ten groups' 90% intervals of (resid + shift) inside +/- tol."""
    return all(abs(r["resid"] + shift) + r["tcrit"] * r["se"] < tol for r in rows)


# -------------------------------------------------------------------------------------------- L4
class L4Ref:
    """Reference-half loadings, point and B Rao-Wu replicates."""

    def __init__(self, ref, B, rng):
        T = L4.Tables(ref[DPQ].to_numpy(int), ref.w7.to_numpy(float), ref.cluster.to_numpy())
        strata = (T.ids // 10).astype(int)
        R = L4.poly_from(T.agg())
        self.lam, _, _ = L4.one_factor_uls(R)
        self.ev = L4.eig_desc(R)
        self.gf = L4.general_factor(self.lam, self.ev)
        self.lam_b = np.array([L4.one_factor_uls(L4.poly_from(T.agg(L4.cluster_boot_mult(T.C, rng, strata))))[0]
                               for _ in range(B)])
        wb = ref[DPQ + ["w7", "cluster", "cell"]].rename(columns=dict(zip(DPQ, L4.ITEMS)))
        wb = wb.rename(columns={"w7": "w"})
        wb = wb[wb.cell != "unclassified"]
        share, Rw, _, _ = L4S.CellUnits(wb, "cell").split()
        lw, _, _, evw = L4S.fit_R(Rw)
        self.within_ratio, self.within_gf, self.between_share = evw[0] / evw[1], L4.general_factor(lw, evw), share.mean()


def l4_eval(X, persona, l4ref, rng, B=None):
    """R1, R2, R3 for one model x framing population (unweighted draws, persona clusters)."""
    B = l4ref.lam_b.shape[0] if B is None else B
    T = L4.Tables(X, np.ones(len(X)), persona)
    R = L4.poly_from(T.agg())
    lam, _, _ = L4.one_factor_uls(R)
    ev = L4.eig_desc(R)
    r1 = L4.general_factor(lam, ev)
    phis, rmsds = np.empty(B), np.empty(B)
    for b in range(B):
        lb = L4.one_factor_uls(L4.poly_from(T.agg(L4.cluster_boot_mult(T.C, rng))))[0]
        phis[b] = L4.congruence(lb, l4ref.lam_b[b])
        rmsds[b] = np.sqrt(np.mean((lb - l4ref.lam_b[b]) ** 2))
    phi_lo, phi_hi = float(np.quantile(phis, 0.05)), float(np.quantile(phis, 0.95))
    rmsd_lo, rmsd_hi = float(np.quantile(rmsds, 0.05)), float(np.quantile(rmsds, 0.95))
    r2v = "pass" if (phi_lo >= 0.95 and rmsd_hi <= R2_RMSD) else \
        ("fail" if (phi_hi < 0.95 or rmsd_lo > R2_RMSD) else "unresolved")
    df = pd.DataFrame(X, columns=L4.ITEMS).assign(w=1.0, cluster=persona, cell=persona)
    share, Rw, _, _ = L4S.CellUnits(df, "cell").split()
    lw, _, _, evw = L4S.fit_R(Rw)
    r3 = L4.general_factor(lw, evw)
    return dict(ev_ratio=ev[0] / ev[1], load_min=lam.min(), R1=r1, phi=L4.congruence(lam, l4ref.lam),
                phi_lo90=phi_lo, R2=bool(phi_lo >= 0.95), between_share=share.mean(),
                phi_hi90=phi_hi, loading_rmsd=float(np.sqrt(np.mean((lam - l4ref.lam) ** 2))),
                loading_rmsd_lo90=rmsd_lo, loading_rmsd_hi90=rmsd_hi,
                R2_size=bool(rmsd_hi <= R2_RMSD), R2_both=bool(phi_lo >= 0.95 and rmsd_hi <= R2_RMSD),
                R2_verdict=r2v,
                within_ev_ratio=evw[0] / evw[1], within_load_min=lw.min(), R3=r3)
