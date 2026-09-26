"""Level-3 rebuild, step c: residuals, standard errors and equivalence tests.

Three estimands, each for the mean PHQ-8 and for the prevalence of PHQ-8 >= 10:

  PS  (primary) post-stratified. The simulated population is a balanced factorial of personas. For a
      group G (overall, a race, a sex or an income band) each of the 48 persona cells c in G gets
      the weight p_c = N_c / sum_{c in G} N_c, where N_c is the NHANES weighted population count of
      cell c (race x sex x income band x marital status). Simulated value: sum_c p_c s_c, with s_c
      the persona cell's mean over draws. Reference: the NHANES weighted mean over the same
      classified population, sum_c p_c y_c. Residual R = sum_c p_c (s_c - y_c).
  DS  design-standardised. Both sides averaged with equal weight over the group's persona cells:
      mean_c s_c against mean_c y_c. This is the level-2 audit's primary reference, carried here so
      the two rungs can share one estimand if Patrick chooses that one.
  MG  marginal, as published (script 67). Simulated: equal-weight mean over the group's cisgender
      persona cells (sex and income marginals include the 12 multiracial cisgender personas, as in
      the paper). Reference: the NHANES weighted mean of every adult in the group (all races,
      uniform cycle weights), i.e. the published anchor.

Standard errors
  Simulated side: the persona cells are fixed by design, so the only sampling is of draws within a
  cell. var(s_c) = s2_c / n_c per framing; the combined framing averages the two framing means,
  var = (v_clin + v_narr) / 4; the pooled model averages the four models, var = sum_m v_m / 16.
  For PS, var_sim = sum_c p_c^2 var(s_c); for DS and MG, sum_c var(s_c) / k^2.
  Reference side: stratified delete-one-PSU jackknife (JKn) on the NHANES design, applied to the
  whole contrast with the persona cell values held fixed. For PS this captures the sampling error of
  y and of the composition weights p_c together. Taylor linearisation is run as a check on the
  primary rows (80c_taylor_vs_jk.csv).
  Total: se = sqrt(var_sim + var_ref); the two sides are independent. Degrees of freedom by
  Satterthwaite with the design df (PSUs minus strata) for the reference and infinity for the
  simulated side.
  Old SE (MG mean only, for comparison): the published rule, SD of the group's cell means over
  sqrt(number of cells), combined with the reference SE in quadrature.

Tolerances (two one-sided tests at alpha .05: pass when the 90% interval lies inside the band)
  mean:        0.2 SD of NHANES PHQ-8 (2005-2018 adults; 80a_tolerance_basis.csv), 1 point, 2 points;
               5 points reported only as the individual-level benchmark the paper used.
  prevalence:  difference within +/-2.5 and +/-5 percentage points; ratio within 0.80 to 1.25, tested
               as two linear one-sided contrasts sim - 0.80 ref > 0 and sim - 1.25 ref < 0, with a 90%
               Fieller interval for the ratio (a log-scale delta method fails when the simulated
               prevalence is near zero, which it is for several middle- and high-income rows).
  delta_min is the smallest symmetric band the row would pass (max |90% limit|).

Specifications (window, income coding, marital coding): see SPECS. Corpora: all rows, and
December-only (clinical rows from dec28_main or dec28_restored, every narrative row). When a
December-only persona cell has no clinical draws the cell is dropped for that model and framing
and the composition is renormalised over the cells that remain; n_cells_missing records it.

Emits analysis/brm/80c_l3_results.csv, 80c_taylor_vs_jk.csv. No randomness.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l3lib", os.path.join(HERE, "80_l3_lib.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

SPECS = [
    ("S1 primary", "2005-2018", "pir_band", "mar"),
    ("S2 cohabiting as married", "2005-2018", "pir_band", "mar_cohab"),
    ("S2b cohabiting as single", "2005-2018", "pir_band", "mar_all"),
    ("S3 2007-2018 PIR (bridge)", "2007-2018", "pir_band", "mar"),
    ("S4 persona-string income", "2007-2018", "p_inc", "mar"),
    ("S5 persona-string, high top-coded", "2007-2018", "p_inc_top", "mar"),
    ("S6 2017-2020 pre-pandemic", "2017-2020", "pir_band", "mar_cohab"),
    ("S7 2021-2023", "2021-2023", "pir_band", "mar_cohab"),
]
MODELS = ["deepseek-chat-v3", "gemini-3-flash-preview", "glm-4.7", "gpt-4o-mini"]
FRAMINGS = ["clinical", "narrative", "combined"]
GROUPS = [("Overall", None, None)] + [(r, "race", r) for r in L.RACES] + \
         [(s, "sex", s) for s in L.SEXES] + [(i, "inc", i) for i in L.INCS]
CELLS = [(r, s, i, m) for r in L.RACES for s in L.SEXES for i in L.INCS for m in L.MARS]
CIDX = {c: k for k, c in enumerate(CELLS)}

tb = pd.read_csv(os.path.join(L.OUTD, "80a_tolerance_basis.csv"))
SD0 = float(tb[(tb.window == "2005-2018") & (tb.population == "all adults 18+")].sd.iloc[0])
MEAN_TOLS = {"0.2SD": 0.2 * SD0, "1pt": 1.0, "2pt": 2.0, "5pt_benchmark": 5.0}
PREV_TOLS = {"2.5pp": 2.5, "5pp": 5.0}
RATIO_BAND = (0.80, 1.25)
print(f"tolerances: 0.2 SD = {0.2 * SD0:.4f} points (SD {SD0:.4f})")

fr = pd.read_csv(os.path.join(L.OUTD, "80a_nhanes_frame.csv"), low_memory=False)
sc = pd.read_csv(os.path.join(L.OUTD, "80b_sim_cells.csv"))
personas = sc[["profile_id", "race", "sex", "inc", "mar"]].drop_duplicates().set_index("profile_id")
assert len(personas) == 60


# ---------- simulated side ----------
def sim_table(corpus, outcome):
    """dict (model, framing) -> DataFrame indexed by profile_id with s, v (NaN when missing)."""
    k = sc[sc.corpus == corpus]
    col, vcol = ("mean", "var") if outcome == "mean" else ("prev10", "prev10_var")
    k = k.assign(s=k[col] * (1 if outcome == "mean" else 100.0),
                 v=(k[vcol] / k.n if outcome == "mean" else k[vcol] * 1e4))
    out = {}
    for mdl in MODELS:
        per = {}
        for f in ["clinical", "narrative"]:
            x = k[(k.model == mdl) & (k.framing == f)].set_index("profile_id")[["s", "v"]]
            per[f] = x.reindex(personas.index)
            out[(mdl, f)] = per[f]
        comb = (per["clinical"] + per["narrative"]) / 2.0
        comb["v"] = (per["clinical"].v + per["narrative"].v) / 4.0
        out[(mdl, "combined")] = comb
    for f in FRAMINGS:
        S = pd.concat([out[(m, f)].s for m in MODELS], axis=1)
        V = pd.concat([out[(m, f)].v for m in MODELS], axis=1)
        cnt = S.notna().sum(axis=1)
        out[("pooled", f)] = pd.DataFrame({"s": S.mean(axis=1), "v": V.sum(axis=1, min_count=1) / cnt ** 2})
    return out


def cell_vec(tab):
    """Order the 48 matched persona cells as CELLS."""
    s = np.full(len(CELLS), np.nan)
    v = np.full(len(CELLS), np.nan)
    for pid, r in personas.iterrows():
        key = (r.race, r.sex, r.inc, r.mar)
        if key in CIDX:
            s[CIDX[key]] = tab.loc[pid, "s"]
            v[CIDX[key]] = tab.loc[pid, "v"]
    return s, v


def gmask(dim, val):
    if dim is None:
        return np.ones(len(CELLS), bool)
    pos = {"race": 0, "sex": 1, "inc": 2}[dim]
    return np.array([c[pos] == val for c in CELLS])


def persona_mask(dim, val):
    """Published marginal: cisgender personas of all five races for sex, income and overall."""
    p = personas
    if dim is None:
        return np.ones(len(p), bool)
    col = {"race": "race", "sex": "sex", "inc": "inc"}[dim]
    return (p[col] == val).values


# ---------- inference ----------
def tost(est, var_sim, var_ref, df_ref):
    var = var_sim + var_ref
    se = float(np.sqrt(var))
    df = var ** 2 / (var_ref ** 2 / df_ref) if var_ref > 0 else 1e9
    tc = float(stats.t.ppf(0.95, df))
    return se, df, tc, est - tc * se, est + tc * se


def jk_moments(des, a, ar, b, br):
    """JKn variance of a, covariance of a and b, variance of b (a = composition-weighted simulated
    value, which moves across replicates only through the weights; b = the reference)."""
    c = (des.rep_nh - 1.0) / des.rep_nh
    da, db = ar - a, br - b
    return float(np.sum(c * da * da)), float(np.sum(c * da * db)), float(np.sum(c * db * db))


def fieller(sim, ref, vsim, mom, df):
    """90% Fieller interval for the prevalence ratio sim / ref, and the two one-sided tests of the
    band 0.80 to 1.25 run as linear contrasts sim - b * ref (no log, so a simulated prevalence near
    zero does not break it). var(sim - b ref) = vsim + A - 2 b B + b^2 C."""
    A, B, C = mom
    t = float(stats.t.ppf(0.95, df))
    qa = ref ** 2 - t ** 2 * C
    qb = -2.0 * (sim * ref - t ** 2 * B)
    qc = sim ** 2 - t ** 2 * (vsim + A)
    disc = qb ** 2 - 4 * qa * qc
    if qa > 0 and disc >= 0:
        lo, hi = (-qb - np.sqrt(disc)) / (2 * qa), (-qb + np.sqrt(disc)) / (2 * qa)
        lo = max(lo, 0.0)       # a ratio of prevalences cannot be negative; the normal approximation can dip below 0
    else:
        lo, hi = np.nan, np.nan
    out = dict(ratio=sim / ref, ratio_ci90_lo=lo, ratio_ci90_hi=hi)
    for b, side in [(RATIO_BAND[0], "lo"), (RATIO_BAND[1], "hi")]:
        se_b = np.sqrt(vsim + A - 2 * b * B + b * b * C)
        out[f"ratio_z_{side}"] = (sim - b * ref) / se_b
    out["ratio_tost_pass"] = bool(out["ratio_z_lo"] > t and out["ratio_z_hi"] < -t)
    return out


rows, tay = [], []
for spec_name, window, incv, marv in SPECS:
    d = L.window_frame(fr, window, asian_adjust=True)
    du = L.window_frame(fr, window, asian_adjust=False)
    des = L.Design(d)
    key = list(zip(d.race, d.sex, d[incv], d[marv]))
    ci = np.array([CIDX.get(k, -1) for k in key])
    onehot = np.zeros((len(d), len(CELLS)))
    ok = ci >= 0
    onehot[np.where(ok)[0], ci[ok]] = 1.0
    ncell = onehot[ok].sum(0)
    # published marginal domains over the full window (uniform weights, all races)
    mdom = {"Overall": np.ones(len(du), bool)}
    for r in L.RACES:
        mdom[r] = (du.race == r).values
    for s in L.SEXES:
        mdom[s] = (du.sex == s).values
    for i in L.INCS:
        mdom[i] = (du[incv] == i).values
    for outcome, ycol, scale in [("mean", "phq8", 1.0), ("prev10", "dep10", 100.0)]:
        y = d[ycol].to_numpy(float) * scale
        w = d.w.to_numpy(float)
        TN = des.psu_totals(onehot * w[:, None])
        TY = des.psu_totals(onehot * (w * y)[:, None])
        Nf, Nr = des.replicate(TN)
        Yf, Yr = des.replicate(TY)
        wu = du.w.to_numpy(float)
        MD = np.column_stack([np.column_stack([wu * m * y, wu * m]) for m in mdom.values()])
        Mf, Mr = des.replicate(des.psu_totals(MD))
        mref = {}
        for j, g in enumerate(mdom):
            th = Mf[2 * j] / Mf[2 * j + 1]
            thr = Mr[:, 2 * j] / Mr[:, 2 * j + 1]
            mref[g] = (th, thr, float(des.jk_var(th, thr)), int(mdom[g].sum()))
        cm_f, cm_r = Yf / Nf, Yr / Nr          # NHANES cell means (full, replicates)
        for corpus in ["all", "dec_only"]:
            if corpus == "dec_only" and spec_name not in ("S1 primary", "S2 cohabiting as married", "S7 2021-2023",
                                                           "S4 persona-string income"):
                continue
            simt = sim_table(corpus, outcome)
            for (mdl, fr_), tab in simt.items():
                s, v = cell_vec(tab)
                avail = ~np.isnan(s)
                for gname, dim, val in GROUPS:
                    g = gmask(dim, val)
                    ga = g & avail
                    base = dict(spec=spec_name, window=window, income_def=incv, marital_def=marv, corpus=corpus,
                                outcome=outcome, framing=fr_, model=mdl, group=gname,
                                n_cells=int(g.sum()), n_cells_missing=int((g & ~avail).sum()))
                    # ----- PS -----
                    Ng, Ngr = Nf[ga].sum(), Nr[:, ga].sum(1)
                    ref_full_g, ref_full_gr = Yf[g].sum() / Nf[g].sum(), Yr[:, g].sum(1) / Nr[:, g].sum(1)
                    p, pr = Nf[ga] / Ng, Nr[:, ga] / Ngr[:, None]
                    simps, simps_r = float(p @ s[ga]), pr @ s[ga]
                    ref, refr = Yf[ga].sum() / Ng, Yr[:, ga].sum(1) / Ngr
                    res, resr = simps - ref, simps_r - refr
                    vref = float(des.jk_var(res, resr))
                    vsim = float(np.sum(p ** 2 * v[ga]))
                    se, df, tc, lo, hi = tost(res, vsim, vref, des.dfree)
                    r = dict(base, estimand="PS", n_nhanes=int(ncell[ga].sum()), sim=simps, ref=ref,
                             ref_se=float(np.sqrt(des.jk_var(ref, refr))), resid=res,
                             se_sim=float(np.sqrt(vsim)), se_ref_part=float(np.sqrt(vref)), se=se, df=df,
                             ci90_lo=lo, ci90_hi=hi, delta_min=max(abs(lo), abs(hi)),
                             ref_full_classified=float(ref_full_g))
                    if outcome == "prev10":
                        r.update(fieller(simps, ref, vsim, jk_moments(des, simps, simps_r, ref, refr), df))
                    rows.append(r)
                    if spec_name == "S1 primary" and corpus == "all" and mdl == "pooled" and fr_ == "combined":
                        # Taylor check of the reference part: d_i = s_c(i) - y_i over the domain
                        dmask = ok & ga[np.clip(ci, 0, None)]
                        dvals = np.where(dmask, s[np.clip(ci, 0, None)] - y, 0.0)
                        _, se_t = des.taylor_mean(dvals, np.where(dmask, w, 0.0))
                        tay.append(dict(outcome=outcome, group=gname, resid=res, se_ref_part_jk=np.sqrt(vref),
                                        se_ref_part_taylor=se_t, ratio=se_t / np.sqrt(vref)))
                    # ----- DS -----
                    ref_ds, ref_dsr = np.nanmean(cm_f[ga]), np.nanmean(cm_r[:, ga], axis=1)
                    k = ga.sum()
                    sim_ds = float(np.mean(s[ga]))
                    res, resr = sim_ds - ref_ds, sim_ds - ref_dsr
                    vref = float(des.jk_var(res, resr))
                    vsim = float(np.sum(v[ga]) / k ** 2)
                    se, df, tc, lo, hi = tost(res, vsim, vref, des.dfree)
                    rows.append(dict(base, estimand="DS", n_nhanes=int(ncell[ga].sum()), sim=sim_ds, ref=ref_ds,
                                     ref_se=float(np.sqrt(vref)), resid=res, se_sim=float(np.sqrt(vsim)),
                                     se_ref_part=float(np.sqrt(vref)), se=se, df=df, ci90_lo=lo, ci90_hi=hi,
                                     delta_min=max(abs(lo), abs(hi))))
                    # ----- MG (as published) -----
                    pm = persona_mask(dim, val)
                    sp, vp = tab.s.values[pm], tab.v.values[pm]
                    av = ~np.isnan(sp)
                    sim_m = float(np.mean(sp[av]))
                    kk = int(av.sum())
                    vsim = float(np.sum(vp[av]) / kk ** 2)
                    th, thr, vref, nref = mref[gname]
                    res = sim_m - th
                    se, df, tc, lo, hi = tost(res, vsim, vref, des.dfree)
                    r = dict(base, estimand="MG", n_cells=int(pm.sum()), n_cells_missing=int(pm.sum() - kk),
                             n_nhanes=nref, sim=sim_m, ref=float(th), ref_se=float(np.sqrt(vref)), resid=res,
                             se_sim=float(np.sqrt(vsim)), se_ref_part=float(np.sqrt(vref)), se=se, df=df,
                             ci90_lo=lo, ci90_hi=hi, delta_min=max(abs(lo), abs(hi)))
                    if outcome == "mean":
                        # published SE: between-cell SD over sqrt(cells); the pooled row spans all models' cells
                        if mdl == "pooled":
                            allc = np.concatenate([simt[(m_, fr_)].s.values[pm] for m_ in MODELS])
                        else:
                            allc = sp
                        allc = allc[~np.isnan(allc)]
                        vold = float(np.var(allc, ddof=1) / len(allc))
                        se_o, df_o, tc_o, lo_o, hi_o = tost(res, vold, vref, des.dfree)
                        r.update(se_sim_old=float(np.sqrt(vold)), se_old=se_o, ci90_lo_old=lo_o, ci90_hi_old=hi_o,
                                 n_cells_old=len(allc))
                    if outcome == "prev10":
                        r.update(fieller(sim_m, th, vsim, jk_moments(des, sim_m, np.full(len(thr), sim_m), th, thr), df))
                    rows.append(r)
    print(f"{spec_name}: design df {des.dfree}, replicates {len(des.rep_psu)}, classified n {int(ok.sum())}")

res = pd.DataFrame(rows)
sd_by_window = tb[tb.population == "all adults 18+"].set_index("window").sd
res["resid_sd_units"] = np.where(res.outcome == "mean", res.resid / res.window.map(sd_by_window), np.nan)
res["resid_sd_units_primary_sd"] = np.where(res.outcome == "mean", res.resid / SD0, np.nan)
for name, tol in MEAN_TOLS.items():
    res[f"pass_{name}"] = np.where(res.outcome == "mean", (res.ci90_lo > -tol) & (res.ci90_hi < tol), np.nan)
for name, tol in PREV_TOLS.items():
    res[f"pass_{name}"] = np.where(res.outcome == "prev10", (res.ci90_lo > -tol) & (res.ci90_hi < tol), np.nan)
res["pass_ratio_0.80_1.25"] = np.where(res.outcome == "prev10", res.ratio_tost_pass.fillna(False).astype(bool), np.nan)
# the Fieller interval and the two linear tests must agree whenever the interval is bounded
chk = res[(res.outcome == "prev10") & res.ratio_ci90_lo.notna()]
agree = ((chk.ratio_ci90_lo > RATIO_BAND[0]) & (chk.ratio_ci90_hi < RATIO_BAND[1])) == chk.ratio_tost_pass.astype(bool)
assert agree.all(), "Fieller interval and ratio TOST disagree"
res["pass_5pt_old_se"] = np.where(res.se_old.notna(), (res.ci90_lo_old > -5) & (res.ci90_hi_old < 5), np.nan)
res["tol_0.2SD_points"] = 0.2 * SD0
num = res.select_dtypes("number").columns
res[num] = res[num].round(6)
res.to_csv(os.path.join(L.OUTD, "80c_l3_results.csv"), index=False)
t = pd.DataFrame(tay).round(6)
t.to_csv(os.path.join(L.OUTD, "80c_taylor_vs_jk.csv"), index=False)
print(t.to_string(index=False))
print(f"\nrows: {len(res)}")
p = res[(res.spec == "S1 primary") & (res.corpus == "all") & (res.framing == "combined")]
pd.set_option("display.width", 250)
for est in ["PS", "MG", "DS"]:
    q = p[(p.estimand == est) & (p.outcome == "mean") & (p.model == "pooled")]
    print(f"\n{est} mean, pooled model, combined framing, 2005-2018:")
    print(q[["group", "sim", "ref", "resid", "se_sim", "se_ref_part", "se", "ci90_lo", "ci90_hi", "delta_min"]].to_string(index=False))
