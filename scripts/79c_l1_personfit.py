"""Level 1 rebuild, step c: person fit (lz*) of every simulated response vector against NHANES.

Inputs from 79a: lz* for every NHANES adult and every corpus row, computed with the NHANES
2005-2018 GRM item parameters (weighted fit; unweighted fit as a sensitivity).

Two readings of the tails.
  Marginal. The NHANES weighted 5th and 95th percentiles of lz* over all adults; the share of a
      model's draws below the 5th (misfit) and above the 95th (overfit). This is the reading the task
      names, but the models' totals are distributed unlike people's, and lz* depends on the total, so
      it mixes pattern with level.
  Total-matched (primary). Within each total-score stratum (exact totals, the top totals merged until
      each stratum holds 100 NHANES respondents), each draw's lz* is placed in the NHANES weighted
      distribution of lz* at the same total. The expected share below the NHANES within-stratum 5th
      percentile and above the 95th is computed with ties split by mass (the exact expectation of the
      randomised probability integral transform), so the reference gives exactly 5% on each side at
      any total mix. The mean mid-PIT is 0.5 in the reference.
  Reference 5% on each side: a ratio (share / 0.05) of 1 means as many misfitting (or overfitting)
  vectors as people show at the same totals.

Pass rule (two-sided, both tails). A model x framing passes person fit when the 90% interval of the
matched misfit ratio and of the matched overfit ratio both lie inside [1/tau, tau]. Readings:
  misfit ratio above tau: less coherent than people (too many unlikely patterns)
  misfit ratio below 1/tau: too prototypical (too few atypical patterns)
  overfit ratio above tau: too prototypical (too many textbook patterns)
  overfit ratio below 1/tau: less coherent (too few regular patterns)
The pass is an intersection-union test (both tails must pass), so no multiplicity adjustment.
tau is set as in 79b: the smallest of {1.10, 1.25, 1.50, 2.00, 3.00} that contains the point ratios of
every NHANES subgroup (sex, race and ethnicity, poverty-income band, cycle, 2021-2023) on both tails.

Intervals: persona-clustered bootstrap for the corpus jointly with the Rao-Wu PSU bootstrap for
NHANES, B = 2000. Ratio intervals 90%, share and mean intervals 95%.

Also: person fit against total (mean lz* and matched tail shares within PHQ bands 0, 1-4, 5-9,
10-14, 15-19, 20-24), item-mean profiles at matched totals, and the share of flat vectors.

Sensitivities: December-only rows; NHANES 2021-2023 as the reference distribution (same item
parameters); the unweighted GRM.

Emits (analysis/brm/):
  l1_personfit_nhanes_ref.csv        NHANES percentiles (marginal and per PHQ band), strata
  l1_personfit_tolerance.csv         NHANES subgroup tail ratios and tau
  l1_personfit_main.csv              per model x framing (x gender group), all samples and references
  l1_personfit_verdicts_tau.csv      verdicts at every tau
  l1_personfit_by_band.csv           person fit within PHQ bands
  l1_personfit_item_profile.csv      item means minus NHANES item means at matched totals
Seeded (20260927).
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

spec = importlib.util.spec_from_file_location("l1", os.path.join(os.path.dirname(__file__), "79_l1_lib.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

B = 2000
rng = np.random.default_rng(20260927)
TAUS = [1.10, 1.25, 1.50, 2.00, 3.00]
PHQB = [(0, 0), (1, 4), (5, 9), (10, 14), (15, 19), (20, 24)]
PHQL = ["0", "1-4", "5-9", "10-14", "15-19", "20-24"]


def phq_band(t):
    out = np.full(len(t), "", dtype=object)
    for (lo, hi), lab in zip(PHQB, PHQL):
        out[(t >= lo) & (t <= hi)] = lab
    return out


class Ref:
    """NHANES reference: conditional (total strata) and marginal CDFs of lz*, with replicates."""

    def __init__(self, d, col, strata_map):
        self.d = d
        self.v = d[col].to_numpy()
        self.w = d.w.to_numpy()
        self.smap = strata_map
        self.s = strata_map[d.total.to_numpy()]
        self.cond = L.TailRef(self.s, self.v)
        self.marg = L.TailRef(np.zeros(len(d), int), self.v)
        self.mult, self.psu_idx = L.raowu_mult(d, B, rng)

    def wrep(self, b):
        return self.w if b is None else self.mult[b, self.psu_idx] * self.w

    def tails(self, tot, val, b=None, loc=None):
        """Matched and marginal (low, high, mid) for queries (total, lz*)."""
        w = self.wrep(b)
        lc, lm = loc if loc is not None else self.locate(tot, val)
        Fm, F = self.cond.cdf(lc, w)
        cl, ch, cmid = L.tail_probs(Fm, F)
        Fm2, F2 = self.marg.cdf(lm, w)
        ml, mh, _ = L.tail_probs(Fm2, F2)
        return cl, ch, cmid, ml, mh

    def locate(self, tot, val):
        return (self.cond.locate(self.smap[np.asarray(tot)], val),
                self.marg.locate(np.zeros(len(val), int), val))

    def cond_mean(self, b=None):
        """Weighted mean lz* per total stratum."""
        w = self.wrep(b)
        num = np.bincount(self.s, weights=w * self.v, minlength=25)
        den = np.bincount(self.s, weights=w, minlength=25)
        return num / np.where(den > 0, den, np.nan)


def group_defs(corpus):
    """(sample, model, framing, subset, row mask) for every reported group."""
    out = []
    for sample in ["full", "dec_only"]:
        base = np.ones(len(corpus), bool) if sample == "full" else corpus.dec_only.to_numpy()
        for mdl in L.MODELS + ["pooled"]:
            mm = base if mdl == "pooled" else base & (corpus.model == mdl).to_numpy()
            for fr in ["clinical", "narrative", "both"]:
                ff = mm if fr == "both" else mm & (corpus.framing == fr).to_numpy()
                out.append((sample, mdl, fr, "all", ff))
                if sample == "full" and fr == "both":
                    for gg in ["cis", "trans"]:
                        out.append((sample, mdl, fr, gg, ff & (corpus.gender_group == gg).to_numpy()))
    return out


def run_reference(corpus, ref, lzcol, tag, groups, tau_holder):
    """All group statistics against one reference. Returns rows."""
    val = corpus[lzcol].to_numpy()
    tot = corpus.total.to_numpy()
    loc = ref.locate(tot, val)
    persona = (corpus.model + "|" + corpus.profile_id).to_numpy()
    p_u, p_inv = np.unique(persona, return_inverse=True)
    p_model = np.array([x.split("|")[0] for x in p_u])
    # point values
    cl, ch, cmid, ml, mh = ref.tails(tot, val, None, loc)
    cm_pt = ref.cond_mean(None)
    s_idx = ref.smap[tot]
    # replicate values per row are too big to store for B x 28,799; accumulate per group instead
    # precompute resampling indices per group
    gi = []
    for sample, mdl, fr, sub, mask in groups:
        pers_in = np.unique(p_inv[mask])
        strata = p_model[pers_in]
        cols = []
        for s in np.unique(strata):
            pos = pers_in[strata == s]
            cols.append(pos[rng.integers(0, len(pos), size=(B, len(pos)))])
        gi.append(np.concatenate(cols, axis=1))
    stats = {k: np.zeros((len(groups), B)) for k in ["cl", "ch", "cmid", "ml", "mh", "lz", "dlz"]}
    n_p = len(p_u)
    for b in range(B):
        bcl, bch, bcmid, bml, bmh = ref.tails(tot, val, b, loc)
        bcm = np.nan_to_num(ref.cond_mean(b))
        dl = val - bcm[s_idx]
        for k, g in enumerate(groups):
            mask = g[4]
            idx = gi[k][b]
            cnt = np.bincount(p_inv[mask], minlength=n_p)[idx].sum()
            for key, arr in [("cl", bcl), ("ch", bch), ("cmid", bcmid), ("ml", bml), ("mh", bmh), ("lz", val),
                             ("dlz", dl)]:
                stats[key][k, b] = np.bincount(p_inv[mask], weights=arr[mask], minlength=n_p)[idx].sum() / cnt
    rows = []
    for k, (sample, mdl, fr, sub, mask) in enumerate(groups):
        r = dict(reference=tag, sample=sample, model=L.SHORT.get(mdl, mdl), framing=fr, subset=sub,
                 n_draws=int(mask.sum()), n_personas=len(np.unique(p_inv[mask])),
                 mean_lzstar=val[mask].mean(), mean_lzstar_ci95_lo=np.quantile(stats["lz"][k], 0.025),
                 mean_lzstar_ci95_hi=np.quantile(stats["lz"][k], 0.975),
                 nhanes_mean_lzstar_at_model_totals=np.nan_to_num(cm_pt)[s_idx[mask]].mean(),
                 matched_lzstar_diff=(val - np.nan_to_num(cm_pt)[s_idx])[mask].mean(),
                 matched_diff_ci95_lo=np.quantile(stats["dlz"][k], 0.025),
                 matched_diff_ci95_hi=np.quantile(stats["dlz"][k], 0.975),
                 marg_share_below_p5=ml[mask].mean(), marg_share_above_p95=mh[mask].mean(),
                 marg_below_ci95_lo=np.quantile(stats["ml"][k], 0.025),
                 marg_below_ci95_hi=np.quantile(stats["ml"][k], 0.975),
                 marg_above_ci95_lo=np.quantile(stats["mh"][k], 0.025),
                 marg_above_ci95_hi=np.quantile(stats["mh"][k], 0.975),
                 matched_share_below_p5=cl[mask].mean(), matched_share_above_p95=ch[mask].mean(),
                 matched_below_ci95_lo=np.quantile(stats["cl"][k], 0.025),
                 matched_below_ci95_hi=np.quantile(stats["cl"][k], 0.975),
                 matched_above_ci95_lo=np.quantile(stats["ch"][k], 0.025),
                 matched_above_ci95_hi=np.quantile(stats["ch"][k], 0.975),
                 misfit_ratio=cl[mask].mean() / 0.05, misfit_ratio_ci90_lo=np.quantile(stats["cl"][k], 0.05) / 0.05,
                 misfit_ratio_ci90_hi=np.quantile(stats["cl"][k], 0.95) / 0.05,
                 overfit_ratio=ch[mask].mean() / 0.05, overfit_ratio_ci90_lo=np.quantile(stats["ch"][k], 0.05) / 0.05,
                 overfit_ratio_ci90_hi=np.quantile(stats["ch"][k], 0.95) / 0.05,
                 mean_mid_pit=cmid[mask].mean(), mid_pit_ci95_lo=np.quantile(stats["cmid"][k], 0.025),
                 mid_pit_ci95_hi=np.quantile(stats["cmid"][k], 0.975))
        rows.append(r)
    return rows


def read_verdict(r, tau):
    """Each tail read R3-style; the overall reading names the pattern of the two tails.

    misfit tail:  'excess misfit' (less coherent than people) / 'misfit deficit' (fewer atypical
                  vectors than people)
    overfit tail: 'excess overfit' (too regular) / 'overfit deficit' (fewer highly regular vectors)
    overall:      pass when both tails are equivalent; 'fail: compressed' when both tails are in
                  deficit (vectors avoid both ends of the human distribution at the same total);
                  otherwise the single-tail reading; 'undetermined' when no tail is decided."""
    vm = L.verdict(r["misfit_ratio_ci90_lo"], r["misfit_ratio_ci90_hi"], tau)
    vo = L.verdict(r["overfit_ratio_ci90_lo"], r["overfit_ratio_ci90_hi"], tau)
    vm = vm.replace("below", "misfit deficit").replace("above", "excess misfit")
    vo = vo.replace("below", "overfit deficit").replace("above", "excess overfit")
    if vm == "equivalent" and vo == "equivalent":
        return vm, vo, "pass"
    dec = {x for x in [vm, vo] if " or " not in x and x not in ("equivalent", "undetermined")}
    if dec == {"misfit deficit", "overfit deficit"}:
        overall = "fail: compressed (both tails in deficit)"
    elif dec:
        overall = "fail: " + " and ".join(sorted(dec))
    else:
        overall = "undetermined"
    return vm, vo, overall


def main():
    nall = pd.read_csv(os.path.join(L.OUT, "l1_scores_nhanes.csv"))
    sc = pd.read_csv(os.path.join(L.OUT, "l1_scores_corpus.csv"))
    corpus = L.load_corpus()
    assert (sc.total.to_numpy() == corpus.total.to_numpy()).all()
    for c in ["lzstar_weighted", "lzstar_unweighted", "lz_weighted", "theta_weighted"]:
        corpus[c] = sc[c].to_numpy()
    n0 = nall[nall.reference == "2005_2018"].reset_index(drop=True)
    n1 = nall[nall.reference == "2021_2023"].reset_index(drop=True)
    smap = L.total_strata(n0.total.to_numpy(), 100)

    # ---------------- reference description ----------------
    ref_rows = []
    for lab, d in [("2005_2018", n0), ("2021_2023", n1)]:
        for col in ["lzstar_weighted", "lz_weighted", "lzstar_unweighted"]:
            ref_rows.append(dict(reference=lab, statistic=col, band="all", n=len(d),
                                 p05=L.wquantile(d[col].to_numpy(), d.w.to_numpy(), 0.05),
                                 p95=L.wquantile(d[col].to_numpy(), d.w.to_numpy(), 0.95),
                                 mean=np.average(d[col], weights=d.w),
                                 spearman_with_total=spearmanr(d[col], d.total)[0]))
        bands = phq_band(d.total.to_numpy())
        for bl in PHQL:
            m = bands == bl
            v = d.lzstar_weighted.to_numpy()[m]
            ref_rows.append(dict(reference=lab, statistic="lzstar_weighted", band=bl, n=int(m.sum()),
                                 p05=L.wquantile(v, d.w.to_numpy()[m], 0.05),
                                 p95=L.wquantile(v, d.w.to_numpy()[m], 0.95),
                                 mean=np.average(v, weights=d.w.to_numpy()[m]), spearman_with_total=np.nan))
    strata_desc = pd.Series(smap).groupby(smap).apply(lambda s: f"{s.index.min()}-{s.index.max()}")
    for lab in strata_desc:
        ref_rows.append(dict(reference="2005_2018", statistic="total stratum", band=lab,
                             n=int(np.isin(n0.total, range(int(lab.split('-')[0]), int(lab.split('-')[1]) + 1)).sum())))
    pd.DataFrame(ref_rows).to_csv(os.path.join(L.OUT, "l1_personfit_nhanes_ref.csv"), index=False)

    # ---------------- tolerance from NHANES subgroups ----------------
    ref0 = Ref(n0, "lzstar_weighted", smap)
    tol_rows = []
    subs = [("sex", s) for s in ["men", "women"]] + \
           [("race_eth", s) for s in ["Hispanic", "NH White", "NH Black", "NH Asian", "Other"]] + \
           [("pir_band", s) for s in ["PIR<1.3", "PIR1.3-3.5", "PIR>3.5", "PIR missing"]] + \
           [("cycle", s) for s in L.CYCLES_0518]
    loc0 = ref0.locate(n0.total.to_numpy(), n0.lzstar_weighted.to_numpy())
    cl, ch, _, _, _ = ref0.tails(n0.total.to_numpy(), n0.lzstar_weighted.to_numpy(), None, loc0)
    rep = np.zeros((B, 2, len(subs)))
    masks = [(n0[v] == s).to_numpy() for v, s in subs]
    for b in range(B):
        w = ref0.wrep(b)
        bcl, bch, _, _, _ = ref0.tails(n0.total.to_numpy(), n0.lzstar_weighted.to_numpy(), b, loc0)
        for k, m in enumerate(masks):
            rep[b, 0, k] = np.average(bcl[m], weights=w[m])
            rep[b, 1, k] = np.average(bch[m], weights=w[m])
    for k, (v, s) in enumerate(subs):
        m = masks[k]
        tol_rows.append(dict(variable=v, level=s, n=int(m.sum()),
                             misfit_ratio=np.average(cl[m], weights=n0.w[m]) / 0.05,
                             misfit_ci90_lo=np.quantile(rep[:, 0, k], 0.05) / 0.05,
                             misfit_ci90_hi=np.quantile(rep[:, 0, k], 0.95) / 0.05,
                             overfit_ratio=np.average(ch[m], weights=n0.w[m]) / 0.05,
                             overfit_ci90_lo=np.quantile(rep[:, 1, k], 0.05) / 0.05,
                             overfit_ci90_hi=np.quantile(rep[:, 1, k], 0.95) / 0.05))
    # 2021-2023 against the 2005-2018 reference (independent samples)
    loc1 = ref0.locate(n1.total.to_numpy(), n1.lzstar_weighted.to_numpy())
    cl1, ch1, _, _, _ = ref0.tails(n1.total.to_numpy(), n1.lzstar_weighted.to_numpy(), None, loc1)
    mult1, pidx1 = L.raowu_mult(n1, B, rng)
    r1 = np.zeros((B, 2))
    for b in range(B):
        w1 = mult1[b, pidx1] * n1.w.to_numpy()
        bcl, bch, _, _, _ = ref0.tails(n1.total.to_numpy(), n1.lzstar_weighted.to_numpy(), b, loc1)
        r1[b] = [np.average(bcl, weights=w1), np.average(bch, weights=w1)]
    tol_rows.append(dict(variable="cycle", level="L (2021-2023)", n=len(n1),
                         misfit_ratio=np.average(cl1, weights=n1.w) / 0.05,
                         misfit_ci90_lo=np.quantile(r1[:, 0], 0.05) / 0.05, misfit_ci90_hi=np.quantile(r1[:, 0], 0.95) / 0.05,
                         overfit_ratio=np.average(ch1, weights=n1.w) / 0.05,
                         overfit_ci90_lo=np.quantile(r1[:, 1], 0.05) / 0.05, overfit_ci90_hi=np.quantile(r1[:, 1], 0.95) / 0.05))
    tol = pd.DataFrame(tol_rows)
    ratios = np.concatenate([tol.misfit_ratio.to_numpy(), tol.overfit_ratio.to_numpy()])
    spread = float(np.exp(np.abs(np.log(ratios)).max()))
    tau = next(t for t in TAUS if t >= spread)
    tol["max_fold_departure"] = spread
    tol["tau_chosen"] = tau
    tol["positive_control_verdict_at_tau"] = [
        read_verdict(dict(misfit_ratio_ci90_lo=a, misfit_ratio_ci90_hi=b_, overfit_ratio_ci90_lo=c,
                          overfit_ratio_ci90_hi=d_), tau)[2]
        for a, b_, c, d_ in zip(tol.misfit_ci90_lo, tol.misfit_ci90_hi, tol.overfit_ci90_lo, tol.overfit_ci90_hi)]
    tol.to_csv(os.path.join(L.OUT, "l1_personfit_tolerance.csv"), index=False)
    print(tol.round(3).to_string(index=False))
    print(f"person-fit tau = {tau} (largest subgroup fold departure {spread:.3f})")

    # ---------------- corpus against references ----------------
    groups = group_defs(corpus)
    rows = run_reference(corpus, ref0, "lzstar_weighted", "2005_2018 weighted GRM", groups, None)
    main_groups = [g for g in groups if g[0] == "full" and g[3] == "all"]
    ref1 = Ref(n1, "lzstar_weighted", smap)
    rows += run_reference(corpus, ref1, "lzstar_weighted", "2021_2023 weighted GRM", main_groups, None)
    refu = Ref(n0, "lzstar_unweighted", smap)
    rows += run_reference(corpus, refu, "lzstar_unweighted", "2005_2018 unweighted GRM", main_groups, None)
    res = pd.DataFrame(rows)
    vrows = []
    out = []
    for _, r in res.iterrows():
        vm, vo, ov = read_verdict(r, tau)
        out.append((vm, vo, ov))
        if r["subset"] == "all":
            for t in TAUS:
                a, b_, c = read_verdict(r, t)
                vrows.append(dict(reference=r["reference"], sample=r["sample"], model=r["model"],
                                  framing=r["framing"], tau=t, misfit=a, overfit=b_, overall=c))
    res["tau"] = tau
    res["misfit_reading"] = [o[0] for o in out]
    res["overfit_reading"] = [o[1] for o in out]
    res["verdict"] = [o[2] for o in out]
    res.to_csv(os.path.join(L.OUT, "l1_personfit_main.csv"), index=False)
    pd.DataFrame(vrows).to_csv(os.path.join(L.OUT, "l1_personfit_verdicts_tau.csv"), index=False)
    cols = ["model", "framing", "subset", "n_draws", "mean_lzstar", "matched_lzstar_diff", "marg_share_below_p5",
            "marg_share_above_p95", "matched_share_below_p5", "matched_share_above_p95", "misfit_ratio_ci90_lo",
            "misfit_ratio_ci90_hi", "overfit_ratio_ci90_lo", "overfit_ratio_ci90_hi", "mean_mid_pit", "verdict"]
    for (rf, sm), g in res.groupby(["reference", "sample"], sort=False):
        print(f"\n=== {rf} / {sm}")
        print(g[cols].round(3).to_string(index=False))

    # ---------------- person fit against total ----------------
    band_rows = []
    nb = phq_band(n0.total.to_numpy())
    cband = phq_band(corpus.total.to_numpy())
    loc_c = ref0.locate(corpus.total.to_numpy(), corpus.lzstar_weighted.to_numpy())
    ccl, cch, cmid, _, _ = ref0.tails(corpus.total.to_numpy(), corpus.lzstar_weighted.to_numpy(), None, loc_c)
    for bl in PHQL:
        m = nb == bl
        band_rows.append(dict(source="NHANES 2005-2018", band=bl, n=int(m.sum()),
                              weighted_share=np.average(m, weights=n0.w),
                              mean_lzstar=np.average(n0.lzstar_weighted[m], weights=n0.w[m]) if m.any() else np.nan,
                              matched_share_below_p5=0.05, matched_share_above_p95=0.05))
        for mdl in L.MODELS:
            mm = (cband == bl) & (corpus.model == mdl).to_numpy()
            if mm.sum() == 0:
                band_rows.append(dict(source=L.SHORT[mdl], band=bl, n=0))
                continue
            band_rows.append(dict(source=L.SHORT[mdl], band=bl, n=int(mm.sum()),
                                  weighted_share=mm.sum() / (corpus.model == mdl).sum(),
                                  mean_lzstar=corpus.lzstar_weighted[mm].mean(),
                                  matched_share_below_p5=ccl[mm].mean(), matched_share_above_p95=cch[mm].mean(),
                                  mean_mid_pit=cmid[mm].mean()))
    for mdl in L.MODELS:
        mm = (corpus.model == mdl).to_numpy()
        band_rows.append(dict(source=L.SHORT[mdl], band="spearman lz* with total",
                              mean_lzstar=spearmanr(corpus.lzstar_weighted[mm], corpus.total[mm])[0]))
    band_rows.append(dict(source="NHANES 2005-2018", band="spearman lz* with total",
                          mean_lzstar=spearmanr(n0.lzstar_weighted, n0.total)[0]))
    band = pd.DataFrame(band_rows)
    band.to_csv(os.path.join(L.OUT, "l1_personfit_by_band.csv"), index=False)
    print(band.round(3).to_string(index=False))

    # ---------------- item profile at matched totals, flat vectors ----------------
    prof = []
    Xn = n0[L.DPQ].to_numpy(float)
    num = np.zeros((25, 8))
    den = np.zeros(25)
    np.add.at(num, n0.total.to_numpy(), Xn * n0.w.to_numpy()[:, None])
    np.add.at(den, n0.total.to_numpy(), n0.w.to_numpy())
    # totals without NHANES respondents do not occur in the corpus range, but guard anyway
    cmean = num / np.where(den > 0, den, np.nan)[:, None]
    flat_nh = np.average((Xn == Xn[:, [0]]).all(axis=1) & (n0.total > 0), weights=n0.w)
    for mdl in L.MODELS + ["pooled"]:
        mm = np.ones(len(corpus), bool) if mdl == "pooled" else (corpus.model == mdl).to_numpy()
        Xc = corpus.loc[mm, L.ITEMS].to_numpy(float)
        ref_items = cmean[corpus.total.to_numpy()[mm]]
        ok = ~np.isnan(ref_items).any(axis=1)
        diff = (Xc[ok] - ref_items[ok]).mean(axis=0)
        flat = ((Xc == Xc[:, [0]]).all(axis=1) & (Xc.sum(axis=1) > 0)).mean()
        # flat share expected from NHANES at the model's totals
        fl_n = np.zeros(25)
        np.add.at(fl_n, n0.total.to_numpy(), ((Xn == Xn[:, [0]]).all(axis=1) & (n0.total > 0)) * n0.w.to_numpy())
        fl_rate = fl_n / np.where(den > 0, den, np.nan)
        exp_flat = np.nanmean(fl_rate[corpus.total.to_numpy()[mm]])
        r = dict(source=L.SHORT.get(mdl, mdl), n=int(ok.sum()), flat_share=flat,
                 nhanes_flat_share_at_model_totals=exp_flat, nhanes_flat_share_overall=flat_nh)
        for j in range(8):
            r[f"item{j + 1}_minus_nhanes"] = diff[j]
        prof.append(r)
    prof = pd.DataFrame(prof)
    prof.to_csv(os.path.join(L.OUT, "l1_personfit_item_profile.csv"), index=False)
    print(prof.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
