"""Level 2 under R3 on the rebuilt reference: both estimands, clustered SEs, per-scope families,
and the "reference too imprecise" stop.

R3 (paper_brm/level2_rule/LEVEL2_RULE.md, adopted 25 Sep; scripts/74) reads the ratio
rho = simulated gap / population gap against four boundaries. Here:

  boundaries   -0.25, 0.25, 0.75, 1.25  (primary)     -0.25, 0.25, 0.80, 1.25  (symmetric kept band)
  regions      reversed | missing | attenuated | kept | steepened   ("attenuated" replaces
               "flattened", which collides with Wang et al.'s "group flattening")
  test         at each boundary c, one-sided, t_c = (g - c p) / sqrt(se_g^2 + c^2 se_p^2)
               (cov = 0: the simulated and NHANES gaps come from separate samples), referred to
               t with Welch-Satterthwaite df from the simulated df and the NHANES design df. The
               interval reported is the inversion of the same test (the Fieller set), found
               numerically.
  naming       one region -> its name; two adjacent -> "A or B"; more -> undetermined.
  multiplicity each scope (pooled; each model; each framing within them) is a family of seven
               contrasts; Bonferroni: every one-sided test at .05 / 7, i.e. a 98.57% interval.
               The unadjusted 90% reading is carried in every row as a secondary column.

Preconditions, checked in this order before R3 is read:
  1. no population gap: the population gap's own interval at the family level
     (p -/+ t_{1-a}(df_p) se_p) covers zero.
  2. reference too imprecise: with a perfectly precise simulated gap g the ratio interval is
     [g / p_hi, g / p_lo] (p oriented positive), where [p_lo, p_hi] is the population gap's
     family-level interval. It spans a multiplicative factor F = p_hi / p_lo = (1 + k) / (1 - k),
     where k = t_{1-a}(df_p) se_p / |p| is that interval's relative half-width.
       Primary (`verdict`, frozen 2026-10-02, LADDER_SPEC.md): the stop fires when k > 1/4 (the
       reference cannot confine any true ratio to the kept region) and the R3 reading does not
       already exclude kept; a reading that excludes kept stands, so an imprecise reference can
       still return a failure but never blocks a pass.
       Conditional (`verdict_conditional`, the rule before the freeze): the stop fires when [g / p_hi, g / p_lo], built from the population
       interval and the simulated point estimate with no simulation error, covers three or more
       regions. The simulation's own error can only widen the interval, so the stop fires only
       where R3 would say undetermined anyway; it names the reference, not the simulation, as
       the reason. It never overrides a determinate verdict.
       Population-only (`verdict_popstop`): the narrowest bounded region is kept (factor
       1.25/0.75 = 5/3; 1.25/0.80 = 1.5625 for the symmetric band); attenuated spans a factor of
       3 and the other regions are unbounded or contain zero. So every true ratio can be confined
       to at most two regions if and only if F <= 5/3, i.e. k <= 1/4 (k <= 0.2195 symmetric).
       This version fires on k alone, before the simulation is seen, in every scope. It is
       reported, not used for the headline, because it also stops contrasts whose intervals sit
       wholly inside one region (see L2.md); the primary rule keeps those verdicts.
     The R3 reading underneath either stop is kept in `r3_unstopped`.

Inputs: analysis/brm/l2_reference.csv (78a), analysis/brm/l2_sim_gaps.csv (78b).
Engine receipt: with 74's df convention (pairs' df for both components) and 90% intervals, the
engine reproduces all 105 verdict_r3 values and the 7 pooled verdict_r3_bonf values of
analysis/level2_ratio_interval.csv.
Emits analysis/brm/l2_verdicts.csv (every configuration), l2_headline.csv, l2_stop.csv,
l2_verdict_summary.csv. Closed form.
"""
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(BASE, "analysis", "brm")
ALPHA = 0.05
FAMILY = 7
LABELS = ["reversed", "missing", "attenuated", "kept", "steepened"]
BANDS = {"asym": np.array([-0.25, 0.25, 0.75, 1.25]), "sym": np.array([-0.25, 0.25, 0.80, 1.25])}
CONTRASTS = ["Black minus White", "Hispanic minus White", "Asian minus White", "Women minus Men",
             "Low minus High SES", "Middle minus High SES", "Low minus Middle SES"]
MODELS = ["openai/gpt-4o-mini", "google/gemini-3-flash-preview", "deepseek/deepseek-chat-v3",
          "z-ai/glm-4.7"]
SHORT = {"reversed": "r", "missing": "m", "attenuated": "a", "kept": "k", "steepened": "s",
         "undetermined": "u", "no population gap": "N", "reference too imprecise": "X"}


def tcrit(a, df):
    df = float(df)
    return float(stats.t.ppf(1 - a, df if np.isfinite(df) and df > 0 else 1e9))


def sdf(se_g, df_g, se_p, df_p, c):
    """Welch-Satterthwaite df for se_g^2 + c^2 se_p^2."""
    vg, vp = se_g ** 2, (c * se_p) ** 2
    if vp == 0:
        return df_g
    den = (vg ** 2 / df_g if np.isfinite(df_g) else 0) + (vp ** 2 / df_p if np.isfinite(df_p) else 0)
    return (vg + vp) ** 2 / den if den > 0 else np.inf


def r3(g, se_g, df_g, p, se_p, df_p, a, bounds, df_mode="satterthwaite"):
    """One row. Returns dict with label, stop flags, interval."""
    s = -1.0 if p < 0 else 1.0
    g, p = g * s, p * s
    tp = tcrit(a, df_p)
    p_lo, p_hi = p - tp * se_p, p + tp * se_p

    def df_at(c):
        return df_g if df_mode == "pairs" else sdf(se_g, df_g, se_p, df_p, c)

    def stat(c):
        return (g - c * p) / np.sqrt(se_g ** 2 + (c * se_p) ** 2)

    above = sum(stat(c) > tcrit(a, df_at(c)) for c in bounds)
    below = sum(stat(c) < -tcrit(a, df_at(c)) for c in bounds)
    lo, hi = above, len(bounds) - below
    if lo == hi:
        lab = LABELS[lo]
    elif hi == lo + 1:
        lab = f"{LABELS[lo]} or {LABELS[hi]}"
    else:
        lab = "undetermined"
    # interval: accepted set of c, by grid + root refinement
    grid = np.linspace(-20, 20, 8001)
    if df_mode == "pairs":
        dfg = np.full(grid.shape, float(df_g))
    else:
        vg, vp = se_g ** 2, (grid * se_p) ** 2
        den = (vg ** 2 / df_g if np.isfinite(df_g) else 0) + \
              (vp ** 2 / df_p if np.isfinite(df_p) else 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            dfg = np.where(den > 0, (vg + vp) ** 2 / den, 1e9)
    tg = stats.t.ppf(1 - a, np.where(np.isfinite(dfg), dfg, 1e9))
    acc = np.abs(stat(grid)) <= tg
    if acc.any():
        i0, i1 = np.argmax(acc), len(acc) - 1 - np.argmax(acc[::-1])
        f = lambda c: abs(stat(c)) - tcrit(a, df_at(c))  # noqa: E731
        ci_lo = grid[0] if i0 == 0 else optimize.brentq(f, grid[i0 - 1], grid[i0])
        ci_hi = grid[-1] if i1 == len(grid) - 1 else optimize.brentq(f, grid[i1], grid[i1 + 1])
        unbounded = i0 == 0 or i1 == len(grid) - 1
        split = not acc[i0:i1 + 1].all()
    else:
        ci_lo = ci_hi = np.nan
        unbounded = split = False
    k = tp * se_p / p
    W = bounds[3] / bounds[2]
    kmax = (W - 1) / (W + 1)
    # conditional: exact g, ratio interval [g/p_hi, g/p_lo]
    if p_lo > 0:
        e_lo, e_hi = sorted([g / p_hi, g / p_lo])
        n_reg = int(np.sum((bounds > e_lo) & (bounds < e_hi))) + 1
    else:
        n_reg = 99
    if p_lo <= 0:
        final = "no population gap"
    elif n_reg >= 3:
        final = "reference too imprecise"
    else:
        final = lab
    if p_lo <= 0:
        final_pop = "no population gap"
    elif k > kmax:
        final_pop = "reference too imprecise"
    else:
        final_pop = lab
    # frozen rule (LADDER_SPEC.md): a contrast whose reference cannot certify 'kept' (k > kmax) is
    # stopped unless its verdict already excludes 'kept', in which case the verdict stands
    if p_lo <= 0:
        final_frozen = "no population gap"
    elif k > kmax and (lab == "undetermined" or "kept" in lab.split(" or ")):
        final_frozen = "reference too imprecise"
    else:
        final_frozen = lab
    return dict(ratio=g / p, ci_lo=ci_lo, ci_hi=ci_hi, ci_unbounded=unbounded, ci_split=split,
                region_lo=LABELS[lo] if lo < 5 else "", region_hi=LABELS[hi] if hi < 5 else "",
                r3_unstopped=lab, pop_ci_lo=s * p_lo if s > 0 else s * p_hi,
                pop_ci_hi=s * p_hi if s > 0 else s * p_lo, k_rel_halfwidth=k, k_max=kmax,
                F_pop=(p_hi / p_lo) if p_lo > 0 else np.inf, stop_population=bool(k > kmax),
                stop_conditional=bool(n_reg >= 3), no_population_gap=bool(p_lo <= 0),
                verdict=final_frozen, verdict_conditional=final, verdict_popstop=final_pop)


def code(v):
    return "/".join(SHORT[x] for x in v.split(" or ")) if " or " in v else SHORT[v]


# ---------------------------------------------------------------------------------- receipt
def engine_receipt():
    d = pd.read_csv(os.path.join(BASE, "analysis", "level2_ratio_interval.csv"))
    fails, n = 0, 0
    for r in d.itertuples():
        out = r3(r.simulated, r.se, r.df, r.population, r.pop_se, r.df, ALPHA,
                 np.array([-0.25, 0.25, 0.75, 1.25]), df_mode="pairs")
        want = r.verdict_r3.replace("flattened", "attenuated")
        fails += out["r3_unstopped"] != want
        n += 1
        if r.scope_type == "pooled":
            out = r3(r.simulated, r.se, r.df, r.population, r.pop_se, r.df, ALPHA / 7,
                     np.array([-0.25, 0.25, 0.75, 1.25]), df_mode="pairs")
            fails += out["r3_unstopped"] != r.verdict_r3_bonf.replace("flattened", "attenuated")
            n += 1
    return n, fails


# ------------------------------------------------------------------------------------- main
def main():
    ref = pd.read_csv(os.path.join(OUTD, "l2_reference.csv"))
    sim = pd.read_csv(os.path.join(OUTD, "l2_sim_gaps.csv"))
    sim["race_set"] = sim.race_set.fillna("n/a")          # pandas reads "n/a" as missing
    n, fails = engine_receipt()
    print(f"engine receipt: {n} verdicts of analysis/level2_ratio_interval.csv, {fails} mismatches")
    if fails:
        raise SystemExit("engine does not reproduce script 74")

    def refrow(contrast, estimand, variant):
        x = ref[(ref.contrast == contrast) & (ref.estimand == estimand) & (ref.variant == variant)]
        if not len(x):
            if estimand == "marginal":                    # marginal has one version per contrast
                x = ref[(ref.contrast == contrast) & (ref.estimand == "marginal") &
                        (ref.variant == "primary")]
            else:
                x = ref[(ref.contrast == contrast) & (ref.estimand == estimand) &
                        (ref.variant == "primary")]
        return x.iloc[0]

    # analysis configurations: each changes one thing from the headline
    H = dict(corpus="all", panel="four models", race_set="four", framing="combined",
             ref_variant="primary", se_type="stratified", ref_se="taylor", band="asym",
             mult="bonferroni7")
    configs = [("headline", {})]
    configs += [(f"framing_{f}", dict(framing=f)) for f in ("clinical", "narrative")]
    configs += [("band_symmetric", dict(band="sym")), ("unadjusted_90", dict(mult="none")),
                ("se_unstratified", dict(se_type="unstratified")),
                ("se_naive_66", dict(se_type="naive")),
                ("ref_se_bootstrap", dict(ref_se="boot")),
                ("five_race", dict(race_set="five", ref_variant="five_race")),
                ("dec_only", dict(corpus="dec_only")),
                ("dec_only_clinical", dict(corpus="dec_only", framing="clinical")),
                ("without_glm", dict(panel="without GLM-4.7")),
                ("without_glm_unadjusted_90", dict(panel="without GLM-4.7", mult="none"))]
    configs += [(f"ref_{v}", dict(ref_variant=v)) for v in
                ("mar_all", "mar_never", "mar_drop1819", "no_marital", "income_alt",
                 "asian_era")]

    rows = []
    for cname, change in configs:
        cfg = dict(H, **change)
        for estimand in ("marginal", "standardised"):
            if cfg["ref_variant"] in ("mar_all", "mar_never", "mar_drop1819", "no_marital",
                                      "five_race") and estimand == "marginal":
                continue                                    # these variants touch the std only
            for contrast in CONTRASTS:
                if cfg["ref_variant"] == "asian_era" and contrast != "Asian minus White":
                    continue
                if cfg["ref_variant"] == "income_alt" and estimand == "marginal" and \
                        "SES" not in contrast:
                    continue
                rr = refrow(contrast, estimand, cfg["ref_variant"])
                rs = "n/a" if contrast in CONTRASTS[:3] else cfg["race_set"]
                ss = sim[(sim.corpus == cfg["corpus"]) & (sim.panel == cfg["panel"]) &
                         (sim.race_set == rs) & (sim.contrast == contrast) &
                         (sim.framing == cfg["framing"])]
                se_p = rr.se_taylor if cfg["ref_se"] == "taylor" else rr.se_boot
                a = ALPHA / FAMILY if cfg["mult"] == "bonferroni7" else ALPHA
                for s in ss.itertuples():
                    if cfg["se_type"] == "stratified":
                        se_g, df_g = s.se, s.df
                    elif cfg["se_type"] == "unstratified":
                        se_g, df_g = s.se_unstratified, s.df_unstratified
                    else:
                        if s.scope != "pooled":
                            continue
                        se_g, df_g = s.se_naive_66, s.df_naive_66
                    if cfg["se_type"] != "stratified" and s.scope != "pooled":
                        continue                            # identical to the headline
                    out = r3(s.simulated, se_g, df_g, rr.estimate, se_p, rr.df, a,
                             BANDS[cfg["band"]])
                    unadj = r3(s.simulated, se_g, df_g, rr.estimate, se_p, rr.df, ALPHA,
                               BANDS[cfg["band"]])
                    rows.append(dict(analysis=cname, **cfg, estimand=estimand, contrast=contrast,
                                     scope=s.scope, n_pairs=s.n_pairs, simulated=s.simulated,
                                     se_sim=se_g, df_sim=df_g, population=rr.estimate,
                                     se_pop=se_p, df_pop=rr.df, alpha_one_sided=a,
                                     interval_level=1 - 2 * a, **out,
                                     verdict_unadjusted_90=unadj["verdict"],
                                     r3_unstopped_90=unadj["r3_unstopped"]))
    res = pd.DataFrame(rows)
    for c in ("ratio", "ci_lo", "ci_hi", "pop_ci_lo", "pop_ci_hi", "k_rel_halfwidth", "k_max",
              "F_pop", "simulated", "se_sim", "df_sim", "population", "se_pop",
              "alpha_one_sided", "interval_level"):
        res[c] = res[c].astype(float).round(4)
    res["code"] = res.verdict.map(code)
    res["code_unstopped"] = res.r3_unstopped.map(code)
    res.to_csv(os.path.join(OUTD, "l2_verdicts.csv"), index=False)

    # ------------------------------------------------------------------------ headline table
    res["code_popstop"] = res.verdict_popstop.map(code)

    def fci(y):
        if y.ci_unbounded:
            return "unbounded"
        return f"{y.ci_lo:.2f} to {y.ci_hi:.2f}" + (" (split)" if y.ci_split else "")

    res.to_csv(os.path.join(OUTD, "l2_verdicts.csv"), index=False)
    hd = res[res.analysis == "headline"]
    wide = []
    for est in ("standardised", "marginal"):
        for c in CONTRASTS:
            x = hd[(hd.estimand == est) & (hd.contrast == c)].set_index("scope")
            r = x.loc["pooled"]
            row = dict(estimand=est, contrast=c, population=r.population, pop_se=r.se_pop,
                       k=r.k_rel_halfwidth, pooled_sim=r.simulated, pooled_se=r.se_sim,
                       pooled_ratio=r.ratio, pooled_ci=fci(r), pooled_verdict=r.verdict,
                       pooled_unstopped=r.r3_unstopped, pooled_popstop=r.verdict_popstop,
                       pooled_unadjusted_90=r.verdict_unadjusted_90)
            for mdl in MODELS:
                y = x.loc[mdl]
                row[mdl.split("/")[-1]] = f"{y.ratio:.2f} [{fci(y)}] {y.code}"
            wide.append(row)
    pd.DataFrame(wide).to_csv(os.path.join(OUTD, "l2_headline.csv"), index=False)

    # ------------------------------------------------------------------------------ stop table
    st = res[(res.analysis.isin(["headline", "unadjusted_90", "band_symmetric", "ref_se_bootstrap"]))
             & (res.scope == "pooled")][["analysis", "estimand", "contrast", "population",
                                         "se_pop", "df_pop", "pop_ci_lo", "pop_ci_hi",
                                         "k_rel_halfwidth", "k_max", "F_pop",
                                         "no_population_gap", "stop_population",
                                         "stop_conditional", "r3_unstopped", "verdict",
                                         "verdict_popstop"]]
    st.to_csv(os.path.join(OUTD, "l2_stop.csv"), index=False)

    # ----------------------------------------------------------------------- summary per config
    summ = []
    for (an, est), g in res.groupby(["analysis", "estimand"]):
        for scope, gg in g.groupby("scope"):
            summ.append(dict(analysis=an, estimand=est, scope=scope,
                             verdicts="; ".join(f"{c.split(' minus ')[0]}-"
                                                f"{c.split(' minus ')[1].replace(' SES', '')}: "
                                                f"{v}" for c, v in zip(gg.contrast, gg.verdict)),
                             n_single_name=int(gg.verdict.isin(LABELS).sum()),
                             n_two_name=int(gg.verdict.str.contains(" or ").sum()),
                             n_stopped=int(gg.verdict.isin(["reference too imprecise",
                                                            "no population gap"]).sum()),
                             verdicts_popstop="; ".join(gg.verdict_popstop.map(code)),
                             verdicts_unadjusted_90="; ".join(
                                 gg.verdict_unadjusted_90.map(code))))
    pd.DataFrame(summ).to_csv(os.path.join(OUTD, "l2_verdict_summary.csv"), index=False)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 60)
    print(pd.DataFrame(wide).to_string(index=False))
    print()
    print(st[st.analysis == "headline"].to_string(index=False))
    for an in [c for c, _ in configs if c != "headline"]:
        x = res[(res.analysis == an) & (res.scope == "pooled")]
        print(f"\n{an}: pooled")
        print(x[["estimand", "contrast", "simulated", "se_sim", "population", "se_pop", "ratio",
                 "ci_lo", "ci_hi", "verdict", "verdict_popstop", "verdict_unadjusted_90"]]
              .to_string(index=False))
    print("\nwritten l2_verdicts.csv, l2_headline.csv, l2_stop.csv, l2_verdict_summary.csv")


if __name__ == "__main__":
    main()
