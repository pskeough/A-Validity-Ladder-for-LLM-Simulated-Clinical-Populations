"""Level 1 rebuild, step f: one summary row per model x framing, ranked.

Collects from the CSVs written by 79b to 79e (no new estimation):
  gateway: violation rate, NHANES rate at the model's totals, O/E with 90% interval, verdict
  person fit (total-matched, weighted GRM, NHANES 2005-2018): matched lz* difference, mean mid-PIT,
      misfit and overfit ratios with 90% intervals, verdict
  sensitivities: person-fit and gateway verdicts on December-only rows, against NHANES 2021-2023,
      and (person fit) under the unweighted GRM
  corrected random-routing null: observed / null (own cell, all draws), descriptive
  pattern reuse: cross-persona collision ratio against NHANES at the same totals

Ranking rule (stated before reading the ranks): distance from people on person fit,
D = |share below the matched 5th percentile - 0.05| + |share above the matched 95th - 0.05|
(0 for the reference, at most 1.9); ties within 0.005 broken by |log O/E| on the gateway. A rank is
a distance, not a pass.

Emits analysis/brm/l1_summary.csv.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("l1", os.path.join(os.path.dirname(__file__), "79_l1_lib.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def main():
    gw = pd.read_csv(os.path.join(L.OUT, "l1_gateway_main.csv"))
    pf = pd.read_csv(os.path.join(L.OUT, "l1_personfit_main.csv"))
    nu = pd.read_csv(os.path.join(L.OUT, "l1_gateway_null.csv"))
    ru = pd.read_csv(os.path.join(L.OUT, "l1_pattern_reuse.csv"))
    rows = []
    for mdl in [L.SHORT[m] for m in L.MODELS] + ["pooled"]:
        for fr in ["clinical", "narrative", "both"]:
            g = gw[(gw.model == mdl) & (gw.framing == fr) & (gw.subset == "all")]
            p = pf[(pf.model == mdl) & (pf.framing == fr) & (pf.subset == "all")]
            g0 = g[(g["sample"] == "full") & (g.reference == "2005_2018")].iloc[0]
            gd = g[(g["sample"] == "dec_only") & (g.reference == "2005_2018")].iloc[0]
            g1 = g[(g["sample"] == "full") & (g.reference == "2021_2023")].iloc[0]
            p0 = p[(p["sample"] == "full") & (p.reference == "2005_2018 weighted GRM")].iloc[0]
            pdc = p[(p["sample"] == "dec_only") & (p.reference == "2005_2018 weighted GRM")].iloc[0]
            p1 = p[(p["sample"] == "full") & (p.reference == "2021_2023 weighted GRM")].iloc[0]
            pu = p[(p["sample"] == "full") & (p.reference == "2005_2018 unweighted GRM")].iloc[0]
            n = nu[(nu.model == mdl) & (nu.framing == fr)].iloc[0]
            r = dict(model=mdl, framing=fr, n_draws=int(p0.n_draws), n_elevated=int(g0.n_elevated),
                     gw_rate=g0.violation_rate, gw_rate_ci95_lo=g0.rate_ci95_lo, gw_rate_ci95_hi=g0.rate_ci95_hi,
                     gw_nhanes_rate_at_model_totals=g0.nhanes_rate_at_model_totals, gw_OE=g0.O_over_E,
                     gw_OE_ci90_lo=g0.OE_ci90_lo, gw_OE_ci90_hi=g0.OE_ci90_hi, gw_verdict=g0.verdict,
                     pf_matched_lz_diff=p0.matched_lzstar_diff, pf_matched_lz_diff_ci95_lo=p0.matched_diff_ci95_lo,
                     pf_matched_lz_diff_ci95_hi=p0.matched_diff_ci95_hi, pf_mid_pit=p0.mean_mid_pit,
                     pf_share_below_p5=p0.matched_share_below_p5, pf_share_above_p95=p0.matched_share_above_p95,
                     pf_misfit_ratio=p0.misfit_ratio, pf_misfit_ci90_lo=p0.misfit_ratio_ci90_lo,
                     pf_misfit_ci90_hi=p0.misfit_ratio_ci90_hi, pf_overfit_ratio=p0.overfit_ratio,
                     pf_overfit_ci90_lo=p0.overfit_ratio_ci90_lo, pf_overfit_ci90_hi=p0.overfit_ratio_ci90_hi,
                     pf_marg_below_p5=p0.marg_share_below_p5, pf_marg_above_p95=p0.marg_share_above_p95,
                     pf_verdict=p0.verdict, pf_verdict_dec_only=pdc.verdict, pf_verdict_nhanes2021=p1.verdict,
                     pf_verdict_unweighted_grm=pu.verdict, gw_verdict_dec_only=gd.verdict,
                     gw_OE_dec_only=gd.O_over_E, gw_verdict_nhanes2021=g1.verdict, gw_OE_nhanes2021=g1.O_over_E,
                     null_corrected=n.null_cell_all_exact, obs_over_null_corrected=n.ratio_obs_to_cell_all_exact,
                     null_published=n.published_null)
            rr = ru[(ru.model == mdl) & (ru.framing == fr)]
            r["reuse_ratio"] = rr.ratio.iloc[0] if len(rr) else np.nan
            # the pass rule is person fit; the gateway row is descriptive (random routing reads as
            # "too prototypical" on it, l1_planted_controls.csv, so it cannot carry a verdict alone)
            r["level1_verdict"] = "pass" if r["pf_verdict"] == "pass" else "fail"
            r["D_personfit"] = abs(r["pf_share_below_p5"] - 0.05) + abs(r["pf_share_above_p95"] - 0.05)
            r["abs_log_gw_OE"] = abs(np.log(r["gw_OE"]))
            rows.append(r)
    s = pd.DataFrame(rows)
    for fr in ["clinical", "narrative", "both"]:
        sub = s[(s.framing == fr) & (s.model != "pooled")].copy()
        sub["Dr"] = (sub.D_personfit / 0.005).round()
        order = sub.sort_values(["Dr", "abs_log_gw_OE"]).index
        s.loc[order, "rank_within_framing"] = np.arange(1, len(order) + 1)
    s.to_csv(os.path.join(L.OUT, "l1_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(s[["model", "framing", "rank_within_framing", "D_personfit", "gw_OE", "pf_verdict", "gw_verdict",
             "pf_verdict_dec_only", "pf_verdict_nhanes2021", "pf_verdict_unweighted_grm", "gw_verdict_dec_only",
             "gw_verdict_nhanes2021", "reuse_ratio"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
