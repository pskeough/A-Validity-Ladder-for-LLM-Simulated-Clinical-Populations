"""Level-3 rebuild, step d: the summary tables the report quotes.

Reads analysis/brm/80c_l3_results.csv and writes, under analysis/brm/:
  80d_se_old_vs_new.csv       published (between-cell) SE against the draw-sampling SE, MG mean rows,
                              primary spec, combined framing, and the one-band pass counts under each
  80d_headline.csv            primary spec, PS estimand, combined framing: mean and prevalence rows
                              per model and pooled, with verdicts at every tolerance
  80d_pass_counts.csv         rows passing each tolerance, per spec, corpus, estimand, framing, outcome
                              (40 model x group rows; pooled rows counted separately)
  80d_model_verdicts.csv      intersection-union verdict per model: passes only if all ten group rows pass
  80d_era.csv                 PS residuals on the three NHANES eras, pooled model, combined framing
  80d_income_anchor.csv       income rows on PIR bands and on the persona-string anchors
  80d_december.csv            all rows against December-only, PS mean, primary and 2021-2023
  80d_framing.csv             clinical against narrative, PS mean, primary spec
  80d_ranges.csv              range of the 40 per-model residuals per spec, estimand, framing, outcome
No randomness.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(BASE, "analysis", "brm")
r = pd.read_csv(os.path.join(OUTD, "80c_l3_results.csv"))
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
P = "S1 primary"
TOLS_M = ["pass_0.2SD", "pass_1pt", "pass_2pt", "pass_5pt_benchmark"]
TOLS_P = ["pass_2.5pp", "pass_5pp", "pass_ratio_0.80_1.25"]
NINE = ["White", "Black", "Asian", "Hispanic", "Men", "Women", "Low", "Middle", "High"]


def save(df, name):
    df = df.copy()
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    df.to_csv(os.path.join(OUTD, name), index=False)
    print(f"\n=== {name} ===")
    print(df.to_string(index=False))


# 1. old versus new SE
q = r[(r.spec == P) & (r.corpus == "all") & (r.estimand == "MG") & (r.outcome == "mean") & (r.framing == "combined")].copy()
q["se_sim_ratio_old_over_new"] = q.se_sim_old / q.se_sim
q["se_total_ratio_old_over_new"] = q.se_old / q.se
# the published script read the old SE with t on (cells - 1) df; reproduce that reading exactly
tc_pub = stats.t.ppf(0.95, q.n_cells_old - 1)
q["ci90_hi_old_pubdf"] = q.resid + tc_pub * q.se_old
q["ci90_lo_old_pubdf"] = q.resid - tc_pub * q.se_old
q["pass_5pt_old_se_pubdf"] = (q.ci90_lo_old_pubdf > -5) & (q.ci90_hi_old_pubdf < 5)
cols = ["model", "group", "resid", "n_cells_old", "se_sim_old", "se_sim", "se_sim_ratio_old_over_new", "se_ref_part",
        "se_old", "se", "se_total_ratio_old_over_new", "ci90_hi_old_pubdf", "ci90_hi_old", "ci90_hi",
        "pass_5pt_old_se_pubdf", "pass_5pt_old_se", "pass_5pt_benchmark"]
save(q[cols], "80d_se_old_vs_new.csv")
pm = q[(q.model != "pooled") & q.group.isin(NINE)]
summ = pd.DataFrame([dict(rows=len(pm), pass_5pt_old_se_pubdf=int(pm.pass_5pt_old_se_pubdf.sum()),
                          pass_5pt_old_se=int(pm.pass_5pt_old_se.sum()),
                          pass_5pt_new_se=int(pm.pass_5pt_benchmark.sum()),
                          median_se_sim_ratio=pm.se_sim_ratio_old_over_new.median(),
                          median_se_total_ratio=pm.se_total_ratio_old_over_new.median(),
                          min_se_total_ratio=pm.se_total_ratio_old_over_new.min(),
                          max_se_total_ratio=pm.se_total_ratio_old_over_new.max())])
save(summ, "80d_se_old_vs_new_summary.csv")

# 2. headline table
h = r[(r.spec == P) & (r.corpus == "all") & (r.estimand == "PS") & (r.framing == "combined")]
hm = h[h.outcome == "mean"][["model", "group", "sim", "ref", "resid", "resid_sd_units", "se", "ci90_lo", "ci90_hi",
                             "delta_min"] + TOLS_M]
hp = h[h.outcome == "prev10"][["model", "group", "sim", "ref", "resid", "se", "ci90_lo", "ci90_hi", "ratio",
                               "ratio_ci90_lo", "ratio_ci90_hi"] + TOLS_P]
hp = hp.rename(columns={c: "prev_" + c for c in ["sim", "ref", "resid", "se", "ci90_lo", "ci90_hi"]})
save(hm.merge(hp, on=["model", "group"]), "80d_headline.csv")
hh = hm.merge(hp, on=["model", "group"])
agg = {"resid": ["min", "max"], "delta_min": "max", "prev_resid": ["min", "max"]}
for t in TOLS_M + TOLS_P:
    agg[t] = "sum"
hs = hh.groupby("model").agg(agg)
hs.columns = ["_".join(c) for c in hs.columns]
save(hs.reset_index(), "80d_headline_by_model.csv")

# 3. pass counts
rows = []
for key, g in r.groupby(["spec", "corpus", "estimand", "framing", "outcome"]):
    pmr = g[g.model != "pooled"]
    pool = g[g.model == "pooled"]
    tols = TOLS_M if key[4] == "mean" else TOLS_P
    d = dict(zip(["spec", "corpus", "estimand", "framing", "outcome"], key), n_rows=len(pmr),
             resid_min=pmr.resid.min(), resid_max=pmr.resid.max(), all_positive=bool((pmr.ci90_lo > 0).all()),
             n_positive_sig=int((pmr.ci90_lo > 0).sum()), n_negative_sig=int((pmr.ci90_hi < 0).sum()))
    for t in tols:
        d[t] = int(pmr[t].sum())
        d[t + "_pooled"] = int(pool[t].sum())
        d[t + "_nine_groups"] = int(pmr[pmr.group.isin(NINE)][t].sum())
    rows.append(d)
pc = pd.DataFrame(rows)
save(pc[(pc.framing == "combined")], "80d_pass_counts.csv")
pc.to_csv(os.path.join(OUTD, "80d_pass_counts_all_framings.csv"), index=False)

# 4. model verdicts (intersection-union)
rows = []
for key, g in r[(r.corpus == "all")].groupby(["spec", "estimand", "framing", "outcome", "model"]):
    tols = TOLS_M if key[3] == "mean" else TOLS_P
    d = dict(zip(["spec", "estimand", "framing", "outcome", "model"], key), groups=len(g),
             max_delta_min=g.delta_min.max(), overall_delta_min=float(g[g.group == "Overall"].delta_min.iloc[0]))
    for t in tols:
        d[t + "_all_groups"] = bool(g[t].astype(bool).all())
        d[t + "_overall"] = bool(g[g.group == "Overall"][t].astype(bool).iloc[0])
    rows.append(d)
mv = pd.DataFrame(rows)
save(mv[(mv.spec == P) & (mv.framing == "combined")], "80d_model_verdicts.csv")
mv.to_csv(os.path.join(OUTD, "80d_model_verdicts_all.csv"), index=False)

# 5. era
e = r[(r.corpus == "all") & (r.estimand == "PS") & (r.framing == "combined") & (r.model == "pooled")
      & r.spec.isin([P, "S2 cohabiting as married", "S6 2017-2020 pre-pandemic", "S7 2021-2023"])]
ev = e.pivot_table(index=["outcome", "group"], columns="spec", values=["ref", "resid", "delta_min"]).reset_index()
ev.columns = ["_".join([c for c in col if c]).replace(" ", "_") for col in ev.columns]
ev["resid_shift_2021_vs_primary"] = ev["resid_S7_2021-2023"] - ev["resid_S1_primary"]
ev["resid_shift_2021_vs_same_coding"] = ev["resid_S7_2021-2023"] - ev["resid_S2_cohabiting_as_married"]
save(ev, "80d_era.csv")
# per-model era shift, mean, against 2005-2018 on the same marital coding
S2 = "S2 cohabiting as married"
e2 = r[(r.corpus == "all") & (r.estimand == "PS") & (r.framing == "combined") & (r.outcome == "mean")
       & r.spec.isin([S2, "S7 2021-2023"]) & (r.model != "pooled")]
w = e2.pivot_table(index=["model", "group"], columns="spec", values="resid").reset_index()
w["shift"] = w["S7 2021-2023"] - w[S2]
save(pd.DataFrame([dict(shift_min=w["shift"].min(), shift_max=w["shift"].max(), shift_mean=w["shift"].mean(),
                        resid_min_2021=w["S7 2021-2023"].min(), resid_max_2021=w["S7 2021-2023"].max())]),
     "80d_era_shift_permodel.csv")

# 6. income anchors
ia = r[(r.corpus == "all") & (r.framing == "combined") & (r.model == "pooled") & r.group.isin(["Low", "Middle", "High", "Overall"])
       & r.spec.isin(["S3 2007-2018 PIR (bridge)", "S4 persona-string income", "S5 persona-string, high top-coded"])
       & r.estimand.isin(["PS", "MG"])]
save(ia[["spec", "estimand", "outcome", "group", "n_nhanes", "sim", "ref", "ref_se", "resid", "ci90_lo", "ci90_hi",
         "delta_min"]].sort_values(["estimand", "outcome", "group", "spec"]), "80d_income_anchor.csv")
ia2 = r[(r.corpus == "all") & (r.framing == "combined") & (r.model != "pooled") & r.group.isin(["Low", "Middle", "High"])
        & r.spec.isin(["S3 2007-2018 PIR (bridge)", "S4 persona-string income"]) & (r.estimand == "PS") & (r.outcome == "mean")]
save(ia2.pivot_table(index=["model", "group"], columns="spec", values="resid").reset_index(), "80d_income_anchor_permodel.csv")

# 7. December-only
dd = r[(r.estimand == "PS") & (r.outcome == "mean") & (r.framing.isin(["clinical", "combined"]))
       & r.spec.isin([P, "S7 2021-2023"])]
dv = dd.pivot_table(index=["spec", "framing", "model", "group"], columns="corpus", values="resid").reset_index()
dv["diff_dec_minus_all"] = dv["dec_only"] - dv["all"]
miss = dd[dd.corpus == "dec_only"][["spec", "framing", "model", "group", "n_cells_missing"]]
dv = dv.merge(miss, on=["spec", "framing", "model", "group"])
save(dv, "80d_december.csv")
save(pd.DataFrame([dict(max_abs_diff=dv.diff_dec_minus_all.abs().max(),
                        max_abs_diff_no_missing=dv[dv.n_cells_missing == 0].diff_dec_minus_all.abs().max(),
                        rows_with_missing_cells=int((dv.n_cells_missing > 0).sum()))]), "80d_december_summary.csv")

# 8. framing
fv = r[(r.spec == P) & (r.corpus == "all") & (r.estimand == "PS") & (r.framing.isin(["clinical", "narrative"]))]
fv = fv.pivot_table(index=["outcome", "model", "group"], columns="framing", values="resid").reset_index()
fv["narrative_minus_clinical"] = fv["narrative"] - fv["clinical"]
save(fv, "80d_framing.csv")
save(fv.groupby(["outcome", "model"]).narrative_minus_clinical.agg(["min", "max", "mean"]).reset_index(),
     "80d_framing_summary.csv")

# 9. ranges
rg = r[(r.corpus == "all") & (r.model != "pooled") & (r.outcome == "mean")].groupby(["spec", "estimand", "framing"]).agg(
    resid_min=("resid", "min"), resid_max=("resid", "max"), sd_units_min=("resid_sd_units", "min"),
    sd_units_max=("resid_sd_units", "max"), delta_min_min=("delta_min", "min")).reset_index()
save(rg, "80d_ranges.csv")

# 10. resolution: the smallest symmetric band a row could pass if its residual were exactly zero
rs = r[(r.corpus == "all") & (r.outcome == "mean") & (r.framing == "combined")].copy()
rs["resolution"] = stats.t.ppf(0.95, rs.df) * rs.se
save(rs.groupby(["spec", "estimand"]).resolution.agg(["median", "max"]).reset_index(), "80d_resolution.csv")

# 11. mean and prevalence disagree in sign: level too high, tail too thin
mp = r[(r.corpus == "all") & (r.framing == "combined") & (r.estimand == "PS")]
w2 = mp.pivot_table(index=["spec", "model", "group"], columns="outcome", values=["ci90_lo", "ci90_hi"]).reset_index()
w2.columns = ["_".join([c for c in col if c]) for col in w2.columns]
w2["mean_high_prev_low"] = (w2.ci90_lo_mean > 0) & (w2.ci90_hi_prev10 < 0)
save(w2.groupby("spec").mean_high_prev_low.agg(["sum", "size"]).reset_index(), "80d_sign_discordance.csv")
save(w2[(w2.spec == P) & w2.mean_high_prev_low][["model", "group"]], "80d_sign_discordance_primary.csv")
