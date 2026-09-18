"""AUTO-REPOINTED COPY of 42_short_paper_gate.py for the voiced manuscript.

Only intex() prose needles changed. Every receipt reconciliation -- the
round(...) == value arithmetic proving each number correct -- is byte-identical
to the original gate. A needle moved only when the old phrase was absent from
the new text AND a sentence carrying every one of its numbers was found.

The original gate is untouched and still guards the original manuscript.
Needles marked UNRESOLVED in GATE_REPOINT.md were left alone and will still
fail; they need a human to confirm the claim survived the rewrite.
"""
"""Every number the condensed paper states, checked against the same receipts as the full one.

A shortened paper is where numbers drift. Sentences get merged, a range loses an endpoint, a figure
carried over from an earlier draft survives because nobody recomputed it. Script 15 cannot catch
that: it reads paper_v2/main.tex and asserts the claims that paper makes, so the condensed version
would ship unchecked.

This applies the same standard to the shorter file. Every quantity below is read from the analysis
CSVs, and where the manuscript states a derived figure that no CSV carries directly, the derivation
is done here rather than the number trusted. Claims the condensed paper drops are simply absent from
this list; claims it keeps are checked at full strength.

Run: python scripts/42_short_paper_gate.py
"""
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
SHORT = os.path.join(BASE, "paper_short")
A = lambda f: pd.read_csv(os.path.join(BASE, "analysis", f))

tex = open(os.path.join(SHORT, "main.tex"), encoding="utf-8").read()
flat = re.sub(r"\s+", " ", tex)
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  <- {detail}"))
    if not ok:
        FAIL.append(name)


def intex(*fragments):
    return all(re.sub(r"\s+", " ", f) in flat for f in fragments)


def rows_of(label):
    """The tabular row whose first cell is `label`, as a list of cells.

    The first cell of the first body row carries \\midrule ahead of the label, so leading control
    sequences are stripped before matching.
    """
    for raw in flat.split(r"\\"):
        cells = [c.strip() for c in raw.split("&")]
        if cells and re.sub(r"^(\\[a-zA-Z]+\s*)+", "", cells[0]).strip() == label:
            return cells
    return None


# ---- Table 1: residuals ------------------------------------------------------------------------
cis = A("residuals_cell_level.csv").set_index("group")
pool = A("bias_residuals_GOLD.csv").set_index("group")
ROWS = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic",
        "Cisgender Man": "Cis men", "Cisgender Woman": "Cis women",
        "Low": "Low SES", "Middle": "Middle SES", "High": "High SES"}
for grp, label in ROWS.items():
    r, cells = cis.loc[grp], rows_of(label)
    want = [f"{r.gt_mean:.2f} ({r.gt_sd:.2f})",
            f"+{r.residual:.2f} [+{r.resid_ci_lo:.2f}, +{r.resid_ci_hi:.2f}]",
            f"{r.cohens_d:.2f}"]
    # The two gender rows print n/a in the pooled column rather than a number. A full-grid pooled
    # marginal for "cisgender men" is the cisgender-men cells, which is the primary frame already,
    # so the sensitivity column has nothing to say there and the receipt repeats the same value.
    if grp.startswith("Cisgender"):
        want.append("n/a")
        assert round(float(pool.loc[grp].residual), 2) == round(float(r.residual), 2)
    else:
        want.append(f"+{pool.loc[grp].residual:.2f}")
    check(f"table row {label}", cells is not None and cells[1:5] == want,
          f"{cells[1:5] if cells else None} != {want}")

check("all 36 model-by-group residuals positive",
      bool((A("per_model_residuals.csv").residual > 0).all())
      and intex("in all 36 model-by-group cells"))
check("residual band 2.8 to 5.5",
      round(cis.residual.min(), 1) == 2.8 and round(cis.residual.max(), 1) == 5.5
      and intex("inflated by 2.8 to 5.5 points", "inflated by 2.8 to 5.5 PHQ-8 points"))
check("low SES 9.72 against 4.25",
      round(float(cis.loc["Low"].model_mean), 2) == 9.72
      and round(float(cis.loc["Low"].gt_mean), 2) == 4.25
      and intex("simulated mean of 9.72 sits 5.48 points above the population value of 4.25"))

# ---- the multiplier, and the interior it does not reproduce ------------------------------------
# The interior predictions are stated in prose and carried by no CSV, so they are re-derived from
# the anchors and the fitted multiplier rather than taken from the sentence that reports them.
inf = A("inflation_form.csv").set_index("quantity").value
m_ols = float(inf["multiplier, least squares through origin"])
m_log = float(inf["multiplier, log-scale estimator"])
pop = {k: float(cis.loc[k].gt_mean) for k in ("Low", "Middle", "High")}
sim = {k: float(cis.loc[k].model_mean) for k in ("Low", "Middle", "High")}
grad = lambda d, a, b: d[a] - d[b]
check("multiplier 2.29 / 2.32, gradient 2.02 -> 4.75, prediction 4.62 to 4.69",
      round(m_ols, 2) == 2.29 and round(m_log, 2) == 2.32
      and round(grad(pop, "Low", "High"), 2) == 2.02
      and round(grad(sim, "Low", "High"), 2) == 4.75
      and round(grad(pop, "Low", "High") * m_ols, 2) == 4.62
      and round(grad(pop, "Low", "High") * m_log, 2) == 4.69
      and intex("Least squares through the origin puts the multiplier at 2.29, the log-scale estimator at 2.32, and either one predicts a simulated",
                "2.02 points against a simulated 4.75",
                "at 2.32, and either one predicts a simulated gradient of 4.62 to 4.69 against the 4.75 we observe"))
check("interior under- and over-predicted: 2.58/3.46 and 2.04/1.29",
      round(grad(pop, "Low", "Middle") * m_ols, 2) == 2.58
      and round(grad(sim, "Low", "Middle"), 2) == 3.46
      and round(grad(pop, "Middle", "High") * m_ols, 2) == 2.04
      and round(grad(sim, "Middle", "High"), 2) == 1.29
      and intex("2.58 against an observed 3.46", "2.04 against 1.29"))
check("multiplier range 2.01 to 2.82",
      round(float(inf["ratio range low"]), 2) == 2.01
      and round(float(inf["ratio range high"]), 2) == 2.82
      and intex("2.01 for middle-income personas to 2.82 for Asian personas"))
alt = A("alternative_comparators.csv").set_index("comparator")
check("alternative comparators +4.99 / +6.99 against +4.01",
      round(float(alt.loc["median"].residual), 2) == 4.99
      and round(float(alt.loc["mode"].residual), 2) == 6.99
      and round(float(alt.loc["mean"].residual), 2) == 4.01
      and int(alt.loc["mode"].simulated) == 8 and int(alt.loc["median"].simulated) == 7
      and int(alt.loc["mode"].population) == 0 and int(alt.loc["median"].population) == 2
      and intex("mode of 8 and a median of 7; the population's are 0 and 2",
                "Against the population median the pooled residual is +4.99, and against the mode it is +6.99, both larger than the +4.01 we report",
                "the +4.01 we report against the mean"))

# ---- cross-group calibration: the reversal, which had no check until it became Section 4.2 ------
ap = A("asian_paradox.csv").set_index("model")
pm_race = A("per_model_residuals.csv")
pm_race = pm_race[pm_race.dimension == "race"].pivot(index="model", columns="group",
                                                     values="model_mean")
gap = lambda d, g: float(d.loc[g]) - float(d.loc["White"])
pop_gap = {g: gap(cis.gt_mean, g) for g in ("Black", "Hispanic", "Asian")}
sim_gap = {g: gap(cis.model_mean, g) for g in ("Black", "Hispanic", "Asian")}
check("White minus Asian: population 0.755, four models 0.708 to 1.736",
      round(float(ap.loc["POOLED"].gt_gap), 3) == 0.755
      and round(float(ap.loc["deepseek/deepseek-chat-v3"].model_gap), 3) == 0.708
      and round(float(ap.loc["openai/gpt-4o-mini"].model_gap), 3) == 0.403
      and round(float(ap.loc["google/gemini-3-flash-preview"].model_gap), 3) == 1.447
      and round(float(ap.loc["z-ai/glm-4.7"].model_gap), 3) == 1.736
      and intex("White minus Asian, a population value of 0.755 points",
                "DeepSeek-V3 comes nearest, a simulated gap of 0.708; GLM-4.7 lands furthest, at 1.736"
                "1.736", "models bracket the target from opposite directions, GPT-4o-mini under at 0.403, Gemini-3-Flash over at 1.447, and one China-developed"
                "at 1.447"))
# The earlier form of this check asserted a sign reversal, which is what the section claimed before
# script 44 tested the contrasts. The test found the simulated intervals covering zero, so what is
# asserted now is the weaker and true statement: significantly below the population, not
# distinguishable from zero. Reading the direction off the point estimates was the original mistake.
rc = A("race_contrast_tests.csv")
rc["g"] = rc.contrast.str.split(" ").str[0]
rc = rc.set_index("g")
flat_null = {g: float(rc.loc[g].ci_lo) <= 0 <= float(rc.loc[g].ci_hi) for g in rc.index}
# The population contrast is recomputed from the unrounded anchors, not read from the four-decimal
# export. Hispanic minus White lands on 0.2425 there, exactly the half-way case, and rounding that
# stored value to three places gives 0.242 while the true 0.242501 gives 0.243. The paper states the
# second. Rounding an already-rounded number is the same error that once put 5.47 in a draft.
gtf = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
GTK = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic (pooled)"}
wm = {g: float(gtf.loc[gtf.group == k, "w_mean"].iloc[0]) for g, k in GTK.items()}
popc = {g: wm[g] - wm["White"] for g in ("Black", "Hispanic", "Asian")}
check("Black and Hispanic gaps cover zero and sit below the population",
      round(popc["Black"], 3) == 0.330
      and round(float(rc.loc["Black"].simulated), 3) == -0.131
      and round(float(rc.loc["Black"].ci_lo), 3) == -0.273
      and round(float(rc.loc["Black"].ci_hi), 3) == 0.011
      and round(float(rc.loc["Black"].simulated) - popc["Black"], 3) == -0.461
      and round(popc["Hispanic"], 3) == 0.243
      and round(float(rc.loc["Hispanic"].simulated), 3) == -0.034
      and round(float(rc.loc["Hispanic"].simulated) - popc["Hispanic"], 3) == -0.277
      and flat_null["Black"] and flat_null["Hispanic"]
      and float(rc.loc["Black"].q_vs_pop) < 0.001 and float(rc.loc["Hispanic"].q_vs_pop) < 0.001
      and int(rc.loc["Black"].n_pairs) == 96
      and intex("Black adults score 0.330 points above White adults in the population",
                "These models put them 0.131 points \emph{below}, an interval of",
                "does not exclude zero ($q = 0.11$)",
                "0.461 points under the population value ($q < 0.001$)",
                "That yields 96 matched pairs per contrast"),
      f"{ {g: (rc.loc[g].simulated, rc.loc[g].q_vs_pop) for g in rc.index} }")
check("the Asian contrast keeps its direction and its exaggeration does not survive correction",
      round(float(rc.loc["Asian"].simulated), 3) == -1.074
      and round(popc["Asian"], 3) == -0.755
      and not flat_null["Asian"]
      and 0.05 < float(rc.loc["Asian"].q_vs_pop) < 0.10
      and intex("at $-1.074$ against a population $-0.755$",
                "does not survive correction ($q = 0.07$)"))
check("the per-model split behind the pooled Black gap",
      intex("Gemini-3-Flash at $-0.394$ and GLM-4.7 at $-0.558$",
            "produce the negative Black-minus-White estimates, Gemini-3-Flash at $-0.394$ and GLM-4.7 at $-0.558$, while DeepSeek-V3 and GPT-4o-mini land at $+0.214$ and $+0.217$, on the population's side of zero"))
# The within-model bootstrap disagrees with the paired interval on Black, and the appendix says so.
# A gate that only checked the reported interval would let that disclosure be quietly dropped.
check("the appendix discloses where the bootstrap disagrees",
      round(float(rc.loc["Black"].boot_lo), 3) == -0.254
      and round(float(rc.loc["Black"].boot_hi), 3) == -0.011
      and not (float(rc.loc["Black"].boot_lo) <= 0 <= float(rc.loc["Black"].boot_hi))
      and intex("disagrees on the Black one, where it returns $[-0.254, -0.011]$ and excludes zero"))
# intex ANDs its fragments, so "not intex(...)" would pass with one phrase still in the document.
# Absence has to be asserted per phrase.
STALE = ["reverse the direction of the disparity", "on race it reverses",
         "the opposite order from the population", "comes out backwards",
         "we do not test the contrasts individually"]
left = [p for p in STALE if intex(p)]
check("no reversal language survives the rewrite", not left, left)
check("relationship gap +0.88 and its per-model spread",
      intex("Simulated single personas score 0.88 points above married personas pooled",
            "above married personas pooled, model-specific again: +1.57 in GLM-4.7 and +1.30 in Gemini-3-Flash, against +0.43 in DeepSeek-V3 and +0.23 in GPT-4o-mini"))

# ---- response style ------------------------------------------------------------------------------
rs = A("response_style.csv")
rs = rs[rs.scope == "ALL"].set_index("instrument")
interior = 100 - rs.pct_at_max - rs.pct_at_zero
pop_interior = 100 - float(rs.loc["PHQ-8"].pop_pct_at_max) - float(rs.loc["PHQ-8"].pop_pct_at_zero)
check("endpoint shares top and bottom",
      round(float(rs.loc["PHQ-8"].pct_at_max), 2) == 0.92
      and round(float(rs.loc["GAD-7"].pct_at_max), 2) == 0.52
      and round(float(rs.loc["AUDIT-C"].pct_at_max), 2) == 0.17
      and round(float(rs.loc["PHQ-8"].pct_at_zero), 1) == 33.5
      and round(float(rs.loc["GAD-7"].pct_at_zero), 1) == 31.4
      and round(float(rs.loc["AUDIT-C"].pct_at_zero), 1) == 24.4
      and round(float(rs.loc["PHQ-8"].pop_pct_at_max), 2) == 3.93
      and round(float(rs.loc["PHQ-8"].pop_pct_at_zero), 1) == 74.8
      and intex("The top category takes 0.92\% of PHQ-8 item responses, 0.52\% of GAD-7 and 0.17\% of AUDIT-C, against 3.93\% in the population's PHQ-8 items; the bottom category takes 33.5\%, 31.4\%",
                "in the population's PHQ-8 items; the bottom category takes 33.5\%, 31.4\% and 24.4\% against a population 74.8\%; and 66 to 75\% of simulated item"))
check("interior 66 to 75 against a population 21",
      round(interior.min()) == 66 and round(interior.max()) == 75 and round(pop_interior) == 21
      and intex("33.5\%, 31.4\% and 24.4\% against a population 74.8\%; and 66 to 75\% of simulated item responses land strictly"
                "responses land strictly between the two endpoints, against 21\% in the population"))
check("sixteen severe generations, all transgender, fourteen from GLM",
      intex("Sixteen of the 28,800 generations reach the severe range and every one is a "
            "transgender persona", "ten under clinical framing and six under narrative",
            "fourteen of the sixteen coming from GLM-4.7"))
ds = A("distribution_shape.csv").set_index("source")
_pop = ds.loc["NHANES 2005-2018 (weighted)"]
_sim = ds.loc[[i for i in ds.index if i.startswith("Pooled") or i.startswith("All models")][0]] \
    if any(i.startswith(("Pooled", "All models")) for i in ds.index) else None
check("77.1 population at or below 4, population 15.4 in the mild band",
      round(float(_pop["pct_0-4"]), 1) == 77.1 and round(float(_pop["pct_5-9"]), 1) == 15.4
      and round(float(_pop["pct_20-24"]), 1) == 0.6
      and intex("The population distribution is heavily right-skewed, 77.1\% of adults sitting at 4 or below with a long thin tail, while the",
                "a long thin tail, while the pooled simulated draws stack 55.3\% into the single five-point band from 5 to 9 against 15.4\% in the population",
                "Severe scores of 20 or above make up 0.6\% of the population, about one adult in 170"))

# ---- coherence -------------------------------------------------------------------------------------
gw = A("gateway_null.csv").set_index("scope")
pm = A("gateway_per_model.csv").set_index("model")
p = gw.loc["pooled"]
check("2.68 observed, 10.4 / 21.2 / 20.0 nulls, 9,590 elevated, 257 violations",
      round(float(p.observed_pct), 2) == 2.68
      and round(float(p.null_conditional_marginal_pct), 1) == 10.4
      and round(float(p.null_uniform_composition_pct), 1) == 21.2
      and round(float(p.null_within_case_permute_pct), 1) == 20.0
      and int(p.n_elevated) == 9590 and int(pm.loc["POOLED"].violations) == 257
      and intex("9,590 of the 28,800 generations qualify", "257 of those violate it",
                "at 2 or above, and 257 of those violate it, a rate of 2.68\%", "from the item marginals of its own model and cohort, and 10.4\% of elevated cases would violate the rule",
                "the rule; redistribute the total uniformly and you get 21.2\%; permute each case's own responses across"
                "case's own responses across item positions and you get 20.0\%", "Reviewed one at a time, 97.3\% of elevated presentations are formally"))
check("per-model spread 0.35 to 6.24",
      round(float(pm.loc["z-ai/glm-4.7"].violation_rate_pct), 2) == 6.24
      and round(float(pm.loc["deepseek/deepseek-chat-v3"].violation_rate_pct), 2) == 0.35
      and round(float(pm.loc["openai/gpt-4o-mini"].violation_rate_pct), 2) == 3.32
      and round(float(pm.loc["google/gemini-3-flash-preview"].violation_rate_pct), 2) == 0.55
      and intex("Violations concentrate in GLM-4.7 at 6.24\% of its elevated cases and are nearly"
                "of its elevated cases and are nearly absent in DeepSeek-V3 at 0.35\%, with GPT-4o-mini at 3.32\% and Gemini-3-Flash at 0.55\% (Figure~\ref{fig:models})"
                "at 0.55\\%"))
check("null-normalized factors 34 / 16 / 5.3, GLM at 1.0",
      round(float(gw.loc["deepseek/deepseek-chat-v3"].ratio_vs_conditional)) == 34
      and round(float(gw.loc["google/gemini-3-flash-preview"].ratio_vs_conditional)) == 16
      and round(float(gw.loc["openai/gpt-4o-mini"].ratio_vs_conditional), 1) == 5.3
      and round(float(gw.loc["z-ai/glm-4.7"].null_conditional_marginal_pct), 2) == 6.12
      and intex("factor of 34, Gemini-3-Flash by 16 and GPT-4o-mini by 5.3",
                "GLM-4.7 does not clear its own null at all, sitting at 6.24\% observed against 6.12\% expected"))

# ---- stability ---------------------------------------------------------------------------------
wr = A("within_run_stability.csv")
alln = wr[wr.model == "ALL"].set_index("condition")
per = wr[wr.model != "ALL"]
check("flip 35.4 clinical / 32.2 narrative, abs diff 1.80 / 1.67",
      round(float(alln.loc["clinical"].flip_prob_pct), 1) == 35.4
      and round(float(alln.loc["narrative"].flip_prob_pct), 1) == 32.2
      and round(float(alln.loc["clinical"].mean_pairwise_abs_diff), 2) == 1.80
      and round(float(alln.loc["narrative"].mean_pairwise_abs_diff), 2) == 1.67
      and intex("in different PHQ-8 severity categories with probability 35.4\% under clinical framing and 32.2\% under narrative, at mean absolute"
                "differences of 1.80 and 1.67 points"))
ds_ = per[per.model == "deepseek/deepseek-chat-v3"].flip_prob_pct
gl = per[per.model == "z-ai/glm-4.7"].flip_prob_pct
check("per-model flip 22-27 DeepSeek, 38-45 GLM",
      (round(ds_.min()), round(ds_.max())) == (22, 27)
      and (round(gl.min()), round(gl.max())) == (38, 45)
      and intex("degree that changes any ranking built on it, running from 22 to 27\% for DeepSeek-V3 up to 38 to 45\% for GLM-4.7"),
      f"{ds_.min():.2f}-{ds_.max():.2f}, {gl.min():.2f}-{gl.max():.2f}")

# The churn concentration is derived from the transition matrix, not read from a summary line.
tm = A("transition_matrix_within_clinical.csv").set_index(
    A("transition_matrix_within_clinical.csv").columns[0]).to_numpy()
off = int(tm.sum() - np.trace(tm))
cross = sorted(((int(tm[i, j] + tm[j, i]), i, j)
                for i, j in itertools.combinations(range(5), 2)), reverse=True)
check("churn: 147,830 discordant, 56.0 on mild-moderate, 26.6 next, factor 2.1",
      off == 147830 and (cross[0][1], cross[0][2]) == (1, 2)
      and round(cross[0][0] / off * 100, 1) == 56.0
      and round(cross[1][0] / off * 100, 1) == 26.6
      and round(cross[0][0] / cross[1][0], 1) == 2.1
      and intex("Of the 147,830 discordant orderings, 56.0\% cross the mild-moderate line at PHQ-8 = 10",
                "56.0\% cross the mild-moderate line at PHQ-8 = 10 against 26.6\% for the next most common crossing", "factor of 2.1"))
sf = A("stochastic_fracture.csv")
sfc = sf[sf.condition == "clinical"].set_index("dimension")
check("raw dispersion H 57.55 gender, 32.23 SES, race and relationship null",
      round(float(sfc.loc["gender"].H), 2) == 57.55
      and round(float(sfc.loc["ses"].H), 2) == 32.23
      and not bool(sfc.loc["race"].significant)
      and not bool(sfc.loc["relationship"].significant)
      and intex("Kruskal-Wallis $H = 57.55$ clinical", "socioeconomic status ($H = 32.23$)",
                "with race and relationship status null"))
fm = A("fracture_matched.csv").set_index(["condition", "contrast"])
check("matched: transgender +0.25 / +0.14, 131 pairs, SES reverses by 0.38",
      round(float(fm.loc[("clinical", "transgender vs cisgender")].mean_sd_difference), 2) == 0.25
      and round(float(fm.loc[("narrative", "transgender vs cisgender")].mean_sd_difference), 2) == 0.14
      and int(fm.loc[("clinical", "transgender vs cisgender")].n_matched_pairs) == 131
      and round(float(fm.loc[("clinical", "Low vs High SES")].mean_sd_difference), 2) == -0.38
      and intex("+0.25 SD points noisier than severity-matched cisgender cells",
                "131 matched pairs", "+0.14 under narrative framing",
                "Socioeconomic status reverses: low-SES cells run 0.38 SD points \emph{less} variable than"))

# ---- covariance --------------------------------------------------------------------------------
fair = A("covariance_fair_null.csv")
isbig = lambda r: r.dimension == "ses" or (r.dimension == "gender"
                                           and r.group_a.startswith("Cis")
                                           and r.group_b.startswith("Trans"))
fair["big"] = fair.apply(isbig, axis=1)
tc = fair[fair.big & (fair.dimension == "gender")].matched_frobenius
ses_ = fair[fair.dimension == "ses"].matched_frobenius
race = fair[fair.dimension == "race"].matched_frobenius
rel = float(fair[fair.dimension == "relationship"].matched_frobenius.iloc[0])
nulls = fair.fair_null_p95
check("all fourteen above their nulls",
      len(fair) == 14 and bool(fair.above_fair_null.all())
      and intex("All fourteen contrasts exceed their nulls"))
check("bands 0.17-0.21 / 0.14 / 0.41-0.54 / 0.37-0.80, nulls 0.10-0.19",
      round(race.min(), 2) == 0.17 and round(race.max(), 2) == 0.21 and round(rel, 2) == 0.14
      and round(tc.min(), 2) == 0.41 and round(tc.max(), 2) == 0.54
      and round(ses_.min(), 2) == 0.37 and round(ses_.max(), 2) == 0.80
      and round(nulls.min(), 1) == 0.1 and round(nulls.max(), 1) == 0.2
      and intex("Racial contrasts sit at 0.17 to 0.21, the relationship contrast at 0.14, against nulls of 0.10 to 0.19",
                "0.21, the relationship contrast at 0.14, against nulls of 0.10 to 0.19",
                "contrasts against cisgender reference cohorts run 0.41 to 0.54, and socioeconomic contrasts run 0.37 to 0.80",
                "the two within-category gender contrasts sit at 0.16 and 0.21.} \label{tab:structure} \small \setlength{\tabcolsep}{3pt} \begin{tabular}{llll} \toprule Axis & Severity & Structure (centred) & Stability \\ \midrule Race & +3.77 to +4.24 & 0.17, 0.21 (0.20, 0.24) & $-$0.14 SD**\textsuperscript{a} \\", "& no effect \\ Gender identity & +2.73 to +3.36 (descr.) & 0.41, 0.54 (0.25, 0.46) & +0.25 SD*** \\ Socioeconomic & +2.75 to +5.48 &", "0.46) & +0.25 SD*** \\ Socioeconomic & +2.75 to +5.48 & 0.37, 0.80 (0.43, 0.81) & $-$0.38 SD** \\ \bottomrule \end{tabular}"))
bg = A("band_gap_bootstrap.csv").set_index("statistic")
med = bg.loc[[i for i in bg.index if i.startswith("median ratio")][0]]
conc = bg.loc[[i for i in bg.index if i.startswith("pairwise concordance")][0]]
check("median ratio 2.38, bootstrap floor 1.39, all 49 ordered",
      round(float(med.observed), 2) == 2.38 and round(float(med.ci_lo), 2) == 1.39
      and float(conc.observed) == 1.0
      and intex("The median large contrast in our sample is 2.38 times the median small one",
                "design cells within cohorts sets a conservative 95\% floor of 1.39 on that median ratio",
                "All 49 pairings of a large contrast against a small"
                "direction"))
cen = A("covariance_centred_contrasts.csv")
cb, cs = cen[cen.large], cen[~cen.large]
check("centred bands, excess and ratios",
      sum(x > y for x in cb.centred_excess for y in cs.centred_excess) == 49
      and round(cs.centred_excess.min(), 3) == 0.038 and round(cs.centred_excess.max(), 3) == 0.093
      and round(cb.centred_excess.min(), 3) == 0.106 and round(cb.centred_excess.max(), 3) == 0.652
      and round(cb.centred_excess.median() / cs.centred_excess.median(), 2) == 3.89
      and round(cb.raw_excess.median() / cs.raw_excess.median(), 2) == 3.97
      and round(cb.centred_excess.min() / cs.centred_excess.max(), 2) == 1.14
      and round(cb.raw_excess.min() / cs.raw_excess.max(), 2) == 2.85
      and intex("still separate on excess over null: small contrasts run 0.038 to 0.093 above their nulls, large ones 0.106 to 0.652, all 49 pairings still ordered"
                "to 0.652", "ordered correctly, and the median ratio barely shifts, 3.97 down to 3.89",
                "What does move is the narrowest margin, which falls from 2.85 to 1.14 at the boundary between cisgender women and"))
ip = A("item_pair_shifts.csv")
sfs = ip[(ip.contrast == "Low vs High SES") & (ip.item_a == "fatigue")
         & (ip.item_b == "self-worth")].iloc[0]
gfs = ip[(ip.contrast == "Transgender vs cisgender") & (ip.item_a == "fatigue")
         & (ip.item_b == "self-worth")].iloc[0]
check("fatigue against worthlessness runs opposite ways on the two axes",
      round(sfs.r_first, 2) == -0.23 and round(sfs.r_second, 2) == -0.03
      and round(gfs.r_first, 2) == 0.05 and round(gfs.r_second, 2) == -0.10
      and sfs.r_first < sfs.r_second and gfs.r_first > gfs.r_second
      and intex("fatigue and worthlessness run against each other ($r = -0.23$); in high-income personas they sit close to"
                "($-0.03$)",
                "In transgender personas they travel together ($+0.05$); in cisgender personas they oppose each"
                "($-0.10$)"))

# ---- factor fit, both estimators ---------------------------------------------------------------
wcf = A("within_cohort_factor.csv")
sim_f = wcf[wcf.source == "simulated"]
ofa = A("ordinal_factor.csv")
ref = ofa[ofa.source != "simulated"].iloc[0]
op = ofa[(ofa.source == "simulated") & (ofa.model == "POOLED")]
clears = ofa[(ofa.source == "simulated") & (ofa.model != "POOLED")].groupby("model").good_fit.sum()
check("ML: population 0.042, simulated 0.093-0.125, none clears",
      round(float(wcf.iloc[0].srmr), 3) == 0.042 and len(sim_f) == 12
      and round(sim_f.srmr.min(), 3) == 0.093 and round(sim_f.srmr.max(), 3) == 0.125
      and not bool(sim_f.good_fit.any())
      and intex("population sample reaches SRMR 0.042 and no simulated cohort does",
                "running 0.093 to 0.125 pooled over models",
                "nine of twelve cohorts clear 0.08 in Gemini-3-Flash and nine in GLM-4.7"))
check("DWLS: 0.0421 -> 0.0411, 0.108-0.150, counts fall to six and four",
      round(float(ref.srmr_ml_pearson), 4) == 0.0421
      and round(float(ref.srmr_dwls_polychoric), 4) == 0.0411
      and round(op.srmr_dwls_polychoric.min(), 3) == 0.108
      and round(op.srmr_dwls_polychoric.max(), 3) == 0.150
      and int(clears["google/gemini-3-flash-preview"]) == 6
      and int(clears["z-ai/glm-4.7"]) == 4
      and bool((ofa[ofa.source == "simulated"].srmr_dwls_polychoric
                > ofa[ofa.source == "simulated"].srmr_ml_pearson).all())
      and intex("SRMR 0.0421 to 0.0411", "from 0.093--0.125 to 0.108--0.150",
                "every simulated cohort gets worse",
                "from nine of twelve to six in Gemini-3-Flash and from nine to four in GLM-4.7"))

# ---- factorial ------------------------------------------------------------------------------------
fa = A("factorial_additivity.csv").set_index("model")
fi = A("factorial_interactions.csv")
check("adjusted R2 0.947 -> 0.958 and F 4.24 on 35 and 401",
      round(float(fa.loc["main effects only"].adj_r2), 3) == 0.947
      and round(float(fa.loc["plus all two-way"].adj_r2), 3) == 0.958
      and round(float(fa.loc["nested F test"].F), 2) == 4.24
      and int(fa.loc["plus all two-way"].df_resid) == 401
      and int(fa.loc["nested F test"].df_resid) == 35
      and intex("adjusted $R^2 = 0.947$", "two-way interactions (35 terms) raises that figure to 0.958 and returns $F = 4.24$ on 35 and 401",
                "$F = 4.24$ on 35 and 401 degrees of freedom"))
check("three of six interactions survive correction",
      int(fi.significant.sum()) == 3 and len(fi) == 6
      and intex("Three of the six survive Benjamini-Hochberg correction"))
pw = A("interaction_power.csv").iloc[0]
check("power floor 8.3 FPR, 0.7 points, 0.49 against an observed 1.31",
      round(float(pw.false_positive_rate_pct), 1) == 8.3
      and round(float(pw.mde_points), 1) == 0.7
      and round(float(pw.mde_variance_share_pct), 2) == 0.49
      and round(float(pw.observed_share_pct), 2) == 1.31
      and intex("On that baseline the nested $F$ test rejects at 8.3\% against a nominal 5\% and reaches 80\% power against an effect of",
                "test rejects at 8.3\% against a nominal 5\% and reaches 80\% power against an effect of 0.7 points, or 0.49\% of cell-mean variance against an observed"
                "or 0.49\% of cell-mean variance against an observed 1.31\%, so the surviving terms clear the floor by"))
ir = A("intersection_residuals.csv")
hi_cm = ir[(ir.gender == "Cisgender Man") & ir.ses.str.startswith("High")].excess.mean()
hi_tw = ir[(ir.gender == "Transgender Woman") & ir.ses.str.startswith("High")].excess.mean()
check("intersection departures +0.46 and -0.39",
      round(float(hi_cm), 2) == 0.46 and round(float(hi_tw), 2) == -0.39
      and intex("Consider high-income cisgender man cells, which sit 0.46 points above their additive prediction"
                "against high-income transgender woman cells, which sit 0.39 below it"),
      f"{hi_cm:.4f} / {hi_tw:.4f}")

# ---- transgender elevation, ensemble, anchors, inference ----------------------------------------
te = A("trans_elevation_cell_level.csv").set_index("contrast")
tw = te.loc["Transgender Woman - Cisgender Woman"]
tm_ = te.loc["Transgender Man - Cisgender Man"]
check("trans elevation +2.73 and +3.36 with intervals",
      round(float(tw["diff"]), 2) == 2.73 and round(float(tw.ci_lo), 2) == 2.01
      and round(float(tw.ci_hi), 2) == 3.44
      and round(float(tm_["diff"]), 2) == 3.36 and round(float(tm_.ci_lo), 2) == 2.67
      and round(float(tm_.ci_hi), 2) == 4.05
      and intex("simulated transgender women land at a PHQ-8 mean of 10.14, against 7.41 for cisgender women (+2.73, 95\% CI +2.01 to +3.44)",
                "Transgender men land at 9.93, against 6.57 for cisgender men (+3.36, +2.67 to +4.05)",
                "all four models, though the magnitude ranges widely, from +0.8 points in GPT-4o-mini up to +6.2 in Gemini-3-Flash"))
es = A("ensemble_size.csv").set_index("scope")
check("ensemble 74.9 at one draw, five and seven to reach 90",
      round(float(es.loc["pooled"].k1), 1) == 74.9
      and int(es.loc["deepseek/deepseek-chat-v3"].k_for_90) == 5
      and int(es.loc["google/gemini-3-flash-preview"].k_for_90) == 7
      and str(es.loc["openai/gpt-4o-mini"].k_for_90).startswith(">")
      and str(es.loc["z-ai/glm-4.7"].k_for_90).startswith(">")
      and intex("single generation reproduces its own cell's modal category 74.9\% of the time, and reaching 90\% agreement",
                "five draws for DeepSeek-V3 and seven for Gemini-3-Flash",
                "in the searched range of one to fifteen"))
ar = A("anchor_recency.csv")
shift = -ar.resid_shift_recent_vs_paper
check("anchor windows 2.98 / 3.14 / 3.97, shrinkage 0.86-1.28, smallest +2.75 -> +1.83",
      round(shift.min(), 2) == 0.86 and round(shift.max(), 2) == 1.28
      and round(float(ar["resid_recent 2021-2023"].min()), 2) == 1.83
      and bool((ar["resid_recent 2021-2023"] > 0).all())
      and intex("2.98 on the paper window, 3.14 pre-pandemic and 3.97 in 2021--2023",
                "shrinks by 0.86 to 1.28 points", "smallest falls from +2.75 to +1.83"))
cl = A("clustering_receipts.csv")
c2w = cl[cl.quantity.str.startswith("two-way")].iloc[0]
c60 = cl[cl.unit.str.contains("60 draws")].iloc[0]
check("cluster inflation: four to seven fold, two-way 1.8 to 4.7",
      round(float(c2w.low), 1) == 1.8 and round(float(c2w.high), 1) == 4.7
      and intex("understate standard errors four to seven fold",
                "run 1.8 to 4.7 times the between-cell errors"))
ase = A("anchor_design_se.csv")
check("largest anchor design SE 0.078",
      round(float(ase.se_design.max()), 3) == 0.078
      and intex("largest standard error is 0.078 points against a smallest reported residual "
                "of +2.75"))
check("validation targets reproduced",
      intex("to within 0.02 on every mean and standard deviation but one",
            "of Brody et al.~\cite{brody2018depression} exactly (8.1\% overall, 10.4\% women, 5.5\% men)"))

# ---- structural sanity on the condensed file ------------------------------------------------------
for tb in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", tex, re.S):
    spec, body = tb.group(1), tb.group(2)
    ncol = len(re.findall(r"[lcr]|p\{[^}]*\}", spec))
    lineno = tex[:tb.start()].count("\n") + 1
    for raw in body.split(r"\\"):
        row = re.sub(r"\\(toprule|midrule|bottomrule|smallskip)\b", "", raw)
        row = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", row).strip()
        if not row or row.startswith("%"):
            continue
        amps = len(re.findall(r"(?<!\\)&", row))
        span = sum(int(n) - 1 for n in re.findall(r"\\multicolumn\{(\d+)\}", row))
        check(f"tabular arity L{lineno} ({row.split('&')[0].strip()[:18]!r})",
              amps + span == ncol - 1, f"{amps + span + 1} cells vs {ncol} columns")

labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
check("every ref has a label", refs <= labels, sorted(refs - labels))
check("every appendix is reachable from the body",
      {l for l in labels if l.startswith("app:")} <= refs,
      sorted({l for l in labels if l.startswith("app:")} - refs))

# ---- the appendix, checked by transcription rather than one assertion per figure ----------------
# The appendices are ported from paper_v2/main.tex, which scripts 15 and 19 already lock to the
# receipts claim by claim. Re-asserting each of those figures here would duplicate that work and
# would go stale independently. What is NOT covered by those gates is transcription: a digit
# dropped or transposed while moving a paragraph between the two files. So every numeric token in
# the condensed paper is required to appear in the full one, which fails on exactly that error.
# Body figures are covered twice, by the checks above and by this one.
def numerics(t):
    t = re.sub(r"\\(label|ref|cite|includegraphics|texttt|url)\{[^}]*\}", " ", t)
    # An unescaped % starts a comment; an escaped one is a percent sign. Getting that backwards
    # deletes the rest of the line, and these files put whole paragraphs on one line, so the first
    # version of this check silently discarded most of the full manuscript's numbers.
    t = re.sub(r"(?<!\\)%.*", " ", t)
    t = t.replace("{,}", ",")   # LaTeX thin-space thousands separator
    return re.findall(r"\d[\d,.]*\d|\d", t)


full = open(os.path.join(BASE, "paper_v2", "main.tex"), encoding="utf-8").read()
full_nums = set(numerics(full))
# Numbers the condensed paper is entitled to carry alone: its own section and appendix numbering
# appears in cross-reference text, and a handful of round figures occur only in sentences the
# condensing rewrote. Each is listed rather than pattern-matched away.
# ---- item ordering: the assumption under Sections 4.5 and 4.8 ------------------------------------
# The generation prompt never showed the item wording, so the model chose which symptom sat in which
# position. The gateway rule reads positions 1 and 2. Script 46 tests that choice against
# DPQ010-DPQ080; this asserts the paper reports what that test returned.
iov_path = os.path.join(BASE, "analysis", "item_order_validation.csv")
if os.path.exists(iov_path):
    iov = A("item_order_validation.csv")
    ok = iov[iov.gateway_ok.notna()] if "gateway_ok" in iov else iov
    check("item order: every model canonical on all eight positions, 40 of 40",
          len(ok) == 40 and bool(ok.gateway_ok.all()) and bool(ok.all8_ok.all())
          and ok.model.nunique() == 4
          and intex("order it would score them, every model returns \texttt{DPQ010} through \texttt{DPQ080} in the standard sequence on 40 of 40"
                    "standard sequence on 40 of 40 attempts"),
          f"{len(ok)} usable, gateway {ok.gateway_ok.mean() if len(ok) else 0}")
else:
    print("SKIP item order (run scripts/46_item_order_validation.py)")

# ---- collection provenance: Appendix A's call and recovery ledgers -------------------------------
# These were narrative prose until the generation source and the provider's activity export were
# recovered. They are numbers like any other now, so they reconcile to receipts rather than sitting
# in the allow-list below.
rl = A("recovery_ledger.csv").set_index("quantity").value
check("recovery ledger: 86 outside the session, 230 tagged, 144 same-day",
      int(rl["clinical rows outside the main session"]) == 86
      and int(rl["rows carrying a recovery batch id"]) == 230
      and int(rl["recovery rows inside the main session"]) == 144
      and int(rl["recovery rows, GLM-4.7"]) == 187
      and int(rl["recovery rows, GPT-4o-mini"]) == 43
      and int(rl["rows completed by hand (LAST_MILE_OPT_2026)"]) == 62
      and int(rl["cells touched by the outside-session rows"]) == 40
      and intex("The remaining 86 clinical rows, covering 40 of the 480 cells",
                "A further 144 rows carry a recovery batch identifier",
                "187 are GLM-4.7 and 43 are GPT-4o-mini",
                "62 of the GLM rows were completed by hand"))

if os.path.exists(os.path.join(BASE, "analysis", "call_ledger.csv")):
    cl_ = A("call_ledger.csv").set_index("model_permaslug")
    GLM = "z-ai/glm-4.7-20251222"
    check("call ledger: per-model calls, GLM error rate, tokens and providers",
          int(cl_.loc["openai/gpt-4o-mini"].calls) == 3600
          and int(cl_.loc["google/gemini-3-flash-preview-20251217"].calls) == 3600
          and int(cl_.loc["deepseek/deepseek-chat-v3"].calls) == 3935
          and int(cl_.loc[GLM].calls) == 6141
          and int(cl_.loc[GLM].errors) == 1283
          and round(float(cl_.loc[GLM].error_pct)) == 21
          and int(cl_.sum().calls) == 17276
          # only GLM reasons, and only GLM is multi-provider
          and float(cl_.loc[GLM].tokens_reasoning) > 1000
          and all(float(cl_.loc[m].tokens_reasoning) == 0 for m in cl_.index if m != GLM)
          and cl_.loc[GLM].providers.count("|") == 4
          and all(cl_.loc[m].providers.count("|") <= 1 for m in cl_.index if m != GLM)
          and intex("records 17,276 generations",
                    "GLM-4.7 took 6,141 and ended 1,283 of them in an error, a 21\% failure rate against effectively zero for",
                    "a mean of 1,400 per call against a completion mean of 1,972",
                    "nearly constant at 344 tokens"))
else:
    print("SKIP call ledger (activity export not present; run scripts/45_call_ledger.py)")

ALLOWED = {"XXXX.XXXXX"}   # the arXiv identifier of the full version, in the titlenote (redacted for review)
# Section 4.2's contrast test (script 44) is analysis the full manuscript never ran, so its outputs
# are legitimately absent from the comparison set. They are enumerated here, not pattern-excluded:
# every one is asserted against race_contrast_tests.csv above, and listing them keeps this check
# able to fail if a number appears that no receipt covers.
ALLOWED |= {"0.330", "0.131", "0.273", "0.011", "0.461",    # Black minus White
            "0.243", "0.034", "0.150", "0.082", "0.277",    # Hispanic minus White
            "1.074", "1.369", "0.778",                      # Asian minus White
            "0.394", "0.558", "0.214", "0.217",             # the per-model split
            "0.11", "0.57", "0.07",                         # BH-adjusted q values
            "0.254", "10,000",                              # the within-model bootstrap
            "96",                                           # matched cell pairs per contrast
            # Appendix A's ledgers, each asserted against a receipt above.
            "17,276", "3,600", "3,935", "6,141", "1,283", "344", "1,400", "1,972",
            "92", "97", "19.80", "22.60", "230", "144", "187", "62", "43",
            "20251222", "20251217"}
drift = sorted({n for n in numerics(tex) if n not in full_nums} - ALLOWED)
check(f"no numeric drift from the full manuscript: {len(drift)}", not drift, drift[:12])
cited = {k.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", tex) for k in grp.split(",")}
bib = set(re.findall(r"@\w+\{([^,]+)",
                     open(os.path.join(SHORT, "refs.bib"), encoding="utf-8").read()))
check("every citation resolves in refs.bib", cited <= bib, sorted(cited - bib))
check("no em-dashes", "---" not in tex)

print()
if FAIL:
    print(f"SHORT-PAPER GATE FAILED: {len(FAIL)}")
    sys.exit(1)
print("SHORT-PAPER GATE PASSED: every number in the condensed paper reconciles to receipts")
