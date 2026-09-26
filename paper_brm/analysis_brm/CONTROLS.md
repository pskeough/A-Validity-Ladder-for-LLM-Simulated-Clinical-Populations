# Positive control and planted failures (26 Sep 2026)

Scripts: `scripts/83_controls_lib.py`, `83a_controls_receipt.py`, `83b_controls_simulate.py`, `83c_controls_summary.py`, `83d_l2_se_check.py`, `83e_l2_power.py`, `83f_l2_parametric_power.py`, `83g_l2_conditional_verdicts.py`. Outputs: `analysis/brm/83*`. Local CPU only, no API calls. Design: `plan/CONTROLS_AND_POWER_DESIGN_2026-09-26.md`. The file named in brackets prints each number below.

## 1. What the controls show

1. **Real people pass every rung.** A pseudo-model built from NHANES respondents is run through the audit's own design and read against a separate half of NHANES. It passes L1 in 99.4% of 800 model-replicates. It passes L3 on all ten groups in 100% at 2 points and 98.1% at 1 point. It passes L4 (R1, R2 and R3, both framings) in 100% of 200. At L2 a determinate wrong verdict is rare: at most 0.1% of per-model readings at dose 1, and at most 1.0% at any dose and scope. [83c_l1.csv, 83c_l3.csv, 83c_l4.csv, 83c_l2.csv]
2. **Each rung detects a planted failure well short of the failures the models show.**

| Rung | Planted failure | Detected in at least 95% of replicates from | Real models |
|---|---|---|---|
| L1 | Share of draws with items shuffled (incoherent) | 40% of draws (50% detection at 20%) | not seen |
| L1 | Share of draws replaced by the most typical human vector at the same total (compressed) | 80% of draws (60% detection at 60%) | misfit ratio 0.00 to 0.18. The 100% dose reaches only 0.33 [L1.md] |
| L2 | Income or sex gap scaled by g | g = 2 pooled (Low-High 99.5%, Women-Men 96.5%); g = 3 per model (97.4 to 100%) | income, standardised: pooled 2.1 to 4.8 times, per model 1.3 to 8.4 (Low-High 1.9 to 5.1) |
| L3 | Every cell mean shifted by d points | per model: d = 0.5 at 0.2 SD (98%), d = 1 at 1 point (100%), d = 2 at 2 points (100%) | +2.1 to +4.7 points overall |
| L4 | Share of draws decorrelated within persona | 50% fails R3 in every replicate; 75% fails R1 in 98.5% | Gemini and GLM within-persona lambda1/lambda2 1.29 to 1.59, which the planted dose reaches between 75% (1.71) and 100% (1.03) |

[83c_l1.csv, 83c_l2.csv, 83c_l3.csv, 83c_l4.csv]

3. **L2 detects distortion and cannot certify faithfulness at NHANES precision.** The positive control is never named "kept" alone, in any of 5,600 per-model or 1,400 pooled readings, at either reference precision. It is usually "undetermined" or given a two-region verdict that contains "kept". The parametric check at the audit's own precision confirms this. There, a faithful model is named "kept" alone at most 37% of the time (Low-High income, smallest draw-only SE), 20% for Women-Men, and 0% for every other contrast [83f_l2_parametric_power.csv]. The "kept" region is 0.75 to 1.25 times the population gap. The gap's own 98.6% interval (the rule's level) is ±18 to ±33% of its value for income and sex, and wider than the gap itself for Black-White and Hispanic-White. The ratio interval adds the simulated side's error on top, so it rarely fits inside a band 0.5 wide. The rule is working as designed, and the reference is too coarse to award "kept". The paper should say so in the L2 section.
4. **Race contrasts are mostly stopped.** At equal income, sex and marital status the NHANES Black-White and Hispanic-White gaps are small (-0.28 and -0.18 points) against SEs of about 0.09. In the positive control, per-model Black-White is stopped in 81% of readings and Hispanic-White in 91% (half-sample reference) [83c_l2.csv]. Stopping is the intended behaviour: the rule declines to read a ratio whose denominator is indistinguishable from zero. The audit's "no race verdict survives" is therefore a property of the reference, and the same design would return it for a faithful model.
5. **R2 is insensitive.** When every draw is decorrelated, Tucker's phi still averages .98. R2 still holds in 20 to 25% of framings (8.5% in both framings), while R1 and R3 fail in all of them [83c_l4.csv]. At 75% decorrelation R2 holds in every replicate while R1 fails in 98.5%. A flat loading vector already scores phi .998 against NHANES [L4.md]. R2 cannot fail a model that R1 passes. It stays in the rung as a check on loading direction, with the flat-vector baseline reported beside it.
6. **The L1 baseline explains tau = 1.5.** Under the persona design the positive control's misfit ratio averages 1.15 (overfit 1.00) [83c_l1.csv]. The design gives equal weight to 48 cells, which over-represents the subgroups with more misfit (L1.md: non-Hispanic Black adults 1.36). A tolerance of 1.25 would fail real people drawn through the audit's own persona design. The misfit tail passes at tau = 1.5 in 99.4% of replicates.

## 2. Calibration

- **L2 intervals cover the true ratio.** Per-model coverage of the true ratio is .986 to 1.000 against a nominal .986, at every dose and contrast [83c_l2_coverage.csv]. The pairs SE is conservative for pseudo-models: it is 1.15 to 1.61 times the draw-to-draw SD measured across the four pseudo-models that share a donor half [83d_l2_se_check.csv]. Pooled coverage dips to .965 to .980 on three contrasts (Middle-High, Women-Men, Low-Middle). The half-split design adds the donor half's own error, which no SE term carries; 83e adds it (section 4).
- **L3 null residuals are unbiased.** Across the ten groups the mean residual is -0.011 to +0.022 points [83c_l3_null.csv]. The 90% intervals cover 83 to 89% per model and 78 to 89% pooled, slightly under nominal, for the same donor-half reason. With the donor half's design variance added to the SE, coverage is 87 to 91% per model and 89 to 94% pooled, so the interval is calibrated once every source of error in the control is counted [83e_l3_null.csv]. A real model has no donor term, so the audit's L3 SE is the right one for the audit.
- **The rung code reproduces the audit.** Script 83a runs the control library on the real corpus with the full NHANES reference and matches every published rung number: 303 of 303 checks, including every L1, L2, L3 and L4 verdict [83a_receipt.csv].

## 3. The L2 standard error

78b's SE for a simulated gap is the SD of the persona-pair differences divided by the square root of the number of pairs. That SD includes real differences in the gap between strata, so for the real models it is usually larger than the draw-only SE: median 1.97 times, range 0.67 to 9.86 [83d_l2_se_check.csv]. The draw-only SE conditions on the fixed persona set, as L3 does. Recomputed with it, 9 of the 35 published verdicts become sharper and none moves outside its published verdict [83g_l2_conditional_verdicts.csv]:

- GPT-4o-mini Asian-White: undetermined to "missing or attenuated"
- Gemini Asian-White and Women-Men: undetermined to "kept or steepened"
- DeepSeek-V3 Black-White: undetermined to "reversed or missing"; Women-Men: "missing or attenuated" to attenuated
- GLM-4.7 Black-White: undetermined to "kept or steepened"; Asian-White: undetermined to steepened
- Pooled Asian-White: undetermined to "kept or steepened"; pooled Women-Men: undetermined to kept

The pooled Women-Men "kept" (ratio 0.97, interval 0.79 to 1.22) averages GPT-4o-mini and DeepSeek-V3, which lose most of the gap, with GLM-4.7, which doubles it. That is why pooled L2 rows are read as description.

## 4. Power at the audit's precision

Two estimates of power, one conservative and one direct.

**Half-split with the donor term (83e, 200 replicates, same panels as 83b).** The simulated side carries the donor half's error and the reference SE is scaled to the audit's full sample. Correct verdict at g = 2 (steepened), per model:

| Contrast | pairs SE | draw-only SE |
|---|---|---|
| Low-High income | 77% | 96% |
| Women-Men | 65% | 83% |
| Middle-High income | 28% | 37% |

Pooled over the panel, g = 2 is named correctly in 96 to 100% of replicates for Low-High and Women-Men under either SE, and in 58 to 62% for Middle-High. At g = 3 every income and sex contrast is correct in at least 97% of replicates. Interval coverage is at least .955 in every cell and has a median of .984 to 1.000, against .986 nominal. A determinate wrong verdict occurs in at most 0.5% of readings. At g = 1 the control is never named "kept" alone. A two-region verdict containing "kept" occurs, for the income and sex contrasts, in 2 to 30% of per-model readings and 8 to 53% of pooled ones, and the rest are undetermined [83e_l2_power.csv].

**Audit precision, parametric (83f).** The population gap is drawn around the full-sample NHANES estimate with its Taylor SE. The simulated gap is drawn around g times it, with the smallest and largest SE the four real models actually had. Each pair is then read with the R3 rule, 20,000 draws per cell. The vectorised rule matches 78c.r3 on a 200-draw check. Correct verdict at g = 2, draw-only SE: 95 to 100% for sex and income (87 to 100% at the largest model SE). With the pairs SE it is 85 to 100% at the smallest model SE and 5 to 85% at the largest (Gemini or GLM-4.7). Black-White reaches 62 to 72% only at g = 3 (20% at the largest pairs SE), and Hispanic-White never exceeds 5%. The largest wrong rate in any cell is 0.16% [83f_l2_parametric_power.csv].

**What this means for the audit's verdicts.** The real Low-High income ratios (1.9 to 5.1 per model, 3.5 pooled) sit where the rule is correct nearly always under either SE, so the income finding does not depend on the SE choice or on power [83g_l2_conditional_verdicts.csv]. The two per-model income ratios near 1.3 (Gemini Middle-High, DeepSeek-V3 Low-Middle) read "kept or steepened", which is what the rule should return at that size. Race verdicts are not interpretable as "kept" or "missing" at this reference precision, for any model, including a faithful one.

## 5. Method (for the paper)

Each replicate splits NHANES 2005-2018 adults at random into two halves, stratified by the 48 persona cells (race by sex by income band by marital status). The donor half supplies respondents and the reference half is the benchmark. A pseudo-model has the audit's design: 120 personas, 2 framings and 30 draws. Each draw is a donor respondent sampled with probability proportional to the examination weight within the persona's cell, and it keeps the respondent's eight item scores. Multiracial personas draw from the NHANES other-race group and transgender personas from the NHANES sex group matching their gender (a transgender man draws from men). Both are placeholders and sit outside the 48 anchored cisgender cells, so, as in the audit, they enter L1 and L4 and are outside L2 and L3. Four independent pseudo-models form a panel, the audit's panel size. There are 200 replicates, and replicate r is seeded (20261001, r).

Planted failures are applied to the same panel at a grid of doses (83b docstring):

- L1 shuffle: permute a draw's eight item values.
- L1 typical: replace a draw with the most typical donor vector at its total.
- L2 gap: shift the first-named group's cells so that the simulated gap is g times the population gap.
- L3 shift: move every cell mean by d points.
- L4 decorrelate: rebuild a draw item by item from random draws of the same persona and framing. This keeps persona means and removes within-persona covariance.

Doses are nested, so a draw altered at dose p is altered at every larger dose. Each rung is read with its audit rule. L1 uses B = 400 bootstrap replicates and L4 uses B = 100, against 2,000 and 1,000 in the audit. At B = 2,000, 83a's interval limits agree within 0.02. The run took 66 minutes on 7 CPU cores.

Limits:

- The reference half has twice the sampling variance of the audit's full sample. L2 is also read with the reference SE divided by the square root of 2, and 83f simulates the audit's precision directly.
- A pseudo-model reproduces its donor half, not the population. L2 and L3 calibration therefore carry the donor half's error, and 83e adds it back.
- R4 (invariance) is not simulated. Its permutation null is already the audit's own calibration (L4.md, section 3).
