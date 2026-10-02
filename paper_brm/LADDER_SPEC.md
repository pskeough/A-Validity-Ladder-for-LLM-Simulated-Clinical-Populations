# Validity Ladder: frozen specification

Version 1.0. Frozen on 2026-10-02 by the commit that adds this file.

This file states every rule, threshold, stop and default of the ladder, with the code that
implements it. The rules were set on the PHQ-8 corpus (NHANES 2005-2018 reference) and then run
once, as a shakedown, on a second instrument and reference: PersonaLLM BFI-44 answers from three
models against Twin-2K-500 humans (`scripts/92_personallm_shakedown.py`). From the freeze on, the
ladder is applied unchanged to every further dataset. Code changes after the freeze are limited
to bug fixes that make the code do what this file says; each is a separate commit naming the rule
it restores.

Notation: a *persona* is one conditioning profile; a *draw* is one call to the model and the answer
vector it returns; a *cell* is one persona under one model and one framing; *k* is the number of
draws averaged into a persona mean. Scores are the instrument's total, items scored 0 to M-1
after reverse keying. *Reference SD* is the standard deviation of the total in the population
reference.

## Reading rules common to every rung

- Every verdict is read per model and per framing. Pooled rows are description.
- A verdict is pass, fail or unresolved, or a named stop where the reference cannot support a
  verdict. "Pass" means the rung's test detected no failure at the resolution the data allow.
- An interval that lies wholly inside a rung's tolerance passes, one wholly outside fails, and one
  that crosses the tolerance is unresolved.
- No rung's result stops another rung, except that level 4's later rules are description when R1
  fails.
- Intervals are 90% unless a rung says otherwise.

## Gate: precision of a persona mean

| Item | Rule |
|---|---|
| Model | One-facet G-study within one model and framing: X_pr = mu + nu_p + e_r:p; components from expected mean squares |
| Statistic | SE(k) = sqrt(s2_r / k); phi(k) = s2_p / (s2_p + s2_r / k) |
| Minimum rule | SE(k) <= 0.25 reference SD in every framing |
| Recommended rule | SE(k) <= 0.125 reference SD in every framing |
| Single draw | within-cell SD sqrt(s2_r) <= 0.25 reference SD |
| k needed | ceil(s2_r / t^2) for tolerance t |
| Separation | phi(k) reported beside every verdict; phi(k) >= .80 (recommended .90) is required only for uses that compare or rank individual personas |
| Intervals | percentile bootstrap over personas, B = 1,000 |
| PHQ-8 values | reference SD 3.935 (NHANES 2005-2018), tolerances 0.98 and 0.49 points |
| Code | `82_gate_lib.one_facet`, `phi_k`, `k_for_se`, `k_for_phi`; `91_ladder_core.gate`, `GATE_MIN`, `GATE_REC`; `88_intermediate_panel.gate_eval` |

## Level 1: individual coherence

| Item | Rule |
|---|---|
| Model | Graded response model fitted to the reference (survey weights where the reference has them) |
| Statistic | l_z* with WLE theta (Snijders correction, polytomous form); each draw placed in the reference distribution of l_z* among respondents with the same total (randomised PIT; totals pooled into strata of at least 100 reference respondents) |
| Ratios | misfit ratio = share below the within-total 5th percentile / .05; overfit ratio = share above the 95th / .05 |
| Pass | both ratios' 90% intervals inside [1/tau, tau] |
| Fail | either tail's interval wholly outside the band; both tails below = "compressed" |
| Tolerance tau | smallest value in {1.50, 2.00, 3.00} that contains the resolved departure of every reference subgroup scored against the pooled reference. Resolved departure of a ratio with 90% interval [lo, hi]: lo if lo > 1, 1/hi if hi < 1, else 1. Subgroups: every demographic attribute the reference records, levels with n >= 100 |
| Floor | tau >= 1.50: real respondents drawn through the persona design pass at 1.5 and not at 1.25 (`83c_l1.csv`) |
| Intervals | persona-clustered bootstrap of the draws, jointly with a bootstrap of the reference (Rao-Wu over PSUs for a survey, iid persons otherwise), B = 2,000 |
| PHQ-8 values | worst resolved subgroup departure 1.27 (non-Hispanic Black, misfit), tau = 1.5 |
| Code | `91_ladder_core.GRM`, `L1Ref`, `l1_eval`, `tau_from_subgroups`, `TAUS`, `TAU_FLOOR`; verdict `79c_l1_personfit.read_verdict` |

## Level 2: subgroup fidelity

| Item | Rule |
|---|---|
| Estimand | simulated gap g = equal-weight mean over strata of the other attributes of within-stratum persona-mean differences; population gap gamma = the reference contrast in the same strata with the same weights (standardised); marginal gap reported beside it |
| Statistic | rho = g / gamma, Fieller interval; Welch-Satterthwaite df combining simulation and design df |
| SEs | g: SD of persona-pair differences / sqrt(pairs) (draw-only SE as sensitivity); gamma: design-based (Taylor linearisation or delete-one-PSU jackknife) |
| Regions | boundaries -0.25, 0.25, 0.75, 1.25: reversed, missing, attenuated, kept, steepened |
| Contrast reading | the region holding the interval; two adjacent regions when it crosses one boundary; undetermined when it crosses more |
| Multiplicity | one family per model; Bonferroni over the family's contrasts (7 for the PHQ-8 design: 98.6% intervals) |
| Stop 1 | "no population gap": gamma's own interval covers 0 |
| Stop 2 | "reference too imprecise": k = t SE_gamma / abs(gamma) > 1/4 (no ratio interval can then fit inside the kept region) and the reading does not already exclude kept. A reading that excludes kept stands |
| Model verdict | pass: every contrast that is not stopped is kept; fail: any contrast's verdict excludes kept; otherwise unresolved |
| Code | `78c_l2_r3_verdicts.r3` (`verdict`; the pre-freeze rule is kept as `verdict_conditional`), `78e_l2_external_r3.r3`, `83f_l2_parametric_power.verdicts`, `88_intermediate_panel.l2_model_verdict` |

## Level 3: population calibration

| Item | Rule |
|---|---|
| Statistic | post-stratified residual R_G = sum_c p_c (s_c - y_c), p_c = reference weighted share of cell c in group G |
| SEs | simulation: sum p_c^2 var(s_c); reference: delete-one-PSU jackknife including the weights; Satterthwaite df |
| Pass | TOST at alpha = .05: 90% interval of R_G inside (-delta, +delta) for every group row |
| Tolerance | named by the user for the use; default 0.5 reference SD (2 points on the PHQ-8); 0.2 SD and 1 point reported |
| Prevalence row | share at or above the cut point: difference bands 2.5 and 5 points, ratio band 0.80-1.25 (Fieller) |
| Code | `80_l3_lib`, `83_controls_lib.l3_eval`, `l3_pass` |

## Level 4: structural fidelity

| Item | Rule |
|---|---|
| Unit | all draws of one model and framing; persona is the resampling unit |
| Fit | two-step polychorics; one-factor ULS |
| R1 | every loading >= .30 and lambda1/lambda2 >= 3; where the reference itself is below a threshold, that item's or the ratio's threshold is the reference's own value. A population without a general factor fails level 4 |
| R2 | pass: Tucker congruence with the reference loadings has lower 90% limit >= .95 AND the RMS loading difference from the reference has upper 90% limit <= .10; fail: congruence upper limit < .95 OR RMSD lower limit > .10; otherwise unresolved. Limits from the persona bootstrap paired with the reference bootstrap |
| R4 | multi-group CFA, metric and scalar steps across each demographic attribute; RMSEA_D with nu from permuting group labels among personas; a step holds if its upper 90% limit < .08, fails if its lower limit > .08, else indeterminate. Pass when every step matches the reference's verdict; fail when a determinate step differs; else unresolved. Margin .05 as sensitivity |
| R3 | diagnostic: one-factor fit to the pooled within-persona covariance (reference: within demographic cell) |
| Level 4 | pass when R1, R2 and R4 pass in every framing; fail when R1 fails, R2 fails or R4 fails in any framing; otherwise unresolved |
| Code | `81_l4_lib`, `81b_l4_structure`, `81c_l4_invariance`, `81g_l4_r2_size`; `83_controls_lib.l4_eval` (`R2_verdict`); `91_ladder_core.L4Ref`, `l4_eval`, `general_factor_vs_ref`, `r2_verdict`, `R1_LOAD`, `R1_RATIO`, `R2_PHI`, `R2_RMSD`; `88_intermediate_panel.r4_eval` |

On NHANES every loading (minimum .67) and the eigenvalue ratio (7.47) exceed the absolute R1
thresholds, so the PHQ-8 code paths that apply the absolute thresholds (`81_l4_lib.general_factor`)
give the frozen R1 verdict.

## Rungs that need persona attributes

Levels 2, 3 and R4 compare groups that the population records. When the personas carry no
attribute the reference records (PersonaLLM's trait instructions), these rungs are not applicable
and are recorded as such. Without repeated draws of one persona under one prompt, the gate and
level 1 are not applicable.

## Changes made before the freeze, from the shakedown and the controls

1. Gate tolerances stated in reference SD units (0.25 and 0.125 SD) so they carry to other
   instruments. On the PHQ-8 this is 0.98 and 0.49 points against the 1.0 and 0.5 points stated
   before; no SE verdict changes (`91a_core_regression.csv`).
2. Gate reads precision only. phi(k) was a pass condition; real NHANES respondents drawn through
   the persona design reach phi(30) of .63 to .76 and failed the gate in every replicate, because a
   faithful simulator has the population's small differences between cells. phi(k) is now the
   separation statistic, required only for persona-comparison uses. On the corpus this changes one
   verdict: GPT-4o-mini's narrative framing passes the recommended rule (SE(30) = 0.32).
3. tau from resolved subgroup departures with a floor of 1.5. The point-ratio rule gave tau = 2.0
   or 3.0 on the Twin-2K BFI scales (worst point departures 1.52 to 2.75) from subgroups of 112 to
   1,361 people whose point ratios carry their own sampling noise; reading the interval without a
   floor gave tau = 1.1 because those subgroups are too small to resolve any departure. On NHANES
   both readings give 1.5.
4. R1 relative to the reference. Twin-2K's Openness scale has one loading of .28 (item 35), so the
   human reference failed the absolute R1.
5. R2 size condition and three-way reading. Congruence ignores the scale of the loadings: with half
   the draws decorrelated the loadings fall from .69 to .52 at their minimum and congruence stays at
   .999 (`83h_r2_size.csv`). An interval that crosses a threshold is unresolved, as at every other
   rung.
6. Level-2 stop 2 on k > 1/4. The earlier stop fired only when the conditional ratio interval
   spanned three or more regions, so a contrast whose reference could never certify kept held a
   faithful model at unresolved on every design. Determinate failures are unchanged.

## Implementation checks

- `scripts/91a_core_regression.py`: `91_ladder_core` returns the audit libraries' numbers on the
  PHQ-8 corpus and NHANES (`analysis/brm/91a_core_regression.csv`).
- `scripts/83i_controls_l2_frozen.py`, `scripts/88e_panel_l2_frozen.py`: level 2 of the controls
  and the intermediate panel recomputed under the frozen rule, each replicate checked against the
  stored ratios and intervals.
- `scripts/83f_l2_parametric_power.py`: the vectorised rule matches `78c.r3` on 200 draws.
