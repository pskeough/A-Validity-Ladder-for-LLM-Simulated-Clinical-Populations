# Intermediate simulator panel (1 Oct 2026)

Scripts: `scripts/88_intermediate_panel.py` (simulation), `scripts/88b_intermediate_summary.py` (tables). Outputs: this folder. Local CPU only, stored NHANES data, no API, no GPU. Nothing under `paper_brm/manuscript/` was edited.

## 1. Question

The controls in CONTROLS.md have one positive control, a NHANES split half. Reviewers asked what the ladder does with simulators that are partly human. This panel runs seven simulator types through every rung with the rung code the audit uses, and records which rung each type trips.

## 2. Design

Each replicate repeats the 83b split. NHANES 2005-2018 adults are split 50/50 within the 48 persona cells into a donor half and a reference half. A pseudo-model has the audit design: 120 personas, 2 framings, 30 draws. Four pseudo-models form a panel. Each draw is a donor respondent sampled within the persona's cell with probability proportional to the MEC weight. The seven types are built from the same donor half in the same replicate.

| Type | Construction | Intended rung |
|---|---|---|
| REAL | 83b positive control, unchanged | none |
| ORACLE-INDEP | each of the 8 items drawn independently from the cell's weighted item marginal | L1 and L4 (no covariance), correct means |
| SHIFTED-2.1 | per-cell exponential tilt of the donor weights so the cell's expected total rises 2.1 points | L3 only |
| SHIFTED-4.7 | same, 4.7 points | L3 only |
| STEEPENED-2x | tilt applied to Low-income personas only, raising them by the full-sample Low-High gap (1.297 SD units, 1.29 points realised), so the Low-High gap doubles | L2 (and L3 Low-income group) |
| COMPRESSED | every draw replaced by the most typical donor vector at its total (83 typical_bank) | L1 only |
| NONINVARIANT | Low-income personas only: items 3, 4, 5 (sleep, fatigue, appetite) resampled independently from the cell marginal; other items and all other personas unchanged | L4 R4 (income metric) only |

The tilt is p_i proportional to w_i exp(beta_c t_i), with beta_c solved per cell (brentq) to hit the target. It moves the total by reweighting real response vectors, so item covariance survives.

Rungs and rules, each taken from the existing library:

- Gate: 82_gate_lib one-facet G-study per model, minimum rule phi(30) >= .80 and SE(30) <= 1.0 in both framings; recommended rule .90 / 0.5.
- L1: 83 l1_eval, tau = 1.5, read per framing on pseudo-model 0. Model passes when both framings pass, fails when either fails, otherwise unresolved.
- L2: 78c r3 Fieller ratio with Bonferroni 98.6%, asymmetric bands, conditional stop. Model rule from 02_ladder.tex: pass when every non-stopped contrast is "kept", fail when any verdict excludes "kept", otherwise unresolved. Four variants per model: pairs SE or draw-only conditional SE, crossed with the half-sample reference SE or that SE divided by sqrt(2) (audit precision). Pooled readings use pairs SE.
- L3: 83 l3_eval, 10 groups, TOST at 2 points (primary) and 1 point.
- L4: R1 (all loadings >= .30 and lambda1/lambda2 >= 3), R2 (phi lower 90% bound >= .95), R3 within-persona diagnostic, on pseudo-model 0 per framing. R4 by 81c.analyse (multi-group CFA, RMSEA_D, permutation over personas, persona bootstrap) for sex, race and income, metric and scalar. Each step verdict is compared with the published NHANES full-sample verdict at margin .08 (primary) and .05. R4 passes when all six steps match, fails when any determinate verdict mismatches, and is otherwise unresolved. There is no R4 verdict without a general factor (R1 failed). Level 4 = R1, R2 and R4 in both framings.

NHANES reference verdicts for R4 (L4.md): at .08 all six steps hold. At .05 sex metric holds, sex scalar fails, race metric holds, race scalar is indeterminate, income metric is indeterminate, income scalar holds.

## 3. Seeds, replicates, runtime

- Master seed 20261088. Replicate r uses `default_rng([20261088, r])` for the split, the draws and the L1/L2/L3/L4 bootstraps. R4 uses `default_rng([20261088, r, type_index, framing, attribute_index])`.
- 70 replicates were launched (0 to 69). Replicate 4 crashed inside 81c (section 7) and was not rerun. 69 replicates are in the tables. Per-model rows (gate, L2, L3) have n = 276. L1 and L4 model-0 rows have n = 69.
- R4 was run on replicates 0 to 24 for REAL and NONINVARIANT (24 completed) and replicates 0 to 4 for the other five types (4 completed). R4 settings: K_PERM 30, bootstrap 100, polychoric bootstrap 2, against 200 / 200 / 100 in the audit.
- L1 bootstrap B = 400, L4 bootstrap B = 100 (as 83b).
- Timing: one replicate without R4 took 92 s on one core. One R4 population took about 55 s. The full run took 39 minutes on 6 workers (17:54 to 18:33; a seventh core was busy with another job). Test runs took 6.4 minutes. Total compute 45.5 minutes, inside the 60-minute budget.

## 4. Confusion table

Share of readings with each verdict. L2 is the per-model reading with pairs SE and half-sample reference unless stated. Source: `88_confusion.csv`.

| Rung | REAL | ORACLE-INDEP | SHIFTED-2.1 | SHIFTED-4.7 | STEEPENED-2x | COMPRESSED | NONINVARIANT |
|---|---|---|---|---|---|---|---|
| Gate, minimum rule pass | 0.000 | 1.000 | 0.000 | 0.000 | 0.130 | 0.000 | 0.007 |
| Gate, recommended rule pass | 0.000 | 0.239 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| L1 pass / fail / unresolved | .928 / 0 / .072 | 0 / 1.000 / 0 | .725 / 0 / .275 | .159 / 0 / .841 | .841 / 0 / .159 | 0 / 1.000 / 0 | .029 / .014 / .957 |
| L2 pass / fail / unresolved | 0 / 0 / 1.000 | 0 / 0 / 1.000 | 0 / 0 / 1.000 | 0 / .007 / .993 | 0 / .924 / .076 | 0 / 0 / 1.000 | 0 / 0 / 1.000 |
| L2 fail, pairs SE, audit precision | 0 | 0 | 0 | .014 | .938 | 0 | 0 |
| L2 fail, draw-only SE, half reference | .011 | .007 | .007 | .018 | .982 | .011 | .004 |
| L2 fail, draw-only SE, audit precision | .022 | .014 | .018 | .033 | .993 | .022 | .022 |
| L2 pooled fail (half / audit precision) | 0 / 0 | 0 / 0 | .014 / .014 | 0 / .029 | 1.000 / 1.000 | 0 / 0 | 0 / 0 |
| L3 pass, 2 points | 1.000 | 1.000 | 0.000 | 0.000 | 0.855 | 1.000 | 1.000 |
| L3 pass, 1 point | 0.982 | 1.000 | 0.000 | 0.000 | 0.004 | 0.982 | 1.000 |
| L4 R1 pass | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.986 | 1.000 |
| L4 R2 pass | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| L4 R3 pass (diagnostic) | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.783 | 0.014 |
| R4 at .08 pass / fail / unresolved / no verdict | 0 / .042 / .958 / 0 | 0 / 0 / 0 / 1.000 | 0 / 0 / 1.000 / 0 | 0 / 0 / 1.000 / 0 | 0 / 0 / 1.000 / 0 | 0 / 0 / 1.000 / 0 | 0 / 1.000 / 0 / 0 |
| R4 at .05 pass / fail / unresolved / no verdict | 0 / 0 / 1.000 / 0 | 0 / 0 / 0 / 1.000 | 0 / 0 / 1.000 / 0 | 0 / 0 / 1.000 / 0 | 0 / .250 / .750 / 0 | 0 / 0 / 1.000 / 0 | 0 / 0 / 1.000 / 0 |
| R4 n | 24 | 4 | 4 | 4 | 4 | 4 | 24 |
| Level 4 at .08 pass | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

R4 fail rate 95% intervals at .08: REAL .042 [.007, .202], NONINVARIANT 1.000 [.862, 1.000]. With n = 4 the other types have an upper bound of .49 on any rate of 0.

Supporting tables: `88_gate.csv`, `88_l1_readings.csv`, `88_l2_contrasts.csv`, `88_l3_groups.csv`, `88_r4_steps.csv`, `88_checks.csv`. Every reading is in `88_panel_verdicts.csv`.

## 5. Findings

**Specificity: each planted failure trips its rung.**

1. SHIFTED fails L3 in every reading at both doses (residual +2.11 and +4.70 points). It keeps R1, R2 and R3 in every framing, and L2 stays unresolved, as for REAL. L1 moves toward unresolved as the tilt grows (pass .725 at 2.1, .159 at 4.7). It does not fail L1.
2. COMPRESSED fails L1 in every replicate (87% "compressed, both tails in deficit", 13% "misfit deficit"; misfit ratio .33, overfit .38). It passes L3. It keeps R1 in 98.6% and R2 in all, and it loses R3 in 22%.
3. ORACLE-INDEP fails L1 in every replicate (misfit ratio 2.67, overfit .44). It has no general factor (lambda1/lambda2 1.33, minimum loading .15), so R1 and R2 fail and R4 has no verdict. It passes L3 and leaves L2 unresolved.
4. STEEPENED-2x fails L2 in 92% of per-model readings with pairs SE and in 98 to 99% with the draw-only SE. Pooled it fails in every replicate. L1, R1, R2 and R3 stay passing. It fails L3 at 1 point in almost every reading and passes at 2 points in 86% (Low-income residual +1.29). Per model, Low-High is named "steepened" alone in 71% (pairs SE, half reference), 74% (pairs, audit precision), 90% (draw-only, half) and 94% (draw-only, audit precision); 97% pooled. The true ratio is 2.03.
5. NONINVARIANT fails R4 at margin .08 in all 24 replicates (48 of 48 framings). The failing step is income metric, RMSEA_D mean .201 (bootstrap interval .176 to .225). R1 and R2 hold in every framing and L2 and L3 match REAL. R3 fails in 99%. L1 goes from 93% pass to 96% unresolved (misfit ratio 1.45), so the plant is not invisible to L1, but L1 does not fail it.

**Which rung no simulator passes.**

6. No type passes L2 at the model level, including REAL. REAL is unresolved in 100% of per-model and pooled readings, under every SE and reference variant. Across all contrasts, variants and scopes, "kept" alone was returned only for ORACLE-INDEP on Low-High income under the draw-only SE at audit precision (22 of 276 per-model readings, 8%). REAL was never named "kept" alone. The only type that L2 ever calls faithful is the one that fails L1 and L4. L2 can fail a simulator and cannot pass one at NHANES precision. This repeats CONTROLS.md finding 3 for every type.
7. No type passes R4, including REAL. At corpus group sizes (24 to 40 personas per group) REAL R4 is unresolved in 23 of 24 replicates at .08 and 24 of 24 at .05. The step that most often blocks a match is race metric (holds in 25% of REAL framings), then sex scalar (40%) and sex metric (48%). Race scalar (98%) and income scalar (100%) hold reliably. Because R4 cannot pass, level 4 is never awarded to any type.
8. Real people fail the gate at k = 30. REAL has median phi(30) .70 and needs about 52 draws for the minimum rule and 116 for the recommended rule. ORACLE-INDEP passes the minimum gate in every reading (phi(30) .90, k_min 14) because independent items have smaller error variance relative to persona variance. The gate rewards between-persona separation, and it is passed by the simulator that fails L1 and L4 hardest.

**R4 calibration and power.**

9. R4 is specific at .08 with one exception. REAL failed R4 in 1 of 24 replicates (replicate 6, narrative framing, sex scalar RMSEA_D .111 with interval .082 to .137). The NHANES sex scalar estimate at full sample is .067, close to .08, so a half-sample panel can exceed .08 by chance. At .05 REAL never failed.
10. R4 detects a strong planted metric violation at corpus group sizes. The NONINVARIANT plant replaces 3 of 8 items with independent draws in one income group. A weaker plant was not tested.
11. At .05 R4 cannot fail the NONINVARIANT plant, because the NHANES income metric step is itself indeterminate at .05 and a mismatch cannot occur on an indeterminate reference step. The plant's failure shows up only at .08. At .05, STEEPENED-2x failed R4 in 1 of 4 replicates on race metric (RMSEA_D above .05 in 1 of 8 framings), a false R4 failure for a simulator with no invariance plant.
12. R4 counts on the five types other than REAL and NONINVARIANT are n = 4 and set only an upper bound.

## 6. Deviations from the requested design, and why

1. SHIFTED and STEEPENED use a per-cell exponential tilt of donor weights. The first version moved each draw to a donor vector at total t + d drawn from the whole donor half. At d = 4.7 that removed most all-zero vectors and broke the item correlation structure (R1 failed: eigenvalue ratio 2.0, minimum loading .14), so a mean shift also failed L4. The tilt keeps real vectors and their covariance. This was changed before the main run.
2. STEEPENED raises only the Low-income personas, as the 83b L2 gap plant raises only the first-named group. The Low-Middle ratio therefore also rises (about 2.9).
3. R4 step verdicts are compared with the published NHANES full-sample verdicts, not with R4 run on the reference half. The audit compares models with the full sample, and running R4 on each reference half would have doubled R4 cost.
4. R4 used K_PERM 30, bootstrap 100 and polychoric bootstrap 2 (audit 200 / 200 / 100), and ran on 24 replicates for REAL and NONINVARIANT and 4 for the others. One R4 population costs about 55 s and full R4 on every replicate and type would not fit the 60-minute budget.
5. L1 and L4 are read on pseudo-model 0 only, as in 83b. Gate, L2 and L3 are read on all four.
6. 70 replicates were launched rather than 100, from the timed single replicate and the R4 cost. Replicate 4 crashed and was not rerun, leaving 69.
7. SHIFTED-4.7 is near the edge of what tilting can reach in low-scoring cells. Tilt effective sample size had median 41 and minimum 1.6 draws per cell, and the smallest realised cell shift was 3.13 against the 4.7 target. The mean realised shift was 4.70. The low ESS is the likely reason L1 drifts to unresolved at that dose.
8. The level-4 rule used here is R1, R2 and R4 in both framings, which is the paper's rule (02_ladder.tex). 83b and 83c scored level 4 on R1, R2 and R3. R3 is reported here as a diagnostic.

## 7. Problems found in existing code

1. `81c_l4_invariance.py` `start_values` (line 191) divides by item SDs without a guard. When an item has zero variance in one group of a bootstrap resample, the correlation matrix has NaN entries and `81_l4_lib.one_factor_uls` raises `LinAlgError: Eigenvalues did not converge`. This killed replicate 4 during an R4 bootstrap fit. The audit's own R4 run did not hit it, but any small-group or low-variance input can.
2. 83b and 83c score the L4 positive control on R1, R2 and R3. The paper's level-4 rule is R1, R2 and R4, with R3 a diagnostic. CONTROLS.md finding 1 ("passes L4 ... in 100%") is therefore a statement about R1 to R3, not about level 4 as the paper defines it.
3. CONTROLS.md says R4 is not simulated. This panel shows that a faithful simulator does not pass R4 at corpus group sizes, so "R4 holds" is not reachable for a faithful model under this design.
4. The 83b L2 gap and L3 shift plants change cell means analytically and never produce draws, so they cannot show that L1 and L4 stay passing under those failures. This panel's tilt-based plants do produce draws.
5. 83b never ran the gate on the positive control. Real people fail the minimum gate at k = 30 in every reading here.
6. 83b reads L1 on the pooled "both" unit. The paper's L1 rule is per framing. Here REAL passes per framing in 93% of replicates and is unresolved in 7%, against 99.4% pass in 83c.
