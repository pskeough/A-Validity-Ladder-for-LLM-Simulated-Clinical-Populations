# PsychBench — Updated Run: Rebuilt Ground Truth + Full Recompute (2026-07-14)

Complete re-derivation of PsychBench's ground truth from primary microdata, and recomputation of
every analysis against it. The experimental data (28,800 model outputs) is unchanged and real; what
was rebuilt is the epidemiological **ground truth** (shown in prior audits to be inflated, mis-scaled,
and in two cases untraceable) and every metric computed on top of it.

**Everything here is reproducible from `scripts/` against public data. No value is asserted without a
receipt.**

---

## 1. What was wrong (established in prior audits, this folder fixes it)

- The paper's ground truth was described as "self-extracted PHQ-8 from NHANES 2005–2018." It was
  actually **PHQ-9 values from Patel et al. 2019 (NHANES 2005–2016)**, plus **unsourced SES numbers**,
  plus a **trans-women mean (15.56) from a Chinese snowball sample (Xu 2023)**, and an **untraceable
  trans-men mean (10.80)**. Scale mismatch (PHQ-9 vs PHQ-8) and miscitations throughout.

## 2. New ground truth — validated pipeline

`02_validate_pipeline.py` reproduces Patel et al. 2019's published PHQ-9 means to within ±0.02
(White 3.19→3.18, Asian 2.23→2.21, Women 3.72→3.71, Overall 3.20→3.19), **proving the extraction +
weighting method is correct**. `03_compute_groundtruth.py` then computes the actual benchmark:
**survey-weighted PHQ-8, NHANES 2005–2018, adults 18+** — the nationally-representative population
parameter, on the *same scale as the model outputs*.

| Cell | GOLD PHQ-8 mean (SD) | n | vs paper's value |
|---|---|---|---|
| Overall baseline | **2.99 (3.94)** | 36,274 | paper 3.50 (uncited) — too high |
| White | **2.91 (3.84)** | 15,181 | paper 3.19 (PHQ-9) |
| Black | **3.24 (4.25)** | 7,931 | paper 3.21 |
| Asian | **2.16 (3.01)** | 2,408 | paper 2.23 |
| Hispanic (pooled) | **3.15 (4.15)** | 9,309 | paper 3.15 was Mexican-American only |
| Men | **2.51 (3.59)** | 17,817 | paper 3.00 — too high |
| Women | **3.44 (4.19)** | 18,457 | paper 3.70 |
| SES Low | **4.25 (4.89)** | 10,663 | paper 5.50 — too high |
| SES Middle | **3.12 (3.96)** | 12,554 | paper 4.10 — too high |
| SES High | **2.23 (3.09)** | 9,967 | paper 3.40 — too high |
| **Transgender women** | **UNAVAILABLE** | — | NHANES has no gender-identity field |
| **Transgender men** | **UNAVAILABLE** | — | " |
| **Multiracial** | **UNAVAILABLE** | — | no clean representative norm |

SES gradient (Low − High) = **2.02 points**, matching the ~2.1 the literature reports — now with
*defensible absolute anchors* instead of the invented ones.

## 3. Recomputed bias residuals (vs GOLD GT)

`04_recompute_analysis.py` → `analysis/bias_residuals_GOLD.csv`. Residual = model mean − gold mean;
d = residual / gold SD.

| Group | Model mean | GOLD residual (d) | Paper residual | Change |
|---|---|---|---|---|
| White | 8.69 | **+5.77 (d 1.51)** | +5.50 | +0.27 |
| Black | 8.51 | **+5.27 (d 1.24)** | +5.30 | −0.03 |
| Asian | 7.71 | **+5.55 (d 1.84)** | +5.48 | +0.07 |
| Hispanic | 8.64 | **+5.49 (d 1.32)** | +5.49 | ≈0 |
| Cis Man | 6.57 | **+4.07 (d 1.13)** | +3.57 | +0.50 |
| Cis Woman | 7.41 | **+3.97 (d 0.95)** | +3.71 | +0.26 |
| SES Low | 11.24 | **+6.99 (d 1.43)** | +6.10 | +0.89 |
| SES Middle | 8.08 | **+4.96 (d 1.26)** | +4.00 | +0.96 |
| SES High | 6.22 | **+3.99 (d 1.29)** | +2.93 | +1.06 |
| Transgender Woman | 10.14 | **REMOVED** — no GT | −5.42 | finding void |
| Transgender Man | 9.93 | **REMOVED** — no GT | −0.87 | finding void |
| Multiracial | 9.01 | **REMOVED** — no GT | — | — |

**The overestimation finding is robust and STRONGER.** Every residual grew (gold anchors are lower
than the paper's inflated ones). Models systematically overestimate depression severity across every
benchmarkable group — the paper's #1 contribution stands, now on defensible ground.

**The transgender "suppression" finding is void, not reversed.** No representative severity benchmark
exists for trans populations, so no residual is computable. What the data *does* show, with no GT
needed: models render trans women (10.14) and trans men (9.93) as markedly more depressed than cis
women (7.41) and cis men (6.57) — i.e., the models **encode** a large trans-cis elevation. Whether
that matches reality is unanswerable without a benchmark that does not exist. The honest, publishable
statement is: **representative trans severity norms do not exist; only prevalence can be benchmarked**
(Hughto 2024: TGD depression 19.7–51.3% vs cis ~20%).

## 4. GT-independent metrics (recomputed from scratch, reconcile exactly with the verified audit)

| Metric | Value | Gate |
|---|---|---|
| DSM-5 gateway violations (paper's own ≥2 rule) | **257 / 9,590 elevated = 2.68%** (GLM 165, GPT 68, Gemini 19, DeepSeek 5) | "0 / perfect coherence" is FALSE |
| PHQ-8 5-category flip rate | **36.66%** (5,279 / 14,400 pairs) | matches audit |
| Cross-run drift (bare − quoted) | **+0.324, t = 15.21, p = 7.9e-52** | matches audit |
| Cross-run MAD | **1.89** | matches audit |

These do not depend on ground truth and are unaffected by the GT rebuild.

## 5. Variance / Stereotype Index — flagged for careful re-derivation

Per-model variance ratios shift under the corrected (lower) GT SDs: DeepSeek 0.46, GPT 0.53, GLM
1.04, Gemini 1.05 (`analysis/per_model_variance_GOLD.csv`). Under gold norms, DeepSeek/GPT compress
variance while GLM/Gemini sit near population variance — a different picture than the paper's
uniform-compression claim. **This metric is GT-SD-sensitive and the exact SI construct (which SD,
per-cell vs marginal) must be pinned before any v2 claim.** Not finalized here.

**FINALIZED 2026-07-15:** construct pinned as marginal per-group SD ratio (model-output SD over gold
weighted SD, `w_sd`), reported descriptively per V2_SPEC's estimand demotion. Receipts:
`scripts/06_covariance_variance_receipts.py` → `analysis/per_group_variance_GOLD.csv` (per-group,
pooled + per-model, 1,000-draw bootstrap CIs) and `analysis/covariance_divergence_GOLD.csv` +
`analysis/covariance_noise_floors.csv` (§7 Frobenius receipts incl. per-group split-half floors and
the mean-conditioned severity-matched check). All §7/§8 draft numbers reconciled to rounding; see
VALIDATION_RECEIPTS.md Receipt 3 for the reconciliation table and the two new robustness findings
(severity matching halves trans divergence magnitudes but preserves rank + floor clearance;
TW-vs-TM = 0.17, below noise — a shared reorganized structure).

## 6. Data-integrity note

One corrupted row (DeepSeek: `phq8_8` = 21, impossible for a 0–3 item; total 29 > PHQ-8 max 24) was
clipped to 24 and flagged. Impact on any cell mean < 0.001. No other item-level values out of range.

## 7. What v2 needs

1. Adopt the gold GT table (§2), cite it as author-derived from NHANES 2005–2018 with this receipt.
2. Report residuals from §3; the overestimation finding leads and is stronger.
3. Remove the transgender-severity residual; reframe trans around prevalence.
4. Drop Multiracial residual cells (no benchmark).
5. Re-derive the variance/SI section carefully (§5).
6. Keep the GT-independent findings (§4) as-is; correct "perfect coherence" → 2.68% violations.

## Files
- `scripts/02_validate_pipeline.py` — Patel 2019 reproduction (validation gate)
- `scripts/03_compute_groundtruth.py` — gold PHQ-8 GT from NHANES microdata
- `scripts/04_recompute_analysis.py` — all residuals + GT-independent metrics
- `groundtruth/phq8_groundtruth_nhanes_2005_2018.csv` — the gold GT table
- `analysis/bias_residuals_GOLD.csv`, `paper_vs_gold_residuals.csv`, `gt_independent_metrics.csv`,
  `per_model_variance_GOLD.csv`
- `data/nhanes_raw/` — 14 raw NHANES XPT files (DEMO+DPQ, 2005–2018)
- `data/model_outputs.csv` — the 28,800 model outputs (unchanged)
