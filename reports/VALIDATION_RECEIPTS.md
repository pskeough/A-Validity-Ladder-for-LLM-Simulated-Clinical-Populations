# Validation Receipts — NHANES Ground-Truth Pipeline

Two independent external validations. The pipeline that computed the gold ground truth reproduces two
different published, peer-reviewed / federal sources — different metrics, subsets, and scales — so the
extraction and weighting are verified by the strictest available standard: matching numbers other
people published.

## Validation 1 — Patel et al. 2019 (mean reproduction)

Source: Patel JS, Oh Y, Rand KL, Wu W, Cyders MA, Kroenke K, Stewart JC. *Measurement invariance of
the PHQ-9 depression screener in U.S. adults across sex, race/ethnicity, and education level: NHANES
2005–2016.* Depression and Anxiety. 2019;36(9):813–823. Table 1 (PHQ-9, unweighted descriptives).

| Group | Patel 2019 published | This pipeline (unweighted) | Δ |
|---|---|---|---|
| Overall | 3.20 (4.27) | 3.19 (4.26) | 0.01 |
| Men | 2.67 (3.87) | 2.66 (3.86) | 0.01 |
| Women | 3.72 (4.57) | 3.71 (4.56) | 0.01 |
| White | 3.19 (4.19) | 3.18 (4.18) | 0.01 |
| Black | 3.21 (4.35) | 3.20 (4.35) | 0.01 |
| Asian | 2.23 (3.07) | 2.21 (3.03) | 0.02 |
| Mexican-American | 3.15 | 3.13 | 0.02 |

Result: **reproduced to ±0.02 on every cell.** Confirms both the extraction and that the paper's
race/gender GT truly originated in this PHQ-9 table (not self-extracted PHQ-8 as claimed).

## Validation 2 — Brody et al. 2018 (prevalence reproduction)

Source: Brody DJ, Pratt LA, Hughes JP. *Prevalence of depression among adults aged 20 and over:
United States, 2013–2016.* NCHS Data Brief No. 303. National Center for Health Statistics, 2018.
(PHQ-9 ≥ 10, MEC-weighted, adults 20+.)

| Group | Brody 2018 published | This pipeline (weighted) | Δ |
|---|---|---|---|
| Overall | 8.1% | 8.1% | 0.0 |
| Women | 10.4% | 10.4% | 0.0 |
| Men | 5.5% | 5.5% | 0.0 |

Result: **exact match.** Confirms the survey-weighting (WTMEC2YR) is applied correctly — the method
used to produce the gold weighted PHQ-8 population means.

## Bearing on the gold ground truth

The gold GT (`groundtruth/phq8_groundtruth_nhanes_2005_2018.csv`) uses the *same* extraction
(validated by #1) and the *same* MEC weighting (validated by #2), applied to PHQ-8 across all 7 cycles
(2005–2018). Both the point-estimate method and the weighting are therefore independently confirmed
against published figures. The only cells with no gold value are those for which **no representative
US data exists at all** (transgender, multiracial) — an absence documented rather than filled.

## Receipt 3 — Section 7/8 recompute (2026-07-15)

The two draft sections that previously cited v1 carry-over numbers with no recomputed backing are now
receipted by `scripts/06_covariance_variance_receipts.py` (SEED=42), which writes:

- `analysis/covariance_divergence_GOLD.csv` — pairwise Frobenius distances between per-cohort 8×8
  PHQ-8 item correlation matrices, raw and severity-matched (PHQ-8 band-matched subsampling, 100
  resamples, 95% CI), with per-pair floor-clearance flags.
- `analysis/covariance_noise_floors.csv` — per-cohort split-half noise floors (200 draws, mean + p95).
- `analysis/per_group_variance_GOLD.csv` — per-group SD ratio vs gold SD with 1,000-draw bootstrap
  CIs, pooled and per model.

**Reconciliation against the draft's §7/§8 numbers — all confirmed to rounding:**

| Draft claim | Recomputed | Verdict |
|---|---|---|
| Trans women divergence 1.14 | 1.137 (vs cisgender men) | ✓ |
| Trans men divergence 1.06 | 1.061 (vs cisgender men) | ✓ |
| Cis women divergence 0.32 | 0.315 (vs cisgender men) | ✓ |
| Asian vs White 0.37 / Black 0.24 / Hispanic 0.23 | 0.374 / 0.239 / 0.229 | ✓ |
| Noise floor 0.17, p95 0.23 (White) | 0.170 / 0.229 | ✓ |
| SES SD ratios 0.67–0.86 | 0.669–0.859 | ✓ |
| Asian SD ratio 1.12 | 1.122 | ✓ |
| Per-model SI DeepSeek 0.46 / GPT 0.53 / GLM 1.04 / Gemini 1.05 | unchanged (04 script) | ✓ |

Reference-group note: the gender divergences are all measured against the **cisgender-man** cohort;
the draft's original phrasing left this ambiguous and has been corrected in §7.

New results from the V2_SPEC-mandated robustness checks (now folded into §7):
- **Mean-conditioned check:** severity-band matching shrinks trans divergences to 0.55 (TW) and
  0.48 (TM) against a matched cis-cis contrast of 0.21 — severity composition accounts for roughly
  half the raw magnitude; rank ordering and floor clearance survive.
- **Per-group floors:** Hispanic-vs-White (0.229) falls inside Hispanic's own p95 floor (0.239);
  Black-vs-White (0.239) clears its floors marginally. Only Asian and the trans contrasts clear
  floors decisively.
- **TW vs TM = 0.170, below both floors** — the two transgender cohorts share a common reorganized
  structure.

## Receipt 4 — FDR ledger (2026-07-15)

The machine-readable per-test ledger promised in draft §3.5 now exists:
`scripts/07_fdr_ledger.py` (SEED=42) → `analysis/fdr_ledger.csv`. The family is the 24 tests behind
every significance claim in the draft: 9 residual one-sample t (§4 Table 2), drift paired t (§6),
flip-rate and gateway-violation model-heterogeneity chi-squares (§6/§5), 2 trans-cis Welch contrasts
(§9), and 10 covariance-divergence label-permutation tests (§7, 500 draws each).
Benjamini-Hochberg is applied across the full family at once: **24/24 significant at q<0.05** —
expected at this N, which is exactly why §3.5 releases the ledger instead of summarizing by count.
Interpretive guard baked into the CSV: T17 (Hispanic-White), T18 (Multiracial-White), and T24 (TW-TM)
are statistically detectable but sit below their per-group p95 split-half floors; the ledger's note
column marks them so it cannot be read as contradicting §7's "within noise" magnitude language.
Detectability (permutation p) and magnitude (floor clearance) answer different questions; the paper's
§7 claims rest on the latter. §8's variance ratios are descriptive by design and carry no tests.
