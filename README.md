# A Validity Ladder for LLM-Simulated Clinical Populations

Code, data and receipts for a methods paper in preparation. Patrick S. Keough, 2026.

The validity ladder is a set of checks to run on a population simulated by a language model before it is used in place of people. A gate checks that persona-level scores are reliable. Four levels then compare the simulated data with a human reference:

| Rung | Question | Reference | Scripts | Report |
|---|---|---|---|---|
| Gate | Is a persona's mean score reliable over k draws? | generalizability theory | 82, 82a-f | `paper_brm/analysis_brm/GATE.md` |
| Level 1 | Does a single draw answer like one person? | NHANES person-fit (lz*) and gateway items at matched totals | 79, 79a-f | `L1.md` |
| Level 2 | Are subgroup gaps the right size? | NHANES gaps; ratio intervals read against fixed regions | 78a-f | `L2.md` |
| Level 3 | Is the overall level right? | NHANES, post-stratified to the persona design; equivalence tests | 80, 80a-d | `L3.md` |
| Level 4 | Do the items hang together as they do in people? | NHANES factor structure and invariance | 81, 81b-e | `L4.md` |
| Controls | Do real people pass, and are planted failures detected? | NHANES split into donor and reference halves | 83, 83a-g | `CONTROLS.md` |

A level-1 failure rules out reading a single draw as one person. Levels 2 to 4 read persona means and still run.

## The worked example

28,800 PHQ-8 assessments: four models (GPT-4o-mini, Gemini-3-Flash, DeepSeek-V3, GLM-4.7), 120 personas, two framings (clinical and narrative) and 30 draws each. The reference is NHANES 2005-2018, with 2017-2020 and 2021-2023 as era checks.

Results, each with its receipt in the report named above:

- **Gate.** Persona means are reliable at k = 30 for every model except GPT-4o-mini under the narrative framing (phi .887 against .90). Averaging both framings gives .967.
- **Level 1.** All four models fail. Three give too few atypical and too few highly regular answer vectors at matched totals. GLM-4.7 shows a misfit deficit only.
- **Level 2.** Income gaps are steepened in every model: standardised ratios of 1.3 to 8.4 per model and 2.1 to 4.8 pooled. Sex is mixed across models. None of the earlier race verdicts survives. At equal income, sex and marital status the NHANES Black-White and Hispanic-White gaps are small and slightly negative.
- **Level 3.** Every model is 2.1 to 4.7 PHQ-8 points above NHANES after post-stratification, and none passes at 0.2 SD, 1 point or 2 points against 2005-2018. Against 2021-2023, Gemini-3-Flash and GLM-4.7 pass the overall row at 2 points.
- **Level 4.** No model passes. DeepSeek-V3 and GPT-4o-mini show no general factor. In GLM-4.7 and Gemini-3-Flash the factor comes from differences between personas, and draws of one persona show none. Income fails invariance.
- **Controls.** A pseudo-model built from NHANES respondents passes levels 1 and 3 and the level-4 structure rules. R4 was not simulated. At levels 3 and 4 and for compression at level 1, the models' real distortions lie beyond the dose at which a planted failure is detected. At level 2, per-model detection needs a threefold gap, and at NHANES precision the level can detect a distorted gap but cannot certify a faithful one.

## Reproducing

```
pip install -r requirements.txt
python scripts/00_download_nhanes.py     # about 50 MB from CDC, checked by SHA-256
python scripts/run_all.py                # about 2 hours on 7 cores; --fast skips the 70-minute simulation
```

Every script checks itself against an earlier computation before it writes (the receipts CSVs), and `run_all.py` stops at the first failed check. No script calls an API. The published outputs are already in `analysis/brm/`, so a rerun can be compared file by file.

## Layout

```
data/model_outputs_v3.csv     the corpus, one row per draw, with row_source and phq8_valid (script 76)
data/model_outputs_v2.csv     the previous release, input to script 76
data/raw/                     the original run outputs script 76 checks against
generation/                   the code that produced the corpus (see generation/README.md)
groundtruth/                  published NHANES PHQ-8 group anchors, used as a check
analysis/brm/                 every output of scripts 76-83g
analysis/*.csv, *.jsonl       earlier outputs the scripts reproduce as a check, and the logged
                              generations of the prompt and decoding controls
scripts/                      the analysis, 00-83g, and run_all.py; 84-87 build the paper's
                              figure, worked case, receipt map (Supplement S2) and supplement tables
paper_brm/manuscript/         the paper and supplement (LaTeX sources and PDFs); a line ending in
                              "% R: file" names the receipt for the numbers on it
paper_brm/analysis_brm/       one report per rung, plus the controls
paper_brm/external/           the ladder on three releases by other groups and the PersonaLLM shakedown (see below)
paper_brm/level2_rule/        the comparison of candidate level-2 rules that led to the ratio rule
```

## Provenance

- **Resent rows.** The December clinical run left 1,532 calls without a usable answer (1,219 GLM-4.7, 313 DeepSeek-V3). In January the same calls were sent again by four scripts in `generation/`: `recovery/verify_run1_refusals.py` (1,188 rows, written in place by `recovery/merge_recovered_data.py`), `retry_failed.py` (157), `recovery/slow_recovery.py` (125) and `recovery/last_mile_recovery_opt.py` (62). The first three send the corpus system prompt; the last sends a shortened one. All four put the persona's registry id where `main.py` put a random id. Script 87a matches every in-place row to the resend outputs in `data/raw/` (paper Section 3 and Supplement S1.8). Every row carries its source in `row_source`. Each rung report gives a sensitivity without the resent rows. Dropping them changes one level-2 verdict (the standardised pooled sex gap) and no other.
- **Restored rows.** 43 GPT-4o-mini rows from the original 28 Dec 2025 run had been overwritten in v2. v3 restores them from the original output file.
- **Frozen rules.** `paper_brm/LADDER_SPEC.md` states every rule, threshold and stop of the ladder. It was frozen by commit `91a1b10` (2 October 2026), after a shakedown on PersonaLLM (`paper_brm/external/personallm/SHAKEDOWN.md`) and before the Bisbee et al. data were opened; later code changes are bug fixes only. No analysis was registered with a public registry. `analysis/prompt_control_design.json`, `analysis/decoding_control_design.json` and the dated `analysis/decoding_control_preregistration.json` were written before those runs.
- **Prompt control.** The "orig" arm of the prompt control asks for 20 PCL-5 items where the original script asked for 4 (`analysis/brm/l2_prompt_control_receipts.csv`).
- **External data.** `paper_brm/external` runs levels 2 and 3 on Meister, Guestrin & Hashimoto (2024, OpinionQA; arXiv:2411.05403, repository commit 36869b5) and Argyle et al. (2023, Study 3; Harvard Dataverse doi:10.7910/DVN/JPV20K). It also runs the gate and levels 2 and 3, on the frozen rules, on Bisbee et al. (2024; Harvard Dataverse doi:10.7910/DVN/VPN481, CC0): `paper_brm/external/scripts/bisbee_*.py`, results in `paper_brm/external/results/bisbee/`, including a reproduction of the authors' row counts and Supplementary Section 8 coefficients (`receipts_rr1.csv`). Raw files are not redistributed here. The results files are included, and the level-2 verdicts for OpinionQA and Argyle under the current rule are in `analysis/brm/l2_external_r3*.csv` (script 78e).

## Earlier release

The repository previously held the preprint-era pipeline (scripts 01-73) and a preprint draft. That version is kept at the tag `preprint-2026-09`. Its level-2 rule, its level-1 rule and several of its results have been replaced by the analysis here.

The corpus was first reported in *Plausible Patients, Impossible Populations* (arXiv:2604.17359).

## AI assistance

The analysis code was written with AI coding assistance to the author's specification. Every reported number is printed by a script into a CSV in `analysis/brm/`.

## Licence

Code under MIT (`LICENSE`). Data, derived files and reports under CC BY-NC-ND 4.0 (`LICENSE-DATA`). NHANES files are public domain.
