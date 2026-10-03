# A Validity Ladder for LLM-Simulated Populations

This repository holds the code, data and receipts for "A Validity Ladder for LLM-Simulated Populations" (Patrick S. Keough, 2026). The validity ladder is a set of checks to run on a sample simulated by prompting a language model with demographic personas, before that sample is used in place of people. A gate tests whether persona-level scores are stable enough to read, and four levels above it each test one use of the sample against human data, from single draws read as individual cases to group gaps, a group's level and the structure of a multi-item scale. A simulated sample supports only the claims whose rungs it passes.

| Rung | Question | Reference in the worked example | Scripts | Report |
|---|---|---|---|---|
| Gate | Is a persona's mean score reliable over k draws? | generalizability theory | 82, 82a-f | `paper_brm/analysis_brm/GATE.md` |
| Level 1 | Does a single draw answer like one person? | NHANES person fit (lz*) at matched totals | 79, 79a-f | `L1.md` |
| Level 2 | Are subgroup gaps the right size? | NHANES gaps, read as Fieller ratio intervals | 78a-f | `L2.md` |
| Level 3 | Is each group's level right? | NHANES, post-stratified to the persona design; equivalence tests | 80, 80a-d | `L3.md` |
| Level 4 | Do the items hang together as they do in people? | NHANES factor structure and invariance | 81, 81b-e | `L4.md` |
| Controls | Do real people pass, and are planted failures detected? | NHANES split into donor and reference halves | 83, 83a-i, 88-90 | `CONTROLS.md` |

The manuscript and supplement, with LaTeX sources and PDFs, are in `paper_brm/manuscript/`.

## Results

The worked example covers 28,800 PHQ-8 assessments from GPT-4o-mini, Gemini-3-Flash, DeepSeek-V3 and GLM-4.7, with 120 personas, two prompt framings and 30 draws each, compared against NHANES 2005-2018. All four models pass the gate, needing 3 to 6 draws under the minimum rule, and fail every level above it. Single draws vary less than people's answers, and the low-to-high income gap runs 1.91 to 5.07 times the population gap. After post-stratification every model sits 2.1 to 4.7 PHQ-8 points above NHANES, and no model reproduces the structure of the scale.

Pseudo-models built from real NHANES respondents are rarely failed, passing the gate and level 3 in every reading and level 1 in 92.9% of replicates, while failures planted into the same respondents are caught at the rung each one targets. On the frozen rules, level 2 keeps 3 of 77 contrasts in the release of Bisbee et al. (2024), 2 of 61 in Argyle et al. (2023, Study 3) and 1 of 154 in OpinionQA (Meister et al.). Five of those six kept contrasts are partisan gaps.

Each result has its receipt in the report named in the table above, and Supplement S2 maps every number in the article to the file that prints it.

## Running the ladder on another sample

Script `scripts/91_ladder_core.py` restates the gate, level 1 and level 4 for any scale of J items with M ordered categories, taking the reference's sampling scheme as an input. Script 91a checks that it returns the PHQ-8 numbers, and script 92 runs it on PersonaLLM BFI-44 answers against the Twin-2K-500 panel. Levels 2 and 3 for other releases are in `paper_brm/external/scripts`, and every rule and threshold is stated in `paper_brm/LADDER_SPEC.md`, frozen at commit `91a1b10` on 2 October 2026 before any external dataset was analysed.

## Reproducing

```
pip install -r requirements.txt
python scripts/00_download_nhanes.py     # about 50 MB from CDC, checked by SHA-256
python scripts/run_all.py                # scripts 76-83g; --fast skips the two long simulations
```

`run_all.py` rebuilds the corpus, every rung and the controls. Every script checks itself against an earlier computation before it writes, and `run_all.py` stops at the first failed check. The later scripts run on their own, in numeric order: 83h-83i, 88-88e and 90-90b (the controls under the frozen rules, the simulator panel and panel power), 84 and 84d-84f (figures), 85 (the worked case), 86 (receipts), 87-87a (supplement tables and resend sources), 89 (level-3 specification range) and 91-92 (the instrument-general ladder and the PersonaLLM shakedown). No analysis script calls an API. The published outputs are already in the repository, so a rerun can be compared file by file.

## Layout

```
data/model_outputs_v3.csv     the corpus, one row per draw, with row_source and phq8_valid (script 76)
data/model_outputs_v2.csv     the previous release, input to script 76
data/raw/                     the original run outputs script 76 checks against
generation/                   the code that produced the corpus (see generation/README.md)
groundtruth/                  published NHANES PHQ-8 group anchors, used as a check
analysis/brm/                 every output of scripts 76-87
analysis/*.csv, *.jsonl       earlier outputs the scripts reproduce as a check, and the logged
                              generations of the prompt and decoding controls
scripts/                      the analysis, 00-92, and run_all.py
paper_brm/LADDER_SPEC.md      the frozen rules
paper_brm/analysis_brm/       one report per rung, the controls, the simulator panel (88) and
                              panel power (90)
paper_brm/external/           the ladder on three releases by other groups and the PersonaLLM shakedown
paper_brm/level2_rule/        the comparison of candidate level-2 rules that led to the ratio rule
paper_brm/manuscript/         the paper and supplement; a line ending in "% R: file" names the
                              receipt for the numbers on it
paper_brm/arxiv/build_arxiv.py   builds the arXiv source package from the manuscript
```

## Provenance

- **Resent rows.** The December clinical run left 1,532 calls without a usable answer (1,219 GLM-4.7, 313 DeepSeek-V3). In January the same calls were sent again by four scripts in `generation/`: `recovery/verify_run1_refusals.py` (1,188 rows, written in place by `recovery/merge_recovered_data.py`), `retry_failed.py` (157), `recovery/slow_recovery.py` (125) and `recovery/last_mile_recovery_opt.py` (62). The first three send the corpus system prompt; the last sends a shortened one. Script 87a matches every in-place row to the resend outputs in `data/raw/` (Supplement S1.8). Every row carries its source in `row_source`, and each rung report gives a sensitivity without the resent rows.
- **Restored rows.** 43 GPT-4o-mini rows from the original 28 Dec 2025 run had been overwritten in v2. v3 restores them from the original output file.
- **Control designs.** `analysis/prompt_control_design.json`, `analysis/decoding_control_design.json` and the dated `analysis/decoding_control_preregistration.json` were written before those runs. The "orig" arm of the prompt control asks for 20 PCL-5 items where the corpus prompt asked for 4 (Supplement S1.9). No analysis was registered with a public registry.
- **External data.** `paper_brm/external` runs levels 2 and 3 on Meister, Guestrin & Hashimoto (2024, OpinionQA; arXiv:2411.05403, repository commit 36869b5) and Argyle et al. (2023, Study 3; Harvard Dataverse doi:10.7910/DVN/JPV20K). It also runs the gate and levels 2 and 3 on Bisbee et al. (2024; Harvard Dataverse doi:10.7910/DVN/VPN481, CC0), including a reproduction of the authors' row counts and Supplementary Section 8 coefficients (`results/bisbee/receipts_rr1.csv`). Raw files are not redistributed here; the results files are included.

## Earlier release

The repository previously held the preprint-era pipeline (scripts 01-73) and a preprint draft, kept at the tag `preprint-2026-09`. Its level-1 and level-2 rules and several of its results have been replaced by the analysis here. The corpus was first reported in *Plausible Patients, Impossible Populations* (arXiv:2604.17359).

## AI assistance

The analysis code was written with AI coding assistance (Claude Code) to the author's specification and reviewed by the author. Every reported number is printed by a script into a file in this repository, and `scripts/86_receipts.py` checks each number in the manuscript against that file.

## Licence

Code under MIT (`LICENSE`). Data, derived files and reports under CC BY-NC-ND 4.0 (`LICENSE-DATA`). NHANES files are public domain.
