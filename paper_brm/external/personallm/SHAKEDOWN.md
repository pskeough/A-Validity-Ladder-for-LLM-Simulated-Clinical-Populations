# Shakedown: the ladder on PersonaLLM BFI-44 against Twin-2K-500

Run 2026-10-02 by `scripts/92_personallm_shakedown.py` on the frozen rules (`paper_brm/LADDER_SPEC.md`).
Outputs in this folder: `92_gate.csv`, `92_l1.csv`, `92_l1_tau.csv`, `92_l4.csv`, `92_unparsed.csv`.
Earlier runs on the pre-freeze rules: `../personallm_prefinal/` and the logs `../personallm_run_*.log`.

## Data

- Simulated: PersonaLLM (Jiang et al., 2024), github.com/hjian42/PersonaLLM at commit 286c149, MIT
  licence. BFI-44 answers from GPT-3.5-turbo-0613, GPT-4-0613 and Llama-2, 32 Big Five persona
  types, 10 draws each at temperature 0.7. 960 answer sets parsed, 0 unparsed.
- Reference: Twin-2K-500 (Toubia et al., 2025), Hugging Face LLM-Digital-Twin/Twin-2K-500, CC BY
  4.0. 2,058 respondents with complete BFI-44, sex, age band and race.
- Five scales, items reverse-keyed and scored 0-4.

## What ran

Gate (k = 10), level 1 and level 4 R1, R2, R3 per model and scale. Levels 2, 3 and R4 are not
applicable, because PersonaLLM's personas are trait instructions with no attribute that the
reference records.

## What broke on the pre-freeze rules, and the fix in the frozen rules

1. **Code.** The audit code was written for 8 items with 4 categories. `91_ladder_core.py` restates
   the gate, level 1 and level 4 for J items and M categories and reproduces the audit libraries
   on the PHQ-8 to 1e-12 (`analysis/brm/91a_core_regression.csv`).
2. **Gate tolerances in points.** Restated as 0.25 and 0.125 reference SD.
3. **Level-1 tau.** The point-ratio rule gave tau = 2.0 on four scales and 3.0 on Neuroticism, set by
   subgroups of 112 to 1,361 people (worst point departures 1.52 to 2.75). Read on their 90%
   intervals, no subgroup departs by more than 1.27, and the frozen rule (resolved departure, floor
   1.5) gives tau = 1.5 on every scale.
4. **R1 against the reference.** Twin-2K's Openness item 35 (reverse-keyed "likes routine work")
   loads .28, so the human reference failed the absolute R1. The frozen R1 lowers a threshold to the
   reference's own value where the reference is below it.
5. **R2.** Congruence passed every model with a general factor (lower limits .960 to .998) while
   their loadings were .07 to .33 RMS away from the humans'. The frozen R2 adds the RMSD
   condition and reads pass / unresolved / fail.

The gate's change to precision only and the level-2 stop on k > 1/4 came from the controls and the
power analysis, not from this run.

## Results on the frozen rules

| Rung | Result |
|---|---|
| Gate, k = 10 | 14 of 15 model-scale cells pass the minimum rule; GPT-4 Agreeableness fails it (SE 1.77 against 1.55). GPT-4 fails the recommended rule on every scale (k needed 16 to 53). Separation phi(10) is .91 to .997 everywhere |
| Level 1 (tau = 1.5) | 10 of 15 fail, 5 undetermined, none pass. Llama-2 fails on every scale (overfit deficit on four, excess misfit on three) |
| R1 | GPT-3.5 and Llama-2 have no general factor on Agreeableness and Conscientiousness, and Llama-2 none on Neuroticism and Openness; GPT-4 has one on every scale |
| R2 | 14 of 15 fail on loading size (RMSD .16 to .48); GPT-3.5 Neuroticism is unresolved (RMSD .071, upper limit .139) |
| R3 (diagnostic) | a within-persona factor only for GPT-4 on Agreeableness, Conscientiousness and Neuroticism |

The pattern on the BFI repeats the PHQ-8's: the models' items hang together more tightly than
people's (GPT-4's minimum loadings .68 to .98 against the humans' .28 to .71), and congruence alone
does not see it.
