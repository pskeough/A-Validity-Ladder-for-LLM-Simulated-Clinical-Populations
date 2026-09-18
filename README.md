# A Validity Ladder for LLM-Simulated Clinical Populations

Code, data and receipts for the preprint. Patrick S. Keough, 2026.

Large language models now generate patient populations for clinician training, screening
studies and clinical audits, and validation of those populations often stops at case review.
A case can pass every reviewer who reads it while the cohort it belongs to sits a full
severity category above the population it was written to represent.

The validity ladder is a protocol for validating a generated population before use. A gate
for regeneration stability sits beneath four levels: individual coherence, subgroup fidelity,
population calibration and structural fidelity. Each level targets a failure the levels below
it cannot detect, and passing a level supports a stated use of the population rather than
general fitness. The ladder is read per model and records a stop wherever no reference exists.

The demonstration runs on 28,800 PHQ-8 assessments from four models (GPT-4o-mini,
Gemini-3-Flash, DeepSeek-V3, GLM-4.7) against survey-weighted NHANES anchors derived here
from primary microdata.

## What the demonstration finds

- **The gate withdraws the single-draw label.** Two draws of one prompt land in different
  PHQ-8 severity categories 35.4% of the time and a single draw reproduces its cell's modal
  category 74.9% of the time, so the gate fails the label rule. It passes the score rule
  pooled, 92.5% of draw pairs landing within five points, and sets the unit for levels 2 and
  3 to the cell mean.
- **Individual coherence passes in three of four models**, which is what makes the rest of
  the ladder necessary: case review alone would clear this corpus.
- **Population calibration fails.** Simulated severity runs above the population anchors in
  every benchmarkable group.
- **Subgroup gaps are preserved by at most one model**, and the level-4 income result holds
  on the draw-level floor but not on the persona-level one, which is reported rather than
  resolved.

A model conditioned on a description returns its expected case for that demographic and
nothing about how people in that demographic vary, so every comparison in the ladder runs on
cell means.

## Layout

```
paper/        the preprint: main.tex (body and appendices), refs.bib, figures/, main.pdf
generation/   the layer that produced the corpus: calling script, both identity registries,
              the narrative conversion, the battery, and the recovery scripts
data/         the 28,800 generations, pooled and with run-provenance labels, plus the two
              per-run exports as originally written
groundtruth/  the survey-weighted PHQ-8 anchors derived from NHANES public microdata
analysis/     every receipt CSV the paper cites, including the ladder-level receipts and the
              FDR ledger
scripts/      the pipeline (01-73) and the gates
paper_library/ citation-verification metadata: what was retrieved, what matched, what did not
reports/      V2_SPEC, MASTER_REPORT, VALIDATION_RECEIPTS, CITATION_VERIFICATION
```

The ladder's own analyses are scripts 63 to 73: the score-scale gate (63), the level-2
contrasts, figure and per-model equivalence (64 to 66), level-3 equivalence (67), the
level-4 population reference (68), the prompt control (69 to 71), level 2 without GLM (72)
and the level-4 floor sensitivity (73). Scripts 01 to 62 build the corpus, the anchors and
the receipts that the ladder reads, and include the gates belonging to the companion
epidemiological audit; they ship here because the ladder's numbers descend from them.

## Provenance and limits worth stating

The generation layer was **rebuilt** from the prompts and registries in the paper's appendix
after the original run's code was not retained. The pre-registration forbids reporting the
control that ran on it as a reproduction of the corpus, and the paper does not.

The anchor derivation is gated on reproducing two published tables (Brody 2018, Patel 2019)
before it is trusted for new numbers; `analysis/validation_patel.csv` is that gate's output.

The analysis scripts were implemented to the author's specification with AI coding
assistance, and every table in the paper regenerates from the released receipts.

`data/nhanes_raw/` is deliberately excluded. It is public CDC microdata, roughly 28 MB,
fetched by `scripts/01_download_nhanes.sh`. The derived anchors are already in
`groundtruth/` and the gates run without it.

## Companion work

The epidemiological audit this corpus was first built for is *Plausible Patients, Impossible
Populations* (arXiv:2604.17359), at
[github.com/pskeough/plausible-patients](https://github.com/pskeough/plausible-patients).
That paper asks what the simulation gets wrong; this one asks what you would have to check
to find out.

## Licence

Code under MIT (`LICENSE`). Paper, data and derived corpus under CC BY-NC-ND 4.0
(`LICENSE-DATA`).
