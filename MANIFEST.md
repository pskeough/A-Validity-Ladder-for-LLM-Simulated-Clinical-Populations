# MANIFEST

What is in this repository, what is not, and why.

The contents are decided by what the paper promises a reader rather than by what was on disk.
The export runs from an allowlist: a file ships only if it is named, and anything present and
unnamed is removed. Secrets scan: clean, and no credential is hardcoded anywhere in the tree.

## Included

| Path | What it is |
|---|---|
| `paper/` | the preprint: `main.tex` (body and appendices), `main.pdf`, `refs.bib`, `figures/` |
| `generation/` | the layer that produced the corpus: the calling script, both identity registries, the narrative conversion, the battery, and the recovery scripts |
| `data/model_outputs.csv` | the 28,800 generations, pooled |
| `data/model_outputs_v2.csv` | the same rows with clinical/narrative labels recovered from run provenance |
| `data/raw/` | the two per-run exports exactly as originally written |
| `groundtruth/` | the survey-weighted PHQ-8 anchors derived from NHANES microdata |
| `analysis/` | every receipt CSV the paper cites, including the ladder-level receipts (`score_scale_gate`, `level2_*`, `level3_equivalence`, `level4_*`, `prompt_control_*`) and the FDR ledger |
| `scripts/` | the pipeline 01-73 and the gates |
| `paper_library/` | citation-verification metadata: what was retrieved, what matched, what did not |
| `reports/` | V2_SPEC, MASTER_REPORT, VALIDATION_RECEIPTS, CITATION_VERIFICATION |

## The ladder's own scripts

Scripts 63 to 73 produce the paper's tables and figures:

```
63_score_scale_gate.py            the regeneration gate on the score scale
64_level2_sex_ses_contrasts.py    level 2, the anchored contrasts
65_level2_figure.py               level 2, the figure
66_level2_permodel_equivalence.py level 2, per model
67_level3_equivalence.py          level 3, population calibration
68_level4_population_reference.py level 4, structural fidelity
69_prompt_control_select.py       the prompt control: selection
70_prompt_control_run.py          the prompt control: run
71_prompt_control_analyze.py      the prompt control: analysis
72_level2_without_glm.py          level 2 with GLM-4.7 removed
73_level4_floor_sensitivity.py    level 4, floor sensitivity (draw-level vs persona-level)
```

Scripts 01 to 62 build the corpus, derive the anchors and produce the receipts the ladder
reads. They include the verification gates belonging to the companion epidemiological audit
(*Plausible Patients, Impossible Populations*, arXiv:2604.17359). Those gates are retained
because the ladder's numbers descend from the same corpus and the same anchor derivation, so
the provenance chain is only complete with them present. Running them requires that paper's
own manuscript sources, which are released in its repository rather than duplicated here.

## Deliberately excluded

- **`data/nhanes_raw/`** — public CDC microdata, roughly 28 MB, fetched by
  `scripts/01_download_nhanes.sh`. The derived anchors are already in `groundtruth/`, and the
  gates run without it.
- **`paper_library/pdf/` and `txt/`** — the retrieved source papers themselves. They are
  copyrighted and not ours to redistribute. The metadata that records which citation resolved
  to which paper, and which title matches failed, is included; the papers are not.
- **The companion paper's manuscript sources** — `paper_short/` and `paper_v2/` live in the
  [plausible-patients](https://github.com/pskeough/plausible-patients) repository. Gates 15,
  19 and 42 assert against them and will skip here.
- **The multi-agent review artifacts** — candidate findings and triage from a fan-out review
  of the manuscript. Process, not receipts, and unreferenced by the paper. Publishing raw
  agent output most of which was rejected would misrepresent what was actually concluded.
- **Figure PNGs** — preview renders emitted alongside the PDFs by
  `scripts/10_make_figures.py`. The PDFs are what the manuscript compiles against.

## Provenance note

The generation layer was rebuilt from the prompts and registries of the paper's appendix
after the original run's code was not retained. The pre-registration forbids reporting the
control that ran on it as a reproduction of the corpus, and the paper does not report it as
one.
