# Panel power at larger persona grids

Run 2026-10-02, frozen rules (`paper_brm/LADDER_SPEC.md` v1.0). 60 replicates of the intermediate
panel (`scripts/90_panel_power.py`, seeds `[20261090, r]`), four pseudo-models per replicate, at
four persona grids. R4 on replicates 0-24; the non-invariant plant on replicates 0-9.
Summaries: `90b_summary.csv`, `90b_l2_contrasts.csv` (`scripts/90b_panel_power_summary.py`) and
`power_l2_contrast_readings.csv` (`power_l2_contrast_readings.py` in this folder).

| Grid | Cells | Attributes |
|---|---|---|
| G48 | 48 | race x sex x income (the corpus design) |
| G144_age | 144 | + age band |
| G144_edu | 144 | + education band |
| G432 | 432 (415 usable) | + age and education |

## Results

Gate. Real respondents pass the minimum rule (SE(30) <= 0.98 points) in every reading on every grid
(SE(30) 0.60 to 0.77) and fail the recommended rule (0.49 points) in every reading. The verdicts are
read from the stored SE(30): runners started before `88_intermediate_panel.py` took the frozen gate
(13:41) stored the pre-freeze verdict (phi(30) >= .80 and SE), which `90b` also prints for the record.

Level 2, model verdict, audit-precision reference.

| Grid | Faithful fails, pairs SE | Faithful fails, draw-only SE | Doubled gap fails, pairs SE | Faithful passes |
|---|---|---|---|---|
| G48 | 0.0% | 3.8% | 97.9% | 0% |
| G144_age | 0.4% | 4.2% | 100% | 0% |
| G144_edu | 1.7% | 8.8% | 97.5% | 0% |
| G432 | 3.3% | 15.4% | 100% | 0% |

The draw-only SE leaves out between-person variation within a cell, so its false-fail rate grows
with grid fineness; the pairs SE is the primary SE.

Level 2, contrast readings of a faithful simulator (share of readings read `kept` alone).

| Contrast | 30 draws, pairs SE | 30 draws, draw-only SE | No simulation error | Median k (reference) |
|---|---|---|---|---|
| Low - High income | 0% | up to 11.7% | 95.0% (G48), 98.3% (G144_age), 11.7% (G144_edu), 36.7% (G432) | .16 to .24 |
| Women - Men | 0% | up to 2.1% | 51.7% (G48, G144_age), 13.3% (G144_edu, G432) | .20 to .23 |
| Black - White | 0% | 0% | 0% | .57 to .74 |
| Hispanic - White | 0% | 0% | 0% | 1.03 to 2.24 |
| Asian - White, Low - Middle, Middle - High | 0% | 0% | 0% to 8.3% | .24 to .52 |

Level 2 cannot pass a faithful simulator at 30 draws on any grid. Without simulation error (the limit
of adding draws) the low-to-high income gap is certified on the 48- and 144-cell age grids; the race
contrasts are not certifiable at any grid because the NHANES gaps are imprecise (k above the 1/4
limit). Adding education raises the income contrast's k from .16 to .24, so a finer grid does not
make the reference more precise.

Level 4. R1 and R2 (frozen, shape and size) pass real respondents in every replicate on every grid.
R4 at margin .08 leaves real respondents unresolved in 92% to 100% of replicates and fails the
non-invariant plant in every replicate on every grid.
