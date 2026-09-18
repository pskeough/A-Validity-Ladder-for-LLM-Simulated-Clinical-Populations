"""Receipts for the score-scale arm of the regeneration gate and for the pooled level-1 composition.

The gate's second rule reads the score rather than the label: two draws of one cell are compared on
the PHQ-8 total, and a pair that differs by five points or more, the instrument's minimal clinically
important change borrowed as a between-draw tolerance, counts as a change. Over the 435 unordered
draw pairs of each cell this gives the share of pairs that differ by five or more, pooled by pair
count and per model, and the share of cells in which at least 90% of pairs stay within five points.

The level-1 composition receipt counts the elevated cases (PHQ-8 total of 10 or more) the gateway
rule is scored on and the share of them that are transgender personas.

Inputs:  data/model_outputs_v2.csv
Outputs: analysis/score_scale_gate.csv, analysis/elevated_composition.csv
"""
import itertools
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
MCID = 5
CELL = ["model", "race", "gender", "ses_normalized", "relationship"]

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"), low_memory=False)
m["phq8_total"] = m["phq8_total"].clip(0, 24)

rows = []
for cond, dc in m.groupby("prompt_condition"):
    cell_rows = []
    for key, sub in dc.groupby(CELL):
        v = sub.phq8_total.to_numpy(float)
        d = np.abs(np.array([a - b for a, b in itertools.combinations(v, 2)]))
        cell_rows.append(dict(zip(CELL, key)) | dict(
            condition=cond, n_pairs=len(d), n_ge_mcid=int((d >= MCID).sum()),
            share_within=float((d < MCID).mean())))
    cl = pd.DataFrame(cell_rows)
    for mdl, sub in list(cl.groupby("model")) + [("POOLED", cl)]:
        rows.append(dict(
            condition=cond, model=mdl, n_cells=len(sub),
            pct_pairs_ge_mcid=round(100 * sub.n_ge_mcid.sum() / sub.n_pairs.sum(), 2),
            pct_cells_pass_90=round(100 * float((sub.share_within >= 0.90).mean()), 1)))
gate = pd.DataFrame(rows)
gate.to_csv(os.path.join(OUT, "score_scale_gate.csv"), index=False)
print(gate.to_string(index=False))

elev = m[m.phq8_total >= 10]
comp = pd.DataFrame([dict(
    n_generations=len(m), n_elevated=len(elev),
    pct_elevated_transgender=round(100 * float(elev.gender.str.contains("Trans", case=False).mean()), 1),
    pct_corpus_transgender=round(100 * float(m.gender.str.contains("Trans", case=False).mean()), 1))])
comp.to_csv(os.path.join(OUT, "elevated_composition.csv"), index=False)
print(comp.to_string(index=False))
