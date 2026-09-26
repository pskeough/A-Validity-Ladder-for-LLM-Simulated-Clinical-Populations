"""Level-3 rebuild, step b: the simulated side, one row per model x persona x framing.

Input data/model_outputs_v3.csv (script 76). The one row with phq8_valid == False is dropped.
Only cisgender personas enter level 3 (NHANES records sex, not gender identity), and multiracial
personas are kept here only so that the published marginal comparison can be reproduced; the
post-stratified estimand uses the 48 personas of four races x two sexes x three income strings x
two relationship statuses.

Per cell and framing: n draws, mean PHQ-8, within-cell variance (ddof 1), the share of draws at or
above 10, and the variance of that share with the plus-four adjustment (p~ = (x + 2) / (n + 4),
var = p~ (1 - p~) / (n + 4)) so that a cell with no draws at 10 or more does not claim zero
sampling error. Two corpora:
  all        every valid row
  dec_only   clinical rows with row_source in {dec28_main, dec28_restored}; all narrative rows

Emits analysis/brm/80b_sim_cells.csv. No randomness.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(BASE, "analysis", "brm")

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v3.csv"), low_memory=False)
assert len(m) == 28800
bad = int((~m.phq8_valid.astype(bool)).sum())
assert bad == 1
m = m[m.phq8_valid.astype(bool)].copy()
m = m[m.gender.isin(["Cisgender Man", "Cisgender Woman"])].copy()
m["sex"] = m.gender.map({"Cisgender Man": "Men", "Cisgender Woman": "Women"})
m["inc"] = m.ses_normalized
m["mar"] = m.relationship
m["y"] = m.phq8_total.astype(float)
assert m.y.between(0, 24).all()
m["dep10"] = (m.y >= 10).astype(float)
m["model"] = m.model.str.split("/").str[1]

dec = m.prompt_condition.eq("narrative") | m.row_source.isin(["dec28_main", "dec28_restored"])
rows = []
for corpus, sub in [("all", m), ("dec_only", m[dec])]:
    g = sub.groupby(["model", "profile_id", "race", "sex", "inc", "mar", "prompt_condition"])
    c = g.agg(n=("y", "size"), mean=("y", "mean"), var=("y", lambda s: s.var(ddof=1)),
              x10=("dep10", "sum")).reset_index()
    c["prev10"] = c.x10 / c.n
    pt = (c.x10 + 2) / (c.n + 4)
    c["prev10_var"] = pt * (1 - pt) / (c.n + 4)
    c["corpus"] = corpus
    rows.append(c)
cells = pd.concat(rows, ignore_index=True)
cells = cells.rename(columns={"prompt_condition": "framing"})
cells.to_csv(os.path.join(OUTD, "80b_sim_cells.csv"), index=False)

full = cells.groupby(["corpus", "framing"]).agg(cells=("n", "size"), draws=("n", "sum"), min_n=("n", "min"),
                                                 zero_var_cells=("var", lambda s: int((s == 0).sum())))
print(full.to_string())
# every model x persona x framing present in both corpora?
for corpus in ["all", "dec_only"]:
    k = cells[cells.corpus == corpus]
    print(corpus, "cells:", len(k), "expected", 4 * 60 * 2)
print("\ndec_only clinical cells with fewer than 20 draws:")
k = cells[(cells.corpus == "dec_only") & (cells.framing == "clinical")]
print(k[k.n < 20][["model", "profile_id", "n"]].to_string(index=False))
print("\ndec_only clinical draws by model:", k.groupby("model").n.sum().to_dict())
