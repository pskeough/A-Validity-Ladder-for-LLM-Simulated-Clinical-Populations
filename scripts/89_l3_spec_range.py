"""Level 3 as a range over the declared reference specifications (review finding A-M4).

Reads analysis/brm/80c_l3_results.csv (script 80c) and reports, per model and framing, the overall
post-stratified residual (PS estimand, mean PHQ-8, full corpus) under each of the eight declared
specifications: S1 primary, the two marital crosswalks, the 2007-2018 bridge, the two persona-string
income crosswalks, and the 2017-2020 and 2021-2023 periods. No new estimation; a tabulation of 80c.

Writes analysis/brm/89_l3_spec_range.csv (one row per model x framing x spec) and
analysis/brm/89_l3_spec_range_summary.csv (one row per model x framing: residual range, lowest
CI bound, specs passing at 2 points).
"""
import os

import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
IN = os.path.join(BASE, "analysis", "brm", "80c_l3_results.csv")
OUT = os.path.join(BASE, "analysis", "brm", "89_l3_spec_range.csv")
OUT_SUM = os.path.join(BASE, "analysis", "brm", "89_l3_spec_range_summary.csv")

d = pd.read_csv(IN)
sel = d[(d.group == "Overall") & (d.estimand == "PS") & (d.outcome == "mean") & (d.corpus == "all")]
cols = ["model", "framing", "spec", "window", "income_def", "marital_def", "sim", "ref", "resid",
        "ci90_lo", "ci90_hi", "pass_1pt", "pass_2pt"]
rows = sel[cols].sort_values(["model", "framing", "spec"]).reset_index(drop=True)
assert rows.groupby(["model", "framing"]).size().eq(8).all(), "expected eight specs per model x framing"
rows.to_csv(OUT, index=False)

summ = rows.groupby(["model", "framing"]).agg(
    resid_min=("resid", "min"), resid_max=("resid", "max"),
    ci90_lo_min=("ci90_lo", "min"), ci90_hi_max=("ci90_hi", "max"),
    specs_pass_2pt=("pass_2pt", "sum"), specs_pass_1pt=("pass_1pt", "sum"),
).reset_index()
summ["specs_passing_2pt"] = (
    rows[rows.pass_2pt == 1].groupby(["model", "framing"])["spec"].apply(lambda s: "; ".join(s))
    .reindex(pd.MultiIndex.from_frame(summ[["model", "framing"]])).fillna("").values
)
summ.to_csv(OUT_SUM, index=False)
print(summ.to_string(index=False))
