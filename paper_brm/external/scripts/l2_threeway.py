"""Three-way level-2 verdicts (kept / not kept / unresolved / not read) for the external datasets.

The article (2 Oct 2026 rewrite) reports each contrast as kept, not kept or unresolved, read from
the Fieller interval against the kept region [0.75, 1.25]; a stopped contrast is 'not read'.
This is a relabelling of the frozen rule (LADDER_SPEC.md): only the kept region decides a model's
pass, and a stop that the interval already overrides is printed by the scorers as its region
verdict, so it is counted here from its interval.

Inputs (no recomputation of any interval):
  analysis/brm/l2_external_r3.csv                       config == headline (OpinionQA, Argyle)
  paper_brm/external/results/bisbee/level2_contrasts_rr1.csv   (Bisbee, primary run)
Output:
  paper_brm/external/results/l2_threeway_summary.csv    counts per dataset, run and framing
  paper_brm/external/results/l2_threeway_rows.csv       one row per contrast with its label
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXT = os.path.join(BASE, "analysis", "brm", "l2_external_r3.csv")
BIS = os.path.join(BASE, "paper_brm", "external", "results", "bisbee", "level2_contrasts_rr1.csv")
OUT = os.path.join(BASE, "paper_brm", "external", "results")
STOPS = {"reference too imprecise", "no population gap"}
LO, HI = 0.75, 1.25


def three_way(verdict, lo, hi):
    if verdict in STOPS:
        return "not read"
    if np.isnan(lo) or np.isnan(hi):
        return "unresolved"
    if lo >= LO and hi <= HI:
        out = "kept"
    elif hi < LO or lo > HI:
        out = "not kept"
    else:
        out = "unresolved"
    # the region verdict printed by the scorer must agree with the interval reading
    if out == "kept":
        assert verdict == "kept", verdict
    elif out == "not kept":
        assert "kept" not in verdict.replace("not kept", ""), verdict
    else:
        assert verdict == "undetermined" or "kept" in verdict, (verdict, lo, hi)
    return out


def main():
    e = pd.read_csv(EXT)
    e = e[e.config == "headline"].copy()
    e["dataset"] = e.source
    e["run"] = e.family.where(e.source != "OpinionQA", "all")
    e["framing"] = ""
    b = pd.read_csv(BIS)
    b["dataset"] = "Bisbee"
    b["run"] = b.model
    rows = pd.concat([
        e[["dataset", "run", "framing", "family", "contrast", "ratio", "ci_lo", "ci_hi", "verdict"]],
        b.assign(family=b.framing)[["dataset", "run", "framing", "family", "contrast", "ratio",
                                    "ci_lo", "ci_hi", "verdict"]].assign(outcome=b.outcome),
    ], ignore_index=True)
    rows["label"] = [three_way(v, lo, hi) for v, lo, hi in
                     zip(rows.verdict, rows.ci_lo.astype(float), rows.ci_hi.astype(float))]
    rows.to_csv(os.path.join(OUT, "l2_threeway_rows.csv"), index=False)
    key = ["dataset", "run", "framing"]
    s = rows.groupby(key + ["label"]).size().unstack(fill_value=0).reset_index()
    for c in ("kept", "not kept", "unresolved", "not read"):
        if c not in s:
            s[c] = 0
    s["total"] = s[["kept", "not kept", "unresolved", "not read"]].sum(axis=1)
    # a reading of the point ratio alone, ignoring the interval: how many sit in the kept region
    rows["point_in_kept"] = rows.ratio.between(LO, HI) & ~rows.verdict.isin(STOPS)
    pk = rows.groupby(key).point_in_kept.sum().rename("point_ratio_in_kept").reset_index()
    s = s.merge(pk, on=key)
    s.to_csv(os.path.join(OUT, "l2_threeway_summary.csv"), index=False)
    print(s.to_string(index=False))
    print("\nArgyle families:", sorted(e[e.source != "OpinionQA"].family.unique())[:10])


if __name__ == "__main__":
    main()
