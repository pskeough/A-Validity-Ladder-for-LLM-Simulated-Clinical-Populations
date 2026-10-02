"""Contrast-level level-2 readings for REAL (faithful) simulators in the panel power replicates,
frozen rule (78c.r3 `verdict`), by grid, reference precision and simulation SE variant. Shares of
replicate x model readings. Writes power_l2_contrast_readings.csv beside this file."""
import glob
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
d = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(os.path.join(D, "reps", "rep_*.csv")))],
              ignore_index=True)
c = d[(d.rung == "L2") & d.rule.str.startswith("contrast") & (d.type == "REAL")].copy()
c["variant"] = c.rule.str.extract(r"\[(.*)\]")[0]
GR = ["G48", "G144_age", "G144_edu", "G432"]
rows = []
for (unit, grid, var), g in c.groupby(["unit", "grid", "variant"]):
    v = g.verdict
    rows.append(dict(contrast=unit, grid=grid, variant=var, n=len(g), reps=g.rep.nunique(),
                     kept=np.mean(v == "kept"),
                     two_region_with_kept=np.mean([(" or " in x) and ("kept" in x) for x in v]),
                     excludes_kept=np.mean([("kept" not in x) and x not in ("undetermined", "no population gap",
                                                                             "reference too imprecise") for x in v]),
                     ref_too_imprecise=np.mean(v == "reference too imprecise"),
                     no_pop_gap=np.mean(v == "no population gap"),
                     undetermined=np.mean(v == "undetermined"),
                     k_rel_median=g.k_rel.median() if "k_rel" in g else np.nan))
o = pd.DataFrame(rows)
o["grid"] = pd.Categorical(o.grid, GR)
o = o.sort_values(["variant", "contrast", "grid"])
o.to_csv(os.path.join(D, "power_l2_contrast_readings.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
print(o[o.variant.isin(["cond_auditprec", "pairs_auditprec", "limit_auditprec"])].round(3).to_string(index=False))
