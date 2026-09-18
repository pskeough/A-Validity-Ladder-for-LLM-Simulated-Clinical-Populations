"""
Cluster-bootstrap intervals for the decoding control.

Table 7 reported every figure as a point estimate. A reviewer noted that the cohort is the natural
cluster, that there are only twelve of them, and that without an interval the per-model ordering
carries no stated uncertainty. That matters most for GLM-4.7, whose change of -3.66 points is the
row the "does not port across endpoints" claim rests on.

Resampling is over COHORTS with replacement, holding the design otherwise fixed, because draws
within a cohort share a vignette and are not independent. Both arms of a resampled cohort travel
together, which preserves the paired structure the contrast depends on.

Writes analysis/decoding_control_intervals.csv.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd

from decoding_control_io import OUT, balanced_cells, both_arms_only, load

B = 2000
SEED = 20260804


def flip_pct(cell_totals, cell_cats):
    flips = pairs = 0
    for t, c in zip(cell_totals, cell_cats):
        for i, j in combinations(range(len(t)), 2):
            pairs += 1
            if c[i] != c[j]:
                flips += 1
    return 100.0 * flips / pairs if pairs else np.nan


df, _ = load(verbose=False)
df = balanced_cells(both_arms_only(df, verbose=False), verbose=False)

# Pre-group so each bootstrap replicate is a lookup rather than a regroup.
cells = {}
for (m, a, p), g in df.groupby(["model", "arm", "profile_id"]):
    cells[(m, a, p)] = (g.total.tolist(), g.cat.tolist())

rng = np.random.default_rng(SEED)
rows = []
for m in sorted(df.model.unique()):
    coh = sorted(df[df.model == m].profile_id.unique())
    obs = {}
    for a in ("default", "temp0"):
        t = [cells[(m, a, p)][0] for p in coh]
        c = [cells[(m, a, p)][1] for p in coh]
        obs[a] = flip_pct(t, c)
    obs_change = obs["temp0"] - obs["default"]

    boot = []
    for _ in range(B):
        pick = rng.choice(coh, size=len(coh), replace=True)
        d_ = flip_pct([cells[(m, "default", p)][0] for p in pick],
                      [cells[(m, "default", p)][1] for p in pick])
        t_ = flip_pct([cells[(m, "temp0", p)][0] for p in pick],
                      [cells[(m, "temp0", p)][1] for p in pick])
        boot.append(t_ - d_)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append(dict(model=m, n_cohorts=len(coh),
                     default_pct=round(obs["default"], 2), temp0_pct=round(obs["temp0"], 2),
                     change=round(obs_change, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                     excludes_zero=bool(lo < 0 and hi < 0) or bool(lo > 0 and hi > 0)))

res = pd.DataFrame(rows).sort_values("change")
pd.set_option("display.width", 140)
print("Cluster bootstrap over %d cohorts, B = %d, seed %d\n" % (res.n_cohorts.iat[0], B, SEED))
print(res.to_string(index=False))

print("\nDo the per-model changes separate?")
for a, b in combinations(res.model.tolist(), 2):
    ra = res[res.model == a].iloc[0]
    rb = res[res.model == b].iloc[0]
    overlap = not (ra.ci_hi < rb.ci_lo or rb.ci_hi < ra.ci_lo)
    print("  %-24s vs %-24s %s" % (a, b, "intervals overlap" if overlap else "separated"))

res.to_csv(os.path.join(OUT, "decoding_control_intervals.csv"), index=False)
print("\nwritten -> analysis/decoding_control_intervals.csv")
