"""
Decoding control: analysis.

Question: the manuscript's regeneration-instability result was measured at each provider's default
decoding, because no sampling parameter was ever set. Does setting temperature to 0 remove it?

Estimators match the main analysis:
  - severity category from the PHQ-8 total on the standard bands
        None 0-4, Mild 5-9, Moderate 10-14, Mod-severe 15-19, Severe 20-24
  - flip probability = share of ordered within-cell draw pairs landing in different categories
  - threshold crossing = share of DISCORDANT pairs that cross PHQ-8 = 10, the line separating
    watchful waiting from active treatment

Writes analysis/decoding_control_results.csv.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd

from decoding_control_io import OUT, both_arms_only, balanced_cells, load

# Loading, deduplication and the both-arms/paired-cohort restrictions live in decoding_control_io so
# that this script and 57 cannot drift apart on them. A model contributes only cohorts it measured
# under BOTH decoding arms, which keeps every contrast reported here within-cell.
df, _report = load()
df = both_arms_only(df)
df = balanced_cells(df)
print("analysing %d generations over %d models and %d cohorts"
      % (len(df), df.model.nunique(), df.profile_id.nunique()))

out = []
for (model, arm), g in df.groupby(["model", "arm"]):
    flips = same = cross = disc = 0
    cells = 0
    for _, cell in g.groupby("profile_id"):
        t = cell.total.tolist()
        c = cell.cat.tolist()
        if len(t) < 2:
            continue
        cells += 1
        for i, j in combinations(range(len(t)), 2):
            if c[i] != c[j]:
                flips += 1
                # does the pair straddle the treatment threshold at 10?
                if (t[i] < 10) != (t[j] < 10):
                    cross += 1
                disc += 1
            else:
                same += 1
    pairs = flips + same
    out.append(dict(model=model, arm=arm, cells=cells, n=len(g), pairs=pairs,
                    flip_pct=round(100 * flips / pairs, 2) if pairs else np.nan,
                    cross_of_discordant_pct=round(100 * cross / disc, 2) if disc else np.nan,
                    cross_of_all_pairs_pct=round(100 * cross / pairs, 2) if pairs else np.nan,
                    mean_total=round(g.total.mean(), 3), sd_total=round(g.total.std(ddof=0), 3),
                    distinct_totals=g.total.nunique()))

res = pd.DataFrame(out).sort_values(["model", "arm"])
pd.set_option("display.width", 160)
print()
print(res.to_string(index=False))

print()
print("%-24s %10s %10s %8s" % ("MODEL", "default", "temp0", "change"))
print("-" * 56)
for model in sorted(res.model.unique()):
    d = res[(res.model == model) & (res.arm == "default")].flip_pct
    t = res[(res.model == model) & (res.arm == "temp0")].flip_pct
    if len(d) and len(t):
        print("%-24s %9.2f%% %9.2f%% %+7.2f" % (model, d.iat[0], t.iat[0], t.iat[0] - d.iat[0]))

pooled = []
for arm in ("default", "temp0"):
    s = res[res.arm == arm]
    w = s.pairs.sum()
    pooled.append((arm, float((s.flip_pct * s.pairs).sum() / w), float((s.cross_of_discordant_pct * s.pairs).sum() / w)))
print()
for arm, f, c in pooled:
    print("pooled %-8s flip %.2f%%   of discordant pairs crossing PHQ-8=10: %.2f%%" % (arm, f, c))

d0 = dict((a, f) for a, f, _ in pooled)
print()
print("VERDICT: temperature 0 changes the pooled flip rate from %.2f%% to %.2f%% (%+.2f points)."
      % (d0["default"], d0["temp0"], d0["temp0"] - d0["default"]))
if d0["temp0"] > 5:
    print("         Instability persists at temperature 0. It is not a sampling-parameter artefact.")

res.to_csv(os.path.join(OUT, "decoding_control_results.csv"), index=False)
print("\nwritten -> %s" % os.path.join(OUT, "decoding_control_results.csv"))
