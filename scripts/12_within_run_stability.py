"""
Within-run stability receipts for v2 Results III.

True stochastic stability: same prompt condition, same cohort, 30 independent iterations.
For every (model, profile_id) cell within each run we compute across its 30 iterations:
  - within-cell SD of PHQ-8 total
  - pairwise band-disagreement probability over all C(30,2)=435 iteration pairs
    (5-band PHQ-8, same cut as 07_fdr_ledger.py)
  - mean absolute pairwise difference
This is the quantity v1 never computed: its "stability" family (36.66% flips, MAD 1.89,
drift +0.324) was computed on CROSS-RUN pairs and therefore mixes prompt-condition,
scoring-path, and collection-window differences with stochastic variation.

Emits analysis/within_run_stability.csv (per model x condition + ALL rows).
Idempotent, no randomness.
"""
import pandas as pd, numpy as np, os
from itertools import combinations

BASE = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.join(BASE, "..")
OUT = os.path.join(BASE, "analysis")

BANDS = [-1, 4, 9, 14, 19, 24]

def cell_stats(vals):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    bands = np.digitize(vals, BANDS[1:-1], right=True)
    _, counts = np.unique(bands, return_counts=True)
    same = (counts * (counts - 1)).sum() / (n * (n - 1))
    diffs = np.abs(vals[:, None] - vals[None, :])
    mad = diffs[np.triu_indices(n, 1)].mean()
    return vals.std(ddof=1), 1 - same, mad

rows = []
for cname, path in (("clinical", os.path.join(ROOT, "runs", "run1", "data", "audit_results.csv")),
                    ("narrative", os.path.join(ROOT, "runs", "run2", "data", "audit_results.csv"))):
    df = pd.read_csv(path)
    df["phq8_total"] = df["phq8_total"].clip(0, 24)
    per_cell = []
    for (mdl, pid), sub in df.groupby(["model", "profile_id"]):
        assert len(sub) == 30, f"{mdl}/{pid}: {len(sub)} iterations"
        sd, pflip, mad = cell_stats(sub.phq8_total.values)
        per_cell.append(dict(model=mdl, profile_id=pid, sd=sd, pflip=pflip, mad=mad))
    pc = pd.DataFrame(per_cell)
    for mdl, sub in pc.groupby("model"):
        rows.append(dict(condition=cname, model=mdl, n_cells=len(sub),
                         mean_within_sd=round(sub.sd.mean(), 4),
                         flip_prob_pct=round(sub.pflip.mean() * 100, 2),
                         mean_pairwise_abs_diff=round(sub.mad.mean(), 4)))
    rows.append(dict(condition=cname, model="ALL", n_cells=len(pc),
                     mean_within_sd=round(pc.sd.mean(), 4),
                     flip_prob_pct=round(pc.pflip.mean() * 100, 2),
                     mean_pairwise_abs_diff=round(pc.mad.mean(), 4)))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "within_run_stability.csv"), index=False)
print(out.to_string(index=False))
a = out[(out.model == "ALL")]
print("\nheadline: within-run flip probability "
      + ", ".join(f"{r.condition} {r.flip_prob_pct:.1f}%" for r in a.itertuples())
      + "  (cross-condition pair flip rate for comparison: 36.66%)")
