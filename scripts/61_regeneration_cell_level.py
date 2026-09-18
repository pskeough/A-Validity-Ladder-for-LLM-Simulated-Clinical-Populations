"""Cell-level regeneration receipts for the ML4H submission.

Script 12 computes per-cell flip probabilities and then averages them away, so the manuscript's
headline instability figures (35.4% clinical, 32.16% narrative) reach the page with no interval.
The stated denominator in the base manuscript is the 417,600 ordered within-cell iteration pairs,
and Table 6's own caption concedes the 435 unordered pairs per cell are not independent. A
reviewer reading that finds a point estimate with no defensible standard error behind it.

The unit that IS independent is the design cell: 120 profiles x 4 models = 480 cells per
condition, each contributing one flip proportion computed over its own 30 iterations. This script
retains that level, so the headline can be reported as the mean of 480 cell proportions with a
between-cell interval -- the same cell-as-unit rule Section 3.4 already applies everywhere else.

Reuses 12_within_run_stability.py's cell_stats verbatim so the aggregate reproduces exactly.

Emits:
  analysis/regeneration_cell_level.csv     one row per (condition, model, profile_id)
  analysis/regeneration_summary.csv        pooled and per-model, with between-cell 95% CIs

Idempotent, no randomness.
"""
import os
from itertools import combinations  # noqa: F401  (parity with script 12's imports)

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.join(BASE, "..")
OUT = os.path.join(BASE, "analysis")

BANDS = [-1, 4, 9, 14, 19, 24]
THRESHOLD = 10  # PHQ-8 >= 10: the line separating watchful waiting from active treatment


def cell_stats(vals):
    """Identical to 12_within_run_stability.py. Do not diverge; the aggregate must reproduce."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    bands = np.digitize(vals, BANDS[1:-1], right=True)
    _, counts = np.unique(bands, return_counts=True)
    same = (counts * (counts - 1)).sum() / (n * (n - 1))
    diffs = np.abs(vals[:, None] - vals[None, :])
    mad = diffs[np.triu_indices(n, 1)].mean()
    return vals.std(ddof=1), 1 - same, mad


def threshold_cross_prob(vals):
    """P(two draws from this cell land on opposite sides of PHQ-8 = 10).

    Same pair-discordance form as the band flip, applied to the binary treatment indicator, so the
    two quantities are computed on one footing.
    """
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    above = (vals >= THRESHOLD).sum()
    below = n - above
    return (2.0 * above * below) / (n * (n - 1))


def mean_ci(x, alpha=0.05):
    """Mean with a normal-approximation interval on the between-cell SE.

    Cells are independent by construction -- separate profiles, separate API calls -- which is what
    licenses this interval where a pair-level one would not be licensed.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    z = 1.959963984540054
    return m, se, m - z * se, m + z * se


def main():
    cells = []
    for cname, path in (
        ("clinical", os.path.join(ROOT, "runs", "run1", "data", "audit_results.csv")),
        ("narrative", os.path.join(ROOT, "runs", "run2", "data", "audit_results.csv")),
    ):
        df = pd.read_csv(path)
        df["phq8_total"] = df["phq8_total"].clip(0, 24)
        for (mdl, pid), sub in df.groupby(["model", "profile_id"]):
            assert len(sub) == 30, f"{mdl}/{pid}: {len(sub)} iterations"
            vals = sub.phq8_total.values
            sd, pflip, mad = cell_stats(vals)
            cells.append(
                dict(
                    condition=cname,
                    model=mdl,
                    profile_id=pid,
                    n_iter=len(sub),
                    cell_mean=round(float(np.mean(vals)), 4),
                    within_sd=round(float(sd), 4),
                    p_band_flip=round(float(pflip), 6),
                    p_threshold_cross=round(float(threshold_cross_prob(vals)), 6),
                    mean_pairwise_abs_diff=round(float(mad), 4),
                )
            )

    cl = pd.DataFrame(cells)
    cl.to_csv(os.path.join(OUT, "regeneration_cell_level.csv"), index=False)

    rows = []
    for cname, sub_c in cl.groupby("condition"):
        for mdl, sub in list(sub_c.groupby("model")) + [("ALL", sub_c)]:
            m_f, se_f, lo_f, hi_f = mean_ci(sub.p_band_flip.values)
            m_t, se_t, lo_t, hi_t = mean_ci(sub.p_threshold_cross.values)
            rows.append(
                dict(
                    condition=cname,
                    model=mdl,
                    n_cells=len(sub),
                    flip_pct=round(m_f * 100, 2),
                    flip_se_pct=round(se_f * 100, 3),
                    flip_ci_lo_pct=round(lo_f * 100, 2),
                    flip_ci_hi_pct=round(hi_f * 100, 2),
                    cross_pct=round(m_t * 100, 2),
                    cross_se_pct=round(se_t * 100, 3),
                    cross_ci_lo_pct=round(lo_t * 100, 2),
                    cross_ci_hi_pct=round(hi_t * 100, 2),
                )
            )

    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "regeneration_summary.csv"), index=False)
    print(summ.to_string(index=False))

    # Reproduction gate against the receipt the manuscript already cites. If this fails, the two
    # scripts have diverged and the manuscript's existing number is no longer the one being framed.
    prior = pd.read_csv(os.path.join(OUT, "within_run_stability.csv"))
    print("\nreproduction check against within_run_stability.csv:")
    ok = True
    for _, p in prior.iterrows():
        q = summ[(summ.condition == p.condition) & (summ.model == p.model)]
        if q.empty:
            continue
        got, want = float(q.flip_pct.iloc[0]), float(p.flip_prob_pct)
        good = abs(got - want) < 0.02
        ok &= good
        print(f"  {'OK  ' if good else 'DIFF'} {p.condition:9s} {p.model:32s} "
              f"{got:6.2f} vs {want:6.2f}")
    print("\nGATE", "PASSED" if ok else "FAILED")

    a = summ[summ.model == "ALL"]
    print("\nheadline, cell as unit (n = 480 cells per condition):")
    for r in a.itertuples():
        print(f"  {r.condition:9s} band flip {r.flip_pct:.1f}% "
              f"[{r.flip_ci_lo_pct:.1f}, {r.flip_ci_hi_pct:.1f}]   "
              f"threshold cross {r.cross_pct:.1f}% "
              f"[{r.cross_ci_lo_pct:.1f}, {r.cross_ci_hi_pct:.1f}]")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
