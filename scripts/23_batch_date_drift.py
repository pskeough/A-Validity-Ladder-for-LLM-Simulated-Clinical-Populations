"""
Does the clinical run drift across its collection window?

The two conditions were not collected in disjoint windows. The narrative run occupies a single
17-hour window on 9-10 January that sits INSIDE the clinical run's 28 December to 11 January span,
which an earlier draft described as the two being collected eleven days apart. They overlap, and
that cuts two ways.

Against the design: a clinical condition spread over two weeks with recovery batches is not a
temporally coherent block. Its within-run flip probability could carry endpoint drift rather than
pure sampling stochasticity, and the estimand argument leans on both conditions being drawn under
the same decoding settings, which an unpinned endpoint over two weeks does not guarantee.

For the design: because the narrative window sits inside the clinical span, a near-contemporaneous
framing contrast can be built from the clinical rows generated in the same window and compared
against the full-span contrast. If the two agree, collection-window drift is not driving the
framing result.

Emits analysis/batch_date_drift.csv and analysis/condition_contrast_datematched.csv.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
m["ts"] = pd.to_datetime(m.timestamp)
m["day"] = m.ts.dt.floor("D")
cond = "condition" if "condition" in m.columns else "prompt_condition"

clin = m[m[cond].str.contains("clin", case=False, na=False)].copy()
narr = m[m[cond].str.contains("narr", case=False, na=False)].copy()
print(f"clinical  n={len(clin):,}  {clin.ts.min()} -> {clin.ts.max()}")
print(f"narrative n={len(narr):,}  {narr.ts.min()} -> {narr.ts.max()}")
overlap_lo, overlap_hi = narr.ts.min(), narr.ts.max()
print(f"narrative window sits inside the clinical span: "
      f"{bool(clin.ts.min() <= overlap_lo and overlap_hi <= clin.ts.max())}")

# ---- 1. does severity or dispersion move across the clinical window? ------------------------
BANDS = [0, 5, 10, 15, 20, 25]
band = lambda v: np.digitize(v, BANDS[1:-1])
rows = []
for d, g in clin.groupby("day"):
    cell = g.groupby(["model", "profile_id"]).phq8_total
    rows.append(dict(day=str(d.date()), n=len(g), n_cells=cell.ngroups,
                     mean=round(float(cell.mean().mean()), 4),
                     within_cell_sd=round(float(cell.std().mean()), 4)))
dd = pd.DataFrame(rows)
print("\nclinical run by collection day:")
print(dd.to_string(index=False))

# Spearman of cell mean on day index, at the cell level so the unit matches the rest of the paper.
cl = clin.groupby(["model", "profile_id", "day"], as_index=False).phq8_total.mean()
cl["dayidx"] = cl.day.rank(method="dense")
from scipy import stats
rho, pv = stats.spearmanr(cl.dayidx, cl.phq8_total)
print(f"\ncell mean vs collection day: Spearman rho = {rho:+.4f}, p = {pv:.4f}, n_cells = {len(cl)}")

# ---- 2. within-run flip probability by day ---------------------------------------------------
flip_rows = []
for d, g in clin.groupby("day"):
    fl = []
    for _, cellg in g.groupby(["model", "profile_id"]):
        b = band(cellg.phq8_total.to_numpy())
        if len(b) > 1:
            fl.append(float((b[:-1] != b[1:]).mean()))
    if fl:
        flip_rows.append(dict(day=str(d.date()), n_cells=len(fl), flip_pct=round(np.mean(fl) * 100, 2)))
fd = pd.DataFrame(flip_rows)
print("\nwithin-run flip probability by collection day:")
print(fd.to_string(index=False))
if len(fd) > 2:
    r2, p2 = stats.spearmanr(range(len(fd)), fd.flip_pct)
    print(f"  flip rate vs day: Spearman rho = {r2:+.4f}, p = {p2:.4f}")
dd.merge(fd, on="day", how="outer").to_csv(os.path.join(OUT, "batch_date_drift.csv"), index=False)

# ---- 3. the framing contrast, recomputed on date-matched clinical rows ------------------------
clin_m = clin[(clin.ts >= overlap_lo) & (clin.ts <= overlap_hi)]
print(f"\nclinical rows inside the narrative window: {len(clin_m):,}")
res = []
for label, cdf in [("full-span clinical", clin), ("date-matched clinical", clin_m)]:
    a = cdf.groupby(["model", "profile_id"]).phq8_total.mean()
    b = narr.groupby(["model", "profile_id"]).phq8_total.mean()
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).to_numpy()
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    res.append(dict(comparison=label, n_cells=len(common),
                    contrast=round(float(diff.mean()), 4), se=round(float(se), 4),
                    ci_lo=round(float(diff.mean() - 1.96 * se), 4),
                    ci_hi=round(float(diff.mean() + 1.96 * se), 4)))
cc = pd.DataFrame(res)
cc.to_csv(os.path.join(OUT, "condition_contrast_datematched.csv"), index=False)
print("\nframing contrast, clinical minus narrative:")
print(cc.to_string(index=False))
d0, d1 = cc.contrast.iloc[0], cc.contrast.iloc[1]
print(f"\n  full-span {d0:+.3f} vs date-matched {d1:+.3f}; difference {abs(d0 - d1):.3f} points")
print(f"  date-matched estimate inside the full-span interval: "
      f"{bool(cc.ci_lo.iloc[0] <= d1 <= cc.ci_hi.iloc[0])}")


# ---- the contrast with the recovery rows removed ---------------------------------------------
# Section 4.4 reports this figure to argue the 86 recovery rows do not carry the condition contrast
# in aggregate. It was previously computed inline and quoted in prose without a receipt, which is
# the one class of error the derivation gate cannot see: a number that never entered a CSV.
_key = ["model", "profile_id", "prompt_condition"]
_pair = lambda df: (df.groupby(_key, as_index=False).phq8_total.mean()
                    .pivot_table(index=_key[:2], columns="prompt_condition", values="phq8_total")
                    .dropna())
_p, _p2 = _pair(m), _pair(m[m.day.dt.strftime("%Y-%m-%d") != "2026-01-11"])
_full = float((_p.clinical - _p.narrative).mean())
_drop = float((_p2.clinical - _p2.narrative).mean())
_nrec = int((m.day.dt.strftime("%Y-%m-%d") == "2026-01-11").sum())
pd.DataFrame([dict(contrast_all_rows=round(_full, 4), contrast_recovery_dropped=round(_drop, 4),
                   n_recovery_rows=_nrec, shift=round(_drop - _full, 4))]
             ).to_csv(os.path.join(OUT, "recovery_row_sensitivity.csv"), index=False)
print(f"condition contrast: {_full:+.4f} all rows, "
      f"{_drop:+.4f} with the {_nrec} recovery rows dropped")
