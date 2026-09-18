"""
Cell-level inference for every comparison whose unit of analysis was previously a generation row.

The design draws 30 iterations per (model, cohort, condition) cell. Those iterations are repeated
draws from one conditional response distribution, not distinguishable individuals, so treating
28,800 rows as exchangeable understates every standard error. Pairing iteration i of one condition
with iteration i of the other is likewise arbitrary: the correlation that licenses pairing lives at
the cell level, not the row level. The effective sample is 480 cells (120 cohorts x 4 models) per
condition, not 14,400 rows.

This script recomputes, at the cell level:
  - severity residuals and CIs against the gold anchors, cisgender frame (Table 2)
  - the condition contrast, pooled and per model (Sections 5, ledger T10/T25-T30)
  - the transgender-cisgender elevation contrasts (ledger T13-T14)
and reports the row-level values alongside, so the inflation is visible.

Emits analysis/{residuals_cell_level,condition_contrast_cell_level,trans_elevation_cell_level}.csv.
Idempotent, no randomness.
"""
import pandas as pd, numpy as np, os
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.join(BASE, "..")
OUT = os.path.join(BASE, "analysis")
CELL = ["model", "profile_id"]
CIS = ["Cisgender Man", "Cisgender Woman"]

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
m["ses_clean"] = m.ses.str.replace('"', '', regex=False).str.split(" ").str[0]
gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
G = {r.group: r for _, r in gt.iterrows()}

# ---- 1. residuals with cell-clustered standard errors --------------------------------------
GROUPS = [("race", "White", "White"), ("race", "Black", "Black"), ("race", "Asian", "Asian"),
          ("race", "Hispanic", "Hispanic (pooled)"),
          ("gender", "Cisgender Man", "Men"), ("gender", "Cisgender Woman", "Women"),
          ("ses_clean", "Low", "Low"), ("ses_clean", "Middle", "Middle"), ("ses_clean", "High", "High")]
rows = []
for col, grp, gtk in GROUPS:
    sel = (m[col] == grp) if col == "gender" else ((m[col] == grp) & m.gender.isin(CIS))
    sub = m[sel]
    g = G[gtk]
    cm = sub.groupby(CELL).phq8_total.mean()          # one value per cell
    mean_cell = cm.mean()
    se_cell = cm.std(ddof=1) / np.sqrt(len(cm))
    se_row = sub.phq8_total.std(ddof=1) / np.sqrt(len(sub))
    res = mean_cell - g.w_mean
    t_cell = res / se_cell
    rows.append(dict(dimension=col.replace("_clean", ""), group=grp,
                     n_cells=len(cm), n_rows=len(sub),
                     model_mean=round(mean_cell, 4),
                     ci_lo=round(mean_cell - 1.96 * se_cell, 4), ci_hi=round(mean_cell + 1.96 * se_cell, 4),
                     gt_mean=round(g.w_mean, 4), gt_sd=round(g.w_sd, 4),
                     residual=round(res, 4),
                     resid_ci_lo=round(res - 1.96 * se_cell, 4), resid_ci_hi=round(res + 1.96 * se_cell, 4),
                     cohens_d=round(res / g.w_sd, 4),
                     t_cell=round(t_cell, 2), p_cell=stats.t.sf(abs(t_cell), len(cm) - 1) * 2,
                     se_cell=round(se_cell, 4), se_row=round(se_row, 4),
                     se_inflation=round(se_cell / se_row, 2)))
rc = pd.DataFrame(rows)
rc.to_csv(os.path.join(OUT, "residuals_cell_level.csv"), index=False)
print("=== 1. residuals, cell-clustered ===")
print(rc[["group", "n_cells", "model_mean", "residual", "resid_ci_lo", "resid_ci_hi",
          "cohens_d", "t_cell", "se_inflation"]].to_string(index=False))
print(f"\n  residual range {rc.residual.min():.2f} to {rc.residual.max():.2f} | "
      f"d {rc.cohens_d.min():.2f} to {rc.cohens_d.max():.2f} | "
      f"CI half-width {1.96*rc.se_cell.min():.3f} to {1.96*rc.se_cell.max():.3f} | "
      f"SE inflation {rc.se_inflation.min():.1f}x to {rc.se_inflation.max():.1f}x")
print(f"  all residual CIs exclude zero: {bool((rc.resid_ci_lo > 0).all())}")

# ---- 2. condition contrast at cell level ---------------------------------------------------
r1 = pd.read_csv(os.path.join(ROOT, "runs", "run1", "data", "audit_results.csv"))
r2 = pd.read_csv(os.path.join(ROOT, "runs", "run2", "data", "audit_results.csv"))
for d in (r1, r2):
    d["phq8_total"] = d.phq8_total.clip(0, 24)
crows = []
for inst in ["phq8_total", "gad7_total", "audit_total"]:
    a = r1.groupby(CELL)[inst].mean(); b = r2.groupby(CELL)[inst].mean()
    diff = (a - b).dropna()
    t = stats.ttest_rel(a.loc[diff.index], b.loc[diff.index])
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    # row-level comparison, as previously reported
    mm = r1.merge(r2, on=["model", "profile_id", "iteration"], suffixes=("_c", "_n"))
    tr = stats.ttest_rel(mm[f"{inst}_c"], mm[f"{inst}_n"])
    crows.append(dict(instrument=inst, model="ALL", n_cells=len(diff),
                      diff=round(diff.mean(), 4),
                      ci_lo=round(diff.mean() - 1.96 * se, 4), ci_hi=round(diff.mean() + 1.96 * se, 4),
                      dz=round(diff.mean() / diff.std(ddof=1), 4),
                      t_cell=round(t.statistic, 3), p_cell=t.pvalue,
                      t_row_previously=round(tr.statistic, 3), p_row_previously=tr.pvalue,
                      significant=bool(t.pvalue < .05)))
    for mdl in sorted(r1.model.unique()):
        aa = r1[r1.model == mdl].groupby("profile_id")[inst].mean()
        bb = r2[r2.model == mdl].groupby("profile_id")[inst].mean()
        dd = (aa - bb).dropna()
        tt = stats.ttest_rel(aa.loc[dd.index], bb.loc[dd.index])
        se2 = dd.std(ddof=1) / np.sqrt(len(dd))
        s = mm[mm.model == mdl]
        trr = stats.ttest_rel(s[f"{inst}_c"], s[f"{inst}_n"])
        crows.append(dict(instrument=inst, model=mdl, n_cells=len(dd),
                          diff=round(dd.mean(), 4),
                          ci_lo=round(dd.mean() - 1.96 * se2, 4), ci_hi=round(dd.mean() + 1.96 * se2, 4),
                          dz=round(dd.mean() / dd.std(ddof=1), 4),
                          t_cell=round(tt.statistic, 3), p_cell=tt.pvalue,
                          t_row_previously=round(trr.statistic, 3), p_row_previously=trr.pvalue,
                          significant=bool(tt.pvalue < .05)))
cc = pd.DataFrame(crows)
cc.to_csv(os.path.join(OUT, "condition_contrast_cell_level.csv"), index=False)
print("\n=== 2. condition contrast, cell level (n=480 cells, was 14,400 rows) ===")
print(cc[cc.instrument == "phq8_total"][["model", "n_cells", "diff", "ci_lo", "ci_hi", "dz",
                                          "t_cell", "p_cell", "t_row_previously", "significant"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
flips = cc[(cc.p_row_previously < .05) & (cc.p_cell >= .05)]
if len(flips):
    print("\n  CLAIMS THAT DO NOT SURVIVE THE CORRECT UNIT OF ANALYSIS:")
    for r in flips.itertuples():
        print(f"    {r.instrument} / {r.model}: row p={r.p_row_previously:.3g} -> cell p={r.p_cell:.3g}")

# ---- 3. trans-cis elevation at cell level --------------------------------------------------
trows = []
for a, b in [("Transgender Woman", "Cisgender Woman"), ("Transgender Man", "Cisgender Man")]:
    ca = m[m.gender == a].groupby(CELL).phq8_total.mean()
    cb = m[m.gender == b].groupby(CELL).phq8_total.mean()
    t = stats.ttest_ind(ca, cb, equal_var=False)
    sp = np.sqrt((ca.var(ddof=1) + cb.var(ddof=1)) / 2)
    se = np.sqrt(ca.var(ddof=1) / len(ca) + cb.var(ddof=1) / len(cb))
    d = ca.mean() - cb.mean()
    trows.append(dict(contrast=f"{a} - {b}", n_cells_a=len(ca), n_cells_b=len(cb),
                      diff=round(d, 4), ci_lo=round(d - 1.96 * se, 4), ci_hi=round(d + 1.96 * se, 4),
                      cohens_d=round(d / sp, 4), t_cell=round(t.statistic, 3), p_cell=t.pvalue,
                      significant=bool(t.pvalue < .05)))
te = pd.DataFrame(trows)
te.to_csv(os.path.join(OUT, "trans_elevation_cell_level.csv"), index=False)
print("\n=== 3. trans-cis elevation, cell level ===")
print(te.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
