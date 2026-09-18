"""Standard errors under the two clustering levels the design actually has.

Section 3.5 rejects the generation as the unit of analysis and computes every residual on cell
means. Two quantities in that argument had no receipt.

The first is the inflation factor itself. It was quoted as five to seven, which is the figure for
the model-by-cohort cell of 60 generations pooled across both prompt framings, the unit
18_cell_level_inference.py uses. Within a single condition the cell holds 30 generations and the
algebraic ceiling on the inflation is sqrt(30) = 5.48, so "seven" is unreachable there. Both are
computed here and the manuscript now names which unit each belongs to.

The second is the level above. Cells are crossed on four models, and cells within a model share
whatever that model does. Two-way cluster-robust standard errors over model and cohort put a bound
on how much the reported between-cell intervals understate that. Cameron-Gelbach-Miller: the
two-way variance is the sum of the two one-way clustered variances minus the variance clustered on
their intersection.

Emits analysis/clustering_receipts.csv.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
GROUPS = {"race": ["White", "Black", "Asian", "Hispanic"],
          "ses": ["Low", "Middle", "High"]}

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
for c in ["race", "gender", "ses", "relationship"]:
    m[c] = m[c].astype(str).str.strip().str.strip('"')
cis = m[m.gender.str.contains("Cis")].copy()

rows = []

# ---- 1. generation-level against cell-level, at both cell definitions -------------------------
for unit, keys in [("model x cohort, both conditions (60 draws)", ["model", "profile_id"]),
                   ("model x cohort x condition (30 draws)", ["model", "profile_id", "prompt_condition"])]:
    infl = []
    for dim, levels in GROUPS.items():
        for lev in levels:
            g = cis[cis[dim].str.startswith(lev)]
            if not len(g):
                continue
            se_gen = g.phq8_total.std(ddof=1) / np.sqrt(len(g))
            cm = g.groupby(keys).phq8_total.mean()
            se_cell = cm.std(ddof=1) / np.sqrt(len(cm))
            infl.append(se_cell / se_gen)
    rows.append(dict(quantity="SE inflation, cell over generation", unit=unit,
                     low=round(float(np.min(infl)), 2), high=round(float(np.max(infl)), 2),
                     n=len(infl)))
    print(f"{unit}: inflation {np.min(infl):.2f}x to {np.max(infl):.2f}x")

# ---- 2. two-way cluster-robust over model and cohort ------------------------------------------
# Fitted on the cell means, the same object Table 2 reports, so the ratio is directly comparable to
# the between-cell standard errors printed there.
cells = cis.groupby(["model", "profile_id", "race", "gender", "ses"],
                    as_index=False).phq8_total.mean()
ratios = []
for dim, levels in GROUPS.items():
    for lev in levels:
        d = cells[cells[dim].str.startswith(lev)].copy()
        if len(d) < 8:
            continue
        d["one"] = 1.0
        base = smf.ols("phq8_total ~ 1", data=d).fit()
        se_plain = float(base.bse.iloc[0])
        v_m = smf.ols("phq8_total ~ 1", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.model}).bse.iloc[0] ** 2
        v_c = smf.ols("phq8_total ~ 1", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.profile_id}).bse.iloc[0] ** 2
        v_i = smf.ols("phq8_total ~ 1", data=d).fit(
            cov_type="cluster",
            cov_kwds={"groups": d.model.astype(str) + "|" + d.profile_id.astype(str)}).bse.iloc[0] ** 2
        two_way = float(np.sqrt(max(v_m + v_c - v_i, 0.0)))
        ratios.append(two_way / se_plain)
rows.append(dict(quantity="two-way cluster-robust SE over between-cell SE",
                 unit="model and cohort, Cameron-Gelbach-Miller",
                 low=round(float(np.min(ratios)), 2), high=round(float(np.max(ratios)), 2),
                 n=len(ratios)))
print(f"two-way cluster-robust over between-cell: {np.min(ratios):.2f}x to {np.max(ratios):.2f}x "
      f"across {len(ratios)} benchmarked groups")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "clustering_receipts.csv"), index=False)

# ---- 3. the mean-controlled dispersion gap, also previously unreceipted -----------------------
c2 = m.groupby(["model", "profile_id", "prompt_condition"], as_index=False).agg(
    sd=("phq8_total", "std"), mu=("phq8_total", "mean"))
fit = smf.ols("sd ~ mu + I(mu**2) + I(mu**3) + C(model)", data=c2).fit()
c2["resid"] = fit.resid
w = c2.pivot_table(index=["model", "profile_id"], columns="prompt_condition", values="resid").dropna()
from scipy import stats as _st
diff = w.clinical - w.narrative
t = _st.ttest_1samp(diff, 0)
print(f"\nmean-controlled dispersion gap (clinical minus narrative): {diff.mean():+.4f} points, "
      f"t = {t.statistic:.2f}, p = {t.pvalue:.2e}")
pd.DataFrame([dict(gap_points=round(float(diff.mean()), 4), t=round(float(t.statistic), 2),
                   p=float(t.pvalue), n_cells=len(diff),
                   rho_mean_sd=round(float(_st.spearmanr(c2.mu, c2.sd)[0]), 3))]
             ).to_csv(os.path.join(OUT, "dispersion_mean_controlled.csv"), index=False)
