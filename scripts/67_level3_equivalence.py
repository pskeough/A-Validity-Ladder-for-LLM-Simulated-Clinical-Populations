"""Level 3 read as an equivalence, not a point comparison.

The body reports the one-band tolerance on point estimates: a model-by-group cell passes when its
residual on the anchor is under five points, and 29 of 36 cells and eight of nine marginals do.
A reviewer objected that a pass on a point estimate is not an affirmative result. This script
reads the same residuals as equivalence tests: a cell passes the use tolerance when the 90%
interval of its residual, with the anchor's design-based standard error folded in as a root sum
of squares, lies inside (-5, +5), which is the two one-sided tests at a bound of one severity
band. The strict tolerance, the anchor's own 95% design interval, is read the same way and never
holds. The Medicaid-matched anchor of Appendix B is run for the low-income cells alongside.

Units follow 18_cell_level_inference.py: a design cell is one model by one profile, its mean taken
over both prompt conditions (60 rows), and a model-by-group residual is the mean of its design
cells with the standard error taken across those cells. The marginal rows reproduce
residuals_cell_level.csv and per_model_residuals.csv reproduces the 36 residuals; both are gates.

Emits analysis/level3_equivalence.csv. No randomness, idempotent.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "analysis")
CIS = ["Cisgender Man", "Cisgender Woman"]
BAND = 5.0
ALPHA = 0.05

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"), low_memory=False)
m = m[m.gender.isin(CIS)].copy()
m["phq8_total"] = m["phq8_total_clipped"]
m["ses"] = m["ses_normalized"]
cells = m.groupby(["model", "profile_id", "race", "gender", "ses"], as_index=False).phq8_total.mean()

gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv")).set_index("group")
dse = pd.read_csv(os.path.join(OUT, "anchor_design_se.csv")).set_index("group")
med = pd.read_csv(os.path.join(OUT, "medicaid_anchor.csv"))
med_row = med[med.definition.str.startswith("D.")].iloc[0]          # income under $35k and Medicaid, the persona's own definition and the paper's Medicaid-matched anchor (+4.21)
GROUPS = [("race", "White", "White", "White"), ("race", "Black", "Black", "Black"),
          ("race", "Asian", "Asian", "Asian"), ("race", "Hispanic", "Hispanic (pooled)", "Hispanic"),
          ("gender", "Cisgender Man", "Men", "Cisgender men"), ("gender", "Cisgender Woman", "Women", "Cisgender women"),
          ("ses", "Low", "Low", "Low"), ("ses", "Middle", "Middle", "Middle"), ("ses", "High", "High", "High")]

prev_marg = pd.read_csv(os.path.join(OUT, "residuals_cell_level.csv")).set_index("group")
prev_model = pd.read_csv(os.path.join(OUT, "per_model_residuals.csv"))


def equivalence(resid, se_cell, se_anchor, df, bound):
    """Two one-sided tests that the residual lies inside (-bound, +bound); returns the larger p
    and the 90% interval used to read it."""
    se = float(np.hypot(se_cell, se_anchor))
    p_low = stats.t.sf((resid + bound) / se, df)
    p_high = stats.t.sf((bound - resid) / se, df)
    tcrit = stats.t.ppf(0.95, df)
    return float(max(p_low, p_high)), resid - tcrit * se, resid + tcrit * se, se


rows = []
for dim, grp, gkey, sekey in GROUPS:
    anchor = float(gt.loc[gkey, "w_mean"]); se_anchor = float(dse.loc[sekey, "se_design"])
    strict = 1.96 * se_anchor
    sub = cells[cells[dim] == grp]
    scopes = [("marginal", "ALL", sub)] + [("model", mdl, g) for mdl, g in sub.groupby("model")]
    for scope, mdl, g in scopes:
        cm = g.phq8_total.to_numpy()
        n = len(cm); mean = float(cm.mean()); se_cell = float(cm.std(ddof=1) / np.sqrt(n)); resid = mean - anchor
        p_band, lo90, hi90, se_tot = equivalence(resid, se_cell, se_anchor, n - 1, BAND)
        p_strict, _, _, _ = equivalence(resid, se_cell, se_anchor, n - 1, strict)
        row = dict(scope=scope, model=mdl, dimension=dim, group=grp, n_cells=n, model_mean=round(mean, 4),
                   anchor=round(anchor, 4), anchor_se=round(se_anchor, 4), residual=round(resid, 4),
                   se_cell=round(se_cell, 4), se_total=round(se_tot, 4),
                   resid_ci90_lo=round(lo90, 4), resid_ci90_hi=round(hi90, 4),
                   point_pass_band=bool(abs(resid) < BAND), p_tost_band=round(p_band, 6),
                   equiv_pass_band=bool(p_band < ALPHA),
                   strict_bound=round(strict, 4), point_pass_strict=bool(abs(resid) < strict),
                   p_tost_strict=round(p_strict, 6), equiv_pass_strict=bool(p_strict < ALPHA))
        if grp == "Low":
            a2 = float(med_row.w_mean); r2 = mean - a2
            # the Medicaid-matched anchor carries no design SE in the receipt; its SRS SE from w_sd and n is used
            se2 = float(med_row.w_sd) / np.sqrt(float(med_row.n))
            p2, lo2, hi2, _ = equivalence(r2, se_cell, se2, n - 1, BAND)
            row.update(medicaid_anchor=round(a2, 4), medicaid_residual=round(r2, 4),
                       medicaid_ci90_lo=round(lo2, 4), medicaid_ci90_hi=round(hi2, 4),
                       medicaid_point_pass_band=bool(abs(r2) < BAND), medicaid_p_tost_band=round(p2, 6),
                       medicaid_equiv_pass_band=bool(p2 < ALPHA))
        rows.append(row)

res = pd.DataFrame(rows)

# gates against the existing receipts
marg = res[res.scope == "marginal"].set_index("group")
for grp in marg.index:
    a, b = float(marg.loc[grp, "residual"]), float(prev_marg.loc[grp, "residual"])
    assert abs(a - b) < 0.0011, f"marginal residual for {grp}: {a} vs receipt {b}"
    assert abs(float(marg.loc[grp, "se_cell"]) - float(prev_marg.loc[grp, "se_cell"])) < 0.0011, f"se for {grp}"
per = res[res.scope == "model"]
for _, r in prev_model.iterrows():
    mine = per[(per.model == r.model) & (per.group == r.group)]
    assert len(mine) == 1, f"missing {r.model} {r.group}"
    assert abs(float(mine.residual.iloc[0]) - float(r.residual)) < 0.0011, f"{r.model} {r.group}: {mine.residual.iloc[0]} vs {r.residual}"
print("gates passed: nine marginal residuals and SEs match residuals_cell_level.csv; 36 residuals match per_model_residuals.csv")

res.to_csv(os.path.join(OUT, "level3_equivalence.csv"), index=False)
pd.set_option("display.width", 250)
cols = ["scope", "model", "group", "n_cells", "residual", "se_total", "resid_ci90_lo", "resid_ci90_hi",
        "point_pass_band", "equiv_pass_band", "strict_bound", "equiv_pass_strict"]
print(res[cols].to_string(index=False))
print(f"\ncells: point pass {int(per.point_pass_band.sum())} of {len(per)}, equivalence pass {int(per.equiv_pass_band.sum())} of {len(per)}"
      f" | strict: point {int(per.point_pass_strict.sum())}, equivalence {int(per.equiv_pass_strict.sum())}")
mg = res[res.scope == "marginal"]
print(f"marginals: point pass {int(mg.point_pass_band.sum())} of {len(mg)}, equivalence pass {int(mg.equiv_pass_band.sum())} of {len(mg)}"
      f" | strict: point {int(mg.point_pass_strict.sum())}, equivalence {int(mg.equiv_pass_strict.sum())}")
low = res[res.group == "Low"][["scope", "model", "residual", "resid_ci90_hi", "equiv_pass_band", "medicaid_residual", "medicaid_ci90_hi", "medicaid_equiv_pass_band"]]
print("\nlow income on the paper anchor and the Medicaid-matched anchor:\n" + low.to_string(index=False))
fail = per[~per.equiv_pass_band][["model", "group", "residual", "resid_ci90_hi"]]
print("\ncells failing the band as an equivalence:\n" + fail.to_string(index=False))
