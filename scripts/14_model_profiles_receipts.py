"""
Per-model profile receipts for v2 Results: Model Behavioral Profiles.

Emits:
  analysis/per_model_residuals.csv        per model x benchmarkable group, cis-only frame
                                          (same frame-matching rule as 09_cisonly_receipts.py)
  analysis/asian_paradox.csv              per model White-Asian gap vs GT gap (0.755)
  analysis/trans_elevation_per_model.csv  per model TW-CW and TM-CM contrasts (descriptive)
  analysis/relationship_descriptive.csv   Single vs Married severity (descriptive; no GT exists)
Idempotent, no randomness.
"""
import pandas as pd, numpy as np, os

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
G = {r.group: r for _, r in gt.iterrows()}
m["phq8_total"] = m.phq8_total.clip(0, 24)
m["ses_clean"] = m.ses.str.replace('"', '', regex=False).str.split(" ").str[0]
CIS = ["Cisgender Man", "Cisgender Woman"]

GROUPS = [("race", "White", "White"), ("race", "Black", "Black"), ("race", "Asian", "Asian"),
          ("race", "Hispanic", "Hispanic (pooled)"),
          ("gender", "Cisgender Man", "Men"), ("gender", "Cisgender Woman", "Women"),
          ("ses_clean", "Low", "Low"), ("ses_clean", "Middle", "Middle"), ("ses_clean", "High", "High")]

rows = []
for mdl, sub in m.groupby("model"):
    for col, grp, gg in GROUPS:
        sel = (sub[col] == grp) if col == "gender" else ((sub[col] == grp) & sub.gender.isin(CIS))
        x = sub.loc[sel, "phq8_total"]
        g = G[gg]
        rows.append(dict(model=mdl, dimension=col.replace("_clean", ""), group=grp, n=len(x),
                         model_mean=round(x.mean(), 3),
                         residual=round(x.mean() - g.w_mean, 3),
                         cohens_d=round((x.mean() - g.w_mean) / g.w_sd, 3)))
pmr = pd.DataFrame(rows)
pmr.to_csv(os.path.join(OUT, "per_model_residuals.csv"), index=False)
summ = pmr.groupby("model").residual.agg(["mean", "min", "max"]).round(3)
print("=== per-model residual summary (cis-only frame) ===")
print(summ.to_string())

# --- Asian paradox: does the model reproduce the White-Asian GT gap? ----------------------
gt_gap = G["White"].w_mean - G["Asian"].w_mean
ap = []
for mdl, sub in m.groupby("model"):
    cis = sub[sub.gender.isin(CIS)]
    mg = cis.loc[cis.race == "White", "phq8_total"].mean() - cis.loc[cis.race == "Asian", "phq8_total"].mean()
    ap.append(dict(model=mdl, model_gap=round(mg, 3), gt_gap=round(gt_gap, 3),
                   calibration_error=round(mg - gt_gap, 3)))
cis_all = m[m.gender.isin(CIS)]
mg_all = cis_all.loc[cis_all.race == "White", "phq8_total"].mean() - cis_all.loc[cis_all.race == "Asian", "phq8_total"].mean()
ap.append(dict(model="POOLED", model_gap=round(mg_all, 3), gt_gap=round(gt_gap, 3),
               calibration_error=round(mg_all - gt_gap, 3)))
apdf = pd.DataFrame(ap)
apdf.to_csv(os.path.join(OUT, "asian_paradox.csv"), index=False)
print("\n=== Asian paradox (White-Asian gap, cis-only) ===")
print(apdf.to_string(index=False))

# --- trans elevation per model (descriptive; no benchmark exists) -------------------------
te = []
for mdl, sub in m.groupby("model"):
    for a, b in [("Transgender Woman", "Cisgender Woman"), ("Transgender Man", "Cisgender Man")]:
        va, vb = sub.loc[sub.gender == a, "phq8_total"], sub.loc[sub.gender == b, "phq8_total"]
        sp = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)
        te.append(dict(model=mdl, contrast=f"{a} - {b}", diff=round(va.mean() - vb.mean(), 3),
                       cohens_d=round((va.mean() - vb.mean()) / sp, 3)))
for a, b in [("Transgender Woman", "Cisgender Woman"), ("Transgender Man", "Cisgender Man")]:
    va, vb = m.loc[m.gender == a, "phq8_total"], m.loc[m.gender == b, "phq8_total"]
    sp = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)
    te.append(dict(model="POOLED", contrast=f"{a} - {b}", diff=round(va.mean() - vb.mean(), 3),
                   cohens_d=round((va.mean() - vb.mean()) / sp, 3)))
tedf = pd.DataFrame(te)
tedf.to_csv(os.path.join(OUT, "trans_elevation_per_model.csv"), index=False)
print("\n=== trans-cis elevation per model (descriptive) ===")
print(tedf.to_string(index=False))

# --- relationship status (descriptive; no representative GT by relationship) --------------
rl = []
for mdl, sub in m.groupby("model"):
    s = sub.groupby("relationship").phq8_total.mean()
    rl.append(dict(model=mdl, single_mean=round(s.get("Single", np.nan), 3),
                   married_mean=round(s.get("Married", np.nan), 3),
                   diff=round(s.get("Single", np.nan) - s.get("Married", np.nan), 3)))
s = m.groupby("relationship").phq8_total.mean()
rl.append(dict(model="POOLED", single_mean=round(s.get("Single", np.nan), 3),
               married_mean=round(s.get("Married", np.nan), 3),
               diff=round(s.get("Single", np.nan) - s.get("Married", np.nan), 3)))
rldf = pd.DataFrame(rl)
rldf.to_csv(os.path.join(OUT, "relationship_descriptive.csv"), index=False)
print("\n=== relationship: Single vs Married (descriptive) ===")
print(rldf.to_string(index=False))
