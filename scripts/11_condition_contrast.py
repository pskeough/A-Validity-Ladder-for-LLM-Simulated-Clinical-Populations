"""
Condition-contrast receipts for v2 Results II (clinical vs narrative prompt framing).

Condition assignment comes from RUN PROVENANCE, not the SES quote artifact:
run1/data/audit_results.csv = clinical (third-person profile injection),
run2/data/audit_results.csv = narrative (first-person self-description).
The two files' run_id sets are disjoint; the mapping is emitted as a receipt
(condition_runid_map.csv) and validated against the pooled file (every row of
data/model_outputs.csv must resolve to exactly one condition, n=14,400 each).

Emits:
  analysis/condition_runid_map.csv        run_id -> condition lookup (23 rows)
  analysis/condition_contrast.csv         paired contrasts, overall + per model x instrument
  analysis/condition_variance_compression.csv  within-cell SD by condition x SES
Idempotent, no randomness.
"""
import pandas as pd, numpy as np, os
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.join(BASE, "..")
OUT = os.path.join(BASE, "analysis")

run1 = pd.read_csv(os.path.join(ROOT, "runs", "run1", "data", "audit_results.csv"))
run2 = pd.read_csv(os.path.join(ROOT, "runs", "run2", "data", "audit_results.csv"))

# --- run_id -> condition map, with disjointness check -----------------------------------
ids1, ids2 = set(run1.run_id.unique()), set(run2.run_id.unique())
assert not (ids1 & ids2), f"run_id sets overlap: {ids1 & ids2}"
cmap = pd.DataFrame(
    [(rid, "clinical") for rid in sorted(ids1)] + [(rid, "narrative") for rid in sorted(ids2)],
    columns=["run_id", "condition"])
cmap.to_csv(os.path.join(OUT, "condition_runid_map.csv"), index=False)

# validate against the pooled release input
pooled = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"), usecols=["run_id"])
cond = pooled.run_id.map(dict(zip(cmap.run_id, cmap.condition)))
assert cond.notna().all(), "pooled rows with unmapped run_id"
assert (cond == "clinical").sum() == 14400 and (cond == "narrative").sum() == 14400

# --- paired contrasts --------------------------------------------------------------------
for df in (run1, run2):
    df["phq8_total"] = df["phq8_total"].clip(0, 24)

KEY = ["model", "profile_id", "iteration"]
merged = run1.merge(run2, on=KEY, suffixes=("_clin", "_narr"), validate="1:1")
assert len(merged) == 14400, f"expected 14,400 matched pairs, got {len(merged)}"

rows = []
def contrast(label, model, a, b):
    d = a - b
    t, p = stats.ttest_rel(a, b)
    n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    rows.append(dict(instrument=label, model=model, n_pairs=n,
                     mean_clinical=round(a.mean(), 3), mean_narrative=round(b.mean(), 3),
                     diff=round(d.mean(), 4), ci_lo=round(d.mean() - 1.96 * se, 4),
                     ci_hi=round(d.mean() + 1.96 * se, 4),
                     dz=round(d.mean() / d.std(ddof=1), 3), t=round(t, 3), p=p))

for inst in ["phq8_total", "gad7_total", "audit_total"]:
    contrast(inst, "ALL", merged[f"{inst}_clin"], merged[f"{inst}_narr"])
    for mdl, sub in merged.groupby("model"):
        contrast(inst, mdl, sub[f"{inst}_clin"], sub[f"{inst}_narr"])

# category crossing between conditions (5-band PHQ-8, same cut as 07_fdr_ledger.py)
cat = lambda x: pd.cut(x, [-1, 4, 9, 14, 19, 24], labels=False)
merged["flip"] = cat(merged.phq8_total_clin) != cat(merged.phq8_total_narr)
rows.append(dict(instrument="phq8_category_flip", model="ALL", n_pairs=len(merged),
                 mean_clinical=np.nan, mean_narrative=np.nan,
                 diff=round(merged.flip.mean() * 100, 2), ci_lo=np.nan, ci_hi=np.nan,
                 dz=np.nan, t=np.nan, p=np.nan))
for mdl, sub in merged.groupby("model"):
    rows.append(dict(instrument="phq8_category_flip", model=mdl, n_pairs=len(sub),
                     mean_clinical=np.nan, mean_narrative=np.nan,
                     diff=round(sub.flip.mean() * 100, 2), ci_lo=np.nan, ci_hi=np.nan,
                     dz=np.nan, t=np.nan, p=np.nan))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "condition_contrast.csv"), index=False)

# --- variance compression by condition x SES ---------------------------------------------
for df, cname in ((run1, "clinical"), (run2, "narrative")):
    df["ses_clean"] = df["ses"].str.replace('"', '', regex=False).str.split(" ").str[0]
vc = []
for cname, df in (("clinical", run1), ("narrative", run2)):
    sd_cell = df.groupby(["model", "profile_id"]).phq8_total.std(ddof=1)
    ses_of = df.groupby(["model", "profile_id"]).ses_clean.first()
    tmp = pd.DataFrame({"sd": sd_cell, "ses": ses_of}).groupby("ses").sd.mean()
    for ses, v in tmp.items():
        vc.append(dict(condition=cname, ses=ses, mean_within_cell_sd=round(v, 4)))
vcdf = pd.DataFrame(vc).pivot(index="ses", columns="condition", values="mean_within_cell_sd")
vcdf["ratio_clin_over_narr"] = (vcdf.clinical / vcdf.narrative).round(4)
vcdf.to_csv(os.path.join(OUT, "condition_variance_compression.csv"))

print(out.to_string(index=False))
print()
print(vcdf.to_string())
print("\nheadline: PHQ-8 clinical-narrative diff "
      f"{out.loc[(out.instrument=='phq8_total') & (out.model=='ALL'),'diff'].iloc[0]:+.3f}, "
      f"flip {out.loc[(out.instrument=='phq8_category_flip') & (out.model=='ALL'),'diff'].iloc[0]:.2f}%")
