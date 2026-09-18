"""
Builds the v2 public release file data/model_outputs_v2.csv from the two RAW run exports,
replacing the earlier quote-artifact-labeled build (08_stability_receipts.py).

Fixes relative to the v1 consolidated export:
  - prompt_condition in {clinical, narrative} assigned from run provenance
    (run1 = clinical, run2 = narrative; run_id sets disjoint; map receipt:
    analysis/condition_runid_map.csv). condition_source = "run_provenance" everywhere.
  - run1's PCL-5 items 6-20 are preserved (the old merge dropped them for all but 46 rows).
  - ses_normalized column (Low/Middle/High) alongside the raw ses string.
  - phq8_total_clipped convenience column (raw totals preserved untouched;
    exactly one run1 row exceeds the instrument maximum, see paper section 3.1).
Idempotent, no randomness.
"""
import pandas as pd, os

BASE = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.join(BASE, "..")

run1 = pd.read_csv(os.path.join(ROOT, "runs", "run1", "data", "audit_results.csv"))
run2 = pd.read_csv(os.path.join(ROOT, "runs", "run2", "data", "audit_results.csv"))
ids1, ids2 = set(run1.run_id), set(run2.run_id)
assert not (ids1 & ids2), "run_id sets overlap"

run1["prompt_condition"] = "clinical"
run2["prompt_condition"] = "narrative"

rel = pd.concat([run1, run2], ignore_index=True)
rel["condition_source"] = "run_provenance"
rel["ses_normalized"] = rel["ses"].str.replace('"', '', regex=False).str.split(" ").str[0]
rel["phq8_total_clipped"] = rel["phq8_total"].clip(0, 24)

assert len(rel) == 28800
assert rel.prompt_condition.value_counts().eq(14400).all()
assert set(rel.ses_normalized) == {"Low", "Middle", "High"}
n_over = int((rel.phq8_total > 24).sum())
assert n_over == 1, f"expected exactly 1 out-of-range PHQ-8 total, got {n_over}"
# every (model, profile_id, iteration) cell appears once per condition
sizes = rel.groupby(["model", "profile_id", "iteration"]).size()
assert (sizes == 2).all()

dest = os.path.join(BASE, "data", "model_outputs_v2.csv")
rel.to_csv(dest, index=False)
print(f"wrote {dest}: {len(rel)} rows, {len(rel.columns)} cols")
print(rel.prompt_condition.value_counts().to_string())
print("pcl5_6..20 non-null (run1 preservation check):",
      int(rel.loc[rel.prompt_condition == "clinical", "pcl5_6"].notna().sum()))
