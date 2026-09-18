"""
Receipts for four analyses present in v1 but absent from the v2 draft, all recomputed on the
correct within-condition frame (v1 computed the first three on cross-run pairs, which mixes the
prompt-framing effect into every quantity):

  A. Refusal / error accounting across all 28,800 generations.
  B. Stochastic fracture: does any demographic dimension predict differential run-to-run
     variability? v1 answered no on all four dimensions using cross-run pairs. Recomputed
     within-run, gender and SES answer yes, so the receipt also carries a mean-conditioned
     check (severity-band-stratified) to separate a genuine dispersion effect from
     mean-variance coupling.
  C. Per-instrument stability: within-run band-flip probability per instrument.
     PCL-5 is reported within-run only; the two runs used different PCL-5 forms
     (20-item vs 5-item), so v1's cross-run PCL-5 stability figure compared unlike totals.
  D. Diagnostic category transition matrix across the two conditions.

Emits analysis/{refusal_accounting,stochastic_fracture,per_instrument_stability,
transition_matrix}.csv. Seeded (SEED=42).
"""
import pandas as pd, numpy as np, os
from scipy import stats

SEED = 42
BASE = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.join(BASE, "..")
OUT = os.path.join(BASE, "analysis")
rng = np.random.default_rng(SEED)

run1 = pd.read_csv(os.path.join(ROOT, "runs", "run1", "data", "audit_results.csv"))
run2 = pd.read_csv(os.path.join(ROOT, "runs", "run2", "data", "audit_results.csv"))
RUNS = (("clinical", run1), ("narrative", run2))
for _, d in RUNS:
    d["phq8_total"] = d.phq8_total.clip(0, 24)
    d["ses_clean"] = d.ses.str.replace('"', '', regex=False).str.split(" ").str[0]

# ---- A. refusals / errors ----------------------------------------------------------------
acc = []
for cond, d in RUNS:
    acc.append(dict(condition=cond, n_rows=len(d),
                    refusals=int((d.refusal_flag == True).sum()),
                    errors=int(d.error_type.notna().sum()) if "error_type" in d else 0,
                    out_of_range_phq8=int((pd.read_csv(
                        os.path.join(ROOT, "runs", "run1" if cond == "clinical" else "run2",
                                     "data", "audit_results.csv")).phq8_total > 24).sum())))
acc_df = pd.DataFrame(acc)
acc_df.to_csv(os.path.join(OUT, "refusal_accounting.csv"), index=False)
print("=== A. refusal accounting ===")
print(acc_df.to_string(index=False))
print(f"  total refusals across {acc_df.n_rows.sum():,} generations: {acc_df.refusals.sum()}")

# ---- B. stochastic fracture ---------------------------------------------------------------
DIMS = [("race", "race"), ("gender", "gender"), ("ses", "ses_clean"), ("relationship", "relationship")]
rows = []
for cond, d in RUNS:
    cell = d.groupby(["model", "profile_id"]).agg(
        sd=("phq8_total", lambda x: x.std(ddof=1)),
        mean=("phq8_total", "mean"),
        race=("race", "first"), gender=("gender", "first"),
        ses_clean=("ses_clean", "first"), relationship=("relationship", "first")).reset_index()
    # mean-conditioned residual dispersion: regress cell SD on cell mean, test the residual.
    # Isolates "is this group noisier than its severity implies" from mean-variance coupling.
    slope, intercept = np.polyfit(cell["mean"], cell.sd, 1)
    cell["sd_resid"] = cell.sd - (slope * cell["mean"] + intercept)
    for label, col in DIMS:
        groups = [g.sd.values for _, g in cell.groupby(col)]
        H, p = stats.kruskal(*groups)
        gr = [g.sd_resid.values for _, g in cell.groupby(col)]
        Hr, pr = stats.kruskal(*gr)
        rows.append(dict(condition=cond, dimension=label, k_groups=len(groups), n_cells=len(cell),
                         H=round(H, 3), p=p, significant=bool(p < .05),
                         H_mean_conditioned=round(Hr, 3), p_mean_conditioned=pr,
                         significant_mean_conditioned=bool(pr < .05)))
frac = pd.DataFrame(rows)
frac.to_csv(os.path.join(OUT, "stochastic_fracture.csv"), index=False)
print("\n=== B. stochastic fracture (within-run) ===")
print(frac.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# per-group mean SD for the dimensions that come out significant, so the manuscript can name
# which cells drive it
detail = []
for cond, d in RUNS:
    cell = d.groupby(["model", "profile_id"]).agg(
        sd=("phq8_total", lambda x: x.std(ddof=1)),
        gender=("gender", "first"), ses_clean=("ses_clean", "first")).reset_index()
    for col in ["gender", "ses_clean"]:
        for grp, g in cell.groupby(col):
            detail.append(dict(condition=cond, dimension=col.replace("_clean", ""),
                               group=grp, n_cells=len(g), mean_within_cell_sd=round(g.sd.mean(), 4)))
det = pd.DataFrame(detail)
det.to_csv(os.path.join(OUT, "stochastic_fracture_detail.csv"), index=False)
print("\n--- driving cells (mean within-cell SD) ---")
print(det[det.condition == "clinical"].to_string(index=False))

# ---- C. per-instrument stability ----------------------------------------------------------
# Standard clinical band cutpoints per instrument.
BANDS = {"phq8_total": [4, 9, 14, 19],      # none/mild/moderate/mod-severe/severe
         "gad7_total": [4, 9, 14],          # minimal/mild/moderate/severe
         "audit_total": [2, 3, 7]}          # AUDIT-C risk bands (sex-neutral operationalization)
inst_rows = []
for cond, d in RUNS:
    for inst, cuts in list(BANDS.items()) + [("pcl5_total", None)]:
        if inst not in d.columns:
            continue
        probs, sds = [], []
        for _, sub in d.groupby(["model", "profile_id"]):
            v = sub[inst].dropna().values
            if len(v) < 2:
                continue
            sds.append(v.std(ddof=1))
            if cuts is not None:
                b = np.digitize(v, cuts, right=True)
                _, cnt = np.unique(b, return_counts=True)
                n = len(v)
                probs.append(1 - (cnt * (cnt - 1)).sum() / (n * (n - 1)))
        inst_rows.append(dict(condition=cond, instrument=inst.replace("_total", "").upper(),
                              n_cells=len(sds),
                              flip_prob_pct=round(np.mean(probs) * 100, 2) if probs else np.nan,
                              mean_within_cell_sd=round(float(np.mean(sds)), 4),
                              note="" if cuts is not None else
                                   "form differs across runs (20-item vs 5-item); within-run only"))
inst = pd.DataFrame(inst_rows)
inst.to_csv(os.path.join(OUT, "per_instrument_stability.csv"), index=False)
print("\n=== C. per-instrument within-run stability ===")
print(inst.to_string(index=False))

# ---- D. transition matrix ------------------------------------------------------------------
m = pd.concat([run1.assign(cond="clinical"), run2.assign(cond="narrative")], ignore_index=True)
w = m.pivot_table(index=["model", "profile_id", "iteration"], columns="cond", values="phq8_total")
LBL = ["None (0-4)", "Mild (5-9)", "Moderate (10-14)", "Mod-severe (15-19)", "Severe (20-24)"]
cat = lambda x: np.digitize(x, [4, 9, 14, 19], right=True)
tab = pd.crosstab(pd.Series(cat(w.clinical.values), name="clinical"),
                  pd.Series(cat(w.narrative.values), name="narrative"))
tab = tab.reindex(index=range(5), columns=range(5), fill_value=0)
tab.index = tab.columns = LBL
tab.to_csv(os.path.join(OUT, "transition_matrix.csv"))
diag = int(np.trace(tab.values)); tot = int(tab.values.sum())
print("\n=== D. diagnostic transition matrix, CROSS-CONDITION (clinical rows x narrative cols) ===")
print(tab.to_string())
print(f"  stable {diag:,}/{tot:,} = {diag/tot*100:.2f}%  (flip {100-diag/tot*100:.2f}%)")
mild_mod = int(tab.iloc[2, 1]); mod_mild = int(tab.iloc[1, 2])
print(f"  mild/moderate boundary traffic: {mild_mod:,} moderate->mild, {mod_mild:,} mild->moderate")

# ---- D2. WITHIN-RUN transition matrix: the object Section 6.2's claim actually needs ------
# All C(30,2)=435 unordered iteration pairs within each cell, symmetrized, at fixed prompt.
for cname, df in RUNS:
    M = np.zeros((5, 5), dtype=np.int64)
    for _, sub in df.groupby(["model", "profile_id"]):
        b = cat(sub.phq8_total.values)
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                M[b[i], b[j]] += 1
                M[b[j], b[i]] += 1
    t2 = pd.DataFrame(M, index=LBL, columns=LBL)
    t2.to_csv(os.path.join(OUT, f"transition_matrix_within_{cname}.csv"))
    d2, n2 = int(np.trace(M)), int(M.sum())
    print(f"\n=== D2. WITHIN-RUN transition matrix, {cname} (fixed prompt, all iteration pairs) ===")
    print(t2.to_string())
    print(f"  same-band {d2:,}/{n2:,} = {d2/n2*100:.2f}%  (flip {100-d2/n2*100:.2f}%)")
    b_mm = int(M[2, 1])
    off = n2 - d2
    print(f"  mild/moderate boundary traffic: {b_mm:,} of {off:,} discordant pairs "
          f"({b_mm/off*100:.1f}% of all crossings)")
