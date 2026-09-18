"""
Receipts for v2 draft sections 7 (output covariance structure) and 8 (variance comparisons).
These are the two sections whose numbers were carried over from v1 without recomputed backing
CSVs (flagged in V2_SPEC: per-group noise floors + mean-conditioned robustness check).

Inputs:  data/model_outputs.csv, groundtruth/phq8_groundtruth_nhanes_2005_2018.csv
Outputs: analysis/covariance_divergence_GOLD.csv   - pairwise Frobenius distances vs reference,
                                                     raw + severity-matched (mean-conditioned)
         analysis/covariance_noise_floors.csv      - per-group split-half noise floors
         analysis/per_group_variance_GOLD.csv      - per-group SD ratio vs gold SD, bootstrap CI,
                                                     pooled + per-model
All resampling seeded (SEED=42). Correlation matrices use rows with complete phq8_1..phq8_8;
dropped-row counts are reported, never silent.
"""
import pandas as pd, numpy as np, os

SEED = 42
N_SPLIT = 200      # split-half draws for noise floors
N_MATCH = 100      # severity-matched resamples for the mean-conditioned check
N_BOOT = 1000      # bootstrap draws for SD-ratio CIs
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
BANDS = [-1, 4, 9, 14, 19, 24]   # PHQ-8 severity bands used throughout the paper

BASE = os.path.join(os.path.dirname(__file__), "..")
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
OUT = os.path.join(BASE, "analysis")
G = lambda grp: gt.loc[gt.group == grp].iloc[0]
rng = np.random.default_rng(SEED)

# same hygiene as 04_recompute_analysis.py: canonical SES + clip the 1 corrupted row
m["ses_clean"] = m["ses"].str.replace('"', '', regex=False).str.split(" ").str[0]
m["phq8_total"] = m["phq8_total"].clip(0, 24)
n_incomplete = int(m[ITEMS].isna().any(axis=1).sum())
c = m.dropna(subset=ITEMS).copy()   # complete-item rows for covariance work

def cohort(df, dim, group):
    return df[df[dim] == group]

def corr_mat(df):
    return df[ITEMS].corr().to_numpy()

def frob(a, b):
    return float(np.linalg.norm(a - b, ord="fro"))

# ---- 1. Noise floors: split-half Frobenius distance within each cohort ----------------
GENDERS = ["Cisgender Man", "Cisgender Woman", "Transgender Woman", "Transgender Man"]
RACES = ["White", "Black", "Asian", "Hispanic", "Multiracial"]
SES = ["Low", "Middle", "High"]
RELATIONSHIP = ["Single", "Married"]
cohorts = ([("gender", g) for g in GENDERS] + [("race", r) for r in RACES]
           + [("ses_clean", s) for s in SES] + [("relationship", r) for r in RELATIONSHIP])

floors = []
for dim, grp in cohorts:
    sub = cohort(c, dim, grp)
    idx = sub.index.to_numpy()
    ds = []
    for _ in range(N_SPLIT):
        perm = rng.permutation(idx)
        half = len(perm) // 2
        ds.append(frob(corr_mat(sub.loc[perm[:half]]), corr_mat(sub.loc[perm[half:]])))
    floors.append(dict(dimension=dim.replace("_clean", ""), group=grp, n=len(sub),
                       floor_mean=round(float(np.mean(ds)), 3),
                       floor_p95=round(float(np.quantile(ds, 0.95)), 3)))
floors = pd.DataFrame(floors)
floors.to_csv(os.path.join(OUT, "covariance_noise_floors.csv"), index=False)

# ---- 2. Pairwise Frobenius divergences (raw + severity-matched) -----------------------
# Race: each group vs White. Gender: every pairwise combination (the draft's referencing is
# ambiguous between "vs cis same-gender" and "vs cis men"; emit all pairs so the receipt
# covers either reading and the manuscript can cite the one it means).
pairs = [("race", r, "White") for r in RACES if r != "White"]
pairs += [("gender", a, b) for i, a in enumerate(GENDERS) for b in GENDERS[i + 1:]]
# SES contrasts: the low-SES cell carries the largest severity residual, so its covariance
# structure is asked the same question as the gender cells.
pairs += [("ses", "Low", "High"), ("ses", "Middle", "High"), ("ses", "Low", "Middle")]
# relationship completes the design: all four factorial axes measured on the same structural test
pairs += [("relationship", "Single", "Married")]

def matched_frob(sa, sb):
    """Mean-conditioned check: within each PHQ-8 severity band, subsample both cohorts to the
    smaller band count, so the two matrices are computed on severity-matched distributions."""
    ba = pd.cut(sa.phq8_total, BANDS, labels=False)
    bb = pd.cut(sb.phq8_total, BANDS, labels=False)
    ds = []
    for _ in range(N_MATCH):
        ia, ib = [], []
        for band in range(5):
            xa, xb = sa.index[ba == band].to_numpy(), sb.index[bb == band].to_numpy()
            k = min(len(xa), len(xb))
            if k < 2:
                continue
            ia.append(rng.choice(xa, k, replace=False))
            ib.append(rng.choice(xb, k, replace=False))
        ds.append(frob(corr_mat(sa.loc[np.concatenate(ia)]), corr_mat(sb.loc[np.concatenate(ib)])))
    return float(np.mean(ds)), float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975))

rows = []
for dim_raw, a, b in pairs:
    dim = "ses_clean" if dim_raw == "ses" else dim_raw
    sa, sb = cohort(c, dim, a), cohort(c, dim, b)
    d_raw = frob(corr_mat(sa), corr_mat(sb))
    d_m, d_lo, d_hi = matched_frob(sa, sb)
    fa = floors.loc[floors.group == a].iloc[0]
    fb = floors.loc[floors.group == b].iloc[0]
    # The flag used to be a single column named `exceeds_both_floors`, computed on the RAW
    # Frobenius while the manuscript tabulates the severity-MATCHED statistic. The name did not
    # say which basis it used, so a reader checking the paper against the release compared the
    # matched column in the table with a raw-basis flag beside it: 11 of 14 on the raw statistic
    # against 7 of 14 on the matched one. Both flags are now emitted, each naming its own basis,
    # and neither carries the old ambiguous name.
    #
    # On this corpus the matched flag is a strict subset of the raw one: 7 of 14 against 11 of 14,
    # and every matched-True row is also raw-True. That is the expected direction, since severity
    # matching removes level differences that inflate the raw distance. It is not guaranteed in
    # general -- matching can raise a distance -- so both columns are computed rather than one
    # being derived from the other.
    rows.append(dict(dimension=dim_raw, group_a=a, group_b=b, n_a=len(sa), n_b=len(sb),
                     frobenius=round(d_raw, 3),
                     matched_frobenius=round(d_m, 3),
                     matched_ci_lo=round(d_lo, 3), matched_ci_hi=round(d_hi, 3),
                     floor_p95_a=fa.floor_p95, floor_p95_b=fb.floor_p95,
                     raw_exceeds_both_floors=bool(d_raw > max(fa.floor_p95, fb.floor_p95)),
                     matched_exceeds_both_floors=bool(d_m > max(fa.floor_p95, fb.floor_p95))))
div = pd.DataFrame(rows)
div.to_csv(os.path.join(OUT, "covariance_divergence_GOLD.csv"), index=False)

# ---- 3. Per-group variance ratios vs gold SD (section 8 receipts) ---------------------
race_map = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic (pooled)"}
gender_map = {"Cisgender Man": "Men", "Cisgender Woman": "Women"}
ses_map = {"Low": "Low", "Middle": "Middle", "High": "High"}

def sd_ratio_rows(df, model_label):
    out = []
    for dim, col, mp in [("race", "race", race_map), ("gender", "gender", gender_map),
                         ("ses", "ses_clean", ses_map)]:
        for mg, gg in mp.items():
            sub = df[df[col] == mg]
            gsd = G(gg).w_sd
            vals = sub.phq8_total.to_numpy()
            ratio = vals.std(ddof=0) / gsd
            boots = [rng.choice(vals, len(vals), replace=True).std(ddof=0) / gsd
                     for _ in range(N_BOOT)]
            out.append(dict(model=model_label, dimension=dim, group=mg, n=len(sub),
                            model_sd=round(vals.std(ddof=0), 3), gt_sd=gsd,
                            sd_ratio=round(ratio, 3),
                            ci_lo=round(float(np.quantile(boots, 0.025)), 3),
                            ci_hi=round(float(np.quantile(boots, 0.975)), 3)))
    return out

var_rows = sd_ratio_rows(m, "POOLED")
for mod in sorted(m.model.unique()):
    var_rows += sd_ratio_rows(m[m.model == mod], mod)
var = pd.DataFrame(var_rows)
var.to_csv(os.path.join(OUT, "per_group_variance_GOLD.csv"), index=False)

# ---- report ----------------------------------------------------------------------------
pd.set_option("display.width", 150)
print(f"rows with incomplete phq8 items dropped from covariance work: {n_incomplete}")
print("\n1. SPLIT-HALF NOISE FLOORS (per cohort)\n" + "=" * 90)
print(floors.to_string(index=False))
print("\n2. PAIRWISE FROBENIUS DIVERGENCE (raw + severity-matched)\n" + "=" * 90)
print(div.to_string(index=False))
print("\n3. PER-GROUP SD RATIO vs GOLD (pooled first, then per model)\n" + "=" * 90)
print(var[var.model == "POOLED"].to_string(index=False))
print("\nDraft numbers to reconcile: TW 1.14, TM 1.06, CW 0.32, Asian 0.37, Black 0.24,")
print("Hispanic 0.23; floor 0.17/p95 0.23 (White); SES ratios 0.67-0.86, Asian 1.12.")
