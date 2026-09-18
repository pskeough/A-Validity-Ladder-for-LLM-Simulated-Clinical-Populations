"""Level 4 under two corrections a statistics reader asked for.

(1) Persona-level floors. In the corpus a row is one draw and a cohort holds 24 to 40 personas, so a
split-half floor over rows puts the same persona's draws on both sides and measures draw noise with
the persona set held fixed. Splitting by persona instead measures what the sample of personas can
resolve. This script recomputes each anchored cohort's floor over 200 persona splits and re-reads
the seven anchored contrasts' matched distances (from level4_population_comparison.csv) against
the larger of the two persona floors.

(2) A size-matched population band. The multiple of the floor grows with sample size under a real
difference, so the NHANES band of 0.51 to 1.36 was measured at 8,000 to 18,000 respondents per
group while the corpus matrices carry 4,000 to 8,000 rows. The NHANES groups are subsampled to the
corpus's matched row counts and the matched distance, the permutation-free floor and the multiple
are recomputed (50 matchings, 100 splits, seeded), giving a band at the corpus's own resolution.

Emits analysis/level4_floor_sensitivity.csv. Seeded, idempotent.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(BASE, "data", "nhanes_raw")
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
BANDS = [-1, 4, 9, 14, 19, 24]
rng = np.random.default_rng(20260909)
N_SPLIT_PERSONA = 200
N_SPLIT_POP = 100
N_MATCH_POP = 50


def cmat(a):
    return np.corrcoef(a, rowvar=False)


def frob(a, b):
    return float(np.linalg.norm(a - b, ord="fro"))


# ---- corpus ----
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"), low_memory=False)
m["phq8_total"] = m["phq8_total_clipped"]
m["ses"] = m["ses_normalized"]
c = m.dropna(subset=ITEMS).copy()
COHORTS = [("gender", "Cisgender Man"), ("gender", "Cisgender Woman"), ("race", "White"), ("race", "Black"),
           ("race", "Asian"), ("race", "Hispanic"), ("ses", "Low"), ("ses", "Middle"), ("ses", "High")]
floors = {}
for dim, grp in COHORTS:
    sub = c[c[dim] == grp]
    personas = sub.profile_id.unique()
    X = sub[ITEMS].to_numpy(float); pid = sub.profile_id.to_numpy()
    ds = []
    for _ in range(N_SPLIT_PERSONA):
        perm = rng.permutation(personas); half = len(perm) // 2
        left = np.isin(pid, perm[:half]); right = np.isin(pid, perm[half:2 * half])
        ds.append(frob(cmat(X[left]), cmat(X[right])))
    floors[grp] = dict(n_rows=len(sub), n_personas=len(personas), floor_persona_mean=float(np.mean(ds)),
                       floor_persona_p95=float(np.quantile(ds, 0.95)))
old = pd.read_csv(os.path.join(OUT, "covariance_noise_floors.csv")).set_index("group")
for grp in floors:
    floors[grp]["floor_row_p95"] = float(old.loc[grp, "floor_p95"])

cmp = pd.read_csv(os.path.join(OUT, "level4_population_comparison.csv"))
PAIRS = {"Women vs Men": ("Cisgender Woman", "Cisgender Man"), "Black vs White": ("Black", "White"),
         "Asian vs White": ("Asian", "White"), "Hispanic vs White": ("Hispanic", "White"),
         "Low vs High income": ("Low", "High"), "Middle vs High income": ("Middle", "High"),
         "Low vs Middle income": ("Low", "Middle")}
rows = []
for contrast, (a, b) in PAIRS.items():
    sim = cmp[(cmp.contrast == contrast) & (cmp.scope == "simulated pooled")].iloc[0]
    pop = cmp[(cmp.contrast == contrast) & (cmp.scope == "population (NHANES)")].iloc[0]
    fp = max(floors[a]["floor_persona_p95"], floors[b]["floor_persona_p95"])
    rows.append(dict(contrast=contrast, source="simulated pooled", n_per_matrix=int(sim.n_per_matrix),
                     matched_distance=float(sim.matched_distance), floor_row_p95=float(sim.binding_floor_p95),
                     multiple_row_floor=float(sim.floor_multiple), floor_persona_p95=fp,
                     multiple_persona_floor=float(sim.matched_distance) / fp,
                     floor_ratio=fp / float(sim.binding_floor_p95)))

# ---- NHANES size-matched band ----
CYCLES = {"D": 2005, "E": 2007, "F": 2009, "G": 2011, "H": 2013, "I": 2015, "J": 2017}
PHQ8 = [f"DPQ0{i}0" for i in range(1, 9)]
frames = []
for suf in CYCLES:
    demo = pd.read_sas(os.path.join(RAW, f"DEMO_{suf}.xpt"), format="xport")
    dpq = pd.read_sas(os.path.join(RAW, f"DPQ_{suf}.xpt"), format="xport")
    cols = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR"] + (["RIDRETH3"] if "RIDRETH3" in demo.columns else [])
    df = demo[cols].merge(dpq[["SEQN"] + PHQ8], on="SEQN", how="left")
    if "RIDRETH3" not in df.columns:
        df["RIDRETH3"] = np.nan
    frames.append(df)
nh = pd.concat(frames, ignore_index=True)
for col in PHQ8:
    nh[col] = nh[col].where(nh[col] <= 3, np.nan)
nh = nh[(nh.RIDAGEYR >= 18)].dropna(subset=PHQ8).copy()
nh["total"] = nh[PHQ8].sum(axis=1)
nh["band"] = pd.cut(nh.total, BANDS, labels=False).astype(int)
nh["sex"] = nh.RIAGENDR.map({1: "Men", 2: "Women"})


def race_of(r):
    if pd.notna(r["RIDRETH3"]) and r["RIDRETH3"] == 6:
        return "Asian"
    return {3: "White", 4: "Black", 1: "Hispanic", 2: "Hispanic"}.get(r["RIDRETH1"], "Other")


nh["race"] = nh.apply(race_of, axis=1)
nh["ses"] = pd.cut(nh.INDFMPIR, [-np.inf, 1.3, 3.5, np.inf], labels=["Low", "Middle", "High"], right=True)
nh["ses"] = nh.ses.astype(object).where(nh.INDFMPIR.notna(), None)
nh.loc[nh.INDFMPIR < 1.3, "ses"] = "Low"
POP = {"Women vs Men": ("sex", "Women", "Men"), "Black vs White": ("race", "Black", "White"), "Asian vs White": ("race", "Asian", "White"),
       "Hispanic vs White": ("race", "Hispanic", "White"), "Low vs High income": ("ses", "Low", "High"),
       "Middle vs High income": ("ses", "Middle", "High"), "Low vs Middle income": ("ses", "Low", "Middle")}
X = nh[PHQ8].to_numpy(float)


def matched_distance(ia, ib, bands, n_match):
    ds = []
    for _ in range(n_match):
        sa, sb = [], []
        for bd in range(5):
            xa = ia[bands[ia] == bd]; xb = ib[bands[ib] == bd]
            k = min(len(xa), len(xb))
            if k < 2:
                continue
            sa.append(rng.choice(xa, k, replace=False)); sb.append(rng.choice(xb, k, replace=False))
        sa, sb = np.concatenate(sa), np.concatenate(sb)
        ds.append(frob(cmat(X[sa]), cmat(X[sb])))
    return float(np.mean(ds))


def split_floor(idx, n_split):
    ds = []
    for _ in range(n_split):
        p = rng.permutation(idx); h = len(p) // 2
        ds.append(frob(cmat(X[p[:h]]), cmat(X[p[h:2 * h]])))
    return float(np.quantile(ds, 0.95))


bands = nh.band.to_numpy()
for contrast, (col, a, b) in POP.items():
    ia_full = np.where(nh[col].to_numpy() == a)[0]; ib_full = np.where(nh[col].to_numpy() == b)[0]
    sim_n = int(cmp[(cmp.contrast == contrast) & (cmp.scope == "simulated pooled")].iloc[0].n_per_matrix)
    # subsample each population group so the matched matrices carry about the corpus's row count
    target = min(len(ia_full), len(ib_full), int(sim_n * 1.15))
    ia = rng.choice(ia_full, min(len(ia_full), target), replace=False)
    ib = rng.choice(ib_full, min(len(ib_full), target), replace=False)
    d_full = float(cmp[(cmp.contrast == contrast) & (cmp.scope == "population (NHANES)")].iloc[0].matched_distance)
    mult_full = float(cmp[(cmp.contrast == contrast) & (cmp.scope == "population (NHANES)")].iloc[0].floor_multiple)
    d_sub = matched_distance(ia, ib, bands, N_MATCH_POP)
    fl = max(split_floor(ia, N_SPLIT_POP), split_floor(ib, N_SPLIT_POP))
    rows.append(dict(contrast=contrast, source="population (NHANES), size-matched", n_per_matrix=int(min(len(ia), len(ib))),
                     matched_distance=d_sub, floor_row_p95=fl, multiple_row_floor=d_sub / fl,
                     floor_persona_p95=np.nan, multiple_persona_floor=np.nan, floor_ratio=np.nan,
                     multiple_full_sample=mult_full))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "level4_floor_sensitivity.csv"), index=False)
fl = pd.DataFrame(floors).T
fl.to_csv(os.path.join(OUT, "level4_persona_floors.csv"))
pd.set_option("display.width", 220)
print(fl.round(3).to_string())
print(res.round(3).to_string(index=False))
