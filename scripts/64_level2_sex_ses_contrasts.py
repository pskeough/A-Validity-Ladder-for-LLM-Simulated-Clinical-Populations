"""Level 2 on the sex and socioeconomic axes, paired exactly as the racial contrasts are.

The racial contrasts (44, 50) pair a contrast cell with a reference cell that shares model,
condition, gender, socioeconomic level and relationship status, so the two differ in race alone.
Here the same construction runs on the two other anchored axes. The sex contrast pairs a cisgender
woman cell with the cisgender man cell sharing model, condition, race, socioeconomic level and
relationship status (240 pairs). The three socioeconomic contrasts pair a low- or middle-income
cell with its high- or middle-income counterpart sharing model, condition, race, gender and
relationship status (160 pairs each). For each contrast:

  paired t over the matched cell pairs, against zero and against the population gap, the second
      carrying the anchor's design-based standard error as the root sum of squares
  per-model contrasts and the count of models carrying the population's sign
  stratum bootstrap resampling cohorts within model, 10,000 draws
  TOST at a bound equal to the population disparity, so a missing gap can be told from a
      low-powered one, as in 50
  the ratio of simulated to population gap with a bootstrap interval, which is the number the
      inflated gradient needs

Benjamini-Hochberg runs across these four contrasts within each question, a family separate from
the three racial contrasts and from the 34-test ledger.

Emits analysis/level2_sex_ses_contrasts.csv. Seeded.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "analysis")
CIS = ["Cisgender Man", "Cisgender Woman"]
N_BOOT = 10_000
RNG = np.random.default_rng(20260909)

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"), low_memory=False)
m = m[m.gender.isin(CIS)].copy()
m["phq8_total"] = m["phq8_total_clipped"]
m["ses"] = m["ses_normalized"]
m["condition"] = m["prompt_condition"]
cell = (m.groupby(["model", "profile_id", "condition", "race", "gender", "ses", "relationship"],
                  as_index=False).phq8_total.mean())

gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
gtm = {g: float(gt.loc[gt.group == g, "w_mean"].iloc[0]) for g in ["Men", "Women", "Low", "Middle", "High"]}
d_se = pd.read_csv(os.path.join(OUT, "anchor_design_se.csv")).set_index("group")
SE_KEY = {"Men": "Cisgender men", "Women": "Cisgender women", "Low": "Low", "Middle": "Middle", "High": "High"}
gtse = {g: float(d_se.loc[k, "se_design"]) for g, k in SE_KEY.items()}

# contrast name, column, high group, low (reference) group, pairing stratum
SPECS = [
    ("Women minus Men", "gender", "Cisgender Woman", "Cisgender Man", ["model", "condition", "race", "ses", "relationship"], "Women", "Men"),
    ("Low minus High SES", "ses", "Low", "High", ["model", "condition", "race", "gender", "relationship"], "Low", "High"),
    ("Middle minus High SES", "ses", "Middle", "High", ["model", "condition", "race", "gender", "relationship"], "Middle", "High"),
    ("Low minus Middle SES", "ses", "Low", "Middle", ["model", "condition", "race", "gender", "relationship"], "Low", "Middle"),
]


def tost(diff, bound):
    """Two one-sided tests that the mean lies inside (-bound, +bound); returns the larger p."""
    n = len(diff); mu = diff.mean(); se = diff.std(ddof=1) / np.sqrt(n)
    p_low = stats.t.sf((mu + bound) / se, n - 1)     # H0: mu <= -bound
    p_high = stats.t.sf((bound - mu) / se, n - 1)    # H0: mu >= +bound
    return float(max(p_low, p_high))


rows = []
for name, col, hi, lo, stratum, ghi, glo in SPECS:
    a = cell[cell[col] == hi].set_index(stratum).phq8_total
    b = cell[cell[col] == lo].set_index(stratum).phq8_total
    for tag, idx in ((hi, a.index), (lo, b.index)):
        if idx.has_duplicates:
            raise SystemExit(f"{name}: {tag} stratum key is not unique")
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_frame("diff").reset_index()
    n = len(d)
    diff = d["diff"].to_numpy()
    obs = float(diff.mean()); se = float(diff.std(ddof=1) / np.sqrt(n))
    t0 = obs / se; p_zero = float(stats.t.sf(abs(t0), n - 1) * 2)
    pop = gtm[ghi] - gtm[glo]
    se_pop = float(np.hypot(gtse[ghi], gtse[glo]))
    se_gap = float(np.hypot(se, se_pop))
    t1 = (obs - pop) / se_gap; p_pop = float(stats.t.sf(abs(t1), n - 1) * 2)
    p_tost = tost(diff, abs(pop))
    by_model = {k: g["diff"].to_numpy() for k, g in d.groupby("model")}
    per_model = {k: round(float(v.mean()), 3) for k, v in by_model.items()}
    picks = [RNG.integers(0, len(v), size=(N_BOOT, len(v))) for v in by_model.values()]
    boot = np.concatenate([v[p] for v, p in zip(by_model.values(), picks)], axis=1).mean(axis=1)
    lo_b, hi_b = np.percentile(boot, [2.5, 97.5])
    ratio = obs / pop; r_lo, r_hi = np.percentile(boot / pop, [2.5, 97.5])
    rows.append(dict(
        contrast=name, n_pairs=n, simulated=round(obs, 4), se=round(se, 4),
        ci_lo=round(obs - 1.96 * se, 4), ci_hi=round(obs + 1.96 * se, 4),
        boot_lo=round(float(lo_b), 4), boot_hi=round(float(hi_b), 4),
        t_vs_zero=round(t0, 2), p_vs_zero=p_zero,
        population=round(pop, 4), pop_se=round(se_pop, 4),
        gap=round(obs - pop, 4), t_vs_pop=round(t1, 2), p_vs_pop=p_pop,
        p_tost_at_population=round(p_tost, 6),
        ratio=round(ratio, 3), ratio_lo=round(float(r_lo), 3), ratio_hi=round(float(r_hi), 3),
        sign_matches_population=bool(np.sign(obs) == np.sign(pop)),
        models_with_population_sign=int(sum(np.sign(v) == np.sign(pop) for v in per_model.values())),
        per_model=str(per_model)))

res = pd.DataFrame(rows)
for col in ("p_vs_zero", "p_vs_pop"):
    order = res[col].rank(method="first")
    res[col.replace("p_", "q_")] = (res[col] * len(res) / order).clip(upper=1.0).round(6)
res.to_csv(os.path.join(OUT, "level2_sex_ses_contrasts.csv"), index=False)
pd.set_option("display.width", 220)
print(res[["contrast", "n_pairs", "simulated", "ci_lo", "ci_hi", "population", "gap", "q_vs_zero", "q_vs_pop",
           "p_tost_at_population", "ratio", "ratio_lo", "ratio_hi", "models_with_population_sign"]].to_string(index=False))
for r in rows:
    print(f"  {r['contrast']:24s} per model: {r['per_model']}")
