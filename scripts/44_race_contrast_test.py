"""Test the racial contrasts individually, which Section 4.2 previously declined to do.

The reversal is a headline claim in the abstract and the conclusion, and the section reporting it
said the contrasts were not tested. That gap is closed here with data already collected: no new
generations, only an analysis the design always supported.

The contrast is paired by construction. A White cell and a Black cell that share a model, gender,
socioeconomic level and relationship status differ in exactly one factor, so the 48 such pairs give
a within-stratum difference and the pairing removes the between-cohort variance that made the
marginal intervals look wide. Three estimates are reported:

  paired t over the 48 matched cell pairs               -- primary
  stratum bootstrap, resampling the 12 strata within model, 10,000 draws
  per-model contrasts                                   -- so pooling can be inspected

Two questions are separable and both are answered. Is the simulated contrast distinguishable from
zero, and is it distinguishable from the population contrast? The second is the reversal claim.
NHANES anchor error enters the second through the contrast standard error, formed as the root sum
of squares of the two group design standard errors, which is valid because the race groups are
disjoint samples.

Benjamini-Hochberg is applied across the three contrasts within each question.

Emits analysis/race_contrast_tests.csv. Seeded, so the bootstrap is reproducible.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
CIS = ["Cisgender Man", "Cisgender Woman"]
STRATUM = ["model", "condition", "gender", "ses", "relationship"]
CONTRASTS = ["Black", "Hispanic", "Asian"]
N_BOOT = 10_000
RNG = np.random.default_rng(20260729)

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
m = m[m.gender.isin(CIS)]

# The raw ses column carries quoting variants from the two export code paths, so it has six
# distinct strings for three levels. Left alone it doubles the strata and the pairing then happens
# within a quoting variant, which tracks the condition only by accident of how each run was written.
# Cleaning it and joining the recovered condition label makes that pairing explicit instead.
m["ses"] = m.ses.str.replace('"', "", regex=False).str.split(" ").str[0]
cmap = pd.read_csv(os.path.join(OUT, "condition_runid_map.csv"))
m = m.merge(cmap, on="run_id", how="left")
if m.condition.isna().any():
    raise SystemExit(f"unlabelled rows: {int(m.condition.isna().sum())}")

# One value per design cell, then one row per (stratum, race).
cell = (m.groupby(["model", "profile_id", "condition", "race", "gender", "ses", "relationship"],
                  as_index=False).phq8_total.mean())

gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
GTK = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic (pooled)"}
gtm = {g: float(gt.loc[gt.group == k, "w_mean"].iloc[0]) for g, k in GTK.items()}

se_path = os.path.join(OUT, "anchor_design_se.csv")
gtse = {}
if os.path.exists(se_path):
    d = pd.read_csv(se_path).set_index("group")
    for g, k in GTK.items():
        # the ground-truth table labels the pooled Hispanic row "Hispanic (pooled)" while the
        # design-SE table labels it "Hispanic"; fall back so the SE is not silently dropped
        key = k if k in d.index else (g if g in d.index else None)
        if key is None:
            raise KeyError("no design SE for %s (tried %r and %r)" % (g, k, g))
        gtse[g] = float(d.loc[key, "se_design"])

rows = []
for grp in CONTRASTS:
    a = cell[cell.race == grp].set_index(STRATUM).phq8_total
    b = cell[cell.race == "White"].set_index(STRATUM).phq8_total
    # Duplicate stratum labels would make this subtraction broadcast instead of pair, silently.
    for name, idx in (("contrast", a.index), ("White", b.index)):
        if idx.has_duplicates:
            raise SystemExit(f"{grp}: {name} stratum key is not unique, pairing would be wrong")
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_frame("diff").reset_index()
    n = len(d)

    obs = float(d["diff"].mean())
    se = float(d["diff"].std(ddof=1) / np.sqrt(n))
    t0 = obs / se
    p_zero = float(stats.t.sf(abs(t0), n - 1) * 2)

    pop = gtm[grp] - gtm["White"]
    # Anchor error is small but not zero; carrying it makes the second test conservative.
    se_pop = float(np.hypot(gtse.get(grp, 0.0), gtse.get("White", 0.0)))
    se_gap = float(np.hypot(se, se_pop))
    t1 = (obs - pop) / se_gap
    p_pop = float(stats.t.sf(abs(t1), n - 1) * 2)

    # Bootstrap the 12 strata within each model, which is the level dependence actually lives at.
    # Resampling is done on integer indices into a plain array: the same draw expressed through
    # pandas .loc inside the loop took longer than the rest of the script by two orders of magnitude.
    d["cohort"] = d.gender + "|" + d.ses + "|" + d.relationship
    by_model = {k: g for k, g in d.groupby("model")}
    blocks = [g["diff"].to_numpy() for g in by_model.values()]
    picks = [RNG.integers(0, len(b), size=(N_BOOT, len(b))) for b in blocks]
    boot = np.concatenate([b[p] for b, p in zip(blocks, picks)], axis=1).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    per_model = {k: round(float(g["diff"].mean()), 3) for k, g in by_model.items()}
    rows.append(dict(
        contrast=f"{grp} minus White", n_pairs=n,
        simulated=round(obs, 4), se=round(se, 4),
        ci_lo=round(obs - 1.96 * se, 4), ci_hi=round(obs + 1.96 * se, 4),
        boot_lo=round(float(lo), 4), boot_hi=round(float(hi), 4),
        t_vs_zero=round(t0, 2), p_vs_zero=p_zero,
        population=round(pop, 4), pop_se=round(se_pop, 4),
        gap=round(obs - pop, 4), t_vs_pop=round(t1, 2), p_vs_pop=p_pop,
        sign_flip=bool(np.sign(obs) != np.sign(pop)),
        models_agreeing_with_pooled_sign=sum(np.sign(v) == np.sign(obs) for v in per_model.values()),
        per_model=str(per_model)))

res = pd.DataFrame(rows)
for col in ("p_vs_zero", "p_vs_pop"):
    order = res[col].rank(method="first")
    res[col.replace("p_", "q_")] = (res[col] * len(res) / order).clip(upper=1.0).round(6)
res.to_csv(os.path.join(OUT, "race_contrast_tests.csv"), index=False)

pd.set_option("display.width", 200)
print(res[["contrast", "n_pairs", "simulated", "ci_lo", "ci_hi", "boot_lo", "boot_hi",
           "q_vs_zero", "population", "gap", "q_vs_pop", "sign_flip",
           "models_agreeing_with_pooled_sign"]].to_string(index=False))
print()
for r in rows:
    print(f"  {r['contrast']:22s} per model: {r['per_model']}")
