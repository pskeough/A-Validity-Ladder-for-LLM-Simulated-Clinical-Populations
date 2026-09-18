"""
Fit the same one-factor model inside every cohort.

Section 2 sets Villarreal-Zegarra and Bellido-Boza against this paper and argues both can hold: a set
of cohorts can each yield a well-fitting single-factor solution while differing from one another in
the structure Section 9 measures. That reconciliation is asserted there and not tested. It is a
falsifiable claim, and this is the test. If within-cohort fit is poor the earlier study's result does
not replicate here and the reconciliation is unnecessary; if fit is good and the loading vectors are
also interchangeable, the reconciliation fails and the covariance divergence needs another account.

Design. One factor, maximum likelihood, on the eight PHQ-8 items, fitted separately within each of
the fourteen cohorts entering the covariance contrasts and within the NHANES adult sample under the
same specification. Two quantities per fit:

  SRMR                standardised root mean square residual between observed and model-implied
                      item correlations. Below .08 is the conventional threshold for good fit.
  Tucker's phi        congruence of the loading vector against a reference. Above .95 is the
                      conventional threshold for treating two loading vectors as equivalent, .85 to
                      .95 as fair.

Items are ordinal with four categories, so Pearson correlations attenuate relative to polychoric.
Attenuation is a property of the scale and applies to every cohort alike, so it biases the fit
indices without favouring any comparison; the congruences, which are the inferential quantity here,
are ratios of loading vectors computed the same way throughout.

Emits analysis/within_cohort_factor.csv and analysis/factor_congruence.csv.
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(BASE, "data", "nhanes_raw")
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
NH_ITEMS = [f"DPQ0{i}0" for i in range(1, 9)]
CYCLES = list("DEFGHIJ")


def fit_one_factor(X):
    """One-factor ML solution; returns loadings, SRMR, and share of item variance explained."""
    X = np.asarray(X, float)
    X = X[:, X.std(axis=0) > 0] if (X.std(axis=0) == 0).any() else X
    if X.shape[1] < 8:
        return None
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    fa = FactorAnalysis(n_components=1, random_state=0).fit(Z)
    lam = fa.components_[0]
    # Orient the factor positively; sign is arbitrary in a one-factor solution and an unoriented
    # vector makes congruence against a reference meaningless.
    if lam.sum() < 0:
        lam = -lam
    R = np.corrcoef(Z, rowvar=False)
    implied = np.outer(lam, lam)
    np.fill_diagonal(implied, 1.0)
    iu = np.triu_indices_from(R, k=1)
    srmr = float(np.sqrt(np.mean((R[iu] - implied[iu]) ** 2)))
    return lam, srmr, float(np.mean(lam ** 2))


def congruence(a, b):
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


# ---- simulated cohorts -----------------------------------------------------------------------
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
for c in ["race", "gender", "ses", "relationship"]:
    m[c] = m[c].astype(str).str.strip().str.strip('"')
m = m[m[ITEMS].notna().all(axis=1)]

COHORTS = ([("race", r) for r in sorted(m.race.unique())]
           + [("gender", g) for g in sorted(m.gender.unique())]
           + [("ses", s) for s in sorted(m.ses.unique())])

# ---- NHANES reference ------------------------------------------------------------------------
frames = []
for c in CYCLES:
    demo = pd.read_sas(os.path.join(RAW, f"DEMO_{c}.xpt"))[["SEQN", "RIDAGEYR", "WTMEC2YR"]]
    frames.append(demo.merge(pd.read_sas(os.path.join(RAW, f"DPQ_{c}.xpt")), on="SEQN"))
d = pd.concat(frames, ignore_index=True)
d[NH_ITEMS] = d[NH_ITEMS].where(d[NH_ITEMS] <= 3)
d = d[(d.RIDAGEYR >= 18) & d[NH_ITEMS].notna().all(axis=1) & (d.WTMEC2YR > 0)]
ref_lam, ref_srmr, ref_var = fit_one_factor(d[NH_ITEMS].to_numpy())

rows = [dict(source="NHANES adults", dimension="population", cohort="All adults 18+", n=len(d),
             srmr=round(ref_srmr, 4), good_fit=ref_srmr < 0.08,
             mean_loading=round(float(ref_lam.mean()), 4),
             pct_variance=round(ref_var * 100, 2), phi_vs_nhanes=1.0)]

lams = {}
for dim, lev in COHORTS:
    g = m[m[dim] == lev]
    out = fit_one_factor(g[ITEMS].to_numpy())
    if out is None:
        continue
    lam, srmr, var = out
    lams[lev] = lam
    rows.append(dict(source="simulated", dimension=dim, cohort=lev, n=len(g),
                     srmr=round(srmr, 4), good_fit=srmr < 0.08,
                     mean_loading=round(float(lam.mean()), 4),
                     pct_variance=round(var * 100, 2),
                     phi_vs_nhanes=round(congruence(lam, ref_lam), 4)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "within_cohort_factor.csv"), index=False)
print(res.to_string(index=False))

sim = res[res.source == "simulated"]
print(f"\nwithin-cohort SRMR across {len(sim)} simulated cohorts: "
      f"{sim.srmr.min():.4f} to {sim.srmr.max():.4f}; "
      f"{int(sim.good_fit.sum())} of {len(sim)} below the .08 threshold")

# ---- pairwise congruence, the quantity the reconciliation turns on ---------------------------
pairs = []
names = list(lams)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        pairs.append(dict(cohort_a=a, cohort_b=b, phi=round(congruence(lams[a], lams[b]), 4)))
cg = pd.DataFrame(pairs).sort_values("phi")
cg["equivalent_at_.95"] = cg.phi >= 0.95
cg.to_csv(os.path.join(OUT, "factor_congruence.csv"), index=False)
print(f"\npairwise loading congruence over {len(cg)} cohort pairs: "
      f"{cg.phi.min():.4f} to {cg.phi.max():.4f}, median {cg.phi.median():.4f}")
print(f"  pairs below the .95 equivalence threshold: {int((~cg['equivalent_at_.95']).sum())}")
print("\nleast congruent pairs:")
print(cg.head(6).to_string(index=False))
print(f"\ncongruence against the NHANES loading vector: "
      f"{sim.phi_vs_nhanes.min():.4f} to {sim.phi_vs_nhanes.max():.4f}")


# ---- does the cohort structure hold inside each model? ---------------------------------------
# Section 11.1 puts model-by-factor variance at roughly twenty times the design-factor interaction
# variance per term, so a cohort matrix pooled over four models is pooling over the largest source
# of variation in the design. The congruence result has to survive being computed within a model.
print("\nwithin-model congruence, high-income against low-income (the extreme pair):")
wm = []
for mod, g in m.groupby("model"):
    lam = {}
    for lev in sorted(g.ses.unique()):
        out = fit_one_factor(g[g.ses == lev][ITEMS].to_numpy())
        if out:
            lam[lev] = out[0]
    hi = [k for k in lam if k.startswith("High")]
    lo = [k for k in lam if k.startswith("Low")]
    if hi and lo:
        wm.append(dict(model=mod, phi_high_vs_low=round(congruence(lam[hi[0]], lam[lo[0]]), 4)))
w = pd.DataFrame(wm).sort_values("phi_high_vs_low")
w.to_csv(os.path.join(OUT, "factor_congruence_within_model.csv"), index=False)
print(w.to_string(index=False))
print(f"  pooled value {congruence(lams['High (>$250k, Concierge)'], lams['Low (<$35k, Medicaid)']):.4f}; "
      f"within-model range {w.phi_high_vs_low.min():.4f} to {w.phi_high_vs_low.max():.4f}")
print(f"  below the .95 threshold in {int((w.phi_high_vs_low < 0.95).sum())} of {len(w)} models")
