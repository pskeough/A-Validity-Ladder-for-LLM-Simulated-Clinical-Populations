"""
Reviewer challenge: the dominant-variance claim rests on n=4 fixed models, and one of them
(z-ai/glm-4.7) is confounded three ways. It alone emitted reasoning tokens, it alone was served
by several upstream providers, and it had a ~21% API failure rate with 62 rows completed by hand.
It also carries the largest value on most axes and 14 of the 16 severe generations. If the
"twenty to one" ratio is a GLM artefact, the sentence in Section 11.1 is not supportable.

The claim under test (main.tex, "Model identity dominates every design factor"):
    "The 30 model-by-factor terms account for roughly 23 percentage points of cell-mean variance
     against 1.31 for the 35 design-factor two-ways, about twenty to one per term."

The decomposition is the one 27_factorial_interactions.py fits, on the same unit (the design cell,
model x cohort, 30 generations averaged into one mean) and the same nested OLS ladder:

    M0  phq8 ~ C(model) + C(race) + C(gender) + C(ses) + C(relationship)
    M1  M0 + C(model):C(race) + C(model):C(gender) + C(model):C(ses) + C(model):C(relationship)
    M2  M1 + all six design-factor two-way interactions

The two published shares are the two INCREMENTAL R-squared steps of that ladder, in that order:
model-by-factor is R2(M1) - R2(M0), design-factor two-way is R2(M2) - R2(M1). That is a SEQUENTIAL
(Type I) decomposition with model-by-factor entered first, which the paper does not state. The
Type III (partial, each block dropped from the full M2) figures are reported alongside so the
reviewer can see whether the ordering is doing any work.

This script reproduces the published numbers with all four models (GATE), then refits with GLM-4.7
excluded, then leaves each model out in turn. Per-term shares divide each block share by its own
degree-of-freedom count, which shrinks with the number of models (30 terms at n=4, 20 at n=3), so
the per-term ratio is reported next to the raw block shares rather than instead of them.

Writes analysis/model_variance_loo.csv. Does not modify any published receipt.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
OUT = os.path.join(BASE, "analysis")
FACTORS = ["race", "gender", "ses", "relationship"]

# published values from analysis/factorial_additivity.csv and interaction_power.csv
PUB_R2_MAIN = 0.9516      # R2 of M1, "main effects only" row of factorial_additivity.csv
PUB_R2_TWO = 0.9647       # R2 of M2, "plus all two-way" row
PUB_DFX_PP = 1.31         # observed_share_pct, interaction_power.csv
PUB_F = 4.24              # nested F on the 35 design-factor two-ways
PUB_MBF_PP = 23.0         # "roughly 23 percentage points", prose only, no receipt
PUB_RATIO = 20.0          # "about twenty to one per term", prose only, no receipt

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
for c in FACTORS:
    m[c] = m[c].astype(str).str.strip().str.strip('"')

cells_all = (m.groupby(["model", "profile_id"] + FACTORS, as_index=False)
               .agg(phq8=("phq8_total", "mean"), n=("phq8_total", "size")))
MODELS = sorted(cells_all.model.unique())
print("%d design cells over %d models, %s generations, %d distinct cohorts"
      % (len(cells_all), len(MODELS), format(cells_all.n.sum(), ","),
         cells_all.groupby(FACTORS).ngroups))
for mod in MODELS:
    print("   %-32s %3d cells" % (mod, int((cells_all.model == mod).sum())))

BY_MODEL = " + ".join("C(model):C(%s)" % f for f in FACTORS)
DESIGN_2W = " + ".join("C(%s):C(%s)" % (a, b)
                       for i, a in enumerate(FACTORS) for b in FACTORS[i + 1:])
F_M0 = "phq8 ~ C(model) + C(race) + C(gender) + C(ses) + C(relationship)"
F_M1 = F_M0 + " + " + BY_MODEL
F_M2 = F_M1 + " + " + DESIGN_2W
F_NO_MBF = F_M0 + " + " + DESIGN_2W   # M2 stripped of the model-by-factor block


def decompose(cells, label):
    """Sequential (Type I) block shares on the M0 -> M1 -> M2 ladder, plus Type III partials."""
    f0 = smf.ols(F_M0, data=cells).fit()
    f1 = smf.ols(F_M1, data=cells).fit()
    f2 = smf.ols(F_M2, data=cells).fit()
    fn = smf.ols(F_NO_MBF, data=cells).fit()

    df_mbf = int(f0.df_resid - f1.df_resid)
    df_dfx = int(f1.df_resid - f2.df_resid)

    seq_mbf = (f1.rsquared - f0.rsquared) * 100.0
    seq_dfx = (f2.rsquared - f1.rsquared) * 100.0
    t3_mbf = (f2.rsquared - fn.rsquared) * 100.0
    t3_dfx = seq_dfx                     # dropping the block from M2 is exactly M1

    anv = sm.stats.anova_lm(f1, f2)
    f_stat = float(anv.F.iloc[1])
    p_val = float(anv["Pr(>F)"].iloc[1])

    per_mbf = seq_mbf / df_mbf
    per_dfx = seq_dfx / df_dfx
    per_mbf3 = t3_mbf / df_mbf
    return dict(
        variant=label,
        n_models=int(cells.model.nunique()),
        n_cells=int(len(cells)),
        df_mbf_terms=df_mbf,
        df_design2w_terms=df_dfx,
        r2_M0=round(f0.rsquared, 6),
        r2_M1_main=round(f1.rsquared, 6),
        r2_M2_twoway=round(f2.rsquared, 6),
        seq_mbf_pp=round(seq_mbf, 4),
        seq_design2w_pp=round(seq_dfx, 4),
        t3_mbf_pp=round(t3_mbf, 4),
        t3_design2w_pp=round(t3_dfx, 4),
        per_term_mbf_pp=round(per_mbf, 5),
        per_term_design2w_pp=round(per_dfx, 5),
        ratio_seq=round(per_mbf / per_dfx, 3) if per_dfx else np.nan,
        ratio_t3=round(per_mbf3 / per_dfx, 3) if per_dfx else np.nan,
        block_ratio_seq=round(seq_mbf / seq_dfx, 3) if seq_dfx else np.nan,
        nested_F=round(f_stat, 3),
        nested_p=float(p_val),
        df_resid_M2=int(f2.df_resid),
    )


rows = [decompose(cells_all, "ALL FOUR (published)")]
rows.append(decompose(cells_all[cells_all.model != "z-ai/glm-4.7"], "DROP GLM-4.7 (n=3)"))
for mod in MODELS:
    rows.append(decompose(cells_all[cells_all.model != mod], "leave out %s" % mod))

res = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
res.to_csv(os.path.join(OUT, "model_variance_loo.csv"), index=False)

# ------------------------------------------------------------------ gate
pub = res.iloc[0]
checks = [
    ("R2 of M1 (main + model-by-factor) == 0.9516", round(float(pub.r2_M1_main), 4) == PUB_R2_MAIN),
    ("R2 of M2 (plus design two-ways) == 0.9647", round(float(pub.r2_M2_twoway), 4) == PUB_R2_TWO),
    ("design-factor two-way share == 1.31 pp", round(float(pub.seq_design2w_pp), 2) == PUB_DFX_PP),
    ("nested F on 35 terms == 4.24", round(float(pub.nested_F), 2) == PUB_F),
    ("residual df of M2 == 401", int(pub.df_resid_M2) == 401),
    ("model-by-factor block has 30 terms", int(pub.df_mbf_terms) == 30),
    ("design two-way block has 35 terms", int(pub.df_design2w_terms) == 35),
    ("model-by-factor share is roughly 23 pp", abs(float(pub.seq_mbf_pp) - PUB_MBF_PP) < 1.0),
    ("per-term ratio is roughly 20 to 1", abs(float(pub.ratio_seq) - PUB_RATIO) < 2.0),
]
print("\n" + "=" * 96)
print("GATE: reproduce the published decomposition with all four models")
print("=" * 96)
for name, ok in checks:
    print("  %-48s %s" % (name, "PASS" if ok else "FAIL"))
gate_ok = all(ok for _, ok in checks)
print("  %-48s %s" % ("OVERALL", "PASS" if gate_ok else "FAIL"))
if not gate_ok:
    print("\n  *** GATE FAILED. Data handling does not match the published receipts.")
    print("  *** Every number below this line is UNTRUSTWORTHY. Do not report it.")

print("\nSums of squares: SEQUENTIAL (Type I) on the M0 -> M1 -> M2 ladder, model-by-factor")
print("entered before the design-factor two-ways. This is what the paper reports and does not say.")
print("Type III (each block dropped from the full M2) is carried in the t3_* columns.")

# ------------------------------------------------------------------ table
pd.set_option("display.width", 200)
show = res[["variant", "n_models", "n_cells", "df_mbf_terms", "seq_mbf_pp", "seq_design2w_pp",
            "per_term_mbf_pp", "per_term_design2w_pp", "ratio_seq", "ratio_t3"]].copy()
show.columns = ["variant", "k", "cells", "mbf_terms", "mbf_pp", "dfx_pp",
                "mbf/term", "dfx/term", "RATIO", "ratio_T3"]
print("\n" + "=" * 96)
print("BLOCK SHARES OF CELL-MEAN VARIANCE, ALL FOUR MODELS AND EVERY LEAVE-ONE-OUT")
print("=" * 96)
print(show.to_string(index=False))
print("\nmbf_pp  = incremental R2 of the model-by-factor block, percentage points")
print("dfx_pp  = incremental R2 of the 35 design-factor two-ways, percentage points")
print("RATIO   = (mbf_pp / mbf_terms) / (dfx_pp / 35), the paper's 'per term' quantity")

# ------------------------------------------------------------------ verdict
noglm = res[res.variant.str.startswith("DROP GLM")].iloc[0]
loo = res[res.variant.str.startswith("leave out")]
print("\n" + "=" * 96)
print("VERDICT")
print("=" * 96)
print("published (n=4)      : %.2f pp / 30 terms  vs  %.2f pp / 35 terms  ->  %.1fx per term"
      % (pub.seq_mbf_pp, pub.seq_design2w_pp, pub.ratio_seq))
print("GLM-4.7 removed (n=3): %.2f pp / %d terms  vs  %.2f pp / 35 terms  ->  %.1fx per term"
      % (noglm.seq_mbf_pp, noglm.df_mbf_terms, noglm.seq_design2w_pp, noglm.ratio_seq))
print("leave-one-out range  : %.1fx to %.1fx  (min dropped: %s)"
      % (loo.ratio_seq.min(), loo.ratio_seq.max(),
         loo.loc[loo.ratio_seq.idxmin(), "variant"].replace("leave out ", "")))
worst = float(loo.ratio_seq.min())
print("\nlowest per-term ratio over any single-model deletion: %.1fx" % worst)
if worst >= 10.0:
    print("  -> the 'roughly twenty to one' claim SURVIVES every leave-one-out at the same order.")
elif worst >= 5.0:
    print("  -> the claim WEAKENS but stays above 5x. The stated multiple needs softening.")
else:
    print("  -> the claim COLLAPSES below 5x on at least one deletion. AS WRITTEN IT IS NOT")
    print("     SUPPORTABLE and the sentence must be rewritten or dropped.")

print("\nWritten -> %s" % os.path.join(OUT, "model_variance_loo.csv"))
