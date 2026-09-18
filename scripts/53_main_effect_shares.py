"""
The manuscript compares two INTERACTION blocks, the 30 model-by-factor terms against the 35
design-factor two-ways, and reports the ratio as "about twenty to one per term". Both main-effect
blocks are excluded from that comparison and neither is reported anywhere, which lets the section
read as a claim that model identity is the largest term in the design. It is not.

This reports the full ladder on exactly the unit 49_model_variance_loo.py and
27_factorial_interactions.py use (the design cell: model x cohort, generations averaged), so the
main-effect shares are directly comparable to the 23.4 and 1.31 already in the text.

Writes analysis/main_effect_shares.csv.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
OUT = os.path.join(BASE, "analysis")
FACTORS = ["race", "gender", "ses", "relationship"]

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
for c in FACTORS:
    m[c] = m[c].astype(str).str.strip().str.strip('"')

cells = (m.groupby(["model", "profile_id"] + FACTORS, as_index=False)
           .agg(phq8=("phq8_total", "mean")))
print("%d design cells over %d models, %d cohorts"
      % (len(cells), cells.model.nunique(), cells.groupby(FACTORS).ngroups))

DEMO = " + ".join("C(%s)" % f for f in FACTORS)
MBF = " + ".join("C(model):C(%s)" % f for f in FACTORS)
TWO = " + ".join("C(%s):C(%s)" % (a, b)
                 for i, a in enumerate(FACTORS) for b in FACTORS[i + 1:])

LADDER = [
    ("model identity alone",           "phq8 ~ C(model)"),
    ("demographic factors alone",      "phq8 ~ " + DEMO),
    ("all five main effects (M0)",     "phq8 ~ C(model) + " + DEMO),
    ("M0 + model-by-factor (M1)",      "phq8 ~ C(model) + " + DEMO + " + " + MBF),
    ("M1 + factor-by-factor (M2)",     "phq8 ~ C(model) + " + DEMO + " + " + MBF + " + " + TWO),
]

r2 = {}
rows = []
for label, f in LADDER:
    fit = smf.ols(f, data=cells).fit()
    r2[label] = fit.rsquared
    rows.append(dict(specification=label, r2=round(fit.rsquared, 6),
                     share_pct=round(100 * fit.rsquared, 3), df_model=int(fit.df_model)))

print()
print("%-30s %8s" % ("SPECIFICATION", "share"))
print("-" * 42)
for label, _ in LADDER:
    print("%-30s %7.1f%%" % (label, 100 * r2[label]))

mbf_pp = 100 * (r2["M0 + model-by-factor (M1)"] - r2["all five main effects (M0)"])
two_pp = 100 * (r2["M1 + factor-by-factor (M2)"] - r2["M0 + model-by-factor (M1)"])
model_pp = 100 * r2["model identity alone"]
demo_pp = 100 * r2["demographic factors alone"]

print()
print("model-by-factor block   %6.2f pp over 30 terms" % mbf_pp)
print("factor-by-factor block  %6.2f pp over 35 terms" % two_pp)
print("per-term ratio          %6.1fx   (manuscript says about twenty to one)" % ((mbf_pp / 30) / (two_pp / 35)))
print()
print("MAIN EFFECTS, excluded from both sides of that comparison:")
print("  model identity        %6.2f pp" % model_pp)
print("  demographic factors   %6.2f pp" % demo_pp)
print("  demographic : model   %6.0fx" % (demo_pp / model_pp))

rows.append(dict(specification="model-by-factor block (30 terms)", r2=np.nan,
                 share_pct=round(mbf_pp, 3), df_model=30))
rows.append(dict(specification="factor-by-factor block (35 terms)", r2=np.nan,
                 share_pct=round(two_pp, 3), df_model=35))

os.makedirs(OUT, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "main_effect_shares.csv"), index=False)
print("\nWritten -> %s" % os.path.join(OUT, "main_effect_shares.csv"))
