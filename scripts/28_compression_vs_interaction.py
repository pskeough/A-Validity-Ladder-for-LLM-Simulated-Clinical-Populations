"""
Are the interactions of Section 11.1 substantive, or an artefact of the bounded scale?

Section 11.1 fits an untransformed linear model to cell means on a 0-24 instrument and reads
departures from additivity as interaction. Three of the reported departures move toward the centre
of the observed range, and their size tracks distance from centre, which is what compression toward
a central tendency produces. This paper documents that compression twice elsewhere: 55.3% of
simulated cisgender outputs sit in the 5-9 band, and output-to-population SD ratios run 0.46 and
0.53 in two models. Section 3.5 states that the relation is concave on this bounded scale and
declines a linear residualization on those grounds, so the same objection applies here.

An earlier version of this script regressed each cohort's departure from the additive prediction on
the prediction itself and read a reliably negative slope as compression. That diagnostic is
degenerate. On a balanced factorial the grand mean plus marginal offsets is exactly the OLS
main-effects fit, so the departures are OLS residuals and are orthogonal to the fitted values by
construction. The regression returns slope zero, r-squared zero and p = 1 whatever the data contain.
It is checked below on synthetic grids carrying compression of known size, where it returns machine
zero at every level, and its output is reported as an identity rather than as evidence.

Two tests replace it, each of which would settle the question differently:

  induced pattern   compression is a monotone map, not a slope. Estimate one from the data by
                    fitting the additive model on the unbounded logit scale and pushing its
                    predictions back through the inverse link. The difference between that and the
                    additive prediction on the raw scale is the departure pattern a bounded scale
                    induces on its own, with no interaction present. Correlating it with the
                    observed departures gives the share of the observed pattern compression
                    accounts for, and unlike a slope it is not orthogonal to anything.
  bounded link      refit the whole additive-versus-interaction comparison on logit(total/24),
                    where a ceiling cannot manufacture curvature, and read the change in the
                    interaction variance as the bound compression places on the objection.

Emits analysis/compression_test.csv and analysis/compression_induced.csv.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
FACTORS = ["race", "gender", "ses", "relationship"]
MAXV = 24

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m["phq8_total"] = m.phq8_total.clip(0, MAXV)
for c in FACTORS:
    m[c] = m[c].astype(str).str.strip().str.strip('"')
cells = (m.groupby(["model", "profile_id"] + FACTORS, as_index=False)
           .agg(phq8=("phq8_total", "mean"), sd=("phq8_total", "std")))

gm = cells.phq8.mean()
off = {f: cells.groupby(f).phq8.mean() - gm for f in FACTORS}
coh = cells.groupby(FACTORS, as_index=False).phq8.mean()
coh["pred"] = gm + sum(coh[f].map(off[f]) for f in FACTORS)
coh["dep"] = coh.phq8 - coh.pred

# ---- 0. the degenerate diagnostic, retained as a demonstration that it is degenerate ----------
sl = stats.linregress(coh.pred, coh.dep)
print(f"discarded shrinkage-slope diagnostic: {sl.slope:+.2e} (r^2 {sl.rvalue ** 2:.2e})")
rng0 = np.random.default_rng(11)
for k in (0.0, 0.3, 0.6):
    t = coh.copy()
    t["phq8"] = coh.pred - k * (coh.pred - coh.pred.mean())     # explicit compression of size k
    g2 = t.phq8.mean()
    p2 = g2 + sum(t[f].map(t.groupby(f).phq8.mean() - g2) for f in FACTORS)
    print(f"  same diagnostic on a grid compressed by k = {k}: "
          f"{stats.linregress(p2, t.phq8 - p2).slope:+.8f}")
print("  the statistic is orthogonal to the fitted values by construction and carries no evidence")

# ---- 1. the pattern a bounded scale induces, against the pattern observed ---------------------
# Fit the additive model where no ceiling operates, then map back. What returns is the departure
# from raw-scale additivity that compression alone produces on this grid.
def to_logit(x):
    p = np.clip(np.asarray(x, float) / MAXV, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


cells["logit"] = to_logit(cells.phq8)
lgm = cells.logit.mean()
loff = {f: cells.groupby(f).logit.mean() - lgm for f in FACTORS}
coh["logit_pred"] = lgm + sum(coh[f].map(loff[f]) for f in FACTORS)
coh["induced"] = MAXV / (1 + np.exp(-coh.logit_pred)) - coh.pred
ind = stats.linregress(coh.induced, coh.dep)
print(f"\ninduced-versus-observed departure over {len(coh)} cohorts: r = {ind.rvalue:+.3f} "
      f"(p = {ind.pvalue:.2e}), so compression accounts for {ind.rvalue ** 2 * 100:.1f}% "
      f"of the departure variance")
print(f"  induced departures span {coh.induced.min():+.3f} to {coh.induced.max():+.3f} points "
      f"against observed {coh.dep.min():+.3f} to {coh.dep.max():+.3f}")
coh.round(4).to_csv(os.path.join(OUT, "compression_induced.csv"), index=False)

# ---- 2. the same model comparison on a link that respects the bound ---------------------------
by_model = " + ".join(f"C(model):C({f})" for f in FACTORS)
base = "C(model) + C(race) + C(gender) + C(ses) + C(relationship) + " + by_model
TERMS = [f"C({a}):C({b})" for i, a in enumerate(FACTORS) for b in FACTORS[i + 1:]]
tw = " + ".join(TERMS)

rows = []
for scale, dv in [("raw PHQ-8", "phq8"), ("logit(total/24)", "logit")]:
    f_m = smf.ols(f"{dv} ~ {base}", data=cells).fit()
    f_2 = smf.ols(f"{dv} ~ {base} + {tw}", data=cells).fit()
    an = sm.stats.anova_lm(f_m, f_2)
    # Drop each term by rebuilding the list. Removing it with a string replace fails silently on
    # the first term, which carries no leading " + ", and that returned a spurious all-null result.
    per = []
    for t in TERMS:
        red = smf.ols(f"{dv} ~ {base} + " + " + ".join(x for x in TERMS if x != t), data=cells).fit()
        per.append((t.replace("C(", "").replace(")", "").replace(":", " x "),
                    float(sm.stats.anova_lm(red, f_2)["Pr(>F)"].iloc[1])))
    q = multipletests([p for _, p in per], method="fdr_bh")[1]
    sig = [n for (n, _), qq in zip(per, q) if qq < 0.05]
    rows.append(dict(scale=scale, omnibus_F=round(float(an.F.iloc[1]), 3),
                     omnibus_p=float(an["Pr(>F)"].iloc[1]),
                     delta_r2_pct=round((f_2.rsquared - f_m.rsquared) * 100, 3),
                     n_significant=len(sig), significant="; ".join(sig) or "none"))
    print(f"\n{scale}: omnibus F = {an.F.iloc[1]:.2f}, p = {an['Pr(>F)'].iloc[1]:.2e}, "
          f"dR2 = {(f_2.rsquared - f_m.rsquared) * 100:.2f}pp")
    print(f"  surviving FDR: {sig if sig else 'none'}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "compression_test.csv"), index=False)

# ---- 3. what the scale change costs the interaction ------------------------------------------
raw_d, log_d = res.loc[res.scale == "raw PHQ-8", "delta_r2_pct"].iloc[0], \
    res.loc[res.scale == "logit(total/24)", "delta_r2_pct"].iloc[0]
removed = (raw_d - log_d) / raw_d * 100
print(f"\ninteraction variance: {raw_d:.2f}pp raw, {log_d:.2f}pp on the logit link")
print(f"  the scale change removes {removed:.0f}% of it and leaves {100 - removed:.0f}% standing")
pd.concat([res, pd.DataFrame([
    dict(scale="scale-change bound", omnibus_F=np.nan, omnibus_p=np.nan,
         delta_r2_pct=round(removed, 1), n_significant=np.nan,
         significant=f"compression accounts for at most {removed:.0f}% of the interaction variance; "
                     f"induced-pattern r = {ind.rvalue:+.3f} ({ind.rvalue ** 2 * 100:.1f}%)")])
           ]).to_csv(os.path.join(OUT, "compression_test.csv"), index=False)
