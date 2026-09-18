"""
Does the bias combine additively across the four design axes?

The design is a complete factorial, race (5) x gender identity (4) x socioeconomic status (3) x
relationship status (2), and the paper has so far reported only its marginals. A crossed design
supports a question the marginals cannot answer: whether the severity a model assigns to a persona
carrying several marked characteristics is the sum of what each characteristic costs on its own, or
more than the sum. Superadditivity would mean the models compound disadvantage; additivity would
mean each axis applies a fixed offset that mitigation can correct one axis at a time.

The unit is the design cell, model x cohort, matching the inference elsewhere in the paper. Cell
means are modelled rather than generation rows, so the 30 iterations inside a cell contribute their
mean and not 30 exchangeable observations.

Every comparison here is internal, cohort against cohort within the same model and decoding
settings, so no population anchor is required and the estimand restriction on variance fidelity does
not apply.

Emits analysis/factorial_additivity.csv, analysis/factorial_interactions.csv and
analysis/intersection_residuals.csv.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
FACTORS = ["race", "gender", "ses", "relationship"]

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
for c in FACTORS:
    m[c] = m[c].astype(str).str.strip().str.strip('"')

cells = (m.groupby(["model", "profile_id"] + FACTORS, as_index=False)
           .agg(phq8=("phq8_total", "mean"), sd=("phq8_total", "std"), n=("phq8_total", "size")))
print(f"{len(cells)} design cells, {cells.n.sum():,} generations, "
      f"{cells.groupby(FACTORS).ngroups} distinct cohorts")

# Model identity must enter interacted with every design factor, not as a main effect alone. The
# paper documents large model-by-factor differences elsewhere: transgender elevation runs +0.8 in
# GPT-4o-mini against +6.2 in Gemini-3-Flash, a 5.4-point model-by-gender interaction. Fitting
# model as a main effect only leaves all of that in the residual, and that residual is the
# denominator of both the F test on the design-factor interactions and the power calculation. An
# earlier version did exactly that, so its minimum detectable effect was measuring the
# misspecification rather than the design's resolution.
by_model = " + ".join(f"C(model):C({f})" for f in FACTORS)
main = ("phq8 ~ C(model) + C(race) + C(gender) + C(ses) + C(relationship) + " + by_model)
two_way = main + " + " + " + ".join(
    f"C({a}):C({b})" for i, a in enumerate(FACTORS) for b in FACTORS[i + 1:])

fit_m = smf.ols(main, data=cells).fit()
fit_2 = smf.ols(two_way, data=cells).fit()
lr = sm.stats.anova_lm(fit_m, fit_2)
f_stat, p_val = float(lr.F.iloc[1]), float(lr["Pr(>F)"].iloc[1])
df_extra = int(lr.df_diff.iloc[1])

print(f"\nmain effects only : R2 {fit_m.rsquared:.4f}  adj {fit_m.rsquared_adj:.4f}  "
      f"resid SD {np.sqrt(fit_m.mse_resid):.4f}")
print(f"plus all two-way  : R2 {fit_2.rsquared:.4f}  adj {fit_2.rsquared_adj:.4f}  "
      f"resid SD {np.sqrt(fit_2.mse_resid):.4f}")
print(f"nested F test on {df_extra} interaction terms: F = {f_stat:.3f}, p = {p_val:.4f}")
print(f"variance explained by all interactions jointly: "
      f"{(fit_2.rsquared - fit_m.rsquared) * 100:.2f} percentage points")

# ---- which two-way terms carry anything, corrected across the family -------------------------
rows = []
for i, a in enumerate(FACTORS):
    for b in FACTORS[i + 1:]:
        red = smf.ols(two_way.replace(f" + C({a}):C({b})", ""), data=cells).fit()
        an = sm.stats.anova_lm(red, fit_2)
        rows.append(dict(interaction=f"{a} x {b}", df=int(an.df_diff.iloc[1]),
                         F=round(float(an.F.iloc[1]), 3), p=float(an["Pr(>F)"].iloc[1])))
inter = pd.DataFrame(rows)
inter["q"] = multipletests(inter.p, method="fdr_bh")[1]
inter["significant"] = inter.q < 0.05
inter.p, inter.q = inter.p.round(4), inter.q.round(4)
inter.to_csv(os.path.join(OUT, "factorial_interactions.csv"), index=False)
print("\ntwo-way interaction terms, Benjamini-Hochberg over the six:")
print(inter.to_string(index=False))

pd.DataFrame([
    dict(model="main effects only", r2=round(fit_m.rsquared, 4), adj_r2=round(fit_m.rsquared_adj, 4),
         resid_sd=round(float(np.sqrt(fit_m.mse_resid)), 4), df_resid=int(fit_m.df_resid)),
    dict(model="plus all two-way", r2=round(fit_2.rsquared, 4), adj_r2=round(fit_2.rsquared_adj, 4),
         resid_sd=round(float(np.sqrt(fit_2.mse_resid)), 4), df_resid=int(fit_2.df_resid)),
    dict(model="nested F test", r2=np.nan, adj_r2=np.nan, resid_sd=np.nan, df_resid=df_extra),
]).assign(F=[np.nan, np.nan, round(f_stat, 3)], p=[np.nan, np.nan, round(p_val, 4)]
          ).to_csv(os.path.join(OUT, "factorial_additivity.csv"), index=False)

# ---- the intersections themselves: observed against what the marginals predict ---------------
# An additive account predicts a cohort's mean as the grand mean plus each factor's own offset.
# The residual from that prediction is what a marginal report cannot see.
gm = cells.phq8.mean()
off = {f: cells.groupby(f).phq8.mean() - gm for f in FACTORS}
coh = cells.groupby(FACTORS, as_index=False).phq8.mean()
coh["additive_pred"] = gm + sum(coh[f].map(off[f]) for f in FACTORS)
coh["excess"] = coh.phq8 - coh.additive_pred
coh = coh.sort_values("excess", ascending=False)
coh.round(4).to_csv(os.path.join(OUT, "intersection_residuals.csv"), index=False)

print(f"\ndeparture from additivity across {len(coh)} cohorts: "
      f"SD {coh.excess.std(ddof=1):.3f} points, range {coh.excess.min():+.2f} to {coh.excess.max():+.2f}")
print(f"  against a within-cell SD of {cells.sd.mean():.3f} and a marginal spread of "
      f"{cells.groupby('ses').phq8.mean().max() - cells.groupby('ses').phq8.mean().min():.2f} on SES")
print("\nmost superadditive cohorts:")
print(coh.head(4)[FACTORS + ["phq8", "additive_pred", "excess"]].round(3).to_string(index=False))
print("\nmost subadditive cohorts:")
print(coh.tail(3)[FACTORS + ["phq8", "additive_pred", "excess"]].round(3).to_string(index=False))

# ---- what size of interaction would this design have caught? ---------------------------------
# A null is only informative with a bound attached, so we ask what size of interaction the nested F
# test would reject at 80% power.
#
# The injection has to sit on a baseline that is additive. An earlier version added the synthetic
# effect to the OBSERVED cell means, which already carry the observed interaction and already reject
# the additive model at F = 4.24, p = 4.8e-13. Power was therefore 1.00 before any injection, the
# loop broke at its first step, and the number it printed was the first grid point rather than a
# detection floor. The baseline here is the main-effects fit plus resampled residuals, which
# reproduces the design's noise while carrying no design-factor interaction, and the calculation is
# checked at delta = 0: a correct baseline must reject at roughly the nominal 5%.
rng = np.random.default_rng(20260726)
mark = ((cells.gender.str.contains("Trans")).astype(int) *
        (cells.ses.str.contains("Low")).astype(int)).to_numpy()
base_fit = fit_m.fittedvalues.to_numpy()
resid = fit_m.resid.to_numpy()


def power_at(delta, reps=60):
    hits = 0
    for _ in range(reps):
        c2 = cells.copy()
        c2["phq8"] = base_fit + rng.choice(resid, len(cells), replace=True) + delta * mark
        a = sm.stats.anova_lm(smf.ols(main, data=c2).fit(), smf.ols(two_way, data=c2).fit())
        hits += float(a["Pr(>F)"].iloc[1]) < 0.05
    return hits / reps


false_positive = power_at(0.0, reps=120)
print(f"\nfalse positive rate of the nested F test on an additive baseline: "
      f"{false_positive * 100:.1f}% (nominal 5%)")

mde = None
for delta in np.arange(0.1, 3.01, 0.1):
    if power_at(delta) >= 0.80:
        mde = float(delta)
        break
# Express the floor as the variance share it corresponds to, which is the unit the text uses.
c2 = cells.copy()
c2["phq8"] = base_fit + rng.choice(resid, len(cells), replace=True) + (mde or 3.0) * mark
share = (smf.ols(two_way, data=c2).fit().rsquared - smf.ols(main, data=c2).fit().rsquared) * 100
print(f"minimum detectable interaction (transgender x low-SES, 80% power, alpha .05): "
      f"{mde if mde else '>3.0'} points, {share:.2f}% of cell-mean variance")
print(f"  against an observed interaction share of {(fit_2.rsquared - fit_m.rsquared) * 100:.2f}%")
print(f"  largest observed departure from additivity: {coh.excess.abs().max():.2f} points")
pd.DataFrame([dict(false_positive_rate_pct=round(false_positive * 100, 2),
                   mde_points=mde, mde_variance_share_pct=round(float(share), 3),
                   observed_share_pct=round((fit_2.rsquared - fit_m.rsquared) * 100, 3))]
             ).to_csv(os.path.join(OUT, "interaction_power.csv"), index=False)

# ---- does the transgender dispersion effect hold inside every racial stratum? ----------------
print("\ntransgender minus cisgender within-cell SD, by racial stratum:")
strat = []
for r, g in cells.groupby("race"):
    t = g[g.gender.str.contains("Trans")].sd.mean()
    c = g[g.gender.str.contains("Cis")].sd.mean()
    strat.append(dict(race=r, trans_sd=round(t, 4), cis_sd=round(c, 4), difference=round(t - c, 4)))
st = pd.DataFrame(strat).sort_values("difference", ascending=False)
print(st.to_string(index=False))
print(f"  positive in {int((st.difference > 0).sum())} of {len(st)} strata")
