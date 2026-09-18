"""
Is the inflation additive or multiplicative, and does the steepened gradient survive either way?

The paper reports a socioeconomic gradient 2.4 times steeper in the simulated populations than in
the population anchors, and reads it as the models exaggerating a disparity they otherwise encode.
If the models instead apply a roughly constant MULTIPLIER to every group, a steepened absolute
gradient follows mechanically and the exaggeration claim has no content beyond the level error.

The two models are fitted on the nine benchmarked cohorts and compared on the dispersion of their
residuals, which is the only thing that distinguishes them here:

  additive        model = population + a          spread of (model - population)
  multiplicative  model = population x b          spread of log(model / population)

Also emits the regional contrast, because the panel was built two US-developed against two
China-developed as a control on development context rather than as a hypothesis, and a control is
only worth stating if it was checked.

Emits analysis/inflation_form.csv and analysis/regional_control.csv.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")

r = pd.read_csv(os.path.join(OUT, "residuals_cell_level.csv"))
r = r[["dimension", "group", "model_mean", "gt_mean", "residual"]].copy()
r["ratio"] = r.model_mean / r.gt_mean
r["logratio"] = np.log(r.ratio)

add_cv = r.residual.std(ddof=1) / r.residual.mean()
mul_cv = r.ratio.std(ddof=1) / r.ratio.mean()
print(f"nine benchmarked cohorts")
print(f"  additive residual : mean {r.residual.mean():.3f}  SD {r.residual.std(ddof=1):.3f}  CV {add_cv:.3f}")
print(f"  ratio             : mean {r.ratio.mean():.3f}  SD {r.ratio.std(ddof=1):.3f}  CV {mul_cv:.3f}")
print(f"  ratio range {r.ratio.min():.2f} to {r.ratio.max():.2f}")
print(f"  CV tightening factor {add_cv / mul_cv:.2f}x")

# CV is not a fair contest across scales. Fit both as one-parameter models and compare on the
# same footing: residual sum of squares on the observed (raw) scale, so neither is advantaged by
# being fitted in its own units.
a_hat = r.residual.mean()
b_hat = float(np.exp(r.logratio.mean()))
pred_add, pred_mul = r.gt_mean + a_hat, r.gt_mean * b_hat
rss_add = float(((r.model_mean - pred_add) ** 2).sum())
rss_mul = float(((r.model_mean - pred_mul) ** 2).sum())
n, k = len(r), 1
aic = lambda rss: n * np.log(rss / n) + 2 * k
print(f"\none-parameter fits, both scored on the raw PHQ-8 scale (n={n}):")
print(f"  additive       a = {a_hat:+.3f}   RSS {rss_add:.4f}   AIC {aic(rss_add):.2f}")
print(f"  multiplicative b = {b_hat:.3f}    RSS {rss_mul:.4f}   AIC {aic(rss_mul):.2f}")
better = "multiplicative" if rss_mul < rss_add else "additive"
print(f"  better fit: {better}  (RSS ratio {max(rss_add, rss_mul) / min(rss_add, rss_mul):.2f}x)")

# Does a constant multiplier reproduce the SES gradient the paper calls steepened?
ses = r[r.dimension == "ses"].set_index("group")
pop_grad = float(ses.loc["Low"].gt_mean - ses.loc["High"].gt_mean)
sim_grad = float(ses.loc["Low"].model_mean - ses.loc["High"].model_mean)
implied = pop_grad * b_hat
print(f"\nSES gradient (low minus high):")
print(f"  population {pop_grad:.3f} | simulated {sim_grad:.3f} | ratio {sim_grad / pop_grad:.3f}")
print(f"  a constant multiplier of {b_hat:.3f} predicts {implied:.3f}; observed {sim_grad:.3f}, "
      f"excess {sim_grad - implied:+.3f} points ({(sim_grad / implied - 1) * 100:+.1f}%)")
print(f"  -> the steepening is {'NOT ' if abs(sim_grad - implied) < 0.5 else ''}separable from the level error")

pd.DataFrame([
    dict(quantity="additive residual CV", value=round(add_cv, 4)),
    dict(quantity="ratio CV", value=round(mul_cv, 4)),
    dict(quantity="additive constant a", value=round(a_hat, 4)),
    dict(quantity="multiplicative constant b", value=round(b_hat, 4)),
    dict(quantity="RSS additive", value=round(rss_add, 4)),
    dict(quantity="RSS multiplicative", value=round(rss_mul, 4)),
    dict(quantity="ratio range low", value=round(float(r.ratio.min()), 4)),
    dict(quantity="ratio range high", value=round(float(r.ratio.max()), 4)),
    dict(quantity="SES gradient ratio observed", value=round(sim_grad / pop_grad, 4)),
    dict(quantity="SES gradient predicted by constant multiplier", value=round(implied, 4)),
    dict(quantity="SES gradient observed", value=round(sim_grad, 4)),
]).to_csv(os.path.join(OUT, "inflation_form.csv"), index=False)

# ---- regional control ------------------------------------------------------------------------
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
US = ["openai/gpt-4o-mini", "google/gemini-3-flash-preview"]
m["region"] = np.where(m.model.isin(US), "US-developed", "China-developed")
cis = m[~m.gender.astype(str).str.contains("rans", case=False, na=False)]
cell = cis.groupby(["region", "model", "profile_id"], as_index=False).phq8_total.mean()

rows = []
for reg, g in cell.groupby("region"):
    per = g.groupby("model").phq8_total.mean()
    rows.append(dict(region=reg, n_cells=len(g), mean=round(float(g.phq8_total.mean()), 4),
                     within_region_range=round(float(per.max() - per.min()), 4),
                     models="; ".join(f"{k.split('/')[-1]} {v:.2f}" for k, v in per.items())))
reg = pd.DataFrame(rows)
print("\n=== regional control (cisgender cells) ===")
print(reg.to_string(index=False))
a = cell[cell.region == "US-developed"].phq8_total
b = cell[cell.region == "China-developed"].phq8_total
t, p = stats.ttest_ind(a, b, equal_var=False)
between = abs(a.mean() - b.mean())
within = max(reg.within_region_range)
print(f"\n  between-region difference {between:.3f} points (t = {t:+.2f}, p = {p:.3f})")
print(f"  widest within-region spread {within:.3f} points")
print(f"  between/within = {between / within:.2f}  -> development context explains "
      f"{'less' if between < within else 'more'} than model identity within a region")
reg.to_csv(os.path.join(OUT, "regional_control.csv"), index=False)

# ---- do the models reproduce the ORDERING of population group differences? -------------------
print("\n=== ordering fidelity, racial contrasts against White ===")
race = r[r.dimension == "race"].set_index("group")
for g in ["Black", "Asian", "Hispanic"]:
    pg = float(race.loc[g].gt_mean - race.loc["White"].gt_mean)
    sg = float(race.loc[g].model_mean - race.loc["White"].model_mean)
    flag = "SIGN REVERSED" if np.sign(pg) != np.sign(sg) else "direction preserved"
    print(f"  {g:9s} vs White: population {pg:+.3f}  simulated {sg:+.3f}   {flag}")
