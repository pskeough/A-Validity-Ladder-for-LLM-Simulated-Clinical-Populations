"""
What does the simulated PHQ-8 distribution look like next to the population one?

The paper argues that means, variances and covariances all diverge, and reports each of those
separately. A reader has so far had to assemble the shape from three scalar summaries. The shape
carries information none of them do: whether the simulated distribution is a shifted copy of the
population, a truncated one, or a differently-shaped object that happens to have a higher mean.

Frame discipline follows the rest of the paper: simulated draws are cisgender personas only, so the
model marginal matches the anchor's gender composition, and the anchor is the pooled weighted NHANES
adult distribution over cycles D-J.

The weighted population histogram uses MEC weights, so the comparison is against the distribution
NHANES estimates for the US adult population rather than the unweighted respondent sample.

Emits paper_v2/figures/fig7_distribution.{pdf,png} and analysis/distribution_shape.csv.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(BASE, "data", "nhanes_raw")
# Both manuscript trees, for the reason script 10 records: writing to paper_v2 alone and copying by
# hand is how a stale figure PDF ships. fig7 in paper_short was 7.02in and downscaled to 0.48x while
# paper_v2 carried a newer one.
FIG = os.path.join(BASE, "paper_v2", "figures")
FIG_SHORT = os.path.join(BASE, "paper_short", "figures")
CYCLES = ["D", "E", "F", "G", "H", "I", "J"]
ITEMS = [f"DPQ0{i}0" for i in range(1, 9)]
MAXV = 24

MODEL_C = {"openai/gpt-4o-mini": "#0072B2", "google/gemini-3-flash-preview": "#E69F00",
           "deepseek/deepseek-chat-v3": "#009E73", "z-ai/glm-4.7": "#CC79A7"}
MODEL_SHORT = {"openai/gpt-4o-mini": "GPT-4o-mini", "google/gemini-3-flash-preview": "Gemini-3-Flash",
               "deepseek/deepseek-chat-v3": "DeepSeek-V3", "z-ai/glm-4.7": "GLM-4.7"}
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "grid.linewidth": 0.4, "grid.alpha": 0.35})

# ---- population ------------------------------------------------------------------------------
frames = []
for c in CYCLES:
    demo = pd.read_sas(os.path.join(RAW, f"DEMO_{c}.xpt"))
    keep = [k for k in ["SEQN", "RIDAGEYR", "WTMEC2YR", "RIAGENDR"] if k in demo.columns]
    frames.append(demo[keep].merge(pd.read_sas(os.path.join(RAW, f"DPQ_{c}.xpt")), on="SEQN"))
d = pd.concat(frames, ignore_index=True)
d[ITEMS] = d[ITEMS].where(d[ITEMS] <= 3)
d = d[(d.RIDAGEYR >= 18) & d[ITEMS].notna().all(axis=1) & (d.WTMEC2YR > 0)].copy()
d["w"] = d.WTMEC2YR / len(CYCLES)
d["phq8"] = d[ITEMS].sum(axis=1)

# ---- simulated, cisgender frame --------------------------------------------------------------
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m["phq8_total"] = m.phq8_total.clip(0, MAXV)
m["gender"] = m.gender.astype(str).str.strip().str.strip('"')
sim = m[m.gender.str.contains("Cis")].copy()
print(f"population n = {len(d):,}; simulated cisgender generations = {len(sim):,}")


def wq(x, w, q):
    """Weighted quantile, linear on the cumulative weight scale."""
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cw, x))


def wecdf(x, w, grid):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    cw = np.cumsum(w) / w.sum()
    return np.interp(grid, x, cw, left=0.0, right=1.0)


grid = np.arange(0, MAXV + 1)
pop_cdf = wecdf(d.phq8, d.w, grid)

BANDS = [(0, 4, "0-4"), (5, 9, "5-9"), (10, 14, "10-14"), (15, 19, "15-19"), (20, 24, "20-24")]


def band_shares(x, w=None):
    w = np.ones(len(x)) if w is None else np.asarray(w)
    tot = w.sum()
    return {lab: float(w[(x >= lo) & (x <= hi)].sum() / tot * 100) for lo, hi, lab in BANDS}


pop_bands = band_shares(d.phq8.to_numpy(), d.w.to_numpy())

rows = [dict(source="NHANES 2005-2018 (weighted)", n=len(d),
             mean=round(float((d.w * d.phq8).sum() / d.w.sum()), 3),
             sd=round(float(np.sqrt((d.w * (d.phq8 - (d.w * d.phq8).sum() / d.w.sum()) ** 2).sum()
                                    / d.w.sum())), 3),
             p10=round(wq(d.phq8, d.w, .10), 2), median=round(wq(d.phq8, d.w, .50), 2),
             p90=round(wq(d.phq8, d.w, .90), 2),
             ks_vs_population=0.0, **{f"pct_{lab}": round(v, 2) for lab, v in pop_bands.items()})]

sim_cdfs = {}
for model, g in sim.groupby("model"):
    x = g.phq8_total.to_numpy()
    cdf = wecdf(x, np.ones(len(x)), grid)
    sim_cdfs[model] = cdf
    rows.append(dict(source=MODEL_SHORT.get(model, model), n=len(x),
                     mean=round(float(x.mean()), 3), sd=round(float(x.std(ddof=1)), 3),
                     p10=float(np.quantile(x, .10)), median=float(np.median(x)),
                     p90=float(np.quantile(x, .90)),
                     ks_vs_population=round(float(np.abs(cdf - pop_cdf).max()), 4),
                     **{f"pct_{lab}": round(v, 2) for lab, v in band_shares(x).items()}))

allx = sim.phq8_total.to_numpy()
pooled_cdf = wecdf(allx, np.ones(len(allx)), grid)
rows.append(dict(source="All models pooled", n=len(allx), mean=round(float(allx.mean()), 3),
                 sd=round(float(allx.std(ddof=1)), 3), p10=float(np.quantile(allx, .10)),
                 median=float(np.median(allx)), p90=float(np.quantile(allx, .90)),
                 ks_vs_population=round(float(np.abs(pooled_cdf - pop_cdf).max()), 4),
                 **{f"pct_{lab}": round(v, 2) for lab, v in band_shares(allx).items()}))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "distribution_shape.csv"), index=False)
print(res.to_string(index=False))

# A shift-only account: slide the population distribution up by the pooled mean residual and see
# whether it reproduces the simulated shape.
#
# The share of KS distance this removes is NOT reported in the manuscript, and this block exists to
# record why. Both distributions are supported on the integers 0 to 24, and the statistic turns on
# the fractional part of the shift: the observed 4.0056 returns exactly what an integer shift of 5
# returns (0.2655), while the integer shift of 4 returns 0.2018. The control below settles it. Data
# constructed as a pure location shift of the population, rounded back onto the same support so the
# true shape difference is zero, returns between 55% and 99% closed depending only on where the
# empirical mean lands between two lattice points. The manuscript rests the shape claim on the
# standard deviations and band shares instead, which no location shift can move.
shift = allx.mean() - float((d.w * d.phq8).sum() / d.w.sum())
shifted = np.clip(d.phq8.to_numpy() + shift, 0, MAXV)
shift_cdf = wecdf(shifted, d.w.to_numpy(), grid)
ks_shift = float(np.abs(pooled_cdf - shift_cdf).max())
ks_raw = float(np.abs(pooled_cdf - pop_cdf).max())
print(f"\nmean shift applied to the population: {shift:+.3f} points")
print(f"KS(simulated, population) = {ks_raw:.4f}; "
      f"KS(simulated, population shifted by the mean) = {ks_shift:.4f}")
print(f"a pure location shift closes {(1 - ks_shift / ks_raw) * 100:.1f}% of the KS distance, "
      f"leaving {ks_shift:.4f}  [NOT REPORTED: lattice-sensitive, see the note above]")
_rng = np.random.default_rng(3)
for _t in range(3):
    _i = _rng.choice(len(d), size=len(allx), replace=True, p=(d.w / d.w.sum()).to_numpy())
    _f = np.clip(np.round(d.phq8.to_numpy()[_i] + 4.0), 0, MAXV)
    _fc = wecdf(_f, np.ones(len(_f)), grid)
    _s = _f.mean() - float((d.w * d.phq8).sum() / d.w.sum())
    _ka = float(np.abs(_fc - wecdf(np.clip(d.phq8.to_numpy() + _s, 0, MAXV), d.w.to_numpy(), grid)).max())
    _kr = float(np.abs(_fc - pop_cdf).max())
    print(f"  control (true shape difference zero): shift {_s:.3f}, "
          f"reports {(1 - _ka / _kr) * 100:.1f}% closed")
pd.DataFrame([dict(mean_shift=round(shift, 4), ks_raw=round(ks_raw, 4),
                   ks_after_shift=round(ks_shift, 4),
                   pct_of_ks_closed_by_shift=round((1 - ks_shift / ks_raw) * 100, 1))]
             ).to_csv(os.path.join(OUT, "distribution_shift_test.csv"), index=False)

# ---- figure ----------------------------------------------------------------------------------
# Stacked. Side by side in one column each panel got about 1.4in, which left the density panel too
# narrow to carry a 0-24 x axis and four direct labels without the labels colliding.
# Side by side at the ML4H text width (6.0in), 2026-09-05: stacked, the figure took half an
# appendix page; horizontal, each panel keeps 2.9in and the pair fits under its paragraph.
fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6))

ax = axes[0]
centres = np.arange(0, MAXV + 1)
# np.isclose, not ==: the SAS export carries denormal zeros (4e-78) for PHQ-8 = 0, and an exact
# comparison dropped 11,955 zero-scoring respondents, the population mode, from the histogram.
popw = np.array([d.w[np.isclose(d.phq8, v)].sum() for v in centres]) / d.w.sum() * 100
ax.bar(centres, popw, width=0.86, color="0.78", edgecolor="none", zorder=1,
       label="NHANES adults (weighted)")
peak = float(popw.max())
for model, g in sim.groupby("model"):
    x = g.phq8_total.to_numpy()
    dens = np.array([(x == v).sum() for v in centres]) / len(x) * 100
    peak = max(peak, float(dens.max()))
    # Model names go in the legend (fidelity audit 2026-09-05: direct labels collided with each
    # other and with the NHANES legend once the population's zero bar was restored).
    ax.plot(centres, dens, color=MODEL_C.get(model, "0.3"), lw=1.2, zorder=3,
            label=MODEL_SHORT.get(model, model))
ax.set_xlabel("PHQ-8 total"); ax.set_ylabel("Percent of draws")
ax.set_xlim(-0.6, 24.6); ax.set_ylim(0, peak * 1.12); ax.grid(axis="y")
ax.legend(frameon=False, fontsize=7, loc="center right")  # clear of the DeepSeek peak at 8

ax = axes[1]
ax.plot(grid, pop_cdf * 100, color="0.35", lw=1.6, label="NHANES adults (weighted)")
ax.plot(grid, shift_cdf * 100, color="0.35", lw=1.0, ls=(0, (3, 2)),
        label=f"NHANES shifted {shift:+.1f} points")
for model, cdf in sim_cdfs.items():
    ax.plot(grid, cdf * 100, color=MODEL_C.get(model, "0.3"), lw=1.2)
ax.set_xlabel("PHQ-8 total"); ax.set_ylabel("Cumulative percent")
ax.set_xlim(0, 24); ax.set_ylim(0, 100); ax.grid(axis="y")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")

fig.set_layout_engine("constrained")
for _figdir in (FIG, FIG_SHORT):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(_figdir, f"fig7_distribution.{ext}"), dpi=220)
plt.close(fig)
print("\nwrote fig7_distribution")


# ---- alternative comparators -----------------------------------------------------------------
# A reviewer's objection to the shape reading also strengthens the level one. If this prompt elicits
# a modal patient rather than a mean one, the mean is the wrong population comparator. Both
# alternatives move the residual the same way, so the reported figure is the conservative choice.
_wmed = wq(d.phq8, d.w, 0.5)
_wmode = int(pd.Series(d.phq8.to_numpy()).groupby(d.phq8.to_numpy())
             .apply(lambda g: d.w.to_numpy()[g.index].sum()).idxmax())
_pm = float((d.w * d.phq8).sum() / d.w.sum())
pd.DataFrame([dict(comparator="mean", population=round(_pm, 3), simulated=round(allx.mean(), 3),
                   residual=round(allx.mean() - _pm, 3)),
              dict(comparator="median", population=round(_wmed, 3), simulated=float(np.median(allx)),
                   residual=round(allx.mean() - _wmed, 3)),
              dict(comparator="mode", population=float(_wmode),
                   simulated=float(pd.Series(allx).mode()[0]),
                   residual=round(allx.mean() - _wmode, 3))]
             ).to_csv(os.path.join(OUT, "alternative_comparators.csv"), index=False)
print(f"\npopulation mean {_pm:.2f}, median {_wmed:.0f}, mode {_wmode}; "
      f"simulated mean {allx.mean():.2f}, median {np.median(allx):.0f}, "
      f"mode {int(pd.Series(allx).mode()[0])}")
print(f"  residual against mean {allx.mean() - _pm:+.2f}, median {allx.mean() - _wmed:+.2f}, "
      f"mode {allx.mean() - _wmode:+.2f}")
