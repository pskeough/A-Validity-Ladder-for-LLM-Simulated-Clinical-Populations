"""The cross-group calibration figure for Section 4.2.

The cross-group calibration failure had no figure. A residual plot will not show it, because every
residual is positive and the failure lives in the differences between them. What shows it is the gap
between each racial group and White personas, plotted against the gap the population actually has.

An earlier version of this figure shaded the rows where the simulated and population signs
disagreed, which drew the eye to a reversal that script 44 then failed to establish: on Black and
Hispanic the simulated interval covers zero. The interval is therefore drawn, and the reader can see
that the population marker sits outside it while zero does not.

Reads analysis/race_contrast_tests.csv, so the figure and the prose cannot drift apart.
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..")
A = lambda f: pd.read_csv(os.path.join(BASE, "analysis", f))
FIG = os.path.join(BASE, "paper_short", "figures")

MODEL_C = {"openai/gpt-4o-mini": "#0072B2", "google/gemini-3-flash-preview": "#E69F00",
           "deepseek/deepseek-chat-v3": "#009E73", "z-ai/glm-4.7": "#CC79A7"}
MODEL_SHORT = {"openai/gpt-4o-mini": "GPT-4o-mini", "google/gemini-3-flash-preview": "Gemini-3-Flash",
               "deepseek/deepseek-chat-v3": "DeepSeek-V3", "z-ai/glm-4.7": "GLM-4.7"}
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "grid.linewidth": 0.4, "grid.alpha": 0.35})

t = A("race_contrast_tests.csv")
t["group"] = t.contrast.str.split(" ").str[0]
t = t.set_index("group")
pm = A("per_model_residuals.csv")
pm = pm[pm.dimension == "race"].pivot(index="model", columns="group", values="model_mean")

CONTRASTS = ["Black", "Hispanic", "Asian"]
pop = {g: float(t.loc[g].population) for g in CONTRASTS}
pooled = {g: float(t.loc[g].simulated) for g in CONTRASTS}
lo = {g: float(t.loc[g].ci_lo) for g in CONTRASTS}
hi = {g: float(t.loc[g].ci_hi) for g in CONTRASTS}
per_model = {m: {g: float(pm.loc[m, g] - pm.loc[m, "White"]) for g in CONTRASTS} for m in pm.index}

fig, ax = plt.subplots(figsize=(4.9, 2.45))
ys = np.arange(len(CONTRASTS))[::-1]

ax.axvline(0, color="0.3", lw=0.8, zorder=1)
for y, g in zip(ys, CONTRASTS):
    for m, c in MODEL_C.items():
        ax.scatter(per_model[m][g], y + 0.15, s=17, color=c, zorder=3)
    # The interval is the point of the figure: on Black and Hispanic it covers zero while the
    # population marker sits outside it, which is the difference between a flattened disparity and
    # a reversed one.
    ax.plot([lo[g], hi[g]], [y - 0.17, y - 0.17], color="0.35", lw=1.1, zorder=2,
            solid_capstyle="butt")
    for x in (lo[g], hi[g]):
        ax.plot([x, x], [y - 0.25, y - 0.09], color="0.35", lw=1.1, zorder=2)
    ax.scatter(pooled[g], y - 0.17, s=24, facecolors="white", edgecolors="0.35", lw=1.1, zorder=4)
    ax.scatter(pop[g], y - 0.17, s=36, marker="D", color="0.15", zorder=5)

ax.set_yticks(ys)
ax.set_yticklabels([f"{g} $-$ White" for g in CONTRASTS])
ax.set_ylim(-0.72, len(CONTRASTS) - 0.32)
ax.set_xlabel("PHQ-8 points relative to White personas")
ax.grid(axis="x")

for m, c in MODEL_C.items():
    ax.scatter([], [], s=17, color=c, label=MODEL_SHORT[m])
# No LaTeX escaping here: matplotlib is not in usetex mode, so a backslash prints as a backslash.
ax.scatter([], [], s=24, facecolors="white", edgecolors="0.35", lw=1.1, label="Pooled, 95% CI")
ax.scatter([], [], s=36, marker="D", color="0.15", label="Population (NHANES)")
ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, -0.29), columnspacing=1.1, handletextpad=0.3)

os.makedirs(FIG, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig8_crossgroup.{ext}"), dpi=220, bbox_inches="tight")
plt.close(fig)

for g in CONTRASTS:
    covers = lo[g] <= 0 <= hi[g]
    print(f"  {g:9s} population {pop[g]:+.3f}   simulated {pooled[g]:+.3f} "
          f"[{lo[g]:+.3f}, {hi[g]:+.3f}]   {'covers zero' if covers else 'excludes zero'}")
print("\nwrote fig8_crossgroup")
