"""Body figure for level 2: the seven anchored contrasts, simulated against the population.

One row per contrast. The filled square is the pooled simulated gap over matched cisgender cell
pairs with its 95% interval (paired t); the small circles are the four per-model gaps; the open
diamond is the NHANES population gap with its design-based standard error. Read from
race_contrast_tests.csv (44) and level2_sex_ses_contrasts.csv (64), never from prose.

Authored at the jmlr[wcp] text width (6.0in) so width=\\linewidth is a no-op and 8.5pt type stays
8.5pt. Emits figures/ml4h_fig4_level2.pdf beside the other ML4H figures.
"""
import ast
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..")
A = lambda f: pd.read_csv(os.path.join(BASE, "analysis", f))  # noqa: E731
OUTS = [os.path.join(BASE, "paper_ml4h", "src", "figures"), os.path.join(BASE, "paper_ml4h", "build", "figures")]
W = 6.0
MODEL_C = {"openai/gpt-4o-mini": "#0072B2", "google/gemini-3-flash-preview": "#E69F00",
           "deepseek/deepseek-chat-v3": "#009E73", "z-ai/glm-4.7": "#CC79A7"}
MODEL_SHORT = {"openai/gpt-4o-mini": "GPT-4o-mini", "google/gemini-3-flash-preview": "Gemini-3-Flash",
               "deepseek/deepseek-chat-v3": "DeepSeek-V3", "z-ai/glm-4.7": "GLM-4.7"}
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "grid.linewidth": 0.4, "grid.alpha": 0.35})

race = A("race_contrast_tests.csv")
ext = A("level2_sex_ses_contrasts.csv")
# per-model verdicts from 66: the label at right is the pooled verdict with the number of models
# whose own verdict on the same rule agrees with it
pm = A("level2_permodel_equivalence.csv")


def share(name, pooled_verdict):
    sub = pm[(pm.contrast == name) & (pm.scope_type == "model")] if "scope_type" in pm.columns else pm[(pm.contrast == name) & pm.scope.str.contains("/") & ~pm.scope.str.contains(r"\|")]
    assert len(sub) == 4, f"{name}: {len(sub)} per-model rows"
    pooled = pm[(pm.contrast == name) & (pm.scope == "pooled")].verdict.iloc[0]
    assert pooled == pooled_verdict, f"{name}: figure says {pooled_verdict}, receipt says {pooled}"
    return int((sub.verdict == pooled).sum())
rows = [  # (label, frame, receipt contrast name, verdict)
    ("Women $-$ men", ext, "Women minus Men", "kept"),
    ("Asian $-$ White", race, "Asian minus White", "kept"),
    ("Black $-$ White", race, "Black minus White", "missing"),
    ("Hispanic $-$ White", race, "Hispanic minus White", "missing"),
    ("Middle $-$ high income", ext, "Middle minus High SES", "steepened"),
    ("Low $-$ middle income", ext, "Low minus Middle SES", "steepened"),
    ("Low $-$ high income", ext, "Low minus High SES", "steepened"),
]

fig, ax = plt.subplots(figsize=(W, 2.3))
n = len(rows)
for i, (label, frame, name, verdict) in enumerate(rows):
    y = n - 1 - i
    r = frame[frame.contrast == name].iloc[0]
    ax.axhline(y, color="0.92", lw=0.5, zorder=0)
    for mdl, v in ast.literal_eval(r.per_model).items():
        ax.plot([v], [y], "o", color=MODEL_C[mdl], ms=3.4, zorder=3, alpha=0.9)
    ax.plot([r.ci_lo, r.ci_hi], [y, y], color="0.25", lw=1.8, solid_capstyle="butt", zorder=4)
    ax.plot([r.simulated], [y], "s", color="white", ms=5.4, zorder=5)
    ax.plot([r.simulated], [y], "s", color="0.25", ms=3.6, zorder=6)
    ax.errorbar([r.population], [y], xerr=[[1.96 * r.pop_se]], fmt="D", color="#B4451F",
                mfc="white", ms=4.2, mew=1.0, elinewidth=0.9, capsize=0, zorder=5)
    ax.text(7.35, y, f"{verdict}, {share(name, verdict)} of 4", fontsize=8.0, va="center", ha="left", color="0.25")
ax.axvline(0, color="0.55", lw=0.7, zorder=1)
ax.text(7.35, n - 0.35, "Verdict (models agreeing)", fontsize=7.2, va="center", ha="left", color="0.25", style="italic")
ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
ax.set_xlim(-2.0, 7.3); ax.set_ylim(-0.6, n + 0.05)
ax.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
ax.set_xlabel("Gap in PHQ-8 points", fontsize=8)
ax.plot([], [], "s", color="0.25", ms=3.6, label="pooled simulated gap")
ax.plot([], [], "D", color="#B4451F", mfc="white", ms=4.2, label="NHANES gap")
for mdl in MODEL_C:
    ax.plot([], [], "o", color=MODEL_C[mdl], ms=2.6, label=MODEL_SHORT[mdl])
ax.legend(fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)
ax.tick_params(axis="x", labelsize=8)
fig.set_constrained_layout(True)
for out in OUTS:
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "ml4h_fig4_level2.pdf"))
    fig.savefig(os.path.join(out, "_ml4h_fig4_level2.png"), dpi=170)
print("wrote ml4h_fig4_level2.pdf to", OUTS)
