"""Figure 2, what each failure looks like in the worked example (3 Oct 2026, Patrick's option H).

One plain panel per rung that the four models fail or pass on:
  gate      draw standard deviation of a persona (one draw) and its standard error at 30 draws,
            against the gate's limit (0.25 NHANES SD) and recommended level (0.125 SD);
  level 1   share of draws more unusual or more regular than 95% of NHANES adults at the same total;
  levels 2-3 post-stratified mean PHQ-8 by income band, NHANES and each model;
  level 4   loadings of the eight items on one factor, NHANES and each model (narrative framing,
            where both Gemini-3-Flash and GLM-4.7 fail R2).
Sketch history: paper_brm/manuscript/figures/fig2_options/. Writes figures/fig2_ladder_panels.pdf/.png."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRM = os.path.join(BASE, "analysis", "brm")
OUT = os.path.join(BASE, "paper_brm", "manuscript", "figures")
INK, GREY, LIGHT, FAINT = "#111111", "#8a8a8a", "#d6d6d6", "#eeeeee"
MODELS = [("deepseek-chat-v3", "DeepSeek-V3"), ("gemini-3-flash-preview", "Gemini-3-Flash"),
          ("gpt-4o-mini", "GPT-4o-mini"), ("glm-4.7", "GLM-4.7")]
NAMES = [n for _, n in MODELS]
TOL = 0.98   # gate limit in PHQ-8 points, 0.25 NHANES SD (gate_dstudy.csv)
REC = 0.49   # recommended level, 0.125 NHANES SD
L4_FRAMING = "narrative"


def label_ends(ax, ends, x, gap, floor):
    ends.sort()
    last = floor - gap
    for yv, name in ends:
        yl = max(yv, last + gap)
        last = yl
        real = name == "Real adults"
        ax.text(x, yl, name, va="center", fontsize=6.8, color=INK if real else GREY,
                fontweight="bold" if real else "normal")


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=8, fontweight="bold")


def gate(ax):
    g = pd.read_csv(os.path.join(BRM, "gate_dstudy.csv"))
    g = g[g.outcome == "PHQ-8 total"]
    sds = [g[g.model == n].within_sd.max() for n in NAMES]
    yy = np.arange(4)[::-1]
    for y_, s in zip(yy, sds):
        ax.plot([0, s], [y_, y_], color=LIGHT, lw=1, zorder=1)
        ax.plot(s, y_, "o", color=GREY, ms=5, zorder=2)
        ax.plot(s / np.sqrt(30), y_, "o", color=INK, ms=5, zorder=3)
    ax.axvline(TOL, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.axvline(REC, color=GREY, lw=0.7, ls=(0, (3, 2)))
    ax.text(TOL + 0.03, 3.62, "limit", fontsize=6.2, color=INK, va="bottom")
    ax.text(REC - 0.03, 3.62, "recom-\nmended", fontsize=6.2, color=GREY, va="bottom", ha="right")
    ax.text(sds[0] + 0.08, 3, "one draw", fontsize=6.3, color=GREY, va="center")
    ax.text(sds[0] / np.sqrt(30) + 0.06, 2.62, "mean of 30", fontsize=6.3, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels(NAMES)
    ax.set_xlim(0, 2.6); ax.set_ylim(-0.5, 4.3)
    ax.set_xlabel("Points a persona's score moves between draws", fontsize=7)
    title(ax, "Gate: one draw is not readable,\nan average of 30 is")


def level1(ax):
    l1 = pd.read_csv(os.path.join(BRM, "l1_summary.csv"))
    yy = np.arange(4)[::-1]
    rows = [l1[(l1.model == n) & (l1.framing == "both")].iloc[0] for n in NAMES]
    for y_, r in zip(yy, rows):
        ax.plot([0, 6.5], [y_, y_], color=FAINT, lw=0.8, zorder=0)
        ax.plot(100 * r.pf_share_below_p5, y_ + 0.14, "o", color=INK, ms=4.5)
        ax.plot(100 * r.pf_share_above_p95, y_ - 0.14, "o", mfc="white", mec=INK, ms=4.5)
    ax.axvline(5, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.text(5.15, 3.62, "real people,\nboth kinds", fontsize=6.2, va="bottom")
    ax.text(100 * rows[0].pf_share_below_p5 + 0.25, 3.14, "unusual", fontsize=6.2, va="center")
    ax.text(100 * rows[0].pf_share_above_p95 + 0.25, 2.86, "overly regular", fontsize=6.2,
            va="center", color=GREY)
    ax.set_yticks(yy); ax.set_yticklabels(NAMES)
    ax.spines["left"].set_position(("outward", 6))
    ax.set_xlim(-0.15, 6.5); ax.set_ylim(-0.5, 4.3)
    ax.set_xlabel("Percent of answers", fontsize=7)
    title(ax, "Level 1: too few unusual\nanswer patterns")


def levels23(ax):
    h = pd.read_csv(os.path.join(BRM, "80d_headline.csv"))
    bands = ["Low", "Middle", "High"]
    x = np.arange(3)
    ends = []
    for key, name in MODELS:
        v = [h[(h.model == key) & (h.group == b)].sim.iloc[0] for b in bands]
        ax.plot(x, v, color=GREY, lw=1, marker="o", ms=3)
        ends.append((v[-1], name))
    ref = [h[(h.model == "pooled") & (h.group == b)].ref.iloc[0] for b in bands]
    ax.plot(x, ref, color=INK, lw=2.2, marker="o", ms=4)
    ends.append((ref[-1], "Real adults"))
    label_ends(ax, ends, 2.1, 0.95, 1.5)
    ax.set_xticks(x); ax.set_xticklabels(["Low\nincome", "Middle", "High"], fontsize=7)
    ax.set_xlim(-0.2, 3.3); ax.set_ylim(0, 12.5)
    ax.set_ylabel("Mean PHQ-8 total", fontsize=7)
    title(ax, "Levels 2 and 3: too high,\nand too steep across income")


def level4(ax):
    l4 = pd.read_csv(os.path.join(BRM, "l4_structure.csv"))
    items = ["Interest", "Mood", "Sleep", "Fatigue", "Appetite", "Self-worth", "Focus", "Moving"]
    cols = [f"load_{i}" for i in range(1, 9)]
    xi = np.arange(8)
    ends = []
    for _, name in MODELS:
        r = l4[(l4.model == name) & (l4.framing == L4_FRAMING)].iloc[0]
        v = r[cols].astype(float).values
        ax.plot(xi, v, color=GREY, lw=1, marker="o", ms=2.5)
        ends.append((v[-1], name))
    v = l4[l4.source == "NHANES"].iloc[0][cols].astype(float).values
    ax.plot(xi, v, color=INK, lw=2.2, marker="o", ms=3.5)
    ends.append((v[-1], "Real adults"))
    ax.axhline(0, color=LIGHT, lw=0.7)
    label_ends(ax, ends, 7.25, 0.17, -0.6)
    ax.set_xticks(xi)
    ax.set_xticklabels(items, fontsize=6.5, rotation=40, ha="right")
    ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.8, 1.1)
    ax.set_ylabel("Loading on one factor", fontsize=7)
    title(ax, "Level 4: items do not load\nas they do in people")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4),
                             gridspec_kw=dict(wspace=0.55, hspace=0.75))
    for ax in axes.flat:
        ax.tick_params(labelsize=7)
    gate(axes[0, 0]); level1(axes[0, 1]); levels23(axes[1, 0]); level4(axes[1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_ladder_panels.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, "fig2_ladder_panels.pdf"))


if __name__ == "__main__":
    main()
