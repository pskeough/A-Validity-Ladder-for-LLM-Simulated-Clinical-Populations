"""Figure 1, the validity ladder as a general map (3 Oct 2026 refocus, Patrick's option A).

Instrument-agnostic by design: no survey, scale or corpus is named. The statistics and thresholds
are in Section 2 and Supplement S7. Writes paper_brm/manuscript/figures/fig1_ladder.pdf and .png.
Sketches that led here: paper_brm/manuscript/figures/fig1_options/."""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "paper_brm", "manuscript", "figures")
INK, MUTED, RULE = "#222222", "#555555", "#bbbbbb"
GATE_FILL, LEVEL_FILL = "#e9e9e9", "#f7f7f7"

# Listed bottom to top: the gate first.
RUNGS = [
    dict(tag="GATE", name="Regeneration stability",
         q="Is a persona's score stable enough to read?",
         claim="The average of k draws can be read as the model's answer for that persona.",
         fail="Answers change from draw to draw, so a single draw is mostly noise.",
         needs="Repeated draws of each persona. Any outcome."),
    dict(tag="LEVEL 1", name="Individual coherence",
         q="Could a real person give this answer, and do answers vary as much as people's?",
         claim="Single draws can serve as individual cases: vignettes, test items, synthetic respondents.",
         fail="Every persona answers alike, or answer patterns are too regular to be human.",
         needs="A multi-item scale and individual human answers on it."),
    dict(tag="LEVEL 2", name="Subgroup fidelity",
         q="Are the gaps between groups the right size and direction?",
         claim="Group differences can be read from the sample: an income gradient, a partisan gap.",
         fail="A gap is exaggerated, shrunk, erased or reversed.",
         needs="A human reference that measures the gap precisely, for groups the personas match."),
    dict(tag="LEVEL 3", name="Population calibration",
         q="Is each group's level right?",
         claim="A group's simulated level (mean, prevalence, share agreeing) can stand in for the real one.",
         fail="The whole sample is shifted, or a right mean hides a wrong prevalence.",
         needs="Human levels for the same groups, and weights to match the persona mix."),
    dict(tag="LEVEL 4", name="Structural fidelity",
         q="Do the items form the same scale as in people, in every group?",
         claim="The total can be scored as one score and compared across groups.",
         fail="The total stops measuring one thing, or measures it differently by group.",
         needs="A multi-item scale, individual human answers and enough personas per group."),
]


def wrap(s, width):
    return "\n".join(textwrap.wrap(s, width))


def main():
    W, H = 6.5, 5.15
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    cols = [("Rung and question", 0.42, 26), ("A pass lets you claim", 2.02, 27),
            ("A failure looks like", 3.62, 24), ("Your study needs", 5.02, 24)]
    top, row_h, gap = H - 0.36, 0.90, 0.06
    for title, x, _ in cols:
        ax.text(x, top + 0.12, title, fontsize=8.2, fontweight="bold", color=INK, va="bottom")
    ax.plot([0.36, W - 0.06], [top + 0.06] * 2, color=INK, lw=0.8)
    for i, r in enumerate(reversed(RUNGS)):
        y1 = top - gap - i * (row_h + gap); y0 = y1 - row_h
        fill = GATE_FILL if r["tag"] == "GATE" else LEVEL_FILL
        ax.add_patch(Rectangle((0.36, y0), W - 0.42, row_h, facecolor=fill, edgecolor=RULE, lw=0.6))
        x = cols[0][1]
        ax.text(x, y1 - 0.08, r["tag"], fontsize=6.6, color=MUTED, va="top")
        ax.text(x, y1 - 0.22, r["name"], fontsize=8, fontweight="bold", color=INK, va="top")
        ax.text(x, y1 - 0.40, wrap(r["q"], cols[0][2]), fontsize=7.2, color=INK, va="top", linespacing=1.25)
        for key, (_, cx, cw) in zip(("claim", "fail", "needs"), cols[1:]):
            ax.text(cx, y1 - 0.08, wrap(r[key], cw), fontsize=7.2, color=INK, va="top", linespacing=1.25)
    yb = top - gap - 4 * (row_h + gap) - row_h + 0.1
    yt = top - 0.15
    ax.add_patch(FancyArrowPatch((0.18, yb), (0.18, yt), arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1))
    ax.text(0.10, (yb + yt) / 2, "broader claims", rotation=90, fontsize=7, color=MUTED, ha="center", va="center")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_ladder.{ext}"), dpi=220)
    plt.close(fig)
    print("wrote", os.path.join(OUT, "fig1_ladder.pdf"))


if __name__ == "__main__":
    main()
