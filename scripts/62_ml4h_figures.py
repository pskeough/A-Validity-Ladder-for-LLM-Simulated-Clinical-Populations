"""Figures for the ML4H 2026 submission, authored at the jmlr[wcp] text width.

Why this exists rather than a flag on 10_make_figures.py: the ML4H build is a single 433.62pt
column (measured, not assumed -- see paper_ml4h/build/probe.log), where the arXiv build is a
two-column acmart at 241.128bp. Six of the eight base figures are authored at the acmart column
and would be magnified 1.79x by width=\\linewidth here, which is how a probe build reached 12
pages. Authoring at final size means the LaTeX scale factor is 1.0 and 8.5pt type stays 8.5pt.

Figure 1 is new. The ML4H version leads on regeneration instability and the repository contains no
figure for it; the base paper reached that result on page 9 and carried it in prose.

All values are read from receipt CSVs. Never from prose.

Emits into paper_ml4h/src/figures/:
  ml4h_fig1_regeneration.pdf   NEW - cell-level flip and threshold-crossing distributions
  ml4h_fig2_residuals.pdf      re-authored at 6.0in from fig1_residuals
  ml4h_fig3_crossgroup.pdf     re-authored at 6.0in from fig8_crossgroup
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..")
A = lambda f: pd.read_csv(os.path.join(BASE, "analysis", f))  # noqa: E731
OUT = os.path.join(BASE, "paper_ml4h", "src", "figures")
os.makedirs(OUT, exist_ok=True)

# 433.62pt / 72.27 pt-per-inch = 6.0003in. Authoring here makes width=\linewidth a no-op.
W = 6.0

BLUE = "#0072B2"
MODEL_C = {
    "openai/gpt-4o-mini": "#0072B2",
    "google/gemini-3-flash-preview": "#E69F00",
    "deepseek/deepseek-chat-v3": "#009E73",
    "z-ai/glm-4.7": "#CC79A7",
}
MODEL_SHORT = {
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "google/gemini-3-flash-preview": "Gemini-3-Flash",
    "deepseek/deepseek-chat-v3": "DeepSeek-V3",
    "z-ai/glm-4.7": "GLM-4.7",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "grid.linewidth": 0.4, "grid.alpha": 0.35,
})


def save(fig, name):
    # No bbox_inches="tight": a tight bbox grows the canvas past figsize to fit labels, and LaTeX
    # then scales it back down, shrinking the type. Constrained layout shrinks the axes instead,
    # so the saved width equals figsize and the scale factor stays 1.
    fig.set_layout_engine("constrained")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=220)
    plt.close(fig)
    print("wrote", name)


def fig1_regeneration():
    """The promoted lead result, shown at the unit the inference is computed on.

    Each dot is one design cell -- one profile under one model, 30 iterations. Showing the 480
    cells rather than a bar of the pooled mean is the point: it makes the cell the visible unit,
    which is the same unit Section 3.4 uses everywhere, and it shows the between-model spread that
    a single pooled number hides.
    """
    cl = A("regeneration_cell_level.csv")
    summ = A("regeneration_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.55))
    panels = [
        ("p_band_flip", "flip", "Severity category changes", "P(two draws differ in band), %"),
        # Shortened: the longer label pushed its trailing "%" 3.3pt past the figure's own mediabox,
        # so the glyph rendered clipped on the body page.
        ("p_threshold_cross", "cross", "Treatment threshold crossed", "P(straddle PHQ-8 = 10), %"),
    ]
    # Same model order as Section 4.1 and the appendix tables (fidelity audit 2026-09-05: four
    # different orderings across figures and tables).
    models = ["openai/gpt-4o-mini", "google/gemini-3-flash-preview",
              "deepseek/deepseek-chat-v3", "z-ai/glm-4.7"]

    rng = np.random.default_rng(0)  # jitter only; no statistic depends on it
    for ax, (col, key, title, xlabel) in zip(axes, panels):
        sub_c = cl[cl.condition == "clinical"]
        for i, mdl in enumerate(models):
            y = len(models) - 1 - i
            vals = sub_c[sub_c.model == mdl][col].values * 100
            ax.scatter(vals, y + rng.uniform(-0.17, 0.17, len(vals)),
                       s=3.2, alpha=0.30, color=MODEL_C[mdl], linewidths=0, zorder=2)
            r = summ[(summ.condition == "clinical") & (summ.model == mdl)].iloc[0]
            m, lo, hi = r[f"{key}_pct"], r[f"{key}_ci_lo_pct"], r[f"{key}_ci_hi_pct"]
            ax.plot([lo, hi], [y, y], color=MODEL_C[mdl], lw=1.9, zorder=3,
                    solid_capstyle="butt")
            ax.plot([m], [y], "o", color="white", ms=5.2, zorder=4)
            ax.plot([m], [y], "o", color=MODEL_C[mdl], ms=3.4, zorder=5)

        pooled = summ[(summ.condition == "clinical") & (summ.model == "ALL")].iloc[0]
        ax.axvline(pooled[f"{key}_pct"], color="0.35", lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.text(pooled[f"{key}_pct"], len(models) - 0.42,
                f" pooled {pooled[f'{key}_pct']:.1f}%", fontsize=7.2, color="0.30",
                va="center", ha="left")

        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_SHORT[m] for m in models[::-1]], fontsize=8)
        ax.set_xlim(0, 68)
        ax.set_ylim(-0.6, len(models) - 0.05)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)

    save(fig, "ml4h_fig1_regeneration")


def fig2_residuals():
    """Severity residuals, cis-only primary with pooled sensitivity. Re-authored at 6.0in.

    Intervals come from the cell-level file, which is what Table 1 prints. The row-level file
    carries an interval 5.2 to 6.9x narrower because it treats the 28,800 generations as
    exchangeable, and drawing those made the arXiv figure contradict its own table.
    """
    cis = A("bias_residuals_CISONLY.csv")
    cell = A("residuals_cell_level.csv").set_index("group")
    cis = cis.set_index("group")
    cis["resid_ci_lo"] = cell.resid_ci_lo
    cis["resid_ci_hi"] = cell.resid_ci_hi
    assert cis.resid_ci_lo.notna().all(), "cell-level intervals missing for some group"
    cis = cis.reset_index()

    pool = A("bias_residuals_GOLD.csv")
    pool = pool[pool.residual.notna()]
    order = ["White", "Black", "Asian", "Hispanic", "Cisgender Man", "Cisgender Woman",
             "Low", "Middle", "High"]
    # Row labels match Table 2 ("Cis men", "Cis women"). The primary marker is a dark neutral
    # rather than #0072B2, which means GPT-4o-mini everywhere else in the paper.
    label = {"Low": "Low SES", "Middle": "Middle SES", "High": "High SES",
             "Cisgender Man": "Cis men", "Cisgender Woman": "Cis women"}
    PRIMARY = "0.15"

    fig, ax = plt.subplots(figsize=(W, 2.75))
    ys = np.arange(len(order))[::-1]
    ci = cis.set_index("group")
    po = pool.set_index("group")

    for y, g in zip(ys, order):
        if g not in ci.index:
            continue
        r = ci.loc[g]
        ax.plot([r.resid_ci_lo, r.resid_ci_hi], [y, y], color=PRIMARY, lw=1.9,
                solid_capstyle="butt", zorder=3)
        if g in po.index:
            # Drawn beneath the primary marker so the sensitivity diamond never hides it.
            ax.plot([po.loc[g].residual], [y], "D", color="0.6", ms=3.4, zorder=3.5)
        ax.plot([r.residual], [y], "o", color="white", ms=5.4, zorder=4)
        ax.plot([r.residual], [y], "o", color=PRIMARY, ms=3.6, zorder=5)

    ax.axvline(0, color="0.2", lw=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([label.get(g, g) for g in order], fontsize=8)
    # One extra row of headroom below High SES so the legend has a band of its own and does not
    # sit on the bottom row's interval (rev 2 review: legend overlapped the High SES bars).
    ax.set_ylim(-1.35, len(order) - 0.45)
    ax.set_xlabel("Simulated minus population PHQ-8 (points)", fontsize=8)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.plot([], [], "o", color=PRIMARY, ms=4, label="Cisgender personas (primary)")
    ax.plot([], [], "D", color="0.6", ms=3.4, label="All personas (sensitivity)")
    ax.legend(fontsize=8, loc="lower right", frameon=False, ncol=2,
              bbox_to_anchor=(1.0, -0.02), borderaxespad=0.2)
    save(fig, "ml4h_fig2_residuals")


def fig3_crossgroup():
    """Population disparities against simulated ones, per model. Re-authored at 6.0in.

    Drawn per model rather than pooled because the pooled contrast averages over a sign
    disagreement: two models run with the population gap and two against it, and a single pooled
    marker makes that read as a shared attenuation.
    """
    try:
        d = A("race_equivalence_tost.csv")
    except FileNotFoundError:
        print("skip ml4h_fig3_crossgroup: race_equivalence_tost.csv absent")
        return

    # The receipt carries one row per (contrast, scope, bound). The estimate and its interval are
    # constant across bound rows, so any one of them gives the geometry; take the row whose bound
    # equals the population disparity, which is the primary pre-specified bound.
    d = d[np.isclose(d.bound, d.population)]

    models = ["openai/gpt-4o-mini", "google/gemini-3-flash-preview",
              "deepseek/deepseek-chat-v3", "z-ai/glm-4.7"]
    contrasts = ["Black minus White", "Hispanic minus White"]
    # The body quotes the z interval (ci95z); draw the same one when the receipt carries it.
    lo_col, hi_col = ("ci95z_lo", "ci95z_hi") if "ci95z_lo" in d.columns else ("ci95_lo", "ci95_hi")

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), sharex=True)
    for ax, con in zip(axes, contrasts):
        sub = d[d.contrast == con]
        pop = float(sub.population.iloc[0])

        # Population disparity, and its mirror. Two models land near +pop and two near -pop, so
        # showing both makes the sign disagreement the visible feature rather than the average.
        ax.axvspan(-0.02, 0.02, color="0.88", zorder=0)
        ax.axvline(pop, color="#B4451F", lw=1.0, ls=(0, (3, 2)), zorder=2)
        # Label to the left of the line so it stays inside the axes (audit 2026-09-05: the
        # right-hand label ran past the text edge).
        ax.text(pop - 0.02, len(models) - 0.30, f"population {pop:.2f} ", fontsize=6.8,
                color="#B4451F", va="center", ha="right")

        for i, mdl in enumerate(models):
            y = len(models) - 1 - i
            r = sub[sub.scope == mdl]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r[lo_col], r[hi_col]], [y, y], color=MODEL_C[mdl], lw=1.9,
                    solid_capstyle="butt", zorder=3)
            ax.plot([r.mean_diff], [y], "o", color="white", ms=5.2, zorder=4)
            ax.plot([r.mean_diff], [y], "o", color=MODEL_C[mdl], ms=3.4, zorder=5)

        p = sub[sub.scope == "pooled"].iloc[0]
        y = -0.72
        ax.plot([p[lo_col], p[hi_col]], [y, y], color="0.25", lw=1.9,
                solid_capstyle="butt", zorder=3)
        ax.plot([p.mean_diff], [y], "s", color="white", ms=5.2, zorder=4)
        ax.plot([p.mean_diff], [y], "s", color="0.25", ms=3.4, zorder=5)

        ax.set_yticks(list(range(len(models))) + [y])
        ax.set_yticklabels([MODEL_SHORT[m] for m in models[::-1]] + ["pooled"], fontsize=8)
        ax.set_title(con.replace(" minus ", " $-$ "), fontsize=8.5, pad=4)
        ax.set_xlabel("Simulated gap (PHQ-8 points)", fontsize=8)
        ax.set_xlim(-0.95, 0.62)
        ax.set_ylim(-1.15, len(models) - 0.05)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)

    save(fig, "ml4h_fig3_crossgroup")


if __name__ == "__main__":
    fig1_regeneration()
    fig2_residuals()
    fig3_crossgroup()
