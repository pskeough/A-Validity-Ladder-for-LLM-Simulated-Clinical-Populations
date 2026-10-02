"""Body figures for level 2 after the 2 Oct 2026 rewrite.

Figure 2 (fig2_level2.pdf/png): verdict grid for the worked example, models x contrasts. Each
  tile is coloured by the article's three-way verdict (kept / not kept / unresolved) or marked
  not read, and prints the gap ratio and, for a contrast that is not kept, the direction.
  Reads figures/fig2_level2_data.csv, written by scripts/84_fig_level2_brm.py from
  analysis/brm/l2_verdicts.csv (78c). The interval chart that 84 draws now goes to the supplement.
Figure 3 (fig3_external_l2.pdf/png): share of level-2 verdicts for the worked example and the
  three external datasets, as stacked bars with counts. Reads
  paper_brm/external/results/l2_threeway_summary.csv (external/scripts/l2_threeway.py) and the
  same figure-2 data for the worked example.
Everything is laid out in inches on one full-figure axes, so nothing depends on tight bounding
boxes and the grid is centred on the page width.
"""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(BASE, "paper_brm", "manuscript", "figures")
DATA = os.path.join(FIG, "fig2_level2_data.csv")
EXT = os.path.join(BASE, "paper_brm", "external", "results", "l2_threeway_summary.csv")

FILL = {"kept": "#7cc48f", "not kept": "#ef9a7a", "unresolved": "#dcdcdc", "not read": "white"}
EDGE = "0.55"
MODELS = ["GPT-4o-mini", "Gemini-3-Flash", "DeepSeek-V3", "GLM-4.7"]
BLOCKS = [["Women − Men"],
          ["Low − High income", "Low − Middle income", "Middle − High income"],
          ["Black − White", "Hispanic − White", "Asian − White"]]
W = 6.5
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})


def label(v):
    if v.startswith("not read"):
        return "not read"
    if v.startswith("not kept"):
        return "not kept"
    return v


def legend(ax, x0, y, items, fs=7.5):
    """Centred row of swatches starting at x0 (inches)."""
    x = x0
    for name, text in items:
        ax.add_patch(Rectangle((x, y - 0.06), 0.22, 0.12, fc=FILL[name], ec=EDGE, lw=0.6))
        ax.text(x + 0.29, y, text, va="center", ha="left", fontsize=fs)
        x += 0.29 + 0.052 * len(text) + 0.25
    return x


def legend_width(items, fs=7.5):
    return sum(0.29 + 0.052 * len(t) + 0.25 for _, t in items) - 0.25


def fig2():
    d = pd.read_csv(DATA)
    d = d[~d.pooled_descriptive]
    lab_w, tile_w, tile_h, gap, bgap = 1.42, 1.22, 0.40, 0.05, 0.16
    grid_w = lab_w + 4 * tile_w + 3 * gap
    x0 = (W - grid_w) / 2
    n_rows = sum(len(b) for b in BLOCKS)
    head, foot = 0.30, 0.42
    H = head + n_rows * (tile_h + gap) + (len(BLOCKS) - 1) * bgap + foot
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    for j, m in enumerate(MODELS):
        ax.text(x0 + lab_w + j * (tile_w + gap) + tile_w / 2, H - head / 2, m, ha="center",
                va="center", fontsize=8.5, fontweight="bold")
    y = H - head
    kept_seen = False
    for bi, block in enumerate(BLOCKS):
        for c in block:
            y -= tile_h
            ax.text(x0 + lab_w - 0.1, y + tile_h / 2, c, ha="right", va="center", fontsize=8.5,
                    fontweight="bold")
            for j, m in enumerate(MODELS):
                r = d[(d.contrast_label == c) & (d.model_label == m)]
                assert len(r) == 1, (c, m)
                r = r.iloc[0]
                lab = label(r.verdict_printed)
                kept_seen |= lab == "kept"
                xl = x0 + lab_w + j * (tile_w + gap)
                ax.add_patch(Rectangle((xl, y), tile_w, tile_h, fc=FILL[lab], ec=EDGE, lw=0.6,
                                       ls="-" if lab != "not read" else (0, (2, 1.5))))
                xc, yc = xl + tile_w / 2, y + tile_h / 2
                if lab == "not read":
                    why = ("no population gap" if "no population" in r.verdict_printed
                           else "survey too imprecise")
                    ax.text(xc, yc, f"not read\n{why}", ha="center", va="center", fontsize=6.8,
                            color="0.4", style="italic", linespacing=1.15)
                else:
                    ax.text(xc, yc + 0.075, f"{r.ratio:.2f}×".replace("-", "−"), ha="center",
                            va="center", fontsize=9, fontweight="bold")
                    sub = r.verdict_printed.replace("not kept, ", "") if lab == "not kept" else lab
                    ax.text(xc, yc - 0.095, sub, ha="center", va="center", fontsize=6.8)
            y -= gap
        y -= bgap
    items = [("kept", "kept" if kept_seen else "kept (none here)"), ("not kept", "not kept"),
             ("unresolved", "unresolved"), ("not read", "not read")]
    lw = legend_width(items)
    gx = x0 + lab_w + (4 * tile_w + 3 * gap) / 2
    legend(ax, gx - lw / 2, foot / 2, items)
    fig.savefig(os.path.join(FIG, "fig2_level2.pdf"))
    fig.savefig(os.path.join(FIG, "fig2_level2.png"), dpi=200)
    plt.close(fig)
    print(f"fig2 {W} x {H:.2f} in; kept tiles: {kept_seen}")


def worked_counts():
    d = pd.read_csv(DATA)
    d = d[~d.pooled_descriptive]
    c = d.verdict_printed.map(label).value_counts()
    return {k: int(c.get(k, 0)) for k in FILL}


def fig3():
    s = pd.read_csv(EXT)

    def row(ds, run, fr):
        r = s[(s.dataset == ds) & (s.run == run) & (s.framing.fillna("") == fr)]
        assert len(r) == 1, (ds, run, fr)
        r = r.iloc[0]
        return {k: int(r[k]) for k in FILL}

    rows = [("Worked example: PHQ-8, four models", worked_counts()),
            ("Bisbee et al.: full profile", row("Bisbee", "rr1", "full")),
            ("Bisbee et al.: political profile", row("Bisbee", "rr1", "pol")),
            ("Bisbee et al.: demographic profile", row("Bisbee", "rr1", "demogs")),
            ("Argyle et al.: Study 3, main run", row("Argyle", "t0.7_main", "")),
            ("OpinionQA (Meister et al.)", row("OpinionQA", "all", ""))]
    lab_w, bar_w, bar_h, gap, nw = 2.2, 3.3, 0.27, 0.13, 0.9
    x0 = (W - (lab_w + bar_w + nw)) / 2
    head, foot = 0.12, 0.55
    H = head + len(rows) * (bar_h + gap) + foot
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    y = H - head
    xb = x0 + lab_w
    for name, cnt in rows:
        y -= bar_h
        n = sum(cnt.values())
        ax.text(xb - 0.1, y + bar_h / 2, name, ha="right", va="center", fontsize=8)
        x = xb
        for k in FILL:
            w = bar_w * cnt[k] / n
            if w <= 0:
                continue
            ax.add_patch(Rectangle((x, y), w, bar_h, fc=FILL[k], ec=EDGE, lw=0.5))
            if w > 0.22:
                ax.text(x + w / 2, y + bar_h / 2, str(cnt[k]), ha="center", va="center",
                        fontsize=7.5)
            x += w
        ax.text(xb + bar_w + 0.1, y + bar_h / 2, f"{cnt['kept']} kept of {n}", ha="left",
                va="center", fontsize=7.5, fontweight="bold" if cnt["kept"] else "normal",
                color="#1b5e36" if cnt["kept"] else "0.3")
        y -= gap
    # axis for shares
    ya = y + gap - 0.04
    ax.plot([xb, xb + bar_w], [ya, ya], color="0.3", lw=0.6)
    for t in (0, 25, 50, 75, 100):
        xt = xb + bar_w * t / 100
        ax.plot([xt, xt], [ya, ya - 0.04], color="0.3", lw=0.6)
        ax.text(xt, ya - 0.1, f"{t}%", ha="center", va="top", fontsize=7)
    items = [(k, k) for k in FILL]
    lw = legend_width(items)
    legend(ax, xb + bar_w / 2 - lw / 2, 0.14, items)
    fig.savefig(os.path.join(FIG, "fig3_external_l2.pdf"))
    fig.savefig(os.path.join(FIG, "fig3_external_l2.png"), dpi=200)
    plt.close(fig)
    pd.DataFrame([dict(row=nm, **c) for nm, c in rows]).to_csv(
        os.path.join(FIG, "fig3_external_l2_data.csv"), index=False)
    print(f"fig3 {W} x {H:.2f} in")
    for nm, c in rows:
        print(f"  {nm}: {c}")


if __name__ == "__main__":
    fig2()
    fig3()
