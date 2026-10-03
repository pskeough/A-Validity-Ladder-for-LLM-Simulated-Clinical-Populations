"""Figure 3, level-2 gaps in the worked example and the three external datasets (3 Oct 2026,
Patrick's option L). One dot per group gap, one row per dataset, with the kept count at the end.

Verdict labels: paper_brm/external/results/l2_threeway_rows.csv (Bisbee full prompt, Argyle
t0.7_main, OpinionQA all), collapsed to the four labels of l2_threeway_summary.csv; worked example
from paper_brm/manuscript/figures/fig2_level2_data.csv (per-model rows). The rung-by-dataset
coverage is Table 4 in 06_external.tex. Sketch history: figures/fig3_options/.
Writes figures/fig3_l2_dots.pdf and .png, and the counts it drew to figures/fig3_l2_dots_data.csv."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 8})
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(BASE, "paper_brm", "external", "results")
OUT = os.path.join(BASE, "paper_brm", "manuscript", "figures")
INK, GREY, FAINT = "#111111", "#8a8a8a", "#e4e4e4"
ORDER = ["kept", "not kept", "unresolved", "not read"]
MARK = {"kept": dict(color=INK, ms=5.2),
        "not kept": dict(color=GREY, ms=4.4),
        "unresolved": dict(mfc="white", mec=GREY, color=GREY, ms=4.4, mew=0.8),
        "not read": dict(color=FAINT, ms=4.0)}


def collapse(label, lo=None, hi=None):
    """Map a printed or three-way label to kept / not kept / unresolved / not read."""
    if label.startswith("kept"):
        return "kept"
    if label.startswith("not kept") or label in ("missing or attenuated", "reversed or missing"):
        return "not kept"
    if label.startswith("not read") or label in ("no population gap", "reference too imprecise"):
        return "not read"
    if label == "undetermined" and lo is not None and not (hi >= 0.75 and lo <= 1.25):
        return "not kept"
    return "unresolved"


def external(dataset, run=None, framing=None):
    rows = pd.read_csv(os.path.join(EXT, "l2_threeway_rows.csv"))
    d = rows[rows.dataset == dataset]
    if run:
        d = d[d.run == run]
    if framing:
        d = d[d.framing == framing]
    return [collapse(l, lo, hi) for l, lo, hi in zip(d.label, d.ci_lo, d.ci_hi)]


def worked():
    f = pd.read_csv(os.path.join(OUT, "fig2_level2_data.csv"))
    f = f[~f.pooled_descriptive.astype(bool)]
    return [collapse(v) for v in f.verdict_printed]


def main():
    rows = [("Worked example", "4 models, PHQ-8", worked()),
            ("Bisbee et al.", "ChatGPT, thermometers", external("Bisbee", framing="full")),
            ("Argyle et al.", "GPT-3, Study 3", external("Argyle", run="t0.7_main")),
            ("OpinionQA", "5 models, Pew questions", external("OpinionQA"))]
    ncol = 40
    fig = plt.figure(figsize=(6.6, 3.6))
    ax = fig.add_axes([0.2, 0.08, 0.66, 0.8]); ax.axis("off")
    y0 = 0.0
    counts = []
    for name, sub, vs in rows:
        vs = sorted(vs, key=ORDER.index)
        counts.append(dict(row=name, total=len(vs), **{k: vs.count(k) for k in ORDER}))
        print(counts[-1])
        nrow = (len(vs) - 1) // ncol + 1
        for i, v in enumerate(vs):
            ax.plot(i % ncol, y0 - (i // ncol), "o", **MARK[v])
        mid = y0 - (nrow - 1) / 2
        ax.text(-1.5, mid + 0.18, name, ha="right", va="center", fontsize=8, fontweight="bold")
        ax.text(-1.5, mid - 0.42, sub, ha="right", va="center", fontsize=6.6, color=GREY)
        k = vs.count("kept")
        ax.text(ncol + 0.8, mid, f"{k} of {len(vs)}\nkept", ha="left", va="center", fontsize=8,
                fontweight="bold" if k else "normal")
        y0 -= nrow + 1.1
    for i, lab in enumerate(ORDER):
        ax.plot(i * 8, 1.3, "o", **MARK[lab])
        ax.text(i * 8 + 0.7, 1.3, lab, va="center", fontsize=7, color=INK if i == 0 else GREY)
    ax.set_xlim(-1, ncol + 4); ax.set_ylim(y0 + 0.6, 1.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_l2_dots.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(counts).to_csv(os.path.join(OUT, "fig3_l2_dots_data.csv"), index=False)
    print("wrote", os.path.join(OUT, "fig3_l2_dots.pdf"), "and fig3_l2_dots_data.csv")


if __name__ == "__main__":
    main()
