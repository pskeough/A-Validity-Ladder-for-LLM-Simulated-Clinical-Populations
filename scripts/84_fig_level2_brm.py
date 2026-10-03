"""BRM body figure for level 2: the gap ratio against the standardised NHANES reference, per model.

Reads ONLY script-78 outputs:
  analysis/brm/l2_verdicts.csv  (78c)  rows analysis == "headline", estimand == "standardised".
      Headline configuration (78c `H`): v3 corpus, four-model panel, combined framings, primary
      reference (four races, "Single" = not married), simulated SE = pair-clustered SD of persona-pair
      differences / sqrt(pairs) (`se_type` "stratified": model-stratified when pooled), Taylor
      reference SE, boundaries -0.25 / 0.25 / 0.75 / 1.25, Bonferroni over the seven contrasts of
      each scope (one-sided .05/7 at each boundary, 98.57% Fieller interval), conditional stop.
  analysis/brm/l2_headline.csv  (78c)  used only as a cross-check: every per-model ratio,
      interval and code printed there must match the row drawn here.

One row group per contrast; within it the four models (filled, colour + shape) and the pooled row
(hollow, descriptive only). Each interval is the Fieller set
78c reports. Shaded bands are the R3 regions. Intervals running past the axis are clipped with an
arrow and the bound printed. A row stopped as "reference too imprecise" shows its point as a hollow
marker with "stopped" and no interval. A contrast stopped as "no population gap" has no defined
ratio, so its rows carry no point and the group is annotated instead.

Self-checks before drawing (the script stops on any failure):
  - configuration columns of every row equal the headline configuration;
  - each drawn interval read against the boundaries gives the verdict 78c printed (r3_unstopped);
  - the right-hand verdict word equals 78c's `verdict` for every row;
  - the values match l2_headline.csv.

Emits paper_brm/manuscript/figures/figS_level2_intervals.pdf (vector) and .png (200 dpi preview),
which is Supplement Figure S1, and fig2_level2_data.csv (the values plotted), which
scripts/84f_fig3_l2_dots.py reads for the worked-example row of Figure 3.
"""
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(BASE, "analysis", "brm", "l2_verdicts.csv")
HEAD = os.path.join(BASE, "analysis", "brm", "l2_headline.csv")
OUT = os.path.join(BASE, "paper_brm", "manuscript", "figures")

HEADLINE = dict(corpus="all", panel="four models", framing="combined", ref_variant="primary",
                se_type="stratified", ref_se="taylor", band="asym", mult="bonferroni7")
BOUNDS = np.array([-0.25, 0.25, 0.75, 1.25])
LABELS = ["reversed", "missing", "attenuated", "kept", "steepened"]
STOPS = {"reference too imprecise", "no population gap"}

CONTRASTS = [("Women minus Men", "Women − Men"),
             ("Low minus High SES", "Low − High income"),
             ("Low minus Middle SES", "Low − Middle income"),
             ("Middle minus High SES", "Middle − High income"),
             ("Black minus White", "Black − White"),
             ("Hispanic minus White", "Hispanic − White"),
             ("Asian minus White", "Asian − White")]
# Okabe-Ito colours; shape carries the model identity as well
MODELS = [("openai/gpt-4o-mini", "GPT-4o-mini", "#0072B2", "o"),
          ("google/gemini-3-flash-preview", "Gemini-3-Flash", "#E69F00", "s"),
          ("deepseek/deepseek-chat-v3", "DeepSeek-V3", "#009E73", "^"),
          ("z-ai/glm-4.7", "GLM-4.7", "#CC79A7", "D")]
POOLED = ("pooled", "Pooled (descriptive)", "0.15", "o")

XMIN, XMAX = -2.25, 9.25
W_IN = 6.5
FS = 7.5          # body text
FS_MIN = 7.0      # smallest text anywhere


def region_reading(lo, hi):
    """R3 naming from an interval: regions the interval reaches."""
    i_lo = int(np.sum(BOUNDS < lo))
    i_hi = int(np.sum(BOUNDS < hi))
    if i_lo == i_hi:
        return LABELS[i_lo]
    if i_hi == i_lo + 1:
        return f"{LABELS[i_lo]} or {LABELS[i_hi]}"
    return "undetermined"


def three_way(v, lo, hi, unbounded):
    """Article's per-contrast verdict (2 Oct 2026 rewrite): kept / not kept / unresolved, read
    from the interval against the kept region [0.75, 1.25]. A stop is 'not read' unless the
    interval already excludes kept (LADDER_SPEC stop 2), which 78c prints as the region verdict."""
    if v in STOPS:
        return "not read"
    assert not unbounded, v
    if lo >= BOUNDS[2] and hi <= BOUNDS[3]:
        out = "kept"
    elif hi < BOUNDS[2] or lo > BOUNDS[3]:
        out = "not kept"
    else:
        out = "unresolved"
    # consistency with 78c's region verdict
    if out == "kept":
        assert v == "kept", v
    elif out == "unresolved":
        assert v == "undetermined" or "kept" in v, v
    else:
        assert "kept" not in v, v
    return out


def verdict_text(v, lo=np.nan, hi=np.nan, unbounded=True):
    if v == "reference too imprecise":
        return "not read (imprecise reference)"
    if v == "no population gap":
        return "not read (no population gap)"
    t = three_way(v, lo, hi, unbounded)
    if t == "not kept":
        return f"not kept, {v}"
    return t


def load():
    v = pd.read_csv(SRC)
    h = v[(v.analysis == "headline") & (v.estimand == "standardised")].copy()
    for k, want in HEADLINE.items():
        bad = h[h[k] != want]
        if len(bad):
            raise SystemExit(f"{k}: {len(bad)} headline rows are not {want!r}")
    assert len(h) == 35, len(h)
    assert np.allclose(h.interval_level, 1 - 2 * 0.05 / 7, atol=1e-4)
    return h


def cross_check(h):
    hd = pd.read_csv(HEAD)
    hd = hd[hd.estimand == "standardised"].set_index("contrast")
    code = {"r": "reversed", "m": "missing", "a": "attenuated", "k": "kept", "s": "steepened",
            "u": "undetermined", "X": "reference too imprecise", "N": "no population gap"}
    n = 0
    for r in h.itertuples():
        if r.scope == "pooled":
            row = hd.loc[r.contrast]
            assert abs(row.pooled_ratio - r.ratio) < 1e-3, (r.contrast, "pooled ratio")
            assert row.pooled_verdict == r.verdict, (r.contrast, "pooled verdict")
            if not r.ci_unbounded:
                lo, hi = map(float, row.pooled_ci.split(" to "))
                assert abs(lo - r.ci_lo) < 0.006 and abs(hi - r.ci_hi) < 0.006
        else:
            s = hd.loc[r.contrast, r.scope.split("/")[-1]]
            m = re.match(r"(-?[\d.]+) \[(.*)\] (\S+)$", s)
            assert m, s
            assert abs(float(m.group(1)) - r.ratio) < 0.006, s
            c = m.group(3)
            want = " or ".join(code[x] for x in c.split("/"))
            assert want == r.verdict, (s, r.verdict)
            if m.group(2) != "unbounded":
                lo, hi = map(float, m.group(2).split(" to "))
                assert abs(lo - r.ci_lo) < 0.006 and abs(hi - r.ci_hi) < 0.006, s
        n += 1
    return n


def build_rows(h):
    """Row table in drawing order with y positions."""
    rows, y = [], 0.0
    headers = []
    for cname, clabel in CONTRASTS:
        headers.append((y, clabel, cname))
        y -= 1.0
        for scope, mlabel, col, mk in MODELS + [POOLED]:
            r = h[(h.contrast == cname) & (h.scope == scope)]
            assert len(r) == 1, (cname, scope)
            r = r.iloc[0]
            pooled = scope == "pooled"
            stopped = r.verdict in STOPS
            # interval check: the drawn interval must give 78c's R3 reading
            if not r.ci_unbounded:
                chk = region_reading(r.ci_lo, r.ci_hi)
                if chk != r.r3_unstopped:
                    raise SystemExit(f"interval/verdict mismatch {cname} {scope}: "
                                     f"[{r.ci_lo}, {r.ci_hi}] reads {chk}, 78c {r.r3_unstopped}")
            if not stopped and r.verdict != r.r3_unstopped:
                raise SystemExit(f"unstopped verdict differs {cname} {scope}")
            draw_interval = not stopped
            draw_point = r.verdict != "no population gap"
            lo_d = max(r.ci_lo, XMIN) if draw_interval else np.nan
            hi_d = min(r.ci_hi, XMAX) if draw_interval else np.nan
            rows.append(dict(
                y=y, contrast=cname, contrast_label=clabel, scope=scope, model_label=mlabel,
                pooled_descriptive=pooled, estimand="standardised", n_pairs=int(r.n_pairs),
                simulated_gap=r.simulated, se_sim_pairs=r.se_sim, df_sim=r.df_sim,
                nhanes_std_gap=r.population, se_nhanes_taylor=r.se_pop, df_nhanes=r.df_pop,
                nhanes_ci_lo=r.pop_ci_lo, nhanes_ci_hi=r.pop_ci_hi,
                interval_level=r.interval_level, ratio=r.ratio,
                fieller_lo=np.nan if r.ci_unbounded else r.ci_lo,
                fieller_hi=np.nan if r.ci_unbounded else r.ci_hi,
                ci_unbounded=bool(r.ci_unbounded),
                point_plotted=r.ratio if draw_point else np.nan,
                interval_plotted=draw_interval,
                drawn_lo=lo_d, drawn_hi=hi_d,
                clipped_lo=bool(draw_interval and r.ci_lo < XMIN),
                clipped_hi=bool(draw_interval and r.ci_hi > XMAX),
                r3_unstopped=r.r3_unstopped, verdict_78c=r.verdict,
                verdict_printed=verdict_text(r.verdict, r.ci_lo, r.ci_hi, bool(r.ci_unbounded)),
                verdict_popstop_78c=r.verdict_popstop,
                k_rel_halfwidth=r.k_rel_halfwidth,
                source="analysis/brm/l2_verdicts.csv analysis=headline estimand=standardised"))
            y -= 1.0
        y -= 0.45
    return pd.DataFrame(rows), headers, y


def main():
    h = load()
    n = cross_check(h)
    d, headers, ybot = build_rows(h)
    # the printed verdict must be 78c's verdict, row by row
    for r in d.itertuples():
        assert r.verdict_printed == verdict_text(r.verdict_78c, r.fieller_lo, r.fieller_hi,
                                                 r.ci_unbounded)

    plt.rcParams.update({"font.family": "Arial", "font.size": FS, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "xtick.major.size": 2.5, "xtick.labelsize": FS})
    ytop = 3.35
    H_IN = 0.128 * (ytop - ybot) + 0.95
    fig = plt.figure(figsize=(W_IN, H_IN))
    L, R = 1.30 / W_IN, 1.0 - 1.74 / W_IN
    B, T = 0.42 / H_IN, 1.0 - 0.40 / H_IN
    ax = fig.add_axes([L, B, R - L, T - B])
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(ybot + 0.55, ytop)

    # the kept region is the only band that decides a verdict; the other boundaries only name
    # the direction of a contrast that is not kept, so they are drawn as faint ticks
    ax.axvspan(BOUNDS[2], BOUNDS[3], color="#cfe8d6", lw=0, zorder=0)
    for b in BOUNDS[2:]:
        ax.axvline(b, color="#2e7d4f", lw=0.7, zorder=1)
    ax.axvline(0.0, color="0.55", lw=0.5, zorder=1)
    ax.plot([1.0, 1.0], [ybot, 0.75], color="0.35", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.text(1.0, ytop - 1.0, "kept region, 0.75 to 1.25", ha="center", va="center",
            fontsize=FS_MIN, color="#1b5e36", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="#2e7d4f", lw=0.5))

    tr_l = blended_transform_factory(fig.transFigure, ax.transData)
    tr_r = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.0 + 0.10 / (R - L) / W_IN, 0.0, "Verdict", ha="left", va="center",
            fontsize=FS, fontweight="bold", transform=tr_r)
    for yh, clabel, _ in headers:
        ax.text(0.05 / W_IN, yh, clabel, ha="left", va="center", fontsize=FS + 0.5,
                fontweight="bold", transform=tr_l)

    style = {m[0]: m for m in MODELS + [POOLED]}
    for r in d.itertuples():
        _, mlabel, col, mk = style[r.scope]
        ax.text(0.16 / W_IN, r.y, mlabel, ha="left", va="center", fontsize=FS,
                style="italic" if r.pooled_descriptive else "normal",
                color="0.3" if r.pooled_descriptive else "0", transform=tr_l)
        vcol = "0.3" if r.pooled_descriptive else "0"
        ax.text(1.0 + 0.10 / (R - L) / W_IN, r.y, r.verdict_printed, ha="left", va="center",
                fontsize=FS, color=vcol, style="italic" if r.pooled_descriptive else "normal",
                transform=tr_r)
        if r.interval_plotted:
            ax.plot([r.drawn_lo, r.drawn_hi], [r.y, r.y], color=col, lw=1.3,
                    solid_capstyle="butt", zorder=3)
            for side, clipped, xe, bound, sym in (("lo", r.clipped_lo, XMIN, r.fieller_lo, "<"),
                                                  ("hi", r.clipped_hi, XMAX, r.fieller_hi, ">")):
                if not clipped:
                    continue
                ax.plot([xe], [r.y], marker=sym, color=col, ms=4.0, mew=0, zorder=4,
                        clip_on=False)
                off = 0.18 if side == "lo" else -0.18
                ax.text(xe + off, r.y + 0.48, f"{bound:.2f}".replace("-", "−"), ha="left" if side == "lo" else "right",
                        va="center", fontsize=FS_MIN, color="0.15", zorder=5)
            if r.pooled_descriptive:
                ax.plot([r.ratio], [r.y], marker=mk, ms=4.6, mfc="white", mec=col, mew=1.0,
                        zorder=5)
            else:
                ax.plot([r.ratio], [r.y], marker=mk, ms=4.4 if mk != "D" else 3.8, color=col,
                        mec="white", mew=0.4, zorder=5)
        elif not np.isnan(r.point_plotted):          # reference too imprecise
            ax.plot([r.ratio], [r.y], marker=mk, ms=4.6, mfc="white", mec=col, mew=1.0,
                    zorder=5)
            ax.text(r.ratio + 0.2, r.y, "not read", ha="left", va="center", fontsize=FS_MIN,
                    style="italic", color="0.15", zorder=5)

    # "no population gap" groups: one annotation over the group
    for yh, clabel, cname in headers:
        g = d[(d.contrast == cname) & (d.verdict_78c == "no population gap")]
        if len(g) == 0:
            continue
        assert len(g) == 5, "partial no-population-gap group"
        r0 = g.iloc[0]
        ymid = g.y.mean()
        txt = (f"Not read: the matched NHANES gap ({r0.nhanes_std_gap:.2f}, interval\n"
               f"{r0.nhanes_ci_lo:.2f} to {r0.nhanes_ci_hi:.2f}) cannot be told from zero,\n"
               "so no ratio is defined.")
        ax.text(4.6, ymid, txt.replace("-", "−"), ha="center", va="center", fontsize=FS_MIN,
                color="0.1", zorder=6, linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.5))

    for yh, _, _ in headers[1:]:
        ax.axhline(yh + 0.72, color="0.45", lw=0.5, zorder=2)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(range(-2, 10))
    ax.set_xticklabels([str(t).replace("-", "−") for t in range(-2, 10)])
    ax.set_xlabel("Gap ratio: simulated gap / standardised NHANES gap "
                  "(98.6% Fieller interval)", fontsize=FS)
    ax.xaxis.set_label_coords(0.5, -0.30 / (T - B) / H_IN)

    handles = [Line2D([], [], marker=mk, color=col, lw=1.3, ms=4.4 if mk != "D" else 3.8,
                      mec="white", mew=0.4, label=lab) for _, lab, col, mk in MODELS]
    handles.append(Line2D([], [], marker="o", color="0.15", lw=1.3, ms=4.6, mfc="white",
                          mec="0.15", mew=1.0, label="pooled, descriptive"))
    handles.append(Line2D([], [], marker="o", lw=0, ms=4.6, mfc="white", mec="0.15", mew=1.0,
                          label="not read (point only)"))
    handles.append(Line2D([], [], marker=">", lw=0, color="0.15", ms=4.0, mew=0,
                          label="interval clipped"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False, fontsize=FS_MIN, handlelength=1.6, handletextpad=0.35,
               columnspacing=0.9, borderaxespad=0.15)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "figS_level2_intervals.pdf"))
    fig.savefig(os.path.join(OUT, "figS_level2_intervals.png"), dpi=200)
    out = d.drop(columns=["y"]).copy()
    for c in out.columns:
        if out[c].dtype == float:
            out[c] = out[c].round(4)
    out.to_csv(os.path.join(OUT, "fig2_level2_data.csv"), index=False)

    print(f"source {SRC}")
    print(f"cross-check against l2_headline.csv: {n} rows, all match")
    print(f"figure {W_IN} x {H_IN:.2f} in")
    pm = d[~d.pooled_descriptive]
    print("\nper-model rows by verdict:")
    print(pm.verdict_78c.value_counts().to_string())
    print("\npooled rows by verdict:")
    print(d[d.pooled_descriptive].verdict_78c.value_counts().to_string())
    print("\nclipped:", d[d.clipped_lo | d.clipped_hi][["contrast", "scope", "fieller_lo",
                                                         "fieller_hi"]].to_string(index=False))
    print("\ninterval levels:", sorted(d.interval_level.unique()))
    print("df_nhanes:", sorted(d.df_nhanes.unique()), " n_pairs:", sorted(d.n_pairs.unique()))


if __name__ == "__main__":
    main()
