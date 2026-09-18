"""
v2 paper figures, rendered from the receipt CSVs (never from prose numbers).
Outputs vector PDF + PNG preview into paper_v2/figures/.
Design: single-hue primary (#0072B2), Okabe-Ito categorical for models (direct-labeled),
thin marks, recessive grid, no dual axes, CVD-safe (validated).
"""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..")
A = lambda f: pd.read_csv(os.path.join(BASE, "analysis", f))
# Figures are written to both manuscript trees. They used to go only to paper_v2 and be copied
# into paper_short by hand, which is how two stale figure PDFs once shipped in a release.
FIG = os.path.join(BASE, "paper_v2", "figures"); os.makedirs(FIG, exist_ok=True)
FIG_SHORT = os.path.join(BASE, "paper_short", "figures"); os.makedirs(FIG_SHORT, exist_ok=True)

BLUE = "#0072B2"
MODEL_C = {"openai/gpt-4o-mini": "#0072B2", "google/gemini-3-flash-preview": "#E69F00",
           "deepseek/deepseek-chat-v3": "#009E73", "z-ai/glm-4.7": "#CC79A7"}
MODEL_SHORT = {"openai/gpt-4o-mini": "GPT-4o-mini", "google/gemini-3-flash-preview": "Gemini-3-Flash",
               "deepseek/deepseek-chat-v3": "DeepSeek-V3", "z-ai/glm-4.7": "GLM-4.7"}
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "grid.linewidth": 0.4, "grid.alpha": 0.35})

def save(fig, name):
    for d in (FIG, FIG_SHORT):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig); print("wrote", name)

# ---------- Fig 1: residuals, cis-only primary vs pooled sensitivity ----------
cis = A("bias_residuals_CISONLY.csv")
pool = A("bias_residuals_GOLD.csv"); pool = pool[pool.residual.notna()]
order = ["White","Black","Asian","Hispanic","Cisgender Man","Cisgender Woman","Low","Middle","High"]
label = {"Low":"Low SES","Middle":"Middle SES","High":"High SES"}
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ys = np.arange(len(order))[::-1]
for y, g in zip(ys, order):
    c = cis[cis.group==g].iloc[0]
    ax.plot([c.resid_ci_lo, c.resid_ci_hi], [y, y], color=BLUE, lw=1.4, zorder=2)
    ax.scatter(c.residual, y, s=26, color=BLUE, zorder=3)
    p = pool[pool.group==g]
    if len(p):
        ax.scatter(p.residual.iloc[0], y, s=30, facecolors="none", edgecolors=BLUE,
                   lw=1.1, zorder=3)
ax.axvline(0, color="0.25", lw=0.8)
ax.set_yticks(ys); ax.set_yticklabels([label.get(g,g) for g in order])
ax.set_xlabel("PHQ-8 residual (model mean $-$ NHANES weighted mean)")
ax.set_xlim(-0.4, 7.6); ax.grid(axis="x")
ax.scatter([],[], s=26, color=BLUE, label="Cisgender personas (primary)")
ax.scatter([],[], s=30, facecolors="none", edgecolors=BLUE, lw=1.1,
           label="Full grid pooled (sensitivity)")
# The legend used to sit inside the axes near x=0, where its two keys landed on the Low SES and
# Middle SES rows and read as data points close to zero for the two groups with the largest
# residuals in the figure. Moved below the plot, the same fix Fig 4 already carries.
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center",
          bbox_to_anchor=(0.5, -0.22), columnspacing=1.6, handletextpad=0.3)
save(fig, "fig1_residuals")

# ---------- Fig 2: model specificity (gateway violations + stability decomposition) ------
gw = A("gateway_per_model.csv"); gw = gw[gw.model!="POOLED"]
wr = A("within_run_stability.csv"); wr = wr[wr.model!="ALL"]
cc = A("condition_contrast.csv")
xflip = cc[(cc.instrument=="phq8_category_flip") & (cc.model!="ALL")].set_index("model")["diff"]
mods = sorted(gw.model, key=lambda m: gw.set_index("model").loc[m,"violation_rate_pct"])
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
ax = axes[0]
v = gw.set_index("model").violation_rate_pct.reindex(mods)
# Each model's own conditional-marginal null is drawn alongside its observed rate. Without it the
# panel reads as a ranking of coherence, and the model with the highest observed rate also has the
# highest null: GLM-4.7 sits at its own null rather than below it (Section 7).
nul = A("gateway_null.csv").set_index("scope").null_conditional_marginal_pct.reindex(mods)
ax.barh(range(len(mods)), v.values, height=0.55, color=BLUE)
ax.scatter(nul.values, range(len(mods)), s=30, marker="|", linewidths=1.6, color="0.25", zorder=4)
for i, x in enumerate(v.values):
    ax.text(x + 19*0.012, i, f"{x:.2f}", va="center", fontsize=7.5)
ax.set_yticks(range(len(mods))); ax.set_yticklabels([MODEL_SHORT[m] for m in mods], fontsize=8)
ax.set_xlabel("DSM-5 gateway violations (% of elevated cases)", fontsize=7.8)
ax.set_xlim(0, 19); ax.grid(axis="x")
# No legend on this panel. Placed low it reads as a label for DeepSeek's tick, placed high it
# overlaps GLM's bar, and the caption identifies the ticks anyway.
ax = axes[1]
wc = wr[wr.condition=="clinical"].set_index("model").flip_prob_pct.reindex(mods)
wn = wr[wr.condition=="narrative"].set_index("model").flip_prob_pct.reindex(mods)
xf = xflip.reindex(mods)
ys = np.arange(len(mods))
ax.scatter(wc, ys, s=24, color=BLUE, zorder=3, label="Within-run (clinical)")
ax.scatter(wn, ys, s=24, color=BLUE, facecolors="white", linewidths=1.2, zorder=3,
           label="Within-run (narrative)")
ax.scatter(xf, ys, s=26, color="0.2", marker="D", zorder=3, label="Cross-condition")
for y in ys:
    ax.plot([min(wc.iloc[y], wn.iloc[y], xf.iloc[y]), max(wc.iloc[y], wn.iloc[y], xf.iloc[y])],
            [y, y], color="0.75", lw=0.8, zorder=2)
ax.set_yticks(ys); ax.set_yticklabels([]); ax.set_xlim(15, 52)
ax.set_xlabel("PHQ-8 category-flip probability (%)", fontsize=7.8)
ax.grid(axis="x"); ax.legend(frameon=False, fontsize=6.8, loc="lower right")
fig.tight_layout(w_pad=1.2)
save(fig, "fig2_model_specificity")

# ---------- Fig 3: covariance divergence with noise floors ----------
cov = A("covariance_divergence_GOLD.csv"); fl = A("covariance_noise_floors.csv")
# matched values and the null they are judged against both come from the fair-null receipt,
# so the figure cannot drift from the text (17_fair_null_receipts.py)
fair = A("covariance_fair_null.csv").set_index(["group_a", "group_b"])
def fair_row(a, b):
    if (a, b) in fair.index: return fair.loc[(a, b)]
    return fair.loc[(b, a)]
rows = [("Low vs high SES", "Low", "High"),
        ("Trans woman vs cis man", "Cisgender Man", "Transgender Woman"),
        ("Trans man vs cis man", "Cisgender Man", "Transgender Man"),
        ("Low vs middle SES", "Low", "Middle"),
        ("Middle vs high SES", "Middle", "High"),
        ("Cis woman vs cis man", "Cisgender Man", "Cisgender Woman"),
        ("Asian vs White", "Asian", "White"),
        ("Black vs White", "Black", "White"),
        ("Hispanic vs White", "Hispanic", "White"),
        ("Trans woman vs trans man", "Transgender Woman", "Transgender Man")]
fig, ax = plt.subplots(figsize=(4.9, 3.6))
ys = np.arange(len(rows))[::-1]
for y, (lab, a, b) in zip(ys, rows):
    r = cov[((cov.group_a==a)&(cov.group_b==b))|((cov.group_a==b)&(cov.group_b==a))].iloc[0]
    f = fair_row(a, b)
    ax.barh(y+0.19, r.frobenius, height=0.34, color=BLUE, label="_")
    ax.barh(y-0.19, f.matched_frobenius, height=0.34, color="#9ECAE1", label="_")
    ax.plot([f.fair_null_p95, f.fair_null_p95], [y-0.42, y+0.42], color="0.15", lw=1.2, zorder=4)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=7.8)
ax.set_xlabel("Frobenius distance between PHQ-8 item correlation matrices")
ax.grid(axis="x")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Patch(facecolor=BLUE, label="Raw"),
           Patch(facecolor="#9ECAE1", label="Severity-matched"),
           Line2D([0],[0], color="0.15", lw=1.2, label="Size-matched permutation null (p95)")]
ax.legend(handles=handles, frameon=False, fontsize=7.3, loc="lower right")
save(fig, "fig3_covariance")

# ---------- Fig 4: variance ratios (descriptive), pooled + per model ----------
pg = A("per_group_variance_GOLD.csv")
grp_order = ["White","Black","Asian","Hispanic","Cisgender Man","Cisgender Woman","Low","Middle","High"]
pooled = pg[pg.model=="POOLED"].set_index("group")
fig, ax = plt.subplots(figsize=(6.2, 2.7))
xs = np.arange(len(grp_order))
ax.axhline(1.0, color="0.25", lw=0.8, ls=(0,(4,3)))
ax.bar(xs, pooled.reindex(grp_order).sd_ratio, width=0.5, color=BLUE, alpha=0.28, label="Pooled")
# Models are dodged horizontally inside each group. Gemini and GLM sit at 1.231 and 1.236 on the
# White cell and at 1.44 and 1.45 on Asian, so plotted at the same x one marker hid the other
# entirely and the figure showed three models where there are four.
for k, (mod, c) in enumerate(MODEL_C.items()):
    mv = pg[pg.model==mod].set_index("group").reindex(grp_order).sd_ratio
    ax.scatter(xs + (k - 1.5) * 0.15, mv, s=15, color=c, label=MODEL_SHORT[mod], zorder=3)
ax.set_xticks(xs); ax.set_xticklabels([label.get(g,g) for g in grp_order], rotation=28, ha="right", fontsize=7.6)
ax.set_ylabel("SD ratio (model / population)")
ax.grid(axis="y")
# The legend used to sit inside the axes at upper left, where it covered Gemini's marker on the
# White column entirely. Moved below the plot, and the top of the axis lifted so no marker touches
# the frame.
ax.set_ylim(0, max(1.55, float(pg.sd_ratio.max()) * 1.08))
ax.legend(frameon=False, fontsize=7.2, ncol=5, loc="upper center",
          bbox_to_anchor=(0.5, -0.34), columnspacing=1.2, handletextpad=0.3)
save(fig, "fig4_variance_ratios")

# ---------- Fig 5: condition contrast per model (PHQ-8 and GAD-7) ----------
cc = A("condition_contrast.csv")
morder = ["ALL","deepseek/deepseek-chat-v3","z-ai/glm-4.7",
          "google/gemini-3-flash-preview","openai/gpt-4o-mini"]
mlab = dict(MODEL_SHORT, ALL="All models")
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True)
for ax, inst, ttl in [(axes[0], "phq8_total", "PHQ-8"), (axes[1], "gad7_total", "GAD-7")]:
    sub = cc[cc.instrument==inst].set_index("model").reindex(morder)
    ys = np.arange(len(morder))[::-1]
    ax.axvline(0, color="0.25", lw=0.8)
    for y, (mdl, r) in zip(ys, sub.iterrows()):
        c = "0.2" if mdl=="ALL" else MODEL_C[mdl]
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=c, lw=1.4, zorder=2)
        ax.scatter(r["diff"], y, s=24, color=c, zorder=3)
    ax.set_yticks(ys); ax.set_title(ttl, fontsize=8.5)
    ax.set_xlabel("Clinical $-$ narrative (points)", fontsize=7.8)
    ax.grid(axis="x")
axes[0].set_yticklabels([mlab[m] for m in morder], fontsize=8)
fig.tight_layout(w_pad=1.4)
save(fig, "fig5_condition_contrast")

# ---------- Fig 6: model profiles (calibration vs stability) ----------
pmr = A("per_model_residuals.csv")
wr = A("within_run_stability.csv"); wr = wr[wr.model!="ALL"]
mean_res = pmr.groupby("model").residual.mean()
mean_flip = wr.groupby("model").flip_prob_pct.mean()
fig, ax = plt.subplots(figsize=(3.6, 3.0))
offsets = {"openai/gpt-4o-mini": (-0.06, 1.6), "google/gemini-3-flash-preview": (0.06, 1.6),
           "deepseek/deepseek-chat-v3": (0.06, 1.6), "z-ai/glm-4.7": (0.06, 1.6)}
for mdl, c in MODEL_C.items():
    x, y = mean_res[mdl], mean_flip[mdl]
    ax.scatter(x, y, s=42, color=c, zorder=3)
    dx, dy = offsets[mdl]
    ax.annotate(MODEL_SHORT[mdl], (x, y), xytext=(x+dx, y+dy),
                fontsize=7.6, ha="center")
ax.set_xlabel("Mean severity residual, cisgender frame (points)")
ax.set_ylabel("Mean within-run flip probability (%)")
ax.set_xlim(2.0, 5.6); ax.set_ylim(18, 48); ax.grid(True)
save(fig, "fig6_profiles")
print("done")
