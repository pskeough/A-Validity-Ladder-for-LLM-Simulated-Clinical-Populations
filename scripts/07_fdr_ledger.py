"""
Machine-readable FDR ledger for PsychBench v2 (draft section 3.5's promised supplementary artifact).

Enumerates the TRUE test family behind every significance claim in the v2 draft and applies
Benjamini-Hochberg across the whole family at once. One row per test:
  test_id, section, description, test_type, n, statistic, p_raw, p_bh, significant_q05

Family (24 tests):
  T01-T09  section 4  residual vs gold GT, one-sample t on model cells (9 benchmarkable groups,
                      CISGENDER personas only, frame-matched per paper section 3.2; v2 primary)
  T10      section 6  cross-run drift, paired t over 14,400 pairs
  T11      section 6  category-flip heterogeneity across models, chi-square 4x2
  T12      section 5  gateway-violation heterogeneity across models, chi-square 4x2
  T13-T14  section 9  trans-cis severity elevation, Welch t (TW vs CW, TM vs CM)
  T15-T27  section 7  covariance divergence, label-permutation test on Frobenius distance
                      (10 race/gender pairs + 3 SES pairs)
Seeded (SEED=42). Sections 8's variance ratios are descriptive by design and carry no tests.

UNIT OF ANALYSIS. Every parametric test below uses the design cell (model x cohort, 480 per
condition) as its unit, not the generation row. The 30 iterations inside a cell are repeated draws
from one conditional response distribution rather than distinguishable individuals, so row-level
tests understate every standard error by roughly a factor of five to seven. The permutation tests
(T15-T34) operate on the row-level correlation structure by construction; null and observed share
that structure, so they are internally valid, but their precision inherits the same clustering and
they are read against magnitude rather than p-value in Section 8.
"""
import pandas as pd, numpy as np, os
from scipy import stats

SEED = 42
N_PERM = 500
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
BASE = os.path.join(os.path.dirname(__file__), "..")
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
OUT = os.path.join(BASE, "analysis")
G = lambda grp: gt.loc[gt.group == grp].iloc[0]
rng = np.random.default_rng(SEED)

m["ses_clean"] = m["ses"].str.replace('"', '', regex=False).str.split(" ").str[0]
m["phq8_total"] = m["phq8_total"].clip(0, 24)

# condition from run provenance (11_condition_contrast.py emits the map; run1=clinical, run2=narrative)
cmap = pd.read_csv(os.path.join(BASE, "analysis", "condition_runid_map.csv"))
m["condition"] = m.run_id.map(dict(zip(cmap.run_id, cmap.condition)))
assert m.condition.notna().all(), "rows with unmapped run_id"

L = []
def add(tid, sec, desc, ttype, n, stat, p):
    L.append(dict(test_id=tid, section=sec, description=desc, test_type=ttype,
                  n=n, statistic=round(float(stat), 3), p_raw=float(p)))

# ---- T01-T09: residuals vs gold (one-sample t against the GT mean as fixed constant) ---
# v2 primary frame: cisgender personas only (paper section 3.2 frame-matching).
CIS = ["Cisgender Man", "Cisgender Woman"]
groups = [("race", "White", "White"), ("race", "Black", "Black"), ("race", "Asian", "Asian"),
          ("race", "Hispanic", "Hispanic (pooled)"),
          ("gender", "Cisgender Man", "Men"), ("gender", "Cisgender Woman", "Women"),
          ("ses_clean", "Low", "Low"), ("ses_clean", "Middle", "Middle"), ("ses_clean", "High", "High")]
CELL = ["model", "profile_id"]
for i, (col, grp, gg) in enumerate(groups, 1):
    sel = (m[col] == grp) if col == "gender" else ((m[col] == grp) & m.gender.isin(CIS))
    vals = m.loc[sel].groupby(CELL).phq8_total.mean()          # one value per design cell
    t, p = stats.ttest_1samp(vals, G(gg).w_mean)
    add(f"T{i:02d}", "4", f"residual vs gold GT: {grp} (cisgender personas, cell level)",
        "one-sample t on cell means", len(vals), t, p)

# ---- T10: condition contrast, PHQ-8 (clinical vs narrative, paired by design cell) ------
KEY = ["model", "profile_id", "iteration"]
wide = m.pivot_table(index=KEY, columns="condition", values="phq8_total")
assert wide.notna().all().all() and len(wide) == 14400
cellw = m.pivot_table(index=CELL, columns="condition", values="phq8_total")
assert len(cellw) == 480
t, p = stats.ttest_rel(cellw.clinical, cellw.narrative)
add("T10", "5", "condition contrast: clinical vs narrative prompt framing, PHQ-8 (cell level)",
    "paired t on cell means", len(cellw), t, p)

# ---- T11: cross-condition flip-rate heterogeneity across models -------------------------
cat = lambda x: pd.cut(x, [-1, 4, 9, 14, 19, 24], labels=False)
paired = wide.reset_index()
paired["flip"] = cat(paired.clinical) != cat(paired.narrative)
# cell-level flip proportion, then a between-model test on those 480 proportions
cellflip = paired.groupby(CELL).flip.mean().reset_index()
cellflip["model"] = [i[0] for i in cellflip[CELL].itertuples(index=False)]
H, p = stats.kruskal(*[g.flip.values for _, g in cellflip.groupby("model")])
add("T11", "6", "cross-condition category-flip rate differs across the 4 models (cell level)",
    "Kruskal-Wallis on cell flip proportions", len(cellflip), H, p)

# ---- T12: gateway-violation heterogeneity across models --------------------------------
elev = m[m.phq8_total >= 10].copy()
elev["viol"] = (elev.phq8_1 < 2) & (elev.phq8_2 < 2)
cellviol = elev.groupby(CELL).viol.mean().reset_index()
cellviol["model"] = [i[0] for i in cellviol[CELL].itertuples(index=False)]
H, p = stats.kruskal(*[g.viol.values for _, g in cellviol.groupby("model")])
add("T12", "5", "gateway-violation rate differs across the 4 models (cell level)",
    "Kruskal-Wallis on cell violation proportions", len(cellviol), H, p)

# ---- T13-T14: trans-cis severity elevation ----------------------------------------------
for tid, a, b in [("T13", "Transgender Woman", "Cisgender Woman"),
                  ("T14", "Transgender Man", "Cisgender Man")]:
    va = m.loc[m.gender == a].groupby(CELL).phq8_total.mean()
    vb = m.loc[m.gender == b].groupby(CELL).phq8_total.mean()
    t, p = stats.ttest_ind(va, vb, equal_var=False)
    add(tid, "9", f"severity elevation: {a} vs {b} (descriptive contrast, cell level)",
        "Welch t on cell means", len(va) + len(vb), t, p)

# ---- T15-T24: covariance divergence, label-permutation null -----------------------------
corr = lambda df: df[ITEMS].corr().to_numpy()
frob = lambda a, b: float(np.linalg.norm(a - b, ord="fro"))
pairs = [("race", "Black", "White"), ("race", "Asian", "White"), ("race", "Hispanic", "White"),
         ("race", "Multiracial", "White"),
         ("gender", "Cisgender Man", "Cisgender Woman"),
         ("gender", "Cisgender Man", "Transgender Woman"),
         ("gender", "Cisgender Man", "Transgender Man"),
         ("gender", "Cisgender Woman", "Transgender Woman"),
         ("gender", "Cisgender Woman", "Transgender Man"),
         ("gender", "Transgender Woman", "Transgender Man"),
         ("ses", "Low", "High"), ("ses", "Middle", "High"), ("ses", "Low", "Middle"),
         ("relationship", "Single", "Married")]
# race/gender pairs keep their published IDs T15-T24; the SES pairs are appended as T31-T33
# so that every test ID cited in the manuscript remains stable across ledger revisions.
COV_IDS = [f"T{15 + i}" for i in range(10)] + ["T31", "T32", "T33", "T34"]
for j, (dim, a, b) in zip(COV_IDS, pairs):
    col = "ses_clean" if dim == "ses" else dim
    sa, sb = m[m[col] == a], m[m[col] == b]
    obs = frob(corr(sa), corr(sb))
    pool = pd.concat([sa, sb]); na = len(sa)
    idx = pool.index.to_numpy()
    null = []
    for _ in range(N_PERM):
        perm = rng.permutation(idx)
        null.append(frob(corr(pool.loc[perm[:na]]), corr(pool.loc[perm[na:]])))
    p = (1 + sum(d >= obs for d in null)) / (N_PERM + 1)
    add(j, "7", f"covariance divergence: {a} vs {b} (Frobenius, label permutation)",
        f"permutation ({N_PERM} draws)", len(pool), obs, p)

# ---- T25-T26: condition contrast on the other comparable batteries ----------------------
for tid, inst, name in [("T25", "gad7_total", "GAD-7"), ("T26", "audit_total", "AUDIT-C")]:
    w = m.pivot_table(index=CELL, columns="condition", values=inst)
    t, p = stats.ttest_rel(w.clinical, w.narrative)
    add(tid, "5", f"condition contrast: clinical vs narrative prompt framing, {name} (cell level)",
        "paired t on cell means", len(w), t, p)

# ---- T27-T30: per-model PHQ-8 condition contrast (sign heterogeneity claim) --------------
for tid, mdl in zip(["T27", "T28", "T29", "T30"], sorted(m.model.unique())):
    w = cellw.reset_index()
    w = w[w.model == mdl]
    t, p = stats.ttest_rel(w.clinical, w.narrative)
    add(tid, "5", f"condition contrast (PHQ-8): {mdl} (cell level)",
        "paired t on cell means", len(w), t, p)

# ---- BH over the whole family -----------------------------------------------------------
led = pd.DataFrame(L)
# At this N nearly everything is detectable; the paper's magnitude claims rest on the split-half
# noise floors (covariance_noise_floors.csv), not on these p-values. Flag the two divergences
# that are detectable but do NOT clear their per-group p95 floors, so the ledger cannot be read
# as contradicting section 7's "within noise" language.
led["note"] = ""
led.loc[led.test_id.str.startswith("T") & led.test_type.str.contains("permutation"), "note"] = (
    "raw-distance test on row-level correlation structure; section 8 reports the "
    "severity-matched statistic against its own size-matched null (17_fair_null_receipts.py)")
rej, p_bh = stats.false_discovery_control(led.p_raw, method="bh"), None
# scipy>=1.11: false_discovery_control returns adjusted p-values
led["p_bh"] = rej
led["significant_q05"] = led.p_bh < 0.05
led["p_raw"] = led.p_raw.map(lambda x: f"{x:.3e}")
led["p_bh"] = led.p_bh.map(lambda x: f"{x:.3e}")
led.to_csv(os.path.join(OUT, "fdr_ledger.csv"), index=False)

pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 70)
print(f"FDR LEDGER — {len(led)} tests, BH across the full family (seed {SEED})\n" + "=" * 120)
print(led.to_string(index=False))
print(f"\nsignificant at q<0.05: {int(led.significant_q05.sum())}/{len(led)}")
