"""
Number-lock gate for the v2 manuscript: every load-bearing numeric claim in
paper_v2/main.tex must reconcile to a receipt CSV. Run after any edit; exits
nonzero on the first mismatch. No claim ships on faith.
"""
import pandas as pd, numpy as np, os, re, sys

BASE = os.path.join(os.path.dirname(__file__), "..")
A = lambda f: pd.read_csv(os.path.join(BASE, "analysis", f))
tex = open(os.path.join(BASE, "paper_v2", "main.tex"), encoding="utf-8").read()

FAIL = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  <- {detail}"))
    if not ok: FAIL.append(name)

def intex(*fragments):
    """Search against whitespace-normalized source.

    LaTeX wraps lines wherever it likes, so a needle that spans a line break fails against the raw
    file even when the sentence is present. That brittleness has produced two false failures and,
    worse, hid a real prose defect from a scan that used the same raw text.
    """
    return all(re.sub(r"\s+", " ", f) in tex_flat for f in fragments)


tex_flat = re.sub(r"\s+", " ", tex)

# ---- LaTeX structural sanity: every tabular row must match its column spec ------------------
# A dropped backslash or a stray & halts the build on "Extra alignment tab"; the PDF still gets
# written from the partial run, so this is caught here rather than by eyeballing the log.
for tb in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", tex, re.S):
    spec, body = tb.group(1), tb.group(2)
    ncol = len(re.findall(r"[lcr]|p\{[^}]*\}", spec))
    lineno = tex[:tb.start()].count("\n") + 1
    for raw in body.split(r"\\"):
        row = re.sub(r"\\(toprule|midrule|bottomrule|smallskip)\b", "", raw)
        row = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", row).strip()
        if not row or row.startswith("%"):
            continue
        amps = len(re.findall(r"(?<!\\)&", row))
        # A \multicolumn{n} cell spans n columns while contributing one &-separated field, so a
        # grouped header row is short on ampersands by design and is not a dropped cell.
        span = sum(int(n) - 1 for n in re.findall(r"\\multicolumn\{(\d+)\}", row))
        check(f"tabular arity L{lineno} ({row.split('&')[0].strip()[:22]!r})",
              amps + span == ncol - 1, f"{amps + span + 1} cells vs {ncol} columns")

# ---- Table 2: cis-only residuals + pooled sensitivity ------------------------------------
cis = A("residuals_cell_level.csv").set_index("group")   # cell-clustered, the reported table
pool = A("bias_residuals_GOLD.csv").set_index("group")
rowspec = {  # group -> (tex label)
    "White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic",
    "Cisgender Man": "Cisgender men", "Cisgender Woman": "Cisgender women",
    "Low": "Low SES", "Middle": "Middle SES", "High": "High SES"}
for grp, lab in rowspec.items():
    c = cis.loc[grp]
    frag = (f"{lab} & {c.model_mean:.2f} [{c.ci_lo:.2f}, {c.ci_hi:.2f}] & "
            f"{c.gt_mean:.2f} ({c.gt_sd:.2f}) & +{c.residual:.2f} "
            f"[+{c.resid_ci_lo:.2f}, +{c.resid_ci_hi:.2f}] & {c.cohens_d:.2f}")
    check(f"table2 {lab}", frag in tex, frag)
    if grp in ("Cisgender Man", "Cisgender Woman"):
        # gender marginals contain no transgender or unbenchmarkable cells, so the pooled
        # sensitivity equals the primary value; the table prints n/a by design
        check(f"table2 {lab} pooled n/a", f"{frag} & n/a" in tex, "expected n/a")
    elif grp in pool.index and not np.isnan(pool.loc[grp].residual):
        p = pool.loc[grp]
        check(f"table2 {lab} pooled", f"+{p.residual:.2f} ({p.cohens_d:.2f})" in tex,
              f"+{p.residual:.2f} ({p.cohens_d:.2f})")

# headline ranges
res = cis.loc[list(rowspec.keys())]
check("abstract residual range 2.8-5.5",
      round(res.residual.min(), 1) == 2.8 and round(res.residual.max(), 1) == 5.5
      and intex("2.8 to 5.5"))
check("abstract d range 0.80-1.30",
      round(res.cohens_d.min(), 2) == 0.80 and round(res.cohens_d.max(), 2) == 1.30
      and intex("0.80 to 1.30"))
lo, hi = cis.loc["Low"], cis.loc["High"]
grad_sim = lo.model_mean - hi.model_mean
gt6 = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv")).set_index("group")
grad_gt = gt6.loc["Low"].w_mean - gt6.loc["High"].w_mean  # full precision, not display-rounded
inf = A("inflation_form.csv").set_index("quantity")
_b = float(inf.loc["multiplicative constant b"].value)
_pred = float(inf.loc["SES gradient predicted by constant multiplier"].value)
check("SES gradient explained by a constant multiplier",
      abs(grad_sim - 4.75) < 0.01 and intex("simulated 4.75", "2.02 points")
      and abs(_pred - grad_gt * _b) < 0.01
      and intex(f"origin puts the multiplier at {float(inf.loc['multiplier, least squares through origin'].value):.2f}")
      and intex(f"predicts a simulated gradient of "
                f"{float(inf.loc['gradient predicted, OLS multiplier'].value):.2f} to "
                f"{float(inf.loc['gradient predicted, log-scale multiplier'].value):.2f}"),
      f"observed {grad_sim:.4f} vs multiplier-predicted {_pred:.4f} "
      f"(excess {(grad_sim/_pred - 1)*100:+.1f}%)")
check("low-SES mean 9.72", abs(lo.model_mean - 9.72) < 0.005 and intex("(9.72)"))

gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv")).set_index("group")
for g, pooled_lab in [("All adults 18+", ""), ("White", ""), ("Black", ""), ("Asian", ""),
                      ("Hispanic (pooled)", " pooled"), ("Men", ""), ("Women", ""),
                      ("Low", ""), ("Middle", ""), ("High", "")]:
    r = gt.loc[g]
    want = f"{r.w_mean:.2f}{pooled_lab} (SD {r.w_sd:.2f})"

# ---- gateway / coherence ------------------------------------------------------------------
gm = A("gateway_per_model.csv").set_index("model")
gti = A("gt_independent_metrics.csv").set_index("metric").value
check("gateway GLM 165/2644 6.24",
      intex("GLM-4.7 (165 of its 2,644 elevated cases, 6.24")
      and gm.loc["z-ai/glm-4.7"].violations == 165)
check("gateway DeepSeek 5/1447 0.35",
      intex("DeepSeek-V3 (5 of 1,447, 0.35") and gm.loc["deepseek/deepseek-chat-v3"].violations == 5)
check("gateway GPT 3.32 / Gemini 0.55",
      intex("GPT-4o-mini at 3.32", "Gemini-3-Flash at 0.55"))
check("coherence complement 97.3", intex("97.3\\%"))

# ---- condition contrast -------------------------------------------------------------------
cc = A("condition_contrast.csv")
ccx = cc.set_index(["instrument", "model"])
allphq = ccx.loc[("phq8_total", "ALL")]
check("condition pooled point estimate unchanged by unit choice",
      abs(allphq["diff"] - 0.3242) < 1e-4)

# variance compression by condition: ratios 1.02-1.13
vc = pd.read_csv(os.path.join(BASE, "analysis", "condition_variance_compression.csv"))
check("condition variance ratios 1.02-1.13",
      round(vc.ratio_clin_over_narr.min(), 2) == 1.02 and round(vc.ratio_clin_over_narr.max(), 2) == 1.13
      and intex("Within-cell standard deviations run higher under clinical framing in all three SES bands, but the raw ratios overstate it"))

# ---- within-run stability -----------------------------------------------------------------
wr = A("within_run_stability.csv")
wall = wr[wr.model == "ALL"].set_index("condition")
check("within-run flips 35.4 / 32.2",
      abs(wall.loc["clinical"].flip_prob_pct - 35.4) < 0.05
      and abs(wall.loc["narrative"].flip_prob_pct - 32.16) < 0.05
      and intex("35.4\\% (clinical) and 32.2\\% (narrative)"))
wm = wr[wr.model != "ALL"]
ds = wm[wm.model == "deepseek/deepseek-chat-v3"].flip_prob_pct
gl = wm[wm.model == "z-ai/glm-4.7"].flip_prob_pct
check("within-run per-model 22-27 DeepSeek, 38-45 GLM",
      22 <= ds.min() < ds.max() <= 27.5 and 37.5 <= gl.min() < gl.max() <= 45
      and intex("22 to 27\\% for DeepSeek-V3 up to 38 to 45\\% for GLM-4.7"))
check("cross stats 36.66 / 1.89 / r 0.75",
      abs(gti["flip_rate_5cat_pct"] - 36.66) < 0.005 and abs(gti["cross_run_MAD"] - 1.8913) < 0.005
      and intex("the corresponding figures are 36.66\\% and 1.89 points", "r = 0.75"))

# ---- profiles -----------------------------------------------------------------------------
pmr = A("per_model_residuals.csv")
mres = pmr.groupby("model").residual.mean()
mflip = wm.groupby("model").flip_prob_pct.mean()
for mdl, res_want, flip_want, frag in [
        ("z-ai/glm-4.7", 2.677, 41.21, "smallest mean calibration residual (2.68 points"),
        ("deepseek/deepseek-chat-v3", 4.816, 24.75, "most severity-inflated (4.82)"),
        ("openai/gpt-4o-mini", 4.926, 39.0, "largest residual (4.93)"),
        ("google/gemini-3-flash-preview", 3.269, 30.16, "(3.27 points, 30\\%, 0.55\\%)")]:
    check(f"profile {mdl}",
          abs(mres[mdl] - res_want) < 0.01 and abs(mflip[mdl] - flip_want) < 0.3 and intex(frag),
          f"res {mres[mdl]:.3f} flip {mflip[mdl]:.1f}")

ap = A("asian_paradox.csv").set_index("model")
check("asian gap GT 0.755",
      abs(ap.loc["POOLED"].gt_gap - 0.755) < 0.001 and intex("a population value of 0.755 points"))
for mdl, gap, txt in [("deepseek/deepseek-chat-v3", 0.708, "DeepSeek-V3 comes closest (simulated gap 0.708, error $-$0.05)"),
                      ("z-ai/glm-4.7", 1.736, "GLM-4.7 is furthest (1.736, error +0.98)"),
                      ("openai/gpt-4o-mini", 0.403, "undershooting at 0.403"),
                      ("google/gemini-3-flash-preview", 1.447, "overshooting at 1.447")]:
    check(f"asian gap {mdl}", abs(ap.loc[mdl].model_gap - gap) < 0.001 and intex(txt))

rel = A("relationship_descriptive.csv").set_index("model")
check("relationship +0.88 pooled",
      abs(rel.loc["POOLED"]["diff"] - 0.88) < 0.005 and intex("0.88 points above married"))
check("relationship per model",
      intex("GLM-4.7 +1.57, Gemini-3-Flash +1.30, DeepSeek-V3 +0.43, GPT-4o-mini +0.23"))

# ---- trans elevation ----------------------------------------------------------------------
te = A("trans_elevation_per_model.csv")
tp = te[te.model == "POOLED"].set_index("contrast")
tw = tp.loc["Transgender Woman - Cisgender Woman"]; tm = tp.loc["Transgender Man - Cisgender Man"]
gpt_min = te[(te.model == "openai/gpt-4o-mini")]["diff"].min()
gem_max = te[(te.model == "google/gemini-3-flash-preview")]["diff"].max()
check("trans per-model span +0.8..+6.2 d 0.40..1.72",
      abs(gpt_min - 0.803) < 0.005 and abs(gem_max - 6.159) < 0.005
      and intex("+0.8 points in GPT-4o-mini to +6.2 in Gemini-3-Flash",
                "those are 0.40 and 1.72; the scaling here is model-side by necessity"))

# ---- covariance ---------------------------------------------------------------------------
cov = A("covariance_divergence_GOLD.csv")
def frob_of(a, b):
    r = cov[((cov.group_a == a) & (cov.group_b == b)) | ((cov.group_a == b) & (cov.group_b == a))]
    return r.iloc[0]
tw_cm = frob_of("Cisgender Man", "Transgender Woman")
check("cov TW 1.14 / TM 1.06 / CW 0.32",
      abs(tw_cm.frobenius - 1.137) < 0.005 and intex("Transgender women diverge from cisgender men by 1.14 and transgender men by 1.06,")
      and intex("against 0.32 for cisgender women"))

# ---- variance ratios (descriptive) --------------------------------------------------------
pg = A("per_group_variance_GOLD.csv")
pooled = pg[pg.model == "POOLED"].set_index("group")
check("SD ratios low 0.67 asian 1.12",
      abs(pooled.loc["Low"].sd_ratio - 0.669) < 0.005 and abs(pooled.loc["Asian"].sd_ratio - 1.122) < 0.005
      and intex("(ratio 0.67, with SES cells spanning 0.67 to 0.86)", "the widest (1.12)"))
pmv = A("per_model_variance_GOLD.csv").set_index("model").mean_stereotype_index
check("per-model SI 0.46/0.53/1.04/1.05",
      abs(pmv["deepseek/deepseek-chat-v3"] - 0.456) < 0.005 and abs(pmv["openai/gpt-4o-mini"] - 0.532) < 0.005
      and abs(pmv["z-ai/glm-4.7"] - 1.039) < 0.005 and abs(pmv["google/gemini-3-flash-preview"] - 1.047) < 0.005
      and intex("DeepSeek-V3 (0.46) and GPT-4o-mini (0.53)", "GLM-4.7 (1.04) and Gemini-3-Flash (1.05)"))

# ---- restored analyses: refusals, fracture, per-instrument, transitions --------------------
ref = A("refusal_accounting.csv")
check("refusals 0 / 28,800",
      ref.refusals.sum() == 0 and ref.errors.sum() == 0 and ref.n_rows.sum() == 28800
      and intex("no model refused an assessment and none returned a malformed response"))
check("one clipped row", ref.out_of_range_phq8.sum() == 1 and intex("clipped to the instrument maximum"))

fr = A("stochastic_fracture.csv").set_index(["condition", "dimension"])
fd = A("stochastic_fracture_detail.csv")
fdc = fd[fd.condition == "clinical"].set_index(["dimension", "group"]).mean_within_cell_sd

pin = A("per_instrument_stability.csv").set_index(["condition", "instrument"])
check("per-instrument PHQ/GAD/AUDIT",
      abs(pin.loc[("clinical", "PHQ8")].flip_prob_pct - 35.40) < 0.05
      and abs(pin.loc[("clinical", "GAD7")].flip_prob_pct - 35.21) < 0.05
      and abs(pin.loc[("clinical", "AUDIT")].flip_prob_pct - 48.10) < 0.05
      and abs(pin.loc[("narrative", "AUDIT")].flip_prob_pct - 46.98) < 0.05
      and intex("35.4\\% and 35.2\\% flip probability under clinical framing, 32.2\\% and 32.5\\%",
                "48.1\\% and 47.0\\%"))

tm = pd.read_csv(os.path.join(BASE, "analysis", "transition_matrix.csv"), index_col=0)
diag, tot = int(np.trace(tm.values)), int(tm.values.sum())
check("transition matrix 9,121 / 63.34",
      diag == 9121 and tot == 14400 and abs(diag / tot * 100 - 63.34) < 0.01
      and intex("9,121 of 14,400 pairs (63.34\\%)"))
check("cross-condition boundary traffic 1,724 / 1,222",
      int(tm.iloc[2, 1]) == 1724 and int(tm.iloc[1, 2]) == 1222
      and intex("1,724 cases moving from moderate under clinical framing to mild under narrative framing and 1,222 moving the other way"))
# within-run transition matrix: the object Section 6.2's operational claim needs
wtm = pd.read_csv(os.path.join(BASE, "analysis", "transition_matrix_within_clinical.csv"), index_col=0)
wd, wt = int(sum(wtm.iloc[i, i] for i in range(5))), int(wtm.values.sum())
disc = wt - wd
for r in range(5):
    vals = " & ".join(f"{int(v):,}" for v in wtm.iloc[r])
    check(f"within-run row {r+1}", vals in tex, vals)
for r, row in enumerate(tm.index):
    vals = " & ".join(f"{int(v):,}" for v in tm.iloc[r])
    check(f"transition row {r+1}", vals in tex, vals)

# ---- SES covariance (restored) --------------------------------------------------------------
ses = cov[cov.dimension == "ses"] if "dimension" in (cov := A("covariance_divergence_GOLD.csv")).columns else None
sesx = ses.set_index(["group_a", "group_b"])
check("SES covariance 1.23 / 0.88 / 0.63",
      abs(sesx.loc[("Low", "High")].frobenius - 1.232) < 0.005
      and abs(sesx.loc[("Low", "Middle")].frobenius - 0.881) < 0.005
      and abs(sesx.loc[("Middle", "High")].frobenius - 0.629) < 0.005
      and intex("low-SES against high-SES personas at 1.23, with low against middle at 0.88 and middle against high at 0.63"))
# ratio band: gender/SES contrasts vs racial range
race_d = cov[cov.dimension == "race"].frobenius
big = [sesx.loc[("Low", "High")].frobenius,
       float(cov[(cov.group_a == "Cisgender Man") & (cov.group_b == "Transgender Woman")].frobenius.iloc[0]),
       float(cov[(cov.group_a == "Cisgender Man") & (cov.group_b == "Transgender Man")].frobenius.iloc[0])]
lo_r, hi_r = min(big) / race_d.max(), max(big) / race_d.min()
check("race covariance incl Multiracial 0.19",
      intex("Asian personas diverge by 0.37, Black by 0.24, Hispanic by 0.23, and Multiracial by 0.19"))

# ---- singleton x condition ------------------------------------------------------------------
check("singleton condition interaction 1.64 -> 0.95 / GPT flat",
      intex("falls from +1.64 under clinical intake to +0.95", "flat at +0.24 and +0.21"))

# ---- claims that previously had NO gate coverage -------------------------------------------
cisr = A("bias_residuals_CISONLY.csv").set_index("group")
poolr = A("bias_residuals_GOLD.csv").set_index("group")
gaps = [poolr.loc[g].residual - cisr.loc[g].residual
        for g in ["White", "Black", "Asian", "Hispanic", "Low", "Middle", "High"]]
check("pooled-vs-cis marginal gap 1.2 to 1.8",
      round(min(gaps), 1) == 1.2 and round(max(gaps), 1) == 1.8
      and intex("1.2 to 1.8 points above their cisgender-only counterparts"),
      f"{min(gaps):.2f}-{max(gaps):.2f}")
ccx2 = A("condition_contrast.csv").set_index(["instrument", "model"])
ratio = ccx2.loc[("phq8_total", "deepseek/deepseek-chat-v3")]["diff"] / cisr.residual.min()
check("framing shift is a quarter of smallest residual",
      0.22 < ratio < 0.30 and intex("is a quarter of the smallest calibration residual"),
      f"ratio {ratio:.3f}")
wr2 = A("within_run_stability.csv"); wr2 = wr2[wr2.model == "ALL"].set_index("condition")
check("within-run MADs 1.80 / 1.67 (like-for-like vs 1.89)",
      abs(wr2.loc["clinical"].mean_pairwise_abs_diff - 1.8011) < 0.005
      and abs(wr2.loc["narrative"].mean_pairwise_abs_diff - 1.6678) < 0.005
      and intex("mean absolute differences between generations of 1.80 and 1.67 points"))
check("Patel gate stated with the one 0.04 SD",
      intex("to within 0.02 on every mean and every standard deviation but one, the Asian standard deviation, which lands at 0.04"))

# ---- corrected inference: fair null + severity-controlled dispersion -----------------------
fair = A("covariance_fair_null.csv")
small = fair[(fair.dimension.isin(["race", "relationship"]))
             | ((fair.dimension == "gender")
                & ~(fair.group_a.str.startswith("Cis") & fair.group_b.str.startswith("Trans")))]
big = fair[((fair.dimension == "gender")
            & fair.group_a.str.startswith("Cis") & fair.group_b.str.startswith("Trans"))
           | (fair.dimension == "ses")]
tc = big[big.dimension == "gender"].matched_frobenius
sesb = big[big.dimension == "ses"].matched_frobenius
lo_r, hi_r = big.matched_frobenius.min() / small.matched_frobenius.max(), big.matched_frobenius.max() / small.matched_frobenius.min()
wcat = fair[(fair.dimension == "gender")
            & ~(fair.group_a.str.startswith("Cis") & fair.group_b.str.startswith("Trans"))]

fs = A("fracture_stratified.csv").set_index(["condition", "dimension"])
fm = A("fracture_matched.csv").set_index(["condition", "contrast"])
tcl = fm.loc[("clinical", "transgender vs cisgender")]
tnr = fm.loc[("narrative", "transgender vs cisgender")]
slh_c = fm.loc[("clinical", "Low vs High SES")]; slh_n = fm.loc[("narrative", "Low vs High SES")]

# ---- raw covariance values quoted in Section 8 ---------------------------------------------
rawc = A("covariance_divergence_GOLD.csv")
def rawv(a, b):
    r = rawc[((rawc.group_a == a) & (rawc.group_b == b)) | ((rawc.group_a == b) & (rawc.group_b == a))]
    return float(r.frobenius.iloc[0])
check("raw race + relationship distances",
      abs(rawv("Asian", "White") - 0.374) < 0.005 and abs(rawv("Black", "White") - 0.239) < 0.005
      and abs(rawv("Hispanic", "White") - 0.229) < 0.005 and abs(rawv("Multiracial", "White") - 0.193) < 0.005
      and abs(rawv("Single", "Married") - 0.251) < 0.005
      and intex("Asian personas diverge by 0.37, Black by 0.24, Hispanic by 0.23, and Multiracial by 0.19, and single against married personas by 0.25"))
check("raw gender + SES distances",
      abs(rawv("Cisgender Man", "Transgender Woman") - 1.137) < 0.005
      and abs(rawv("Cisgender Man", "Transgender Man") - 1.061) < 0.005
      and abs(rawv("Cisgender Man", "Cisgender Woman") - 0.315) < 0.005
      and abs(rawv("Low", "High") - 1.232) < 0.005 and abs(rawv("Low", "Middle") - 0.881) < 0.005
      and abs(rawv("Middle", "High") - 0.629) < 0.005
      and intex("Transgender women diverge from cisgender men by 1.14 and transgender men by 1.06, against 0.32 for cisgender women",
                "low-SES against high-SES personas at 1.23, with low against middle at 0.88 and middle against high at 0.63"))

# ---- raw (uncontrolled) dispersion statistics quoted in Section 6.3 -------------------------
frr = A("stochastic_fracture.csv").set_index(["condition", "dimension"])
check("raw KW gender 57.55 / 53.15",
      abs(frr.loc[("clinical", "gender")].H - 57.55) < 0.02
      and abs(frr.loc[("narrative", "gender")].H - 53.15) < 0.02
      and intex("(Kruskal-Wallis H = 57.55 clinical, 53.15 narrative)"))
check("raw KW ses 32.23 / 19.65",
      abs(frr.loc[("clinical", "ses")].H - 32.23) < 0.02
      and abs(frr.loc[("narrative", "ses")].H - 19.65) < 0.02 and intex("(H = 32.23, 19.65)"))
check("raw KW race/relationship null",
      frr.loc[("clinical", "race")].p > .05 and frr.loc[("clinical", "relationship")].p > .05
      and frr.loc[("narrative", "race")].p > .05 and frr.loc[("narrative", "relationship")].p > .05
      and intex("(H = 2.76 and 2.45 clinical, both p \\textgreater{} 0.05, same under narrative framing)"))
fdd = A("stochastic_fracture_detail.csv")
fdc = fdd[fdd.condition == "clinical"].set_index(["dimension", "group"]).mean_within_cell_sd
check("raw driving SDs 1.87 / 1.38 / 1.51 / 1.85 / 1.52",
      abs(fdc[("gender", "Transgender Woman")] - 1.8769) < 0.005
      and abs(fdc[("gender", "Cisgender Man")] - 1.3813) < 0.005
      and abs(fdc[("gender", "Cisgender Woman")] - 1.5067) < 0.005
      and abs(fdc[("ses", "Low")] - 1.8474) < 0.005 and abs(fdc[("ses", "Middle")] - 1.5203) < 0.005
      and intex("mean within-cell SD of 1.87 sits above cisgender men at 1.38 and cisgender women at 1.51",
                "low-SES cells at 1.85 against 1.52 for middle-SES"))
check("Table 4 severity spans",
      intex("+3.77 to +4.24", "+2.75 to +5.48", "+2.73 to +3.36 (descr.)", "+0.88 (descr.)"))

# ---- cell-level inference (Sections 3.5, 5, 7, 9) -------------------------------------------
rcl = A("residuals_cell_level.csv").set_index("group")
check("residual SE inflation 5.2-6.8x disclosed",
      round(rcl.se_inflation.min(), 1) == 5.2 and round(rcl.se_inflation.max(), 1) == 6.8
      and intex("by a factor of five to seven on", "60-generation model-by-cohort unit"))
check("all cell-level residual CIs exclude zero", bool((rcl.resid_ci_lo > 0).all()))
check("unit of analysis stated",
      intex("The unit of analysis is the design cell", "480 per condition",
            "$d_{\\mathrm{pop}}$"))
# Previously asserted that the omission was disclosed. The standard errors are now computed, so
# the check asserts the design variables are named and the resulting magnitude is reported.
_ds = A("anchor_design_se.csv")
check("anchor sampling error computed, not excluded",
      intex("SDMVSTRA") and intex("SDMVPSU")
      and intex(f"largest is {_ds.se_design.max():.3f} points")
      and "anchor sampling error is therefore excluded" not in tex,
      f"max design SE {_ds.se_design.max():.4f}, widest interval {(_ds.ci_hi - _ds.ci_lo).max():.3f}")

ccl = A("condition_contrast_cell_level.csv").set_index(["instrument", "model"])
a = ccl.loc[("phq8_total", "ALL")]
check("condition pooled cell-level +0.32 t 7.29",
      abs(a["diff"] - 0.3242) < 1e-4 and abs(a.t_cell - 7.293) < 0.01
      and intex("0.32 points (95\\% CI 0.24 to 0.41, paired t = 7.29 over 480 design cells, dz = 0.33)"))
check("condition GAD/AUDIT cell-level t 7.18 / 2.84",
      abs(ccl.loc[("gad7_total", "ALL")].t_cell - 7.178) < 0.01
      and abs(ccl.loc[("audit_total", "ALL")].t_cell - 2.842) < 0.01
      and intex("(+0.32, t = 7.18)", "(+0.07, t = 2.84)"))
ds = ccl.loc[("phq8_total", "deepseek/deepseek-chat-v3")]
gp = ccl.loc[("phq8_total", "openai/gpt-4o-mini")]
check("DeepSeek +0.74 t 14.38 dz 1.31",
      abs(ds["diff"] - 0.7414) < 1e-3 and abs(ds.t_cell - 14.380) < 0.01 and abs(ds.dz - 1.313) < 0.01
      and intex("+0.74 points under clinical framing (t = 14.38, dz = 1.31)"))
check("GPT-4o-mini null at cell level",
      (not bool(gp.significant)) and abs(gp.t_cell + 1.476) < 0.01 and abs(gp.p_cell - 0.1425) < 0.001
      and intex("its point estimate is $-$0.10 and its interval spans zero ($-$0.23 to +0.03, t = $-$1.48, p = 0.14, ledger T29)")
      and intex("GPT-4o-mini shows no detectable framing effect at all"))
check("no sign-reversal claim survives anywhere",
      "reverses sign" not in tex and "sign reversal" not in tex)

tcl = A("trans_elevation_cell_level.csv").set_index("contrast")
tw = tcl.loc["Transgender Woman - Cisgender Woman"]; tm = tcl.loc["Transgender Man - Cisgender Man"]
check("trans elevation cell-level CIs and t",
      abs(tw["diff"] - 2.728) < 0.005 and abs(tw.t_cell - 7.481) < 0.01
      and abs(tm["diff"] - 3.358) < 0.005 and abs(tm.t_cell - 9.509) < 0.01
      and intex("(+2.73, 95\\% CI +2.01 to +3.44 across cells, t = 7.48)",
                "(+3.36, +2.67 to +4.05, t = 9.51)"))

led2 = A("fdr_ledger.csv").set_index("test_id")
check("ledger 33 of 34, T29 the exception",
      len(led2) == 34 and int(led2.significant_q05.sum()) == 33
      and not bool(led2.loc["T29"].significant_q05)
      and intex("Thirty-three of the 34 clear q \\textless{} 0.05", "All but T29 clear"))
check("ledger tests are cell level",
      bool(led2.test_type.str.contains("cell means").any())
      and intex("Every test uses the design cell as its unit"))

# ---- claims previously uncovered by any check ------------------------------------------------
check("elevated defined as PHQ-8 >= 10",
      intex("We call a generation elevated when its PHQ-8 total reaches 10", "9,590 of the 28,800 generations qualify"))
check("gateway 257 of 9,590 = 2.68",
      gti["gateway_violations"] == 257 and gti["gateway_elevated_n"] == 9590
      and intex("257 of those 9,590 violate the rule, a rate of 2.68"))
check("PIR stated as policy cutpoints not terciles",
      intex("used in federal program eligibility and NCHS reporting, and they do not divide the sample into equal thirds")
      and "poverty-income-ratio terciles" not in tex)
check("persona/anchor construct mismatch disclosed",
      intex("a Medicaid-enrolled persona sits at or below the 1.30 cutpoint that defines the low stratum"))
check("Asian window disclosed",
      intex("The Asian anchor therefore rests on 2011--2018 while the others rest on 2005--2018"))
check("condition label not circularly validated",
      intex("A severity comparison could not establish the labels in any case, since the labels are "
            "later used to test severity")
      and "the mapping is confirmed by the per-run means it implies" not in tex
      # The assignment must rest on evidence independent of severity. The three bases are batch
      # identifiers, export schema and collection window; the ordinal that names the last of them
      # moved from "fourth" to "third" when the design's own template count stopped being counted
      # as a basis, so the check names the bases rather than their position in a list.
      and intex("batch identifier") and intex("export schema")
      and intex("Collection time is a third basis"))
check("NHIS access argument replaces blanket absence",
      intex("The National Health Interview Survey administered the full PHQ-8 to all 27,651 sample adults in 2022",
            "not available through the Center either, following Executive Order 14168")
      and "carry no gender-identity field" not in tex)

# ---- severity-controlled inference (Sections 6.3, 8, 10) -----------------------------------
fair = A("covariance_fair_null.csv")
isbig = lambda r: r.dimension == "ses" or (r.dimension == "gender"
                                           and r.group_a.startswith("Cis") and r.group_b.startswith("Trans"))
fair["big"] = fair.apply(isbig, axis=1)
big, small = fair[fair.big].matched_frobenius, fair[~fair.big].matched_frobenius
check("fair null: all 14 contrasts above",
      len(fair) == 14 and fair.above_fair_null.all()
      and intex("all fourteen contrasts exceed their nulls",
                "all fourteen cohort contrasts in the design diverge detectably"))
check("small band 0.14-0.21",
      round(small.min(), 2) == 0.14 and round(small.max(), 2) == 0.21
      and intex("narrow band from 0.14 to 0.21"), f"{small.min():.4f}-{small.max():.4f}")
tc = fair[fair.big & (fair.dimension == "gender")].matched_frobenius
ses = fair[fair.dimension == "ses"].matched_frobenius
check("trans-cis band 0.41-0.54",
      round(tc.min(), 2) == 0.41 and round(tc.max(), 2) == 0.54
      and intex("run 0.41 to 0.54", "0.41--0.54"), f"{tc.min():.4f}-{tc.max():.4f}")
check("SES band 0.37-0.80",
      round(ses.min(), 2) == 0.37 and round(ses.max(), 2) == 0.80
      and intex("socioeconomic contrasts 0.37 to 0.80", "0.37--0.80"), f"{ses.min():.4f}-{ses.max():.4f}")
lo_r, hi_r = big.min() / small.max(), big.max() / small.min()
rb = fair[fair.dimension == "race"].matched_frobenius
check("race band 0.17-0.21",
      round(rb.min(), 2) == 0.17 and round(rb.max(), 2) == 0.21
      and intex("racial contrasts land at 0.17 to 0.21", "0.17--0.21"), f"{rb.min():.4f}-{rb.max():.4f}")
rel = float(fair[fair.dimension == "relationship"].matched_frobenius.iloc[0])
check("relationship 0.14", round(rel, 2) == 0.14 and intex("relationship contrast at 0.14", "& 0.14 (0.14) & no effect"))

# ---- cell-centred contrasts: the decomposition Table 6 and Table 7 now carry ------------------
# Centring raises the noise floor along with the statistic, so the claim the paper makes is about
# excess over each contrast's own recomputed null, not about the raw centred distance. Both are
# asserted here because the table prints the distance and the prose argues from the excess.
cen = A("covariance_centred_contrasts.csv")
cen_big = cen[cen.large]
cen_small = cen[~cen.large]
c_tc = cen_big[cen_big.dimension == "gender"].centred_frobenius
c_ses = cen_big[cen_big.dimension == "ses"].centred_frobenius
c_race = cen[cen.dimension == "race"].centred_frobenius
c_rel = float(cen[cen.dimension == "relationship"].centred_frobenius.iloc[0])
check("centred: all 14 above their own centred null",
      len(cen) == 14 and bool((cen.centred_frobenius > cen.centred_null_p95).all())
      and intex("All fourteen exceed their nulls on both statistics"))
check("centred trans-cis band 0.25-0.46",
      round(c_tc.min(), 2) == 0.25 and round(c_tc.max(), 2) == 0.46
      and intex("(0.25--0.46)", "from 0.41--0.54 to 0.25--0.46"), f"{c_tc.min():.4f}-{c_tc.max():.4f}")
check("centred SES band 0.43-0.81",
      round(c_ses.min(), 2) == 0.43 and round(c_ses.max(), 2) == 0.81
      and intex("(0.43--0.81)"), f"{c_ses.min():.4f}-{c_ses.max():.4f}")
check("centred race band 0.20-0.24",
      round(c_race.min(), 2) == 0.19 and round(c_race.max(), 2) == 0.24
      and intex("(0.20--0.24)"), f"{c_race.min():.4f}-{c_race.max():.4f}")
check("centred relationship 0.14", round(c_rel, 2) == 0.14)
for tag in ("raw", "centred"):
    b, s = cen_big[f"{tag}_excess"], cen_small[f"{tag}_excess"]
    ordered = sum(x > y for x in b for y in s)
    med, ext = b.median() / s.median(), b.min() / s.max()
    check(f"{tag} excess bands and ratios",
          ordered == 49 and round(med, 2) == (3.97 if tag == "raw" else 3.89)
          and round(ext, 2) == (2.85 if tag == "raw" else 1.14),
          f"ordered {ordered}/49, median {med:.2f}, extremum {ext:.2f}")
check("centred excess bands in prose",
      round(cen_small.centred_excess.min(), 3) == 0.038
      and round(cen_small.centred_excess.max(), 3) == 0.093
      and round(cen_big.centred_excess.min(), 3) == 0.106
      and round(cen_big.centred_excess.max(), 3) == 0.652
      and intex("contrasts run 0.038 to 0.093 above their nulls and large ones 0.106 to 0.652",
                "the small contrasts run 0.038 to 0.093 and the large ones 0.106 to 0.652"))
check("centred narrowest margin is cis woman vs trans man",
      cen_big.loc[cen_big.centred_excess.idxmin(), "group_b"] == "Transgender Man"
      and intex("where cisgender women meet transgender men"))

# ---- ordinal factor refit: the estimator objection, run rather than conceded -----------------
ofa = A("ordinal_factor.csv")
ref = ofa[ofa.source != "simulated"].iloc[0]
op = ofa[(ofa.source == "simulated") & (ofa.model == "POOLED")]
om = ofa[(ofa.source == "simulated") & (ofa.model != "POOLED")]
clears = om.groupby("model").good_fit.sum()
check("DWLS: population 0.0421 -> 0.0411",
      round(ref.srmr_ml_pearson, 4) == 0.0421 and round(ref.srmr_dwls_polychoric, 4) == 0.0411
      and intex("SRMR 0.0421 to 0.0411"))
check("DWLS: pooled cohorts 0.093-0.125 -> 0.108-0.150",
      len(op) == 12 and round(op.srmr_ml_pearson.min(), 3) == 0.093
      and round(op.srmr_ml_pearson.max(), 3) == 0.125
      and round(op.srmr_dwls_polychoric.min(), 3) == 0.108
      and round(op.srmr_dwls_polychoric.max(), 3) == 0.150
      and intex("from 0.093--0.125 to 0.108--0.150"),
      f"{op.srmr_dwls_polychoric.min():.4f}-{op.srmr_dwls_polychoric.max():.4f}")
check("DWLS: within-model counts fall to 6 and 4",
      int(clears["google/gemini-3-flash-preview"]) == 6 and int(clears["z-ai/glm-4.7"]) == 4
      and int(clears["deepseek/deepseek-chat-v3"]) == 0 and int(clears["openai/gpt-4o-mini"]) == 0
      and intex("from nine of twelve to six in Gemini-3-Flash and from nine to four in GLM-4.7"),
      dict(clears))
check("DWLS: no simulated cohort improves under the better estimator",
      bool((ofa[ofa.source == "simulated"].srmr_dwls_polychoric
            > ofa[ofa.source == "simulated"].srmr_ml_pearson).all())
      and intex("Every simulated cohort gets worse"))
wc = fair[(fair.dimension == "gender") & ~fair.big].set_index("group_a").matched_frobenius
# This check used to require the literal string "sit at 0.16 and 0.21, inside the race band",
# which made it structurally incapable of noticing that the claim is false: 0.157 rounds to 0.16
# and sits BELOW the race band's 0.170 floor. Asserting a sentence cannot test a sentence. The
# relation is now computed from the receipts and the manuscript is checked against the result.
_small = fair[~fair.big].matched_frobenius
_wc_lo, _wc_hi = float(wc.min()), float(wc.max())
_in_race = rb.min() <= _wc_lo <= rb.max()
_in_small = _small.min() <= _wc_lo <= _small.max()
check("within-category gender contrasts placed correctly",
      round(wc["Cisgender Man"], 2) == 0.21 and round(wc["Transgender Woman"], 2) == 0.16
      and not _in_race and _in_small
      and intex("cisgender man against cisgender woman at 0.21, and transgender woman against transgender man at 0.16")
      and intex(f"brackets the race band ({rb.min():.2f}--{rb.max():.2f})")
      and intex(f"inside the small-contrast band ({_small.min():.2f}--{_small.max():.2f})"),
      f"wc {_wc_lo:.4f}-{_wc_hi:.4f} vs race {rb.min():.4f}-{rb.max():.4f} "
      f"vs small {_small.min():.4f}-{_small.max():.4f}")
twtm = fair[(fair.group_a == "Transgender Woman") & (fair.group_b == "Transgender Man")].iloc[0]
check("TW-TM narrow margin p=0.027",
      abs(twtm.p_perm - 0.027) < 0.002 and intex("whose margin over its null is narrow (p = 0.027)"))
check("null range 0.10-0.19",
      round(fair.fair_null_p95.min(), 2) == 0.10 and round(fair.fair_null_p95.max(), 2) == 0.19
      and intex("against nulls between 0.10 and 0.19"))

pm = A("covariance_fair_null_permodel.csv")
pmb = pm[pm.is_large]
nsep = sum(g[g.is_large].matched_frobenius.min() > g[~g.is_large].matched_frobenius.max()
           for _, g in pm.groupby("model"))
check("per-model: every large contrast above its own null",
      bool(pmb.above_fair_null.all()) and len(pmb) == 28
      and intex("Each large contrast exceeds its own null within every model separately"))
check("per-model: band separates in 2 of 4",
      nsep == 2 and intex("resolves cleanly in two of the four models at one quarter of the data"), f"{nsep}/4")

fs = A("fracture_stratified.csv").set_index(["condition", "dimension"])
check("stratified: gender survives, others do not",
      fs.loc[("clinical", "gender")].survives and fs.loc[("narrative", "gender")].survives
      and fs.loc[("clinical", "gender")].p_stratified < 0.001
      and not any(fs.loc[(c, d)].survives for c in ("clinical", "narrative")
                  for d in ("race", "ses", "relationship"))
      and intex("the gender statistic clears its null decisively in both conditions (p = 0.0005)",
                "(p = 0.97 clinical, 0.12 narrative)"))
fm = A("fracture_matched.csv").set_index(["condition", "contrast"])
check("matched: trans elevated +0.25 / +0.14",
      abs(fm.loc[("clinical", "transgender vs cisgender")].mean_sd_difference - 0.25) < 0.01
      and abs(fm.loc[("narrative", "transgender vs cisgender")].mean_sd_difference - 0.14) < 0.01
      and fm.loc[("clinical", "transgender vs cisgender")].verdict == "elevated"
      and fm.loc[("narrative", "transgender vs cisgender")].verdict == "elevated"
      and intex("+0.25 SD points noisier than severity-matched cisgender cells under clinical framing",
                "and +0.14 under narrative framing"))
check("matched: n=131 pairs", int(fm.loc[("clinical", "transgender vs cisgender")].n_matched_pairs) == 131
      and intex("131 matched pairs"))
lh = [fm.loc[(c, "Low vs High SES")] for c in ("clinical", "narrative")]
mh = [fm.loc[(c, "Middle vs High SES")] for c in ("clinical", "narrative")]
check("matched: SES reverses, all four sig",
      all(r.verdict == "lower" for r in lh + mh) and all(r.p < 0.01 for r in lh + mh)
      and abs(lh[0].mean_sd_difference + 0.38) < 0.01 and abs(lh[1].mean_sd_difference + 0.37) < 0.01
      and abs(mh[0].mean_sd_difference + 0.34) < 0.01 and abs(mh[1].mean_sd_difference + 0.32) < 0.01
      and intex("$-$0.38 and $-$0.37 SD points \\emph{less} variable than severity-matched high-SES cells",
                "middle-SES cells $-$0.34 and $-$0.32"))
# This check previously enumerated three contrasts and asserted the prose claim "race and
# relationship status show nothing." It omitted Hispanic vs White, which is significant under
# clinical framing, so the check confirmed a claim its own receipt contradicts. It now covers
# every race and relationship contrast and asserts only what holds: none is ELEVATED, and the one
# that reaches significance runs lower and does not replicate.
_rr = [k for k in fm.index.get_level_values(1).unique()
       if "White" in k or "Married" in k]
_elev = [(c, k) for c in ("clinical", "narrative") for k in _rr
         if fm.loc[(c, k)].verdict == "higher"]
_sig = [(c, k, fm.loc[(c, k)].mean_sd_difference, fm.loc[(c, k)].p)
        for c in ("clinical", "narrative") for k in _rr if fm.loc[(c, k)].verdict != "n.s."]
check("matched: no race or relationship contrast is elevated",
      not _elev and len(_sig) == 1 and _sig[0][1] == "Hispanic vs White"
      and intex("Neither race nor relationship status shows elevated dispersion under either "
                "control, in either condition")
      and intex("Hispanic against White cells sitting 0.14 SD points less variable"),
      f"elevated {_elev}; significant {[(c, k, round(d, 4)) for c, k, d, _ in _sig]}")
check("Table 4 stability cells",
      intex("+0.25 SD***", "$-$0.38 SD**\\textsuperscript{b}", "& no effect"))

# ---- cross-sectional structure synthesis (Section 10 / Table 4) ----------------------------
cv = A("covariance_divergence_GOLD.csv")
cv["floor_max"] = cv[["floor_p95_a", "floor_p95_b"]].max(axis=1)
cv["matched_over"] = cv.matched_frobenius > cv.floor_max
TRANS_CIS = [("Cisgender Man", "Transgender Woman"), ("Cisgender Man", "Transgender Man"),
             ("Cisgender Woman", "Transgender Woman"), ("Cisgender Woman", "Transgender Man")]
WITHIN = [("Cisgender Man", "Cisgender Woman"), ("Transgender Woman", "Transgender Man")]
sel = lambda a, b: cv[(cv.group_a == a) & (cv.group_b == b)].iloc[0]
# Table 4 severity spans must match the receipts they summarize
race_res = cis.loc[["White", "Black", "Asian", "Hispanic"]].residual
ses_res = cis.loc[["Low", "Middle", "High"]].residual
check("structure table SES span +2.75 to +5.48",
      abs(ses_res.min() - 2.75) < 0.005 and abs(ses_res.max() - 5.48) < 0.005
      and intex("+2.75 to +5.48"))
check("structure table trans span +2.73 to +3.36", intex("+2.73 to +3.36"))

# ---- ledger claims ------------------------------------------------------------------------
led = A("fdr_ledger.csv")
t29 = led[led.test_id == "T29"].iloc[0]

# ---- run provenance -----------------------------------------------------------------------
cmap = A("condition_runid_map.csv")
check("runid map 23 ids", len(cmap) == 23 and intex("23 batch identifiers"))

print()
if FAIL:
    print(f"GATE FAILED: {len(FAIL)} mismatches"); sys.exit(1)
print("GATE PASSED: every checked claim reconciles to receipts")
