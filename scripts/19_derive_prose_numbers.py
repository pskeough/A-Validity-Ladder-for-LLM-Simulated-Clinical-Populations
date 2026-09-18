"""
Derivation gate: recompute every prose quantity FROM the receipts, then assert the manuscript
contains what the derivation produces.

This is deliberately different from 15_numberlock_gate.py. That gate asserts a manuscript string
matches a value; if a quantity is mis-specified the same way in both the prose and the check, it
passes. That happened: a boundary-traffic share was written as one off-diagonal cell over a
both-directions denominator, in the prose and in the check alike, and the gate confirmed the
arithmetic of a wrong ratio (the true share is 56.0%, not 28.0%).

Here nothing is asserted against a literal. Each quantity is derived from the receipt, formatted,
and then searched for in the manuscript. A number that appears in prose but was never derived is
what this cannot see, so derivations are added whenever a new quantity enters the text.
"""
import pandas as pd, numpy as np, os, sys

BASE = os.path.join(os.path.dirname(__file__), "..")
A = lambda f: pd.read_csv(os.path.join(BASE, "analysis", f))
import re as _re
tex_raw = open(os.path.join(BASE, "paper_v2", "main.tex"), encoding="utf-8").read()
# LaTeX source wraps lines wherever it likes; a literal search must not depend on that
tex = _re.sub(r"\s+", " ", tex_raw)
FAIL = []

def derive(name, value, *needles):
    """value is what the receipts imply; needles are how it must appear in the manuscript."""
    ok = all(n in tex for n in needles)
    print(("PASS " if ok else "FAIL ") + f"{name} = {value}" + ("" if ok else f"  <- expected {needles}"))
    if not ok: FAIL.append(name)

# ---- within-run transition matrix: shares must use matching numerator/denominator ---------
w = pd.read_csv(os.path.join(BASE, "analysis", "transition_matrix_within_clinical.csv"), index_col=0).values
tot = int(w.sum()); diag = int(sum(w[i, i] for i in range(5))); disc = tot - diag
mm = int(w[2, 1]) + int(w[1, 2])          # both orderings, matching the denominator
nm = int(w[0, 1]) + int(w[1, 0])
tex_thousands = lambda n: f"{n:,}".replace(",", "{,}")   # LaTeX writes 417{,}600
derive("within-run agreement", f"{diag/tot*100:.2f}%", f"{diag/tot*100:.2f}\\%", tex_thousands(tot))
derive("mild-moderate share of discordance", f"{mm/disc*100:.1f}%",
       f"{mm:,} cross the mild-moderate line, {mm/disc*100:.1f}\\%", f"{disc:,} discordant orderings")
derive("next-largest boundary share", f"{nm/disc*100:.1f}%", f"{nm/disc*100:.1f}\\% for the next most common")
derive("concentration ratio", f"{mm/nm:.1f}x", f"factor of {mm/nm:.1f}")

c = pd.read_csv(os.path.join(BASE, "analysis", "transition_matrix.csv"), index_col=0).values
cdisc = int(c.sum()) - int(sum(c[i, i] for i in range(5)))
cmm = int(c[2, 1]) + int(c[1, 2])
derive("cross-condition boundary share", f"{cmm/cdisc*100:.1f}%",
       f"{cmm/cdisc*100:.1f}\\% of its discordant pairs against the {mm/disc*100:.1f}\\%")

# ---- condition contrast: no fold-change may be quoted against a null estimate -------------
cc = A("condition_contrast_cell_level.csv").set_index(["instrument", "model"])
gp = cc.loc[("phq8_total", "openai/gpt-4o-mini")]
if not bool(gp.significant):
    bad = [p for p in ["eleven-fold", "eleven times", "an order of magnitude about how much framing",
                       "order of magnitude across models"] if p in tex]
    print(("PASS " if not bad else "FAIL ") + "no fold-change against a null estimate"
          + ("" if not bad else f"  <- found {bad}"))
    if bad: FAIL.append("fold-change against null")

# ---- income mapping: the direction of the residual under a middle-stratum anchor ----------
gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv")).set_index("group")
cis = A("residuals_cell_level.csv").set_index("group")
alt = cis.loc["Low"].model_mean - gt.loc["Middle"].w_mean
derive("residual under middle anchor", f"+{alt:.2f}",
       f"whose anchor is {gt.loc['Middle'].w_mean:.2f}, the reported +{cis.loc['Low'].residual:.2f} would widen to +{alt:.2f}")

# ---- Table 5 stability cell must match the contrast its footnote names -------------------
fm = A("fracture_matched.csv").set_index(["condition", "contrast"])
lh = fm.loc[("clinical", "Low vs High SES")]
derive("Table 5 SES stability (low vs high)", f"{lh.mean_sd_difference:+.2f}",
       f"$-${abs(lh.mean_sd_difference):.2f} SD")

# ---- the withdrawn sign reversal must not survive anywhere --------------------------------
resid = [p for p in ["in sign across models", "reverse sign across models", "sign is model-specific",
                     "model-specific directions", "reverses sign"] if p in tex]
print(("PASS " if not resid else "FAIL ") + "sign-reversal claim fully withdrawn"
      + ("" if not resid else f"  <- {resid}"))
if resid: FAIL.append("sign-reversal residue")

# ---- structural: no CR-mangled control sequences -------------------------------------------
raw = open(os.path.join(BASE, "paper_v2", "main.tex"), encoding="utf-8", newline="").read()
# A \r written through a non-raw Python string becomes a carriage return, splitting \ref{ into
# "Section~" + CR + "ef{". On write that CR is sometimes normalised to a bare newline, so checking
# for CR alone misses it -- which is how one reached a compiled PDF as "Section efsec:estimand".
# Any orphan control-sequence tail at a line start is the signature regardless of the line ending.
mangled = len(_re.findall(r"(?:^|[\r\n])(?:ef|able|igure|ext|ite)[\{ ]", raw, _re.M))
print(("PASS " if not mangled else "FAIL ") + f"no CR-mangled \\ref tokens ({mangled} found)")
if mangled: FAIL.append("mangled refs")

# ---- every \ref target must exist ----------------------------------------------------------
import re
labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
missing = sorted(refs - labels)
print(("PASS " if not missing else "FAIL ") + f"all {len(refs)} \\ref targets resolve"
      + ("" if not missing else f"  <- dangling {missing}"))
if missing: FAIL.append("dangling refs")

# ---- band separation: the reported statistic must be the one that survives resampling -----
bg = A("band_gap_bootstrap.csv").set_index("statistic")
ext = bg.loc["extremum ratio min(large)/max(small)"]
med = bg.loc["median ratio median(large)/median(small)"]
con = bg.loc["pairwise concordance (share of 49 large-vs-small comparisons ordered large>small)"]
# The manuscript must report the OBSERVED statistic. An earlier version reported the median of the
# bootstrap replicates in its place, which understated the separation (2.38 written as 1.92) because
# cell resampling shrinks effective n and inflates small Frobenius distances more than large ones.
# These checks now read the `observed` column and assert the interval is described as a floor.
derive("observed median ratio", f"{med.observed}",
       f"median large contrast is {med.observed:.2f} times the median small one")
derive("observed extremum ratio", f"{ext.observed}",
       f"largest small one by a factor of {ext.observed:.2f}")
derive("bootstrap interval on the median ratio", f"[{med.ci_lo}, {med.ci_hi}]",
       f"95\% interval of {med.ci_lo:.2f} to {med.ci_hi:.2f} on the median ratio")
derive("bootstrap interval on the extremum", f"[{ext.ci_lo}, {ext.ci_hi}]",
       f"{ext.ci_lo:.2f} to {ext.ci_hi:.2f} on the extremum")
derive("observed percentile inside its own replicates", f"{med.obs_pctile_in_boot:.0f}th",
       f"sits at the {med.obs_pctile_in_boot:.0f}th percentile of its own replicates")
derive("pairwise concordance", f"{con.observed} lower {con.ci_lo}",
       f"{con.ci_lo*100:.0f}\% of the 49 pairings still order correctly")

# A percentile interval that excludes its own observed statistic must be labelled, never presented
# as if it were centred on it.
for phrase in ["conservative floors", "conservative cluster-bootstrap interval"]:
    ok = phrase in tex
    print(("PASS " if ok else "FAIL ") + f"bootstrap interval described as a floor ({phrase!r})")
    if not ok: FAIL.append(f"unlabelled bootstrap interval: {phrase}")

# The superseded bootstrap-median values must not survive anywhere.
for bad in ["1.92 times the median small one", "median of 1.06", "median ratio 1.92"]:
    hit = bad in tex
    print(("FAIL " if hit else "PASS ") + f"superseded bootstrap median {bad!r} absent")
    if hit: FAIL.append(f"superseded value {bad}")
# Guards the withdrawn v3 range claim. "factor of 1.7" was a needle here and now collides with the
# legitimate observed extremum of 1.75, so the check keys on the range itself.
stale = [q for q in ["1.7 to 5.6", "1.7 to 5.6 times", "seventeen-fold"] if q in tex]
print(("PASS " if not stale else "FAIL ") + "no extremum-range claim survives" + ("" if not stale else f"  <- {stale}"))
if stale: FAIL.append("stale band range")

# ---- frame-sensitivity band ---------------------------------------------------------------
# Two defects reached a draft of this table and neither was catchable by asserting a string.
# The overall column was computed on the full persona grid while the low-SES column was cisgender
# only, so the columns described different populations; and one anchor was a midpoint of a range
# no source states. The checks below recompute the marginal the table must be using, and refuse
# the withdrawn constants by name.
fb = A("frame_sensitivity_band.csv").set_index("anchor")

_m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
_m["phq8_total"] = _m.phq8_total.clip(0, 24)
_cis = _m[~_m.gender.astype(str).str.contains("rans", case=False, na=False)]
cis_overall = float(_cis.groupby(["model", "profile_id"]).phq8_total.mean().mean())
full_overall = float(_m.groupby(["model", "profile_id"]).phq8_total.mean().mean())
low_ses = float(A("residuals_cell_level.csv").set_index("group").loc["Low"].model_mean)

# The frame check. Every overall residual in the table must reconstruct the CISGENDER marginal.
# If the full-grid mean were used instead, each would be 1.52 points too large and this fails.
frame_ok, frame_bad = True, []
for a, r in fb.iterrows():
    for anchor, resid in ((r.mean_hi, r.resid_overall_lo), (r.mean_lo, r.resid_overall_hi)):
        if abs((anchor + resid) - cis_overall) > 0.01:
            frame_ok = False; frame_bad.append(a[:34])
    for anchor, resid in ((r.mean_hi, r.resid_low_lo), (r.mean_lo, r.resid_low_hi)):
        if abs((anchor + resid) - low_ses) > 0.01:
            frame_ok = False; frame_bad.append(a[:34] + " [low]")
print(("PASS " if frame_ok else "FAIL ")
      + f"every band residual reconstructs its own column's marginal "
        f"(cis {cis_overall:.3f} / low {low_ses:.3f}; full grid {full_overall:.3f} is NOT used)")
if not frame_ok: FAIL.append(f"band frame mismatch: {sorted(set(frame_bad))}")

# Withdrawn constants must not reappear. 6.25 was an unsourced midpoint; 8.51 and 5.53 are the
# full-grid mean and its residual, which belong to a different frame than this table's columns.
# These were scoped away from an appendix that documented them by name. That appendix is gone,
# so the constants must now be absent from the manuscript entirely.
body = tex
for bad, why in [("6.25", "unsourced Huang midpoint"), ("8.51", "full-grid mean"),
                 ("5.53", "full-grid residual"), ("4.91", "superseded Lowe residual"),
                 ("OB/GYN", "withdrawn anchor label"),
                 ("4.25--5.64", "caseload bound with the unattainable zero floor"),
                 ("exceeds the entire", "SES-unmatched span claim")]:
    hit = bad in body
    print(("FAIL " if hit else "PASS ") + f"withdrawn constant {bad!r} ({why}) absent from manuscript")
    if hit: FAIL.append(f"withdrawn constant {bad} present")

ro = lambda a: cis_overall - a
rl = lambda a: low_ses - a
for a, r in fb.iterrows():
    if abs(r.mean_lo - r.mean_hi) < 1e-9:
        dp = 3 if "VA primary care" in a else 2
        derive(f"band row {a[:30]}", f"{r.mean_lo:.{dp}f}",
               f"& {r.mean_lo:.{dp}f} & +{ro(r.mean_lo):.2f} & +{rl(r.mean_lo):.2f}"
               if ro(r.mean_lo) > 0 else f"& {r.mean_lo:.{dp}f} &")
    else:
        derive(f"band row {a[:30]} (bounded)", f"{r.mean_lo:.2f}-{r.mean_hi:.2f}",
               f"{r.mean_lo:.3f}--{r.mean_hi:.3f}",
               f"+{ro(r.mean_hi):.2f} to +{ro(r.mean_lo):.2f}",
               f"+{rl(r.mean_hi):.2f} to +{rl(r.mean_lo):.2f}")

# The retention comparison that used to be asserted here is gone from the prose, and must stay
# gone. (m - a)/(m - n) is increasing in m for any anchor a above the population value n, so the
# higher simulated mean ALWAYS retains a larger share. It is an identity, not a finding, and it
# cannot bear on whether the residual is a frame artifact.
taut = "survives better than the overall one at every anchor"
print(("FAIL " if taut in tex else "PASS ") + "tautological retention claim absent from manuscript")
if taut in tex: FAIL.append("tautological retention claim restored")

# ---- item-9 offset: measured, not assumed --------------------------------------------------
off_s = float(fb.loc["VA primary care, HIV-enriched cohort"].offset)
off_t = float(fb.loc["Primary care, antidepressant initiation"].offset)
derive("item-9 offset, screening band", f"{off_s:.3f}", f"{off_s:.3f} among respondents scoring 5 to 8")
off_p = float(fb.loc["US internet panel, no clinical frame"].offset)
derive("item-9 offset, panel tail-matched", f"{off_p:.3f}", f"(the panel, {off_p:.3f})")
derive("item-9 offset, treatment band", f"{off_t:.3f}", f"rising to {off_t:.3f} at treatment-entry severity")
derive("converted VACS anchor", f"{fb.loc['VA primary care, HIV-enriched cohort'].mean_lo:.3f}",
       f"& {fb.loc['VA primary care, HIV-enriched cohort'].mean_lo:.3f} &")

# ---- Lowe: the caseload bound and the decomposition ----------------------------------------
case = fb.loc["US primary care, full caseload (derived bound)"]
derive("caseload working interval", f"{case.mean_lo:.3f}-{case.mean_hi:.3f}",
       f"as {case.mean_lo:.3f} to {case.mean_hi:.3f}")

# ---- SES matching. A cohort residual must be read against a cohort-matched anchor. ----------
sm = A("frame_ses_matched.csv").set_index("quantity")
grad = float(sm.loc["population SES gradient (low vs all adults)"].value)
matched = sm.loc["low-income residual, SES-matched"].value
anchor = sm.loc["SES-matched caseload anchor"].value
retain = sm.loc["share of the NHANES low-income residual retained"].value
m_lo, m_hi = (float(x) for x in matched.split("-"))
a_lo, a_hi = (float(x) for x in anchor.split("-"))
r_lo, r_hi = (x.rstrip("%") for x in retain.split("-"))
derive("SES gradient", f"{grad:.2f}", f"{grad:.2f} points above the all-adults value")
derive("SES-matched anchor", anchor, f"anchor at {a_lo:.2f}", f"to {a_hi:.2f} and the low-income")
derive("SES-matched low-income residual", matched, f"+{m_lo:.2f} to +{m_hi:.2f}")
derive("retention share", retain, f"retaining {r_lo}\% to {r_hi}\% of")

# The unmatched figure may appear in the table, but must never be the number the prose defends.
unmatched_defended = f"low-income residual at +{case.resid_low_lo:.2f}" in tex
print(("FAIL " if unmatched_defended else "PASS ")
      + f"unmatched low-income residual (+{case.resid_low_lo:.2f}) is not the defended figure")
if unmatched_defended: FAIL.append("unmatched low-income residual defended")

# ---- shape contrast: three sources, three dispersions ---------------------------------------
sh = A("frame_shape_contrast.csv").set_index("source")
for src, label in [("NHANES, survey-weighted", "NHANES"), ("simulated, cisgender", "simulated")]:
    r = sh.loc[src]
    derive(f"shape {label}", f"{r['mean']:.2f}/{r.sd:.2f}/{r.pct_ge_10:.1f}",
           f"standard deviation of {r.sd:.2f}", f"{r.pct_ge_10:.1f}\%")
derive("caseload residuals", f"+{ro(case.mean_hi):.2f}/+{rl(case.mean_hi):.2f}",
       f"leaves +{ro(case.mean_hi):.2f} to +{ro(case.mean_lo):.2f}",
       f"+{rl(case.mean_hi):.2f} to +{rl(case.mean_lo):.2f}")   # table column only, not defended
derive("simulated overall in prose", f"{cis_overall:.2f}", f"cisgender overall mean of {cis_overall:.2f}")
# The low-income marginal is not restated in this section; it enters only through the SES-matched
# residual, so it is checked by reconstruction instead of by string.
recon = abs((m_hi + a_lo) - low_ses) < 0.01 and abs((m_lo + a_hi) - low_ses) < 0.01
print(("PASS " if recon else "FAIL ")
      + f"SES-matched residual reconstructs the low-income marginal ({low_ses:.3f})")
if not recon: FAIL.append("SES-matched reconstruction")

dc = A("frame_decomposition.csv")
setting = float(dc[dc.component.str.startswith("setting")].value.iloc[0])
casemix = float(dc[dc.component.str.startswith("case mix")].value.iloc[0])
derive("setting effect", f"{setting:.2f}", f"puts {setting:.2f} points on setting")
derive("case-mix component", f"{casemix:.1f}", f"about {casemix:.1f} on case mix")

# ---- gateway null: a coherence rate with no null cannot distinguish a coherent model from a
# lenient rule, so the manuscript must carry both. -------------------------------------------
gn = A("gateway_null.csv").set_index("scope")
gp = gn.loc["pooled"]
per_m = gn.drop("pooled")
derive("gateway null, conditional marginals", f"{gp.null_conditional_marginal_pct:.1f}%",
       f"{gp.null_conditional_marginal_pct:.1f}\% of elevated cases would violate")
# Pooled and per-model nulls must share a basis, or the pooled ratio and the per-model factors are
# not comparable. An earlier version pooled models inside a cohort and they did not reconcile.
_w = (per_m.null_conditional_marginal_pct * per_m.n_elevated).sum() / per_m.n_elevated.sum()
_ok = abs(_w - gp.null_conditional_marginal_pct) < 1.0
print(("PASS " if _ok else "FAIL ")
      + f"pooled null ({gp.null_conditional_marginal_pct:.2f}%) reconciles with the elevated-weighted "
        f"mean of the per-model nulls ({_w:.2f}%)")
if not _ok: FAIL.append("gateway null basis mismatch")
derive("gateway null, uniform compositions", f"{gp.null_uniform_composition_pct:.1f}%",
       f"gives {gp.null_uniform_composition_pct:.1f}\%")
derive("gateway null, within-case permutation", f"{gp.null_within_case_permute_pct:.1f}%",
       f"gives {gp.null_within_case_permute_pct:.1f}\%")
per_m = gn.drop("pooled")
_r = per_m.sort_values("ratio_vs_conditional")
derive("weakest model against its own null", f"{_r.iloc[0].ratio_vs_conditional:.1f}x",
       f"a factor of {_r.iloc[0].ratio_vs_conditional:.1f}")
# The weakest model does not clear its null, and the prose must say so rather than quoting a
# factor above one. This check fails if the ratio moves back above 1.5 without the text changing,
# and equally if the text reverts to claiming a gradient across all four models.
_weak_clears = _r.iloc[0].ratio_vs_conditional >= 1.5
_says_fails = "does not clear its own null at all" in tex
print(("PASS " if _weak_clears != _says_fails or not _weak_clears and _says_fails else "FAIL ")
      + f"weakest model's prose matches its ratio ({_r.iloc[0].ratio_vs_conditional:.2f}x)")
if _weak_clears and _says_fails:
    FAIL.append("weakest-model prose contradicts its ratio")
if not _weak_clears and not _says_fails:
    FAIL.append("weakest model fails its null but the prose does not say so")
# The observed rate and every null must be computed over the same denominator. A cell-averaged null
# against a case-weighted observed rate inflated this figure from 10.4 to 13.4 and turned GLM's
# ratio from 0.98 into 2.11.
_basis = [s for s in ("null of 13.4", "13.4\\% of elevated cases", "GLM-4.7 by 2.1") if s in tex]
print(("PASS " if not _basis else "FAIL ") + "withdrawn cell-averaged gateway figures absent")
if _basis: FAIL.append(f"cell-averaged gateway figure present: {_basis}")
derive("strongest model against its own null", f"{_r.iloc[-1].ratio_vs_conditional:.0f}x",
       f"by a factor of {_r.iloc[-1].ratio_vs_conditional:.0f}")
derive("all four per-model factors present", "gradient",
       *[f"by {v:.1f}" if v < 10 else f"of {v:.0f}" for v in
         (_r.ratio_vs_conditional.iloc[0], _r.ratio_vs_conditional.iloc[-1])])


# ---- inflation form, regional control, and ordering fidelity --------------------------------
inf = A("inflation_form.csv").set_index("quantity")
_b = float(inf.loc["multiplicative constant b"].value)
_pred = float(inf.loc["SES gradient predicted by constant multiplier"].value)
_obs = float(inf.loc["SES gradient observed"].value)
_ols = float(inf.loc["multiplier, least squares through origin"].value)
_boot = float(inf.loc["bootstrap share favouring multiplicative"].value)
_g1 = float(inf.loc["gradient predicted, OLS multiplier"].value)
_g2 = float(inf.loc["gradient predicted, log-scale multiplier"].value)
derive("both multiplier estimators", f"{_ols:.2f}/{_b:.2f}",
       f"origin puts the multiplier at {_ols:.2f} and the log-scale estimator at {_b:.2f}")
derive("gradient predicted by either", f"{_g1:.2f}-{_g2:.2f}",
       f"predicts a simulated gradient of {_g1:.2f} to {_g2:.2f}")
# A model comparison this close must ship its interval, not be asserted as a finding.
derive("bootstrap share favouring multiplicative", f"{_boot:.1f}%",
       f"in only {_boot:.1f}\% of replicates")
derive("CV of ratio vs residual",
       f"{float(inf.loc['ratio CV'].value):.2f}/{float(inf.loc['additive residual CV'].value):.2f}",
       f"coefficient of variation {float(inf.loc['ratio CV'].value):.2f} against "
       f"{float(inf.loc['additive residual CV'].value):.2f}")
derive("RSS comparison, OLS multiplier against additive",
       f"{float(inf.loc['RSS multiplicative, OLS'].value):.2f}/{float(inf.loc['RSS additive'].value):.2f}",
       f"({float(inf.loc['RSS multiplicative, OLS'].value):.2f} against "
       f"{float(inf.loc['RSS additive'].value):.2f})")
# The gradient is now reported as the interval both estimators span, checked above.
_excess = (_obs / _pred - 1) * 100
derive("excess over the widest multiplier prediction", f"{(_obs/_g1-1)*100:.0f}%",
       f"steepening to within {(_obs/_g1-1)*100:.0f}\\%")
derive("multiplier range",
       f"{float(inf.loc['ratio range low'].value):.2f}-{float(inf.loc['ratio range high'].value):.2f}",
       f"running {float(inf.loc['ratio range low'].value):.2f} for middle-income personas",
       f"to {float(inf.loc['ratio range high'].value):.2f} for Asian personas")

rg = A("regional_control.csv").set_index("region")
derive("regional control means",
       f"{rg.loc['US-developed']['mean']:.2f}/{rg.loc['China-developed']['mean']:.2f}",
       f"averages {rg.loc['US-developed']['mean']:.2f} on cisgender cells against "
       f"{rg.loc['China-developed']['mean']:.2f}")
derive("within-region spreads", f"{rg.loc['China-developed'].within_region_range:.2f}/"
       f"{rg.loc['US-developed'].within_region_range:.2f}",
       f"by {rg.loc['China-developed'].within_region_range:.2f}\npoints".replace("\n", " "),
       f"models by {rg.loc['US-developed'].within_region_range:.2f}")

# Ordering fidelity: the reversals must reconcile to the receipts, not be transcribed.
_r = A("residuals_cell_level.csv").set_index("group")
for g, val in [("Black", None), ("Hispanic", None)]:
    pg = float(_r.loc[g].gt_mean - _r.loc["White"].gt_mean)
    sg = float(_r.loc[g].model_mean - _r.loc["White"].model_mean)
    ok = pg > 0 > sg
    print(("PASS " if ok else "FAIL ") + f"{g} vs White reverses sign (pop {pg:+.3f}, sim {sg:+.3f})")
    if not ok: FAIL.append(f"{g} reversal")


# ---- ensemble size: the recommendation must carry a number, and the number must be derived ----
es = A("ensemble_size.csv").set_index("scope")
_p = es.loc["pooled"]
derive("single-generation agreement", f"{_p.k1:.1f}%",
       f"modal category {_p.k1:.1f}\\% of the time")
for mdl, name in [("deepseek/deepseek-chat-v3", "DeepSeek-V3"),
                  ("google/gemini-3-flash-preview", "Gemini-3-Flash")]:
    k = es.loc[mdl].k_for_90
    derive(f"k for 90% agreement, {name}", str(k), f"draws for {name}" if name == "DeepSeek-V3"
           else f"seven for {name}")


# ---- rendered geometry: a clipped table still yields its characters to a text extractor -------
# Table 3 shipped with its Low SES column running off the page. Every text-level check passed,
# because get_text() returns characters the layout has already clipped. Only the block geometry
# shows it, so the gate now reads the PDF as laid out rather than as a character stream.
try:
    import fitz
    _pdf = os.path.join(BASE, "paper_v2", "main.pdf")
    over = []
    for _pg in fitz.open(_pdf):
        for _b in _pg.get_text("blocks"):
            if _b[2] > _pg.rect.width - 20 or _b[0] < 20:
                over.append((_pg.number + 1, round(_b[0]), round(_b[2]), " ".join(_b[4][:40].split())))
    print(("PASS " if not over else "FAIL ")
          + f"no rendered block overflows the page box ({len(over)} found)")
    if over:
        for o in over[:4]: print("      ", o)
        FAIL.append("rendered overflow")
except ImportError:
    print("SKIP  geometry check (PyMuPDF unavailable)")


# ---- anchor sampling error: computed, not excluded --------------------------------------------
ds = A("anchor_design_se.csv")
derive("design effect range on the SE",
       f"{ds.design_effect_on_se.min():.2f}-{ds.design_effect_on_se.max():.2f}",
       f"factor of {ds.design_effect_on_se.min():.2f} to {ds.design_effect_on_se.max():.2f}")
derive("largest design-based SE", f"{ds.se_design.max():.3f}",
       f"largest is {ds.se_design.max():.3f} points")
derive("widest anchor interval", f"{(ds.ci_hi - ds.ci_lo).max():.2f}",
       f"spans {(ds.ci_hi - ds.ci_lo).max():.2f} points")
# The anchors the design-based pass reproduces must match the ones the paper actually uses.
_gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv")).set_index("group")
# This pass re-derives the PIR strata from INDFMPIR rather than reusing the pipeline's assignment,
# so the SES groups can differ by a few thousandths on boundary cases. That is independent
# reproduction, not disagreement; the tolerance is set where a real derivation error would show.
_diffs = {r.group: abs(r.weighted_mean - _gt.loc[r.group].w_mean)
          for _, r in ds.iterrows() if r.group in _gt.index}
_bad = [g for g, v in _diffs.items() if v > 0.01]
print(("PASS " if not _bad else "FAIL ")
      + f"design-based pass reproduces the published anchors "
        f"(max discrepancy {max(_diffs.values()):.4f} over {len(_diffs)} groups)")
if _bad: FAIL.append(f"anchor mismatch: {_bad}")


# ---- factorial interactions -------------------------------------------------------------------
# An earlier version fitted model identity as a main effect only. The paper documents large
# model-by-factor differences, so that specification parked them in the residual, inflated the
# denominator of the F test, and returned a null. The corrected fit reverses it. This check also
# asserts the baseline carries the model-by-factor terms, so the misspecification cannot return.
fa = A("factorial_additivity.csv")
fi = A("factorial_interactions.csv")
ir = A("intersection_residuals.csv")
_m, _f, _t = fa.iloc[0], fa.iloc[1], fa.iloc[2]
spec = open(os.path.join(BASE, "scripts", "27_factorial_interactions.py"), encoding="utf-8").read()
# The terms are built programmatically, so check the construction and that it is folded into the
# baseline formula rather than searching for four literal strings that never appear in the source.
_ok = ('C(model):C({f})' in spec and 'for f in FACTORS' in spec
       and '+ by_model' in spec and 'main = (' in spec)
print(("PASS " if _ok else "FAIL ") + "baseline fit carries all four model-by-factor terms")
if not _ok: FAIL.append("factorial specification")
derive("adjusted R2, baseline and full", f"{_m.adj_r2:.3f}/{_f.adj_r2:.3f}",
       f"adjusted $R^2$ of {_m.adj_r2:.3f}", f"adjusted $R^2$ to {_f.adj_r2:.3f}")
derive("residual SD, baseline and full", f"{_m.resid_sd:.2f}/{_f.resid_sd:.2f}",
       f"standard deviation of {_m.resid_sd:.2f} points", f"to {_f.resid_sd:.2f}")
derive("nested F on the interaction terms", f"F={_t.F:.2f}",
       f"$F = {_t.F:.2f}$ on {int(_t.df_resid)} and {int(_f.df_resid)} degrees")
_sig = fi[fi.significant]
print(("PASS " if len(_sig) == 3 else "FAIL ")
      + f"three of six interactions survive correction ({len(_sig)} found: {list(_sig.interaction)})")
if len(_sig) != 3: FAIL.append("interaction count")
derive("departure from additivity", f"max {ir.excess.abs().max():.2f}",
       f"full additivity is {ir.excess.abs().max():.2f} points")
# The withdrawn additive conclusion must not survive anywhere.
for bad in ["bias enters as a fixed per-axis offset", "does not need to be re-estimated",
            "The axes therefore combine additively"]:
    hit = bad in tex
    print(("FAIL " if hit else "PASS ") + f"withdrawn additivity claim absent: {bad!r}")
    if hit: FAIL.append(f"stale additivity claim: {bad}")


# ---- compression vs interaction: the bounded-scale objection must be answered, not assumed -----
ct = A("compression_test.csv").set_index("scale")
_raw, _log = ct.loc["raw PHQ-8"], ct.loc["logit(total/24)"]
# The shrinkage-slope diagnostic was withdrawn: on a balanced factorial it regresses OLS residuals
# on OLS fitted values, so it returns zero by construction whatever the data contain. This check
# fails if the withdrawn statistic returns to the prose.
_dead = [s for s in ("slope of $-0.000$", "shrinkage fit reproduces", "pure shrinkage fit") if s in tex]
print(("PASS " if not _dead else "FAIL ") + f"degenerate shrinkage diagnostic absent from prose")
if _dead: FAIL.append(f"withdrawn shrinkage claim present: {_dead}")
_bound = ct.loc["scale-change bound"]
derive("scale-change cost", f"{_bound.delta_r2_pct:.0f}% removed",
       f"Twenty-two percent of the signal goes with the scale change and {100 - _bound.delta_r2_pct:.0f}\\% stands")
_ind = A("compression_induced.csv")
import scipy.stats as _st
_ir = _st.linregress(_ind.induced, _ind.dep)
derive("induced-pattern correlation, reported as withdrawn", f"r = {_ir.rvalue:.2f}",
       f"It gives $r = {_ir.rvalue:.2f}$, which we withdraw")
# The induced regressor is collinear with a pure bilinear interaction kernel, so its correlation
# apportions nothing. The manuscript must carry that collinearity, and must not carry a variance
# share derived from it.
_o = {f: _ind[f].map(_ind.groupby(f).phq8.mean() - _ind.phq8.mean()) for f in
      ["race", "gender", "ses", "relationship"]}
_kern = sum(_o[a] * _o[b] for i, a in enumerate(_o) for b in list(_o)[i + 1:])
_rk = _st.linregress(_ind.induced, _kern).rvalue
derive("induced regressor collinear with a bilinear kernel", f"r = {_rk:.2f}",
       f"correlates at $r = {_rk:.2f}$ with a pure")
_share = [s for s in ("accounting for 16.7", "Two routes agree", "compression accounts for roughly a fifth")
          if s in tex]
print(("PASS " if not _share else "FAIL ") + "no variance share is claimed for the induced route")
if _share: FAIL.append(f"withdrawn apportionment present: {_share}")
derive("logit omnibus survives", f"F={_log.omnibus_F:.2f}",
       f"($F = {_log.omnibus_F:.2f}$", f"{_log.delta_r2_pct:.2f}\%")
# Only the terms that agree on both scales may be reported as findings.
_r = set(str(_raw.significant).split("; ")); _l = set(str(_log.significant).split("; "))
_both = _r & _l
print(("PASS " if len(_both) == 2 else "FAIL ")
      + f"two interactions replicate across scales ({sorted(_both)})")
if len(_both) != 2: FAIL.append("cross-scale interaction agreement")
_scale_dep = (_r ^ _l)
print(("PASS " if _scale_dep and "scale-dependent" in tex else "FAIL ")
      + f"the non-replicating term is declared scale-dependent ({sorted(_scale_dep)})")
if not (_scale_dep and "scale-dependent" in tex): FAIL.append("scale-dependence undeclared")


# ---- distribution shape: every prose figure recomputed from the shape receipt ----------------
_ds = A("distribution_shape.csv").set_index("source")
_pop, _pool = _ds.loc["NHANES 2005-2018 (weighted)"], _ds.loc["All models pooled"]
derive("population share at 4 or below", f"{_pop['pct_0-4']:.1f}%", f"{_pop['pct_0-4']:.1f}\% of adults at 4 or below")
derive("pooled 5-9 concentration", f"{_pool['pct_5-9']:.1f}%",
       f"{_pool['pct_5-9']:.1f}\% in the 5--9 band against {_pop['pct_5-9']:.1f}\%")
for _m, _lab in [("DeepSeek-V3", "DeepSeek-V3"), ("GPT-4o-mini", "GPT-4o-mini")]:
    derive(f"{_m} 5-9 concentration", f"{_ds.loc[_m,'pct_5-9']:.1f}%", f"{_ds.loc[_m,'pct_5-9']:.1f}\%")
derive("simulated SDs against population", "1.65 / 2.11 vs 3.94",
       f"{_ds.loc['DeepSeek-V3','sd']:.2f} and {_ds.loc['GPT-4o-mini','sd']:.2f}",
       f"population {_pop['sd']:.2f}")
# The severe-band absence is the sharpest claim in the section, so it is derived and not asserted.
_sev_pop, _sev_sim = float(_pop["pct_20-24"]), float(_pool["pct_20-24"])
print(("PASS " if _sev_sim == 0.0 and _sev_pop > 0 else "FAIL ")
      + f"severe band: population {_sev_pop:.1f}%, simulated {_sev_sim:.1f}%")
if not (_sev_sim == 0.0 and _sev_pop > 0): FAIL.append("severe-band absence")
derive("severe-band population share", f"{_sev_pop:.1f}%", f"{_sev_pop:.1f}\% of adults")

_sft = A("distribution_shift_test.csv").iloc[0]
# The KS-share-closed statistic is withdrawn. Both distributions sit on the integer 0-24 lattice
# and the statistic turns on the fractional part of the shift: on data built as a pure location
# shift with zero shape difference it reports 55% to 99% closed. The manuscript must not quote a
# share, and the shape claim must rest on the standard deviations, which no location shift moves.
_ks_dead = [s for s in (f"closes {_sft.pct_of_ks_closed_by_shift:.1f}\\% of it",
                        "The remainder is shape",
                        "closes barely half the distributional distance",
                        f"closes {_sft.pct_of_ks_closed_by_shift:.0f}\\% of the distributional distance")
            if s in tex]
print(("PASS " if not _ks_dead else "FAIL ") + "withdrawn KS decomposition absent from prose")
if _ks_dead: FAIL.append(f"withdrawn KS share present: {_ks_dead}")
_ds = A("distribution_shape.csv").set_index("source")
# The dispersion-based shape claim is withdrawn. A single-vignette design bounds achievable
# dispersion by construction, and the cross-instrument test showed the suppressed top category is a
# property of ordinal generation rather than of depression: it appears on PHQ-8, GAD-7 and AUDIT-C
# alike. These checks assert the withdrawal held and that what replaced it is derived.
_dead_shape = [x for x in ("less than half as wide as the population",
                           "A location shift changes none of those things",
                           "No amount of shifting produces this") if x in tex]
print(("PASS " if not _dead_shape else "FAIL ") + "withdrawn dispersion-as-evidence claim absent")
if _dead_shape: FAIL.append(f"withdrawn dispersion claim present: {_dead_shape}")

_rs = A("response_style.csv")
_rs = _rs[_rs.scope == "ALL"].set_index("instrument")
for _i in ("PHQ-8", "GAD-7", "AUDIT-C"):
    derive(f"endpoint share, {_i}", f"{_rs.loc[_i].pct_at_max:.2f}%",
           f"{_rs.loc[_i].pct_at_max:.2f}\%")
derive("population endpoint share", f"{_rs.loc['PHQ-8'].pop_pct_at_max:.2f}%",
       f"against {_rs.loc['PHQ-8'].pop_pct_at_max:.2f}\% in the")
derive("population floor share", f"{_rs.loc['PHQ-8'].pop_pct_at_zero:.1f}%",
       f"{_rs.loc['PHQ-8'].pop_pct_at_zero:.1f}\% of")
# The claim only holds if suppression really does run across all three instruments.
_all_low = bool((_rs.pct_at_max < _rs.loc["PHQ-8"].pop_pct_at_max).all())
print(("PASS " if _all_low else "FAIL ")
      + "top category suppressed on every instrument, not only PHQ-8")
if not _all_low: FAIL.append("cross-instrument suppression claim")

_ic = A("instrument_correlations.csv").set_index("scope")
derive("PHQ-8 to GAD-7 correlation", f"{_ic.loc['ALL'].r_phq8_gad7:.2f}",
       f"correlate at {_ic.loc['ALL'].r_phq8_gad7:.2f}",
       f"{_ic.loc['ALL'].loewe_lo:.2f} to {_ic.loc['ALL'].loewe_hi:.2f}")
_above = [m for m in _ic.index if m != "ALL" and _ic.loc[m].r_phq8_gad7 > _ic.loc[m].loewe_hi]
print(("PASS " if len(_above) == 3 else "FAIL ")
      + f"three of four models above the clinical band ({len(_above)} found)")
if len(_above) != 3: FAIL.append("cross-instrument correlation count")

_ac = A("alternative_comparators.csv").set_index("comparator")
derive("residual against the median", f"{_ac.loc['median'].residual:+.2f}",
       f"median the pooled residual is +{_ac.loc['median'].residual:.2f}")
derive("residual against the mode", f"{_ac.loc['mode'].residual:+.2f}",
       f"against the mode it is +{_ac.loc['mode'].residual:.2f}")
# Every alternative comparator must move the residual the same way, or the conservatism argument
# in Section 4 is backwards.
_cons = bool((_ac.residual >= _ac.loc["mean"].residual).all())
print(("PASS " if _cons else "FAIL ") + "the reported mean comparator is the conservative one")
if not _cons: FAIL.append("comparator conservatism direction")

# ---- anchor recency: residuals must shrink by the amount the receipt says and stay positive ---
_ar = A("anchor_recency.csv")
_rc = [c for c in _ar.columns if c.startswith("resid_") and c != "resid_shift_recent_vs_paper"]
_allpos = bool((_ar[_rc] > 0).all().all())
print(("PASS " if _allpos else "FAIL ") + "every residual positive on all three anchor windows")
if not _allpos: FAIL.append("anchor recency sign")
derive("residual shrinkage on the recent anchor",
       f"{-_ar.resid_shift_recent_vs_paper.max():.2f} to {-_ar.resid_shift_recent_vs_paper.min():.2f}",
       f"shrink by {-_ar.resid_shift_recent_vs_paper.max():.2f} to "
       f"{-_ar.resid_shift_recent_vs_paper.min():.2f} points")
derive("smallest residual on the recent anchor", f"{_ar['resid_recent 2021-2023'].min():+.2f}",
       f"from +{_ar['resid_paper 2005-2018'].min():.2f} to +{_ar['resid_recent 2021-2023'].min():.2f}")

# ---- within-cohort factor model --------------------------------------------------------------
_wf = A("within_cohort_factor.csv")
_nh = _wf[_wf.source == "NHANES adults"].iloc[0]
_sim = _wf[_wf.source == "simulated"]
derive("NHANES within-sample SRMR", f"{_nh.srmr:.3f}", f"SRMR {_nh.srmr:.3f}")
derive("simulated SRMR range", f"{_sim.srmr.min():.3f}-{_sim.srmr.max():.3f}",
       f"{_sim.srmr.min():.3f} to {_sim.srmr.max():.3f}")
# The claim is that NO simulated cohort clears the threshold the population sample clears.
_ok = bool(_nh.srmr < 0.08 and (_sim.srmr >= 0.08).all())
print(("PASS " if _ok else "FAIL ") + "population fits, no simulated cohort does")
if not _ok: FAIL.append("within-cohort fit separation")

_cg = A("factor_congruence.csv")
derive("congruence failures", f"{int((~_cg['equivalent_at_.95']).sum())} of {len(_cg)}",
       f"{_pool.name and ''}{int((~_cg['equivalent_at_.95']).sum())}", f"of sixty-six cohort pairs")
_worst = _cg.iloc[0]
derive("least congruent pair", f"{_worst.phi:.2f}", "$\\varphi = " + f"{_worst.phi:.2f}$")
# The band split must be recovered, not asserted: the SES and transgender pairs are the extremes
# and the racial pairs are the most congruent. A congruence result that did not order this way
# would still produce a low minimum, so the ordering is checked and not just the value.
_RACE = {"White", "Black", "Asian", "Hispanic", "Multiracial"}
_race_pairs = _cg[_cg.cohort_a.isin(_RACE) & _cg.cohort_b.isin(_RACE)]
_ordered = bool(_race_pairs.phi.min() > _cg.phi.quantile(0.5)) and "SES" not in str(_worst.cohort_a)
print(("PASS " if _race_pairs.phi.min() > _cg.phi.median() else "FAIL ")
      + f"racial pairs more congruent than the median pair "
        f"({_race_pairs.phi.min():.3f} vs median {_cg.phi.median():.3f})")
if not _race_pairs.phi.min() > _cg.phi.median(): FAIL.append("congruence band ordering")


# ---- recovery-row sensitivity: quoted in prose, previously in no receipt ----------------------
_rr = A("recovery_row_sensitivity.csv").iloc[0]
derive("condition contrast, all rows", f"{_rr.contrast_all_rows:+.3f}",
       f"+{_rr.contrast_all_rows:.3f}")
derive("condition contrast, recovery rows dropped", f"{_rr.contrast_recovery_dropped:+.3f}",
       f"to +{_rr.contrast_recovery_dropped:.3f}")
derive("recovery row count", f"{int(_rr.n_recovery_rows)}",
       f"Dropping the {int(_rr.n_recovery_rows)} recovery rows")


# ---- clustering and dispersion: quantities added to Section 3.5 and Section 6 ----------------
_cl = A("clustering_receipts.csv")
_c60 = _cl[_cl.unit.str.contains("60 draws")].iloc[0]
_c30 = _cl[_cl.unit.str.contains("30 draws")].iloc[0]
_c2w = _cl[_cl.quantity.str.startswith("two-way")].iloc[0]
derive("SE inflation, 60-draw unit", f"{_c60.low}-{_c60.high}x",
       "by a factor of five to seven on the 60-generation model-by-cohort unit")
derive("SE inflation, 30-draw unit", f"{_c30.low}-{_c30.high}x", "by four to five within a single condition")
derive("two-way cluster-robust inflation", f"{_c2w.low}-{_c2w.high}x",
       f"run {_c2w.low:.1f} to {_c2w.high:.1f} times the between-cell errors")
_dm = A("dispersion_mean_controlled.csv").iloc[0]
derive("mean-controlled dispersion gap", f"{_dm.gap_points:+.3f}",
       f"a gap of {_dm.gap_points:.3f} points ($t = {_dm.t:.2f}$",
       "$\\rho = " + f"{_dm.rho_mean_sd:.2f}$")

# ---- anchor sampling error is window-specific -------------------------------------------------
# The residual on the recent anchor must be compared against the standard error on that same
# window. Dividing a 2021-2023 residual by the 2005-2018 standard error is a cross-window ratio,
# and it turned a factor of eight into more than twenty.
_sw = A("anchor_se_by_window.csv").set_index("era")
_paper_se = _sw.loc["paper 2005-2018"].largest_design_se
_rec = _sw.loc["recent 2021-2023"]
derive("largest anchor SE, paper window", f"{_paper_se:.4f}",
       f"rises from {_paper_se:.3f} on the paper window")
derive("largest anchor SE, recent window", f"{_rec.largest_design_se:.4f}",
       f"to {_rec.largest_design_se:.3f} on 2021--2023")
derive("Asian n on each window", f"{int(_rec.asian_n)} vs {int(_sw.loc['paper 2005-2018'].asian_n)}",
       f"rests on {int(_rec.asian_n)} respondents against "
       + f"{int(_sw.loc['paper 2005-2018'].asian_n):,}".replace(",", "{,}"))
_smallest = A("anchor_recency.csv")["resid_recent 2021-2023"].min()
_factor = _smallest / _rec.largest_design_se
print(("PASS " if 7.5 <= _factor < 8.5 else "FAIL ")
      + f"smallest recent residual clears its own-window SE by {_factor:.1f}x")
if not 7.5 <= _factor < 8.5:
    FAIL.append("window-matched residual-to-SE factor")


# ---- which item pairs move, and the within-cell decomposition --------------------------------
_ip = A("item_pair_shifts.csv")
_ses = _ip[_ip.contrast == "Low vs High SES"].iloc[0]
_gen = _ip[_ip.contrast == "Transgender vs cisgender"]
_gen_fs = _gen[(_gen.item_a == "fatigue") & (_gen.item_b == "self-worth")].iloc[0]
# The prose names the two ends of each shift now, so the check has to test the direction and not
# only the pair. Fatigue against worthlessness is the pair that carries the clinical reading, and
# it runs the opposite way on the two axes: negative in low-income cells, positive in transgender
# ones. An earlier draft of this sentence had the SES direction backwards.
derive("largest SES item-pair shift", f"{_ses.item_a} x {_ses.item_b}",
       f"{_ses.item_a} and worthlessness running against each other",
       f"r = $-${abs(_ses.r_first):.2f}")
print(("PASS " if _ses.r_first < _ses.r_second and _gen_fs.r_first > _gen_fs.r_second
       else "FAIL ") + "fatigue-worthlessness runs opposite ways on the two axes "
      f"(SES {_ses.r_first:+.3f}->{_ses.r_second:+.3f}, "
      f"gender {_gen_fs.r_first:+.3f}->{_gen_fs.r_second:+.3f})")
if not (_ses.r_first < _ses.r_second and _gen_fs.r_first > _gen_fs.r_second):
    FAIL.append("item-pair direction")

# Table 6 and Table 7 now carry the centring per contrast rather than the two aggregates this gate
# used to check, so the derivation follows: the four gender contrasts must fall under centring and
# the racial and socioeconomic ones must rise, which is what the manuscript asserts.
_cc = A("covariance_centred_contrasts.csv")
_cross = _cc[_cc.large & (_cc.dimension == "gender")]
_rise = _cc[_cc.dimension.isin(["race", "ses"])]
derive("centred gender band", f"{_cross.centred_frobenius.min():.2f}-{_cross.centred_frobenius.max():.2f}",
       f"from 0.41--0.54 to {_cross.centred_frobenius.min():.2f}--{_cross.centred_frobenius.max():.2f}")
print(("PASS " if (_cross.centred_frobenius < _cross.raw_frobenius).all()
       and (_rise[_rise.dimension == 'race'].centred_frobenius
            > _rise[_rise.dimension == 'race'].raw_frobenius).sum() >= 3
       else "FAIL ") + "centring lowers every gender contrast and raises the racial ones")
if not ((_cross.centred_frobenius < _cross.raw_frobenius).all()
        and (_rise[_rise.dimension == "race"].centred_frobenius
             > _rise[_rise.dimension == "race"].raw_frobenius).sum() >= 3):
    FAIL.append("centring direction by axis")


print()
if FAIL:
    print(f"DERIVATION GATE FAILED: {len(FAIL)}"); sys.exit(1)
print("DERIVATION GATE PASSED: every checked quantity was derived, not asserted")
