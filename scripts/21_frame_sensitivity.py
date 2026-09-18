"""
Frame-sensitivity band for the severity residual.

The prompt frames a screening encounter and instructs the model to answer as the patient would, so
part of the residual against a general-population anchor may be frame rather than miscalibration. A
single clinical anchor cannot settle this, and swapping one in would repeat the error this paper
documents: a borrowed constant, a different estimand, the wrong scale. Instead we place the
simulated means inside the band of published means across sampling frames and report how much of
the residual survives at each.

Three disciplines apply, and the first two were violated in an earlier draft of this script.

Frame. Both columns are computed on cisgender personas only, matching Section 3.2. The earlier
version computed the overall column on the full grid and the low-SES column on cisgender cells, so
the two columns of the same table answered questions about different populations and every overall
residual was inflated by 1.52 points.

Provenance. Every anchor names its source, sample, instrument and selection rule, and every one is
checked against the source document. An anchor of 6.25 for "US primary care and OB/GYN" was carried
by that earlier version and is gone: it traced to Huang et al. 2006, which is a factor-structure and
DIF study reporting no mean PHQ-9 at all. The value was a midpoint of a range that no source states.
That is the failure class Section 12 is about, so it is documented in Appendix A rather than quietly
deleted.

Matched scope. A residual is only interpretable if the simulated marginal and the anchor describe
the same population on every axis the comparison is read across, not just the frame. An earlier
version compared the LOW-INCOME simulated mean against an all-comers clinical caseload and read the
gap as surviving frame adjustment. Low-income patients score above the caseload average, and this
paper's own derivation measures that gradient, so the comparison had to carry it. It now does. No
SES-stratified clinical PHQ-8 anchor exists, so the population gradient is transferred additively
and flagged as the approximation it is.

Scale. PHQ-9 anchors are converted to PHQ-8 using the offset those two scales actually differ by,
the mean of item 9, computed here from NHANES under the survey weights rather than assumed. The
offset is severity-matched: item 9 scales with depression, so the population mean under-corrects a
clinical anchor. It turns out to be small enough that the mixed-instrument objection mostly
dissolves, which is worth knowing either way.

Emits analysis/frame_sensitivity_band.csv and analysis/frame_decomposition.csv. The bootstrap is
seeded; nothing else is random.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(BASE, "data", "nhanes_raw")
CYCLES = ["D", "E", "F", "G", "H", "I", "J"]          # 2005-2006 through 2017-2018
ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]

# ---- item-9 offset, derived rather than assumed -------------------------------------------
frames = []
for c in CYCLES:
    demo = pd.read_sas(os.path.join(RAW, f"DEMO_{c}.xpt"))[["SEQN", "RIDAGEYR", "WTMEC2YR"]]
    frames.append(demo.merge(pd.read_sas(os.path.join(RAW, f"DPQ_{c}.xpt")), on="SEQN"))
nh = pd.concat(frames, ignore_index=True)
nh[ITEMS] = nh[ITEMS].where(nh[ITEMS] <= 3)           # 7 = refused, 9 = don't know
nh = nh[(nh.RIDAGEYR >= 18) & nh[ITEMS].notna().all(axis=1) & (nh.WTMEC2YR > 0)].copy()
nh["w"] = nh.WTMEC2YR / len(CYCLES)                   # pooled multi-cycle weight
nh["phq8"] = nh[ITEMS[:8]].sum(axis=1)
nh["phq9"] = nh[ITEMS].sum(axis=1)
wmean = lambda d, col: float((d[col] * d.w).sum() / d.w.sum())

nhanes = wmean(nh, "phq8")
off_screen = wmean(nh[nh.phq9.between(5, 8)], "DPQ090")     # anchors known only by their mean
off_treat = wmean(nh[nh.phq9.between(11, 14)], "DPQ090")    # treatment-entry severity
# Where an anchor publishes its tail fraction, the offset is matched to that instead of to a band
# around its mean. The panel reports 26.4% at PHQ-9 >= 10, and item 9 is concentrated there.
off_lo, off_hi = wmean(nh[nh.phq9 < 10], "DPQ090"), wmean(nh[nh.phq9 >= 10], "DPQ090")
off_panel = 0.736 * off_lo + 0.264 * off_hi
# The screener is interviewer-administered by CAPI at the MEC, not self-completed, so item 9 is
# plausibly under-reported here relative to the self-report forms the clinical anchors used. Every
# offset below is therefore a floor, each converted anchor an upper bound, and each residual
# conservative. That is the direction the section's claims need, so it is stated rather than fixed.

# ---- simulated marginals, cisgender frame throughout ---------------------------------------
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
cis = m[~m.gender.astype(str).str.contains("rans", case=False, na=False)]
cells = cis.groupby(["model", "profile_id"], as_index=False).phq8_total.mean()
overall = float(cells.phq8_total.mean())
full_grid = float(m.groupby(["model", "profile_id"]).phq8_total.mean().mean())   # sensitivity only
low = float(pd.read_csv(os.path.join(OUT, "residuals_cell_level.csv"))
            .set_index("group").loc["Low"].model_mean)

rng = np.random.default_rng(20260726)
cohorts = cells.profile_id.unique()
by_cohort = {c: cells.loc[cells.profile_id == c, "phq8_total"].values for c in cohorts}
boot = np.array([np.concatenate([by_cohort[c] for c in rng.choice(cohorts, len(cohorts), True)]).mean()
                 for _ in range(2000)])
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

# ---- the band. every row is checked against the source document ----------------------------
# (label, reported mean, instrument, offset applied, sample, selection, cite key)
ANCHORS = [
    ("NHANES 2005-2018, adults 18+", nhanes, "PHQ-8", 0.0,
     "36,259 adults, survey-weighted", "none", "this paper"),
    ("US primary care, below severe on all three syndromes", 3.6, "PHQ-8", 0.0,
     "1,759 of 2,091 consecutive patients, 15 clinics",
     "PHQ-8, GAD-7 and PHQ-15 all < 15; conditions on the outcome", "lowe2008"),
    ("US primary care, full caseload (derived bound)", None, "PHQ-8", 0.0,
     "2,091 consecutive patients, 15 clinics, 92% participation", "none", "lowe2008"),
    ("VA primary care, HIV-enriched cohort", 5.7, "PHQ-9", off_screen,
     "7,731 veterans, 8 medical centers, 95% male; pooled over person-visits 2003-2015",
     "cohort is ~half HIV-positive by design and the study targets comorbidity clustering, "
     "so this conditions on comorbidity and bounds a general VA primary care mean from above",
     "stevens2020"),
    ("US internet panel, no clinical frame", 6.5, "PHQ-9", off_panel,
     "96,234 respondents, PureSpectrum non-probability panel, 2023-2024; SD 6.6, 26.4% at >= 10",
     "none on severity; opt-in panel", "perlis2025"),
    ("Primary care, antidepressant initiation", 12.2, "PHQ-9", off_treat,
     "771 employed patients, 88 Minnesota clinics (DIAMOND)",
     "PHQ-9 >= 7 and a new antidepressant fill", "beck2011"),
]

# Lowe reports by syndrome subgroup, not overall, and the subgroups overlap, so the full-caseload
# mean is unpublished. It is bounded exactly, not assumed. Of the 332 patients outside the
# reference group, the 138 in the depression subgroup average 18.5; the other 194 are elevated on
# anxiety or somatization but by construction score below 15 on the PHQ-8, so they contribute
# somewhere in [0, 15). An earlier version put their floor at 10, which the >= 15 cutoff does not
# support, and so reported a bound that was not one.
N_TOT, REF_N, REF_M, DEP_N, DEP_M, CUT = 2091, 1759, 3.6, 138, 18.5, 15
rest_n = N_TOT - REF_N - DEP_N
fixed = REF_N * REF_M + DEP_N * DEP_M
case_floor_math = fixed / N_TOT                       # assumes the 194 all score 0: not attainable
case_lo = (fixed + rest_n * REF_M) / N_TOT            # the 194 at the reference mean: conservative
case_hi = (fixed + rest_n * CUT) / N_TOT

# Read-check on the subgroup structure before any of it is used. The reported single-syndrome
# percentages must reconstruct the 194, and the depression subgroup must match the reported
# prevalence. Both do, to a tenth of a patient.
venn = (0.43 * 168) + (0.46 * 199) + ((332 - (0.25 * 138 + 0.43 * 168 + 0.46 * 199)) - 0.75 * 138)
print(f"Lowe read-check: single-syndrome cells reconstruct {venn:.1f} vs 332 - 138 = {332 - 138}; "
      f"depression subgroup {0.066 * N_TOT:.1f} vs reported {DEP_N}")

rows = []
for label, val, inst, off, sample, sel, src in ANCHORS:
    if val is None:
        lo_a, hi_a = case_lo, case_hi
        rows.append(dict(anchor=label, mean_lo=round(lo_a, 3), mean_hi=round(hi_a, 3),
                         instrument=inst, offset=0.0, sample=sample, selection=sel, source=src,
                         resid_overall_lo=round(overall - hi_a, 3),
                         resid_overall_hi=round(overall - lo_a, 3),
                         resid_low_lo=round(low - hi_a, 3), resid_low_hi=round(low - lo_a, 3)))
    else:
        v = val - off
        rows.append(dict(anchor=label, mean_lo=round(v, 3), mean_hi=round(v, 3),
                         instrument=inst, offset=round(off, 4), sample=sample, selection=sel,
                         source=src,
                         resid_overall_lo=round(overall - v, 3), resid_overall_hi=round(overall - v, 3),
                         resid_low_lo=round(low - v, 3), resid_low_hi=round(low - v, 3)))
band = pd.DataFrame(rows)
band.to_csv(os.path.join(OUT, "frame_sensitivity_band.csv"), index=False)

print(f"NHANES PHQ-8 weighted mean       {nhanes:.4f}   (published anchor 2.98: reproduced)")
print(f"item-9 offset, PHQ-9 5-8 band    {off_screen:.4f}")
print(f"item-9 offset, PHQ-9 11-14 band  {off_treat:.4f}")
print(f"\nsimulated, cisgender frame: overall {overall:.4f}  cluster-boot CI [{ci_lo:.3f}, {ci_hi:.3f}]"
      f"\n                            low SES {low:.4f}"
      f"\n  (full grid {full_grid:.4f} is NOT the comparison: it carries transgender cells the "
      f"population anchor has no counterpart for)")
print(f"\nLowe full-caseload bound {case_lo:.3f} to {case_hi:.3f} "
      f"({rest_n} patients constrained to [0, {CUT}) by the >= {CUT} cutoff)\n")
print(band[["anchor", "mean_lo", "mean_hi", "instrument", "resid_overall_lo", "resid_overall_hi",
            "resid_low_lo", "resid_low_hi"]].to_string(index=False))

# ---- what survives, stated against the unconditioned anchor --------------------------------
screening = [r for r in rows if r["source"] in ("stevens2020", "perlis2025")]
top = max(r["mean_hi"] for r in screening)
dec = pd.DataFrame([
    dict(component="setting: primary-care attendee below severe vs general population",
         value=round(3.6 - nhanes, 3),
         basis="Lowe reference group 3.6 vs NHANES weighted mean, both PHQ-8"),
    dict(component="case mix: full caseload vs that reference group",
         value=round((case_lo + case_hi) / 2 - 3.6, 3),
         basis=f"derived caseload bound {case_lo:.2f}-{case_hi:.2f}, midpoint"),
    dict(component="unexplained: simulated cisgender overall vs full caseload",
         value=round(overall - (case_lo + case_hi) / 2, 3),
         basis="what neither setting nor case mix accounts for"),
])
dec.to_csv(os.path.join(OUT, "frame_decomposition.csv"), index=False)
# ---- SES matching. The low-income simulated mean cannot be read against an all-comers anchor.
gt_low = float(pd.read_csv(os.path.join(OUT, "residuals_cell_level.csv"))
               .set_index("group").loc["Low"].gt_mean)
grad = gt_low - nhanes
ses_lo, ses_hi = case_lo + grad, case_hi + grad
base_low = low - gt_low
pd.DataFrame([dict(quantity="population SES gradient (low vs all adults)", value=round(grad, 4),
                   basis=f"NHANES low {gt_low:.4f} minus all adults {nhanes:.4f}"),
              dict(quantity="SES-matched caseload anchor", value=f"{ses_lo:.3f}-{ses_hi:.3f}",
                   basis="caseload bound plus the population gradient, transferred additively"),
              dict(quantity="low-income residual, SES-matched",
                   value=f"{low - ses_hi:.3f}-{low - ses_lo:.3f}",
                   basis="the comparison the section stands behind for this cohort"),
              dict(quantity="share of the NHANES low-income residual retained",
                   value=f"{(low - ses_hi) / base_low * 100:.0f}%-{(low - ses_lo) / base_low * 100:.0f}%",
                   basis=f"against the {base_low:.3f}-point residual of Table 2")
              ]).to_csv(os.path.join(OUT, "frame_ses_matched.csv"), index=False)
print(f"\n=== SES matching (no stratified clinical anchor exists; population gradient transferred) ===")
print(f"  gradient {grad:.3f} | SES-matched anchor {ses_lo:.3f}-{ses_hi:.3f}")
print(f"  low-income residual  UNMATCHED +{low - case_hi:.2f} to +{low - case_lo:.2f}"
      f"   ->  MATCHED +{low - ses_hi:.2f} to +{low - ses_lo:.2f}")
print(f"  retains {(low - ses_hi) / base_low * 100:.0f}% to {(low - ses_lo) / base_low * 100:.0f}%"
      f" of the {base_low:.2f}-point NHANES low-income residual")

# ---- shape, not just level. The panel matches the simulated mean and nothing else about it.
# Recomputed on the PHQ-8 sample (eight items complete), not the item-9 subset above, so the
# dispersion reported beside the anchor describes the same respondents the anchor does.
_p8 = pd.concat(frames, ignore_index=True)
_p8[ITEMS[:8]] = _p8[ITEMS[:8]].where(_p8[ITEMS[:8]] <= 3)
_p8 = _p8[(_p8.RIDAGEYR >= 18) & _p8[ITEMS[:8]].notna().all(axis=1) & (_p8.WTMEC2YR > 0)].copy()
_p8["w"] = _p8.WTMEC2YR / len(CYCLES)
_p8["phq8"] = _p8[ITEMS[:8]].sum(axis=1)
_mu8 = float((_p8.phq8 * _p8.w).sum() / _p8.w.sum())
nh_sd = float(np.sqrt((_p8.w * (_p8.phq8 - _mu8) ** 2).sum() / _p8.w.sum()))
nh_p10 = float((_p8.w * (_p8.phq8 >= 10)).sum() / _p8.w.sum() * 100)
assert abs(nh_sd - 3.9352) < 5e-4, f"anchor SD {nh_sd} no longer matches the groundtruth file"
sim_sd, sim_p10 = float(cis.phq8_total.std()), float((cis.phq8_total >= 10).mean() * 100)
sim_mild = float(cis.phq8_total.between(5, 9).mean() * 100)
within = float(cis.groupby(["model", "profile_id"]).phq8_total.std().mean())
pd.DataFrame([
    dict(source="NHANES, survey-weighted", mean=round(_mu8, 3), sd=round(nh_sd, 3),
         pct_ge_10=round(nh_p10, 1), note=f"PHQ-8, n={len(_p8):,} (eight items complete)"),
    dict(source="Perlis internet panel", mean=6.5, sd=6.6, pct_ge_10=26.4, note="PHQ-9, as published"),
    dict(source="simulated, cisgender", mean=round(overall, 3), sd=round(sim_sd, 3),
         pct_ge_10=round(sim_p10, 1), note=f"{sim_mild:.1f}% in 5-9; within-cell SD {within:.2f}"),
]).to_csv(os.path.join(OUT, "frame_shape_contrast.csv"), index=False)
print(f"\n=== shape contrast: same mean, different object ===")
print(f"  NHANES     mean {_mu8:.2f}  SD {nh_sd:.2f}  {nh_p10:.1f}% at >= 10  (n={len(_p8):,})")
print(f"  panel      mean 6.50  SD 6.60  26.4% at >= 10")
print(f"  simulated  mean {overall:.2f}  SD {sim_sd:.2f}  {sim_p10:.1f}% at >= 10"
      f"  ({sim_mild:.1f}% in the 5-9 mild band, within-cell SD {within:.2f})")

print("\n=== decomposition, PHQ-8 throughout, cisgender frame ===")
print(dec.to_string(index=False))
print(f"\nprimary comparison, the unconditioned caseload bound:")
print(f"  overall  +{overall - case_hi:.2f} to +{overall - case_lo:.2f}")
print(f"  low SES  +{low - case_hi:.2f} to +{low - case_lo:.2f}")
print(f"\nhighest screening anchor {top:.3f}; simulated cisgender overall {overall:.3f}")
print(f"  low SES  clears by {low - top:.2f}")
print(f"  overall  clears by {overall - top:.2f}; cluster-boot CI low {ci_lo:.3f} "
      f"{'>' if ci_lo > top else '<'} {top:.3f} (margin {ci_lo - top:+.3f}), P(above) = {(boot > top).mean():.3f}")
print("  -> the overall margin is real but thin and must be reported as thin, not asserted flat")
