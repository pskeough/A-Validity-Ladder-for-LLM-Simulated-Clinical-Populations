"""Level 4, part 4: per-model verdicts from 81b, 81c and 81d.

Pass rules (one model, one framing; a model passes level 4 only if it passes in both framings):
  R1 general factor   every one-factor polychoric loading >= .30 and lambda1/lambda2 >= 3 in the
                      model's whole simulated population. Failing R1 fails level 4; R2 to R4 are
                      then not interpretable and are reported as description.
  R2 loadings         Tucker's phi with the NHANES loadings, lower one-sided 95% bootstrap bound
                      (5th percentile) >= .95.
  R3 factor source    the general factor also holds within cells (within-persona covariance for the
                      model, within-demographic-cell covariance for NHANES), as it does in NHANES.
  R4 invariance       for sex, race and income, the metric and scalar verdicts at e0 = .05
                      (holds / fails / indeterminate, 81c) are compared with the NHANES verdicts.
                      A step matches when both are the same determinate verdict, mismatches when
                      both are determinate and differ, and is unresolved otherwise. R4 passes when
                      all six steps match, fails when any step mismatches, and is unresolved
                      otherwise. The same comparison at e0 = .08 is reported as a sensitivity.

usage: python 81e_l4_summary.py [all|dec]
Emits analysis/brm/l4_summary{_dec}.csv and l4_invariance_pattern{_dec}.csv.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "analysis", "brm")
SUBSET = sys.argv[1] if len(sys.argv) > 1 else "all"
SUF = "" if SUBSET == "all" else "_" + SUBSET
MODELS = ["DeepSeek-V3", "Gemini-3-Flash", "GLM-4.7", "GPT-4o-mini"]


def main():
    st = pd.read_csv(os.path.join(OUT, f"l4_structure{SUF}.csv"))
    wb = pd.read_csv(os.path.join(OUT, f"l4_between_within{SUF}.csv"))
    inv = pd.read_csv(os.path.join(OUT, f"l4_invariance{SUF}.csv"))
    nh_inv = inv[inv.population.str.startswith("NHANES")].set_index(["attribute", "step"])
    nh_wb = wb[(wb.population == "NHANES") & (wb.frame == "full")].iloc[0]

    def compare(mv, nv):
        if mv.startswith("not interpretable"):
            return "not interpretable"
        if mv == nv and mv != "indeterminate":
            return "match"
        if mv == "indeterminate" or nv == "indeterminate":
            return "unresolved"
        return "mismatch"

    pat = []
    for _, r in inv[~inv.population.str.startswith("NHANES")].iterrows():
        ref = nh_inv.loc[(r.attribute, r.step)]
        pat.append(dict(population=r.population, attribute=r.attribute, step=r.step,
                        model_rmsea_d=r.rmsea_d, model_lo90=r.rmsea_d_lo90, model_hi90=r.rmsea_d_hi90,
                        nhanes_rmsea_d=ref.rmsea_d, nhanes_lo90=ref.rmsea_d_lo90,
                        nhanes_hi90=ref.rmsea_d_hi90,
                        model_verdict=r.verdict_e05, nhanes_verdict=ref.verdict_e05,
                        pattern=compare(r.verdict_e05, ref.verdict_e05),
                        model_verdict_e08=r.verdict_e08, nhanes_verdict_e08=ref.verdict_e08,
                        pattern_e08=compare(r.verdict_e08, ref.verdict_e08)))
    pat = pd.DataFrame(pat)

    def r4_of(p, col):
        """pass: every step matches; fail: at least one determinate mismatch; else unresolved."""
        if (p[col] == "not interpretable").any():
            return "not interpretable"
        if (p[col] == "mismatch").any():
            return "fail"
        if (p[col] == "match").all():
            return "pass"
        return "unresolved"
    pat.to_csv(os.path.join(OUT, f"l4_invariance_pattern{SUF}.csv"), index=False)

    rows = []
    for mdl in MODELS:
        for fr in ("clinical", "narrative"):
            lab = f"{mdl} {fr}"
            s = st[st.population == lab].iloc[0]
            w = wb[(wb.population == lab) & (wb.frame == "full")].iloc[0]
            p = pat[(pat.population == lab) & (pat.step != "metric (polychoric)")]
            pp = pat[(pat.population == lab) & (pat.step == "metric (polychoric)")]
            r1 = bool(s.general_factor)
            r2 = bool(r1 and s.phi_nhanes_lo90 >= 0.95)
            r3 = bool(r1 and w.within_general_factor)
            n_match = int((p.pattern == "match").sum())
            n_mis = int((p.pattern == "mismatch").sum())
            n_unr = int((p.pattern == "unresolved").sum())
            r4v = r4_of(p, "pattern")
            r4v08 = r4_of(p, "pattern_e08")
            r4 = bool(r1 and r4v == "pass")
            fails = [k for k, v in (("R1 general factor", r1), ("R2 loadings", r2),
                                    ("R3 factor source", r3), ("R4 invariance pattern", r4)) if not v]
            rows.append(dict(model=mdl, framing=fr, subset=SUBSET,
                             ev_ratio_12=s.ev_ratio_12, load_min=s.load_min,
                             n_load_negative=s.n_load_negative, gf_boot_share=s.general_factor_boot_share,
                             R1_general_factor=r1,
                             phi_nhanes=s.phi_nhanes, phi_lo90=s.phi_nhanes_lo90,
                             phi_lo95=s.phi_nhanes_lo95, phi_hi95=s.phi_nhanes_hi95,
                             loading_rmsd=s.loading_rmsd_nhanes, R2_loadings=r2,
                             between_share=w.between_share_mean,
                             nhanes_between_share=nh_wb.between_share_mean,
                             within_ev_ratio=w.within_ev_ratio,
                             nhanes_within_ev_ratio=nh_wb.within_ev_ratio,
                             within_load_min=w.within_load_min, R3_factor_source=r3,
                             inv_steps=len(p), inv_match=n_match, inv_mismatch=n_mis,
                             inv_unresolved=n_unr,
                             inv_poly_match=int((pp.pattern == "match").sum()),
                             inv_mismatch_e08=int((p.pattern_e08 == "mismatch").sum()),
                             inv_match_e08=int((p.pattern_e08 == "match").sum()),
                             R4_verdict_e05=r4v, R4_verdict_e08=r4v08,
                             R4_invariance=r4, level4_pass=not fails,
                             failed_rules="; ".join(fails)))
    out = pd.DataFrame(rows)
    by_model = out.groupby("model").agg(pass_both=("level4_pass", "all"),
                                        rules_passed_min=("failed_rules", lambda x: 4 - max(
                                            len(v.split("; ")) if v else 0 for v in x)))
    out = out.merge(by_model, left_on="model", right_index=True)
    out.to_csv(os.path.join(OUT, f"l4_summary{SUF}.csv"), index=False)

    # floor-multiple reading (81d): does the band 0.5-1.4 exclude the no-difference null, and do
    # models show a detectable structural difference on the same contrasts as NHANES?
    fl = pd.read_csv(os.path.join(OUT, f"l4_floor_null{SUF}.csv"))
    nh_fl = fl[fl.population.str.startswith("NHANES")].set_index("contrast")
    frows = []
    for pop, g in fl.groupby("population", sort=False):
        agree = [bool(r.above_null) == bool(nh_fl.loc[r.contrast, "above_null"]) for _, r in g.iterrows()]
        frows.append(dict(population=pop, n_contrasts=len(g),
                          null_multiple_min=g.null_multiple.min(), null_multiple_max=g.null_multiple.max(),
                          n_null_inside_band=int(g.null_inside_band.sum()),
                          n_obs_inside_band=int(g.obs_inside_band.sum()),
                          n_above_null=int(g.above_null.sum()),
                          n_above_null_agree_nhanes=int(sum(agree)),
                          n_xfit_ci_excludes_0=int((g.xfit_lo95 > 0).sum())))
    fsum = pd.DataFrame(frows)
    fsum.to_csv(os.path.join(OUT, f"l4_floor_summary{SUF}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))
    print()
    print(pat.round(3).to_string(index=False))
    print()
    print(fsum.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
