"""What the gate licenses: the k each downstream rung needs, per model, and whether k = 30 meets it.

Pass rule for cell-mean use, stated before the per-model results were read against it:
  minimum      phi(k) >= .80 and SE(k) <= 1.0 PHQ-8 point in each framing
  recommended  phi(k) >= .90 and SE(k) <= 0.5 point in each framing
  single draw  the single-draw use passes when phi(1) >= .80 and SE(1) <= 1.0
where phi and SE are the per-model, per-framing D-study quantities of 82a (all 120 personas). The
rungs that read cell means (levels 2 and 3) run on the 60 cisgender personas, so their D-study is
recomputed on that subset here.

Rung-specific units (from scripts 66 and 67):
  level 2  a matched pair difference of two per-framing cell means (k draws each), averaged over the
           pairs of a model; its dependability is phi_gap(k) = tau2 / (tau2 + noise(k)), where tau2
           is the variance of the pairs' expected differences and noise(k) the mean draw-noise
           variance of a pair difference (s2_a / k + s2_b / k).
  level 3  a design cell averaged over both framings (2k draws); phi_L3(k) =
           (s2_p + s2_pf / 2) / (s2_p + s2_pf / 2 + s2_e / (2k)); the model-by-group mean is the mean
           of its design cells, and the draw-noise share of its squared SE is reported at k = 30.
  level 4  if the persona-level floor becomes primary, item-level cell means are the unit: per item,
           a one-facet phi on the item score.
Population gaps (the level-2 bounds) and anchor design SEs (the strict level-3 tolerance) are read
from the files scripts 66 and 67 use.

Emits (analysis/brm/): gate_license_models.csv, gate_license_level2.csv, gate_license_level3.csv,
gate_license_items.csv, gate_license_cis_dstudy.csv, gate_tolerance_rationale.csv.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("gl", os.path.join(HERE, "82_gate_lib.py"))
gl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gl)

K_AVAIL = 30
CIS = ["Cisgender Man", "Cisgender Woman"]
GT_KEY = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic (pooled)",
          "Men": "Men", "Women": "Women", "Low": "Low", "Middle": "Middle", "High": "High"}
SE_KEY = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic",
          "Men": "Cisgender men", "Women": "Cisgender women", "Low": "Low", "Middle": "Middle", "High": "High"}
RACE_STRATUM = ["condition", "gender", "ses", "relationship"]
SES_STRATUM = ["condition", "race", "gender", "relationship"]
SEX_STRATUM = ["condition", "race", "ses", "relationship"]
SPECS = [
    ("Black minus White", "race", "Black", "White", RACE_STRATUM, "Black", "White"),
    ("Hispanic minus White", "race", "Hispanic", "White", RACE_STRATUM, "Hispanic", "White"),
    ("Asian minus White", "race", "Asian", "White", RACE_STRATUM, "Asian", "White"),
    ("Women minus Men", "gender", "Cisgender Woman", "Cisgender Man", SEX_STRATUM, "Women", "Men"),
    ("Low minus High SES", "ses", "Low", "High", SES_STRATUM, "Low", "High"),
]
L3_GROUPS = [("race", "White", "White"), ("race", "Black", "Black"), ("race", "Hispanic", "Hispanic"),
             ("race", "Asian", "Asian"), ("gender", "Cisgender Man", "Men"),
             ("gender", "Cisgender Woman", "Women"), ("ses", "Low", "Low"), ("ses", "Middle", "Middle"),
             ("ses", "High", "High")]


def k_min(rule_phi, rule_se, rows):
    """Smallest k meeting phi >= rule_phi and SE <= rule_se in every framing row."""
    ks = []
    for r in rows.itertuples():
        ks.append(max(gl.k_for_phi(r.s2_p, r.s2_e, rule_phi), gl.k_for_se(r.s2_e, rule_se)))
    return max(ks)


def main():
    v = gl.load_corpus()
    v["ses"] = v.ses_normalized
    v["condition"] = v.framing
    gt = pd.read_csv(os.path.join(gl.BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
    gtm = {g: float(gt.loc[gt.group == k, "w_mean"].iloc[0]) for g, k in GT_KEY.items()}
    dse = pd.read_csv(os.path.join(gl.BASE, "analysis", "anchor_design_se.csv")).set_index("group")
    gtse = {g: float(dse.loc[k, "se_design"]) for g, k in SE_KEY.items()}

    # ---- tolerance rationale: population gaps and anchor SEs the tolerances are set against
    tr = [dict(quantity=f"NHANES gap {s[0]}", value=gtm[s[5]] - gtm[s[6]]) for s in SPECS]
    tr += [dict(quantity=f"NHANES anchor design SE {g}", value=gtse[g]) for g in gtse]
    pd.DataFrame(tr).to_csv(os.path.join(gl.OUTD, "gate_tolerance_rationale.csv"), index=False,
                            float_format="%.4f")

    # ---- per-model licensing from 82a (all 120 personas)
    ds = pd.read_csv(os.path.join(gl.OUTD, "gate_dstudy.csv"))
    ds = ds[ds.outcome == "PHQ-8 total"]
    lic = []
    for m in gl.MODELS:
        r = ds[ds.model == gl.SHORT[m]]
        k_lo = k_min(0.80, 1.0, r)
        k_hi = k_min(0.90, 0.5, r)
        single = all((gl.phi_k(x.s2_p, x.s2_e, 1) >= 0.80) and (np.sqrt(x.s2_e) <= 1.0) for x in r.itertuples())
        lic.append(dict(model=gl.SHORT[m], phi1_clinical=float(r[r.framing == "clinical"].phi_nested_1.iloc[0]),
                        phi1_narrative=float(r[r.framing == "narrative"].phi_nested_1.iloc[0]),
                        within_sd_max=float(np.sqrt(r.s2_e.max())),
                        single_draw_passes=single,
                        k_minimum_rule=k_lo, k_recommended_rule=k_hi,
                        k30_meets_minimum=k_lo <= K_AVAIL, k30_meets_recommended=k_hi <= K_AVAIL,
                        phi30_min_over_framings=float(r.phi_nested_30.min()),
                        se30_max_over_framings=float(r.se_at_30.max()),
                        k_se05_q90cell_max=float(r.k_se_tol1_q90cell.max())))
    lic = pd.DataFrame(lic)
    lic.to_csv(os.path.join(gl.OUTD, "gate_license_models.csv"), index=False, float_format="%.4f")

    # ---- cisgender-subset D-study (the personas levels 2 and 3 use)
    c = v[v.gender.isin(CIS)]
    cis_personas = sorted(c.profile_id.unique())
    assert len(cis_personas) == 60
    s = gl.cell_summaries(c, "y")
    mean, n, ss = gl.to_arrays(s, cis_personas, gl.MODELS, gl.FRAMES)
    cis_rows, l3_rows = [], []
    for mi, m in enumerate(gl.MODELS):
        for fi, f in enumerate(gl.FRAMES):
            o = gl.one_facet(mean[:, mi, fi], n[:, mi, fi], ss[:, mi, fi])
            cis_rows.append(dict(model=gl.SHORT[m], framing=f, n_persona=o["n_cells"], s2_p=o["p"], s2_e=o["e"],
                                 phi_1=gl.phi_k(o["p"], o["e"], 1), phi_30=gl.phi_k(o["p"], o["e"], 30),
                                 k_phi80=gl.k_for_phi(o["p"], o["e"], 0.8), k_phi90=gl.k_for_phi(o["p"], o["e"], 0.9),
                                 k_se05=gl.k_for_se(o["e"], 0.5), k_se10=gl.k_for_se(o["e"], 1.0)))
        t = gl.two_facet(mean[:, mi, :], n[:, mi, :], ss[:, mi, :])
        sp_bar = gl.pos(t["p"]) + gl.pos(t["pf"]) / 2
        phiL3 = lambda k: sp_bar / (sp_bar + gl.pos(t["e"]) / (2 * k))  # noqa: E731
        k80 = int(np.ceil(4 * gl.pos(t["e"]) / (2 * sp_bar))) if sp_bar > 0 else np.inf
        k90 = int(np.ceil(9 * gl.pos(t["e"]) / (2 * sp_bar))) if sp_bar > 0 else np.inf
        # model-by-group means over design cells (both framings pooled, 60 draws per cell)
        cm = c[c.model == m]
        dcell = cm.groupby(["profile_id", "race", "gender", "ses"]).y.agg(["mean", "var", "size"]).reset_index()
        for col, lev, key in L3_GROUPS:
            gcells = dcell[dcell[col] == lev]
            ncell = len(gcells)
            var_cm = gcells["mean"].var(ddof=1)
            se_obs = np.sqrt(var_cm / ncell)
            # draw-noise variance of one design cell mean at k per framing (2k draws): s2_e / (2k)
            noise30 = gl.pos(t["e"]) / (2 * K_AVAIL)
            l3_rows.append(dict(model=gl.SHORT[m], group=key, n_design_cells=ncell,
                                phi_L3_1=phiL3(1), phi_L3_30=phiL3(K_AVAIL), k_phiL3_80=k80, k_phiL3_90=k90,
                                se_group_mean_obs_k30=se_obs,
                                se_group_mean_draw_only_k30=np.sqrt(noise30 / ncell),
                                se_group_mean_draw_only_k1=np.sqrt(gl.pos(t["e"]) / 2 / ncell),
                                draw_share_of_se2_k30=noise30 / var_cm,
                                anchor_design_se=gtse[key], use_tolerance=5.0))
    cisd = pd.DataFrame(cis_rows)
    cisd.to_csv(os.path.join(gl.OUTD, "gate_license_cis_dstudy.csv"), index=False, float_format="%.5g")
    l3 = pd.DataFrame(l3_rows)
    l3.to_csv(os.path.join(gl.OUTD, "gate_license_level3.csv"), index=False, float_format="%.5g")

    # ---- level 2: matched pair differences of per-framing cell means
    cell = (c.groupby(["model", "profile_id", "condition", "race", "gender", "ses", "relationship"])
             .y.agg(["mean", "var", "size"]).reset_index())
    l2 = []
    for m in gl.MODELS:
        cm = cell[cell.model == m]
        for name, col, a, b, strat, ka, kb in SPECS:
            A = cm[cm[col] == a].set_index(strat)
            Bb = cm[cm[col] == b].set_index(strat)
            j = A.join(Bb, lsuffix="_a", rsuffix="_b", how="inner")
            diff = j.mean_a - j.mean_b
            noise30 = (j.var_a / j.size_a + j.var_b / j.size_b).mean()
            var_d = diff.var(ddof=1)
            tau2 = var_d - noise30
            npair = len(j)
            if tau2 > 0:
                k80 = int(np.ceil(noise30 * K_AVAIL / (0.25 * tau2)))
                k90 = int(np.ceil(noise30 * K_AVAIL / (tau2 / 9)))
                phi30 = tau2 / (tau2 + noise30)
            else:
                k80 = k90 = np.inf
                phi30 = 0.0
            pop = gtm[ka] - gtm[kb]
            l2.append(dict(model=gl.SHORT[m], contrast=name, n_pairs=npair, gap_k30=diff.mean(),
                           population_gap=pop, se_gap_obs_k30=np.sqrt(var_d / npair),
                           se_gap_draw_only_k30=np.sqrt(noise30 / npair),
                           se_gap_draw_only_k1=np.sqrt(noise30 * K_AVAIL / npair),
                           draw_share_of_se2_k30=noise30 / var_d, tau2=tau2, phi_gap_30=phi30,
                           k_phi_gap80=k80, k_phi_gap90=k90,
                           se_draw_k1_over_abs_pop_gap=np.sqrt(noise30 * K_AVAIL / npair) / abs(pop)))
    l2 = pd.DataFrame(l2)
    l2.to_csv(os.path.join(gl.OUTD, "gate_license_level2.csv"), index=False, float_format="%.5g")

    # ---- level 4 option: item-level cell means
    it_rows = []
    for m in gl.MODELS:
        for f in gl.FRAMES:
            d = v[(v.model == m) & (v.framing == f)]
            for i in range(1, 9):
                col = f"phq8_{i}"
                s_i = gl.cell_summaries(d.assign(it=d[col].astype(float)), "it")
                o = gl.one_facet(s_i["mean"].to_numpy(), s_i["n"].to_numpy(float), s_i["ss"].to_numpy())
                it_rows.append(dict(model=gl.SHORT[m], framing=f, item=i, s2_p=o["p"], s2_e=o["e"],
                                    phi_1=gl.phi_k(o["p"], o["e"], 1), phi_30=gl.phi_k(o["p"], o["e"], 30),
                                    k_phi80=gl.k_for_phi(o["p"], o["e"], 0.8),
                                    k_phi90=gl.k_for_phi(o["p"], o["e"], 0.9)))
    items = pd.DataFrame(it_rows)
    items.to_csv(os.path.join(gl.OUTD, "gate_license_items.csv"), index=False, float_format="%.5g")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(pd.DataFrame(tr).round(3).to_string(index=False))
    print("\nMODELS\n", lic.round(3).to_string(index=False))
    print("\nCIS D-STUDY\n", cisd.round(3).to_string(index=False))
    print("\nLEVEL 3\n", l3.round(3).to_string(index=False))
    print("\nLEVEL 2\n", l2.round(3).to_string(index=False))
    agg = items.groupby(["model", "framing"]).agg(min_phi30=("phi_30", "min"), max_k80=("k_phi80", "max"),
                                                  max_k90=("k_phi90", "max"),
                                                  items_phi30_below_80=("phi_30", lambda x: int((x < 0.8).sum())))
    print("\nITEMS\n", agg.round(3).to_string())
    agg.reset_index().to_csv(os.path.join(gl.OUTD, "gate_license_items_summary.csv"), index=False,
                             float_format="%.5g")


if __name__ == "__main__":
    main()
