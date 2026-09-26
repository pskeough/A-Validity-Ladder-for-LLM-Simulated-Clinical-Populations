"""Band flip as a descriptive row: recomputed on v3, the modal-category departure defined, and the
dependence of band flip on a cell's distance to the nearest cut point.

Definitions (cell = one model x one persona x one framing, n draws):
  band flip          P(two distinct draws of the cell fall in different PHQ-8 severity categories)
                     = 1 - sum_b c_b (c_b - 1) / (n (n - 1)), averaged over cells (scripts 12 and 61)
  threshold cross    P(two distinct draws fall on opposite sides of PHQ-8 = 10)
  modal departure    share of a cell's draws that fall outside the cell's most common category,
                     1 - max_b c_b / n, averaged over cells. It equals the probability that one
                     draw, taken at random, does not land in the cell's modal category. Script 25
                     estimated it by simulation (k = 1 column) on cells pooled over both framings
                     (60 draws); both pooled and per-framing versions are printed here, exactly.
  distance to cut    |cell mean - nearest category boundary|, boundaries 4.5, 9.5, 14.5, 19.5
                     (0 = on a boundary, 2.5 = centre of a category)

Location against dispersion: band flip is regressed on distance to cut and within-cell SD, with
partial R^2 for each, and compared with the flip predicted by a normal with the cell's own mean and
SD discretised at the boundaries (a pure location-plus-dispersion model with no free parameters).

Emits (analysis/brm/): gate_bandflip_summary.csv, gate_modal_departure.csv,
gate_bandflip_by_distance.csv, gate_bandflip_regression.csv, gate_bandflip_cells.csv.
Deterministic.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

_spec = importlib.util.spec_from_file_location(
    "gl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "82_gate_lib.py"))
gl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gl)

Z = stats.norm.ppf(0.975)
EDGES = np.array([-np.inf, 4.5, 9.5, 14.5, 19.5, np.inf])


def cell_table(v):
    rows = []
    for (m, p, f), g in v.groupby(["model", "profile_id", "framing"]):
        y = g.y.to_numpy()
        b = g.band.to_numpy()
        n = len(y)
        c = np.bincount(b, minlength=5)
        above = (y >= gl.THRESHOLD).sum()
        mu, sd = y.mean(), y.std(ddof=1)
        rows.append(dict(model=gl.SHORT[m], profile_id=p, framing=f, n=n, cell_mean=mu, within_sd=sd,
                         p_band_flip=1 - (c * (c - 1)).sum() / (n * (n - 1)),
                         p_threshold_cross=2.0 * above * (n - above) / (n * (n - 1)),
                         modal_share=c.max() / n,
                         dist_to_cut=np.min(np.abs(mu - np.array(gl.CUT_BOUNDARIES))),
                         pred_flip_normal=pred_flip(mu, sd)))
    return pd.DataFrame(rows)


def pred_flip(mu, sd):
    """Flip probability for two independent draws from N(mu, sd) cut at the category boundaries."""
    if sd == 0:
        return 0.0
    cdf = stats.norm.cdf((EDGES - mu) / sd)
    pi = np.diff(cdf)
    return 1 - (pi ** 2).sum()


def mean_ci(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return m, m - Z * se, m + Z * se


def main():
    v = gl.load_corpus()
    ct = cell_table(v)
    ct.to_csv(os.path.join(gl.OUTD, "gate_bandflip_cells.csv"), index=False, float_format="%.6g")

    # ---- summary, per framing x model with cell-level intervals (script 61's unit)
    prior = pd.read_csv(os.path.join(gl.BASE, "analysis", "regeneration_summary.csv"))
    prior["mshort"] = prior.model.map(lambda s: gl.SHORT.get(s, s))
    rows = []
    for f, gf in ct.groupby("framing"):
        for m, g in list(gf.groupby("model")) + [("ALL", gf)]:
            fl, fl_lo, fl_hi = mean_ci(g.p_band_flip)
            cr, cr_lo, cr_hi = mean_ci(g.p_threshold_cross)
            md, md_lo, md_hi = mean_ci(1 - g.modal_share)
            pr = prior[(prior.condition == f) & (prior.mshort == m)]
            rows.append(dict(framing=f, model=m, n_cells=len(g),
                             band_flip_pct=100 * fl, band_flip_lo=100 * fl_lo, band_flip_hi=100 * fl_hi,
                             published_flip_pct=float(pr.flip_pct.iloc[0]) if len(pr) else np.nan,
                             threshold_cross_pct=100 * cr, cross_lo=100 * cr_lo, cross_hi=100 * cr_hi,
                             published_cross_pct=float(pr.cross_pct.iloc[0]) if len(pr) else np.nan,
                             modal_departure_pct=100 * md, modal_dep_lo=100 * md_lo, modal_dep_hi=100 * md_hi,
                             mean_within_sd=g.within_sd.mean(),
                             pct_cells_mean_within_1_of_cut=100 * np.mean(g.dist_to_cut < 1)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(gl.OUTD, "gate_bandflip_summary.csv"), index=False, float_format="%.4f")

    # ---- modal departure: per framing (30 draws) and pooled over framings (60 draws, script 25)
    md_rows = []
    for f, gf in ct.groupby("framing"):
        for m, g in list(gf.groupby("model")) + [("ALL", gf)]:
            md_rows.append(dict(cell_definition=f"model x persona, {f} only", model=m, n_cells=len(g),
                                draws_per_cell=int(g.n.median()),
                                modal_departure_pct=100 * (1 - g.modal_share).mean(),
                                single_draw_matches_mode_pct=100 * g.modal_share.mean()))
    pooled = []
    for (m, p), g in v.groupby(["model", "profile_id"]):
        c = np.bincount(g.band.to_numpy(), minlength=5)
        pooled.append(dict(model=gl.SHORT[m], modal_share=c.max() / len(g), n=len(g)))
    pooled = pd.DataFrame(pooled)
    for m, g in list(pooled.groupby("model")) + [("ALL", pooled)]:
        md_rows.append(dict(cell_definition="model x persona, both framings pooled (script 25)", model=m,
                            n_cells=len(g), draws_per_cell=int(g.n.median()),
                            modal_departure_pct=100 * (1 - g.modal_share).mean(),
                            single_draw_matches_mode_pct=100 * g.modal_share.mean()))
    e25 = pd.read_csv(os.path.join(gl.BASE, "analysis", "ensemble_size.csv"))
    for r in e25.itertuples():
        md_rows.append(dict(cell_definition="script 25 receipt (simulated, model_outputs.csv)",
                            model="ALL" if r.scope == "pooled" else gl.SHORT.get(r.scope, r.scope),
                            n_cells=r.n_cells, draws_per_cell=60, modal_departure_pct=100 - r.k1,
                            single_draw_matches_mode_pct=r.k1))
    mdf = pd.DataFrame(md_rows)
    mdf.to_csv(os.path.join(gl.OUTD, "gate_modal_departure.csv"), index=False, float_format="%.3f")

    # ---- flip by distance to cut, overall and within SD tertiles
    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.51]
    labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5"]
    ct["dist_bin"] = pd.cut(ct.dist_to_cut, bins, right=False, labels=labels)
    ct["sd_tertile"] = pd.qcut(ct.within_sd, 3, labels=["low SD", "mid SD", "high SD"])
    tert_edges = np.quantile(ct.within_sd, [0, 1 / 3, 2 / 3, 1])
    by = []
    for f, gf in list(ct.groupby("framing")) + [("both", ct)]:
        for sdg, gs in [("all", gf)] + list(gf.groupby("sd_tertile", observed=True)):
            for dbin, g in gs.groupby("dist_bin", observed=True):
                by.append(dict(framing=f, sd_group=sdg, dist_bin=dbin, n_cells=len(g),
                               band_flip_pct=100 * g.p_band_flip.mean(),
                               threshold_cross_pct=100 * g.p_threshold_cross.mean(),
                               mean_within_sd=g.within_sd.mean()))
    byd = pd.DataFrame(by)
    byd.to_csv(os.path.join(gl.OUTD, "gate_bandflip_by_distance.csv"), index=False, float_format="%.4f")

    # ---- regression: location vs dispersion
    reg = []
    for f, g in list(ct.groupby("framing")) + [("both", ct)]:
        X = pd.DataFrame({"dist": g.dist_to_cut, "sd": g.within_sd})
        y = g.p_band_flip
        full = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC3")
        r_d = sm.OLS(y, sm.add_constant(X[["sd"]])).fit().rsquared
        r_s = sm.OLS(y, sm.add_constant(X[["dist"]])).fit().rsquared
        corr_pred = np.corrcoef(g.pred_flip_normal, y)[0, 1]
        reg.append(dict(framing=f, n_cells=len(g), r2_full=full.rsquared,
                        partial_r2_distance=full.rsquared - r_d, partial_r2_sd=full.rsquared - r_s,
                        r2_distance_alone=r_s, r2_sd_alone=r_d,
                        b_distance_per_point=full.params["dist"], b_distance_lo=full.conf_int().loc["dist", 0],
                        b_distance_hi=full.conf_int().loc["dist", 1],
                        b_sd_per_point=full.params["sd"], b_sd_lo=full.conf_int().loc["sd", 0],
                        b_sd_hi=full.conf_int().loc["sd", 1],
                        r2_normal_prediction=corr_pred ** 2,
                        mean_pred_flip_pct=100 * g.pred_flip_normal.mean(),
                        mean_obs_flip_pct=100 * y.mean(),
                        sd_tertile_edges="/".join(f"{x:.3f}" for x in tert_edges)))
    rg = pd.DataFrame(reg)
    rg.to_csv(os.path.join(gl.OUTD, "gate_bandflip_regression.csv"), index=False, float_format="%.5g")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(summ.round(2).to_string(index=False))
    print()
    print(mdf.round(2).to_string(index=False))
    print()
    print(byd[byd.framing == "both"].round(2).to_string(index=False))
    print()
    print(rg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
