"""Level 1 rebuild, step b: the gateway rule read against NHANES, two-sided, matched on total.

Rule. An elevated case (PHQ-8 total >= 10) violates the gateway rule when item 1 (anhedonia) and
item 2 (depressed mood) are both below 2. Violations are impossible above a total of 20.

The violation rate falls steeply with the total (in NHANES from about half of cases at 10 to under a
tenth above 16), and the models' elevated totals are not distributed like people's. So the rate is
compared at matched totals, two ways:
  O/E (indirect standardisation, primary). E is the NHANES violation rate at each case's own total
      band, averaged over the model's elevated cases: the rate people would show if they had the
      model's total mix. O/E below 1 means the model routes elevated load onto items 1 and 2 more
      reliably than people with the same totals (more prototypical); above 1 means less coherent.
  Direct standardisation (secondary). The model's band-specific rates averaged over the NHANES
      elevated total mix, on the bands where the model has at least 20 elevated cases (common
      support); the NHANES rate on the same support is the comparator. The NHANES mass on the
      support is reported.
Total bands: 10, 11, 12, 13, 14, 15-16, 17-18, 19-20, 21-24 (21-24 has no possible violation).

Uncertainty. Corpus side: persona-clustered bootstrap (a persona is a profile_id within a model; both
framings of a persona go together when framings are pooled; pooled-model rows resample personas within
model). NHANES side: Rao-Wu rescaling bootstrap over PSUs within strata (SDMVSTRA, SDMVPSU), weights
WTMEC2YR / 7. The two are drawn jointly in each of B = 2000 replicates. The NHANES rate also gets a
Taylor-linearised design SE. Ratio intervals are 90% percentile intervals (two one-sided 5% tests);
rate intervals are 95%.

Tolerance. The equivalence band is [1/tau, tau] on O/E. tau is fixed as the smallest value in
{1.10, 1.25, 1.50, 2.00, 3.00} that contains the point O/E of every NHANES subgroup of 100 or more
elevated respondents (sex, race and ethnicity, poverty-income band, survey cycle, and the 2021-2023
cycle), each computed against the pooled 2005-2018 band rates. The rule is: the population's own
subgroups must pass. Verdicts are also given at every tau in the list.

Sensitivities: December-only rows (clinical rows from dec28_main or dec28_restored, all narrative
rows); NHANES 2021-2023 as the reference.

Emits (analysis/brm/):
  l1_gateway_nhanes.csv            NHANES rate, n, linearised and bootstrap SE, by reference
  l1_gateway_nhanes_by_total.csv   band rates, both references
  l1_gateway_tolerance.csv         NHANES subgroup O/E and the tau chosen
  l1_gateway_main.csv              per model x framing (x gender group), all samples and references
  l1_gateway_verdicts_tau.csv      verdicts at every tau
Seeded (20260926).
"""
import importlib.util
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("l1", os.path.join(os.path.dirname(__file__), "79_l1_lib.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

B = 2000
rng = np.random.default_rng(20260926)
BANDS = [[10], [11], [12], [13], [14], [15, 16], [17, 18], [19, 20], [21, 22, 23, 24]]
BLAB = ["10", "11", "12", "13", "14", "15-16", "17-18", "19-20", "21-24"]
TAUS = [1.10, 1.25, 1.50, 2.00, 3.00]
MIN_SUPPORT = 20


def band_of(total):
    out = np.full(len(total), -1)
    for k, bnd in enumerate(BANDS):
        out[np.isin(total, bnd)] = k
    return out


def nh_band_rates(e, W):
    """Band violation rates for NHANES elevated frame e under weight matrix W (R, n_e)."""
    bd = band_of(e.total.to_numpy())
    v = ((e.DPQ010 < 2) & (e.DPQ020 < 2)).to_numpy(float)
    num = np.zeros((W.shape[0], len(BANDS)))
    den = np.zeros_like(num)
    for k in range(len(BANDS)):
        m = bd == k
        num[:, k] = W[:, m] @ v[m]
        den[:, k] = W[:, m].sum(axis=1)
    return num / np.where(den > 0, den, np.nan), den


def stratified_idx(cl_strata, B, rng):
    """Resample clusters within strata; cl_strata is the stratum of each cluster position."""
    cols = []
    for s in np.unique(cl_strata):
        pos = np.where(cl_strata == s)[0]
        cols.append(pos[rng.integers(0, len(pos), size=(B, len(pos)))])
    return np.concatenate(cols, axis=1)


def sim_group_stats(g, v_pt, v_bt, p_nh_pt, p_nh_bt, rng):
    """g: corpus rows of one group. Returns dict of point estimates and bootstrap arrays."""
    e = g[g.total >= 10]
    clus = (g.model + "|" + g.profile_id).to_numpy()
    cl_u, cl_inv = np.unique(clus, return_inverse=True)
    cl_strata = np.array([c.split("|")[0] for c in cl_u])
    bd = band_of(e.total.to_numpy())
    viol = ((e.phq8_1 < 2) & (e.phq8_2 < 2)).to_numpy(float)
    e_cl = np.searchsorted(cl_u, (e.model + "|" + e.profile_id).to_numpy())
    C = np.zeros((len(cl_u), len(BANDS)))
    V = np.zeros_like(C)
    np.add.at(C, (e_cl, bd), 1.0)
    np.add.at(V, (e_cl, bd), viol)
    N = np.bincount(cl_inv, minlength=len(cl_u)).astype(float)
    idx = stratified_idx(cl_strata, B, rng)
    Cb, Vb, Nb = C[idx].sum(axis=1), V[idx].sum(axis=1), N[idx].sum(axis=1)
    Ct, Vt = C.sum(axis=0), V.sum(axis=0)
    vz = np.nan_to_num(v_pt)
    vbz = np.nan_to_num(v_bt)
    O = Vt.sum() / Ct.sum()
    E = (Ct @ vz) / Ct.sum()
    Ob = Vb.sum(axis=1) / Cb.sum(axis=1)
    Eb = (Cb * vbz).sum(axis=1) / Cb.sum(axis=1)
    # direct standardisation on common support
    sup = Ct >= MIN_SUPPORT
    sup[-1] = True                       # 21-24: violation impossible on both sides
    w = p_nh_pt * sup
    vm = np.where(Ct > 0, Vt / np.maximum(Ct, 1), 0.0)
    DS = (w * vm).sum() / w.sum()
    DSn = (w * vz).sum() / w.sum()
    wb = p_nh_bt * sup[None, :]
    vmb = np.where(Cb > 0, Vb / np.maximum(Cb, 1), 0.0)
    DSb = (wb * vmb).sum(axis=1) / wb.sum(axis=1)
    DSnb = (wb * vbz).sum(axis=1) / wb.sum(axis=1)
    return dict(n_rows=int(N.sum()), n_personas=len(cl_u), n_elevated=int(Ct.sum()),
                elevated_share=Ct.sum() / N.sum(), O=O, Ob=Ob, E=E, Eb=Eb, DS=DS, DSb=DSb, DSn=DSn,
                DSnb=DSnb, support_mass=float((p_nh_pt * sup).sum()),
                support_bands=";".join(BLAB[k] for k in range(len(BANDS)) if sup[k]))


def q(a, lo, hi):
    return np.nanquantile(a, lo), np.nanquantile(a, hi)


def main():
    nall = pd.read_csv(os.path.join(L.OUT, "l1_scores_nhanes.csv"))
    corpus = L.load_corpus()
    refs = {}
    rows_nh, rows_band = [], []
    for ref in ["2005_2018", "2021_2023"]:
        d = nall[nall.reference == ref].reset_index(drop=True)
        e_mask = (d.total >= 10).to_numpy()
        We = L.raowu_weights(d, B, rng, rows=e_mask)
        e = d[e_mask].reset_index(drop=True)
        viol = ((e.DPQ010 < 2) & (e.DPQ020 < 2)).to_numpy(float)
        R, se_lin = L.lin_se_ratio(d, ((d.DPQ010 < 2) & (d.DPQ020 < 2)) & (d.total >= 10), d.total >= 10)
        Rb = (We @ viol) / We.sum(axis=1)
        lo, hi = q(Rb, 0.025, 0.975)
        rows_nh.append(dict(reference=ref, n_adults=len(d), n_elevated=len(e),
                            elevated_share_weighted=np.average(d.total >= 10, weights=d.w),
                            violation_rate_weighted=R, violation_rate_unweighted=viol.mean(),
                            se_linearised=se_lin, se_bootstrap=Rb.std(), ci95_lo=lo, ci95_hi=hi))
        v_pt, den_pt = nh_band_rates(e, e.w.to_numpy()[None, :])
        v_bt, den_bt = nh_band_rates(e, We)
        p_pt = den_pt[0] / den_pt[0].sum()
        p_bt = den_bt / den_bt.sum(axis=1, keepdims=True)
        for k in range(len(BANDS)):
            m = band_of(e.total.to_numpy()) == k
            rows_band.append(dict(reference=ref, band=BLAB[k], n=int(m.sum()), share_of_elevated=p_pt[k],
                                  violation_rate=v_pt[0, k], boot_se=np.nanstd(v_bt[:, k])))
        refs[ref] = dict(d=d, e=e, We=We, v_pt=v_pt[0], v_bt=v_bt, p_pt=p_pt, p_bt=p_bt)
    pd.DataFrame(rows_nh).to_csv(os.path.join(L.OUT, "l1_gateway_nhanes.csv"), index=False)
    pd.DataFrame(rows_band).to_csv(os.path.join(L.OUT, "l1_gateway_nhanes_by_total.csv"), index=False)
    print(pd.DataFrame(rows_nh).round(4).to_string(index=False))

    # ---------------- tolerance from NHANES subgroups ----------------
    r0 = refs["2005_2018"]
    e, We = r0["e"], r0["We"]
    bd = band_of(e.total.to_numpy())
    viol = ((e.DPQ010 < 2) & (e.DPQ020 < 2)).to_numpy(float)
    tol_rows = []
    subgroups = [("sex", s) for s in ["men", "women"]] + \
                [("race_eth", s) for s in ["Hispanic", "NH White", "NH Black", "NH Asian", "Other"]] + \
                [("pir_band", s) for s in ["PIR<1.3", "PIR1.3-3.5", "PIR>3.5", "PIR missing"]] + \
                [("cycle", s) for s in L.CYCLES_0518]
    for var, lev in subgroups:
        m = (e[var] == lev).to_numpy()
        w = e.w.to_numpy() * m
        O = (w @ viol) / w.sum()
        Ev = (w @ np.nan_to_num(r0["v_pt"][bd])) / w.sum()
        Wm = We * m[None, :]
        Ob = (Wm @ viol) / Wm.sum(axis=1)
        vb = np.nan_to_num(r0["v_bt"])[:, bd]
        Eb = (Wm * vb).sum(axis=1) / Wm.sum(axis=1)
        lo, hi = q(Ob / Eb, 0.05, 0.95)
        tol_rows.append(dict(variable=var, level=lev, n_elevated=int(m.sum()), observed=O, expected=Ev,
                             O_over_E=O / Ev, ci90_lo=lo, ci90_hi=hi))
    # the 2021-2023 cycle against the 2005-2018 band rates (independent samples)
    r1 = refs["2021_2023"]
    e1 = r1["e"]
    bd1 = band_of(e1.total.to_numpy())
    viol1 = ((e1.DPQ010 < 2) & (e1.DPQ020 < 2)).to_numpy(float)
    O = np.average(viol1, weights=e1.w)
    Ev = np.average(np.nan_to_num(r0["v_pt"][bd1]), weights=e1.w)
    Ob = (r1["We"] @ viol1) / r1["We"].sum(axis=1)
    Eb = (r1["We"] * np.nan_to_num(r0["v_bt"])[:, bd1]).sum(axis=1) / r1["We"].sum(axis=1)
    lo, hi = q(Ob / Eb, 0.05, 0.95)
    tol_rows.append(dict(variable="cycle", level="L (2021-2023)", n_elevated=len(e1), observed=O, expected=Ev,
                         O_over_E=O / Ev, ci90_lo=lo, ci90_hi=hi))
    tol = pd.DataFrame(tol_rows)
    elig = tol[tol.n_elevated >= 100]
    spread = float(np.exp(np.abs(np.log(elig.O_over_E)).max()))
    tau = next(t for t in TAUS if t >= spread)
    tol["positive_control_verdict_at_tau"] = [
        L.verdict(lo_, hi_, next(t for t in TAUS if t >= float(np.exp(np.abs(np.log(
            tol[tol.n_elevated >= 100].O_over_E)).max()))))
        .replace("below", "too prototypical").replace("above", "less coherent")
        for lo_, hi_ in zip(tol.ci90_lo, tol.ci90_hi)]
    tol["eligible_n100"] = tol.n_elevated >= 100
    tol["max_eligible_fold_departure"] = spread
    tol["tau_chosen"] = tau
    tol.to_csv(os.path.join(L.OUT, "l1_gateway_tolerance.csv"), index=False)
    print(tol.round(3).to_string(index=False))
    print(f"largest fold departure among eligible NHANES subgroups: {spread:.3f}; tau = {tau}")

    # ---------------- corpus ----------------
    groups = []
    for sample in ["full", "dec_only"]:
        cs = corpus if sample == "full" else corpus[corpus.dec_only]
        for mdl in L.MODELS + ["pooled"]:
            cm = cs if mdl == "pooled" else cs[cs.model == mdl]
            for fr in ["clinical", "narrative", "both"]:
                cf = cm if fr == "both" else cm[cm.framing == fr]
                groups.append((sample, mdl, fr, "all", cf))
                if sample == "full" and fr == "both":
                    for gg in ["cis", "trans"]:
                        groups.append((sample, mdl, fr, gg, cf[cf.gender_group == gg]))
    rows, vrows = [], []
    for ref in ["2005_2018", "2021_2023"]:
        rr = refs[ref]
        for sample, mdl, fr, sub, g in groups:
            if ref == "2021_2023" and not (sample == "full" and sub == "all"):
                continue
            s = sim_group_stats(g, rr["v_pt"], rr["v_bt"], rr["p_pt"], rr["p_bt"], rng)
            oe_b = s["Ob"] / s["Eb"]
            ds_b = s["DSb"] / s["DSnb"]
            o_lo, o_hi = q(s["Ob"], 0.025, 0.975)
            r_lo, r_hi = q(oe_b, 0.05, 0.95)
            d_lo, d_hi = q(ds_b, 0.05, 0.95)
            row = dict(sample=sample, reference=ref, model=L.SHORT.get(mdl, mdl), framing=fr, subset=sub,
                       n_rows=s["n_rows"], n_personas=s["n_personas"], n_elevated=s["n_elevated"],
                       elevated_share=s["elevated_share"], violation_rate=s["O"], rate_ci95_lo=o_lo,
                       rate_ci95_hi=o_hi, nhanes_rate_at_model_totals=s["E"], O_over_E=s["O"] / s["E"],
                       OE_ci90_lo=r_lo, OE_ci90_hi=r_hi, direct_std_model=s["DS"], direct_std_nhanes=s["DSn"],
                       direct_std_ratio=s["DS"] / s["DSn"] if s["DSn"] > 0 else np.nan, DS_ci90_lo=d_lo,
                       DS_ci90_hi=d_hi, support_nhanes_mass=s["support_mass"], support_bands=s["support_bands"])
            v = L.verdict(r_lo, r_hi, tau)
            row["verdict_tau"] = tau
            row["verdict"] = v.replace("below", "too prototypical").replace("above", "less coherent")
            rows.append(row)
            if sub == "all":
                for t in TAUS:
                    vv = L.verdict(r_lo, r_hi, t).replace("below", "too prototypical").replace("above", "less coherent")
                    vrows.append(dict(sample=sample, reference=ref, model=row["model"], framing=fr, tau=t,
                                      OE=row["O_over_E"], lo=r_lo, hi=r_hi, verdict=vv))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(L.OUT, "l1_gateway_main.csv"), index=False)
    pd.DataFrame(vrows).to_csv(os.path.join(L.OUT, "l1_gateway_verdicts_tau.csv"), index=False)
    show = res[(res["sample"] == "full") & (res.reference == "2005_2018")]
    cols = ["model", "framing", "subset", "n_elevated", "violation_rate", "rate_ci95_lo", "rate_ci95_hi",
            "nhanes_rate_at_model_totals", "O_over_E", "OE_ci90_lo", "OE_ci90_hi", "direct_std_ratio",
            "support_nhanes_mass", "verdict"]
    print(show[cols].round(4).to_string(index=False))
    print(res[(res["sample"] != "full") | (res.reference != "2005_2018")][
        ["sample", "reference"] + cols[:2] + ["n_elevated", "violation_rate", "O_over_E", "OE_ci90_lo",
                                              "OE_ci90_hi", "verdict"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
