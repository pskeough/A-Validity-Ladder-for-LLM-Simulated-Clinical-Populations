"""Level 1 rebuild, step a: fit a graded response model to NHANES PHQ-8 and score every response
vector (NHANES 2005-2018, NHANES 2021-2023, and all 28,799 valid corpus rows) for person fit.

Reference model. Samejima's graded response model with a standard normal trait, fitted by marginal
maximum likelihood (81-point quadrature on -6..6) to NHANES 2005-2018 adults with all eight DPQ
items in 0..3 and a positive MEC weight. The main fit is weighted: MEC weights / 7, rescaled to sum
to n, entered as a pseudo-likelihood. An unweighted fit is kept as a sensitivity.

Person fit. Theta by Warm's weighted likelihood estimate (finite for the all-zero and all-three
vectors, where the ML estimate does not exist). Two statistics:
  lz   polytomous lz (Drasgow, Levine & Williams 1985), standardised at theta-hat as if known;
  lz*  the same statistic with the Snijders (2001) correction for estimated theta, in the
       polytomous form of Sinharay (2016), including the r0 = J / (2I) term that WLE adds.
Low values mean the vector is less likely than the model expects at its theta (misfit). High values
mean it is more likely than expected (overfit: too regular, too Guttman-like).

Checks written to CSV:
  - analytic gradient against central finite differences at the start values;
  - model-implied against observed (weighted) total-score distribution;
  - lz and lz* calibration on vectors simulated from the fitted model at fixed theta.

Emits (analysis/brm/):
  l1_nhanes_sample_flow.csv, l1_grm_params.csv, l1_grm_fit_checks.csv, l1_grm_total_fit.csv,
  l1_personfit_calibration.csv, l1_scores_nhanes.csv, l1_scores_corpus.csv
Seeded (20260925).
"""
import importlib.util
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("l1", os.path.join(os.path.dirname(__file__), "79_l1_lib.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

rng = np.random.default_rng(20260925)
os.makedirs(L.OUT, exist_ok=True)


def main():
    flow = []
    nh = L.load_nhanes("2005_2018", flow)
    nl = L.load_nhanes("2021_2023", flow)
    pd.DataFrame(flow).to_csv(os.path.join(L.OUT, "l1_nhanes_sample_flow.csv"), index=False)
    print(f"NHANES 2005-2018 n = {len(nh)}; 2021-2023 n = {len(nl)}")

    X = L.nhanes_items(nh)

    # gradient check on a subsample at the start values
    U, f, _ = L.patterns(X[:3000])
    p0 = L.pack(np.full(8, 1.3), np.tile([0.4, 1.4, 2.6], (8, 1)))
    _, g = L._negll_grad(p0, U, f)
    eps = 1e-5
    fd = np.array([(L._negll_grad(p0 + eps * e, U, f)[0] - L._negll_grad(p0 - eps * e, U, f)[0]) / (2 * eps)
                   for e in np.eye(len(p0))])
    grad_err = float(np.max(np.abs(fd - g)) / max(1.0, np.max(np.abs(fd))))

    fits = {}
    for lab, w in [("weighted", nh.w.to_numpy()), ("unweighted", None)]:
        fits[lab] = L.fit_grm(X, w)
        r = fits[lab]
        print(f"GRM {lab}: converged={r['converged']} nit={r['nit']} loglik={r['loglik']:.2f} "
              f"patterns={r['n_patterns']}")

    rows = []
    for lab, r in fits.items():
        for j in range(8):
            rows.append(dict(fit=lab, item=L.DPQ[j], a=r["a"][j], b1=r["b"][j, 0], b2=r["b"][j, 1],
                             b3=r["b"][j, 2]))
    pd.DataFrame(rows).to_csv(os.path.join(L.OUT, "l1_grm_params.csv"), index=False)

    chk = [dict(check="max relative error analytic vs finite-difference gradient", value=grad_err)]
    for lab, r in fits.items():
        chk += [dict(check=f"{lab} converged", value=float(r["converged"])),
                dict(check=f"{lab} iterations", value=r["nit"]),
                dict(check=f"{lab} pseudo-loglik", value=r["loglik"]),
                dict(check=f"{lab} unique patterns", value=r["n_patterns"])]

    # observed vs model-implied total-score distribution
    tf = []
    obs_w = np.bincount(nh.total, weights=nh.w, minlength=25)
    obs_w = obs_w / obs_w.sum()
    obs_u = np.bincount(nh.total, minlength=25) / len(nh)
    for lab, r in fits.items():
        exp = L.expected_total_dist(r["a"], r["b"])
        obs = obs_w if lab == "weighted" else obs_u
        for t in range(25):
            tf.append(dict(fit=lab, total=t, observed=obs[t], expected=exp[t]))
        chk.append(dict(check=f"{lab} total-score distribution, sum |obs - exp| / 2",
                        value=0.5 * np.abs(obs - exp).sum()))
    pd.DataFrame(tf).to_csv(os.path.join(L.OUT, "l1_grm_total_fit.csv"), index=False)

    # calibration of lz and lz* on vectors simulated from the weighted fit
    a, b = fits["weighted"]["a"], fits["weighted"]["b"]
    cal = []
    for th in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        P, _, _ = L.cat_probs(a, b, np.array([th]))
        cp = np.cumsum(P[0], axis=1)
        u = rng.random((20000, 8))
        Xs = (u[:, :, None] > cp[None, :, :]).sum(axis=2)
        Xs = np.minimum(Xs, 3)
        _, lz, lzs = L.score_vectors(a, b, Xs)
        tot = Xs.sum(axis=1)
        cal.append(dict(theta=th, n=len(Xs), share_all_zero=float((tot == 0).mean()),
                        lz_mean=lz.mean(), lz_sd=lz.std(), lzstar_mean=lzs.mean(), lzstar_sd=lzs.std(),
                        lzstar_p05=np.quantile(lzs, 0.05), lzstar_p95=np.quantile(lzs, 0.95),
                        share_lzstar_below_m1645=float((lzs < -1.645).mean()),
                        share_lzstar_above_1645=float((lzs > 1.645).mean())))
    cal = pd.DataFrame(cal)
    cal.to_csv(os.path.join(L.OUT, "l1_personfit_calibration.csv"), index=False)
    print(cal.round(3).to_string(index=False))

    # score NHANES (both references) and the corpus with both parameter sets
    m = L.load_corpus()
    out_n = []
    for ref, d in [("2005_2018", nh), ("2021_2023", nl)]:
        keep = d[["SEQN", "cycle", "w", "stratum", "psu", "total", "sex", "race_eth", "pir_band"] + L.DPQ].copy()
        keep["reference"] = ref
        for lab, r in fits.items():
            th, lz, lzs = L.score_vectors(r["a"], r["b"], L.nhanes_items(d))
            keep[f"theta_{lab}"], keep[f"lz_{lab}"], keep[f"lzstar_{lab}"] = th, lz, lzs
        out_n.append(keep)
    out_n = pd.concat(out_n, ignore_index=True)
    out_n.to_csv(os.path.join(L.OUT, "l1_scores_nhanes.csv"), index=False)

    keep = m[["model", "short", "profile_id", "framing", "iteration", "row_source", "dec_only",
              "gender_group", "race", "gender", "ses_normalized", "relationship", "total"] + L.ITEMS].copy()
    for lab, r in fits.items():
        th, lz, lzs = L.score_vectors(r["a"], r["b"], m[L.ITEMS].to_numpy())
        keep[f"theta_{lab}"], keep[f"lz_{lab}"], keep[f"lzstar_{lab}"] = th, lz, lzs
    keep.to_csv(os.path.join(L.OUT, "l1_scores_corpus.csv"), index=False)

    n0 = out_n[out_n.reference == "2005_2018"]
    chk += [dict(check="NHANES 2005-2018 weighted mean lz*", value=np.average(n0.lzstar_weighted, weights=n0.w)),
            dict(check="NHANES 2005-2018 weighted SD lz*",
                 value=np.sqrt(np.cov(n0.lzstar_weighted, aweights=n0.w))),
            dict(check="NHANES 2005-2018 weighted share total = 0", value=np.average(n0.total == 0, weights=n0.w)),
            dict(check="corpus share total = 0", value=float((keep.total == 0).mean())),
            dict(check="theta at grid boundary, NHANES", value=float((np.abs(n0.theta_weighted) >= 9.99).sum())),
            dict(check="theta at grid boundary, corpus", value=float((np.abs(keep.theta_weighted) >= 9.99).sum()))]
    pd.DataFrame(chk).to_csv(os.path.join(L.OUT, "l1_grm_fit_checks.csv"), index=False)
    print(pd.DataFrame(chk).to_string(index=False))
    print(pd.read_csv(os.path.join(L.OUT, "l1_grm_params.csv")).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
