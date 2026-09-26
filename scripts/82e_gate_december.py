"""December-only sensitivity for the gate G-study.

Rows kept: clinical rows with row_source dec28_main or dec28_restored, and every narrative row.
December failures were whole cells or runs of draws, so 49 clinical cells (DeepSeek-V3 9, GLM-4.7
40) are absent and four are short (7, 11, 17, 23 draws). The per-model-and-framing one-facet
estimator handles the imbalance exactly (unbalanced one-way method of moments); the per-model and
joint estimators drop personas with any missing cell. A REML check runs on the one-facet designs.

Runs 82a's pipeline with december_only=True (outputs suffixed _december), then writes
analysis/brm/gate_december_compare.csv: the D-study quantities side by side, full v3 against
December-only, and the band flip on December-only clinical cells.
"""
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gl = _load("gl", "82_gate_lib.py")
ga = _load("ga", "82a_gate_gstudy.py")


def reml_one_facet(d):
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("y ~ 1", d, groups=d["profile_id"])
        for meth in ("lbfgs", "bfgs", "powell", "nm"):
            try:
                r = md.fit(reml=True, method=meth, maxiter=2000)
            except Exception:
                continue
            if np.isfinite(r.llf) and (best is None or r.llf > best.llf + 1e-9):
                best = r
    return float(best.cov_re.iloc[0, 0]), float(best.scale)


def main():
    ga.main(december_only=True)

    full = pd.read_csv(os.path.join(gl.OUTD, "gate_dstudy.csv"))
    dec = pd.read_csv(os.path.join(gl.OUTD, "gate_dstudy_december.csv"))
    keys = ["outcome", "model", "framing"]
    cols = ["n_persona", "s2_p", "s2_e", "phi_1", "phi_30", "k_phi80", "k_phi90", "k_se_tol1", "k_se_tol2"]
    cmp_ = full[keys + cols].merge(dec[keys + cols], on=keys, suffixes=("_full", "_dec"))

    v = gl.load_corpus(december_only=True)
    reml = []
    for m in gl.MODELS:
        for f in gl.FRAMES:
            d = v[(v.model == m) & (v.framing == f)]
            sp, se = reml_one_facet(d)
            flips = []
            for _, c in d.groupby("profile_id"):
                b = c.band.to_numpy()
                n = len(b)
                if n < 2:
                    continue
                cnt = np.bincount(b, minlength=5)
                flips.append(1 - (cnt * (cnt - 1)).sum() / (n * (n - 1)))
            reml.append(dict(outcome="PHQ-8 total", model=gl.SHORT[m], framing=f, reml_s2_p_dec=sp,
                             reml_s2_e_dec=se, n_draws_dec=len(d), n_cells_dec=d.profile_id.nunique(),
                             band_flip_pct_dec=100 * np.mean(flips)))
    cmp_ = cmp_.merge(pd.DataFrame(reml), on=keys, how="left")
    cmp_.to_csv(os.path.join(gl.OUTD, "gate_december_compare.csv"), index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\nDECEMBER COMPARE")
    print(cmp_.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
