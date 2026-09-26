"""REML check on the ANOVA variance components of 82a (statsmodels MixedLM).

Model and framing enter REML as fixed effects; every persona-involving component is random:
  per model and framing   y ~ 1 + (1 | persona)
  per model               y ~ framing + (1 | persona) + (0 + framing | persona) as a variance component
  joint                   y ~ model * framing + persona + persona:model + persona:framing
                          + persona:model:framing, the last four as variance components
Under the unrestricted mixed model the persona-involving expected mean squares are the same whether
model and framing are fixed or random, so these REML estimates target the same quantities as the
all-random ANOVA estimates of 82a (p, pm, pf, pmf, e). Model and framing main effects have no REML
counterpart here.

Emits analysis/brm/gate_reml_check.csv (REML beside ANOVA from gate_components.csv).
Deterministic.
"""
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

_spec = importlib.util.spec_from_file_location(
    "gl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "82_gate_lib.py"))
gl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gl)

OUTCOME_LABEL = {"y": "PHQ-8 total", "elev": "PHQ-8 >= 10"}


def fit(formula, data, vc=None, methods=("lbfgs", "bfgs", "powell", "nm")):
    """REML fit; several optimizers are tried and the highest REML log-likelihood is kept, because
    a single lbfgs run can stop at the zero boundary for the persona variance and still report
    convergence."""
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm(formula, data, groups=data["profile_id"], re_formula="1", vc_formula=vc)
        for meth in methods:
            try:
                r = md.fit(reml=True, method=meth, maxiter=2000)
            except Exception:
                continue
            if np.isfinite(r.llf) and (best is None or r.llf > best.llf + 1e-9):
                best = r
    r = best
    out = {"p": float(r.cov_re.iloc[0, 0]), "e": float(r.scale), "converged": bool(r.converged),
           "llf": float(r.llf)}
    if vc:
        for name, val in zip(r.model.exog_vc.names, r.vcomp):
            out[name] = float(val)
    return out


def fit_joint(d, anova):
    """Joint REML fit from the default start and from the ANOVA estimates (negative ones set to a
    small positive value); the higher REML log-likelihood is kept. Also returns the REML
    log-likelihood evaluated at the ANOVA point, so a reader can see which point the data prefer."""
    from statsmodels.regression.mixed_linear_model import MixedLMParams
    vc = {"pm": "0 + C(mm)", "pf": "0 + C(framing)", "pmf": "0 + C(mm):C(framing)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("yy ~ C(mm) * C(framing)", d, groups=d["profile_id"], re_formula="1", vc_formula=vc)
        e = anova["e"]
        names = md.exog_vc.names
        vcomp = np.array([max(anova[nm], 1e-4 * e) / e for nm in names])
        start = MixedLMParams.from_components(cov_re=np.array([[max(anova["p"], 1e-4 * e) / e]]), vcomp=vcomp)
        md.cov_pen, md.fe_pen, md.reml = None, None, True      # set by fit(); needed by loglike()
        llf_at_anova = float(md.loglike(start, profile_fe=True))
        best, tag = None, None
        for label, kw in (("default", {}), ("anova", {"start_params": start})):
            for meth in ("lbfgs", "bfgs"):
                try:
                    r = md.fit(reml=True, method=meth, maxiter=3000, **kw)
                except Exception:
                    continue
                if np.isfinite(r.llf) and (best is None or r.llf > best.llf + 1e-9):
                    best, tag = r, f"{label}/{meth}"
    out = {"p": float(best.cov_re.iloc[0, 0]), "e": float(best.scale), "converged": bool(best.converged),
           "llf": float(best.llf), "start": tag}
    for name, val in zip(best.model.exog_vc.names, best.vcomp):
        out[name] = float(val)
    return out, llf_at_anova


def main():
    v = gl.load_corpus()
    anova = pd.read_csv(os.path.join(gl.OUTD, "gate_components.csv"))
    rows = []
    for outcome in ("y", "elev"):
        lab = OUTCOME_LABEL[outcome]
        d = v[["model", "profile_id", "framing", outcome]].rename(columns={outcome: "yy"})
        d["mm"] = d.model.map(gl.SHORT)

        def a(scope, model, framing, comp):
            s = anova[(anova.outcome == lab) & (anova.scope == scope) & (anova.model == model)
                      & (anova.framing == framing) & (anova.component == comp)]
            return float(s.estimate_raw.iloc[0])

        for m in gl.MODELS:
            for f in gl.FRAMES:
                r = fit("yy ~ 1", d[(d.model == m) & (d.framing == f)])
                for c in ("p", "e"):
                    rows.append(dict(outcome=lab, scope="model_x_framing", model=gl.SHORT[m], framing=f,
                                     component=c, reml=r[c], anova=a("model_x_framing", gl.SHORT[m], f, c),
                                     converged=r["converged"], llf=r["llf"]))
            r = fit("yy ~ C(framing)", d[d.model == m], vc={"pf": "0 + C(framing)"})
            for c in ("p", "pf", "e"):
                rows.append(dict(outcome=lab, scope="model", model=gl.SHORT[m], framing="both",
                                 component=c, reml=r[c], anova=a("model", gl.SHORT[m], "both", c),
                                 converged=r["converged"], llf=r["llf"]))
            print("done", lab, gl.SHORT[m], flush=True)
        r, llf_at_anova = fit_joint(d, {c: a("joint", "all", "both", c) for c in ("p", "pm", "pf", "pmf", "e")})
        for c in ("p", "pm", "pf", "pmf", "e"):
            rows.append(dict(outcome=lab, scope="joint", model="all", framing="both", component=c,
                             reml=r[c], anova=a("joint", "all", "both", c), converged=r["converged"],
                             llf=r["llf"], llf_at_anova=llf_at_anova, start=r["start"]))
        print("done", lab, "joint", flush=True)

    out = pd.DataFrame(rows)
    out["abs_diff"] = (out.reml - out.anova).abs()
    out["rel_diff_pct"] = 100 * out.abs_diff / out.anova.abs().clip(lower=1e-9)
    out.to_csv(os.path.join(gl.OUTD, "gate_reml_check.csv"), index=False, float_format="%.5g")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print("\nmax abs diff, total:", out[out.outcome == "PHQ-8 total"].abs_diff.max())
    print("max abs diff, indicator:", out[out.outcome == "PHQ-8 >= 10"].abs_diff.max())


if __name__ == "__main__":
    main()
