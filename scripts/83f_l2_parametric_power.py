"""Level 2 power at the audit's own precision, by parametric simulation of the R3 rule.

The half-split controls (83b, 83e) compare a pseudo-model built from one NHANES half with the other
half, so the benchmark noise enters twice and the reference is half the audit's size. A real audit
has neither handicap. Here the two estimates are drawn directly:
    population gap  P_hat ~ N(P, se_P^2)        P, se_P, df_P: full-sample NHANES (the audit's reference)
    simulated gap   G_hat ~ N(g P, se_G^2)      se_G: a real model's simulated-gap SE
and 78c's R3 rule (a = .05/7, asymmetric bounds, frozen stop) is applied to each draw, with the
label logic vectorised (it matches 78c.r3 exactly; checked against it on the first 200 draws).

se_G settings per contrast: the smallest and largest conditional SE among the four audited models
(83d), and the smallest and largest 78b pairs SE, so the range the audit actually faced is covered.
Doses g: -0.5, 0, 0.5, 1, 1.5, 2, 3 (the true ratio). 20,000 draws each.

Emits analysis/brm/83f_l2_parametric_power.csv.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c83", os.path.join(HERE, "83_controls_lib.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
R2M = C.R2M
DOSES = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
N = 20000
A, B = C.L2_ALPHA, np.asarray(C.L2_BANDS, float)


def verdicts(g, p, se_g, df_g, se_p, df_p):
    """Vectorised 78c.r3 'verdict' (label with the frozen stop) for arrays g, p."""
    s = np.where(p < 0, -1.0, 1.0)
    g, p = g * s, p * s
    tp = R2M.tcrit(A, df_p)
    p_lo, p_hi = p - tp * se_p, p + tp * se_p
    above = np.zeros(len(g), int)
    below = np.zeros(len(g), int)
    for c in B:
        t = R2M.tcrit(A, R2M.sdf(se_g, df_g, se_p, df_p, c))
        st = (g - c * p) / np.sqrt(se_g ** 2 + (c * se_p) ** 2)
        above += st > t
        below += st < -t
    lo, hi = above, len(B) - below
    lab = np.where(lo == hi, np.array(R2M.LABELS + ["?"])[np.minimum(lo, 5)], "undetermined").astype(object)
    two = hi == lo + 1
    lab[two] = [f"{R2M.LABELS[a]} or {R2M.LABELS[b]}" for a, b in zip(lo[two], hi[two])]
    with np.errstate(divide="ignore", invalid="ignore"):
        e1, e2 = g / p_hi, g / p_lo
        k = tp * se_p / p
    # frozen stop (78c.r3): k > kmax and the reading does not already exclude kept
    kmax = (B[3] / B[2] - 1) / (B[3] / B[2] + 1)
    allows_kept = np.array([v == "undetermined" or "kept" in v.split(" or ") for v in lab])
    out = np.where(p_lo <= 0, "no population gap",
                   np.where((k > kmax) & allows_kept, "reference too imprecise", lab))
    return out


def cls(v, true):
    if v in ("reference too imprecise", "no population gap"):
        return "stopped"
    if v == "undetermined":
        return "undetermined"
    names = v.split(" or ")
    return "correct" if names == [true] else ("compatible" if true in names else "wrong")


def main():
    ref = C.l2_reference(C.CellRef(C.load_frame()))
    se = pd.read_csv(os.path.join(C.OUTD, "83d_l2_se_check.csv"))
    se = se[se.source.str.startswith("corpus")]
    rng = np.random.default_rng(20261002)
    # exactness check of the vectorised rule against 78c.r3
    P, sP, dP = ref["Low minus High SES"]
    gg, pp = 2.0 * P + 0.1 * rng.standard_normal(200), P + sP * rng.standard_normal(200)
    v = verdicts(gg, pp, 0.1, np.inf, sP, dP)
    ref_v = [R2M.r3(a, 0.1, np.inf, b, sP, dP, A, B)["verdict"] for a, b in zip(gg, pp)]
    assert list(v) == ref_v, "vectorised rule disagrees with 78c.r3"
    rows = []
    for name, dim, hi, lo in C.L2_CONTRASTS:
        dpairs = int((C.l2_coef(dim, hi, lo)[0] > 0).sum()) - 1          # 78b: pairs - 1
        P, sP, dP = ref[name]
        x = se[se.contrast == name]
        settings = {"conditional, smallest": (x.se_conditional.min(), np.inf),
                    "conditional, largest": (x.se_conditional.max(), np.inf),
                    "pairs, smallest": (x.mean_se_78b.min(), dpairs),
                    "pairs, largest": (x.mean_se_78b.max(), dpairs)}
        for lab, (sG, dG) in settings.items():
            for g in DOSES:
                true = C.true_region(g)
                pp = P + sP * rng.standard_normal(N)
                gg = g * P + sG * rng.standard_normal(N)
                c = pd.Series([cls(vv, true) for vv in verdicts(gg, pp, sG, dG, sP, dP)])
                rows.append(dict(contrast=name, P=P, se_P=sP, df_P=dP, se_G_setting=lab, se_G=sG, dose=g,
                                 true_region=true, correct=(c == "correct").mean(),
                                 correct_or_compatible=c.isin(["correct", "compatible"]).mean(),
                                 wrong=(c == "wrong").mean(), undetermined=(c == "undetermined").mean(),
                                 stopped=(c == "stopped").mean()))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.OUTD, "83f_l2_parametric_power.csv"), index=False)
    pd.set_option("display.width", 220)
    print(out.drop(columns=["df_P"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
