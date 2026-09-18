"""Level 2 pooled over the three models that clear level 1.

GLM-4.7 does not clear its own level-1 null, and the protocol says results above level 1 then
support no use for that model. This script reruns the seven anchored contrasts with GLM-4.7
withheld, so the pooled verdicts can be read on the panel the protocol would license. Pairing,
tests and the verdict rule follow 66_level2_permodel_equivalence.py exactly: paired t against zero
and against the population gap with the anchor's design error folded in, two one-sided tests at a
bound equal to the population disparity against the population value and against zero, the
tightest bound delta_min, and the verdict in the paper's order (missing, kept, steepened or
flattened, reversed, undetermined). Emits analysis/level2_without_glm.csv. No randomness.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "analysis")
CIS = ["Cisgender Man", "Cisgender Woman"]
ALPHA = 0.05
DROP = "z-ai/glm-4.7"

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"), low_memory=False)
m = m[m.gender.isin(CIS)].copy()
m["phq8_total"] = m["phq8_total_clipped"]
m["ses"] = m["ses_normalized"]
m["condition"] = m["prompt_condition"]
cell = (m.groupby(["model", "profile_id", "condition", "race", "gender", "ses", "relationship"],
                  as_index=False).phq8_total.mean())
assert DROP in set(cell.model)

gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
GTK = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic (pooled)",
       "Men": "Men", "Women": "Women", "Low": "Low", "Middle": "Middle", "High": "High"}
gtm = {g: float(gt.loc[gt.group == k, "w_mean"].iloc[0]) for g, k in GTK.items()}
d_se = pd.read_csv(os.path.join(OUT, "anchor_design_se.csv")).set_index("group")
SE_KEY = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic", "Men": "Cisgender men",
          "Women": "Cisgender women", "Low": "Low", "Middle": "Middle", "High": "High"}
gtse = {g: float(d_se.loc[k, "se_design"]) for g, k in SE_KEY.items()}

# contrast, column, high group, reference group, pairing stratum, anchor keys
SPECS = [
    ("Black minus White", "race", "Black", "White", ["model", "condition", "gender", "ses", "relationship"], "Black", "White"),
    ("Hispanic minus White", "race", "Hispanic", "White", ["model", "condition", "gender", "ses", "relationship"], "Hispanic", "White"),
    ("Asian minus White", "race", "Asian", "White", ["model", "condition", "gender", "ses", "relationship"], "Asian", "White"),
    ("Women minus Men", "gender", "Cisgender Woman", "Cisgender Man", ["model", "condition", "race", "ses", "relationship"], "Women", "Men"),
    ("Low minus High SES", "ses", "Low", "High", ["model", "condition", "race", "gender", "relationship"], "Low", "High"),
    ("Middle minus High SES", "ses", "Middle", "High", ["model", "condition", "race", "gender", "relationship"], "Middle", "High"),
    ("Low minus Middle SES", "ses", "Low", "Middle", ["model", "condition", "race", "gender", "relationship"], "Low", "Middle"),
]


def tost(centre, se, df, bound):
    p_low = stats.t.sf((centre + bound) / se, df)
    p_high = stats.t.sf((bound - centre) / se, df)
    return float(max(p_low, p_high))


def verdict(sign_ok, ratio, p_sep, p_eq_pop, p_eq_zero):
    separable = p_sep < ALPHA
    if p_eq_zero < ALPHA and separable:
        return "missing"
    if sign_ok and p_eq_pop < ALPHA and not separable:
        return "kept"
    if separable and sign_ok and ratio > 1:
        return "steepened"
    if separable and sign_ok and ratio < 1:
        return "flattened"
    if separable and not sign_ok:
        return "reversed"
    return "undetermined"


def run(sub, label):
    rows = []
    for name, col, hi, lo, stratum, ghi, glo in SPECS:
        a = sub[sub[col] == hi].set_index(stratum).phq8_total
        b = sub[sub[col] == lo].set_index(stratum).phq8_total
        common = a.index.intersection(b.index)
        diff = (a.loc[common] - b.loc[common]).to_numpy()
        n = len(diff); df = n - 1
        est = float(diff.mean()); se = float(diff.std(ddof=1) / np.sqrt(n))
        pop = gtm[ghi] - gtm[glo]; se_pop = float(np.hypot(gtse[ghi], gtse[glo])); se_gap = float(np.hypot(se, se_pop))
        p_zero = float(stats.t.sf(abs(est / se), df) * 2)
        p_pop = float(stats.t.sf(abs((est - pop) / se_gap), df) * 2)
        bound = abs(pop)
        p_eq_pop = tost(est - pop, se_gap, df, bound)
        p_eq_zero = tost(est, se, df, bound)
        tcrit = stats.t.ppf(0.95, df)
        rows.append(dict(panel=label, contrast=name, n_pairs=n, simulated=round(est, 4), se=round(se, 4),
                         ci95_lo=round(est - 1.96 * se, 4), ci95_hi=round(est + 1.96 * se, 4),
                         population=round(pop, 4), ratio=round(est / pop, 3), p_vs_zero=p_zero, p_vs_pop=p_pop,
                         p_tost_pop=round(p_eq_pop, 6), delta_min_pop=round(abs(est - pop) + tcrit * se_gap, 4),
                         p_tost_zero=round(p_eq_zero, 6), sign_matches=bool(np.sign(est) == np.sign(pop)),
                         verdict=verdict(bool(np.sign(est) == np.sign(pop)), est / pop, p_pop, p_eq_pop, p_eq_zero)))
    return rows


full = run(cell, "four models")
three = run(cell[cell.model != DROP], "without GLM-4.7")
res = pd.DataFrame(full + three)
# gate: the four-model rows must reproduce the published pooled receipts
ref = pd.read_csv(os.path.join(OUT, "level2_permodel_equivalence.csv"))
ref = ref[ref.scope == "pooled"].set_index("contrast")
for r in full:
    want = ref.loc[r["contrast"]]
    assert abs(r["simulated"] - float(want.simulated)) < 0.0011, (r["contrast"], r["simulated"], want.simulated)
    assert r["verdict"] == want.verdict, (r["contrast"], r["verdict"], want.verdict)
print("gate passed: the four-model panel reproduces level2_permodel_equivalence.csv")
res.to_csv(os.path.join(OUT, "level2_without_glm.csv"), index=False)
pd.set_option("display.width", 220)
print(res[["panel", "contrast", "n_pairs", "simulated", "ci95_lo", "ci95_hi", "population", "ratio", "p_vs_pop", "p_tost_pop", "p_tost_zero", "verdict"]].round(3).to_string(index=False))
