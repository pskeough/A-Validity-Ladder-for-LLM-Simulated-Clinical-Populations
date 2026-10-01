"""85: the worked case for the BRM paper, rebuilt on corpus v3 and the rebuilt rung frames.

The case is the persona the paper follows through every rung: GPT-4o-mini, a Black cisgender
woman, single, on Medicaid with income under $35,000 (profile P_BLACK_CW_LOW_S). The 24 Sep plan
compared her clinical cell mean (10.3) with 4.25 (all low-income adults, PIR < 1.3) and 5.5 (the
Medicaid-matched anchor). This script recomputes those comparators on the level-3 frame and adds
the comparator the rebuilt level 3 actually uses, her own NHANES cell (Black women, PIR < 1.3,
not married), with a delete-one-PSU jackknife SE.

Reads: data/model_outputs_v3.csv, analysis/brm/80a_nhanes_frame.csv,
       analysis/brm/l1_scores_corpus.csv, analysis/brm/l1_scores_nhanes.csv
Writes: analysis/brm/85_worked_case.csv (one row per quantity), 85_worked_case_draws.csv
Local CPU only.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib  # noqa: E402

l1 = importlib.import_module("79_l1_lib")
l3 = importlib.import_module("80_l3_lib")

BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(BASE, "analysis", "brm")
PID = "P_BLACK_CW_LOW_S"
MODEL = "openai/gpt-4o-mini"
BANDS = [0, 5, 10, 15, 20, 25]  # PHQ-8 severity categories: 0-4, 5-9, 10-14, 15-19, 20-24

rows = []


def add(q, value, se=np.nan, n=np.nan, note=""):
    rows.append(dict(quantity=q, value=value, se=se, n=n, note=note))


# ---------------------------------------------------------------- the persona's draws
d = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v3.csv"), low_memory=False)
d = d[d.phq8_valid.astype(bool)]
c = d[(d.model == MODEL) & (d.profile_id == PID)].copy()
assert len(c) == 60, len(c)
c["band"] = np.digitize(c.phq8_total, BANDS[1:-1])
keep = ["prompt_condition", "iteration", "row_source", "phq8_total", "band"] + [f"phq8_{i}" for i in range(1, 9)]
c.sort_values(["prompt_condition", "iteration"])[keep].to_csv(
    os.path.join(OUT, "85_worked_case_draws.csv"), index=False)

for f, g in c.groupby("prompt_condition"):
    x = g.phq8_total.to_numpy(float)
    add(f"cell_mean_{f}", x.mean(), x.std(ddof=1) / np.sqrt(len(x)), len(x), "mean PHQ-8 over the cell's draws")
    add(f"cell_sd_{f}", x.std(ddof=1), n=len(x))
    add(f"cell_min_{f}", x.min(), n=len(x))
    add(f"cell_max_{f}", x.max(), n=len(x))
    add(f"share_ge10_{f}", (x >= 10).mean(), n=len(x), note="share of draws at or above the screening cut")
    b = g.band.to_numpy()
    modal = np.bincount(b, minlength=5).argmax()
    add(f"modal_band_{f}", modal, n=len(x), note="0=0-4, 1=5-9, 2=10-14, 3=15-19, 4=20-24")
    add(f"outside_modal_{f}", (b != modal).mean(), n=len(x), note="share of draws outside the modal category")
    # probability two distinct draws fall in different categories
    cnt = np.bincount(b, minlength=5)
    m = len(b)
    add(f"bandflip_{f}", 1 - (cnt * (cnt - 1)).sum() / (m * (m - 1)), n=m, note="P(two distinct draws differ in category)")
    first = g.sort_values("iteration").iloc[0]
    add(f"first_draw_total_{f}", first.phq8_total, note=f"iteration {int(first.iteration)}")
x = c.phq8_total.to_numpy(float)
add("cell_mean_combined", c.groupby("prompt_condition").phq8_total.mean().mean(), n=len(x),
    note="average of the two framing means (the level-3 unit)")

# ---------------------------------------------------------------- level 1 for her draws
sc = pd.read_csv(os.path.join(OUT, "l1_scores_corpus.csv"))
sc = sc[(sc.model == MODEL) & (sc.profile_id == PID)]
nh = pd.read_csv(os.path.join(OUT, "l1_scores_nhanes.csv"))
nh = nh[nh.reference == "2005_2018"]
lab = l1.total_strata(nh.total.to_numpy(int))
ref = l1.TailRef(lab[nh.total.to_numpy(int)], nh.lzstar_weighted.to_numpy())
loc = ref.locate(lab[sc.total.to_numpy(int)], sc.lzstar_weighted.to_numpy())
Fm, F = ref.cdf(loc, nh.w.to_numpy())
lo, hi, mid = l1.tail_probs(Fm, F)
add("l1_below_p5_share", lo.mean(), n=len(sc), note="share of her draws below the NHANES within-total 5th percentile of lz*")
add("l1_above_p95_share", hi.mean(), n=len(sc), note="share above the 95th")
add("l1_mid_pit", mid.mean(), n=len(sc), note="mean mid-PIT (NHANES = 0.5)")
el = sc[sc.total >= 10]
viol = ((el.phq8_1 < 2) & (el.phq8_2 < 2))
add("l1_elevated_draws", len(el), note="draws at 10 or more")
add("l1_gateway_violations", int(viol.sum()), n=len(el), note="elevated draws with items 1 and 2 both below 2")

# ---------------------------------------------------------------- level 3 comparators
fr = pd.read_csv(os.path.join(OUT, "80a_nhanes_frame.csv"), low_memory=False)
w05 = l3.window_frame(fr, "2005-2018", asian_adjust=True)
des = l3.Design(w05)


def dmean(mask, label, note):
    mask = mask.fillna(False).to_numpy(bool)
    w = np.where(mask, w05.w.to_numpy(), 0.0)
    mu, se = l3.domain_mean_jk(des, w05.phq8.to_numpy(float), w)
    p, sp = l3.domain_mean_jk(des, w05.dep10.to_numpy(float), w)
    add(f"nhanes_mean_{label}", mu, se, int(mask.sum()), note)
    add(f"nhanes_ge10_{label}", p, sp, int(mask.sum()), note)


dmean(w05.pir_band == "Low", "low_income_all", "all adults PIR < 1.3 (the 24 Sep comparator, 4.25)")
dmean((w05.race == "Black") & (w05.sex == "Women") & (w05.pir_band == "Low") & (w05.mar == "Single"),
      "own_cell", "Black women, PIR < 1.3, not married (her level-3 cell)")
dmean((w05.race == "Black") & (w05.sex == "Women") & (w05.pir_band == "Low"),
      "black_women_low", "Black women, PIR < 1.3, any marital status")
dmean(w05.p_low_any.astype(bool), "medicaid_under35k_all",
      "household income under $35,000 and Medicaid, all adults")
dmean(w05.p_low_any.astype(bool) & (w05.race == "Black") & (w05.sex == "Women") & (w05.mar == "Single"),
      "medicaid_under35k_own_cell", "under $35,000 and Medicaid, Black women, not married")

# ---------------------------------------------------------------- level 2: her income stratum
# The Low-High contrast for her stratum (Black, cisgender woman, single): her persona mean against
# her high-income counterpart's, beside the NHANES gap within the same stratum.
for lev, cpid in [("middle", "P_BLACK_CW_MIDDLE_S"), ("high", "P_BLACK_CW_HIGH_S")]:
    cc = d[(d.model == MODEL) & (d.profile_id == cpid)]
    assert len(cc) == 60, (cpid, len(cc))
    add(f"counterpart_mean_combined_{lev}", cc.groupby("prompt_condition").phq8_total.mean().mean(), n=len(cc),
        note=f"{cpid}, average of the two framing means")
    band = "Middle" if lev == "middle" else "High"
    dmean((w05.race == "Black") & (w05.sex == "Women") & (w05.pir_band == band) & (w05.mar == "Single"),
          f"own_stratum_{lev}", f"Black women, PIR band {band}, not married")
own = {r["quantity"]: r["value"] for r in rows}
add("stratum_gap_low_high_sim", own["cell_mean_combined"] - own["counterpart_mean_combined_high"],
    note="her persona mean minus her high-income counterpart's (GPT-4o-mini)")
add("stratum_gap_low_high_nhanes", own["nhanes_mean_own_cell"] - own["nhanes_mean_own_stratum_high"],
    note="NHANES: Black women not married, PIR < 1.3 minus PIR > 3.5 (one stratum of the standardised gap)")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "85_worked_case.csv"), index=False)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
