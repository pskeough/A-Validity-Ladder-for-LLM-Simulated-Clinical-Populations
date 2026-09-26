"""Level-3 rebuild, step a: the NHANES reference frame.

Builds one person-level frame over 2005-2018 (D..J), 2017-March 2020 (P) and 2021-2023 (L) with
the persona attributes coded as far as NHANES carries them (see 80_l3_lib.py for every rule), and
checks it against the published receipts before anything downstream uses it.

Gates (asserted):
  - the 2005-2018 group means reproduce groundtruth/phq8_groundtruth_nhanes_2005_2018.csv;
  - income < $35k AND Medicaid on 2005-2018 reproduces persona_anchor_se.csv (script 59) in mean,
    and its JKn SE is within 5% of script 59's Taylor SE;
  - the 2017-2020 and 2021-2023 race and sex anchors reproduce anchor_recency.csv (script 30).
Printed, not asserted: JKn SEs of the nine published anchors against anchor_design_se.csv
(script 26 subset the file before linearising, so small differences are expected).

Emits, under analysis/brm/:
  80a_nhanes_frame.csv        person-level frame (no weights beyond the raw MEC weight)
  80a_gates.csv               every gate value, receipt value and difference
  80a_tolerance_basis.csv     population mean, SD and prevalence at and around the cut, per window
  80a_cell_counts.csv         NHANES n and weighted share of the 48 persona cells, per window and
                              coding
  80a_coverage.csv            how much of each window the 48 cells cover, and what is dropped
No randomness.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l3lib", os.path.join(HERE, "80_l3_lib.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

os.makedirs(L.OUTD, exist_ok=True)
AN = os.path.join(L.BASE, "analysis")

fr = L.load_all()
fr.to_csv(os.path.join(L.OUTD, "80a_nhanes_frame.csv"), index=False)
print("frame rows by cycle:", fr.cycle.value_counts().sort_index().to_dict())

gates = []


def gate(name, got, want, tol, hard=True):
    diff = got - want
    ok = abs(diff) <= tol
    gates.append(dict(gate=name, value=round(got, 6), receipt=round(want, 6), diff=round(diff, 6),
                      tolerance=tol, asserted=hard, passed=bool(ok)))
    if hard:
        assert ok, f"gate failed: {name}: {got} vs {want}"


# 1. published 2005-2018 anchors (uniform weights, all races)
d = L.window_frame(fr, "2005-2018", asian_adjust=False)
des = L.Design(d)
gt = pd.read_csv(os.path.join(L.BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv")).set_index("group")
dse = pd.read_csv(os.path.join(AN, "anchor_design_se.csv")).set_index("group")
doms = {
    "All adults 18+": (np.ones(len(d), bool), "All adults 18+"),
    "White": (d.race == "White", "White"), "Black": (d.race == "Black", "Black"),
    "Asian": (d.race == "Asian", "Asian"), "Hispanic (pooled)": (d.race == "Hispanic", "Hispanic"),
    "Men": (d.sex == "Men", "Cisgender men"), "Women": (d.sex == "Women", "Cisgender women"),
    "Low": (d.pir_band == "Low", "Low"), "Middle": (d.pir_band == "Middle", "Middle"),
    "High": (d.pir_band == "High", "High"),
}
for g, (mask, sekey) in doms.items():
    w = np.where(mask, d.w, 0.0)
    mu, se = L.domain_mean_jk(des, d.phq8.values, w)
    gate(f"2005-2018 mean {g}", mu, float(gt.loc[g, "w_mean"]), 5e-4)
    gate(f"2005-2018 JKn SE vs script 26 Taylor SE {g}", se, float(dse.loc[sekey, "se_design"]), 0.01, hard=False)
    mu_t, se_t = des.taylor_mean(d.phq8.values, w)
    gate(f"2005-2018 Taylor (domain) SE vs JKn SE {g}", se_t, se, 0.005, hard=False)

# 2. persona-matched low-income anchor, script 59
p59 = pd.read_csv(os.path.join(AN, "persona_anchor_se.csv")).iloc[0]
w = np.where(d.p_low_any, d.w, 0.0)
mu, se = L.domain_mean_jk(des, d.phq8.values, w)
gate("2005-2018 income<$35k & Medicaid mean (script 59)", mu, float(p59.weighted_mean), 5e-4)
gate("2005-2018 income<$35k & Medicaid JKn SE (script 59 Taylor)", se, float(p59.se_design), 0.05 * float(p59.se_design))

# 3. era anchors, script 30
rec = pd.read_csv(os.path.join(AN, "anchor_recency.csv")).set_index("group")
for window, col in [("2017-2020", "anchor_pre-pandemic 2017-Mar 2020"), ("2021-2023", "anchor_recent 2021-2023")]:
    dw = L.window_frame(fr, window, asian_adjust=False)
    for g, mask in {"White": dw.race == "White", "Black": dw.race == "Black", "Asian": dw.race == "Asian",
                    "Hispanic": dw.race == "Hispanic", "Cisgender Man": dw.sex == "Men",
                    "Cisgender Woman": dw.sex == "Women"}.items():
        mu = float(np.average(dw.phq8[mask], weights=dw.w[mask]))
        gate(f"{window} mean {g} (script 30)", mu, float(rec.loc[g, col]), 6e-4)
    # income: script 30 used <= 1.30 for Low; ours uses < 1.3 (script 03). Printed only.
    for g in L.INCS:
        mask = dw.pir_band == g
        mu = float(np.average(dw.phq8[mask], weights=dw.w[mask]))
        gate(f"{window} mean {g} (script 30, boundary differs)", mu, float(rec.loc[g, col]), 0.02, hard=False)

pd.DataFrame(gates).to_csv(os.path.join(L.OUTD, "80a_gates.csv"), index=False)
print(pd.DataFrame(gates).to_string(index=False))

# 4. tolerance basis: population distribution per window (uniform weights, all adults)
rows = []
for window in L.WINDOWS:
    dw = L.window_frame(fr, window, asian_adjust=False)
    dsn = L.Design(dw)
    for popname, mask in [("all adults 18+", np.ones(len(dw), bool)),
                          ("four persona races", dw.race.isin(L.RACES).values)]:
        w = np.where(mask, dw.w, 0.0)
        mu, se_mu = L.domain_mean_jk(dsn, dw.phq8.values, w)
        sd = float(np.sqrt(np.sum(w * (dw.phq8 - mu) ** 2) / w.sum()))
        r = dict(window=window, population=popname, n=int(mask.sum()), design_df=dsn.dfree,
                 mean=round(mu, 4), se_mean=round(se_mu, 4), sd=round(sd, 4),
                 sd_0p2=round(0.2 * sd, 4), sd_0p5=round(0.5 * sd, 4))
        for cut in (8, 9, 10, 11, 12):
            p, sp = L.domain_mean_jk(dsn, (dw.phq8 >= cut).astype(float).values, w)
            r[f"prev_ge{cut}"] = round(100 * p, 3)
            if cut == 10:
                r["se_prev_ge10"] = round(100 * sp, 3)
        # prevalence change produced by shifting every score by +1 / +2 / -1 points
        r["prev_shift_plus1_pp"] = round(r["prev_ge9"] - r["prev_ge10"], 3)
        r["prev_shift_plus2_pp"] = round(r["prev_ge8"] - r["prev_ge10"], 3)
        r["prev_shift_minus1_pp"] = round(r["prev_ge11"] - r["prev_ge10"], 3)
        rows.append(r)
tb = pd.DataFrame(rows)
tb.to_csv(os.path.join(L.OUTD, "80a_tolerance_basis.csv"), index=False)
print("\n" + tb.to_string(index=False))

# 5. cell counts: 48 persona cells per window and coding
codings = [("2005-2018", "pir_band", "mar"), ("2005-2018", "pir_band", "mar_cohab"),
           ("2005-2018", "pir_band", "mar_all"),
           ("2007-2018", "pir_band", "mar"), ("2007-2018", "p_inc", "mar"), ("2007-2018", "p_inc_top", "mar"),
           ("2017-2020", "pir_band", "mar_cohab"), ("2021-2023", "pir_band", "mar_cohab")]
crow, cov = [], []
for window, inc, mar in codings:
    dw = L.window_frame(fr, window, asian_adjust=True)
    four = dw.race.isin(L.RACES)
    cl = four & dw.sex.notna() & dw[inc].notna() & dw[mar].notna()
    cov.append(dict(window=window, income=inc, marital=mar, n_adults=len(dw), n_four_races=int(four.sum()),
                    n_classified=int(cl.sum()),
                    wshare_four_races=round(float(dw.w[four].sum() / dw.w.sum()), 4),
                    wshare_classified_of_four=round(float(dw.w[cl].sum() / dw.w[four].sum()), 4),
                    n_missing_income_four=int((four & dw[inc].isna()).sum()),
                    n_missing_marital_four=int((four & dw[mar].isna()).sum()),
                    n_young_imputed_single=int((cl & dw.young_imputed_single).sum())))
    c = dw[cl].groupby(["race", "sex", inc, mar]).agg(n=("phq8", "size"), wsum=("w", "sum"),
                                                      wmean=("phq8", lambda s: np.nan)).reset_index()
    tot = dw.w[cl].sum()
    for _, r in c.iterrows():
        sub = dw[cl & (dw.race == r.race) & (dw.sex == r.sex) & (dw[inc] == r[inc]) & (dw[mar] == r[mar])]
        crow.append(dict(window=window, income=inc, marital=mar, race=r.race, sex=r.sex, inc=r[inc],
                         mar=r[mar], n=int(r.n), wshare=round(float(r.wsum / tot), 5),
                         wmean_phq8=round(float(np.average(sub.phq8, weights=sub.w)), 4),
                         prev10=round(100 * float(np.average(sub.dep10, weights=sub.w)), 3)))
cc = pd.DataFrame(crow)
cc.to_csv(os.path.join(L.OUTD, "80a_cell_counts.csv"), index=False)
cv = pd.DataFrame(cov)
cv.to_csv(os.path.join(L.OUTD, "80a_coverage.csv"), index=False)
print("\n" + cv.to_string(index=False))
print("\ncells per coding (min n):")
print(cc.groupby(["window", "income", "marital"]).n.agg(["count", "min", "median"]).to_string())
