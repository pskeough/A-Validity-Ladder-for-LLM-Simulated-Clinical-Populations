"""
Reviewer challenge: the Low-SES persona says "I make less than $35,000 a year and I am on
Medicaid", but the Low-SES anchor conditions on the poverty-income ratio alone (INDFMPIR < 1.3).
Medicaid adults are selected on disability and SSI and score higher on the PHQ-8, so some part
of the headline +5.48 residual may be the model correctly reading a cue the anchor discarded.

This script recomputes the Low-SES anchor under four definitions, all MEC-weighted on the same
NHANES 2005-2018 frame and the same PHQ-8 construction as 03_compute_groundtruth.py:

  A. PIR < 1.3                        (current paper anchor; must reproduce 4.25)
  B. PIR < 1.3 AND Medicaid
  C. household income < $35,000
  D. household income < $35,000 AND Medicaid   (matches the persona text)

Writes analysis/medicaid_anchor.csv. Does not modify the ground-truth table.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "nhanes_raw")
OUT = os.path.join(HERE, "..", "analysis")

CYCLES = {"D": 2005, "E": 2007, "F": 2009, "G": 2011, "H": 2013, "I": 2015, "J": 2017}
PHQ8 = ["DPQ0%d0" % i for i in range(1, 9)]

# INDHHIN2 annual household income categories. 1-6 and 13 are all strictly under $35,000.
#  1 $0-4,999   2 $5,000-9,999   3 $10,000-14,999   4 $15,000-19,999
#  5 $20,000-24,999   6 $25,000-34,999   7 $35,000-44,999 ...
# 12 "$20,000 and over" and 13 "under $20,000" are coarse fallbacks; 77/99 refused/DK.
UNDER_35K = {1, 2, 3, 4, 5, 6, 13}


def load():
    frames = []
    for suf in CYCLES:
        demo = pd.read_sas(os.path.join(RAW, "DEMO_%s.xpt" % suf), format="xport")
        dpq = pd.read_sas(os.path.join(RAW, "DPQ_%s.xpt" % suf), format="xport")
        hiq = pd.read_sas(os.path.join(RAW, "HIQ_%s.xpt" % suf), format="xport")

        keep = ["SEQN", "RIDAGEYR", "INDFMPIR", "WTMEC2YR"]
        if "INDHHIN2" in demo.columns:
            keep.append("INDHHIN2")
        elif "INDHHINC" in demo.columns:
            keep.append("INDHHINC")
        df = demo[[c for c in keep if c in demo.columns]].copy()
        if "INDHHIN2" not in df.columns:
            df["INDHHIN2"] = df["INDHHINC"] if "INDHHINC" in df.columns else np.nan

        df = df.merge(dpq[["SEQN"] + [c for c in PHQ8 if c in dpq.columns]],
                      on="SEQN", how="left")

        # Medicaid. HIQ031D carries the value 17 when the respondent reports Medicaid
        # coverage. Some cycles also expose HIQ032D. Take either.
        med_cols = [c for c in ("HIQ031D", "HIQ032D") if c in hiq.columns]
        h = hiq[["SEQN"] + med_cols].copy()
        if med_cols:
            flags = [(h[c] == 17) for c in med_cols]
            h["medicaid"] = np.logical_or.reduce(flags)
        else:
            h["medicaid"] = np.nan
        df = df.merge(h[["SEQN", "medicaid"]], on="SEQN", how="left")

        df["cycle"] = suf
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def phq8_total(df):
    d = df[PHQ8].copy()
    for c in PHQ8:
        d[c] = d[c].where(d[c] <= 3, np.nan)
    return d.sum(axis=1, min_count=len(PHQ8))


def wstats(sub):
    x = sub["PHQ8"].to_numpy(float)
    w = sub["WTMEC2YR"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if len(x) == 0:
        return dict(n=0, w_mean=np.nan, w_sd=np.nan)
    wm = np.sum(w * x) / np.sum(w)
    wsd = np.sqrt(np.sum(w * (x - wm) ** 2) / np.sum(w))
    return dict(n=len(x), w_mean=round(wm, 4), w_sd=round(wsd, 4))


df = load()
df["PHQ8"] = phq8_total(df)
a = df[(df.RIDAGEYR >= 18)].dropna(subset=["PHQ8"]).copy()

a["pir_low"] = a["INDFMPIR"] < 1.3
a["inc_low"] = a["INDHHIN2"].isin(UNDER_35K)
a["medicaid"] = a["medicaid"].fillna(False).astype(bool)

defs = [
    ("A. PIR < 1.3 (paper anchor)", a[a.pir_low]),
    ("B. PIR < 1.3 AND Medicaid", a[a.pir_low & a.medicaid]),
    ("C. household income < $35k", a[a.inc_low]),
    ("D. income < $35k AND Medicaid (persona)", a[a.inc_low & a.medicaid]),
    ("   all adults 18+ (reference)", a),
]

SIM_LOW = 9.7261  # simulated low-SES cell mean, cisgender frame (paper Table 1)

rows = []
for name, sub in defs:
    s = wstats(sub)
    s["definition"] = name
    s["sim_mean"] = SIM_LOW
    s["residual"] = round(SIM_LOW - s["w_mean"], 4) if np.isfinite(s["w_mean"]) else np.nan
    rows.append(s)

out = pd.DataFrame(rows)[["definition", "n", "w_mean", "w_sd", "sim_mean", "residual"]]
os.makedirs(OUT, exist_ok=True)
out.to_csv(os.path.join(OUT, "medicaid_anchor.csv"), index=False)

pd.set_option("display.width", 140)
print("LOW-SES ANCHOR UNDER FOUR DEFINITIONS  (NHANES 2005-2018, adults 18+, MEC-weighted)")
print("=" * 100)
print(out.to_string(index=False))

base = out.loc[out.definition.str.startswith("A."), "w_mean"].iat[0]
print("\nGATE: definition A must reproduce the published Low-SES anchor of 4.25.")
print("      got %.4f  ->  %s" % (base, "PASS" if abs(base - 4.25) < 0.01 else "FAIL"))

med = out.loc[out.definition.str.startswith("D."), "residual"].iat[0]
pap = out.loc[out.definition.str.startswith("A."), "residual"].iat[0]
if np.isfinite(med):
    print("\nHeadline residual under the paper anchor : %+.2f" % pap)
    print("Headline residual matching the persona   : %+.2f" % med)
    print("Shrinkage                                : %.1f%% of the original"
          % (100.0 * med / pap))
print("\nWritten -> %s" % os.path.join(OUT, "medicaid_anchor.csv"))
