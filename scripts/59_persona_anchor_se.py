"""
Design-based standard error for the persona-matched low-income anchor.

Script 47 derived that anchor (household income under $35,000 AND Medicaid, matching the persona
text) and reported a weighted mean and a residual with no standard error. The figure then became a
headline: the paper's severity range is quoted as "2.8 to 4.2" on this anchor in the abstract, the
introduction, the discussion and the conclusion. A reviewer noted that it is the one anchor carrying
a headline and no interval, and that Section 3.4's claim about the largest anchor standard error
(0.078 points) is computed over a table this anchor is absent from.

This closes both. Same estimator as script 26: Taylor-series linearisation for a ratio estimator,
strata SDMVSTRA, PSUs SDMVPSU nested within stratum, MEC examination weights. Same subpopulation
rule as script 47, and the domain is handled correctly by keeping the full sample and zeroing the
weights outside the domain, which is what a subpopulation estimate requires under a complex design.

Writes analysis/persona_anchor_se.csv.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(BASE, "data", "nhanes_raw")
SUFFIXES = ["D", "E", "F", "G", "H", "I", "J"]
ITEMS = ["DPQ0%d0" % i for i in range(1, 9)]
UNDER_35K = {1, 2, 3, 4, 5, 6, 13}          # INDHHIN2 categories strictly under $35,000
SIM_LOW = 9.7261                             # simulated low-income cisgender mean, from script 47


def load():
    frames = []
    for suf in SUFFIXES:
        demo = pd.read_sas(os.path.join(RAW, "DEMO_%s.xpt" % suf), format="xport")
        dpq = pd.read_sas(os.path.join(RAW, "DPQ_%s.xpt" % suf), format="xport")
        hiq = pd.read_sas(os.path.join(RAW, "HIQ_%s.xpt" % suf), format="xport")
        keep = ["SEQN", "RIDAGEYR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]
        inc = "INDHHIN2" if "INDHHIN2" in demo.columns else "INDHHINC"
        keep.append(inc)
        df = demo[[k for k in keep if k in demo.columns]].copy()
        df = df.rename(columns={inc: "INC"})
        df = df.merge(dpq[["SEQN"] + [c for c in ITEMS if c in dpq.columns]], on="SEQN")
        med_cols = [c for c in ("HIQ031D", "HIQ032D") if c in hiq.columns]
        if med_cols:
            m = hiq[["SEQN"] + med_cols].copy()
            m["MEDICAID"] = (m[med_cols] == 17).any(axis=1)
            df = df.merge(m[["SEQN", "MEDICAID"]], on="SEQN", how="left")
        else:
            df["MEDICAID"] = False
        df["MEDICAID"] = df["MEDICAID"].fillna(False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def phq8(df):
    x = df[ITEMS].copy()
    x = x.where(x <= 3)                       # 7 refused, 9 don't know
    return x.sum(axis=1, min_count=8)


def taylor_mean_se(y, w, strata, psu):
    """Linearised SE of a weighted mean, treated as a ratio estimator of totals."""
    mu = np.sum(w * y) / np.sum(w)
    z = w * (y - mu)                          # linearised residual
    W = np.sum(w)
    var = 0.0
    for h in np.unique(strata):
        m = strata == h
        psus = np.unique(psu[m])
        n_h = len(psus)
        if n_h < 2:
            continue
        tot = np.array([z[m & (psu == p)].sum() for p in psus])
        var += n_h / (n_h - 1.0) * np.sum((tot - tot.mean()) ** 2)
    return mu, np.sqrt(var) / W


d = load()
d["phq8"] = phq8(d)
d = d[(d.RIDAGEYR >= 18) & d.phq8.notna() & (d.WTMEC2YR > 0)].copy()
d["phq8"] = d.phq8.clip(0, 24)

# Domain estimation: keep every PSU, zero the weight outside the domain. Subsetting the file first
# would drop PSUs and understate the variance.
dom = d.INC.isin(UNDER_35K) & d.MEDICAID
d["w_dom"] = np.where(dom, d.WTMEC2YR, 0.0)

mu, se = taylor_mean_se(d.phq8.values, d.w_dom.values,
                        d.SDMVSTRA.values, d.SDMVPSU.values)
n = int(dom.sum())
srs = d.loc[dom, "phq8"].std(ddof=1) / np.sqrt(n)

print("persona-matched anchor (income < $35k AND Medicaid)")
print("  n              %d" % n)
print("  weighted mean  %.4f" % mu)
print("  design SE      %.4f   (Taylor linearisation, %d strata)" % (se, d.SDMVSTRA.nunique()))
print("  SRS SE         %.4f" % srs)
print("  design effect  %.3f" % (se / srs))
print("  95%% CI        [%.3f, %.3f]" % (mu - 1.96 * se, mu + 1.96 * se))
print()
print("  residual vs simulated %.4f: %.4f  [%.3f, %.3f]"
      % (SIM_LOW, SIM_LOW - mu, SIM_LOW - mu - 1.96 * se, SIM_LOW - mu + 1.96 * se))

pd.DataFrame([dict(anchor="income < $35k AND Medicaid (persona-matched)", n=n,
                   weighted_mean=round(mu, 4), se_design=round(se, 4), se_srs=round(srs, 4),
                   design_effect_on_se=round(se / srs, 3),
                   ci_lo=round(mu - 1.96 * se, 4), ci_hi=round(mu + 1.96 * se, 4),
                   residual=round(SIM_LOW - mu, 4),
                   resid_ci_lo=round(SIM_LOW - mu - 1.96 * se, 4),
                   resid_ci_hi=round(SIM_LOW - mu + 1.96 * se, 4))]
             ).to_csv(os.path.join(OUT, "persona_anchor_se.csv"), index=False)
print("\nwritten -> analysis/persona_anchor_se.csv")
