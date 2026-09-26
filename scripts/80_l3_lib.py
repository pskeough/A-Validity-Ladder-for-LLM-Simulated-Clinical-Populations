"""Shared helpers for the level-3 rebuild (scripts 80a to 80d). Not run on its own.

NHANES person frame
    One row per adult (18+) with a valid PHQ-8 (all eight items 0..3) and a positive MEC weight,
    for the cycles 2005-2006 (D) through 2017-2018 (J), 2017-March 2020 pre-pandemic (P) and
    2021-2023 (L). Derived columns:
      race      White, Black, Hispanic, Asian, Other. Asian is RIDRETH3 == 6, which exists from 2011
                (cycle G) onward; the other groups come from RIDRETH1 (3 White, 4 Black, 1 and 2
                Hispanic), exactly as scripts 03, 26 and 30 define them. Before 2011 Asian adults sit
                in Other.
      sex       RIAGENDR (sex recorded at interview). Read as the anchor for cisgender personas.
      pir_band  INDFMPIR < 1.3 Low, <= 3.5 Middle, > 3.5 High (script 03's rule).
      p_inc     persona-string income band, cycles E..J only (INDHHIN2; D carries INDHHINC, whose top
                code is $75,000 and over, so middle and high cannot be separated there):
                  Low     household income under $35,000 (codes 1-6, 13) AND Medicaid (HIQ031D == 17)
                  Middle  $55,000 to $99,999 (codes 9, 10, 14) AND private insurance (HIQ031A == 14)
                  High    $100,000 and over (code 15) AND private insurance
                Everyone else is unclassified on this definition.
      p_inc_top the same with High further restricted to INDFMPIR at its top code (>= 5).
      mar       D..J only, the level-2 coding of script 78a: Married = DMDMARTL 1; Single = 2-5
                (widowed, divorced, separated, never married); living with a partner (6) is left
                out of both. Primary.
      mar_cohab Married = married or living with a partner; Single = widowed, divorced, separated
                or never married. The only split DMDMARTZ (P and L) supports, so the era windows
                use it, and on D..J it is the bridge to them.
      mar_all   D..J only: Married = DMDMARTL 1; Single = every other status, cohabiting included
                (the audit check's construction).
                In every coding, adults aged 18-19 who were not asked marital status are Single;
                refused / don't know are unclassified.

Survey variance
    Design: strata SDMVSTRA (unique across cycles), PSUs SDMVPSU nested in strata. Replicate
    variance by the stratified delete-one-PSU jackknife (JKn): replicate (h, j) drops PSU j of
    stratum h and multiplies the weights of the stratum's other PSUs by n_h / (n_h - 1);
    v = sum_h (n_h - 1) / n_h * sum_j (theta_hj - theta)^2. Strata with one PSU are skipped, the
    same convention as scripts 26 and 59. Every estimator here is a smooth function of cell totals,
    so replicates are computed from PSU-by-cell totals. Domains keep every PSU (zero weight outside
    the domain). Taylor linearisation is provided as a cross-check for domain means.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(BASE, "data", "nhanes_raw")
OUTD = os.path.join(BASE, "analysis", "brm")
ITEMS = [f"DPQ0{i}0" for i in range(1, 9)]
R3_CYCLES = set("GHIJPL")            # cycles that carry RIDRETH3
WINDOWS = {
    "2005-2018": dict(cycles=list("DEFGHIJ"), wcol="WTMEC2YR"),
    "2007-2018": dict(cycles=list("EFGHIJ"), wcol="WTMEC2YR"),
    "2017-2020": dict(cycles=["P"], wcol="WTMECPRP"),
    "2021-2023": dict(cycles=["L"], wcol="WTMEC2YR"),
}
UNDER35 = {1, 2, 3, 4, 5, 6, 13}
MID_P = {9, 10, 14}
RACES = ["White", "Black", "Asian", "Hispanic"]
SEXES = ["Men", "Women"]
INCS = ["Low", "Middle", "High"]
MARS = ["Married", "Single"]


def _files(c):
    if c == "P":
        return "P_DEMO", "P_DPQ", None
    if c == "L":
        return "DEMO_L", "DPQ_L", None
    return f"DEMO_{c}", f"DPQ_{c}", f"HIQ_{c}"


def load_cycle(c):
    fd, fq, fh = _files(c)
    demo = pd.read_sas(os.path.join(RAW, fd + ".xpt"), format="xport")
    dpq = pd.read_sas(os.path.join(RAW, fq + ".xpt"), format="xport")
    wcol = "WTMECPRP" if c == "P" else "WTMEC2YR"
    d = demo.merge(dpq[["SEQN"] + ITEMS], on="SEQN", how="inner")
    x = d[ITEMS].where(d[ITEMS] <= 3)
    d["phq8"] = x.sum(axis=1, min_count=8)
    d = d[(d.RIDAGEYR >= 18) & d.phq8.notna() & (d[wcol] > 0)].copy()
    out = pd.DataFrame({"SEQN": d.SEQN.astype(int), "cycle": c, "age": d.RIDAGEYR,
                        "wraw": d[wcol].astype(float), "stratum": d.SDMVSTRA.astype(int),
                        "psu": d.SDMVPSU.astype(int), "phq8": d.phq8.astype(float)})
    out["dep10"] = (out.phq8 >= 10).astype(float)
    r1 = d.RIDRETH1
    r3 = d.RIDRETH3 if "RIDRETH3" in d.columns else pd.Series(np.nan, index=d.index)
    race = np.select([r3 == 6, r1 == 3, r1 == 4, r1.isin([1, 2])],
                     ["Asian", "White", "Black", "Hispanic"], "Other")
    out["race"] = race
    out["sex"] = d.RIAGENDR.map({1: "Men", 2: "Women"})
    pir = d.INDFMPIR
    out["pir"] = pir
    out["pir_band"] = np.select([pir < 1.3, pir <= 3.5, pir > 3.5], INCS, None)
    out.loc[pir.isna(), "pir_band"] = None
    young = d.RIDAGEYR < 20
    if "DMDMARTL" in d.columns:
        m = d.DMDMARTL
        out["mar"] = np.select([m == 1, m.isin([2, 3, 4, 5])], MARS, None)
        out["mar_cohab"] = np.select([m.isin([1, 6]), m.isin([2, 3, 4, 5])], MARS, None)
        out["mar_all"] = np.select([m == 1, m.isin([2, 3, 4, 5, 6])], MARS, None)
        cods = ["mar", "mar_cohab", "mar_all"]
    else:
        m = d.DMDMARTZ
        out["mar"] = None
        out["mar_cohab"] = np.select([m == 1, m.isin([2, 3])], MARS, None)
        out["mar_all"] = None
        cods = ["mar_cohab"]
    miss = m.isna() & young
    for cc in cods:
        out.loc[miss.values, cc] = "Single"
    out["young_imputed_single"] = miss.values
    out["p_inc"] = None
    out["p_inc_top"] = None
    out["p_low_any"] = False
    if fh is not None:
        hiq = pd.read_sas(os.path.join(RAW, fh + ".xpt"), format="xport")
        h = pd.DataFrame({"SEQN": hiq.SEQN.astype(int), "medicaid": hiq.HIQ031D == 17,
                          "private": hiq.HIQ031A == 14})
        out = out.merge(h, on="SEQN", how="left")
        out["medicaid"] = out.medicaid.fillna(False).astype(bool)
        out["private"] = out.private.fillna(False).astype(bool)
        inc = d.set_index(d.SEQN.astype(int))["INDHHIN2" if "INDHHIN2" in d.columns else "INDHHINC"]
        out["hhinc"] = out.SEQN.map(inc)
        out["p_low_any"] = out.hhinc.isin(UNDER35) & out.medicaid      # valid for D too
        if "INDHHIN2" in d.columns:
            low = out.p_low_any
            mid = out.hhinc.isin(MID_P) & out.private
            high = (out.hhinc == 15) & out.private
            top = high & (out.pir >= 5)
            out["p_inc"] = np.select([low, mid, high], INCS, None)
            out["p_inc_top"] = np.select([low, mid, top], INCS, None)
            out.loc[~(low | mid | high), "p_inc"] = None
            out.loc[~(low | mid | top), "p_inc_top"] = None
    else:
        out["medicaid"] = np.nan
        out["private"] = np.nan
        out["hhinc"] = np.nan
    return out


def load_all():
    return pd.concat([load_cycle(c) for c in list("DEFGHIJ") + ["P", "L"]], ignore_index=True)


def window_frame(fr, window, asian_adjust):
    """Rows of one window with weight w. Multi-cycle windows divide the MEC weight by the number
    of cycles. With asian_adjust, Asian adults (identifiable only from 2011) are divided by the
    number of RIDRETH3 cycles in the window instead, so the Asian population total is an average
    over the cycles that can see it rather than 4/7 of it."""
    spec = WINDOWS[window]
    d = fr[fr.cycle.isin(spec["cycles"])].copy()
    k = len(spec["cycles"])
    ka = len([c for c in spec["cycles"] if c in R3_CYCLES])
    d["w"] = d.wraw / k
    if asian_adjust and ka < k:
        d.loc[d.race == "Asian", "w"] = d.loc[d.race == "Asian", "wraw"] / ka
    return d


class Design:
    """PSU structure of one window and the JKn replicate machinery."""

    def __init__(self, d):
        key = d.stratum.astype(str) + "_" + d.psu.astype(str)
        self.codes, uniq = pd.factorize(key)
        pstrat = np.array([int(u.split("_")[0]) for u in uniq])
        self.s_of_psu, suniq = pd.factorize(pstrat)
        self.K = len(uniq)
        self.n_h = np.bincount(self.s_of_psu)
        self.dfree = int(self.K - len(suniq))
        self.rep_psu = np.where(self.n_h[self.s_of_psu] >= 2)[0]
        self.rep_h = self.s_of_psu[self.rep_psu]
        self.rep_nh = self.n_h[self.rep_h]
        self.stratum_arr = pstrat

    def psu_totals(self, X):
        """X: (n_people, m) person-level contributions -> (K, m) PSU totals."""
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        T = np.zeros((self.K, X.shape[1]))
        np.add.at(T, self.codes, X)
        return T

    def replicate(self, T):
        """PSU totals (K, m) -> full totals (m,) and replicate totals (R, m)."""
        full = T.sum(0)
        S = np.zeros((len(self.n_h), T.shape[1]))
        np.add.at(S, self.s_of_psu, T)
        f = (self.rep_nh / (self.rep_nh - 1.0))[:, None]
        reps = full[None, :] - S[self.rep_h] + f * (S[self.rep_h] - T[self.rep_psu])
        return full, reps

    def jk_var(self, theta, theta_reps):
        c = ((self.rep_nh - 1.0) / self.rep_nh)
        dev = theta_reps - theta
        return np.nansum(c.reshape((-1,) + (1,) * (dev.ndim - 1)) * dev ** 2, axis=0)

    def taylor_mean(self, y, w_dom):
        """Linearised SE of the ratio mean sum(w y) / sum(w) over a domain (w_dom zero outside)."""
        y = np.asarray(y, float)
        W = w_dom.sum()
        mu = np.sum(w_dom * np.nan_to_num(y)) / W
        z = w_dom * (np.nan_to_num(y) - mu) / W
        zt = np.bincount(self.codes, weights=z, minlength=self.K)
        var = 0.0
        for h in range(len(self.n_h)):
            if self.n_h[h] < 2:
                continue
            t = zt[self.s_of_psu == h]
            var += self.n_h[h] / (self.n_h[h] - 1.0) * np.sum((t - t.mean()) ** 2)
        return mu, float(np.sqrt(var))


def domain_mean_jk(des, y, w_dom):
    """Weighted domain mean and its JKn SE."""
    T = des.psu_totals(np.column_stack([w_dom * np.nan_to_num(y), w_dom]))
    full, reps = des.replicate(T)
    th = full[0] / full[1]
    thr = reps[:, 0] / reps[:, 1]
    return float(th), float(np.sqrt(des.jk_var(th, thr)))
