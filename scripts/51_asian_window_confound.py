"""
Reviewer challenge: the Asian racial contrast is period-confounded.

The Asian anchor is identified from RIDRETH3, which NHANES released only from the 2011 cycle
onward, so it rests on cycles G-J (2011-2018). Every other racial anchor pools D-J (2005-2018).
Section 3 of the paper establishes that population depression rose across the anchor window
(see analysis/anchor_recency.csv). If that is right, the Asian anchor is drawn from a
systematically later and higher-prevalence period than the White anchor, the White-minus-Asian
POPULATION gap is biased downward, and the paper's claim that the models exaggerate the Asian
contrast "by about two fifths" is inflated by the mismatch rather than by the models.

This matters because Asian is the only racial contrast the paper claims the models preserve.

Everything here uses the conventions of 03_compute_groundtruth.py (PHQ-8 = DPQ010..DPQ080 with
values above 3 set missing, complete responders only, adults 18+, MEC weights WTMEC2YR, Asian
keyed off RIDRETH3 == 6 and the remaining groups off RIDRETH1) and the Taylor-linearised design
standard errors of 26_design_based_se.py. The simulated side is untouched: it is the same 96
matched cell pairs as 44_race_contrast_test.py, recomputed here from model_outputs_v2.csv.

GATES (all four must pass or nothing downstream is trustworthy):
  G1  White anchor, 2005-2018   ->  2.91   (published)
  G2  Asian anchor, 2005-2018   ->  2.16   (published)
  G3  White design SE, 2005-2018 -> 0.0498 (published, anchor_design_se.csv)
  G4  simulated Asian minus White contrast -> -1.074 (published)

Writes analysis/asian_window_confound.csv (anchors by window) and
analysis/asian_window_contrasts.csv (contrast tests under both anchor sets).
Does not modify the ground-truth table or any existing receipt.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(BASE, "data", "nhanes_raw")
OUT = os.path.join(BASE, "analysis")

CYCLES = {"D": 2005, "E": 2007, "F": 2009, "G": 2011, "H": 2013, "I": 2015, "J": 2017}
MATCHED = ["G", "H", "I", "J"]          # the cycles that carry RIDRETH3
PHQ8 = ["DPQ0%d0" % i for i in range(1, 9)]

CIS = ["Cisgender Man", "Cisgender Woman"]
STRATUM = ["model", "prompt_condition", "gender", "ses_normalized", "relationship"]
CONTRASTS = ["Black", "Hispanic", "Asian"]


# ----------------------------------------------------------------------------- NHANES side
def load():
    frames = []
    for suf, yr in CYCLES.items():
        demo = pd.read_sas(os.path.join(RAW, "DEMO_%s.xpt" % suf), format="xport")
        dpq = pd.read_sas(os.path.join(RAW, "DPQ_%s.xpt" % suf), format="xport")
        keep = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3", "INDFMPIR",
                "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]
        df = demo[[c for c in keep if c in demo.columns]].copy()
        if "RIDRETH3" not in df.columns:
            df["RIDRETH3"] = np.nan
        df = df.merge(dpq[["SEQN"] + [c for c in PHQ8 if c in dpq.columns]], on="SEQN", how="left")
        df["cycle"] = suf
        df["year"] = yr
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def phq8_total(df):
    d = df[PHQ8].copy()
    for c in PHQ8:
        d[c] = d[c].where(d[c] <= 3, np.nan)
    return d.sum(axis=1, min_count=len(PHQ8))


def race_of(r):
    if pd.notna(r["RIDRETH3"]) and r["RIDRETH3"] == 6:
        return "Asian"
    return {3: "White", 4: "Black", 1: "MexAm", 2: "OtherHisp"}.get(r["RIDRETH1"], "Other/Multi")


def wmean(sub):
    x = sub["PHQ8"].to_numpy(float)
    w = sub["WTMEC2YR"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if len(x) == 0:
        return dict(n=0, w_mean=np.nan, w_sd=np.nan)
    mu = np.sum(w * x) / np.sum(w)
    sd = np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w))
    return dict(n=len(x), w_mean=round(mu, 6), w_sd=round(sd, 6))


def design_se(df):
    """Taylor linearisation for a weighted mean, with-replacement PSU variance (as in 26)."""
    d = df[np.isfinite(df.WTMEC2YR) & (df.WTMEC2YR > 0) & df.PHQ8.notna()]
    if len(d) == 0:
        return np.nan
    W = d.WTMEC2YR.sum()
    mu = (d.WTMEC2YR * d.PHQ8).sum() / W
    z = d.WTMEC2YR * (d.PHQ8 - mu) / W
    var = 0.0
    for _, st in pd.DataFrame({"stratum": d.SDMVSTRA.astype(int),
                               "psu": d.SDMVPSU.astype(int), "z": z}).groupby("stratum"):
        psu = st.groupby("psu").z.sum()
        n = len(psu)
        if n < 2:
            continue
        var += n / (n - 1) * ((psu - psu.mean()) ** 2).sum()
    return float(np.sqrt(var))


raw = load()
raw["PHQ8"] = phq8_total(raw)
adults = raw[raw.RIDAGEYR >= 18].dropna(subset=["PHQ8"]).copy()
adults["race"] = adults.apply(race_of, axis=1)

GROUPS = ["White", "Black", "Asian", "MexAm", "OtherHisp"]
WINDOWS = {"paper 2005-2018": list(CYCLES), "matched 2011-2018": MATCHED}


def anchor(window_cycles, group):
    sub = adults[adults.cycle.isin(window_cycles)]
    sub = sub[sub.race.isin(["MexAm", "OtherHisp"])] if group == "Hispanic (pooled)" \
        else sub[sub.race == group]
    s = wmean(sub)
    s["se_design"] = round(design_se(sub), 4) if s["n"] else np.nan
    return s


rows = []
for wname, wcyc in WINDOWS.items():
    for g in GROUPS + ["Hispanic (pooled)"]:
        s = anchor(wcyc, g)
        s.update(window=wname, group=g)
        rows.append(s)
anchors = pd.DataFrame(rows)[["window", "group", "n", "w_mean", "w_sd", "se_design"]]

A = {(r.window, r.group): r for r in anchors.itertuples()}


# ----------------------------------------------------------------------------- model side
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
m = m[m.gender.isin(CIS)]
cell = (m.groupby(["model", "profile_id", "prompt_condition", "race", "gender",
                   "ses_normalized", "relationship"], as_index=False).phq8_total.mean())

sim = {}
for g in CONTRASTS:
    a = cell[cell.race == g].set_index(STRATUM).phq8_total
    b = cell[cell.race == "White"].set_index(STRATUM).phq8_total
    for nm, idx in (("contrast", a.index), ("White", b.index)):
        if idx.has_duplicates:
            raise SystemExit("%s: %s stratum key is not unique, pairing would be wrong" % (g, nm))
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).to_numpy(float)
    sim[g] = dict(n=len(diff), obs=float(diff.mean()),
                  se=float(diff.std(ddof=1) / np.sqrt(len(diff))))


# ----------------------------------------------------------------------------- gates
gw = float(A[("paper 2005-2018", "White")].w_mean)
ga = float(A[("paper 2005-2018", "Asian")].w_mean)
gse = float(A[("paper 2005-2018", "White")].se_design)
gsim = sim["Asian"]["obs"]

gates = [("G1 White anchor 2005-2018 == 2.91", gw, 2.91, 0.005),
         ("G2 Asian anchor 2005-2018 == 2.16", ga, 2.16, 0.005),
         ("G3 White design SE 2005-2018 == 0.0498", gse, 0.0498, 0.0005),
         ("G4 simulated Asian-White == -1.074", gsim, -1.0736, 0.002)]
gate_ok = all(abs(v - t) < tol for _, v, t, tol in gates)


# ----------------------------------------------------------------------------- contrasts
GTK = {"Black": "Black", "Hispanic": "Hispanic (pooled)", "Asian": "Asian"}
crows = []
for wname in WINDOWS:
    for g in CONTRASTS:
        pop = float(A[(wname, GTK[g])].w_mean) - float(A[(wname, "White")].w_mean)
        se_pop = float(np.hypot(A[(wname, GTK[g])].se_design, A[(wname, "White")].se_design))
        s = sim[g]
        se_gap = float(np.hypot(s["se"], se_pop))
        t1 = (s["obs"] - pop) / se_gap
        p_pop = float(stats.t.sf(abs(t1), s["n"] - 1) * 2)
        ratio = abs(s["obs"]) / abs(pop) if pop else np.nan
        crows.append(dict(
            anchor_window=wname, contrast="%s minus White" % g, n_pairs=s["n"],
            simulated=round(s["obs"], 4), sim_se=round(s["se"], 4),
            population=round(pop, 4), pop_se=round(se_pop, 4),
            gap=round(s["obs"] - pop, 4), t_vs_pop=round(t1, 2), p_vs_pop=p_pop,
            exaggeration_ratio=round(ratio, 4),
            exaggeration_pct=round(100 * (ratio - 1), 1),
            sign_flip=bool(np.sign(s["obs"]) != np.sign(pop))))
contr = pd.DataFrame(crows)
# Benjamini-Hochberg across the three contrasts, within each anchor window (as in 44).
contr["q_vs_pop"] = np.nan
for wname in WINDOWS:
    sel = contr.anchor_window == wname
    order = contr.loc[sel, "p_vs_pop"].rank(method="first")
    contr.loc[sel, "q_vs_pop"] = (contr.loc[sel, "p_vs_pop"] * sel.sum() / order).clip(upper=1.0)
contr["q_vs_pop"] = contr.q_vs_pop.round(6)

# Per-cycle White and Asian means, to show the trend the confound depends on.
trend = []
for suf, yr in CYCLES.items():
    r = {"cycle": suf, "years": "%d-%d" % (yr, yr + 1)}
    for g in ("White", "Asian"):
        s = wmean(adults[(adults.cycle == suf) & (adults.race == g)])
        r["%s_n" % g] = s["n"]
        r["%s_mean" % g] = round(s["w_mean"], 4) if s["n"] else np.nan
    trend.append(r)
trend = pd.DataFrame(trend)

os.makedirs(OUT, exist_ok=True)
anchors.to_csv(os.path.join(OUT, "asian_window_confound.csv"), index=False)
contr.to_csv(os.path.join(OUT, "asian_window_contrasts.csv"), index=False)


# ----------------------------------------------------------------------------- report
pd.set_option("display.width", 170)
print("IS THE ASIAN RACIAL CONTRAST PERIOD-CONFOUNDED?")
print("NHANES adults 18+, PHQ-8, MEC-weighted, conventions of 03_compute_groundtruth.py")
print("=" * 110)

print("\nGATE")
for name, val, tgt, tol in gates:
    print("  %-42s got %9.4f  target %9.4f  %s"
          % (name, val, tgt, "PASS" if abs(val - tgt) < tol else "FAIL"))
print("  OVERALL GATE: %s" % ("PASS" if gate_ok else "*** FAIL ***"))
if not gate_ok:
    print("\n  !!! GATE FAILED. Data handling does not match the published pipeline.")
    print("  !!! Do NOT treat any number below as trustworthy.")

print("\n[1-3] ANCHORS BY WINDOW")
print(anchors.to_string(index=False))

print("\nPer-cycle White and Asian anchors (the trend the confound depends on)")
print(trend.to_string(index=False))

wp = float(A[("paper 2005-2018", "White")].w_mean)
wm = float(A[("matched 2011-2018", "White")].w_mean)
ap = float(A[("paper 2005-2018", "Asian")].w_mean)
am = float(A[("matched 2011-2018", "Asian")].w_mean)
print("\n  White  2005-2018 %.4f -> 2011-2018 %.4f   (%+.4f)" % (wp, wm, wm - wp))
print("  Asian  2005-2018 %.4f -> 2011-2018 %.4f   (%+.4f, should be ~0 by construction)"
      % (ap, am, am - ap))

print("\n[4] WHITE MINUS ASIAN POPULATION GAP")
print("  (a) paper's mismatched windows : %.4f points" % (wp - ap))
print("  (b) matched 2011-2018 window   : %.4f points" % (wm - am))
print("      period confound in the gap : %.4f points (%.1f%% of the published gap)"
      % ((wm - am) - (wp - ap), 100 * ((wm - am) - (wp - ap)) / (wp - ap)))

print("\n[5-6] CONTRAST TESTS UNDER BOTH ANCHOR SETS")
print(contr[["anchor_window", "contrast", "n_pairs", "simulated", "population", "gap",
             "exaggeration_ratio", "exaggeration_pct", "q_vs_pop", "sign_flip"]]
      .to_string(index=False))

pa = contr[(contr.anchor_window == "paper 2005-2018") & contr.contrast.str.startswith("Asian")].iloc[0]
ma = contr[(contr.anchor_window == "matched 2011-2018") & contr.contrast.str.startswith("Asian")].iloc[0]
print("\n  Asian contrast, published    : simulated %.4f vs population %.4f -> ratio %.3f (+%.1f%%)"
      % (pa.simulated, pa.population, pa.exaggeration_ratio, pa.exaggeration_pct))
print("  Asian contrast, matched window: simulated %.4f vs population %.4f -> ratio %.3f (+%.1f%%)"
      % (ma.simulated, ma.population, ma.exaggeration_ratio, ma.exaggeration_pct))
share = 1 - (ma.exaggeration_ratio - 1) / (pa.exaggeration_ratio - 1)
print("  Share of the published exaggeration attributable to the window mismatch: %.1f%%"
      % (100 * share))
print("  Residual exaggeration after matching: %+.1f%% (published claim: about two fifths, +42%%)"
      % ma.exaggeration_pct)

print("\nWritten -> %s" % os.path.join(OUT, "asian_window_confound.csv"))
print("Written -> %s" % os.path.join(OUT, "asian_window_contrasts.csv"))
