"""
Validation gate: reproduce Patel et al. 2019 (Depression & Anxiety, NHANES 2005-2016)
published PHQ-9 descriptive means, to PROVE this pipeline is correct before it is trusted
to compute new ground truth.

Patel 2019 Table 1 published values (PHQ-9, 0-27), which our audit verified are the true
origin of the paper's race/gender GT:
    Overall  3.20 (SD 4.27)
    Men      2.67 (SD 3.87)   Women 3.72 (SD 4.57)
    White    3.19 (SD 4.19)   Black 3.21 (SD 4.35)   Asian 2.23 (SD 3.07)
    MexAm    3.15
If our computed values match these, the extraction + weighting method is validated.
"""
import pandas as pd, numpy as np, glob, os

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "nhanes_raw")
CYCLES = {  # suffix -> (start_year, in Patel 2005-2016 window?)
    "D": (2005, True), "E": (2007, True), "F": (2009, True),
    "G": (2011, True), "H": (2013, True), "I": (2015, True),
    "J": (2017, False),
}
PHQ9_ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]   # DPQ010..DPQ090

def load_all():
    frames = []
    for suf, (yr, in_patel) in CYCLES.items():
        demo = pd.read_sas(os.path.join(RAW, f"DEMO_{suf}.xpt"), format="xport")
        dpq  = pd.read_sas(os.path.join(RAW, f"DPQ_{suf}.xpt"),  format="xport")
        keep_demo = ["SEQN","RIDAGEYR","RIAGENDR","RIDRETH1","INDFMPIR","WTMEC2YR","SDMVSTRA","SDMVPSU"]
        if "RIDRETH3" in demo.columns: keep_demo.append("RIDRETH3")
        demo = demo[[c for c in keep_demo if c in demo.columns]]
        df = demo.merge(dpq[["SEQN"]+[c for c in PHQ9_ITEMS if c in dpq.columns]], on="SEQN", how="left")
        df["cycle"] = suf; df["year"] = yr; df["in_patel"] = in_patel
        if "RIDRETH3" not in df.columns: df["RIDRETH3"] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def phq(df, items):
    d = df[items].copy()
    for c in items:
        d[c] = d[c].where(d[c] <= 3, np.nan)   # 7=refused,9=dontknow,.=missing -> NaN
    return d.sum(axis=1, min_count=len(items))  # require ALL items present

def wmean_wsd(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    mean = np.sum(w*x)/np.sum(w)
    var = np.sum(w*(x-mean)**2)/np.sum(w)     # population (weighted) variance
    return mean, np.sqrt(var), len(x)

df = load_all()
df["PHQ9"] = phq(df, PHQ9_ITEMS)
adult = df[(df.RIDAGEYR >= 18)].copy()

# Patel window: 2005-2016 (6 cycles). Pooled MEC weight = WTMEC2YR / n_cycles
pat = adult[adult.in_patel].copy()
n_cycles = pat["cycle"].nunique()
pat["w"] = pat["WTMEC2YR"] / n_cycles

# race: combine RIDRETH1 (all cycles) for White/Black/MexAm/OtherHisp; Asian only from RIDRETH3 (2011+)
def race_of(r):
    e1, e3 = r["RIDRETH1"], r["RIDRETH3"]
    if pd.notna(e3) and e3 == 6: return "Asian"
    if e1 == 3: return "White"
    if e1 == 4: return "Black"
    if e1 == 1: return "MexAm"
    if e1 == 2: return "OtherHisp"
    return "Other/Multi"
pat["race"] = pat.apply(race_of, axis=1)
pat["sex"] = pat["RIAGENDR"].map({1:"Men", 2:"Women"})

PATEL = {"Overall":(3.20,4.27),"Men":(2.67,3.87),"Women":(3.72,4.57),
         "White":(3.19,4.19),"Black":(3.21,4.35),"Asian":(2.23,3.07),"MexAm":(3.15,None)}

print("="*74)
print("VALIDATION GATE: reproduce Patel et al. 2019 PHQ-9 (NHANES 2005-2016)")
print("="*74)
print(f"{'group':<10} {'Patel':>14} | {'WEIGHTED mean(SD) n':>26} | {'UNWEIGHTED':>16}")
def row(label, sub):
    pm, ps = PATEL.get(label,(None,None))
    wm, ws, n = wmean_wsd(sub["PHQ9"], sub["w"])
    um, us, _ = wmean_wsd(sub["PHQ9"], np.ones(len(sub)))
    tag = f"{pm} ({ps})" if pm is not None else "-"
    print(f"{label:<10} {tag:>14} | {wm:6.2f} ({ws:4.2f}) n={n:<7} | {um:5.2f} ({us:4.2f})")

pv = pat.dropna(subset=["PHQ9"])
row("Overall", pv)
for g in ["Men","Women"]: row(g, pv[pv.sex==g])
for g in ["White","Black","Asian","MexAm"]: row(g, pv[pv.race==g])
print("\nAsian n-cycles note: Asian uses RIDRETH3, available",
      sorted(pat.loc[pat.RIDRETH3.notna(),'year'].unique().tolist()))
