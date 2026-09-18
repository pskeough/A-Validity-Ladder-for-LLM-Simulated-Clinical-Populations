"""
GOLD ground truth: author-derived PHQ-8 population norms from NHANES 2005-2018 microdata.
Same scale as the PsychBench model outputs (PHQ-8, 0-24). Survey-weighted (MEC) means + SDs,
which are the nationally-representative population parameters (the correct epidemiological
benchmark). Unweighted reported alongside for continuity with Patel 2019.

Pipeline validated in 02_validate_pipeline.py (reproduces Patel 2019 PHQ-9 to +/-0.02).
Note: constant pooling divisors cancel in weighted mean/SD, so raw WTMEC2YR is used for
point estimates (design-based SEs would need SDMVSTRA/SDMVPSU; not required for a benchmark).
"""
import pandas as pd, numpy as np, os

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "nhanes_raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "groundtruth")
CYCLES = {"D":2005,"E":2007,"F":2009,"G":2011,"H":2013,"I":2015,"J":2017}
PHQ8_ITEMS = [f"DPQ0{i}0" for i in range(1, 9)]   # DPQ010..DPQ080 (excludes item 9 suicidality)
PHQ9_ITEMS = PHQ8_ITEMS + ["DPQ090"]

def load_all():
    frames=[]
    for suf, yr in CYCLES.items():
        demo = pd.read_sas(os.path.join(RAW,f"DEMO_{suf}.xpt"), format="xport")
        dpq  = pd.read_sas(os.path.join(RAW,f"DPQ_{suf}.xpt"),  format="xport")
        cols=["SEQN","RIDAGEYR","RIAGENDR","RIDRETH1","INDFMPIR","WTMEC2YR"]
        if "RIDRETH3" in demo.columns: cols.append("RIDRETH3")
        df=demo[[c for c in cols if c in demo.columns]].merge(
            dpq[["SEQN"]+[c for c in PHQ9_ITEMS if c in dpq.columns]], on="SEQN", how="left")
        if "RIDRETH3" not in df.columns: df["RIDRETH3"]=np.nan
        df["cycle"]=suf; df["year"]=yr
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def total(df, items):
    d=df[items].copy()
    for c in items: d[c]=d[c].where(d[c]<=3, np.nan)
    return d.sum(axis=1, min_count=len(items))

def wstats(sub, col):
    x=sub[col].to_numpy(float); w=sub["WTMEC2YR"].to_numpy(float)
    m=np.isfinite(x)&np.isfinite(w)&(w>0); x,w=x[m],w[m]
    if len(x)==0: return dict(n=0, w_mean=np.nan, w_sd=np.nan, u_mean=np.nan, u_sd=np.nan)
    wm=np.sum(w*x)/np.sum(w); wsd=np.sqrt(np.sum(w*(x-wm)**2)/np.sum(w))
    # 6 decimals: 2-decimal display rounding in the paper must never sit on a x.xx5 boundary
    return dict(n=len(x), w_mean=round(wm,6), w_sd=round(wsd,6),
                u_mean=round(x.mean(),6), u_sd=round(x.std(ddof=0),6))

df=load_all()
df["PHQ8"]=total(df,PHQ8_ITEMS); df["PHQ9"]=total(df,PHQ9_ITEMS)
a=df[df.RIDAGEYR>=18].copy()

def race_of(r):
    if pd.notna(r["RIDRETH3"]) and r["RIDRETH3"]==6: return "Asian"
    return {3:"White",4:"Black",1:"MexAm",2:"OtherHisp"}.get(r["RIDRETH1"],"Other/Multi")
a["race"]=a.apply(race_of,axis=1)
a["sex"]=a["RIAGENDR"].map({1:"Men",2:"Women"})
# Income terciles via poverty-income ratio (standard NHANES/USDA thresholds)
def ses_of(pir):
    if pd.isna(pir): return None
    if pir<1.3: return "Low"
    if pir<=3.5: return "Middle"
    return "High"
a["ses"]=a["INDFMPIR"].apply(ses_of)

av=a.dropna(subset=["PHQ8"])
rows=[]
def add(dim, group, sub):
    s=wstats(sub,"PHQ8"); s.update(dimension=dim, group=group); rows.append(s)

add("overall","All adults 18+", av)
for g in ["White","Black","Asian","MexAm","OtherHisp"]: add("race", g, av[av.race==g])
add("race","Hispanic (pooled)", av[av.race.isin(["MexAm","OtherHisp"])])
for g in ["Men","Women"]: add("sex", g, av[av.sex==g])
for g in ["Low","Middle","High"]: add("ses", g, av[av.ses==g])

gt=pd.DataFrame(rows)[["dimension","group","n","w_mean","w_sd","u_mean","u_sd"]]
os.makedirs(OUT, exist_ok=True)
gt.to_csv(os.path.join(OUT,"phq8_groundtruth_nhanes_2005_2018.csv"), index=False)

pd.set_option("display.width",120)
print("GOLD PHQ-8 GROUND TRUTH  (NHANES 2005-2018, adults 18+, MEC-weighted)")
print("="*78)
print(gt.to_string(index=False))
ov=gt[gt.group=="All adults 18+"].iloc[0]
print(f"\nSanity: PHQ-8 overall weighted {ov.w_mean} vs PHQ-9 overall "
      f"{wstats(av,'PHQ9')['w_mean']} (PHQ-8 should be slightly lower). ")
print("SES gradient (Low - High, weighted):",
      round(gt[gt.group=='Low'].w_mean.iat[0]-gt[gt.group=='High'].w_mean.iat[0],2),"points")
print(f"\nWritten -> {os.path.join(OUT,'phq8_groundtruth_nhanes_2005_2018.csv')}")
