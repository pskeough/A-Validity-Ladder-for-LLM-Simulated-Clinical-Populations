"""
Second independent validation of the NHANES pipeline: reproduce the published national depression
PREVALENCE (Brody, Pratt & Hughes 2018, NCHS Data Brief No. 303): among US adults 20+, 2013-2016,
PHQ-9 >= 10 => 8.1% overall (women 10.4%, men 5.5%). A different source, metric, and subset than the
Patel 2019 mean-reproduction (02), so a genuinely independent check.
"""
import pandas as pd, numpy as np, os
RAW=os.path.join(os.path.dirname(__file__),"..","data","nhanes_raw")
frames=[]
for suf in ("H","I"):  # 2013-14 + 2015-16 = 2013-2016
    demo=pd.read_sas(os.path.join(RAW,f"DEMO_{suf}.xpt"),format="xport")
    dpq=pd.read_sas(os.path.join(RAW,f"DPQ_{suf}.xpt"),format="xport")
    items=[f"DPQ0{i}0" for i in range(1,10)]
    df=demo[["SEQN","RIDAGEYR","RIAGENDR","WTMEC2YR"]].merge(dpq[["SEQN"]+items],on="SEQN",how="left")
    for c in items: df[c]=df[c].where(df[c]<=3,np.nan)
    df["PHQ9"]=df[items].sum(axis=1,min_count=9); frames.append(df)
a=pd.concat(frames); a=a[(a.RIDAGEYR>=20)&a.PHQ9.notna()].copy(); a["w"]=a.WTMEC2YR/2
prev=lambda s: 100*np.sum(s.w*(s.PHQ9>=10))/np.sum(s.w)
print("NHANES 2013-2016 depression prevalence (PHQ-9>=10, 20+) vs Brody 2018:")
print(f"  Overall {prev(a):.1f}%  (pub 8.1)  | Women {prev(a[a.RIAGENDR==2]):.1f}%  (pub 10.4)"
      f"  | Men {prev(a[a.RIAGENDR==1]):.1f}%  (pub 5.5)")
