"""
Receipts for the v2 primary Table 2: severity residuals on CISGENDER personas only,
frame-matching the model marginal to the population anchor (see paper Section 3.2).
Pooled full-grid residuals (04_recompute_analysis.py) are retained as the sensitivity column.
Emits analysis/bias_residuals_CISONLY.csv. Idempotent, no randomness.
"""
import pandas as pd, numpy as np, os

BASE = os.path.join(os.path.dirname(__file__), "..")
m = pd.read_csv(os.path.join(BASE,"data","model_outputs.csv"))
gt = pd.read_csv(os.path.join(BASE,"groundtruth","phq8_groundtruth_nhanes_2005_2018.csv"))
G = {r.group: r for _, r in gt.iterrows()}
m["phq8_total"]=m.phq8_total.clip(0,24)
m["ses_clean"]=m.ses.str.replace('"','',regex=False).str.split(" ").str[0]
CIS = ["Cisgender Man","Cisgender Woman"]

rows=[]
def emit(dim, grp, sub, gtkey):
    x=sub.phq8_total.values; mm=x.mean(); sd=x.std(ddof=1); n=len(x); se=sd/np.sqrt(n)
    g=G[gtkey]; res=mm-g.w_mean
    # 4 decimals so 2-decimal display rounding never lands on a boundary
    rows.append(dict(dimension=dim, group=grp, n=n,
        model_mean=round(mm,4), model_sd=round(sd,4),
        ci_lo=round(mm-1.96*se,4), ci_hi=round(mm+1.96*se,4),
        gt_mean=round(g.w_mean,4), gt_sd=round(g.w_sd,4),
        residual=round(res,4), resid_ci_lo=round(res-1.96*se,4), resid_ci_hi=round(res+1.96*se,4),
        cohens_d=round(res/g.w_sd,4), t_one_sample=round(res/se,2)))
for mg,gg in [("White","White"),("Black","Black"),("Asian","Asian"),("Hispanic","Hispanic (pooled)")]:
    emit("race",mg,m[(m.race==mg)&(m.gender.isin(CIS))],gg)
for mg,gg in [("Cisgender Man","Men"),("Cisgender Woman","Women")]:
    emit("gender",mg,m[m.gender==mg],gg)
for mg,gg in [("Low","Low"),("Middle","Middle"),("High","High")]:
    emit("ses",mg,m[(m.ses_clean==mg)&(m.gender.isin(CIS))],gg)

df=pd.DataFrame(rows)
df.to_csv(os.path.join(BASE,"analysis","bias_residuals_CISONLY.csv"),index=False)
print(df.to_string(index=False))
lo=m[(m.ses_clean=="Low")&(m.gender.isin(CIS))].phq8_total.mean()
hi=m[(m.ses_clean=="High")&(m.gender.isin(CIS))].phq8_total.mean()
print(f"\nSES gradient cis-only: simulated {lo-hi:.3f} vs GT {G['Low'].w_mean-G['High'].w_mean:.3f} "
      f"(ratio {(lo-hi)/(G['Low'].w_mean-G['High'].w_mean):.2f})")
print(f"headline: residual {df.residual.min():+.2f} to {df.residual.max():+.2f}, "
      f"d {df.cohens_d.min():.2f} to {df.cohens_d.max():.2f}")
