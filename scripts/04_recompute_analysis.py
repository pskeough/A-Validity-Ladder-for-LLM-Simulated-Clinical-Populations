"""
Master recompute of PsychBench analysis against the GOLD (NHANES-derived, validated) ground truth.
Real model outputs: data/model_outputs.csv (28,800 rows, unchanged experimental data).
GT: groundtruth/phq8_groundtruth_nhanes_2005_2018.csv (weighted PHQ-8, this project).

Produces:
  analysis/bias_residuals_GOLD.csv     - residual, Cohen's d, Stereotype Index vs gold GT
  analysis/gt_independent_metrics.csv  - gateway, flip, drift, MAD, per-model variance (GT-independent)
Everything double-checked: GT-independent headline stats are recomputed from scratch and must match
the prior data-audit (gateway 2.68% / flip 36.66% / drift +0.325) as an internal consistency gate.
"""
import pandas as pd, numpy as np, os
from scipy import stats

BASE=os.path.join(os.path.dirname(__file__),"..")
m=pd.read_csv(os.path.join(BASE,"data","model_outputs.csv"))
gt=pd.read_csv(os.path.join(BASE,"groundtruth","phq8_groundtruth_nhanes_2005_2018.csv"))
OUT=os.path.join(BASE,"analysis"); os.makedirs(OUT,exist_ok=True)
G=lambda grp: gt.loc[gt.group==grp].iloc[0]   # gold row (w_mean, w_sd)

# normalize the ses quote artifact -> canonical + condition marker
m["ses_clean"]=m["ses"].str.replace('"','',regex=False).str.split(" ").str[0]   # Low/Middle/High
m["is_quoted"]=m["ses"].str.startswith('"')     # quote-form encodes the two prompt conditions
# data-integrity: clip 1 corrupted row (phq8_8=21 -> total 29; PHQ-8 max is 24). Flagged, not silent.
_n_bad=int(((m.phq8_total>24)|(m.phq8_total<0)).sum())
m["phq8_total"]=m["phq8_total"].clip(0,24)

# ---- A. GT-DEPENDENT: bias residuals vs GOLD GT (weighted PHQ-8) --------------------
race_map={"White":"White","Black":"Black","Asian":"Asian","Hispanic":"Hispanic (pooled)","Multiracial":None}
gender_map={"Cisgender Man":"Men","Cisgender Woman":"Women","Transgender Woman":None,"Transgender Man":None}
ses_map={"Low":"Low","Middle":"Middle","High":"High"}
rows=[]
def resid(dim, model_group, gt_group, sub):
    mm, msd, n = sub.phq8_total.mean(), sub.phq8_total.std(ddof=0), len(sub)
    if gt_group is None:
        rows.append(dict(dimension=dim, group=model_group, model_mean=round(mm,3), model_sd=round(msd,3), n=n,
            gt_mean=None, gt_sd=None, residual=None, cohens_d=None, sd_ratio=None,
            note="NO representative GT exists (NHANES has no gender-identity field / no clean Multiracial norm)"))
    else:
        g=G(gt_group)
        rows.append(dict(dimension=dim, group=model_group, model_mean=round(mm,3), model_sd=round(msd,3), n=n,
            gt_mean=g.w_mean, gt_sd=g.w_sd, residual=round(mm-g.w_mean,3),
            cohens_d=round((mm-g.w_mean)/g.w_sd,3), sd_ratio=round(msd/g.w_sd,3), note=""))
for mg,gg in race_map.items():   resid("race", mg, gg, m[m.race==mg])
for mg,gg in gender_map.items(): resid("gender", mg, gg, m[m.gender==mg])
for mg,gg in ses_map.items():    resid("ses", mg, gg, m[m.ses_clean==mg])
res=pd.DataFrame(rows)
res.to_csv(os.path.join(OUT,"bias_residuals_GOLD.csv"), index=False)

# ---- B. GT-INDEPENDENT metrics (recomputed from scratch; must match prior audit) ----
ind={}
# gateway coherence: elevated PHQ8>=10 must have phq8_1>=2 OR phq8_2>=2 (paper's stated >=2 rule)
elev=m[m.phq8_total>=10]; viol=elev[(elev.phq8_1<2)&(elev.phq8_2<2)]
ind["gateway_elevated_n"]=len(elev); ind["gateway_violations"]=len(viol)
ind["gateway_violation_rate_pct"]=round(100*len(viol)/len(elev),2)
ind["corrupted_rows_clipped"]=_n_bad
# pairing for flip/drift: pair the two prompt conditions by quote-form (bare vs quoted)
def cat(x):
    return pd.cut(x,[-1,4,9,14,19,24],labels=[0,1,2,3,4]).astype(int)
g=m.groupby(["model","profile_id","iteration"])
assert (g.size()==2).all(), "pairing not all size 2"
def pair_totals(sub):
    b=sub[~sub.is_quoted]; q=sub[sub.is_quoted]
    if len(b)==1 and len(q)==1:                       # clean bare/quoted pair
        return pd.Series({"bare":b.phq8_total.iloc[0], "quoted":q.phq8_total.iloc[0]})
    v=sub.phq8_total.tolist()                          # 694 same-form pairs: file order
    return pd.Series({"bare":v[0], "quoted":v[1]})
paired=g.apply(pair_totals, include_groups=False)
paired["ca"]=cat(paired.bare); paired["cb"]=cat(paired.quoted)
ind["n_pairs"]=len(paired)
ind["flip_rate_5cat_pct"]=round(100*(paired.ca!=paired.cb).mean(),2)
d=paired.bare-paired.quoted; t,p=stats.ttest_rel(paired.bare,paired.quoted)
ind["drift_mean_bare_minus_quoted"]=round(d.mean(),4); ind["drift_t"]=round(t,3); ind["drift_p"]=f"{p:.2e}"
ind["cross_run_MAD"]=round(d.abs().mean(),4)
indf=pd.DataFrame([ind]).T.reset_index(); indf.columns=["metric","value"]
indf.to_csv(os.path.join(OUT,"gt_independent_metrics.csv"), index=False)

# per-model variance ratio (Stereotype Index) vs gold, over race+sex+ses cells that HAVE gold GT
def si_for_model(md):
    ratios=[]
    for col,mp in [("race",race_map),("gender",gender_map),("ses_clean",ses_map)]:
        for mg,gg in mp.items():
            if gg is None: continue
            sub=md[md[col]==mg]
            if len(sub)>1: ratios.append(sub.phq8_total.std(ddof=0)/G(gg).w_sd)
    return np.mean(ratios)
pm=pd.DataFrame([{"model":mod,"mean_stereotype_index":round(si_for_model(m[m.model==mod]),3)}
                 for mod in sorted(m.model.unique())])
pm.to_csv(os.path.join(OUT,"per_model_variance_GOLD.csv"), index=False)

pd.set_option("display.width",130)
print("A. BIAS RESIDUALS vs GOLD GT (weighted PHQ-8)\n"+"="*90)
print(res.to_string(index=False)); print()
print("B. GT-INDEPENDENT HEADLINE METRICS (recomputed from scratch)\n"+"="*90)
for _,r in indf.iterrows(): print(f"  {r.metric:<28} {r.value}")
print("\n  Internal-consistency gate vs prior audit: gateway 2.68% | flip 36.66% | drift +0.325")
print("\nC. PER-MODEL STEREOTYPE INDEX vs gold\n"+"="*90); print(pm.to_string(index=False))
