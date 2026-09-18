"""
Receipts for the four stability/coherence numbers that previously lived only in prose:
individual test-retest r + Spearman, per-model 5-category flip rates, per-model gateway violations.
Also builds the v2 release file: model_outputs_v2.csv with an explicit prompt_condition column
(reconstructed from the SES quote-form artifact; 694 same-form pairs assigned by file order and
flagged condition_source='file_order') and normalized SES labels.
Idempotent; SEED-free (no randomness).
"""
import pandas as pd, numpy as np, os
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), "..")
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
OUT = os.path.join(BASE, "analysis")

m["ses_clean"] = m["ses"].str.replace('"','',regex=False).str.split(" ").str[0]
m["is_quoted"] = m["ses"].str.startswith('"')
m["phq8_total"] = m["phq8_total"].clip(0, 24)

# ---- per-model gateway violations (elevated PHQ8>=10 needs phq8_1>=2 OR phq8_2>=2) ----
rows=[]
elev_all = m[m.phq8_total>=10]
viol_all = elev_all[(elev_all.phq8_1<2)&(elev_all.phq8_2<2)]
rows.append(dict(model="POOLED", elevated=len(elev_all), violations=len(viol_all),
                 violation_rate_pct=round(100*len(viol_all)/len(elev_all),2)))
for mod in sorted(m.model.unique()):
    e = elev_all[elev_all.model==mod]; v = viol_all[viol_all.model==mod]
    rows.append(dict(model=mod, elevated=len(e), violations=len(v),
                     violation_rate_pct=round(100*len(v)/len(e),2)))

# ---- pairing: reconstruct condition; flag file-order fallback pairs ----
g = m.groupby(["model","profile_id","iteration"])
assert (g.size()==2).all()
def pair(sub):
    b = sub[~sub.is_quoted]; q = sub[sub.is_quoted]
    if len(b)==1 and len(q)==1:
        return pd.Series({"bare":b.phq8_total.iloc[0],"quoted":q.phq8_total.iloc[0],
                          "i_bare":b.index[0],"i_quoted":q.index[0],"fallback":0})
    i0,i1 = sub.index[0], sub.index[1]
    return pd.Series({"bare":sub.phq8_total.iloc[0],"quoted":sub.phq8_total.iloc[1],
                      "i_bare":i0,"i_quoted":i1,"fallback":1})
P = g.apply(pair, include_groups=False).reset_index()
cat = lambda x: pd.cut(x,[-1,4,9,14,19,24],labels=[0,1,2,3,4]).astype(int)
P["ca"],P["cb"] = cat(P.bare), cat(P.quoted)

pear = stats.pearsonr(P.bare,P.quoted); spear = stats.spearmanr(P.bare,P.quoted)
stab=[dict(metric="pearson_r",value=round(pear[0],4)),
      dict(metric="spearman_rho",value=round(spear[0],4)),
      dict(metric="n_pairs",value=len(P)),
      dict(metric="fallback_pairs_file_order",value=int(P.fallback.sum())),
      dict(metric="flip_rate_5cat_pct_POOLED",value=round(100*(P.ca!=P.cb).mean(),2))]
for mod in sorted(P.model.unique()):
    pm=P[P.model==mod]
    stab.append(dict(metric=f"flip_rate_5cat_pct_{mod}",value=round(100*(pm.ca!=pm.cb).mean(),2)))

pd.DataFrame(rows).to_csv(os.path.join(OUT,"gateway_per_model.csv"),index=False)
pd.DataFrame(stab).to_csv(os.path.join(OUT,"stability_receipts.csv"),index=False)

# ---- v2 release file: explicit condition column + normalized SES ----
m2 = m.copy()
m2["prompt_condition"]="bare"; m2["condition_source"]="quote_artifact"
qidx = P.set_index("i_quoted").index
m2.loc[m2.index.isin(qidx),"prompt_condition"]="quoted"
fb = P[P.fallback==1]
m2.loc[m2.index.isin(fb.i_bare)|m2.index.isin(fb.i_quoted),"condition_source"]="file_order"
m2["ses"]=m2["ses_clean"]
m2 = m2.drop(columns=["ses_clean","is_quoted"])
m2.to_csv(os.path.join(BASE,"data","model_outputs_v2.csv"),index=False)

print("gateway_per_model.csv:");  print(pd.DataFrame(rows).to_string(index=False))
print("\nstability_receipts.csv:"); print(pd.DataFrame(stab).to_string(index=False))
print(f"\nmodel_outputs_v2.csv written: {len(m2)} rows, prompt_condition counts:")
print(m2.groupby(["prompt_condition","condition_source"]).size())
