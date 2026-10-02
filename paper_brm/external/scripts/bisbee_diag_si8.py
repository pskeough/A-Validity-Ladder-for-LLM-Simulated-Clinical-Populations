"""Diagnose the SI Section 8 receipt: full-prompt row total, human-missing rows, and anes_simp.csv
against anes_simp rebuilt from the ANES cumulative file with the authors' recodes (2_DATA script)."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw", "bisbee2024")
GROUPS = ["Democratic Party", "Republican Party", "Black Americans", "White Americans", "Asian Americans",
          "Gays and Lesbians", "Muslims", "Jews", "Liberals", "Conservatives", "Christians"]
long = pd.read_parquet(os.path.join(RAW, "derived", "rr1_long.parquet"))
full = long[long.framing == "full"]
print("full rows", len(full), "authors 2,493,589 (2,235,705 + 257,884)")
h = pd.read_csv(os.path.join(RAW, "PA_replication", "data", "raw", "anes_simp.csv"))
h["respID"] = h.respID.astype(str)
hl = h.melt(id_vars=["respID", "year"], value_vars=[f"therm_{g}" for g in GROUPS], var_name="group", value_name="y")
hl["group"] = hl.group.str.replace("therm_", "", regex=False)
m = full.merge(hl, on=["respID", "group"], how="left")
print("human NA rows (csv)", int(m.y.isna().sum()), "authors 257,884")

# rebuild from the dta as in the R script
cols = dict(VCF0006a="respID", VCF0004="year", VCF0105b="raceth", VCF0101="age", VCF0104="gender", VCF0803="ideo",
            VCF0302="PID", VCF0114="income", VCF0703="regis", VCF0140="education", VCF9259="interest",
            VCF0147="marst", VCF0218="Democratic Party", VCF0224="Republican Party", VCF0206="Black Americans",
            VCF0207="White Americans", VCF0227="Asian Americans", VCF0232="Gays and Lesbians", VCF9267="Muslims",
            VCF0205="Jews", VCF0211="Liberals", VCF0212="Conservatives", VCF9269="Christians")
a = pd.read_stata(os.path.join(RAW, "PA_replication", "data", "raw", "anes_timeseries_cdf_stata_20220916.dta"),
                  columns=list(cols), convert_categoricals=False)
a = a.rename(columns=cols)
a = a[a.year.isin([2016, 2020])].dropna()
print("dta rows 2016/2020 complete", len(a))
for g in GROUPS:
    x = a[g]
    a[g] = np.where((x > 97) | (x < 0), np.nan, np.where(x == 97, 100, x))
a["respID"] = a.respID.astype("int64").astype(str)
r = a.set_index("respID")
hh = h.set_index("respID")
common = hh.index.intersection(r.index)
print("csv respondents", len(hh), "in rebuilt", len(common))
for g in GROUPS:
    x, y = hh.loc[common, f"therm_{g}"], r.loc[common, g]
    diff = ~((x == y) | (x.isna() & y.isna()))
    print(f"{g:18s} csv NA {int(x.isna().sum()):5d} rebuilt NA {int(y.isna().sum()):5d} differ {int(diff.sum())}")
