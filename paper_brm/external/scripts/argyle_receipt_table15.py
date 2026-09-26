"""Receipt: rebuild Argyle et al.'s complete-case subset (Study3Analysis.R, anesgpt3_sub)
and check our outcome coding against their Appendix Table 15 means (N = 1,781)."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "raw" / "argyle2023"
OUT = Path(__file__).resolve().parents[1] / "results" / "argyle"
d = pd.read_csv(RAW / "anesgpt3_task3.csv")
na = lambda c: d[c].where(d[c] != -1)
cols = pd.DataFrame({
    "age_gpt3": na("age_gpt3"), "age_anes": d.V161267.where(d.V161267 > 0),
    "church_gpt3": na("church_goer_gpt3"), "church_anes": d.V161244.where(d.V161244 > 0),
    "discuss_gpt3": na("discuss_politics_gpt3"), "discuss_anes": d.V162174.where(d.V162174.isin([1, 2])),
    "race_gpt3": na("race_gpt3"), "race_anes": d.V161310x.where(d.V161310x.isin([1, 2, 3, 5])),
    "educ_gpt3": na("education_gpt3"), "educ_anes": d.V161270.where(d.V161270.between(1, 16)),
    "gender_gpt3": na("gender_gpt3"), "gender_anes": d.V161342.where(d.V161342.isin([1, 2])),
    "ideology_gpt3": na("ideology_gpt3"), "ideology_anes": d.V161126.where(d.V161126.between(1, 7)),
    "patriotism_gpt3": na("patriotism_gpt3"), "patriotism_anes": d.V162125x.where(d.V162125x > 0),
    "pid7_gpt3": na("pid7_gpt3"), "pid7_anes": d.V161158x.where(d.V161158x > 0),
    "interest_gpt3": na("political_interest_gpt3"), "interest_anes": d.V162256.where(d.V162256 > 0),
})
voted_g = na("voted_2016_gpt3")
vc_g = na("votechoice_2016_gpt3")
cols["vote_gpt3"] = np.where(voted_g == 0, "DNV", vc_g.astype("Int64").astype(str))
cols.loc[voted_g.isna() | ((voted_g == 1) & vc_g.isna()), "vote_gpt3"] = np.nan
voted_a = d.V162031x.where(d.V162031x.isin([0, 1]))
vc_a = d.V162062x.map({1: "C", 2: "T", 3: "O", 4: "O", 5: "O"})
cols["vote_anes"] = np.where(voted_a == 0, "DNV", vc_a)
cols.loc[voted_a.isna() | ((voted_a == 1) & vc_a.isna()), "vote_anes"] = np.nan
cc = cols.dropna()
rows = [
    ("N", len(cc), 1781),
    ("pid7_gpt3", cc.pid7_gpt3.mean(), 4.4), ("pid7_anes", cc.pid7_anes.mean(), 3.9),
    ("ideology_gpt3", cc.ideology_gpt3.mean(), 4.0), ("ideology_anes", cc.ideology_anes.mean(), 4.1),
    ("patriotism_gpt3", cc.patriotism_gpt3.mean(), 1.5), ("patriotism_anes", cc.patriotism_anes.mean(), 2.0),
    ("interest_gpt3", cc.interest_gpt3.mean(), 1.7), ("interest_anes", cc.interest_anes.mean(), 2.0),
    ("church_gpt3 (1=yes)", (cc.church_gpt3 == 1).mean(), 0.6), ("church_anes", (cc.church_anes == 1).mean(), 0.6),
    ("voted_gpt3", (cc.vote_gpt3 != "DNV").mean(), 0.8), ("voted_anes", (cc.vote_anes != "DNV").mean(), 0.9),
    ("trump_gpt3 | voted", (cc.vote_gpt3[cc.vote_gpt3 != "DNV"] == "2").mean(), 0.2),
    ("other_gpt3 | voted", (cc.vote_gpt3[cc.vote_gpt3 != "DNV"] == "42").mean(), 0.5),
    ("trump_anes | voted", (cc.vote_anes[cc.vote_anes != "DNV"] == "T").mean(), 0.4),
]
r = pd.DataFrame(rows, columns=["statistic", "ours", "argyle_table15"])
r["ours_rounded"] = r.ours.round(1)
r["match"] = (r.ours_rounded - r.argyle_table15).abs() < 0.051
r.to_csv(OUT / "receipt_table15.csv", index=False)
print(r.to_string())
