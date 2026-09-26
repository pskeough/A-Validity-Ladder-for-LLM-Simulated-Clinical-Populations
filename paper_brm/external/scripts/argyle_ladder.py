"""Validity ladder levels 2 and 3 on Argyle et al. (2023) Study 3 silicon samples,
"Out of One, Many", Political Analysis 31(3). Replication data: Harvard Dataverse
doi:10.7910/DVN/JPV20K, files Study3_Data/anesgpt3_task3*.csv.

Design, from the released generation code: for each ANES 2016 respondent and each of
12 interview questions, GPT-3 (davinci) is shown the respondent's own answers to the
other 11 questions in interview form and asked the omitted one. Every silicon answer
therefore has a human twin: the same respondent's real answer. Main file at
temperature 0.7 (README); robustness files at 0.001 and 1.0.

Level 2: group gap in the silicon answers against the group gap in the same
respondents' real answers, verdict by the manuscript's rule (scripts/66 in the
PsychBench repo), inference by respondent bootstrap.
Level 3: silicon group mean against the real group mean, strict tolerance (the real
mean's own 95% sampling half-width) and a use tolerance stated per outcome.

Outputs go to ../results/argyle/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "argyle2023"
OUT = ROOT / "results" / "argyle"
OUT.mkdir(parents=True, exist_ok=True)
B = 2000
Z95, Z90 = 1.959964, 1.644854


def clean(path):
    d = pd.read_csv(path)
    g = lambda c: d[c].where(d[c] != -1)  # GPT-3 unparsed answers are coded -1
    o = pd.DataFrame(index=d.index)
    # grouping attributes: the respondent's own ANES answers (they sit in every backstory)
    o["gender"] = d.V161342.map({1: "Male", 2: "Female"})
    o["race"] = d.V161310x.map({1: "White", 2: "Black", 3: "Asian", 5: "Hispanic"})
    ed = d.V161270
    o["educ"] = np.select([(ed > 0) & (ed <= 9), (ed >= 10) & (ed <= 12), ed == 13, (ed >= 14) & (ed <= 16)],
                          ["HS or less", "Some college", "Bachelor's", "Graduate"], None)
    age = d.V161267.where(d.V161267 > 0)
    o["agegrp"] = np.select([age.between(18, 29), age >= 65], ["18-29", "65+"], None)
    pid = d.V161158x.where(d.V161158x > 0)
    o["party"] = np.select([pid <= 3, pid >= 5], ["Democrat", "Republican"], None)
    # outcomes, human (_h) and silicon (_s), coded as in Argyle's Study3Analysis.R / Table 15
    o["pid7_h"], o["pid7_s"] = pid, g("pid7_gpt3")
    ideo = d.V161126.where(d.V161126.between(1, 7))
    o["ideology_h"], o["ideology_s"] = ideo, g("ideology_gpt3")
    o["patriotism_h"], o["patriotism_s"] = d.V162125x.where(d.V162125x > 0), g("patriotism_gpt3")
    o["interest_h"], o["interest_s"] = d.V162256.where(d.V162256 > 0), g("political_interest_gpt3")
    o["church_h"] = d.V161244.map({1: 1.0, 2: 0.0})
    o["church_s"] = g("church_goer_gpt3").map({1: 1.0, 2: 0.0})
    o["discuss_h"] = d.V162174.map({1: 1.0, 2: 0.0})
    o["discuss_s"] = g("discuss_politics_gpt3").map({1: 1.0, 2: 0.0})
    o["voted_h"] = d.V162031x.map({0: 0.0, 1: 1.0})
    o["voted_s"] = g("voted_2016_gpt3").map({0: 0.0, 1: 1.0})
    vc_h = d.V162062x.map({1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0})
    o["trump_h"] = vc_h.where(o.voted_h == 1)
    vc_s = g("votechoice_2016_gpt3").map({1: 0.0, 2: 1.0, 42: 0.0})
    o["trump_s"] = vc_s.where(o.voted_s == 1)
    # two-party share: Trump among Trump + Clinton voters in each source
    o["trump2p_h"] = d.V162062x.map({1: 0.0, 2: 1.0}).where(o.voted_h == 1)
    o["trump2p_s"] = g("votechoice_2016_gpt3").map({1: 0.0, 2: 1.0}).where(o.voted_s == 1)
    return o


OUTCOMES = {  # name: (label, scale points or None for a proportion, use tolerance)
    "trump": ("Trump share among voters", None, 0.05),
    "trump2p": ("Trump two-party share", None, 0.05),
    "voted": ("Turnout 2016", None, 0.05),
    "church": ("Attends church", None, 0.05),
    "discuss": ("Ever discusses politics", None, 0.05),
    "pid7": ("Party ID (1-7)", 7, 1.0),
    "ideology": ("Ideology (1-7)", 7, 1.0),
    "patriotism": ("Patriotism (1-7, 1 = flag feels extremely good)", 7, 1.0),
    "interest": ("Political interest (1-4, 1 = very)", 4, 1.0),
}
CONTRASTS = [
    ("gender", "Female", "Male"),
    ("race", "Black", "White"),
    ("race", "Hispanic", "White"),
    ("race", "Asian", "White"),
    ("educ", "HS or less", "Graduate"),
    ("agegrp", "18-29", "65+"),
    ("party", "Republican", "Democrat"),
]


def verdict(mg, se_mg, dd, se_dd, hg, frac=1.0):
    """Manuscript rule (PsychBench scripts/66_level2_permodel_equivalence.py):
    missing: equivalent to zero at the bound and separable from the population value;
    kept: sign matches, equivalent to the population value, not separable from it;
    steepened / flattened: separable, sign kept, ratio above / below 1;
    reversed: separable, sign differs; otherwise undetermined. Bound = |population gap|."""
    b = frac * abs(hg)
    separable = abs(dd) > Z95 * se_dd
    eq_zero = (mg - Z90 * se_mg > -b) and (mg + Z90 * se_mg < b)
    eq_pop = (dd - Z90 * se_dd > -b) and (dd + Z90 * se_dd < b)
    same = np.sign(mg) == np.sign(hg)
    if eq_zero and separable:
        return "missing"
    if same and not separable and eq_pop:
        return "kept"
    if same and separable:
        return "steepened" if abs(mg) > abs(hg) else "flattened"
    if (not same) and separable:
        return "reversed"
    return "undetermined"


def run(tag, path, rng):
    o = clean(path)
    L2, L3 = [], []
    for out, (label, pts, tol) in OUTCOMES.items():
        h, s = f"{out}_h", f"{out}_s"
        for gv, a, b in CONTRASTS:
            if out == "pid7" and gv == "party":
                continue  # party is built from pid7
            sub = o[(o[gv].isin([a, b])) & o[h].notna() & o[s].notna()]
            A, Bb = sub[sub[gv] == a], sub[sub[gv] == b]
            if len(A) < 20 or len(Bb) < 20:
                continue
            hg = A[h].mean() - Bb[h].mean()
            mg = A[s].mean() - Bb[s].mean()
            ah, as_, bh_, bs_ = A[h].values, A[s].values, Bb[h].values, Bb[s].values
            ra = rng.integers(0, len(A), size=(B, len(A)))   # resample respondents, keeping
            rb = rng.integers(0, len(Bb), size=(B, len(Bb)))  # each human-silicon pair together
            bh = ah[ra].mean(1) - bh_[rb].mean(1)
            bm = as_[ra].mean(1) - bs_[rb].mean(1)
            bd = bm - bh
            se_mg, se_dd = bm.std(ddof=1), bd.std(ddof=1)
            dd = mg - hg
            L2.append(dict(run=tag, outcome=label, contrast=f"{a} - {b}", n_a=len(A), n_b=len(Bb),
                           human_gap=hg, human_gap_se=bh.std(ddof=1),
                           human_gap_nonzero=abs(hg) > Z95 * bh.std(ddof=1), silicon_gap=mg, silicon_gap_ci95_lo=mg - Z95 * se_mg,
                           silicon_gap_ci95_hi=mg + Z95 * se_mg, ratio=mg / hg if hg else np.nan,
                           diff=dd, p_separable=2 * stats.norm.sf(abs(dd) / se_dd),
                           dmin_zero_frac=max(abs(mg - Z90 * se_mg), abs(mg + Z90 * se_mg)) / abs(hg),
                           dmin_pop_frac=max(abs(dd - Z90 * se_dd), abs(dd + Z90 * se_dd)) / abs(hg),
                           verdict=verdict(mg, se_mg, dd, se_dd, hg),
                           verdict_half_bound=verdict(mg, se_mg, dd, se_dd, hg, 0.5),
                           # sufficient statistics for the ratio-interval rule (paper_brm/level2_rule)
                           se_model_gap=se_mg, se_human_gap=bh.std(ddof=1),
                           cov_model_human=np.cov(bm, bh, ddof=1)[0, 1],
                           ratio_boot_ci90_lo=np.quantile(bm / bh, 0.05) if (bh != 0).all() else np.nan,
                           ratio_boot_ci90_hi=np.quantile(bm / bh, 0.95) if (bh != 0).all() else np.nan))
        # level 3 per group, all groups that appear in a contrast plus the whole sample
        groups = [("all", "All respondents")] + sorted({(gv, g) for gv, x, y in CONTRASTS for g in (x, y)})
        for gv, grp in groups:
            sub = o[o[h].notna() & o[s].notna()] if gv == "all" else o[(o[gv] == grp) & o[h].notna() & o[s].notna()]
            if len(sub) < 20:
                continue
            res = sub[s] - sub[h]
            r = res.mean()
            se_r = res.std(ddof=1) / np.sqrt(len(sub))   # paired: same respondents
            se_h = sub[h].std(ddof=1) / np.sqrt(len(sub))
            strict = Z95 * se_h
            lo, hi = r - Z90 * se_r, r + Z90 * se_r
            L3.append(dict(run=tag, outcome=label, group=f"{gv}:{grp}", n=len(sub),
                           human_mean=sub[h].mean(), silicon_mean=sub[s].mean(), residual=r,
                           residual_ci90_lo=lo, residual_ci90_hi=hi,
                           strict_tol=strict, passes_strict=(lo > -strict) and (hi < strict),
                           use_tol=tol, passes_use=(lo > -tol) and (hi < tol),
                           individual_agreement=(sub[s] == sub[h]).mean()))
    return pd.DataFrame(L2), pd.DataFrame(L3), o


rng = np.random.default_rng(20260924)
runs = {"t0.7_main": "anesgpt3_task3.csv", "t0.001": "anesgpt3_task3_temp001.csv", "t1.0": "anesgpt3_task3_temp10.csv"}
all2, all3, frames = [], [], {}
for tag, f in runs.items():
    l2, l3, o = run(tag, RAW / f, rng)
    all2.append(l2)
    all3.append(l3)
    frames[tag] = o
L2 = pd.concat(all2, ignore_index=True)
L3 = pd.concat(all3, ignore_index=True)
L2["q_separable"] = np.nan
for _, g in L2.groupby(["run", "outcome"]):
    L2.loc[g.index, "q_separable"] = stats.false_discovery_control(g.p_separable.values)
L2.to_csv(OUT / "level2_contrasts.csv", index=False)
L3.to_csv(OUT / "level3_groups.csv", index=False)

# Gate diagnostic (not the gate): the same respondents' silicon answers across the three
# temperature runs. Two runs at different temperatures bound regeneration stability from
# above; the gate itself needs repeat draws at one setting, which were not released.
G = []
for out in OUTCOMES:
    s = f"{out}_s"
    for x, y in [("t0.7_main", "t1.0"), ("t0.7_main", "t0.001"), ("t0.001", "t1.0")]:
        a, b = frames[x][s], frames[y][s]
        m = a.notna() & b.notna()
        row = dict(outcome=OUTCOMES[out][0], runs=f"{x} vs {y}", n=int(m.sum()),
                   identical=(a[m] == b[m]).mean())
        if OUTCOMES[out][1]:
            row["within_one_point"] = ((a[m] - b[m]).abs() <= 1).mean()
        G.append(row)
pd.DataFrame(G).to_csv(OUT / "gate_diagnostic_temperature_runs.csv", index=False)

# receipt: Argyle's Appendix Table 15 means (complete cases on their 22 columns)
o = frames["t0.7_main"]
print(L2[L2.run == "t0.7_main"].verdict.value_counts())
print("rows", len(L2), len(L3))
