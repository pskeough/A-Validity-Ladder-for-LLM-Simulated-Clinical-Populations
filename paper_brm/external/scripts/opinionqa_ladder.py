"""Validity ladder levels 2 and 3 on OpinionQA steered distributions released by
Meister, Guestrin & Hashimoto (2024), "Benchmarking Distributional Alignment of LLMs"
(arXiv 2411.05403; github.com/nicolemeister/benchmarking-distributional-alignment,
commit 36869b5). 100 contested Pew ATP questions, 5 models, 3 elicitation methods,
persona (task1) and few-shot (task3_easy_hard) steering.

Scalar per group and question: the expected ordinal position of the answer
distribution, using OpinionQA's own option_ordinal field (Santurkar et al. 2023),
rescaled to 0-1 over the question's ordinal range. Options without an ordinal
("Refused", "Not sure") are dropped and the rest renormalised.

Outputs go to ../results/opinionqa/.
"""
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "meister2024"
OUT = ROOT / "results" / "opinionqa"
OUT.mkdir(parents=True, exist_ok=True)
WAVE = "Pew_American_Trends_Panel_disagreement_100"
HUMAN = RAW / "opinions_qa" / "data" / "human_resp"

CONTRASTS = [  # (groupvar, group_a, group_b): gap = a - b
    ("POLPARTY", "Republican", "Democrat"),
    ("SEX", "Female", "Male"),
    ("RACE", "Black", "White"),
    ("RACE", "Hispanic", "White"),
    ("RACE", "Asian", "White"),
    ("EDUCATION", "Less than high school", "College graduate or some postgrad"),
    ("INCOME", "Less than $30,000", "$100,000 or more"),
    ("CREGION", "South", "Northeast"),
]
METHODS = ["express_distribution", "sequence", "model_logprobs"]
TASKS = {"task1": "persona", "task3_easy_hard": "few-shot"}
LETTERS = [chr(ord("A") + i) for i in range(26)]

# ---------------------------------------------------------------- reference data
info = pd.read_csv(HUMAN / "Pew_American_Trends_Panel_disagreement_500" / "info.csv")
info = info.drop_duplicates("key").set_index("key")


def ordinal_map(q):
    refs = ast.literal_eval(info.loc[q, "references"])
    ords = ast.literal_eval(info.loc[q, "option_ordinal"])
    m = {refs[i]: float(ords[i]) for i in range(len(ords))}
    lo, hi = min(m.values()), max(m.values())
    return {k: (v - lo) / (hi - lo) for k, v in m.items()}, refs


counts = {}  # groupvar -> json with raw counts (only POLPARTY, SEX, RACE released)
for g in ["POLPARTY", "SEX", "RACE"]:
    counts[g] = json.load(open(HUMAN / WAVE / f"{g}_data.json", encoding="utf-8"))

# human proportions for every group, letter-keyed, from the task0 file (same for all models)
t0 = json.load(open(RAW / "results" / "opinionqa" / "express_distribution" / "gpt-4" / "task0" / WAVE / "NONE" / "Democrat.json", encoding="utf-8"))
# two questions are nominal (all ordinals equal) and carry no ordinal mean; excluded
NOMINAL = [q for q in t0 if len(set(ast.literal_eval(info.loc[q, "option_ordinal"]))) < 2]
QIDS = [q for q in t0 if q not in NOMINAL]

audit = []


def option_names(q, gv, grp, n_letters):
    """Letter order used in Meister's prompt for this group: the keys of the group's
    count dict. Where counts are released we read them; otherwise we infer the
    sorted full option set and accept it only if the letter count matches."""
    if gv in counts:
        return list(counts[gv][q][grp].keys())
    _, refs = ordinal_map(q)
    names = sorted(refs)
    if len(names) != n_letters:
        return None
    return names


def to_named(dist_letters, names):
    return {names[LETTERS.index(k)]: v for k, v in dist_letters.items() if LETTERS.index(k) < len(names)}


def ord_mean(named, q):
    om, _ = ordinal_map(q)
    p = np.array([named.get(k, 0.0) for k in om])
    x = np.array(list(om.values()))
    if p.sum() <= 0:
        return np.nan, np.nan
    p = p / p.sum()
    return float((p * x).sum()), float((p * (x - (p * x).sum()) ** 2).sum())


def modal(named, q):
    om, _ = ordinal_map(q)
    sub = {k: named.get(k, 0.0) for k in om}
    return max(sub, key=sub.get) if sum(sub.values()) > 0 else None


def human_group(q, gv, grp):
    key = f"expected_results_{gv}_{grp}"
    if key not in t0[q]:
        return None
    letters = t0[q][key]
    names = option_names(q, gv, grp, len(letters))
    if names is None:
        audit.append((q, gv, grp, "human letters could not be mapped"))
        return None
    named = to_named(letters, names)
    m, v = ord_mean(named, q)
    n = None
    if gv in counts:
        c = counts[gv][q][grp]
        n = sum(cnt for k, cnt in c.items() if k in ordinal_map(q)[0])
        # receipt: proportions in Meister's file equal counts/n over all options
        tot = sum(c.values())
        for k, cnt in c.items():
            assert abs(named[k] - cnt / tot) < 1e-9, (q, gv, grp, k)
    return dict(named=named, mean=m, var=v, n=n, names=names, modal=modal(named, q))


def model_group(method, model, task, q, gv, grp, names):
    f = RAW / "results" / "opinionqa" / method / model / task / WAVE / gv / f"{grp}.json"
    if not f.exists():
        return "nofile"
    d = json.load(open(f, encoding="utf-8"))
    if q not in d:
        return None
    rec = d[q]
    # receipt: the human target stored with the model output matches the task0 target
    exp = rec.get("expected_results")
    ref = t0[q].get(f"expected_results_{gv}_{grp}")
    if exp is not None and ref is not None:
        assert all(abs(exp[k] - ref[k]) < 1e-9 for k in ref), (method, model, task, q, gv, grp)
    avg = rec.get("avg_actual_results") or {}
    avg = {k: v for k, v in avg.items() if v is not None and k in LETTERS}
    if not avg or sum(avg.values()) <= 0:
        return None
    named = to_named(avg, names)
    m, _ = ord_mean(named, q)
    return dict(named=named, mean=m, modal=modal(named, q))


MODELS = sorted(os.listdir(RAW / "results" / "opinionqa" / "express_distribution"))

# ---------------------------------------------------------------- build long table
rows = []
groups_needed = sorted({(gv, g) for gv, a, b in CONTRASTS for g in (a, b)})
H = {}
for q in QIDS:
    for gv, grp in groups_needed:
        h = human_group(q, gv, grp)
        if h is not None:
            H[(q, gv, grp)] = h
for method in METHODS:
    for model in MODELS:
        for task in TASKS:
            for q in QIDS:
                for gv, grp in groups_needed:
                    h = H.get((q, gv, grp))
                    if h is None:
                        continue
                    r = model_group(method, model, task, q, gv, grp, h["names"])
                    if r == "nofile":
                        continue
                    rows.append(dict(method=method, model=model, task=task, q=q, groupvar=gv, group=grp,
                                     h_mean=h["mean"], h_var=h["var"], h_n=h["n"], h_modal=h["modal"],
                                     m_mean=None if r is None else r["mean"],
                                     m_modal=None if r is None else r["modal"],
                                     parse_fail=r is None))
long = pd.DataFrame(rows)
long.to_csv(OUT / "opinionqa_long.csv", index=False)

# ---------------------------------------------------------------- receipt: reproduce Meister TV
ev = pd.read_csv(RAW / "results" / "eval_disagreement_bootstrapping.csv")
chk = []
for method in METHODS:
    for model in MODELS:
        for gv, grp in [("POLPARTY", "Democrat"), ("SEX", "Female")]:
            f = RAW / "results" / "opinionqa" / method / model / "task1" / WAVE / gv / f"{grp}.json"
            if not f.exists():
                continue
            d = json.load(open(f, encoding="utf-8"))
            tvs = []
            for q, rec in d.items():
                exp = rec["expected_results"]
                act = {k: v for k, v in (rec.get("avg_actual_results") or {}).items() if v is not None}
                if not act:
                    continue
                s = sum(act.values())
                act = {k: v / s for k, v in act.items()} if s > 0 else act
                keys = set(exp) | set(act)
                tvs.append(0.5 * sum(abs(exp.get(k, 0) - act.get(k, 0)) for k in keys))
            sel = ev[(ev["Output Type"] == method) & (ev.Model == model) & (ev["Task Type"] == "task1") &
                     (ev["Demographic Group/Avg"] == gv) & (ev["Demographic/Avg"] == grp) & (ev.Dataset == "opinionqa")]
            chk.append(dict(method=method, model=model, group=grp, ours_mean_tv=np.mean(tvs),
                            meister_tv=sel.TV.iloc[0] if len(sel) else np.nan, n_q=len(tvs)))
pd.DataFrame(chk).to_csv(OUT / "receipt_tv_reproduction.csv", index=False)

# ---------------------------------------------------------------- level 2
def level2_verdict(m_gap, h_gap, diff, bound):
    """Ladder rule, section 3 of the manuscript. m_gap, diff: arrays over questions of
    the oriented model gap and (model - human) gap; bound = mean human gap."""
    n = len(diff)
    tcrit95 = stats.t.ppf(0.975, n - 1)
    tcrit90 = stats.t.ppf(0.95, n - 1)
    mg, se_mg = m_gap.mean(), m_gap.std(ddof=1) / np.sqrt(n)
    dd, se_dd = diff.mean(), diff.std(ddof=1) / np.sqrt(n)
    hg = h_gap.mean()
    separable = abs(dd) > tcrit95 * se_dd
    same_sign = np.sign(mg) == np.sign(hg)
    # tightest bounds at which each TOST passes, as fractions of the population gap
    dmin_zero = max(abs(mg - tcrit90 * se_mg), abs(mg + tcrit90 * se_mg)) / hg
    dmin_pop = max(abs(dd - tcrit90 * se_dd), abs(dd + tcrit90 * se_dd)) / hg

    def rule(b):
        eq_zero = (mg - tcrit90 * se_mg > -b) and (mg + tcrit90 * se_mg < b)
        eq_pop = (dd - tcrit90 * se_dd > -b) and (dd + tcrit90 * se_dd < b)
        if eq_zero and separable:
            return "missing", eq_zero, eq_pop
        if same_sign and not separable and eq_pop:
            return "kept", eq_zero, eq_pop
        if same_sign and separable:
            return ("steepened" if abs(mg) > abs(hg) else "flattened"), eq_zero, eq_pop
        if (not same_sign) and separable:
            return "reversed", eq_zero, eq_pop
        return "undetermined", eq_zero, eq_pop

    v, eq_zero, eq_pop = rule(bound)
    v_half, _, _ = rule(0.5 * bound)
    p_sep = 2 * stats.t.sf(abs(dd) / se_dd, n - 1)
    return dict(n_questions=n, human_gap=hg, model_gap=mg, model_gap_ci95_lo=mg - tcrit95 * se_mg,
                model_gap_ci95_hi=mg + tcrit95 * se_mg, ratio=mg / hg, diff=dd, p_separable=p_sep,
                separable=separable, equiv_zero=eq_zero, equiv_pop=eq_pop,
                dmin_zero_frac=dmin_zero, dmin_pop_frac=dmin_pop,
                share_q_reversed=float((m_gap < 0).mean()), verdict=v, verdict_half_bound=v_half,
                # sufficient statistics for the ratio-interval rule (paper_brm/level2_rule)
                se_model_gap=se_mg, se_human_gap=h_gap.std(ddof=1) / np.sqrt(n),
                cov_model_human=np.cov(m_gap, h_gap, ddof=1)[0, 1] / n, df=n - 1)


L2 = []
ok = long[~long.parse_fail]
for (method, model, task), sub in ok.groupby(["method", "model", "task"]):
    idx = sub.set_index(["q", "groupvar", "group"])
    for gv, a, b in CONTRASTS:
        qs = [q for q in QIDS if (q, gv, a) in idx.index and (q, gv, b) in idx.index]
        if len(qs) < 20:
            continue
        ha = np.array([idx.loc[(q, gv, a), "h_mean"] for q in qs])
        hb = np.array([idx.loc[(q, gv, b), "h_mean"] for q in qs])
        ma = np.array([idx.loc[(q, gv, a), "m_mean"] for q in qs])
        mb = np.array([idx.loc[(q, gv, b), "m_mean"] for q in qs])
        keep = ~np.isnan(ha) & ~np.isnan(hb) & ~np.isnan(ma) & ~np.isnan(mb)
        ha, hb, ma, mb = ha[keep], hb[keep], ma[keep], mb[keep]
        dh = ha - hb
        s = np.sign(dh)
        s[s == 0] = 1
        h_gap = s * dh               # human gap, oriented positive on every question
        m_gap = s * (ma - mb)        # model gap in the same orientation
        r = level2_verdict(m_gap, h_gap, m_gap - h_gap, h_gap.mean())
        L2.append(dict(method=method, model=model, steering=TASKS[task], contrast=f"{a} - {b}", **r))
L2 = pd.DataFrame(L2)
# Benjamini-Hochberg over the separation tests within each model x method x steering family
L2["q_separable"] = np.nan
for _, g in L2.groupby(["method", "model", "steering"]):
    L2.loc[g.index, "q_separable"] = stats.false_discovery_control(g.p_separable.values)
L2.to_csv(OUT / "level2_contrasts.csv", index=False)

# sensitivity: orient only on questions where the human gap is separable from zero
L2s = []
for (method, model, task), sub in ok.groupby(["method", "model", "task"]):
    idx = sub.set_index(["q", "groupvar", "group"])
    for gv, a, b in CONTRASTS:
        if gv not in counts:
            continue
        qs = [q for q in QIDS if (q, gv, a) in idx.index and (q, gv, b) in idx.index]
        keep = []
        for q in qs:
            ra, rb = idx.loc[(q, gv, a)], idx.loc[(q, gv, b)]
            se = np.sqrt(ra.h_var / ra.h_n + rb.h_var / rb.h_n)
            if abs(ra.h_mean - rb.h_mean) > 1.96 * se and not (np.isnan(ra.m_mean) or np.isnan(rb.m_mean)):
                keep.append(q)
        if len(keep) < 20:
            continue
        dh = np.array([idx.loc[(q, gv, a), "h_mean"] - idx.loc[(q, gv, b), "h_mean"] for q in keep])
        dm = np.array([idx.loc[(q, gv, a), "m_mean"] - idx.loc[(q, gv, b), "m_mean"] for q in keep])
        s = np.sign(dh)
        r = level2_verdict(s * dm, s * dh, s * dm - s * dh, (s * dh).mean())
        L2s.append(dict(method=method, model=model, steering=TASKS[task], contrast=f"{a} - {b}", **r))
pd.DataFrame(L2s).to_csv(OUT / "level2_contrasts_sensitivity_separable_questions.csv", index=False)

# ---------------------------------------------------------------- level 3
# Per group: share of questions inside (a) the strict tolerance, the human sample's own
# 95% interval for the ordinal mean, and (b) the use tolerance, same plurality answer.
# Reference: the share a fresh human sample of the same size would pass, by parametric
# bootstrap from the human counts (only where counts are released).
rng = np.random.default_rng(20260924)


def human_resample_pass(q, gv, grp, B=400):
    c = counts[gv][q][grp]
    om, _ = ordinal_map(q)
    names = [k for k in c if k in om]
    cnt = np.array([c[k] for k in names], dtype=float)
    n = int(cnt.sum())
    if n < 2:
        return np.nan, np.nan
    p = cnt / n
    x = np.array([om[k] for k in names])
    mu = (p * x).sum()
    se = np.sqrt((p * (x - mu) ** 2).sum() / n)
    top = names[int(np.argmax(p))]
    draws = rng.multinomial(n, p, size=B) / n
    mus = draws @ x
    strict = np.mean(np.abs(mus - mu) <= 1.96 * se)
    plural = np.mean([names[int(np.argmax(dw))] == top for dw in draws])
    return strict, plural


REF = {}
for q in QIDS:
    for gv in counts:
        for grp in {g for v, a, b in CONTRASTS if v == gv for g in (a, b)}:
            if (q, gv, grp) in H:
                REF[(q, gv, grp)] = human_resample_pass(q, gv, grp)

L3 = []
for (method, model, task, gv, grp), sub in long.groupby(["method", "model", "task", "groupvar", "group"]):
    sub = sub[~sub.parse_fail]
    if len(sub) < 20:
        continue
    res = sub.m_mean - sub.h_mean
    row = dict(method=method, model=model, steering=TASKS[task], group=f"{gv}:{grp}", n_questions=len(sub),
               mean_abs_residual=res.abs().mean(), median_abs_residual=res.abs().median(),
               share_plurality_match=(sub.m_modal == sub.h_modal).mean())
    if gv in counts:
        se = np.sqrt(sub.h_var / sub.h_n)
        row["share_within_strict"] = (res.abs() <= 1.96 * se).mean()
        refs = [REF[(q, gv, grp)] for q in sub.q]
        row["human_resample_strict"] = np.mean([r[0] for r in refs])
        row["human_resample_plurality"] = np.mean([r[1] for r in refs])
    row["passes_use_tolerance_90"] = row["share_plurality_match"] >= 0.90
    L3.append(row)
L3 = pd.DataFrame(L3)
L3.to_csv(OUT / "level3_groups.csv", index=False)

# parse failures
pf = long.groupby(["method", "model", "task"]).parse_fail.agg(["sum", "count"]).reset_index()
pf.to_csv(OUT / "parse_failures.csv", index=False)
pd.DataFrame(audit, columns=["q", "groupvar", "group", "issue"]).drop_duplicates().to_csv(OUT / "mapping_audit.csv", index=False)
print("long rows", len(long), "L2 rows", len(L2), "L3 rows", len(L3))
