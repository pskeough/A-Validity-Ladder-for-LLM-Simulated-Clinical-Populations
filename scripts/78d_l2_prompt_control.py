"""Level 2 for the prompt-control arms (scripts 69-71) under R3 on the rebuilt reference.

No API is called. The generations are the ones 70 logged in analysis/prompt_control_raw.jsonl;
loading and deduplication follow 71 rule for rule (first write wins per model x arm x cohort x
draw; a model counts only if it carries all three arms; cohorts must be present in every arm).
The gate reproduces 71's level-2 gaps (analysis/prompt_control_results.csv, outcome level2_gap:
estimate, se and n for every arm x model x contrast) before anything is written.

Design of the control (analysis/prompt_control_design.json): 12 cohorts, race White or
Black x cisgender man or woman x income Low, Middle, High, relationship fixed at Single; clinical
framing only; 30 draws. Five contrasts. Every persona pair is observed once per arm, so there is
no framing pseudo-replication here. Per model: 6 pairs (sex, race) or 4 (income), SE sd/sqrt(n).
Pooled: equal model weights, SE stratified by model, Satterthwaite df.

References (78a): marginal (published) and standardised to THIS design (variant pc_design:
race in {White, Black}, sex, income, relationship Single). Families: the five contrasts of one
arm x scope, Bonferroni at .05/5. Boundaries -0.25, 0.25, 0.75, 1.25; stops as in 78c.

The "orig" arm's system prompt is not the December prompt: it asks for 20 PCL-5 items where the
December main.py asked for 4. The line-by-line comparison is written as a receipt.

Emits analysis/brm/l2_prompt_control_gaps.csv, l2_prompt_control_verdicts.csv,
l2_prompt_control_receipts.csv.
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(BASE, "analysis", "brm")
RAW = os.path.join(BASE, "analysis", "prompt_control_raw.jsonl")
PREREG = os.path.join(BASE, "analysis", "prompt_control_design.json")  # written before the run; not registered
DEC_MAIN = os.path.join(BASE, "generation", "main.py")  # the 28 Dec main.py; differs from the backup in file paths only
ALPHA = 0.05

spec = importlib.util.spec_from_file_location(
    "r3mod", os.path.join(os.path.dirname(__file__), "78c_l2_r3_verdicts.py"))
r3mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3mod)


def load(cutoff=None):
    rows = []
    for line in open(RAW, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("ok"):
            continue
        if cutoff is not None and r["ts"] > cutoff:
            continue
        try:
            phq = json.loads(r["raw"])["PHQ8"]
        except Exception:
            continue
        if not isinstance(phq, list) or len(phq) != 8 or not all(isinstance(v, int) for v in phq):
            continue
        rows.append(dict(model=r["model"].split("/")[-1], arm=r["arm"], profile_id=r["profile_id"],
                         draw=r["draw"], ts=r["ts"], total=int(np.clip(sum(phq), 0, 24))))
    df = pd.DataFrame(rows)
    key = ["model", "arm", "profile_id", "draw"]
    df = df.sort_values("ts").drop_duplicates(subset=key, keep="first")
    return df


def cells(pre, arms, cutoff=None):
    cohort = {c["profile_id"]: c for c in pre["cohorts"]}
    df = load(cutoff)
    have = df.groupby("model").arm.nunique()
    df = df[df.model.isin(have[have == len(arms)].index)]
    keep = []
    for m, g in df.groupby("model"):
        common = set.intersection(*[set(a.profile_id) for _, a in g.groupby("arm")])
        keep.append(g[g.profile_id.isin(common)])
    df = pd.concat(keep)
    for f in ("race", "gender", "ses_level"):
        df[f] = df.profile_id.map(lambda p: cohort[p][f])
    cell = (df.groupby(["arm", "model", "profile_id", "race", "gender", "ses_level"],
                       as_index=False).agg(y=("total", "mean"), n_draws=("total", "size")))
    return cell, sorted(cell.model.unique())


def gaps_from(cell, models, pre, arms):
    rows = []
    for sp in pre["contrasts"]:
        name, factor, hi, lo = sp["name"], sp["factor"], sp["high"], sp["low"]
        stratum = ["model"] + sp["stratum"]
        for arm in arms:
            s = cell[cell.arm == arm]
            a = s[s[factor] == hi].set_index(stratum).y
            b = s[s[factor] == lo].set_index(stratum).y
            common = a.index.intersection(b.index)
            d = (a.loc[common] - b.loc[common]).rename("diff").reset_index()
            per = {}
            for m in models:
                x = d[d.model == m]["diff"].to_numpy(float)
                mu, se, n = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), len(x)
                per[m] = (mu, se, n)
                rows.append(dict(arm=arm, contrast=name, scope=m, n_pairs=n, simulated=mu,
                                 se=se, df=n - 1))
            K = len(models)
            v = np.array([per[m][1] ** 2 for m in models]) / K ** 2
            dfs = np.array([per[m][2] - 1 for m in models], float)
            mu = float(np.mean([per[m][0] for m in models]))
            rows.append(dict(arm=arm, contrast=name, scope="pooled", n_pairs=len(d),
                             simulated=mu, se=float(np.sqrt(v.sum())),
                             df=float(v.sum() ** 2 / np.sum(v ** 2 / dfs)),
                             se_naive_71=float(d["diff"].std(ddof=1) / np.sqrt(len(d)))))
    return pd.DataFrame(rows)


def main():
    pre = json.load(open(PREREG, encoding="utf-8"))
    arms = [a["name"] for a in pre["arms"]]
    ref = pd.read_csv(os.path.join(OUTD, "l2_reference.csv"))
    pub = pd.read_csv(os.path.join(BASE, "analysis", "prompt_control_results.csv"))
    pub = pub[pub.outcome == "level2_gap"]
    rc, fails = [], 0

    # 71 wrote its results at 2026-09-09 18:55:41 +10:00; the raw log kept growing until 19:29
    # (a second GLM-4.7 pass that fills draws the first pass lost). The gate replays 71 on the log
    # as it stood when 71 ran; the analysis then uses the complete log under the same
    # first-write-wins rule, and the change is written as a receipt.
    cutoff = "2026-09-09T08:55:41.056451+00:00"
    cell_71, models_71 = cells(pre, arms, cutoff)
    g71 = gaps_from(cell_71, models_71, pre, arms)
    for got in g71.itertuples():
        p = pub[(pub.arm == got.arm) & (pub.model == got.scope) & (pub.group == got.contrast)].iloc[0]
        se_got = got.se_naive_71 if got.scope == "pooled" else got.se
        for lab, g_, w_ in (("estimate", got.simulated, p.estimate), ("se", se_got, p.se),
                            ("n", got.n_pairs, p.n)):
            ok = abs(g_ - w_) <= 6e-5
            fails += not ok
            rc.append(dict(check="reproduces 71 level2_gap on the log as of 71's run",
                           item=f"{got.arm} | {got.scope} | {got.contrast} | {lab}",
                           got=round(float(g_), 6), want=float(w_), ok=bool(ok)))
    if fails:
        pd.DataFrame(rc).to_csv(os.path.join(OUTD, "l2_prompt_control_receipts.csv"), index=False)
        raise SystemExit(f"GATE FAIL: {fails}")

    cell, models = cells(pre, arms)
    assert models == models_71
    gaps = gaps_from(cell, models, pre, arms)
    mm = cell.merge(cell_71, on=["arm", "model", "profile_id"], suffixes=("", "_71"))
    for r in mm[(mm.n_draws != mm.n_draws_71) | (abs(mm.y - mm.y_71) > 1e-9)].itertuples():
        rc.append(dict(check="cell changed by the log after 71 ran",
                       item=f"{r.arm} | {r.model} | {r.profile_id} | draws {r.n_draws_71} -> {r.n_draws}",
                       got=round(r.y, 4), want=round(r.y_71, 4), ok=True))
    rc.append(dict(check="cells with the full 30 draws, complete log", item="count / total",
                   got=f"{int((cell.n_draws == 30).sum())} / {len(cell)}", want=np.nan, ok=True))
    rc.append(dict(check="cells with the full 30 draws, log as of 71", item="count / total",
                   got=f"{int((cell_71.n_draws == 30).sum())} / {len(cell_71)}", want=np.nan, ok=True))
    gaps.to_csv(os.path.join(OUTD, "l2_prompt_control_gaps.csv"), index=False)

    fam = len(pre["contrasts"])
    out = []
    bounds = r3mod.BANDS["asym"]
    for g in gaps.itertuples():
        for est, var in (("marginal", "primary"), ("standardised", "pc_design")):
            rr = ref[(ref.contrast == g.contrast) & (ref.estimand == est) &
                     (ref.variant == var)].iloc[0]
            o = r3mod.r3(g.simulated, g.se, g.df, rr.estimate, rr.se_taylor, rr.df,
                         ALPHA / fam, bounds)
            u = r3mod.r3(g.simulated, g.se, g.df, rr.estimate, rr.se_taylor, rr.df, ALPHA, bounds)
            out.append(dict(arm=g.arm, contrast=g.contrast, scope=g.scope, estimand=est,
                            n_pairs=g.n_pairs, simulated=round(g.simulated, 4),
                            se=round(g.se, 4), df=round(g.df, 2),
                            population=round(rr.estimate, 4), se_pop=round(rr.se_taylor, 4),
                            ratio=round(o["ratio"], 4),
                            ci_lo=round(o["ci_lo"], 4), ci_hi=round(o["ci_hi"], 4),
                            ci_unbounded=o["ci_unbounded"], verdict=o["verdict"],
                            r3_unstopped=o["r3_unstopped"], verdict_popstop=o["verdict_popstop"],
                            verdict_unadjusted_90=u["verdict"],
                            code=r3mod.code(o["verdict"])))
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(OUTD, "l2_prompt_control_verdicts.csv"), index=False)

    # prompt comparison receipt
    src = open(DEC_MAIN, encoding="utf-8").read()
    i = src.index('return """ROLE:') + len('return """')
    dec_prompt = src[i:src.index('"""', i)]
    orig = pre["arms"][0]["system_prompt"]
    dl, ol = dec_prompt.splitlines(), orig.splitlines()
    rc.append(dict(check="orig arm vs December system prompt", item="identical",
                   got=int(dec_prompt == orig), want=np.nan, ok=True))
    for k, (x, y) in enumerate(zip(dl, ol)):
        if x != y:
            rc.append(dict(check="orig arm vs December system prompt, differing line",
                           item=f"line {k + 1}: December '{x.strip()}' | orig arm '{y.strip()}'",
                           got=k + 1, want=np.nan, ok=True))
    rc.append(dict(check="orig arm vs December system prompt", item="line counts December / orig",
                   got=f"{len(dl)} / {len(ol)}", want=np.nan, ok=True))
    for a in pre["arms"]:
        line = [x.strip() for x in a["system_prompt"].splitlines() if '"PCL5"' in x]
        rc.append(dict(check="PCL5 line in arm system prompt", item=a["name"],
                       got=line[0] if line else "", want=np.nan, ok=True))
    for e in pre["endpoints"]:
        rc.append(dict(check="prompt-control endpoint (prereg)", item=e["original"],
                       got=e["control"], want=f"substituted={e['substituted']}", ok=True))
    pd.DataFrame(rc).to_csv(os.path.join(OUTD, "l2_prompt_control_receipts.csv"), index=False)

    pd.set_option("display.width", 250)
    p = res[res.scope == "pooled"].copy()
    p["cell"] = p.ratio.round(2).astype(str) + " " + p.code
    print(p.pivot_table(index=["estimand", "contrast"], columns="arm", values="cell",
                        aggfunc="first").to_string())
    m = res[res.scope != "pooled"].copy()
    m["cell"] = m.ratio.round(2).astype(str) + " " + m.code
    print(m.pivot_table(index=["estimand", "contrast", "scope"], columns="arm", values="cell",
                        aggfunc="first").to_string())
    print(pd.DataFrame(rc[-6:]).to_string(index=False))
    print(f"gate: {sum(1 for r in rc if r['check'].startswith('reproduces'))} checks against 71 pass")


if __name__ == "__main__":
    main()
