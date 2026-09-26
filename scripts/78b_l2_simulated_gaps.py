"""Level 2 simulated gaps with standard errors clustered on the persona pair.

Pairing is script 66's, unchanged: cisgender frame, one value per design cell (the mean over its
draws), and a stratum that holds every attribute but the contrasted one fixed, so the two cells of
a pair differ in one attribute alone. The gate below reruns that pairing on data/model_outputs_v2.csv
and must reproduce analysis/level2_permodel_equivalence.csv (simulated gap and 66's standard error,
pooled, per model and per condition) before anything is written.

What changes from 66:
  - Corpus: data/model_outputs_v3.csv (43 December GPT-4o-mini rows restored, see script 76), with
    the one row whose PHQ-8 items are not all in 0..3 dropped (phq8_valid == False).
  - Unit: a persona pair is the same two personas within one model; 66 counted it twice, once per
    framing. Here each persona pair counts once. Per framing, a model's gap is the mean of its
    persona-pair differences with SE sd / sqrt(n). Framings combined, each pair's two differences
    are averaged first (the cluster mean), and the SE is sd(cluster means) / sqrt(pairs).
  - Pooled rows: the estimand is the average over this fixed four-model panel (equal model
    weights), so the pooled SE is stratified by model, sqrt(sum_m se_m^2) / K, with Satterthwaite
    df. The unstratified cluster SE (persona pairs across all models as one sample, which counts
    between-model disagreement as noise) and 66's naive SE are carried as sensitivities.
  - Race set: the sex and income contrasts drop the Multiracial pairs in the primary analysis
    (no defensible NHANES anchor; see 78a). race_set = "five" keeps them, as 66 did.

Emits, in analysis/brm/:
  l2_sim_gaps.csv      corpus x panel x race_set x contrast x scope x framing
  l2_sim_pairs.csv     every persona-pair difference (framing-specific), primary corpus
  l2_sim_receipts.csv  gate against 66, design balance, clustering inflation, framing correlation
Corpora: "all" (v3 minus the invalid row) and "dec_only" (clinical rows with row_source
dec28_main or dec28_restored, all narrative rows). Panels: "four models", "without GLM-4.7".
Closed form, nothing stochastic.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(BASE, "analysis", "brm")
CIS = ["Cisgender Man", "Cisgender Woman"]
DROP = "z-ai/glm-4.7"
MODELS = ["openai/gpt-4o-mini", "google/gemini-3-flash-preview", "deepseek/deepseek-chat-v3",
          "z-ai/glm-4.7"]
RACE_ST = ["model", "condition", "gender", "ses", "relationship"]
SES_ST = ["model", "condition", "race", "gender", "relationship"]
SEX_ST = ["model", "condition", "race", "ses", "relationship"]
SPECS = [
    ("Black minus White", "race", "Black", "White", RACE_ST),
    ("Hispanic minus White", "race", "Hispanic", "White", RACE_ST),
    ("Asian minus White", "race", "Asian", "White", RACE_ST),
    ("Women minus Men", "gender", "Cisgender Woman", "Cisgender Man", SEX_ST),
    ("Low minus High SES", "ses", "Low", "High", SES_ST),
    ("Middle minus High SES", "ses", "Middle", "High", SES_ST),
    ("Low minus Middle SES", "ses", "Low", "Middle", SES_ST),
]
EXPECTED_STRATA = {"race": 12, "four": {"gender": 24, "ses": 16}, "five": {"gender": 30, "ses": 20}}


def cells(m):
    m = m[m.gender.isin(CIS)].copy()
    m["phq8"] = m["phq8_total_clipped"]
    m["ses"] = m["ses_normalized"]
    m["condition"] = m["prompt_condition"]
    return (m.groupby(["model", "profile_id", "condition", "race", "gender", "ses",
                       "relationship"], as_index=False)
             .agg(y=("phq8", "mean"), n_draws=("phq8", "size")))


def paired(cell, col, hi, lo, stratum):
    a = cell[cell[col] == hi].set_index(stratum).y
    b = cell[cell[col] == lo].set_index(stratum).y
    assert not a.index.has_duplicates and not b.index.has_duplicates
    common = a.index.intersection(b.index)
    return (a.loc[common] - b.loc[common]).to_frame("diff").reset_index()


def mse(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else np.nan, np.nan, n
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)), n


def estimate(d, stratum, models):
    """Rows for every scope x framing for one contrast's pair table d."""
    key = [c for c in stratum if c != "condition"]            # persona pair, model included
    rows = []
    for framing in ("clinical", "narrative", "combined"):
        if framing == "combined":
            u = d.groupby(key, as_index=False)["diff"].mean()     # cluster mean over framings
            nfr = d.groupby(key).size()
            n_obs = len(d)
        else:
            u = d[d.condition == framing][key + ["diff"]]
            nfr = None
            n_obs = len(u)
        per = {}
        for mdl in models:
            x = u[u.model == mdl]["diff"]
            mu, se, n = mse(x)
            per[mdl] = (mu, se, n)
            rows.append(dict(scope=mdl, framing=framing, n_pairs=int(n),
                             n_obs=int((d.model == mdl).sum()) if framing == "combined" else int(n),
                             simulated=mu, se=se, df=n - 1, se_unstratified=se,
                             df_unstratified=n - 1, se_naive_66=np.nan, df_naive_66=np.nan))
        K = len(models)
        mus = np.array([per[m][0] for m in models])
        v = np.array([per[m][1] ** 2 for m in models]) / K ** 2
        dfs = np.array([per[m][2] - 1 for m in models], float)
        mu = float(mus.mean())
        se = float(np.sqrt(v.sum()))
        df = float(v.sum() ** 2 / np.sum(v ** 2 / dfs))
        mu_u, se_u, n_u = mse(u["diff"])
        # 66's naive SE: every framing-specific pair a separate unit
        xs = d["diff"] if framing == "combined" else d[d.condition == framing]["diff"]
        _, se_n, n_n = mse(xs)
        rows.append(dict(scope="pooled", framing=framing, n_pairs=int(len(u)), n_obs=int(n_obs),
                         simulated=mu, se=se, df=df, se_unstratified=se_u,
                         df_unstratified=n_u - 1, se_naive_66=se_n, df_naive_66=n_n - 1,
                         mean_unweighted=mu_u))
    return rows


def run(cell, corpus, panel, race_set, models):
    out = []
    pairs = []
    for name, col, hi, lo, st in SPECS:
        c = cell[cell.model.isin(models)]
        if col != "race" and race_set == "four":
            c = c[c.race != "Multiracial"]
        if col == "race" and race_set == "five":
            continue                                  # race contrasts have no race stratum
        d = paired(c, col, hi, lo, st)
        for r in estimate(d, st, models):
            out.append(dict(corpus=corpus, panel=panel,
                            race_set="n/a" if col == "race" else race_set, contrast=name, **r))
        d = d.assign(contrast=name, corpus=corpus, panel=panel, race_set=race_set)
        pairs.append(d)
    return out, pairs


def main():
    os.makedirs(OUT, exist_ok=True)
    rc = []

    # ------------------------------------------------------------------ gate: 66 on v2 exactly
    v2 = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"), low_memory=False)
    c2 = cells(v2)
    ref = pd.read_csv(os.path.join(BASE, "analysis", "level2_permodel_equivalence.csv"))
    fails = 0
    for name, col, hi, lo, st in SPECS:
        d = paired(c2, col, hi, lo, st)
        scopes = [("pooled", d)] + [(k, g) for k, g in d.groupby("model")] + \
                 [(k, g) for k, g in d.groupby("condition")]
        for scope, g in scopes:
            mu, se, n = mse(g["diff"])
            r = ref[(ref.contrast == name) & (ref.scope == scope)].iloc[0]
            for lab, got, want in (("simulated", mu, r.simulated), ("se", se, r.se),
                                   ("n_pairs", n, r.n_pairs)):
                ok = abs(got - want) <= 6e-5
                fails += not ok
                rc.append(dict(check="v2 pairing reproduces 66", item=f"{name} | {scope} | {lab}",
                               got=round(float(got), 6), want=float(want), ok=bool(ok)))
    if fails:
        pd.DataFrame(rc).to_csv(os.path.join(OUT, "l2_sim_receipts.csv"), index=False)
        raise SystemExit(f"GATE FAIL: {fails} checks against 66")
    print(f"gate: {len(rc)} checks against level2_permodel_equivalence.csv, all pass")

    # ---------------------------------------------------------------------------- v3 corpora
    v3 = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v3.csv"), low_memory=False)
    assert len(v3) == 28800
    n_invalid = int((~v3.phq8_valid.astype(bool)).sum())
    v3 = v3[v3.phq8_valid.astype(bool)].copy()
    items = [f"phq8_{i}" for i in range(1, 9)]
    assert (v3[items].sum(axis=1) == v3.phq8_total_clipped).all()
    rc.append(dict(check="corpus", item="v3 rows dropped, phq8_valid False", got=n_invalid,
                   want=1, ok=n_invalid == 1))
    dec = v3[(v3.prompt_condition == "narrative") |
             v3.row_source.isin(["dec28_main", "dec28_restored"])]
    corpora = {"all": cells(v3), "dec_only": cells(dec)}
    for k, c in corpora.items():
        cc = c[c.gender.isin(CIS)]
        rc.append(dict(check="corpus", item=f"{k}: cisgender design cells", got=len(cc),
                       want=4 * 2 * 60, ok=True))
        rc.append(dict(check="corpus", item=f"{k}: cisgender clinical draws",
                       got=int(cc[cc.condition == "clinical"].n_draws.sum()), want=np.nan, ok=True))

    rows, pairs = [], []
    for corpus, cell in corpora.items():
        for panel, models in (("four models", MODELS),
                              ("without GLM-4.7", [m for m in MODELS if m != DROP])):
            for race_set in ("four", "five"):
                r, p = run(cell, corpus, panel, race_set, models)
                rows += r
                pairs += p
    res = pd.DataFrame(rows)
    num = ["simulated", "se", "df", "se_unstratified", "se_naive_66", "mean_unweighted"]
    res[num] = res[num].astype(float).round(6)
    res.to_csv(os.path.join(OUT, "l2_sim_gaps.csv"), index=False)
    pp = pd.concat(pairs, ignore_index=True)
    pp[(pp.corpus == "all") & (pp.panel == "four models")].to_csv(
        os.path.join(OUT, "l2_sim_pairs.csv"), index=False)

    # -------------------------------------------------------------------------------- receipts
    prim = pp[(pp.corpus == "all") & (pp.panel == "four models")]
    for name, col, hi, lo, st in SPECS:
        for rs in (["four", "five"] if col != "race" else ["four"]):
            d = prim[(prim.contrast == name) & (prim.race_set == rs)]
            for (mdl, cond), g in d.groupby(["model", "condition"]):
                want = 12 if col == "race" else EXPECTED_STRATA[rs][col]
                ok = len(g) == want
                fails += not ok
                rc.append(dict(check="design balance: strata per model x framing",
                               item=f"{name} | {rs} | {mdl} | {cond}", got=len(g), want=want,
                               ok=bool(ok)))
            key = [c for c in st if c != "condition"]
            w = d.pivot_table(index=key, columns="condition", values="diff")
            rc.append(dict(check="clinical-narrative correlation of pair differences, pooled",
                           item=f"{name} | {rs}", got=round(float(w.corr().iloc[0, 1]), 4),
                           want=np.nan, ok=True))
            wr = [g.pivot_table(index=[k for k in key if k != "model"], columns="condition",
                                values="diff").corr().iloc[0, 1] for _, g in d.groupby("model")]
            rc.append(dict(check="clinical-narrative correlation, mean within model",
                           item=f"{name} | {rs}", got=round(float(np.mean(wr)), 4), want=np.nan,
                           ok=True))
    base = res[(res.corpus == "all") & (res.panel == "four models") & (res.scope == "pooled")
               & (res.framing == "combined")]
    for r in base.itertuples():
        rc.append(dict(check="SE ratio, unstratified cluster / 66 naive",
                       item=f"{r.contrast} | {r.race_set}",
                       got=round(r.se_unstratified / r.se_naive_66, 4), want=np.nan, ok=True))
        rc.append(dict(check="SE ratio, model-stratified cluster / 66 naive",
                       item=f"{r.contrast} | {r.race_set}", got=round(r.se / r.se_naive_66, 4),
                       want=np.nan, ok=True))
    pd.DataFrame(rc).to_csv(os.path.join(OUT, "l2_sim_receipts.csv"), index=False)
    if fails:
        raise SystemExit(f"FAIL: {fails} design-balance checks")

    pd.set_option("display.width", 250)
    show = res[(res.corpus == "all") & (res.panel == "four models") & (res.race_set != "five")]
    print(show[["contrast", "scope", "framing", "n_pairs", "simulated", "se", "df",
                "se_unstratified", "se_naive_66"]].to_string(index=False))
    print(pd.DataFrame(rc[-40:]).to_string(index=False))
    print("written l2_sim_gaps.csv, l2_sim_pairs.csv, l2_sim_receipts.csv")


if __name__ == "__main__":
    main()
