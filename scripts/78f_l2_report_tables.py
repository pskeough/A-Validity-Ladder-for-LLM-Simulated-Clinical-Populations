"""Report tables for paper_brm/analysis_brm/L2.md, assembled from the 78a-78e receipts so every
number in the report sits in one CSV.

  l2_report_table.csv     one row per contrast x scope (pooled, four models): the published
                          verdict and ratio (analysis/level2_permodel_equivalence.csv, script 66),
                          and the rebuilt headline, December-only and without-GLM readings under
                          both estimands (ratio, Bonferroni interval, verdict, stop reading).
  l2_report_rank.csv      the ranking used for the summary. A reading is "determinate" when R3
                          names one region or two adjacent ones (not undetermined, not stopped).
                          Score = determinate pooled readings (0-2, both estimands) + determinate
                          per-model readings (0-8), ties broken by the pooled standardised ratio's
                          distance from 1 on the log scale. Contrasts are ranked by score.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(BASE, "analysis", "brm")
UNDET = {"undetermined", "reference too imprecise", "no population gap"}
SCOPES = ["pooled", "openai/gpt-4o-mini", "google/gemini-3-flash-preview",
          "deepseek/deepseek-chat-v3", "z-ai/glm-4.7"]


def main():
    v = pd.read_csv(os.path.join(OUTD, "l2_verdicts.csv"))
    pub = pd.read_csv(os.path.join(BASE, "analysis", "level2_permodel_equivalence.csv"))
    pub = pub[pub.scope.isin(SCOPES)].set_index(["contrast", "scope"])
    rows = []
    for (c, s), _ in pub.groupby(level=[0, 1]):
        r = dict(contrast=c, scope=s, published_verdict=pub.loc[(c, s)].verdict,
                 published_ratio=round(float(pub.loc[(c, s)].ratio), 4),
                 published_n_pairs=int(pub.loc[(c, s)].n_pairs))
        for an in ("headline", "dec_only", "without_glm"):
            for est in ("marginal", "standardised"):
                x = v[(v.analysis == an) & (v.estimand == est) & (v.contrast == c) & (v.scope == s)]
                if not len(x):
                    continue
                x = x.iloc[0]
                pre = f"{an}_{est}"
                r[f"{pre}_ratio"] = round(x.ratio, 4)
                r[f"{pre}_ci"] = ("unbounded" if x.ci_unbounded else
                                  f"{x.ci_lo:.2f} to {x.ci_hi:.2f}")
                r[f"{pre}_verdict"] = x.verdict
                if an == "headline":
                    r[f"{pre}_n_pairs"] = int(x.n_pairs)
                    r[f"{pre}_sim"] = round(x.simulated, 4)
                    r[f"{pre}_se_sim"] = round(x.se_sim, 4)
                    r[f"{pre}_pop"] = round(x.population, 4)
                    r[f"{pre}_se_pop"] = round(x.se_pop, 4)
                    r[f"{pre}_unstopped"] = x.r3_unstopped
                    r[f"{pre}_popstop"] = x.verdict_popstop
                    r[f"{pre}_unadjusted_90"] = x.verdict_unadjusted_90
        rows.append(r)
    t = pd.DataFrame(rows)
    t["scope"] = pd.Categorical(t.scope, SCOPES)
    t = t.sort_values(["contrast", "scope"])
    t.to_csv(os.path.join(OUTD, "l2_report_table.csv"), index=False)

    rk = []
    for c, g in t.groupby("contrast"):
        pooled = g[g.scope == "pooled"].iloc[0]
        per = g[g.scope != "pooled"]
        dp = sum(pooled[f"headline_{e}_verdict"] not in UNDET for e in ("marginal", "standardised"))
        dm = sum((per[f"headline_{e}_verdict"].map(lambda z: z not in UNDET)).sum()
                 for e in ("marginal", "standardised"))
        rk.append(dict(contrast=c, determinate_pooled=int(dp), determinate_per_model=int(dm),
                       score=int(dp + dm),
                       log_distance_std=round(abs(np.log(abs(pooled.headline_standardised_ratio))), 4),
                       pooled_marginal=pooled.headline_marginal_verdict,
                       pooled_standardised=pooled.headline_standardised_verdict,
                       published_pooled=pooled.published_verdict))
    rk = pd.DataFrame(rk).sort_values(["score", "log_distance_std"], ascending=False)
    rk.insert(0, "rank", range(1, len(rk) + 1))
    rk.to_csv(os.path.join(OUTD, "l2_report_rank.csv"), index=False)
    pd.set_option("display.width", 250)
    print(rk.to_string(index=False))
    print(t[["contrast", "scope", "published_verdict", "published_ratio",
             "headline_marginal_ratio", "headline_marginal_verdict",
             "headline_standardised_ratio", "headline_standardised_verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
