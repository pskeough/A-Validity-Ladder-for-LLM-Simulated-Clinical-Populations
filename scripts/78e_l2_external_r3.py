"""Level 2 of the two external demos (paper_brm/external) re-read under R3 with the same
multiplicity and stop rules as the PHQ-8 headline (78c).

Inputs are the published level-2 rows; nothing is regenerated and no API is called.
  OpinionQA (Meister et al. 2024): results/opinionqa/level2_contrasts.csv. g = model gap, p = Pew
    gap, both averaged over the same questions (the pairing unit), with cov(g, p) from the
    question-level pairing and df = questions - 1.
  Argyle et al. 2023 Study 3: results/argyle/level2_contrasts.csv. g = silicon gap, p = human
    gap on the same ANES respondents, respondent-bootstrap SEs and covariance, df infinite.

Test at boundary c (oriented so p > 0): t_c = (g - c p) / sqrt(se_g^2 + c^2 se_p^2 - 2 c cov),
one-sided at level a, df as above. Because g and p share one df the inversion is the closed-form
Fieller set. Boundaries -0.25, 0.25, 0.75, 1.25 (primary) and -0.25, 0.25, 0.80, 1.25.

Families (Bonferroni, a = .05 / m):
  OpinionQA   one model x method x steering cell; m = the contrasts it carries (8, 5 or 3).
  Argyle      one run (temperature); m = 61. Sensitivity: one outcome within a run (m = 5 to 7).
Preconditions as 78c: "no population gap" when p's family-level interval covers zero, then the
conditional reference stop (exact-g interval [g/p_hi, g/p_lo] spans three or more regions);
the population-only stop (k = t se_p / |p| > 1/4) is carried as verdict_popstop.

Gate: at a = .05 unadjusted with only the no-population-gap precondition, the engine reproduces
the R3_interval column of paper_brm/level2_rule/results/applied_all_rules.csv for every external
row ("flattened" read as "attenuated").

Emits analysis/brm/l2_external_r3.csv (every row x configuration), l2_external_r3_summary.csv
(verdict counts), l2_external_r3_receipts.csv.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(BASE, "analysis", "brm")
EXT = os.path.join(BASE, "paper_brm", "external", "results")
APPLIED = os.path.join(BASE, "paper_brm", "level2_rule", "results", "applied_all_rules.csv")
ALPHA = 0.05
LABELS = ["reversed", "missing", "attenuated", "kept", "steepened"]
BANDS = {"asym": np.array([-0.25, 0.25, 0.75, 1.25]), "sym": np.array([-0.25, 0.25, 0.80, 1.25])}


def tq(a, df):
    df = np.asarray(df, float)
    return stats.t.ppf(1 - a, np.where(np.isfinite(df), df, 1e9))


def r3cov(g, se_g, p, se_p, cov, df, a, bounds):
    s = -1.0 if p < 0 else 1.0
    g, p = g * s, p * s
    t = float(tq(a, df))
    above = below = 0
    for c in bounds:
        z = (g - c * p) / np.sqrt(se_g ** 2 + c ** 2 * se_p ** 2 - 2 * c * cov)
        above += z > t
        below += z < -t
    lo, hi = above, len(bounds) - below
    lab = LABELS[lo] if lo == hi else (f"{LABELS[lo]} or {LABELS[hi]}" if hi == lo + 1
                                       else "undetermined")
    A = p ** 2 - t ** 2 * se_p ** 2
    B = g * p - t ** 2 * cov
    C = g ** 2 - t ** 2 * se_g ** 2
    disc = B ** 2 - A * C
    if A > 0 and disc >= 0:
        ci_lo, ci_hi = (B - np.sqrt(disc)) / A, (B + np.sqrt(disc)) / A
    else:
        ci_lo = ci_hi = np.nan
    p_lo, p_hi = p - t * se_p, p + t * se_p
    k = t * se_p / p
    W = bounds[3] / bounds[2]
    kmax = (W - 1) / (W + 1)
    if p_lo > 0:
        e_lo, e_hi = sorted([g / p_hi, g / p_lo])
        n_reg = int(np.sum((bounds > e_lo) & (bounds < e_hi))) + 1
    else:
        n_reg = 99
    nopop = p_lo <= 0
    verdict = "no population gap" if nopop else ("reference too imprecise" if n_reg >= 3 else lab)
    popstop = "no population gap" if nopop else ("reference too imprecise" if k > kmax else lab)
    return dict(ratio=g / p, ci_lo=ci_lo, ci_hi=ci_hi, r3_unstopped=lab,
                r3_nopop_only="no population gap" if nopop else lab, k_rel_halfwidth=k,
                stop_conditional=bool(n_reg >= 3 and not nopop),
                stop_population=bool(k > kmax and not nopop), no_population_gap=bool(nopop),
                verdict=verdict, verdict_popstop=popstop)


def load():
    oq = pd.read_csv(os.path.join(EXT, "opinionqa", "level2_contrasts.csv"))
    oq = pd.DataFrame(dict(
        source="OpinionQA", reference="marginal on both sides",
        row=oq.contrast + " | " + oq.model + " | " + oq.method + " | " + oq.steering,
        family=oq.model + " | " + oq.method + " | " + oq.steering, subfamily="",
        contrast=oq.contrast, g=oq.model_gap, se_g=oq.se_model_gap, p=oq.human_gap,
        se_p=oq.se_human_gap, cov=oq.cov_model_human, df=oq.df.astype(float),
        published_verdict=oq.verdict))
    ar = pd.read_csv(os.path.join(EXT, "argyle", "level2_contrasts.csv"))
    ar = pd.DataFrame(dict(
        source="Argyle", reference="same respondents (identical composition)",
        row=ar.outcome + " | " + ar.contrast + " | " + ar.run, family=ar.run,
        subfamily=ar.run + " | " + ar.outcome, contrast=ar.contrast, g=ar.silicon_gap,
        se_g=ar.se_model_gap, p=ar.human_gap, se_p=ar.se_human_gap, cov=ar.cov_model_human,
        df=np.inf, published_verdict=ar.verdict))
    return pd.concat([oq, ar], ignore_index=True)


def short(v):
    m = {"reversed": "r", "missing": "m", "attenuated": "a", "kept": "k", "steepened": "s",
         "undetermined": "u", "no population gap": "N", "reference too imprecise": "X"}
    return "/".join(m[x] for x in v.split(" or "))


def main():
    d = load()
    d["m_family"] = d.groupby("family").row.transform("size")
    d["m_subfamily"] = d.groupby("subfamily").row.transform("size")

    # gate
    ap = pd.read_csv(APPLIED)
    ap = ap[ap.source != "paper (PHQ-8)"].set_index("row").R3_interval
    rc, fails = [], 0
    for r in d.itertuples():
        o = r3cov(r.g, r.se_g, r.p, r.se_p, r.cov, r.df, ALPHA, BANDS["asym"])
        want = ap.loc[r.row].replace("flattened", "attenuated")
        ok = o["r3_nopop_only"] == want
        fails += not ok
        if not ok:
            rc.append(dict(check="gate mismatch", item=r.row, got=o["r3_nopop_only"], want=want))
    rc.insert(0, dict(check="reproduces R3_interval of applied_all_rules.csv (90%, external rows)",
                      item="rows matched / rows", got=f"{len(d) - fails} / {len(d)}", want="all"))
    if fails:
        pd.DataFrame(rc).to_csv(os.path.join(OUTD, "l2_external_r3_receipts.csv"), index=False)
        raise SystemExit(f"GATE FAIL {fails}")

    configs = [("headline", "family", "asym"), ("band_symmetric", "family", "sym"),
               ("unadjusted_90", None, "asym"), ("argyle_within_outcome", "subfamily", "asym")]
    out = []
    for name, fam, band in configs:
        bnd = BANDS[band]
        for r in d.itertuples():
            if name == "argyle_within_outcome" and r.source != "Argyle":
                continue
            m = 1 if fam is None else (r.m_family if fam == "family" else r.m_subfamily)
            o = r3cov(r.g, r.se_g, r.p, r.se_p, r.cov, r.df, ALPHA / m, bnd)
            out.append(dict(config=name, source=r.source, reference=r.reference, row=r.row,
                            family=r.family if fam != "subfamily" else r.subfamily,
                            family_size=m, contrast=r.contrast, g=r.g, se_g=r.se_g, p=r.p,
                            se_p=r.se_p, cov=r.cov, df=r.df, published_verdict=r.published_verdict,
                            **o, code=short(o["verdict"])))
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(OUTD, "l2_external_r3.csv"), index=False)

    summ = (res.groupby(["config", "source", "verdict"]).size().rename("n").reset_index())
    tot = res.groupby(["config", "source"]).size().rename("rows").reset_index()
    summ = summ.merge(tot, on=["config", "source"])
    # by contrast, headline
    h = res[res.config == "headline"]
    byc = (h.groupby(["source", "contrast", "verdict"]).size().rename("n").reset_index())
    byc["config"] = "headline by contrast"
    med = (h.groupby(["source", "contrast"]).ratio.median().rename("median_ratio").reset_index())
    byc = byc.merge(med, on=["source", "contrast"])
    stops = (h.groupby("source")[["no_population_gap", "stop_conditional", "stop_population"]]
             .sum().reset_index())
    stops["config"] = "headline stop counts"
    byrun = (res[(res.source == "Argyle") & (res.config != "argyle_within_outcome")]
             .groupby(["config", "family", "verdict"]).size().rename("n").reset_index()
             .rename(columns={"family": "run"}))
    byrun["source"] = "Argyle by run"
    pd.concat([summ, byc, stops, byrun], ignore_index=True).to_csv(
        os.path.join(OUTD, "l2_external_r3_summary.csv"), index=False)
    # the published label on the same rows, for the change table
    chg = (h.assign(published=h.published_verdict.str.replace("flattened", "attenuated"))
           .groupby(["source", "published", "verdict"]).size().rename("n").reset_index())
    chg["check"] = "published verdict -> headline R3 verdict"
    rc += chg.rename(columns={"published": "item", "verdict": "got", "n": "want"}).assign(
        item=lambda x: x.source + " | " + x["item"]).drop(columns="source").to_dict("records")
    pc = (h.assign(published=h.published_verdict.str.replace("flattened", "attenuated"))
          .groupby(["source", "published"]).size().rename("want").reset_index())
    rc += pc.assign(check="published verdict count (all runs)",
                    item=pc.source + " | " + pc.published, got="").drop(
        columns=["source", "published"]).to_dict("records")
    pd.DataFrame(rc).to_csv(os.path.join(OUTD, "l2_external_r3_receipts.csv"), index=False)

    pd.set_option("display.width", 250)
    print(rc[0])
    print(summ.pivot_table(index=["source", "verdict"], columns="config", values="n",
                           fill_value=0).to_string())
    print(byc.pivot_table(index=["source", "contrast"], columns="verdict", values="n",
                          fill_value=0).join(med.set_index(["source", "contrast"])).to_string())
    print(stops.to_string(index=False))
    a = h[(h.source == "Argyle") & (h.family == "t0.7_main")]
    print(a[["row", "ratio", "ci_lo", "ci_hi", "verdict", "verdict_popstop", "published_verdict"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
