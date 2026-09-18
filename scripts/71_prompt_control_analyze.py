"""
Prompt-template control: analysis.

Question: the level-2 gaps and the level-3 elevation were both measured under one system prompt
that assigns a "Clinical Simulation Engine" role and instructs the model to reference NSDUH/CDC
population trends. Do they survive deleting that instruction (arm B), and do they survive removing
the role assignment and the population reference together (arm C)?

Estimators match the main analysis:
  - PHQ-8 total is eight integer items, summed and clipped to 0-24
  - severity category on the standard bands: None 0-4, Mild 5-9, Moderate 10-14,
    Mod-severe 15-19, Severe 20-24
  - a design cell is one model by one cohort within one arm, its mean taken over the draws, as in
    18_cell_level_inference.py and 67_level3_equivalence.py
  - level-3 residual = cell mean minus the survey-weighted NHANES anchor, with the anchor's
    design-based standard error folded in as a root sum of squares, as in 67
  - level-2 gap = paired difference over strata that share everything but the contrast factor, as
    in 44_race_contrast_test.py and 64_level2_sex_ses_contrasts.py, tested against zero and against
    the population gap
  - flip probability = share of ordered within-cell draw pairs landing in different categories,
    as in 56_decoding_control_analyze.py

Loading, deduplication and the all-arms/paired-cohort restrictions follow decoding_control_io.py
rule for rule, and are restated here rather than imported because that module is bound to the
decoding control's raw log and pre-registration:

  1. DUPLICATE DRAWS. For each (model, arm, profile_id, draw) keep the generation with the earliest
     timestamp. First write wins. The rule is applied before any outcome is computed and cannot see
     the totals, so it cannot select on the thing being measured.
  2. OVER-LENGTH CELLS. After deduplication a cell holds at most one generation per draw index, so
     it is capped at n_draws by construction. The cap is asserted rather than assumed.
  3. A model contributes only cohorts it measured under ALL THREE arms, which keeps every
     between-arm contrast within-cell.

Runs on the smoke output without crashing. With one cohort and two draws most standard errors are
undefined and print as NaN, which is the correct answer and not a failure.

Writes analysis/prompt_control_results.csv and analysis/prompt_control_arm_contrasts.csv.
"""
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(OUT, "prompt_control_raw.jsonl")
PREREG = os.path.join(OUT, "prompt_control_preregistration.json")

BANDS = [(0, 4, "None"), (5, 9, "Mild"), (10, 14, "Moderate"),
         (15, 19, "Mod-severe"), (20, 24, "Severe")]
BASELINE_ARM = "orig"


def band(t):
    for lo, hi, name in BANDS:
        if lo <= t <= hi:
            return name
    return None


def mean_se(x):
    """Mean and standard error of a sample; se is undefined below two observations."""
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, 0
    if len(x) < 2:
        return float(x.mean()), np.nan, len(x)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), len(x)


def t_p(estimate, se, df):
    """Two-sided t p-value, NaN wherever it is not defined."""
    if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0 or df < 1:
        return np.nan
    return float(stats.t.sf(abs(estimate / se), df) * 2)


def ci95(estimate, se, df):
    if not np.isfinite(estimate) or not np.isfinite(se) or df < 1:
        return np.nan, np.nan
    t = float(stats.t.ppf(0.975, df))
    return estimate - t * se, estimate + t * se


def load(verbose=True, raw_path=None):
    """Return (df, report). df has one row per kept generation."""
    path = raw_path or RAW
    rows, n_lines, n_failed, n_unparsed = [], 0, 0, 0
    for line in open(path, encoding="utf-8"):
        n_lines += 1
        r = json.loads(line)
        if not r.get("ok"):
            n_failed += 1
            continue
        try:
            phq = json.loads(r["raw"])["PHQ8"]
        except Exception:
            n_unparsed += 1
            continue
        if not isinstance(phq, list) or len(phq) != 8 or not all(isinstance(v, int) for v in phq):
            n_unparsed += 1
            continue
        rows.append(dict(model=r["model"].split("/")[-1], arm=r["arm"],
                         profile_id=r["profile_id"], draw=r["draw"], ts=r["ts"],
                         total=int(np.clip(sum(phq), 0, 24)), items=tuple(phq),
                         provider=r.get("provider"),
                         latency_s=r.get("latency_s"), cost_usd=r.get("cost_usd") or 0.0,
                         completion_tokens=r.get("completion_tokens")))

    df = pd.DataFrame(rows)
    n_parsed = len(df)
    key = ["model", "arm", "profile_id", "draw"]
    if n_parsed:
        df = (df.sort_values("ts", ascending=True)
                .drop_duplicates(subset=key, keep="first")
                .sort_values(key).reset_index(drop=True))
        df["cat"] = df.total.map(band)
        n_draws = json.load(open(PREREG, encoding="utf-8"))["n_draws"]
        sizes = df.groupby(key[:3]).size()
        assert sizes.max() <= n_draws, "cell exceeds n_draws after dedup: %s" % (sizes.idxmax(),)
    n_dupes = n_parsed - len(df)

    report = dict(lines=n_lines, failed=n_failed, unparsed=n_unparsed,
                  parsed=n_parsed, duplicates_dropped=n_dupes, kept=len(df),
                  spend_usd=round(float(df.cost_usd.sum()), 4) if len(df) else 0.0)
    if verbose:
        print("raw lines %d | api failures %d | unparsed %d | duplicate draws dropped %d | kept %d"
              % (n_lines, n_failed, n_unparsed, n_dupes, len(df)))
    return df, report


def all_arms_only(df, arms, verbose=True):
    """Restrict to models that carry every arm. A model missing an arm answers nothing."""
    if not len(df):
        return df
    have = df.groupby("model").arm.nunique()
    dropped = sorted(have[have < len(arms)].index)
    if dropped and verbose:
        print("excluded, not all %d arms collected: %s" % (len(arms), ", ".join(dropped)))
    return df[df.model.isin(sorted(have[have == len(arms)].index))].copy()


def balanced_cells(df, arms, verbose=True):
    """Restrict to cohorts a model measured in EVERY arm, so between-arm contrasts are within-cell."""
    if not len(df):
        return df
    keep = []
    for m, g in df.groupby("model"):
        by_arm = dict(list(g.groupby("arm")))
        if len(by_arm) < len(arms):
            continue
        common = set.intersection(*[set(a.profile_id) for a in by_arm.values()])
        dropped = set(g.profile_id) - common
        if dropped and verbose:
            print("  %s: %d cohort(s) missing from at least one arm, dropped from paired tests: %s"
                  % (m, len(dropped), ", ".join(sorted(dropped))))
        keep.append(g[g.profile_id.isin(common)])
    return pd.concat(keep, ignore_index=True) if keep else df.iloc[:0]


# ----------------------------------------------------------------------------- design and anchors
pre = json.load(open(PREREG, encoding="utf-8"))
ARMS = [a["name"] for a in pre["arms"]]
COHORT = {c["profile_id"]: c for c in pre["cohorts"]}

gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
gtm = {r.group: float(r.w_mean) for r in gt.itertuples()}
dse = pd.read_csv(os.path.join(OUT, "anchor_design_se.csv")).set_index("group")
ANCHOR = {g["group"]: (gtm[g["gt_row"]], float(dse.loc[g["se_row"], "se_design"]))
          for g in pre["level3_groups"]}

df, report = load()
if not len(df):
    raise SystemExit("no parseable generations in %s; run 70_prompt_control_run.py first" % RAW)
df = all_arms_only(df, ARMS)
df = balanced_cells(df, ARMS)
if not len(df):
    raise SystemExit("no model carries all %d arms yet; nothing to analyse" % len(ARMS))

for f in ("race", "gender", "ses_level"):
    df[f] = df.profile_id.map(lambda p: COHORT[p][f])

print("analysing %d generations over %d models, %d arms and %d cohorts"
      % (len(df), df.model.nunique(), df.arm.nunique(), df.profile_id.nunique()))

# design cell = one model x one cohort within one arm, averaged over its draws
cell = (df.groupby(["arm", "model", "profile_id", "race", "gender", "ses_level"], as_index=False)
          .agg(cell_mean=("total", "mean"), n_draws=("total", "size")))

rows = []


def emit(arm, model, outcome, group, n, estimate, se, df_, p=None,
         population=np.nan, ratio=np.nan, sign=""):
    lo, hi = ci95(estimate, se, df_)
    rows.append(dict(arm=arm, model=model, outcome=outcome, group=group, n=n,
                     estimate=round(estimate, 4) if np.isfinite(estimate) else np.nan,
                     se=round(se, 4) if np.isfinite(se) else np.nan,
                     ci_lo=round(lo, 4) if np.isfinite(lo) else np.nan,
                     ci_hi=round(hi, 4) if np.isfinite(hi) else np.nan,
                     p=p if p is not None else np.nan,
                     population=round(population, 4) if np.isfinite(population) else np.nan,
                     ratio=round(ratio, 4) if np.isfinite(ratio) else np.nan,
                     sign=sign))


# ------------------------------------------------------------------------------- 1. cohort means
for (arm, model, pid), g in df.groupby(["arm", "model", "profile_id"]):
    mu, se, n = mean_se(g.total)
    emit(arm, model, "cohort_mean", pid, n, mu, se, n - 1)

# ------------------------------------------------------------- 2. level-3 residuals on the anchors
for g in pre["level3_groups"]:
    name, axis, level = g["group"], g["axis"], g["level"]
    anchor, se_anchor = ANCHOR[name]
    sub = cell[cell[axis] == level]
    for arm in sorted(sub.arm.unique()):
        s = sub[sub.arm == arm]
        for model in sorted(s.model.unique()) + ["pooled"]:
            v = s.cell_mean if model == "pooled" else s[s.model == model].cell_mean
            mu, se_cell, n = mean_se(v)
            resid = mu - anchor
            # the anchor is a survey estimate, so its design-based error enters the residual
            se = float(np.hypot(se_cell, se_anchor)) if np.isfinite(se_cell) else np.nan
            emit(arm, model, "level3_residual", name, n, resid, se, n - 1,
                 p=t_p(resid, se, n - 1), population=anchor,
                 sign="+" if np.isfinite(resid) and resid > 0 else
                      ("-" if np.isfinite(resid) else ""))

# ---------------------------------------------------------------------- 3. level-2 paired contrasts
for spec in pre["contrasts"]:
    name, factor, hi, lo = spec["name"], spec["factor"], spec["high"], spec["low"]
    stratum = ["model"] + spec["stratum"]
    pop = gtm[spec["anchor_high"]] - gtm[spec["anchor_low"]]
    se_pop = float(np.hypot(float(dse.loc[{"Men": "Cisgender men", "Women": "Cisgender women"}
                                          .get(spec["anchor_high"], spec["anchor_high"]), "se_design"]),
                            float(dse.loc[{"Men": "Cisgender men", "Women": "Cisgender women"}
                                          .get(spec["anchor_low"], spec["anchor_low"]), "se_design"])))
    for arm in sorted(cell.arm.unique()):
        s = cell[cell.arm == arm]
        a = s[s[factor] == hi].set_index(stratum).cell_mean
        b = s[s[factor] == lo].set_index(stratum).cell_mean
        if a.index.has_duplicates or b.index.has_duplicates:
            raise SystemExit("%s: stratum key is not unique in arm %s" % (name, arm))
        common = a.index.intersection(b.index)
        d = (a.loc[common] - b.loc[common]).reset_index().rename(columns={"cell_mean": "diff"})
        per_model = {k: float(v["diff"].mean()) for k, v in d.groupby("model")} if len(d) else {}
        for model in sorted(per_model) + ["pooled"]:
            v = d["diff"] if model == "pooled" else d[d.model == model]["diff"]
            mu, se, n = mean_se(v)
            emit(arm, model, "level2_gap", name, n, mu, se, n - 1,
                 p=t_p(mu, se, n - 1), population=pop,
                 ratio=mu / pop if np.isfinite(mu) and pop else np.nan,
                 sign="matches_population" if np.isfinite(mu) and np.sign(mu) == np.sign(pop)
                      else ("opposes_population" if np.isfinite(mu) else ""))
            # the reversal question: is the simulated gap distinguishable from the population gap,
            # with the anchor's own design error folded into the contrast standard error
            se_gap = float(np.hypot(se, se_pop)) if np.isfinite(se) else np.nan
            emit(arm, model, "level2_gap_vs_population", name, n, mu - pop, se_gap, n - 1,
                 p=t_p(mu - pop, se_gap, n - 1), population=pop,
                 ratio=mu / pop if np.isfinite(mu) and pop else np.nan)
        nm = sum(1 for v in per_model.values() if np.sign(v) == np.sign(pop))
        emit(arm, "pooled", "level2_models_with_population_sign", name,
             len(per_model), float(nm), np.nan, 0, population=pop)

# ----------------------------------------------------------- 4. within-cell category flip probability
for (arm, model), g in df.groupby(["arm", "model"]):
    flips = same = cross = disc = cells = 0
    for _, c in g.groupby("profile_id"):
        t = c.total.tolist()
        k = c.cat.tolist()
        if len(t) < 2:
            continue
        cells += 1
        for i, j in combinations(range(len(t)), 2):
            if k[i] != k[j]:
                flips += 1
                if (t[i] < 10) != (t[j] < 10):     # straddles the treatment threshold
                    cross += 1
                disc += 1
            else:
                same += 1
    pairs = flips + same
    emit(arm, model, "flip_probability", "all cohorts", pairs,
         100.0 * flips / pairs if pairs else np.nan, np.nan, 0)
    emit(arm, model, "cross_of_discordant_pct", "all cohorts", disc,
         100.0 * cross / disc if disc else np.nan, np.nan, 0)

# --------------------------------------------------------------------------- 5. between-arm contrasts
wide = cell.pivot_table(index=["model", "profile_id"], columns="arm", values="cell_mean")
arm_rows = []
for arm in ARMS:
    if arm == BASELINE_ARM or arm not in wide.columns or BASELINE_ARM not in wide.columns:
        continue
    d = (wide[arm] - wide[BASELINE_ARM]).dropna().reset_index().rename(columns={0: "diff"})
    d.columns = ["model", "profile_id", "diff"]
    for model in sorted(d.model.unique()) + ["pooled"]:
        v = d["diff"] if model == "pooled" else d[d.model == model]["diff"]
        mu, se, n = mean_se(v)
        lo, hi = ci95(mu, se, n - 1)
        arm_rows.append(dict(contrast="%s minus %s" % (arm, BASELINE_ARM), model=model,
                             n_cohorts=n,
                             estimate=round(mu, 4) if np.isfinite(mu) else np.nan,
                             se=round(se, 4) if np.isfinite(se) else np.nan,
                             ci_lo=round(lo, 4) if np.isfinite(lo) else np.nan,
                             ci_hi=round(hi, 4) if np.isfinite(hi) else np.nan,
                             p=t_p(mu, se, n - 1),
                             baseline_mean=round(float(
                                 (wide[BASELINE_ARM] if model == "pooled"
                                  else wide.loc[model, BASELINE_ARM]).mean()), 4)))
        emit(arm, model, "arm_contrast_vs_%s" % BASELINE_ARM, "cohort means", n, mu, se, n - 1,
             p=t_p(mu, se, n - 1))

res = pd.DataFrame(rows)
arm_contrasts = pd.DataFrame(arm_rows)
os.makedirs(OUT, exist_ok=True)
res.to_csv(os.path.join(OUT, "prompt_control_results.csv"), index=False)
arm_contrasts.to_csv(os.path.join(OUT, "prompt_control_arm_contrasts.csv"), index=False)

pd.set_option("display.width", 200)
print()
print("ARM MEANS over design cells")
print(cell.groupby(["arm", "model"]).cell_mean.mean().round(3).to_string())
print()
print("BETWEEN-ARM CONTRASTS on cohort means, paired within model")
if len(arm_contrasts):
    print(arm_contrasts.to_string(index=False))
else:
    print("  none: the baseline arm '%s' is not present alongside another arm" % BASELINE_ARM)
print()
print("LEVEL-3 RESIDUALS, pooled over models")
l3 = res[(res.outcome == "level3_residual") & (res.model == "pooled")]
print(l3[["arm", "group", "n", "estimate", "ci_lo", "ci_hi", "population"]].to_string(index=False)
      if len(l3) else "  none")
print()
print("LEVEL-2 GAPS, pooled over models")
l2 = res[(res.outcome == "level2_gap") & (res.model == "pooled")]
print(l2[["arm", "group", "n", "estimate", "ci_lo", "ci_hi", "p", "population", "ratio", "sign"]]
      .to_string(index=False) if len(l2) else "  none")
print()
print("FLIP PROBABILITY by arm")
fl = res[res.outcome == "flip_probability"]
print(fl[["arm", "model", "n", "estimate"]].to_string(index=False) if len(fl) else "  none")

print()
print("written -> %s" % os.path.join(OUT, "prompt_control_results.csv"))
print("written -> %s" % os.path.join(OUT, "prompt_control_arm_contrasts.csv"))
print("spend represented in this analysis: $%.4f" % report["spend_usd"])
