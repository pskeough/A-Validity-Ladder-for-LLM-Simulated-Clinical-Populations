"""
Decoding control: diagnostics.

56 reported that temperature 0 cuts the within-cell severity-category flip rate from 28.5% to 4.9%.
That number is the OBSERVABLE flip rate. It does not by itself distinguish two very different worlds:

  (W1) Temperature 0 made the instrument more reliable. The model has a stable reading of each
       vignette and sampling noise was obscuring it. Removing the noise reveals the reading.

  (W2) Temperature 0 made the sampler deterministic. The model's uncertainty is unchanged; greedy
       decoding just stops displaying it and returns one arbitrary point from the same distribution.
       Nothing about the measurement improved. We only stopped looking.

These have opposite implications for the manuscript, so they get separated here before anything is
written. The discriminating tests:

  A  Determinism.        Under W2, temp0 cells collapse to a single repeated total. Under W1 they
                         retain some spread that is merely narrower.
  B  Representativeness. Under W1 the temp0 value sits near the centre of the default distribution
                         for the same cell. Under W2 its position is arbitrary, so the percentiles
                         spread out and land off 50 as often as on it.
  C  Location shift.     Under W2 greedy decoding is a mode-seeking rule, not a mean-seeking one, so
                         it should bias toward the modal item response and shift the totals.
  D  Category agreement. Does temp0 land on the same severity category the default arm most often
                         gives, and does it fall on the same side of PHQ-8 = 10?
  E  Cohort ordering.    The manuscript's claims are about between-cohort structure. Does the temp0
                         arm preserve the cohort ranking, or is it a different measurement?
  F  Residual flips.     What survives at temp0, and is it concentrated?
  G  Drift.              Default-arm cell means vs the December values in the prereg, which is an
                         endpoint-drift check independent of the decoding question.

Writes analysis/decoding_control_diagnostics.csv and prints a verdict on W1 vs W2.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from decoding_control_io import BASE, OUT, band, both_arms_only, balanced_cells, load

df, _report = load()
df = both_arms_only(df)
df = balanced_cells(df)
print("analysing %d generations, %d models, %d cohorts\n" % (len(df), df.model.nunique(), df.profile_id.nunique()))

MODELS = sorted(df.model.unique())
BAR = "=" * 78


# ----------------------------------------------------------------- A. determinism
print(BAR)
print("A. DETERMINISM  -- did temperature 0 collapse each cell to one repeated answer?")
print(BAR)
print("%-24s %-8s %6s %10s %12s %14s" % ("MODEL", "ARM", "cells", "mean uniq", "1-value %", "modal share %"))
print("-" * 78)

det = []
for m in MODELS:
    for arm in ("default", "temp0"):
        g = df[(df.model == m) & (df.arm == arm)]
        uniq, single, modal, iuniq = [], 0, [], []
        for _, c in g.groupby("profile_id"):
            u = c.total.nunique()
            uniq.append(u)
            if u == 1:
                single += 1
            modal.append(c.total.value_counts().iat[0] / len(c))
            iuniq.append(c["items"].nunique())
        n = len(uniq)
        det.append(dict(model=m, arm=arm, cells=n, mean_unique_totals=np.mean(uniq),
                        pct_single_valued=100 * single / n, modal_share=100 * np.mean(modal),
                        mean_unique_item_vectors=np.mean(iuniq)))
        print("%-24s %-8s %6d %10.2f %12.1f %14.1f" % (m, arm, n, np.mean(uniq), 100 * single / n, 100 * np.mean(modal)))

detdf = pd.DataFrame(det)
print()
print("mean distinct ITEM VECTORS per cell (30 draws) -- the finer-grained view:")
for m in MODELS:
    d = detdf[(detdf.model == m) & (detdf.arm == "default")].mean_unique_item_vectors.iat[0]
    t = detdf[(detdf.model == m) & (detdf.arm == "temp0")].mean_unique_item_vectors.iat[0]
    print("  %-24s default %5.1f / 30    temp0 %5.1f / 30" % (m, d, t))


# ------------------------------------------------- B. is the temp0 value representative?
print()
print(BAR)
print("B. REPRESENTATIVENESS  -- where does the temp0 answer sit inside the default spread?")
print(BAR)
print("If temp0 recovers a stable central reading, percentiles cluster near 50.")
print("If greedy decoding just freezes one arbitrary point, they scatter.\n")
print("%-24s %8s %10s %12s %12s" % ("MODEL", "cells", "med pctile", "mean |d-50|", "outside 25-75 %"))
print("-" * 78)

repr_rows = []
for m in MODELS:
    d0 = df[(df.model == m) & (df.arm == "default")]
    t0 = df[(df.model == m) & (df.arm == "temp0")]
    pcts = []
    for pid, c in t0.groupby("profile_id"):
        ref = d0[d0.profile_id == pid].total.values
        if len(ref) == 0:
            continue
        v = c.total.value_counts().index[0]          # temp0 modal total
        # midpoint percentile: handles ties without inflating toward 0 or 100
        p = 100 * (np.sum(ref < v) + 0.5 * np.sum(ref == v)) / len(ref)
        pcts.append(p)
        repr_rows.append(dict(model=m, profile_id=pid, temp0_modal_total=int(v),
                              default_mean=float(ref.mean()), default_median=float(np.median(ref)),
                              pctile_in_default=round(p, 1)))
    pcts = np.array(pcts)
    print("%-24s %8d %10.1f %12.1f %12.1f"
          % (m, len(pcts), np.median(pcts), np.mean(np.abs(pcts - 50)),
             100 * np.mean((pcts < 25) | (pcts > 75))))

reprdf = pd.DataFrame(repr_rows)
allp = reprdf.pctile_in_default.values
print("\npooled: median percentile %.1f | mean |dev from 50| %.1f | outside 25-75 in %.1f%% of cells"
      % (np.median(allp), np.mean(np.abs(allp - 50)), 100 * np.mean((allp < 25) | (allp > 75))))
print("        cells where temp0 sits in the default distribution's tail (<10 or >90): %.1f%%"
      % (100 * np.mean((allp < 10) | (allp > 90))))


# ----------------------------------------------------------------- C. location shift
print()
print(BAR)
print("C. LOCATION SHIFT  -- does temperature 0 move the estimate, not just narrow it?")
print(BAR)
print("%-24s %10s %10s %9s %12s %10s" % ("MODEL", "def mean", "t0 mean", "delta", "Wilcoxon p", "max |cell d|"))
print("-" * 78)

shift_rows = []
for m in MODELS:
    a = df[(df.model == m) & (df.arm == "default")].groupby("profile_id").total.mean()
    b = df[(df.model == m) & (df.arm == "temp0")].groupby("profile_id").total.mean()
    j = pd.concat([a.rename("d"), b.rename("t")], axis=1).dropna()
    delta = j.t - j.d
    try:
        p = stats.wilcoxon(j.t, j.d).pvalue
    except Exception:
        p = np.nan
    print("%-24s %10.2f %10.2f %+9.2f %12.4f %10.2f"
          % (m, j.d.mean(), j.t.mean(), delta.mean(), p, delta.abs().max()))
    for pid in j.index:
        shift_rows.append(dict(model=m, profile_id=pid, default_mean=round(j.d[pid], 3),
                               temp0_mean=round(j.t[pid], 3), delta=round(delta[pid], 3)))

sh = pd.DataFrame(shift_rows)
print("\npooled mean shift %+.2f PHQ-8 points | mean |shift| %.2f | cells shifting >=1 point: %.1f%%"
      % (sh.delta.mean(), sh.delta.abs().mean(), 100 * (sh.delta.abs() >= 1).mean()))
print("cells shifting >= 5 points (the PHQ-8 minimum important difference): %.1f%%"
      % (100 * (sh.delta.abs() >= 5).mean()))


# ------------------------------------------------------------- D. category agreement
print()
print(BAR)
print("D. CATEGORY AGREEMENT  -- is the temp0 answer the one the default arm usually gives?")
print(BAR)
print("%-24s %8s %16s %18s" % ("MODEL", "cells", "same category %", "same side of 10 %"))
print("-" * 78)

agree_rows = []
for m in MODELS:
    d0 = df[(df.model == m) & (df.arm == "default")]
    t0 = df[(df.model == m) & (df.arm == "temp0")]
    sc = ss = n = 0
    for pid, c in t0.groupby("profile_id"):
        ref = d0[d0.profile_id == pid]
        if not len(ref):
            continue
        n += 1
        tcat = c.cat.value_counts().index[0]
        dcat = ref.cat.value_counts().index[0]
        tside = c.total.value_counts().index[0] >= 10
        dside = (ref.total >= 10).mean() >= 0.5
        sc += tcat == dcat
        ss += tside == dside
        agree_rows.append(dict(model=m, profile_id=pid, temp0_cat=tcat, default_modal_cat=dcat,
                               cat_match=tcat == dcat, side10_match=tside == dside,
                               default_pct_above10=round(100 * (ref.total >= 10).mean(), 1)))
    print("%-24s %8d %16.1f %18.1f" % (m, n, 100 * sc / n, 100 * ss / n))

ag = pd.DataFrame(agree_rows)
print("\npooled: temp0 category matches the default modal category in %.1f%% of cells" % (100 * ag.cat_match.mean()))
print("        temp0 falls on the default majority side of PHQ-8=10 in %.1f%% of cells" % (100 * ag.side10_match.mean()))
amb = ag[(ag.default_pct_above10 > 20) & (ag.default_pct_above10 < 80)]
print("        cells where the DEFAULT arm was itself split on the threshold (20-80%% above 10): %d of %d"
      % (len(amb), len(ag)))
if len(amb):
    print("        -> in these, temp0 reports a single side with no indication the case was borderline.")


# --------------------------------------------------------------- E. cohort ordering
print()
print(BAR)
print("E. COHORT ORDERING  -- do the two arms measure the same between-cohort structure?")
print(BAR)
for m in MODELS:
    a = df[(df.model == m) & (df.arm == "default")].groupby("profile_id").total.mean()
    b = df[(df.model == m) & (df.arm == "temp0")].groupby("profile_id").total.mean()
    j = pd.concat([a.rename("d"), b.rename("t")], axis=1).dropna()
    rho = stats.spearmanr(j.d, j.t).statistic
    r = stats.pearsonr(j.d, j.t).statistic
    print("  %-24s within-model across arms:  Spearman %.3f   Pearson %.3f" % (m, rho, r))

print()
for arm in ("default", "temp0"):
    piv = df[df.arm == arm].groupby(["model", "profile_id"]).total.mean().unstack(0)
    rhos = []
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            # Pairwise complete. Models cover different numbers of cohorts once the paired
            # restriction is applied, so a listwise drop would discard every cohort GLM lacks.
            pair = piv[[a, b]].dropna()
            if len(pair) >= 4:
                rhos.append(stats.spearmanr(pair[a], pair[b]).statistic)
    print("  cross-model cohort-ordering agreement, %-8s mean Spearman %.3f  (min %.3f, max %.3f, over %d model pairs)"
          % (arm, np.mean(rhos), np.min(rhos), np.max(rhos), len(rhos)))


# ------------------------------------------------------------- F. residual instability
print()
print(BAR)
print("F. RESIDUAL INSTABILITY AT TEMPERATURE 0")
print(BAR)
t0 = df[df.arm == "temp0"]
resid = []
for (m, pid), c in t0.groupby(["model", "profile_id"]):
    if c.cat.nunique() > 1:
        vc = c.total.value_counts()
        resid.append(dict(model=m, profile_id=pid, n=len(c), distinct=c.total.nunique(),
                          spread="%d-%d" % (c.total.min(), c.total.max()),
                          cats="/".join(sorted(c.cat.unique())),
                          crosses10=bool((c.total < 10).any() and (c.total >= 10).any()),
                          modal_share=round(100 * vc.iat[0] / len(c), 1)))
rz = pd.DataFrame(resid)
if len(rz):
    print("%d of %d temp0 cells still change severity category across 30 draws:\n" % (len(rz), t0.groupby(["model", "profile_id"]).ngroups))
    print(rz.to_string(index=False))
    print("\nof those, %d straddle PHQ-8 = 10." % rz.crosses10.sum())
else:
    print("none")


# ------------------------------------------------------- G. replication and drift
print()
print(BAR)
print("G. REPLICATION  -- August default arm vs the December corpus, per model, like for like")
print(BAR)
print("The December collection used the same design on these cohorts: clinical framing, 30 draws per")
print("cell, provider default decoding, same estimator. So the default arm is a direct re-measurement")
print("eight months on, and both the level and the flip rate can be compared per model.\n")

DEC2AUG = {"openai/gpt-4o-mini": "gpt-4o-mini",
           "deepseek/deepseek-chat-v3": "deepseek-chat",
           "google/gemini-3-flash-preview": "gemini-3-flash-preview",
           "z-ai/glm-4.7": "glm-4.7"}

dec = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
dec = dec[(dec.prompt_condition == "clinical") & dec.profile_id.isin(df.profile_id.unique())].copy()
dec["model"] = dec.model.map(DEC2AUG)
dec["total"] = dec.phq8_total.clip(0, 24).astype(int)
dec["cat"] = dec.total.map(band)


def flip_rate(g):
    """Same estimator as the headline: ordered within-cell pairs landing in different categories."""
    flips = pairs = cross = disc = 0
    for _, c in g.groupby("profile_id"):
        t, cc = c.total.tolist(), c.cat.tolist()
        for i, j_ in combinations(range(len(t)), 2):
            pairs += 1
            if cc[i] != cc[j_]:
                flips += 1
                disc += 1
                if (t[i] < 10) != (t[j_] < 10):
                    cross += 1
    return (100 * flips / pairs if pairs else np.nan,
            100 * cross / disc if disc else np.nan)


# Drift is a within-arm question and needs no pairing, so use the full deduplicated default arm
# rather than the paired subset, and compare each model against itself on the cohorts it covers.
full, _ = load(verbose=False)
aug = full[full.arm == "default"]

print("%-24s %9s %9s %8s   %9s %9s %8s" % ("MODEL", "dec mean", "aug mean", "delta", "dec flip", "aug flip", "delta"))
print("-" * 78)
for m in sorted(set(aug.model) & set(dec.model.dropna())):
    a = aug[aug.model == m]
    d = dec[(dec.model == m) & dec.profile_id.isin(a.profile_id.unique())]
    af, ac = flip_rate(a)
    dfp, dc = flip_rate(d)
    print("%-24s %9.2f %9.2f %+8.2f   %8.2f%% %8.2f%% %+7.2f"
          % (m, d.total.mean(), a.total.mean(), a.total.mean() - d.total.mean(), dfp, af, af - dfp))

print()
cm = dec.groupby("profile_id").total.mean().rename("december")
am = aug.groupby("profile_id").total.mean().rename("august")
j = pd.concat([cm, am], axis=1).dropna()
j["delta"] = j.august - j.december
print(j.round(2).to_string())
print("\ncohort means: mean shift %+.2f | mean |shift| %.2f | Spearman %.3f | Pearson %.3f"
      % (j.delta.mean(), j.delta.abs().mean(),
         stats.spearmanr(j.december, j.august).statistic,
         stats.pearsonr(j.december, j.august).statistic))
print("note: two of the four routes are undated substitutes for retired December snapshots, so a")
print("      per-model match is evidence the served build is behaviourally close, not proof it is")
print("      the same build. The pre-registration forbids claiming a reproduction of the corpus.")


# -------------------------------------------------------------------- verdict
print()
print(BAR)
print("VERDICT")
print(BAR)
# W1 and W2 are not properties of the experiment, they are properties of an ENDPOINT, so a pooled
# statistic is the wrong summary: averaging a frozen endpoint with an unaffected one reports a
# middle that describes neither. Classify per model instead.
print("%-24s %11s %11s %13s   %s" % ("MODEL", "flip delta", "modal share", "single-valued", "READING"))
print("-" * 78)
for m in MODELS:
    d = df[(df.model == m) & (df.arm == "default")]
    t = df[(df.model == m) & (df.arm == "temp0")]
    delta = flip_rate(t)[0] - flip_rate(d)[0]
    row = detdf[(detdf.model == m) & (detdf.arm == "temp0")]
    modal, single = row.modal_share.iat[0], row.pct_single_valued.iat[0]
    if abs(delta) < 5:
        read = "W0  temperature 0 not honoured; instability unchanged"
    elif modal >= 80:
        read = "W2  output frozen, not stabilised"
    else:
        read = "W1  genuinely narrower spread"
    print("%-24s %+10.1f %10.1f%% %12.1f%%   %s" % (m, delta, modal, single, read))

print()
print("pooled, for reference only: temp0 answer lands in a tail (<10 or >90 pctile) of the default")
print("distribution in %.1f%% of cells, and differs from the default modal category in %.1f%%."
      % (100 * np.mean((allp < 10) | (allp > 90)), 100 * (1 - ag.cat_match.mean())))
print()
print("-> The remedy is endpoint-dependent. Where the serving stack honours temperature 0 the flip")
print("   rate collapses, and it collapses because the output stops varying rather than because the")
print("   judgement steadied. Where the stack does not honour it, nothing changes. Neither case")
print("   makes the underlying case ambiguity go away: %d of %d cells had the DEFAULT arm split"
      % (len(amb), len(ag)))
print("   20-80%% across PHQ-8 = 10, and in those cells temperature 0 reports one side with no")
print("   indication the case was borderline.")

pd.concat([detdf.assign(block="determinism")], ignore_index=True).to_csv(
    os.path.join(OUT, "decoding_control_diagnostics.csv"), index=False)
reprdf.to_csv(os.path.join(OUT, "decoding_control_representativeness.csv"), index=False)
sh.merge(ag, on=["model", "profile_id"]).to_csv(os.path.join(OUT, "decoding_control_cellwise.csv"), index=False)
print("\nwritten -> decoding_control_diagnostics.csv, _representativeness.csv, _cellwise.csv")
