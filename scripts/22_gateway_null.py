"""
A null for the DSM-5 gateway coherence rate.

Section "Results IV" reports that 2.68% of elevated cases violate the gateway rule (PHQ-8 total at
or above 10 while neither of the two cardinal items reaches 2), and reads the complement as evidence
of case-level coherence. That reading was unsupported as it stood, because the rule is easy to
satisfy by arithmetic alone: with both gateway items at 1, the remaining six items can still supply
18, so a case can clear 10 without either cardinal symptom. Every other structural claim in this
paper carries a permutation null. This one did not, and a rate with no null cannot distinguish a
model that routes symptom load onto the gateway items from a rule that is simply lenient.

Three nulls, each destroying a different thing and preserving the rest:

  conditional-marginal  each case keeps its own PHQ-8 total; its eight item responses are redrawn
                        from its own cohort's item marginals, conditioned on that total by
                        rejection. Preserves total, cohort, and per-item response tendencies;
                        destroys only the routing of load onto particular items. This is the
                        null the coherence claim is actually against.
  uniform-composition   each case keeps its total; the total is redistributed uniformly at random
                        over all compositions into eight parts of 0-3. Preserves nothing but the
                        total. The most permissive null and the loosest test.
  within-case-permute   each case's own eight responses are shuffled across item positions.
                        Preserves the total and the case's exact multiset of responses; destroys
                        only which item received which response.

Emits analysis/gateway_null.csv, pooled and per model. Seeded.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
GATE = ["phq8_1", "phq8_2"]          # the two cardinal DSM-5 items
ELEV, CARD, MAXV, NDRAW = 10, 2, 3, 200

rng = np.random.default_rng(20260726)
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
m["phq8_total"] = m[ITEMS].sum(axis=1).clip(0, 24)
elev = m[m.phq8_total >= ELEV].copy()
violates = lambda a: (a[:, 0] < CARD) & (a[:, 1] < CARD)


def compositions_uniform(total, n, k=8):
    """n uniform draws from the compositions of `total` into k parts each in [0, MAXV]."""
    out = np.empty((n, k), dtype=int)
    filled = 0
    while filled < n:
        want = (n - filled) * 3
        cand = rng.integers(0, MAXV + 1, size=(want, k))
        ok = cand[cand.sum(axis=1) == total]
        take = min(len(ok), n - filled)
        if take:
            out[filled:filled + take] = ok[:take]
            filled += take
        elif total > k * MAXV or total < 0:
            return None
    return out


def null_conditional_marginal(df):
    """Redraw items from the cohort's own per-item marginals, conditioned on the case total.

    Stratified by model as well as cohort. An earlier version pooled models inside a cohort, which
    made the pooled null a draw from a wider marginal than any single model's and inflated it to
    15.2% against an elevated-weighted mean of 12.9% over the per-model nulls. The pooled figure and
    the per-model factors were then not on the same scale, so the ratios could not be compared.

    The return is violations over retained draws, the same numerator and denominator the observed
    rate uses.
    """
    # Violations and retained draws are accumulated across cells and divided at the end. An earlier
    # version averaged per-cell rates with np.mean, which put this null on a different basis from
    # the observed rate it is compared against: the observed rate is violations over all elevated
    # cases, while an unweighted mean over cells gives a cell holding one case the same weight as a
    # cell holding sixty. Cells here run 1 to 60 elevated cases and cell size is associated with
    # violation rate, so the mismatch did not wash out. It inflated the pooled null from 10.5 to
    # 13.4 and, for GLM-4.7 alone, turned a ratio of 0.97 into 2.11. The sibling nulls below were
    # always case-weighted, so the three quoted in one sentence did not share a basis.
    viol = kept = 0
    for _, grp in df.groupby(["model", "profile_id"]):
        pools = [grp[c].to_numpy() for c in ITEMS]
        for _ in range(NDRAW // 20):
            drawn = np.stack([rng.choice(p, len(grp)) for p in pools], axis=1)
            # condition on the total by re-drawing rows that miss it, a bounded number of times
            for _ in range(60):
                bad = drawn.sum(axis=1) != grp.phq8_total.to_numpy()
                if not bad.any():
                    break
                drawn[bad] = np.stack([rng.choice(p, bad.sum()) for p in pools], axis=1)
            keep = drawn.sum(axis=1) == grp.phq8_total.to_numpy()
            if keep.sum():
                viol += int(violates(drawn[keep][:, :2]).sum())
                kept += int(keep.sum())
    return viol / kept * 100


def null_uniform(df):
    rates = []
    for total, grp in df.groupby("phq8_total"):
        c = compositions_uniform(int(total), min(NDRAW, 400))
        if c is not None:
            rates.append((violates(c[:, :2]).mean(), len(grp)))
    w = np.array([r[1] for r in rates], dtype=float)
    return float(np.average([r[0] for r in rates], weights=w)) * 100


def null_within_permute(df):
    a = df[ITEMS].to_numpy()
    hits = []
    for _ in range(NDRAW // 10):
        perm = np.apply_along_axis(rng.permutation, 1, a)
        hits.append(violates(perm[:, :2]).mean())
    return float(np.mean(hits)) * 100


rows = []
for label, d in [("pooled", elev)] + [(mm, g) for mm, g in elev.groupby("model")]:
    obs = violates(d[GATE].to_numpy()).mean() * 100
    cm, un, wp = null_conditional_marginal(d), null_uniform(d), null_within_permute(d)
    rows.append(dict(scope=label, n_elevated=len(d), observed_pct=round(obs, 2),
                     null_conditional_marginal_pct=round(cm, 2),
                     null_uniform_composition_pct=round(un, 2),
                     null_within_case_permute_pct=round(wp, 2),
                     ratio_vs_conditional=round(cm / obs, 2) if obs else np.nan))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "gateway_null.csv"), index=False)
print(res.to_string(index=False))

p = res[res.scope == "pooled"].iloc[0]
print(f"\npooled: observed {p.observed_pct:.2f}% against nulls of "
      f"{p.null_conditional_marginal_pct:.2f}%, {p.null_uniform_composition_pct:.2f}% and "
      f"{p.null_within_case_permute_pct:.2f}%")
print(f"  the rule is lenient, and the models still clear it by {p.ratio_vs_conditional:.1f}x "
      f"against the strictest null")
worst = res[res.scope != "pooled"].sort_values("ratio_vs_conditional").iloc[0]
print(f"  weakest model: {worst.scope} at {worst.observed_pct:.2f}% against its own null of "
      f"{worst.null_conditional_marginal_pct:.2f}% ({worst.ratio_vs_conditional:.1f}x)")
