"""Two checks the previous revision should have run and did not.

A reviewer caught both, and each is the mirror of a fix already applied elsewhere in the paper.

FIRST: the cross-instrument correlation is pooled, and pooling inflates it.
Section 4.1 reports PHQ-8 and GAD-7 correlating at 0.79 across 14,400 generations spanning 120
cohorts. Cells whose persona reads as distressed sit high on both instruments, so pooling induces
correlation that has nothing to do with within-person comorbidity. Loewe's comparison band is a
between-person correlation inside one sample. Section 8 already fixes exactly this error by centring
each cell on its own means before pooling; the same fix belongs here. If the within-cell figure
drops toward 0.6 the finding dissolves and must come out of the abstract.

SECOND: the endpoint test was run at one end only.
Section 4.1 reports 74.8% population zeros against 33.5% simulated, which is under-use of the bottom
category, and then attributes only the top-category deficit to response format. If the models also
under-emit zero on GAD-7 and AUDIT-C, both deficits are the same artefact and the honest statement
is one mechanism: these models compress ordinal responses toward the interior on every instrument
administered.

Emits analysis/within_cell_correlation.csv and analysis/endpoint_both_ends.csv.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(BASE, "data", "nhanes_raw")
PHQ = [f"phq8_{i}" for i in range(1, 9)]
GAD = [f"gad7_{i}" for i in range(1, 8)]
AUD = [f"audit_{i}" for i in range(1, 4)]
NH_PHQ = [f"DPQ0{i}0" for i in range(1, 9)]

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m["gender"] = m.gender.astype(str).str.strip().str.strip('"')
cis = m[m.gender.str.contains("Cis")].copy()

# ---- 1. pooled against within-cell -----------------------------------------------------------
rows = []
for scope, df in [("ALL", cis)] + [(mm, g) for mm, g in cis.groupby("model")]:
    d = df[["model", "profile_id", "phq8_total", "gad7_total"]].dropna()
    pooled = float(d.phq8_total.corr(d.gad7_total))
    # Centre each cell on its own means, which removes between-cell mean structure entirely and
    # leaves only the within-cell covariation Loewe's between-person figure is comparable to.
    g = d.groupby(["model", "profile_id"])
    cp = d.phq8_total - g.phq8_total.transform("mean")
    cg = d.gad7_total - g.gad7_total.transform("mean")
    within_pooled = float(np.corrcoef(cp, cg)[0, 1])
    # Also the plain mean of per-cell correlations, which weights every cell equally.
    percell = d.groupby(["model", "profile_id"]).apply(
        lambda x: x.phq8_total.corr(x.gad7_total), include_groups=False).dropna()
    rows.append(dict(scope=scope, n=len(d), r_pooled=round(pooled, 4),
                     r_within_cell=round(within_pooled, 4),
                     r_mean_per_cell=round(float(percell.mean()), 4),
                     n_cells=len(percell)))
cor = pd.DataFrame(rows)
cor.to_csv(os.path.join(OUT, "within_cell_correlation.csv"), index=False)
print(cor.to_string(index=False))
_a = cor[cor.scope == "ALL"].iloc[0]
print(f"\npooled {_a.r_pooled:.3f} -> within-cell {_a.r_within_cell:.3f} "
      f"(mean per-cell {_a.r_mean_per_cell:.3f})")
print(f"  Loewe reports 0.64 to 0.75 across three scales in a German primary-care sample; the "
      f"within-cell figure is the comparable object")
print("  VERDICT: " + ("survives, still above the band" if _a.r_within_cell > 0.75
                       else "inside or below the band; the claim does not hold"))

# ---- 2. both endpoints, all three instruments ------------------------------------------------
frames = []
for c in "DEFGHIJ":
    d = pd.read_sas(os.path.join(RAW, f"DEMO_{c}.xpt"))[["SEQN", "RIDAGEYR", "WTMEC2YR"]]
    frames.append(d.merge(pd.read_sas(os.path.join(RAW, f"DPQ_{c}.xpt")), on="SEQN"))
pop = pd.concat(frames, ignore_index=True)
pop[NH_PHQ] = pop[NH_PHQ].where(pop[NH_PHQ] <= 3)
pop = pop[(pop.RIDAGEYR >= 18) & pop[NH_PHQ].notna().all(axis=1) & (pop.WTMEC2YR > 0)].copy()
w = (pop.WTMEC2YR / 7).to_numpy()
pa = np.rint(pop[NH_PHQ].to_numpy(float))
pop_top = float((pa == 3).sum(axis=1) @ w / (w.sum() * 8) * 100)
pop_bot = float((pa == 0).sum(axis=1) @ w / (w.sum() * 8) * 100)

erows = []
for name, items, top in [("PHQ-8", PHQ, 3), ("GAD-7", GAD, 3), ("AUDIT-C", AUD, 4)]:
    a = np.rint(cis[items].to_numpy(float))
    a = a[~np.isnan(a).any(axis=1)]
    # Interior use is the unifying quantity: neither endpoint, on a scale of its own width.
    interior = float(((a > 0) & (a < top)).sum() / a.size * 100)
    erows.append(dict(instrument=name, max_category=top,
                      pct_at_max=round(float((a == top).sum() / a.size * 100), 3),
                      pct_at_zero=round(float((a == 0).sum() / a.size * 100), 3),
                      pct_interior=round(interior, 3),
                      pop_pct_at_max=round(pop_top, 3) if name == "PHQ-8" else np.nan,
                      pop_pct_at_zero=round(pop_bot, 3) if name == "PHQ-8" else np.nan))
ep = pd.DataFrame(erows)
ep.to_csv(os.path.join(OUT, "endpoint_both_ends.csv"), index=False)
print("\n" + ep.to_string(index=False))
_pi = float((np.rint(pop[NH_PHQ].to_numpy(float)) > 0).sum(axis=1) @ w / (w.sum() * 8) * 100) \
    - float((np.rint(pop[NH_PHQ].to_numpy(float)) == 3).sum(axis=1) @ w / (w.sum() * 8) * 100)
print(f"\npopulation PHQ-8 interior use (neither endpoint): {_pi:.2f}%")
print(f"simulated interior use: " + ", ".join(f"{r.instrument} {r.pct_interior:.1f}%"
                                              for r in ep.itertuples()))
_both = bool((ep.pct_at_zero < pop_bot).all() and (ep.pct_at_max < pop_top).all())
print(f"\nboth endpoints under-used on every instrument: {_both}")
print("  if true, the floor deficit and the ceiling deficit are one mechanism, not two")
