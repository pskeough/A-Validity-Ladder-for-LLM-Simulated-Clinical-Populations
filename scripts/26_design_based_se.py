"""
Design-based standard errors for the population anchors.

NHANES is a complex multistage sample. Treating it as simple random sampling understates the
standard error of a weighted mean, usually by a factor of one to two on health measures. Section 3.5
previously excluded anchor sampling error on the grounds that the derivation did not use the stratum
and cluster identifiers. They are in the DEMO files the pipeline already downloads, so the exclusion
was a choice rather than a constraint and is closed here.

Taylor-series linearisation for a ratio estimator, the standard design-based form for a weighted
mean, with strata as SDMVSTRA and PSUs as SDMVPSU nested within stratum. Emits
analysis/anchor_design_se.csv.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(BASE, "data", "nhanes_raw")
CYCLES = ["D", "E", "F", "G", "H", "I", "J"]
ITEMS = [f"DPQ0{i}0" for i in range(1, 9)]

frames = []
for c in CYCLES:
    demo = pd.read_sas(os.path.join(RAW, f"DEMO_{c}.xpt"))
    keep = ["SEQN", "RIDAGEYR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA", "RIAGENDR", "INDFMPIR",
            "RIDRETH1", "RIDRETH3"]
    demo = demo[[k for k in keep if k in demo.columns]]
    frames.append(demo.merge(pd.read_sas(os.path.join(RAW, f"DPQ_{c}.xpt")), on="SEQN"))
d = pd.concat(frames, ignore_index=True)
d[ITEMS] = d[ITEMS].where(d[ITEMS] <= 3)
d = d[(d.RIDAGEYR >= 18) & d[ITEMS].notna().all(axis=1) & (d.WTMEC2YR > 0)].copy()
d["w"] = d.WTMEC2YR / len(CYCLES)
d["phq8"] = d[ITEMS].sum(axis=1)
# Strata repeat across cycles, so make them unique per cycle before pooling.
d["stratum"] = d.SDMVSTRA.astype(int)
d["psu"] = d.SDMVPSU.astype(int)


def design_se(df):
    """Taylor linearisation for a weighted mean, with-replacement PSU variance."""
    W = df.w.sum()
    mu = (df.w * df.phq8).sum() / W
    df = df.assign(z=df.w * (df.phq8 - mu) / W)
    var = 0.0
    for _, st in df.groupby("stratum"):
        psu = st.groupby("psu").z.sum()
        n = len(psu)
        if n < 2:
            continue                      # lone PSU contributes nothing under this convention
        var += n / (n - 1) * ((psu - psu.mean()) ** 2).sum()
    return mu, float(np.sqrt(var))


def srs_se(df):
    W = df.w.sum()
    mu = (df.w * df.phq8).sum() / W
    v = (df.w * (df.phq8 - mu) ** 2).sum() / W
    return float(np.sqrt(v / len(df)))


groups = {"All adults 18+": d}
groups["Low"] = d[d.INDFMPIR <= 1.30]
groups["Middle"] = d[(d.INDFMPIR > 1.30) & (d.INDFMPIR <= 3.50)]
groups["High"] = d[d.INDFMPIR > 3.50]
groups["Cisgender men"] = d[d.RIAGENDR == 1]
groups["Cisgender women"] = d[d.RIAGENDR == 2]

# The four racial anchors were omitted from an earlier version of this pass, which left the
# fixed-constant simplification licensed for five of the nine benchmarked anchors while the paper
# claimed it for all. Asian is the thinnest anchor and carries the largest effect in the design, so
# it is the one most in need of a design-based interval. RIDRETH3 separates Asian from 2011 onward;
# earlier cycles carry RIDRETH1 only, which is why the Asian anchor rests on the shorter window.
# Race assignment must match 03_compute_groundtruth.py exactly or the two receipts describe
# different samples. That script keys Asian off RIDRETH3 == 6, which exists only from 2011, and
# takes White, Black and the two Hispanic strata from RIDRETH1 across all seven cycles. Keying
# every group off RIDRETH3 silently drops the pre-2011 cycles from the non-Asian anchors.
if "RIDRETH3" not in d.columns:
    d["RIDRETH3"] = np.nan
is_asian = d.RIDRETH3 == 6
groups["Asian"] = d[is_asian]
groups["White"] = d[(d.RIDRETH1 == 3) & ~is_asian]
groups["Black"] = d[(d.RIDRETH1 == 4) & ~is_asian]
groups["Hispanic"] = d[d.RIDRETH1.isin([1, 2]) & ~is_asian]

rows = []
for name, g in groups.items():
    if not len(g):
        continue
    mu, dse = design_se(g)
    sse = srs_se(g)
    rows.append(dict(group=name, n=len(g), weighted_mean=round(mu, 4),
                     se_design=round(dse, 4), se_srs=round(sse, 4),
                     design_effect_on_se=round(dse / sse, 3),
                     ci_lo=round(mu - 1.96 * dse, 4), ci_hi=round(mu + 1.96 * dse, 4)))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "anchor_design_se.csv"), index=False)
print(res.to_string(index=False))
print(f"\nlargest design-based SE across anchors: {res.se_design.max():.4f}")
print(f"SE inflation over simple random sampling: {res.design_effect_on_se.min():.2f}x to "
      f"{res.design_effect_on_se.max():.2f}x")
print(f"widest anchor 95% interval: {(res.ci_hi - res.ci_lo).max():.3f} points")
print("compare against the smallest residual in Table 2 (+2.75) and the cell-clustered model-side "
      "interval widths, which run about 0.7 points")
