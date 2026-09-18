"""
Does the residual depend on which NHANES era the anchor comes from?

The anchors pool seven two-year cycles, 2005-2006 through 2017-2018. Population depression rose over
that window and rose again after 2020, so an anchor built on 2005-2018 could understate a contemporary
population mean and inflate every residual the paper reports. The models were queried in 2025 and a
reader is entitled to ask whether they are being compared against a population that no longer exists.

Three anchor sets, same derivation and same frame each time:

  paper          2005-2006 through 2017-2018, seven cycles, WTMEC2YR / 7
  pre-pandemic   2017-March 2020, the CDC combined pre-pandemic file, WTMECPRP
  recent         2021-2023, WTMEC2YR

Weights are never pooled across incompatible designs. Each anchor set uses its own MEC weight and its
own strata and PSUs, and the three are compared as separate estimates rather than combined.

The comparison that matters is not whether the anchors differ but whether any difference is large
enough to move a residual. Emits analysis/anchor_recency.csv.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(BASE, "data", "nhanes_raw")
ITEMS = [f"DPQ0{i}0" for i in range(1, 9)]

ERAS = {
    "paper 2005-2018": ([(f"DEMO_{c}", f"DPQ_{c}") for c in "DEFGHIJ"], "WTMEC2YR", 7),
    "pre-pandemic 2017-Mar 2020": ([("P_DEMO", "P_DPQ")], "WTMECPRP", 1),
    "recent 2021-2023": ([("DEMO_L", "DPQ_L")], "WTMEC2YR", 1),
}


def load(pairs, wcol, ncycles):
    frames = []
    for demo_f, dpq_f in pairs:
        demo = pd.read_sas(os.path.join(RAW, f"{demo_f}.xpt"))
        keep = [k for k in ["SEQN", "RIDAGEYR", wcol, "SDMVPSU", "SDMVSTRA", "RIAGENDR",
                            "INDFMPIR", "RIDRETH1", "RIDRETH3"] if k in demo.columns]
        frames.append(demo[keep].merge(pd.read_sas(os.path.join(RAW, f"{dpq_f}.xpt")), on="SEQN"))
    d = pd.concat(frames, ignore_index=True)
    d[ITEMS] = d[ITEMS].where(d[ITEMS] <= 3)
    d = d[(d.RIDAGEYR >= 18) & d[ITEMS].notna().all(axis=1) & (d[wcol] > 0)].copy()
    d["w"] = d[wcol] / ncycles
    d["phq8"] = d[ITEMS].sum(axis=1)
    return d


def subgroups(d):
    """Group definitions identical to 03_compute_groundtruth.py."""
    is_asian = d.RIDRETH3 == 6
    return {
        "White": d[(d.RIDRETH1 == 3) & ~is_asian],
        "Black": d[(d.RIDRETH1 == 4) & ~is_asian],
        "Asian": d[is_asian],
        "Hispanic": d[d.RIDRETH1.isin([1, 2]) & ~is_asian],
        "Cisgender Man": d[d.RIAGENDR == 1],
        "Cisgender Woman": d[d.RIAGENDR == 2],
        "Low": d[d.INDFMPIR <= 1.30],
        "Middle": d[(d.INDFMPIR > 1.30) & (d.INDFMPIR <= 3.50)],
        "High": d[d.INDFMPIR > 3.50],
    }


def wmean(g):
    return float((g.w * g.phq8).sum() / g.w.sum())


def design_se(df):
    """Taylor linearisation for a weighted mean, matching 26_design_based_se.py.

    Each window carries its own sampling error, and one cycle is noisier than seven. Comparing a
    residual computed on the recent anchor against the standard error computed on the paper window
    mixes the two, which is how a factor of eight was once written as more than twenty.
    """
    W = df.w.sum()
    mu = (df.w * df.phq8).sum() / W
    df = df.assign(z=df.w * (df.phq8 - mu) / W)
    var = 0.0
    for _, st in df.groupby(df.SDMVSTRA.astype(int)):
        psu = st.groupby(st.SDMVPSU.astype(int)).z.sum()
        if len(psu) < 2:
            continue
        var += len(psu) / (len(psu) - 1) * ((psu - psu.mean()) ** 2).sum()
    return float(np.sqrt(var))


sim = pd.read_csv(os.path.join(OUT, "bias_residuals_CISONLY.csv")).set_index("group")

anchors, overall, ses_by_era = {}, {}, []
for era, (pairs, wcol, n) in ERAS.items():
    d = load(pairs, wcol, n)
    overall[era] = (wmean(d), len(d))
    groups = subgroups(d)
    anchors[era] = {k: (wmean(g), len(g)) for k, g in groups.items() if len(g)}
    ses = {k: design_se(g) for k, g in groups.items() if len(g)}
    worst = max(ses, key=ses.get)
    ses_by_era.append(dict(era=era, largest_design_se=round(ses[worst], 4), largest_se_group=worst,
                           asian_n=len(groups["Asian"])))
    print(f"{era}: n = {len(d):,}, all-adult weighted mean {overall[era][0]:.3f}, "
          f"largest design SE {ses[worst]:.4f} ({worst}), Asian n = {len(groups['Asian'])}")
pd.DataFrame(ses_by_era).to_csv(os.path.join(OUT, "anchor_se_by_window.csv"), index=False)

rows = []
for group in sim.index:
    r = dict(group=group, model_mean=round(float(sim.loc[group, "model_mean"]), 3))
    for era in ERAS:
        mu, n = anchors[era].get(group, (np.nan, 0))
        r[f"anchor_{era}"] = round(mu, 3)
        r[f"resid_{era}"] = round(float(sim.loc[group, "model_mean"]) - mu, 3)
        r[f"n_{era}"] = n
    r["resid_shift_recent_vs_paper"] = round(
        r["resid_recent 2021-2023"] - r["resid_paper 2005-2018"], 3)
    rows.append(r)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "anchor_recency.csv"), index=False)
cols = ["group", "model_mean"] + [c for c in res.columns if c.startswith(("anchor_", "resid_"))]
print("\n" + res[cols].to_string(index=False))

sh = res.resid_shift_recent_vs_paper
print(f"\nsmallest residual on the paper anchor: {res['resid_paper 2005-2018'].min():+.3f}")
print(f"smallest residual on the 2021-2023 anchor: {res['resid_recent 2021-2023'].min():+.3f}")
print(f"residual shift from moving to the recent anchor: {sh.min():+.3f} to {sh.max():+.3f} points, "
      f"mean {sh.mean():+.3f}")
print(f"every residual remains positive on all three anchors: "
      f"{bool((res[[c for c in res.columns if c.startswith('resid_')]].drop(columns=['resid_shift_recent_vs_paper']) > 0).all().all())}")
