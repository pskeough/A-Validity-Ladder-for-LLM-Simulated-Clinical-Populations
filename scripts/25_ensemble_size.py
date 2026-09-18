"""
How many generations per cell does a stable severity category take?

Section 13.3 tells builders never to record a single generation. That is a warning rather than a
specification, and the design can supply the specification. For each cell we take the modal PHQ-8
severity category over all 30 iterations as the target, draw k iterations without replacement,
and ask how often the modal category of the draw matches the target. The reported k is the smallest
ensemble reaching 90% and 95% agreement.

A draw counts as agreeing only when the target is the UNIQUE mode of the draw. A tie is not an
answer a pipeline can record, so counting ties as hits would inflate even-numbered ensembles: at
k = 2 any disagreement makes both values modes, which is why an earlier version of this script
reported that two generations suffice.

Emits analysis/ensemble_size.csv. Seeded.
"""
import os
from collections import Counter

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
CUTS = [5, 10, 15, 20]          # none / mild / moderate / mod-severe / severe
DRAWS, KMAX = 400, 15

rng = np.random.default_rng(20260726)
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
m["band"] = np.digitize(m.phq8_total, CUTS)


def agreement(vals, k):
    target = Counter(vals).most_common(1)[0][0]
    hits = 0
    for _ in range(DRAWS):
        d = rng.choice(vals, k, replace=False)
        c = Counter(d)
        top = max(c.values())
        modes = [v for v, n in c.items() if n == top]
        hits += len(modes) == 1 and modes[0] == target
    return hits / DRAWS


rows = []
for label, g in [("pooled", m)] + [(mm, gg) for mm, gg in m.groupby("model")]:
    cells = [v.to_numpy() for _, v in g.groupby(["model", "profile_id"]).band if len(v) >= KMAX]
    rec = {"scope": label, "n_cells": len(cells)}
    for k in range(1, KMAX + 1):
        rec[f"k{k}"] = round(float(np.mean([agreement(c, k) for c in cells])) * 100, 1)
    for thr in (90, 95):
        hit = next((k for k in range(1, KMAX + 1) if rec[f"k{k}"] >= thr), None)
        rec[f"k_for_{thr}"] = hit if hit else f">{KMAX}"
    rows.append(rec)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "ensemble_size.csv"), index=False)
cols = ["scope", "n_cells", "k1", "k3", "k5", "k7", "k9", "k11", "k_for_90", "k_for_95"]
print(res[cols].to_string(index=False))
p = res[res.scope == "pooled"].iloc[0]
print(f"\npooled: a single generation matches the cell's modal category {p.k1:.1f}% of the time")
print(f"  90% agreement needs k = {p.k_for_90}; 95% needs k = {p.k_for_95}")
per = res[res.scope != "pooled"]
print(f"  per model, k for 90%: " + ", ".join(f"{r.scope.split('/')[-1]} {r.k_for_90}" for _, r in per.iterrows()))
