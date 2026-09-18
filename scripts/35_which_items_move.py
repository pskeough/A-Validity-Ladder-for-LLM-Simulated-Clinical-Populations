"""Which item pairs move between cohorts, not just how far the matrices sit apart.

A Frobenius distance over 28 off-diagonal elements tells a fairness audience that structure differs
and tells a clinical audience nothing. The interesting question for anyone in psychometrics is which
symptoms decouple. The matrices are already computed, so this reports the pairs that move most on
the two largest contrasts, severity-matched so the comparison is not reading a level difference.

Second question, raised in the same review. A cohort marginal pools 24 cells drawn from 24 distinct
vignettes, so its correlation matrix mixes within-cell item covariance with between-cell mean
structure. If the contrast is driven by the latter then "they reorganize which symptoms travel
together" overstates it. The within-cell version below centres every cell on its own item means
before pooling, which removes between-cell mean structure entirely, and reports whether the contrast
survives.

Emits analysis/item_pair_shifts.csv and analysis/covariance_within_cell.csv.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
LABEL = {1: "anhedonia", 2: "depressed mood", 3: "sleep", 4: "fatigue",
         5: "appetite", 6: "self-worth", 7: "concentration", 8: "psychomotor"}

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
for c in ["gender", "ses"]:
    m[c] = m[c].astype(str).str.strip().str.strip('"')
m = m[m[ITEMS].notna().all(axis=1)].copy()
m["phq8_total"] = m[ITEMS].sum(axis=1)

CONTRASTS = [
    ("Low vs High SES", m.ses.str.startswith("Low"), m.ses.str.startswith("High")),
    ("Transgender vs cisgender", m.gender.str.contains("Trans"), m.gender.str.contains("Cis")),
]


def matched(a, b, tol=0.75):
    """Match cells one to one on mean severity so the comparison is not reading a level gap."""
    ca = a.groupby(["model", "profile_id"])[ITEMS + ["phq8_total"]].mean()
    cb = b.groupby(["model", "profile_id"])[ITEMS + ["phq8_total"]].mean()
    used, pairs = set(), []
    for ia, ra in ca.iterrows():
        cand = cb[(abs(cb.phq8_total - ra.phq8_total) <= tol)
                  & (~cb.index.isin(used)) & (cb.index.get_level_values(0) == ia[0])]
        if len(cand):
            j = (cand.phq8_total - ra.phq8_total).abs().idxmin()
            used.add(j)
            pairs.append((ia, j))
    ka = a.set_index(["model", "profile_id"]).loc[[p[0] for p in pairs]]
    kb = b.set_index(["model", "profile_id"]).loc[[p[1] for p in pairs]]
    return ka, kb


def within_cell(df):
    """Centre each cell on its own item means, so only within-cell covariance remains."""
    g = df.groupby(["model", "profile_id"])[ITEMS]
    return df[ITEMS] - g.transform("mean")


rows, wrows = [], []
for name, ma, mb in CONTRASTS:
    a, b = matched(m[ma], m[mb])
    Ra, Rb = a[ITEMS].corr().to_numpy(), b[ITEMS].corr().to_numpy()
    iu = np.triu_indices(8, k=1)
    d = Rb - Ra
    order = np.argsort(-np.abs(d[iu]))
    print(f"\n{name}: {len(a):,} vs {len(b):,} severity-matched generations")
    print(f"  Frobenius distance over the off-diagonal: {np.sqrt((d[iu] ** 2).sum()):.3f}")
    for k in order[:4]:
        i, j = iu[0][k], iu[1][k]
        rows.append(dict(contrast=name, item_a=LABEL[i + 1], item_b=LABEL[j + 1],
                         r_first=round(float(Ra[i, j]), 3), r_second=round(float(Rb[i, j]), 3),
                         shift=round(float(d[i, j]), 3)))
        print(f"    {LABEL[i+1]:15s} x {LABEL[j+1]:15s} {Ra[i,j]:+.3f} -> {Rb[i,j]:+.3f} "
              f"({d[i,j]:+.3f})")

    # within-cell only
    Wa, Wb = within_cell(a).corr().to_numpy(), within_cell(b).corr().to_numpy()
    dw = Wb - Wa
    f_raw = float(np.sqrt((d[iu] ** 2).sum()))
    f_within = float(np.sqrt((dw[iu] ** 2).sum()))
    wrows.append(dict(contrast=name, frobenius_pooled=round(f_raw, 4),
                      frobenius_within_cell=round(f_within, 4),
                      retained_pct=round(f_within / f_raw * 100, 1)))
    print(f"  within-cell only (between-cell mean structure removed): {f_within:.3f}, "
          f"{f_within / f_raw * 100:.0f}% of the pooled distance")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "item_pair_shifts.csv"), index=False)
pd.DataFrame(wrows).to_csv(os.path.join(OUT, "covariance_within_cell.csv"), index=False)
