"""Every covariance contrast recomputed with between-cell mean structure removed.

Section 9 reports one aggregate version of this check (script 35): pooled over the four crossing
contrasts, the transgender-versus-cisgender distance retains 59% of its magnitude once each design
cell is centred on its own item means, against 79% for low-versus-high SES. That aggregate is not
enough for Table 6, which reports a range per axis and whose gender row carries the paper's headline
equity claim. Reporting the axis with the largest between-cell share on its uncentred number only is
the weakest place in the table.

So this runs the centring per contrast, on all fourteen, inside script 17's machinery: one
severity-band-matched draw, then a null built by permuting labels within band on that same draw. The
null is recomputed on the centred columns rather than borrowed from the uncentred run, and it has to
be. Centring subtracts a mean estimated from 30 draws, which adds estimation noise to every
correlation, so the centred statistic sits on a higher noise floor than the uncentred one and
comparing it against the uncentred null would read that floor as structure. Several small contrasts
come back larger after centring for exactly this reason.

Cell means are taken over the cohort's full set of rows before matching, since a cell's item means
are a property of the cell and not of whichever subsample the matcher drew.

Emits analysis/covariance_centred_contrasts.csv.
"""
import os

import numpy as np
import pandas as pd

SEED = 42
N_PERM = 400
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
CENT = [f"c_{i}" for i in ITEMS]
BANDS = [-1, 4, 9, 14, 19, 24]

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
rng = np.random.default_rng(SEED)

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
m["ses_clean"] = m["ses"].str.replace('"', '', regex=False).str.split(" ").str[0]
m["phq8_total"] = m["phq8_total"].clip(0, 24)
c = m.dropna(subset=ITEMS).copy()
c[CENT] = c[ITEMS] - c.groupby(["model", "profile_id"])[ITEMS].transform("mean")
c["band"] = pd.cut(c.phq8_total, BANDS, labels=False).astype(int)

PAIRS = [("race", "Black", "White"), ("race", "Asian", "White"), ("race", "Hispanic", "White"),
         ("race", "Multiracial", "White"),
         ("gender", "Cisgender Man", "Cisgender Woman"),
         ("gender", "Cisgender Man", "Transgender Woman"),
         ("gender", "Cisgender Man", "Transgender Man"),
         ("gender", "Cisgender Woman", "Transgender Woman"),
         ("gender", "Cisgender Woman", "Transgender Man"),
         ("gender", "Transgender Woman", "Transgender Man"),
         ("ses", "Low", "High"), ("ses", "Middle", "High"), ("ses", "Low", "Middle"),
         ("relationship", "Single", "Married")]


def is_large(dim, a, b):
    return dim == "ses" or (dim == "gender" and a.startswith("Cis") and b.startswith("Trans"))


def frob(a, b):
    return float(np.linalg.norm(np.corrcoef(a, rowvar=False)
                                - np.corrcoef(b, rowvar=False), ord="fro"))


def contrast(dim, a, b):
    col = "ses_clean" if dim == "ses" else dim
    sa, sb = c[c[col] == a], c[c[col] == b]
    ia, ib = [], []
    for bd in range(5):
        xa, xb = sa.index[sa.band == bd].to_numpy(), sb.index[sb.band == bd].to_numpy()
        k = min(len(xa), len(xb))
        if k < 2:
            continue
        ia.append(rng.choice(xa, k, replace=False))
        ib.append(rng.choice(xb, k, replace=False))
    ia, ib = np.concatenate(ia), np.concatenate(ib)
    out = dict(dimension=dim, group_a=a, group_b=b, n_matched_per_matrix=len(ia),
               large=is_large(dim, a, b))
    for tag, cols in (("raw", ITEMS), ("centred", CENT)):
        X = c[cols].to_numpy()
        pos = {v: i for i, v in enumerate(c.index.to_numpy())}
        gi = np.array([pos[v] for v in ia])
        gj = np.array([pos[v] for v in ib])
        obs = frob(X[gi], X[gj])
        pooled = np.concatenate([gi, gj])
        pband = c.band.to_numpy()[pooled]
        null = np.empty(N_PERM)
        for t in range(N_PERM):
            L, R = [], []
            for bd in range(5):
                idx = pooled[pband == bd]
                if len(idx) < 4:
                    continue
                p = rng.permutation(idx)
                h = len(p) // 2
                L.append(p[:h])
                R.append(p[h:2 * h])
            null[t] = frob(X[np.concatenate(L)], X[np.concatenate(R)])
        out[f"{tag}_frobenius"] = round(obs, 4)
        out[f"{tag}_null_p95"] = round(float(np.percentile(null, 95)), 4)
        out[f"{tag}_excess"] = round(obs - float(null.mean()), 4)
        out[f"{tag}_p_perm"] = round((1 + int((null >= obs).sum())) / (N_PERM + 1), 5)
    out["retained_pct"] = round(out["centred_frobenius"] / out["raw_frobenius"] * 100, 1)
    out["excess_retained_pct"] = round(out["centred_excess"] / out["raw_excess"] * 100, 1)
    return out


rows = [contrast(*p) for p in PAIRS]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "covariance_centred_contrasts.csv"), index=False)
for _, r in df.iterrows():
    print(f"  {r.dimension:12s} {r.group_a[:17]:18s} vs {r.group_b[:17]:18s} "
          f"raw {r.raw_frobenius:.3f} (null {r.raw_null_p95:.3f})  "
          f"centred {r.centred_frobenius:.3f} (null {r.centred_null_p95:.3f})  "
          f"excess {r.raw_excess:.3f} -> {r.centred_excess:.3f}")

big, small = df[df.large], df[~df.large]
print(f"\nabove own null: raw {int((df.raw_frobenius > df.raw_null_p95).sum())}/14, "
      f"centred {int((df.centred_frobenius > df.centred_null_p95).sum())}/14")
for tag in ("raw", "centred"):
    b, s = big[f"{tag}_excess"], small[f"{tag}_excess"]
    print(f"{tag:8s} excess over null: small {s.min():.3f}-{s.max():.3f}  "
          f"large {b.min():.3f}-{b.max():.3f}  "
          f"median ratio {b.median() / s.median():.2f}x  extremum {b.min() / s.max():.2f}x  "
          f"ordered {sum(x > y for x in b for y in s)}/{len(b) * len(s)}")
cross = big[big.dimension == "gender"]
ses = big[big.dimension == "ses"]
print(f"\ncentred excess: trans x cis {cross.centred_excess.min():.3f}-"
      f"{cross.centred_excess.max():.3f}, socioeconomic {ses.centred_excess.min():.3f}-"
      f"{ses.centred_excess.max():.3f}, small {small.centred_excess.min():.3f}-"
      f"{small.centred_excess.max():.3f}")
