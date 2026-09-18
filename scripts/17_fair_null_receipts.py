"""
Severity-controlled inference for the structure and stability comparisons.

Two comparisons in this paper pit cohorts against each other that also differ in mean severity,
so both need nulls that hold severity fixed rather than assuming a functional form for it.

(1) COVARIANCE. The statistic is a Frobenius distance between PHQ-8 item correlation matrices
    computed on severity-matched subsamples. Its null applies the IDENTICAL matched-subsample
    procedure to labels permuted within severity band, so null and observed share size and band
    composition. A split-half floor estimated on half-cohort matrices is not a valid comparison
    here: correlation-matrix estimation error scales with sample size, and the half-cohort
    matrices differ in size from the matched subsamples by a factor that varies from pair to pair.
    Emitted both pooled and per model, since the per-model result bounds how much of the pooled
    band separation survives at one quarter of the sample.

(2) DISPERSION BY DESIGN AXIS. Cell dispersion is compared between axis levels using cells
    matched one-to-one on mean severity within model, without replacement, so no control cell is
    reused and the compared quantities are independent. A second control permutes axis labels
    within model and severity quintile. Residualizing cell dispersion on cell mean with a pooled
    linear fit is not adequate on a bounded scale, where the relation is concave and
    model-specific and socioeconomic status is close to collinear with the mean.

Emits analysis/{covariance_fair_null,covariance_fair_null_permodel,fracture_matched,
fracture_stratified}.csv. Seeded (SEED=42), idempotent.
"""
import pandas as pd, numpy as np, os
from scipy import stats

SEED = 42
N_PERM = 1000          # pooled null
N_PERM_MODEL = 250     # per-model null (secondary, 4x the work)
N_STRAT = 2000
MATCH_TOL = 0.75
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
BANDS = [-1, 4, 9, 14, 19, 24]
BASE = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.join(BASE, "..")
OUT = os.path.join(BASE, "analysis")
rng = np.random.default_rng(SEED)

def cmat(a):
    """Correlation matrix of an (n x 8) array, via numpy for speed."""
    return np.corrcoef(a, rowvar=False)

def frob(a, b):
    return float(np.linalg.norm(a - b, ord="fro"))

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
is_big = lambda dim, a, b: dim == "ses" or (dim == "gender" and a.startswith("Cis") and b.startswith("Trans"))

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
m["ses_clean"] = m.ses.str.replace('"', '', regex=False).str.split(" ").str[0]
c = m.dropna(subset=ITEMS).copy()
c["band"] = pd.cut(c.phq8_total, BANDS, labels=False).astype(int)

def fair_null_row(sub, dim, a, b, nperm):
    col = "ses_clean" if dim == "ses" else dim
    sa, sb = sub[sub[col] == a], sub[sub[col] == b]
    ia, ib = [], []
    for bd in range(5):
        xa = sa.index[sa.band == bd].to_numpy(); xb = sb.index[sb.band == bd].to_numpy()
        k = min(len(xa), len(xb))
        if k < 2: continue
        ia.append(rng.choice(xa, k, replace=False)); ib.append(rng.choice(xb, k, replace=False))
    ia, ib = np.concatenate(ia), np.concatenate(ib)
    X = sub[ITEMS].to_numpy()
    pos = {v: i for i, v in enumerate(sub.index.to_numpy())}
    gi = np.array([pos[v] for v in ia]); gj = np.array([pos[v] for v in ib])
    obs = frob(cmat(X[gi]), cmat(X[gj]))
    pooled = np.concatenate([gi, gj])
    pband = sub.band.to_numpy()[pooled]
    null = np.empty(nperm)
    for t in range(nperm):
        L, R = [], []
        for bd in range(5):
            idx = pooled[pband == bd]
            if len(idx) < 4: continue
            p = rng.permutation(idx); h = len(p) // 2
            L.append(p[:h]); R.append(p[h:2 * h])
        null[t] = frob(cmat(X[np.concatenate(L)]), cmat(X[np.concatenate(R)]))
    return dict(dimension=dim, group_a=a, group_b=b, n_matched_per_matrix=len(ia),
                matched_frobenius=round(obs, 4),
                fair_null_mean=round(float(null.mean()), 4),
                fair_null_p95=round(float(np.percentile(null, 95)), 4),
                p_perm=round((1 + int((null >= obs).sum())) / (nperm + 1), 5),
                above_fair_null=bool(obs > np.percentile(null, 95)))

rows = [fair_null_row(c, d, a, b, N_PERM) for d, a, b in PAIRS]
cf = pd.DataFrame(rows)
cf.to_csv(os.path.join(OUT, "covariance_fair_null.csv"), index=False)
print(f"=== (1a) covariance vs size-matched fair null, pooled ({N_PERM} perms) ===")
print(cf.to_string(index=False))
bigv = cf[[is_big(*r) for r in zip(cf.dimension, cf.group_a, cf.group_b)]].matched_frobenius
smallv = cf[[not is_big(*r) for r in zip(cf.dimension, cf.group_a, cf.group_b)]].matched_frobenius
print(f"\n  small band {smallv.min():.3f}-{smallv.max():.3f} | large band {bigv.min():.3f}-{bigv.max():.3f}")
print(f"  ratio {bigv.min()/smallv.max():.2f}x to {bigv.max()/smallv.min():.2f}x | above null {int(cf.above_fair_null.sum())}/{len(cf)}")

# ---- per model: does the band separation survive at n/4? ----------------------------------
prow = []
for mdl, sub in c.groupby("model"):
    sub = sub.copy()
    r = [fair_null_row(sub, d, a, b, N_PERM_MODEL) for d, a, b in PAIRS]
    df = pd.DataFrame(r); df["model"] = mdl
    prow.append(df)
pm = pd.concat(prow, ignore_index=True)
pm["is_large"] = [is_big(*x) for x in zip(pm.dimension, pm.group_a, pm.group_b)]
pm.to_csv(os.path.join(OUT, "covariance_fair_null_permodel.csv"), index=False)
print(f"\n=== (1b) per model ({N_PERM_MODEL} perms) ===")
for mdl, g in pm.groupby("model"):
    b_, s_ = g[g.is_large].matched_frobenius, g[~g.is_large].matched_frobenius
    print(f"  {mdl:32s} small {s_.min():.3f}-{s_.max():.3f} | large {b_.min():.3f}-{b_.max():.3f} "
          f"| separated: {b_.min() > s_.max()} | large above null {int(g[g.is_large].above_fair_null.sum())}/{len(g[g.is_large])}")
ses_ok = pm[(pm.dimension == "ses")].groupby("model").above_fair_null.sum()
tc_ok = pm[pm.is_large & (pm.dimension == "gender")].above_fair_null.sum()
print(f"  SES contrasts above per-model null: {int(ses_ok.sum())}/{len(pm[pm.dimension=='ses'])}")
print(f"  trans-cis contrasts above per-model null: {int(tc_ok)}/{len(pm[pm.is_large & (pm.dimension=='gender')])}")
print(f"  models with a clean band gap: {sum(g[g.is_large].matched_frobenius.min() > g[~g.is_large].matched_frobenius.max() for _, g in pm.groupby('model'))}/4")

# ---- (2) dispersion: 1:1 matched cells + stratified permutation ----------------------------
def cells(path):
    d = pd.read_csv(path); d["phq8_total"] = d.phq8_total.clip(0, 24)
    d["ses_clean"] = d.ses.str.replace('"', '', regex=False).str.split(" ").str[0]
    g = d.groupby(["model", "profile_id"]).agg(
        sd=("phq8_total", lambda x: x.std(ddof=1)), mean=("phq8_total", "mean"),
        race=("race", "first"), gender=("gender", "first"),
        ses=("ses_clean", "first"), relationship=("relationship", "first")).reset_index()
    g["is_trans"] = g.gender.str.startswith("Transgender")
    return g

def pair_one_to_one(cell, mhi, mlo):
    """Greedy nearest-mean matching without replacement: each control cell is used at most once,
    so the paired differences are independent."""
    A, B = cell[mhi], cell[mlo]
    used, deltas = set(), []
    for _, ra in A.sort_values("mean").iterrows():
        cand = B[(B.model == ra.model) & (~B.index.isin(used)) & ((B["mean"] - ra["mean"]).abs() <= MATCH_TOL)]
        if not len(cand): continue
        j = (cand["mean"] - ra["mean"]).abs().idxmin()
        used.add(j); deltas.append(ra.sd - B.loc[j].sd)
    if len(deltas) < 5: return np.nan, np.nan, len(deltas)
    return float(np.mean(deltas)), float(stats.ttest_1samp(deltas, 0).pvalue), len(deltas)

mrows, srows = [], []
for cond, path in (("clinical", os.path.join(ROOT, "runs", "run1", "data", "audit_results.csv")),
                   ("narrative", os.path.join(ROOT, "runs", "run2", "data", "audit_results.csv"))):
    cell = cells(path)
    TESTS = [("transgender vs cisgender", cell.is_trans, ~cell.is_trans),
             ("Low vs High SES", cell.ses == "Low", cell.ses == "High"),
             ("Middle vs High SES", cell.ses == "Middle", cell.ses == "High"),
             ("Low vs Middle SES", cell.ses == "Low", cell.ses == "Middle"),
             ("Single vs Married", cell.relationship == "Single", cell.relationship == "Married"),
             ("Black vs White", cell.race == "Black", cell.race == "White"),
             ("Asian vs White", cell.race == "Asian", cell.race == "White"),
             ("Hispanic vs White", cell.race == "Hispanic", cell.race == "White")]
    for label, hi, lo in TESTS:
        d, p, n = pair_one_to_one(cell, hi, lo)
        v = "insufficient" if d != d else ("elevated" if d > 0 and p < .05
                                           else "lower" if d < 0 and p < .05 else "n.s.")
        mrows.append(dict(condition=cond, contrast=label,
                          mean_sd_difference=round(d, 4) if d == d else np.nan,
                          p=p, n_matched_pairs=n, verdict=v))
    cell["q"] = cell.groupby("model")["mean"].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
    for dim in ["race", "gender", "ses", "relationship"]:
        obs = stats.kruskal(*[g.sd.values for _, g in cell.groupby(dim)]).statistic
        null = np.empty(N_STRAT)
        for t in range(N_STRAT):
            lab = cell.groupby(["model", "q"])[dim].transform(lambda s: rng.permutation(s.values))
            null[t] = stats.kruskal(*[g.sd.values for _, g in cell.groupby(lab)]).statistic
        pv = (1 + int((null >= obs).sum())) / (N_STRAT + 1)
        srows.append(dict(condition=cond, dimension=dim, H_observed=round(float(obs), 3),
                          null_median=round(float(np.median(null)), 3),
                          null_p95=round(float(np.percentile(null, 95)), 3),
                          p_stratified=round(pv, 5), survives=bool(pv < .05)))

fm = pd.DataFrame(mrows); fs = pd.DataFrame(srows)
fm.to_csv(os.path.join(OUT, "fracture_matched.csv"), index=False)
fs.to_csv(os.path.join(OUT, "fracture_stratified.csv"), index=False)
print("\n=== (2a) one-to-one severity-matched cell pairs ===")
print(fm.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print("\n=== (2b) model x severity-quintile stratified permutation of KW ===")
print(fs.to_string(index=False))
