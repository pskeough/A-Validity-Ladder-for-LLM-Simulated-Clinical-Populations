"""
A population reference for level 4, structural fidelity.

Level 4 asks whether a model's PHQ-8 item correlation structure differs between two demographic
cohorts. The statistic is a Frobenius distance between item correlation matrices computed on
severity-matched subsamples (script 17), read against a permutation null that holds severity fixed
(script 17) and against a within-cohort split-half floor (script 06). What the paper has so far
lacked is the corresponding quantity in the population the personas are meant to stand for: whether
US adults themselves show a between-group difference in PHQ-8 item structure at the same
resolution. The current reference is a citation to published invariance tests, and on the income
axis there is no citation at all. NHANES carries the same eight items on the same response scale,
so the population's own between-group structural distance can be computed with the identical
procedure rather than borrowed.

WHAT IS AND IS NOT COMPARABLE ACROSS THE TWO SOURCES

A Frobenius distance between two ESTIMATED correlation matrices is not a fixed property of the two
populations. It carries estimation error, and that error shrinks with sample size. Two groups drawn
from one identical distribution will still show a positive distance, larger at n = 1,000 than at
n = 10,000. The NHANES contrasts here run from about 2,400 to 16,600 rows per matrix; the simulated
contrasts run from about 4,400 to 7,800 pooled and about 1,200 to 2,200 per model. So a raw
distance of 0.20 in NHANES and a raw distance of 0.20 in the corpus are not the same finding, and
comparing the two numbers directly would be an error of the kind this paper is about.

The two quantities that ARE comparable are internal to each source:

  floor multiple   the severity-matched distance divided by the binding split-half floor, that is,
                   the larger of the two cohorts' own p95 split-half distances. The floor is
                   estimated on the same rows at a comparable size, so it absorbs the sample-size
                   term. A multiple near or below 1 means the between-group distance is no larger
                   than the distance one gets by splitting a single cohort in half.

  null position    whether the matched distance clears the p95 of that source's own within-band
                   permutation null, with the permutation p. The null is built by permuting labels
                   within severity band on those same rows, so it too is sized to the source.

The raw distances and the group sizes are still emitted, because the reader needs to see that the
sample sizes differ; they are simply not the axis of comparison.

WHAT THIS SCRIPT COMPUTES

1. NHANES 2005-2018 adults (RIDAGEYR >= 18) with all eight PHQ-8 items present after recoding
   values above 3 to missing. Sex from RIAGENDR, race with RIDRETH3 == 6 as Asian and the RIDRETH1
   mapping otherwise, Hispanic pooling Mexican American with Other Hispanic, income terciles from
   INDFMPIR at the standard 1.3 and 3.5 poverty-income-ratio cuts, PHQ-8 total, the five severity
   bands the paper uses, and WTMEC2YR. Other/Multi race is not a contrast group; those rows still
   enter the sex and income contrasts, which is the population marginal the paper anchors to.

2. Seven population contrasts, the ones with a counterpart in the corpus design: Women vs Men,
   Black vs White, Asian vs White, Hispanic vs White, Low vs High income, Middle vs High income,
   Low vs Middle income. For each: the severity-matched Frobenius distance as a mean over 100
   matchings (script 06's matched_frob) and as the single matching script 17 scores, the
   within-band permutation null over 1,000 permutations with its p95 and permutation p, each
   group's 200-split split-half floor, the unmatched raw distance, a MEC-weighted sensitivity
   using weighted Pearson correlations on the two full groups, and Tucker's congruence between the
   two groups' one-factor loading vectors computed as in script 31.

3. The same seven contrasts on the simulated corpus, pooled and per model. The sex contrast is read
   in the cisgender frame, Cisgender Woman against Cisgender Man, since the population has no
   transgender counterpart; the race and income contrasts are computed on the full grid, which is
   the scope of the existing receipts and therefore the scope the gate pins.

GATE

The pooled simulated pass replays script 17's fourteen-pair pooled loop and script 06's fourteen-
cohort floor loop with the same seed and the same order of random draws, so the reproduction is
exact rather than approximate. Any drift beyond 0.01 against analysis/covariance_fair_null.csv or
analysis/covariance_noise_floors.csv stops the script, because it would mean the procedure applied
to NHANES is not the procedure the paper reports.

Emits analysis/covariance_population_reference.csv and analysis/level4_population_comparison.csv.
Seeded (SEED = 42 for the replayed receipts, SEED_NEW = 68 for everything new), idempotent, and it
writes nothing else.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

SEED = 42          # the seed the replayed receipts were produced under
SEED_NEW = 68      # everything this script computes fresh
N_PERM = 1000      # within-band permutation null
N_MATCH = 100      # severity-matched resamples averaged for the stable matched distance
N_SPLIT = 200      # split-half draws per cohort, as in script 06
TOL = 0.01         # gate tolerance against the existing receipts

ITEMS = [f"phq8_{i}" for i in range(1, 9)]
NH_ITEMS = [f"DPQ0{i}0" for i in range(1, 9)]
BANDS = [-1, 4, 9, 14, 19, 24]
CYCLES = list("DEFGHIJ")   # NHANES 2005-06 through 2017-18

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(BASE, "data", "nhanes_raw")


# ---- statistics, taken from scripts 06, 17 and 31 -------------------------------------------
def cmat(a):
    """Correlation matrix of an (n x 8) array, via numpy for speed (script 17)."""
    return np.corrcoef(a, rowvar=False)


def frob(a, b):
    return float(np.linalg.norm(a - b, ord="fro"))


def weighted_corr(X, w):
    """Weighted Pearson correlation matrix. Constant pooling divisors on the NHANES two-year
    weights cancel here as they do in the weighted means of script 03, so raw WTMEC2YR is used."""
    w = np.asarray(w, float)
    w = w / w.sum()
    mu = w @ X
    Z = X - mu
    cov = (Z * w[:, None]).T @ Z
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def fair_null_row(sub, dim, a, b, nperm, rng):
    """Script 17's fair_null_row, body unchanged; the generator is passed in rather than taken
    from module scope so the replay of the receipt and the new work can hold separate streams."""
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


def matched_frob(sa, sb, rng, n_match=N_MATCH):
    """Script 06's mean-conditioned check: within each severity band subsample both cohorts to the
    smaller band count, so the two matrices are computed on severity-matched distributions. The
    mean over n_match matchings is the stable estimate; a single matching is what script 17 scores."""
    Xa, Xb = sa[ITEMS].to_numpy(), sb[ITEMS].to_numpy()
    pa = {v: i for i, v in enumerate(sa.index.to_numpy())}
    pb = {v: i for i, v in enumerate(sb.index.to_numpy())}
    ba, bb = sa.band.to_numpy(), sb.band.to_numpy()
    ds = []
    for _ in range(n_match):
        ia, ib = [], []
        for band in range(5):
            xa = sa.index.to_numpy()[ba == band]; xb = sb.index.to_numpy()[bb == band]
            k = min(len(xa), len(xb))
            if k < 2: continue
            ia.append(rng.choice(xa, k, replace=False)); ib.append(rng.choice(xb, k, replace=False))
        gi = np.array([pa[v] for v in np.concatenate(ia)])
        gj = np.array([pb[v] for v in np.concatenate(ib)])
        ds.append(frob(cmat(Xa[gi]), cmat(Xb[gj])))
    return (float(np.mean(ds)), float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975)),
            int(len(gi)))


def split_half_floor(X, rng, n_split=N_SPLIT):
    """Script 06's per-cohort noise floor: repeated random halves of one cohort, Frobenius between
    the two half-matrices. Permuting positions consumes the same draws as permuting the label array
    script 06 permutes, and selects the same rows in the same order."""
    n = len(X)
    ds = []
    for _ in range(n_split):
        perm = rng.permutation(n)
        h = n // 2
        ds.append(frob(cmat(X[perm[:h]]), cmat(X[perm[h:]])))
    return float(np.mean(ds)), float(np.quantile(ds, 0.95))


def fit_one_factor(X):
    """Script 31's one-factor ML solution; returns the sign-oriented loading vector."""
    X = np.asarray(X, float)
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    fa = FactorAnalysis(n_components=1, random_state=0).fit(Z)
    lam = fa.components_[0]
    if lam.sum() < 0:
        lam = -lam
    return lam


def congruence(a, b):
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


# ---- corpora ---------------------------------------------------------------------------------
def load_simulated():
    m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"), low_memory=False)
    m["phq8_total"] = m["phq8_total_clipped"]
    m["ses_clean"] = m["ses_normalized"]
    c = m.dropna(subset=ITEMS).copy()
    c["band"] = pd.cut(c.phq8_total, BANDS, labels=False).astype(int)
    return c


def load_nhanes():
    frames = []
    for suf in CYCLES:
        demo = pd.read_sas(os.path.join(RAW, f"DEMO_{suf}.xpt"), format="xport")
        dpq = pd.read_sas(os.path.join(RAW, f"DPQ_{suf}.xpt"), format="xport")
        cols = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "WTMEC2YR"]
        if "RIDRETH3" in demo.columns:
            cols.append("RIDRETH3")
        d = demo[[c for c in cols if c in demo.columns]].merge(
            dpq[["SEQN"] + [c for c in NH_ITEMS if c in dpq.columns]], on="SEQN", how="left")
        if "RIDRETH3" not in d.columns:
            d["RIDRETH3"] = np.nan
        d["cycle"] = suf
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df[NH_ITEMS] = df[NH_ITEMS].where(df[NH_ITEMS] <= 3)          # 7 refused, 9 don't know
    a = df[(df.RIDAGEYR >= 18) & df[NH_ITEMS].notna().all(axis=1)].copy()

    def race_of(r):
        if pd.notna(r["RIDRETH3"]) and r["RIDRETH3"] == 6:
            return "Asian"
        return {3: "White", 4: "Black", 1: "MexAm", 2: "OtherHisp"}.get(r["RIDRETH1"], "Other/Multi")

    a["race"] = a.apply(race_of, axis=1).replace({"MexAm": "Hispanic", "OtherHisp": "Hispanic"})
    a["gender"] = a["RIAGENDR"].map({1: "Men", 2: "Women"})

    def ses_of(pir):
        if pd.isna(pir): return None
        if pir < 1.3: return "Low"
        if pir <= 3.5: return "Middle"
        return "High"

    a["ses_clean"] = a["INDFMPIR"].apply(ses_of)
    a = a.rename(columns={old: new for old, new in zip(NH_ITEMS, ITEMS)})
    a["phq8_total"] = a[ITEMS].sum(axis=1)
    a["band"] = pd.cut(a.phq8_total, BANDS, labels=False).astype(int)
    return a[ITEMS + ["phq8_total", "band", "gender", "race", "ses_clean", "WTMEC2YR",
                      "cycle"]].reset_index(drop=True)


# ---- gate: replay the two existing receipts exactly -------------------------------------------
PAIRS_17 = [("race", "Black", "White"), ("race", "Asian", "White"), ("race", "Hispanic", "White"),
            ("race", "Multiracial", "White"),
            ("gender", "Cisgender Man", "Cisgender Woman"),
            ("gender", "Cisgender Man", "Transgender Woman"),
            ("gender", "Cisgender Man", "Transgender Man"),
            ("gender", "Cisgender Woman", "Transgender Woman"),
            ("gender", "Cisgender Woman", "Transgender Man"),
            ("gender", "Transgender Woman", "Transgender Man"),
            ("ses", "Low", "High"), ("ses", "Middle", "High"), ("ses", "Low", "Middle"),
            ("relationship", "Single", "Married")]

COHORTS_06 = ([("gender", g) for g in
               ["Cisgender Man", "Cisgender Woman", "Transgender Woman", "Transgender Man"]]
              + [("race", r) for r in ["White", "Black", "Asian", "Hispanic", "Multiracial"]]
              + [("ses_clean", s) for s in ["Low", "Middle", "High"]]
              + [("relationship", r) for r in ["Single", "Married"]])


def replay_receipts(c):
    """Reproduce covariance_fair_null.csv and covariance_noise_floors.csv from this corpus."""
    rng = np.random.default_rng(SEED)
    fair = pd.DataFrame([fair_null_row(c, d, a, b, N_PERM, rng) for d, a, b in PAIRS_17])

    rng6 = np.random.default_rng(SEED)
    rows = []
    for dim, grp in COHORTS_06:
        X = c.loc[c[dim] == grp, ITEMS].to_numpy()
        fm, f95 = split_half_floor(X, rng6)
        rows.append(dict(dimension=dim.replace("_clean", ""), group=grp, n=len(X),
                         floor_mean=round(fm, 3), floor_p95=round(f95, 3)))
    return fair, pd.DataFrame(rows)


def run_gate(fair, floors):
    ref_f = pd.read_csv(os.path.join(OUT, "covariance_fair_null.csv"))
    ref_n = pd.read_csv(os.path.join(OUT, "covariance_noise_floors.csv"))
    key = ["dimension", "group_a", "group_b"]
    a = fair.set_index(key).matched_frobenius
    b = ref_f.set_index(key).matched_frobenius
    d_fair = (a - b).abs().sort_values(ascending=False)
    k2 = ["dimension", "group"]
    d_mean = (floors.set_index(k2).floor_mean - ref_n.set_index(k2).floor_mean).abs()
    d_p95 = (floors.set_index(k2).floor_p95 - ref_n.set_index(k2).floor_p95).abs()
    d_floor = pd.concat([d_mean, d_p95], axis=1).max(axis=1).sort_values(ascending=False)
    print("GATE against the existing receipts")
    print(f"  covariance_fair_null.csv    max |delta matched_frobenius| over 14 pairs   "
          f"{d_fair.max():.6f}  (worst: {d_fair.index[0]})")
    print(f"  covariance_noise_floors.csv max |delta floor_mean / floor_p95| over 14     "
          f"{d_floor.max():.6f}  (worst: {d_floor.index[0]})")
    ok = bool(d_fair.max() <= TOL and d_floor.max() <= TOL)
    print(f"  gate tolerance {TOL}: {'PASS' if ok else 'FAIL'}\n")
    if not ok:
        sys.exit("GATE FAILED: the reimplemented procedure does not reproduce the receipts.")
    return ok


# ---- contrasts --------------------------------------------------------------------------------
# label, dimension, population group a, population group b, simulated group a, simulated group b
CONTRASTS = [
    ("Women vs Men",           "gender", "Women",  "Men",   "Cisgender Woman", "Cisgender Man"),
    ("Black vs White",         "race",   "Black",  "White", "Black",           "White"),
    ("Asian vs White",         "race",   "Asian",  "White", "Asian",           "White"),
    ("Hispanic vs White",      "race",   "Hispanic", "White", "Hispanic",      "White"),
    ("Low vs High income",     "ses",    "Low",    "High",  "Low",             "High"),
    ("Middle vs High income",  "ses",    "Middle", "High",  "Middle",          "High"),
    ("Low vs Middle income",   "ses",    "Low",    "Middle", "Low",            "Middle"),
]


def contrast_stats(frame, dim, a, b, rng, nperm=N_PERM, weights=None, floors_cache=None):
    """Every level-4 statistic for one contrast on one frame."""
    col = "ses_clean" if dim == "ses" else dim
    sa, sb = frame[frame[col] == a], frame[frame[col] == b]
    Xa, Xb = sa[ITEMS].to_numpy(), sb[ITEMS].to_numpy()

    single = fair_null_row(frame, dim, a, b, nperm, rng)
    m_mean, m_lo, m_hi, n_matched = matched_frob(sa, sb, rng)

    def floor_of(name, X):
        if floors_cache is not None and name in floors_cache:
            return floors_cache[name]
        v = split_half_floor(X, rng)
        if floors_cache is not None:
            floors_cache[name] = v
        return v

    fa_mean, fa_p95 = floor_of(a, Xa)
    fb_mean, fb_p95 = floor_of(b, Xb)
    binding = max(fa_p95, fb_p95)

    out = dict(dimension=dim, group_a=a, group_b=b, n_a=len(sa), n_b=len(sb),
               n_matched_per_matrix=single["n_matched_per_matrix"],
               matched_frobenius_mean100=round(m_mean, 4),
               matched_ci_lo=round(m_lo, 4), matched_ci_hi=round(m_hi, 4),
               matched_frobenius_single=single["matched_frobenius"],
               fair_null_mean=single["fair_null_mean"],
               fair_null_p95=single["fair_null_p95"],
               p_perm=single["p_perm"], above_fair_null=single["above_fair_null"],
               floor_mean_a=round(fa_mean, 4), floor_p95_a=round(fa_p95, 4),
               floor_mean_b=round(fb_mean, 4), floor_p95_b=round(fb_p95, 4),
               binding_floor_p95=round(binding, 4),
               floor_multiple=round(m_mean / binding, 3),
               above_floor=bool(m_mean > binding),
               frobenius_raw=round(frob(cmat(Xa), cmat(Xb)), 4),
               tucker_phi=round(congruence(fit_one_factor(Xa), fit_one_factor(Xb)), 4))
    if weights is not None:
        wa = sa[weights].to_numpy(float); wb = sb[weights].to_numpy(float)
        out["frobenius_weighted"] = round(frob(weighted_corr(Xa, wa), weighted_corr(Xb, wb)), 4)
    return out


def main():
    print("=" * 100)
    print("LEVEL 4 POPULATION REFERENCE")
    print("=" * 100 + "\n")

    sim = load_simulated()
    print(f"simulated corpus: {len(sim)} complete-item rows, "
          f"{sim.model.nunique()} models, conditions {sorted(sim.prompt_condition.unique())}\n")

    fair, floors = replay_receipts(sim)
    run_gate(fair, floors)

    # --- population -----------------------------------------------------------------------
    nh = load_nhanes()
    print("NHANES 2005-2018 adults 18+ with all eight PHQ-8 items present: "
          f"{len(nh)} respondents")
    for lab, col in (("sex", "gender"), ("race", "race"), ("income", "ses_clean")):
        counts = nh[col].value_counts(dropna=False).to_dict()
        print(f"  {lab:7s} " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print("  severity bands " + ", ".join(
        f"{lo}-{hi}: {int((nh.band == i).sum())}"
        for i, (lo, hi) in enumerate([(0, 4), (5, 9), (10, 14), (15, 19), (20, 24)])))
    print()

    rng_pop = np.random.default_rng(SEED_NEW)
    pop_floors = {}
    pop_rows = []
    for label, dim, pa, pb, _, _ in CONTRASTS:
        r = contrast_stats(nh, dim, pa, pb, rng_pop, weights="WTMEC2YR", floors_cache=pop_floors)
        r["contrast"] = label
        pop_rows.append(r)
    pop = pd.DataFrame(pop_rows)
    pop = pop[["contrast", "dimension", "group_a", "group_b", "n_a", "n_b",
               "n_matched_per_matrix", "matched_frobenius_mean100", "matched_ci_lo",
               "matched_ci_hi", "matched_frobenius_single", "fair_null_mean", "fair_null_p95",
               "p_perm", "above_fair_null", "floor_mean_a", "floor_p95_a", "floor_mean_b",
               "floor_p95_b", "binding_floor_p95", "floor_multiple", "above_floor",
               "frobenius_raw", "frobenius_weighted", "tucker_phi"]]
    pop.to_csv(os.path.join(OUT, "covariance_population_reference.csv"), index=False)

    pd.set_option("display.width", 240)
    print("POPULATION REFERENCE (NHANES, script 17 procedure unchanged)")
    print("-" * 100)
    print(pop.drop(columns=["floor_mean_a", "floor_mean_b", "matched_ci_lo",
                            "matched_ci_hi"]).to_string(index=False))
    print()

    # --- simulated, pooled and per model --------------------------------------------------
    rng_sim = np.random.default_rng(SEED_NEW)
    comp = []

    def emit(scope, label, r):
        comp.append(dict(contrast=label, scope=scope,
                         n_per_matrix=r["n_matched_per_matrix"],
                         matched_distance=r["matched_frobenius_mean100"],
                         matched_distance_single_matching=r["matched_frobenius_single"],
                         null_p95=r["fair_null_p95"], p_perm=r["p_perm"],
                         binding_floor_p95=r["binding_floor_p95"],
                         floor_multiple=r["floor_multiple"],
                         above_null=r["above_fair_null"], above_floor=r["above_floor"],
                         tucker_phi=r["tucker_phi"]))

    for r in pop_rows:
        emit("population (NHANES)", r["contrast"], r)

    # pooled: take the single-matching statistic and its null from the replayed receipt run, so the
    # emitted numbers are the published ones; everything else is computed here.
    ref = fair.set_index(["dimension", "group_a", "group_b"])
    # the pooled floors are the replayed receipt values, so the pooled row carries the published
    # floor rather than a second estimate of it
    pooled_floors = {r.group: (r.floor_mean, r.floor_p95) for r in floors.itertuples()}
    for label, dim, _, _, sa_, sb_ in CONTRASTS:
        r = contrast_stats(sim, dim, sa_, sb_, rng_sim, floors_cache=pooled_floors)
        k = (dim, sa_, sb_) if (dim, sa_, sb_) in ref.index else (dim, sb_, sa_)
        pub = ref.loc[k]
        r.update(n_matched_per_matrix=int(pub.n_matched_per_matrix),
                 matched_frobenius_single=float(pub.matched_frobenius),
                 fair_null_mean=float(pub.fair_null_mean),
                 fair_null_p95=float(pub.fair_null_p95), p_perm=float(pub.p_perm),
                 above_fair_null=bool(pub.above_fair_null))
        r["contrast"] = label
        emit("simulated pooled", label, r)

    for mdl, g in sim.groupby("model"):
        g = g.copy()
        mf = {}
        for label, dim, _, _, sa_, sb_ in CONTRASTS:
            r = contrast_stats(g, dim, sa_, sb_, rng_sim, floors_cache=mf)
            r["contrast"] = label
            emit(mdl, label, r)

    cmp_df = pd.DataFrame(comp)
    order = {lab: i for i, (lab, *_) in enumerate(CONTRASTS)}
    scope_order = ["population (NHANES)", "simulated pooled"] + sorted(sim.model.unique())
    cmp_df["_c"] = cmp_df.contrast.map(order)
    cmp_df["_s"] = cmp_df.scope.map({s: i for i, s in enumerate(scope_order)})
    cmp_df = cmp_df.sort_values(["_c", "_s"]).drop(columns=["_c", "_s"]).reset_index(drop=True)
    cmp_df.to_csv(os.path.join(OUT, "level4_population_comparison.csv"), index=False)

    print("LEVEL 4 COMPARISON, population against simulation")
    print("-" * 100)
    print("floor_multiple = severity-matched distance / binding split-half floor p95, each source")
    print("on its own floor. above_null is the matched distance against that source's own")
    print("within-band permutation null. Raw distances are not comparable across sources.")
    print("-" * 100)
    print(cmp_df.to_string(index=False))
    print()

    print("SIDE BY SIDE on the two comparable quantities")
    print("-" * 100)
    hdr = (f"{'contrast':<24}{'pop mult':>9}{'pop p':>9}{'pop phi':>9}   "
           f"{'sim mult':>9}{'sim p':>9}{'sim phi':>9}   {'per-model mult range':>22}")
    print(hdr)
    for label, *_ in CONTRASTS:
        p = cmp_df[(cmp_df.contrast == label) & (cmp_df.scope == "population (NHANES)")].iloc[0]
        s = cmp_df[(cmp_df.contrast == label) & (cmp_df.scope == "simulated pooled")].iloc[0]
        pm = cmp_df[(cmp_df.contrast == label) & (~cmp_df.scope.isin(
            ["population (NHANES)", "simulated pooled"]))].floor_multiple
        print(f"{label:<24}{p.floor_multiple:>9.2f}{p.p_perm:>9.4f}{p.tucker_phi:>9.4f}   "
              f"{s.floor_multiple:>9.2f}{s.p_perm:>9.4f}{s.tucker_phi:>9.4f}   "
              f"{pm.min():>10.2f} to {pm.max():<9.2f}")
    print()
    print(f"population contrasts above their own floor: "
          f"{int(pop.above_floor.sum())} of {len(pop)}; above their own null: "
          f"{int(pop.above_fair_null.sum())} of {len(pop)}")
    sp = cmp_df[cmp_df.scope == "simulated pooled"]
    print(f"simulated pooled above its own floor: {int(sp.above_floor.sum())} of {len(sp)}; "
          f"above its own null: {int(sp.above_null.sum())} of {len(sp)}")
    print("\nwritten -> analysis/covariance_population_reference.csv")
    print("written -> analysis/level4_population_comparison.csv")


if __name__ == "__main__":
    main()
