"""Level 2 subgroup fidelity, re-stated as an equivalence claim and disaggregated by model.

Two reviewer objections are answered here, on data already collected. No new generations, no edits
to any other file.

  (a) A "kept" gap in 44/64 is a failure to separate the simulated gap from the population value.
      Failure to reject is not evidence of agreement. Every contrast is therefore also given a
      formal equivalence test against the population value, at a bound equal to the population gap
      itself, plus delta_min_pop: the tightest bound at which that equivalence still holds. The
      symmetric statement, equivalence to zero at the same bound, is what distinguishes a gap that
      is absent from one that is merely unresolved; 50 supplies exactly that construction for the
      racial contrasts and it is reused unchanged.

  (b) The verdicts pool four models, so a panel can pass an axis that no single model earns. The
      sex axis is the case in point: per-model gaps of 0.24, 0.99, 0.32 and 1.81 against a
      population gap of 0.94. Every quantity is therefore recomputed on the same matched pairs
      within each model, within each prompt condition, and within each model x condition cell.

Pairing is taken from 44/50/64 without modification. Cisgender frame, one value per design cell,
and a stratum that holds every factor but the contrasted one fixed, so the two cells of a pair
differ in one factor alone: 96 pairs per racial contrast, 240 for sex, 160 for each socioeconomic
contrast. Inputs are the v2 release columns (prompt_condition, ses_normalized, phq8_total_clipped).
The gate below is what proves the pairing still reproduces the published numbers.

Anchor error enters the test against the population value as the root sum of squares of the two
group design standard errors, se_gap = hypot(se, se_pop), which is 64's construction. It enters the
equivalence test against the population value the same way. The equivalence test against zero uses
the paired standard error alone, because no anchor is involved in the null of no gap.

Verdicts, per row:

  missing       equivalent to zero at the bound, and separable from the population value
  kept          sign matches, equivalent to the population value, not separable from it
  steepened     separable from the population value, sign kept, ratio > 1
  flattened     separable from the population value, sign kept, ratio < 1
  reversed      separable from the population value, sign differs
  undetermined  neither equivalence nor separation established

Benjamini-Hochberg runs across the seven contrasts within each test, on the pooled rows only; the
per-model and per-condition rows carry raw p-values, since they are a disaggregation of the pooled
claim rather than seven further independent claims. Both p and q are reported either way.

Emits analysis/level2_permodel_equivalence.csv. Idempotent. Nothing here is stochastic; the seed
below exists so that stays true if a resampling estimate is ever added.
"""
import ast
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "analysis")
CIS = ["Cisgender Man", "Cisgender Woman"]
ALPHA = 0.05
SEED = 20260909
RNG = np.random.default_rng(SEED)  # deliberately unused: every estimate below is closed form

# ------------------------------------------------------------------------------------------ data
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"), low_memory=False)
m = m[m.gender.isin(CIS)].copy()
m["phq8_total"] = m["phq8_total_clipped"]
m["ses"] = m["ses_normalized"]
m["condition"] = m["prompt_condition"]
if m.condition.isna().any():
    raise SystemExit("unlabelled rows in prompt_condition")

cell = (m.groupby(["model", "profile_id", "condition", "race", "gender", "ses", "relationship"],
                  as_index=False).phq8_total.mean())

gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
# the ground-truth table labels the pooled Hispanic row "Hispanic (pooled)"; the design-SE table
# labels it "Hispanic". Both keys are spelled out rather than resolved by fallback.
GT_KEY = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic (pooled)",
          "Men": "Men", "Women": "Women", "Low": "Low", "Middle": "Middle", "High": "High"}
SE_KEY = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic",
          "Men": "Cisgender men", "Women": "Cisgender women",
          "Low": "Low", "Middle": "Middle", "High": "High"}
gtm = {g: float(gt.loc[gt.group == k, "w_mean"].iloc[0]) for g, k in GT_KEY.items()}
dse = pd.read_csv(os.path.join(OUT, "anchor_design_se.csv")).set_index("group")
gtse = {g: float(dse.loc[k, "se_design"]) for g, k in SE_KEY.items()}

RACE_STRATUM = ["model", "condition", "gender", "ses", "relationship"]
SES_STRATUM = ["model", "condition", "race", "gender", "relationship"]
SEX_STRATUM = ["model", "condition", "race", "ses", "relationship"]

# contrast name, factor column, contrast level, reference level, pairing stratum, anchor keys
SPECS = [
    ("Black minus White", "race", "Black", "White", RACE_STRATUM, "Black", "White"),
    ("Hispanic minus White", "race", "Hispanic", "White", RACE_STRATUM, "Hispanic", "White"),
    ("Asian minus White", "race", "Asian", "White", RACE_STRATUM, "Asian", "White"),
    ("Women minus Men", "gender", "Cisgender Woman", "Cisgender Man", SEX_STRATUM, "Women", "Men"),
    ("Low minus High SES", "ses", "Low", "High", SES_STRATUM, "Low", "High"),
    ("Middle minus High SES", "ses", "Middle", "High", SES_STRATUM, "Middle", "High"),
    ("Low minus Middle SES", "ses", "Low", "Middle", SES_STRATUM, "Low", "Middle"),
]
CONTRASTS = [s[0] for s in SPECS]
MODELS = sorted(cell.model.unique())
CONDITIONS = sorted(cell.condition.unique())


def paired(col, hi, lo, stratum):
    """One row per matched pair of design cells differing in the contrasted factor alone."""
    a = cell[cell[col] == hi].set_index(stratum).phq8_total
    b = cell[cell[col] == lo].set_index(stratum).phq8_total
    # Duplicate stratum labels would make this subtraction broadcast instead of pair, silently.
    for tag, idx in ((hi, a.index), (lo, b.index)):
        if idx.has_duplicates:
            raise SystemExit(f"{col} {hi}-{lo}: {tag} stratum key is not unique, pairing is wrong")
    common = a.index.intersection(b.index)
    return (a.loc[common] - b.loc[common]).to_frame("diff").reset_index()


# ------------------------------------------------------------------------------------ statistics
def tost(centre, se, df, bound):
    """Two one-sided tests that `centre` lies inside (-bound, +bound). Returns the larger p."""
    t_lo = (centre + bound) / se            # H0a: mu <= -bound
    t_hi = (centre - bound) / se            # H0b: mu >= +bound
    p_lo = float(stats.t.sf(t_lo, df))
    p_hi = float(stats.t.cdf(t_hi, df))
    return t_lo, p_lo, t_hi, p_hi, max(p_lo, p_hi)


def delta_min(centre, se, df):
    """Tightest symmetric bound at which the TOST still rejects: the 1-2a interval's far edge."""
    return abs(centre) + stats.t.ppf(1 - ALPHA, df) * se


def bh(p):
    """Benjamini-Hochberg, in the form 44 and 64 use within their own families."""
    p = pd.Series(p, dtype=float)
    order = p.rank(method="first")
    return (p * len(p) / order).clip(upper=1.0)


def verdict(sign_ok, ratio, p_sep, p_eq_pop, p_eq_zero):
    """The rule the paper states, applied in the order the paper states it."""
    if any(pd.isna(v) for v in (p_sep, p_eq_pop, p_eq_zero)):
        return "undetermined"
    separable = p_sep < ALPHA
    if p_eq_zero < ALPHA and separable:
        return "missing"
    if sign_ok and p_eq_pop < ALPHA and not separable:
        return "kept"
    if separable and sign_ok and ratio > 1:
        return "steepened"
    if separable and sign_ok and ratio < 1:
        return "flattened"
    if separable and not sign_ok:
        return "reversed"
    return "undetermined"


def estimate(name, x, pop, se_pop, scope, scope_type, model, condition):
    """Every quantity for one contrast within one scope, on the pairs handed in."""
    n = len(x)
    df = n - 1
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    row = dict(contrast=name, scope=scope, scope_type=scope_type, model=model,
               condition=condition, n_pairs=n, df=df,
               simulated=round(mean, 4), sd_diff=None if pd.isna(sd) else round(sd, 4),
               population=round(pop, 4), pop_se=round(se_pop, 4),
               # full precision, for the gate only: 44 and 64 print per-model means at 3 dp and one
               # of them (deepseek Low-High, 2.4475000000000002) sits on the rounding boundary, so
               # a 4-dp column cannot be compared to a 3-dp receipt by tolerance.
               mean_exact=mean)
    # A degenerate scope (one pair, or zero variance within a scope) would divide by zero and
    # report an infinite t rather than an absent one. Emit the means and stop there.
    if not (n > 1 and np.isfinite(se) and se > 0):
        row.update({k: np.nan for k in (
            "se", "se_gap", "ci95_lo", "ci95_hi", "ci95z_lo", "ci95z_hi", "ci90_lo", "ci90_hi",
            "t_vs_zero", "p_vs_zero", "t_vs_pop", "p_vs_pop", "bound",
            "t_lower_pop", "p_lower_pop", "t_upper_pop", "p_upper_pop", "p_tost_pop",
            "delta_min_pop", "t_lower_zero", "p_lower_zero", "t_upper_zero", "p_upper_zero",
            "p_tost_zero", "delta_min_zero", "ratio")})
        row.update(gap=round(mean - pop, 4), equivalent_pop=False, equivalent_zero=False,
                   sign_matches_population=bool(np.sign(mean) == np.sign(pop)),
                   verdict="undetermined", verdict_basis="degenerate scope")
        return row

    se_gap = float(np.hypot(se, se_pop))
    t95 = float(stats.t.ppf(1 - ALPHA / 2, df))
    t90 = float(stats.t.ppf(1 - ALPHA, df))

    t0 = mean / se
    p_zero = float(stats.t.sf(abs(t0), df) * 2)
    diff = mean - pop
    t1 = diff / se_gap
    p_pop = float(stats.t.sf(abs(t1), df) * 2)

    bound = abs(pop)
    tl_p, pl_p, tu_p, pu_p, p_eq_pop = tost(diff, se_gap, df, bound)
    tl_z, pl_z, tu_z, pu_z, p_eq_zero = tost(mean, se, df, bound)

    row.update(
        se=round(se, 4), se_gap=round(se_gap, 4),
        # 44 and 64 print the interval with the normal multiplier; both are carried so the
        # published receipt values are readable straight off this file.
        ci95z_lo=round(mean - 1.96 * se, 4), ci95z_hi=round(mean + 1.96 * se, 4),
        ci95_lo=round(mean - t95 * se, 4), ci95_hi=round(mean + t95 * se, 4),
        ci90_lo=round(mean - t90 * se, 4), ci90_hi=round(mean + t90 * se, 4),
        t_vs_zero=round(t0, 3), p_vs_zero=p_zero,
        gap=round(diff, 4), t_vs_pop=round(t1, 3), p_vs_pop=p_pop,
        bound=round(bound, 4),
        t_lower_pop=round(tl_p, 3), p_lower_pop=pl_p,
        t_upper_pop=round(tu_p, 3), p_upper_pop=pu_p,
        p_tost_pop=p_eq_pop, equivalent_pop=bool(p_eq_pop < ALPHA),
        delta_min_pop=round(delta_min(diff, se_gap, df), 4),
        t_lower_zero=round(tl_z, 3), p_lower_zero=pl_z,
        t_upper_zero=round(tu_z, 3), p_upper_zero=pu_z,
        p_tost_zero=p_eq_zero, equivalent_zero=bool(p_eq_zero < ALPHA),
        delta_min_zero=round(delta_min(mean, se, df), 4),
        sign_matches_population=bool(np.sign(mean) == np.sign(pop)),
        ratio=round(mean / pop, 4))
    return row


rows = []
pairs = {}
for name, col, hi, lo, stratum, ghi, glo in SPECS:
    d = paired(col, hi, lo, stratum)
    pairs[name] = d
    pop = gtm[ghi] - gtm[glo]
    se_pop = float(np.hypot(gtse[ghi], gtse[glo]))
    scopes = [("pooled", "pooled", "", "", d)]
    scopes += [(k, "model", k, "", g) for k, g in d.groupby("model")]
    scopes += [(k, "condition", "", k, g) for k, g in d.groupby("condition")]
    scopes += [(f"{k[0]} | {k[1]}", "model_x_condition", k[0], k[1], g)
               for k, g in d.groupby(["model", "condition"])]
    for scope, stype, mdl, cond, sub in scopes:
        rows.append(estimate(name, sub["diff"].to_numpy(float), pop, se_pop,
                             scope, stype, mdl, cond))

res = pd.DataFrame(rows)

# ---------------------------------------------------------------- multiplicity, pooled rows only
FAMILIES = [("p_vs_zero", "q_vs_zero"), ("p_vs_pop", "q_vs_pop"),
            ("p_tost_pop", "q_tost_pop"), ("p_tost_zero", "q_tost_zero")]
pool_mask = res.scope_type == "pooled"
for pcol, qcol in FAMILIES:
    res[qcol] = np.nan
    res.loc[pool_mask, qcol] = bh(res.loc[pool_mask, pcol]).round(9).to_numpy()

# The pooled claim is one of seven, so its verdict is taken on the corrected p-value; the
# disaggregations are read as a decomposition of that claim and keep the raw one. verdict_raw_p is
# carried on every row so the two can be compared rather than assumed identical.
res["verdict"] = [
    verdict(r.sign_matches_population, r.ratio,
            r.q_vs_pop if r.scope_type == "pooled" else r.p_vs_pop,
            r.q_tost_pop if r.scope_type == "pooled" else r.p_tost_pop,
            r.q_tost_zero if r.scope_type == "pooled" else r.p_tost_zero)
    for r in res.itertuples()]
res["verdict_raw_p"] = [
    verdict(r.sign_matches_population, r.ratio, r.p_vs_pop, r.p_tost_pop, r.p_tost_zero)
    for r in res.itertuples()]
res["verdict_basis"] = np.where(pool_mask, "BH q across 7 contrasts", "raw p")
res.loc[res.se.isna(), "verdict_basis"] = "degenerate scope"

# ------------------------------------------------------------------------------------------ gate
# Nothing is written until the pairing is shown to reproduce the published receipts exactly.
fails = []


def check(label, got, want, tol):
    ok = abs(float(got) - float(want)) <= tol
    if not ok:
        fails.append((label, float(got), float(want)))
    return ok, f"  [{'ok  ' if ok else 'FAIL'}] {label:56s} got {float(got):+.10g}  want {float(want):+.10g}"


print("GATE 1: pooled rows reproduce the published level-2 receipts")
print("=" * 108)
lines = []
for fn in ("race_contrast_tests.csv", "level2_sex_ses_contrasts.csv"):
    ref = pd.read_csv(os.path.join(OUT, fn))
    for _, r in ref.iterrows():
        g = res[(res.contrast == r.contrast) & (res.scope == "pooled")].iloc[0]
        for c, tol in (("n_pairs", 0), ("simulated", 5e-5), ("ci95z_lo", 5e-5),
                       ("ci95z_hi", 5e-5), ("population", 5e-5), ("gap", 5e-5)):
            src = "ci_lo" if c == "ci95z_lo" else ("ci_hi" if c == "ci95z_hi" else c)
            lines.append(check(f"{fn[:5]} {r.contrast} {src}", g[c], r[src], tol)[1])
        lines.append(check(f"{fn[:5]} {r.contrast} p_vs_pop", g.p_vs_pop, r.p_vs_pop,
                           max(1e-12, abs(r.p_vs_pop) * 1e-6))[1])
        for k, v in ast.literal_eval(r.per_model).items():
            pm = res[(res.contrast == r.contrast) & (res.scope == k)].iloc[0]
            # Compared at the receipt's own precision, which is where it was rounded, and with
            # Python's round on a cast float, which is what 44 and 64 used. numpy's round is
            # banker's rounding and disagrees on exact halves: two per-model means land on one
            # (deepseek Low-High 2.4475, gpt-4o-mini Middle-High 1.3325).
            lines.append(check(f"{fn[:5]} {r.contrast} per_model[{k}]",
                               round(float(pm["mean_exact"]), 3), v, 0)[1])
print("\n".join(lines))
print(f"  -> {len(lines)} checks, {len(fails)} failed")

print("\nGATE 2: pooled equivalence-to-zero at the population bound matches 50")
print("=" * 108)
eq = pd.read_csv(os.path.join(OUT, "race_equivalence_tost.csv"))
eq = eq[(eq.scope == "pooled") & (eq.bound_type == "population disparity")]
n2 = 0
for _, r in eq.iterrows():
    g = res[(res.contrast == r.contrast) & (res.scope == "pooled")].iloc[0]
    for lab, got, want, tol in (("bound", g.bound, r.bound, 5e-5),
                                ("p_tost_zero", g.p_tost_zero, r.p_tost, abs(r.p_tost) * 1e-6),
                                ("delta_min_zero", g.delta_min_zero, r.delta_min_equiv, 5e-5),
                                ("ci90_lo", g.ci90_lo, r.ci90_lo, 5e-5),
                                ("ci90_hi", g.ci90_hi, r.ci90_hi, 5e-5)):
        print(check(f"50 {r.contrast} {lab}", got, want, tol)[1])
        n2 += 1
missing = [c for c in CONTRASTS[:3] if c not in set(eq.contrast)]
if missing:
    print(f"  note: 50 covers Black and Hispanic only; no receipt row for {', '.join(missing)}")
# 64 carries the same construction for the four sex and socioeconomic contrasts, under the name
# p_tost_at_population, rounded to 6 dp.
for _, r in pd.read_csv(os.path.join(OUT, "level2_sex_ses_contrasts.csv")).iterrows():
    g = res[(res.contrast == r.contrast) & (res.scope == "pooled")].iloc[0]
    print(check(f"64 {r.contrast} p_tost_zero", round(float(g.p_tost_zero), 6),
                r.p_tost_at_population, 0)[1])
    n2 += 1
print(f"  -> {n2} checks")

if fails:
    print("\nGATE FAIL. Data handling does not match the published analysis. Nothing written.")
    for label, got, want in fails:
        print(f"  {label}: got {got:+.10g}, want {want:+.10g}")
    raise SystemExit(1)
print("\nGATE PASS")

# ----------------------------------------------------------------------------------------- write
os.makedirs(OUT, exist_ok=True)
COLS = ["contrast", "scope", "scope_type", "model", "condition", "n_pairs", "df",
        "simulated", "sd_diff", "se", "ci95_lo", "ci95_hi", "ci95z_lo", "ci95z_hi",
        "ci90_lo", "ci90_hi", "t_vs_zero", "p_vs_zero", "q_vs_zero",
        "population", "pop_se", "se_gap", "gap", "t_vs_pop", "p_vs_pop", "q_vs_pop",
        "bound", "t_lower_pop", "p_lower_pop", "t_upper_pop", "p_upper_pop",
        "p_tost_pop", "q_tost_pop", "equivalent_pop", "delta_min_pop",
        "t_lower_zero", "p_lower_zero", "t_upper_zero", "p_upper_zero",
        "p_tost_zero", "q_tost_zero", "equivalent_zero", "delta_min_zero",
        "sign_matches_population", "ratio", "verdict", "verdict_raw_p", "verdict_basis"]
res = res[COLS].sort_values(
    ["contrast", "scope_type", "scope"],
    key=lambda s: s.map({c: i for i, c in enumerate(CONTRASTS)}).fillna(
        s.map({"pooled": 0, "model": 1, "condition": 2, "model_x_condition": 3})) if s.name in
    ("contrast", "scope_type") else s).reset_index(drop=True)
path = os.path.join(OUT, "level2_permodel_equivalence.csv")
res.to_csv(path, index=False)

# --------------------------------------------------------------------------------------- summary
pd.set_option("display.width", 240)
short = {m: m.split("/")[-1] for m in MODELS}


def get(c, s):
    return res[(res.contrast == c) & (res.scope == s)].iloc[0]


print("\n\nLEVEL 2, POOLED: separation, equivalence, and the tightest defensible bound")
print("=" * 108)
p = res[res.scope_type == "pooled"].copy()
show = p[["contrast", "n_pairs", "simulated", "ci95_lo", "ci95_hi", "population", "gap",
          "ratio", "q_vs_pop", "p_tost_pop", "delta_min_pop", "p_tost_zero", "delta_min_zero",
          "verdict"]].copy()
for c in ("q_vs_pop", "p_tost_pop", "p_tost_zero"):
    show[c] = show[c].map(lambda v: f"{v:.3g}")
print(show.to_string(index=False))

print("\n\nVERDICT BY SCOPE (pooled, then the four models, then the two prompt conditions)")
print("=" * 108)
W = 21
hdr = (f"{'contrast':<22} {'pooled':<13} "
       + " ".join(f"{short[m]:<{W}}" for m in MODELS)
       + " " + " ".join(f"{c:<{W}}" for c in CONDITIONS))
print(hdr)
print("-" * len(hdr))
for c in CONTRASTS:
    cells = [f"{get(c, 'pooled').verdict:<13}"]
    for scope in list(MODELS) + list(CONDITIONS):
        r = get(c, scope)
        cells.append(f"{r.simulated:+.3f} {r.verdict}".ljust(W))
    print(f"{c:<22} " + " ".join(cells))

print("\n\nPER-MODEL GAPS AGAINST THE POPULATION VALUE")
print("=" * 108)
for c in CONTRASTS:
    r0 = get(c, "pooled")
    pm = {short[m]: round(float(get(c, m).simulated), 3) for m in MODELS}
    agree = sum(get(c, m).verdict == r0.verdict for m in MODELS)
    print(f"  {c:<22} population {r0.population:+.3f}  pooled {r0.simulated:+.3f} ({r0.verdict})")
    print(f"  {'':<22} per model  {pm}")
    print(f"  {'':<22} models sharing the pooled verdict: {agree}/4")

print("\n\nWHERE THE PANEL DISAGREES WITH ITSELF")
print("=" * 108)
d = res[(res.scope_type.isin(["model", "condition"]))].copy()
for c in CONTRASTS:
    v = d[d.contrast == c]
    kinds = sorted(set(v.verdict))
    if len(kinds) > 1 or kinds != [get(c, "pooled").verdict]:
        detail = ", ".join(f"{r.scope.split('/')[-1]}={r.verdict}" for r in v.itertuples())
        print(f"  {c:<22} pooled={get(c, 'pooled').verdict};  {detail}")

flip = res[res.verdict != res.verdict_raw_p]
print(f"\n  rows where the BH-corrected verdict differs from the raw-p verdict: {len(flip)}")
for r in flip.itertuples():
    print(f"    {r.contrast} / {r.scope}: {r.verdict_raw_p} -> {r.verdict}")

print("\nWritten -> %s  (%d rows, %d contrasts x %d scopes)"
      % (path, len(res), res.contrast.nunique(), res.scope.nunique()))
