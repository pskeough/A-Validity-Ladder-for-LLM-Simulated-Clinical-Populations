"""Put a formal equivalence bound on the racial null of Section 4.2.

Reviewer objection: the paper says the population Black-White and Hispanic-White disparities "do
not survive into the simulation" and that a study run on the simulated cohort "would find the
groups equivalent and report a null". The support offered is an interval covering zero plus a
separate test against the population value. A confidence interval covering zero is a failure to
reject, not evidence of absence, and for a politically consequential claim the paper carries no
equivalence test, no smallest effect of interest, and no power statement. This script supplies all
three, on data already collected.

Pairing is taken from 44_race_contrast_test.py without modification: cisgender frame, one value per
design cell, and a stratum of (model, condition, gender, ses, relationship) so that a White cell and
a contrast cell differ in race alone. That gives 96 matched pairs per contrast, 24 within each of
the four models. Inputs are the v2 release columns (prompt_condition, ses_normalized,
phq8_total_clipped), which are the same quantities 44 rebuilt by hand from the v1 export; the gate
below is what proves it.

Computed, pooled and separately within each model:

  1. GATE   -- the published Black-White simulated contrast and its interval
  2. TOST   -- two one-sided tests at a bound set to the population disparity itself
               (0.330 Black-White, 0.243 Hispanic-White). Substantively motivated: rejecting both
               one-sided nulls is rejecting that the simulated gap is as large as the real one, in
               either direction.
  3. TOST   -- at half that bound, and at the lower 95% confidence limit of the population
               disparity, so the claim's dependence on the bound is visible rather than assumed.
  4. delta_min -- the tightest bound at which equivalence still holds, which is the number the
               paper can actually write down.
  5. Power  -- minimum detectable effect at 80% for the realised n and variance, power to have
               detected the population disparity, and the power of the equivalence test itself
               under a true difference of zero (Monte Carlo, seeded).

Emits analysis/race_equivalence_tost.csv. No new generations, no edits to any existing file.
"""
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "analysis")
CIS = ["Cisgender Man", "Cisgender Woman"]
STRATUM = ["model", "condition", "gender", "ses", "relationship"]
CONTRASTS = ["Black", "Hispanic"]
ALPHA = 0.05
N_MC = 200_000
RNG = np.random.default_rng(20260803)

# ---------------------------------------------------------------- data, paired exactly as 44 does
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m = m[m.gender.isin(CIS)].copy()
m["phq8_total"] = m["phq8_total_clipped"]
m["ses"] = m["ses_normalized"]
m["condition"] = m["prompt_condition"]
if m.condition.isna().any():
    raise SystemExit("unlabelled rows in prompt_condition")

cell = (m.groupby(["model", "profile_id", "condition", "race", "gender", "ses", "relationship"],
                  as_index=False).phq8_total.mean())

gt = pd.read_csv(os.path.join(BASE, "groundtruth", "phq8_groundtruth_nhanes_2005_2018.csv"))
GTK = {"White": "White", "Black": "Black", "Hispanic": "Hispanic (pooled)"}
gtm = {g: float(gt.loc[gt.group == k, "w_mean"].iloc[0]) for g, k in GTK.items()}

# The design standard errors are keyed on the short label, not the ground-truth label. 44 looked
# them up with the ground-truth key, so "Hispanic (pooled)" missed and that contrast's own anchor
# error silently dropped out of its pop_se. Reproduced below as pop_se_asreported for the gate, and
# corrected in the bound arithmetic, where it makes the conservative bound tighter rather than looser.
dse = pd.read_csv(os.path.join(OUT, "anchor_design_se.csv")).set_index("group")
gtse = {g: float(dse.loc[g, "se_design"]) for g in ("White", "Black", "Hispanic")}


def paired(grp):
    a = cell[cell.race == grp].set_index(STRATUM).phq8_total
    b = cell[cell.race == "White"].set_index(STRATUM).phq8_total
    for name, idx in (("contrast", a.index), ("White", b.index)):
        if idx.has_duplicates:
            raise SystemExit(f"{grp}: {name} stratum key is not unique, pairing would be wrong")
    common = a.index.intersection(b.index)
    return (a.loc[common] - b.loc[common]).to_frame("diff").reset_index()


# ------------------------------------------------------------------------------------ statistics
def tost(mean, se, df, bound):
    """Two one-sided tests against a symmetric equivalence interval (-bound, +bound)."""
    t_lo = (mean + bound) / se          # H0: mu <= -bound
    t_hi = (mean - bound) / se          # H0: mu >= +bound
    p_lo = float(stats.t.sf(t_lo, df))
    p_hi = float(stats.t.cdf(t_hi, df))
    p = max(p_lo, p_hi)
    return t_lo, p_lo, t_hi, p_hi, p, bool(p < ALPHA)


def delta_min(mean, se, df):
    """Smallest symmetric bound at which the TOST still rejects: the 1-2a interval's far edge."""
    return abs(mean) + stats.t.ppf(1 - ALPHA, df) * se


def power_t(delta, se, df, alpha=ALPHA):
    """Power of the two-sided paired t-test against a true effect of delta, se held at observed.
    The far tail of the noncentral t underflows to NaN at large noncentrality, where it is
    negligible by construction, so each tail is floored at zero rather than propagated."""
    crit = stats.t.ppf(1 - alpha / 2, df)
    ncp = delta / se
    hi = stats.nct.sf(crit, df, ncp)
    lo = stats.nct.cdf(-crit, df, ncp)
    hi = 0.0 if not np.isfinite(hi) else float(hi)
    lo = 0.0 if not np.isfinite(lo) else float(lo)
    return min(1.0, hi + lo)


def mde(se, df, target=0.80):
    return float(optimize.brentq(lambda d: power_t(d, se, df) - target, 1e-9, 20 * se))


def tost_power_at_zero(bound, sd, n, n_mc=N_MC):
    """Probability the equivalence test rejects when the true paired difference is exactly zero,
    with the sd estimated from the same n. Monte Carlo, because the sd is not known."""
    df = n - 1
    mbar = RNG.normal(0.0, sd / np.sqrt(n), size=n_mc)
    s = sd * np.sqrt(RNG.chisquare(df, size=n_mc) / df)
    se = s / np.sqrt(n)
    crit = stats.t.ppf(1 - ALPHA, df)
    ok = ((mbar + bound) / se > crit) & ((mbar - bound) / se < -crit)
    return float(ok.mean())


rows = []
for grp in CONTRASTS:
    d = paired(grp)
    pop = gtm[grp] - gtm["White"]
    pop_se = float(np.hypot(gtse[grp], gtse["White"]))
    pop_se_asreported = float(np.hypot(gtse[grp] if grp == "Black" else 0.0, gtse["White"]))
    pop_lo95 = pop - 1.96 * pop_se

    scopes = [("pooled", d)] + [(k, g) for k, g in d.groupby("model")]
    for scope, sub in scopes:
        x = sub["diff"].to_numpy(float)
        n = len(x)
        df = n - 1
        mean = float(x.mean())
        sd = float(x.std(ddof=1))
        se = sd / np.sqrt(n)
        t95 = stats.t.ppf(0.975, df)
        t90 = stats.t.ppf(1 - ALPHA, df)
        dmin = delta_min(mean, se, df)

        bounds = [("population disparity", pop),
                  ("half population disparity", pop / 2.0),
                  ("population lower 95% CL", pop_lo95),
                  ("delta_min (tightest passing)", dmin)]
        for bname, b in bounds:
            t_lo, p_lo, t_hi, p_hi, p, eq = tost(mean, se, df, b)
            rows.append(dict(
                contrast=f"{grp} minus White", scope=scope, n_pairs=n,
                mean_diff=round(mean, 4), sd_diff=round(sd, 4), se=round(se, 4), df=df,
                # 44 formed the published interval with the normal multiplier; the gate has to
                # match that construction, so both are carried. The TOST uses t throughout.
                ci95z_lo=round(mean - 1.96 * se, 4), ci95z_hi=round(mean + 1.96 * se, 4),
                ci95_lo=round(mean - t95 * se, 4), ci95_hi=round(mean + t95 * se, 4),
                ci90_lo=round(mean - t90 * se, 4), ci90_hi=round(mean + t90 * se, 4),
                population=round(pop, 4), pop_se=round(pop_se, 4),
                pop_se_asreported=round(pop_se_asreported, 4),
                bound_type=bname, bound=round(float(b), 4),
                t_lower=round(t_lo, 3), p_lower=p_lo,
                t_upper=round(t_hi, 3), p_upper=p_hi,
                p_tost=p, equivalent=eq,
                delta_min_equiv=round(dmin, 4),
                mde80=round(mde(se, df), 4),
                power_at_population=round(power_t(pop, se, df), 4),
                tost_power_at_zero=round(tost_power_at_zero(b, sd, n), 4) if b > 0 else np.nan,
            ))

res = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
res.to_csv(os.path.join(OUT, "race_equivalence_tost.csv"), index=False)

# ------------------------------------------------------------------------------------------ gate
pool = res[(res.scope == "pooled") & (res.bound_type == "population disparity")].set_index("contrast")
bw = pool.loc["Black minus White"]
hw = pool.loc["Hispanic minus White"]
checks = [
    ("Black-White simulated contrast = -0.131", bw.mean_diff, -0.1306, 0.001),
    ("Black-White se = 0.0724", bw.se, 0.0724, 0.0001),
    ("Black-White CI low  = -0.273", bw.ci95z_lo, -0.2726, 0.001),
    ("Black-White CI high = +0.011", bw.ci95z_hi, 0.0114, 0.001),
    ("Black-White n_pairs = 96", bw.n_pairs, 96, 0),
    ("Hispanic-White simulated contrast = -0.034", hw.mean_diff, -0.0340, 0.001),
    ("Hispanic-White CI low  = -0.150", hw.ci95z_lo, -0.1497, 0.001),
    ("Hispanic-White CI high = +0.082", hw.ci95z_hi, 0.0816, 0.001),
    ("Hispanic-White n_pairs = 96", hw.n_pairs, 96, 0),
    ("population Black-White = +0.330", bw.population, 0.3299, 0.001),
    ("population Hispanic-White = +0.243", hw.population, 0.2425, 0.001),
]
ok = all(abs(float(got) - float(want)) <= tol for _, got, want, tol in checks)

pd.set_option("display.width", 220)
print("GATE: reproduce the published racial contrast numbers from the same pairing")
print("=" * 104)
for label, got, want, tol in checks:
    flag = "ok  " if abs(float(got) - float(want)) <= tol else "FAIL"
    print(f"  [{flag}] {label:46s} got {float(got):+.4f}  want {float(want):+.4f}")
print(f"\n  GATE {'PASS' if ok else 'FAIL'}")
if not ok:
    print("  *** GATE FAILED. Data handling does not match the published analysis.")
    print("  *** Nothing below this line is trustworthy. Stop.")

print("\n\nEQUIVALENCE TESTS (TOST, alpha = 0.05, symmetric bound, paired t on design cells)")
print("=" * 104)
show = ["contrast", "scope", "n_pairs", "mean_diff", "se", "ci90_lo", "ci90_hi",
        "bound_type", "bound", "p_lower", "p_upper", "p_tost", "equivalent"]
for grp in CONTRASTS:
    c = f"{grp} minus White"
    t = res[(res.contrast == c) & (res.bound_type != "delta_min (tightest passing)")].copy()
    for col in ("p_lower", "p_upper", "p_tost"):
        t[col] = t[col].map(lambda v: f"{v:.4g}")
    print(f"\n{c}")
    print(t[show[1:]].to_string(index=False))

print("\n\nPOWER AND MINIMUM DETECTABLE EFFECT")
print("=" * 104)
p = res[res.bound_type == "population disparity"][
    ["contrast", "scope", "n_pairs", "sd_diff", "se", "mde80", "population",
     "power_at_population", "delta_min_equiv", "tost_power_at_zero"]]
print(p.to_string(index=False))

print("\n\nTIGHTEST DEFENSIBLE EQUIVALENCE CLAIM (pooled)")
print("=" * 104)
for grp in CONTRASTS:
    r = res[(res.contrast == f"{grp} minus White") & (res.scope == "pooled")
            & (res.bound_type == "delta_min (tightest passing)")].iloc[0]
    print(f"  {grp:9s} vs White: equivalent to zero within +/- {r.bound:.3f} PHQ-8 points "
          f"(90% CI [{r.ci90_lo:+.3f}, {r.ci90_hi:+.3f}]), "
          f"population disparity {r.population:+.3f}, ratio {r.bound / r.population:.2f}")

print("\n\nSIDE FINDING: the Hispanic anchor standard error dropped out of the published gap test")
print("=" * 104)
# 44 keyed anchor_design_se.csv with the ground-truth label, so "Hispanic (pooled)" found nothing
# and the Hispanic contrast carried only White's design error. Redone with the right key, and the
# published Asian and Black p-values reused, since their lookups resolved.
prev = pd.read_csv(os.path.join(OUT, "race_contrast_tests.csv")).set_index("contrast")
pv = {}
for c in prev.index:
    grp = c.split(" ")[0]
    if grp == "Hispanic":
        r = res[(res.contrast == c) & (res.scope == "pooled")].iloc[0]
        se_gap = float(np.hypot(r.se, r.pop_se))
        t = (r.mean_diff - r.population) / se_gap
        pv[c] = float(stats.t.sf(abs(t), r.df) * 2)
    else:
        pv[c] = float(prev.loc[c, "p_vs_pop"])
s = pd.Series(pv).sort_values()
q = (s.to_numpy() * len(s) / np.arange(1, len(s) + 1)).clip(max=1.0)
q = np.minimum.accumulate(q[::-1])[::-1]
for (c, p), qq in zip(s.items(), q):
    print(f"  {c:22s} p_vs_pop {p:.6g}   q {qq:.6g}   (published q {prev.loc[c, 'q_vs_pop']:.6g})")
print("  Only the Hispanic row moves. It stays significant, but no longer at q < 0.001.")

print("\nWritten -> %s" % os.path.join(OUT, "race_equivalence_tost.csv"))
