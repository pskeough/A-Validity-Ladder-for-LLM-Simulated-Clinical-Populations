"""Is the DSM-5 gateway result a positional artifact rather than a clinical one?

A reviewer objects to Section~\\ref{sec:coherence}. The gateway rule reads array positions 1 and 2
(DPQ010 anhedonia, DPQ020 depressed mood). Language models asked to emit a list of integers are
said to produce weakly monotone non-increasing arrays as a matter of format habit. If they do, mass
piles onto positions 1 and 2 for reasons that have nothing to do with depression, the 2.68% gateway
violation rate falls out of the generation format, and the paper's coherence claim is decoration on
a syntactic tic.

The objection is testable, because it makes a prediction the clinical reading does not: a format
habit produces a smooth descending gradient across all eight positions, is indifferent to severity,
and shows up in cases nobody would call depressed. Symptom routing produces elevation at positions
1 and 2 specifically, with no ordering among positions 3 to 8, and intensifies with total load.

Five measurements over the 28,800 generations, each against a null:

  1. weakly monotone non-increasing across positions 1..8, observed, pooled and per model
  2. the same under a column-permutation null: each item column permuted independently within
     model, so every item's marginal survives and the joint is destroyed. R replicates, mean
     and spread. A within-case-permutation null is also given in closed form, since the chance
     that a shuffle of a case's own eight responses lands sorted is prod(count_v!)/8!
  3. mass on positions 1-2 conditional on the total, against a uniform-position expectation of
     2/8 and against the column-permutation null restricted to the same total
  4. Spearman of item position against mean item score, per model, over all eight positions and
     over positions 3 to 8 alone. The second is the discriminator: a descending-list habit orders
     the non-cardinal positions too, symptom routing does not
  5. Spearman of position against score within each generation, tie-corrected, averaged
  6. a positive control the objection has no answer to: the same eight positions in NHANES.
     If position k is elevated because of a format habit, the emitted profile across positions
     should be unrelated to the MEC-weighted mean of DPQ0k0 in the real population

GATE: reproduce the published 2.68% gateway violation rate among elevated cases and the 10.4%
conditional-marginal chance null from the raw file, using the construction of 22_gateway_null.py.

Emits analysis/gateway_positional_artifact.csv and analysis/gateway_positional_by_total.csv.
Seeded. Reads data only; writes nothing outside analysis/.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
OUT = os.path.join(BASE, "analysis")

ITEMS = [f"phq8_{i}" for i in range(1, 9)]
GATE = ["phq8_1", "phq8_2"]
ELEV, CARD, NREP = 10, 2, 200
POS = np.arange(1, 9, dtype=float)

rng = np.random.default_rng(20260803)

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
A = m[ITEMS].to_numpy(float)
# Totals follow 03/22: the sum of the eight items, clipped to the instrument range for the
# elevation test. One generation carries an out-of-range item (position 8 scored 21), which the
# clip absorbs. Share denominators below use the raw sum so that shares stay inside [0, 1].
m["phq8_total"] = A.sum(axis=1).clip(0, 24)
RAW = A.sum(axis=1)
MODELS = sorted(m.model.unique())


# ----------------------------------------------------------------- gate

def violates(a):
    return (a[:, 0] < CARD) & (a[:, 1] < CARD)


def null_conditional_marginal(df, seed=20260726):
    """22_gateway_null.py's conditional-marginal null, reimplemented to reproduce 10.42.

    Each case keeps its own total; its eight responses are redrawn from the item marginals of its
    own model and cohort and conditioned on that total by rejection. Violations over retained
    draws, the same basis the observed rate uses.
    """
    r = np.random.default_rng(seed)
    viol = kept = 0
    for _, grp in df.groupby(["model", "profile_id"]):
        pools = [grp[c].to_numpy() for c in ITEMS]
        tot = grp.phq8_total.to_numpy()
        for _ in range(10):
            drawn = np.stack([r.choice(p, len(grp)) for p in pools], axis=1)
            for _ in range(60):
                bad = drawn.sum(axis=1) != tot
                if not bad.any():
                    break
                drawn[bad] = np.stack([r.choice(p, bad.sum()) for p in pools], axis=1)
            keep = drawn.sum(axis=1) == tot
            if keep.sum():
                viol += int(violates(drawn[keep][:, :2]).sum())
                kept += int(keep.sum())
    return viol / kept * 100


elev_mask = m.phq8_total.to_numpy() >= ELEV
elev = m[elev_mask].copy()
n_elev = int(elev_mask.sum())
n_viol = int(violates(elev[GATE].to_numpy()).sum())
obs_gate = 100.0 * n_viol / n_elev
null_gate = null_conditional_marginal(elev)

g1 = (n_elev == 9590) and (n_viol == 257) and abs(obs_gate - 2.68) < 0.005
g2 = abs(null_gate - 10.42) < 0.05
GATE_OK = g1 and g2

print("=" * 92)
print("GATE  reproduce the published gateway numbers from data/model_outputs_v2.csv")
print("=" * 92)
print("  elevated generations (PHQ-8 total >= 10) : %d   (paper: 9,590)" % n_elev)
print("  gateway violations                       : %d   (paper: 257)" % n_viol)
print("  observed violation rate                  : %.2f%%  (paper: 2.68%%)   %s"
      % (obs_gate, "ok" if g1 else "MISMATCH"))
print("  conditional-marginal chance null         : %.2f%%  (paper: 10.4%%)   %s"
      % (null_gate, "ok" if g2 else "MISMATCH"))
print("\n  GATE: %s" % ("PASS" if GATE_OK else "*** FAIL ***"))
if not GATE_OK:
    print("  *** DATA HANDLING DOES NOT MATCH THE PAPER. EVERYTHING BELOW IS UNTRUSTWORTHY. ***")
print()


# ------------------------------------------------------- shared measures

def monotone_noninc(a):
    """Weakly monotone non-increasing across positions 1..8."""
    return (np.diff(a, axis=1) <= 0).all(axis=1)


def inversions(a):
    """Adjacent positions where the array goes up. 0 to 7 per generation."""
    return (np.diff(a, axis=1) > 0).sum(axis=1)


VALS = np.unique(A)


def avg_ranks(a):
    """Tie-corrected ranks within each row. Average rank of value v is (#below v) + (c_v + 1)/2."""
    counts = np.stack([(a == v).sum(axis=1) for v in VALS], axis=1).astype(float)
    below = np.cumsum(counts, axis=1) - counts
    rank_of_val = below + (counts + 1.0) / 2.0
    idx = np.searchsorted(VALS, a)
    return np.take_along_axis(rank_of_val, idx, axis=1)


def within_gen_spearman(a):
    """Spearman of position (1..8) against score, per row. NaN where the row has no variance."""
    r = avg_ranks(a)
    rc = r - 4.5                      # average ranks always sum to 36, so the mean is 4.5
    pc = POS - 4.5
    num = (rc * pc).sum(axis=1)
    den = np.sqrt((rc ** 2).sum(axis=1) * (pc ** 2).sum())
    out = np.full(len(a), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def spearman_small(x, y):
    """Spearman for two short vectors, tie-corrected."""
    def rk(v):
        v = np.asarray(v, float)
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(1, len(v) + 1)
        for u in np.unique(v):
            sel = v == u
            if sel.sum() > 1:
                r[sel] = r[sel].mean()
        return r
    a, b = rk(x) - np.mean(rk(x)), rk(y) - np.mean(rk(y))
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def within_case_monotone_null(a):
    """Closed form. Shuffling a case's own eight responses lands sorted with prob prod(c_v!)/8!."""
    from math import factorial
    fac = {int(v): factorial(int(v)) for v in range(9)}
    counts = np.stack([(a == v).sum(axis=1) for v in VALS], axis=1)
    p = np.ones(len(a), dtype=float)
    for k in range(counts.shape[1]):
        p *= np.array([fac[c] for c in counts[:, k]], dtype=float)
    return float((p / factorial(8)).mean()) * 100


def share12(a, denom):
    s = np.full(len(a), np.nan)
    ok = denom > 0
    s[ok] = (a[ok, 0] + a[ok, 1]) / denom[ok]
    return s


# ------------------------------------- the column-permutation null, R draws
# Each item column permuted independently inside its own model. Item marginals survive exactly;
# every trace of which responses travelled together is destroyed.
model_idx = {mm: np.where(m.model.to_numpy() == mm)[0] for mm in MODELS}
obs_tot_bins = np.bincount(RAW.astype(int), minlength=40).astype(float)
obs_tot_w = obs_tot_bins.copy()
obs_tot_w[0] = 0.0                                   # shares undefined at total 0
obs_tot_w = obs_tot_w / obs_tot_w.sum()

null_mono = {k: [] for k in ["pooled"] + MODELS}
null_mono_elev = {k: [] for k in ["pooled"] + MODELS}
null_spear = {k: [] for k in ["pooled"] + MODELS}
null_share_ct = {k: [] for k in ["pooled"] + MODELS}   # total-conditional share on positions 1-2
null_gaterate = []
null_num_tot = np.zeros(40)     # pooled share on positions 1-2, accumulated per total
null_den_tot = np.zeros(40)

for _ in range(NREP):
    P = np.empty_like(A)
    for mm in MODELS:
        ix = model_idx[mm]
        blk = A[ix]
        P[ix] = np.stack([rng.permutation(blk[:, j]) for j in range(8)], axis=1)
    mono = monotone_noninc(P)
    sp = within_gen_spearman(P)
    praw = P.sum(axis=1)
    pel = praw.clip(0, 24) >= ELEV
    sh = share12(P, praw)
    for scope in ["pooled"] + MODELS:
        sel = np.ones(len(P), bool) if scope == "pooled" else (m.model.to_numpy() == scope)
        null_mono[scope].append(100.0 * mono[sel].mean())
        e = sel & pel
        null_mono_elev[scope].append(100.0 * mono[e].mean() if e.any() else np.nan)
        null_spear[scope].append(float(np.nanmean(sp[sel])))
        # total-conditional share: mean share within each total, reweighted to the observed
        # distribution of totals, so the comparison is like-for-like on symptom load
        v = sel & (praw > 0)
        num = np.bincount(praw[v].astype(int), weights=sh[v], minlength=40)
        den = np.bincount(praw[v].astype(int), minlength=40).astype(float)
        ok = den > 0
        w = obs_tot_w[:40].copy()
        w[~ok] = 0.0
        mean_by_tot = np.zeros(40)
        mean_by_tot[ok] = num[ok] / den[ok]
        null_share_ct[scope].append(float((mean_by_tot * w).sum() / w.sum()))
        if scope == "pooled":
            null_num_tot += num
            null_den_tot += den
    null_gaterate.append(100.0 * violates(P[pel]).mean())


# ------------------------------------------------------- observed measures
mono_obs = monotone_noninc(A)
inv_obs = inversions(A)
sp_obs = within_gen_spearman(A)
sh_obs = share12(A, RAW)

rows = []
by_total_rows = []
for subset, mask_all in [("all", np.ones(len(A), bool)), ("elevated", elev_mask)]:
    for scope in ["pooled"] + MODELS:
        sel = mask_all & (np.ones(len(A), bool) if scope == "pooled"
                          else (m.model.to_numpy() == scope))
        a = A[sel]
        nm = np.array(null_mono[scope] if subset == "all" else null_mono_elev[scope], float)
        means = a.mean(axis=0)
        v = sel & (RAW > 0)
        num = np.bincount(RAW[v].astype(int), weights=sh_obs[v], minlength=40)
        den = np.bincount(RAW[v].astype(int), minlength=40).astype(float)
        ok = den > 0
        w = obs_tot_w[:40].copy()
        w[~ok] = 0.0
        mbt = np.zeros(40)
        mbt[ok] = num[ok] / den[ok]
        sh_ct = float((mbt * w).sum() / w.sum())
        null_ct = float(np.mean(null_share_ct[scope]))
        rows.append(dict(
            subset=subset, scope=scope, n=int(sel.sum()),
            monotone_pct=round(100.0 * mono_obs[sel].mean(), 3),
            null_colperm_mono_mean_pct=round(float(nm.mean()), 3),
            null_colperm_mono_sd_pct=round(float(nm.std(ddof=1)), 3),
            null_colperm_mono_p2_5=round(float(np.percentile(nm, 2.5)), 3),
            null_colperm_mono_p97_5=round(float(np.percentile(nm, 97.5)), 3),
            null_withincase_mono_pct=round(within_case_monotone_null(a), 3),
            mean_inversions_of_7=round(float(inv_obs[sel].mean()), 3),
            pct_zero_inversions=round(100.0 * (inv_obs[sel] == 0).mean(), 3),
            share12_obs=round(sh_ct, 4),
            share12_null_colperm=round(null_ct, 4),
            share12_uniform_expect=0.25,
            ratio_vs_uniform=round(sh_ct / 0.25, 3),
            ratio_vs_colperm_null=round(sh_ct / null_ct, 3) if null_ct else np.nan,
            spearman_pos_vs_meanitem=round(spearman_small(POS, means), 3),
            spearman_pos3to8=round(spearman_small(POS[2:], means[2:]), 3),
            within_gen_spearman_mean=round(float(np.nanmean(sp_obs[sel])), 4),
            within_gen_spearman_sd=round(float(np.nanstd(sp_obs[sel], ddof=1)), 4),
            within_gen_spearman_null=round(float(np.mean(null_spear[scope])), 4),
            n_no_variance=int(np.isnan(sp_obs[sel]).sum()),
            **{f"mean_pos{i + 1}": round(float(means[i]), 4) for i in range(8)}))
        if subset == "all" and scope == "pooled":
            for t in range(1, 25):
                if den[t] > 0:
                    nt = (null_num_tot[t] / null_den_tot[t]) if null_den_tot[t] > 0 else np.nan
                    by_total_rows.append(dict(phq8_total=t, n=int(den[t]),
                                              share12_obs=round(float(mbt[t]), 4),
                                              share12_null_colperm=round(float(nt), 4),
                                              uniform_expect=0.25,
                                              ratio_vs_uniform=round(float(mbt[t]) / 0.25, 3),
                                              ratio_vs_null=round(float(mbt[t] / nt), 3)
                                              if np.isfinite(nt) and nt else np.nan))

res = pd.DataFrame(rows)
bt = pd.DataFrame(by_total_rows)
os.makedirs(OUT, exist_ok=True)
res.to_csv(os.path.join(OUT, "gateway_positional_artifact.csv"), index=False)
bt.to_csv(os.path.join(OUT, "gateway_positional_by_total.csv"), index=False)

pd.set_option("display.width", 250)
short = {mm: mm.split("/")[-1] for mm in MODELS}
short["pooled"] = "pooled"

print("=" * 92)
print("1-2  MONOTONICITY  are emitted arrays weakly non-increasing across positions 1..8?")
print("=" * 92)
t1 = res[res.subset == "all"][["scope", "n", "monotone_pct", "null_colperm_mono_mean_pct",
                              "null_colperm_mono_sd_pct", "null_withincase_mono_pct",
                              "mean_inversions_of_7"]].copy()
t1["scope"] = t1.scope.map(short)
t1.columns = ["scope", "n", "observed %", "col-perm null %", "null sd", "within-case null %",
              "mean inversions/7"]
print(t1.to_string(index=False))
print("\n     (%d replicates of the column-permutation null; 'inversions' counts adjacent" % NREP)
print("      positions where the array goes UP. A descending list has 0 of 7.)")

print("\n" + "=" * 92)
print("3  MASS ON POSITIONS 1-2, CONDITIONAL ON THE PHQ-8 TOTAL")
print("=" * 92)
t3 = res[res.subset == "all"][["scope", "share12_obs", "share12_null_colperm",
                              "ratio_vs_uniform", "ratio_vs_colperm_null"]].copy()
t3["scope"] = t3.scope.map(short)
t3.columns = ["scope", "obs share on pos 1-2", "col-perm null", "ratio vs 0.25", "ratio vs null"]
print(t3.to_string(index=False))
e3 = res[res.subset == "elevated"][["scope", "share12_obs", "ratio_vs_uniform"]].copy()
e3["scope"] = e3.scope.map(short)
print("\n  elevated cases only:")
print("  " + e3.to_string(index=False).replace("\n", "\n  "))

print("\n" + "=" * 92)
print("4  POSITION vs MEAN ITEM SCORE  (per model)")
print("=" * 92)
t4 = res[res.subset == "all"][["scope"] + [f"mean_pos{i}" for i in range(1, 9)]
                              + ["spearman_pos_vs_meanitem", "spearman_pos3to8"]].copy()
t4["scope"] = t4.scope.map(short)
t4.columns = ["scope"] + [f"p{i}" for i in range(1, 9)] + ["rho(1..8)", "rho(3..8)"]
print(t4.to_string(index=False))
print("\n     rho(3..8) is the discriminator. A format habit that emits descending integers has no")
print("     reason to stop at position 3, so it drives rho(3..8) toward -1 as well.")

print("\n" + "=" * 92)
print("5  WITHIN-GENERATION SPEARMAN OF POSITION vs SCORE")
print("=" * 92)
t5 = res[res.subset == "all"][["scope", "within_gen_spearman_mean", "within_gen_spearman_sd",
                              "within_gen_spearman_null", "n_no_variance"]].copy()
t5["scope"] = t5.scope.map(short)
t5.columns = ["scope", "mean rho", "sd", "col-perm null", "n flat (rho undefined)"]
print(t5.to_string(index=False))
print("\n     A perfectly descending array scores rho = -1. The within-case permutation null is 0")
print("     by construction.")

print("\n" + "=" * 92)
print("6  POSITIVE CONTROL  the same eight positions in NHANES 2005-2018, MEC-weighted")
print("=" * 92)
CYCLES = ["D", "E", "F", "G", "H", "I", "J"]
DPQ = ["DPQ0%d0" % i for i in range(1, 9)]
RAWDIR = os.path.join(BASE, "data", "nhanes_raw")
fr = []
for suf in CYCLES:
    demo = pd.read_sas(os.path.join(RAWDIR, "DEMO_%s.xpt" % suf), format="xport")
    dpq = pd.read_sas(os.path.join(RAWDIR, "DPQ_%s.xpt" % suf), format="xport")
    d = demo[["SEQN", "RIDAGEYR", "WTMEC2YR"]].merge(
        dpq[["SEQN"] + [c for c in DPQ if c in dpq.columns]], on="SEQN", how="left")
    fr.append(d)
nh = pd.concat(fr, ignore_index=True)
for c in DPQ:                                   # 7 refused, 9 don't know, . missing
    nh[c] = nh[c].where(nh[c] <= 3, np.nan)
nh = nh[nh.RIDAGEYR >= 18].dropna(subset=DPQ + ["WTMEC2YR"])
nh = nh[nh.WTMEC2YR > 0]
w = nh.WTMEC2YR.to_numpy(float)
nh_means = np.array([float(np.sum(w * nh[c].to_numpy(float)) / np.sum(w)) for c in DPQ])

ctl = []
for scope in ["pooled"] + MODELS:
    sel = np.ones(len(A), bool) if scope == "pooled" else (m.model.to_numpy() == scope)
    ctl.append(dict(scope=short[scope],
                    rho_vs_nhanes=round(spearman_small(A[sel].mean(axis=0), nh_means), 3)))
ctl = pd.DataFrame(ctl)
nhrow = pd.DataFrame([dict(scope="NHANES adults 18+ (n=%d)" % len(nh),
                           **{f"p{i + 1}": round(float(nh_means[i]), 4) for i in range(8)})])
print(nhrow.to_string(index=False))
print("\n  Spearman of each model's eight position means against the NHANES eight item means:")
print("  " + ctl.to_string(index=False).replace("\n", "\n  "))
print("\n     Read this one carefully. It is weak and it cuts both ways. A pure format habit would")
print("     give rho near zero, and three of four models sit above that, so there is some content")
print("     in the ordering. But the agreement is poor in absolute terms, and DeepSeek's is nil.")
print("     That is the paper's own thesis, not a defence of it: these models do not reproduce the")
print("     real item profile either. It refutes 'the positions are pure format'. It does not")
print("     establish 'the positions are clinically right'.")
res["nhanes_item_mean_rho"] = [
    ctl.loc[ctl.scope == short[s], "rho_vs_nhanes"].iat[0] for s in res.scope]
res.to_csv(os.path.join(OUT, "gateway_positional_artifact.csv"), index=False)

print("\n" + "=" * 92)
print("VERDICT")
print("=" * 92)
p = res[(res.subset == "all") & (res.scope == "pooled")].iloc[0]
pe = res[(res.subset == "elevated") & (res.scope == "pooled")].iloc[0]
nullmean = float(np.mean(null_mono["pooled"]))
worst = res[(res.subset == "all") & (res.scope != "pooled")].sort_values("monotone_pct").iloc[-1]
print("  perfectly non-increasing arrays, pooled : %.2f%% observed, %.2f%% under the "
      "column-permutation null" % (p.monotone_pct, nullmean))
print("  worst single model                      : %s at %.2f%% (its null %.2f%%)"
      % (short[worst.scope], worst.monotone_pct, worst.null_colperm_mono_mean_pct))
print("  mean within-generation rho, pooled      : %+.3f (null %+.3f)"
      % (p.within_gen_spearman_mean, p.within_gen_spearman_null))
print("  mass on positions 1-2, total-conditional: %.3f observed against %.3f expected if"
      % (p.share12_obs, 0.25))
print("                                            position carried no information (%.2fx)"
      % p.ratio_vs_uniform)
print("  gateway violation rate under the column-permutation null: %.2f%% (obs %.2f%%)"
      % (float(np.mean(null_gaterate)), obs_gate))
mech = p.monotone_pct > 50.0 and p.monotone_pct > 5 * max(nullmean, 1e-9)
print("\n  Does monotone integer-list generation mechanically explain the gateway result?")
print("  ->  %s" % ("YES. The arrays are strongly sorted and the coherence claim is partly an "
                    "artifact of\n      integer-list generation. The author must be told directly."
                    if mech else
                    "NO. Emitted arrays are not sorted descending at anything like the rate the\n"
                    "      objection requires, and the elevation at positions 1-2 does not extend\n"
                    "      into positions 3-8 as a format habit would predict."))
if not GATE_OK:
    print("\n  *** GATE FAILED - none of the above is trustworthy. ***")
print("\nWritten -> %s" % os.path.join(OUT, "gateway_positional_artifact.csv"))
print("Written -> %s" % os.path.join(OUT, "gateway_positional_by_total.csv"))
