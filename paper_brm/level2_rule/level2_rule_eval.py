"""Candidate level-2 verdict rules: operating characteristics by simulation, then every rule
applied to the paper's own level-2 rows and to the two external releases.

Every rule here reads the same five numbers per contrast: the simulated gap g and its SE, the
population gap p and its SE, and their covariance (zero where the two come from different
samples, as in the paper). All rules are stated on the gap ratio rho = g / p.

  R0  current     scripts/66 as published. Bound = |p| for both equivalence tests.
  R1  half        R0 with the bound at |p| / 2.
  R2  reorder     R0 with the flattened test moved ahead of the missing test.
  R3  interval    The 90% Fieller interval for rho, read against four fixed boundaries:
                    reversed   rho <= -m0
                    missing    -m0 < rho < m0
                    flattened  m0 <= rho <= 1 - m1
                    kept       1 - m1 < rho < 1 + m1
                    steepened  rho >= 1 + m1
                  A verdict is issued when the whole interval sits in one region. An interval
                  that crosses one boundary gets both names ("missing or flattened"). Wider than
                  that is undetermined. A population gap whose own 90% interval covers zero
                  gets "no population gap" before anything else is read.

Each boundary test is the one-sided test of rho > c, i.e. of g - c*p > 0, with
se_c = sqrt(se_g^2 + c^2 se_p^2 - 2 c cov). At c = 0 it is the paper's test against zero; at
c = 1 it is the paper's test against the population value. R3 therefore reuses the paper's two
tests and adds two more at stated margins.

Outputs to ./results/. Deterministic (seeded). Runs in about a minute.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                       # PsychBench/UpdatedRun
EXT = HERE.parent / "external" / "results"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
ALPHA = 0.05
RNG = np.random.default_rng(20260925)
LABELS = ["reversed", "missing", "flattened", "kept", "steepened"]


# --------------------------------------------------------------------------------------- rules
def _crit(df, two_sided):
    df = np.asarray(df, float)
    q = 1 - ALPHA / 2 if two_sided else 1 - ALPHA
    return stats.t.ppf(q, np.where(np.isfinite(df), df, 1e9))


def paper_rule(g, se_g, p, se_p, cov, df, frac=1.0, order="paper"):
    """scripts/66 verdict(), vectorised. Separability uses the anchor-inclusive SE; the
    equivalence test against zero uses the simulated gap's SE alone, as in 66."""
    g, se_g, p, se_p, cov = map(lambda a: np.asarray(a, float), (g, se_g, p, se_p, cov))
    t95, t90 = _crit(df, True), _crit(df, False)
    d = g - p
    se_d = np.sqrt(se_g ** 2 + se_p ** 2 - 2 * cov)
    b = frac * np.abs(p)
    sep = np.abs(d) > t95 * se_d
    eq0 = (g - t90 * se_g > -b) & (g + t90 * se_g < b)
    eqp = (d - t90 * se_d > -b) & (d + t90 * se_d < b)
    same = np.sign(g) == np.sign(p)
    ratio_gt1 = np.abs(g) > np.abs(p)
    tests = {
        "missing": eq0 & sep,
        "kept": same & ~sep & eqp,
        "steepened": sep & same & ratio_gt1,
        "flattened": sep & same & ~ratio_gt1,
        "reversed": sep & ~same,
    }
    seq = ["missing", "kept", "steepened", "flattened", "reversed"]
    if order == "reorder":
        seq = ["flattened", "missing", "kept", "steepened", "reversed"]
    out = np.full(g.shape, "undetermined", dtype=object)
    done = np.zeros(g.shape, bool)
    for lab in seq:
        hit = tests[lab] & ~done
        out[hit] = lab
        done |= hit
    return out


def boundaries(m0, m1):
    return np.array([-m0, m0, 1 - m1, 1 + m1])


def interval_rule(g, se_g, p, se_p, cov, df, m0=0.25, m1=0.25):
    """R3. Returns the label and the region index range [lo, hi] the 90% interval covers."""
    g, se_g, p, se_p, cov = map(lambda a: np.asarray(a, float), (g, se_g, p, se_p, cov))
    s = np.where(p < 0, -1.0, 1.0)            # orient so the population gap is positive
    g, p, cov = g * s, p * s, cov              # cov(s g, s p) = s^2 cov = cov
    t90 = _crit(df, False)
    B = boundaries(m0, m1)
    above = np.zeros(g.shape, int)            # boundaries the ratio is confidently above
    below = np.zeros(g.shape, int)            # boundaries it is confidently below
    for c in B:
        se_c = np.sqrt(se_g ** 2 + c ** 2 * se_p ** 2 - 2 * c * cov)
        t = (g - c * p) / se_c
        above += t > t90
        below += t < -t90
    lo, hi = above, len(B) - below
    nopop = p - t90 * se_p <= 0
    out = np.empty(g.shape, dtype=object)
    for i in range(g.size):
        if nopop.flat[i]:
            out.flat[i] = "no population gap"
        elif lo.flat[i] == hi.flat[i]:
            out.flat[i] = LABELS[lo.flat[i]]
        elif hi.flat[i] == lo.flat[i] + 1:
            out.flat[i] = f"{LABELS[lo.flat[i]]} or {LABELS[hi.flat[i]]}"
        else:
            out.flat[i] = "undetermined"
    return out, lo, hi


def fieller(g, se_g, p, se_p, cov, df, level=0.90):
    """Fieller interval for g/p. NaN bounds where the population gap is not separable from zero
    at this level (the set is then unbounded)."""
    t = _crit(df, False) if level == 0.90 else stats.norm.ppf(0.5 + level / 2)
    a = p ** 2 - t ** 2 * se_p ** 2
    bq = g * p - t ** 2 * cov
    c = g ** 2 - t ** 2 * se_g ** 2
    disc = bq ** 2 - a * c
    ok = (a > 0) & (disc >= 0)
    r = np.sqrt(np.where(ok, disc, np.nan))
    lo = np.where(ok, (bq - r) / np.where(a > 0, a, np.nan), np.nan)
    hi = np.where(ok, (bq + r) / np.where(a > 0, a, np.nan), np.nan)
    return lo, hi


RULES = {
    "R0_current": lambda *a: paper_rule(*a),
    "R1_half_bound": lambda *a: paper_rule(*a, frac=0.5),
    "R2_reorder": lambda *a: paper_rule(*a, order="reorder"),
    "R3_interval": lambda *a: interval_rule(*a)[0],
}
MARGINS = [(0.2, 0.2), (0.25, 0.25), (1 / 3, 1 / 3), (0.25, 0.5)]


# ---------------------------------------------------------------------------------- simulation
def simulate(reps=4000):
    """P(verdict | true ratio, precision). p = 1 without loss of generality."""
    rhos = np.round(np.arange(-0.75, 2.0001, 0.025), 3)
    rows = []
    for se_p in (0.0, 0.15):
        for se_g in (0.05, 0.10, 0.20, 0.40):
            for rho in rhos:
                gh = RNG.normal(rho, se_g, reps)
                ph = RNG.normal(1.0, se_p, reps) if se_p else np.ones(reps)
                args = (gh, np.full(reps, se_g), ph, np.full(reps, se_p), np.zeros(reps), np.full(reps, 200.0))
                for name, f in RULES.items():
                    v = pd.Series(f(*args)).value_counts(normalize=True)
                    for lab, pr in v.items():
                        rows.append(dict(rule=name, se_p=se_p, se_g=se_g, rho=rho, verdict=lab, prob=pr))
    return pd.DataFrame(rows)


def region_of(rho, m0=0.25, m1=0.25):
    return LABELS[int(np.searchsorted(boundaries(m0, m1), rho, side="right"))]


def summarise(sim):
    """Rule-agnostic properties, plus one scored table against the labels' plain meanings."""
    out = []
    for (rule, se_p, se_g), s in sim.groupby(["rule", "se_p", "se_g"]):
        P = s.pivot_table(index="rho", columns="verdict", values="prob", fill_value=0)
        get = lambda lab, r: float(P.loc[r, lab]) if lab in P.columns else 0.0

        def share_containing(lab, r):  # R3 straddle labels count toward both names
            return sum(float(P.loc[r, c]) for c in P.columns if lab in c.split(" or "))
        miss = P[[c for c in P.columns if c == "missing"]].sum(axis=1) if "missing" in P.columns else P.iloc[:, 0] * 0
        pos = miss[(miss.index > 0) & (miss > 0.10)]
        # plain-meaning accuracy: the verdict names the region the true ratio is in (straddles
        # count if either name is right); undetermined counts as wrong but is tallied separately
        acc, wrong, undet, single, single_ok = [], [], [], [], []
        for r in P.index:
            truth = region_of(r)
            ok = sum(float(P.loc[r, c]) for c in P.columns if truth in c.split(" or "))
            u = sum(float(P.loc[r, c]) for c in P.columns if c in ("undetermined", "no population gap"))
            acc.append(ok)
            single.append(sum(float(P.loc[r, c]) for c in P.columns if c in LABELS))
            single_ok.append(float(P.loc[r, truth]) if truth in P.columns else 0.0)
            undet.append(u)
            wrong.append(1 - ok - u)
        out.append(dict(
            rule=rule, se_p=se_p, se_g=se_g,
            largest_ratio_called_missing_10pct=float(pos.index.max()) if len(pos) else np.nan,
            p_missing_at_0_50=get("missing", 0.5), p_missing_at_0_75=get("missing", 0.75),
            p_missing_at_0_90=get("missing", 0.9),
            p_flattened_at_0_50=share_containing("flattened", 0.5),
            p_kept_at_1_00=share_containing("kept", 1.0), p_kept_at_0_90=share_containing("kept", 0.9),
            p_kept_at_1_10=share_containing("kept", 1.1),
            p_reversed_at_minus_0_50=share_containing("reversed", -0.5),
            labels_ever_modal=",".join(sorted(set(P.idxmax(1)))),
            mean_single_name=np.mean(single), mean_single_name_right=np.mean(single_ok),
            mean_right_incl_two_names=np.mean(acc), mean_wrong=np.mean(wrong), mean_undetermined=np.mean(undet),
            max_wrong=np.max(wrong)))
    return pd.DataFrame(out)


def plot(sim, se_p, path):
    rules = list(RULES)
    ses = sorted(sim.se_g.unique())
    order = ["reversed", "reversed or missing", "missing", "missing or flattened", "flattened",
             "flattened or kept", "kept", "kept or steepened", "steepened", "undetermined",
             "no population gap"]
    colours = {"reversed": "#7b3294", "reversed or missing": "#b58bc4", "missing": "#c2a5cf",
               "missing or flattened": "#e0d0e6", "flattened": "#fdb863", "flattened or kept": "#fde0b8",
               "kept": "#5aae61", "kept or steepened": "#a6dba0", "steepened": "#1b7837",
               "undetermined": "#d9d9d9", "no population gap": "#969696"}
    fig, axes = plt.subplots(len(rules), len(ses), figsize=(4 * len(ses), 2.6 * len(rules)),
                             sharex=True, sharey=True)
    for i, rule in enumerate(rules):
        for j, se in enumerate(ses):
            ax = axes[i, j]
            s = sim[(sim.rule == rule) & (sim.se_p == se_p) & (sim.se_g == se)]
            P = s.pivot_table(index="rho", columns="verdict", values="prob", fill_value=0)
            cols = [c for c in order if c in P.columns]
            ax.stackplot(P.index, [P[c] for c in cols], colors=[colours[c] for c in cols],
                         labels=cols, linewidth=0)
            for x in (0, 1):
                ax.axvline(x, color="k", lw=0.6, ls=":")
            if i == 0:
                ax.set_title(f"SE of simulated gap = {se:g} x population gap", fontsize=9)
            if j == 0:
                ax.set_ylabel(rule, fontsize=9)
            if i == len(rules) - 1:
                ax.set_xlabel("true ratio (simulated gap / population gap)", fontsize=8)
            ax.set_ylim(0, 1)
    h, l = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    idx = sorted(range(len(l)), key=lambda k: order.index(l[k]))
    fig.legend([h[k] for k in idx], [l[k] for k in idx], loc="lower center", ncol=6, fontsize=8,
               frameon=False)
    fig.suptitle(f"Probability of each level-2 verdict by true ratio (SE of population gap = {se_p:g})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------------------- inputs
def load_inputs():
    frames = []
    pap = pd.read_csv(REPO / "analysis" / "level2_permodel_equivalence.csv")
    pap = pap[pap.se.notna()]
    frames.append(pd.DataFrame(dict(
        source="paper (PHQ-8)", row=pap.contrast + " | " + pap.scope, scope_type=pap.scope_type,
        g=pap.simulated, se_g=pap.se, p=pap.population, se_p=pap.pop_se, cov=0.0, df=pap.df,
        published_verdict=pap.verdict)))
    oq = pd.read_csv(EXT / "opinionqa" / "level2_contrasts.csv")
    frames.append(pd.DataFrame(dict(
        source="OpinionQA (Meister 2024)",
        row=oq.contrast + " | " + oq.model + " | " + oq.method + " | " + oq.steering,
        scope_type="model x method x steering", g=oq.model_gap, se_g=oq.se_model_gap, p=oq.human_gap,
        se_p=oq.se_human_gap, cov=oq.cov_model_human, df=oq.df, published_verdict=oq.verdict)))
    ar = pd.read_csv(EXT / "argyle" / "level2_contrasts.csv")
    frames.append(pd.DataFrame(dict(
        source="Argyle 2023 Study 3", row=ar.outcome + " | " + ar.contrast + " | " + ar.run,
        scope_type=ar.run, g=ar.silicon_gap, se_g=ar.se_model_gap, p=ar.human_gap,
        se_p=ar.se_human_gap, cov=ar.cov_model_human, df=np.inf, published_verdict=ar.verdict,
        boot_lo=ar.ratio_boot_ci90_lo, boot_hi=ar.ratio_boot_ci90_hi)))
    return pd.concat(frames, ignore_index=True)


def apply_rules(d):
    a = (d.g.values, d.se_g.values, d.p.values, d.se_p.values, d["cov"].values, d.df.values)
    for name, f in RULES.items():
        d[name] = f(*a)
    for m0, m1 in MARGINS:
        d[f"R3_m{m0:.2f}_{m1:.2f}"] = interval_rule(*a, m0=m0, m1=m1)[0]
    d["ratio"] = d.g / d.p
    d["ratio_ci90_lo"], d["ratio_ci90_hi"] = fieller(*a)
    return d


# ---------------------------------------------------------------------------------------- main
if __name__ == "__main__":
    # receipt 1: R0 here reproduces the published verdict column on every source. The paper's
    # pooled rows use BH q in 66, so they are compared on the per-model / per-condition rows,
    # which use raw p there as here.
    d = apply_rules(load_inputs())
    chk = d[~((d.source == "paper (PHQ-8)") & (d.scope_type == "pooled"))]
    agree = (chk.R0_current == chk.published_verdict).groupby(chk.source).mean()
    print("R0 reproduces published verdicts (share of rows):\n", agree.round(4).to_string())
    assert (agree == 1).all(), "R0 does not reproduce the published rule"
    # receipt 2: Fieller against the Argyle respondent bootstrap
    a = d[d.source == "Argyle 2023 Study 3"].dropna(subset=["boot_lo", "ratio_ci90_lo"])
    a = a[np.isfinite(a.ratio_ci90_hi) & (a.ratio_ci90_hi - a.ratio_ci90_lo < 5)]
    print("Fieller vs bootstrap 90%% ratio bounds, Argyle, median |diff|: lo %.3f hi %.3f (n=%d)" % (
        (a.ratio_ci90_lo - a.boot_lo).abs().median(), (a.ratio_ci90_hi - a.boot_hi).abs().median(), len(a)))
    d.to_csv(OUT / "applied_all_rules.csv", index=False)

    # crosstabs, current against each candidate, per source
    with open(OUT / "crosstabs.txt", "w", encoding="utf-8") as fh:
        for src, s in d.groupby("source"):
            if src == "Argyle 2023 Study 3":
                s = s[s.scope_type == "t0.7_main"]
            for r in ["R1_half_bound", "R2_reorder", "R3_interval"]:
                fh.write(f"\n=== {src}: R0_current (rows) x {r} (columns), n={len(s)}\n")
                fh.write(pd.crosstab(s.R0_current, s[r], margins=True).to_string() + "\n")
            fh.write(f"\n--- {src}: R3 margin sensitivity (verdict counts)\n")
            cols = [c for c in d.columns if c.startswith("R3_m")]
            fh.write(pd.concat({c: s[c].value_counts() for c in cols}, axis=1).fillna(0).astype(int).to_string() + "\n")
            # how many rows change label as the margins move (against the 0.25/0.25 default)
            for c in cols:
                fh.write(f"  {c}: {int((s[c] != s['R3_m0.25_0.25']).sum())} of {len(s)} rows differ from m0.25_0.25\n")

    # the paper's own rows, pooled and per model, in full
    p = d[(d.source == "paper (PHQ-8)") & d.scope_type.isin(["pooled", "model"])]
    p[["row", "g", "p", "ratio", "ratio_ci90_lo", "ratio_ci90_hi", "published_verdict",
       "R1_half_bound", "R2_reorder", "R3_interval", "R3_m0.20_0.20", "R3_m0.33_0.33"]].round(3).to_csv(
        OUT / "paper_pooled_and_permodel.csv", index=False)

    sim = simulate()
    sim.to_csv(OUT / "simulation_verdict_probabilities.csv", index=False)
    summ = summarise(sim)
    summ.round(3).to_csv(OUT / "simulation_summary.csv", index=False)
    plot(sim, 0.0, OUT / "fig_operating_characteristics_anchor_known.png")
    plot(sim, 0.15, OUT / "fig_operating_characteristics_anchor_se015.png")
    pd.set_option("display.width", 250)
    print(summ.round(2).to_string(index=False))
