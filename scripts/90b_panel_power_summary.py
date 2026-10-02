"""Summary of the panel power replicates (90_panel_power.py).

Per grid, the rate of each verdict over replicates x pseudo-models:
  gate      REAL pass rate (minimum rule, both framings).
  L2        model verdict per variant and type, under three model rules:
              current   88.l2_model_verdict on the stopped verdicts (conditional stop at >= 3 regions);
              popstop   the same rule on the population-stopped verdicts (a contrast whose reference
                        relative half-width k exceeds kmax = 1/4 is stopped and not judged);
              keptable  pass when every contrast with k <= kmax and a population gap is 'kept';
                        fail when any contrast with a population gap has an unstopped verdict
                        that excludes 'kept'; otherwise unresolved.
            Per contrast: median k and the share of replicates with k <= kmax.
  L4        R1, R2, R4 (.08, .05) and level 4: REAL pass rates and NONINVARIANT fail rates, under
            R2 as run in 88 (congruence only) and under the frozen R2 (congruence and loading RMSD,
            pass / unresolved / fail, 83_controls_lib.l4_eval R2_verdict).
Output: paper_brm/analysis_brm/panel_power/90b_summary.csv, 90b_l2_contrasts.csv
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "..", "paper_brm", "analysis_brm", "panel_power")
KMAX = 0.25
STOPS = ("reference too imprecise", "no population gap")
GRIDS = ["G48", "G144_age", "G144_edu", "G432"]
_B = pd.read_csv(os.path.join(HERE, "..", "analysis", "brm", "80a_tolerance_basis.csv"))
SD0 = float(_B[(_B.window == "2005-2018") & (_B.population == "all adults 18+")].sd.iloc[0])
TOL_MIN, TOL_REC = 0.25 * SD0, 0.125 * SD0   # LADDER_SPEC.md gate: 0.98 and 0.49 PHQ-8 points


def model_rule(verdicts):
    live = [v for v in verdicts if v not in STOPS]
    if any(v != "undetermined" and "kept" not in v.split(" or ") for v in live):
        return "fail"
    if live and all(v == "kept" for v in live):
        return "pass"
    return "unresolved"


def keptable_rule(rows):
    gap = rows[rows.verdict != "no population gap"]
    if any(v != "undetermined" and "kept" not in v.split(" or ") for v in gap.verdict_unstopped):
        return "fail"
    judged = gap[gap.k_rel <= KMAX]
    if len(judged) and (judged.verdict_unstopped == "kept").all():
        return "pass"
    return "unresolved"


def rates(s):
    n = len(s)
    c = s.value_counts()
    return dict(n=n, **{f"rate_{k}": c.get(k, 0) / n for k in ("pass", "fail", "unresolved")})


def main():
    files = sorted(glob.glob(os.path.join(OUTD, "reps", "rep_*.csv")))
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"{len(files)} replicates", flush=True)
    out = []

    def add(grid, rung, type_, rule, s):
        out.append(dict(grid=grid, rung=rung, type=type_, rule=rule, **rates(s)))

    for g in GRIDS:
        x = d[d.grid == g]
        # Frozen gate (precision only), re-read from the stored SE(30): runners that loaded 88 before
        # its gate change (13:41) stored verdicts under the pre-freeze rule (phi(30) >= .80 and SE).
        gt = x[(x.rung == "gate") & (x.type == "REAL")]
        ok_min = (gt.se30_clinical <= TOL_MIN) & (gt.se30_narrative <= TOL_MIN)
        ok_rec = (gt.se30_clinical <= TOL_REC) & (gt.se30_narrative <= TOL_REC)
        add(g, "gate", "REAL", "minimum, both framings", ok_min.map({True: "pass", False: "fail"}))
        add(g, "gate", "REAL", "recommended, both framings", ok_rec.map({True: "pass", False: "fail"}))
        add(g, "gate", "REAL", "stored verdict (pre-freeze rule in early runners)", gt.verdict)

        c = x[(x.rung == "L2") & x.rule.str.startswith("contrast")]
        for (t, rule), cc in c.groupby(["type", "rule"]):
            var = rule[rule.index("[") + 1:-1]
            cur, pop, kep = [], [], []
            for _, m in cc.groupby(["rep", "model"]):
                cur.append(model_rule(list(m.verdict)))
                pop.append(model_rule(list(m.verdict_popstop)))
                kep.append(keptable_rule(m))
            for name, v in (("current", cur), ("popstop", pop), ("keptable", kep)):
                add(g, "L2", t, f"{var} / {name}", pd.Series(v))

        for rung in ("L4-R1", "L4-R2", "L4-R4", "L4"):
            y = x[(x.rung == rung) & (x.unit == "model")]
            for (t, rule), yy in y.groupby(["type", "rule"]):
                add(g, rung, t, rule, yy.verdict)

        # frozen R2 (shape and size, three-way) and level 4 recombined with it
        fr = x[(x.rung == "L4") & (x.rule == "framing")]
        for t, ff in fr.groupby("type"):
            r2 = ff.groupby("rep").R2_verdict.agg(
                lambda v: "fail" if (v == "fail").any() else ("pass" if (v == "pass").all() else "unresolved"))
            add(g, "L4-R2", t, "frozen: shape and size", r2)
            r1 = ff.assign(r1=ff.R1.astype(str) == "True").groupby("rep").r1.all()
            r4 = x[(x.rung == "L4-R4") & (x.type == t) & (x.rule == "both framings, margin .08")].set_index("rep").verdict
            l4 = []
            for rep in r4.index:
                a, b, c = r1[rep], r2[rep], r4[rep]
                l4.append("pass" if (a and b == "pass" and c == "pass") else
                          ("fail" if (not a or b == "fail" or c == "fail") else "unresolved"))
            if l4:
                add(g, "L4", t, "frozen: R1, R2 (shape and size), R4 at .08", pd.Series(l4))

    res = pd.DataFrame(out)
    res.to_csv(os.path.join(OUTD, "90b_summary.csv"), index=False)

    c = d[(d.rung == "L2") & d.rule.str.startswith("contrast") & (d.type == "REAL") & (d.model == "limit")]
    k = c.groupby(["grid", "rule", "unit"]).agg(
        n=("k_rel", "size"), k_median=("k_rel", "median"),
        share_k_le_kmax=("k_rel", lambda v: float(np.mean(v <= KMAX))),
        share_no_gap=("verdict", lambda v: float(np.mean(v == "no population gap"))),
        share_kept_limit=("verdict", lambda v: float(np.mean(v == "kept")))).reset_index()
    k["grid"] = pd.Categorical(k.grid, GRIDS)
    k = k.sort_values(["rule", "unit", "grid"])
    k.to_csv(os.path.join(OUTD, "90b_l2_contrasts.csv"), index=False)

    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(k.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
