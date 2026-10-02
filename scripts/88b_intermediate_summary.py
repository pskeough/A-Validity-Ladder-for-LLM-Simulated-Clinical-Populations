"""Summary of the 88 intermediate-panel replicates.

Reads paper_brm/analysis_brm/intermediate_panel/reps/rep_*.csv, with level 2 from l2_frozen_reps (88e) and
R2 and level 4 from r2_size_reps (88d) under the frozen rules, and writes, in the same folder:
  88_panel_verdicts.csv   every verdict row (type x replicate x rung/rule x model x unit)
  88_confusion.csv        one row per simulator type: pass rate per rung (and fail / unresolved
                          shares for the three-state rungs L1, L2, R4 and level 4), with n
  88_l2_contrasts.csv     per-contrast L2 verdict shares per type, scope and SE/reference variant
  88_r4_steps.csv         R4 step verdict shares and mean RMSEA_D per type, framing and step
  88_l1_readings.csv      L1 per-framing reading shares per type
  88_l3_groups.csv        L3 mean residual and pass share per type and group (per model)
  88_gate.csv             gate statistics per type (per model and framing)
  88_checks.csv           realised shifts and tilt diagnostics
Prints the tables used in INTERMEDIATE_PANEL.md.
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "..", "paper_brm", "analysis_brm", "intermediate_panel")
TYPES = ["REAL", "ORACLE-INDEP", "SHIFTED-2.1", "SHIFTED-4.7", "STEEPENED-2x", "COMPRESSED", "NONINVARIANT"]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def shares(v, states=("pass", "fail", "unresolved")):
    n = len(v)
    out = {"n": n}
    for s in states:
        out[s] = float((v == s).mean()) if n else np.nan
    return out


def combine(vs):
    return "fail" if "fail" in vs else ("pass" if all(v == "pass" for v in vs) else "unresolved")


def frozen_rules(df, R):
    """Replace level 2 with 88e's recomputation and R2 / level 4 with 88d's re-read, both under the
    frozen rules (LADDER_SPEC.md). Level 2 rows are exact recomputations on the same data; R2 adds
    the loading-size condition and reads pass / unresolved / fail."""
    def load(sub):
        f = sorted(glob.glob(os.path.join(OUTD, sub, "rep_*.csv")))
        assert len(f) == R, f"{sub} has {len(f)} of {R} replicates"
        return pd.concat([pd.read_csv(x, low_memory=False, keep_default_na=False, na_values=[""]) for x in f],
                         ignore_index=True)

    # gate: precision only, SE(30) <= 0.25 and 0.125 NHANES SD in both framings, from the stored SEs
    tb = pd.read_csv(os.path.join(OUTD, "..", "..", "..", "analysis", "brm", "80a_tolerance_basis.csv"))
    sd0 = float(tb[(tb.window == "2005-2018") & (tb.population == "all adults 18+")].sd.iloc[0])
    g = df.rung == "gate"
    se = df.loc[g, ["se30_clinical", "se30_narrative"]].apply(pd.to_numeric).max(axis=1)
    df.loc[g, "verdict"] = np.where(se <= 0.25 * sd0, "pass", "fail")
    df.loc[g, "pass_rec"] = np.where(se <= 0.125 * sd0, "True", "False")

    l2 = load("l2_frozen_reps")
    l2["model"] = l2.model.astype(str)
    df = pd.concat([df[df.rung != "L2"], l2], ignore_index=True)

    r2 = load("r2_size_reps")
    r2v = r2.groupby(["rep", "type"]).R2_verdict.agg(lambda v: combine(list(v)))
    r1 = r2.assign(r1=r2.R1.astype(str) == "True").groupby(["rep", "type"]).r1.all()
    is_r2 = (df.rung == "L4-R2") & (df.unit == "model")
    df.loc[is_r2, "verdict"] = [r2v[(r, t)] for r, t in zip(df.rep[is_r2], df.type[is_r2])]
    for e in ("08", "05"):
        r4 = df[(df.rung == "L4-R4") & (df.rule == f"both framings, margin .{e}")].set_index(["rep", "type"]).verdict
        is_l4 = (df.rung == "L4") & (df.rule == f"level 4 (R1, R2, R4 at .{e})")
        new = []
        for r, t in zip(df.rep[is_l4], df.type[is_l4]):
            a, b, c = r1[(r, t)], r2v[(r, t)], r4[(r, t)]
            new.append("pass" if (a and b == "pass" and c == "pass") else
                       ("fail" if (not a or b == "fail" or c == "fail") else "unresolved"))
        df.loc[is_l4, "verdict"] = new
    return df


def main():
    files = sorted(glob.glob(os.path.join(OUTD, "reps", "rep_*.csv")))
    df = pd.concat([pd.read_csv(f, low_memory=False, keep_default_na=False, na_values=[""]) for f in files],
                   ignore_index=True)
    df["model"] = df.model.astype(str)
    R = df.rep.nunique()
    print(f"{R} replicates ({df.rep.min()}..{df.rep.max()})")
    df = frozen_rules(df, R)
    df.to_csv(os.path.join(OUTD, "88_panel_verdicts.csv"), index=False)
    pm = df.model.isin(["0", "1", "2", "3"])
    tf = {"True": True, "False": False, True: True, False: False}

    rows = []
    for t in TYPES:
        x = df[df.type == t]
        r = {"type": t}
        g = x[(x.rung == "gate") & pm.loc[x.index]]
        r["gate_min_pass"], r["gate_n"] = (g.verdict == "pass").mean(), len(g)
        r["gate_rec_pass"] = g.pass_rec.map(tf).mean()
        l1 = x[(x.rung == "L1") & (x.unit == "model")]
        s = shares(l1.verdict)
        r.update({"L1_pass": s["pass"], "L1_fail": s["fail"], "L1_unresolved": s["unresolved"], "L1_n": s["n"]})
        for var, tag in (("pairs_half", ""), ("pairs_auditprec", "_auditprec"), ("cond_half", "_cond"),
                         ("cond_auditprec", "_cond_auditprec")):
            l2 = x[(x.rung == "L2") & (x.rule == f"model [{var}]") & pm.loc[x.index]]
            s = shares(l2.verdict)
            r.update({f"L2{tag}_pass": s["pass"], f"L2{tag}_fail": s["fail"], f"L2{tag}_unresolved": s["unresolved"]})
            if tag == "":
                r["L2_n"] = s["n"]
        l2p = x[(x.rung == "L2") & (x.rule == "model [pairs_half]") & (x.model == "pooled")]
        s = shares(l2p.verdict)
        r.update({"L2_pooled_pass": s["pass"], "L2_pooled_fail": s["fail"], "L2_pooled_unresolved": s["unresolved"]})
        l3 = x[(x.rung == "L3") & (x.unit == "model") & pm.loc[x.index]]
        r["L3_pass_2pt"], r["L3_n"] = (l3.verdict == "pass").mean(), len(l3)
        r["L3_pass_1pt"] = l3.pass_1pt.map(tf).mean()
        for k in ("R1", "R2", "R3"):
            v = x[(x.rung == f"L4-{k}")]
            r[f"L4_{k}_pass"] = (v.verdict == "pass").mean()
        r["L4_n"] = len(x[x.rung == "L4-R1"])
        for e in ("08", "05"):
            v = x[(x.rung == "L4-R4") & (x.rule == f"both framings, margin .{e}")].copy()
            # a framing without a general factor gets no R4 verdict (the rung's rule); 88 scored it as
            # fail in the both-framing row, so it is relabelled here
            v["verdict"] = np.where(v.detail.str.contains("no verdict"), "no verdict", v.verdict)
            s = shares(v.verdict, ("pass", "fail", "unresolved", "no verdict"))
            r.update({f"R4_e{e}_pass": s["pass"], f"R4_e{e}_fail": s["fail"], f"R4_e{e}_unresolved": s["unresolved"],
                      f"R4_e{e}_noverdict": s["no verdict"], f"R4_e{e}_n": s["n"]})
            if s["n"]:
                lo, hi = wilson(int((v.verdict == "fail").sum()), s["n"])
                r[f"R4_e{e}_fail_lo95"], r[f"R4_e{e}_fail_hi95"] = lo, hi
            v = x[(x.rung == "L4") & (x.rule == f"level 4 (R1, R2, R4 at .{e})")]
            s = shares(v.verdict)
            r.update({f"L4level_e{e}_pass": s["pass"], f"L4level_e{e}_fail": s["fail"],
                      f"L4level_e{e}_unresolved": s["unresolved"]})
        # R4 no-verdict share among framings (no general factor)
        fr = x[(x.rung == "L4") & (x.rule == "framing") & x.R4_e08.notna()]
        r["R4_framings_no_factor"] = float(fr.R4_e08.str.startswith("no verdict").mean()) if len(fr) else np.nan
        rows.append(r)
    conf = pd.DataFrame(rows)
    conf.to_csv(os.path.join(OUTD, "88_confusion.csv"), index=False, float_format="%.4f")

    # ---------------------------------------------------------------- L2 per contrast
    c2 = df[(df.rung == "L2") & df.rule.str.startswith("contrast")].copy()
    c2["variant"] = c2.rule.str.extract(r"\[(.*)\]")[0]
    c2["scope"] = np.where(c2.model == "pooled", "pooled", "per model")
    c2["kept_alone"] = c2.verdict == "kept"
    c2["steep_alone"] = c2.verdict == "steepened"
    c2["has_steep"] = c2.verdict.str.contains("steepened")
    c2["excludes_kept"] = ~c2.verdict.isin(["undetermined", "reference too imprecise", "no population gap"]) & \
        ~c2.verdict.str.split(" or ").map(lambda z: "kept" in z)
    c2["stopped"] = c2.verdict.isin(["reference too imprecise", "no population gap"])
    c2["undetermined"] = c2.verdict == "undetermined"
    t2 = c2.groupby(["type", "variant", "scope", "unit"]).agg(
        n=("verdict", "size"), kept_alone=("kept_alone", "mean"), steepened_alone=("steep_alone", "mean"),
        names_steepened=("has_steep", "mean"), excludes_kept=("excludes_kept", "mean"),
        undetermined=("undetermined", "mean"), stopped=("stopped", "mean"), ratio_mean=("ratio", "mean"),
        gap_full_sample=("g_true", "first")).reset_index()
    t2.to_csv(os.path.join(OUTD, "88_l2_contrasts.csv"), index=False, float_format="%.4f")

    # ---------------------------------------------------------------- R4 steps
    st = df[df.rung == "L4-R4 step"].copy()
    if len(st):
        st["fails08"], st["holds08"] = st.v08 == "fails", st.v08 == "holds"
        st["fails05"], st["holds05"] = st.v05 == "fails", st.v05 == "holds"
        t4 = st.groupby(["type", "rule"]).agg(n=("v08", "size"), reps=("rep", "nunique"),
                                              rmsea_d_mean=("rmsea_d", "mean"), lo_mean=("lo", "mean"),
                                              hi_mean=("hi", "mean"), holds_08=("holds08", "mean"),
                                              fails_08=("fails08", "mean"), holds_05=("holds05", "mean"),
                                              fails_05=("fails05", "mean"), p_perm_mean=("p_perm", "mean"),
                                              converged=("converged", "mean")).reset_index()
        t4.to_csv(os.path.join(OUTD, "88_r4_steps.csv"), index=False, float_format="%.4f")
    else:
        t4 = pd.DataFrame()

    # ---------------------------------------------------------------- L1 readings
    l1 = df[(df.rung == "L1") & (df.rule == "framing")]
    t1 = l1.groupby(["type"]).verdict.value_counts(normalize=True).rename("share").reset_index()
    m1 = l1.groupby("type")[["misfit_ratio", "overfit_ratio"]].mean().reset_index()
    t1 = t1.merge(m1, on="type")
    t1.to_csv(os.path.join(OUTD, "88_l1_readings.csv"), index=False, float_format="%.4f")

    # ---------------------------------------------------------------- L3 groups
    l3 = df[(df.rung == "L3") & (df.rule == "group") & pm]
    t3 = l3.groupby(["type", "unit"]).agg(n=("resid", "size"), resid_mean=("resid", "mean"),
                                          se_mean=("se", "mean"),
                                          pass_2pt=("verdict", lambda v: (v == "pass").mean())).reset_index()
    t3.to_csv(os.path.join(OUTD, "88_l3_groups.csv"), index=False, float_format="%.4f")

    # ---------------------------------------------------------------- gate
    g = df[(df.rung == "gate") & pm]
    gcols = [c for c in g.columns if any(c.startswith(p) for p in ("phi30_", "phi1_", "se30_", "s2p_", "s2e_",
                                                                     "k_min_", "k_rec_"))]
    tg = g.groupby("type")[gcols].median().reset_index()
    tg.to_csv(os.path.join(OUTD, "88_gate.csv"), index=False, float_format="%.4f")

    # ---------------------------------------------------------------- checks
    ck = df[df.rung == "check"]
    cols = [c for c in ["shift_total", "shift_low", "shift_notlow", "realised_min", "realised_mean", "ess_min",
                        "ess_median", "n_cells"] if c in ck]
    tc = ck.groupby(["type", "rule"])[cols].agg(["mean", "min"]).reset_index()
    tc.columns = ["_".join(c).strip("_") for c in tc.columns]
    tc.to_csv(os.path.join(OUTD, "88_checks.csv"), index=False, float_format="%.4f")

    # ---------------------------------------------------------------- print
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    show = ["type", "gate_min_pass", "L1_pass", "L1_fail", "L1_unresolved", "L2_pass", "L2_fail", "L2_unresolved",
            "L2_cond_auditprec_pass", "L2_cond_auditprec_fail", "L3_pass_2pt", "L4_R1_pass", "L4_R2_pass",
            "L4_R3_pass", "R4_e08_pass", "R4_e08_fail", "R4_e08_unresolved", "R4_e08_noverdict", "R4_e08_n", "R4_e05_fail",
            "L4level_e08_pass"]
    print("\nCONFUSION")
    print(conf[show].round(3).to_string(index=False))
    print("\nn:", conf[["type", "gate_n", "L1_n", "L2_n", "L3_n", "L4_n", "R4_e08_n"]].to_string(index=False))
    print("\nL2, Low minus High and Low minus Middle, per model")
    z = t2[(t2.scope == "per model") & t2.unit.isin(["Low minus High SES", "Low minus Middle SES"])]
    print(z.round(3).to_string(index=False))
    print("\nL2 pooled, income contrasts")
    z = t2[(t2.scope == "pooled") & t2.unit.str.contains("SES")]
    print(z.round(3).to_string(index=False))
    if len(t4):
        print("\nR4 steps")
        print(t4.round(3).to_string(index=False))
    print("\nL1 readings")
    print(t1.round(3).to_string(index=False))
    print("\nL3 groups (per model)")
    print(t3[t3.unit.isin(["Overall", "Low", "Middle", "High"])].round(3).to_string(index=False))
    print("\nGate (medians)")
    print(tg.round(3).to_string(index=False))
    print("\nChecks")
    print(tc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
