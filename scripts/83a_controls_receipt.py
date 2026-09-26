"""Receipt for script 83: the simplified rung evaluators in 83_controls_lib reproduce the published
rung results on the real corpus, with the full NHANES sample as the reference.

  L1  matched misfit / overfit ratio points and the verdict, per model (framings pooled), against
      l1_personfit_main.csv (2005-2018 weighted GRM, full sample, subset all). Intervals use B = 400
      here against 2,000 there, so interval limits are compared loosely and verdicts exactly.
  L2  standardised reference estimates against l2_reference.csv (primary); jackknife SE against its
      Taylor SE; per-model simulated gaps and SEs against l2_sim_gaps.csv; per-model verdicts
      against l2_verdicts.csv (headline, standardised).
  L3  PS residual and total SE per model and group against 80c_l3_results.csv (S1 primary, all,
      mean, combined).
  L4  R1, phi and R3 per model x framing against l4_summary.csv; phi's lower bound uses B = 200
      here against 1,000 there.

Emits analysis/brm/83a_receipt.csv. Seeded (20260930).
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c83", os.path.join(HERE, "83_controls_lib.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

rng = np.random.default_rng(20260930)
OUT = C.OUTD
MODELS = {"deepseek/deepseek-chat-v3": "DeepSeek-V3", "google/gemini-3-flash-preview": "Gemini-3-Flash",
          "z-ai/glm-4.7": "GLM-4.7", "openai/gpt-4o-mini": "GPT-4o-mini"}
rows = []


def add(rung, item, got, want, tol):
    ok = (got == want) if isinstance(want, (str, bool, np.bool_)) else bool(abs(float(got) - float(want)) <= tol)
    rows.append(dict(rung=rung, item=item, got=got, want=want, tol=tol, ok=ok))


t0 = time.time()
d = C.load_frame()
pers = C.load_personas()
print(f"frame {len(d)} adults, {int((d.c48 >= 0).sum())} in the 48 anchored cells; personas {len(pers)}")

# --------------------------------------------------------------------------------------------- L1
sc = pd.read_csv(os.path.join(OUT, "l1_scores_corpus.csv"), low_memory=False)
main = pd.read_csv(os.path.join(OUT, "l1_personfit_main.csv"))
main = main[(main.reference == "2005_2018 weighted GRM") & (main["sample"] == "full") &
            (main.framing == "both") & (main.subset == "all")].set_index("model")
ref1 = C.L1Ref(d, 400, rng)
for m, short in MODELS.items():
    x = sc[sc.model == m]
    persona = pd.factorize(x.profile_id)[0]
    r = C.l1_eval(ref1, x.total.to_numpy(int), x.lzstar_weighted.to_numpy(), persona, rng)
    w = main.loc[short]
    add("L1", f"{short} misfit ratio", r["misfit_ratio"], w.misfit_ratio, 1e-9)
    add("L1", f"{short} overfit ratio", r["overfit_ratio"], w.overfit_ratio, 1e-9)
    add("L1", f"{short} misfit ci90 hi", r["misfit_ratio_ci90_hi"], w.misfit_ratio_ci90_hi, 0.08)  # B = 400 vs 2,000; checked at B = 2,000 (within 0.02)
    add("L1", f"{short} overfit ci90 hi", r["overfit_ratio_ci90_hi"], w.overfit_ratio_ci90_hi, 0.08)
    want = C.PF.read_verdict(w, C.TAU)[2]
    add("L1", f"{short} verdict", r["verdict"], want, 0)
print(f"L1 done {time.time() - t0:.0f}s")

# --------------------------------------------------------------------------------------------- L2
cref = C.CellRef(d)
l2ref = C.l2_reference(cref)
pub = pd.read_csv(os.path.join(OUT, "l2_reference.csv"))
pub = pub[(pub.estimand == "standardised") & (pub.variant == "primary")].set_index("contrast")
for name, (est, se, df) in l2ref.items():
    add("L2", f"reference {name} estimate", est, pub.loc[name, "estimate"], 1e-4)
    add("L2", f"reference {name} JK/Taylor SE ratio", se / pub.loc[name, "se_taylor"], 1.0, 0.10)
    add("L2", f"reference {name} df", df, pub.loc[name, "df"], 1)  # 78a: 108 for Asian (2011-2018 PSUs)

cells = pd.read_csv(os.path.join(OUT, "80b_sim_cells.csv"))
cells = cells[cells.corpus == "all"]
key = pers.set_index("profile_id").c48
gaps = pd.read_csv(os.path.join(OUT, "l2_sim_gaps.csv"))
gaps = gaps[(gaps.corpus == "all") & (gaps.panel == "four models") & (gaps.framing == "combined") &
            (gaps.race_set.fillna("n/a").isin(["four", "n/a"]))]
ver = pd.read_csv(os.path.join(OUT, "l2_verdicts.csv"))
ver = ver[(ver.analysis == "headline") & (ver.estimand == "standardised")]
sim_cells = {}
for m in MODELS:
    mm = m.split("/")[1]
    s = {}
    for f in ("clinical", "narrative"):
        x = cells[(cells.model == mm) & (cells.framing == f)].set_index("profile_id")
        a = np.full(48, np.nan)
        v = np.full(48, np.nan)
        for pid, r in x.iterrows():
            if key[pid] >= 0:
                a[key[pid]] = r["mean"]
                v[key[pid]] = r["var"] / r.n
        s[f] = (a, v)
    sim_cells[m] = s
    sim = C.l2_sim(s["clinical"][0], s["narrative"][0])
    for name, _, _, _ in C.L2_CONTRASTS:
        g = gaps[(gaps.scope == m) & (gaps.contrast == name)].iloc[0]
        add("L2", f"{MODELS[m]} {name} sim gap", sim[name][0], g.simulated, 1e-5)
        add("L2", f"{MODELS[m]} {name} sim se", sim[name][1], g.se, 1e-5)
        out = C.l2_verdict(sim, l2ref, name)
        vv = ver[(ver.scope == m) & (ver.contrast == name)].iloc[0]
        add("L2", f"{MODELS[m]} {name} verdict", out["verdict"], vv.verdict, 0)
    sim_cells[m]["sim"] = sim
pooled = C.l2_pool([sim_cells[m]["sim"] for m in MODELS])
for name, _, _, _ in C.L2_CONTRASTS:
    g = gaps[(gaps.scope == "pooled") & (gaps.contrast == name)].iloc[0]
    add("L2", f"pooled {name} sim gap", pooled[name][0], g.simulated, 1e-5)
    add("L2", f"pooled {name} sim se", pooled[name][1], g.se, 1e-5)
    add("L2", f"pooled {name} sim df", pooled[name][2], g.df, 1e-2)
    vv = ver[(ver.scope == "pooled") & (ver.contrast == name)].iloc[0]
    add("L2", f"pooled {name} verdict", C.l2_verdict(pooled, l2ref, name)["verdict"], vv.verdict, 0)
print(f"L2 done {time.time() - t0:.0f}s")

# --------------------------------------------------------------------------------------------- L3
l3 = pd.read_csv(os.path.join(OUT, "80c_l3_results.csv"))
l3 = l3[(l3.spec == "S1 primary") & (l3.corpus == "all") & (l3.outcome == "mean") &
        (l3.framing == "combined") & (l3.estimand == "PS")]
for m in MODELS:
    (sc_, vc), (sn, vn) = sim_cells[m]["clinical"], sim_cells[m]["narrative"]
    res = C.l3_eval(cref, (sc_ + sn) / 2, (vc + vn) / 4)
    for r in res:
        w = l3[(l3.model == m.split("/")[1]) & (l3.group == r["group"])].iloc[0]
        add("L3", f"{MODELS[m]} {r['group']} resid", r["resid"], w.resid, 1e-4)
        add("L3", f"{MODELS[m]} {r['group']} se", r["se"], w.se, 1e-4)
    for tn, tol in C.L3_TOLS.items():
        want = bool(l3[l3.model == m.split("/")[1]].apply(
            lambda y: (y.ci90_lo > -tol) and (y.ci90_hi < tol), axis=1).all())
        add("L3", f"{MODELS[m]} all-groups pass at {tn}", C.l3_pass(res, 0.0, tol), want, 0)
print(f"L3 done {time.time() - t0:.0f}s")

# --------------------------------------------------------------------------------------------- L4
l4ref = C.L4Ref(d, 200, rng)
st = pd.read_csv(os.path.join(OUT, "l4_structure.csv"))
nh = st[st.source == "NHANES"].iloc[0]
add("L4", "NHANES loadings min", l4ref.lam.min(), nh.load_min, 1e-4)
add("L4", "NHANES ev ratio", l4ref.ev[0] / l4ref.ev[1], nh.ev_ratio_12, 1e-3)
summ = pd.read_csv(os.path.join(OUT, "l4_summary.csv"))
summ = summ[summ.subset == "all"].set_index(["model", "framing"])
v3 = pd.read_csv(os.path.join(C.BASE, "data", "model_outputs_v3.csv"), low_memory=False)
v3 = v3[v3.phq8_valid.astype(bool)]
for m, short in MODELS.items():
    for f in ("clinical", "narrative"):
        x = v3[(v3.model == m) & (v3.prompt_condition == f)]
        r = C.l4_eval(x[C.L4.ITEMS].to_numpy(int), pd.factorize(x.profile_id)[0], l4ref, rng)
        w = summ.loc[(short, f)]
        add("L4", f"{short} {f} ev ratio", r["ev_ratio"], w.ev_ratio_12, 1e-3)
        add("L4", f"{short} {f} R1", r["R1"], bool(w.R1_general_factor), 0)
        add("L4", f"{short} {f} phi", r["phi"], w.phi_nhanes, 1e-3)
        add("L4", f"{short} {f} phi lo90", r["phi_lo90"], w.phi_lo90, 0.05)  # B = 200 vs 1,000
        add("L4", f"{short} {f} R2", r["R2"], bool(w.R2_loadings), 0)
        add("L4", f"{short} {f} within ev ratio", r["within_ev_ratio"], w.within_ev_ratio, 1e-3)
        add("L4", f"{short} {f} R3", r["R3"], bool(w.R3_factor_source), 0)
print(f"L4 done {time.time() - t0:.0f}s")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "83a_receipt.csv"), index=False)
pd.set_option("display.width", 200)
print(out.groupby("rung").ok.agg(["sum", "size"]))
bad = out[~out.ok]
print("mismatches:" if len(bad) else "all checks pass")
if len(bad):
    print(bad.to_string(index=False))
