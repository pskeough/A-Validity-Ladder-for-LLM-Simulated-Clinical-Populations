"""The audit's level 2 verdicts under the two simulated-gap SE conventions.

pairs        78b: SD of persona-pair differences / sqrt(pairs), df pairs - 1 (published verdicts)
conditional  draw variance only, conditional on the fixed persona set (level 3's convention),
             sqrt(sum over cells coef^2 (v_clin + v_narr) / 4), df infinite; pooled: sqrt(sum se^2) / K
Reference: the published standardised NHANES estimate with its Taylor SE and df (l2_reference.csv).
The pairs column reproduces l2_verdicts.csv (checked). Emits analysis/brm/83g_l2_conditional_verdicts.csv.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c83", os.path.join(HERE, "83_controls_lib.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
OUT = C.OUTD
MODELS = ["gpt-4o-mini", "gemini-3-flash-preview", "deepseek-chat-v3", "glm-4.7"]

pub = pd.read_csv(os.path.join(OUT, "l2_reference.csv"))
pub = pub[(pub.estimand == "standardised") & (pub.variant == "primary")].set_index("contrast")
ref = {k: (pub.loc[k, "estimate"], pub.loc[k, "se_taylor"], pub.loc[k, "df"]) for k in pub.index}
ver = pd.read_csv(os.path.join(OUT, "l2_verdicts.csv"))
ver = ver[(ver.analysis == "headline") & (ver.estimand == "standardised")]

key = C.load_personas().set_index("profile_id").c48
cells = pd.read_csv(os.path.join(OUT, "80b_sim_cells.csv"))
cells = cells[cells.corpus == "all"]
sims = {}
for m in MODELS:
    s, v = {}, {}
    for f in ("clinical", "narrative"):
        x = cells[(cells.model == m) & (cells.framing == f)].set_index("profile_id")
        a, b = np.full(48, np.nan), np.full(48, np.nan)
        for pid, r in x.iterrows():
            if key[pid] >= 0:
                a[key[pid]], b[key[pid]] = r["mean"], r["var"] / r.n
        s[f], v[f] = a, b
    sims[m] = C.l2_sim(s["clinical"], s["narrative"], v["clinical"], v["narrative"])
K = len(MODELS)
scopes = {m: (sims[m], C.l2_conditional(sims[m])) for m in MODELS}
scopes["pooled"] = (C.l2_pool([sims[m] for m in MODELS]),
                    {k: (np.mean([sims[m][k][0] for m in MODELS]),
                         float(np.sqrt(np.sum([sims[m][k][3] ** 2 for m in MODELS]))) / K, np.inf)
                     for k in sims[MODELS[0]]})
rows = []
for scope, (sp, sc) in scopes.items():
    for name, *_ in C.L2_CONTRASTS:
        op, oc = C.l2_verdict(sp, ref, name), C.l2_verdict(sc, ref, name)
        pv = ver[ver.contrast == name]
        pv = pv[pv.scope.str.endswith(scope)] if scope != "pooled" else pv[pv.scope.str.contains("pool")]
        rows.append(dict(scope=scope, contrast=name, sim_gap=sp[name][0], se_pairs=sp[name][1], se_cond=sc[name][1],
                         ratio=op["ratio"], verdict_pairs=op["verdict"],
                         published=pv.verdict.iloc[0] if len(pv) else "",
                         ci_pairs=f"[{op['ci_lo']:.2f}, {op['ci_hi']:.2f}]", verdict_conditional=oc["verdict"],
                         ci_conditional=f"[{oc['ci_lo']:.2f}, {oc['ci_hi']:.2f}]"))
out = pd.DataFrame(rows)
out["changed"] = out.verdict_pairs != out.verdict_conditional
out.to_csv(os.path.join(OUT, "83g_l2_conditional_verdicts.csv"), index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string(index=False))
print("pairs reproduces published:", bool((out.verdict_pairs == out.published).all()))
