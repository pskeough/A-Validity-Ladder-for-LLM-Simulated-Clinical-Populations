"""Is 78b's simulated-gap SE (SD of persona-pair differences / sqrt(pairs)) the right size for a fixed
persona design? Within one 83b replicate the four pseudo-models share one donor half, so the
spread of their gaps is pure draw-to-draw noise, the only sampling a fixed persona design has.
Compare it with the mean 78b SE, and with the conditional SE implied by the draws alone,
sqrt(sum over cells of coef^2 var(cell mean)), the convention level 3 uses.

Also reads the real corpus: 78b SE against the conditional SE for each model and contrast.
Emits analysis/brm/83d_l2_se_check.csv.
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c83", os.path.join(HERE, "83_controls_lib.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
OUT = C.OUTD

files = sorted(glob.glob(os.path.join(OUT, "83b_reps", "rep_*.csv")))
df = pd.concat([pd.read_csv(f, low_memory=False, keep_default_na=False, na_values=[""]) for f in files])
l2 = df[(df.rung == "L2") & (df.condition == "null") & (df.model != "pooled")].copy()
l2["sim_gap"] = pd.to_numeric(l2.sim_gap)
l2["sim_se"] = pd.to_numeric(l2.sim_se)
rows = []
for name, g in l2.groupby("unit"):
    within = g.groupby("rep").sim_gap.var(ddof=1)          # across the 4 pseudo-models, same donors
    rows.append(dict(source="83b pseudo-models", contrast=name, n_reps=g.rep.nunique(),
                     empirical_sd_within_rep=float(np.sqrt(within.mean())),
                     mean_se_78b=float(g.sim_se.mean())))
emp = pd.DataFrame(rows)
emp["se_78b_over_empirical"] = emp.mean_se_78b / emp.empirical_sd_within_rep

# real corpus: 78b SE vs conditional SE from within-cell draw variance
pers = C.load_personas()
key = pers.set_index("profile_id").c48
cells = pd.read_csv(os.path.join(OUT, "80b_sim_cells.csv"))
cells = cells[cells.corpus == "all"]
real = []
for mdl in ["deepseek-chat-v3", "gemini-3-flash-preview", "glm-4.7", "gpt-4o-mini"]:
    s, v = {}, {}
    for f in ("clinical", "narrative"):
        x = cells[(cells.model == mdl) & (cells.framing == f)].set_index("profile_id")
        a, b = np.full(48, np.nan), np.full(48, np.nan)
        for pid, r in x.iterrows():
            if key[pid] >= 0:
                a[key[pid]], b[key[pid]] = r["mean"], r["var"] / r.n
        s[f], v[f] = a, b
    sim = C.l2_sim(s["clinical"], s["narrative"])
    vc = (v["clinical"] + v["narrative"]) / 4                  # variance of the framing-averaged cell mean
    for name, dim, hi, lo in C.L2_CONTRASTS:
        coef, _ = C.l2_coef(dim, hi, lo)
        se_cond = float(np.sqrt(np.nansum(coef ** 2 * vc)))
        real.append(dict(source=f"corpus {mdl}", contrast=name, mean_se_78b=sim[name][1], se_conditional=se_cond,
                         se_78b_over_conditional=sim[name][1] / se_cond))
out = pd.concat([emp, pd.DataFrame(real)], ignore_index=True)
out.to_csv(os.path.join(OUT, "83d_l2_se_check.csv"), index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
