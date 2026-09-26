"""Link the gate's G-study to the decoding control: variance components and the D-study at
temperature 0 against provider defaults on the 12 control cohorts. No API calls; reads the existing
decoding-control log through decoding_control_io (the same loader, dedup and pairing as 56, 57, 60).

For each endpoint and arm, a one-facet design (cohort p, draws nested, 30 per cell) gives s2_p and
s2_e, phi(k) = s2_p / (s2_p + s2_e / k), the k for phi >= .80 and .90, and the k for SE <= 0.5 and
1.0 point. The released clinical corpus on the same 12 cohorts is the third column, and the
120-persona clinical figures from 82a are carried alongside to show how the 12-cohort sample
compares with the full grid (the 12 were drawn one per stratum of the pooled clinical mean, so
their between-cohort spread need not match any one model's full-grid spread).

Intervals: percentile bootstrap over the 12 cohorts, B = 2000, seeded; a resampled cohort carries
both arms and its corpus cell, so every contrast stays paired.

Emits analysis/brm/gate_decoding_link.csv.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decoding_control_io import balanced_cells, both_arms_only, load  # noqa: E402

_spec = importlib.util.spec_from_file_location("gl", os.path.join(HERE, "82_gate_lib.py"))
gl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gl)

B = 2000
SEED = 20260926
CTRL2CORPUS = {"gpt-4o-mini": "openai/gpt-4o-mini", "deepseek-chat": "deepseek/deepseek-chat-v3",
               "gemini-3-flash-preview": "google/gemini-3-flash-preview", "glm-4.7": "z-ai/glm-4.7"}


def summaries(df, cohorts):
    """Arrays (mean, n, ss) over the ordered cohort list."""
    g = df.groupby("profile_id").total
    s = g.agg(n="size", mean="mean", var=lambda x: x.var(ddof=1)).reindex(cohorts)
    return s["mean"].to_numpy(float), s["n"].to_numpy(float), (s["var"] * (s["n"] - 1)).to_numpy(float)


def flip(df):
    out = []
    for _, c in df.groupby("profile_id"):
        b = np.digitize(c.total.to_numpy(), gl.CUTS)
        cnt = np.bincount(b, minlength=5)
        n = len(b)
        out.append(1 - (cnt * (cnt - 1)).sum() / (n * (n - 1)))
    return 100 * np.mean(out)


def dstudy(c):
    return dict(s2_p=c["p"], s2_e=c["e"], within_sd=np.sqrt(gl.pos(c["e"])),
                phi_1=gl.phi_k(c["p"], c["e"], 1), phi_30=gl.phi_k(c["p"], c["e"], 30),
                k_phi80=gl.k_for_phi(c["p"], c["e"], 0.80), k_phi90=gl.k_for_phi(c["p"], c["e"], 0.90),
                k_se05=gl.k_for_se(c["e"], 0.5), k_se10=gl.k_for_se(c["e"], 1.0))


def main():
    dc, _ = load(verbose=False)
    dc = balanced_cells(both_arms_only(dc, verbose=False), verbose=False)
    v = gl.load_corpus()
    vc = v[v.framing == "clinical"]
    d120 = pd.read_csv(os.path.join(gl.OUTD, "gate_dstudy.csv"))
    d120 = d120[(d120.outcome == "PHQ-8 total") & (d120.framing == "clinical")].set_index("model")

    rng = np.random.default_rng(SEED)
    rows = []
    for cm, full in CTRL2CORPUS.items():
        g = dc[dc.model == cm]
        cohorts = sorted(g.profile_id.unique())
        assert len(cohorts) == 12
        src = {"default": g[g.arm == "default"], "temp0": g[g.arm == "temp0"],
               "corpus_clinical": vc[(vc.model == full) & vc.profile_id.isin(cohorts)].rename(columns={"y": "total"})}
        arr = {k: summaries(x, cohorts) for k, x in src.items()}
        point = {k: gl.one_facet(*a) for k, a in arr.items()}
        boots = {k: [] for k in arr}
        for _ in range(B):
            ix = rng.integers(0, 12, 12)
            for k, a in arr.items():
                boots[k].append(gl.one_facet(*(x[ix] for x in a)))
        for k in arr:
            rec = dict(model=gl.SHORT[full], source=k, n_cohorts=12, n_draws=point[k]["n_draws"],
                       grand_mean=point[k]["grand"], band_flip_pct=flip(src[k]))
            rec.update(dstudy(point[k]))
            bp = np.array([b["p"] for b in boots[k]])
            be = np.array([b["e"] for b in boots[k]])
            rec["s2_e_lo"], rec["s2_e_hi"] = np.percentile(be, [2.5, 97.5])
            rec["s2_p_lo"], rec["s2_p_hi"] = np.percentile(bp, [2.5, 97.5])
            bk = [gl.k_for_phi(a, e, 0.90) for a, e in zip(bp, be)]
            rec["k_phi90_lo"], rec["k_phi90_hi"] = np.percentile(bk, [2.5, 97.5], method="inverted_cdf")
            bk = [gl.k_for_se(e, 0.5) for e in be]
            rec["k_se05_lo"], rec["k_se05_hi"] = np.percentile(bk, [2.5, 97.5], method="inverted_cdf")
            if k == "temp0":
                bd = np.array([b["e"] for b in boots["default"]])
                ratio = be / np.where(bd > 0, bd, np.nan)
                rec["s2_e_ratio_to_default"] = point["temp0"]["e"] / point["default"]["e"]
                rec["ratio_lo"], rec["ratio_hi"] = np.nanpercentile(ratio, [2.5, 97.5])
            if k == "corpus_clinical":
                r120 = d120.loc[gl.SHORT[full]]
                rec["full_grid_s2_p"] = r120.s2_p
                rec["full_grid_s2_e"] = r120.s2_e
                rec["full_grid_k_phi90"] = r120.k_phi90
            one_valued = (src[k].groupby("profile_id").total.nunique() == 1).mean()
            rec["pct_one_valued_cells"] = 100 * one_valued
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(gl.OUTD, "gate_decoding_link.csv"), index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
