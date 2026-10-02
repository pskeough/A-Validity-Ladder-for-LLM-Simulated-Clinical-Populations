"""Per-framing counts from the Bisbee ladder outputs (bisbee_ladder.py), for the manuscript's receipts.
Writes ../results/bisbee/summary_{model}.csv: one row per model x framing."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results", "bisbee")
model = sys.argv[1] if len(sys.argv) > 1 else "rr1"
gt = pd.read_csv(os.path.join(R, f"gate_{model}.csv"))
l2 = pd.read_csv(os.path.join(R, f"level2_contrasts_{model}.csv"))
l2m = pd.read_csv(os.path.join(R, f"level2_model_{model}.csv"))
l3 = pd.read_csv(os.path.join(R, f"level3_outcome_{model}.csv"))
rows = []
for fr, g in gt.groupby("framing"):
    c = l2[l2.framing == fr]
    m = l2m[l2m.framing == fr].iloc[0]
    o = l3[l3.framing == fr]
    rp = c[c.contrast == "Republican - Democrat"]
    steep = rp[rp.verdict == "steepened"]
    rows.append(dict(
        model=model, framing=fr, thermometers=len(g), personas=int(g.n_personas.max()),
        gate_pass_min=int(g.pass_min.sum()), gate_pass_rec=int(g.pass_rec.sum()),
        gate_pass_single=int(g.pass_single.sum()), gate_k_min_lo=int(g.k_min.min()), gate_k_min_hi=int(g.k_min.max()),
        gate_k_rec_lo=int(g.k_rec.min()), gate_k_rec_hi=int(g.k_rec.max()),
        gate_se30_lo=g.se_k.min(), gate_se30_hi=g.se_k.max(), gate_within_sd_lo=g.within_sd.min(),
        gate_within_sd_hi=g.within_sd.max(), ref_sd_lo=g.sd_ref.min(), ref_sd_hi=g.sd_ref.max(),
        l2_contrasts=len(c), l2_no_pop_gap=int((c.verdict == "no population gap").sum()),
        l2_ref_too_imprecise=int((c.verdict == "reference too imprecise").sum()),
        l2_kept=int((c.verdict == "kept").sum()), l2_excludes_kept=int(m.excludes_kept), l2_model_verdict=m.verdict,
        l2_steepened=int((c.verdict == "steepened").sum()),
        l2_party_steepened=len(steep), l2_party_steepened_ratio_lo=steep.ratio.min() if len(steep) else None,
        l2_party_steepened_ratio_hi=steep.ratio.max() if len(steep) else None,
        l3_pass_05sd=int(o.pass_05sd.sum()), l3_pass_02sd=int(o.pass_02sd.sum()),
        l3_outcomes_with_failing_group=int((o.groups_failing_05sd > 0).sum())))
s = pd.DataFrame(rows)
s.to_csv(os.path.join(R, f"summary_{model}.csv"), index=False)
print(s.T.to_string())
