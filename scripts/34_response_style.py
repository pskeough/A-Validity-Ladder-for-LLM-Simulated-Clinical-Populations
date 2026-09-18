"""Is the compressed severity range a psychiatric result or a general ordinal response style?

A reviewer raised the alternative this paper had not tested. Endpoint avoidance and central-tendency
pull are documented general properties of LLM responses on ordinal scales, and they produce exactly
the signature Section 4.1 reports: mass in the interior bands, the top category under-emitted,
dispersion around half of the population's. If that is what is happening, the compression is a
property of ordinal generation rather than of how these models represent depression, and the
epidemiological reading is wrong at the level of mechanism.

The design already carries the discriminating test. Every generation answers three instruments in
one response: PHQ-8 and GAD-7 on 0-3, and AUDIT-C on 0-4. A general response-style artefact has no
reason to respect instrument boundaries, so it should suppress the top category on all three alike.
A representation specific to depression should not.

Two comparisons:

  endpoint usage   share of item responses at the scale maximum, per instrument, against the
                   population share where one exists. NHANES supplies PHQ-8 item marginals; GAD-7
                   and AUDIT-C have no NHANES counterpart, so those two are read against each other
                   and against PHQ-8 within the same generations rather than against a population.
  cross-instrument the PHQ-8 to GAD-7 correlation the models produce, against the 0.64 to 0.75 band
    correlation   Loewe et al. report for the same instrument pair in primary care. A model that
                  reproduces severity but collapses the distinction between depression and anxiety
                  is a different failure from a level error, and it is measurable here without any
                  population anchor.

Emits analysis/response_style.csv and analysis/instrument_correlations.csv.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(BASE, "data", "nhanes_raw")

PHQ = [f"phq8_{i}" for i in range(1, 9)]
GAD = [f"gad7_{i}" for i in range(1, 8)]
AUD = [f"audit_{i}" for i in range(1, 4)]
NH_PHQ = [f"DPQ0{i}0" for i in range(1, 9)]
CYCLES = list("DEFGHIJ")

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m["gender"] = m.gender.astype(str).str.strip().str.strip('"')
cis = m[m.gender.str.contains("Cis")]

# ---- population item marginals, PHQ-8 only ---------------------------------------------------
frames = []
for c in CYCLES:
    d = pd.read_sas(os.path.join(RAW, f"DEMO_{c}.xpt"))[["SEQN", "RIDAGEYR", "WTMEC2YR"]]
    frames.append(d.merge(pd.read_sas(os.path.join(RAW, f"DPQ_{c}.xpt")), on="SEQN"))
pop = pd.concat(frames, ignore_index=True)
pop[NH_PHQ] = pop[NH_PHQ].where(pop[NH_PHQ] <= 3)
pop = pop[(pop.RIDAGEYR >= 18) & pop[NH_PHQ].notna().all(axis=1) & (pop.WTMEC2YR > 0)].copy()
pop["w"] = pop.WTMEC2YR / len(CYCLES)


# NHANES XPT files store a zero response as a denormalized float near 5.4e-79 rather than exact
# zero, so `== 0` silently matches nothing and the population floor share comes back as 0.00%.
# Sums are unaffected, which is why every total in this paper is correct, but any equality test on
# an item value has to round first. Rounding also protects the model-side comparison against any
# float written by the export.
def _codes(df, items):
    a = df[items].to_numpy(float)
    a = a[~np.isnan(a).any(axis=1)]
    return np.rint(a)


def endpoint_share(df, items, top):
    """Share of all item responses sitting at the scale maximum."""
    a = _codes(df, items)
    return float((a == top).sum() / a.size * 100) if a.size else np.nan


def floor_share(df, items):
    a = _codes(df, items)
    return float((a == 0).sum() / a.size * 100) if a.size else np.nan


w = pop.w.to_numpy()
pa = np.rint(pop[NH_PHQ].to_numpy(float))
pop_top = float((pa == 3).sum(axis=1) @ w / (w.sum() * len(NH_PHQ)) * 100)
pop_floor = float((pa == 0).sum(axis=1) @ w / (w.sum() * len(NH_PHQ)) * 100)
print(f"population PHQ-8: {pop_top:.2f}% of item responses at the top category, "
      f"{pop_floor:.2f}% at zero")

rows = []
for scope, df in [("ALL", cis)] + [(mm, g) for mm, g in cis.groupby("model")]:
    for name, items, top in [("PHQ-8", PHQ, 3), ("GAD-7", GAD, 3), ("AUDIT-C", AUD, 4)]:
        rows.append(dict(scope=scope, instrument=name, max_category=top,
                         pct_at_max=round(endpoint_share(df, items, top), 3),
                         pct_at_zero=round(floor_share(df, items), 3),
                         pop_pct_at_max=round(pop_top, 3) if name == "PHQ-8" else np.nan,
                         pop_pct_at_zero=round(pop_floor, 3) if name == "PHQ-8" else np.nan))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "response_style.csv"), index=False)
print("\n" + res.to_string(index=False))

allc = res[res.scope == "ALL"].set_index("instrument")
print(f"\nendpoint usage, cisgender generations, all four models pooled:")
for inst in ("PHQ-8", "GAD-7", "AUDIT-C"):
    print(f"  {inst:8s} {allc.loc[inst].pct_at_max:5.2f}% at max, "
          f"{allc.loc[inst].pct_at_zero:5.2f}% at zero")
print(f"  population PHQ-8 reference: {pop_top:.2f}% at max, {pop_floor:.2f}% at zero")
# The discriminating statement. A general ordinal artefact suppresses the top category everywhere.
_suppressed = [i for i in ("PHQ-8", "GAD-7", "AUDIT-C") if allc.loc[i].pct_at_max < 1.0]
print(f"\ninstruments with under 1% of responses at the scale maximum: {_suppressed or 'none'}")

# ---- cross-instrument correlation ------------------------------------------------------------
# Loewe et al. report 0.64 to 0.75 across PHQ-8, GAD-7 and PHQ-15 in a primary-care sample; that
# band is the only anchor available for this comparison and it is already cited in the paper.
LOEWE_LO, LOEWE_HI = 0.64, 0.75
crows = []
for scope, df in [("ALL", cis)] + [(mm, g) for mm, g in cis.groupby("model")]:
    d = df[["phq8_total", "gad7_total", "audit_total"]].dropna()
    if len(d) < 30:
        continue
    crows.append(dict(scope=scope, n=len(d),
                      r_phq8_gad7=round(float(d.phq8_total.corr(d.gad7_total)), 4),
                      r_phq8_audit=round(float(d.phq8_total.corr(d.audit_total)), 4),
                      r_gad7_audit=round(float(d.gad7_total.corr(d.audit_total)), 4),
                      loewe_lo=LOEWE_LO, loewe_hi=LOEWE_HI))
cor = pd.DataFrame(crows)
cor.to_csv(os.path.join(OUT, "instrument_correlations.csv"), index=False)
print("\n" + cor.to_string(index=False))
_r = cor[cor.scope == "ALL"].iloc[0].r_phq8_gad7
print(f"\npooled PHQ-8 to GAD-7 correlation: {_r:.3f} against a clinical reference band of "
      f"{LOEWE_LO} to {LOEWE_HI}")
print("  above the band means the models are collapsing depression and anxiety toward one "
      "underlying severity dimension")
