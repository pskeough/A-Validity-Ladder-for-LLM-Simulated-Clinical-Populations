"""
PRE-REGISTRATION for the decoding control experiment.

The manuscript reports that a third of same-cell regenerations change severity category and one in
five cross PHQ-8 = 10, measured at each provider's DEFAULT decoding because no sampling parameter
was ever set. A reviewer can answer that in one line: "set temperature to 0". This control tests it.

Design, fixed here before any call is made:

  4 endpoints x 12 cohorts x 30 draws x 2 decoding arms = 2,880 generations, clinical framing only.

  arm A  temperature = 0.0, top_p = 1.0     (the reviewer's proposed remedy)
  arm B  provider default, nothing set      (the original run's configuration)

Both arms are collected in the SAME session so the contrast is internal. That matters because two
of the four December 2025 routes no longer exist: google/gemini-3-flash-preview-20251217 and
z-ai/glm-4.7-20251222 are gone, and only their undated routes survive. The control therefore
measures decoding sensitivity on endpoints as served in August 2026. It does NOT reproduce the
December corpus and no claim of that kind may be made from it.

Cohort selection is stratified, not chosen. The 120 cohorts are ranked by their observed clinical
mean PHQ-8 in the released data, split into 12 equal strata, and one cohort is drawn per stratum
under a fixed seed recorded below. That spans the severity range without selecting on the outcome
the experiment measures.

Writes analysis/decoding_control_preregistration.json. Run this before 55_decoding_control_run.py.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
OUT = os.path.join(BASE, "analysis")

SEED = 20260803          # the date this design was fixed; not tuned
N_STRATA = 12
N_DRAWS = 30

# Route substitutions forced by endpoint retirement, recorded so the paper can state them.
ENDPOINTS = [
    {"original": "openai/gpt-4o-mini",                      "control": "openai/gpt-4o-mini",            "substituted": False},
    {"original": "deepseek/deepseek-chat-v3",               "control": "deepseek/deepseek-chat",        "substituted": False,
     "note": "the v3 slug was a route alias for deepseek-chat; analysis/openrouter_deepseek_resolution.json records the resolution"},
    {"original": "google/gemini-3-flash-preview-20251217",  "control": "google/gemini-3-flash-preview", "substituted": True,
     "note": "dated snapshot retired; undated route is a different build"},
    {"original": "z-ai/glm-4.7-20251222",                   "control": "z-ai/glm-4.7",                  "substituted": True,
     "note": "dated snapshot retired; undated route is a different build"},
]

ARMS = [
    {"name": "temp0",   "params": {"temperature": 0.0, "top_p": 1.0}},
    {"name": "default", "params": {}},
]

FACTORS = ["race", "gender", "ses", "relationship"]

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"))
m["phq8_total"] = m.phq8_total.clip(0, 24)
clin = m[m.prompt_condition == "clinical"].copy()

cohort = (clin.groupby(["profile_id"] + FACTORS, as_index=False)
              .agg(mean_phq8=("phq8_total", "mean"), n=("phq8_total", "size"))
              .sort_values("mean_phq8")
              .reset_index(drop=True))
assert len(cohort) == 120, len(cohort)

rng = np.random.default_rng(SEED)
edges = np.linspace(0, len(cohort), N_STRATA + 1).astype(int)
picked = []
for i in range(N_STRATA):
    lo, hi = edges[i], edges[i + 1]
    j = int(rng.integers(lo, hi))
    r = cohort.iloc[j]
    picked.append(dict(stratum=i + 1, profile_id=r.profile_id,
                       race=r.race, gender=r.gender, ses=r.ses, relationship=r.relationship,
                       observed_mean_phq8=round(float(r.mean_phq8), 4)))

sel = pd.DataFrame(picked)
print("SELECTED COHORTS (seed %d, %d strata by observed clinical mean PHQ-8)" % (SEED, N_STRATA))
print(sel.to_string(index=False))
print()
print("severity span: %.2f to %.2f  (full range across 120 cohorts: %.2f to %.2f)"
      % (sel.observed_mean_phq8.min(), sel.observed_mean_phq8.max(),
         cohort.mean_phq8.min(), cohort.mean_phq8.max()))
print()
print("axis balance in the 12 selected:")
for f in FACTORS:
    print("  %-13s %s" % (f, dict(sel[f].value_counts())))

n_calls = len(ENDPOINTS) * len(sel) * N_DRAWS * len(ARMS)
print("\nplanned generations: %d x %d x %d x %d = %d"
      % (len(ENDPOINTS), len(sel), N_DRAWS, len(ARMS), n_calls))

prereg = dict(
    fixed_utc=datetime.now(timezone.utc).isoformat(),
    seed=SEED,
    n_strata=N_STRATA,
    n_draws=N_DRAWS,
    condition="clinical",
    endpoints=ENDPOINTS,
    arms=ARMS,
    planned_generations=n_calls,
    selection_rule=("120 cohorts ranked by observed clinical mean PHQ-8 in the released data, "
                    "split into 12 equal strata, one drawn per stratum under the fixed seed"),
    primary_outcomes=[
        "within-cell severity-category flip probability, per model per arm",
        "share of discordant ordered pairs crossing PHQ-8 = 10, per model per arm",
    ],
    cohorts=picked,
)
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "decoding_control_preregistration.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(prereg, f, indent=2)

digest = hashlib.sha256(json.dumps(prereg, sort_keys=True).encode()).hexdigest()[:16]
print("\nwritten -> %s" % p)
print("prereg sha256[:16] = %s   (cite this in the appendix)" % digest)
