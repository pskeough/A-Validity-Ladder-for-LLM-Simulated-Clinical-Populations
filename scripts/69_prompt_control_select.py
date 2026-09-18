"""
PRE-REGISTRATION for the prompt-template control experiment.

The manuscript's level-2 (between-group gaps) and level-3 (severity against the NHANES anchor)
verdicts were both measured under ONE system prompt, which casts the model as a "Clinical
Simulation Engine" and, in operational constraint 2, instructs it to "Base your responses on the
statistical likelihood of symptoms for this specific demographic intersection in the US population
(referencing NSDUH/CDC trends)". A reviewer can answer the whole results section in one line: the
gaps and the elevation are properties of that instruction, not of the models. This control tests it.

Design, fixed here before any call is made:

  4 endpoints x 12 cohorts x 30 draws x 3 prompt arms = 4,320 generations, clinical user prompt
  only, provider default decoding in every arm (nothing set), which is the configuration the
  released corpus ran under.

  arm A  the original system prompt, verbatim                     (the within-session control)
  arm B  the same prompt with operational constraint 2, the NSDUH/CDC population instruction,
         deleted and the remaining constraints renumbered         (isolates that one line)
  arm C  a first-person template with no role assignment and no population reference: the model is
         told it is the person in the profile and answers the four questionnaires as that person,
         keeping NO MORALIZING, SINGLE-SHOT and NUMERIC ONLY and the identical OUTPUT FORMAT block

The OUTPUT FORMAT block is byte-identical across the three arms, so the parser is unchanged and no
arm can differ from another through parse success. The user prompt is byte-identical across the
three arms as well, so the entire manipulation sits in the system message. One consequence is
recorded rather than hidden: the shared user prompt closes with "TASK: Simulate this participant's
responses to the Standardized Clinical Battery", so arm C removes the simulation framing from the
system message while the user message still carries the word. Arm C is therefore a partial removal
of the framing, not a clean first-person condition, and no claim beyond that may be made from it.

All three arms are collected in the SAME session so the contrast is internal. Two of the four
December 2025 routes no longer exist: google/gemini-3-flash-preview-20251217 and
z-ai/glm-4.7-20251222 are gone, and only their undated routes survive. The endpoint list is read
verbatim from the decoding control's pre-registration so the two controls stand on identical
infrastructure. This control therefore measures prompt sensitivity on endpoints as served now. It
does NOT reproduce the December corpus and no claim of that kind may be made from it.

Cohort selection is by rule, not by outcome. The cohorts are the full crossing

    race in {White, Black} x gender in {Cisgender Man, Cisgender Woman}
    x socioeconomic status in {Low, Middle, High}

with relationship status held at one level drawn under the fixed seed recorded below. Nothing about
the released severity of a cohort enters the choice. The crossing is complete, so inside the control
every contrast the paper reports on these axes exists as a matched pair differing in one attribute:
6 sex pairs, 6 Black-minus-White pairs, and 4 pairs for each of the three income contrasts. Those
pairs are enumerated here, before collection, and asserted.

Writes analysis/prompt_control_preregistration.json. Run this before 70_prompt_control_run.py.
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
DECODING_PREREG = os.path.join(OUT, "decoding_control_preregistration.json")

SEED = 20260909          # the date this design was fixed; not tuned
N_DRAWS = 30
RELATIONSHIP_LEVELS = ["Married", "Single"]

RACES = ["White", "Black"]
GENDERS = ["Cisgender Man", "Cisgender Woman"]
SES_LEVELS = ["Low", "Middle", "High"]

# The OUTPUT FORMAT block is shared by all three arms byte for byte. Reconstructed from Appendix A
# exactly as 55_decoding_control_run.py reconstructs it: the clinical condition administered a
# 20-item PCL-5, Appendix A prints the format line without an item count, and the released data
# carry 20 populated PCL-5 item columns under clinical framing.
OUTPUT_FORMAT = """OUTPUT FORMAT:
{
  "PHQ8": [int, int...], // 8 items, 0-3 scale
  "GAD7": [int, int...], // 7 items, 0-3 scale
  "AUDIT_C": [int, int...], // 3 items, 0-4 scale
  "PCL5": [int, int...] // 20 items, 0-4 scale
}"""

# Arm A. The released corpus's system prompt, verbatim.
PROMPT_A = """ROLE:
You are a Clinical Simulation Engine designed for a synthetic data study.
Your task is to generate PROBABILISTIC responses for a "Standardized Patient" based on a provided demographic profile.

OPERATIONAL CONSTRAINTS (STRICT):
1. NO MORALIZING: Do not offer help, advice, or disclaimers. You are simulating data, not treating a patient.
2. PROBABILISTIC ACCURACY: Base your responses on the statistical likelihood of symptoms for this specific demographic intersection in the US population (referencing NSDUH/CDC trends).
3. SINGLE-SHOT OUTPUT: You must output the results for ALL four screening tools in a single valid JSON object.
4. NUMERIC ONLY: Do not generate text explanations. Return integers.

""" + OUTPUT_FORMAT

# Arm B. Arm A with operational constraint 2 removed and 3 and 4 renumbered. Every other character
# of the role block, the surviving constraints and the output format is unchanged.
PROMPT_B = """ROLE:
You are a Clinical Simulation Engine designed for a synthetic data study.
Your task is to generate PROBABILISTIC responses for a "Standardized Patient" based on a provided demographic profile.

OPERATIONAL CONSTRAINTS (STRICT):
1. NO MORALIZING: Do not offer help, advice, or disclaimers. You are simulating data, not treating a patient.
2. SINGLE-SHOT OUTPUT: You must output the results for ALL four screening tools in a single valid JSON object.
3. NUMERIC ONLY: Do not generate text explanations. Return integers.

""" + OUTPUT_FORMAT

# Arm C. No role assignment, no reference to a population, no reference to statistical likelihood
# and no mention of simulation or of a study. The NO MORALIZING clause loses its trailing "You are
# simulating data, not treating a patient", which is simulation framing rather than a constraint.
PROMPT_C = """You are the person described in the profile below.
Answer the four screening questionnaires as that person, about yourself.

CONSTRAINTS (STRICT):
1. NO MORALIZING: Do not offer help, advice, or disclaimers.
2. SINGLE-SHOT OUTPUT: You must output the results for ALL four screening tools in a single valid JSON object.
3. NUMERIC ONLY: Do not generate text explanations. Return integers.

""" + OUTPUT_FORMAT

# Identical in all three arms; the clinical user prompt of Appendix A.
USER_TEMPLATE = """PROFILE_ID: {profile_id}
DEMOGRAPHICS:
- Race: {race}
- Gender Identity: {gender}
- Socioeconomic Status: {ses}
- Relationship Status: {relationship}

TASK: Simulate this participant's responses to the Standardized Clinical Battery."""

ARMS = [
    {"name": "orig", "params": {}, "system_prompt": PROMPT_A,
     "description": "the released corpus's system prompt, verbatim"},
    {"name": "no_pop", "params": {}, "system_prompt": PROMPT_B,
     "description": "arm A with operational constraint 2 (PROBABILISTIC ACCURACY, the NSDUH/CDC "
                    "population instruction) deleted and the remaining constraints renumbered"},
    {"name": "first_person", "params": {}, "system_prompt": PROMPT_C,
     "description": "first-person, no role assignment and no population reference; NO MORALIZING, "
                    "SINGLE-SHOT and NUMERIC ONLY and the OUTPUT FORMAT block retained"},
]

# contrast name, factor, high level, reference level, pairing stratum, anchor group high, anchor low
CONTRASTS = [
    ("Women minus Men",       "gender",    "Cisgender Woman", "Cisgender Man", ["race", "ses_level"], "Women",  "Men"),
    ("Black minus White",     "race",      "Black",           "White",         ["gender", "ses_level"], "Black", "White"),
    ("Low minus High SES",    "ses_level", "Low",             "High",          ["race", "gender"],    "Low",    "High"),
    ("Middle minus High SES", "ses_level", "Middle",          "High",          ["race", "gender"],    "Middle", "High"),
    ("Low minus Middle SES",  "ses_level", "Low",             "Middle",        ["race", "gender"],    "Low",    "Middle"),
]

# Level-3 anchor rows. The severity anchor lives in groundtruth/, its design-based standard error in
# analysis/anchor_design_se.csv under a different row label for the two sex rows.
LEVEL3_GROUPS = [
    {"group": "Low",    "axis": "ses_level", "level": "Low",             "gt_row": "Low",    "se_row": "Low"},
    {"group": "Middle", "axis": "ses_level", "level": "Middle",          "gt_row": "Middle", "se_row": "Middle"},
    {"group": "High",   "axis": "ses_level", "level": "High",            "gt_row": "High",   "se_row": "High"},
    {"group": "Men",    "axis": "gender",    "level": "Cisgender Man",   "gt_row": "Men",    "se_row": "Cisgender men"},
    {"group": "Women",  "axis": "gender",    "level": "Cisgender Woman", "gt_row": "Women",  "se_row": "Cisgender women"},
    {"group": "White",  "axis": "race",      "level": "White",           "gt_row": "White",  "se_row": "White"},
    {"group": "Black",  "axis": "race",      "level": "Black",           "gt_row": "Black",  "se_row": "Black"},
]

# Endpoints are not re-decided here. The decoding control fixed the substitutions forced by endpoint
# retirement in August 2026 and both controls must stand on the same routes to be comparable.
dec = json.load(open(DECODING_PREREG, encoding="utf-8"))
ENDPOINTS = dec["endpoints"]

m = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"), low_memory=False)
m["phq8_total"] = m["phq8_total_clipped"]
clin = m[m.prompt_condition == "clinical"].copy()

# One relationship level, drawn rather than chosen, so that the choice cannot be read as tuned. The
# level is held constant across all 12 cohorts, which is what makes every contrast a one-attribute
# difference.
rng = np.random.default_rng(SEED)
relationship = RELATIONSHIP_LEVELS[int(rng.integers(0, len(RELATIONSHIP_LEVELS)))]

reg = (clin[["profile_id", "race", "gender", "ses", "relationship", "ses_normalized"]]
       .drop_duplicates())
obs = clin.groupby("profile_id").phq8_total.mean()

picked = []
for race in RACES:
    for gender in GENDERS:
        for lvl in SES_LEVELS:
            r = reg[(reg.race == race) & (reg.gender == gender)
                    & (reg.ses_normalized == lvl) & (reg.relationship == relationship)]
            assert len(r) == 1, (race, gender, lvl, relationship, len(r))
            r = r.iloc[0]
            picked.append(dict(profile_id=r.profile_id, race=r.race, gender=r.gender,
                               ses=r.ses, ses_level=r.ses_normalized, relationship=r.relationship,
                               # descriptive only; the released mean plays no part in selection
                               released_mean_phq8=round(float(obs[r.profile_id]), 4)))

sel = pd.DataFrame(picked)
assert len(sel) == 12 and sel.profile_id.nunique() == 12

# Enumerate the matched pairs the design promises, and assert the counts, before any call is made.
pairs = {}
for name, factor, hi, lo, stratum, _ghi, _glo in CONTRASTS:
    a = sel[sel[factor] == hi].set_index(stratum).profile_id
    b = sel[sel[factor] == lo].set_index(stratum).profile_id
    assert not a.index.has_duplicates and not b.index.has_duplicates, name
    common = a.index.intersection(b.index)
    pairs[name] = [dict(stratum=dict(zip(stratum, k if isinstance(k, tuple) else (k,))),
                        high=a.loc[k], low=b.loc[k]) for k in common]
EXPECTED = {"Women minus Men": 6, "Black minus White": 6, "Low minus High SES": 4,
            "Middle minus High SES": 4, "Low minus Middle SES": 4}
for name, want in EXPECTED.items():
    assert len(pairs[name]) == want, (name, len(pairs[name]), want)

print("SELECTED COHORTS (seed %d; relationship status drawn = %s, held constant)"
      % (SEED, relationship))
print(sel[["profile_id", "race", "gender", "ses_level", "relationship", "released_mean_phq8"]]
      .to_string(index=False))
print()
print("selection rule is the complete 2 x 2 x 3 crossing, so no cohort was chosen on its severity;")
print("released clinical means of the 12 span %.2f to %.2f"
      % (sel.released_mean_phq8.min(), sel.released_mean_phq8.max()))
print()
print("matched pairs available inside the control:")
for name, _f, _h, _l, _s, _gh, _gl in CONTRASTS:
    print("  %-24s %d pairs" % (name, len(pairs[name])))
print()
print("level-3 anchor rows: %s" % ", ".join(g["group"] for g in LEVEL3_GROUPS))
print()
for arm in ARMS:
    print("=" * 78)
    print("ARM %s  (%s)" % (arm["name"], arm["description"]))
    print("-" * 78)
    print(arm["system_prompt"])
print("=" * 78)
print("USER PROMPT, identical in all three arms")
print("-" * 78)
print(USER_TEMPLATE)
print("=" * 78)

n_calls = len(ENDPOINTS) * len(sel) * N_DRAWS * len(ARMS)
print("\nplanned generations: %d arms x %d endpoints x %d cohorts x %d draws = %d"
      % (len(ARMS), len(ENDPOINTS), len(sel), N_DRAWS, n_calls))

prereg = dict(
    fixed_utc=datetime.now(timezone.utc).isoformat(),
    seed=SEED,
    n_draws=N_DRAWS,
    condition="clinical",
    decoding="provider default in every arm, nothing set, matching the released corpus",
    relationship_level=relationship,
    relationship_draw_rule=("one level drawn from %s under the fixed seed and held constant across "
                            "all 12 cohorts" % RELATIONSHIP_LEVELS),
    endpoints=ENDPOINTS,
    endpoints_source="analysis/decoding_control_preregistration.json, verbatim",
    arms=ARMS,
    user_template=USER_TEMPLATE,
    planned_generations=n_calls,
    selection_rule=("the complete crossing race in {White, Black} x gender in {Cisgender Man, "
                    "Cisgender Woman} x socioeconomic status in {Low, Middle, High}, with "
                    "relationship status fixed at one seeded draw; no cohort is selected on any "
                    "released outcome"),
    contrasts=[dict(name=n, factor=f, high=h, low=l, stratum=s,
                    anchor_high=gh, anchor_low=gl, n_pairs=len(pairs[n]), pairs=pairs[n])
               for n, f, h, l, s, gh, gl in CONTRASTS],
    level3_groups=LEVEL3_GROUPS,
    primary_outcomes=[
        "cohort mean PHQ-8 total, per arm per model per cohort",
        "level-3 residual against the NHANES anchor for Low, Middle, High, Men, Women, White and "
        "Black, per arm per model, the anchor's design-based standard error folded in as a root "
        "sum of squares (groundtruth/phq8_groundtruth_nhanes_2005_2018.csv, "
        "analysis/anchor_design_se.csv)",
        "level-2 paired gap for each of the five contrasts, per arm, paired t against zero and "
        "against the population gap, the count of models carrying the population's sign, and the "
        "ratio of simulated to population gap",
        "within-cell severity-category flip probability, per arm per model, estimated as in "
        "56_decoding_control_analyze.py",
        "between-arm contrasts B minus A and C minus A on cohort means, paired over cohorts "
        "within model",
    ],
    scope=("This control measures prompt-template sensitivity on the four endpoints AS SERVED NOW. "
           "Two of the four December 2025 routes have been retired and only their undated "
           "successors survive, so the control does not reproduce the December corpus and no claim "
           "of that kind may be made from it. Arm A is the within-session control against which "
           "arms B and C are read; the released corpus is not."),
    known_limitation=("The user prompt is held byte-identical across arms and still closes with "
                      "'TASK: Simulate this participant's responses to the Standardized Clinical "
                      "Battery'. Arm C therefore removes the role assignment and the population "
                      "reference from the system message only, and is a partial rather than a "
                      "complete removal of the simulation framing."),
    cohorts=picked,
)
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "prompt_control_preregistration.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(prereg, f, indent=2)

digest = hashlib.sha256(json.dumps(prereg, sort_keys=True).encode()).hexdigest()[:16]
print("\nwritten -> %s" % p)
print("prereg sha256[:16] = %s   (cite this in the appendix)" % digest)
