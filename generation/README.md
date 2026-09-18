# Generation layer

Everything that produced the 28,800 responses: the script that made the calls, the stimulus it drew
from, and the recovery scripts that refilled cells whose first attempt failed.

## Files

| File | What it is |
|---|---|
| `main.py` | The generation engine. Builds the user prompt from a registry profile, calls the model through OpenRouter, validates the return, scores the four instruments, writes one CSV row per iteration. |
| `generate_identities.py` | Builds the clinical registry from the factorial design. |
| `convert_registry_to_narrative.py` | Derives the narrative registry from the clinical one. The whole framing manipulation is the two lookup tables at the top of this file. |
| `retry_failed.py` | Reissues cells left flagged after the main run. |
| `export_manual_prompts.py` | Emits the prompts for cells completed by hand. |
| `recovery/` | The per-model recovery scripts used on 11 January, plus the two refusal diagnostics. |
| `registries/identities_registry.json` | 120 clinical profiles. `injection_text` is the exact demographic block the clinical condition put in the user turn. |
| `registries/identities_registry_narrative.json` | The same 120 profiles in first person. `original_tag_text` preserves the clinical string each was derived from. |
| `registries/diagnostic_battery.json` | Item counts and response scales for the four instruments. |

## The retention rule

This is the part a reader auditing the release should check, because the provider ledger records
more billed calls than there are rows and the difference has to be accounted for.

`main.py` runs a fixed number of iterations per cell and **writes exactly one row per iteration**.
A row is never dropped for its scores. Inside an iteration, a call is reissued (`MAX_RETRIES = 2`,
so at most two reissues) when the response:

1. contains `cannot`, `unable to`, or `as an ai` — the refusal check;
2. fails `json.loads` after code fences are stripped;
3. is missing any of `PHQ8`, `GAD7`, `AUDIT_C`, `PCL5`.

If all attempts fail, the row is still written with `refusal_flag = True`.

**The rule reads refusal text, JSON validity and key presence. It never reads a value.** There is
no array-length check and no in-range check in this version, so the discarded attempts cannot be
the high-scoring ones or the low-scoring ones. Those reissued attempts are billed and retained
nowhere, and they are the whole of the ledger shortfall the paper reports.

## What this pins down

Every `profile_id` in `data/model_outputs_v2.csv` appears in the registries and every registry
profile appears in the corpus: **120 of 120, no residue on either side.**

## What a replication still cannot get

The endpoints. Two of the four routes are undated substitutes for snapshots since retired.
Appendix L of the paper measures what that costs by rerunning twelve cohorts eight months later.

## Checksums

```
f2966be8191867f0c89e98b2afc22933  registries/identities_registry.json
0b48f66e87d5e33de0e0c66662a4fcb0  registries/identities_registry_narrative.json
49a561b0c0490d9a25184e7539c7ba7b  registries/diagnostic_battery.json
a65e1eb47383f14b9dd0a90a042dec80  convert_registry_to_narrative.py
```

Both registry files are byte-identical to two independently retained copies.
