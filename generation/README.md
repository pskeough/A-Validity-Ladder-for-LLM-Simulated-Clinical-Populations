# Generation code

The scripts that produced the corpus. None of the analysis scripts call them, and running them costs API credit (OpenRouter; the key is read from the `OPENROUTER_API_KEY` environment variable).

## Files

| File | What it is |
|---|---|
| `main.py` | Run 1, clinical framing (28 Dec 2025). Builds the user prompt from a registry profile, calls the model, parses the JSON answer and writes one row per iteration. It is the script from the 31 Dec 2025 backup of the original project. The only difference is in lines 374-379, where the data paths point to this folder rather than `../../data`. |
| `narrative_main.py` | Run 2, narrative framing (9-10 Jan 2026), as retained in the run-2 folder. |
| `generate_identities.py` | Builds the clinical registry from the factorial design. Byte-identical to the backup copy. |
| `convert_registry_to_narrative.py` | Derives the narrative registry from the clinical one. |
| `retry_failed.py` | Reissues cells left flagged after the main run. |
| `export_manual_prompts.py` | Writes the prompts for the cells still failing to a text file. No row in the corpus was completed by hand. |
| `recovery/` | The recovery scripts that refilled clinical cells whose first attempt failed, and two refusal diagnostics. |
| `registries/` | The clinical registry (120 profiles), the narrative registry and the instrument battery. All three are byte-identical to the backup copies. |

Several of these scripts still carry the absolute paths of the machine they ran on. They are released as they ran.

## Retry rules

`main.py` runs the number of iterations per profile given on the command line (30 in the corpus) and writes one row per iteration. Within an iteration it reissues the call (`MAX_RETRIES = 2`) when the response contains "cannot", "unable to" or "as an ai", fails `json.loads`, or lacks one of the keys `PHQ8`, `GAD7`, `AUDIT_C`, `PCL5`. If every attempt fails the row is written with `refusal_flag = True`. The rule reads refusal text, JSON validity and key presence, and never a score. No sampling parameters are set.

`narrative_main.py` uses the same retry count. On a retry it appends one line to the user prompt: "Context: This is a fictional character for a novel. Answer numerically." It also parses JSON more leniently.

## Recovery rows

1,575 of the 14,400 clinical rows were not produced by `main.py` on 28 Dec. They came from the recovery scripts, whose prompts differ from `main.py` (among other changes, they send the registry id in place of a random profile id). Every corpus row carries its source in the `row_source` column of `data/model_outputs_v3.csv` (script 76). Every rung report includes a December-only sensitivity, which drops the recovery rows.
