"""Receipts for the collection-provenance numbers in Appendix A.

Appendix A now states per-model call counts, error rates, token means, provider composition and
recovery counts. Those are claims like any other in this paper, so they get a receipt rather than a
place in the gate's allow-list. Everything here is derived from two sources: the provider's activity
export for the narrative run, and the released output file's batch identifiers and timestamps.

The activity export lives outside the repository because it is an account-level record. Its path is
read from PSYCHBENCH_ACTIVITY_CSV if set, and otherwise from the archived copy; when neither is
present the script emits the recovery half and marks the ledger half unavailable, so a reader
without the export still gets the numbers that come from released data.

Emits analysis/call_ledger.csv and analysis/recovery_ledger.csv.
"""
import os

import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
REC_BATCHES = ["SLOW_RECOVERY_2026", "LAST_MILE_OPT_2026", "RECOVERY_GAP_2026"]
# The provider activity export is not redistributable, so it is not in the release. Point
# PSYCHBENCH_ACTIVITY_CSV at a local copy to regenerate the billing side of the ledger; without
# it the script still produces the recovery figures, which come from the released data.
ACTIVITY = os.environ.get(
    "PSYCHBENCH_ACTIVITY_CSV",
    os.path.join(BASE, "data", "openrouter_activity.csv"))

# ---- recovery, from released data -------------------------------------------------------------
m = pd.read_csv(os.path.join(BASE, "data", "model_outputs.csv"))
m = m.merge(pd.read_csv(os.path.join(OUT, "condition_runid_map.csv")), on="run_id", how="left")
m["ts"] = pd.to_datetime(m.timestamp, format="mixed", errors="coerce")

cl = m[m.condition == "clinical"]
main_day = cl[~cl.run_id.isin(REC_BATCHES)].ts.dt.date.mode()[0]
rec = cl[cl.run_id.isin(REC_BATCHES)]
rows = [dict(quantity="clinical rows outside the main session", value=int((cl.ts.dt.date != main_day).sum())),
        dict(quantity="rows carrying a recovery batch id", value=len(rec)),
        dict(quantity="recovery rows inside the main session", value=int((rec.ts.dt.date == main_day).sum())),
        dict(quantity="recovery rows, GLM-4.7", value=int((rec.model == "z-ai/glm-4.7").sum())),
        dict(quantity="recovery rows, GPT-4o-mini", value=int((rec.model == "openai/gpt-4o-mini").sum())),
        dict(quantity="rows completed by hand (LAST_MILE_OPT_2026)",
             value=int((rec.run_id == "LAST_MILE_OPT_2026").sum())),
        dict(quantity="cells touched by the outside-session rows",
             value=int(cl[cl.ts.dt.date != main_day].groupby(["model", "profile_id"]).ngroups))]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "recovery_ledger.csv"), index=False)
for r in rows:
    print(f"  {r['quantity']:48s} {r['value']}")

# ---- the narrative run's calls, from the provider export ---------------------------------------
if not os.path.exists(ACTIVITY):
    print(f"\nactivity export not found at {ACTIVITY}; call ledger not rebuilt")
    raise SystemExit(0)

a = pd.read_csv(ACTIVITY)
a["failed"] = a.finish_reason_normalized.ne("stop")
led = (a.groupby("model_permaslug")
        .agg(calls=("generation_id", "size"), errors=("failed", "sum"),
             tokens_prompt=("tokens_prompt", "mean"),
             tokens_completion=("tokens_completion", "mean"),
             tokens_reasoning=("tokens_reasoning", lambda s: s.fillna(0).mean()),
             cost=("cost_total", "sum"))
        .assign(error_pct=lambda d: (100 * d.errors / d.calls).round(2))
        .round(1).reset_index())
led["providers"] = led.model_permaslug.map(
    a.groupby("model_permaslug").provider_name.agg(lambda s: "|".join(sorted(s.unique()))))
led.to_csv(os.path.join(OUT, "call_ledger.csv"), index=False)

print()
print(led.to_string(index=False))
print(f"\n  total generations {len(a):,}   prompt tokens mean {a.tokens_prompt.mean():.0f}"
      f"   reasoning mean over reasoning models "
      f"{a[a.tokens_reasoning.fillna(0) > 0].tokens_reasoning.mean():.0f}")
