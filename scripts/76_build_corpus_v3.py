"""Build data/model_outputs_v3.csv: v2 with the 43 overwritten GPT-4o-mini rows restored and a
provenance column on every row.

Why. The 31 Dec 2025 backup of the original project holds audit_results.csv as main.py left it on
28 Dec 2025 (copied to data/raw/run1_clinical_2025-12-28.csv). Against it:
  - 12,825 clinical rows of v2 are December successes, value for value;
  - 1,532 December rows were failures with no scores; v2 fills them from later recovery passes;
  - 43 valid December GPT-4o-mini rows (21:45 to 21:47 on 28 Dec) are missing from v2 and were
    replaced by RECOVERY_GAP_2026 rows generated on 11 Jan 2026 under a shortened prompt.
The December rows are the original data, so v3 puts them back. Everything else is carried over from
v2 unchanged; the recovery rows stay in the corpus and are labelled, and analyses report a
December-only sensitivity (script 75).

Columns added:
  row_source   dec28_main | dec28_restored | recovery_inplace | recovery_slow | recovery_last_mile |
               narrative
  row_batch    the run_id as released (for restored rows, the December run_id)
  phq8_valid   every PHQ-8 item is an integer in 0..3. One DeepSeek row has phq8_8 = 21; v2 kept it
               and clipped the total to 24. New scripts drop invalid rows instead.
  narr_source  narrative rows only: narr_main | narr_rerun_same_script (a failed call rerun on 9-10 Jan
               2026 by the same run-2 script). Empty for clinical rows.

v2 is not modified. Emits data/model_outputs_v3.csv and analysis/brm/corpus_v3_provenance.csv.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEC = os.path.join(BASE, "data", "raw", "run1_clinical_2025-12-28.csv")
OUTD = os.path.join(BASE, "analysis", "brm")
KEY = ["model", "profile_id", "iteration", "timestamp"]
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
GPT = "openai/gpt-4o-mini"
R2 = os.path.join(BASE, "data", "raw", "run2_narrative")
# Snapshots that carry refusal rows. The 39-column files mix two schemas after row 2,460, so
# on_bad_lines="skip" reads their first block only; the 40-column snapshot covers the full run.
R2_SNAPSHOTS = ["narrative_audit_results_backup_20260109_230913.csv",
                "narrative_audit_results_before_cleanup_20260110_130136.csv"]


def mkey(df):
    return pd.MultiIndex.from_frame(df[KEY].astype(str))


def main():
    os.makedirs(OUTD, exist_ok=True)
    v = pd.read_csv(os.path.join(BASE, "data", "model_outputs_v2.csv"), low_memory=False)
    dec = pd.read_csv(DEC)
    assert v.shape == (28800, 60) and dec.shape == (14400, 39)
    assert int(dec.refusal_flag.sum()) == 1532

    cl = v.prompt_condition == "clinical"
    ok_dec = dec[dec.refusal_flag == False]  # noqa: E712
    ref_dec = dec[dec.refusal_flag == True]  # noqa: E712
    in_ok = mkey(v).isin(mkey(ok_dec)) & cl
    in_ref = mkey(v).isin(mkey(ref_dec)) & cl
    assert int(in_ok.sum()) == 12825 and int(in_ref.sum()) == 1489

    rb = v.run_id.astype(str)
    src = np.where(~cl, "narrative", "unassigned").astype(object)
    src[in_ok.values] = "dec28_main"
    src[(in_ref & ~rb.isin(["SLOW_RECOVERY_2026", "LAST_MILE_OPT_2026"])).values] = "recovery_inplace"
    src[(cl & rb.eq("SLOW_RECOVERY_2026")).values] = "recovery_slow"
    src[(cl & rb.eq("LAST_MILE_OPT_2026")).values] = "recovery_last_mile"
    v["row_source"] = src
    v["row_batch"] = rb

    # The 43 GPT rows: v2 rows with no December key, replaced December rows with no v2 key.
    gap = cl & (v.model == GPT) & (v.row_source == "unassigned")
    assert int(gap.sum()) == 43 and set(rb[gap]) == {"RECOVERY_GAP_2026"}
    dg = dec[dec.model == GPT]
    lost = dg[~mkey(dg).isin(mkey(v[cl & (v.model == GPT)]))]
    assert len(lost) == 43 and not lost.refusal_flag.any()
    assert lost.timestamp.min() >= "2025-12-28T21:45" and lost.timestamp.max() < "2025-12-28T21:48"
    repl = v[gap].set_index(["profile_id", "iteration"])
    lk = lost.set_index(["profile_id", "iteration"])
    assert repl.index.sort_values().equals(lk.index.sort_values()), "restored rows do not pair one to one"

    restored = []
    for idx, r in lk.iterrows():
        row = repl.loc[idx].copy()          # carries prompt_condition, ses_normalized, etc.
        for c in dec.columns:
            if c in ("profile_id", "iteration"):
                continue
            row[c] = r[c]
        for i in range(5, 21):              # December asked for 4 PCL-5 items
            row[f"pcl5_{i}"] = np.nan
        row["phq8_total_clipped"] = min(max(float(r.phq8_total), 0.0), 24.0)
        row["row_source"] = "dec28_restored"
        row["row_batch"] = r.run_id
        row["profile_id"], row["iteration"] = idx
        restored.append(row)
    restored = pd.DataFrame(restored)[v.columns]

    old = v[gap][["profile_id", "iteration", "phq8_total"]].merge(
        restored[["profile_id", "iteration", "phq8_total"]], on=["profile_id", "iteration"],
        suffixes=("_recovery", "_december"))
    v3 = pd.concat([v[~gap], restored], ignore_index=True)
    assert (v3.row_source == "unassigned").sum() == 0
    v3["phq8_valid"] = v3[ITEMS].apply(lambda s: s.between(0, 3) & (s == s.round())).all(axis=1)

    # Narrative provenance. The original run-2 folder held the narrative main.py (generation/narrative_main.py)
    # and the output file at each cleanup step (the final file and two snapshots are in data/raw/run2_narrative). Its final file equals the released
    # narrative rows value for value. A narrative row is a same-day rerun when its (model, profile_id,
    # iteration) failed in an earlier snapshot; the reruns used the same script and prompt.
    nar = v3.prompt_condition == "narrative"
    fin = pd.read_csv(os.path.join(R2, "narrative_audit_results.csv"), low_memory=False)
    assert len(fin) == 14400 and mkey(fin).isin(mkey(v3[nar])).all()
    failed = set()
    for f in R2_SNAPSHOTS:
        d = pd.read_csv(os.path.join(R2, f), low_memory=False, on_bad_lines="skip")
        bad = d[d.refusal_flag.astype(str).str.lower().eq("true")]
        failed |= set(map(tuple, bad[["model", "profile_id", "iteration"]].astype(str).values))
    failed.add(("deepseek/deepseek-chat-v3", "P_WHITE_CW_HIGH_S", "19"))  # fix_single_missing_value.py
    trip = pd.Series(list(map(tuple, v3[["model", "profile_id", "iteration"]].astype(str).values)), index=v3.index)
    v3["narr_source"] = np.where(~nar, "", np.where(trip.isin(failed), "narr_rerun_same_script", "narr_main"))

    order = {"clinical": 0, "narrative": 1}
    v3 = v3.sort_values(["prompt_condition", "model", "profile_id", "iteration"],
                        key=lambda s: s.map(order) if s.name == "prompt_condition" else s,
                        kind="stable").reset_index(drop=True)
    cells = v3.groupby(["prompt_condition", "model", "profile_id"]).size()
    assert len(v3) == 28800 and cells.eq(30).all() and len(cells) == 960
    assert not v3.duplicated(["prompt_condition", "model", "profile_id", "iteration"]).any()
    v3.to_csv(os.path.join(BASE, "data", "model_outputs_v3.csv"), index=False)

    prov = (v3.groupby(["prompt_condition", "model", "row_source"]).size()
            .rename("rows").reset_index())
    prov.to_csv(os.path.join(OUTD, "corpus_v3_provenance.csv"), index=False)
    print(prov.to_string(index=False))
    print(f"\nrestored 43 GPT-4o-mini rows; December minus recovery PHQ-8, paired by cell and "
          f"iteration: mean {float((old.phq8_total_december - old.phq8_total_recovery).mean()):+.3f}")
    print(f"invalid PHQ-8 rows (an item outside 0..3): {int((~v3.phq8_valid).sum())}")
    print("narrative provenance:", v3[v3.narr_source != ""].groupby(["model", "narr_source"]).size().to_dict())
    print("clinical GLM-4.7 rows by race and source:")
    g = v3[(v3.prompt_condition == "clinical") & (v3.model == "z-ai/glm-4.7")]
    print(pd.crosstab(g.race, g.row_source).to_string())


if __name__ == "__main__":
    main()
