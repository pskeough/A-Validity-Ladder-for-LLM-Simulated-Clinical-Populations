"""
Shared loader for the decoding control. Both 56 (headline) and 57 (diagnostics) import from here so
the parsing and hygiene rules cannot drift apart between them.

Two hygiene problems exist in the raw log and are handled here, not in the analysis scripts:

  1. DUPLICATE DRAWS. Two runner processes wrote analysis/decoding_control_raw.jsonl concurrently
     during the August 3 collection, so 176 (model, arm, profile_id, draw) keys appear twice in the
     GLM-4.7 default arm. All 176 pairs are distinct generations, not replayed writes, so they are
     real draws that simply exceed the design. Keeping them would give those cells more ordered
     pairs than the rest and silently overweight them in every pooled estimate.

     Rule: for each (model, arm, profile_id, draw), keep the generation with the earliest timestamp.
     First write wins. The rule is applied before any outcome is computed and cannot see the totals,
     so it cannot select on the thing being measured.

  2. OVER-LENGTH CELLS. After deduplication a cell holds at most one generation per draw index, so
     it is capped at n_draws by construction. The cap is asserted rather than assumed.

Parsing matches the main analysis: PHQ-8 is eight integer items, summed and clipped to 0-24, and
binned on the standard severity bands.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
OUT = os.path.join(BASE, "analysis")
RAW = os.path.join(OUT, "decoding_control_raw.jsonl")
PREREG = os.path.join(OUT, "decoding_control_design.json")  # written before the run; not registered

BANDS = [(0, 4, "None"), (5, 9, "Mild"), (10, 14, "Moderate"),
         (15, 19, "Mod-severe"), (20, 24, "Severe")]


def band(t):
    for lo, hi, name in BANDS:
        if lo <= t <= hi:
            return name
    return None


def load(verbose=True, raw_path=None):
    """Return (df, report). df has one row per kept generation."""
    path = raw_path or RAW
    rows, n_lines, n_failed, n_unparsed = [], 0, 0, 0
    for line in open(path, encoding="utf-8"):
        n_lines += 1
        r = json.loads(line)
        if not r.get("ok"):
            n_failed += 1
            continue
        try:
            phq = json.loads(r["raw"])["PHQ8"]
        except Exception:
            n_unparsed += 1
            continue
        if not isinstance(phq, list) or len(phq) != 8 or not all(isinstance(v, int) for v in phq):
            n_unparsed += 1
            continue
        rows.append(dict(model=r["model"].split("/")[-1], arm=r["arm"],
                         profile_id=r["profile_id"], draw=r["draw"], ts=r["ts"],
                         total=int(np.clip(sum(phq), 0, 24)), items=tuple(phq),
                         provider=r.get("provider"), has_provenance="provider" in r,
                         latency_s=r.get("latency_s"), cost_usd=r.get("cost_usd") or 0.0,
                         completion_tokens=r.get("completion_tokens")))

    df = pd.DataFrame(rows)
    n_parsed = len(df)

    # Tie-break on provenance first, then time. Part of the August 3 collection predates the build
    # that records which upstream provider served the call, and GLM-4.7 is routed across eight of
    # them at differing quantization. Where the same draw exists both with and without that field,
    # the provenance-bearing row is the one that can be defended. Neither field is an outcome, so
    # the rule cannot select on the totals being measured.
    key = ["model", "arm", "profile_id", "draw"]
    df = (df.sort_values(["has_provenance", "ts"], ascending=[False, True])
            .drop_duplicates(subset=key, keep="first")
            .sort_values(key).reset_index(drop=True))
    n_dupes = n_parsed - len(df)

    df["cat"] = df.total.map(band)

    n_draws = json.load(open(PREREG, encoding="utf-8"))["n_draws"]
    sizes = df.groupby(key[:3]).size()
    assert sizes.max() <= n_draws, "cell exceeds n_draws after dedup: %s" % sizes.idxmax()

    report = dict(lines=n_lines, failed=n_failed, unparsed=n_unparsed,
                  parsed=n_parsed, duplicates_dropped=n_dupes, kept=len(df),
                  spend_usd=round(df.cost_usd.sum(), 4))
    if verbose:
        print("raw lines %d | api failures %d | unparsed %d | duplicate draws dropped %d | kept %d"
              % (n_lines, n_failed, n_unparsed, n_dupes, len(df)))
    return df, report


def both_arms_only(df, verbose=True):
    """Restrict to models that have both decoding arms. A one-armed model answers nothing."""
    have = df.groupby("model").arm.nunique()
    dropped = sorted(have[have < 2].index)
    if dropped and verbose:
        print("excluded, only one arm collected: %s" % ", ".join(dropped))
    return df[df.model.isin(sorted(have[have == 2].index))].copy()


def balanced_cells(df, verbose=True):
    """Restrict to cohorts a model measured in BOTH arms, so every contrast is within-cell."""
    keep = []
    for m, g in df.groupby("model"):
        arms = dict(list(g.groupby("arm")))
        if len(arms) < 2:
            continue
        common = set.intersection(*[set(a.profile_id) for a in arms.values()])
        dropped = set(g.profile_id) - common
        if dropped and verbose:
            print("  %s: %d cohort(s) present in only one arm, dropped from paired tests: %s"
                  % (m, len(dropped), ", ".join(sorted(dropped))))
        keep.append(g[g.profile_id.isin(common)])
    return pd.concat(keep, ignore_index=True) if keep else df.iloc[:0]
