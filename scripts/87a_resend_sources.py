"""Trace the recovery_inplace rows of corpus v3 to the January scripts that produced them.

Why. The December clinical run left 1,532 calls without a parsed answer. In January
generation/recovery/verify_run1_refusals.py resent those prompts once with a tolerant JSON parser,
and generation/recovery/merge_recovered_data.py wrote the parsed replies into the December rows,
keeping their run_id and timestamp (row_source recovery_inplace). The calls that still failed were
resent by generation/retry_failed.py, and later passes (slow_recovery.py, last_mile_recovery_opt.py)
handled the rest. The outputs of the first two scripts were recovered from the original project
folder and copied to data/raw:
  run1_resend_verification_2026-01.csv   verify_run1_refusals.py output (reply preview per call)
  run1_retry_results_2026-01.csv         retry_failed.py output

Checks:
  1. every recovery_inplace row matches one of the two files on model, profile, iteration and all
     eight PHQ-8 items;
  2. the system prompt in both scripts is the same string as in generation/main.py, and the user
     prompt differs from main.py's only in the PROFILE_ID line.

Output: analysis/brm/87a_resend_sources.csv
"""
import ast
import json
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "generation")
ITEMS = [f"phq8_{i}" for i in range(1, 9)]
KEY = ["model", "profile_id", "iteration"]


def system_prompt(path):
    """The first triple-quoted string that starts with ROLE: in a script."""
    src = open(path, encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("ROLE:"):
            return node.value
    raise ValueError(f"no system prompt in {path}")


def user_template(path):
    src = open(path, encoding="utf-8").read()
    m = re.search(r'f"""(PROFILE_ID: .*?)"""', src, re.S)
    return m.group(1)


def main():
    v3 = pd.read_csv(os.path.join(ROOT, "data", "model_outputs_v3.csv"), low_memory=False)
    inplace = v3[v3.row_source == "recovery_inplace"]
    ver = pd.read_csv(os.path.join(ROOT, "data", "raw", "run1_resend_verification_2026-01.csv"))
    retry = pd.read_csv(os.path.join(ROOT, "data", "raw", "run1_retry_results_2026-01.csv"))

    fixed = ver[ver.verification_result == "json_error_fixed"].copy()
    # The preview keeps the first 200 characters of the reply, which always include the PHQ8 array.
    phq = fixed.raw_response_preview.map(
        lambda s: json.loads(re.search(r'"PHQ8"\s*:\s*(\[[^\]]*\])', s).group(1)))
    for i, c in enumerate(ITEMS):
        fixed[c] = phq.map(lambda v: v[i])

    def matches(src):
        m = inplace[KEY + ITEMS].merge(src[KEY + ITEMS], on=KEY, how="left", suffixes=("", "_s"))
        return (m[ITEMS].values == m[[c + "_s" for c in ITEMS]].values).all(axis=1)

    by_verify = matches(fixed)
    by_retry = matches(retry)
    script = pd.Series("unmatched", index=inplace.index)
    script[by_retry] = "retry_failed.py"
    script[by_verify] = "verify_run1_refusals.py"
    out = (inplace.assign(script=script.values).groupby(["model", "script"]).size()
           .rename("rows").reset_index())
    out.to_csv(os.path.join(ROOT, "analysis", "brm", "87a_resend_sources.csv"), index=False)
    print(out.to_string(index=False))
    assert (script != "unmatched").all(), "a recovery_inplace row matches neither January file"

    main_py = os.path.join(GEN, "main.py")
    sys_main, usr_main = system_prompt(main_py), user_template(main_py)
    for f in [os.path.join(GEN, "recovery", "verify_run1_refusals.py"), os.path.join(GEN, "retry_failed.py")]:
        same_sys = system_prompt(f) == sys_main
        usr = user_template(f)
        same_rest = usr.split("\n", 1)[1] .replace("p_data", "profile").replace("row", "profile") == \
            usr_main.split("\n", 1)[1]
        print(f"{os.path.basename(f)}: system prompt identical to main.py: {same_sys}; "
              f"PROFILE_ID line: {usr.splitlines()[0]!r} vs {usr_main.splitlines()[0]!r}; "
              f"rest of user prompt identical: {same_rest}")
        assert same_sys


if __name__ == "__main__":
    main()
