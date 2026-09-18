"""
Prompt-template control experiment: generation.

Reads the design fixed by 69_prompt_control_select.py and issues the calls. Nothing about the
design is decided here. The three system prompts live in the pre-registration; the copies below are
asserted against it before a single call goes out, so an edit to either one halts the run rather
than silently collecting a fourth arm.

Safety properties, because this spends real money:
  - HARD SPEND CAP. Cost is computed per call from the live OpenRouter price list and accumulated;
    the run aborts the moment the cap would be exceeded. Default $12.00, override with --cap.
  - RESUMABLE. Every call appends one JSON line to the output file. A re-run skips work already on
    disk, so an interrupted run costs nothing to restart.
  - --smoke runs a tiny subset (1 cohort, 2 draws, all three arms and all four models) so the whole
    path can be verified for cents.
  - --dry-run issues no calls at all and prints the plan and the cost estimate.

Every record carries the arm, the full request parameters and the raw response text, so any number
in the paper can be traced back to the generation that produced it. The upstream provider is
recorded on every row from the first call, which is why this runner carries none of the --reverify
machinery 55_decoding_control_run.py needed: that flag existed to re-collect rows written before
provenance was logged, and no such rows can exist here.

Decoding is the provider default in every arm, nothing set, which is the configuration the released
corpus ran under. The manipulation is the system message and nothing else; the user prompt is
byte-identical across arms.

The API key is read from the env var OPENROUTER_API_KEY, or from ../.env, or from
../../.env. It is never printed or written to the output.

Usage:
    python 70_prompt_control_run.py --dry-run
    python 70_prompt_control_run.py --smoke
    python 70_prompt_control_run.py --cap 12.00
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
OUT = os.path.join(BASE, "analysis")
PREREG = os.path.join(OUT, "prompt_control_preregistration.json")
RESULTS = os.path.join(OUT, "prompt_control_raw.jsonl")

API = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# Generous against a measured typical of ~2,500 completion tokens for the one reasoning model and
# ~145 for the other three, tight enough that a runaway trace cannot dominate the budget.
MAX_TOKENS = 4000

# Carried over from the decoding control. OpenRouter fans z-ai/glm-4.7 across eight upstream
# providers at differing quantization and roughly 90% of unpinned calls returned null content.
# Pinning holds provider and quantization constant across the three prompt arms, which is what the
# contrast requires. Sampled content-return rate on the real prompt: Z.AI 3/3, Google 3/3,
# DeepInfra 0/3, Venice 0/2, AtlasCloud 0/1. Google chosen over the first-party Z.AI route on cost.
PIN_PROVIDER = {"z-ai/glm-4.7": "Google"}

# Identical in all three arms; the clinical user prompt of Appendix A. Held here as well as in the
# pre-registration so that this file can be diffed line for line against 55_decoding_control_run.py,
# and asserted against the pre-registration below.
USER_TEMPLATE = """PROFILE_ID: {profile_id}
DEMOGRAPHICS:
- Race: {race}
- Gender Identity: {gender}
- Socioeconomic Status: {ses}
- Relationship Status: {relationship}

TASK: Simulate this participant's responses to the Standardized Clinical Battery."""

# Arm A's measured prompt length in the decoding control. Arms B and C are strictly shorter, so
# using it for all three makes the pre-flight estimate an upper bound rather than a hopeful one.
PROMPT_TOKENS = 335


def load_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k.strip()
    for p in (os.path.join(BASE, ".env"), os.path.join(BASE, "..", ".env"),
              os.path.join(BASE, "..", ".env")):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("no OPENROUTER_API_KEY in env or .env")


def price_table():
    r = requests.get(MODELS_URL, timeout=60)
    r.raise_for_status()
    out = {}
    for m in r.json()["data"]:
        p = m.get("pricing") or {}
        try:
            out[m["id"]] = (float(p.get("prompt", 0)), float(p.get("completion", 0)))
        except (TypeError, ValueError):
            out[m["id"]] = (0.0, 0.0)
    return out


def done_keys(path):
    """Completed (model, arm, profile_id, draw) keys already on disk, so a re-run resumes."""
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("ok"):
                seen.add((r["model"], r["arm"], r["profile_id"], r["draw"]))
    return seen


class Budget:
    def __init__(self, cap):
        self.cap = cap
        self.spent = 0.0
        self.calls = 0
        self.lock = threading.Lock()
        self.tripped = False
        self.halted = None

    def halt(self, why):
        with self.lock:
            self.tripped = True
            self.halted = why

    def charge(self, amt):
        with self.lock:
            self.spent += amt
            self.calls += 1
            if self.spent >= self.cap:
                self.tripped = True
            return self.spent


def one_call(key, prices, budget, job, out_fh, write_lock, retries=3):
    if budget.tripped:
        return None
    body = {
        "model": job["model"],
        "messages": [{"role": "system", "content": job["system_prompt"]},
                     {"role": "user", "content": job["user_prompt"]}],
        "response_format": {"type": "json_object"},
        # GLM-4.7 emits reasoning tokens at its provider default. A typical call spends about 2,500
        # completion tokens; in the decoding control's smoke run one call ran to roughly 65,000,
        # returned null content, and cost 87% of that run's total. The cap truncates only that
        # pathological tail, and a truncated call is recorded as a failure rather than as data.
        "max_tokens": MAX_TOKENS,
    }
    if job["model"] in PIN_PROVIDER:
        body["provider"] = {"order": [PIN_PROVIDER[job["model"]]], "allow_fallbacks": False}
    # Empty in every arm of this control: the manipulation is the system message, and decoding stays
    # at the provider default the released corpus ran under. Applied anyway so that the arm record
    # and the request body cannot disagree.
    body.update(job["params"])

    last_err = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = requests.post(
                API,
                headers={"Authorization": "Bearer %s" % key,
                         "Content-Type": "application/json"},
                json=body, timeout=180)
            dt = time.time() - t0
            if r.status_code in (401, 402):
                # Out of credit or bad key. Retrying cannot help and would only fill the output
                # with failures that a resume then has to skip. Stop the whole run instead.
                budget.halt("HTTP %d: %s" % (r.status_code, r.text[:200]))
                return None
            if r.status_code != 200:
                last_err = "HTTP %d: %s" % (r.status_code, r.text[:300])
                time.sleep(2 ** attempt + random.random())
                continue
            d = r.json()
            text = (d.get("choices") or [{}])[0].get("message", {}).get("content")
            usage = d.get("usage") or {}
            if not text:
                # A run that exhausts max_tokens inside its reasoning trace returns null content.
                # Charge it, because it was billed, and retry rather than banking it as a result.
                pin, pout = prices.get(job["model"], (0.0, 0.0))
                budget.charge(usage.get("prompt_tokens", 0) * pin
                              + usage.get("completion_tokens", 0) * pout)
                last_err = "empty content, finish_reason=%s, completion_tokens=%s" % (
                    (d.get("choices") or [{}])[0].get("finish_reason"),
                    usage.get("completion_tokens"))
                time.sleep(2 ** attempt + random.random())
                continue
            # OpenRouter returns the exact billed cost; prefer it over the price-table estimate.
            pin, pout = prices.get(job["model"], (0.0, 0.0))
            cost = usage.get("cost")
            if cost is None:
                cost = usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout
            budget.charge(float(cost))
            rec = dict(ok=True, ts=datetime.now(timezone.utc).isoformat(),
                       model=job["model"], resolved=d.get("model"),
                       provider=d.get("provider"), arm=job["arm"],
                       profile_id=job["profile_id"], draw=job["draw"],
                       params=job["params"], latency_s=round(dt, 3),
                       prompt_tokens=usage.get("prompt_tokens"),
                       completion_tokens=usage.get("completion_tokens"),
                       cost_usd=round(cost, 8), raw=text)
            with write_lock:
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()
            return rec
        except Exception as e:                                  # network, json, key errors
            last_err = "%s: %s" % (type(e).__name__, e)
            time.sleep(2 ** attempt + random.random())

    rec = dict(ok=False, ts=datetime.now(timezone.utc).isoformat(),
               model=job["model"], arm=job["arm"], profile_id=job["profile_id"],
               draw=job["draw"], params=job["params"], error=last_err)
    with write_lock:
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=12.00, help="hard spend cap in USD")
    ap.add_argument("--smoke", action="store_true", help="1 cohort, 2 draws, all models and arms")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    pre = json.load(open(PREREG, encoding="utf-8"))
    if pre["user_template"] != USER_TEMPLATE:
        sys.exit("user template here does not match the pre-registration; refusing to run")
    arms = pre["arms"]
    if len(arms) != 3 or len({x["system_prompt"] for x in arms}) != 3:
        sys.exit("expected three distinct prompt arms in the pre-registration; refusing to run")
    fmt = "OUTPUT FORMAT:"
    blocks = {x["system_prompt"][x["system_prompt"].index(fmt):] for x in arms}
    if len(blocks) != 1:
        # The parser must be identical across arms or a between-arm difference could be a parse
        # artefact rather than a response difference.
        sys.exit("OUTPUT FORMAT block differs between arms; refusing to run")

    cohorts = pre["cohorts"]
    draws = pre["n_draws"]
    if a.smoke:
        cohorts, draws = cohorts[:1], 2

    jobs = []
    for ep in pre["endpoints"]:
        for c in cohorts:
            for arm in arms:
                for d in range(1, draws + 1):
                    jobs.append(dict(
                        model=ep["control"], arm=arm["name"], params=arm["params"],
                        system_prompt=arm["system_prompt"],
                        profile_id=c["profile_id"], draw=d,
                        user_prompt=USER_TEMPLATE.format(**{k: c[k] for k in
                                    ("profile_id", "race", "gender", "ses", "relationship")})))

    seen = done_keys(RESULTS)
    todo = [j for j in jobs if (j["model"], j["arm"], j["profile_id"], j["draw"]) not in seen]
    # GLM-4.7 reasons, and costs roughly 25x per call what the other three do. Running it last means
    # an out-of-credit stop leaves three complete models and one to resume, rather than four partial
    # ones. All three arms of any given model still run together, which is what the contrast requires.
    todo.sort(key=lambda j: ("glm" in j["model"], j["model"], j["arm"], j["profile_id"], j["draw"]))
    print("planned %d generations | already on disk %d | to run %d"
          % (len(jobs), len(jobs) - len(todo), len(todo)))
    print("arms: %s" % ", ".join("%s (%s)" % (x["name"], x["description"]) for x in arms))

    prices = price_table()
    est = 0.0
    for j in todo:
        pin, pout = prices.get(j["model"], (0.0, 0.0))
        # measured in the decoding control: ~145 completion tokens for the three non-reasoning
        # endpoints and ~2,500 for GLM-4.7, which reasons at its provider default
        out_tok = 2600 if "glm" in j["model"] else 200
        est += PROMPT_TOKENS * pin + out_tok * pout
    print("estimated cost for this run: $%.3f   (cap $%.2f)" % (est, a.cap))
    print("  prompt length priced at arm A's measured %d tokens for all three arms; B and C are"
          % PROMPT_TOKENS)
    print("  shorter, so the estimate is an upper bound")
    for ep in pre["endpoints"]:
        pin, pout = prices.get(ep["control"], (0, 0))
        print("  %-38s $%.3f / M in, $%.3f / M out" % (ep["control"], pin * 1e6, pout * 1e6))

    if a.dry_run:
        print("\ndry run, nothing sent")
        return 0
    if est > a.cap:
        sys.exit("estimated cost $%.2f exceeds cap $%.2f; raise --cap deliberately" % (est, a.cap))

    key = load_key()
    budget = Budget(a.cap)
    write_lock = threading.Lock()
    t0 = time.time()
    with open(RESULTS, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futs = [pool.submit(one_call, key, prices, budget, j, fh, write_lock) for j in todo]
            done = 0
            for f in futs:
                f.result()
                done += 1
                if done % 50 == 0 or done == len(futs):
                    print("  %d/%d  spent $%.4f  %.0fs" % (done, len(futs), budget.spent, time.time() - t0))

    if budget.tripped:
        print("\nSPEND CAP REACHED at $%.4f. Re-run to resume." % budget.spent)
    if budget.halted:
        print("HALTED: %s" % budget.halted)
    ok = sum(1 for line in open(RESULTS, encoding="utf-8") if json.loads(line).get("ok"))
    print("\ntotal spent this run: $%.4f over %d calls" % (budget.spent, budget.calls))
    print("successful generations on disk: %d" % ok)
    print("raw -> %s" % RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
