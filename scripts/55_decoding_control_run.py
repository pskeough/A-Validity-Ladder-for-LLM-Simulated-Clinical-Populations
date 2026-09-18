"""
Decoding control experiment: generation.

Reads the design fixed by 54_decoding_control_select.py and issues the calls. Nothing about the
design is decided here.

Safety properties, because this spends real money:
  - HARD SPEND CAP. Cost is computed per call from the live OpenRouter price list and accumulated;
    the run aborts the moment the cap would be exceeded. Default $6.00, override with --cap.
  - RESUMABLE. Every call appends one JSON line to the output file. A re-run skips work already on
    disk, so an interrupted run costs nothing to restart.
  - --smoke runs a tiny subset (1 cohort, 2 draws) so the whole path can be verified for cents.
  - --dry-run issues no calls at all and prints the plan and the cost estimate.

Every record carries the full request and the raw response text, so any number in the paper can be
traced back to the generation that produced it.

The API key is read from PSYCHBENCH env var OPENROUTER_API_KEY, or from ../.env, or from
../../PsychBench/.env. It is never printed or written to the output.

Usage:
    python 55_decoding_control_run.py --dry-run
    python 55_decoding_control_run.py --smoke
    python 55_decoding_control_run.py --cap 6.00
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
PREREG = os.path.join(OUT, "decoding_control_preregistration.json")
RESULTS = os.path.join(OUT, "decoding_control_raw.jsonl")

API = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# Generous against a measured typical of ~2,500 completion tokens for the one reasoning model and
# ~145 for the other three, tight enough that a runaway trace cannot dominate the budget.
MAX_TOKENS = 4000

# Sampled content-return rate on the real prompt: Z.AI 3/3, Google 3/3, DeepInfra 0/3, Venice 0/2,
# AtlasCloud 0/1. Google chosen over the first-party Z.AI route on cost.
PIN_PROVIDER = {"z-ai/glm-4.7": "Google"}

# Reconstructed from Appendix A. The clinical condition administered a 20-item PCL-5; Appendix A
# prints the format line without an item count, and the released data carry 20 populated PCL-5
# item columns under clinical framing. That reconstruction is recorded here rather than assumed.
SYSTEM_PROMPT = """ROLE:
You are a Clinical Simulation Engine designed for a synthetic data study.
Your task is to generate PROBABILISTIC responses for a "Standardized Patient" based on a provided demographic profile.

OPERATIONAL CONSTRAINTS (STRICT):
1. NO MORALIZING: Do not offer help, advice, or disclaimers. You are simulating data, not treating a patient.
2. PROBABILISTIC ACCURACY: Base your responses on the statistical likelihood of symptoms for this specific demographic intersection in the US population (referencing NSDUH/CDC trends).
3. SINGLE-SHOT OUTPUT: You must output the results for ALL four screening tools in a single valid JSON object.
4. NUMERIC ONLY: Do not generate text explanations. Return integers.

OUTPUT FORMAT:
{
  "PHQ8": [int, int...], // 8 items, 0-3 scale
  "GAD7": [int, int...], // 7 items, 0-3 scale
  "AUDIT_C": [int, int...], // 3 items, 0-4 scale
  "PCL5": [int, int...] // 20 items, 0-4 scale
}"""

USER_TEMPLATE = """PROFILE_ID: {profile_id}
DEMOGRAPHICS:
- Race: {race}
- Gender Identity: {gender}
- Socioeconomic Status: {ses}
- Relationship Status: {relationship}

TASK: Simulate this participant's responses to the Standardized Clinical Battery."""


def load_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k.strip()
    for p in (os.path.join(BASE, ".env"), os.path.join(BASE, "..", ".env"),
              os.path.join(BASE, "..", "PsychBench", ".env")):
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
    """Map completed key -> whether any row for it records the upstream provider.

    The provenance flag matters because a later build of this runner began writing which upstream
    provider served each call. GLM-4.7 is routed across eight of them at differing quantization, so
    a decoding contrast drawn from rows that do not name the provider rests on an assumption that
    both arms were served identically. --reverify re-collects exactly those rows.
    """
    seen = {}
    if not os.path.exists(path):
        return seen
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("ok"):
                k = (r["model"], r["arm"], r["profile_id"], r["draw"])
                seen[k] = seen.get(k, False) or ("provider" in r)
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
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": job["user_prompt"]}],
        "response_format": {"type": "json_object"},
        # GLM-4.7 emits reasoning tokens at its provider default. A typical call spends about 2,500
        # completion tokens; in the smoke run one call ran to roughly 65,000, returned null content,
        # and cost 87% of that run's total. The cap truncates only that pathological tail, and a
        # truncated call is recorded as a failure rather than as data.
        "max_tokens": MAX_TOKENS,
    }
    # See the amendment in the pre-registration. Under default routing GLM-4.7 is served by eight
    # upstream providers at differing quantization, and roughly 90% of calls came back with null
    # content. Pinning holds provider and quantization constant across both decoding arms.
    if job["model"] in PIN_PROVIDER:
        body["provider"] = {"order": [PIN_PROVIDER[job["model"]]], "allow_fallbacks": False}
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
    ap.add_argument("--cap", type=float, default=6.00, help="hard spend cap in USD")
    ap.add_argument("--smoke", action="store_true", help="1 cohort, 2 draws, all models and arms")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reverify", default="", help="comma-separated model slugs whose rows lacking "
                                                   "a recorded provider should be re-collected, "
                                                   "e.g. 'z-ai/glm-4.7'")
    a = ap.parse_args()

    pre = json.load(open(PREREG, encoding="utf-8"))
    cohorts = pre["cohorts"]
    draws = pre["n_draws"]
    if a.smoke:
        cohorts, draws = cohorts[:1], 2

    jobs = []
    for ep in pre["endpoints"]:
        for c in cohorts:
            for arm in pre["arms"]:
                for d in range(1, draws + 1):
                    jobs.append(dict(
                        model=ep["control"], arm=arm["name"], params=arm["params"],
                        profile_id=c["profile_id"], draw=d,
                        user_prompt=USER_TEMPLATE.format(**{k: c[k] for k in
                                    ("profile_id", "race", "gender", "ses", "relationship")})))

    seen = done_keys(RESULTS)
    # --reverify re-collects only those rows of a model that carry no provider field, which is the
    # minimum needed to put both decoding arms on verifiably constant infrastructure. Old rows stay
    # in the log; the loader's tie-break prefers the provenance-bearing copy.
    reverify = {s.strip() for s in a.reverify.split(",") if s.strip()}
    if reverify:
        released = [k for k, prov in seen.items() if k[0] in reverify and not prov]
        for k in released:
            del seen[k]
        print("re-verifying %s: %d rows lack a recorded provider and will be re-collected"
              % (sorted(reverify), len(released)))
    todo = [j for j in jobs if (j["model"], j["arm"], j["profile_id"], j["draw"]) not in seen]
    # GLM-4.7 reasons, and costs roughly 25x per call what the other three do. Running it last means
    # an out-of-credit stop leaves three complete models and one to resume, rather than four partial
    # ones. Both arms of any given model still run together, which is what the contrast requires.
    todo.sort(key=lambda j: ("glm" in j["model"], j["model"], j["arm"], j["profile_id"], j["draw"]))
    print("planned %d generations | already on disk %d | to run %d"
          % (len(jobs), len(jobs) - len(todo), len(todo)))

    prices = price_table()
    est = 0.0
    for j in todo:
        pin, pout = prices.get(j["model"], (0.0, 0.0))
        # measured in the smoke run: ~335 prompt tokens throughout; ~145 completion for the three
        # non-reasoning endpoints and ~2,500 for GLM-4.7, which reasons at its provider default
        out_tok = 2600 if "glm" in j["model"] else 200
        est += 335 * pin + out_tok * pout
    print("estimated cost for this run: $%.3f   (cap $%.2f)" % (est, a.cap))
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
    ok = sum(1 for line in open(RESULTS, encoding="utf-8") if json.loads(line).get("ok"))
    print("\ntotal spent this run: $%.4f over %d calls" % (budget.spent, budget.calls))
    print("successful generations on disk: %d" % ok)
    print("raw -> %s" % RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
