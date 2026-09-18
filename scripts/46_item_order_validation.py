"""Do these models order the PHQ-8 the way the scoring code assumes?

The generation prompt named the instrument and gave an item count, never the item wording, so each
model chose which symptom went in which position of the returned array. Two results depend on that
choice being canonical. Section 4.8's gateway rule reads positions 1 and 2 as anhedonia and
depressed mood, and Section 4.5's correlation matrices assume position k means the same symptom in
every model. Neither was ever checked.

This asks each endpoint, ten times, to enumerate the eight items in the order it would score them,
and matches what comes back against DPQ010 through DPQ080 by keyword. The two gateway positions are
reported separately, because they are the ones a headline result rests on.

Same four routes as the study, same JSON response format. 40 calls.

Emits analysis/item_order_validation.csv.
Run: python scripts/46_item_order_validation.py
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "analysis")
MODELS = ["openai/gpt-4o-mini", "google/gemini-3-flash-preview",
          "deepseek/deepseek-chat-v3", "z-ai/glm-4.7"]
N = 10

# DPQ010--DPQ080. Each entry is the set of keywords that identify the item; matching is on content
# so that paraphrase does not count as a mismatch, since the question is ordering and not wording.
# Keys must not collide across items. A first pass used "down" for item 2, which also matched
# item 6's "let yourself or your family down" and scored every model 7/8 on the matcher's error
# rather than the model's. Each string here appears in exactly one canonical item.
CANON = [
    ("anhedonia", ("interest or pleasure", "anhedonia", "little interest")),
    ("depressed mood", ("hopeless", "depressed", "feeling down", "down, depress")),
    ("sleep", ("sleep", "asleep", "insomnia")),
    ("energy", ("tired", "energy", "fatigu")),
    ("appetite", ("appetite", "overeating", "eating")),
    ("self-worth", ("bad about yourself", "failure", "family down", "worthless")),
    ("concentration", ("concentrat", "focus")),
    ("psychomotor", ("slowly", "fidget", "restless", "psychomotor")),
]

PROMPT = ("List the eight items of the PHQ-8 in the standard order in which they are scored. "
          'Return only JSON: {"items": ["item 1 text", ..., "item 8 text"]}')


def classify(text):
    """Which canonical item does this string describe? Best match by key count, None if no key hits."""
    t = text.lower()
    scored = [(sum(k in t for k in keys), name) for name, keys in CANON]
    best, name = max(scored)
    return name if best else None


def call(model, key):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    # A reasoning model can return null content with the answer in the reasoning field, and several
    # routes wrap JSON in a markdown fence despite json_object being requested.
    txt = msg.get("content") or msg.get("reasoning") or ""
    return re.sub(r"^\s*```(?:json)?|```\s*$", "", txt.strip()).strip()


def main():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        env = os.path.join(BASE, "..", ".env")
        if os.path.exists(env):
            for line in open(env, encoding="utf-8"):
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("set OPENROUTER_API_KEY, or put it in PsychBench/.env")

    import pandas as pd
    rows = []
    for model in MODELS:
        for i in range(1, N + 1):
            try:
                raw = call(model, key)
                obj = json.loads(raw)
                # Some routes return the array under a different key, or bare.
                items = obj.get("items") if isinstance(obj, dict) else obj
                if items is None and isinstance(obj, dict):
                    items = next((v for v in obj.values() if isinstance(v, list)), [])
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                    KeyError, TypeError, TimeoutError) as e:
                rows.append(dict(model=model, iteration=i, error=type(e).__name__))
                print(f"  {model:32s} {i:2d}  FAILED {type(e).__name__}")
                continue
            got = [classify(re.sub(r"^\s*\d+[.)]\s*", "", str(x))) for x in items]
            exact = [got[k] == CANON[k][0] if k < len(got) else False for k in range(8)]
            rows.append(dict(model=model, iteration=i, n_returned=len(items),
                             pos1=got[0] if got else None,
                             pos2=got[1] if len(got) > 1 else None,
                             gateway_ok=bool(exact[0] and exact[1]),
                             all8_ok=bool(all(exact)),
                             n_correct=sum(exact), order="|".join(str(g) for g in got)))
            print(f"  {model:32s} {i:2d}  gateway={'ok ' if exact[0] and exact[1] else 'NO '}"
                  f" correct={sum(exact)}/8")
            time.sleep(0.3)

    d = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "item_order_validation.csv"), index=False)

    ok = d[d.get("gateway_ok").notna()] if "gateway_ok" in d else d
    print("\n" + "=" * 62)
    if len(ok):
        g = ok.groupby("model").agg(n=("iteration", "size"),
                                    gateway=("gateway_ok", "mean"),
                                    all8=("all8_ok", "mean"),
                                    mean_correct=("n_correct", "mean")).round(3)
        print(g.to_string())
        print(f"\npositions 1 and 2 canonical in {100 * ok.gateway_ok.mean():.1f}% of "
              f"{len(ok)} responses; all eight in {100 * ok.all8_ok.mean():.1f}%")
    if "error" in d and d.error.notna().any():
        print(f"failed calls: {int(d.error.notna().sum())}")


if __name__ == "__main__":
    main()
