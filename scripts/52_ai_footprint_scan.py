"""
AI-footprint scan for LaTeX or Markdown drafts.

The old contamination gate looked for em-dashes and nothing else. Em-dashes are the easiest tell
to remove and the least informative one, so a draft can pass it while still reading as machine
output. This scans for the constructions that actually survive a careless de-AI pass, all six
families drawn from a hand-marked review of the manuscript v12:

  1 comma-connective  a complete clause, a comma, then a connective appending an explanation or
                      consequence the reader did not ask for. Delete everything after the comma;
                      if the sentence still says what you meant, the tail was gloss.
  2 copula-pointer    "is what", "what X is", "that is", "it is that", "which is". The copula
                      points at a noun instead of asserting, deferring the real verb.
  3 anaphora          three or more consecutive sentences opening with the same words.
  4 meta-textual      the document talking about itself instead of asserting.
  5 long-sentence     45+ words, or 5+ commas.
  6 register-slip     conversational hedges, debate-club objection scaffolding, evaluative filler.

Plus two hard failures: em-dashes, and sentence-initial LLM connectives.

Also flags OVER-scrubbing: runs of four or more consecutive short declarative sentences, which is
what an over-aggressive fix produces and is its own tell.

Usage:
    python 52_ai_footprint_scan.py PATH [PATH ...] [--max-per-1k N] [--quiet]

Exit code 1 if any hard failure is present or the weighted rate exceeds --max-per-1k
(default 12 findings per 1,000 words), so it can be wired in as a gate.
"""
import argparse
import os
import re
import sys

HARD = [
    ("em-dash", r"---|\u2014",
     "never acceptable; use a comma, colon, or full stop"),
    ("llm-connective", r"(?m)(?:^|(?<=[.!?]\s))(Therefore|Thus|Moreover|Furthermore|Additionally|"
                       r"Consequently|Hence|Notably|Importantly|In essence|Ultimately),",
     "sentence-initial connective that reads as generated"),
]

SOFT = [
    ("comma-connective", r", (so|and so|and that|which is|and it is|making it|meaning that) ",
     "clause + comma + appended gloss"),
    ("copula-pointer", r"\b(is what|it is that|and that is|That is what)\b|"
                       r"\bWhat [a-z][\w' ]{2,40}? is\b",
     "copula pointing at a noun instead of asserting"),
    ("meta-textual", r"\b(this paper|this section|what we find is|we make no claim|"
                     r"it is worth noting|as noted above|in this work)\b",
     "the document talking about itself"),
    ("register-slip", r"\b(surprised us|one objection|a second objection|needs one \w+ first|"
                      r"argues otherwise|indicate otherwise|easy to state|worth naming|"
                      r"worth reporting|deserves its own)\b",
     "conversational or debate-club scaffolding"),
]

STRIP = [
    (r"(?m)^\s*%.*$", " "),                       # LaTeX comments
    (r"\\(begin|end)\{[^}]*\}", " "),             # environments
    (r"\\(cite|ref|autoref|label|url)\{[^}]*\}", " X "),
    (r"\$[^$]*\$", " N "),                        # inline math
]


def clean(text):
    for pat, rep in STRIP:
        text = re.sub(pat, rep, text)
    return text


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", text.replace("\n", " ")) if s.strip()]


def scan(path):
    raw = open(path, encoding="utf-8", errors="replace").read()
    body = clean(raw)
    lines = raw.split("\n")
    findings = []

    for name, pat, why in HARD + SOFT:
        hard = any(name == h[0] for h in HARD)
        for i, ln in enumerate(lines, 1):
            if re.match(r"^\s*%", ln):
                continue
            for m in re.finditer(pat, ln):
                a, b = max(0, m.start() - 45), min(len(ln), m.end() + 45)
                findings.append((("HARD" if hard else "soft"), name, i, ln[a:b].strip(), why))

    sents = sentences(body)
    for s in sents:
        w, c = len(s.split()), s.count(",")
        if w >= 45 or c >= 5:
            findings.append(("soft", "long-sentence", 0, s[:110], "%dw %dc" % (w, c)))

    # anaphora: 3+ consecutive sentences sharing their first two words
    for i in range(len(sents) - 2):
        heads = [" ".join(x.split()[:2]).lower() for x in sents[i:i + 3]]
        if len(set(heads)) == 1 and len(heads[0]) > 3:
            findings.append(("soft", "anaphora", 0, sents[i][:110],
                             "3 sentences open '%s'" % heads[0]))

    # over-scrub: 4+ consecutive short declaratives
    run = 0
    for s in sents:
        run = run + 1 if len(s.split()) <= 11 else 0
        if run == 4:
            findings.append(("soft", "over-scrubbed", 0, s[:110],
                             "4+ consecutive short sentences reads as machine-flattened"))
    return findings, len(body.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--max-per-1k", type=float, default=12.0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    failed = False
    for p in a.paths:
        if not os.path.exists(p):
            print("missing: %s" % p)
            failed = True
            continue
        findings, words = scan(p)
        hard = [f for f in findings if f[0] == "HARD"]
        counts = {}
        for f in findings:
            counts[f[1]] = counts.get(f[1], 0) + 1
        rate = 1000.0 * len(findings) / max(words, 1)

        print("\n=== %s  (%d words) ===" % (p, words))
        for k in sorted(counts, key=lambda k: -counts[k]):
            print("  %-18s %d" % (k, counts[k]))
        print("  %-18s %.1f per 1,000 words" % ("RATE", rate))

        if not a.quiet:
            for sev, name, ln, txt, why in findings:
                if sev == "HARD" or name in ("anaphora", "over-scrubbed"):
                    print("  [%s] %-16s L%-5s %s" % (sev, name, ln or "-", txt[:95]))

        if hard:
            print("  FAIL: %d hard finding(s)" % len(hard))
            failed = True
        if rate > a.max_per_1k:
            print("  FAIL: rate %.1f exceeds %.1f per 1,000 words" % (rate, a.max_per_1k))
            failed = True

    print("\n%s" % ("AI-FOOTPRINT SCAN FAILED" if failed else "AI-FOOTPRINT SCAN PASSED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
