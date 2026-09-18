"""
Reference verification gate.

Appendix A discloses that version 1 of this paper carried a citation that does not exist as
described. arXiv's CS moderation names hallucinated citations and fabricated references as the
trigger for its strongest sanction, so for this manuscript reference checking is not hygiene, it is
the risk category the paper actually carries. Every entry is verified here rather than trusted.

Each entry with a DOI is resolved against the DOI registry by content negotiation, and the returned
title, first author and year are compared against the bibliography. Entries without a DOI are
listed for manual confirmation, since a URL resolving is weaker evidence than a registry record.

Emits analysis/reference_check.csv. Exits non-zero if any DOI fails to resolve or disagrees with
the entry, so the check can sit alongside the two number gates.
"""
import json
import os
import re
import sys
import time
import urllib.request

import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
BIB = os.path.join(BASE, "paper_v2", "refs.bib")


def field(entry, key):
    """Brace-balanced field read. A regex to the closing brace breaks on nested braces, which
    several entries carry in LaTeX-escaped author names."""
    m = re.search(r"\b" + key + r"\s*=\s*\{", entry, re.I)
    if not m:
        return ""
    i, depth, out = m.end(), 1, []
    while depth:
        c = entry[i]
        depth += (c == "{") - (c == "}")
        if depth:
            out.append(c)
        i += 1
    return "".join(out)


def norm(s):
    # Punctuation is stripped before whitespace is collapsed, not after; doing it the other way
    # leaves a double space wherever a comma was and fails an otherwise identical title.
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


def resolve(doi):
    req = urllib.request.Request(
        "https://doi.org/" + doi,
        headers={"Accept": "application/vnd.citationstyles.csl+json",
                 "User-Agent": "reference-verification (research)"})
    return json.load(urllib.request.urlopen(req, timeout=30))


entries = re.split(r"\n(?=@)", open(BIB, encoding="utf-8").read().strip())
rows, failures = [], []
for e in entries:
    key = re.match(r"@\w+\{([^,]+)", e).group(1)
    doi, title, year = field(e, "doi"), field(e, "title"), field(e, "year")
    authors = field(e, "author")
    if not doi:
        rows.append(dict(key=key, doi="", resolves="no DOI", title_match="", author_match="",
                         registry_year="", bib_year=year))
        continue
    try:
        d = resolve(doi)
        rt = d.get("title")
        rt = " ".join(rt) if isinstance(rt, list) else (rt or "")
        got, want = norm(rt), norm(re.sub(r"[{}\\$]", "", title))
        ok_t = bool(want[:40] and (want[:40] in got or got[:40] in want))
        fam = (d.get("author") or [{}])[0].get("family", "")
        ok_a = (not fam) or norm(fam).split()[-1] in norm(authors)
        ryear = ((d.get("issued", {}) or {}).get("date-parts") or [[None]])[0][0]
        rows.append(dict(key=key, doi=doi, resolves="yes", title_match=ok_t, author_match=ok_a,
                         registry_year=ryear, bib_year=year))
        if not (ok_t and ok_a):
            failures.append(key)
            print(f"FLAG [{key}]\n     bib:      {want[:90]}\n     registry: {got[:90]}")
        else:
            note = "" if str(ryear) == year[:4] else f"  (year {ryear} vs {year})"
            print(f"OK   [{key}]{note}")
    except Exception as ex:
        rows.append(dict(key=key, doi=doi, resolves=f"ERROR: {ex}", title_match=False,
                         author_match=False, registry_year="", bib_year=year))
        failures.append(key)
        print(f"FAIL [{key}] {doi}: {ex}")
    time.sleep(0.2)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(BASE, "analysis", "reference_check.csv"), index=False)
nod = res[res.resolves == "no DOI"]
print(f"\n{int((res.resolves == 'yes').sum())} of {len(res)} entries verified against the DOI "
      f"registry; {len(nod)} carry no DOI and need manual confirmation:")
for k in nod.key:
    print(f"  {k}")
if failures:
    print(f"\nREFERENCE GATE FAILED: {len(failures)} ({failures})")
    sys.exit(1)
print("\nREFERENCE GATE PASSED: every DOI resolves to the cited work")
