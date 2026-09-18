"""Fetch full texts for the bibliography, by whatever route each reference actually has.

The paper-library skill resolves against arXiv, which suits an ML preprint bibliography. This one is
mostly clinical journals, federal survey documentation and conference proceedings, so the arXiv path
reached 4 of 40. Two things went wrong there and both are recorded here rather than worked around:

  * the skill's arXiv-ID regex matches DOI fragments. 10.1001/jamainternmed.2024.2544 became the
    arXiv id "2024.2544" and 10.1016/j.ypmed.2022.106988 became "2022.10698". All four such
    phantoms 404'd, so they failed safely, but a DOI whose tail happens to match a real arXiv id
    would silently download an unrelated paper, which is the exact failure the skill's
    title-verification exists to catch.
  * 32 entries came back unresolved, which is the honest answer for a bibliography like this one.

Unpaywall says 23 of the 25 DOI-bearing references are open access, so the routes below are ordered
by reliability: publisher OA link, then PubMed Central, then arXiv, then a direct URL for the
government documents. Every download is title-verified against the bibliography entry, because a
resolver returning the wrong paper is the failure that survives casual checking.

Emits paper_library/pdf/, paper_library/txt/, paper_library/verification.json.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
BIB = BASE / "paper_v2" / "refs.bib"
LIB = BASE / "paper_library"
# Crossref and Unpaywall ask for a contact address so they can route their polite pool and reach
# whoever is hammering them. Set CONTACT_EMAIL before running; the fallback is the address printed
# on the paper, which is where a query would land anyway.
EMAIL = os.environ.get("CONTACT_EMAIL", "anonymous@example.org")
UA = "Mozilla/5.0 (academic literature retrieval)"

# Entries with no DOI, resolved by hand to a stable public source. Recorded explicitly so a reader
# can see exactly where each came from; nothing here is guessed.
MANUAL = {
    "raji2021everything": "https://arxiv.org/pdf/2111.15366",
    "blodgett2021salmon": "https://aclanthology.org/2021.acl-long.81.pdf",
    "northcutt2021pervasive": "https://arxiv.org/pdf/2103.14749",
    "padmakumar2024diversity": "https://arxiv.org/pdf/2309.05196",
    "santurkar2023whose": "https://arxiv.org/pdf/2303.17548",
    "dominguezolmedo2024questioning": "https://arxiv.org/pdf/2306.07951",
    "wang2024patientpsi": "https://arxiv.org/pdf/2405.19660",
    "cheng2023compost": "https://aclanthology.org/2023.emnlp-main.669.pdf",
    "cheng2023marked": "https://aclanthology.org/2023.acl-long.84.pdf",
    "brody2018depression": "https://www.cdc.gov/nchs/data/databriefs/db303.pdf",
    "nchs2022srvydesc":
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2022/srvydesc-508.pdf",
}


def field(entry, key):
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
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


def curl(url, dest):
    r = subprocess.run(["curl", "-sSL", "-A", UA, "--max-time", "60", "-o", str(dest),
                        "-w", "%{http_code}"], capture_output=True, text=True)
    return r.stdout.strip()


def curl_url(url, dest):
    r = subprocess.run(["curl", "-sSL", "-A", UA, "--max-time", "60", "-o", str(dest),
                        "-w", "%{http_code}", url], capture_output=True, text=True)
    return r.stdout.strip()


def get_json(url):
    out = subprocess.run(["curl", "-sS", "-L", "-A", UA, "--max-time", "25", url],
                         capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except Exception:
        return {}


def routes_for(key, doi):
    """Ordered candidate URLs. Publisher OA first, then PMC, then any manual entry."""
    urls = []
    if doi:
        up = get_json(f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}")
        for loc in ([up.get("best_oa_location")] or []) + (up.get("oa_locations") or []):
            if not loc:
                continue
            for u in (loc.get("url_for_pdf"), loc.get("url")):
                if u and u not in urls:
                    urls.append(u)
        # PubMed Central, which serves a clean PDF where the publisher link is an HTML landing page
        pm = get_json("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
                      f"?ids={doi}&format=json&tool=lit&email={EMAIL}")
        for rec in pm.get("records", []):
            if rec.get("pmcid"):
                urls.append(f"https://europepmc.org/api/fulltextRepo?pprId={rec['pmcid']}&type=FILE&fileName=EMS.pdf")
                urls.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{rec['pmcid']}/pdf/")
    if key in MANUAL:
        urls.insert(0, MANUAL[key])
    return urls


def main():
    (LIB / "pdf").mkdir(parents=True, exist_ok=True)
    (LIB / "txt").mkdir(parents=True, exist_ok=True)
    import fitz

    entries = re.split(r"\n(?=@)", BIB.read_text(encoding="utf-8").strip())
    only = sys.argv[1:] or None
    results = []
    for e in entries:
        key = re.match(r"@\w+\{([^,]+)", e).group(1)
        if only and key not in only:
            continue
        title, doi = field(e, "title"), field(e, "doi")
        pdf = LIB / "pdf" / f"{key}.pdf"
        rec = {"key": key, "title": re.sub(r"[{}\\]", "", title), "doi": doi,
               "status": "no route", "source": "", "pages": 0, "title_match": None}

        if pdf.exists() and pdf.stat().st_size > 20000:
            rec["status"] = "cached"
        else:
            for url in routes_for(key, doi):
                code = curl_url(url, pdf)
                if code == "200" and pdf.exists() and pdf.stat().st_size > 20000:
                    head = pdf.read_bytes()[:5]
                    if head.startswith(b"%PDF"):
                        rec["status"], rec["source"] = "downloaded", url
                        break
                    pdf.unlink(missing_ok=True)
                time.sleep(0.4)

        if pdf.exists() and pdf.stat().st_size > 20000:
            try:
                d = fitz.open(pdf)
                text = "".join(p.get_text() for p in d)
                # A landing page saved as PDF parses cleanly and contains no article, which is the
                # quietest failure here: pages > 0, text non-empty, nothing of the paper in it.
                if any(k in text[:1500] for k in ("Skip to main content",
                                                  "An official website of the United States",
                                                  "Log in Dashboard", "Access keys NCBI Homepage")):
                    rec["status"] = "landing page, not article"
                    pdf.unlink(missing_ok=True)
                    results.append(rec)
                    print(f"  {'LANDING':9s} {rec['status']:12s} {key[:30]:32s}   0p")
                    continue
                (LIB / "txt" / f"{key}.txt").write_text(text, encoding="utf-8")
                rec["pages"] = d.page_count
                # Title verification: the claimed title must appear in the first page's text.
                first = norm(d[0].get_text())[:4000]
                want = norm(re.sub(r"[{}\\$]", "", title))
                rec["title_match"] = bool(want[:45] in first or
                                          norm(re.sub(r"\s", "", want))[:40] in norm(
                                              re.sub(r"\s", "", first)))
                if rec["status"] == "no route":
                    rec["status"] = "cached"
            except Exception as exc:
                rec["status"] = f"parse failed: {exc}"
        results.append(rec)
        flag = ("OK " if rec["title_match"] else
                "MISMATCH" if rec["title_match"] is False else "    ")
        print(f"  {flag:9s} {rec['status']:12s} {key[:30]:32s} {rec['pages']:3d}p")

    (LIB / "verification.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    got = [r for r in results if r["pages"]]
    mism = [r for r in results if r["title_match"] is False]
    print(f"\n{len(got)}/{len(results)} retrieved with text; "
          f"{sum(1 for r in got if r['title_match'])} title-verified, {len(mism)} mismatched")
    for r in mism:
        print(f"  MISMATCH {r['key']}: {r['title'][:70]}")
    missing = [r["key"] for r in results if not r["pages"]]
    if missing:
        print(f"\nnot retrieved ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    main()
