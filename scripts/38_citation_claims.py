"""Does each cited work support the sentence it is attached to?

Title verification catches a citation pointing at the wrong paper. It cannot catch a citation
pointing at the right paper and claiming something the paper does not say, which is the failure this
manuscript's own Appendix A documents in version 1. That needs the full text.

For every load-bearing citation, this states the claim the manuscript makes and searches the
retrieved source for the phrase that would support it. A claim whose support cannot be located is
printed for manual reading, not silently passed.

Emits analysis/citation_claims.csv.
"""
import os
import re

import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
LIB = os.path.join(BASE, "paper_library", "txt")
OUT = os.path.join(BASE, "analysis")

# Each entry: the citation key, what the manuscript claims from it, and the strings that would have
# to appear in the source for the claim to hold. Written from the manuscript, not from the sources.
CLAIMS = [
    ("villarrealzegarra2026synthetic",
     "fitted confirmatory factor models to synthetic PHQ-9/PHQ-8/GAD-7 and reported good fit and "
     "high internal consistency",
     [r"confirmatory factor", r"internal consistency|Cronbach|omega", r"PHQ|GAD"]),
    ("dominguezolmedo2024questioning",
     "LLM survey responses carry ordering and labeling artifacts, and models trend toward uniform "
     "responding once answer order is randomized",
     [r"order", r"random", r"uniform"]),
    ("cheng2023marked",
     "documents strong racial stereotyping in persona-conditioned text",
     [r"stereotyp", r"persona|portrayal", r"racial|race"]),
    ("cheng2023compost",
     "characterizes caricature in LLM simulations and finds marginalized groups most susceptible",
     [r"caricature", r"marginaliz|identit"]),
    ("bisbee2024synthetic",
     "reports distortion in LLM-generated survey data, qualitatively",
     [r"synthetic|simulat", r"survey", r"variance|distribution|distort"]),
    ("wu2020equivalency",
     "PHQ-8 and PHQ-9 correlate at 0.996 to 0.998",
     [r"0\.99", r"PHQ-8", r"PHQ-9"]),
    ("argyle2023outofone",
     "conditions language models on demographic profiles to simulate human samples",
     [r"silicon sampl|simulat", r"demographic"]),
    ("santurkar2023whose",
     "measures whose opinions language models reflect across demographic groups",
     [r"opinion", r"demographic"]),
    ("northcutt2021pervasive",
     "label errors in benchmark test sets destabilize published rankings",
     [r"label error", r"test set", r"benchmark"]),
    ("raji2021everything",
     "benchmarks function as stand-ins for general capability and that framing has limits",
     [r"benchmark", r"general"]),
    ("nchs2022srvydesc",
     "NHIS 2022 administered the full PHQ-8 to all sample adults and fielded gender-identity items "
     "not released on the public-use file",
     [r"PHQ", r"gender identity|Gender Identity", r"Research Data Center|RDC|not.{0,40}public"]),
    ("obermeyer2019dissecting",
     "an algorithm used to manage population health showed racial bias",
     [r"racial bias|race", r"algorithm"]),
    ("omiye2023racebased",
     "large language models propagate race-based medicine",
     [r"race-based|race based", r"language model"]),
    ("pfohl2024toolbox",
     "a toolbox for surfacing health equity harms in LLMs",
     [r"equity", r"harm"]),
]

rows = []
for key, claim, needles in CLAIMS:
    path = os.path.join(LIB, f"{key}.txt")
    if not os.path.exists(path):
        rows.append(dict(key=key, claim=claim, status="source not retrieved", hits="", evidence=""))
        print(f"  {'NO SOURCE':13s} {key}")
        continue
    t = re.sub(r"\s+", " ", open(path, encoding="utf-8", errors="replace").read())
    hits, ev = [], ""
    for n in needles:
        m = re.search(n, t, re.I)
        hits.append(bool(m))
        if m and not ev:
            ev = t[max(0, m.start() - 130):m.start() + 190].strip()
    ok = all(hits)
    rows.append(dict(key=key, claim=claim,
                     status="supported" if ok else "PARTIAL, read manually",
                     hits=f"{sum(hits)}/{len(hits)}", evidence=ev[:300]))
    print(f"  {'supported' if ok else 'PARTIAL':13s} {key:32s} {sum(hits)}/{len(hits)}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "citation_claims.csv"), index=False)
n_ok = sum(1 for r in rows if r["status"] == "supported")
n_src = sum(1 for r in rows if r["status"] != "source not retrieved")
print(f"\n{n_ok}/{n_src} claims supported by their retrieved source; "
      f"{len(rows) - n_src} sources not retrieved")


# ---- numeric claims, which is where version 1 actually failed --------------------------------
# A citation can point at the right paper and still carry a number that paper does not report.
# Appendix A documents exactly that failure. These are the specific figures the manuscript draws
# from each clinical anchor, checked against the retrieved text.
NUMERIC = [
    ("beck2011severity", "12.2 at antidepressant initiation among 771 patients scoring 7 or higher",
     [r"12\.2", r"771", r"\b7\b"]),
    ("stevens2020depression", "VA cohort with roughly equal numbers with and without HIV, "
     "person-visits 2003 to 2015",
     [r"HIV", r"2003", r"2015"]),
    ("patel2019phq9", "PHQ-9 descriptive table this pipeline reproduces to within 0.02",
     [r"PHQ-9", r"invariance"]),
    ("brody2018depression", "national prevalence 8.1% overall, 10.4% women, 5.5% men",
     [r"8\.1", r"10\.4", r"5\.5"]),
    ("xu2023phq9transwomen", "PHQ-9 validation, 198 transgender women, Shenyang China",
     [r"198", r"Shenyang", r"transgender women"]),
    ("lowe2008depression", "3.6 for 1,759 patients (84%) scoring below 15 on all three scales; "
     "intercorrelations 0.64 to 0.75; caseload reconstructed as 4.583 to 5.641",
     [r"3\.6", r"1,?759", r"0\.6[0-9]"]),
    ("perlis2025clinical", "internet panel converting to 6.38 with 26.4% at or above 10",
     [r"26\.4", r"PHQ"]),
]
print("\nnumeric claims:")
nrows = []
for key, claim, needles in NUMERIC:
    path = os.path.join(LIB, f"{key}.txt")
    if not os.path.exists(path):
        nrows.append(dict(key=key, claim=claim, status="SOURCE NOT RETRIEVED", hits="", evidence=""))
        print(f"  {'NO SOURCE':13s} {key:30s} <- numbers unverifiable from the library")
        continue
    t = re.sub(r"\s+", " ", open(path, encoding="utf-8", errors="replace").read())
    hits, ev = [], ""
    for n in needles:
        m = re.search(n, t)
        hits.append(bool(m))
        if m and not ev:
            ev = t[max(0, m.start() - 150):m.start() + 200].strip()
    ok = all(hits)
    nrows.append(dict(key=key, claim=claim, status="figures present" if ok else "PARTIAL",
                      hits=f"{sum(hits)}/{len(hits)}", evidence=ev[:320]))
    print(f"  {'figures present' if ok else 'PARTIAL':16s} {key:30s} {sum(hits)}/{len(hits)}")
    if ok and ev:
        print(f"      {ev[:190]}")
pd.DataFrame(nrows).to_csv(os.path.join(OUT, "citation_numeric_claims.csv"), index=False)
