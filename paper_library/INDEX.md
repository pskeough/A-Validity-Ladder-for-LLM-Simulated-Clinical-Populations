# Literature library

40 references from `paper_v2/refs.bib`. **22 retrieved with full text, 20 title-verified, 0 genuine
mismatches.** Retrieved sources are not redistributed here for copyright reasons; per-entry retrieval status in `verification.json`;
claim checks in `../analysis/citation_claims.csv` and `citation_numeric_claims.csv`.

## What this library is for

Three questions it answers, in the order they matter for this paper:

1. **Does each citation point at the paper it claims?** Answered for all 40: 25 by DOI-registry
   resolution (`scripts/32_verify_references.py`), 20 additionally by page-1 title match here.
2. **Does each cited work support the sentence it is attached to?** Answered for 12 load-bearing
   claims, all supported. This is the check that catches the version-1 failure class, where a
   citation pointed at a real paper that did not say the thing.
3. **Do the numbers drawn from each source appear in that source?** Answered for 2 of 7. **Five
   sources carrying specific figures could not be retrieved, and this is the library's main gap.**

## The gap that matters

| source | figures the manuscript draws from it | retrievable |
|---|---|---|
| `lowe2008depression` | mean 3.6 for 1,759 patients (84%); intercorrelations 0.64–0.75; caseload reconstructed to 4.583–5.641; 75% comorbidity | **no — closed access** |
| `perlis2025clinical` | internet panel converting to 6.38, 26.4% at or above 10 | no |
| `patel2019phq9` | PHQ-9 descriptive table reproduced to within 0.02 | no |
| `brody2018depression` | 8.1% / 10.4% / 5.5% national prevalence | no |
| `xu2023phq9transwomen` | the withdrawn v1 constant: 198 transgender women, Shenyang | no |

`lowe2008depression` is the sharpest. It supplies the reconstructed caseload band that Section 4.2's
frame bound rests on, and the correlation band Section 4.1 uses. Both are closed-access figures
transcribed from the published table, which is the practice Section 12 argues against. The paper
now says so about the correlation band; it does not yet say so about the caseload reconstruction.

Two others, `patel2019phq9` and `brody2018depression`, are the pipeline's validation targets. Their
numbers are independently confirmed by the pipeline reproducing them, which is a stronger check than
reading the table, so the retrieval gap there is not a verification gap.

## Routing

**Positioning and prior work on LLM population simulation** — `argyle2023outofone` (not retrieved),
`bisbee2024synthetic`, `santurkar2023whose`, `cheng2023compost`, `cheng2023marked`.
`bisbee2024synthetic` is what Related Work positions the severity residuals against; the dispersion
comparison that used to sit there was withdrawn in this revision.

**The result that most constrains our claims** — `villarrealzegarra2026synthetic`. Fitted CFA to
synthetic PHQ data and reported good fit, which Section 9 tests directly and finds model-dependent.
Retrieved and verified.

**The alternative explanation that cost us a finding** — `dominguezolmedo2024questioning`. Ordering
and labeling artifacts in LLM survey responses. This was a caveat in Section 9.1 and is now
load-bearing in Section 4.1, where the cross-instrument endpoint test attributes the compressed
range to ordinal response format. Retrieved and verified.

**Benchmark validity, which is Section 12's argument** — `northcutt2021pervasive` (label errors
destabilize rankings), `raji2021everything`, `raji2020closing`, `jacobs2021measurement`,
`gebru2021datasheets`, `bender2018datastatements`, `paullada2021discontents`.

**Clinical anchors** — `beck2011severity` and `stevens2020depression` retrieved and their conditions
verbatim-verified; `lowe2008depression` and `perlis2025clinical` not retrievable.

**Federal measurement infrastructure** — `nchs2022srvydesc` retrieved (135pp) and verified to carry
both the PHQ-8 and the withheld gender-identity items, which is Section 10's central factual claim.
`nchs2024srvydesc`, `eo14168`, `cdc2015brfssmodules`, `census2022hps` not retrieved; all are public
documents whose relevant content is quoted in Section 10 from the agency text.

## Retrieval notes

The `paper-library` skill resolves against arXiv, which suits an ML preprint bibliography. This one
is mostly clinical journals and federal documentation, so that path reached 4 of 40. Two things went
wrong and both are recorded rather than worked around:

- **The skill's arXiv-ID regex matches DOI fragments.** `10.1001/jamainternmed.2024.2544` became the
  arXiv id `2024.2544`; `10.1016/j.ypmed.2022.106988` became `2022.10698`. All four phantoms 404'd,
  so they failed safely, but a DOI whose tail matched a real arXiv id would have downloaded an
  unrelated paper — the exact failure title-verification exists to catch.
- **PMC's `/pmc/articles/<id>/pdf/` route now returns HTML.** Two landing pages were saved as PDFs,
  parsed cleanly, and contained no article. `scripts/37_fetch_library.py` now detects and rejects
  them.

Also fixed here: a 1,200-character title-verification window flagged `stevens2020depression` as a
mismatch because journal boilerplate pushed its title to character 1,937. Widened to 4,000. The two
remaining flagged mismatches are a `ﬂ` ligature (`santurkar2023whose`) and a LaTeX `$\Psi$`
(`wang2024patientpsi`); both were read and are correct.
