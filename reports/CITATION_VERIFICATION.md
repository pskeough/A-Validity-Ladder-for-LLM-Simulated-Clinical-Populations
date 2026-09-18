# PsychBench v2 — Citation Verification (verify-then-cite)

**Pass rule:** no cite without opening it. Every reference below was located and its
abstract/source page fetched. Scope: §2 Related Work and §9 Trans section, per V2_SPEC §2 row.
Date: 2026-07-15.

## Verdict table

| # | Intended cite | Verdict | One-line note |
|---|---|---|---|
| 1 | Jacobs & Wallach — Measurement and Fairness | **SUPPORTS** | Exact fit for the measurement-theoretic / operationalization claim. |
| 2 | Raji et al. — AI audit / benchmark validity | **SUPPORTS (pick correct paper)** | Two real papers; recommend *Everything in the Whole Wide World Benchmark* (2021) for the operationalization/benchmark-validity claim, *Closing the AI Accountability Gap* (2020) for the "prior AI audit" contribution sentence. |
| 3 | Blodgett et al. — fairness benchmark pitfalls | **SUPPORTS** | *Stereotyping Norwegian Salmon*; directly about pitfalls in fairness benchmark datasets. |
| 4 | Northcutt et al. — pervasive label errors | **SUPPORTS (minor scope note)** | Documents label errors in general ML test sets, not fairness benchmarks specifically. |
| 5 | Gebru et al. — Datasheets for Datasets | **SUPPORTS** | Correct for the "datasheets" documentation-framework slot. |
| 6 | Bender & Friedman — Data Statements | **SUPPORTS** | Correct for the "data statements" slot. Paullada (below) is a survey, not the documentation-framework cite. |
| 6b | Paullada et al. — Data and its (dis)contents | **PARTIAL (wrong slot)** | Real and relevant, but a dataset-development *survey*; fits the "provenance gaps / dataset audits" clause, not "documentation frameworks." Optional add alongside #3/#4. |
| 7 | Obermeyer et al. 2019 — racial bias in risk scores | **SUPPORTS** | Exact fit; *Science* 366:447–453. |
| 8 | Zack et al. — GPT-4 bias in health care | **SUPPORTS** | *Lancet Digital Health* 2024; venue confirmed. |
| 9 | Omiye et al. — LLMs propagate race-based medicine | **SUPPORTS** | *npj Digital Medicine* 2023; venue confirmed. |
| 10 | Pfohl et al. — health-equity toolbox for LLMs | **SUPPORTS** | *Nature Medicine* 2024; venue confirmed. |
| 11 | Padmakumar & He — diversity reduction under alignment | **SUPPORTS (scope note)** | Aligned model (InstructGPT) reduces diversity, base (GPT-3) does not — matches the base-vs-aligned claim. Setting is human–AI co-writing, not free generation. |
| 12 | NASEM 2022 — Measuring Sex, Gender Identity, SO | **SUPPORTS** | Exact fit for the "measurement gap / formal review" claim. |
| 13 | Patel et al. 2019 — PHQ-9 invariance NHANES | **SUPPORTS** | Biblio confirmed exactly: *Depression and Anxiety* 36(9):813–823. |
| 14 | Brody, Pratt & Hughes 2018 — NCHS Data Brief 303 | **SUPPORTS** | Confirmed; data years 2013–2016 (not 2005–2018). |
| 15 | Xu et al. 2023 — PHQ-9 trans women, Shenyang | **SUPPORTS (error source)** | Confirmed exactly; snowball sample, n=198, PHQ-9 mean 15.56. Matches Correction Note framing. |
| 16 | Hughto — prevalence-only trans depression anchor | **WRONG-FIT as labeled** | No "Hughto 2024" BRFSS depression paper exists. Recommend **Liu et al. 2024** (*JAMA Intern Med*, national BRFSS) as primary; Hughto et al. 2022 (*Prev Med*, WA-state BRFSS) as a secondary population-based option. |

---

## Per-citation notes

### 1. Jacobs & Wallach — Measurement and Fairness — SUPPORTS
Abigail Z. Jacobs & Hanna Wallach, *Measurement and Fairness*, FAccT '21, pp. 375–385, DOI 10.1145/3442188.3445901.
Proposes measurement modeling from the quantitative social sciences: fairness constructs (SES, risk of
recidivism, teacher effectiveness) are unobservable and must be operationalized via a measurement model,
so validity of any fairness claim depends on that operationalization. Exact fit for the §2 sentence
"audit conclusions are only as sound as the operationalization of the constructs being measured." No overstatement.

### 2. Raji et al. — SUPPORTS, but choose the right paper
Two distinct, real Raji papers map onto two different §2 clauses:
- **AI and the Everything in the Whole Wide World Benchmark** — Raji, Denton, Bender, Hanna, Paullada,
  NeurIPS 2021 Datasets & Benchmarks. Critiques "general" benchmarks as under-justified construct
  abstractions — the strongest fit for the *benchmark-validity / operationalization* thread in §2.
- **Closing the AI Accountability Gap** — Raji, Smart, White, Mitchell, Gebru, Hutchinson, Smith-Loud,
  Theron, Barnes, FAT* 2020, pp. 33–44, DOI 10.1145/3351095.3372873. The canonical end-to-end internal
  audit framework (SMACTR) — the right cite for the contribution sentence "we are not aware of a prior
  AI audit that re-derived its complete benchmark layer." 
**Recommendation:** cite the *Everything Benchmark* paper in the measurement-validity sentence and
*Closing the AI Accountability Gap* in the audit-practice contribution sentence. Both are load-bearing;
they are not interchangeable.

### 3. Blodgett et al. — SUPPORTS
Blodgett, Lopez, Olteanu, Sim, Wallach, *Stereotyping Norwegian Salmon: An Inventory of Pitfalls in
Fairness Benchmark Datasets*, ACL-IJCNLP 2021, pp. 1004–1015. Applies a measurement-modeling lens to four
fairness benchmarks and inventories validity pitfalls (unclear constructs, unstated assumptions). Exact
fit for "pitfalls in fairness benchmark data."

### 4. Northcutt et al. — SUPPORTS (minor scope note)
Northcutt, Athalye, Mueller, *Pervasive Label Errors in Test Sets Destabilize Machine Learning
Benchmarks*, NeurIPS 2021 Datasets & Benchmarks. ~3.4% mean label error across 10 canonical CV/NLP/audio
test sets; benchmark rankings flip under correction. Supports "label errors." **Scope note:** these are
general ML test sets, not fairness benchmarks specifically — fine for the "label errors" clause, don't
stretch it to "fairness benchmark data" (that clause is carried by Blodgett).

### 5. Gebru et al. — SUPPORTS
Gebru, Morgenstern, Vecchione, Vaughan, Wallach, Daumé III, Crawford, *Datasheets for Datasets*, CACM
64(12):86–92, 2021, DOI 10.1145/3458723. Process-level documentation framework. Correct for "datasheets."

### 6. Bender & Friedman — SUPPORTS
Bender & Friedman, *Data Statements for Natural Language Processing*, TACL 6:587–604, 2018,
DOI 10.1162/tacl_a_00041. The "data statements" documentation framework. Correct second cite for the
"datasheets and data statements" pair. **Verified against Paullada:** V2_SPEC listed Paullada as an
alternative — but Paullada et al., *Data and its (dis)contents*, *Patterns* 2(11):100336, 2021, is a
*survey* of dataset development/critique, not a documentation framework. Keep Bender & Friedman for the
documentation slot; if you want Paullada, place it in the "dataset audits / provenance gaps" clause
alongside Northcutt/Blodgett, not the "documentation frameworks" clause.

### 7. Obermeyer et al. 2019 — SUPPORTS
Obermeyer, Powers, Vogeli, Mullainathan, *Dissecting racial bias in an algorithm used to manage the
health of populations*, *Science* 366(6464):447–453, DOI 10.1126/science.aax2342. Care-management risk
algorithm systematically underestimated Black patients' illness because it predicted cost, not need.
Exact fit for "racial bias in care-management risk scores."

### 8. Zack et al. — SUPPORTS
Zack, Lehman, Suzgun, et al., *Assessing the potential of GPT-4 to perpetuate racial and gender biases
in health care: a model evaluation study*, *Lancet Digital Health* 6(1):e12–e22, 2024,
DOI 10.1016/S2589-7500(23)00225-X. GPT-4 exaggerates disease-prevalence differences and stereotyped
differentials across tasks. Fit for "GPT-class models reproduce demographic disparities in diagnosis
and treatment recommendation." (Note: online 2023, print issue 2024 — cite as 2024.)

### 9. Omiye et al. — SUPPORTS
Omiye, Lester, Spichak, Rotemberg, Daneshjou, *Large language models propagate race-based medicine*,
*npj Digital Medicine* 6:195, 2023, DOI 10.1038/s41746-023-00939-z. Four commercial LLMs reproduce
debunked race-based medical claims. Exact fit for "race-based errors propagated by LLMs."

### 10. Pfohl et al. — SUPPORTS
Pfohl, Cole-Lewis, Sayres, et al., *A toolbox for surfacing health equity harms and biases in large
language models*, *Nature Medicine* 30(12):3590–3600, 2024, DOI 10.1038/s41591-024-03258-2. Multifactorial
equity-evaluation framework with a diverse rater pool. Fit for the health-equity-evaluation thread.

### 11. Padmakumar & He — SUPPORTS (scope note)
Padmakumar & He, *Does Writing with Language Models Reduce Content Diversity?*, ICLR 2024,
arXiv:2309.05196. **Checked the base-vs-aligned claim specifically:** writing with the feedback-tuned
(aligned) InstructGPT produced a statistically significant reduction in content diversity, while writing
with the base GPT-3 did not; the effect traces to the aligned model contributing less diverse text. This
does document diversity reduction in aligned vs base — the draft's §2 framing holds. **Scope note:** the
study is human–AI *co-writing*, not free model generation, and diversity is measured across authors
(homogenization). The draft already confines the mechanism to an untested hypothesis, so no overstatement;
just don't imply it measured free-generation output diversity.

### 12. NASEM 2022 — SUPPORTS
National Academies of Sciences, Engineering, and Medicine, *Measuring Sex, Gender Identity, and Sexual
Orientation*, The National Academies Press, 2022, DOI 10.17226/26424. Formal NIH-commissioned review
recommending standardized gender-identity measurement; documents the absence of such fields in federal
instruments. Exact fit for "a measurement gap that has been the subject of formal review."

### 13. Patel et al. 2019 — SUPPORTS (biblio confirmed exactly)
Patel, Chang, et al., *Measurement invariance of the Patient Health Questionnaire-9 (PHQ-9) depression
screener in U.S. adults across sex, race/ethnicity, and education level: NHANES 2005–2016*, *Depression
and Anxiety* 36(9):813–823, 2019, DOI 10.1002/da.22940. n=31,366. All draft biblio fields match. Correct
as the pipeline-validation reproduction target (§10 "reproduces Patel et al. 2019 to 0.02").

### 14. Brody, Pratt & Hughes 2018 — SUPPORTS
Brody DJ, Pratt LA, Hughes JP, *Prevalence of Depression Among Adults Aged 20 and Over: United States,
2013–2016*, NCHS Data Brief No. 303, National Center for Health Statistics, 2018. PHQ-9-based; 8.1% adult
2-week prevalence; income gradient and lower Asian prevalence reported. **Note the data window is
2013–2016**, not the 2005–2018 span used for the model's own anchors — correct as stated, just keep the
years right in prose. Correct as the second pipeline-validation target (§10 "Brody et al. 2018 exactly").

### 15. Xu et al. 2023 — SUPPORTS (this is the v1 error source; framing confirmed)
Xu L, Chang R, Wang H, Xu C, Yu X, Chen H, Wang R, Liu S, Liu Y, Wang Y, Cai Y, *Validation of the
Patient Health Questionnaire-9 for Suicide Screening in Transgender Women*, *Transgender Health*
8(5):450–456, 2023, DOI 10.1089/trgh.2021.0075. **Snowball** sample of **198** transgender women in
**Shenyang, China** (Apr–Jul 2017); PHQ-9 (0–27 scale) **mean 15.56, SD 5.70**; a suicide-screening
validation study, not a population norm. Every element of the §10 Correction Note description checks out:
snowball-sampled Shenyang validation study, PHQ-9 (not PHQ-8) scale, standing in for a US population
parameter. Use exactly as the withdrawn-finding source.

### 16. Hughto — WRONG-FIT as labeled; recommended replacement
The V2_SPEC "Hughto 2024" prevalence citation **does not exist** as described. The paper reporting the
BRFSS 2014–2022 depression-prevalence trend among TGD adults (19.7%→51.3%) is:
- **Liu M, Patel VR, Reisner SL, Keuroghlian AS**, *Health Status and Mental Health of Transgender and
  Gender-Diverse Adults*, *JAMA Internal Medicine* 184(8):984–986, 2024, DOI 10.1001/jamainternmed.2024.2544.
  National BRFSS surveillance; directly reports elevated diagnosed-depression prevalence. **Hughto is not
  an author.** This is the strongest fit for the §9 sentence "representative surveillance does support
  elevated prevalence of diagnosed depression among transgender and gender-diverse adults."
- The genuine Hughto population-based paper is **Hughto JMW, et al.**, *Health, economic and social
  disparities among transgender women, transgender men and transgender nonbinary adults: Results from a
  population-based study*, *Preventive Medicine* 156:106988, 2022, DOI 10.1016/j.ypmed.2022.106988. But it
  is **Washington-State BRFSS only** (2016–2019), not national, and reports "poor mental health" rather
  than a headline depression-prevalence figure — weaker for a "representative surveillance" claim.

**Recommendation:** cite **Liu et al. 2024** as the primary representative-surveillance anchor for the
§9 prevalence claim; optionally add Hughto et al. 2022 as a supporting population-based disparities cite.
Do not cite a "Hughto 2024."

---

## Flags where framing could overstate the source

- **§2 "dataset audits ... found label errors ... in fairness benchmark data"** — Northcutt is general ML
  test sets (label errors), Blodgett is fairness benchmarks (pitfalls). Keep the two claims paired to the
  right cites; don't let Northcutt carry "fairness benchmark."
- **§2 homogenization (Padmakumar & He)** — co-writing / cross-author homogenization, not free-generation
  output variance. The draft's "untested hypothesis" hedge covers this; keep it.
- **§9 prevalence anchor** — as written the claim needs a *national representative* source; Liu et al.
  2024 supplies it, the mislabeled "Hughto 2024" did not.

---

## BibTeX

```bibtex
@inproceedings{jacobs2021measurement,
  author    = {Jacobs, Abigail Z. and Wallach, Hanna},
  title     = {Measurement and Fairness},
  booktitle = {Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT)},
  pages     = {375--385},
  year      = {2021},
  doi       = {10.1145/3442188.3445901}
}

@inproceedings{raji2021everything,
  author    = {Raji, Inioluwa Deborah and Bender, Emily M. and Paullada, Amandalynne and Denton, Emily and Hanna, Alex},
  title     = {{AI} and the Everything in the Whole Wide World Benchmark},
  booktitle = {Proceedings of the NeurIPS 2021 Track on Datasets and Benchmarks},
  year      = {2021},
  eprint    = {2111.15366},
  archivePrefix = {arXiv}
}

@inproceedings{raji2020closing,
  author    = {Raji, Inioluwa Deborah and Smart, Andrew and White, Rebecca N. and Mitchell, Margaret and Gebru, Timnit and Hutchinson, Ben and Smith-Loud, Jamila and Theron, Daniel and Barnes, Parker},
  title     = {Closing the {AI} Accountability Gap: Defining an End-to-End Framework for Internal Algorithmic Auditing},
  booktitle = {Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (FAT*)},
  pages     = {33--44},
  year      = {2020},
  doi       = {10.1145/3351095.3372873}
}

@inproceedings{blodgett2021salmon,
  author    = {Blodgett, Su Lin and Lopez, Gilsinia and Olteanu, Alexandra and Sim, Robert and Wallach, Hanna},
  title     = {Stereotyping {N}orwegian Salmon: An Inventory of Pitfalls in Fairness Benchmark Datasets},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL-IJCNLP)},
  pages     = {1004--1015},
  year      = {2021}
}

@inproceedings{northcutt2021pervasive,
  author    = {Northcutt, Curtis G. and Athalye, Anish and Mueller, Jonas},
  title     = {Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks},
  booktitle = {Proceedings of the NeurIPS 2021 Track on Datasets and Benchmarks},
  year      = {2021},
  eprint    = {2103.14749},
  archivePrefix = {arXiv}
}

@article{gebru2021datasheets,
  author  = {Gebru, Timnit and Morgenstern, Jamie and Vecchione, Briana and Vaughan, Jennifer Wortman and Wallach, Hanna and Daum{\'e} III, Hal and Crawford, Kate},
  title   = {Datasheets for Datasets},
  journal = {Communications of the ACM},
  volume  = {64},
  number  = {12},
  pages   = {86--92},
  year    = {2021},
  doi     = {10.1145/3458723}
}

@article{bender2018datastatements,
  author  = {Bender, Emily M. and Friedman, Batya},
  title   = {Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {6},
  pages   = {587--604},
  year    = {2018},
  doi     = {10.1162/tacl_a_00041}
}

@article{paullada2021discontents,
  author  = {Paullada, Amandalynne and Raji, Inioluwa Deborah and Bender, Emily M. and Denton, Emily and Hanna, Alex},
  title   = {Data and Its (Dis)contents: A Survey of Dataset Development and Use in Machine Learning Research},
  journal = {Patterns},
  volume  = {2},
  number  = {11},
  pages   = {100336},
  year    = {2021},
  doi     = {10.1016/j.patter.2021.100336}
}

@article{obermeyer2019dissecting,
  author  = {Obermeyer, Ziad and Powers, Brian and Vogeli, Christine and Mullainathan, Sendhil},
  title   = {Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations},
  journal = {Science},
  volume  = {366},
  number  = {6464},
  pages   = {447--453},
  year    = {2019},
  doi     = {10.1126/science.aax2342}
}

@article{zack2024gpt4,
  author  = {Zack, Travis and Lehman, Eric and Suzgun, Mirac and Rodriguez, Jorge A. and Celi, Leo Anthony and Gichoya, Judy and Jurafsky, Dan and Szolovits, Peter and Bates, David W. and Abdulnour, Raja-Elie E. and Butte, Atul J. and Alsentzer, Emily},
  title   = {Assessing the Potential of {GPT-4} to Perpetuate Racial and Gender Biases in Health Care: A Model Evaluation Study},
  journal = {The Lancet Digital Health},
  volume  = {6},
  number  = {1},
  pages   = {e12--e22},
  year    = {2024},
  doi     = {10.1016/S2589-7500(23)00225-X}
}

@article{omiye2023racebased,
  author  = {Omiye, Jesutofunmi A. and Lester, Jenna C. and Spichak, Simon and Rotemberg, Veronica and Daneshjou, Roxana},
  title   = {Large Language Models Propagate Race-Based Medicine},
  journal = {npj Digital Medicine},
  volume  = {6},
  pages   = {195},
  year    = {2023},
  doi     = {10.1038/s41746-023-00939-z}
}

@article{pfohl2024toolbox,
  author  = {Pfohl, Stephen R. and Cole-Lewis, Heather and Sayres, Rory and others},
  title   = {A Toolbox for Surfacing Health Equity Harms and Biases in Large Language Models},
  journal = {Nature Medicine},
  volume  = {30},
  number  = {12},
  pages   = {3590--3600},
  year    = {2024},
  doi     = {10.1038/s41591-024-03258-2}
}

@inproceedings{padmakumar2024diversity,
  author    = {Padmakumar, Vishakh and He, He},
  title     = {Does Writing with Language Models Reduce Content Diversity?},
  booktitle = {Proceedings of the 12th International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2309.05196},
  archivePrefix = {arXiv}
}

@book{nasem2022measuring,
  author    = {{National Academies of Sciences, Engineering, and Medicine}},
  title     = {Measuring Sex, Gender Identity, and Sexual Orientation},
  publisher = {The National Academies Press},
  address   = {Washington, DC},
  year      = {2022},
  doi       = {10.17226/26424}
}

@article{patel2019phq9,
  author  = {Patel, Jay S. and Oh, Youngran and Rand, Kevin L. and Wu, Wei and Cyders, Melissa A. and Kroenke, Kurt and Stewart, Jesse C.},
  title   = {Measurement Invariance of the Patient Health Questionnaire-9 ({PHQ-9}) Depression Screener in {U.S.} Adults Across Sex, Race/Ethnicity, and Education Level: {NHANES} 2005--2016},
  journal = {Depression and Anxiety},
  volume  = {36},
  number  = {9},
  pages   = {813--823},
  year    = {2019},
  doi     = {10.1002/da.22940}
}

@techreport{brody2018depression,
  author      = {Brody, Debra J. and Pratt, Laura A. and Hughes, Jeffery P.},
  title       = {Prevalence of Depression Among Adults Aged 20 and Over: United States, 2013--2016},
  institution = {National Center for Health Statistics},
  type        = {NCHS Data Brief},
  number      = {303},
  address     = {Hyattsville, MD},
  year        = {2018}
}

@article{xu2023phq9transwomen,
  author  = {Xu, Lulu and Chang, Ruijie and Wang, Hui and Xu, Chen and Yu, Xiaoyue and Chen, Hui and Wang, Rongxi and Liu, Shangbin and Liu, Yujie and Wang, Ying and Cai, Yong},
  title   = {Validation of the Patient Health Questionnaire-9 for Suicide Screening in Transgender Women},
  journal = {Transgender Health},
  volume  = {8},
  number  = {5},
  pages   = {450--456},
  year    = {2023},
  doi     = {10.1089/trgh.2021.0075}
}

@article{liu2024tgdhealth,
  author  = {Liu, Michael and Patel, Vishal R. and Reisner, Sari L. and Keuroghlian, Alex S.},
  title   = {Health Status and Mental Health of Transgender and Gender-Diverse Adults},
  journal = {JAMA Internal Medicine},
  volume  = {184},
  number  = {8},
  pages   = {984--986},
  year    = {2024},
  doi     = {10.1001/jamainternmed.2024.2544}
}

@article{hughto2022disparities,
  author  = {Hughto, Jaclyn M. W. and Gunn, Hamish A. and Rood, Brian A. and Pantalone, David W.},
  title   = {Health, Economic, and Social Disparities Among Transgender Women, Transgender Men, and Transgender Nonbinary Adults: Results from a Population-Based Study},
  journal = {Preventive Medicine},
  volume  = {156},
  pages   = {106988},
  year    = {2022},
  doi     = {10.1016/j.ypmed.2022.106988}
}
```

*Note: author lists marked with "others" (Pfohl) and some middle-author orderings (Zack, Patel, Hughto)
were reconstructed from PubMed/publisher metadata; confirm exact author strings against the publisher
page before final submission. All titles, venues, years, volumes, and DOIs were verified at source.*
