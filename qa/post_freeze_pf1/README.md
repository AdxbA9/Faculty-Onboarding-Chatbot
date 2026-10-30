# Official guide inventory, knowledge-gap mapping and ingestion readiness (post-freeze phase PF-1)

The controlled preparation phase before any new document enters the production knowledge base. The frozen runtime
(`runtime-freeze-i9`, commit `44ada15`, merged to `main`) and its metrics are unchanged: 192 questions, 55 served, 137
fallback, 28.6% coverage, 55 correct, 0 limited, 0 wrong, 100% precision, 0 false acceptances, 11 false fallbacks, 6,674
tests. PF-1 inventories the candidate guides, measures what can be measured, classifies what cannot, and writes the PF-2
ingestion plan. Nothing was ingested.

| File | Purpose |
|---|---|
| `inventory_scan.py` | measures every PDF and DOCX in the source folder: size, SHA-256, page count, stored properties, per-page text volume and images, text-sparse and image-dominated pages, Arabic-script share, replacement characters, URLs, e-mails, phones, numbered steps, menu paths, a text sample; PDF text through PyMuPDF as production reads it, DOCX text through the standard library; no OCR |
| `classification.json` | the manual audit of every file: canonical title, issuing unit, system, topics, audience, language, date evidence, officiality, relevance, extraction class, screenshot dependence, duplicate or version status, specialist mapping, handbook overlap, status, priority, notes |
| `build_inventory.py` | merges the scan and the classification into the master inventory and prints the counts |
| `document_inventory.json`, `document_inventory.csv` | the master document inventory (25 documents) with the count summary |
| `gap_mapping.json` | the I-8 false-fallback mapping, other recorded fallbacks the new data addresses, taxonomy, document-to-specialist matrix, handbook overlap, high-value knowledge, conflicts, precedence proposal, metadata requirements, pipeline and chunking readiness, retrieval risks, cross-document cases, citation design, storage and security findings |
| `pf2_question_draft.json` | 34 draft evaluation questions for PF-2 with the document and page each rests on (not run in PF-1) |
| `pilot_plan.md` | the PF-2 pilot selection and the smallest set of changes PF-2 needs |

Reproduce (the source folder is given on the command line and recorded by base name only):

    python -X utf8 qa/post_freeze_pf1/inventory_scan.py <source folder> <scan.json>
    python -X utf8 qa/post_freeze_pf1/build_inventory.py <scan.json> qa/post_freeze_pf1/classification.json qa/post_freeze_pf1

## Where the documents are

The approved source set is the `Tutorials` folder of the Senior Project (milestone 4) working directory, one level above
the repository's parent folder, next to a `Tutorials.rar` archive (26.1 MB). The folder holds 25 files: 21 PDF and 4
DOCX. The archive is a container that no extractor on this machine can open; by size it is consistent with a compressed
copy of the same set and it was not inventoried. No other UOS guide exists in the repository, in the handover folder or
in the other project copies (those contain only further copies of the Faculty Handbook).

## Counts

| Measure | Value |
|---|---|
| Candidate documents found | 25 (21 PDF, 4 DOCX; plus 1 RAR container not inventoried) |
| Approved official and relevant candidates | 12 |
| Held for review | 9 |
| Rejected, duplicate or irrelevant | 4 (1 duplicate, 3 vendor or student material) |
| PDF pages, all candidates | 225 (measured) |
| DOCX pages, all candidates | 7 declared by the files' own properties (word-based estimate 8) |
| Pages of the approved set | 91 (88 PDF pages measured, 3 DOCX pages declared) |
| Faculty Handbook pages | 272 (verified against the production cache) |
| Potential corpus if every approved document is ingested | 363 pages |
| Potential corpus if every candidate were ingested | 504 pages |

File types: PDF 21 (supported by the loader), DOCX 4 (not supported by the loader; text extractable with the standard
library), PPT/PPTX 0, XLS/XLSX 0, other 1 (RAR). No exact duplicates by hash. One content duplicate: the Performance
Appraisal Phase 2 guide exists as DOCX and PDF (token overlap 0.99); the PDF is canonical. Two student-and-instructor
pairs cover the same tool from each side (Collaborate; Blackboard login). No document carries a version history; dates
come from the file properties or from a stated edition ("Version 1.0 September 2026", "Edition 1 September 2026",
"Version 2.5 August 2021"); where neither exists the inventory says DATE UNKNOWN.

Languages: 17 English, 6 bilingual (Grade Transfer, grade categories, iThenticate, Blackboard student brief, Collaborate
student guide, Connect exam support), 2 Arabic only (both Turnitin guides; one is misnamed "EN"). Arabic text extracts
with reversed segments and broken ligatures, and the production embedder and reranker are English models, so Arabic
pages cannot be processed safely by the current pipeline; bilingual guides are usable through their English pages only
(the grade-category guide needs an in-page split). This is a readiness observation; no Arabic support was built.

Extraction: GOOD 8, GOOD WITH MINOR CLEANUP 7, NEEDS STRUCTURED CLEANUP 6 (the bilingual pages), OCR or alternative
extraction needed in part 2 (McGraw Hill Connect sync guide, Coursera learner guide: screenshot-dominated pages), BLOCKED
for the current English pipeline 2 (the Arabic guides). The DOCX files extract cleanly but need conversion or a reader
before the loader can take them. Screenshot-heavy pages: the UOS Blackboard guides write their steps out, so the
surrounding text is sufficient; the Akindi guide has one image-only step; the MyUOS guide's image-upload steps and the
vendor guides depend on screenshots; no OCR was run.

Officiality: the 12 approved documents are ITC Academic Computing or HR material with UOS marks, UOS URLs or the service
desk contact (ten PDFs) plus two generated DOCX guides that name the University (Microsoft Forms) or are the DOCX twin of
a UOS-headed PDF. Held: three unattributed files (Akindi, Business Mission, DocuSign note), one third-party compilation
("compiled from Anthology release notes"), two vendor documents, one Arabic instructor guide, two student guides.

## Specialist mapping

Primary specialist over the 25 documents: Teaching & Learning 16, Institutional Navigation 5, Faculty Services & HR 3,
Research & Innovation 1. Multi-domain: the Blackboard faculty short guide and the Grade Transfer guide (Teaching with
Institutional for access, dashboards and Banner), the Business Mission guide (HR with Research for conference missions),
the Turnitin instructor guide (Teaching with Research for thesis courses), the student Blackboard brief. No fifth
specialist is proposed; the matrix with routing terminology is in `gap_mapping.json`.

## Handbook overlap and conflicts

Nothing in the approved set is already covered by the handbook at procedure level: eight documents are new step-by-step
workflows, three new procedures, six new systems; new contact information is the ITC service desk e-mail and the help-desk
extension and the e-learning URL. Conflicts: C1 the handbook's page-224 sentence that Blackboard forwards grades to Banner
automatically against the 2024 Grade Transfer Dashboard with Chair and Dean approval (LIKELY UPDATED INFORMATION, serve as
policy plus procedure, never merged); C2 the iThenticate quota, 30 documents per year on the English page and 15 on the
Arabic page of the same guide (UNRESOLVED CONFLICT, raise with ITC, exclude or mark the sentence); C4 the website menu path
to MyUOS (2022 "Quick Links" versus 2024 "Helpful resources"; LIKELY UPDATED); C5 the 2026 vendor compilation's layout
names (DIFFERENT CONTEXT, held); C3, C6, C7, C8 no conflict. Precedence proposal: policy source for entitlements, limits,
approvals and obligations; newest official system manual for interface steps, URLs and system contacts; vendor material
never answers a UOS policy or procedure; official-versus-official conflicts are reported, not merged.

## Readiness

Ingestion pipeline: READY WITH STRUCTURAL CHANGES. The registry already holds several records, the chunk metadata is
source-aware and the retrieval scope is source-aware, but the loader reads the first PDF in `data/` only, the cache is
keyed by that one file, the section map applies to one source, DOCX is unreadable, there is no page-language filter,
the interface shows one document chip, and the four frozen specialists compile their retrieval scope with the handbook's
source id and page ranges, so no other source can reach a specialist until a scope-extension seam exists in non-frozen
code or a decision record reopens the specialists. The legacy path searches every indexed chunk, so new documents reach
the legacy answer as soon as they are indexed. Chunking: paragraph windows, rows and row windows fit narrative, tables
and contact lines; numbered procedures need a `procedure_step` and a `procedure_window` chunk type and bilingual pages a
language filter; no FAQ-pair or navigation-path type is needed for the approved set. Metadata additions and the
citation rule (`canonical title · Page n`, one chip per source, source ids in Developer mode only) are specified in
`gap_mapping.json`.

## I-8 false fallbacks

None of the 11 I-8 false fallbacks is a data gap the new guides fill: three are verifier entity binding on WLAM profile
sentences, two are routing ambiguity, three are specialist partial support on handbook facts, one is a verifier numeral
reading, one is the frozen Institutional directory lexicon, and all concern handbook content. The new documents address
other recorded fallbacks instead: the grade entry and transfer procedure, Blackboard content upload steps, the attendance
tool, the ITC service-desk contact, and a procedure area the handbook never had (performance appraisal workflow). No
coverage improvement is claimed; PF-2 measures it.

## Priority and pilot

P0 (5): Blackboard faculty short guide, Grade Transfer Dashboard guide, Blackboard gradebook guide, Performance Appraisal
Phase 2 guide (PDF), iThenticate guidelines. P1 (5): assignments and tests, communication tools, Collaborate instructor
guide, attendance tool, grade categories. P2 (2): MyUOS announcement guide, Microsoft Forms guide. HOLD (13): the nine
held documents and the four rejected ones. The PF-2 pilot is the five P0 documents (43 English pages; see
`pilot_plan.md`), chosen for official provenance, clean extraction, new knowledge the handbook lacks, and coverage of
all four specialists; the pilot also exercises the bilingual page filter, the precedence rule on C1 and the handling of
the unresolved C2.

## Storage, security, tests

Storage: MIXED, commit the approved official guides PF-2 ingests under `data/` with registry records (about 11 MB for
the P0 and P1 set); keep vendor, student, Arabic-only, unverified DOCX files and the archive outside Git. Security and
privacy: PASS, no credential, key, student record or personal staff data in any document text; author names exist in
file properties only and were not copied into these artifacts; the appraisal guide is marked Internal Use (confirm
before serving). Tests (no runtime code changed): `test_sources`, `test_retrieval_scope`, `test_planf_imports`,
`test_compat`, `test_clause_retrieval_intent` all passed (207 tests). Frozen specialists: the six files are content-identical to
the freeze commit `44ada15` (git blob hashes equal at the freeze commit, at HEAD and for the LF-normalised working copy,
6 of 6). Note for later checks: the working copy on this branch was re-written with CRLF line endings at checkout
(`core.autocrlf`), so a raw byte hash of `navigation_cues.py` and `institutional.py` no longer equals the value recorded
at freeze time; compare git blobs or LF-normalised content, not raw working-copy bytes.
