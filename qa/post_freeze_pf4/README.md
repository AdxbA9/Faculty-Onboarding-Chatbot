# Post-freeze phase PF-4: controlled P1 guide expansion and held-out evaluation

Date: 2026-10-31. Branch `feature/post-freeze-knowledge-expansion`, starting commit `75d81cd` (the PF-3D tree after the
history rewrite that removed old co-author trailers; file content identical to PF-3D). Frozen control:
`runtime-freeze-i9` (D-108). Decision record: D-113.

## Outcome in one paragraph

Three of the five PF-1 P1 guides are now in the knowledge base (attendance tool, Collaborate Ultra, communication tools;
8 enabled sources, 309 pages, 15,000 chunks) and pass through the same D-111/D-112 rules as the P0 guides; two are held
(Assignments and Tests: 25 pages collide with frozen page-keyed indexes; Grade Categories: interleaved English and
Arabic). Nothing served is wrong: P0 is unchanged (19 of 37 core guide questions, 100% precision, every PF-3D answer
byte-identical), the P1 questions are served 7 of 41 times (5 correct, 2 acceptable but limited, 0 wrong), and every
conflict, policy, wrong-source, unsupported, appraisal, HR and annual-leave case falls back. The held-out set, written
blind and frozen before any P1 text was read, is the decisive result: **0 of 50 served, 0 of the 15 questions an
enabled source answers**. The deterministic guide path is precise but does not yet generalise to natural wording.
`PLAN_F_GUIDE_SCOPES` stays off by default.

## Sources (`p1_source_review.json`, `selected_sources.json`)

| PF-1 P1 candidate | Decision | Why |
|---|---|---|
| Attendance Tool in Blackboard Ultra | ACTIVATED | official ITC guide, clean English (2 pages), new tool workflow; handbook keeps the absence policy |
| Blackboard Collaborate Ultra Guide | ACTIVATED | official, names UOS, 2026, clean English (4 pages); virtual sessions |
| Blackboard Learn Ultra: Communication Tools | ACTIVATED (journal section blocked) | official, clean English (5 pages); its 2022 journal recommendation is a registered unresolved conflict |
| Blackboard Ultra Assignments and Tests Guide | HOLD (architectural) | 25 pages: the frozen Teaching calendar-row and Institutional page-start indexes are keyed by page number without a source filter; the collision check would refuse the whole Plan F registry |
| Setting Grade Categories in the Blackboard Gradebook | HOLD (language) | English and Arabic interleave on its only page; the english_only filter drops it |

All three are approved for Teaching & Learning only. The attendance tool guide never answers a policy question (the
student absence policy is handbook 12.15); the Collaborate guide is named by the product, never by the verb
"collaborate". File hashes equal the PF-1 inventory; no Internal Use, personal or contact data in the content.

## Held-out methodology (`held_out_questions.jsonl`, `held_out_labels.json`)

An independent agent, given only the document titles, the systems and the category mix, wrote 50 questions; it opened
no project file. The file was frozen (SHA-256 `86db3759...`, recorded in the labels and checked by a test) before the
P1 text was read and before any evaluation. Labels were added afterwards from the official documents: 15 answerable
from an enabled source, 19 partially, 16 not at all. No rule was changed in response to a held-out result.

## Results (`metrics.json`; three sets, never one headline)

| | questions | Plan F served | correct / limited / wrong | precision | coverage |
|---|---|---|---|---|---|
| P0 core guide (PF-3D set) | 37 | 19 | 19 / 0 / 0 | 100% | 51.4% (PF-3D 51.4%) |
| P0 all guide answers (100 questions) | 100 | 33 | 32 / 1 / 0 | 100% | unchanged, byte-identical |
| P1 source questions | 28 | 6 | 5 / 1 / 0 | 100% | 21.4% |
| P1 all (incl. negatives, cross-document) | 41 | 7 | 5 / 2 / 0 | 100% | |
| HELD-OUT | 50 (15 answerable) | 0 | 0 / 0 / 0 | n/a | 0% of all, 0% of answerable |

P1 per source: attendance 1 of 10, Collaborate 2 of 10, communication 3 of 8. Served P1 answers (`p1_results.json`):
excused attendance record, default Collaborate role, attendance overview and export (complete three-step procedure),
discussion titles, class conversations (correct); creating a discussion and the announcement-plus-discussion pair
(limited: the two creation entry points without the New Discussion page fields).

Held-out answerable fallbacks (`metrics.json` held_out.false_fallbacks_answerable): grounding, an asked word the guide
does not print (H03 "final course" grades, H06 "all the students"); conflict over-blocking (H04, the iThenticate upload
list shares chunks with the stamped quota); structure (H05, a procedure mixing A./1./bullets); source scope, a
"where/who" clause handed to Institutional Navigation, which is not approved for that guide (H09, H12, H19);
misspellings (H26-H29); multi-clause composition, conversational or cross-source wording split into clauses one of which
is unsupported (H30, H32, H40, H41). None is a safety problem.

## Safety

Conflict (8), handbook contamination (6), wrong source (22), appraisal, authority, known HR, annual leave, policy
boundary and unsupported questions: none served from a guide (P11, a guide question that must be answered from the
gradebook guide, and C29, relabelled complete in PF-3D, are reviewed CORRECT). Held-out traps (policy versus system,
wrong-source collisions, false premises, unsupported): none served. Real-model wrong-source check
(`wrong_source_results.json`): 36 cases (the 16 PF-2 cases unchanged plus 20 on the P1 vocabulary): 0 wrong-source acceptances, 0 violations; Plan F served 8, all from expected sources (7 correct, 1 acceptable but limited: a discussion-grading question answered with the whole Discussions section); retrieval put an expected source first 35 of 36 times (P14: the attendance guide outranks the Collaborate guide for 'attendance overview', and Plan F still served the Collaborate guide). SHADOW: 191 of 191 identical to the direct legacy call. Guide scopes off
versus on: 40 differences, all reviewed answers. Determinism: 368 repeat records, 0 differences. Exceptions 0. Plan F
model calls 0.

Rules refined while reviewing the P1 guides (each refuses only; tests in `tests/test_p1_guides.py` and
`tests/test_guide_structure.py`): ordered bullets under an action heading in the D-111 index; labelled sub-sections
titled after their object; a list whose last item continues on the next page; academic-standing policy words.

## Frozen baseline (`frozen_baseline_comparison.json`)

The 192-question master set with guide scopes on, final PF-4 code: Plan F serves the same 55 questions as I-9 with 0 wrong and 0 false acceptances; no master question selects the guide registry; SHADOW is identical to the direct legacy call 192 of 192; 0 model calls. One question differs from I-9, on the legacy path only: T07 'Is attendance mandatory for students?' (a legacy fallback in I-9 and now) cites attendance guide page 2 in addition to handbook pages 193 and 228, because the unscoped legacy retrieval sees the new source. Reviewed: no Plan F answer changed; the legacy citation is a related tool page, not a policy source (listed under limitations).

## Application (`app_smoke.json`, `legacy_isolation.json`)

ACTIVE 16 turns: one served answer per P1 guide (attendance statement page 2; Collaborate procedure page 4 "D- Checking
Session Attendance"; communication procedure page 4 "Conversations"), the P0 announcement, automatic zeros, grade
transfer and iThenticate answers unchanged, the held-out H06 fallback, the quota and journal conflicts, the
"collaborate" collision, handbook teaching load, HR, annual leave, a P0+P1 cross-document aggregate and the held
appraisal guide. SHADOW 2 turns: the visible answer stays legacy. LEGACY: fresh process, 12 of 12 minimal records; no Plan F, D-111 or D-112 module imported; no runtime cache.

Legacy retrieval now also ranks the P1 guides (as PF-2 established for every enabled source). One effect: for "What is
the attendance policy for students in Blackboard courses?" the unscoped legacy retrieval now cites attendance guide
page 1 instead of handbook pages 222/228; with a model key the legacy answer could draw on the tool guide. Plan F is
unaffected (it never answers policy from a guide).

## Performance

Cold knowledge-base build 358.9 s (the embedding cache is keyed by the whole enabled corpus, so adding a source
re-embeds every chunk once); cached start 13.9 s; embeddings 15,000 x 768 (46.1 MB; PF-3D 44.7 MB). Guide index (D-111 procedures and D-112 structures of seven guides) built in 20 ms per runtime build (PF-3D 13 ms). Per guide clause:
scoped retrieval p50 600 ms (PF-3D 549 ms: "Blackboard" now names five guides), selection p50 1.4 ms (p95 9 ms),
verification about 1 ms. Served guide answer p50 613 ms; served P1 answer p50 592 ms; guide-scope fallback p50 2.06 s
(PF-3D 1.97 s); scopes off p50 1.43 s; legacy p50 0.81 s. No material regression.

## Tests

Targeted gate (31 modules: registry, loader and corpus, chunking via the guide tests, grounding, structure, scopes, specialists, Coordinator, verification, aggregation, adapter, runtime, UI, legacy compatibility, conflicts, P1 and held-out safety): **2,495 passed**, 0 failed, 0 skipped. Full suite: **6,947 passed, 0 failed, 0 skipped, 0 xfail** (6,926 at the start; 21 new). State-pinning tests of the PF-2/PF-3 source set (counts, approved sets, conflict stamps) were updated to the eight-source state, each keeping its assertion. `pip check`: no broken requirements. Security scan of the changed and new files: 0 problems.

## Recommendation

`PLAN_F_GUIDE_SCOPES` default: **OFF**. Precision holds everywhere, but 0 of 15 answerable held-out questions are
served, so generalisation is not credible. Next phase: **PF-5B, targeted false-fallback refinement** of the systematic
defects the held-out set exposed (asked-word strictness for modifiers and synonyms, conversational multi-clause
splitting, Institutional handoff of guide-system "where/who" clauses, misspellings of system names, chunk-level conflict
over-blocking on the iThenticate page), each measured on a NEW held-out set before and after. The Assignments and Tests
guide needs its own decision record (source-aware frozen page indexes) before it can be ingested.

## Files

| File | Content |
|---|---|
| `p1_source_review.json`, `selected_sources.json` | the revalidation of the five candidates; the batch, scopes and claim types |
| `held_out_questions.jsonl`, `held_out_labels.json` | the frozen held-out set (hash recorded) and its labels |
| `held_out_results.json`, `held_out_review.json` | per held-out question what ACTIVE did; served held-out answers with verdicts (none) |
| `p1_questions.jsonl`, `p1_results.json` | the P1 evaluation questions; per question results, verdicts and the procedure QA |
| `p0_regression.json` | P0 guide answers carried byte-identical from PF-3D, changes (none) |
| `frozen_baseline_comparison.json` | the 192-question master set against I-9 |
| `wrong_source_results.json`, `cross_document_results.json` | wrong-source adversarial (retrieval and Plan F); every cross-document question |
| `metrics.json` | the three sets, safety groups, invariance, latency, corpus |
| `app_smoke.json`, `legacy_isolation.json` | application and LEGACY checks |
| `run_eval.py`, `make_review.py`, `analyze.py`, `derive.py`, `wrong_source.py`, `legacy_isolation.py` | the scripts |
