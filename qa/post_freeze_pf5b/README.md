# Post-freeze phase PF-5B: natural-language guide generalization and fresh held-out validation

Date: 2026-10-31. Branch `feature/post-freeze-knowledge-expansion`, starting commit `925d341` (PF-4). Frozen control:
`runtime-freeze-i9` (D-108). Decision record: D-114. No new document, no Arabic, no voice, no UI feature; the corpus stays
8 enabled sources, 309 pages, 15,000 chunks.

## Outcome in one paragraph

PF-5B replaced the PF-4 false-fallback causes with generic, deterministic rules (wording, software-versus-office routing,
misspelled system names, claim-level conflicts, mixed-label procedures and cut text), had them attacked by a four-lens
adversarial review and by a PF-4-versus-PF-5B worst-case differential over 1,943 probe questions, and kept only what
refuses more or serves a verified answer. Nothing served is wrong in any set: P0 rises from 19 to 21 of 37 core guide
questions (19 correct, 2 acceptable but limited; every PF-4 answer byte-identical), P1 is unchanged (7 of 41), the frozen
192-question set serves the same 55 answers as I-9, the 60-case wrong-source check has 0 wrong-source acceptances, and the
quota and journal conflicts, the appraisal hold and the policy boundary hold. Generalization did not improve materially:
the PF-4 held-out set goes from 0 to 1 of 50 (1 of 15 answerable), and the **fresh held-out set, frozen before any rule was
written, serves 0 of 56 (0 of 20 answerable)**. The dominant causes are structural, not lexical: 12 of the 18 answerable
guide questions never name the system the guide documents, and conversational questions add a clause no source supports.
`PLAN_F_GUIDE_SCOPES` stays off by default.

## The PF-4 starting defects and the generic fixes (`false_fallback_analysis.json`, `old_held_out_before_after.json`)

| Area | Rule (D-114) | Effect on the PF-4 diagnostic case |
|---|---|---|
| A wording | discourse words are not terms; the default "final" may be unstated only in a procedure ask, attributively, when the registry record declares it and the guide text names neither final nor midterm; compounds ("log in"/"login"); two noun synonym pairs (online/virtual, student/learner) | H03 served (correct) |
| B routing | a software clause an approved guide binds stays with its specialist (no Institutional handoff), and a one-clause software question the Coordinator gives to Institutional Navigation goes to the specialist owning the guide; physical places never bind; organizational units, roles and contact asks keep Institutional | H09, H12 kept by their specialist (guide does not state the asked words); P0 G14, G35 now served (limited) |
| C misspellings | only Blackboard, Collaborate, iThenticate, Gradebook, Banner; bounded edit distance; "colab" only in product position with product context; never policy words; typed text kept for display, logs and served clauses | H26, H27, H29: system names corrected, other misspelled words are not |
| D multi-part | unchanged all-or-nothing contract (no partial answers) | H30, H32, H40, H41 still fall back |
| E conflicts | a VALUE conflict blocks the units that state the disputed value; other conflicts keep blocking the chunk | H04: upload steps no longer blocked; the clause asks words the guide does not print |
| F structure | wrapped lines joined; lettered sections over numbered sub-sections recovered; cut units, statements ending with ":" or a next-item label, statements that continue an unstated step, the cut last step of a flagged procedure, lower-case cut openings refused | H05: section recovered, refused (its text is cut) |

Rules tried and removed after the adversarial review, because each produced an answer PF-4 refused and that was wrong:
tolerance for marked or predicate modifiers ("Is recording required ...?"), -ment/-tion derivations, verb synonyms, scope
words as non-critical, "once/until" as function words, binding "in my Blackboard course" with the system name, the alias
"collab", verb and room uses of "colab".

## Adversarial review and differential (`metrics.json` adversarial_review)

Four lenses (false acceptance, overfitting, typo and routing, code correctness), one independent skeptic per finding: 31
findings, 17 verified real (6 high, 6 medium, 5 low), 14 not reproduced; every reproduced issue was fixed by refusing more.
Three later findings came from the application smoke and the reruns (a Coordinator-routed software "where" question, the
chip naming the original specialist, two statements that were not self-contained) and were fixed the same way. The
worst-case differential (every chunk of each named guide retrieved) over 1,943 probe questions (P0, P1, PF-4 held-out,
wrong-source and the reviewers' probes; the fresh set excluded) compares the PF-4 code with the final code: 12 newly
accepted answers (11 correct, 1 acceptable but limited, 0 wrong), 25 refusals (several of them PF-4 false acceptances),
12 weak yes/no probe answers that pick another unit (served by no evaluation set).

## Fresh held-out methodology (`fresh_held_out_questions.jsonl`, `fresh_held_out_labels.json`)

Before any PF-5B rule was written, three independent writer agents, given only the eight documents' titles with a one-line
purpose, a persona, a topic slice and a style mix, wrote 56 questions (no project file, no PF-4 question, no failure reason,
no guide wording). The file was frozen at 2026-10-31T09:37:16Z, SHA-256 `966b9664...e0a5`, pinned by a test. Labels were
added afterwards by two independent labelers per question reading only the source texts (43 agreed, 13 adjudicated from the
source text): 20 answerable, 23 partially, 13 not at all; Plan F expected to serve 18, must not serve 22, either 16.
Exposure, recorded: the implementing session saw the questions when freezing them; the overfitting reviewer ran them
against PF-4 and an intermediate draft (0 served in both) and found five list entries that occur only in fresh wording,
which were removed. No rule was added or changed to serve a fresh question; the result below is from the final code.

## Results (`metrics.json`; four sets, never one headline)

| | questions | Plan F served | correct / limited / wrong | precision | coverage |
|---|---|---|---|---|---|
| P0 core guide (PF-3D set) | 37 | 21 (PF-4: 19) | 19 / 2 / 0 | 100% | 56.8% (PF-4 51.4%) |
| P0 all guide answers | 100 | 35 (PF-4: 33) | 32 / 3 / 0 | 100% | 33 byte-identical, 2 new (G14, G35) |
| P1 (incl. negatives, cross-document) | 41 | 7 | 5 / 2 / 0 | 100% | byte-identical to PF-4 |
| PF-4 held-out (diagnostic) | 50 (15 answerable) | 1 (PF-4: 0) | 1 / 0 / 0 | 100% | 2.0% of all, 6.7% of answerable |
| FRESH held-out | 56 (20 yes, 23 partial, 13 no) | 0 | 0 / 0 / 0 | n/a | 0% of all, 0% of answerable |

Fresh fallbacks (`fresh_held_out_results.json`, `false_fallback_analysis.json`): of the 20 answerable questions, 7 are
multi-part with an unsupported clause (often a statement or complaint), 6 find no specialist and 4 are decided by the
handbook path because no guide system is named, and single questions stop at structure, grounding or routing ambiguity.
Only 17 of the 56 fresh questions name a guide's system at all. Every "must not serve" fresh question falls back.

## Safety

Conflicts (iThenticate quota, journal recommendation), handbook contamination, wrong-source, appraisal, HR, annual leave,
authority, policy-boundary and unsupported groups: nothing served from a guide (P11 and C29 are the reviewed correct
exceptions carried from PF-4). Wrong-source check (`wrong_source_results.json`): 60 cases (16 PF-2, 20 PF-4, 24 PF-5B on
typo collisions, software versus office "where", the verb "collaborate", attendance policy versus tool, grade versus
Gradebook, submission versus Grade Transfer, research journal versus Blackboard journal): 0 wrong-source acceptances, 0
violations; Plan F served 10, all correct. Quota questions stay blocked; the iThenticate upload and login procedures are
served. Cross-document (`cross_document_results.json`): 14 questions, 6 served (handbook plus a P0 guide, and P0 plus P1),
0 wrong. Procedure QA (`procedure_qa.json`): every served procedure, section or list complete, verbatim and in printed
order.

## Frozen baseline, SHADOW, LEGACY, determinism

Frozen 192 with guide scopes on (`frozen_baseline_comparison.json`): the same 55 Plan F answers as I-9, 0 wrong, 0 false
acceptances, SHADOW 192 of 192; only T07's legacy-path citation differs (documented in PF-4). The master set never selects
the guide registry (0 of 384 records), so no D-114 rule runs on it. SHADOW on the PF-5B sets: 247 of 247 identical to the
direct legacy call. Scopes off versus on: 47 differences, all reviewed answers or explained (H09, H11, H12 kept by their
specialist; H27 routed by the corrected name; same visible legacy answer). LEGACY fresh process (`legacy_isolation.json`):
16 of 16 minimal records, no Plan F, D-111, D-112 or D-114 module imported (`name_spelling` included), no runtime cache.
Determinism: 480 repeat records, 0 differences. Plan F model calls: 0. Exceptions: 0.

## Application (`app_smoke.json`)

ACTIVE, 19 turns on the real chat page: natural and misspelled Blackboard and Collaborate questions served with their
guide, page and printed heading (the user's message is displayed as typed), Gradebook, Grade Transfer ("final course
grades"), the iThenticate upload procedure, the attendance tool, communication tools and a two-guide, two-specialist
multi-part answer (each clause under its own heading and chips) served; the quota conflict, the attendance policy trap, the
cut mixed-label Collaborate procedure, the research-group journal, Turnitin and the held appraisal guide fall back; the
software "where" question goes to the guide's specialist, the support office stays with Institutional Navigation. 0 wrong,
0 model calls, no internal id in the visible text.

## Performance (`metrics.json` performance_idle_before_after)

Measured idle, back to back, on the same 191 questions with the PF-4 code and the final PF-5B code: cached knowledge-base
startup 23.6 s and 23.4 s; guide index build p50 19.5 ms and 20.1 ms; scoped retrieval per clause p50 575 ms and 564 ms;
selection per clause p50 1.3 ms and 2.4 ms (the only change above 20%, +1.1 ms: compound stems and the per-guide vocabulary
built once per runtime); served guide answer p50 591 ms and 582 ms; guide-scope fallback p50 1,978 ms and 1,916 ms. No
material regression.

## Tests

`tests/test_pf5b_generalization.py` (135 tests): every fix area with positive and negative cases, the appraisal hold and
held sources, and the frozen fresh-set hash. Targeted gate (32 modules): 2,630 passed. Full suite: 7,082 passed, 0 failed,
0 skipped, 0 xfail (6,947 at the start; 135 new). `pip check`: no broken requirements. Security scan of the 28 changed and
new files: 0 problems.

## Recommendation

`PLAN_F_GUIDE_SCOPES` default: **OFF**. Every safety criterion holds, but fresh held-out coverage of the answerable
questions is 0% (criterion: at least 50% at 100% precision), so generalization is not credible. The remaining gap is not a
wording rule: most answerable fresh questions never name the system a guide documents, or combine a supported ask with an
unsupported statement. Closing it needs architectural decisions (inferring the system a question is about, and a
per-clause partial-answer contract) that change the D-110/D-111 and I-5 boundaries.

## Files

| File | Content |
|---|---|
| `fresh_held_out_questions.jsonl`, `fresh_held_out_labels.json` | the frozen fresh set (hash recorded) and its two-labeler labels |
| `fresh_held_out_results.json`, `fresh_held_out_review.json` | per fresh question what ACTIVE did and why; served fresh answers with verdicts (none) |
| `old_held_out_before_after.json` | the PF-4 held-out set before and after, with each starting defect's fix and result |
| `false_fallback_analysis.json` | the PF-4 taxonomy and the fresh stop layers |
| `p0_regression.json`, `p1_regression.json` | P0 and P1 answers carried byte-identical; P0 G14 and G35 new and reviewed |
| `frozen_baseline_comparison.json` | the 192-question master set against I-9 |
| `wrong_source_results.json`, `cross_document_results.json`, `procedure_qa.json` | adversarial, cross-document and procedure checks |
| `app_smoke.json`, `legacy_isolation.json` | application and LEGACY checks |
| `metrics.json` | compact metrics of every set, safety, invariance, review, latency and performance |
| `run_eval.py`, `make_review.py`, `analyze.py`, `derive.py`, `wrong_source.py`, `legacy_isolation.py` | the scripts |
