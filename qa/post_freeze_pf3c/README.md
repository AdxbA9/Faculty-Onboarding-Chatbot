# Post-freeze phase PF-3C: guide grounding, evidence selection and verified procedure assembly

Date: 2026-10-31. Branch `feature/post-freeze-knowledge-expansion`, built on the uncommitted PF-3B working tree over the
PF-2 commit `fd14af8`. Frozen control: `runtime-freeze-i9` (commit `44ada15`, D-108). Decision records: D-110 (PF-3B,
finalised) and D-111 (this phase).

## Outcome in one paragraph

Guide answers are now served, and only verified ones. With `PLAN_F_GUIDE_SCOPES` on, Plan F serves **10 of the 37 core
guide questions (27.0%; PF-3B 0), all 10 correct** on hand review, with 0 wrong. Across all 100 evaluation questions 22
guide answers are served: 21 correct, 1 acceptable but limited, 0 wrong. They include complete procedures, a numbered list
continued across two pages, and four cross-document answers in which a handbook clause and a guide procedure are each
verified on their own. The evidence the D-111 path selects is on target for every question it selects for (12 of 12;
PF-3B 5 of 24). Every conflict, contamination, approval, wrong-source, appraisal and incomplete-procedure case falls back.
The frozen 192-question set is identical to I-9, SHADOW is unchanged (100 of 100), and Plan F makes 0 model calls.
`PLAN_F_GUIDE_SCOPES` stays off by default; enabling it is the project owner's decision.

## Measure first: why PF-3B quoted off-target guide evidence

`evidence_selection.py` (before any change) ran the 37 guide questions against hand-labelled target passages
(`guide_targets.json`):

| retrieval measure (before) | value |
|---|---|
| whole-corpus retrieval: expected guide first / in the top five | 36 / 37 of 37 |
| guide-scoped retrieval: on-target chunk first | 30 of 37 (partial 2, off target 5) |
| guide-scoped retrieval: on-target chunk in the top five | 35 of 37 |
| guide-scoped retrieval: top chunk is the guide's generic introduction window | 11 of 37 |
| a procedure chunk in the top five | 29 of 37 |

Retrieval already finds the right passage. The top chunk, however, is usually a page-wide prose window, and the frozen
specialists quoted from it the sentence that names the system (their grounding rewards the system name), which is
mostly the introduction: 14 of 24 PF-3B quotes were off target. PF-3C therefore fixes unit-level selection and
grounding, not the ranking, and adds no score constant.

## What changed

| Area | Change |
|---|---|
| `handbook_bot/guide_grounding.py` (new) | the D-111 rules: guide intent (the I-4 request reader plus three guide readings), the binding of declared system names and of the faculty actor, statement units and intent checks, the procedure index with heading provenance, page continuations and cut-step completion, completeness rules, selection, the independent verification, rendering of a procedure heading |
| `handbook_bot/agents/specialists/guide_evidence.py` (new) | guide-scoped retrieval and selection (`GuideEvidence`) and the wrapper that puts the I-9 specialist behind the guide path (`GuideCapableSpecialist`) |
| `handbook_bot/plan_f_runtime.py` | the guide registry wraps the same I-9 specialist objects; the runtime keeps the guide index; the adapter verifies a guide clause as one answer (`guide_guard`), refuses guide evidence outside the D-111 path, copies the D-111 record into the evidence item and renders a procedure heading on its own line |
| `handbook_bot/plan_f_verification.py`, `plan_f_aggregation.py` | a set-level guide branch in the findings verifier and the aggregator (the handbook branches are unchanged) |
| `handbook_bot/source_scopes.py` | `SYSTEM_NAMES` (the declared names a guide may bind), `FACULTY_AUDIENCES`, `faculty_audience`, `usable_record` |
| `handbook_bot/sources.py`, `knowledge/sources.json` | the registry field `audience` (four guides: faculty and instructor or researcher; handbook: faculty; held appraisal record: none) |
| `ui/pipeline.py`, `ui/presentation.py` | diagnostics `guide_answer_forms`, `guide_answer_parts`; Developer row "Guide answer" |
| frozen specialists | untouched in PF-3C (the PF-3B D-110 boundary option stays, tested; the runtime no longer uses it) |

## Results

### Guide evaluation (`results.json`, `review.json`, `metrics.json`; 48 PF-2 + 22 PF-3B + 30 PF-3C questions)

| Core guide questions (G01-G37) | PF-3B | PF-3C |
|---|---|---|
| served by Plan F / fallback | 0 / 37 | 10 / 27 |
| coverage | 0% | 27.0% |
| correct / limited / wrong | - | 10 / 0 / 0 |
| precision | n/a | 100% |
| answer forms | - | 6 complete procedures, 1 unordered-list item, 3 statements |

Per source (served of questions): Blackboard short guide 4 of 13, gradebook guide 1 of 10, grade transfer guide 3 of 6,
iThenticate 2 of 8. Per specialist (routed / served): Teaching 28 / 8, Research 6 / 2, Institutional Navigation 4 / 0
(not approved for the gradebook guide or iThenticate, D-110).

Why the other 27 core questions fall back: the D-111 path found no provably complete, fully covered evidence 21 (steps
printed without list markers, lists cut or mixed by the PDF reading, headings carried from the page, an asked word the
guide does not print), no approved guide for the owning specialist 3 (routed to Institutional Navigation), the aggregator
2 (cross-document questions with an unsupported handbook clause), routing ambiguity 1.

All served guide answers (any category, reviewed by hand in `review.json`): 22, of which 21 CORRECT, 1 ACCEPTABLE BUT
LIMITED (C04: the dashboard by its direct link only), 0 WRONG. They include the PF-3C questions C01, C04, C05, C06, C07
(the gradebook list continued across pages 2 and 3, cited on both), C09, C11, the cross-document answers P16, C15, C16,
C17 (handbook clause plus guide procedure, each verified on its own, no bridge text) and P11 (the Post option in the
gradebook column header).

Safety groups, none served: conflict variants 6 (the quota never served), handbook contamination 6, the grade transfer
authority question, wrong-source collisions 18 (approval, submission, review, attendance, support, login, grade,
research, originality, dashboard, assessment), the held appraisal guide, incomplete procedures 2, known HR and annual
leave. SHADOW identical to the direct legacy call 100 of 100 (Plan F accepted 26 internally). Guide scopes off versus on:
22 differences, every one a reviewed guide answer. Determinism: 186 repeat records, 0 differences. Exceptions 0. Plan F
model calls 0.

### Evidence selection before and after (`evidence_before.json`, `evidence_after.json`, `review.json`)

| | PF-3B (frozen extraction) | PF-3C (D-111 selection) |
|---|---|---|
| guide questions with selected guide evidence | 24 | 12 |
| on target / partial / off target (by hand) | 5 / 5 / 14 | 12 / 0 / 0 |
| generic introduction selected | most of the 14 | 0 |

For the other 25 questions the D-111 path selects nothing rather than off-target text. The guide-scoped retrieval
numbers in the first table are unchanged by PF-3C (it changes selection, not ranking).

### Frozen 192-question set with guide scopes ON (`baseline_comparison.json`)

| | I-9 | PF-3B | PF-3C |
|---|---|---|---|
| served / fallback / coverage | 55 / 137 / 28.6% | 55 / 137 / 28.6% | 55 / 137 / 28.6% |
| wrong / false acceptances / false fallbacks | 0 / 0 / 11 | 0 / 0 / 11 | 0 / 0 / 11 |
| questions changed versus I-9 | - | 0 | 0 (ACTIVE and SHADOW) |
| SHADOW identical to legacy | 192 | 192 | 192 |
| Plan F model calls | 0 | 0 | 0 |

No master question names a guide system, so none selects the guide registry.

### Application smoke (`app_smoke.json`, real chat page, guide scopes on)

ACTIVE, 12 turns: the Blackboard access procedure, the multi-page gradebook list, the send-grades procedure and the
iThenticate upload procedure render directly (printed heading on its own line, steps in order, the guide's title and
page chips, Developer row "Guide answer: procedure (n parts)"); the grade transfer approval question, the disputed quota,
the known HR question, annual leave, the Banner Self Service collision and the held appraisal question fall back; the
handbook teaching load is served as at I-9; the cross-document question aggregates handbook page 81 with the iThenticate
access procedure. SHADOW, 3 turns: legacy visible, Plan F guide acceptance internal only. LEGACY: fresh-process checks
(I-8: 30 of 30 minimal records; `legacy_isolation.py`: 8 of 8, no Plan F or D-111 module imported, no runtime cache).

### Performance

| | value |
|---|---|
| guide index build (29 procedures, 0 pages skipped) | about 7 ms, once per knowledge base |
| guide registry | wraps the I-9 specialist objects: no second build (0.1 ms) |
| guide path per clause: retrieval / selection | p50 557 ms, p95 1,101 ms / p50 0.8 ms |
| D-111 verification in the adapter (guide guard, findings verifier, legacy Verifier) | about 1 ms each |
| served guide answer, wall time, same questions scopes on versus off | p50 586 ms versus 1,479 ms (no legacy run follows) |
| guide question that falls back, same questions scopes on versus off | p50 +595 ms, p95 +1.16 s (the extra scoped retrieval) |
| master set ACTIVE p50 / p95 | 1,194 / 2,450 ms (I-9 1,252 / 2,590) |

### Tests and checks

Targeted gate (the 26 PF-3B gate modules plus `test_guide_grounding.py`): 2,252 passed, 0 failed, 0 skipped. Full
suite: **6,858 passed, 0 failed, 0 skipped** (6,770 at the start; 88 new D-111 tests). `pip check`: no broken
requirements. Security scan of the changed and new files: 0 problems. Real-model wrong-source retrieval
(`wrong_source_retrieval.json`): 16 of 16 unchanged.

## Recommendation

PF-3C meets its criteria. Before expanding to the remaining approved guides, the most valuable targeted refinement is
the guide reading itself (steps printed without list markers, mixed lists), which accounts for most of the remaining
fallbacks; it needs its own decision record and the same evaluation. `PLAN_F_GUIDE_SCOPES` may be considered for the
default only by the project owner.

## Files

| File | Purpose |
|---|---|
| `guide_targets.json` | the hand-labelled target passage of every guide question |
| `evidence_selection.py`, `evidence_before.json`, `evidence_after.json` | retrieval and evidence-selection measurement before PF-3C and with the D-111 selector |
| `pf3c_questions.jsonl` | 30 new labelled questions: grade transfer workflow and authority, Blackboard versus gradebook, iThenticate, cross-document pairs, shared-term collisions, appraisal, incomplete procedures |
| `run_eval.py`, `results.json` | the real-KB run through `ui.pipeline` (100 questions): ACTIVE and SHADOW with guide scopes on, DIRECT with the guide path record, ACTIVE with scopes off, LEGACY, repeats |
| `make_review.py`, `review.json` | the manual review of every served guide answer and of the selected evidence |
| `analyze.py`, `metrics.json` | guide, per-source, per-specialist, safety-group, SHADOW, control, determinism, latency and model-call metrics |
| `baseline_comparison.json` | the 192-question rerun with guide scopes on against I-9 |
| `wrong_source_retrieval.json` | the PF-2 real-model wrong-source retrieval check, rerun |
| `app_smoke.json` | the application smoke test (ACTIVE 12 turns, SHADOW 3, LEGACY checks) |
| `legacy_isolation.py`, `legacy_isolation.json` | the LEGACY fresh-process check including the D-111 modules |
| `decision_summary.md` | the D-111 decision in brief |

Reproduce:

    python -X utf8 qa/post_freeze_pf3c/evidence_selection.py . <before.json>              (on the PF-3B tree)
    python -X utf8 qa/post_freeze_pf3c/evidence_selection.py . <after.json> --pf3c
    python -X utf8 qa/post_freeze_pf3c/run_eval.py . <out dir>
    python -X utf8 qa/post_freeze_pf3c/make_review.py <out dir>/results.json <after.json> qa/post_freeze_pf3c/review.json
    python -X utf8 qa/post_freeze_pf3c/analyze.py <out dir>/results.json qa/post_freeze_pf3c/review.json <metrics.json>
    PLAN_F_GUIDE_SCOPES=1 python -X utf8 qa/plan_f_i8/master_eval.py . <rerun dir>
    python -X utf8 qa/post_freeze_pf2/baseline_compare.py qa/plan_f_i8/results_2026-10-30.json <rerun dir>/i8_results.json <rerun dir>/metrics.json <out json>
