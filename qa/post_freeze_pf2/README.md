# Post-freeze phase PF-2: controlled official-guide ingestion and post-ingestion evaluation

Date: 2026-10-30. Branch `feature/post-freeze-knowledge-expansion`, on top of the PF-1 commit `5213736`.
Frozen baseline: `runtime-freeze-i9` (commit `44ada15`, D-108). Decision record of this phase: D-109.

## What PF-2 did

1. **The knowledge base is registry-driven and multi-document.** `knowledge/sources.json` (schema 2) lists six records;
   the loader (`handbook_bot/knowledge_base.py`, `assemble_corpus`) reads the enabled ones in registry order. The Faculty
   Handbook comes first and is chunked exactly as before (13,688 chunks, ids 0..13687, texts byte-identical to a
   one-document build). Each further document keeps its own 1-based page numbers, is loaded on its own (an error names
   the source), passes a deterministic page-level language filter (`english_only`: Arabic-script share of letters above
   0.30 drops a page, above 0.50 drops a row), is chunked with the procedure-aware chunk types (`procedure_step`,
   `procedure_window`, system manuals only; the four classic types are unchanged) and is annotated with its record's
   identity (`source_id`, title, `citation_label`, `source_page`, `document_type`, `authority_level`, `system`,
   `department`, `language`, `content_type`, `source_conflict`). No handbook section is forced onto a guide: `chapter`,
   `section_no`, `section_title` are null and `page_section_nos` is empty on every guide chunk.
2. **Four approved P0 guides are ingested** (copies under `data/guides/`, stable ids, sha256 in the registry):
   `uos_blackboard_faculty_short_guide` (11 pages), `uos_grade_transfer_dashboard_guide` (pages 1-3 indexed, Arabic
   pages 4-6 excluded), `uos_blackboard_gradebook_guide` (10 pages), `uos_ithenticate_guidelines` (page 1 indexed,
   Arabic page 2 excluded, near-empty page 3 yields no chunk). Corpus: 298 indexed pages, 14,549 chunks (861 from the
   guides: 889 paragraph, 8,821 row, 4,684 row window, 109 procedure step, 46 procedure window in all).
3. **Performance Appraisal Phase 2 User Guide: INGESTION HOLD - INTERNAL USE REVIEW REQUIRED.** Every page carries the
   running header "Internal Use". Its record is in the registry, disabled; the file is not committed, not loaded, not
   served. The pilot proceeds with four documents.
4. **The iThenticate yearly limit is a registered conflict** (English page: 30 documents per year; Arabic page: 15). The
   loader stamps the eight chunks that carry it (`source_conflict`); the adapter rejects any Plan F finding that rests on
   such a chunk (`PLAN_F_REJECTED_BY_SOURCE_CONFLICT`, pass-through and aggregation paths). Nothing is merged or chosen.
5. **Cache identity**: `cache/corpus_<16 hex>_v13_*` keyed by the ordered enabled sources (id, file sha256, page filter,
   chunking, section map name) and the cache format; no path takes part. `CACHE_VERSION` is v13. No cache is committed.
6. **Scope-extension seam** (`handbook_bot/source_scopes.py`, `RetrievalScope.also_source_ids`): every record names the
   specialists that may retrieve from it (`domains`, the PF-1 mapping: Teaching = handbook + Blackboard short guide +
   gradebook + grade transfer; Research = handbook + iThenticate; Faculty Services = handbook only while the appraisal
   guide is held; Institutional = handbook + Blackboard short guide + grade transfer). `build_specialist_registry` applies
   it when `PLAN_F_GUIDE_SCOPES` is on (default off) and, whatever the switch, refuses to build Plan F when a non-handbook
   document has page numbers inside a specialist's handbook page ranges (the frozen section guard attributes chunks by
   page alone; the pilot guides have 2-11 pages, the lowest scoped handbook page is 14, so there is no collision).
7. **Citations**: one document chip per cited document, each followed by its own page chips (`Page N`, with the page's
   heading when the evidence names one); a one-document answer renders exactly as before. The Developer panel gains a
   "Source ids" row; the diagnostics gain `legacy_sources` and `plan_f_sources` (ids only).
8. **Frozen files untouched**: the six frozen specialist files and their helper contracts are content-identical to
   `44ada15` (git blob ids; the raw sha256 of two of them differs only because this checkout wrote them with CRLF).

## The limit this phase measured rather than worked around

Each frozen specialist rejects, inside its own file, any retrieved chunk whose `source_id` is not the handbook
(`teaching.py`, `research.py`, `faculty_services.py`, `institutional.py`), and the section guard groups chunks by page
number without a source filter. Until a decision record reopens those lines, **no Plan F answer can rest on guide
evidence**; the unscoped legacy retrieval already cites the guides. The seam is in place and tested; turning it on
changes nothing served today (measured below).

## Results

### Frozen 192-question master set (`qa/plan_f_i8/master_eval.py` rerun on the multi-source corpus; `baseline_comparison.json`)

| | I-9 frozen | PF-2 rerun |
|---|---|---|
| questions / served / fallback / coverage | 192 / 55 / 137 / 28.6% | 192 / 55 / 137 / 28.6% |
| correct / limited / wrong (I-8 review) | 55 / 0 / 0 | 55 / 0 / 0 |
| false acceptances / false fallbacks | 0 / 11 | 0 / 11 (same ids) |
| questions with any change (served, outcome, routing, handoff, specialist calls, path, findings reason, answer, pages) | - | **0 in ACTIVE, 0 in SHADOW** |
| SHADOW identical to the direct legacy call | 192 / 192 | 192 / 192 |
| determinism subset (24 x 4) | stable | stable |
| LEGACY sample, fresh process | 30 minimal records, no Plan F import | 30 minimal records, no Plan F import, no runtime cache |
| Plan F model calls | 0 | 0 |
| ACTIVE wall p50 / p95 | 1,252 / 2,590 ms | 1,185 / 2,370 ms |
| Plan F attempt p50 | 653 ms | 598 ms |
| knowledge base load inside the harness | 22.9 s | 23.5 s |

Verdict: IDENTICAL. The 137 legacy fallbacks keep their pages too: no guide chunk displaced a handbook page in the top
evidence of any master question.

### Guide evaluation (`guide_questions.jsonl`, 48 questions; `results.json`, `review.json`, `metrics.json`)

37 guide questions (13 Blackboard short guide, 10 gradebook, 6 grade transfer, 8 iThenticate including 2
cross-document asks), 1 ambiguous, 4 wrong-source or held-document controls, 6 handbook baseline and regression
questions. Every question ran in ACTIVE, SHADOW, ACTIVE with the seam on, and LEGACY; 41 ran twice more in ACTIVE.

| Metric | Value |
|---|---|
| Plan F served, guide questions | **0 of 37** (expected: the frozen source guard); outcomes UNSAFE_RESULT 29, NOT_APPLICABLE 5 (iThenticate: no Coordinator cue), REJECTED_BY_AGGREGATOR 2, REJECTED_BY_ROUTING 1 |
| Plan F served, all 48 | 4 (the handbook baseline questions T01, R01, I01, H01; answers, pages and outcomes identical to I-9) |
| served guide answers reviewed CORRECT / LIMITED / WRONG | 0 / 0 / 0 (none served) |
| legacy retrieval, expected document first | 36 of 37 (97.3%); never the handbook instead of a guide |
| legacy retrieval, expected document in the top five | 37 of 37 |
| legacy retrieval, expected page in the top five | 36 of 37 (miss: G21, the 255-character title limit) |
| manual review of the top-five evidence | RETRIEVAL_CORRECT 33, PARTIAL 3 (G16 posting step cut, G36 and G37 one-sided cross-document evidence), MISS 1, CONFLICT 1, WRONG_SOURCE_RISK 2, correct no-answer 2 |
| candidate pool from the guides (mean over guide questions) | 86% |
| conflict question G31 | not served by Plan F (NOT_APPLICABLE in every mode); the legacy evidence items carry the conflicted rows (5 of 5) and the page shows no evidence text; a model answer on the legacy path would need the same guard |
| wrong-source controls (G39-G42) | 0 served; no appraisal content anywhere |
| wrong-source retrieval, real models (`wrong_source_retrieval.json`, 16 questions on shared terms: approval, submission, review, attendance, support, login, and system-versus-policy pairs) | expected document first 16 of 16; expected page in the top five 16 of 16 |
| baseline questions against the I-9 run | 6 of 6 unchanged (served, outcome, answer, pages) |
| SHADOW | 48 of 48 identical to the direct legacy call; same Plan F outcome as ACTIVE 48 of 48 |
| seam on (`PLAN_F_GUIDE_SCOPES`) | registry built with Teaching +3, Research +1, Institutional +2 sources, 0 collisions; **0 of 48 questions changed** (the specialists' own guard still rejects guide chunks) |
| determinism | 82 repeat records, 0 differences |
| latency ACTIVE p50 / p95 (guide questions) | 1,361 / 2,306 ms; legacy p50 807 ms; cold first request 2,504 ms (registry 679 ms) |
| exceptions, model calls | 0, 0 |

### Performance

| | I-9 | PF-2 |
|---|---|---|
| corpus | 272 pages, 13,688 chunks | 298 pages, 14,549 chunks (+6.3%) |
| cold embedding build (models + chunking + embeddings) | not re-measured in I-9 | 330 s and 334 s on two runs (models 21 s, assembly 1.4 s, embeddings 305-310 s, CPU) |
| cached startup | 22.9 s | 23.2 s first run, 15.5 s with a warm model cache (assembly 1.4 s, cache load 0.2-0.3 s) |
| embedding matrix / cache on disk | 13,688 x 768 | 14,549 x 768 (44.7 MB); 65.4 MB in four cache files |
| registry build (cold) | 712 ms | 679-721 ms |

### Tests

Targeted gate (I-8 list plus `test_multi_source.py` and `test_retrieval_scope.py`): 1,641 passed. Full suite:
**6,730 passed, 0 failed** (6,674 at the freeze plus the 56 new multi-source tests). `pip check`: no broken
requirements. Security scan of the changed and new files: 0 problems.

### Application smoke (`app_smoke.json`)

ACTIVE (11 turns), SHADOW (4) and LEGACY (3) on the real chat page with Developer mode: guide questions show the
document chip and its page chips and fall back to the legacy path; the held appraisal question refuses with no chip;
the handbook teaching-load question is served by Plan F in ACTIVE (page 91) and stays legacy-visible in SHADOW; the
known HR and annual-leave regressions are not served; LEGACY shows no Plan F rows and the 14,549-chunk corpus.

## Acceptance against the PF-2 thresholds

- Frozen baseline reproduced: yes (0 changed questions, metrics identical).
- No wrong supported answer: yes (no served guide answer; the four served handbook answers are the reviewed I-9 ones).
- Internal Use document not served: yes (held, not committed).
- Conflict blocked from direct Plan F service: yes (guard in place and tested; no specialist can reach the chunk today).
- No Arabic chunk indexed from the guides: yes (filter report in `pilot_source_manifest.json`; the handbook's own seven
  chunks with a few Arabic characters on pages 95 and 128 are pre-existing and unchanged).
- Guide coverage by Plan F: **0%**, below any useful threshold, for a structural reason outside PF-2's scope (frozen
  source guard). This is the finding the phase delivers, not a defect of the ingestion.

## Recommendation: PF-3B (fix the structural defect before expanding)

Open a decision record to reopen, in the four frozen specialists, the one-line source guard (`source_id !=
APPROVED_SOURCE_ID`) in favour of the registry's domain policy already carried by each specialist's extended scope, and
to give the section guard a source filter (or to skip it for non-section-mapped sources). Then turn `PLAN_F_GUIDE_SCOPES`
on, rerun this folder's evaluation (served guide answers are reviewed one by one) and the 192-question comparison. Also
add Coordinator cues for iThenticate (five guide questions reached no specialist). Only after that should PF-3 add the
remaining eight approved guides; the page-collision refusal means any approved document with 14 or more pages disables
Plan F until the section guard filters by source.

## Files

| File | Purpose |
|---|---|
| `guide_questions.jsonl` | the 48 labelled questions (source, pages, domain, specialist, answer type, served expectation, conflict flag, baseline flag) |
| `run_guide_eval.py` | the real-KB run through `ui.pipeline` in ACTIVE, SHADOW, ACTIVE with the seam on, LEGACY (with the candidate pool by source), repeats |
| `results.json` | every record of that run (answers, pages, evidence items with source ids and citation checks, diagnostics, timings) |
| `make_review.py`, `review.json` | the manual review (verdicts and notes per question) |
| `analyze_guide_eval.py`, `metrics.json` | the metrics above, with the performance inputs |
| `baseline_compare.py`, `baseline_comparison.json` | the 192-question rerun against the frozen I-9 run |
| `build_manifest.py`, `pilot_source_manifest.json` | every P0 document with its decision, identity, hash and what was indexed |
| `perf_snapshot.py` | cached-startup timing |
| `security_scan.py` | hygiene scan of the changed and new files |
| `wrong_source_retrieval.py`, `wrong_source_retrieval.json` | real-model retrieval on terms shared across documents and on the system/policy boundary |
| `app_smoke.json` | the chat-page smoke session in the three modes |
| `freeze_manifest_after_pf2.json` | the refreshed I-9 manifest (raw hashes, git blob ids, the v13 corpus cache, the registry hash, test counts) |

Reproduce:

    python -X utf8 qa/post_freeze_pf2/run_guide_eval.py . <out dir>
    python -X utf8 qa/post_freeze_pf2/make_review.py <out dir>/results.json qa/post_freeze_pf2/review.json
    python -X utf8 qa/post_freeze_pf2/analyze_guide_eval.py <out dir>/results.json qa/post_freeze_pf2/review.json <metrics.json>
    python -X utf8 qa/plan_f_i8/master_eval.py . <rerun dir>
    python -X utf8 qa/post_freeze_pf2/baseline_compare.py qa/plan_f_i8/results_2026-10-30.json <rerun dir>/i8_results.json <rerun dir>/metrics.json <out json>
