# Development Status

Handoff log for the Part 2 implementation. Newest entry first. Update this
file at the end of every meaningful work session, before the final push, so
the next developer can start from the branch and commit named here.

Rules: one entry per session; commit hashes are the short hashes shown by
`git log --oneline`; wording stays factual and neutral.

---

## 2026-10-31 - Controlled P1 guide expansion and held-out evaluation (post-freeze phase PF-4)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/post-freeze-knowledge-expansion` |
| Starting commit | `75d81cd` (PF-3D after the history rewrite that removed old co-author trailers; file tree identical to PF-3D) |
| Ending state | COMPLETE. Three P1 guides activated under D-113 (two held); 0 wrong Plan F answers in every set; P0 unchanged; held-out generalisation measured honestly (0 of 15 answerable questions served); one local commit for the project owner to push; `PLAN_F_GUIDE_SCOPES` stays off by default |
| Freeze status | I-9 (`runtime-freeze-i9`, D-108) remains the frozen baseline and comparison control; the frozen specialists are not changed |
| Elapsed | 2h08m08s (start 2026-10-31T06:32:37Z, end 2026-10-31T08:40:45Z; includes the background evaluation runs) |

### Before and after

| | PF-3D | PF-4 |
|---|---|---|
| enabled sources / pages / chunks | 5 / 298 / 14,549 | 8 / 309 / 15,000 (451 new) |
| P0 core guide questions served (correct / wrong) | 19 of 37 (19 / 0) | 19 of 37 (19 / 0), answers byte-identical |
| P1 questions served (correct / limited / wrong) | - | 7 of 41 (5 / 2 / 0); 6 of the 28 source questions |
| held-out questions served (answerable) | - | 0 of 50 (0 of 15) |
| frozen 192-question set | 55 served, 0 wrong | 55 served (same set), 0 wrong, 0 false acceptances; 1 legacy citation page differs (T07) |

### Completed

- P1 revalidation of the five PF-1 candidates (hash, officiality, language, extraction, conflicts, privacy): three activated
  (Attendance Tool in Blackboard Ultra, Blackboard Collaborate Ultra Guide, Blackboard Learn Ultra: Communication Tools),
  two held (Assignments and Tests: 25 pages collide with frozen page-keyed indexes; Grade Categories: interleaved
  English and Arabic). Registry records, Teaching-only scopes, system vocabulary (Collaborate by product, never by the
  verb), a registered unresolved conflict for the 2022 journal recommendation.
- A held-out set of 50 questions written blind by an independent agent and frozen (hash pinned by a test) before any P1
  text was read; labelled afterwards (15 answerable, 19 partial, 16 not).
- Four generic refinements that only refuse (ordered bullets in the D-111 index, labelled sub-sections titled after their
  object, lists whose last item continues on the next page, academic-standing policy words). D-113 recorded.

### Tests performed

- PF-4 evaluation (191 questions: P0 100, P1 41, held-out 50; ACTIVE, DIRECT, SHADOW, scopes off, LEGACY, two repeats):
  results above; SHADOW 191 of 191 identical; scopes off versus on: 40 differences, all reviewed answers; 368 repeat
  records, 0 differences; 0 exceptions; 0 model calls. Every served P1 answer reviewed by hand.
- Frozen 192-question set with guide scopes on: Plan F identical to I-9 (55 served, 0 wrong, 0 false acceptances, SHADOW
  192 of 192); T07 (legacy fallback) now also cites attendance guide page 2 through the unscoped legacy retrieval.
- Wrong-source adversarial with the real models: 36 cases, 0 wrong-source acceptances, 0 violations.
- Application smoke: ACTIVE 16 turns (one answer per P1 guide, P0 answers unchanged, held-out fallback, conflicts,
  collision, handbook, HR, annual leave, cross-document, appraisal); SHADOW 2; LEGACY fresh process 12 of 12 minimal.
- Targeted gate (31 modules): 2,495 passed. Full suite: 6,947 passed, 0 failed, 0 skipped (6,926 at the start; 21 new).
  `pip check`: no broken requirements. Security scan: 0 problems.

### Known limitations

- Held-out generalisation: 0 of 15 answerable questions served. Causes: asked words the guide does not print (modifiers,
  synonyms), conversational and cross-source questions split into clauses one of which is unsupported, "where/who"
  clauses handed to Institutional Navigation (not approved for those guides), misspelled system names, chunk-level
  over-blocking by the iThenticate quota conflict, procedures mixing A./1./bullets.
- P1 coverage is low (6 of 28 source questions); attendance-guide steps are wrapped beside screenshots and read as
  fragments, which the rules refuse.
- The unscoped legacy retrieval now cites P1 tool pages for some student-attendance policy questions (T07, C20); Plan F
  never answers policy from a guide, but a legacy answer with a model key could draw on them.
- Adding a source re-embeds the whole corpus once (about 6 minutes cold).
- The Assignments and Tests guide needs a decision record making the frozen page-keyed indexes source-aware.

### Next task

PF-5B: targeted false-fallback refinement of the systematic defects the held-out set exposed, measured on a NEW held-out
set before and after; no new documents until held-out coverage is credible. `PLAN_F_GUIDE_SCOPES` default stays a
project-owner decision (recommended OFF).

### Blockers

None.

### Files changed

`knowledge/sources.json` (three P1 records, one conflict), `data/guides/` (three P1 PDFs), `handbook_bot/source_scopes.py`,
`handbook_bot/guide_grounding.py`, `handbook_bot/guide_structure.py`, `tests/test_p1_guides.py` (new),
`tests/test_guide_structure.py`, `tests/test_guide_grounding.py`, `tests/test_multi_source.py`, `tests/test_sources.py`,
`qa/post_freeze_pf4/` (new), `docs/DECISIONS.md` (D-113), `docs/STATUS.md`.

---

## 2026-10-31 - Deterministic guide structure recovery and coverage refinement (post-freeze phase PF-3D)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/post-freeze-knowledge-expansion` |
| Starting commit | `b1d4bd1` (Enable verified guide-grounded specialist answers) |
| Ending state | COMPLETE. Plan F reads the printed structure of the guides under D-112 and serves 19 of 37 core guide questions (PF-3C 10), all correct; 33 guide answers over the 100-question evaluation, 32 correct, 1 limited, 0 wrong; the frozen 192-question set is identical to I-9; one local commit for the project owner to push; `PLAN_F_GUIDE_SCOPES` stays off by default |
| Freeze status | I-9 (`runtime-freeze-i9`, D-108) remains the frozen baseline and comparison control. D-112 adds structure recovery inside the D-111 guide path, used only with `PLAN_F_GUIDE_SCOPES` on; the frozen specialists and the handbook path are unchanged |
| Elapsed | 1h43m41s (start 2026-10-31T03:47:20Z, end 2026-10-31T05:31:01Z; includes the background evaluation runs) |

### Before and after

| | PF-3C | PF-3D |
|---|---|---|
| core guide questions served by Plan F | 10 of 37 (27.0%) | 19 of 37 (51.4%) |
| core served correct / limited / wrong | 10 / 0 / 0 | 19 / 0 / 0 |
| Blackboard short guide / gradebook / grade transfer / iThenticate | 4 of 13 / 1 of 10 / 3 of 6 / 2 of 8 | 8 of 13 / 6 of 10 / 3 of 6 / 2 of 8 |
| structural false fallbacks (identified before any change) | 10 | 9 recovered, 1 remaining (refused correctly) |
| all served guide answers (100 questions): correct / limited / wrong | 21 / 1 / 0 | 32 / 1 / 0 |
| guide evidence selected: on target / partial / off target | 12 / 0 / 0 | 21 / 0 / 0 (set measure) |
| frozen 192-question set | 55 served, 0 wrong, 0 changed | 55 served, 0 wrong, 0 changed |

### Completed

- Measured before changing: the 27 PF-3C core fallbacks classified by hand (8 legitimate, 10 structural, 9 other false
  fallbacks) in `qa/post_freeze_pf3d/fallback_classification.json`.
- D-112 recorded: structure may be recovered only from deterministic formatting and text evidence (no LLM, no invented
  steps, no guessed sequence): printed blocks, headings, section boundaries, page continuation only under no heading,
  list lead-ins and list types, exact quotes of every unit, completeness proof (one ordered label list, a heading that
  names the asked action, no missing last step); bullets never renumbered; unordered lists never presented as steps.
- New module `handbook_bot/guide_structure.py`; `guide_grounding.py` (structure index, `structure_problem`, answer form
  `section`, adapter verification, missing-last-step rule for D-111 procedures too, `citation_heading`);
  `guide_evidence.py` (structure ranks); `plan_f_runtime.py` (`guide_citation_items`: verified guide items cited under
  their printed heading instead of the carried page hint).
- Recovered and reviewed: keyboard reorder, learning module, announcement, assignment (pages 10 and 11), the two
  gradebook views, automatic zeros, gradebook download, accommodations, child-course filter; the cross-document P14; C29
  (labelled incomplete in PF-3C from the PF-2 misreading of a section label; complete on the printed page).

### Tests performed

- Frozen 192-question set with guide scopes on (final code): identical to I-9 and to the PF-3C rerun (0 changed
  questions in ACTIVE and SHADOW; 55 served, 0 wrong, 0 false acceptances, 11 false fallbacks, SHADOW 192 of 192, 0
  model calls; no master question selects the guide registry).
- Guide evaluation (100 questions; ACTIVE, SHADOW, DIRECT, scopes off, LEGACY, two repeats): core 19 of 37 served, 19
  correct, 0 wrong (precision 100%); all served guide answers 33 (32 correct, 1 limited, 0 wrong); every PF-3C answer
  unchanged; conflict, contamination, wrong-source, appraisal, authority, known HR and annual leave all fall back; SHADOW
  100 of 100 identical; scopes off versus on: 33 differences, all reviewed guide answers; determinism 0 differences;
  0 exceptions; 0 model calls.
- Evidence selection (`evidence_selection.py --pf3c`, final code): 21 of 37 with evidence, all on target as a set, 0 off
  target; PF-3C's 12 unchanged.
- Adversarial: first, middle and last step missing, continuation not indexed, heading of another procedure, two
  procedures mixed, forged or incomplete answers, unordered lists: all refused (`adversarial_results.json`).
- Application smoke with guide scopes on: ACTIVE 13 turns (recovered Blackboard and gradebook sections, grade transfer,
  iThenticate, quota fallback, handbook teaching load, known HR, annual leave, incomplete procedure, Banner Self Service,
  cross-document, held appraisal, guide statement); SHADOW 3; LEGACY fresh process 10 of 10 minimal, no D-112 module
  imported.
- Real-model wrong-source retrieval: 16 of 16 unchanged.
- Targeted gate (the PF-3C gate plus `test_guide_structure.py`): 2,320 passed. Full suite: 6,926 passed, 0 failed,
  0 skipped (6,858 at the start; 68 new D-112 tests). `pip check`: no broken requirements. Security scan of the changed and new files: 0 problems.

### Known limitations

- 18 of 37 core guide questions still fall back: 8 legitimately (no answer in the guide, specialist not approved for the
  guide, the policy/system boundary, the quota conflict, an ambiguous ask), G20 (an option inside a capability list,
  refused by design) and 9 for reasons other than structure (an asked word the guide does not print, evidence below the
  reranker gate).
- The D-112 rules were developed and measured on the same 47 guide questions; no held-out guide question set exists yet.
- A guide question that still falls back pays one scoped retrieval before the legacy path (p50 about 0.5 s more).
- `PLAN_F_GUIDE_SCOPES` stays off by default.

### Next task

Controlled P1 guide expansion under D-109, D-111 and D-112, with a held-out guide question set written without
reference to the guides' layout; the Performance Appraisal guide stays on HOLD. Enabling `PLAN_F_GUIDE_SCOPES` by
default is the project owner's decision.

### Blockers

None.

### Files changed

`handbook_bot/guide_structure.py` (new), `handbook_bot/guide_grounding.py`,
`handbook_bot/agents/specialists/guide_evidence.py`, `handbook_bot/plan_f_runtime.py`, `tests/test_guide_structure.py`
(new), `qa/post_freeze_pf3d/` (new), `docs/DECISIONS.md` (D-112), `docs/STATUS.md`.

---

## 2026-10-31 - Guide grounding, evidence selection and verified procedure assembly (post-freeze phase PF-3C)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/post-freeze-knowledge-expansion` |
| Starting commit | `fd14af8` (Add controlled multi-source guide ingestion), with the uncommitted PF-3B working tree on top (preserved; backed up before any change) |
| Ending state | COMPLETE. Plan F serves verified guide answers under D-111 (10 of 37 core guide questions, all correct; 22 guide answers over the 100-question evaluation, 21 correct, 1 limited, 0 wrong); the frozen 192-question set is identical to I-9; PF-3B and PF-3C committed together as one local commit for the project owner to push; `PLAN_F_GUIDE_SCOPES` stays off by default |
| Freeze status | I-9 (`runtime-freeze-i9`, D-108) remains the frozen baseline and comparison control. PF-3B (D-110) reopened source acceptance; PF-3C (D-111) adds one deterministic guide evidence path in front of the unchanged I-9 specialists, used only with `PLAN_F_GUIDE_SCOPES` on |
| Elapsed | 1h39m26s (start 2026-10-31T01:45:07Z, end 2026-10-31T03:24:33Z; includes the background evaluation runs) |

### Before and after

| | PF-3B | PF-3C |
|---|---|---|
| core guide questions served by Plan F | 0 of 37 | 10 of 37 (27.0%) |
| served guide answers correct / limited / wrong | - | 10 / 0 / 0 |
| guide evidence reviewed by hand: on target / partial / off target | 5 / 5 / 14 (of 24 returned) | 12 / 0 / 0 (of 12 selected; nothing selected for the rest) |
| frozen 192-question set | 55 served, 0 wrong, 0 changed | 55 served, 0 wrong, 0 changed |

### Completed

- Measured before changing: guide-scoped retrieval already ranks an on-target chunk first for 30 of 37 guide questions
  (35 in the top five); the off-target PF-3B quotes came from unit selection inside page-wide windows (the sentence naming
  the system, usually the introduction), not from ranking.
- D-110 finalised (superseded in part) and D-111 recorded: one guide evidence path; declared system names and the faculty
  actor bound by the registry (new `audience` field); every other content word stated by the evidence; complete
  procedures only (ordered labels, real heading, no page continuation left open, cut steps completed only from the page's
  own prose); policy, authority, timing and handbook-subject asks never answered from a guide; independent set-level
  verification in the adapter.
- New modules `handbook_bot/guide_grounding.py` and `handbook_bot/agents/specialists/guide_evidence.py`; adapter,
  findings verifier and aggregator guide branches; diagnostics `guide_answer_forms`, `guide_answer_parts`; Developer row
  "Guide answer". The frozen specialists are not changed by PF-3C.

### Tests performed

- Frozen 192-question set with guide scopes on: identical to I-9 (0 changed questions in ACTIVE and SHADOW; 55 served,
  0 wrong, 0 false acceptances, 11 false fallbacks, SHADOW 192 of 192, 0 model calls; no master question selects the
  guide registry).
- Guide evaluation (48 PF-2 + 22 PF-3B + 30 PF-3C questions; ACTIVE, SHADOW, DIRECT, scopes off, LEGACY, two repeats):
  core guide questions 10 of 37 served (27.0%), 10 correct, 0 limited, 0 wrong (precision 100%); all served guide answers 22 (21 correct, 1 limited, 0 wrong), including a numbered gradebook list continued across two pages and four cross-document answers; conflict, contamination, approval, wrong-source, appraisal and incomplete-procedure cases all fall back; SHADOW 100 of 100 identical; scopes off versus on: 22 differences, all reviewed guide answers; determinism 0 differences; 0 exceptions; 0 model calls.
- Evidence selection before and after (`evidence_selection.py`): guide-scoped retrieval unchanged (on-target chunk first 30 of 37, top five 35); the evidence selected by the D-111 path is on target for 12 of 12 questions (PF-3B frozen extraction: 5 on target, 5 partial, 14 off target of 24); nothing is selected for the other 25.
- Application smoke with guide scopes on: ACTIVE 12 turns (Blackboard procedure, multi-page gradebook list, grade transfer
  procedure, grade transfer authority fallback, iThenticate procedure, quota fallback, handbook teaching load, known HR,
  annual leave, Banner Self Service collision, cross-document aggregate, held appraisal); SHADOW 3; LEGACY fresh-process
  checks (I-8: 30 of 30 minimal; PF-3C: 8 of 8 minimal, no Plan F or D-111 module imported).
- Real-model wrong-source retrieval rerun: 16 of 16 unchanged.
- Targeted gate: 2,252 passed. Full suite: 6,858 passed, 0 failed, 0 skipped (6,770 at the start). `pip check`: no
  broken requirements. Security scan of the changed and new files: 0 problems (49 files).

### Known limitations

- 27 of 37 core guide questions still fall back: guide steps printed without list markers (the short guide's
  learning-module, announcement and assignment pages), lists cut or mixed by the PDF reading, headings carried from the
  page, asks whose words the guide does not print ("upload" for a file size, "cannot" versus "unable"), approval asks, and
  routing to Institutional Navigation for guides it is not approved for.
- Evidence below the production reranker gate is not considered (G21's 255-character limit).
- `PLAN_F_GUIDE_SCOPES` stays off by default.

### Next task

Evidence-driven expansion of the remaining approved guides, or targeted refinement of the guide reading (unmarked steps,
mixed lists) under its own decision record, each with the D-111 evaluation and a hand review of every served answer.
Enabling `PLAN_F_GUIDE_SCOPES` by default is the project owner's decision.

### Blockers

None.

### Files changed

`handbook_bot/guide_grounding.py` (new), `handbook_bot/agents/specialists/guide_evidence.py` (new),
`handbook_bot/plan_f_runtime.py`, `plan_f_verification.py`, `plan_f_aggregation.py`, `source_scopes.py`, `sources.py`,
`knowledge/sources.json`, `ui/pipeline.py`, `ui/presentation.py`, `tests/test_guide_grounding.py` (new),
`tests/test_multi_source.py`, `qa/post_freeze_pf3c/` (new), `docs/DECISIONS.md` (D-110 finalised, D-111),
`docs/STATUS.md`; with the PF-3B files carried in the same commit.

---

## 2026-10-31 - Controlled specialist source reopening and guide activation (post-freeze phase PF-3B)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/post-freeze-knowledge-expansion` |
| Starting commit | `fd14af8` (Add controlled multi-source guide ingestion) |
| Ending state | INCOMPLETE on the coverage criterion, changes left UNCOMMITTED in the working tree; carried into PF-3C, which builds on them and commits PF-3B and PF-3C together if PF-3C passes. The specialist source-acceptance boundary is reopened under control (D-110) and is safe; the frozen 192-question set is unchanged with guide scopes on; Plan F serves 0 of 37 guide questions because the frozen ask gate and procedure rule of the specialists (and the I-1 subject guard) refuse second-person system-manual evidence, rules this phase was not allowed to redesign; `PLAN_F_GUIDE_SCOPES` stays off by default |
| Freeze status | I-9 (`runtime-freeze-i9`, D-108) remains the frozen baseline and the comparison control. PF-3B is a controlled post-freeze reopening of ONE contract boundary (source acceptance); claim interpretation, grounding, the ask gate, completeness, binding and safety gates stay frozen |
| Elapsed | 0h57m40s (start 2026-10-30T20:34:27Z, end 2026-10-30T21:32:07Z; includes long background evaluation runs) |

### Before and after

| | before (I-9 / PF-2) | after (PF-3B) |
|---|---|---|
| sources a specialist accepts | handbook only, hard-coded | handbook plus the D-110 approved guides when `PLAN_F_GUIDE_SCOPES` is on; handbook only otherwise |
| guide evidence reaching the specialists | none | 24 of 37 guide questions |
| guide answers served by Plan F | 0 of 37 | 0 of 37 |
| frozen 192-question set | 55 served, 0 wrong | 55 served, 0 wrong, 0 changed questions |

### Completed

- D-110 recorded: why the source guard is reopened, exactly what may change, what stays frozen, the approved source
  sets, the same verification standard plus three source rules, fallback for anything unsafe, the iThenticate routing
  decision, the conflict behaviour, the rollback condition and the baseline comparison requirement.
- Specialist files changed (source boundary only): `teaching.py`, `research.py`, `institutional.py` (constructor and
  `from_knowledge_base` take `approved_sources`; one source check each, Institutional statements only), `scoping.py`
  (approved source set, guide sources beside the unchanged handbook scope, source-aware section guard),
  `domain_cues.py` (iThenticate cue). Faculty Services & HR, claim model and navigation cues unchanged.
- Source policy (`handbook_bot/source_scopes.py`): code allow-list intersected with the registry (enabled, not internal
  use, not held, specialist listed in `domains`); system vocabulary per guide ("course banner image" is not Banner);
  source-authority, policy-boundary and procedure-completeness checks.
- Runtime (`handbook_bot/plan_f_runtime.py`): a guide-enabled registry beside the I-9 one; a question uses it only when it
  names a guide's system; `PLAN_F_REJECTED_BY_SOURCE_AUTHORITY`; guards on pass-through findings, provision slices and
  aggregation parts. Diagnostics and Developer panel: source scope, named guide sources, guide-served flag (ids only).
- iThenticate questions route to Research; SafeAssign and Turnitin course questions stay with Teaching.

### Tests performed

- Frozen 192-question set with guide scopes on: identical to I-9 and PF-2 (0 changed questions in ACTIVE and SHADOW; 55
  served, 55 correct, 0 wrong, 0 false acceptances, 11 false fallbacks, SHADOW 192 of 192, 0 model calls).
- Guide evaluation (48 PF-2 + 22 new PF-3B questions) in ACTIVE and SHADOW with scopes on, a DIRECT pass, ACTIVE with scopes
  off and LEGACY, plus two repeats: guide served 0 of 37; stop layers: ask gate 20, procedure rule 6, unrecognised ask 3,
  Institutional without an approved statement 3, aggregator 2, other 3; conflict variants, contamination both ways,
  cross-document and wrong-source cases all fall back; SHADOW 70 of 70 identical; scopes off versus on 0 differences;
  determinism 0 differences; 0 exceptions; 0 model calls. Manual review of the 24 returned guide quotes: 5 on target, 5
  partial, 14 off target.
- Layer probe: a stubbed supported guide finding is refused by the I-1 subject guard; the quota by the conflict guard; a
  single step by the procedure guard.
- Application smoke with guide scopes on: ACTIVE 12 turns, SHADOW 3, LEGACY fresh-process check (30 minimal records, no
  Plan F import); behaviour as expected, correct chips and Developer rows.
- Real-model wrong-source retrieval rerun: 16 of 16 unchanged.
- Targeted gate: 2,164 passed. Full suite: 6,770 passed, 0 failed (6,730 at the start). `pip check`: no broken
  requirements. Security scan of the changed and new files: 0 problems. Frozen-file blob check: six specialist files
  compared with `44ada15`; only the three listed above changed.

### Known limitations

- No guide answer is served (the coverage criterion of PF-3B is not met).
- The specialists' guide evidence selection prefers generic introductions that name the system (14 of 24 top quotes off
  target): relaxing the grounding rules alone would serve wrong answers.
- Five iThenticate questions are owned by Research but read as unrecognised "how" asks; "where do I find ..." questions
  are refined to Institutional Navigation, which is not approved for iThenticate.
- The Teaching and Research limitation text still says no procedural guide is indexed (diagnostics only).
- Guide-scoped questions add about 60 ms at the median and 280 ms at p95 to the Plan F attempt.

### Next task

PF-3C: a decision record (proposed D-111) on source-context binding (a guide's declared system binds the system named in
the question, "you" binds the faculty actor) in the specialists' ask gate and the I-1 subject guard, together with
procedure-aware guide evidence selection and complete-procedure answers, each with its own evaluation and a hand review of
every served answer. The project owner decides whether the PF-3B working tree is committed as a checkpoint.

### Blockers

The coverage criterion needs changes to frozen grounding and completeness rules, which the PF-3B brief does not
authorise.

### Files changed

`handbook_bot/agents/specialists/teaching.py`, `research.py`, `institutional.py`, `scoping.py`, `domain_cues.py`,
`handbook_bot/source_scopes.py`, `handbook_bot/plan_f_runtime.py`, `ui/pipeline.py`, `ui/presentation.py`,
`tests/test_multi_source.py`, `qa/post_freeze_pf3b/` (new), `docs/DECISIONS.md` (D-110), `docs/STATUS.md`.

---

## 2026-10-30 - Controlled multi-source guide ingestion and post-ingestion evaluation (post-freeze phase PF-2)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/post-freeze-knowledge-expansion` |
| Starting commit | `5213736` (Audit official guides for knowledge expansion) |
| Ending state | the knowledge base is registry-driven and multi-document; four approved official ITC guides are ingested beside the Faculty Handbook (298 indexed pages, 14,549 chunks, 861 from the guides); the Internal Use appraisal guide is on INGESTION HOLD; the iThenticate yearly-limit conflict is registered and blocked from Plan F; the frozen 192-question set reproduces I-9 exactly (0 changed questions); no Plan F answer rests on guide evidence because the frozen specialists' own source guard rejects non-handbook chunks (measured, recorded in D-109, the reason for the PF-3B recommendation); frozen specialists unchanged; local commit on top of `5213736`, not pushed by the developer tooling |
| Frozen baseline (unchanged) | 192 questions; 55 served, 137 fallback, 28.6% coverage; 55 correct, 0 limited, 0 wrong, 100% precision; 0 false acceptances, 11 false fallbacks; SHADOW mismatch 0; Plan F model calls 0; 6,674 tests at the freeze |
| Plan F switches | `PLAN_F_ENABLED`, `PLAN_F_SHADOW` and the new `PLAN_F_GUIDE_SCOPES` default to off |
| Elapsed | 6h56m08s (start 2026-10-30T13:25:28Z, end 2026-10-30T20:21:36Z; includes long background runs) |

### Completed

- Objective: make the knowledge base safely multi-document, ingest the PF-1
  P0 pilot under the provenance, language and conflict rules, and measure
  the result against the frozen I-9 baseline without changing any frozen
  specialist.
- Pilot documents: Blackboard Ultra Faculty Short Guide (11 pages), Grade
  Transfer Dashboard Guide (English pages 1-3 of 6), Blackboard Ultra Guide:
  Gradebook (10 pages), iThenticate Guidelines (English page 1 of 3), copied
  to `data/guides/` with stable names and registered with their sha256.
  Performance Appraisal Phase 2 User Guide: INGESTION HOLD - INTERNAL USE
  REVIEW REQUIRED (running header "Internal Use" on every page); registry
  record disabled, file not committed, not loaded, not served.
- Multi-source registry (`knowledge/sources.json` schema 2,
  `handbook_bot/sources.py`): new optional record fields (language,
  document type, authority level, publication date, department, system,
  citation label, access, enabled, sha256, page filter, domains,
  conflicts); a one-record registry of the frozen runtime still loads;
  disabled records are listed, never loaded, and need no file; validation
  checks every enabled file and its registered hash.
- Loader (`handbook_bot/knowledge_base.py`): enabled records in registry
  order, the handbook first and chunked exactly as before (13,688 chunks,
  ids and texts identical to a one-document build); every further
  document keeps its own page numbers, is loaded on its own with
  source-named errors, and continues the global chunk numbering (the
  frozen section guard requires `chunk_id == index`); `kb.pages` stays
  the handbook's pages for the legacy extractors; `kb.sources`,
  `kb.source_pages`, `kb.corpus_key` and per-source stats are new;
  `build_knowledge_base(pdf_path=...)` still builds one document.
- Cache identity: `cache/corpus_<16 hex>_v13_*`, keyed by the ordered
  enabled sources (id, file sha256, page filter, chunking, section map
  name) and the cache format; no path takes part. `CACHE_VERSION` v13.
- Chunk metadata: `source_page`, `citation_label`, `document_type`,
  `authority_level`, `system`, `department`, `language`, `content_type`,
  `source_conflict` on every chunk; guide chunks carry no handbook section
  (chapter and section fields null), nothing fabricated.
- Language control: deterministic `english_only` page filter (Arabic-script
  share of letters above 0.30 drops a page, above 0.50 drops a row);
  grade-transfer pages 4-6 and iThenticate page 2 are not indexed; no guide
  chunk contains Arabic script. No Arabic support was added.
- Procedure-aware chunking (`handbook_bot/chunking.py`, system manuals
  only): `procedure_step` (one numbered, lettered or bulleted step with its
  continuation, prefixed by its procedure heading) and `procedure_window`
  (three consecutive steps); 109 and 46 chunks; the paragraph, row and
  row-window chunks are unchanged and the handbook gets none.
- Source conflict guard: the iThenticate yearly limit (English 30, Arabic
  15) is a registered unresolved conflict; eight chunks carry
  `source_conflict`; the adapter rejects any finding resting on one
  (`PLAN_F_REJECTED_BY_SOURCE_CONFLICT`, pass-through and aggregation
  paths). Nothing is merged or chosen.
- Source-scope seam (`handbook_bot/source_scopes.py`,
  `RetrievalScope.also_source_ids`): the domain-to-source policy is read
  from each record's `domains` (Teaching: handbook, short guide, gradebook,
  grade transfer; Research: handbook, iThenticate; Faculty Services:
  handbook while the appraisal guide is held; Institutional: handbook,
  short guide, grade transfer) and applied in `build_specialist_registry`
  when `PLAN_F_GUIDE_SCOPES` is on; independently of the switch the
  registry is refused when a guide's page numbers fall inside a
  specialist's handbook page ranges (none for the pilot).
- Citations (`ui/presentation.py`): one document chip per cited document,
  each followed by its own page chips; a one-document answer renders as
  before; the Developer panel gains "Source ids"; the diagnostics gain
  `legacy_sources` and `plan_f_sources` (ids only).
- Legacy path: it now retrieves from the four guides as well; this is the
  intended, documented change. No handbook-only answer of the master set
  changed (pages, answers and outcomes identical).
- Decision record D-109 appended to `docs/DECISIONS.md`.

### Tests performed

- Frozen 192-question master set rerun on the multi-source corpus and
  compared field by field with the I-9 run (served path, outcome, routing
  reason, handoff outcome, specialist calls, path, findings reason, answer,
  pages): 0 changes in ACTIVE, 0 in SHADOW; SHADOW identical to the direct
  legacy call 192 of 192; determinism stable; LEGACY fresh-process check: 30
  minimal records, no Plan F import, no runtime cache.
- Guide evaluation (48 labelled questions: 37 guide including 2
  cross-document, 1 ambiguous, 4 wrong-source or held-document controls, 6
  handbook baseline and regression) in ACTIVE, SHADOW, ACTIVE with the seam
  on, LEGACY and two repeats: guide questions served by Plan F 0 of 37
  (fallback 37, coverage 0%, correct 0, limited 0, wrong 0, precision not
  applicable); the 4 served handbook questions are identical to I-9; the
  seam changes 0 of 48 outcomes; legacy retrieval puts the expected
  document first for 36 of 37 and the expected page in the top five for
  36 of 37; manual review of the top-five evidence: 33 correct, 3 partial
  (posting step, two one-sided cross-document asks), 1 miss (title length
  limit), 1 conflict, 2 wrong-source risks (password reset, direct Banner
  entry), 2 correct no-answer controls.
- Wrong-source retrieval with the real models (16 questions on shared
  terms such as approval, submission, review, attendance, support, login,
  and system-versus-policy pairs): expected document first 16 of 16,
  expected page in the top five 16 of 16.
- Application smoke on the real chat page with Developer mode: ACTIVE 11
  turns, SHADOW 4, LEGACY 3; guide answers cite "document title, Page N";
  the held appraisal question refuses with no chip; the handbook teaching
  load is served by Plan F in ACTIVE only; the known HR and annual-leave
  regressions are not served.
- Performance: cold corpus embedding 330-334 s on CPU (models 21 s,
  assembly 1.4 s); cached start 23.2 s (15.5 s with a warm model cache)
  against 22.9 s at I-9; embedding matrix 14,549 x 768 (44.7 MB), cache 65.4
  MB; ACTIVE medians on the master set: served single 601 ms (I-9 660),
  aggregate 1,186 ms (1,307), fallback 1,355 ms (1,487), direct legacy 772
  ms (834); no material regression.
- Targeted gate (the I-8 list plus the multi-source and retrieval-scope
  modules): 1,641 passed. Full suite: 6,730 passed, 0 failed (6,674 at
  the freeze plus 56 new multi-source tests). `pip check`: no broken
  requirements. Security scan of the changed and new files: 0 problems.

### Known limitations

- The frozen specialists reject every non-handbook chunk in their own
  files and the section guard has no source filter, so Plan F serves no
  guide answer; the seam is in place but changes nothing served.
- The legacy path can cite the conflicted iThenticate rows; with a model
  client configured it would need the same conflict guard.
- Five iThenticate questions reach no specialist (no Coordinator cue).
- Cross-document legacy evidence is one-sided for the two cross-document
  asks; the aggregator correctly refuses them in ACTIVE.
- An approved document with 14 or more pages would disable Plan F through
  the page-collision refusal until the section guard filters by source.
- Guide page chips show the page's printed heading; on a guide's first
  page it repeats the document title (cosmetic).

### Next task

PF-3B: a decision record to reopen the four specialists' source guard in
favour of the registry domain policy and to give the section guard a
source filter; add Coordinator cues for iThenticate; turn
`PLAN_F_GUIDE_SCOPES` on, rerun `qa/post_freeze_pf2/` and the 192-question
comparison, review every served guide answer. Only then expand to the
remaining approved guides (PF-3A). Pending outside the code: Human
Resources on the appraisal guide, ITC on the iThenticate limit.

### Blockers

None for the code. The appraisal guide and the iThenticate limit wait for
the owning units.

### Files changed

`knowledge/sources.json`, `data/guides/` (four PDFs),
`handbook_bot/sources.py`, `handbook_bot/knowledge_base.py`,
`handbook_bot/chunking.py`, `handbook_bot/retrieval.py`,
`handbook_bot/source_scopes.py` (new), `handbook_bot/plan_f_runtime.py`,
`handbook_bot/config.py`, `ui/presentation.py`, `ui/pipeline.py`,
`tests/test_multi_source.py` (new), `tests/test_sources.py`,
`tests/test_teaching_specialist.py`, `qa/plan_f_i9/make_manifest.py`,
`qa/post_freeze_pf2/` (new), `docs/DECISIONS.md` (D-109),
`docs/STATUS.md`.

---

## 2026-10-30 - Official guide inventory, knowledge-gap mapping and ingestion readiness (post-freeze phase PF-1)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/post-freeze-knowledge-expansion` (created and pushed by the developer from `main` after the I-9 merge) |
| Starting commit | `bec494f` (Merge frozen Plan F runtime integration) |
| Ending state | the 25 candidate guides of the Tutorials folder (21 PDF, 4 DOCX, plus a RAR container not inventoried) are inventoried, measured, classified and mapped; 12 approved official and relevant candidates (91 pages), 9 held for review, 4 rejected or duplicate; ingestion pipeline READY WITH STRUCTURAL CHANGES; PF-2 pilot of five P0 documents selected; no document ingested, no runtime code changed, the frozen baseline `runtime-freeze-i9` and its metrics unchanged; local commit on top of `bec494f`, not pushed by the developer tooling |
| Frozen baseline (unchanged) | 192 questions; 55 served, 137 fallback, 28.6% coverage; 55 correct, 0 limited, 0 wrong, 100% precision; 0 false acceptances, 11 false fallbacks; 6,674 tests |
| Elapsed | 0h17m49s (start 2026-10-30T12:51:58Z, end 2026-10-30T13:09:47Z) |

### Completed

- Purpose: the controlled preparation before any new document enters the
  production knowledge base: what exists, how many pages, what is official
  and relevant, duplicates and versions, extraction quality, specialist
  ownership, pipeline support, metadata, conflicts with the Faculty
  Handbook, priority, and the exact PF-2 plan.
- Documents found: 25 in the `Tutorials` folder of the Senior Project
  (milestone 4) directory (recorded by base name, no absolute path): 21
  PDF and 4 DOCX, 225 measured PDF pages and 7 declared DOCX pages; a
  `Tutorials.rar` sibling (26.1 MB) is a container no local tool can open
  and was not inventoried. The Faculty Handbook stays at 272 pages
  (verified against the production cache); the potential corpus is 363
  pages with every approved document and 504 with every candidate.
- Classification: APPROVED 12 (Blackboard faculty short guide, gradebook,
  assignments and tests, communication tools, Collaborate instructor
  guide, attendance tool, grade categories, Grade Transfer Dashboard,
  iThenticate guidelines, Performance Appraisal Phase 2 PDF, MyUOS
  announcement guide, Microsoft Forms guide); HOLD FOR REVIEW 9 (Akindi
  guide, Business Mission guide and DocuSign note without an issuing unit;
  a 2026 Blackboard compilation from vendor release notes; the McGraw Hill
  Connect sync and Coursera learner guides as vendor material; the Arabic
  Turnitin instructor guide; two student guides); REJECT 4 (the appraisal
  DOCX duplicate of the PDF, the Coursera administrator guide, the Connect
  student exam sheet, the Arabic Turnitin student guide misnamed EN). No
  exact duplicate by hash; no version history; dates from file properties
  or stated editions, otherwise DATE UNKNOWN.
- Extraction (PyMuPDF for PDF as production reads it; standard-library
  DOCX text; no OCR run): GOOD 8, GOOD WITH MINOR CLEANUP 7, NEEDS
  STRUCTURED CLEANUP 6 (bilingual pages), OCR or alternative extraction
  needed in part 2 (screenshot-dominated vendor guides), BLOCKED for the
  English pipeline 2 (Arabic only). Languages: 17 English, 6 bilingual, 2
  Arabic; Arabic text extracts reversed or with broken ligatures and the
  embedder and reranker are English models, so Arabic pages are a
  readiness observation only. Screenshots: the UOS guides write their
  steps out; one Akindi step and the vendor guides depend on images.
- Domain mapping to the four frozen specialists (no fifth): Teaching 16,
  Institutional Navigation 5, Faculty Services & HR 3, Research 1 as
  primary; five multi-domain documents. Taxonomy: Blackboard, grade
  transfer, originality tools, assessment tools, MyUOS portal, HR portal,
  productivity and IT services, third-party platforms.
- Handbook overlap: eight new step-by-step workflows, three new
  procedures, six new systems, new ITC service-desk contact and e-learning
  URL; nothing already covered at procedure level. Conflicts: C1 the
  page-224 "forwarded automatically to Banner" sentence versus the 2024
  Grade Transfer Dashboard with Chair and Dean approval (LIKELY UPDATED,
  serve as policy plus procedure); C2 the iThenticate quota of 30 (English
  page) versus 15 (Arabic page) in one guide (UNRESOLVED, raise with ITC);
  C4 the website menu path to MyUOS (LIKELY UPDATED); C5 the 2026 vendor
  compilation's layout (DIFFERENT CONTEXT, held); C3, C6, C7, C8 none.
  Precedence proposal: policy source for entitlements and approvals,
  newest official system manual for interface steps and system contacts,
  vendor material never for UOS policy or procedure, official-versus-
  official conflicts reported not merged.
- I-8 false-fallback mapping: 0 of the 11 are data gaps the guides fill
  (verifier entity binding, routing, specialist partial support, numeral
  reading, the Institutional lexicon, all on handbook content); the guides
  address other recorded fallbacks (grade entry and transfer procedure,
  content upload, attendance tool, service-desk contact) and add the
  appraisal workflow. No coverage improvement is claimed.
- Readiness: pipeline READY WITH STRUCTURAL CHANGES (first-PDF loader,
  single-file cache key, one-source section map, no DOCX, no page-language
  filter, one document chip, and the frozen specialists' retrieval scope
  pinned to the handbook's source id, so new sources reach the legacy path
  only until a scope-extension seam in non-frozen code or a decision
  record exists). Chunking: add `procedure_step` and `procedure_window`
  and a page-language filter; no FAQ or navigation-path type needed.
  Metadata additions, retrieval collision risks ("grade", "attendance",
  "announcement", "approval", "login", "submission", "review", "mission",
  "support", "Banner"), cross-document cases for the I-5 aggregator and
  the citation rule (`canonical title · Page n`, one chip per source) are
  recorded in `qa/post_freeze_pf1/gap_mapping.json`.
- Priority: P0 five (Blackboard faculty short guide, Grade Transfer
  Dashboard, gradebook, Performance Appraisal Phase 2, iThenticate), P1
  five, P2 two, HOLD thirteen. PF-2 pilot: the five P0 documents, 43
  English pages, covering all four specialists; plan and required changes
  in `qa/post_freeze_pf1/pilot_plan.md`; 34 draft PF-2 questions in
  `pf2_question_draft.json` (not run).
- Storage: MIXED (commit the approved official guides PF-2 ingests under
  `data/` with registry records; keep vendor, student, Arabic-only,
  unverified DOCX files and the archive outside Git). Security and
  privacy: PASS (no credential, key, student record or personal staff data
  in any text; author names in file properties only, not copied; the
  appraisal guide is marked Internal Use, to confirm before serving).

### Verification

- Focused tests (no runtime code changed): `tests/test_sources.py`,
  `tests/test_retrieval_scope.py`, `tests/test_planf_imports.py`,
  `tests/test_compat.py`, `tests/test_clause_retrieval_intent.py`: 207
  passed, 0 failed, 0 skipped (22 s).
- Frozen specialists: content-identical to the freeze commit `44ada15` (git
  blob hashes equal at the freeze commit, at HEAD and for the LF-normalised
  working copy, 6 of 6). The checkout of this branch re-wrote
  `navigation_cues.py` and `institutional.py` with CRLF line endings, so a
  raw byte hash of those two files differs from the value recorded at
  freeze time; later checks compare git blobs or LF-normalised content.
- Artifacts are reproducible (`inventory_scan.py`, `build_inventory.py`),
  carry no absolute path, no personal name from document metadata and no
  secret; no document text is committed beyond short samples in the scan
  output (not committed).
- Elapsed phase runtime: 0h17m49s (start 2026-10-30T12:51:58Z, end 2026-10-30T13:09:47Z).

### Files

- `qa/post_freeze_pf1/` (new): `README.md`, `inventory_scan.py`,
  `classification.json`, `build_inventory.py`, `document_inventory.json`,
  `document_inventory.csv`, `gap_mapping.json`, `pf2_question_draft.json`,
  `pilot_plan.md`.
- `docs/STATUS.md` (this entry). No code and no source document changed.

### Next

- PF-2, controlled official-guide ingestion and post-ingestion evaluation:
  decide the Internal Use question and the held documents with the owner,
  build the smallest multi-document changes listed in the pilot plan,
  ingest the five pilot documents, prove the 192 frozen answers unchanged
  with `qa/plan_f_i8/compare_runs.py`, then evaluate the draft questions
  with the I-8 review discipline. Ingestion is not complete.

---

## 2026-10-30 - Plan F runtime integration freeze (runtime integration, phase I-9) - Runtime Integration I-1 through I-9 COMPLETE

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `eeb516f` |
| Ending state | RUNTIME FROZEN - READY WITH MINOR KNOWN LIMITATIONS, freeze identifier `runtime-freeze-i9` (documentation and repository metadata only): the I-8 harness re-run on the frozen tree reproduced the final I-8 results (0 differences on every compared field for all 192 questions in ACTIVE and SHADOW), the three mode contracts, the routing, handoff, aggregation, verification and adapter contracts and the I-7 interface rules verified and documented in `docs/RUNTIME_FREEZE_I9.md`, a machine-readable freeze manifest in `qa/plan_f_i9/freeze_manifest.json`, the known-limitations register and the post-freeze change policy recorded (D-108 in `docs/DECISIONS.md`); no code under `handbook_bot/` or `ui/` changed in this phase; both Plan F switches still default to off; Plan F remains CONNECTED BEHIND FLAGS; local commit on top of `eeb516f`, not pushed by the developer tooling; `main` untouched |
| Frozen metrics (I-8 final, unchanged in I-9) | 192 questions; 55 served, 137 fallback, coverage 28.6%; 55 correct, 0 limited, 0 wrong, precision 100%; 0 false acceptances, 11 false fallbacks; SHADOW mismatches 0; Plan F model calls 0 |
| Elapsed | 0h24m41s (start 2026-10-30T12:14:45Z, end 2026-10-30T12:39:26Z) |

### Completed

- Objective: prove that the integrated runtime is stable, reproducible,
  documented, safe and regression-clean, and freeze it as the Senior
  Project runtime baseline. No new capability, no architecture change, no
  acceptance increase, no reopened specialist.
- Starting state recorded: HEAD `eeb516f`, clean tree, tracking ref equal
  to HEAD (I-8 pushed by the developer), Python 3.11.9 on Windows,
  `pip check` clean, knowledge-base cache v12 present, six frozen hashes
  matched. Committed I-8 artifacts verified as the final post-fix state
  (the two first-run defects recorded as fallbacks, 55 served, no secret,
  no personal path, no raw log).
- Freeze artifacts: `docs/RUNTIME_FREEZE_I9.md` (safety statement,
  architecture summary with the three modes, frozen metrics, mode
  contracts, routing, handoff, aggregation, verification and adapter
  contracts with their complete reason-code vocabularies, interface rules,
  the I-9 verification record, the post-I-9 change policy and the
  known-limitations register L1 to L11 classified SAFETY / COVERAGE / UX /
  DATA / ENVIRONMENT); `qa/plan_f_i9/make_manifest.py` and
  `freeze_manifest.json` (SHA-256 of the integration modules, the seam and
  interface files, the six frozen specialists and their helpers, the
  evaluation set and QA artifacts, the v12 cache files; git HEAD; Python
  and package versions; test counts; no absolute path);
  `qa/plan_f_i9/concurrency_smoke.py`; D-108 in `docs/DECISIONS.md`.
- Safety statement (verbatim in the freeze document): the system does not
  guarantee perfect answers; Plan F achieved 100% precision on the 55
  answers it served in the final 192-question I-8 evaluation, with 0 wrong
  served answers; overall coverage was 28.6% and unsupported cases fell
  back.

### Verification

- I-8 reproducibility: the I-8 harness re-run on the frozen tree with the cached knowledge base (935 s class run, 0 exceptions) and compared with the committed `qa/plan_f_i8/results_2026-10-30.json` by `compare_runs.py` plus a field-level check: 0 differences in 192 ACTIVE and 192 SHADOW records on served or fallback decision, outcome, answer text, cited pages, routing code and final specialists, specialist calls, handoff outcome, aggregation outcome and coverage, findings reason, the served evidence items (chunk, page, quote), the direct attempts and the LEGACY sample; SHADOW identical to the direct legacy call 192 of 192; the frozen metrics reproduced exactly (55 served, 55 correct, 0 limited, 0 wrong, 0 false acceptances, 11 false fallbacks, 0 model calls). I-8 final and I-9 freeze: unchanged.
- Critical regression cases and determinism: all as in the final I-8 run (known HR wrong-support blocked at the I-1 gate; annual-leave partial blocked alone and in the maternity aggregate; the I-8 open-function case blocked with ENUMERATION_INCOMPLETE; the I-8 false-premise yes/no case blocked with RELATION_MISMATCH; WLAM relation mismatch blocked; consultancy provision slices, conference-travel and research-grant authority served; housing ambiguous with 0 specialist calls; the Research Institute handoff executed at depth 1 and fallen back; same-domain, cross-domain and multi-page aggregates served; the SHADOW internally accepted cases returned the legacy answer); determinism 24 questions x 4 observations, 24 stable on route, specialist count, decision, answer, pages and handoff outcome.
- Mode contracts: LEGACY PASS (fresh-process sample of 30 with the minimal
  record, no Plan F module imported, no runtime cache, 0 exceptions; app
  smoke without a Plan F row); SHADOW PASS (192 of 192 visible outputs
  identical to the direct legacy call in the rerun, 0 leaks; app smoke with
  legacy answers and chips only and the attempt in Developer mode only);
  ACTIVE PASS (served only adapter- and verifier-accepted results, every
  other case a seamless legacy fallback with its reason; chips and labels
  from the diagnostics).
- Routing freeze PASS (`plan-f-routing-1`, no score, registration-order or
  tie-break choice; ambiguous ownership falls back); handoff freeze PASS
  (`MAX_DEPTH = 1`, self-loop, visited and repeated target, unknown target,
  second hop, unresolved clause, unsupported remainder, unsafe target and
  exception all blocked, the fixtures in `tests/test_plan_f_routing.py`
  green, three real rounds executed at depth 1 and none accepted);
  aggregation freeze PASS (request order, provenance, exact quotes,
  duplicate collapse, conflict rejection, all-or-nothing completeness,
  child verification, no generative rewrite); verification freeze PASS
  (I-4 guarantees plus the I-8 open-function and yes/no rules); adapter
  freeze PASS (known HR gate and annual-leave protection intact; the
  false-premise yes/no case and the "for new faculty" case stay blocked).
- Concurrency and cache: `qa/plan_f_i9/concurrency_smoke.py`, eight threads and sixteen calls on a fresh runtime state: the registry built exactly once (as the sequential reference), every concurrent answer equal to its sequential answer, sixteen diagnostics records with no foreign pages, 0 exceptions, 0 model calls; the existing `test_cache_is_safe_under_concurrent_first_use` and the `RuntimeCache` suite green in the gate; v12 knowledge-base cache reused at every start (23 s), registry built once per process and reused on every later request, never built in LEGACY.
- App smoke: ACTIVE with Developer mode, twelve questions (greeting, Teaching, Research, HR and Institutional singles, a two-part multi-page aggregate, the routing-refined conference-travel answer, the known HR fallback, the annual-leave fallback, the housing ambiguity with 0 specialist calls, the Research Institute handoff at depth 1, an off-topic refusal): no crash, no stale bubble, chips and panels from the diagnostics, registry built once and reused eleven times, 0 exceptions, 12 log records without question or answer text, process memory 1,293 MB; SHADOW two questions: legacy answers and chips only, the attempt in Developer mode only; LEGACY three questions: no Plan F row, section, chip or log record. PASS.
- Performance snapshot: median / p90 / max, I-9 rerun vs I-8 final: knowledge base 23.0 s (22.9 s); cold first ACTIVE request 2,069 ms with an 851 ms registry build (1,755 ms, 697 ms); served single 626 / 826 / 996 ms (660 / 744 / 852); served aggregate 1,179 / 1,305 / 1,326 ms (1,307 / 1,408 / 1,456); routing rejection attempt 1 / 2 / 2 ms (same); legacy fallback end to end 1,397 / 2,426 / 3,285 ms (1,487 / 2,560 / 3,489); total ACTIVE 1,151 / 2,336 / 3,285 ms (1,239 / 2,516 / 3,489); direct legacy 760 / 1,657 / 1,758 ms (834 / 1,860 / 2,048). Ordinary variance, no material regression.
- Diagnostics hygiene: 486 seam records of the rerun and 17 app records: 0 raw questions, 0 raw answers, 0 tracebacks, 0 filesystem paths, 0 credentials.
- Automated: targeted gate (22 files) 1,534 collected, 1,534 passed, 0 failed, 0 skipped, 0 xfail (91 s); full suite 6,674 collected, 6,674 passed, 0 failed, 0 skipped, 0 xfail (220 s; the I-8 baseline 6,674); `pip check`
  clean. Frozen specialist hashes unchanged (6 of 6) before and after.
  Security scan of the freeze change set and QA artifacts: no key, token,
  password, `.env` content, personal path, model cache, embeddings, large
  generated file, screenshot, AI attribution, co-author trailer or
  generated-by notice.
- Elapsed phase runtime: 0h24m41s (start 2026-10-30T12:14:45Z, end 2026-10-30T12:39:26Z).

### Files

- `docs/RUNTIME_FREEZE_I9.md` (new): the freeze document.
- `docs/DECISIONS.md` (modified): D-108.
- `qa/plan_f_i9/` (new): `make_manifest.py`, `freeze_manifest.json`,
  `concurrency_smoke.py`, `README.md`.
- `docs/STATUS.md` (this entry). No code changed.

### Next

- Post-freeze review (user-controlled): decide selective improvements
  against the baseline, additional official-guide ingestion, final polish,
  and the eventual merge to `main`. Frozen behaviour is not changed
  silently; see the change policy in the freeze document.

---

## 2026-10-30 - Full end-to-end adversarial application QA of the integrated runtime (runtime integration, phase I-8)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `8d6280f` |
| Ending state | the integrated runtime was evaluated as a product on the real knowledge base through the real seam and the real chat page: a 192-question master set (40 adversarial by category, 98 by tag, 75 regression) in ACTIVE, DIRECT and SHADOW, a determinism subset four times, a LEGACY sample in a fresh process, a 21-turn app session in ACTIVE plus SHADOW and LEGACY app checks; two defects found and fixed in the I-4 verification layer (an open "what does HR do for ..." ask served with one service; a false-premise yes/no ask served with the wrong category's value sentence); after the fixes 55 of 192 served, all 55 correct on manual evidence review, 0 false acceptances, 0 wrong supported answers, SHADOW identical to legacy 192 of 192, handoff depth 1 with no cycle, 0 Plan F model calls, the four frozen specialists unchanged (hashes verified); readiness for I-9: READY WITH MINOR KNOWN LIMITATIONS; Plan F remains CONNECTED BEHIND FLAGS; local commit on top of `8d6280f`, not pushed by the developer tooling |
| Plan F switches | `PLAN_F_ENABLED` and `PLAN_F_SHADOW` still default to off |
| Elapsed | 1h16m53s (start 2026-10-30T10:40:19Z, end 2026-10-30T11:57:12Z) |

### Completed

- Objective: decide from evidence whether the integrated runtime
  (Coordinator, routing refinement, the four frozen specialists, retrieval,
  reranking, guards, findings verifier, aggregator, bounded handoffs,
  legacy Verifier, fallback, source presentation, the Developer panel, the
  three modes, diagnostics, security, performance) is correct, safe,
  stable and presentable enough to freeze in I-9.
- Master evaluation set (`qa/plan_f_i8/question_set.py`): 192 unique
  real-handbook questions, each with domain, test category, tags, the
  behaviour a correct runtime may show (serve 58, fallback required 58,
  either 76), the handbook fact and page, and the specialist owners a
  correct routing may select. Teaching 33, Research 39, HR 40,
  Institutional Navigation 40, multipart and cross-domain 30, no domain
  10; direct, numeric, conditional, authority, contact and location,
  enumeration, same-domain multipart, cross-domain, ambiguous, unsupported,
  general, paraphrase (10 groups), typo and fragment (13 imperfect inputs)
  and adversarial (subject and entity binding, numeric values, page
  mismatch, negation, conditions, partial answers, enumeration
  completeness, conflicting clauses, misleading evidence, routing
  ambiguity, handoff, unsupported synthesis, general HR, cross-domain
  confusion, duplicated facts, wrong contact, near-match terminology,
  injection, comparison).
- Harness (`master_eval.py`, `legacy_check.py`, `review_served.py`,
  `analyze.py`, `compare_runs.py`): every question through
  `ui.pipeline.ask_with_diagnostics` in ACTIVE with per-item citation
  checks (quote in chunk, chunk page is the cited page, quote in the page
  text), the decision, routing and adapter attempt on the same runtime,
  every question in SHADOW compared field by field with the direct legacy
  call, 24 questions repeated three more times, the LEGACY sample in
  process and in a fresh process; the review sheet of every served answer
  and the manual verdicts in `review.json`; the metrics in `metrics.json`.
- Defects found and fixed (I-4 verification layer,
  `handbook_bot/plan_f_verification.py`; no frozen file touched):
  (D1, MAJOR) "What does HR do for new faculty?" was served with the one
  Employee Handbook sentence because the function-ask pattern required the
  clause to end right after "do"; the pattern now accepts a trailing
  for/to/with/in/on/about/regarding/during phrase, the ask is enumerative
  and falls back with ENUMERATION_INCOMPLETE. (D2, CRITICAL) "Is the
  teaching load for lecturers 12 credit hours per semester?" was served
  with the Lecturer Researcher and senior-lecturer sentences that carry
  the proposed value, a wrong supported answer, because a yes/no ask whose
  action cannot be read had no relation check; such an ask is now answered
  only by a statement that itself grants, forbids, entitles or requires,
  never by a plain value statement. Four regression tests added
  (`tests/test_plan_f_verification.py`). The whole evaluation was rerun on
  the fixed tree: exactly those two questions changed (both now
  fallbacks), everything else byte-identical.
- Open items recorded for post-I-9 (all safe fallbacks, coverage not
  safety): three-clause aggregates whose evidence spans four pages are
  rejected by the unchanged legacy Verifier's three-page citation cap (2
  questions); the evidence section label is the page's dominant section
  (page right); an approval clause settled in single-clause form stays
  ambiguous inside a two-clause question (2); the College of Graduate
  Studies and Research directory row is not found by Institutional
  Navigation (frozen lexicon); WLAM profile loads for named categories
  fail the entity check because the category is named in the preceding
  sentence (3).

### Verification

- Final run (real knowledge base from the v12 cache in 22.9 s, no model
  client, 935 s for all passes): 192 questions, Plan F served 55, fallback
  137, coverage 28.6%; served answers reviewed one by one against the
  handbook page: 55 CORRECT, 0 LIMITED, 0 WRONG, precision 100%; false
  acceptances 0 (2 in the first run, both fixed); false fallbacks 11 in
  five patterns; exceptions 0. Per group (questions / served / correct):
  teaching 24 / 8 / 8, research 29 / 13 / 13, HR 25 / 9 / 9, institutional
  25 / 13 / 13, cross-domain 22 / 11 / 11, adversarial and unsupported
  67 / 1 / 1 (98 adversarially tagged questions: 9 served, all correct).
  Citations: 67 evidence items, 0 defects, page list equals evidence pages
  55 of 55.
- Routing: codes unchanged 142, refined by request kind 22, clauses scoped
  15, ambiguous 12, none 1; classification CORRECT 160, ACCEPTABLE 7,
  AMBIGUOUS-SAFE-FALLBACK 20, WRONG 5 (all domain-cue misses that reached
  the legacy path, none served). Aggregation: 27 attempts, 11 served (all
  correct, parts in request order, no duplicate), rejections 9 missing
  clause, 5 child finding rejected, 2 legacy Verifier page cap. Handoffs:
  6 rounds attempted, 3 executed (Research to Institutional Navigation,
  depth 1), 0 accepted, 1 blocked by the visited-target protection, no
  second hop, no cycle.
- Modes: SHADOW 192 of 192 visible outputs identical to the direct legacy
  call, 0 leaks, 0 exceptions; LEGACY sample of 30 in a fresh process with
  the minimal diagnostics record, no Plan F module imported, no runtime
  cache, 0 exceptions. Determinism: 24 questions x 4 observations, 24
  stable. Known cases: "What does HR do?" blocked (I-1 gate), annual leave
  blocked (partial), WLAM relation mismatch blocked, housing ambiguous,
  Research Institute handoff executed and fallen back, consultancy
  provision slices and the research-grant and conference-travel authority
  cases served and correct.
- Performance (median / p90 / max): served single answer 660 / 744 / 852
  ms, served aggregate 1,307 / 1,408 / 1,456 ms, routing rejection attempt
  1 / 2 / 2 ms, Plan F attempt before a fallback 646 / 776 / 2,085 ms,
  legacy fallback end to end 1,487 / 2,560 / 3,489 ms, all ACTIVE
  responses 1,239 / 2,516 / 3,489 ms, direct legacy 834 / 1,860 / 2,048
  ms; cold first ACTIVE request 1,755 ms including the 697 ms registry
  build, registry reused on every later request; specialist calls per
  attempt 0 for 25, 1 for 154, 2 for 12, 3 for 1; Plan F model calls 0.
- App (built-in browser, the final tree): ACTIVE 21 sequential turns with
  Developer mode on (greeting, single findings, routing-refined answer,
  three-part aggregate, known HR fallback, ambiguous routing, the handoff
  round, refusal, annual leave, two-page aggregate, typo, adversarial
  maternity ask, the two fixed cases shown as fallbacks with their new
  reasons): one panel per answer, every earlier panel keeping its own
  served path and pages, no duplicate, no stale loading bubble, loading
  text observed with role status and aria-live polite, process memory
  1,314 MB after 17 turns and 1,327 MB after 21, 375 px without horizontal
  overflow; SHADOW 3 questions with Developer mode off: legacy answers and
  chips only, no Plan F text, the internal attempt visible in the panel
  only; LEGACY 4 questions: Mode LEGACY, Served path Legacy, no Plan F,
  routing, verification, aggregation or handoff row, no Plan F record in
  the launcher log. Diagnostics hygiene: 486 seam records and 24 app
  records without question or answer text.
- Automated: targeted gate (22 files) 1,534 collected, 1,534 passed, 0
  failed, 0 skipped, 0 xfail (85 s); full suite 6,674 passed, 0 failed, 0
  skipped, 0 xfail (219 s; baseline 6,670 plus 4 new tests); `pip check`
  clean. Frozen specialist hashes unchanged (6 of 6) before and after the
  fixes. Security scan of the change set and the run logs: no key, token,
  `.env` content, local path or traceback (the only pattern hits are the
  pre-existing redaction fixtures of the verification tests).
- Elapsed phase runtime: 1h16m53s (start 2026-10-30T10:40:19Z, end 2026-10-30T11:57:12Z).

### Files

- `handbook_bot/plan_f_verification.py` (modified): the function-ask
  pattern accepts a trailing prepositional phrase; a yes/no ask with no
  readable action is answered only by a permission, prohibition,
  entitlement or requirement statement.
- `tests/test_plan_f_verification.py` (modified): two classification rows
  and two real-knowledge-base regression tests for the two fixes.
- `qa/plan_f_i8/` (new): `README.md`, `question_set.py`, `master_eval.py`,
  `legacy_check.py`, `review_served.py`, `review.json`, `analyze.py`,
  `compare_runs.py`, `results_2026-10-30.json`, `metrics.json`.
- `docs/STATUS.md` (this entry).

### Next

- I-9, final runtime freeze: decide the default mode, freeze the runtime
  and the documentation, and record the open coverage items of this phase
  for selective reopening afterwards. Nothing in this phase claims the
  freeze.

---

## 2026-10-30 - Interface compatibility for the Plan F runtime (runtime integration, phase I-7)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `8331475` |
| Ending state | the chat page presents every answer kind the pipeline returns (legacy extractor, legacy model path, refusal, verified single finding, verified aggregate, routing-refined answer, handoff round, legacy fallback after a Plan F attempt, shadow run) from the per-turn diagnostics of `ui.pipeline.ask_with_diagnostics`, through a pure helper module; users see the answer, the handbook and page-with-section chips and a domain chip only for a verified specialist answer; Developer mode shows Runtime, Routing, Verification, Aggregation, Handoff, Evidence, Performance, LLM and Knowledge-base sections with N/A for anything unrecorded; no exception text reaches the page; both Plan F switches still default to off; nothing under `handbook_bot/` changed (frozen hashes verified); Plan F remains CONNECTED BEHIND FLAGS; local commit on top of `8331475`, to be pushed by the user; I-8 not started |

### Completed

- Objective: make the existing NiceGUI interface state accurately what
  produced each answer and what it cost, without the interface deciding
  anything itself. The pipeline's diagnostics record is the only source of
  truth; the interface adds no architecture logic and imports no Plan F
  module (`ui/chat.py` still contains none of the runtime module names,
  the Coordinator's name or the switch names).
- Presentation helpers: `ui/presentation.py` (new, pure functions, no
  NiceGUI import). `answer_kind`, `served_label`, `path_label` and
  `domain_labels` read the served path, the Plan F path, the routing and
  handoff outcome and the legacy result's own route; `source_chips` builds
  one chip per distinct cited page with that page's section from the
  evidence ("Page 54 · Contact Information"), the handbook title once in
  front, an OCR marker, nothing for a refusal that cites nothing, and never
  a placeholder section; `performance_rows` prints recorded timings only
  (a Plan F answer: specialist execution, routing refinement, findings
  verification, aggregation, handoff round, legacy Verifier, Plan F total,
  with retrieval and rerank N/A because the specialists do not time them
  separately; a fallback: the Plan F attempt apart from the legacy stages;
  every turn: the end-to-end wall time measured around the whole call and
  the registry build or reuse); `dev_sections` assembles the Developer
  panel and leaves out sections that do not apply; `escape_identifiers`
  keeps underscores inside identifiers out of Markdown emphasis;
  `assistant_message` gives each turn its own copy of its diagnostics;
  `error_diagnostics` keeps the exception class only.
- Chat page: `_call_pipeline` calls `ask_with_diagnostics` and measures the
  wall time; each assistant message stores its result, diagnostics and wall
  time; the Developer panel renders the sections of that message; the
  loading bubble reads "Searching official university sources…" (`role`
  status, `aria-live` polite); a pipeline exception yields a fixed safe
  reply and a knowledge-base load failure a message with the exception
  class only (the traceback goes to the server log); the message box, the
  send button, the theme button and the Developer switch carry `aria-label`s.
  LEGACY mode renders exactly the legacy result (no Plan F row, chip or
  import at work); SHADOW renders the legacy answer and chips with the
  shadow attempt visible in Developer mode only.
- Styles: chip variants for the handbook and the verified domain, the
  loading text, and narrow-screen rules (540 px and below: stacked Developer
  rows, full-width chips, the user-bubble width cap moved from the
  inline-block bubble to its flex wrapper, which fixed short messages
  wrapping mid-word at 820 px and below).
- Defects closed (log with before, fix and verification in
  `qa/plan_f_i7/README.md`): Plan F answers labelled "Deterministic
  extractor" (MAJOR); invented "0.0 ms" retrieval and rerank and
  evidence-count "candidates" for Plan F answers (MAJOR); fallback timing
  without the Plan F attempt or the wall time (MAJOR); exception text in the
  user's reply and the status banner (MAJOR); duplicate-looking page chips
  (MINOR); Markdown eating the underscores of `GROQ_API_KEY` (MINOR);
  dots-only loading state (MINOR); unlabelled controls (MINOR); mid-word
  wrapping of short user messages on narrow screens (MINOR); a handbook chip
  on a refusal (MINOR); clipped Developer values on narrow screens
  (COSMETIC); repeated section names in the Evidence row (COSMETIC). Left as
  is and recorded: the handbook's own list glyphs inside verbatim aggregate
  parts (the interface does not alter answer text).

### Verification

- Browser matrix (`app.py` launched locally with the switch in the process
  environment only, no model key, the real cached knowledge base, Developer
  mode on): LEGACY greeting, HR telephone (extractor), annual leave (no-key
  reply with intact underscores), off-topic refusal; SHADOW HR telephone
  (accepted internally, legacy answer shown), teaching load plus sick leave
  (aggregate accepted internally), "What does HR do?" (rejected
  internally); ACTIVE HR telephone (single finding, "Verified · Institutional
  Navigation"), teaching load plus sick leave (aggregate, three parts, three
  page chips), "Who approves conference travel?" (routing refined), "What
  does HR do?" and annual leave (legacy fallback with the Plan F attempt
  time shown apart), faculty housing (ambiguous routing, 0 specialist
  calls), HR telephone plus Passport Unit fax (multi-page aggregate), "Where
  is the Research Institute?" (a real executed handoff Research to
  Institutional Navigation at depth 1, rendered in the Handoff section,
  served by legacy fallback), off-topic refusal without chips. No exception,
  no developer vocabulary outside the panel, each panel bound to its own
  turn after later fallbacks, the loading text observed in the DOM, the
  launcher's logs free of question and answer text. Responsive review at the
  desktop width in both themes and at 375 x 812 px. Plan F model calls in
  every run: 0. Visible answers in SHADOW identical to LEGACY for the shared
  question (HR telephone) and in every case the legacy result's own text and
  pages.
- Backend regression: the I-6 real-knowledge-base evaluation (126 questions in SHADOW and 126 in ACTIVE) re-run on this code and compared record by record with `qa/plan_f_i6/results_2026-10-30.json`: 252 of 252 records equal in served path, outcome, answer text, pages, routing reason, handoff outcome and specialist calls; 56 ACTIVE records (55 unique questions) served by Plan F, every answer byte-identical to I-6; SHADOW 126 of 126 identical to the direct legacy call; the known HR wrong-support case still `PLAN_F_UNSAFE_RESULT`, the ambiguous housing question still `PLAN_F_REJECTED_BY_ROUTING` with 0 calls; model calls 0, exceptions 0 (output kept outside the repository).
- Automated: `tests/test_ui_presentation.py` (25 tests, labels,
  chips, escaping, N/A timing rules, Developer sections for every served
  path, per-turn isolation, source constraints). Targeted gate (the I-6 gate
  plus the new file, 20 files): 1,416 collected, 1,416 passed, 0 failed, 0 skipped, 0 xfail (82 s). Full suite: 6,670 collected, 6,670 passed, 0 failed, 0 skipped, 0 xfail (221 s)
  (baseline 6,645 before this phase). `pip check`: no broken requirements. Frozen
  specialist hashes unchanged (6 of 6); no file under `handbook_bot/`
  changed. Security scan of the change set: no key, token, `.env` content or
  local path.
- Elapsed phase runtime: 0h41m19s (start 2026-10-30T08:12:01Z, end 2026-10-30T08:53:20Z).

### Files

New: `ui/presentation.py`, `tests/test_ui_presentation.py`,
`qa/plan_f_i7/README.md`. Changed: `ui/chat.py`, `ui/styles.py`,
`docs/STATUS.md` (this entry). No frozen file changed; nothing under
`handbook_bot/` changed.

### Next

I-8: full end-to-end adversarial application QA on the integrated system
(both switches still default to off; Plan F stays behind flags).

---

## 2026-10-30 - Bounded Plan F handoffs and routing refinement (runtime integration, phase I-6)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `8b2cc6e` |
| Ending state | a deterministic routing refinement now settles one owner per request clause before any specialist runs, and a specialist's handoff request is executed once, at depth one, under visited-specialist and second-hop protection; every result still passes the I-1 guards, the I-4 verifier and the I-5 all-or-nothing aggregation; both Plan F switches still default to off; the four frozen specialists, the dispatcher and the legacy core unchanged (hashes verified; the Coordinator gained one additive read-only helper); Plan F is CONNECTED BEHIND FLAGS and verified, not released; local commit on top of `8b2cc6e`, pushed by the user; I-7 not started |

### Completed

- Objective: resolve requests that could not complete because the
  Coordinator assigned a clause to several specialists (I-5 Category C) or
  a specialist declined a clause to another specialist (Category D), with
  one deterministic routing-refinement step before dispatch and ONE bounded
  handoff round after it. Not an agent conversation: zero model calls, no
  query invention, no cycles, no second hop. Coverage was not a target.
- Routing refinement: `handbook_bot/plan_f_routing.py` (new). `refine(decision)`
  returns the decision to execute (the original object when nothing changed)
  and an immutable `RoutingResult` (`RoutingCode`: `ROUTE_UNCHANGED`,
  `ROUTE_CLAUSES_SCOPED`, `ROUTE_REFINED_BY_REQUEST_KIND`, `ROUTE_AMBIGUOUS`,
  `ROUTE_NO_OWNER`, `ROUTE_ERROR`; owners per clause before and after, the
  rules applied, dropped and scoped tasks). Ownership policy, derived from
  the specialists' own boundaries and the Coordinator's rules: (1) a contact
  or location clause belongs to Institutional Navigation alone (the
  Coordinator's channel rule; Teaching, Research and HR hand such clauses to
  Institutional Navigation; Institutional Navigation keeps them "whoever's
  unit it names"); (2) an authority or responsibility clause about a subject
  ("who approves / handles / is responsible for X") belongs to the one subject
  specialist whose cues claim it (Institutional Navigation's own boundary
  hands every such ask to the subject owner); (3) a "which office / unit ...
  handles X" clause with a subject claim stays ambiguous (the two hand it to
  each other): no specialist runs; (4) two subject specialists claiming one
  clause stay ambiguous: nothing is chosen by score, registration order or
  chance; (5) a single specialist that received a multipart question as one
  clause gets one focus clause per request clause, assigned by the
  Coordinator's own cue rule (`coordinator.cue_scores`, one additive public
  helper); the clause texts are the Coordinator's split, unchanged. Request
  kinds come from the I-4 request reader (`read_request`), so domain and
  requested relation decide together. An unresolved clause rejects the
  attempt before dispatch (`PLAN_F_REJECTED_BY_ROUTING`, no specialist call).
- Handoff round: `handbook_bot/plan_f_handoff.py` (new). `run_handoffs`
  validates every request of the dispatched runs before any execution and
  runs each valid target exactly once (`MAX_DEPTH` = 1). A request is valid
  only when: the target is one of the four specialists and registered; it is
  not the source; it has not executed for this request (first-round runs and
  earlier targets: no A to B to A, no repeated target); the request carries
  the specialist's own handoff task (the target's id, the source as requester,
  exactly one focus clause, passed to the target unchanged); that clause is a
  request clause the source owned and declined (report outcome `handed_off`;
  the whole of a multipart question, a clause the source answered itself or
  text outside the request bind to nothing); and every first-round result
  the request still depends on is supported (otherwise no target runs). A
  target that asks for a handoff is the blocked second hop; a target that
  raises or reports a model call rejects the round. Codes (`HandoffCode`):
  `HANDOFF_NONE`, `HANDOFF_EXECUTED`, `HANDOFF_TARGET_INVALID`,
  `HANDOFF_SELF_LOOP`, `HANDOFF_TARGET_ALREADY_VISITED`,
  `HANDOFF_CLAUSE_UNRESOLVED`, `HANDOFF_REMAINDER_UNSUPPORTED`,
  `HANDOFF_SECOND_HOP_BLOCKED`, `HANDOFF_UNSAFE_RESULT`, `HANDOFF_EXCEPTION`;
  after aggregation the adapter reports `HANDOFF_ACCEPTED`,
  `HANDOFF_TARGET_REJECTED` or `HANDOFF_CONFLICT`. The settled execution the
  aggregator reads: the source task keeps the clauses it did not hand off (or
  is dropped), the handoff task joins the decision with a validated stamp,
  the executed request leaves the source's pending list, the target's run
  joins the dispatch result. The aggregator (`plan_f_aggregation`) accepts a
  settled handoff task only with that stamp and the recorded round, skips a
  source task's handed-off clauses, and still answers `HANDOFF_REQUIRED` for
  a dispatch with a pending request; the dispatcher never executes handoffs.
- Seam: `plan_f_runtime.PlanFRuntime.attempt` refines the decision right
  after the Coordinator (a `routing` trace entry only when something changed,
  so the pass-through trace of phases I-1 to I-4 is unchanged); a specialist
  result with handoff requests, on either path, goes to `_handoff_round` and
  then to the aggregator (`_finish_aggregate`). New outcomes
  `PLAN_F_REJECTED_BY_ROUTING` and `PLAN_F_REJECTED_BY_HANDOFF`;
  `AdapterResult.routing`, `handoff`, `handoff_outcome`, `specialist_calls`;
  timings `routing_ms`, `handoff_ms`, `handoff_specialist_ms`. `ui/pipeline.py`:
  diagnostics gain `routing_refined`, `routing_reason`, `routing_initial`,
  `routing_final`, `routing_ms`, `handoff_attempted`, `handoff_source`,
  `handoff_target`, `handoff_depth`, `handoff_visited`, `handoff_outcome`,
  `handoff_ms`, `specialist_calls` (codes, ids and numbers only). LEGACY is
  untouched and imports nothing of Plan F; SHADOW still returns the legacy
  answer unchanged; ACTIVE serves only what the aggregate or the single
  verification record accepted.
- HR as a named entity in contact and location asks (section 22 of the
  brief): decision A, routing. A contact or location clause names its office;
  Institutional Navigation owns the ask kind, and HR hands such a clause to
  Institutional Navigation anyway. Routing it directly saves the HR call and
  the handoff round for the same answer; HR still runs when the clause is an
  HR policy, entitlement or benefit ask. The frozen specialists are unchanged.
- Same-domain multipart questions (section 23): the Coordinator module's
  decomposition is unchanged; the refinement applies the Coordinator's own
  clause split and cue rule to the single specialist's task, so the frozen
  specialist processes each request clause separately and reports per
  clause. No split happens that the Coordinator does not already make.
- Real-knowledge-base run (`qa/plan_f_i6/`, 2026-10-30; v12 cache, 272 pages,
  13,688 chunks; 62 I-3 questions, 14 I-4 probes, the 26-question I-5
  aggregation set and the 24-question routing and handoff set: 117 unique
  questions, 126 runs per mode): every one of the 35 answers the I-5 ACTIVE
  run served is served again with byte-identical text and pages (28 single
  findings, the 5 aggregates, the 2 provision slices); the two same-fact
  aggregates now come through the clause-scoped path with the shared finding
  rendered once. Known cases unchanged: "What does Human Resources do?" and
  "What does HR do?" stop at the I-1 gate (general ask), "How many days of
  annual leave do faculty members get?" at `status partial`; the mixed
  maternity-and-annual-leave question falls back at the HR run's partial
  status. Over the 126 runs the refinement left 85 decisions unchanged,
  settled 17 by request kind, scoped 19 single-specialist multipart tasks and
  rejected 5 as ambiguous before any specialist ran; 12 attempts made no
  specialist call, 101 one, 13 two, none three.
- Category C re-evaluation (the 8 unique I-5 cross-domain or shared-routing
  cases, 11 runs): "Who approves conference travel?" and "Who approves the
  funding ceiling for research grants?" route to Research alone
  (`ROUTE_REFINED_BY_REQUEST_KIND`, 1 specialist call each, Institutional
  Navigation never run) and are served (pages 117 and 165); "How many theses
  can a faculty member supervise at once and what is the phone number of the
  Human Resources Department?" routes the contact clause to Institutional
  Navigation alone (HR dropped) and is served as a two-specialist aggregate
  (pages 81, 54). "Who approves research leave?" and "Who approves Hajj
  leave?" route to Research and HR respectively, which return `partial`
  (1 call each): fallback. "Who handles faculty visas and where is that
  office?" routes clause 1 to HR and clause 2 to Institutional Navigation;
  both return nothing usable (not found, partial; 2 calls): fallback. "Which
  office handles faculty housing?" and "Can I reduce my teaching load if I
  have a funded research project, and who approves it?" stay
  `ROUTE_AMBIGUOUS` (an office-selection ask with a subject claim; a clause
  claimed by Research and Teaching) and run no specialist. Newly resolved: 3
  of 8 unique cases; the other 5 fall back for a named reason, with fewer or
  equal specialist calls than in I-5.
- Category D re-evaluation (the I-5 handoff case, 2 runs): "Where is the
  Research Institute and what research funding is available?" is now
  clause-scoped to Research; Research declines the location clause to
  Institutional Navigation with a well-formed handoff task and returns
  `not_found` for the funding clause, so the round stops at
  `HANDOFF_REMAINDER_UNSUPPORTED` without running the target (1 call):
  fallback. Two real handoffs did execute in the run: "Who do I contact
  about research grants?" (Research declines to Institutional Navigation,
  which returns `out_of_scope`) and "Where is the Research Institute?"
  (Research declines to Institutional Navigation, which returns `not_found`);
  both complete the round at depth one and fall back on the target's result
  (2 calls each). No real handoff completed a request.
- Twenty answers newly served, every one reviewed against the production
  chunk and the printed page (`review_accepted.py --new-only`): CORRECT 20,
  ACCEPTABLE BUT LIMITED 0, WRONG 0. By cause: routing by request kind, 3
  single findings ("Who approves conference travel?", page 117, the last rule
  of the conference-travel list; "Who approves the funding ceiling for
  research grants?", page 165; "Who approves the petty cash advance for a
  principal investigator?", page 166, the sentence stating approval by the
  VCRGS) and 3 aggregates (theses plus HR telephone, pages 81 and 54; entry
  visas plus the Passport Unit telephone, pages 61 and 53; theses plus
  conference-travel approval, pages 81 and 117); clause scoping of a single
  specialist's multipart task, 9 aggregates (credit hours plus office hours,
  pages 96, 91, 79; travel plus employment offers, page 61; theses plus
  petty cash, pages 81 and 166; HR telephone plus Passport Unit fax, pages 54
  and 53; funding ceiling plus the IRB, pages 165 and 265; lecturers' load
  plus the 20% absence rule, pages 96 and 228; travel plus entry visas, page
  61; colleges plus the Finance telephone, pages 24 and 54; HR telephone plus
  HR fax, the one directory row rendered once, page 54); and 5 questions of
  the new set that the I-5 machinery already handled (the HR fax row; four
  cross-domain aggregates: office hours plus petty cash, lecturers' load plus
  the Registration telephone, sick leave plus the funding ceiling, the 20%
  absence rule plus maternity leave). Every part is a verbatim quote with its
  own page; no part was omitted, no connective text was added, every
  specialist assignment matches the clause's domain and relation.
- Acceptance (ACTIVE served by Plan F, unique questions): I-3/I-4 corpus 21
  of 62 (34%) in I-5 to 22 of 62 (35%); I-4 probes 9 to 10 of 14; I-5
  aggregation set 6 to 16 of 26; routing and handoff set 8 of 24; all 117
  unique questions: 55 served, 62 fallback. No coverage target was pursued;
  the 62 fallbacks are ambiguous or two-subject clauses, partial, not-found or
  general specialist results, declined clauses no specialist supports, and
  questions no specialist owns.
- Latency (adapter timings, real knowledge base): routing refinement median
  0.3 ms (0.2 to 0.5 ms) when a decision changed, below 0.05 ms otherwise; a
  pre-dispatch routing rejection costs 1.0 ms for the whole attempt (0.4 to
  1.3 ms) with no specialist call; the handoff round's own work 0.1 ms (0.0
  to 0.2 ms); the two executed targets 0.1 to 0.2 ms each (Institutional
  Navigation declining or finding nothing for a clause reads its lexicon and
  runs no retrieval). First-round specialists median 617 ms (up to 1,946 ms
  for two specialists); I-4 verification 0.6 ms; aggregation 1.8 ms (1.2 to
  4.2 ms, 22 aggregates). Total Plan F attempt when served median 644 ms;
  ACTIVE wall served 644 ms, legacy fallback 1,410 ms (0.1 to 3,223 ms, two
  specialists plus the legacy path at the top). SHADOW: 117 of 117 unique
  questions identical to the direct legacy call, 0 exceptions. Plan F model
  calls: 0 on every attempt, dispatch and diagnostic record. Diagnostics
  hygiene: 0 of 252 records carry the question or the answer text.
- Controlled application smoke test: `app.py` launched locally with the
  switches in the process environment only (never in `.env`) and the
  pipeline's debug diagnostics logged by the launcher (no code change),
  driven through the browser at localhost with Developer mode on. ACTIVE:
  "Who approves conference travel?" rendered the page-117 rule with chips
  Page 117 and Participation in Conferences (676 ms, registry built once in
  742 ms; logged `ROUTE_REFINED_BY_REQUEST_KIND`, research and institutional
  to research, 1 specialist call); "How many theses can a faculty member
  supervise at once and what is the phone number of the Human Resources
  Department?" rendered two paragraphs with chips Page 54, Page 81 and
  Responsibility to Research (678 ms; routing dropped HR, aggregate accepted,
  2 calls); "What is the phone number of the Human Resources Department and
  what is the fax number of the Passport Unit?" rendered the two directory
  rows with chips Page 53, Page 54 and Contact Information (3.5 ms;
  `ROUTE_CLAUSES_SCOPED`, 1 call); "Which office handles faculty housing?"
  was served by legacy (the configured no-key reply) after
  `PLAN_F_REJECTED_BY_ROUTING` with 0 specialist calls. The developer panel
  stayed usable on every answer; registry reused after the first question;
  no raw exception, no crash; the logged diagnostics carry the routing and
  handoff keys and no question or answer text. SHADOW (a second launch): the
  same two served questions showed the legacy no-key reply (Pages 117, 119;
  Page 81) while the log recorded Plan F accepting them internally with the
  same routing records (691 and 672 ms, registry built once in 716 ms); no
  exception. No UI change was needed; the I-7 observations stand.
- Tests: `tests/test_plan_f_routing.py` (new, 38 tests): codes and bounds;
  unchanged decisions returned as the same object with the pass-through trace
  intact; party asks about a subject routed to the subject specialist and
  served without running Institutional Navigation; contact and location
  clauses routed to Institutional Navigation; the office-selection and the
  two-subject asks ambiguous with no specialist run; same-domain multipart
  tasks clause-scoped and served per clause, or nothing when a clause is not
  found; the Coordinator's split followed exactly (no false split, no missed
  split, conditions kept with their clause, request order); a clause with only
  foreign cues without owner; a raising refinement redacted; determinism and
  text-free records. Handoffs: a valid request executed once and verified
  (trace, record, items, settled dispatch); the target rejected by the
  verifier; a handoff completing a cross-domain aggregate only when the target
  answers; a conflict with the target's evidence; a target already executed;
  a repeated target from two sources; a second hop; a cycle back to the
  source; the validation edges (no task, two clauses, wrong requester, self
  loop, unregistered target, text outside the request, a clause the source
  answered, the whole of a two-clause question); the whole-question binding of
  a one-clause request; an unsettled handoff task never aggregated; an
  unsupported remainder blocking the round without a call; a raising or
  model-reporting target redacted; the I-1 bare request never executed;
  determinism; SHADOW diagnostics without touching the answer; ACTIVE serving
  routed and handed-off answers and falling back on blocked rounds; LEGACY
  untouched; the frozen Research specialist resolving a Category C case on
  the test corpus. Updated: `tests/test_plan_f_runtime.py` (the bare handoff
  stub is `PLAN_F_REJECTED_BY_HANDOFF`, never executed; the four complex
  questions carry the routing or handoff code), `tests/test_plan_f_aggregation.py`
  (the shared-report mechanism tested through direct aggregator calls; the
  adapter path now clause-scopes; the bare handoff case), `tests/test_planf_imports.py`
  (the two new modules).

### Tests performed

Targeted gate (19 files: routing and handoff, aggregation, verifier, adapter,
pipeline, imports, four-specialist dispatch, Coordinator, contracts,
registry, legacy Verifier, compatibility, orchestrator, the frozen
specialists' integration, support-gate and acceptance suites): 1,391
collected, 1,391 passed, 0 failed, 0 skipped, 0 xfail (81 s).
Full suite: 6,645 collected, 6,645 passed, 0 failed, 0 skipped, 0 xfail
(215.9 s): the I-5 baseline of 6,604 plus the 39 new routing and handoff tests
and two new import-budget cases.
`pip check`: no broken requirements. Frozen specialist hashes unchanged
before and after (Teaching, Research, Faculty Services & HR and its claim
model, Institutional and its cues; D-103 and D-107 values); the dispatcher,
the legacy core and the shared specialist modules unchanged; the Coordinator
changed by the one additive helper `cue_scores` only.
- Actual phase runtime, recorded locally from the first command of the phase to
  the local commit: 0h 51m 52s. Major measured components: two full
  real-knowledge-base evaluations of 126 runs in three modes each (527 s the
  recorded one, about the same for the first run before the shared-coverage
  projection), the targeted gate 81 s, the full suite 216 s, the four
  application launches for the smoke tests about 70 s each, the knowledge-base
  build 22.5 s per process; the rest is design, implementation, the tests and
  the review.

### Known issues at handoff

- A "which office / unit handles X" ask with a subject claim stays
  ambiguous by design (the specialists hand it to each other); the
  Institutional Navigation specialist answers responsible-office asks only
  from its own statements. A two-subject clause ("reduce my teaching load if
  I have a funded research project") also stays ambiguous.
- No real handoff completed a request in the recorded corpus: the frozen
  specialists decline the clauses whose owner the routing refinement now
  selects directly, and the two declines that did execute reached a target
  that holds no evidence for the clause. The round is exercised end to end,
  acceptance included, by the synthetic fixtures.
- The legacy model path remains REAL MODEL EVALUATION BLOCKED - NO AVAILABLE
  CONFIGURED CLIENT (no key configured); I-6 does not depend on it.
- UI observations stay on the I-7 list: Plan F labelled "Deterministic
  extractor", retrieval and rerank shown as 0.0 ms for Plan F answers,
  candidate counts, fallback timing, duplicate-looking page chips, the
  Markdown underscore rendering. No UI change was needed in I-6.

### Do not redo

- Do not choose between two valid owners by score or order; ambiguous stays
  fallback.
- Do not raise the handoff depth or follow a target's handoff.
- Do not relax the aggregator's stamp check for handoff tasks.

### Next task

I-7: UI compatibility and interface improvement, after approval.

### Files changed

`handbook_bot/plan_f_routing.py` (new), `handbook_bot/plan_f_handoff.py` (new),
`handbook_bot/plan_f_aggregation.py` (settled handoff tasks, handed-off
clauses), `handbook_bot/plan_f_runtime.py` (routing before dispatch, the
handoff round, new outcomes and fields), `handbook_bot/agents/coordinator.py`
(`cue_scores`), `ui/pipeline.py` (diagnostics keys), `tests/test_plan_f_routing.py`
(new), `tests/test_plan_f_runtime.py`, `tests/test_plan_f_aggregation.py`,
`tests/test_planf_imports.py` (expectations), `qa/plan_f_i6/` (new:
`README.md`, `routing_eval.py`, `analyze.py`, `review_accepted.py`,
`results_2026-10-30.json`), `docs/STATUS.md` (this entry). No frozen
specialist, dispatcher, legacy core or shared specialist module changed.

---

## 2026-10-30 - Deterministic Plan F aggregation (runtime integration, phase I-5)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `0b179f7` |
| Ending state | a deterministic multi-finding aggregation layer now decides every decision with several request clauses or several specialists; the single-clause pass-through path of phases I-1 to I-4 is unchanged except that a quote rejected as excess evidence may be served as its provision slice; both Plan F switches still default to off; the four frozen specialists, the Coordinator, the dispatcher and the legacy core unchanged (hashes verified); Plan F is CONNECTED BEHIND FLAGS and verified, not released; local commit on top of `0b179f7`, pushed by the user; I-6 not started |

### Completed

- Objective: allow Plan F to serve an answer that needs MORE THAN ONE
  independently verified finding when the findings the current execution
  already produced cover every clause of the request completely; never
  free-form synthesis, never a model call, never a handoff execution, never
  a routing change, never a partial answer served as a whole. Acceptance
  rate was not a target.
- Aggregator: `handbook_bot/plan_f_aggregation.py` (new).
  `aggregate(decision, dispatched, chunks, metadata)` returns an immutable
  `Aggregation` (an `AggregationResult` plus the ordered parts to render):
  code, clause and coverage counts, per-clause coverage with its own
  findings-aware verification record, ordered evidence references (finding,
  task, specialist, source, page, chunk, the clauses it covers, the original
  of a slice), duplicates, conflicts, specialists, pages, completeness,
  `safe_to_render`; `to_dict` is JSON-serialisable, deterministic (no wall
  time) and free of question and answer text. Codes (`AggregationCode`, 11):
  `AGGREGATED_ACCEPTED`, `NO_AGGREGATION_REQUIRED`, `MULTIPART_UNSUPPORTED`,
  `UNRESOLVED_CLAUSE`, `CROSS_DOMAIN_UNRESOLVED`, `HANDOFF_REQUIRED`,
  `UNSAFE_RESULT`, `MISSING_REQUIRED_CLAUSE`, `CHILD_FINDING_REJECTED` (the
  child's I-4 code in the detail), `VALUE_CONFLICT`, `AGGREGATION_ERROR`.
  It never raises: an exception becomes `AGGREGATION_ERROR` with the
  exception category only. Standard library only; zero model calls.
- Clause coverage (structured, never by word similarity): the request
  clauses are the Coordinator's own split (`metadata["analysis"]["clauses"]`,
  in order). A clause is owned by the task whose `focus_clauses` names it
  exactly (a cross-domain decision); a clause owned by two tasks is
  `CROSS_DOMAIN_UNRESOLVED` and one owned by none `UNRESOLVED_CLAUSE`, both
  knowable from the decision alone, so no specialist runs for them
  (`request_problem`). When one specialist received the whole question as its
  one clause (a same-domain multipart decision), its single report is shared
  by every request clause and tried one finding at a time; the eligible
  findings are then verified together, so the primary-first and value-conflict
  rules of I-4 still apply. Run-level safety first, on every executed run:
  `supported` status, no handoff, no `missing` entry, no model call, a clause
  report list, no unprocessed clause (`plan_f_verification.run_code`). Then
  per clause the complete I-4 verifier on the clause's own view
  (`clause_view`: the report and the findings whose `metadata["clauses"]`
  name it): `report_code` (outcome, ask shape, requested details, relation;
  the Institutional outcome and tier), the quote guard, the subject binding,
  requested detail, entities, population, modifiers, conditions, relation,
  provision boundary. Every clause covered or no aggregate (all or nothing).
- Conflicts and duplicates: two request clauses that read as the same ask
  (equal `RequestShape`) answered with different values (a number, or the
  bound contact value of a contact ask) are `VALUE_CONFLICT`; nothing is
  chosen by score. A quote already rendered for an earlier clause (same
  chunk, same text) is rendered once; the clause stays covered and the
  duplicate is recorded; a duplicate never covers a clause it did not verify
  for. Ordering: request clause order, then the specialist's finding order,
  then first occurrence; nothing depends on set or dict iteration.
- Rendering: the verbatim quotes of the accepted parts, one paragraph each,
  in request order; no label, no connective sentence, no rewording (the
  shape every Plan F answer has had since I-1). Labels made of the user's
  clause text were evaluated and not used: the legacy Verifier treats every
  line as a claim and a directory row does not cover a clause's words. The
  part-to-clause mapping travels in `items[*]["meta"]["plan_f"]["clause_index"]`
  and in `QAResult.sub_questions` (one entry per request clause with its
  task's route). Every rendered part keeps its own page: the legacy Verifier
  must cite every evidence page, or the attempt is `PLAN_F_REJECTED_BY_VERIFIER`.
- Provision slice (the I-4 consultancy case, section 24 of the brief):
  `plan_f_verification.provision_slice` derives, from a quote the verifier
  rejected as `EXCESS_EVIDENCE`, the exact substring that holds the answering
  unit's own provision (every sentence of that list item, numbered item or
  section passage the quote contains) and the lead-in sentences immediately
  before it; sentences of OTHER provisions before or after are left out.
  The derived finding keeps source, page, chunk and metadata and records
  `derived_from`, the `slice` offsets and the provision; its id is the
  original id plus `/slice`. `verify_findings_sliced` serves the slice only
  when it passes every I-4 check on its own, the quote guard included;
  otherwise the original rejection stands. It applies on both paths. The
  consultancy question is now served as the 16.1.7 sentence alone.
- Seam: `plan_f_runtime.PlanFRuntime.attempt` now branches after the
  Coordinator: one request clause and one task take the unchanged
  pass-through path (`_single_clause`); several clauses or several
  specialists take `_aggregate` (every task dispatched once, nothing
  re-routed; the aggregator; adaptation of the parts; the unchanged legacy
  Verifier; the `QAResult`). New outcome `PLAN_F_REJECTED_BY_AGGREGATOR`;
  `AdapterResult.aggregation`, `AdapterResult.path`
  (`single_finding_passthrough` / `multi_finding_aggregate`), `verified` now
  requires the aggregate AND every clause's own record on the aggregation
  path; trace entries `aggregator` (shape, then accepted or rejected with the
  record); timings `aggregation_ms` and its parts. `ui/pipeline.py`: ACTIVE
  serves an aggregate only when `_verified` holds for the aggregate and every
  clause; diagnostics gain `plan_f_path` and twelve `aggregation_*` keys
  (counts, codes, pages, specialists, timing; never text); the `findings_*`
  keys of an aggregate summarise the clause records. LEGACY is untouched and
  still imports nothing of Plan F; SHADOW still returns the legacy answer
  unchanged.
- Unsupported shapes (always fallback): a clause shared by two specialists,
  a specialist that answers no request clause, an uncovered clause, a
  handoff request anywhere, a partial, not found, out of scope or error run,
  an unprocessed clause, a general or enumerative clause, a rejected child
  finding, a value conflict, a model call, an unexecuted task. Handoff
  requests are preserved in the dispatch result and never executed (I-6).
- Enumeration (section 23): investigated, not enabled. The HR case "What
  does HR provide to new faculty members?" has no bounded list in the source
  (page 61 lists pre-joining formalities under 2.2.3.8 and visa, travel and
  accommodation steps under 2.2.3.9; the specialist returns one bullet and a
  recruitment bullet from page 57). The frozen specialists return findings
  by relevance, not a list's items, and carry no list-boundary metadata, so
  completeness cannot be proven from the findings alone; proving it from
  chunk text the specialist did not select would break the provenance rule.
  `ENUMERATION_INCOMPLETE` stands, inside aggregates too.
- Real-knowledge-base run (`qa/plan_f_i5/`, 2026-10-30; v12 cache, 272 pages,
  13,688 chunks, knowledge base in 22.2 s; 62 I-3 questions, 14 I-4 probes and
  the 26-question aggregation set, 96 unique questions, 102 runs per mode):
  all 28 answers the I-4 ACTIVE run served (20 corpus, 8 probes) are served
  again with byte-identical answer text and pages, every one through the
  single-finding pass-through path (`plan_f_path` = `single_finding_passthrough`,
  the aggregator never consulted). The three former limited cases: WLAM still
  `RELATION_MISMATCH`, HR provision still `ENUMERATION_INCOMPLETE`, the
  consultancy question now served as the 16.1.7 sentence (page 260, item
  `task-1-f1/slice` derived from `task-1-f1`, slice offsets 117 to 314 of the
  original quote); the probe "What percentage of their time can faculty spend
  on consultancy?" is served the same way. Known cases unchanged: "What does
  Human Resources do?" and "What does HR do?" stop at the I-1 gate (general
  ask), "How many days of annual leave do faculty members get?" at `status
  partial`; the multipart visa, housing, reduced-load and Research Institute
  questions fall back with `CROSS_DOMAIN_UNRESOLVED` (no specialist run) or
  `HANDOFF_REQUIRED` (the handoff preserved, not executed).
- Five real aggregates served in ACTIVE mode, each reviewed against the
  production chunk and the printed page: "How many credit hours is a full
  teaching load and how many days of sick leave with full pay are faculty
  entitled to?" (Teaching + HR; pages 96, 91, 107; the two workload quotes of
  clause 1 both state 12 credit hours per semester, the sick-leave sentence
  answers clause 2); "How many credit hours is a full teaching load and how
  many theses can a faculty member supervise at once?" (Teaching + Research;
  pages 96, 91, 81); "Does HR assist new faculty members in obtaining entry
  visas, and how many colleges does the University of Sharjah have?" (HR +
  Institutional stated fact; pages 61, 24); "How many office hours must
  faculty hold, and how many office hours per week must faculty hold?" and
  "How many days of sick leave with full pay are faculty entitled to, and how
  many days of sick leave are faculty entitled to?" (one specialist, the
  whole question as its clause, one verified sentence covering both clauses,
  rendered once with the duplicate recorded; pages 79 and 107). Every part is
  a verbatim quote with its own page; `sub_questions` carries the request
  clauses; the legacy Verifier cited every evidence page.
- Acceptance (ACTIVE served by Plan F): I-3/I-4 corpus 20 of 62 (32%) in I-4
  to 21 of 62 (34%), fallback 42 to 41 (the consultancy slice); I-4 probes 8
  of 14 to 9 of 14; aggregation set 6 of 26 served (5 aggregates, 1 single
  finding); all 96 unique questions: 35 served, 61 fallback. No acceptance
  target was pursued.
- Multipart classification (every question with several request clauses or
  several specialists, 32 runs over 26 unique questions): A (findings
  available and independently verifiable, aggregated) 5; B (a requested
  clause not supported by the frozen specialist: the single specialist given
  the whole question reports the combined clause `partial`, or `not_found`
  for a contact clause naming two offices) 14; C (cross-domain, a clause
  assigned to two specialists or a specialist that would need another: no
  specialist run) 11; D (handoff required) 2; E (synthesis or general
  explanation) 0; F (unsupported) 0. Only A was activated. The same-domain
  multipart questions of the set (12) are all B except the two same-fact
  questions above: with the Coordinator unchanged, one specialist receives
  the whole multipart question as one clause and reports it partial.
- Review of every answer the ACTIVE run served (`qa/plan_f_i5/review_accepted.py`:
  the production chunk and the printed page text of every part): the 5
  aggregates CORRECT (every request clause answered; every rendered part a
  verbatim quote of the cited page, the quote an exact substring of its
  chunk; subject, relation, value and condition as asked; no omitted clause;
  no connecting text; no cross-domain contamination; every page preserved),
  0 ACCEPTABLE BUT LIMITED, 0 WRONG. In the two teaching-load aggregates the
  first clause renders two quotes (pages 96 and 91) that both state 12 credit
  hours per semester for regular ranked faculty: the I-4 rule that keeps a
  secondary finding stating the same value, applied per clause. The 2 sliced
  answers (consultancy days per week; consultancy percentage of time) are
  CORRECT: the served text is the complete 16.1.7 sentence and the removed
  text is the last sentence of 16.1.6 on liability, a different numbered
  provision. The 28 single-finding answers are byte-identical to the I-4 run
  reviewed there.
- Aggregation latency (adapter timings, real knowledge base): the five
  accepted aggregates median 2.6 ms (1.5 to 5.3 ms) in total, of which clause
  mapping 0.01 ms, per-clause verification 1.9 ms (0.9 to 4.5), conflict
  check 0.6 ms (0.5 to 0.8), deduplication 0.01 ms; the rendering plus the
  legacy Verifier 0.3 ms (0.1 to 0.6); the 16 dispatched rejections median
  0.04 ms; the 11 pre-dispatch rejections 0.5 ms for the whole attempt (no
  specialist run). Against: I-4 findings verification median 0.6 ms (0.3 to
  1.7 ms) on the pass-through path, specialist retrieval and rerank median
  783 ms (695 to 1,284 ms) for the aggregate attempts (two specialists run)
  and 679 ms over all dispatched aggregation attempts. ACTIVE wall: served
  aggregates median 760 ms (706 to 1,257 ms), all served answers 614 ms,
  legacy fallback 1,374 ms. SHADOW: 96 of 96 unique questions identical to
  the direct legacy call; 0 exceptions. Plan F model calls: 0 on every
  attempt, dispatch and diagnostic record. Diagnostics hygiene: 0 of 204
  records carry the question or the answer text.
- Controlled application smoke test: `app.py` launched locally with the
  switches in the process environment only (never in `.env`) and the
  pipeline's debug diagnostics logged by the launcher (no code change),
  driven through the browser at localhost with Developer mode on. ACTIVE:
  "How many credit hours is a full teaching load and how many days of sick
  leave with full pay are faculty entitled to?" rendered as three verbatim
  paragraphs with chips Page 91, Page 96, Page 107 and the section chip
  Workload Allocation Model (WLAM); "Does HR assist new faculty members in
  obtaining entry visas, and how many colleges does the University of Sharjah
  have?" as two paragraphs with chips Page 24, Page 61 and Recruitment and
  Hiring; "How many days of sick leave with full pay are faculty entitled to,
  and how many days of sick leave are faculty entitled to?" as one paragraph
  with chip Page 107 and Leaves and Absences; "What does HR do?" served by
  legacy (the configured no-key reply). The developer panel labels every
  Plan F answer "Deterministic extractor", shows the cited pages and best
  section and counts the considered findings and rendered parts as
  candidates and reranked chunks; totals 1,379 ms (first question, registry
  built in 678 ms), 830 ms and 754 ms. The logged diagnostics name the path
  `multi_finding_aggregate`, `AGGREGATED_ACCEPTED`, 2 of 2 clauses, the pages
  and specialists, aggregation 2.7, 1.4 and 4.6 ms, registry reused, 0 model
  calls, no error, and hold no question or answer text. No raw exception, no
  crash; no UI change was needed. SHADOW (a second launch): the same aggregate question showed the
  legacy no-key reply with its own chips (Pages 94, 96, 107) while the log
  recorded the aggregate accepted internally (`served` legacy, path
  `multi_finding_aggregate`, pages 96, 91, 107, 2.6 ms, registry built once
  in 670 ms); the HR telephone control question showed the legacy
  extractor answer while Plan F accepted its single finding internally;
  registry reused; no exception. The duplicated "Page 54" chip of that
  legacy answer is a pre-existing legacy-path observation for the I-7 list.
- Tests: `tests/test_plan_f_aggregation.py` (new, 50 tests): the contract
  (codes, immutable, serialisable, text-free, deterministic, no model or
  network import), decision reading (clauses, ownership, pre-dispatch
  problems), two specialists covering two clauses, deterministic output
  across runs and across direct calls, request order over task order,
  multi-page and same-page evidence, a shared report mapped by verification,
  one finding covering two clauses rendered once, the same quote returned
  twice, the pass-through path untouched (the aggregator never consulted),
  the enumerative clause inside a multipart request, the provision slice
  (unit, served consultancy case, slice rejected on its own merits,
  `clause_view` projection), and the negatives: missing clause, partial run,
  contradicted modifier, wrong subject, missing condition, wrong entity and
  wrong field of a contact clause, conflicting numeric values, conflicting
  contact values, a duplicate that must not create completeness, a handoff
  (target never run), a general ask, a synthesis question, wrong page, wrong
  chunk, tampered quote, a finding without its clause index, a raising
  aggregator (three injection points, redacted), a pending task, a reported
  model call, a legacy Verifier that cites one page only, reordered findings;
  pipeline SHADOW non-invasive with the aggregation diagnostics, ACTIVE
  serving the aggregate and falling back on a rejected child, `_verified`
  requiring every clause record, LEGACY untouched, and the real frozen
  specialists aggregating the cross-domain question on the test corpus.
  Updated: `tests/test_plan_f_runtime.py` (the multi-clause questions now
  carry the aggregator's code; a shared clause runs no specialist, a
  multipart clause set is dispatched once), `tests/test_plan_f_verification.py`
  (the consultancy quote: still excess for the verifier, served as its slice
  by the adapter), `tests/test_plan_f_pipeline.py` (one SHADOW outcome
  string), `tests/test_planf_imports.py` (the new module in `NEW_MODULES`).

### Tests performed

Targeted gate (18 files: aggregation, verifier, adapter, pipeline, imports,
four-specialist dispatch, Coordinator, contracts, registry, legacy Verifier,
compatibility, orchestrator, the frozen specialists' integration, support-gate
and acceptance suites): 1,350 collected, 1,350 passed, 0 failed, 0 skipped,
0 xfail (96 s).
Full suite: 6,604 collected, 6,604 passed, 0 failed, 0 skipped, 0 xfail
(204.5 s): the I-4 baseline of 6,553 plus the 50 new aggregation tests and one
new import-budget case.
`pip check`: no broken requirements. Frozen specialist hashes unchanged
before and after (Teaching, Research, Faculty Services & HR and its claim
model, Institutional and its cues; D-103 and D-107 values); the legacy core,
the Coordinator, the dispatcher and the shared specialist modules unchanged.

### Known issues at handoff

- Same-domain multipart questions remain fallback in practice: the unchanged
  Coordinator gives a single specialist the whole question as ONE clause, and
  the frozen specialist then reports the combined clause `partial` ("not
  stated together in one place") or, for a contact clause naming two
  offices, `not_found`; a partial run is never aggregated. The aggregator
  serves such a question only when one verified sentence answers every
  clause. Per-clause decomposition by the Coordinator is a Coordinator
  change, outside I-5.
- A contact clause that names Human Resources also selects the HR
  specialist, which hands it off; the shared clause and the handoff make
  such cross-domain questions fall back.
- The legacy model path remains REAL MODEL EVALUATION BLOCKED - NO AVAILABLE
  CONFIGURED CLIENT (no key configured); I-5 does not depend on it.
- UI observations from I-3 stay on the I-7 list; no UI change was needed.

### Do not redo

- Do not serve a clause subset: all or nothing is the I-5 rule.
- Do not resolve a value conflict by retrieval score, and do not execute a
  handoff to complete an aggregate (I-6).
- Do not label aggregate parts with generated text.

### Next task

I-6: handoff round and Coordinator routing integration, after approval.

### Files changed

`handbook_bot/plan_f_aggregation.py` (new), `handbook_bot/plan_f_verification.py`
(`run_code` / `report_code` split, `clause_view`, `provision_slice`,
`verify_findings_sliced`), `handbook_bot/plan_f_runtime.py` (the aggregation
path, the slice on the pass-through path, `AdapterResult.aggregation` and
`path`, the new outcome, `clause_index` on adapted items), `ui/pipeline.py`
(ACTIVE condition, diagnostics keys), `tests/test_plan_f_aggregation.py`
(new), `tests/test_plan_f_runtime.py`, `tests/test_plan_f_verification.py`,
`tests/test_plan_f_pipeline.py`, `tests/test_planf_imports.py` (expectations),
`qa/plan_f_i5/` (new: `README.md`, `aggregate_eval.py`, `analyze.py`,
`review_accepted.py`, `results_2026-10-30.json`), `docs/STATUS.md` (this
entry). No frozen specialist, Coordinator, dispatcher, legacy core or shared
specialist module changed.

---

## 2026-10-30 - Findings-aware Plan F verification (runtime integration, phase I-4)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `1a4ed1b` |
| Ending state | a findings-aware verification layer now sits between the I-1 guards and rendering; both Plan F switches still default to off; the four frozen specialists and the legacy core unchanged (hashes verified); Plan F is CONNECTED BEHIND FLAGS and verified, not released; local commit on top of `1a4ed1b`, pushed by the user; I-5 not started |

### Completed

- Objective: make the final acceptance decision aware of the STRUCTURED
  specialist finding before it becomes a user-visible `QAResult`. A result may
  be supported by a source and still be unsuitable as the direct answer to the
  user's exact request; I-4 strengthens that boundary without touching the
  frozen specialists, the Coordinator, the dispatcher or the legacy Verifier.
  Acceptance rate was not a target.
- Verifier: `handbook_bot/plan_f_verification.py` (new). `read_request(clause)`
  reads the user's clause deterministically into a `RequestShape` (request
  kind, named entities, content noun phrases with their modifiers, condition
  phrases, the counted unit of a quantity ask, the asked action, contact
  field, body noun, population). `verify_findings(clause, findings, chunks,
  metadata)` returns an immutable `FindingsVerification` (accepted, reason
  code, finding status, request kind, subject / relation / requested-detail /
  evidence-integrity verdicts, completeness class, `safe_to_render`, the ids
  of the findings that may be rendered, per-finding verdicts; `to_dict` is
  JSON-serialisable, deterministic and free of question or answer text). It
  never raises: an exception becomes `VERIFICATION_ERROR` with the exception
  category only. Zero model calls; standard library only.
- Reason-code model (`ReasonCode`, 19 codes): `ACCEPTED`, `UNSUPPORTED_STATUS`,
  `PARTIAL_FINDING`, `NOT_FOUND`, `OUT_OF_SCOPE`, `HANDOFF_REQUIRED`,
  `MULTIPART_UNSUPPORTED`, `GENERAL_ASK_MISMATCH`, `UNRECOGNISED_ASK`,
  `SUBJECT_MISMATCH`, `RELATION_MISMATCH`, `REQUESTED_DETAIL_MISSING`,
  `CONDITION_MISMATCH`, `MODIFIER_MISMATCH`, `ENUMERATION_INCOMPLETE`,
  `EXCESS_EVIDENCE`, `VALUE_CONFLICT`, `INVALID_EVIDENCE_REFERENCE`,
  `VERIFICATION_ERROR`; completeness classes `complete`, `elaborated`,
  `specialist_contract`, `partial`, `enumeration_unproven`, `excess`,
  `not_assessed`; 19 request kinds (contact, location, quantity, amount,
  percentage, duration, frequency, timing, value, definition, authority,
  responsibility, selection, eligibility, obligation, yes/no, enumeration,
  general, unrecognised).
- Checks, in order: status and structure (the I-1 `result_problem`
  conditions as codes; a consistency test pins that the two never disagree);
  per finding the I-1 quote guard (evidence integrity) and the I-1 subject
  binding, reused, plus an independent lexicon check that an Institutional
  finding's office is one the clause names; then on the ONE local evidence
  unit that answers: the requested detail of the request kind (the bound
  contact field, a location entry, a number tied to the asked unit noun, a
  money amount, a percentage, a duration, a frequency, a timing), the
  clause's named entities and its asked population or role, no contradicted
  category modifier (closed contrast sets: leave types, pay status,
  population categories, day types, hour types, examination types, visa
  types, contact fields), every condition phrase stated (per-period wording,
  idioms such as "at once" / "simultaneously"), the asked relation (a copular
  definition of the asked subject; the asked action by a named party before
  it or as its passive agent; a nominal action bound to a party; the asked
  kind of body; a requirement for the asked actor; an eligibility statement;
  the asked predicate with the asked polarity), and the provision boundary
  (a sentence of another list item or numbered section before or after the
  answering unit is `EXCESS_EVIDENCE`; same-provision sentences are
  `elaborated`). An enumerative ask (what does X provide, what are the X,
  what does X do, requirements) is `ENUMERATION_INCOMPLETE`: one finding
  cannot prove the set complete (I-5). A secondary finding must pass on its
  own and is dropped with `VALUE_CONFLICT` when it states another value for
  the asked quantity. A request kind the verifier does not read falls back
  to the specialist's contract plus the I-1 guards (`specialist_contract`).
- Trimming evaluated, not performed: a quote that spans two provisions is
  rejected, not cut to its answering unit. Cutting would be non-generative
  and quote-guard checkable, but it would change the evidence the frozen
  specialist selected and scored, break the provenance equality between the
  rendered item and the finding's `evidence_quote`, and belongs with the
  rendering decisions of I-5; rejection with fallback is the safe choice now.
- Seam: `plan_f_runtime.PlanFRuntime.attempt` calls `verify_findings` after
  the quote and subject guards and before rendering (lazy import; the
  verifier builds on the adapter's guard primitives); rejection is the new
  outcome `PLAN_F_REJECTED_BY_FINDINGS_VERIFIER`; `AdapterResult.verification`
  and `AdapterResult.verified`; trace entry `findings_verifier` with the
  verification record; `timings["findings_verify_ms"]`. `ui/pipeline.py`:
  ACTIVE serves a Plan F answer only when the adapter accepted it AND
  `attempt.verified` (an attempt without a verification record is never
  served); diagnostics gain `findings_verified`, `findings_accepted`,
  `findings_reason`, `findings_completeness`, `findings_request_kind`,
  `findings_ms` (codes only). LEGACY is untouched and still never imports
  Plan F; SHADOW still returns the legacy answer unchanged. The I-1 gates
  stay in place (defence in depth; none removed).
- I-3 23-case re-evaluation (real knowledge base, ACTIVE): 20 still served
  by Plan F, 3 now fall back, and the 3 are exactly the I-3 ACCEPTABLE BUT
  LIMITED cases: "What is the Workload Allocation Model?" ->
  `RELATION_MISMATCH` (the quote names WLAM in its heading and then states
  the legal basis; WLAM is not the subject of a defining statement);
  "How many days per week can faculty spend on consultancy?" ->
  `EXCESS_EVIDENCE` (the quote opens with the liability sentence of 16.1.6
  before the answering 16.1.7 sentence); "What does HR provide to new
  faculty members?" -> `ENUMERATION_INCOMPLETE` (an enumerative ask answered
  by one bullet of a list). All 20 previously CORRECT cases remain accepted;
  "How many office hours must faculty hold?" is narrowed to its primary
  finding (five office hours per week; the evening-course finding states
  another value and is dropped with `VALUE_CONFLICT`). Per-case table in
  `qa/plan_f_i4/` (`analyze.py` on the recorded results).
- Real-knowledge-base rerun (`qa/plan_f_i4/reverify.py`; production build
  from the v12 cache in 22.6 s, 272 pages, 13,688 chunks; registry 666 ms on
  its first construction in the process): the 62 I-3 questions plus 14
  probes through the verifier directly, through the adapter, and through
  `ui.pipeline.ask_with_diagnostics` in SHADOW and ACTIVE mode. SHADOW: 62 of
  62 identical to the direct legacy call, Plan F 0 model calls, median wall
  1,338 ms. ACTIVE: 20 served by Plan F (32%; I-3: 23, 37%), 42 by legacy
  (68%); accepted wall median 627 ms (1 to 719 ms), fallback median 1,255 ms
  (max 2,507 ms); every served answer `used_llm` False, `llm_calls` 0,
  `findings_accepted` True. Adapter outcomes on the 62: ACCEPTED 20,
  REJECTED_BY_FINDINGS_VERIFIER 3, REJECTED_BY_SUBJECT_GUARD 2, UNSAFE_RESULT
  32, NOT_APPLICABLE 5; per domain Teaching 5 of 12, Research 4 of 12, HR 5
  of 14, Institutional 6 of 14, special 0 of 10. Verifier codes on the 50
  dispatched results: ACCEPTED 20, PARTIAL_FINDING 13, GENERAL_ASK_MISMATCH 6,
  NOT_FOUND 6, SUBJECT_MISMATCH 2, RELATION_MISMATCH 1, EXCESS_EVIDENCE 1,
  ENUMERATION_INCOMPLETE 1; request kinds read: quantity 7, contact 5,
  yes/no 3, authority 2, obligation, definition, value, percentage, amount,
  selection, enumeration and timing 1 each, unrecognised 25 (all of them
  results the status gate had already stopped). Verification latency,
  measured apart from retrieval and reranking: median 0.6 ms (0.3 to
  1.2 ms) inside the adapter, 0.2 ms (0.0 to 1.5 ms) called directly,
  against a specialist retrieval-and-rerank median of 590 ms; no measurable
  change to the SHADOW or ACTIVE wall times. Probes: the mutated asks
  (calendar days, part-time faculty, visiting faculty) are already `partial`
  at the frozen specialists; the consultancy percentage ask is
  `EXCESS_EVIDENCE` like its day-count twin; "What information does HR
  provide to new faculty members?" (a selection ask) is served. Diagnostics
  hygiene: 152 records, none holds the question or the answer text. Legacy
  model path: REAL MODEL EVALUATION BLOCKED - NO AVAILABLE CONFIGURED CLIENT
  (unchanged; I-4 does not depend on it).
- Every answer served by Plan F in the ACTIVE run reviewed against its
  production chunk and the printed page (`review_accepted.py`: quote present
  in the chunk, page and source correct, subject correct, no wrong number,
  date or contact, no cross-domain contamination, no second provision): the
  20 I-3 questions and the 8 served probes are all CORRECT; ACCEPTABLE BUT
  LIMITED 0; WRONG 0. The three I-3 limited answers no longer reach the user.
- Known HR cases: "What does Human Resources do?" and "What does HR do?"
  remain blocked by the I-1 gate (general ask) before the verifier runs;
  called directly on the dispatched result the verifier classifies them
  `GENERAL_ASK_MISMATCH`. "How many days of annual leave do faculty members
  get?" remains blocked (`status partial`; verifier `PARTIAL_FINDING`); the
  maternity sentence is additionally caught by the modifier contrast
  (`MODIFIER_MISMATCH`) when forced through as supported. Multipart,
  cross-domain and synthesis questions still stop at the decision gate;
  handoff results are `HANDOFF_REQUIRED` and their handoff requests are left
  untouched for I-6.
- Tests: `tests/test_plan_f_verification.py` (new, 104 tests): the contract
  (immutable, serialisable, text-free, deterministic, no model dependency),
  request reading (33 clauses), status codes and their agreement with the
  I-1 gate (15 shapes), evidence-integrity mutations, contact and entity
  mutations (field, office, mislabelled report, labelled row, location asked
  of a row, another office's row), numeric and policy mutations (leave type,
  day type, unit noun, period condition, population, idiom), relation
  mutations (definition, party position, passive agent, body kind, actor,
  polarity, enumeration versus selection, general, unrecognised), provision
  boundary, value conflict, verifier failure safety, adapter integration
  (trace order, outcome, timings, the four I-1 examples, the known cases,
  forced misleading shapes, rendering of verified findings only) and the
  pipeline (SHADOW non-invasive, ACTIVE requires the verified record,
  diagnostics codes only, LEGACY untouched). `tests/test_plan_f_runtime.py`:
  two expectations updated (the trace now holds `findings_verifier`; office
  hours renders the primary finding only). `tests/test_planf_imports.py`:
  the new module in `NEW_MODULES`.

### Tests performed

Targeted gate (`tests/test_plan_f_verification.py`, `test_plan_f_runtime.py`,
`test_plan_f_pipeline.py`, `test_planf_imports.py`,
`test_four_specialist_dispatch.py`, `test_planf_coordinator.py`,
`test_planf_contracts.py`, `test_planf_registry.py`, `test_verifier.py`,
`test_compat.py`, `test_orchestrator.py` and the frozen specialists'
integration, support-gate and acceptance suites; 17 files): 1,299 collected,
1,299 passed, 0 failed, 0 skipped, 0 xfailed (69 s).
Full suite: 6,553 collected, 6,553 passed, 0 failed, 0 skipped, 0 xfailed
(201 s; the I-3 baseline of 6,448 plus the 104 new verification tests and
one new import-budget case).
`pip check`: no broken requirements. Frozen specialist hashes unchanged
before and after (Teaching, Research, Faculty Services & HR and its claim
model, Institutional and its cues; D-103 and D-107 values); the legacy core,
the Coordinator, the dispatcher and the shared specialist modules unchanged.

### Known issues at handoff

- The verifier is conservative by design: enumerative asks never render from
  one finding, a quote spanning two provisions is rejected rather than cut,
  and a second finding with another value is dropped. I-5 (aggregation) owns
  rendering several findings with explicit labels and gaps.
- Request reading is pattern-based: a clause outside its patterns is
  `unrecognised` and decided by the specialist's contract plus the I-1
  guards, as before I-4.
- The legacy model path remains REAL MODEL EVALUATION BLOCKED - NO AVAILABLE
  CONFIGURED CLIENT (no key configured); I-4 does not depend on it.
- UI observations from I-3 stay on the I-7 list, unchanged.

### Do not redo

- Do not raise acceptance by loosening the verifier or the I-1 gates; the
  metric that matters is wrong accepted answers = 0.
- Do not trim quotes in the adapter before the I-5 rendering decision.

### Next task

I-5: deterministic multi-finding aggregation (several supported findings,
explicit labels and gaps, no composed sentences), after approval.

### Files changed

`handbook_bot/plan_f_verification.py` (new), `handbook_bot/plan_f_runtime.py`
(seam, outcome, `verification` field), `ui/pipeline.py` (ACTIVE condition,
diagnostics keys), `tests/test_plan_f_verification.py` (new),
`tests/test_plan_f_runtime.py` (two expectations), `tests/test_planf_imports.py`
(module list), `qa/plan_f_i4/` (new: `README.md`, `reverify.py`, `analyze.py`,
`review_accepted.py`, `results_2026-10-30.json`), `docs/STATUS.md` (this
entry). No frozen specialist, Coordinator, dispatcher, legacy core or shared
specialist module changed.

---

## 2026-10-30 - Plan F verified on the real knowledge base (runtime integration, phase I-3)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `7c7efae` |
| Ending state | measurement and verification only: no production file changed, both Plan F switches still default to off, the four frozen specialists and the legacy core unchanged (hashes verified); Plan F is CONNECTED BEHIND FLAGS and verified on the real knowledge base, not released; local commit on top of `7c7efae`, pushed by the user; I-4 not started |

### Completed

- Real knowledge base through the production path (`build_knowledge_base`,
  real embedder and cross-encoder reranker, production chunking, metadata,
  source registry and section map): source `uos_faculty_handbook_2025_26`
  (registered), 272 pages, 13,688 chunks (847 paragraph, 8,412 row, 4,429
  row window), 206 section records, embeddings 13,688 x 768 float32,
  `IndexFlatIP` with 13,688 vectors, OCR off. First-time model download and
  load 32.6 s (embedder) and 22.2 s (reranker); first build 331.9 s (the
  embedding run; the v12 cache files were created, the archived v11 cache is
  not used); cached rebuild 14.8 s with identical embeddings; no error, the
  only warnings are Hugging Face notes about symlinks and unauthenticated
  downloads.
- All four specialists built from the real knowledge base with their own
  `from_knowledge_base` builders: Teaching 9 ms, Research 10 ms, Faculty
  Services & HR 8 ms, Institutional 777 ms for its first construction in a
  process (it reads the directory and the statements once); the registry
  through `build_runtime` 97 ms afterwards (88 to 117 ms across runs; 712 ms
  for the first build inside the running application). Source and section
  guards passed (shared pages 15, 11, 11 and 10; no unresolved page); no
  partial registry, no exception, no model call.
- Runtime cache on the real knowledge base: first attempt built the runtime
  (97 ms) and the second reused it; a second knowledge-base object and a
  second cached-build knowledge base each built their own runtime (97 and
  117 ms); 3 builds, 3 entries, distinct runtimes and registries.
- 62 realistic questions (Teaching 12, Research 12, HR 14, Institutional 14,
  special 10: greeting, general handbook question, unsupported, ambiguous,
  multipart, cross-domain, three-specialist, mutual decline, handoff-producing,
  out of domain) run through `ui.pipeline.ask_with_diagnostics` in LEGACY,
  SHADOW and ACTIVE mode, each compared with the direct `answer_question`
  call where applicable. Runner, analyser, review script and the recorded
  results are in `qa/plan_f_i3/`.
- LEGACY: 62 of 62 identical to the direct call in every visible field and
  in the trace; Plan F neither imported nor built (cache absent). Median
  wall 827 ms (0 to 1,997 ms): real retrieval and rerank, no model client.
- SHADOW: 62 of 62 identical to the direct call; Plan F attempted on all 62,
  accepted 23, 0 model calls; registry built once and reused 61 times.
  Median wall 1,363 ms; overhead over the legacy wall median +389 ms (range
  -285 to +792 ms, the spread being run-to-run variation of legacy retrieval).
  Plan F attempt median 569 ms (max 832): Coordinator under 1 ms, specialist
  retrieval and rerank with the real cross-encoder median 595 ms, guards and
  Verifier under 1 ms; Institutional answers in about 1 ms (no retrieval).
- Outcome distribution (one attempt per question): ACCEPTED 23 (37%);
  UNSAFE_RESULT 32 (partial 13, multipart or cross-domain or synthesis 7,
  general ask 6, not found or out of scope 6); NOT_APPLICABLE 5;
  REJECTED_BY_SUBJECT_GUARD 2; quote guard 0; Verifier 0; errors 0; registry
  unavailable 0. Fallback 39 (63%). Per domain: Teaching 6 of 12, Research 5
  of 12, HR 6 of 14, Institutional 6 of 14, special 0 of 10.
- Every accepted answer reviewed against its production chunk and the
  printed page (quote present, page and source correct, subject correct, no
  wrong number, date or contact, no cross-domain contamination): 20 CORRECT,
  3 ACCEPTABLE BUT LIMITED ("What is the Workload Allocation Model?" quotes
  its legal basis only; the consultancy quote unit opens with an unrelated
  liability sentence before the one-day-per-week rule; "What does HR provide
  to new faculty members?" quotes one of several items), 0 WRONG. With the
  real reranker the teaching-load answer is the page 91 line (12 credit
  hours per regular semester for ranked faculty) instead of the fixture-run
  page 96 line; both are correct.
- Known HR cases: "What does Human Resources do?" and "What does HR do?" are
  still returned as `supported` by the frozen HR specialist and rejected by
  the adapter as general asks; "How many days of annual leave do faculty
  members get?" is `partial`; all three served by legacy in SHADOW and ACTIVE.
- ACTIVE: 23 questions served by Plan F (`used_llm` False, `llm_calls` 0 on
  every one), 39 by legacy; accepted wall median 634 ms (1 to 741 ms),
  fallback wall median 1,314 ms (max 2,597 ms: Plan F plus legacy).
- Controlled application smoke test: `app.py` launched locally three times
  with the switches in the process environment only (never in `.env`) and
  driven through the browser at localhost. LEGACY: the knowledge base
  loaded, an extractor answer with page chips, the configured no-key reply
  for a policy question, the greeting. SHADOW: visible answers identical to
  legacy for two questions Plan F accepted internally and for the known HR
  case. ACTIVE: the Human Resources directory row (chips Page 54, Contact
  Information) and the four-theses quote (chips Page 81, Responsibility to
  Research) served by Plan F; the known HR and annual-leave questions served
  by legacy. In-application diagnostics (debug logging enabled by the
  launcher, no code change): registry built once in 712 ms then reused, 0
  Plan F model calls, no error. A second SHADOW launch with the same logging showed Plan F accepting the
  four-theses question internally (page 81, the same page the legacy answer
  cited) while the visible answer stayed the legacy reply, and the multipart
  visa question falling back; registry built once in 710 ms then reused. No raw exception, no crash.
  With Developer mode on, a Plan F answer's panel shows query type, cited
  pages and best section correctly, labels the answer "Deterministic
  extractor", shows retrieval and rerank at 0.0 ms (Plan F records its own
  timing keys) and counts candidates and reranked chunks as 1; the panel of
  an ACTIVE fallback answer shows the legacy timings only, without the Plan F
  attempt time.
- Legacy model path: REAL MODEL EVALUATION BLOCKED - NO AVAILABLE CONFIGURED
  CLIENT (no key configured; 52 of the 62 legacy answers were the no-key
  reply). Plan F made 0 model calls throughout.
- Diagnostics hygiene: 124 records inspected, none holds the question or the
  answer text; JSON serialisable; buffer capped at 200 (I-2 test); the
  pipeline writes nothing to disk.
- Tests added: none; the real-knowledge-base run revealed no missing
  integration guarantee.

### Tests performed

Full suite after the phase: 6,448 collected, 6,448 passed, 0 failed, 0 skipped,
0 xfailed (206 s); no test added, so the I-2 baseline count is unchanged.
`pip check`: no broken requirements. Frozen specialist hashes unchanged
before and after (Teaching, Research, Faculty Services & HR and its claim
model, Institutional and its cues; D-103 and D-107 values); every legacy
core file, the seam, the adapter and the UI unchanged.

### Known issues at handoff

- Acceptance is 37% by design; partial results (13) are the largest fallback
  class, then multipart and cross-domain questions (7) and general asks (6).
  These belong to I-4 (findings-aware verification) and I-5 (aggregation).
- First-question cost in a fresh process: about 0.7 s for the first
  Institutional construction, about 0.1 s for later registry builds; the
  one-time real embedding of the handbook takes about 5.5 minutes, after
  which the cache loads in about 15 s.
- ACTIVE fallback does double work (Plan F about 0.6 s plus legacy about
  0.8 s); SHADOW adds about 0.4 s per question. Performance: ACCEPTABLE FOR
  LOCAL DEMO; no project latency target exists.
- UI observations for I-7, not changed here: the configured no-key reply
  loses its underscores in Markdown ("GROQAPIKEY"); a page chip and a section
  chip can show the same "Page N" label when the section hint is a page; a
  Plan F answer is not marked as such in the interface; the developer panel calls a Plan F answer "Deterministic extractor",
  shows its retrieval and rerank as 0.0 ms and its candidate counts as 1, and
  omits the Plan F attempt time from a fallback answer's total.
- The known frozen HR defect stands (general asks answered with topical
  sentences); it stays blocked by the adapter.

### Do not redo

- Do not raise acceptance by loosening the I-1 gates or changing the
  Coordinator; measure, then design I-4 and I-5.
- Do not commit the generated model or embedding caches.

### Next task

I-4: findings-aware verification (quote-in-source and subject binding as a
Verifier mode, coverage by focus clause, conflict detection), after approval.

### Files changed

`qa/plan_f_i3/` (new: `README.md`, `runtime_measure.py`, `analyze.py`,
`review_accepted.py`, `results_2026-10-30.json`), `docs/STATUS.md` (this
entry). No production, test or documentation file other than this entry.

---

## 2026-10-30 - Feature-flagged Plan F runtime seam and shadow mode (runtime integration, phase I-2)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `b80afbd` |
| Ending state | `ui/pipeline.py` is the one approved seam between the legacy pipeline and the Plan F adapter; both switches default to off, so a start without new environment variables is the legacy chatbot exactly as before; SHADOW and ACTIVE modes exist behind the switches and are not enabled anywhere; the four frozen specialists and the legacy core modules unchanged (hashes verified); committed on top of `b80afbd` and pushed with a normal push; I-3 (real knowledge base and models) not started |

### Completed

- `handbook_bot/config.py`: `PLAN_F_SHADOW` (default False) beside the
  existing `PLAN_F_ENABLED` (default False), both through the existing
  `_env_bool` convention; the comment block now says that only
  `ui/pipeline.py` reads them. Modes: LEGACY (both off), SHADOW (`PLAN_F_SHADOW`
  on; it takes precedence when both are on), ACTIVE (`PLAN_F_ENABLED` on
  alone).
- `ui/pipeline.py`: `ask(kb, question) -> QAResult` is unchanged for the UI.
  LEGACY calls `answer_question` exactly as before and neither imports nor
  builds Plan F. SHADOW produces the legacy answer first and returns it
  unchanged, then runs the I-1 adapter for diagnostics only. ACTIVE runs the
  adapter first and returns its `QAResult` only when the outcome is
  `PLAN_F_ACCEPTED`; every other outcome, a registry failure or an exception
  returns the legacy answer. The adapter is imported lazily inside the
  flagged branch; no Plan F failure can raise through `ask`; legacy failures
  keep their existing behaviour. `ask_with_diagnostics` returns a
  JSON-serialisable diagnostic beside the result (mode, served, attempted,
  outcome, specialist, status, fallback reason, Plan F and legacy pages,
  answer equality, page overlap, Plan F timings, registry built or reused,
  Plan F and legacy LLM counts, redacted error); it never holds the question
  or the answer text. `recent_plan_f_diagnostics` keeps the last 200 records
  in memory; nothing is written to disk; `reset_plan_f_state` clears the
  cache and the records.
- `handbook_bot/plan_f_runtime.py`: `RuntimeCache`, a bounded, thread-safe
  cache that builds the `PlanFRuntime` once per knowledge-base object (keyed
  by identity, validated through a weak reference, released when the object
  is collected), remembers a failed build for that object instead of
  rebuilding on every question, and never shares a registry between two
  knowledge bases. The pipeline creates one cache on the first SHADOW or
  ACTIVE question.
- `tests/test_plan_f_pipeline.py` (new, 66 tests): the flag matrix and the
  SHADOW precedence; environment parsing through the existing convention in
  a fresh interpreter; LEGACY identity against `answer_question` for
  greeting, contact, count, date, synthesis, refusal, ambiguous and no-key
  questions (every user-visible field, the trace and the timing keys); no
  Plan F import or build in LEGACY mode (fresh interpreter); SHADOW returns
  the legacy answer unchanged for accepted and rejected Plan F outcomes,
  records the known HR and annual-leave rejections, builds the runtime once
  and reuses it, adds no model call to the user's question, and survives
  eleven Plan F failure shapes (registry unavailable, no specialist,
  rejection, partial, quote guard, subject guard, Verifier rejection,
  specialist exception, adapter exception, seam exception, import failure)
  with the legacy answer intact and every error redacted; ACTIVE returns an
  accepted answer of each specialist and falls back for eleven rejected
  shapes, never serves a result without an accepted outcome, and shares one
  cache with SHADOW; a second knowledge base gets its own runtime; the cache
  unit tests (once per object, failure remembered and isolated, raising
  builder redacted, collected objects released, bounded, concurrent first
  use builds once, non-weak-referenceable objects built but not cached); the
  diagnostics buffer; the legacy core free of Plan F. The I-1 isolation
  test now names `ui/pipeline.py` as the approved seam (one narrow change).

### Tests performed

Targeted gate (new seam tests, adapter, four-specialist dispatch, Plan F Coordinator,
registry, contracts, imports, Verifier, compatibility, orchestrator): 687 passed,
0 failed (51 s); the 66 seam tests and the 79 adapter tests among them.
Full suite after the change: 6,448 collected, 6,448 passed, 0 failed, 0 skipped,
0 xfailed (201 s): the 6,382 baseline plus the 66 seam tests.
`pip check`: no broken requirements. Frozen specialist hashes unchanged
(Teaching, Research, Faculty Services & HR with its claim model,
Institutional with its cues). Legacy core unchanged: `qa.py`,
`orchestrator.py`, `router.py`, `retrieval.py`, `synthesis.py`, `verifier.py`,
`types.py`, `knowledge_base.py`, `app.py`, `ui/chat.py`. Changed on purpose:
`ui/pipeline.py`, `config.py`, `plan_f_runtime.py` (cache only).
Controlled smoke run of `ui.pipeline.ask_with_diagnostics` in all four
switch combinations on the real-page fixture knowledge base without a
server: LEGACY served legacy with Plan F unloaded; SHADOW served legacy with
Plan F recorded (accepted on the HR telephone and theses questions, rejected
on the known HR and annual-leave cases, fallback on the multipart and
no-match cases); ACTIVE served the two accepted Plan F answers and legacy
for the rest; both switches on behaved as SHADOW; 0 Plan F model calls;
no exception.

### Known issues at handoff

- Plan F is CONNECTED BEHIND FLAGS, not released: nothing has run on the
  real embedder, reranker or model, and registry construction, latency,
  accepted and fallback rates on real retrieval are unmeasured (I-3).
- Fixture timings only: registry build about 0.7 s, adapter 1 to 260 ms per
  question with the fake embedder.
- A failed registry build is remembered per knowledge-base object for the
  life of the process; a rebuilt knowledge base is a new object and builds
  afresh.
- The I-1 limits stand: general asks, partial results, handoffs and
  multipart questions all fall back to the legacy answer.

### Do not redo

- Do not set `PLAN_F_ENABLED` or `PLAN_F_SHADOW` in a committed file or
  change their defaults; enable them per machine through the environment
  for I-3 measurement only.
- Do not add Plan F references to the legacy core, the UI or the evaluation
  runner; the seam is `ui/pipeline.py`.

### Next task

I-3: run the connected architecture on the real knowledge base and models in
SHADOW mode and measure registry construction, latency, accepted and
fallback rates and failure behaviour, after approval.

### Files changed

`ui/pipeline.py`, `handbook_bot/config.py`, `handbook_bot/plan_f_runtime.py`
(cache), `tests/test_plan_f_pipeline.py` (new), `tests/test_plan_f_runtime.py`
(one isolation assertion), `docs/STATUS.md` (this entry).

---

## 2026-10-30 - Plan F runtime adapter (runtime integration, phase I-1)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `d95084f` |
| Ending state | adapter implemented and tested; NOT wired into the live chatbot (`ui/pipeline.ask` still calls `answer_question`, `PLAN_F_ENABLED` is read by nothing, no shadow mode); the four frozen specialists and the legacy runtime modules unchanged (hashes verified before and after); committed as the I-1 commit on top of `d95084f` and pushed with a normal push |

### Completed

- `handbook_bot/plan_f_runtime.py` (new): the safe seam between the Plan F
  path and the legacy `QAResult` contract. Registry factory from the real
  `KnowledgeBase` through the four specialists' own `from_knowledge_base`
  builders, fail-closed (a complete registry or none, every builder failure
  redacted; no module-level registry or cache). `PlanFRuntime.attempt` runs
  the Coordinator and the dispatcher on an explicit registry and accepts only
  the I-1 safe case: one selected specialist, one task, one focus clause, no
  Synthesis requirement, no multipart question, no uncovered clause; a
  `supported` result with no handoff and no `missing` entry; a clause report
  that is a recognised specific ask with every requested detail covered and a
  strong relation where one is reported, or an Institutional `read` outcome
  at tier E1, E2 or stated fact. Quote guard: the quote is an exact substring
  of `chunks[chunk_id]` on the cited page of the cited source, no
  normalisation. Subject guard: every named entity of the clause is in the
  quote, the specialist's requested details are covered by the finding and
  the clause's content words are covered under the unchanged Verifier's
  rules; an Institutional finding must name the office the clause asks about
  (or the University) and a contact answer needs a complete directory row.
  Extractive renderer (verbatim quotes, pages from accepted findings only),
  the unchanged `agents.verifier.verify` as the final gate, a complete
  `QAResult` (`used_llm=False`, `llm_calls=0`, `retried=False`, the Plan F
  trace in `agent_trace`), nine explicit `PlanFOutcome` values, and a
  test-only `answer_with_fallback` that the application does not call.
- Required cases blocked: "What does Human Resources do?" and "What does HR
  do?" (the frozen HR specialist still returns `supported`; the clause is a
  general ask, so it is not eligible) and "How many days of annual leave do
  faculty members get?" (`partial`; never rendered, the sixty-day maternity
  sentence never leaves the trace). The fixture-corpus shape of the HR
  defect, a supported quote that never names Human Resources, is rejected by
  the subject guard.
- `tests/test_plan_f_runtime.py` (new, 79 tests) on a knowledge base built
  from the full real handbook with the project's loader, chunker and
  metadata (fake embedder, stand-in reranker): registry (all four builders,
  the HR builder included; defective knowledge bases; one builder failing;
  a builder producing the wrong specialist; no partial registry; redaction),
  evidence adapter, quote guard (13 unsafe shapes), subject guard, status
  gating for every status and for handoffs, one supported example of each
  specialist rendered through the adapter, complex questions falling back,
  Verifier integration, the pinned `QAResult` contract, zero model calls,
  determinism, the fallback wrapper and live-runtime isolation.
  `tests/test_planf_imports.py`: the adapter added to the modules that must
  import alone without loading the orchestrator (one additive line).
- Fixture facts: registry construction about 0.7 s on the real pages with the
  fake embedder (Institutional reads its directory and statements at
  construction); adapter time per question 1 to 260 ms. Real-model timings
  remain UNKNOWN until I-3.

### Tests performed

Targeted gate (adapter, Plan F Coordinator, registry, contracts, imports,
four-specialist dispatch, Verifier, compatibility): 598 passed, 0 failed
(26 s).
Full suite after the change: 6,382 collected, 6,382 passed, 0 failed, 0 skipped,
0 xfailed (183 s): the 6,302 baseline plus 79 adapter tests plus one added
import-isolation case.
`pip check`: no broken requirements. Frozen and live-runtime hashes
identical before and after: the six frozen specialist files, the shared
specialist helpers, `coordinator.py`, `dispatch.py`, `app.py`, `ui/chat.py`,
`ui/pipeline.py`, `qa.py`, `orchestrator.py`, `router.py`, `retrieval.py`,
`synthesis.py`, `verifier.py`, `types.py`, `config.py`.

### Known issues at handoff

- The adapter is deliberately narrow: a general ask ("What does the
  Technology Transfer Office do?") and a topic question whose content words
  the quote does not cover ("Is attendance mandatory for students?") fall
  back to the legacy path even when the specialist answered well; coverage
  is measured in shadow mode (I-2, I-3), not raised by relaxing the gates.
- A Research quote unit may span two sentences; both are rendered
  (extractive by design).
- Rendered answers are the verbatim quotes with page chips and no
  attribution sentence; richer deterministic rendering belongs to I-5.
- Real-model registry construction and latency are unmeasured (I-3).

### Do not redo

- Do not wire the adapter into `ui/pipeline.py`, read `PLAN_F_ENABLED`, add
  shadow mode, build the registry at startup, execute handoffs or aggregate
  findings before I-2 is approved.
- Do not relax the I-1 gates to gain coverage.

### Next task

I-2: the feature-flagged runtime seam in `ui/pipeline.ask` with shadow
mode, after approval.

### Files changed

`handbook_bot/plan_f_runtime.py` (new), `tests/test_plan_f_runtime.py`
(new), `tests/test_planf_imports.py` (one additive entry), `docs/STATUS.md`
(this entry).

---

## 2026-10-29 - Institutional Navigation freeze review and Step 4.16 closure (Milestone 4, Step 4.16 freeze; D-107)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `c316698` |
| Ending state | Step 4.16 COMPLETE; Institutional Navigation frozen for Milestone 4 (D-107); all four specialists frozen and executable through the default Plan F dispatch; the Step 4.16 work committed on `feature/remaining-specialists` as one completion commit on top of `c316698` (normal push to `origin/feature/remaining-specialists`, no force, no rebase, no amend); Plan F still not wired into the live chatbot (`PLAN_F_ENABLED` False); runtime integration not started |

### Completed

- Independent adversarial freeze review of the Step 4.16-C1 code (read-only;
  the production manifest unchanged before and after): full suite 6,302
  passed before and after; Institutional 2,293 and four-specialist 216
  passed; contact 226/226, evidence 690/690, benchmark byte-identical to the
  C1 baseline (12/48, 0 wrong), holdout 5/24, 0 wrong; 346 fresh adversarial
  questions (171 blind, 175 white-box) on both paths and both rankings: 43
  supported answers, all verified correct, 0 wrong supported answers, 34
  partial findings on target, 0 non-verbatim quotes, every negative control,
  ambiguity and cross-specialist case refused or handed off; mutation 6/6
  and 11/11 killed, the B4 and C1 switches guarded by their flip tests;
  determinism identical across repeats, fresh instances and rankings; no
  regressions. Verdict: freeze approved, no blocking finding.
- D-107 recorded: Institutional Navigation frozen for Milestone 4 with the
  frozen SHA-256 hashes of `institutional.py` and `navigation_cues.py`, the
  reviewed hashes, the accepted evidence, the component boundary and the
  D-103 change-control rule; runtime integration explicitly outside the
  freeze.
- Documentation-only corrections found by the review: the `institutional.py`
  module docstring (C1 paragraph added; "not in the default dispatch"
  replaced by the B6/D-105, freeze and runtime state), `PHASE = "C1"` and
  the phase label of the limitation text (five phase-label assertions in
  the Institutional tests updated from "B4" to "C1"; no other test
  change); the shadowed duplicate `navigation_cues._ABBREVIATION` (the C1
  copy, reached by no call) removed, behaviour unchanged;
  `docs/COORDINATOR.md` dispatch paragraph (all four specialists
  executable); `docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md` status header,
  "Freeze status (D-107)" section and phase list.
- Step 4.16 closed:

  | Step | State |
  |---|---|
  | 4.16-A design | complete |
  | 4.16-B1 contract, scope, lexicon (D-104) | complete |
  | 4.16-B2 directory lookups | complete |
  | 4.16-B3 responsibilities, structure, locations (D-106) | complete |
  | 4.16-B4 boundaries and handoffs | complete |
  | 4.16-B5 QA and acceptance | complete |
  | 4.16-B6 default dispatch (D-105) | complete |
  | 4.16-C1 natural phrasing | complete |
  | Independent freeze review | complete |
  | D-107 Institutional Navigation frozen | recorded |

  Step 4.16: COMPLETE. Runtime integration: NOT started (next stage).
- Four-specialist state: Teaching & Learning frozen; Research & Innovation
  frozen; Faculty Services & HR frozen (D-103); Institutional Navigation
  frozen (D-107); all four in `dispatch.EXECUTABLE_SPECIALISTS` and run by
  `run_question` when registered (`tests/test_four_specialist_dispatch.py`).
- Git: the Step 4.16 work (Institutional production files, tests, QA
  assets, the D-105 dispatcher change, documentation, D-105 to D-107)
  committed in one commit on top of `c316698`; pushed to `origin/feature/remaining-specialists` with a normal upstream push (the
  remote branch did not exist before).

### Tests performed

Targeted, after the documentation-only changes: Institutional 2,293, four-specialist
dispatch 216, Coordinator 143, Plan F contracts/registry/imports 108, the frozen
specialists' integration and alignment tests 159, shared helpers 65: 2,984 passed,
0 failed, 0 skipped, 0 xfailed (46.5 s).
QA replays on the frozen files: contact accuracy 226/226 as expected, 6/6 `_B2_*`
switches killed, restored replay 113/113; evidence binding 690/690 as expected, 11/11
`_B3_*` switches killed, restored replay 345/345; the 62-question benchmark through the
default dispatch byte-identical to `coverage_baseline_c1.jsonl` (system 12/48, direct
10/48, 0 wrong, 14/14 refusals, routing 34/62 exact); holdout system 5/24, direct 5/24,
0 wrong, 12/12 refusals. No nondeterministic case.
Full suite, final gate before the commit: 6,302 collected, 6,302 passed, 0 failed, 0 skipped, 0 xfailed (170.65 s).
`pip check`: no broken requirements.

### Known issues at handoff

- Coordinator routing gap (DEP-2): Institutional Navigation is not selected
  for part of the answerable navigation questions (18 of 48 benchmark, 11
  of 24 holdout, 35 of 107 answerable blind questions, 33 of which the
  specialist would not answer either); bare structure and count questions
  reach no specialist; same-domain double asks are not split. Classified in
  the freeze review, not fixed; a separate approved step.
- Recall limitations accepted with the freeze (D-107): duty words shared
  with the office name, synonyms, referral wordings, context outside the
  closed shapes, lowercase acronyms and typos fail closed; split, sub-row
  and page 52 directory entries give no number; "Where is the Women's Main
  Library?" fails closed.
- Frozen HR specialist, outside this step: "What does Human Resources do?"
  is routed to HR alone, which returns a supported answer quoting an
  unrelated page 64 contract-termination sentence; recorded for the HR
  change log, not corrected (D-103 change control).
- The live UI issues noted on 2026-10-28 (raw exception text, `0.0.0.0`,
  no authentication) stay for the integration step.

### Do not redo

- Do not reopen any of the four frozen specialists without a new decision
  record (D-103, D-107).
- Do not start runtime integration (adapter, `ui/pipeline.py`,
  `PLAN_F_ENABLED`, Synthesis aggregation, Verifier findings mode) before
  its decision record.
- The holdout (`f6367ccf...`) and the benchmark (`2c2e14cf...`) stay
  unchanged.

### Next task

Multi-agent runtime integration planning and implementation: the
integration decision record for the adapter behind `ui/pipeline.ask`
(`APPLICATION_INTEGRATION_READINESS.md`), then its implementation; the
Coordinator routing step (DEP-2) separately.

### Files changed

`docs/DECISIONS.md` (D-107), `docs/STATUS.md` (this entry),
`docs/COORDINATOR.md`, `docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`,
`handbook_bot/agents/specialists/institutional.py` (docstring, `PHASE`,
limitation text), `handbook_bot/agents/specialists/navigation_cues.py`
(duplicate `_ABBREVIATION` removed), `tests/test_institutional_contract.py`,
`tests/test_institutional_directory.py`,
`tests/test_institutional_integration.py`,
`tests/test_institutional_offices.py` (phase label). Review deliverables
outside the repository:
`SENIOR_PROJECT_HANDOVER/INSTITUTIONAL_STEP_4_16_FREEZE/` (freeze review
report and baseline manifest).

---

## 2026-10-28 - Application integration readiness analysis (Milestone 4, step 4.16-C1 secondary objective; read-only)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `c316698` |
| Ending state | documentation only; no production, test or QA file changed; the C1 work of 2026-10-26 stands unchanged (production hashes identical to its final gate run); Plan F still not wired into the live chatbot; `PLAN_F_ENABLED` False; nothing committed |

### Completed

- Re-verified the C1 state before proceeding: `institutional.py` and
  `navigation_cues.py` hash-identical to the final C1 gate run, 40/40 frozen
  files unchanged, nothing staged, HEAD `c316698`.
- Read-only inspection of the live path (`app.py` -> `ui/chat.py` ->
  `ui/pipeline.ask` -> `handbook_bot.answer_question` -> `orchestrator.run`)
  and of the Plan F entry points (`coordinator.coordinate`,
  `dispatch.run_question` / `dispatch`, the four `from_knowledge_base(kb)`
  builders, `build_knowledge_base`). Confirmed: no API module, no
  authentication, in-memory per-tab sessions, `PLAN_F_ENABLED` read by no
  production module, no production registry builder, no findings renderer,
  no Milestone 1 fallback in code.
- Deliverable `SENIOR_PROJECT_HANDOVER/INSTITUTIONAL_STEP_4_16_C1/
  APPLICATION_INTEGRATION_READINESS.md`: confirmed entry points with
  file:line references; the minimum safe architecture (an adapter module
  behind the `ui/pipeline.ask` seam, deterministic rendering of findings
  into the unchanged `QAResult`, fallback to the Milestone 1 path, the flag
  as the engine switch); options A (NiceGUI), B (JSON API in the same
  FastAPI-based process), C (mobile over B); request/response structures,
  sessions, citations, errors and timeouts, security, configuration,
  rollback, integration tests, the files likely to change.
- The C1 QA report gained the gate mapping to the 2026-10-28 brief and the
  ten acceptance criteria; the C1 handoff gained the integration paragraph.

### Tests performed

None run in this step (no code changed). The C1 gates of 2026-10-26 stand:
full suite 6302 passed; contact 226/226; evidence 690/690; benchmark 12/48
system, 10/48 direct, 0 wrong; holdout 5/24 and 5/24, 0 wrong.

### Known issues at handoff

- Setting `PLAN_F_ENABLED=True` connects nothing today; the adapter does not
  exist.
- The live UI shows raw exception text in its catch-all (`ui/chat.py`
  `_submit`), binds `0.0.0.0` and has no authentication: to fix in the
  integration step, not here.
- Not solved by the proposed architecture: the Coordinator routing gap
  (DEP-2), handoff rounds, Synthesis aggregation, the Verifier findings mode,
  the real-model evaluation.

### Do not redo

- Do not implement the adapter, change `PLAN_F_ENABLED`, or touch the UI
  before the freeze review and an integration decision record.

### Next task

Institutional freeze review (independent adversarial pass on the final C1
code); then the integration decision record for the adapter; the Coordinator
routing step separately.

### Files changed

`docs/STATUS.md` (this entry) only. Deliverables outside the repository:
`SENIOR_PROJECT_HANDOVER/INSTITUTIONAL_STEP_4_16_C1/APPLICATION_INTEGRATION_READINESS.md`
(new), `INSTITUTIONAL_C1_QA_REPORT.md` and
`INSTITUTIONAL_C1_FREEZE_READINESS_HANDOFF.md` (sections added).

---

## 2026-10-26 - Institutional Navigation natural-language coverage correction (Milestone 4, step 4.16-C1)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `c316698` |
| Ending state | implemented locally; not yet committed; Institutional Navigation reads polite, introduced and indirect questions through a closed allowlist; not frozen; Coordinator, dispatcher and the three frozen specialists unchanged; Plan F still not wired into the live chatbot |

### Completed

- `navigation_cues.natural_ask`: a focus clause is read as the plain question it
  asks (request frames, greetings, contractions, filler words in one closed
  position, embedded order, count phrasing), and every B2/B3 check reads the
  result. Context is set aside only in closed shapes (before the question: a
  greeting or thanks, a self-introduction, or the asker's own unit; after it:
  only thanks). Four switches with negative controls (`_C1_NATURAL_ASK`,
  `_C1_CONTEXT_GUARD`, `_C1_KIND_CHANGE_CHECK`, `_C1_COMPOSITION_CONTINUES`).
- Eight `contact=False` chapter units with their own statements added to the
  lexicon (CLD, SSU, Transport Section, Mail Office, Facilities Management
  Section, Projects Management Section, Deans Council, UAAC); five
  never-equivalent pairs; a few general plain-form wordings.
- A blind 36-question holdout frozen before any code change
  (`qa/institutional_4_16/holdout_cases.jsonl`); runner option `--cases`;
  stored C1 benchmark run `coverage_baseline_c1.jsonl`.
- Four adversarial review rounds (1,523 probes). The first C1 version had
  4 blockers (corrections, sub-units and fields in the context set aside; the
  ownership check skipped); the open context handling was replaced by the
  closed allowlist. Round 4 found one minor issue, fixed.
- Tests: new `tests/test_institutional_natural_phrasing.py` (102); deliberate
  updates in the acceptance, four-specialist and offices tests; B3 case
  expectations D09, U17, S01 changed with the lexicon (handbook evidence
  recorded). Docs: `docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`, the QA
  README. Deliverables in `SENIOR_PROJECT_HANDOVER/INSTITUTIONAL_STEP_4_16_C1/`.

### Tests performed

Full suite: 6302 collected, 6302 passed, 0 failed, 0 skipped, 0 xfailed
(6174 before C1; +128 new Institutional tests).
Institutional family and four-specialist tests: 2509 passed. Contact accuracy
226/226 and evidence binding 690/690 with every `_B2_*` / `_B3_*` switch
killed. The unchanged 62-question benchmark through the default dispatcher:
coverage 12/48 (was 6/48), Institutional alone 10/48 (was 4/48), 0 wrong
answers, 0 answers lost, 14/14 refusals kept, Coordinator selections
identical. Holdout: 5/24 system (was 2/24), 5/24 direct (was 0/24), 0 wrong.
Every run repeated byte-identical. No real-model run, no paid API, no FAISS
rebuild; the 25 guides not ingested.

### Known issues at handoff

- The Coordinator still does not route 18 of 48 benchmark Institutional
  questions to Institutional Navigation (DEP-2), most of them to no specialist;
  Institutional alone answers 5 of the never-routed benchmark and holdout
  questions, so the rest of the gap also needs Institutional work.
- Holdout gains are partly contaminated (HS02, HS07, HR08, HS04; see the C1
  coverage comparison); conservative reading 4/24 system, 1/24 direct.
- `navigation_cues._ABBREVIATION` (C1 section) is shadowed by an older
  definition; safe (fails closed); remove at the freeze review.
- Coverage stays low by design: context outside the closed shapes, name
  variants the handbook does not print, referral wordings and duty paraphrases
  fail closed.
- Stale text recorded for the freeze review: the `institutional.py` module
  docstring and `PHASE`, `docs/COORDINATOR.md`, two frozen test names.

### Do not redo

- The holdout is frozen (SHA-256 `f6367ccf...`); do not edit it. The benchmark is
  unchanged (`2c2e14cf...`).
- Word-list context filters were broken twice by the review; keep context
  handling to closed shapes.

### Next task

Formal freeze review of Institutional Navigation (with an independent
adversarial pass on the final code); then a separately approved Coordinator
routing step.

### Files changed

`handbook_bot/agents/specialists/institutional.py`,
`handbook_bot/agents/specialists/navigation_cues.py`,
`tests/test_institutional_natural_phrasing.py` (new),
`tests/test_institutional_acceptance.py`, `tests/test_four_specialist_dispatch.py`,
`tests/test_institutional_offices.py`, `qa/institutional_4_16/holdout_cases.jsonl`
(new), `qa/institutional_4_16/coverage_baseline_c1.jsonl` (new),
`qa/institutional_4_16/coverage_runner.py`, `qa/institutional_4_16/build_evidence_cases.py`,
`qa/institutional_4_16/evidence_cases.jsonl`, `qa/institutional_4_16/README.md`,
`docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`, `docs/STATUS.md`.

---

## 2026-10-26 - Institutional Navigation enabled in the default Plan F dispatch (Milestone 4, step 4.16-B6)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `c316698` |
| Ending state | implemented locally; not yet committed; four specialists executable in the default Plan F dispatch (D-105); Plan F still not wired into the live chatbot; Teaching, Research and Faculty Services & HR frozen and unchanged; Institutional Navigation not frozen |

### Completed

- D-105: `dispatch.EXECUTABLE_SPECIALISTS` gains `SpecialistId.INSTITUTIONAL`
  (`handbook_bot/agents/specialists/dispatch.py`: the tuple, its comment and
  the module docstring; the only production change). All four specialists
  execute through the default `dispatch()` / `run_question()` when
  registered; an unregistered specialist or a handoff-produced task stays
  pending; handoffs remain data.
- Test updates: the dispatcher change failed 48 cases in 9 test functions,
  all phase-state assertions (Institutional absent from the executable set or
  pending by default); only those were updated. Frozen specialists' test
  files: one assertion each in `tests/test_faculty_services_integration.py`
  and `tests/test_research_integration.py`; no behaviour assertion changed.
  The plan-listed `test_teaching_integration.py`, `test_teaching_alignment.py`
  and `test_clause_retrieval_intent.py` needed no change (their registries do
  not register Institutional, so it stays pending there).
- New `tests/test_four_specialist_dispatch.py` (215); benchmark runner option
  `--default-dispatch`; stored B5 benchmark result
  `qa/institutional_4_16/coverage_baseline_b5.jsonl`. A bounded review (two
  lenses, a skeptic per finding) found no blocker or major issue; its minor
  findings (two tautological test groups, a vacuous assertion, stale comments
  and names) were fixed or recorded.
- `docs/AGENT_CONTRACTS.md`, `docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`
  and D-105 updated. Deliverables in
  `SENIOR_PROJECT_HANDOVER/INSTITUTIONAL_STEP_4_16_B6/`.

### Tests performed

Full suite: 6174 collected, 6174 passed, 0 failed, 0 skipped, 0 xfailed
(5959 before B6; +215 new). Families: Faculty Services & HR 1309 (the D-103
count), Research 1221, Teaching 670, Institutional 2166, four-specialist 215,
Plan F 251. Evidence binding 690/690 and contact accuracy 226/226 with every
`_B2_*` / `_B3_*` switch killed. The 62-question blind benchmark through the
default dispatcher is byte-identical to Step 4.16-B5: routing 34/62 exact,
Institutional recall 25/46, coverage 6/48 (12.5%), 0 wrong or unsafe
supported answers, citations 6/6. No real-model run, no paid API, no FAISS
rebuild; the 25 guides not ingested.

### Known issues at handoff

End-to-end coverage stays 12.5% (an integration and readiness risk); the
Coordinator misses 21 of 46 Institutional-owned questions (DEP-2);
HR / Research referrals are declined by both specialists in the first round
and handoffs are not executed yet; no Teaching handoff (DEP-3). Stale text
recorded for the freeze review: the `institutional.py` docstring,
`docs/COORDINATOR.md` (dispatch paragraph), two frozen test names.
Institutional is not frozen; live chatbot integration and real-model
evaluation not started.

### Files changed

Production: `handbook_bot/agents/specialists/dispatch.py`. Tests:
`tests/test_faculty_services_integration.py`,
`tests/test_research_integration.py` (phase-state assertions only),
`tests/test_institutional_contract.py`, `test_institutional_directory.py`,
`test_institutional_offices.py`, `test_institutional_integration.py`,
`test_institutional_alignment.py`; new `tests/test_four_specialist_dispatch.py`.
QA: `qa/institutional_4_16/coverage_runner.py`, `README.md`, new
`coverage_baseline_b5.jsonl`. Docs: `docs/DECISIONS.md` (D-105),
`docs/AGENT_CONTRACTS.md`, `docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`,
`docs/STATUS.md`.

## 2026-10-26 - Institutional Navigation final QA and acceptance (Milestone 4, step 4.16-B5)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `c316698` |
| Ending state | QA only, no production change; not yet committed; architecture inactive; Teaching, Research and Faculty Services & HR frozen and unchanged; Institutional Navigation accepted with documented non-blocking limitations, ready for B6 (default dispatch, D-105), not executable by default |

### Completed

- Requirements traceability of C1 to C12, U1 to U4, R1 to R11 and the
  status semantics to functions, tests, handbook evidence and results; the
  R9 / routing-document conflict reconciled (R9 authoritative; no
  information Institutional Navigation could give is lost; the questions
  stay unanswered for the user until integration).
- Test groups G (evidence conflicts) and H (determinism independent of
  retrieval): new `tests/test_institutional_conflicts.py`.
- Independent evaluation: 62 faculty-style questions written by blind
  authors from the handbook pages only, expected quotes verified on their
  pages before any run (`qa/institutional_4_16/coverage_cases.jsonl`,
  `coverage_runner.py`), run through the Coordinator and the dispatcher with
  all four specialists and directly; new
  `tests/test_institutional_acceptance.py` (strict safety assertions,
  measured coverage, the known limitations A to I, the redacted error path).
- Findings classified: no P0, no P1; P2 non-blocking for B6: low
  natural-language coverage (F-01) and Coordinator routing gaps (F-02, frozen
  Coordinator); ten P3 accepted limitations. Deliverables in
  `SENIOR_PROJECT_HANDOVER/INSTITUTIONAL_STEP_4_16_B5/`.

### Tests performed

Independent evaluation: answer correctness on answered questions 6/6
(system) and 4/4 (direct); 0 unsafe supported and 0 wrong answers; 14/14
correct refusals; citations on their expected pages 6/6 and 4/4; coverage
of source-answerable questions 6/48 (system) and 4/48 (direct); Coordinator
exact selection 34/62, Institutional inclusion 38/62; second run
byte-identical. New B5 tests: 25 passed. B4 gate results reused for the
unchanged production code (B2 226/226 and B3 690/690 with every switch
killed; 296 B4 tests; 448 Coordinator, dispatcher and frozen integration
tests). Full suite, run once from the repository root: 5959 passed, 0
failed, 0 skipped (5934 before B5). No real-model run, no model download,
no FAISS rebuild, no paid API; the 25 new documents not ingested.

### Known issues at handoff

The narrow question forms answer few natural phrasings (a greeting, a
request verb or a context sentence fails closed); the Coordinator does not
select Institutional Navigation for 21 of 46 Institutional-owned questions
(DEP-2); HR / Research referrals are declined by both specialists in the
first round; no Teaching handoff (DEP-3); lexicon gaps for printed units;
multi-line directory entries; C7, C8 and C10 not answered. A bounded
coverage correction is recommended after B6 and before the freeze review.
B6 not started; the live chatbot does not use Plan F.

### Files changed

New: `tests/test_institutional_conflicts.py`,
`tests/test_institutional_acceptance.py`,
`qa/institutional_4_16/coverage_cases.jsonl`, `coverage_runner.py`.
Modified: `qa/institutional_4_16/README.md`,
`docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`, `docs/STATUS.md`. No
production file changed.

## 2026-10-26 - Institutional Navigation cross-specialist boundaries and handoffs (Milestone 4, step 4.16-B4)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `c316698` |
| Ending state | implemented locally; not yet committed; architecture inactive; Teaching, Research and Faculty Services & HR frozen and unchanged; Institutional Navigation in development (phase B4 of B1 to B6), not executable by default |

### Completed

- Phase B4 of the Institutional Navigation specialist (under D-104): before a
  non-contact clause is answered, its subject is checked with the shared
  `hr_owned` / `research_owned` cues (unchanged) and the B3 local lists. An
  HR or Research subject the shared cues name gets no finding and is
  returned as a `HandoffRequest` to that owner (never executed); a clause
  is never handed back to the specialist whose handoff task it came in; a
  Teaching subject, a clause only the local lists claim, and a shared-cue
  clause that also has a Teaching subject get no finding and no handoff
  (no target is guessed); contact and location asks are never handed off,
  and an office's own name never makes a clause another specialist's;
  approvals are never answered with an office. A task with no clause of its
  own is `out_of_scope`. Four `_B4_*` switches, each shown necessary.
- D-106 recorded: the team's four provisional acceptance conditions for the
  B3 design; no condition violated, condition 3 (real-model coverage) open.
  D-103 unchanged; D-105 stays reserved for default dispatch.
- New `tests/test_institutional_integration.py` (a 48-question matrix
  through the unchanged Coordinator and the dispatcher with all four
  specialists) and `tests/test_institutional_alignment.py` (Coordinator
  selections unchanged; every frozen specialist's result identical whether
  Institutional Navigation runs or stays pending).
- A bounded review (three lenses, one skeptic each) confirmed two minor text
  defects (a not-handed-back note and the approval note claimed facts about
  the handbook the code never checked); both fixed with regression asserts.
  Its plan and boundary lenses failed four times on network errors; the plan
  check was done inline and an earlier partial boundary probe led to the
  shared-cue-only handoff rule.

### Tests performed

Focused B4: 296 passed (integration 185, alignment 111). Institutional B1
to B3: 1845 passed (scope 132, contract 53, directory 498, offices 1162).
Coordinator, dispatcher and frozen specialists' integration and alignment
tests: 448 passed. Evidence binding 690 of 690 runs as expected (110
supported and 60 partial unchanged; 23 foreign-subject cases now
out_of_scope, 9 with a handoff), every `_B3_*` switch killed; contact
accuracy 226 of 226, the same 118 numbers, every `_B2_*` switch killed;
second replays and case-file rebuilds byte-identical. Full suite: 5934
passed, 0 failed, 0 skipped (5641 before B4), run from the repository root.
No real-model run, no model download, no FAISS rebuild, no paid API.
Passing cases do not prove accuracy on every phrasing.

### Known issues at handoff

Separate tasks: for "which office handles / who should I contact about <HR
or Research subject>" the subject specialist hands navigation to
Institutional Navigation and it hands the subject back, so neither answers
in the first round (bounded by one handoff round). No Teaching handoff
exists (DEP-3), so a Teaching subject the Coordinator gives only to
Institutional Navigation is unanswered. Ownership is read with word cues
(over- and under-reach documented). The routing document's rows for faculty
housing, employment questions and the leave balance conflict with R9; B4
follows R9 (for team review). Governance-level authority (C8) is not
answered. Coordinator cue gaps unchanged (DEP-2). QA campaign (B5) and
default dispatch (B6) not started; the live chatbot does not use Plan F.

### Files changed

New: `tests/test_institutional_integration.py`,
`tests/test_institutional_alignment.py`. Modified (all untracked since B1):
`handbook_bot/agents/specialists/institutional.py`,
`tests/institutional_support.py`, `tests/test_institutional_contract.py`,
`tests/test_institutional_directory.py`, `tests/test_institutional_offices.py`,
`qa/institutional_4_16/build_evidence_cases.py`, `evidence_cases.jsonl`,
`build_contact_cases.py`, `contact_cases.jsonl`, `README.md`,
`docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`. Modified: `docs/STATUS.md`,
`docs/DECISIONS.md` (D-106 appended). `navigation_cues.py` unchanged in B4.

## 2026-10-25 - Institutional Navigation responsibilities, structure and locations (Milestone 4, step 4.16-B3)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `c316698` |
| Ending state | implemented locally; not yet committed; architecture inactive; Teaching, Research and Faculty Services & HR frozen and unchanged; Institutional Navigation in development (phase B3 of B1 to B6), not executable by default |

### Completed

- Phase B3 of the Institutional Navigation specialist (under D-104; no new
  decision): office responsibilities (E2), organisational structure (chair,
  composition, subgroups, the bare organisation ask, counts, colleges) and
  the four documented library locations (section 11.6), each answered only
  from a handbook statement that binds the office: the named subject with a
  responsibility verb, a strict anaphor, or a list lead-in within one
  passage. Statements are read once from in-scope paragraph passages, never
  from the directory pages 52 to 55. A clause is answered only in one of the
  plain question forms; procedures, approvals, other specialists' subjects
  and mixed asks are read and not answered (phase B4). No floor, room, hour,
  map, GPS, officeholder or reporting line is given.
- Eleven evidence-binding protections (`_B3_*`), each shown necessary by
  mutation. `qa/institutional_4_16/`: 345 evidence-binding cases (15
  categories, including the confirmed defects of five adversarial reviews),
  case builder, runner with mutation mode.
- Team decisions recorded in `docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`
  (no D-number; D-105 stays reserved for default dispatch): Decision A,
  bounded direct directory reading, accepted with its six properties checked
  (property 6 on the build path only, Plan F not being live); Decision B,
  page-52 contact expansion deferred, no telephone or fax answer added.
- For team review: (1) B3 uses its own statement reader instead of the
  shared `grounding.AskGate` named in the 4.16-A plan (the unmodified gate
  cannot bind the handbook's E2 statements; no shared module changed); (2)
  statement reading extends Decision A; (3) inside `_B2_REFERENT_CHECK`, a
  contact ask now needs a plain contact form (all 113 B2 cases unchanged, a
  few natural phrasings now fail closed).

### Tests performed

New `tests/test_institutional_offices.py` (1165 passed). B1/B2 regression:
`tests/test_institutional_contract.py` (53), `tests/test_institutional_directory.py`
(498) made phase-aware, `tests/test_institutional_scope.py` unchanged (132,
parametrised over the 49-entry lexicon): 683 passed. Plan F contracts,
registry, imports, Coordinator and clause retrieval intent: 289 passed.
Evidence-binding evaluation: 690 runs (345 cases, two stand-in rankings),
690 as expected, 0 wrong-office, wrong-location or invented answers,
deterministic; every `_B3_*` protection switched off alone lets a wrong
answer out. Contact-accuracy evaluation: 226 of 226 as expected, the same
118 numbers, every `_B2_*` protection still needed. Case files rebuild
byte-identical. Full suite: 5641 passed, 0 failed, 0 skipped (4468 before
B3). No real-model run, no model download, no FAISS rebuild, no paid API.
Passing cases do not prove accuracy on every phrasing.

### Known issues at handoff

Rule-based English reader: five adversarial reviews found defects in every
round; all confirmed wrong answers were fixed and kept as cases, and the
design fails closed outside the plain forms, but new phrasings may find
gaps. Coverage gaps (fail closed): department lists of colleges, the OIR
list beyond item 1, British spellings, restricted one-word duties, purpose
clauses, "health and safety" asks (partial), two-ask clauses, negated
location asks, "&"/"and" college-name variants, the page-52 directory
entries. Handoffs and mixed questions (B4), the QA campaign (B5) and default
dispatch enablement (B6, a separate decision) are not started; the live
chatbot does not use Plan F.

### Files changed

New: `tests/test_institutional_offices.py`,
`qa/institutional_4_16/evidence_cases.jsonl`, `build_evidence_cases.py`,
`evidence_binding_runner.py`. Modified (all untracked since B1):
`handbook_bot/agents/specialists/institutional.py`,
`handbook_bot/agents/specialists/navigation_cues.py`,
`tests/institutional_support.py`, `tests/test_institutional_contract.py`,
`tests/test_institutional_directory.py`, `qa/institutional_4_16/README.md`,
`build_contact_cases.py`, `contact_cases.jsonl`,
`contact_accuracy_runner.py`, `docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`.
Modified: `docs/STATUS.md`. `docs/DECISIONS.md` unchanged in B3.

## 2026-10-24 - Institutional Navigation directory lookups (Milestone 4, step 4.16-B2)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `c316698` |
| Ending state | implemented locally; not yet committed; architecture inactive; Teaching, Research and Faculty Services & HR frozen and unchanged; Institutional Navigation in development (phase B2 of B1 to B6), not executable by default |

### Completed

- Phase B2 of the Institutional Navigation specialist (under D-104; no new
  decision): telephone and fax numbers from the contact directory (section
  1.16). A number is returned only from one directory line that starts with
  the asked office's printed name, holds exactly two complete numbers, has no
  neighbouring line running into it and sits under a header naming the
  telephone and fax columns; the asked office must be the unit asked about.
  Seventeen entries are answerable; one-number, labelled, multi-line and
  split entries are `partial` (office name only, no number) or `not_found`.
  E-mail and website asks are not answered. Numbers are reported as printed,
  not as verified to be active; shared numbers are flagged.
- `qa/institutional_4_16/`: 113 contact-accuracy cases (expected numbers taken
  from the printed directory rebuilt from the PDF's geometry), case builder,
  replay runner with a mutation mode for the six row-binding protections.

### Tests performed

New `tests/test_institutional_directory.py` (494); `tests/test_institutional_contract.py`
made phase-aware (53); `tests/test_institutional_scope.py` unchanged (128):
675 passed. Plan F contracts, registry, imports, Coordinator and clause
retrieval intent: 289 passed. Contact-accuracy evaluation: 226 runs, 0
wrong-office or wrong-number answers, deterministic under both stand-in
rankings; every protection switched off alone lets a wrong answer out. Full
suite: 4468 passed, 0 failed, 0 skipped. No real-model run, no model
download, no FAISS rebuild, no paid API.

### Known issues at handoff

The directory starts on page 52; its page-52 entries (Chancellor's Office,
Vice Chancellors, Admissions) are not in the office lexicon and are not
answered. "IT Center" is not printed in the handbook and is not an alias.
Office responsibility, structure and locations (B3), handoffs (B4) and
default dispatch enablement (B6, a separate decision) are not started; the
live chatbot does not use Plan F.

### Files changed

New: `tests/test_institutional_directory.py`, `qa/institutional_4_16/`
(`README.md`, `contact_cases.jsonl`, `build_contact_cases.py`,
`contact_accuracy_runner.py`). Modified (all untracked since B1):
`handbook_bot/agents/specialists/institutional.py`,
`handbook_bot/agents/specialists/navigation_cues.py`,
`tests/institutional_support.py`, `tests/test_institutional_contract.py`,
`docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`. Modified: `docs/STATUS.md`.

---

## 2026-10-24 - Institutional Navigation specialist, design and first phase (Milestone 4, steps 4.16-A and 4.16-B1)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `c316698` |
| Ending state | implemented locally; not yet committed; architecture inactive; Teaching, Research and Faculty Services & HR frozen and unchanged; Institutional Navigation in development (phase B1 of B1 to B6), not executable by default |

### Completed

- Step 4.16-A (design only): requirements, architecture and contract,
  routing and boundaries (from the real Coordinator output), test strategy
  and implementation plan, in
  `SENIOR_PROJECT_HANDOVER/INSTITUTIONAL_STEP_4_16_A/`.
- Step 4.16-B1 (D-104): `specialists/institutional.py`
  (`InstitutionalResources`, `InstitutionalNavigationSpecialist`, scope,
  the fail-closed phase-B1 run) and `specialists/navigation_cues.py` (ask
  kinds, evidence tiers E1 / E2 / E3, the 46-entry office lexicon checked
  against the handbook PDF); `docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`.
  The specialist answers nothing yet: every result is `not_found` with the
  clause read and recorded.

### Tests performed

New: `tests/test_institutional_scope.py` (128) and
`tests/test_institutional_contract.py` (56), with
`tests/institutional_support.py` (real handbook pages, project PDF loader and
chunker, fake embedder): 184 passed, together with the Plan F contract,
registry and import tests (108 passed). Full suite: see the Step 4.16-B1
handover reports. No real-model run, no model download, no FAISS rebuild, no
paid API.

### Known issues at handoff

Institutional Navigation answers nothing in phase B1. The directory rows are
chunked unevenly (some name chunks hold part of a row, for example the
College of Sciences chunk holds only its fax number), so phase B2 must bind a
number only within one complete row. The handbook has no e-mail address and
two URLs. The Coordinator selects no specialist for "Who is the
Chancellor?" and "What does the Business Services Department do?". Default
dispatch enablement (B6) needs a separate decision and the phase-state
assertions of five existing test files.

### Files changed

New: `handbook_bot/agents/specialists/institutional.py`,
`handbook_bot/agents/specialists/navigation_cues.py`,
`tests/institutional_support.py`, `tests/test_institutional_scope.py`,
`tests/test_institutional_contract.py`,
`docs/INSTITUTIONAL_NAVIGATION_SPECIALIST.md`. Modified: `docs/DECISIONS.md`
(D-104), `docs/STATUS.md`.

---

## 2026-10-24 - Faculty Services & HR specialist frozen (Milestone 4, step 4.15)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | HR frozen for Milestone 4 (D-103); not yet committed; architecture inactive; Teaching, Research and HR frozen; Institutional Navigation (Step 4.16) not started |

### Completed

- Step 4.15: the freeze evidence of Step 4.14-W1 verified against the
  current files (the recorded results postdate the last code edit; the code
  hashes match the Step 4.14-W1 record; nothing re-run because nothing was
  missing or inconsistent); the freeze recorded as D-103 with the baseline
  hashes, the accepted test counts, the accepted limitations, the component
  boundary and the change-control rule; a freeze manifest and the Git
  checkpoint instructions prepared for the team in
  `SENIOR_PROJECT_HANDOVER/HR_STEP_4_15/`.

### Tests performed

None re-run in this step. Accepted results (Step 4.14-W1, on the frozen code):
focused 1,167 passed; HR suite 1,309 passed; full suite 3,793 passed; 0
failures, 0 errors, 0 skipped, 0 xfailed; replay 0 changed runs; mutation 15 /
19 (targeted), 4 / 19 (reconstructed), 17 distinct.

### Known issues at handoff

The accepted limitations of D-103 and `docs/FACULTY_SERVICES_SPECIALIST.md`
("Freeze status"). The Milestone 4 work since `bb8432d` (Teaching, Research,
HR, the shared modules, the Coordinator changes, their tests, `qa/` and the
docs) is uncommitted; the team's Git checkpoint is the next action;
`.venv_broken_backup/` is not ignored by `.gitignore` and must never be
staged.

### Files changed

`docs/DECISIONS.md` (D-103), `docs/STATUS.md`,
`docs/FACULTY_SERVICES_SPECIALIST.md` (freeze status). No code, test or QA file
changed.

---

## 2026-10-24 - HR conditional-language correction (Milestone 4, steps 4.14-R2 and 4.14-W1)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; HR ready for the Step 4.15 freeze decision; Step 4.15 not started |

### Completed

- Step 4.14-R2 (final freeze-readiness review, read-only): D-101 verified
  correct and monotonic (in code and over 466 instrumented runs: 0 looser
  decisions), D-100 still holding, every recorded gate genuine; one
  freeze-blocking finding, HR-R2-W1: the N1 evidence with "when" instead of
  "if" was supported (also "whenever" and a deontic variant), because D-098 /
  D-099 accepted every "when" trigger.
- Step 4.14-W1 (D-102, Option A chosen by the team): the D-099 class-C rule
  reads a "when" / "whenever" restriction of the asked predicate exactly as an
  "if" restriction (same exemptions; a concessive "even when" exempt; "once",
  "upon", "after", approval gates and existential claims unchanged); switch
  `_C12_WHEN_CONDITION` (nineteen `_C12_*` switches, all on). Two expectations
  moved, each with a companion that asks the condition: the "when promoted"
  salary row of the class-C table and the page-63 "Does confirmation of
  employment include the probation period?" (now partial; "... when
  calculating the End of Service Benefit?" supported).

### Tests performed

The W1 cases reproduced as SUPPORTED on the unchanged code ("when", "whenever",
deontic; the "if" control and HR-414-N1 partial). Focused gate (corrections,
support gate, composite, claims, statements, grounding): 1,167 passed. HR suite
(eleven files): 1,309 passed, 0 failed, 0 skipped, 0 xfailed (was 1,271; +34 in
`tests/test_faculty_services_corrections.py`, +4 in
`tests/test_faculty_services_support_gate.py`). Full suite: 3,755 to 3,793
passed, 0 failed, 0 skipped, 0 xfailed; Teaching 670 and Research 1,221 tests
unchanged and green. The 194 reconstructed cases under both rankings: 0 changed
runs against Step 4.14-N1 (0 supported to partial, 0 partial to supported), D1
and D2 still corrected against the Step 4.14-A baseline, a second replay
byte-identical; the real page-71 "even when" answer stays supported. Switch
mutations: R11 set plus the addendum (42 cases) 15 of 19 killed
(`_C12_WHEN_CONDITION` by SK-A5, SK-A6, SK-A7; no earlier kill lost; survivors
`_C12_ENTITY_SUBJECT`, `_C12_PAREN_NUMERAL`, `_C12_HEDGE`, `_C12_LEADS`);
reconstructed rounds (136 cases) 4 of 19, the same four as before; 17 distinct
switches killed over both sets; deterministic replay in every run. No
real-model run, no model download, no FAISS rebuild, no paid API.
`PLAN_F_ENABLED=False`. teaching.py, research.py, claim_model.py and every
shared module, the Coordinator, the dispatcher, the orchestrator, retrieval,
the knowledge base and the UI byte-identical to the 2026-10-24 snapshot.

### Known issues at handoff

No known materially wrong supported HR answer in the agreed regression scope
(the fixture pages, the 194 reconstructed cases, the R11 and adversarial sets,
the crafted N1 and W1 sets). Accepted conservative downgrades: an omitted
"when" event condition now gives partial (the two moved expectations; asking
with the condition is supported). Unchanged limits: a negative question that
asks its own "when" condition stays partial through the asked-role check;
"once", "upon" and "after" triggers are still accepted; the 27 conservative
misses of the reconstructed set and the F1 / F2 wrapper backlog; crafted-only
evidence for the N1 / W1 shapes; stand-in rankings; real-model validation and
runtime integration outstanding. The Step 4.15 freeze decision and the Git
checkpoint for steps 4.14-A to 4.14-W1 are pending the team (nothing staged or
committed).

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_corrections.py`,
`tests/test_faculty_services_support_gate.py`,
`tests/test_faculty_services_grounding.py` (one expectation moved, one
companion added), `qa/hr_4_14_a/` (SK-A5 to SK-A7, `after_4_14_w1/`, README),
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-102),
`docs/STATUS.md`.

---

## 2026-10-24 - HR condition-ownership correction (Milestone 4, steps 4.14-R and 4.14-N1)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; final HR freeze review pending; Step 4.15 not started |

### Completed

- Step 4.14-R (independent review, read-only): every Step 4.14 number
  confirmed from its artefact; P1 and P2 verified against their bounded
  designs; HR-414-N1 confirmed as a reproducible, materially wrong supported
  answer with its root cause in the HR-local own-content exemption
  `_c11_own`; the smallest correction prototyped (no change over the 194
  reconstructed cases); verdict: bounded defect, targeted correction
  required.
- Step 4.14-N1 (D-101): `_c11_own` also judges a restriction on the words
  that distinguish it from its statement's main clause (`_c11_main_roots`),
  so a condition that restates the asked entity ("... the housing allowance
  if they receive university housing") stays a condition; the rule is
  monotonic (it only withdraws a D-098 exemption, never grants one); a
  condition the question asks itself keeps its exemption through
  `_c12_asked_condition`; the three HR call sites pass the claim; switch
  `_C12_OWN_CONTENT` (eighteen `_C12_*` switches, all on). The strict xfail
  that recorded HR-414-N1 is now an ordinary passing test.

### Tests performed

The N1 strict xfail reproduced on the unchanged code and turned XPASS when
the correction landed. Focused gate (corrections, support gate, composite,
claims, statements): 974 passed. HR suite (eleven files): 1,271 passed, 0
failed, 0 xfailed (1,271 collected; was 1,243 passed + 1 xfailed; +27 tests in
`tests/test_faculty_services_corrections.py`, +1 switch row in
`tests/test_faculty_services_support_gate.py`). Full suite: 3,755 passed, 0
failed, 0 xfailed (3,755 collected; was 3,727 + 1 xfailed); Teaching 670 and
Research 1,221 tests unchanged in count and green. The 194 reconstructed cases
under both rankings: 0 changed runs against the final Step 4.14 results,
byte-identical to the Step 4.14-R prototype replay, D1 and D2 still corrected
against the Step 4.14-A baseline, a second replay byte-identical. Seven
crafted N1-shaped claims (negative, affirmative, another entity, heavy
vocabulary overlap, the same word on both sides, the entity restated whole,
eligibility) are partial under both rankings and supported again with the
switch off; asked conditions, the "subject to" case, unconditional claims, D1,
D2 and the true-claim guards are unchanged. Switch mutations: R11 set plus the
addendum (39 cases) 14 of 18 killed (`_C12_OWN_CONTENT` by SK-A2, SK-A3 and
SK-A4; every earlier kill reproduced; survivors `_C12_ENTITY_SUBJECT`,
`_C12_PAREN_NUMERAL`, `_C12_HEDGE`, `_C12_LEADS`, as before); reconstructed
rounds (136 cases) 4 of 18 killed, the same four as in Step 4.14; 16 distinct
switches killed over both sets; deterministic replay in every run. No
real-model run, no model download, no FAISS rebuild, no paid API.
`PLAN_F_ENABLED=False`. teaching.py, research.py, claim_model.py and every
shared module, the Coordinator, the dispatcher, the orchestrator, retrieval,
the knowledge base and the UI byte-identical to the 2026-10-24 snapshot.

### Known issues at handoff

No known materially wrong supported HR answer on the fixture pages or the
crafted sets. HR-414-N1 was only ever shown on crafted sentences; its
incidence on the real handbook is unknown and the full handbook was not
scanned. "Does the furniture allowance cover furniture?" stays supported from
page 106 ("is provided, which can be utilized over four years"): the
documented D-098 / D-099 reader limit that a unit mentioning the asked object
answers a verb it never states; unchanged by D-101. The 27 conservative
misses of the reconstructed set and the F1 / F2 wrapper backlog are
unchanged. Both rankings are stand-ins; real-model validation outstanding.
The final independent HR freeze review and the Git checkpoint for steps
4.14-A to 4.14-N1 are pending the team (nothing staged or committed).

### Files changed

`handbook_bot/agents/specialists/faculty_services.py` (+36 / -11),
`tests/test_faculty_services_corrections.py`,
`tests/test_faculty_services_support_gate.py` (one row),
`qa/hr_4_14_a/` (`hr_4_14_cases_switchkillers_addendum.jsonl` SK-A2 to SK-A4,
`compare_replay.py`, `after_4_14_n1/`, README), `docs/FACULTY_SERVICES_SPECIALIST.md`,
`docs/DECISIONS.md` (D-101), `docs/STATUS.md`.

---

## 2026-10-24 - HR bounded defect correction (Milestone 4, steps 4.14-A and 4.14)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; Step 4.15 not started |

### Completed

- Step 4.14-A (evidence recovery and QA, repository read-only): the Step
  4.12-R12 / 4.13 audit artefacts (35 reported wrong supported answers) were
  not recoverable from any location; 194 labelled cases were reconstructed on
  the fixture pages and replayed under both stand-in rankings: exactly two
  materially wrong supported answers (a statement about another entity in the
  same sentence, "Is the housing allowance paid once?"; a parenthesised
  numeral, "Is the probation period two (2) years?"), 27 conservative misses
  and 5 defensible over-supports; both corrections prototyped as
  process-local patches with no other change over the 194 cases and the 1,147
  HR tests green. The assets now live in `qa/hr_4_14_a/`.
- Step 4.14 (D-100): P2, `_C12_PAREN_NUMERAL` (a bracketed numeral that
  repeats its number words is dropped from the text the readers read), then
  the refined P1, `_C12_ENTITY_SUBJECT` (a claim with no role actor and no
  object binds its asked entity in the supporting clause when the sentence
  names that entity in another clause); both in the D-099 block of
  `faculty_services.py` (+105 / -5 lines), each behind its own switch; the two
  defect tests ported as ordinary passing tests. The switch-mutation
  investigation found and fixed an over-veto of the ported P1 on a true claim
  ("Does health insurance cover the spouse of faculty members?": a verb the
  claim reader does not anchor had leaked into the entity phrase), classified
  the three former mutation survivors (`_C12_HEDGE` equivalent at the status
  level, `_C12_LEADS` visible at clause level only, `_C12_NEGATIVE_CONDITION`
  without a triggering fixture, now killed by a crafted case) and recorded one
  new, out-of-scope reader gap (HR-414-N1) as a strict xfail.

### Tests performed

Full suite 3,727 passed, 1 xfailed (3,728 tests, 0 failures, 0 errors) in
334 s against the 3,631 baseline (+97 new tests); HR 1,243 passed, 1 xfailed
(1,244) in 253 s against 1,147 (new: `tests/test_faculty_services_corrections.py`,
91 tests, and six rows in `tests/test_faculty_services_support_gate.py`);
the one xfail is the documented HR-414-N1 gap. The two Step 4.14-A defects
reproduced as strict xfails on the unchanged code before any edit, flipped to
XPASS when each correction landed, and now pass as ordinary tests. Replay of
the 194 reconstructed cases under both rankings (388 runs): the only changed
runs are the two defects (supported to partial under both rankings); no
supported true claim regressed; no quoted evidence changed; a second replay
was byte-identical. Switch mutations: the R11 set plus the addendum (36
cases) kill 13 of 17 switches, the survivors being `_C12_ENTITY_SUBJECT` and
`_C12_PAREN_NUMERAL` (killed by R2-07a and R2-12j on the reconstructed set),
`_C12_HEDGE` and `_C12_LEADS` (status-equivalent; focused tests assert their
reading and clause-level effect); the reconstructed set (136 cases) kills 4 of
17 (defence in depth, as in Step 4.14-A); `_C12_LEDGER` is killed by SK-12
again after the P1 refinement; deterministic replay in every run. No
real-model run, no model download, no FAISS rebuild, no paid API.
`PLAN_F_ENABLED=False`. Research files, teaching.py, claim_model.py,
grounding.py, evidence.py, details.py, base.py, contracts.py, registry.py,
scoping.py, domain_cues.py, retrieval_intent.py, coordinator.py, dispatch.py,
orchestrator.py, retrieval.py, sources.py, config.py, the knowledge base and
the UI byte-identical to the 2026-10-24 snapshot.

### Known issues at handoff

HR-414-N1: a claim whose only evidence is conditional is supported when the
condition shares a content word with the claim ("Are faculty members not
entitled to the housing allowance?" against "... not entitled to the housing
allowance if they receive university housing"; `_c11_own` in `_c11_unasked`
treats the condition as the claim's own content). Crafted reproduction, both
rankings; incidence on the real pages unknown; strict xfail in the
corrections file; recommended as a bounded item for the next step with its
own replay. The 27 conservative misses of the reconstructed set (ten wrapped
true claims, four out-of-scope routings, thirteen strict-rule partials) are
unchanged and documented in `docs/FACULTY_SERVICES_SPECIALIST.md`; F1 / F2
(recall through wrappers) are backlog pending a design confirmation.
Real-model validation outstanding. The Git checkpoint for Steps 4.14-A and
4.14 is pending the team's review (nothing staged or committed).

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_support_gate.py` (six rows),
`tests/test_faculty_services_corrections.py` (new), `qa/hr_4_14_a/` (new:
case files, runner, mutation harness, before / after results, README),
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-100),
`docs/STATUS.md`.

---

## 2026-10-05 - HR specific-claim support gate (Milestone 4, steps 4.12-R11 and 4.12-C12)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; the combined Step 4.12-R12 / 4.13 final audit pending |

### Completed

- Step 4.12-R11 (read-only confirmation of D-098, verdict: bounded
  correction required): 26 materially wrong supported answers (17 real, 9 on
  crafted pages) in six classes, the largest one a specific claim that the
  claim reader could not read (a conversational wrapper or a semantic
  prefix) and that the earlier details then supported.
- Step 4.12-C12-A (validation, scratchpad only): the fail-closed contract
  simulated as an in-memory patch over every replay set under both
  rankings; the 43 supported-to-partial changes classified by four judges
  and four adversarial verifiers (28 correct catches, 15 conservative
  losses; the two changes flagged as over-blocks were examined: a hedge
  exemption that would have fixed one let a hedged statement about another
  appointment support a claim and was not kept, the other is a reader limit
  on a nominalised predicate); no stop condition.
- Step 4.12-C12-B (D-099): one HR-local support-gate owner,
  `_specific_claim_support_gate`, in `faculty_services.py`: the specific /
  general boundary, wrapper normalisation that keeps every semantic
  operator, the gate (no claim, no frame or no statement never supports), a
  ledger with no static operator, the clause, comparison, condition (no
  copular bypass), modifier and alternative-filler rules, a lead split off
  by the Coordinator joined back, and the polarity of a fallback statement;
  fifteen module switches. The implementation reproduces the validated
  prototype byte for byte over every replay set before the three additions
  the C12 adversarial transformations and cross-domain gate called for.

### Tests performed

Full suite 3631 passed (0 failures, 0 errors), `pip check` clean, HR 1147
(the 922 earlier tests with five expectations moved to D-099 and one
asked-condition case added, 923 in all, and 224 new in
`tests/test_faculty_services_support_gate.py`), Teaching focused gate 1009,
Research regressions 1221, Teaching mutations 56 / 56, Research mutations 116 / 116,
historical 34 / 34, HR mutations 304 / 304 (the C11 280 plus 24 D-099
ones; 0 survivors, 0 errors). The 26 R11 wrong answers are partial under both
rankings. The 2,672 HR baseline questions, the C10 and R10 campaigns (143
and 150), the 16 R10 analogs, the 35 DR2 probes, the 279 R11 real and 307
R11 synthetic questions and 128 C11 synthetic questions were replayed under
both stand-in rankings: 43 answers changed, all supported to partial (28
correct catches, 15 conservative losses), none partial to supported; no
general question changed its status, quotes or message; no answer changed
its quoted evidence. The 3,290 distinct real-fixture questions split into
3,011 specific and 279 general ones; no specific question without a claim
or frame and no specific answer not covered by the composite is supported.
Adversarial transformations (38 real-fixture questions combining wrappers
and prefixes with must, may, all, every, always, only, under, over, if,
unless and additional, both rankings): no wrong support. Cross-domain:
`llm_calls` 0, handoffs as data, Institutional pending, no foreign-section
quote. In one process, alternating rounds over 800 baseline questions, the
D-099 rules cost 0.0% (96.69 s against 96.70 s with every D-099 switch
off); the replay wall-clock ranged from 131 s to 180 s across runs against
165 s for C11 at the start of the step (machine load); the HR focused gate
(the nine earlier files) took 72.80 s against 73.09 s.
`PLAN_F_ENABLED=False`; executable specialists Teaching, Research and
Faculty Services & HR. Research files, teaching.py, claim_model.py,
grounding.py, evidence.py, details.py, base.py, coordinator.py and
dispatch.py byte-identical.

### Known issues at handoff

Fifteen true or defensible answers are partial under D-099 (leading hedges,
conditions the question does not ask, a time inferred across clauses, an
anaphoric condition, a stated hedge on an exclusive claim, a nominalised
predicate; see `docs/FACULTY_SERVICES_SPECIALIST.md`); the Milestone 5
Verifier may recover them. Reader limits: a clause that only mentions the
asked object still answers a claim whose verb the returned unit never
states; a wrapper's words are part of the retrieval query. Real-model
validation outstanding. The three-specialist cap of a four-domain question
is unchanged. The combined Step 4.12-R12 / 4.13 final audit not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_support_gate.py` (new),
`tests/test_faculty_services_composite.py` (four expectations moved to
D-099), `tests/test_faculty_services_grounding.py` (one expectation moved
to D-099), `tests/faculty_services_support.py` (the verbatim page-104
committee composition), `docs/FACULTY_SERVICES_SPECIALIST.md`,
`docs/DECISIONS.md` (D-099), `docs/STATUS.md`.

---

## 2026-10-04 - HR statement-level composite claims (Milestone 4, steps 4.12-R10, 4.12-DR2, 4.12-C11-DESIGN and 4.12-C11)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; Step 4.12-R11 confirmation pending |

### Completed

- Step 4.12-R10 (read-only confirmation of D-097, verdict: HR-local design
  review required): four wrong supported answers whose universal, exclusive
  or frequency operator no D-097 dimension evaluated on the sentence that
  answered them.
- Step 4.12-DR2 (design review, read-only) and Step 4.12-C11-DESIGN (design
  only): one statement-level comparison (one predicate statement, one joint
  comparison, one operator ledger, vetoes only from the rest of the returned
  quote) instead of further per-dimension rules; six more probes of the same
  shape.
- Step 4.12-C11 (D-098): `claim_model.py` gains the statement-level schema
  and tables (standard library only; the D-097 claim and statement types
  unchanged); `faculty_services.py` reads each returned unit clause by clause
  into predicate statements and supports a specific question only when one
  statement states the whole claim (one joint comparison of eleven
  dimensions, one operator ledger, vetoes only from the rest of the returned
  quote), as one more detail that decides support and never changes the
  evidence an answer reports; nine module switches, enabled phase by phase
  against an exact replay of the C10 outputs under both stand-in rankings.
- The final replay of every phase found three defects in the new subject
  reading and evidence ranking (an actor read from outside the question's
  subject or from a modifier of the evidence's subject, an actor named after
  a leading antecedent, partial answers reporting other evidence); each was
  corrected within D-098 and every phase was replayed again from the start.
- The ten R10 and DR2 failures: partial with the failed dimension named under
  both rankings.

### Tests performed

Full suite 3406 passed (0 failures, 0 errors), `pip check` clean, HR 922
(the 658 earlier tests and 264 new: 49 in
`tests/test_faculty_services_statements.py`, 215 in
`tests/test_faculty_services_composite.py`), Teaching focused gate 1009,
Research regressions 1221, Teaching mutations 56 / 56, Research mutations
116 / 116, historical 34 / 34, HR mutations 280 / 280 (the D-097 223 plus 57
D-098 ones; 0 survivors, 0 errors; four that stricter rules masked received
direct tests). The 2,672 HR baseline questions, the C10 and R10 campaigns
(143 and 150 questions), the 16 R10 analogs, the 35 DR2 probes and 128
synthetic questions were replayed under both stand-in rankings after each
enablement (the joint comparison; the operator ledger, frequency and safe
leading frames; universal slots; enumerations; alternatives; unconditional
claims; window vetoes): with every switch off the outputs equal C10 byte for
byte; each enablement changed exactly the statuses the design listed for it;
the final state differs from C10 in 15 statuses of real questions, all
supported to partial under both rankings (the four R10 failures, the six
DR2 probes and five true or defensible answers, see below), and in 11
synthetic classes. No answer changed its quoted evidence (the 15 changed
answers keep their findings), and every earlier missing message is kept
with the composite's message added. The replay with every switch on took
124 s against 117 s with every switch off (+6%); the HR focused gate took
64.25 s against 73.49 s at the start of the step. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, evidence.py, details.py, base.py, domain_cues.py, scoping.py,
retrieval_intent.py, coordinator.py, dispatch.py and router.py
byte-identical.

### Known issues at handoff

The statement-level composite (D-098) is strict: five true or defensible
answers are now partial ("Is the end-of-service gratuity always one month's
salary per year of service?", "Is the maximum children's education
allowance 75,000 Dirhams for all children?", "Is the visiting contract
extended with Chancellor approval?", "Is the department chair's appointment
renewable only once?" and a department-level promotions committee question
with "equal to or higher than"). Reader limits: a clause whose verb has no
auxiliary is not split from the clause before it; a role named in a leading
antecedent of the evidence is tolerated rather than read as the actor; a
stated universal with a number in it never answers a universal question,
even one asking the same number. Real-model validation outstanding. Step
4.12-R11 (confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/claim_model.py`,
`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_statements.py` (new),
`tests/test_faculty_services_composite.py` (new),
`tests/test_faculty_services_grounding.py` (one expectation moved to D-098
universal slots), `tests/faculty_services_support.py` (C11 fixture pages),
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-098),
`docs/STATUS.md`.

---

## 2026-10-04 - HR structured claims (Milestone 4, steps 4.12-DR1, 4.12-C10-DESIGN and 4.12-C10)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; Step 4.12-R10 confirmation pending |

### Completed

- Step 4.12-DR1 (HR-local design review, read-only, verdict A) and Step
  4.12-C10-DESIGN (design only, verdict A): a bounded HR-local structured
  claim model for the three claim dimensions the D-096 contracts do not read.
- Step 4.12-C10 (D-097): new `handbook_bot/agents/specialists/claim_model.py`
  (standard library only; imported only by `faculty_services.py`) with the
  claim and statement schema, the operator readers and the compatibility
  tables; HR adapters in `faculty_services.py` reading modality bound to the
  asked action, quantifier scope (universal, generic, existential,
  proportional), exclusivity with its focus, condition roles, fail-closed
  operators and a window veto, each behind a module switch. Enabled phase by
  phase against an exact 2,672-question replay of the C9 outputs under both
  stand-in rankings.
- The 11 known failures: partial with the dimension named under both
  rankings; the 8 paired controls supported.
- Fresh campaign: 143 new questions (modality, scope, exclusivity, condition
  roles, unknown operators, combination attacks, controls) under both
  rankings, every supported answer read (33 questions): 0 materially wrong.
  Against the exact C9 module, 30 C9 supported answers became partial: 25
  were wrong or over-claimed at C9 (four of them wrong in ways this campaign
  first exposed, closed within the D-097 dimensions) and 5 true universal
  answers fell to the strict universal rule; one more C9 answer kept its
  status with a corrected quote (a visiting-contract sentence no longer
  answers a probation question).

### Tests performed

Full suite 3142 passed (0 failures, 0 errors), `pip check` clean, HR 658
(the 297 earlier tests and 361 in `tests/test_faculty_services_claims.py`),
Teaching focused gate 1009, Research regressions 1221, Teaching mutations
56 / 56, Research mutations 116 / 116, historical 34 / 34, HR mutations
223 / 223 (the C9 179 plus 44 D-097 ones; 0 survivors, 0 errors). The 2,672
HR baseline questions replayed under both stand-in rankings after each
enablement (modality; scope; exclusivity; condition roles; operators; the
window veto): with every switch off the outputs equal C9 on all 2,672; the
final state differs from C9 in 18 statuses only (the 11 known failures, "Can
sick leave without pay be granted only in special circumstances?", two
existential negatives now supported, the generic "Are faculty members
entitled to Hajj leave?" and three strict generic narrowings) and, among
answers supported in both, in one primary quote (more focused) and six or
seven secondary findings. The 80-operator set: 65 / 65 kept supported,
10 / 10 now partial, five read by hand. The 23 baseline questions with an
unverified operator are all partial. Cross-domain runs (HR alone and with
Teaching, Research and Institutional, both rankings): 0 flags; handoffs as
data only, no LLM call, Institutional pending, no HR quote from a foreign
section. `PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists
Teaching, Research and Faculty Services & HR; Institutional pending. Research
files, teaching.py, grounding.py, evidence.py, details.py, domain_cues.py,
scoping.py, retrieval_intent.py, coordinator.py, dispatch.py and router.py
byte-identical.

### Known issues at handoff

Strict universal and generic scope (D-097) leave some true answers partial:
"May all emeritus faculty teach courses as needed?" (the handbook states the
rule generically), "Are female faculty eligible for sixty days of maternity
leave with full pay?" (stated for full-time female faculty). Reader limits:
a relative clause with an object in a question's subject, a compound noun
subject whose second noun can be a verb, and "-ent" / "-ment" verbs outside
the HR verb list leave the asked action unread (the clause then fails closed
or keeps its C9 result). The Teaching modal issue stays documented only.
Real-model validation outstanding. Step 4.12-R10 (confirmation) and 4.13
(adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/claim_model.py` (new),
`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_claims.py` (new),
`tests/test_faculty_services_grounding.py` (one R7 expectation moved to
D-097 strict generic scope), `tests/faculty_services_support.py` (C10
fixture pages), `docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md`
(D-097), `docs/STATUS.md`.

---

## 2026-10-04 - HR semantic consolidation (Milestone 4, steps 4.12-R8 and 4.12-C9)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; HR-local design review recommended |

### Completed

- Step 4.12-R8 (read-only confirmation, verdict C): 43 materially wrong
  supported answers in ten classes (scope narrowing, conditional evidence
  read as fact, "Is not any X ...?", compound or coordinated labels read as
  actors, role verbs outside the HR inventory dropped, predicate polarity
  outside the D-094 list, relative-clause polarity, clausal absent
  conditions, condition identity, maximum / minimum qualifier binding);
  42 of them present before C8.
- Step 4.12-C9 (D-096, `faculty_services.py` only): the HR-local
  safeguards consolidated into one claim-compatibility layer with ten
  contracts (scope, conditional fact, negative existence, labels, explicit
  verbs, generic predicate polarity, population modifiers, clausal
  conditions, condition identity, qualifier identity), six of them behind
  a module switch; contracted negation read with either apostrophe; shared
  grounding unchanged.
- Tests: 297 HR tests across the six files (11 new D-096 invariant tests
  with metamorphic pairs; two C8 assertions that let a narrower negation
  settle a broader question were updated).
- The 43 R8 cases: 43 / 43 now `partial` under both stand-in rankings, and
  the HR sub-task of both dispatch repeats `partial`. Every earlier campaign
  replayed against the C8 outputs with the intended changes only. A fresh
  campaign of 137 new questions and six dispatch runs, every SUPPORTED read:
  0 materially wrong; against the exact pre-C9 module (reconstructed from
  the session record, md5 matching the recorded pre-C9 hash) ten C8-wrong
  answers became `partial` and one true negative was narrowed by scope.
- Breadth sets of 56 and 34 questions outside the ten classes: 9 wrong
  supported answers in three claim dimensions the contracts do not read,
  identical at C8 (modal strength, universal quantification / affirmative
  scope, "only" exclusivity). Not patched: an HR-local design review is
  recommended.

### Tests performed

Full suite 2781 passed (0 failures, 0 errors), `pip check` clean, HR 297,
Teaching focused gate 1009, Research regressions 1221, Teaching mutations
56 / 56, Research mutations 116 / 116, historical 34 / 34, HR mutations
179 / 179 (the accepted 150, one of them re-pointed because the generic
verb fallback now keeps an explicit role verb, plus 29 new; 0 survivors,
0 errors). The six Teaching corpora replayed with 0 changed rows; the 45
earlier HR probe sets replayed against the C8 outputs with the 43 R8 cases
now `partial`, two true negative questions narrowed by scope and no other
change. `PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists
Teaching, Research and Faculty Services & HR; Institutional pending.
Research files, teaching.py, grounding.py, domain_cues.py, coordinator.py
and dispatch.py byte-identical.

### Known issues at handoff

HR-local design review recommended (D-096): modal strength ("Must ...?"
settled by "may"; the shared deontic gate accepts any normative marker),
universal quantification and affirmative scope ("Are all faculty members
eligible / entitled ...?"), "only" exclusivity. Real-model validation
outstanding. No personal HR data source or workflow system exists.
Conservative misses as before, plus D-096 scope narrowings ("Does the award
not impact salary?" against "the monthly salary"). Step 4.12-R9
(confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_grounding.py`, `tests/faculty_services_support.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-096),
`docs/STATUS.md`.

---

## 2026-10-02 - HR negative-ask polarity, entitlement-synonym, multi-negation and role-verb correction (Milestone 4, steps 4.12-R7 and 4.12-C8)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R7 (read-only confirmation, verdict C): every C1 to C7
  protection held; four classes surfaced (negative asks settled by the
  affirmative rule, entitlement synonyms read without polarity, a second
  "without" frame swallowed by the pay status, a role-subject verb dropped
  and a policy label read as an actor).
- Step 4.12-C8 (D-095, `faculty_services.py` only): a negative ask needs
  the same negation of the same subject (value, relation, predicate and
  "there is no" existence asks); qualified / receive / get / given / provided / awarded join the
  entitlement family with polarity; each absence frame is read on its own;
  role-subject action verbs are kept; policy, section, chapter and
  guideline labels are not actors.
- Tests: 286 HR tests across the six files.
- Every R7 wrong case replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh campaign of 100 questions and two
  breadth rounds of 44 and 27 on the real fixture under both stand-in
  rankings, every SUPPORTED read, 0 materially wrong.

### Tests performed

Full suite 2770 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 150 / 150 (the accepted 125 plus 25 new), the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. No personal HR data source or workflow
system exists. Conservative misses as before, plus "Does the Chancellor
extend the visiting contract?" and two paraphrased true negative questions
now `partial`. Step 4.12-R8 (confirmation) and 4.13 (adversarial review) not
yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_grounding.py`, `tests/test_faculty_services_specialist.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-095),
`docs/STATUS.md`.

---

## 2026-10-02 - HR negation-scope, evidence-polarity, leave-state and verb-binding correction (Milestone 4, steps 4.12-R6 and 4.12-C7)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R6 (read-only confirmation, verdict C): every C1 to C6
  protection held; five classes surfaced (an absence window wider than its
  clause, missing "need no / require no" frames, affirmative asks settled
  by negated evidence, an unread leave-state phrasing, verbs dropped for
  non-role and "is X verbed" asks).
- Step 4.12-C7 (D-094, `faculty_services.py` only): negation governs its
  own clause; new absence frames; an affirmative predicate is bound against
  negated evidence (a negative rule now gives `partial` with the negation
  named, superseding the 4.11 reading); leave states read in more words;
  explicit verbs kept for "is X verbed (by Y)" and active non-role asks,
  with the agent bound to the actor; relational verbs never stand for an
  action.
- Tests: 278 HR tests across the six files (one 4.11 specialist test
  rewritten to the D-094 contract).
- Every R6 wrong case replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh campaign and a breadth round on the
  real fixture under both stand-in rankings, every SUPPORTED read, 0
  materially wrong.

### Tests performed

Full suite 2762 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 125 / 125 (the accepted 101 plus 24 new), the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. No personal HR data source or workflow
system exists. Conservative misses as before (the two D-093 losses, the
"Does termination for cause require the Chancellor's approval?" retrieval
miss), plus "Is the visiting contract extended by the Chancellor?" now
`partial` (the handbook says "with Chancellor approval"). Step 4.12-R7
(confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_grounding.py`, `tests/test_faculty_services_specialist.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-094),
`docs/STATUS.md`.

---

## 2026-10-02 - HR condition-polarity and residual passive-binding correction (Milestone 4, steps 4.12-R5 and 4.12-C6)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R5 (read-only confirmation, verdict C): every C1 to C5
  protection held with no overblocking; a condition stated as absent was
  settled by quotes stating it as required (six questions), and an
  agentless passive ask was settled through a relation the shared parser
  read from a noun ("Can the probation period be confirmed?").
- Step 4.12-C6 (D-093, `faculty_services.py` only): a negated condition
  needs a stated absence; a leave state keeps its polarity; "without pay"
  stays a pay status; the passive binding runs also through noun relations,
  with a noun form bound only through its "of" phrase, an active object
  never read after a passive auxiliary or a preposition, and relational
  verbs ("includes") stating no action.
- Tests: 266 HR tests across the six files, with a verbatim p63
  confirmation fixture kept out of the shared page list.
- Every R5 wrong case replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh campaign with breadth rounds on the
  real fixture under both stand-in rankings, every SUPPORTED read, 0
  materially wrong.

### Tests performed

Full suite 2750 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 101 / 101 (the accepted 87 plus 14 new), the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. No personal HR data source or workflow
system exists. Conservative misses as before, plus "Does termination for
cause require the Chancellor's approval?" (`not_found` under both stand-in
rankings: the candidate chunks are printed prose lines; "Is Chancellor
approval required for termination for cause?" is supported). Step 4.12-R6
(confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_grounding.py`, `tests/faculty_services_support.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-093),
`docs/STATUS.md`.

---

## 2026-10-02 - HR sameness-binding, personal-status and passive-action correction (Milestone 4, steps 4.12-R4 and 4.12-C5)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R4 (read-only confirmation, verdict C): every C1 to C4
  protection held; sameness evidence was not bound to the asked comparison
  (four promotion questions) and a decision on the asker's own case was
  answered from policy (five questions).
- Step 4.12-C5 (D-092, `faculty_services.py` only): sameness evidence bound
  to the asked subject, comparison and scope, bare "equal" and "at the same
  time" no evidence; grant, approval, acceptance and other decision status of
  the asker's own case fail closed while policy and eligibility questions
  stay answerable; the campaign's passive class ("Can the probation period
  be extended?" quoting the visiting-contract sentence) closed by binding an
  agentless passive verb to the asked subject.
- Tests: 258 HR tests across the six files, with verbatim fixtures for
  p69, p148 B, p149 D, p150 D and p152 kept out of the shared page list.
- Every R4 wrong case replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh campaign and breadth rounds on the real
  fixture under both stand-in rankings, every SUPPORTED read, 0 materially
  wrong.

### Tests performed

Full suite 2742 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 87 / 87 (the accepted 69 plus 18 new), the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. No personal HR data source or workflow
system exists. Conservative misses as before (D-088 base asks, the "earn"
and single-bound salary phrasings, the department committee chair sentence),
plus explicit sameness statements that use other words than the question
("Is the minimum residence period for promotion irrespective of prior years
of service?" stays partial). Step 4.12-R5 (confirmation) and 4.13
(adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_grounding.py`,
`tests/test_faculty_services_specialist.py`, `tests/faculty_services_support.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-092),
`docs/STATUS.md`.

---

## 2026-10-02 - HR universality, qualified-rank and personal-record correction (Milestone 4, steps 4.12-R3 and 4.12-C4)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R3 (read-only confirmation, verdict C): every C1 to C3
  protection held; three older classes surfaced under new phrasings:
  sameness claims settled by topic overlap, "senior" ranks inheriting the
  base salary row, and the asker's own leave usage answered from policy.
- Step 4.12-C4 (D-091, `faculty_services.py` only): sameness and
  independence claims need stated sameness or independence; a qualified
  rank has no salary row ("full professor" the one alias, its category
  stated by the Professor row); personal-record asks fail closed with a
  limitation; hypothetical and policy asks stay answerable.
- Tests: 242 HR tests across the six files (sameness and independence
  frames and evidence, qualified ranks and the alias, personal usage,
  remaining and balance asks against policy asks).
- Every R3 reproduction replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh campaign with breadth probes on the
  real fixture under both stand-in rankings, every SUPPORTED read, 0
  materially wrong.

### Tests performed

Full suite 2726 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 69 / 69, the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. No personal HR data source exists.
Conservative misses as before (D-088 base asks, the "earn" and single-bound
salary phrasings, the department committee chair sentence, "Are senior
lecturers appointed on a definite contract for two years?"). Step 4.12-R4
(confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_tables.py`, `tests/test_faculty_services_grounding.py`,
`tests/test_faculty_services_specialist.py`, `docs/FACULTY_SERVICES_SPECIALIST.md`,
`docs/DECISIONS.md` (D-091), `docs/STATUS.md`.

---

## 2026-10-02 - HR salary-header and named-source frame correction (Milestone 4, steps 4.12-R2 and 4.12-C3)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R2 (read-only confirmation, verdict C): every C1 and C2
  protection held; the salary-table header settled unrelated salary
  propositions, and "Does the Employee Handbook say that ..." and the full
  Benefits Chart name in prepositional frames escaped the named-source guard.
- Step 4.12-C3 (D-090, `faculty_services.py` only): the salary header is
  schema, never a row, a quote or evidence; a salary data row settles only
  its rank's amounts and currency; strong source frames (say / state /
  require / specify / provide that; according to / under / in / per; the
  full chart name) are contents asks; "as per the ..." inside the asked
  proposition stays a mention.
- Tests: 231 HR tests across the six files (header and row predicates,
  header never quoted, header and row grounding, thirteen unsupported salary
  propositions, the strong source frames, mention versus content). The 4.11
  assertion that the header is a table row now states the D-090 contract.
- Every R2 reproduction replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh targeted campaign on the real fixture
  under both stand-in rankings, every SUPPORTED read, 0 materially wrong.

### Tests performed

Full suite 2715 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 60 / 60, the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. Conservative misses as before (D-088
base asks, the department committee chair sentence, the Faculty Handbook
"provides an Employee Handbook" phrasing). Step 4.12-R3 (confirmation) and
4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_tables.py`, `tests/test_faculty_services_specialist.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-090),
`docs/STATUS.md`.

---

## 2026-10-01 - HR actor preservation and named-source correction (Milestone 4, steps 4.12-R1 and 4.12-C2)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R1 (read-only confirmation, verdict C): the Step 4.12
  numeric and leave classes stayed fixed; two bounded classes remained: a
  named actor lost when the shared parser builds no or a degenerate
  relation ("Does the Chancellor form the College Academic Promotions
  Committee?"), and a mentioned-only source answering for its contents
  ("What does the Employee Handbook say about settling in the UAE?").
- Step 4.12-C2 (D-089, `faculty_services.py` only): named roles stay
  demanded; an HR-built relation binds the named actor to the stated
  action and is stated only by an occurrence whose subject or agent is
  that actor; navigation segments demand no actor; the department,
  college and university promotion committees are bound as distinct
  bodies; Employee Handbook and Benefits Chart contents asks fail closed
  with the mention quoted.
- Tests: 219 HR tests across the six files (actor matrix on the
  promotion pages, form / chair collisions, verb-less role asks, committee
  identity, the administrative-role axis, named source detection, Employee
  Handbook and Benefits Chart contents versus mention).
- Every R1 reproduction and breadth case replayed safe; every earlier
  campaign replayed with the intended changes only; a fresh targeted
  campaign on the real fixture under both stand-in rankings, every
  SUPPORTED read, 0 materially wrong.

### Tests performed

Full suite 2703 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 52 / 52, the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. Conservative misses: "Does the Faculty
Handbook say HR provides an Employee Handbook?" is partial (the parser
reads "provides" oddly); the D-088 base-ask misses; the department
committee chair sentence with "The Committee" as subject. Step 4.12-R2
(confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/faculty_services_support.py`, `tests/test_faculty_services_grounding.py`,
`tests/test_faculty_services_specialist.py`, `tests/test_faculty_services_tables.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-089), `docs/STATUS.md`.

---

## 2026-10-01 - HR isolation QA and fact-grounding correction (Milestone 4, steps 4.12 and 4.12-C1)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12 (read-only isolation QA, verdict C): ownership, shared pages
  and shared sections, Teaching / Research / Institutional boundaries,
  source gaps, Coordinator cap and routing all held (0 foreign quote leaks,
  0 silent domain loss); five materially wrong SUPPORTED results were found
  in the HR fact gate on owned pages (probation "six months" from the
  resignation notice, maternity "thirty days" and sick leave "30 days" from
  the additional unpaid tranches, notice "one year" from the probation
  period, the department-level procedure from the college-level heading).
- Step 4.12-C1 (D-088, `faculty_services.py` only): the fact's object must
  measure the asked subject; notice, contract and appointment are demanded
  anchors; a base ask is never settled by a number marked additional, up to
  / maximum, minimum or without / half pay; promotion-procedure asks bind
  the department / college / university level to the item's first-named
  body; headings and list lead-ins carry no level and are never quotes;
  "what is the ... procedure" is a process request; a nominalised HR concept
  or "pay" in a pay-status phrase builds no relation; a dropped relation
  keeps the roles the ask names and HR verbs (chair, form, nominate...)
  keep theirs; "level" is wording.
- Tests: 212 HR tests across the six files (new matrices for
  probation / notice, maternity, sick leave, the demanded anchors, the
  tranche rules, the organisational levels and the heading rule; fixtures
  `P64_APPT`, `P149_HEAD`, `P150`).
- Every Step 4.12 reproduction replayed safe; the 4.12 campaigns and the
  4.11 smoke replayed with the intended changes only; a fresh targeted
  campaign on the real fixture under both stand-in rankings, every
  SUPPORTED read, 0 materially wrong.

### Tests performed

Full suite 2696 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 43 / 43, the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py and
grounding.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. Conservative misses added by D-088: a
base ask against an "up to" entitlement ("Is Hajj leave twenty days?", "Is
sick leave 15 working days?") is partial. Step 4.12 MINORs left for later
steps: general-ask quote order under the stand-in rankings, page-level
section labels on shared pages, the Coordinator cap attaching a dropped
domain's clause through a weak hint, actor-only probation scope, harmless HR
co-selection on "rank" and "pay". Step 4.12-R1 (isolation confirmation) and
4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/faculty_services_support.py`, `tests/test_faculty_services_grounding.py`,
`tests/test_faculty_services_tables.py`, `docs/FACULTY_SERVICES_SPECIALIST.md`,
`docs/DECISIONS.md` (D-088), `docs/STATUS.md`.

---

## 2026-10-01 - Faculty Services & HR Specialist implemented (Milestone 4, steps 4.9 to 4.11)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.9 (read-only audit, passed with source gaps) and Step 4.10 (design,
  passed with minor open questions): HR ownership, boundaries, source
  coverage, routing collisions, reuse, grounding needs, test and mutation
  architecture.
- Step 4.11 (implementation, D-085 to D-087): the Faculty Services & HR
  Specialist (`faculty_services.py`) built on the shared helpers and the
  shared HR cue lexicon added to `domain_cues.py`; section-level ownership
  with explicit exclusions and bounded shared sections; the HR fact gate
  (leave type, pay status, category, administrative role, rank, money
  ranges and qualifiers, extension and stage wording, one local item); the
  salary-table row shape; the compensation list split by role; procedure
  requests at most partial; source gaps fail closed; navigation handed to
  Institutional; the Coordinator's inline HR cue block replaced by the
  shared lexicon; `EXECUTABLE_SPECIALISTS` extended to Faculty Services &
  HR; the package docstring updated. Both open questions resolved: the VCAA
  and the Vice Chancellor stay distinct roles; the chapter 2 preamble (p56)
  is the one extra page (verified with `compile_scope` against the real
  section map), the chapter 4 lead-in needs none.
- Tests: `tests/test_faculty_services_scope.py`, `_specialist.py`,
  `_grounding.py`, `_tables.py`, `_alignment.py`, `_integration.py` and
  the shared fixtures `tests/faculty_services_support.py` (201 tests
  across the six files).
- Implementation smoke on the real fixture (two stand-in rankings, mixed
  questions through the dispatcher): every SUPPORTED read, 0 materially
  wrong SUPPORTED; the R2 to R6 Teaching corpora replayed unchanged.

### Tests performed

Full suite 2685 passed (0 failures, 0 errors; 2484 before this step),
`pip check` clean, Teaching focused gate 1009, Research regressions 1221,
Teaching mutations 56 / 56, Research mutations 116 / 116, historical
34 / 34, HR mutations 31 / 31, 4 shared-page guard fixtures 0 leaks, 0 D-078
route violations. `PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable
specialists Teaching, Research and Faculty Services & HR; Institutional
pending; no Verifier, synthesis, orchestrator or inter-specialist execution.
Research files byte-identical; teaching.py and grounding.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. Source gaps (Milestone 4.5): the
Employee Handbook, the Benefits Chart (vacation days), leave procedures,
payroll, visa and passport renewal, ID cards, parking, HR forms. Safe false
negatives: approval asks whose object stands far before the passive verb,
asks whose verb the text lacks, deontic asks against informational text,
a possessive role before a number, hyphenated percentage ranges, the
unscoped resignation-notice ask (the probation sentence is scoped). Formal
isolation (4.12) and adversarial (4.13) reviews not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py` (new),
`handbook_bot/agents/specialists/domain_cues.py`,
`handbook_bot/agents/coordinator.py` (HR cue block replaced by the shared
lexicon), `handbook_bot/agents/specialists/dispatch.py`
(`EXECUTABLE_SPECIALISTS`), `handbook_bot/agents/specialists/__init__.py`
(docstring), six `tests/test_faculty_services_*.py` files and
`tests/faculty_services_support.py` (new), `docs/FACULTY_SERVICES_SPECIALIST.md`
(new), `docs/DECISIONS.md` (D-085 to D-087), `docs/STATUS.md`,
`docs/AGENT_CONTRACTS.md`, `docs/COORDINATOR.md`.

---

## 2026-10-01 - Teaching hyphenated period normalisation (Milestone 4, temporary steps 4.8T-R5 and 4.8T-C6)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T-R5 (final independent read-only freeze review of D-083): every
  numeric, actor, period, table and historical gate held, except one class:
  a hyphenated period in the question ("per-semester", "per-year") was read
  as no period, so "Is the per-semester workload of visiting faculty 24
  credit hours?" was SUPPORTED by the p91 row and "Does the department chair
  review teaching loads per-year?" by a sentence stating each semester.
- Step 4.8T-C6 (correction, D-084): `frequencies_of` in `grounding.py`
  normalises its text before matching (one call to the existing
  `normalise`), so hyphenated and spaced periods are the same; units stay
  distinct. No change to numeric facts, the relation parser, the
  Coordinator, retrieval, the dispatcher, the contracts or Research.
- Tests: four tests added to `tests/test_teaching_numeric_binding.py`
  (hyphen/spaced equivalence of the reader for per/each/every and every
  unit, unit distinctness, the p91 row and p79 rule with hyphenated
  periods, the chair/dean periods with hyphenated periods).
- Replays: the R5 reproductions PARTIAL; the D-083 cases (p91 pairs, p79
  rule, update/check periods, chair/dean periods) unchanged in spaced and
  hyphenated wording; representative historical replay unchanged; a fresh
  bounded campaign weighted to hyphenated periods, every SUPPORTED read; the
  4 shared-page guard fixtures (0 leaks).

### Tests performed

Focused suites 1009 passed (1005 before this step), full suite 2484
passed (0 failures, 0 errors; 2480 before), `pip check` clean, Teaching
mutations 56 / 56 (the 55 of Step 4.8T-C5 plus the hyphenated ask period
dropped), Research mutations 116 / 116, historical 34 / 34, Research
regressions 1221 passed. `PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`;
executable specialists Teaching and Research; Faculty Services & HR and
Institutional pending; no Verifier, synthesis, orchestrator or
inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Safe false negatives as documented at
Step 4.8T-C5.

### Files changed

`handbook_bot/agents/specialists/grounding.py`,
`tests/test_teaching_numeric_binding.py`, `docs/DECISIONS.md` (D-084),
`docs/TEACHING_SPECIALIST.md`, `docs/STATUS.md`.

---

## 2026-09-30 - Teaching numeric / value-frequency binding (Milestone 4, temporary steps 4.8T-R4 and 4.8T-C5)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T-R4 (final independent read-only freeze review of D-082): every
  earlier class stayed fixed, but a numeric ask was checked word by word
  across the unit: the real workload row "12 credit hours per regular
  semester or 24 credit hours per academic year" supported "24 credit hours
  per semester", and "five office hours per week, spread over a minimum of
  two days" supported "two office hours per week".
- Step 4.8T-C5 (correction, D-083) in the shared grounding module
  (`grounding.py`): `numeric_facts` reads each run of numbers with the
  object it qualifies and the period of its clause (up to the next comma);
  a specific numeric ask must find one local fact with the same values, the
  same object and the asked period (stated after or before the number),
  whose clause names no other subject or category than the asked actor
  (a generic asked subject must stand in that clause), and which does not
  sit in another independent clause without the asked action; an action's
  period must sit in the clause of that action by the asked actor; clauses
  split at ";", ":", a sentence end, "and", "or", "but".
  Ask parser: a role heading a concept phrase is the actor, a concept noun
  after a role is no verb. No change to research.py, the Research tests or
  docs, the contracts, the Coordinator, the dispatcher, the section guard,
  the source scope or the evidence primitives.
- Tests: `tests/test_teaching_numeric_binding.py` (18 tests: the
  fact reader, the four workload-row pairings, the office-hours rule, two
  actions with two periods, values with their objects, an actor with its
  own value and period, a category with its own value, a value in the
  clause of its action, a value of another independent clause, a value of
  a subordinate clause, a generic subject with its own value, a
  coordinated predicate, a period before the number, a period stopping at
  a comma, the parser corrections, table rows, correct facts). One parse case of
  `tests/test_teaching_final_grounding.py` reworded ("course team": the
  coordinator is a lexicon role and now the actor).
- Replays after the correction (real fixture, two rankings): the Step
  4.8T-R4 cases PARTIAL; every earlier set replayed and read; a fresh check
  weighted to several numbers, periods and pairs in one unit; the 4
  shared-page guard fixtures (0 leaks).

### Tests performed

Focused suites, full suite 2480 passed (0 failures, 0 errors; 2462 before
this step), `pip check` clean, Teaching mutations 55 / 55 (the 44 of
Step 4.8T-C4 plus value, object and frequency matched anywhere,
value-object and value-frequency binding disabled, multi-pair cross
product, office-hours value from the days number, semester and year pairs
crossing, action frequency crossing clauses, a period read past a comma, a
fact taken from another independent clause),
Research mutations 116 / 116, historical 34 / 34. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching and Research; Faculty
Services & HR and Institutional pending; no Verifier, synthesis,
orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Safe false negatives: an ask with a
relative clause before its verb, a constructed prose line with three or
more digit cells (read as a table row), and the earlier documented
misses.

### Files changed

`handbook_bot/agents/specialists/grounding.py`,
`tests/test_teaching_numeric_binding.py` (new),
`tests/test_teaching_final_grounding.py` (one parse case),
`docs/DECISIONS.md` (D-083), `docs/TEACHING_SPECIALIST.md`,
`docs/AGENT_CONTRACTS.md`, `docs/STATUS.md`.

---

## 2026-09-30 - Teaching generic-subject / irregular-passive / sentence-scope grounding (Milestone 4, temporary steps 4.8T-R3 and 4.8T-C4)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T-R3 (final independent read-only freeze re-review of D-081):
  every earlier class stayed fixed, but three residual classes reproduced:
  a subject outside the role lexicon fell to the word bag ("Does the
  quality office assess course files?" from the chairs' duty, "Does the
  academic advisor remove the Incomplete mark?" from a calendar row), an
  agentless passive ask with an irregular participle received no relation
  ("Are final grades sent to the dean?"), and the agent rule read the actor
  from another sentence ("Do faculty transfer grades to Banner?").
- Step 4.8T-C4 (correction, D-082) in the shared grounding module
  (`grounding.py`): generic subject phrases bound as the evidence clause's
  subject (never the word bag); a table row settles no ask that names an
  actor; irregular participles in the passive-ask patterns; the agent rule
  scoped to the sentence of the occurrence, with actor pronouns kept and
  patients ("They are transferred", "Students are informed by") excluded;
  ", which" keeps its antecedent. No change to research.py, the Research
  tests or docs, the contracts, the Coordinator, the dispatcher, the
  section guard, the source scope or the evidence primitives.
- Tests: `tests/test_teaching_final_grounding.py` (22 tests: the
  three classes each with negatives and positives, the generic positive
  control, the table-row actor rule with table values kept, the irregular
  participles with their agents, coordinated verbs, actor and object
  pronouns, ", which").
- Replays after the correction (real fixture, two rankings): every Step
  4.8T-R3 case PARTIAL; the Step 4.8T, 4.8T-R, 4.8T-R2 and 4.8T-R3 sets
  replayed, every `supported` result read; a fresh check weighted to
  unknown subjects, table rows with actors, irregular passives, sentence
  scope, pronouns, coordination, direction, receiving, frequency and mixed
  questions; the 4 shared-page guard fixtures (0 leaks).

### Tests performed

Focused suites, full suite 2462 passed (0 failures, 0 errors; 2440 before
this step), `pip check` clean, Teaching mutations 44 / 44 (the 36 of
Step 4.8T-C3 plus generic subject ignored, generic actor word bag, table
row satisfies an actor ask, irregular "sent" / "kept" / "held" and "made"
removed, actor lookup widened to the unit, object pronoun inherits the
actor), Research mutations 116 / 116, historical 34 / 34.
`PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists Teaching
and Research; Faculty Services & HR and Institutional pending; no Verifier,
synthesis, orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Safe false negatives: an agentless
action in the sentence after an instructor duty ("Must I upload the
syllabus on Blackboard?"), an actor ask answered only by a table row, a
generic subject with a light verb, and the earlier documented misses.

### Files changed

`handbook_bot/agents/specialists/grounding.py`,
`tests/test_teaching_final_grounding.py` (new), `docs/DECISIONS.md`
(D-082), `docs/TEACHING_SPECIALIST.md`, `docs/AGENT_CONTRACTS.md`,
`docs/STATUS.md`.

---

## 2026-09-30 - Teaching residual actor / passive / light-verb / frequency grounding (Milestone 4, temporary steps 4.8T-R2 and 4.8T-C3)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T-R2 (final independent read-only re-review of D-080): the
  positional classes stayed fixed, but five residual classes reproduced on
  real handbook sentences: the person asking exempt from role grounding
  ("Must I attend the final examination?" from the student rule), a
  subject-less action given to a role named only as recipient ("Do students
  upload the syllabi on Blackboard?"), a passive ask losing its agent before
  "of" ("Are course files assessed by faculty members of the department?"),
  receive / get / obtain bound to nothing ("Do instructors receive an
  Incomplete grade?"), and frequency units dropped as generic ("five office
  hours per semester" from the per-week rule).
- Step 4.8T-C3 (correction, D-081) in the shared grounding module
  (`grounding.py`): the self role demanded as a grammatical subject (or an
  imperative addressed to the reader); `agent_roles` of a unit; the
  passive-ask agent tail; no object-anywhere fallback; receive / get /
  obtain as actions with recipient, object and source; typed frequency
  units (`frequencies_of`) with day / week / month / year words kept as
  terms; "enter" separated from the publish verb class. No change to
  research.py, the Research tests or docs, the contracts, the Coordinator,
  the dispatcher, the section guard, the source scope or the evidence
  primitives.
- Tests: `tests/test_teaching_residual_grounding.py` (33 tests: the
  five classes each with negatives and positives, the agent-phrase tails,
  receive / get / obtain forms, the frequency pairs, the publish / enter
  separation, the earlier position classes). One position test moved its
  impersonal light-verb positive to PARTIAL by D-081.
- Replays after the correction (real fixture, two rankings): every Step
  4.8T-R2 real-fixture MAJOR case PARTIAL; the constructed reversed
  receive / informed-by cases PARTIAL; the 204-question constructed set,
  the 101-question re-review campaign and a fresh position-weighted check
  (self role, scope, subject-less verbs, agents, receiving, frequency,
  mixed), every `supported` result read; the 4 shared-page guard fixtures
  (0 leaks).

### Tests performed

Focused suites, full suite 2440 passed (0 failures, 0 errors; 2407 before
this step), `pip check` clean, Teaching mutations 36 / 36 (the 25 of
Step 4.8T-C2 plus self-role exemption restored, subject-less action for a
recipient allowed, passive agent truncated before "of" and before "whose",
passive object-anywhere restored, receive recipient and source binding
bypassed, semester and day dropped from frequency units, periodically read
as semester), Research mutations 116 / 116, historical 34 / 34.
`PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists Teaching
and Research; Faculty Services & HR and Institutional pending; no Verifier,
synthesis, orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Safe false negatives: an impersonal rule
with no faculty subject ("Office hours may be held online") now PARTIAL for
a faculty ask; "each semester" against "each term"; a source asked with
"from" but stated only as a destination; the earlier safe misses ("Who
supervises laboratory sessions?", "How many service hours are expected per
week?", "Does the chair receive approval from the dean?", "each year"
against "periodically", the annual evaluation of performance).

### Files changed

`handbook_bot/agents/specialists/grounding.py`,
`tests/test_teaching_residual_grounding.py` (new),
`tests/test_teaching_position.py`, `docs/DECISIONS.md` (D-081),
`docs/TEACHING_SPECIALIST.md`, `docs/AGENT_CONTRACTS.md`, `docs/STATUS.md`.

---

## 2026-09-30 - Teaching subject / position grounding (Milestone 4, temporary steps 4.8T-R and 4.8T-C2)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T-R (independent read-only re-review of D-079): the original
  MAJOR class stayed fixed, but deliberate grammatical-position attacks
  reproduced wrong-actor, wrong-object, reversed-direction and sense
  collisions marked `supported` ("Must faculty attend all theoretical
  lectures?" from the students' rule; "Does the dean review course files?"
  from the chairs' duty; "Does Banner forward grades to Blackboard?"; the
  calendar line "02 - 12 May 15 -25 Final exams" as a grade percentage;
  "absence of material errors" as attendance).
- Step 4.8T-C2 (correction, D-080): relation binding in the shared
  grounding module (`grounding.py`: `RoleLexicon`, `parse_relation`,
  `RelationChecker`), vocabulary-injected and domain-neutral: the asked
  actor, action, adjective, object and destination must be stated in one
  evidence clause; a different named subject disqualifies; direction is
  read from the preposition class; action nouns bind their agent; the
  recipient of an agentless passive and the object of a causative are not
  its agent; a personal pronoun takes its antecedent; a subject that is no
  role ("course files", "office hours") must be the evidence clause's
  subject; the antecedent pair cannot migrate an actor. Teaching side:
  `GROUNDING_ROLES` (chair, council, dean, registrar, registration,
  faculty, lecturer, student, systems, the full institution name ...),
  role concepts checked by role, the attendance alias restricted to class
  or student context, the percentage family typed (a calendar line is no
  percentage, a grade-table line is), the regrade pattern anchored at a
  word boundary ("Are grades ..." is no regrade). No change to
  research.py, the Research tests or docs, the contracts, the Coordinator,
  the dispatcher, the section guard, the source scope or the evidence
  primitives.
- Tests: `tests/test_teaching_position.py` (32 tests: the role lexicon and
  the parser, the seven confirmed wrong-actor cases each with its positive
  control, the attendance sense, the calendar and percentage typing,
  subject / object swaps, passive agents, recipients, a filler in verb
  position, pronoun antecedents, direction by preposition, party asks
  without a party, a light verb after a concept subject, the object read
  before "against"). One alignment test moved its role assertion from
  concepts to the lexicon.
- Replays after the correction (real fixture, two rankings): the 28 original
  phrasings (0 of 56 runs `supported`); the 23 confirmed cases of Step
  4.8T-C (0 `supported`); the 34 targeted limit attacks (all wrong-actor
  cases `partial`; two safe misses unchanged); the 4 shared-page guard
  fixtures (0 leaks); the 85-question Step 4.8T-R campaign and a fresh
  91-question position-weighted check, every `supported` result read.

### Tests performed

Focused suites, full suite 2407 passed (0 failures, 0 errors; 2375 before
this step), `pip check` clean, Teaching mutations 25 / 25 (the sixteen of
Step 4.8T-C plus subject check bypassed, different subject ignored,
direction reversed, dean and chair merged, registrar and registration
merged, antecedent actor migration allowed, attendance alias widened,
percentage accepts calendar rows, relation parse removed), Research
mutations 116 / 116, historical 34 / 34. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching and Research; Faculty
Services & HR and Institutional pending; no Verifier, synthesis,
orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Safe false negatives: a party ask on a
one-vocabulary clause ("Who supervises laboratory sessions?"), a clause
without a teaching cue ("How many service hours are expected per week?"),
a recipient ask not stated as such ("Does the chair receive approval from
the dean?"), "each year" against "periodically", "Who reviews the workload
allocation each year?" against the annual evaluation of performance. An
impersonal rule ("Office hours may be held online") still answers any
actor, by design.

### Files changed

`handbook_bot/agents/specialists/grounding.py`,
`handbook_bot/agents/specialists/teaching.py`,
`tests/test_teaching_position.py` (new), `tests/test_teaching_alignment.py`,
`docs/DECISIONS.md` (D-080), `docs/TEACHING_SPECIALIST.md`,
`docs/AGENT_CONTRACTS.md`, `docs/STATUS.md`.

---

## 2026-09-29 - Teaching conservative-supported alignment (Milestone 4, temporary steps 4.8T and 4.8T-C)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T (read-only audit): confirmed one MAJOR Teaching class, 23
  concrete cases: a specific ask marked `supported` on evidence that was on
  the topic but did not establish the asked proposition, relation, deadline,
  quantity, authority or location. Teaching's Milestone 3 contract required
  relevance plus the explicitly requested detail families only.
- Step 4.8T-C (correction, D-079): a shared, domain-neutral ask gate
  (`handbook_bot/agents/specialists/grounding.py`) injected with the Teaching
  vocabulary; general / specific recognised / specific unknown shapes;
  grounding of the ask's specifics in one local unit (concepts, content
  words by stem with verb classes, a normative marker for a deontic
  proposition, a party with the asked action, a number for a value ask, a
  contents statement for a contents ask; table rows name a specific and
  settle no rule; exclusion phrases state nothing); location evidence
  narrowed to systems, electronic places and locative phrases; a pure place
  or contact clause handed to Institutional Navigation and never answered
  from teaching text, a mixed clause given a strict contact detail; the
  substantive hint passed to the D-078 route helper; the grounding detail of
  a pronoun-initial sentence read with its antecedent. No change to
  research.py, the Research tests or docs, the contracts, the Coordinator,
  the dispatcher, the section guard, the source scope or the evidence
  primitives; base.py still imports only the contracts.
- Tests: `tests/test_teaching_alignment.py` (57 tests: the gate's shapes and
  checks, the five Step 4.8 cases, the fresh yes/no, relation and navigation
  classes, deontic, value, party, location, table-row and antecedent rules,
  positive controls, the mixed contact clause, D-078 unchanged, navigation
  intent). Seven existing Teaching tests moved from the permissive contract
  to the conservative one (documented in place); the D-078 Teaching tests
  now expect a pure contact clause to be handed off.
- Replays after the correction (real fixture, two rankings): 22 of the 23
  confirmed cases are `partial` or handed off; the 23rd ("Who reviews the
  workload allocation each year?") is now `supported` by the annual
  evaluation sentence in which the department chair evaluates performance
  against the workload plan each year, a correct answer. The 66-question
  Step 4.8T audit and a fresh 96-question check were rerun and every
  `supported` result read.

### Tests performed

Focused suites, full suite 2375 passed (0 failures, 0 errors; 2318 before
this step), `pip check` clean, Teaching gate mutations 16 / 16 (a scratchpad
harness: gate bypassed, unknown ask allowed, concept, term, deontic, party,
contents, value, location, navigation, antecedent, table-row, verb-form,
yes/no shape, calendar "May", contact detail), Research mutations
116 / 116, historical 34 / 34.
`PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists Teaching
and Research; Faculty Services & HR and Institutional pending; no Verifier,
synthesis, orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Word-bag grounding does not bind
grammatical positions (a student attendance rule states an instructor
attendance question); a permission asked of a descriptive sentence stays
`partial` ("Can a course be delivered fully online?" from "Courses offered
fully online use an LMS"); "each year" is not stated by "periodically".
These are safe false negatives or Verifier work.

### Files changed

`handbook_bot/agents/specialists/grounding.py` (new),
`handbook_bot/agents/specialists/teaching.py`,
`tests/test_teaching_alignment.py` (new), `tests/test_teaching_specialist.py`,
`tests/test_teaching_integration.py`, `tests/test_clause_retrieval_intent.py`,
`docs/DECISIONS.md` (D-079), `docs/TEACHING_SPECIALIST.md`,
`docs/AGENT_CONTRACTS.md`, `docs/STATUS.md`.

---

## 2026-09-29 - Research integration review and freeze (Milestone 4, step 4.8)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | reviewed; no production change; Research frozen for Milestone 4; not yet committed |

### Completed

- Read-only integration review of the Research & Innovation Specialist with
  the Coordinator, the registry, the dispatcher, the Teaching specialist,
  the handoff contracts and the source scope. 21 representative Research
  cues route to Research; 6 negative controls, 4 compounds (tasks, focus
  clauses, full question preserved, nothing executed by the Coordinator).
  Registry: deterministic lookup by id or string, duplicates and non
  specialists rejected, unknown ids raise, no aliases, no fallback.
  Dispatch: Teaching and Research execute, Faculty Services & HR and
  Institutional stay pending, an unregistered or non-executable Research
  stays pending, a raising specialist yields a redacted `error`, unknown
  specialist ids fail before dispatch, handoffs stay data.
- 51 fresh integration questions through the real Coordinator and
  dispatcher on the real fixture, two Research rankings: 0 route-rule
  violations (D-078 holds in both clause orders, three-clause and
  three-way shapes), no clause lost, no specialist executed unexpectedly,
  0 LLM calls. Output contract inspected on supported, partial, not_found,
  out_of_scope and error results: every quote traceable to its chunk and
  page, routes and relations reported per clause and per finding.
- Isolation rerun identical to Steps 4.5A-D and 4.7 (0 leaks, 66 / 66
  shared-page checks, 107 ok lines). Import hygiene: the shared route
  helper imports only the router and the route names; no specialist
  module loads the Coordinator, orchestrator, synthesis or Verifier.
- Decision: the Research specialist is frozen for Milestone 4. D-076
  (conservative `supported`) and D-078 (retrieval intent per focus clause)
  remain; only regression fixes if later cross-specialist QA exposes a
  genuine issue; its contracts are the reference pattern for Faculty
  Services & HR and Institutional Navigation (D-077).

### Tests performed

Focused suites, full suite 2318 passed (0 failures, 0 errors), `pip check`
clean. Mutation evidence unchanged (Research 116 / 116, historical 34 / 34;
the four D-078 mutations remain in the harness). Production flags unchanged
(`PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`).

### Known issues at handoff

Real-model validation outstanding (no embedding model or reranker locally;
no download attempted; no accuracy claimed). MINOR, unchanged: safe
Research false negatives on the hashed-embedder fixture, some same-domain
compounds not split, "extension" alone is not a contact word, a clause
without cap vocabulary ("Can I supervise five theses") reaches no
specialist and is reported unowned, and "How is the University's share of
licensing income divided?" selects no specialist (no cue matches "licensing
income"). Observed on the Teaching side, outside
the Research freeze: Teaching still uses its Milestone 3 contract, so a
yes/no or open ask can be `supported` on topic-level evidence and a shared
navigation clause ("where is the curriculum committee office") can be
`supported` from policy text while its handoff to Institutional Navigation
is kept; a bounded Teaching alignment with D-076 is a candidate for a later
step or for the Milestone 5 Verifier.

### Files changed

`docs/STATUS.md` only.

---

## 2026-09-29 - Research MAJOR correction: retrieval intent per focus clause (Milestone 4, step 4.7)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; ready for Step 4.8 |

### Completed

- Step 4.6 (formal adversarial QA, read-only, 134 fresh questions, three
  layers) found one MAJOR: a mixed question such as "Who approves the petty
  cash advance and what is the VCRGS phone number?" gives the Research task
  the Coordinator's joined intent `contact`; the specialist retrieved every
  focus clause under that route, whose boost fills the candidate window
  with printed prose lines it rejects, and the answered substantive clause
  came back `not_found`. Reproduced on one clause (intent `policy`:
  `supported`; intent `contact`: `not_found`), 8 of 18 targeted probes,
  1 of 134 campaign questions, 1 of 66 questions of the Step 4.5B review.
  Teaching had the same path (3 of 4 probes with a Blackboard or office
  hours contact clause).
- D-078: the retrieval route is resolved per focus clause with one shared
  deterministic helper, `specialists/retrieval_intent.py`
  (`resolve_clause_retrieval_intent`): a sole focus clause keeps the task
  intent; a clause of a multi-clause task is classified on its own with the
  Coordinator's router; `contact` is kept only by a clause that is itself a
  contact ask; a clause with a substantive half retrieves as `policy`.
  Research and Teaching call it per clause; the clause report and each
  finding carry `route`. No change to the Coordinator, `SpecialistTask`,
  `CoordinatorDecision`, `SpecialistFindings`, handoffs, ownership, the ask
  shape, the relation contract, the section guard or the source scope.
  `base.py` still imports only the contracts.
- Regression tests (`tests/test_clause_retrieval_intent.py`, 38 tests): the
  exact minimised Step 4.6 reproducer on a crowd-out fixture (policy lines
  beside a thirty-line phone directory in scope) plus a guard that the
  fixture still fails under the retired rule; mixed tasks in both clause
  orders, the TTO case, two substantive clauses beside one contact clause,
  the real Coordinator and dispatcher (Research executed, Institutional
  pending, no clause lost, handoffs as data); pure-contact and pure-policy
  controls; the same shapes for Teaching. Import hygiene covers the new
  module (2 tests).
- Mutations: four new (`clause_route_from_task_intent`,
  `teaching_clause_route_from_task_intent`, `contact_route_self_check_removed`,
  `per_clause_classification_bypassed`), all caught; Research harness
  116 / 116; historical 34 / 34.
- Replays after the correction: the minimised MAJOR and 18 mixed probes
  (17 research halves `supported` through dispatch; the remaining one is
  the unsplit compound "What is the maximum research advance and whom do I
  call about it?", `partial` with its evidence, a known MINOR); the 134
  campaign questions (only change: the research-group-awards mixed question
  is now `supported` in both rankings; 966 findings, 0 traceability
  violations; misleading and excluded rankings unchanged); the 66-question
  Step 4.5B review (only change: the VCRGS contact + petty cash pair is now
  `supported`); source isolation and shared-page checks identical to Step
  4.5A-D (0 leaks; page 175, section guard, source bounds unchanged).

### Tests performed

Focused suites, full suite 2318 passed (0 failures, 0 errors; 2278 before
this step), `pip check` clean, Research mutations 116 / 116, historical
34 / 34. `PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists
Teaching and Research; Faculty Services & HR and Institutional pending; no
Verifier, synthesis, orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding (embedding model and reranker not
available locally; no download attempted; no accuracy claimed). Remaining
MINOR findings from Step 4.6: two `supported` results whose condition or
recipient sits in another sentence (conference eligibility; "TTO royalty
share?"), six questions the Coordinator does not route to Research, two
same-domain compounds not split, fixture recall losses under the hashed
embedder ("Who owns my invention?", "How many theses can I supervise at
once?").

### Files changed

`handbook_bot/agents/specialists/retrieval_intent.py` (new),
`handbook_bot/agents/specialists/research.py`,
`handbook_bot/agents/specialists/teaching.py`,
`tests/test_clause_retrieval_intent.py` (new), `tests/test_planf_imports.py`,
`docs/DECISIONS.md` (D-078), `docs/RESEARCH_SPECIALIST.md`,
`docs/TEACHING_SPECIALIST.md`, `docs/AGENT_CONTRACTS.md`,
`docs/COORDINATOR.md`, `docs/STATUS.md`.

---

## 2026-09-29 - Research semantic boundary decision and conservative SUPPORTED contract (Milestone 4, step 4.5A-D)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; ready for Step 4.5B |

### Completed

- Architectural decision (D-076): domain specialists stay deterministic
  evidence specialists; a recognised specific ask is `supported` only with
  a STRONG deterministic relation, an ambiguous relation is `partial`, false
  negatives are preferred to false-positive `supported` results, semantic
  relation verification belongs to the Milestone 5 Verifier and Milestone 4
  does not implement it.
- Relation contract in the Research specialist: a `RelationFrame` read from
  the question (roles, action classes with their objects, purpose,
  receiver, share owner, required thing, permission scope, proposition,
  period, subject / agent / patient positions, prepositional arguments,
  object restrictions, negation); `_relation_ok` proves it in the statement
  that states the answer; the clause report carries `relation` and each
  finding of a recognised ask carries `metadata["relation"]`; an ambiguous
  relation is `partial` with the reason in `missing`.
- Position binding (after the fresh adversarial relation check): a party is
  proved in the position the question gives it; a negated or contrasting
  occurrence states nothing; arguments are stated with their preposition;
  named bodies (IRB, IACUC, Library, Head of ...) are roles; nominate is not
  appoint.
- Milestone 5 Verifier boundary (design only, `docs/AGENT_CONTRACTS.md`):
  inputs, outcomes (verified, downgraded, rejected, retry requested) and the
  rule that only the Verifier may treat an ambiguous relation as
  established. HR and Institutional reuse rule (D-077).
- Tests: `tests/test_research_relation_boundary.py` (new, 271): strong,
  ambiguous, wrong action, wrong role, wrong purpose or beneficiary, general
  controls, position binding, and the rule helpers.

### Adversarial results

- Replay of the 178 Step 4.5A-R failure questions: 4 `supported` under some
  ranking, all correct answers (research committees review grant
  applications; the Board approves major equipment funding).
- Fresh read-only relation check (1069 fresh questions, seven probe
  dimensions, two rankings, real handbook fixture): 29 `supported`
  results, each inspected; 0 MAJOR relation false positives; 4 MINOR
  (a stated relation of looser scope: the outside-authors item for faculty
  book funding, the Dean-approved invoices for the invoice deadline, the
  per-day local per diem for a whole trip); the check was run before and
  after each correction cut, the last run on the final code.
- Recorded false negatives (all `partial` with the evidence quoted) are
  listed in `docs/RESEARCH_SPECIALIST.md`.

### Not done on purpose

No Verifier, no LLM or semantic model inside a specialist, no change to the
Router, retrieval, knowledge base, sources registry, section map,
Synthesis, orchestrator, UI or evaluation; no handoff execution;
`PLAN_F_ENABLED` False, `MAX_LLM_CALLS` 2; only Teaching and Research
executable; no model downloaded, no real-model claim.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows:
`tests/test_research_relation_boundary.py` 271,
`tests/test_research_ask_shape.py` 202, `tests/test_research_specialist.py`
310, `tests/test_research_corrections.py` 415,
`tests/test_research_integration.py` 23, `tests/test_planf_coordinator.py`
143, `tests/test_teaching_specialist.py` 467,
`tests/test_teaching_integration.py` 37, helpers 65,
`tests/test_planf_imports.py` and `tests/test_planf_contracts.py` 91; full
`pytest -q` 2278 passed, 0 failures, 0 errors; `pip check` clean. Scratch
mutation checks: 34 historical and 112 Research mutations caught (100%),
including the relation and position mutations (ambiguous treated as strong,
downgrade removed, action / role / purpose / subject / agent / argument /
patient / object / period / negation / conditioned-role bindings removed,
named bodies removed, heading strip removed, nominate merged). Real-fixture
isolation: unchanged from Step 4.5A-R (no excluded-text leak, shared pages
intact, page 175 handling unchanged).

### Next task

Step 4.5B (not started).

---

## 2026-09-29 - Research completeness root-cause correction (Milestone 4, step 4.5A-R)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; architectural decision required before Step 4.5B |

### Completed

- Root cause located: `ResearchInnovationSpecialist.run()` computed the
  uncovered details only over the details the specialist recognised, so an
  unrecognised ask requested nothing and topic relevance made it
  `supported` (fail-open).
- Root rule (D-074): each substantive segment is `general`,
  `specific_recognised` or `specific_unknown` from its question structure;
  an unknown specific ask gets a requested detail no evidence covers; a
  recognised one is grounded (unmapped words, object head, yes/no subject,
  approval act, purpose, beneficiary) in the answering statement; an
  elliptical clause inherits an unambiguous head's ask; overview questions
  stay `supported`. Contact evidence needs a named body; orphan lettered
  items need a modified context.
- Shared Research ownership lexicon (D-075): `specialists/domain_cues.py`,
  read by the Coordinator, the Research and the Teaching specialists; a
  channel clause that asks a subject question keeps its subject specialist.
- Tests: `tests/test_research_ask_shape.py` (new, 202).

### Open (architectural decision required)

A fresh read-only adversarial check (8 dimensions, 1,136 fresh questions)
reported 88 fail-open, 47 misbinding and 14 other cases; after one
structural tightening, a replay of those 178 questions still found 72
`supported` under at least one ranking (some of them arguably correct). The survivors fall into two classes:
permissive classifier paths (telegraphic yes/no without an auxiliary,
existence forms with scaffolding specifics) and relation misbinding of
recognised asks (the answer statement contains the asked words but states
another relation). Lexical grounding checks co-occurrence, not relations.

### Not done on purpose

No change to the Router, retrieval, knowledge base, sources registry,
section map, Synthesis, Verifier, orchestrator, UI or evaluation; no handoff
execution; no model; `PLAN_F_ENABLED` False, `MAX_LLM_CALLS` 2.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows:
`tests/test_research_specialist.py` 310, `tests/test_research_corrections.py`
415, `tests/test_research_integration.py` 23, `tests/test_research_ask_shape.py`
202, `tests/test_planf_coordinator.py` 143, `tests/test_teaching_specialist.py`
467, `tests/test_teaching_integration.py` 37, helpers 65,
`tests/test_planf_imports.py` and `tests/test_planf_contracts.py` 91; full
`pytest -q` 2007 passed, 0 failures, 0 errors; `pip check` clean. Scratch
mutation checks: 34 of 34 historical and 71 of 71 Research mutations caught
(15 of them for the root rule). Real-fixture isolation: 30 of 30
excluded-text attacks and 66 of 66 shared-page checks without a leak.

### Next task

Architectural decision on relation-level answer checking (semantic check in
the Verifier, a semantic ask classifier, or a narrower `supported`
contract); Step 4.5B only after that decision.

---

## 2026-09-29 - Research & Innovation Specialist correction pass (Milestone 4, step 4.5A)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; design review required before Step 4.5B |

### Completed

- Step 4.5 findings corrected in the Research & Innovation Specialist, the
  Faculty Onboarding Coordinator and the Teaching & Learning Specialist's
  handoff cues: navigation and contact (M4-F6), the modal "may" and
  deadline end points (M4-F7), value, approver, eligibility and timing
  binding (M4-F8), topic-only support (M4-F9), compound questions (M4-F10),
  reachability (M4-F11), vocabulary (M4-F12), detail false negatives and
  list completeness (M4-F13), the reranker length contract (M4-F14);
  decisions D-063 to D-068. M4-F15 (quote-level section label) deferred;
  M4-F1 kept as a synthetic-only regression; M4-F5 tracked.
- Second round after an adversarial verification of the first: per-sentence
  timing binding with the asked subject and action, end points with a
  time-bearing head, recommenders and adjectives never approvers, clause-level
  approval targets, new specific objects, orphan lettered items, yes/no
  property focus, no prefix word matching, broader navigation, clause
  continuation and greeting handling in the Coordinator, navigation-first and
  multi-domain handoffs in Teaching, foreign subjects of a shared clause,
  redaction of spaced, UNC and forward-slash UNC paths, and dispatch keeping
  non-Coordinator tasks pending; decisions D-069 to D-073.
- Tests: `tests/test_research_corrections.py` (new), redaction cases in
  `tests/test_specialist_helpers.py`, updated Research, Coordinator and
  Teaching tests.

### Re-check result (design review required)

A read-only re-check of the second round (six agents, 726 variants): 40 of
45 confirmed repros fixed and 5 partly reproducing (TIM-03, NAV-05, NAV-07,
SAF-3, BIND-07), with 56 new MAJOR paraphrase variants of the same classes
(timing wordings with no recognised detail, approvals of another action,
yes/no properties beside a slot focus, navigation phrasings such as
"where's", click questions outside the cue list, lexicon gaps between the
Coordinator, Teaching and Research). Each round closed its enumerated
repros, but fresh paraphrases keep producing wrong `supported` results: the
specialist decides completeness by recognising the requested detail
lexically, and an unrecognised ask is `supported` on topic relevance
(fail-open). Choosing a fail-closed default, moving detail-coverage checking
to the Verifier, or accepting a documented residual rate is a design
decision outside this bounded pass.

### Not done on purpose

No change to the Router, retrieval, knowledge base, sources registry,
section map, Synthesis, Verifier, orchestrator, UI or evaluation. No handoff
execution. No source ingested, no index rebuilt, no model downloaded.
`PLAN_F_ENABLED` stays False, `MAX_LLM_CALLS` 2. Real-model validation
remains outstanding.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows:
`tests/test_research_specialist.py` 310 passed;
`tests/test_research_corrections.py` 415 passed;
`tests/test_research_integration.py` 23 passed;
`tests/test_planf_coordinator.py` 143 passed;
`tests/test_teaching_specialist.py` 467 passed;
`tests/test_teaching_integration.py` 37 passed; helpers 65 passed;
`tests/test_planf_imports.py` 15 passed; `tests/test_planf_contracts.py`
76 passed; full `pytest -q` 1805 passed, 0 failures; `pip check` clean.
Scratch mutation checks (not committed): 34 of 34 historical and 56 of 56 Research
mutations caught. Real-fixture isolation re-run: 30 of 30 excluded-text
attacks and 66 of 66 shared-page checks without a leak.

### Next task

Design review of the fail-open completeness rule (see the Step 4.5A
report); Step 4.5B independent re-review only after that decision.

---

## 2026-09-28 - Research & Innovation Specialist implementation (Milestone 4, step 4.4)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive |

### Completed

- Shared helpers extracted unchanged from the Teaching & Learning
  Specialist: `specialists/scoping.py` (scope compilation with optional
  extra pages, section guard), `specialists/evidence.py` (quote units and
  quality, quote selection with an opt-in lead-in join, per-unit detail
  coverage), `specialists/details.py` (value-shaped numbers, money, timing,
  limits, configurable authority); one `base.redact_error`. Teaching keeps
  its names and behaviour (467 tests, guard oracle and 34 historical
  mutations unchanged).
- Faculty Onboarding Coordinator: bounded research cues (IACUC, IRB,
  royalties, inventors, commercialisation, licensing, TTO, research office,
  board, units, consultancy and others), the WLAM demotion for
  "research-intensive faculty", and the thesis supervision split by detail
  cue.
- `specialists/research.py`: the Research & Innovation Specialist,
  deterministic and extractive, scope 3.5, 5.4, 8.1 to 8.11, chapter 9
  (with the printed 12.3 and the page-175 preamble), 16.1 to 16.4, shared
  6.4, 7.2, 7.7; section guard on the eleven shared pages; research
  vocabulary; detail families including eligibility, duration, required
  documents, funding source, role and publication type; procedural
  safety; handoffs to Teaching, Faculty Services & HR and Institutional
  Navigation, never executed.
- Dispatch: Research is executable with Teaching; Faculty Services & HR and
  Institutional Navigation stay pending. Teaching-phase integration tests
  pin the Teaching-only executable set.
- Documentation: `docs/RESEARCH_SPECIALIST.md`; Coordinator, contracts,
  Teaching and decisions documents (D-058 to D-062).

### Not done on purpose

No change to the Router, retrieval, knowledge base, sources registry,
section map, Synthesis, Verifier, orchestrator, UI or evaluation. No
handoff execution. No source ingested, no index rebuilt, no model
downloaded. `PLAN_F_ENABLED` stays False, `MAX_LLM_CALLS` 2. Real-model
validation remains outstanding. Known limitations recorded for Step 4.6:
the row-join analogue of Milestone 3 R-1, start-of-window timing counted as
a deadline, unbound limit values, and Coordinator gaps for "chief editor"
and "petty cash" questions.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: helpers 49 passed;
`tests/test_teaching_specialist.py` 467 passed;
`tests/test_teaching_integration.py` 37 passed;
`tests/test_planf_coordinator.py` 143 passed;
`tests/test_research_specialist.py` 310 passed;
`tests/test_research_integration.py` 23 passed; the focused multi-agent
set 1259 passed; `tests/test_planf_*.py` 249 passed; full `pytest -q` 1374
passed, 0 failures; `pip check` clean. Scratch mutation checks (not
committed): 34 of 34 historical and 18 of 18 Research mutations caught.

### Next task

Step 4.5: Research isolation and unit QA.

---

## 2026-09-28 - Final bounded correction pass after the major adversarial QA gate (Milestone 3B, step 3.12A)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | corrections implemented locally; not yet committed; architecture inactive |

### Completed

Step 3.11 (major adversarial QA) passed with no blocker and no major
finding; its minor and cosmetic findings are corrected or deferred here.

- F-1 compound institutional wording: the Faculty Onboarding Coordinator
  recognises "who manages / supports / maintains ... a support, service,
  system, office or unit" and "where is / where can I find" a named
  service (Registrar, admissions, finance, HR, IT services ...); bare
  "who" and "where" are not cues and "office hours" is never a place. The
  Teaching & Learning Specialist hands the same phrasings off and treats a
  "who manages / supports / is responsible for" clause as fully supported
  only when one evidence unit names the responsible party.
- F-2 grading and exam topic anchors ("grading policy / rules / scheme",
  "exam rules / regulations / policy") that imply their base word; bare
  "policy" and "rules" stay generic; research, HR, parking, travel and
  conference wording carries no concept; "e-learning" is the LMS topic.
- F-3 hyphen-like separators are read as spaces for cue and concept
  matching only (Coordinator and specialist); add/drop and e-learning
  are Coordinator teaching cues; questions and quotes stay verbatim.
- F-4 minimum, maximum and fee details need value-shaped numbers (unit,
  percent, currency or an attached limit phrase); years, section numbers
  and identifiers never count.
- F-5 "should" satisfies a requirement question by design, documented and
  tested; quotes keep the original wording.
- F-6 "What are the <subject> requirements?" is a requirement question;
  ownership is unchanged.
- F-7 lettered list items ("a.", "b.", "A.", "B.") are list items.
- F-8 "48 hours of office hours" satisfies an office-hours count; "48
  hours of training" does not.
- F-9 deferred as cosmetic (row boundary inside a paragraph window needs
  the chunker); F-10 two cheap dispatch checks for a decision altered
  after validation; F-11 documented only.
- Documentation: Coordinator and specialist documents, D-052 to D-057.

### Not done on purpose

No change to Router, retrieval, knowledge base, sources registry, section
map, Synthesis, Verifier, orchestrator, UI, evaluation or production
activation. `PLAN_F_ENABLED` stays False, `MAX_LLM_CALLS` 2; the dispatch
seam stays inactive. No source ingested, no index rebuilt. Real-model
validation remains outstanding.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_planf_coordinator.py`
98 passed; `tests/test_teaching_specialist.py` 467 passed;
`tests/test_teaching_integration.py` 37 passed; sources, retrieval scope,
contracts, registry, Coordinator, Teaching and integration together 817
passed; the multi-agent modules 204 passed; full `pytest -q` 947 passed, 0
failures; `pip check` clean. Mutation checks (scratch, not committed):
every correction reverted in isolation makes at least one test fail; the
historical section-guard oracle and mutation set still pass.

### Next task

Step 3.12B: independent freeze re-review of the first specialist pattern.

---

## 2026-09-28 - Coordinator to Teaching & Learning Specialist integration (Milestone 3B, step 3.10)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | integration seam implemented locally; not yet committed; architecture inactive |

### Completed

- `handbook_bot/agents/specialists/dispatch.py`: `dispatch(decision,
  registry)` and `run_question(question, registry)` run the Teaching &
  Learning Specialist on the tasks the Faculty Onboarding Coordinator
  assigns to it and return the decision plus findings (`DispatchResult`
  with per-task `SpecialistRun` records in Coordinator order). Other
  specialists' tasks stay pending and visible; handoff requests are
  preserved, not executed; a raising specialist becomes an `error` finding
  with redacted text; no LLM call, no Synthesis, no Verifier.
- `tests/test_teaching_integration.py`: Teaching-only, single and
  multi-clause focus, Teaching with research, faculty services and
  institutional tasks, non-Teaching questions with a call-counting
  specialist, procedure, not_found, partial, supported, execution error and
  raising specialist, handoff preserved but not executed, Coordinator
  specialist limit, Teaching subtask limit, unregistered specialist,
  no-default-to-Teaching, task identity and order, determinism, and
  architecture inactivity.
- Documentation: Coordinator and specialist documents, package docstring,
  D-051.

### Not done on purpose

No change to the Coordinator, the Teaching specialist, Router, retrieval,
orchestrator, QA, Synthesis, Verifier, contracts, base, registry, knowledge
files or UI. No handoff execution, no Synthesis, no Verifier collaboration.
No new source ingested; `knowledge/sources.json` and
`knowledge/handbook_sections.json` unchanged. `PLAN_F_ENABLED` stays False,
`MAX_LLM_CALLS` 2. Step 3.9F MINOR findings (minimum, maximum and fee
bare-number semantics; "should" versus "must"; "what are the X
requirements"; lettered list items; a quantity false negative) are deferred
to Step 3.11. Real-model validation remains outstanding.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_teaching_integration.py`
25 passed; `tests/test_teaching_specialist.py` 382 passed;
`tests/test_planf_coordinator.py` 73 passed; sources, scope, contracts,
registry, Coordinator, Teaching and integration together 695 passed;
the multi-agent modules 179 passed; full `pytest -q` 825 passed,
0 failures; `pip check` clean.

### Next task

Step 3.11: major QA of the Coordinator to Teaching path, including the
deferred Step 3.9F MINOR findings and, once the models are available
locally, real-model validation.

---

## 2026-09-28 - Teaching specialist final correction pass after the final independent re-test (Milestone 3B, step 3.9E)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | final corrections implemented locally; not yet committed; awaiting Step 3.9F |

### Completed

- `handbook_bot/agents/specialists/teaching.py`: completeness judged
  within one local evidence unit of one finding, never unioned across
  findings, sentences or list items (F-1); quantity requires a number tied
  to a count noun naming the counted subject, with a separate frequency
  family (F-2); professional, faculty and teaching development, training
  modules and workshops, training for faculty and "new faculty" are anchor
  concepts (F-3); list markers are stripped before the antecedent check
  and sibling items or headings are never antecedents (F-4); a requirement
  family for what must be contained, included, provided or required (F-5);
  filesystem paths redacted from error text (F-6). F-7 unchanged and
  deferred as documented. Still deterministic, extractive, no LLM.
- `tests/test_teaching_specialist.py`: the real page-79 layout regression,
  a same-quote union attack, eight cross-finding union attacks with wrong
  pieces scored highest, same-unit positive controls, number-semantics
  negatives, development and training anchors with unrelated-training
  controls, numbered-pronoun antecedent tests, requirement-family tests and
  path-redaction tests. Twenty-four in-process mutations are each caught.
- `docs/TEACHING_SPECIALIST.md` updated; `docs/DECISIONS.md` D-047 to
  D-050.

### Not done on purpose

No frozen module changed (Router, Coordinator, retrieval, orchestrator,
QA, Synthesis, Verifier, contracts, base, registry, knowledge files, UI,
evaluation). No production wiring: `PLAN_F_ENABLED` stays False,
`MAX_LLM_CALLS` 2. No model download: real-model validation is still
outstanding. F-7 (construction-time invariant) left as designed.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_teaching_specialist.py`
382 passed; sources, scope, contracts, registry, Coordinator and
Teaching together 670 passed; the multi-agent modules 179
passed; full `pytest -q` 800 passed, 0 failures; `pip check` clean.
Real-text probes with stand-in ranking: no excluded-section quote, no
fragment or antecedent-less quote, the page-79 part-time question `partial`
with the gap named.

### Next task

Step 3.9F: independent re-test of the final corrections, then real-model
validation once the two models are available locally, then Coordinator to
Teaching integration testing (Step 3.10).

---

## 2026-09-28 - Teaching specialist second correction pass after the independent re-test (Milestone 3B, step 3.9C)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | second corrections implemented locally; not yet committed; awaiting the final independent Teaching re-test (step 3.9D) |

### Completed

- `handbook_bot/agents/specialists/teaching.py`: window-independent
  paragraph attribution in `SectionGuard` with a construction-time id
  invariant (F-A, F-H); a completeness gate separating relevance from
  answerability with eleven requested-detail families and faculty
  categories, plain-language missing entries and detail-first candidate
  ordering (F-B); ownership cues made a superset of the Coordinator's
  teaching cues plus natural phrasings (F-C); grade-table lines with
  textual ranges (F-D); bare "attend" demoted to plain vocabulary (F-E);
  antecedent-dependent sentences quoted with their antecedent or not at
  all (F-F); procedural requests `partial` only with evidence about the
  system or the request (F-G). Still deterministic, extractive, no LLM.
- `tests/test_teaching_specialist.py`: overlap-layout fixtures at several
  heading positions with a character-level consistency invariant; a
  ten-case adversarial completeness matrix where the right evidence is
  absent and wrong-but-related evidence ranks highest, plus positive
  controls; ownership, grade-row, attendance, antecedent, procedure and
  invariant tests. Seventeen in-process mutations (including the old
  overlap rule, a removed detail gate, ignored part-time, authority and
  category checks, narrowed ownership, the old grade-row rule, bare attend
  as an anchor, removed antecedent and procedural gates) are each caught.
- `docs/TEACHING_SPECIALIST.md` rewritten; `docs/DECISIONS.md` D-042 to
  D-046 (D-046 defers the one-intent-per-task concern, F-I).

### Not done on purpose

No frozen module changed (Router, Coordinator, retrieval, orchestrator,
QA, Synthesis, Verifier, contracts, registry, knowledge files, UI,
evaluation). No production wiring: `PLAN_F_ENABLED` stays False,
`MAX_LLM_CALLS` 2. No model download: the embedding model and cross-encoder
remain unavailable locally, so real-model validation is still outstanding.
F-I (one intent per task) is deferred, not fixed.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_teaching_specialist.py`
326 passed; sources, scope, contracts, registry, Coordinator and Teaching
together 614 passed; the multi-agent modules 179 passed; full `pytest -q`
744 passed, 0 failures; `pip check` clean. Real-text probes with stand-in
ranking: no excluded-section quote on any shared page, zero quotable
false allows or rejects against an independent page-text oracle, no leak
at any synthetic overlap layout.

### Next task

Step 3.9D: final independent Teaching re-test, then real-model validation
once the two models are available locally, then Coordinator to Teaching
integration testing (Step 3.10).

---

## 2026-09-28 - Teaching specialist correction pass after the dedicated test gate (Milestone 3B, step 3.9A)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | corrections implemented locally; not yet committed; awaiting the Step 3.9 re-test |

### Completed

- `handbook_bot/agents/specialists/teaching.py`: five evidence gates. New
  `SectionGuard` attributes chunks on shared pages to printed sections and
  rejects excluded-section text (F-1); concept vocabulary with aliases,
  anchor-or-two-concepts relevance rule and a generic-term list (F-2, F-6);
  quote quality rules, sentence units with list-item context, table-row
  detection, quote deduplication (F-3); phrase-level handoff cues,
  positive-cue ownership, unowned clauses not searched (F-4); calendar
  term and year context and filtering (F-5); page-level section-label
  caveat in finding metadata (F-7). `FINAL_K` is no longer applied inside
  the specialist (D-038). Still deterministic, extractive, no LLM.
- `tests/test_teaching_specialist.py`: rewritten around synthetic handbook
  pages chunked by the project's chunker and annotated with the real
  section map; a raw-overlap reranker independent of the specialist's
  normalisation and a scripted reranker for adversarial cases (high-scoring
  wrong candidate, lower-ranked relevant evidence, heading chunks, duplicate
  row and row window, excluded-section text on shared pages, generic-word
  distractors, wrong-year calendar line, morphological variants, ambiguous
  cue words). Mutation checks (guard removed, one-word relevance, heading
  filter removed, dedupe removed, "who approves" cue restored, calendar
  year check removed, table-row rule removed, ownership default removed)
  are each caught by at least one test.
- `docs/TEACHING_SPECIALIST.md` rewritten; `docs/DECISIONS.md` D-035 to
  D-041.

### Not done on purpose

No frozen module changed (retrieval, sources, knowledge base, config,
Router, Coordinator, orchestrator, QA, Synthesis, Verifier, contracts, base,
registry, knowledge files, UI, evaluation). No other specialist. No
production wiring: `PLAN_F_ENABLED` stays False, `MAX_LLM_CALLS` 2. No
model download: the embedding model and cross-encoder remain unavailable
locally, so real-model validation is still outstanding.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_teaching_specialist.py`
236 passed; sources, scope, contracts, registry, Coordinator and Teaching
together 524 passed; the multi-agent modules 179 passed; full `pytest -q`
654 passed, 0 failures; `pip check` clean; a fresh interpreter with sockets
blocked imports the specialist without loading a model.

### Next task

Step 3.9 re-test of the corrected specialist, then real-model validation
once the two models are available locally, then Coordinator to Teaching
integration testing (Step 3.10).

---

## 2026-09-28 - Teaching & Learning Specialist, deterministic version 1 (Milestone 3B, step 3.8)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | implemented locally on the branch; not yet committed; awaiting dedicated testing and review (step 3.9) |

### Completed

- `handbook_bot/agents/specialists/teaching.py`: `TeachingLearningSpecialist`
  (id `teaching`) implementing `Specialist.run(task) -> SpecialistFindings`
  with no LLM call. Scope compiled from the section map for the owned
  sections (1.14; 3.1 to 3.4, 3.7, 3.10 to 3.12; 5.1, 5.2, 5.3, 5.5; 10.6;
  12.7, 12.12 to 12.18; 16.6) into `RetrievalScope(source_ids, page_ranges)`.
  Per-clause scoped retrieval through the shared `gather_candidates`,
  production rerank and gate, extractive findings with exact quotes,
  deterministic status aggregation, handoffs for misrouted or shared
  clauses, procedure-request detection for Blackboard, Banner and MyUOS,
  unapproved-source refusal, redacted error results, dependency injection
  through `TeachingResources` (`from_knowledge_base` available, unused by
  the runtime).
- `tests/test_teaching_specialist.py`: import safety, contract and registry,
  scope compiled from the real map (included and excluded pages, chapter 5
  handled section by section), supported facts, procedure safety, ownership
  boundaries and handoffs, no Coordinator call, boundary-page safety, zero
  evidence, unapproved source, retrieval failure, multi-part tasks,
  deduplication, clause limit, Coordinator compatibility, quote selection.
- `docs/TEACHING_SPECIALIST.md`; `docs/DECISIONS.md` D-030 to D-034; one
  docstring paragraph in `handbook_bot/agents/specialists/__init__.py`.

### Not done on purpose

No orchestrator, QA, retrieval, sources, knowledge base, config, Router,
Coordinator, Synthesis, Verifier, contract, registry, UI or evaluation
change. No other specialist. No production wiring: `PLAN_F_ENABLED` stays
False and `MAX_LLM_CALLS` 2. No date-extractor hook (the calendar row is
retrieved as evidence directly). No new source.

### Source reality

The handbook remains the only indexed document. Blackboard, Banner and
MyUOS procedures are not in it; the specialist returns partial or not found
for them and never generates instructions.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_teaching_specialist.py`
105 passed; the multi-agent, source and scope modules together 303 passed;
full `pytest -q` 523 passed, 0 failures; `pip check` clean; a fresh
interpreter imports the specialist, registry, Coordinator, retrieval,
orchestrator, Verifier and UI modules without loading a model, with
`PLAN_F_ENABLED` False, `MAX_LLM_CALLS` 2 and `CACHE_VERSION` v12.

### Next task

Step 3.9: dedicated adversarial testing and independent review of the
Teaching specialist, then the QA gate that decides whether the specialist
pattern is frozen for the remaining specialists.

---

## 2026-09-28 - Source metadata and scoped retrieval foundation (Milestone 3A)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` (created from `feature/faculty-coordinator`) |
| Starting commit | `1a583f8` |
| Ending state | source-scoped retrieval foundation completed, independently reviewed, corrected and committed on `feature/teaching-specialist`; see Git history for the commit |

### Completed

- `knowledge/sources.json` and `handbook_bot/sources.py`: source registry
  (`SourceRecord`, `SourceRegistry`, `load_source_registry`,
  `find_source_for_path`, `validate_source_registry`, `unregistered_source`).
  One record: `uos_faculty_handbook_2025_26`, repository-relative path.
- `knowledge/handbook_sections.json`: 206-record page-range map of the
  handbook (4 front-matter parts, 16 chapters, 186 level-2 sections),
  derived from the document's table of contents on pages 4 to 13 and
  checked against every chapter and section heading on its listed page.
  `SectionMap.resolve(page)` is deterministic; no page is unmapped; document
  numbering quirks (5.2, 7.8, "12.3" in chapter 9) recorded with notes.
- `annotate_metadata`: every chunk gains `source_id`, `source_title`,
  `source_type`, `source_version`, `chapter`, `chapter_title`, `section_no`,
  `section_title`, `page_section_nos`; old keys and chunk text unchanged
  (13,688 chunks before and after). Called from `build_knowledge_base`;
  an unlisted document gets an `unregistered_...` identity and no sections.
- `handbook_bot/retrieval.py`: optional `RetrievalScope` on
  `gather_candidates` (source ids, chapters, section numbers, section
  prefixes, page ranges). Dense search restricted at the index level with a
  FAISS id selector, lexical search restricted to allowed chunks, empty
  scope returns nothing, `scope=None` unchanged.
- `CACHE_VERSION` v11 to v12 (cache rebuilds once per machine).
- Tests: `tests/test_sources.py`, `tests/test_retrieval_scope.py`.
- Docs: `docs/SOURCE_INVENTORY.md`, `docs/DECISIONS.md` D-021 to D-025.

### Correction pass (same day, after independent review)

- F-1: section and prefix constraints are chapter-consistent; the section
  printed as 12.3 on page 180 (chapter 9) is no longer admitted by
  `section_prefixes={"12"}` or `section_numbers={"12.3"}`; the metadata
  still records it as printed.
- F-3: `find_source_for_path` no longer falls back to file-name matching;
  a same-named file elsewhere is unregistered. A section map is applied
  only when the loaded page count matches its `page_count`; otherwise the
  source identity is kept and section labels are left null with a warning.
- F-4: the scoped dense fallback requires an inner-product index and no
  longer catches `RuntimeError`.
- F-2, F-5, F-7: page-granular section scopes, non-unique printed section
  numbers, strict identity and the single-document runtime are documented
  in `docs/SOURCE_INVENTORY.md`; decisions D-026 to D-029.
- F-8: tests added for all of the above, a global scope invariant across
  scope types, multi-seed brute-force checks, L2 rejection, error
  propagation, empty corpus and single-chunk cases.
- Deferred: F-6 (authority and status vocabulary), content fingerprinting,
  text-level section segmentation, a unique section-record identifier,
  multi-document ingestion.

### Not done on purpose

No Teaching specialist, no other specialist, no Coordinator, orchestrator,
Synthesis, Verifier, UI or evaluation change; `EvidenceResult.doc_id` not
populated; no new source added; no web access. The multi-agent runtime
stays inactive (`PLAN_F_ENABLED` False, `MAX_LLM_CALLS` 2).

### Source reality

The handbook is the only indexed document. It references Blackboard and
Banner at policy level only; no standalone procedural guide for either
system exists locally. Detailed Blackboard or Banner workflow questions must
return not found or partial until verified guides are added.

### Tests performed

See the phase report: targeted tests, all multi-agent tests, full
`pytest -q`, `pip check`, fresh-interpreter imports and a real-handbook
metadata probe, all with `.\.venv\Scripts\python.exe` on Windows.

### Next task

The Teaching specialist on the
handbook's teaching sections with an explicit `RetrievalScope`, returning
findings whose `evidence_quote` is verbatim chunk text and whose
`source_id` and `page` come from chunk metadata.

---

## 2026-09-24 - Plan F Coordinator foundation

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/plan-f-coordinator` (created from `feature/plan-f-core`) |
| Starting commit | `db8f63e` |
| Ending state | Milestone 2 Coordinator foundation completed, independently reviewed and committed on `feature/plan-f-coordinator`; see Git history for the commit |

### Completed

- `handbook_bot/agents/coordinator.py`: `coordinate(question) ->
  CoordinatorDecision` and `analyze(question)`. Deterministic-first on top of
  the existing Router (arbitration off): domain scoring from cue lexicons,
  level (department, college, university), systems (blackboard, banner,
  myuos), complexity (greeting, unknown, simple, multipart, cross_domain,
  journey), specialist selection capped by the Plan F budgets, one task per
  specialist carrying the full question plus `context["focus_clauses"]`,
  `requires_synthesis` by rule. Contact questions are conditional (channel
  lookup: Institutional alone; contact request about a subject: subject
  specialist plus Institutional). Bare Blackboard or Banner questions go to
  Teaching; bare MyUOS selects nobody. Confidence is a heuristic ordinal.
  Uniform metadata shape for every decision. No LLM call; arbitration
  deferred (D-PLANF-018).
- Review corrections applied (F-1 MyUOS owner removed, F-2 conditional
  contact rule, F-3 full question in every task, F-4 uniform metadata,
  F-5 journey no longer forces Synthesis, F-8 wording).
- `tests/test_planf_coordinator.py`: the A to Z behaviour list plus the
  review cases (MyUOS with and without context, contact with a subject,
  task contextual completeness, metadata shape, journey and Synthesis).
- `docs/COORDINATOR.md` (rules as implemented), `docs/DECISIONS.md`
  D-PLANF-015 to 020, one docstring line in `handbook_bot/agents/__init__.py`.

### Not done on purpose (later phases)

No specialist implementation, no retrieval, Synthesis, Verifier or `QAResult`
change, no handoff execution, no orchestration, no UI or evaluation change.
The Coordinator is not imported by the production pipeline; `PLAN_F_ENABLED`
stays False; the Milestone 1 pipeline is unchanged.

### Tests performed

See the phase report: targeted Coordinator tests, all Plan F tests, full
`pytest -q`, `pip check` and fresh-interpreter imports, all with
`.\.venv\Scripts\python.exe` on Windows.

### Next task

Phase 2, knowledge layer (source registry, section map from the handbook
table of contents, chunk metadata, cache version bump, scope filter and
boosts), then the Teaching Agent once the Blackboard and Banner guides are
obtained and reviewed.

---

## 2026-09-24 - Plan F core foundation: contracts, specialist framework, registry

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/plan-f-core` (created from `feature/verifier-tests`) |
| Starting commit | `78cd26d` |
| Ending state | committed on `feature/plan-f-core`; see Git history for the commit |

### Completed

- `handbook_bot/agents/specialists/contracts.py`: `SpecialistId` and
  `FindingStatus` enums; `SpecialistTask`, `Finding`, `SpecialistFindings`,
  `HandoffRequest`, `CoordinatorDecision` dataclasses with validation and
  `to_dict()`. `claim` and `evidence_quote` are separate fields by design.
- `handbook_bot/agents/specialists/base.py`: `Specialist` abstract base
  (`run(task) -> SpecialistFindings`) and `check_findings()`.
- `handbook_bot/agents/specialists/registry.py`: `SpecialistRegistry`;
  unknown ids and duplicates rejected explicitly; no global instance.
- `handbook_bot/config.py`: inactive `PLAN_F_ENABLED` and `PLAN_F_MAX_*`
  budgets (3 specialists, 3 subtasks, handoff depth 1, 1 retry, 5 LLM calls,
  4 agent calls). `MAX_LLM_CALLS` and `MAX_VERIFY_RETRIES` unchanged.
- Tests: `tests/test_planf_contracts.py`, `tests/test_planf_registry.py`,
  `tests/test_planf_imports.py` (fresh-interpreter imports, static import
  guard, budgets, `QAResult` unchanged).
- Docs: `docs/AGENT_CONTRACTS.md`, `docs/DECISIONS.md` (D-PLANF-001..014).

### Not done on purpose (later phases)

No Coordinator logic, no specialist implementation, no retrieval, Synthesis
or Verifier change, no `QAResult` change, no UI or evaluation change. Plan F
is not active; the Milestone 1 pipeline is unchanged.

### Tests performed

See the phase report: targeted Plan F tests, full `pytest -q`, `pip check`
and fresh-interpreter imports, all with `.\.venv\Scripts\python.exe` on
Windows.

### Next task

Phase 2, knowledge layer: source registry (`knowledge/sources.yaml` or JSON),
section map from the handbook table of contents, chunk metadata (doc id,
section, domains, systems, level, scope), cache version bump, scope filter
and boosts in retrieval. Blackboard and Banner guides must be obtained and
reviewed by the team before the Teaching Agent phase.

---

## 2026-09-21 - Verifier agent, committed tests, evaluation metadata

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/verifier-tests` (created from `feature/router-orchestrator`) |
| Starting commit | `6b92261` |
| Ending commit | `dd3103c` (code); this file is committed on top of it |

### Completed

- `handbook_bot/agents/verifier.py`: deterministic Verifier agent. Checks
  every sentence of an answer against every retrieved chunk (the Part 1
  check used the top chunk only). Numbers, phone numbers, e-mail addresses
  and date words must appear in the supporting chunk. Cited pages come only
  from supporting chunks; unsupported claimed pages are dropped; no page is
  ever invented. Rejections carry targeted retry feedback. No LLM call.
- `tests/` (115 tests): router matrix and `number` regression, ambiguity
  and LLM-fallback failure modes, verifier support and citation rules,
  orchestrator paths (greeting, extractors, synthesis, retry, one-retry cap,
  LLM budget, API error, missing key, verifier fallback), compatibility of
  `QAResult`, `answer_question()`, the UI fields and the Pages parser.
  Runs offline: fake models and a scripted LLM client.
- `eval/run_eval.py` records `llm_calls`, `retried`, `sub_questions`,
  `agent_trace` and the router, budget and planner settings.
  `eval/score.py` prints an LLM-call and verifier-decision summary for runs
  that carry these fields; Part 1 run files still score unchanged.
- `pytest.ini`, `pytest` in `requirements.txt`, `.pytest_cache/` ignored.

### Tests performed

```
python -m compileall handbook_bot ui eval app.py tests   -> OK
pytest -q                                                 -> 115 passed
```

Both in the system interpreter and in a fresh virtual environment built
from `requirements.txt`.

Verifier calibration against the golden set (no API): of the 33 cases with
an evidence quote, 30 reference answers were accepted against their own
evidence placed after four distractor chunks, with the expected page cited
in all 30; all 33 were rejected when only foreign evidence was supplied.
The three rejections are reference answers that contain facts absent from
their quote.

### Known issues

- Real-model evaluation is not yet run: the development sandbox blocks
  huggingface.co and api.groq.com (HTTP 403 from the network policy), so
  `python eval/run_eval.py --limit 3` fails at model download. Run it on a
  machine with network access and a `GROQ_API_KEY`, then the full golden
  set, then `python eval/score.py`. No accuracy figure for the Milestone 1
  pipeline exists until then.
- Pre-existing tool issues, not changed here: the count extractor takes the
  first regex match in whichever chunks were retrieved; the contact
  extractor accepts loosely spaced digit runs as phone numbers; the date
  extractor still calls the Part 1 classifier internally. Scope of
  `feature/tool-fixes`.
- `docs/IMPLEMENTATION_SPEC.md` and `docs/WORK_SPLIT.md` are cited by the
  code but are not in the repository yet.
- Milestone 2 (Planner, Evidence agent) is not started; `PLANNER_ENABLED`
  stays off.

### Files changed

```
handbook_bot/agents/verifier.py   new
tests/conftest.py                 new
tests/test_router.py              new
tests/test_verifier.py            new
tests/test_orchestrator.py        new
tests/test_compat.py              new
pytest.ini                        new
eval/run_eval.py                  metadata fields
eval/score.py                     cost summary section
requirements.txt                  pytest
.gitignore                        .pytest_cache/
docs/STATUS.md                    new
```

### Do not redo

Router scoring rules, the LLM-call budget and retry clamp in the
orchestrator, the verifier auto-discovery hook, the Pages parser.

### Next task

1. On a machine with network access: `pip install -r requirements.txt`,
   set `GROQ_API_KEY` in `.env`, run `python eval/run_eval.py --limit 3`,
   then the full set, then `python eval/score.py`. Commit the run file
   under `eval/runs/` and record the numbers here.
2. Open pull requests towards an integration branch: `part2-eval`, then
   `feature/router-orchestrator`, then `feature/verifier-tests`.
3. Commit `docs/IMPLEMENTATION_SPEC.md` and `docs/WORK_SPLIT.md`.

### Blockers

Network access to the model host and the LLM API from the development
environment.

---

## 2026-09-21 - Router, Synthesis, Orchestrator (Milestone 1 foundation)

| Field | Value |
|---|---|
| Developer | Mohammad Abdoljalil |
| Branch | `feature/router-orchestrator` (created from `part2-eval`) |
| Starting commit | `3ab4d58` |
| Ending commit | `6b92261` |

### Completed

- `handbook_bot/agents/types.py`: shared interface contract
  (`RouteDecision`, `SubQuestion`, `EvidenceResult`, `SynthesisResult`,
  `VerifyResult`, `TraceEntry`). Route name `policy_yesno` kept.
- `handbook_bot/agents/router.py`: hybrid Router; all intents scored,
  ambiguity margin, one bounded LLM arbitration call only on a tie, never
  raises. Fixes the Part 1 `number` mis-route (count vs contact).
- `handbook_bot/agents/synthesis.py`: typed wrapper over the Part 1 prompt
  and Groq call; prompt rule 1 no longer caps answers at two sentences.
- `handbook_bot/orchestrator.py`: deterministic control flow, LLM-call
  budget, retry clamped to one, agent trace, verifier auto-discovery with
  fallback to the Part 1 check.
- `handbook_bot/config.py`: agent and orchestration flags.
- `handbook_bot/qa.py`: `answer_question()` delegates to the orchestrator;
  `QAResult` gains `agent_trace`, `llm_calls`, `retried`, `sub_questions`.

### Tests performed

Local test scripts and review rounds; not committed to the repository.

### Known issues at handoff

Verifier and committed tests outstanding (done in the entry above); final
parity run on the real index not recorded.

### Files changed

`handbook_bot/agents/__init__.py`, `agents/types.py`, `agents/router.py`,
`agents/synthesis.py`, `handbook_bot/orchestrator.py`,
`handbook_bot/config.py`, `handbook_bot/qa.py`.
