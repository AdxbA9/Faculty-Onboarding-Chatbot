# Development Status

Handoff log for the Part 2 implementation. Newest entry first. Update this
file at the end of every meaningful work session, before the final push, so
the next developer can start from the branch and commit named here.

Rules: one entry per session; commit hashes are the short hashes shown by
`git log --oneline`; wording stays factual and neutral.

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
