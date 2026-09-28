# Architecture Decisions

Decisions that shape Plan F, recorded so they are not re-argued by accident.
Each entry: what was decided, why, and when to revisit. THIS PHASE DOES NOT
ACTIVATE PLAN F: entries D-PLANF-001 to D-PLANF-012 describe the target
design; only the contracts, base class, registry and inactive budgets exist
in code today.

## D-PLANF-001 Plan F is a findings-based two-layer architecture
- Date: 2026-09-24
- Decision: a Coordinator selects domain specialists; each specialist returns structured findings (claim, evidence quote, source, page, confidence) rather than a final answer; the orchestrator renders or aggregates the findings and the Verifier checks them.
- Why: a specialist that writes the final answer cannot be verified against its evidence, and its prose cannot be combined with another specialist's without losing source boundaries. Findings keep every statement traceable to a source.
- Revisit: if the Teaching Agent gate shows findings add latency without improving verifiability.

## D-PLANF-002 The Coordinator owns routing and decomposition
- Decision: one component decides domains, level, systems, selected specialists and subtasks. It reuses the existing Router for technical intent and adds domain scoring; one structured LLM call is allowed only for ambiguous, multipart or cross-domain questions.
- Why: two competing classifiers would disagree; a deterministic-first Coordinator keeps simple questions at zero LLM calls, as the Router does today.
- Revisit: after domain-accuracy measurement on the labelled set.

## D-PLANF-003 Four specialists, no more
- Decision: `teaching`, `research`, `faculty_services`, `institutional`, centralised in `SpecialistId`.
- Why: they match the handbook's chapter structure and the sources the team can obtain. Every candidate for a fifth (Orientation, Blackboard, Banner, College, Department) either owns no unique decision or is a knowledge source.
- Revisit: only with evidence that a candidate owns decisions none of the four can make.

## D-PLANF-004 Specialists never call each other; the orchestrator executes handoffs
- Decision: a specialist that needs another domain returns a `HandoffRequest`; the deterministic orchestrator validates and executes it.
- Why: direct calls make budgets unenforceable, hide loops, and make traces incomplete.
- Revisit: never; this is a safety property.

## D-PLANF-005 Maximum handoff depth is one round
- Decision: `PLAN_F_MAX_HANDOFF_DEPTH = 1`. Requests produced by the handoff round are recorded, not executed.
- Why: bounded latency and cost; every collaboration case in the candidate set is satisfied by one round.
- Revisit: if reviewed cases show a required second round.

## D-PLANF-006 The orchestrator stays deterministic Python
- Decision: no LLM decides control flow, parallelism, budgets or retries.
- Why: testable, reproducible, and the same reason Milestone 1 rejected an LLM manager.
- Revisit: never.

## D-PLANF-007 One shared, scope-aware evidence retrieval
- Decision: one FAISS index over all approved sources; a hard source-scope filter per specialist; soft system and domain boosts.
- Why: four indexes would quadruple build time and cache size for no retrieval benefit; scope is a filter, not a separate store.
- Revisit: only if benchmarking shows scoped retrieval quality suffers in a shared index.

## D-PLANF-008 Plan F budgets are declared now and inactive
- Decision: `PLAN_F_ENABLED = False` and `PLAN_F_MAX_*` constants exist in `config.py`; the Milestone 1 pipeline keeps `MAX_LLM_CALLS = 2` and `MAX_VERIFY_RETRIES = 1`.
- Why: raising the live budget before the Coordinator exists would change Milestone 1 behaviour and its measured cost for no benefit. Namespacing avoids two meanings of one name.
- Revisit: when the Plan F orchestration path is wired in, `PLAN_F_ENABLED` becomes the switch.

## D-PLANF-009 The Verifier stays independent and gains a findings mode later
- Decision: the existing sentence-level Verifier remains; a later phase adds quote-in-source, claim-supported-by-quote, subtask coverage, citation origin and conflict checks. It trusts no specialist.
- Why: the Milestone 1 rules already prevent invented citations; findings mode extends them without replacing them.
- Revisit: at the Teaching Agent gate.

## D-PLANF-010 Synthesis is used conditionally
- Decision: one supported specialist on a simple question is rendered deterministically without an LLM; several specialists, multipart or partial results go through Synthesis in an aggregation mode that receives structured findings only.
- Why: the common case stays at one LLM call; aggregation is needed only when findings must be combined.
- Revisit: if deterministic rendering reads poorly in acceptance testing.

## D-PLANF-011 Blackboard and Banner are systems and sources, not agents
- Decision: the Teaching & Learning specialist owns them; their guides are ingested as sources with a `system` tag.
- Why: they own no decision; a "Blackboard agent" would be retrieval with a label.
- Revisit: never.

## D-PLANF-012 Orientation is a Coordinator journey mode
- Decision: an orientation question makes the Coordinator select specialists in stage order and Synthesis render an ordered checklist. No Orientation specialist.
- Why: orientation is a sequence over the other domains, not a domain.
- Revisit: only with evidence that orientation owns decisions the Coordinator cannot express.

## D-PLANF-013 Contracts are plain dataclasses with self-validation
- Decision: `contracts.py` uses `@dataclass` with `__post_init__` validation and `to_dict()`, and `str` enums for ids and statuses; no Pydantic, no new dependency.
- Why: matches `agents/types.py`; Pydantic is only a transitive dependency of the Groq SDK and the project does not use it.
- Revisit: if a later phase needs schema export for the Coordinator's structured LLM call.

## D-PLANF-014 QAResult is not extended in the foundation phase
- Decision: no Plan F fields are added to `QAResult` until a runtime component produces them.
- Why: fields with no producer cannot be tested for meaning, and every addition touches `to_dict()`, the UI and the run files.
- Revisit: in the first phase that returns a `CoordinatorDecision` or `SpecialistFindings` through `answer_question()`.

## D-PLANF-015 The Coordinator is deterministic-first and built on the Router
- Date: 2026-09-24
- Decision: `agents/coordinator.py` takes technical intent, the multipart flag and route scores from the existing Router with its LLM arbitration switched off, and adds domain, level, system and complexity classification from cue lexicons. The Router is not modified or duplicated.
- Why: two intent classifiers would disagree; the Router already handles the route vocabulary and its regression suite; the Coordinator only adds what the Router does not know (domains, levels, systems, specialists).
- Revisit: if domain accuracy on the labelled set shows the cue lexicons need a different mechanism.

## D-PLANF-016 Domain, level and system rules are cue lexicons with conditional overrides
- Decision (amended 2026-09-24 after review): each domain has strong cues (1.0) and hints (0.5); a domain is a candidate at 1.0. Level is the most specific of department, college, university. Systems are blackboard, banner, myuos. Contact questions are conditional: a direct channel lookup (phone, fax, e-mail address, extension, "number of the") selects Institutional Navigation alone; a contact request about a subject keeps the subject specialist(s) and adds Institutional. A bare Blackboard or Banner mention selects Teaching, which owns those guides; a bare MyUOS mention selects nobody, because MyUOS is the general portal. Rules are documented in `docs/COORDINATOR.md`.
- Why: transparent, testable and cheap; the vocabulary is routing language, not policy content. The first version selected Institutional alone for every contact route and gave MyUOS to Teaching; review showed both erased or fabricated the subject domain.
- Revisit: after measuring domain accuracy; hints and weights may change without touching the contract.

## D-PLANF-017 One task per selected specialist, full question plus focus, synthesis by rule
- Decision (amended 2026-09-24 after review): candidates are capped at `min(PLAN_F_MAX_SPECIALISTS, PLAN_F_MAX_SUBTASKS)`; each selected specialist receives one task whose `question` is always the complete original question, with the specialist's own clauses in `context["focus_clauses"]` (shared clauses allowed, dropped-domain clauses reported as uncovered, whole question as focus when no clause is its own); `requires_synthesis` is True for two or more specialists or a single-specialist multipart question, never for the journey label alone.
- Why: a clause fragment such as "who approves it" cannot be answered on its own; the full question keeps every task self-contained while the focus tells the specialist its part. The count stays bounded and the Verifier can check coverage against explicit tasks and uncovered clauses.
- Revisit: if the Teaching Agent gate shows specialists need a different focus representation.

## D-PLANF-018 Coordinator LLM arbitration is deferred
- Decision: no LLM call in the Coordinator; `used_llm` is always False.
- Why: the Router must pick exactly one route, so a tie needs a tie-breaker; the Coordinator may select several specialists when domains tie, and the orchestrator, Synthesis and the Verifier reconcile their findings. An arbitration call would only save the cost of an extra specialist run, which is a measurement for the Teaching Agent gate, not a design need today.
- Revisit: at the Teaching Agent gate, with measured specialist-call counts on cross-domain questions.

## D-PLANF-019 Unknown questions select nobody; the Coordinator stays inactive
- Decision: a question with no domain cue and no system gets no specialist, confidence 0.0 and complexity `unknown`, so the future orchestrator can fall back to the Milestone 1 handbook path; a greeting gets no specialist; orientation questions are classified as `journey` but not routed by stage. The Coordinator is not wired into `answer_question`, the orchestrator, the UI or the evaluation runtime, and `PLAN_F_ENABLED` remains False.
- Why: a fabricated domain decision on nonsense would send a specialist to answer something it cannot ground; the handbook path already refuses safely. Activation is a separate, reviewable phase.
- Revisit: when the Plan F orchestration path is built.

## D-PLANF-020 Coordinator confidence is a heuristic ordinal; decision metadata has one shape
- Date: 2026-09-24
- Decision: `CoordinatorDecision.confidence` is a heuristic ordinal signal for ordering and traces (formula in `docs/COORDINATOR.md`); it is not calibrated, not a probability and not a measured accuracy, and no document or report may present it as such. Every decision, greeting and empty question included, carries the same `metadata` keys (`METADATA_KEYS`), with empty values where nothing applies.
- Why: a number without calibration invites false precision in the report; a uniform metadata shape lets the orchestrator and the UI read decisions without special cases.
- Revisit: if routing accuracy is measured and a calibrated score becomes available.

## D-021 One source registry and per-source section maps instead of a heading heuristic
- Date: 2026-09-28
- Decision: documents the corpus may be built from are declared in `knowledge/sources.json` (`handbook_bot/sources.py`); the handbook's chapters and level-2 sections are a page-range map in `knowledge/handbook_sections.json`, derived from the document's own table of contents and checked heading by heading. The loader's heading guess (`section`, "Page N" on almost every page) stays untouched and unused.
- Why: source-scoped retrieval needs a stable source identity and a reliable section label on every chunk; the table of contents is authoritative and machine-checkable, the heuristic is not. Repository-relative paths keep the registry portable across machines.
- Revisit: when a second document is added (a second map file, same schema) or if a finer, level-3 map is needed.

## D-022 Section ranges share boundary pages; a page carries all its sections plus one primary
- Decision: ranges are inclusive and a section may share its first page with the section that ends there. `page_section_nos` lists every section on the page; `section_no` is the first section starting on the page, otherwise the one in progress. Section scoping matches any section present on the page.
- Why: page-granular metadata cannot split a page; excluding boundary pages would lose real content (the academic calendar continues onto page 51, where 1.15 starts; page 222 holds 12.10, 12.11 and 12.12). Recall at the boundary is preferred, the reranker and the Verifier decide relevance.
- Revisit: if chunk-level heading detection is ever added.

## D-023 Document numbering quirks are recorded as printed, chapter comes from page ranges
- Decision: the handbook prints 5.2 and 7.8 twice and numbers the copyright policy in chapter 9 as 12.3. The map keeps the printed numbers with a note on each record; `chapter` is always derived from the chapter page ranges, never from the section number.
- Why: repairing the document's numbering would invent labels that do not appear in the source; deriving the chapter from pages keeps chapter and prefix scoping correct regardless.
- Revisit: only if a corrected handbook edition is issued.

## D-024 Retrieval scope is generic, optional and enforced at the index level
- Decision: `RetrievalScope` (source ids, chapters, exact section numbers, dotted section prefixes, page ranges) is an optional argument of `gather_candidates`. Dense search runs as an exact FAISS search restricted by an id selector over the allowed chunks; lexical search scores only allowed chunks; a scope matching nothing returns nothing. `scope=None` is the previous code path, pinned by a snapshot test.
- Why: post-filtering a global top-k can lose in-scope chunks that rank below the global cut-off, which would turn a scoped search into a silent miss; restricting at the index level cannot. Falling back to the whole corpus on an empty scope would defeat the purpose of scoping and is a hallucination risk, so it is forbidden. The scope names no domain and no specialist: those policies belong to later phases.
- Revisit: if a non-flat index type is adopted (the fallback scoring assumes an inner-product flat index).

## D-025 Cache version v12; no runtime component uses the new metadata yet
- Decision: `CACHE_VERSION` v11 to v12 because cached metadata changed; `EvidenceResult.doc_id` stays unpopulated and the orchestrator, Router, Synthesis, Verifier and UI are unchanged. Chunk metadata alone carries source identity for now.
- Why: the phase adds infrastructure only; wiring it into the pipeline is reviewed separately with the first consumer.
- Revisit: with the first component that passes a scope or reads `source_id`.

## D-026 Section metadata is source-faithful; section retrieval is chapter-consistent
- Date: 2026-09-28 (correction pass after independent review)
- Decision: chunk metadata records section numbers exactly as the document prints them, including the chapter-9 section printed as 12.3. `RetrievalScope` section and prefix constraints accept a printed number only when its leading number equals the chunk's resolved `chapter`, which comes from the chapter page ranges. `chapters={9}` with `section_numbers={"12.3"}` therefore matches nothing, by design; that section is reached by chapter or page range.
- Why: the review showed `section_prefixes={"12"}` admitting intellectual-property text from chapter 9. Rewriting the printed number would invent a label the source does not contain; matching by resolved chapter keeps source fidelity and closes the leak.
- Revisit: never for fidelity; the matching rule may change if a unique section-record identifier is introduced.

## D-027 Section scopes are page-granular and section numbers are not identifiers
- Decision: section constraints restrict pages associated with a section (every section present on the page counts) and do not isolate text on shared pages; `section_no` is metadata as printed and may repeat (5.2, 7.8). Callers prefer `source_ids` plus `chapters`, and page ranges where physical isolation matters.
- Why: 115 of 272 pages carry more than one section; text-level splitting would need reliable heading detection inside pages, which the loader does not provide. Documenting the contract is safer than implying a precision that does not exist.
- Revisit: if chunk-level heading detection or a unique section-record identifier is added.

## D-028 Registered-source identity is strict path identity; incompatible section maps are not applied
- Decision: `find_source_for_path` matches only the file at the registered repository-relative location, in any spelling (the operating system decides when both files exist); a same-named file elsewhere is unregistered. A section map is applied only when the loaded page count and highest page index equal its `page_count`; otherwise the source keeps its identity, chunks get no section labels, and a warning is printed.
- Why: the file-name fallback let any same-named file claim the handbook's identity, and a different edition would have received wrong section labels silently. Length is a cheap, model-free edition check; content fingerprinting is deferred.
- Revisit: when a size or hash field is added to source records.

## D-029 Scoped dense fallback supports inner-product indexes only and never masks search failures
- Decision: the id-selector path is tried first; the direct-scoring fallback runs only when the index cannot take search parameters (`TypeError`, `AttributeError`) and only for an index whose `metric_type` is inner product; any other metric raises `TypeError`. A `RuntimeError` from the index propagates.
- Why: scoring an L2 index as inner product would return misleading similarities; catching `RuntimeError` hid genuine failures behind a fallback.
- Revisit: if a non-flat or non-inner-product index is adopted.
