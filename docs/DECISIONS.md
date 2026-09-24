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
