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

## D-030 The first specialist is deterministic and extractive
- Date: 2026-09-28
- Decision: `TeachingLearningSpecialist` makes no LLM call. Every finding's claim is the quoted handbook text and the quote is an exact slice of a retrieved chunk; source id and page come from chunk metadata. Synthesis remains the only place where user-facing prose is written.
- Why: the specialist is the architecture proof; removing generation from it removes the possibility of invented facts while the contract, scoping, status rules and handoffs are validated. A future reasoning step, if measurements justify one, would sit behind the same quote-validation rule.
- Revisit: at the Teaching QA gate, with measured evidence quality on the real index.

## D-031 Teaching scope is a curated list of sections compiled to page ranges
- Decision: the specialist owns the sections in `OWNED_SECTIONS` (chapter and printed number), compiled from the section map into `RetrievalScope(source_ids, page_ranges)` at construction. Chapter 5 is not taken whole: 5.4 (conferences) is excluded. Ambiguous areas (2.9, 3.5, 3.6, 6.2, 6.3, 12.8 to 12.11, 12.19, 12.20) are excluded and listed in `EXCLUDED_SECTIONS`.
- Why: ownership must follow the text, not chapter titles; excluding an ambiguous area costs recall on a few questions, including one costs leakage of another specialist's material. Compiling from the map keeps the scope in step with the document and fails loudly when a section is missing.
- Revisit: after the Teaching QA gate, section by section, with recorded evidence.

## D-032 Scope eligibility is not evidence; relevance and the gate decide
- Decision: a chunk inside the scope becomes a finding only when it passes `MIN_RERANK_SCORE` (applied to every kept item) and one of its sentences shares at least one stemmed content word with the focus clause; the quote is that sentence, extended by the next when it also matches.
- Why: section labels are page-granular, so an included page may carry a neighbouring section's text. The one-word rule is a simple, documented relevance filter, not a calibrated threshold, and the existing gate is reused rather than a new number invented.
- Revisit: when the cross-encoder's behaviour on scoped candidates has been measured.

## D-033 System procedures are never generated
- Decision: a clause asking for a step-by-step procedure in Blackboard, Banner or MyUOS is at most `partial`; policy evidence is returned, `missing` names the absent procedure and `limitations` states that no procedural guide is indexed. Fact questions about a system ("how do grades reach Banner") may be `supported`.
- Why: the corpus contains policy references only; anything more would be invented.
- Revisit: when verified guides are registered and indexed.

## D-034 Specialists reason over assigned clauses only and never re-route
- Decision: the specialist works on `task.context["focus_clauses"]` (the whole question when absent), classifies each clause with its own small cue rules, hands off a misrouted clause and answers a shared clause while also handing it off. It never calls the Coordinator and never scans the full question for other specialists' clauses.
- Why: re-routing would couple specialist and Coordinator, duplicate handoffs for clauses already assigned elsewhere, and make domain decisions inconsistent.
- Revisit: when handoff execution is built in the orchestrator.

## D-035 Chunk-level section eligibility is decided from the headings printed on the page
- Date: 2026-09-28
- Decision: on an in-scope page that also carries a non-owned section, `SectionGuard` attributes each chunk to a section using the section headings printed on that page (number plus the first two title words, located as printed lines and inside paragraph windows) and the chunk order the chunker emits. Text after a non-owned heading is never quoted. A shared page whose headings cannot be located is rejected whole. The page-range `RetrievalScope` is unchanged and remains the first boundary.
- Why: the Step 3.9 gate showed excluded-section text on boundary pages (3.5 on page 80, 12.8 and 12.9 to 12.11 on pages 221 and 222, 5.4 on 117 and 119) becoming supported Teaching findings. Every heading on the fifteen shared pages is printed as its own line, so attribution is reliable without positional guesswork; conservative rejection covers the case where it is not.
- Revisit: if the chunker or the section map changes, or when a chunk-level section label is added to the shared metadata foundation.

## D-036 Relevance requires one anchor concept or two vocabulary concepts; generic words never count
- Decision: the one-content-word rule is replaced by a small explicit Teaching vocabulary with aliases. A quoted unit qualifies when it shares one anchor concept (syllabus, office hours, teaching load, Blackboard, Banner, LMS, attendance, add/drop, final exams and similar) or two vocabulary concepts (exam, grade, teach, course, class and similar) with the focus clause. Generic terms (policy, process, require, contact, fee, assignment, information, faculty, student, university, semester) carry no concept. No stemmer and no calibrated threshold.
- Why: one shared generic word made unrelated fragments supported ("payment of a fixed fee" for a make-up exam fee question) and `not_found` was unreachable. Aliases replace fragile suffix stripping for the small vocabulary that matters.
- Revisit: with measured false negatives on the real cross-encoder; add vocabulary, do not lower the rule.

## D-037 Quote quality rules and sentence units with list-item context
- Decision: headings, numbered list titles, colon-ending lead-ins, lower-case continuations, prose units under five words, truncated last sentences and printed prose lines are not evidence. Paragraph units are sentences; a sentence inside a list item is quoted with the item's opening; a unit is extended by a relevant following unit that does not open a new item. Table rows are quoted whole. Findings are deduplicated by normalised quote on the same page, so a row and its row window yield one finding. The two-findings-per-clause cap is kept.
- Why: heading fragments ("3.1 Teaching Responsibilities", "4. Office hours") and duplicate row and row-window quotes consumed the two finding slots and displaced usable evidence.
- Revisit: at the Teaching QA gate with real ranking; the cap changes only if a real supported question cannot be represented otherwise.

## D-038 The production FINAL_K cut is not applied inside the specialist
- Decision: all reranked candidates above `MIN_RERANK_SCORE` are examined in rank order; the per-clause finding cap bounds the output.
- Why: most chunks are printed lines that the quote rules never accept; cutting to five before the quality rules left no eligible paragraph for common questions. `FINAL_K` limits answer context in the production pipeline, not evidence eligibility.
- Revisit: if candidate examination becomes a measurable cost on the real index.

## D-039 Ownership needs a positive teaching cue; other-domain cues are phrase-level
- Decision: a clause without a teaching cue is not searched (misrouted clauses are handed off, cue-less clauses are reported as unowned). Faculty-services, research and institutional cues are explicit phrases (annual leave, employment contract, research ethics, "who do I contact about", help desk). Bare "contract", "appointment", "benefits", "funding", "extension" and "who approves" are not cues; approval questions about courses stay with Teaching because the curricula-approval policy is Teaching evidence.
- Why: the gate found spurious shared handoffs on ordinary Teaching questions and cue-less questions ("internship requirements") being searched inside the Teaching scope.
- Revisit: alongside the Coordinator's cue tables when handoff execution is built.

## D-040 Calendar lines carry term and academic-year context and are filtered by them
- Decision: for lines on the Academic Calendar pages the term and year are read from the line or the nearest preceding semester header and recorded in the finding; a line from another academic year than the one asked for (default: the registry version's year) or another term than the one named is not evidence; "classes end" is not evidence for "classes begin". Calendar paragraph chunks are not evidence units. No date extractor.
- Why: the gate selected the next year's "Classes begin for Fall 2026-2027" line first and rows lacked semester context.
- Revisit: when a new handbook edition changes the calendar layout.

## D-041 Section labels on findings are page-level and say so
- Decision: every finding carries `section_label_page_level: True`, `page_section_nos`, `shared_page` and `boundary_guard`; `section_no` remains the page's primary section. No more precise number is fabricated; source and page are authoritative.
- Why: on shared pages the primary label can differ from the quoted text's section (12.18.5 text labelled 12.19).
- Revisit: when chunk-level section labels exist in the shared metadata.

## D-042 Paragraph attribution on shared pages is window-independent
- Date: 2026-09-28
- Decision: inside a paragraph window, text after a heading belongs to that heading's section until the next heading, and text before the first heading belongs to the section that precedes that heading in the page's section order. A window without any heading lies in the section the previous window ended in. The region before a heading is never attributed from state remembered by an overlapping window. `SectionGuard` also verifies at construction that chunk metadata ids match list order.
- Why: the Step 3.9B re-test showed that a heading inside the 50-word paragraph overlap let excluded text before it inherit the owned segment reached by the previous window. Windows overlap by more words than a heading marker, so a heading is never missed, and the preceding-section rule attributes every character the same way in every window.
- Revisit: if the chunker's overlap drops below the marker length or chunk-level section labels arrive in the shared metadata.

## D-043 Relevance and completeness are separate gates
- Decision: relevance (one anchor or two vocabulary concepts) establishes the topic only. When a clause explicitly asks for a detail, the clause is fully `supported` only when a finding contains that detail in the same local unit as a clause concept; otherwise it is `partial` and `missing` names the detail. Families: fee or cost, penalty or consequence, deadline or date, number or frequency, minimum, maximum, approving authority, percentage or range, part-time and full-time, location or system, and the faculty category named in the clause. Evidence covering more requested details is chosen first, within a chunk and among candidates.
- Why: one shared anchor made related but incomplete evidence fully `supported` (the full-time office-hours rule for a part-time question, the Regular Faculty load for a research-intensive question). Separating answerability from topic keeps relevance permissive and support honest without a parser or an LLM.
- Revisit: with real-model evidence quality; add families, do not loosen the local-unit rule.

## D-044 Ownership cues are a superset of the Coordinator's teaching cues
- Decision: the specialist's positive teaching cues include every Coordinator teaching cue plus natural phrasings (consultation time and student consultations, class sections and a section being cancelled, first week of classes, timetables, learners, generative tools). Positive ownership remains required; internship, research grant, salary, visa, help desk and parking questions stay unowned.
- Why: a task the Coordinator assigns to Teaching must not be reported unowned because the specialist's vocabulary was narrower.
- Revisit: whenever the Coordinator cue tables change.

## D-045 Antecedent-dependent sentences and tangential procedure evidence
- Decision: a sentence opening with Such, They, These, This, Those or It is quoted with the sentence before it when that sentence is inside the same allowed span and list item and is not a heading; otherwise it is not quoted. For a procedural system request, evidence counts only when it names the system asked about or shares an anchor concept with the request; otherwise the clause is `not_found`. Grade-table lines with textual ranges ("F Below 60 0.00") are table rows. Bare "attend" is plain vocabulary; classroom attendance is the noun, absence, or "attend" applied to classes, lectures, sessions or exams.
- Why: the re-test found antecedent-less quotes, a tangential evening-course sentence making a MyUOS procedure question `partial`, the F grade row rejected, and conference attendance passing as attendance evidence.
- Revisit: at the Teaching QA gate with real ranking.

## D-046 One intent per task is deferred to Coordinator and orchestration work
- Decision: the Coordinator assigns one intent per task and the specialist applies it to every focus clause. Per-clause intents are not addressed in the specialist.
- Why: changing it requires Coordinator and orchestration design, outside the specialist's file boundary.
- Revisit: during Coordinator to Teaching integration (Step 3.10).

## D-047 Requested details are satisfied only within one local evidence unit
- Date: 2026-09-28
- Decision: a clause is fully `supported` only when ONE local evidence unit (a list item, a table row or a sentence) of ONE finding contains every requested detail together with a clause concept. Details found in different units, different findings, different list items or different faculty categories are never combined. `missing` says when a detail is present somewhere but not stated together with the rest.
- Why: the Step 3.9D re-test showed the union of "part-time" from one page-79 item and "five hours" from the other making a part-time office-hours question fully `supported` on the real handbook.
- Revisit: only to add qualifier families; the one-unit rule stays.

## D-048 A quantity is a number tied to a count noun that names the counted subject
- Decision: for "how many", "how much", "number of" and "how long", the evidence unit must contain a number followed by a count noun (hours, credit hours, days, weeks, students, courses, classes, sessions, times, percent, points and similar) whose phrase shares a concept with the counted subject of the question, or its head noun when the subject carries no concept. "How often" is a separate frequency family (weekly, per week, N times). Room, page and section numbers, dates, list markers and counts of something else do not count.
- Why: any number in a relevant sentence satisfied the quantity check.
- Revisit: with real-model probes on numeric questions.

## D-049 Requirement questions need an obligation statement; development and training are anchors
- Decision: questions asking what must be contained, included or provided, what is required or what the requirements are, are answered only by a unit that expresses an obligation (must, shall, required, responsible for, expected to) about the subject. Professional, faculty, teaching or instructional development, development training, training modules, workshops or programmes, training for or of faculty, and "new faculty" are anchor concepts; bare "training" stays plain, so security, safety or compliance training is not faculty-development evidence.
- Why: topic-only questions were `supported` by topically related lines that answered nothing, and sections 3.7 and 5.2 questions were `not_found` by gate rejection.
- Revisit: at the Teaching QA gate.

## D-050 List markers are stripped before the antecedent check; error text hides paths
- Decision: a numbered or lettered list item that opens with a context-dependent word is quoted with the list's lead-in or the preceding plain sentence, never with a sibling item or a heading, and not at all when no such antecedent is inside the allowed span. Error messages in results have key-shaped secrets and filesystem paths (Windows drive paths, home and system trees, other absolute paths) replaced. The section guard's construction-time invariant is unchanged (F-7 stays deferred as documented).
- Why: "2. These syllabi …" bypassed the antecedent rule, and an index failure message carried a local path.
- Revisit: none planned.

## D-051 Coordinator to specialist dispatch is a minimal, inactive seam
- Date: 2026-09-28
- Decision: `handbook_bot/agents/specialists/dispatch.py` takes the Coordinator's `CoordinatorDecision`, walks its tasks in order, runs each task whose specialist is executable in the current phase (`EXECUTABLE_SPECIALISTS`, Teaching only) through the specialist registry, and records every other task as pending with the Coordinator's task kept intact. Tasks are never reinterpreted, reordered or reconstructed; handoff requests are preserved but not executed; a raising specialist yields an `error` finding with secrets and paths redacted; no LLM call, no Synthesis, no Verifier. The production runtime does not import the module and `PLAN_F_ENABLED` stays False.
- Why: the Coordinator and the Teaching specialist were validated separately; the smallest seam that proves decision to task to findings, without a general framework, keeps both validated components unchanged and leaves handoff rounds, Synthesis and Verifier collaboration to their own milestones.
- Revisit: when the next specialist becomes executable (extend `EXECUTABLE_SPECIALISTS`) and when handoff execution is designed.

## D-052 Compound institutional wording is a cue-coverage correction, not a routing redesign
- Date: 2026-09-28
- Decision: the Coordinator's institutional cues gain "who manages / supports / maintains / administers / runs / provides ... a support, help desk, service, system, portal, office, unit, department or centre" and "where is / where can I find" a named service (Registrar, registration, admissions or finance office, HR, IT services, reception, security, bookstore, cafeteria, parking); "office hours" is excluded from the place nouns and bare "who" or "where" are never cues. The Teaching specialist recognises the same phrasings as institutional handoff cues, and a "who manages / supports / is responsible for" clause carries a "responsible party or unit" detail: it is fully supported only when one evidence unit names a responsible party with a responsibility verb form ("The IT department administers the LMS"). Only the Coordinator-provided focus clauses are classified.
- Why: "Who manages Blackboard support and what is my teaching load?" and "Where is the Registrar and what is the grading policy?" reached Teaching whole and the non-teaching half was dropped without a trace (Step 3.11, F-1). A shared clause is still searched by Teaching because it names a Teaching system; the detail rule and the handoff make the gap explicit instead of silent.
- Revisit: when Institutional Navigation becomes executable and handoff rounds are designed.

## D-053 Grading and exam topic anchors imply their base word
- Date: 2026-09-28
- Decision: "grading policy / rules / scheme / criteria / regulations / procedures / guidelines" join the grading-system anchor and "exam(ination) policy / rules / regulations / procedures / guidelines / conduct / instructions" form the exam-policy anchor. Each anchor implies its base word (grade, exam) as plain vocabulary of the clause, and a text that uses the base word shares the anchor's topic (`_ANCHOR_IMPLIES`). Bare "policy", "rules" and "regulations" remain generic. "e-learning" is read as the LMS topic.
- Why: the Coordinator routes "grading" and "exam" questions to Teaching, but the relevance gate needed one anchor or two concepts and returned `not_found` for "What is the grading policy?" and "What are the exam rules?" (Step 3.11, F-2). The implied base keeps the rule "one anchor or two concepts" while letting a topic question be answered by sentences that do not repeat the topic phrase; research policy, HR policy, parking rules, travel rules and conference regulations still carry no concept.
- Revisit: if real-model validation shows the implied base admits off-topic grade or exam sentences.

## D-054 Hyphen-like separators are normalised for cue and concept matching only
- Date: 2026-09-28
- Decision: before cue matching in the Coordinator (`cue_text`) and before concept and cue matching in the Teaching specialist, hyphen-like separators between word characters (`-`, Unicode hyphens and dashes, `/`) are read as spaces, one character for one, so positions do not shift. The question, its clauses and every quote stay verbatim. The Coordinator gains add/drop ("add/drop", "add-drop", "add drop", "add and drop") and e-learning as teaching cues; the academic calendar's add/drop lines and the LMS policy are Teaching-owned.
- Why: "office-hours" and "peer-observation" were not routed at all, and "add/drop" and "e-learning" had no cue (Step 3.11, F-3). Normalising the classification text is smaller and safer than duplicating every multi-word cue in hyphenated form.
- Revisit: none planned.

## D-055 Minimum, maximum and fee details require value-shaped numbers
- Date: 2026-09-28
- Decision: a number satisfies a minimum or maximum detail only when it is followed by a count noun, a percent sign or a currency, or is attached to the limit phrase ("not exceed 40"); a year, a dotted section number and an identifier ("room 204", "examination 101", "page 12") are never values, and a table row's numeric cells are its values. A fee or cost detail needs a monetary value: a currency next to a number, an explicit "amount / price / fee of N", or "free of charge". The quantity family additionally accepts "N <unit> of <subject>" when the "of" phrase names the counted subject ("48 hours of office hours", not "48 hours of training").
- Why: "The minimum is described in section 3.4", "a maximum in 2024" and "fee applies to examination 101" satisfied the families on synthetic sentences (Step 3.11, F-4), and "48 hours of office hours" did not satisfy the quantity family (F-8). The same value principle already applied to quantities is extended, with positive controls kept ("minimum of five office hours per week", "maximum of 20 students", "fee of AED 100", "cost is 100 AED", "90 percent maximum", "not exceed two months' basic salary").
- Revisit: if the approved source gains a fee table without currency words.

## D-056 "Should" satisfies a requirement question by design
- Date: 2026-09-28
- Decision: the obligation check for requirement questions ("what must ... include", "what are the <subject> requirements") accepts must, shall, required, responsible for, expected to and should. Quotes keep the original wording; "should" is never rewritten as "must", and no deontic logic is built. The requirement family also recognises "What are the <subject> requirements?" with a subject of up to four words; requirement detection and ownership stay separate ("What are the research requirements?" is handed off).
- Why: the handbook states many of its requirements with "should" (Step 3.11, F-5, F-6); refusing them would turn real requirements into `partial`. The choice is recorded so it is deliberate, not accidental.
- Revisit: only if Synthesis needs to distinguish advisory from mandatory wording.

## D-057 Lettered items, deferred row boundary, minimal dispatch checks, page-level labels
- Date: 2026-09-28
- Decision: lettered list markers ("a.", "b.", "A.", "B." after punctuation and before a capital) open list items exactly like numbered markers: the marker stays with its item, a lowercase marker is not a dangling fragment, and a pronoun-opening item never borrows its sibling as antecedent (F-7). The page-224 bullet that flows into a new capitalised sentence without punctuation stays joined (F-9, cosmetic, deferred): the paragraph chunk carries no line boundaries and a capital-letter rule would split proper nouns. The dispatch seam adds two cheap checks for a decision altered after validation, a non-task subtask (`TypeError`) and a repeated task id (`ValueError`), and nothing else (F-10). `section_no` stays page-primary metadata with `page_section_nos` and `boundary_guard` carrying the shared-page truth (F-11).
- Why: Step 3.11 findings F-7, F-9, F-10 and F-11; each is either a local safe rule or explicitly deferred with its reason.
- Revisit: F-9 when the chunker exposes row boundaries inside paragraph windows.
