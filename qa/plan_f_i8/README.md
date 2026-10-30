# Full end-to-end application QA of the integrated Plan F runtime (runtime integration, phase I-8)

Application-level evaluation of the integrated runtime on the real knowledge base, through the real
seam (`ui/pipeline.py`) and the real chat page: Coordinator, routing refinement, the four frozen
specialists, retrieval and reranking, the I-1 guards, the findings-aware verifier, the aggregator,
bounded handoffs, the legacy Verifier, the legacy fallback, source presentation, the Developer panel,
the three operating modes, diagnostics hygiene, security and performance. The goal was to decide
whether the runtime is correct, safe, stable and presentable enough to freeze in I-9.

| File | Purpose |
|---|---|
| `question_set.py` | the I-8 master evaluation set: 192 unique questions with domain, test category, tags, the behaviour a correct runtime may show (`serve`, `fallback`, `either`), the handbook fact and page; the routing owners of cross-domain and ambiguous questions; the determinism subset (24) and the LEGACY sample (30) |
| `master_eval.py` | the real-KB run through `pipeline.ask_with_diagnostics`: every question in ACTIVE (with per-item citation checks), the Coordinator decision, routing refinement and adapter attempt on the same runtime object (DIRECT), every question in SHADOW compared field by field with the direct legacy call, the determinism subset three more times, the LEGACY sample in process; switches set in process only, no `.env`, no model call |
| `legacy_check.py` | LEGACY isolation in a fresh process: the sample through the seam with both switches off; asserts the minimal diagnostics record, no Plan F module imported, no runtime cache, no exception |
| `review_served.py` | the evidence review sheet of every served answer (clauses, routing, handoff, per-clause aggregation coverage, the answer as served, every evidence item with its citation checks, chunk and page text) |
| `review.json` | the manual verdict of every served answer (CORRECT / LIMITED / WRONG, citation OK / DEFECT, note) |
| `analyze.py` | coverage and precision separately, false acceptances and false fallbacks, per-domain metrics, routing classification, aggregation, handoffs, fallback reasons, SHADOW invariance, determinism, latency, specialist-call efficiency, model calls, diagnostics hygiene, citation checks, paraphrase and typo robustness, known regression cases; writes `metrics.json` |
| `compare_runs.py` | question-by-question comparison of two recorded runs (used to confirm the two I-8 fixes changed only the two questions they were meant to change) |
| `results_2026-10-30.json` | the recorded final run (the production chunk text of each evidence item is dropped; quote, chunk index, page and the check flags are kept) |
| `metrics.json` | the metrics of the recorded final run |

Run from the repository root with the project interpreter:

    python -X utf8 qa/plan_f_i8/master_eval.py . <output dir>
    python -X utf8 qa/plan_f_i8/legacy_check.py .
    python -X utf8 qa/plan_f_i8/review_served.py . <output dir>/i8_results.json [ids...]
    python -X utf8 qa/plan_f_i8/analyze.py <output dir>/i8_results.json qa/plan_f_i8/review.json
    python -X utf8 qa/plan_f_i8/compare_runs.py <before>/i8_results.json <after>/i8_results.json

## Master evaluation set

192 unique questions (the brief's target was about 120 to 180; 75 keep the exact wording of the
I-3 to I-6 corpora, 117 are new). By domain label: Teaching & Learning 33, Research & Innovation
39, Faculty Services & HR 40, Institutional Navigation 40, cross-domain and same-domain multipart
30, no domain 10. By test category: numeric 31, direct 4, conditional 14, authority 13,
contact/location 21, enumeration 4, same-domain multipart 12, cross-domain 10, ambiguous 8,
unsupported 5, general 12, paraphrase 7, typo or fragment 11, adversarial 40. The adversarial
subset is 40 questions by category and 98 by tag (subject binding 5, entity binding 7, numeric 9,
page mismatch 1, negation 7, conditions 8, partial answers 8, enumeration completeness 4,
conflicting clauses 2, misleading evidence 4, routing ambiguity 10, handoff 5, unsupported
synthesis 3, general HR 5, cross-domain confusion 2, duplicated facts 6, wrong contact 9,
near-match terminology 19, injection 1, comparison 1). Ten paraphrase groups and 13 imperfect
inputs (misspellings, fragments, casual phrasing, a lone `?` and whitespace). Expected behaviour:
serve 58, fallback required 58, either 76.

For the per-domain tables the analysis groups questions by what they test: teaching 24, research
29, HR 25, institutional 25, cross-domain 22, adversarial/unsupported 67 (every question whose
category is adversarial, general, ambiguous or unsupported, or that has no domain).

## Recorded run (2026-10-30, CPU, Windows, final local tree)

Knowledge base from the v12 cache in 22.9 s (272 pages, 13,688 chunks); no model client
configured, so the legacy model path returns its no-key reply; Plan F made 0 model calls. The full
run (ACTIVE, DIRECT, SHADOW, 3 x 24 repetitions, LEGACY sample) took 935 s.

| Metric | Value |
|---|---|
| Total unique questions | 192 |
| Plan F served | 55 |
| Fallback | 137 |
| Coverage | 28.6% |
| Correct served | 55 |
| Limited served | 0 |
| Wrong served | 0 |
| Precision | 100.0% |
| False acceptances | 0 (2 in the first run, both fixed in I-8, see the defect log) |
| False fallbacks | 11 |
| Routing correct / acceptable | 160 / 7 |
| Routing ambiguous-safe fallback | 20 |
| Routing wrong | 5 (all fell back safely: no domain cue recognised, legacy answered) |
| Aggregates served | 11 of 27 aggregation attempts |
| Handoffs executed | 3 of 6 rounds attempted (all at depth 1, Research to Institutional Navigation) |
| Handoffs accepted | 0 |
| SHADOW visible mismatches | 0 of 192 |
| Exceptions | 0 (486 seam records, 192 direct attempts) |
| Plan F LLM calls | 0 |

Per domain (served answers reviewed one by one against the handbook page):

| Group | Questions | Served | Fallback | Correct | Limited | Wrong |
|---|---|---|---|---|---|---|
| Teaching & Learning | 24 | 8 | 16 | 8 | 0 | 0 |
| Research & Innovation | 29 | 13 | 16 | 13 | 0 | 0 |
| Faculty Services & HR | 25 | 9 | 16 | 9 | 0 | 0 |
| Institutional Navigation | 25 | 13 | 12 | 13 | 0 | 0 |
| Cross-domain / multipart | 22 | 11 | 11 | 11 | 0 | 0 |
| Adversarial / unsupported | 67 | 1 | 66 | 1 | 0 | 0 |

The one served adversarial-group question is "What is the fax number of the Media Center?" (a
duplicated-facts probe: the row shares its numbers with Public Relations; the Media Center row is
its own chunk and was answered correctly). Of the 98 adversarially tagged questions 9 were served,
all correct, 89 fell back.

### Evidence and citations

Every served answer was reviewed for subject, relation, value, conditions, supporting evidence,
source page and multipart completeness (`review.json`): 55 CORRECT, 0 LIMITED, 0 WRONG. All 67
evidence items pass the programmatic checks (quote is an exact substring of the production chunk
it names, that chunk's page is the cited page, the quote occurs in the printed page text); every
served answer's page list equals its evidence pages. One citation observation: the evidence section
label is the page's dominant section, so the page-80 chunk about evening office hours (section 3.4)
carries the label "Responsibility to Research" (3.5); the page is right (MINOR, ingestion metadata).

### Routing

Routing codes over the 192 attempts: unchanged 142, refined by request kind 22, clauses scoped
15, ambiguous 12, none 1 (greeting). Classification against the owners each question may select:
CORRECT 160, ACCEPTABLE 7 (a plausible owner among several, or the Research-first route of a
research-unit location that is handed to Institutional Navigation), AMBIGUOUS-SAFE-FALLBACK 20,
WRONG 5. All five WRONG routings are domain-cue misses that reached the legacy path untouched
("how many ofice hours do i need", "who aproves confernce travel", "What are the weekly
consultation-hour requirements for full-time faculty?", "Tell me everything about leave.", "Does
the handbook say faculty may supervise an unlimited number of theses?"); no wrong routing produced
a served answer.

### Aggregation

The aggregation path ran 27 times and served 11 aggregates (8 two-clause same-domain or
cross-domain pairs, 1 with two supporting findings for one clause, 1 whose two clauses are the one
directory row with the duplicate collapsed, all with the parts in request order and no duplicated
part). Rejections: a required clause not supported 9, a clause's own finding rejected by the
findings verifier 5, accepted by the aggregator but rejected by the unchanged legacy Verifier 2
(the two three-clause questions whose evidence spans four pages; see D3).

### Handoffs

Six rounds were attempted, all Research to Institutional Navigation: three executed at depth one
("Where is the Research Institute?", "Where is the Research Institute of Sciences and
Engineering?", "Who do I contact about research grants?"; the target found nothing or was out of
scope, legacy answered), one blocked because the request bound to no declined clause (the fragment
"where research institute"), one blocked because Research kept a clause it did not support, one
blocked by the visited-target protection ("What is the fax number of Human Resources, and where is
the Research Institute?": Institutional Navigation had already run for the fax clause). Maximum
depth 1, no second hop, no cycle; the structural edges (self loop, repeat target, unknown target,
second hop, target failure) stay covered by `tests/test_plan_f_routing.py`.

### Fallback reasons (137)

Partial support 52; nothing found 25; findings verifier 12 (subject, relation, enumeration or
detail); no domain cue, legacy answered 12; routing ambiguity 12; general ask shape 6; handoff
unresolved 6; aggregator 5; subject guard 2; enumeration incomplete 2; legacy Verifier page cap 2;
greeting 1. The 11 false fallbacks (the handbook states the fact, Plan F did not serve it) fall
into five patterns: the WLAM profile sentences name the category in the preceding sentence of the
bullet, so the value sentence fails the entity check (Active Research, Research Intensive, Teaching
Track faculty loads); a specialist reported partial support for a paraphrase or a second-sentence
fact (office hours "per week should I keep", who arranges travel, who approves unpaid maternity
leave); the directory lexicon has no entry for the College of Graduate Studies and Research; a
number word was not read as the asked unit ("maximum number of theses"); and an approval clause
that the single-clause refinement settles is left ambiguous inside a two-clause question
("Department Chair" also draws Institutional Navigation for the 48-hour results deadline). All are
coverage items; none weakens safety.

### SHADOW, LEGACY, determinism

SHADOW: 192 of 192 visible outputs identical to the direct legacy call (answer, pages, section,
query type, evidence, items, model flags, sub-questions, candidate counts), no Plan F answer, page
or chip reached the result, 0 exceptions. LEGACY: the 30-question sample in a fresh process gives
the minimal `{mode, served, attempted}` record for every question, imports no Plan F module, builds
no runtime cache and raises nothing; the same in process after the ACTIVE and SHADOW passes.
Determinism: 24 questions (single findings, aggregates, routing-refined, handoff, ambiguous,
unsupported) observed four times each: routing, served or fallback decision, answer text, pages,
specialist calls and handoff outcome identical in all 24.

### Performance (measured)

| Measure | Median | p90 | Max | n |
|---|---|---|---|---|
| Served single Plan F answer, ACTIVE wall | 660 ms | 744 ms | 852 ms | 43 |
| Served aggregate, ACTIVE wall | 1,307 ms | 1,408 ms | 1,456 ms | 11 |
| Routing rejection, Plan F attempt | 1 ms | 2 ms | 2 ms | 12 |
| Plan F attempt before a legacy fallback | 646 ms | 776 ms | 2,085 ms | 137 |
| Legacy fallback, ACTIVE wall (attempt plus legacy) | 1,487 ms | 2,560 ms | 3,489 ms | 137 |
| Total ACTIVE response, all warm questions | 1,239 ms | 2,516 ms | 3,489 ms | 191 |
| Direct legacy call | 834 ms | 1,860 ms | 2,048 ms | 192 |
| SHADOW response (legacy plus attempt) | 1,497 ms | 2,551 ms | 3,608 ms | 191 |
| LEGACY sample, in process | 712 ms | 1,674 ms | 1,730 ms | 30 |

Cold: knowledge base 22.9 s from the cache; the first ACTIVE request 1,755 ms including the
specialist registry build (697 ms); the registry was reused on the other 191 questions and on
every later request of the app (built once per process, 20 of 21 app requests reused). Directory
answers of Institutional Navigation take 1 to 5 ms end to end. Specialist calls per attempt: 0 for
25, 1 for 154, 2 for 12, 3 or more for 1; the 7 attempts with two or more calls that fell back are
the two-clause questions whose second clause failed and the three executed handoffs, none a
gratuitous fan-out. Plan F model calls 0; legacy model calls 0 (no client).

### Diagnostics hygiene and security

486 seam records (ACTIVE, SHADOW, repetitions, LEGACY) and the 24 app-log records contain neither
the question nor the answer text. The change set and the QA artifacts contain no key, token,
credential, `.env` content, local path or traceback; the Developer panel shows exception categories
only (none occurred).

## App end-to-end review (browser, built-in pane, the final tree)

ACTIVE, 21 sequential turns in one session with Developer mode on: greeting (not applicable),
office hours (single finding), conference travel approval (routing refined), HR telephone
(directory row, 3 ms), teaching load plus sick leave (three-part aggregate, three page chips),
"What does HR do?" (known HR fallback), housing (ambiguous routing, 0 specialist calls), "Where is
the Research Institute?" (Handoff section: Research to Institutional Navigation, depth 1, visited
both), capital of France (refusal, no chips), annual leave (partial, fallback), HR telephone plus
Passport Unit fax (two-page aggregate), "how many thesis can i supervise" (fallback), maternity
leave for male faculty (fallback), instructors' load (single finding), HR telephone plus capital of
France (aggregator rejected, legacy extractor answered the contact clause), the lecturer 12-hour
false premise (fallback with the new yes/no reason), "What does HR do for new faculty?" (fallback
with ENUMERATION_INCOMPLETE), then four more questions for the memory check. After 21 turns: 42
message rows, one Developer panel per answer, every earlier panel still showing its own served path
and pages, no duplicated message, no stale loading bubble, scrolled to the latest message. The
loading text "Searching official university sources…" was observed with `role="status"` and
`aria-live="polite"`. Process memory 1,314 MB after 17 turns and 1,327 MB after 21. At 375 px the
sidebar is hidden, the Developer rows stack, no horizontal overflow. SHADOW, 3 questions with
Developer mode off: the visible answers are the legacy extractor row and the no-key reply with
legacy chips, no "Verified" chip and no Plan F text anywhere in the transcript; with Developer mode
on the panels read SHADOW, "Legacy (Plan F in shadow)", the internal outcome (accepted single
finding, accepted aggregate, unsafe result) and "built (827 ms)" then "reused"; switching the mode
off removes every panel. LEGACY, 4 questions: see the STATUS entry for the recorded rows.

## Defect log

| # | Defect | Severity | Status |
|---|---|---|---|
| D1 | "What does HR do for new faculty?" was served with the one sentence about the Employee Handbook, so one of HR's services stood for the answer to an open "what does X do" ask (a false acceptance; the known "What does HR do?" stayed blocked by the I-1 gate). Cause: the verifier's function-ask pattern required the clause to end right after "do", so the trailing "for new faculty" made it a selection ask. | MAJOR | fixed: `_FUNCTION_ASK` accepts a trailing for/to/with/in/on/about/regarding/during phrase; the ask is enumerative and falls back with `ENUMERATION_INCOMPLETE`; regression tests added |
| D2 | "Is the teaching load for lecturers 12 credit hours per semester?" (a false premise) was served with the Lecturer Researcher and senior-lecturer sentences that carry the proposed value, instead of the lecturer load of 15: a wrong supported answer. Cause: a yes/no ask whose action cannot be read had no relation check, so retrieval steered by the proposed number passed on subject words alone. | CRITICAL | fixed: a yes/no ask with no readable action is answered only by a statement that itself grants, forbids, entitles or requires; a plain value statement is rejected (`RELATION_MISMATCH`); "Can a faculty member be PI for two projects?" and the verb path are unchanged; regression tests added |
| D3 | Three-clause aggregates whose evidence spans four pages are rejected by the unchanged legacy Verifier, which cites at most three pages, so the adapter's "every rendered part keeps its page" rule fails ("verifier did not cite every evidence page"). Two questions, both safe fallbacks. | MINOR | open, documented; a decision for post-I-9 (per-part verification in the adapter or a verifier page-cap decision), not a casual change to the legacy Verifier |
| D4 | The evidence section label of a chip and of the Developer panel is the page's dominant section, so a chunk at the end of one section carries the next section's title (page 80). The page is correct. | MINOR | open; ingestion metadata, outside I-8 scope |
| D5 | An approval clause that the single-clause refinement settles ("Who approves conference travel?") stays ambiguous inside a two-clause question, so two aggregates that could be served fall back. | MINOR | open; routing refinement rule scope, post-I-9 |
| D6 | The directory row of the College of Graduate Studies and Research (page 54) is not found by Institutional Navigation. | MINOR | open; frozen specialist lexicon, post-I-9 reopening by a D-record |
| D7 | The WLAM profile loads for Active Research, Research Intensive and Teaching Track faculty fall back because the value sentence names the category only in the preceding sentence of the bullet. | MINOR | open; verifier entity binding, post-I-9 |
| D8 | The first DIRECT pass of `master_eval.py` read a non-existent attribute of the Coordinator decision and recorded an exception for every question (harness only). | MINOR (test harness) | fixed before the final run |

Observed and left as is: the legacy contact extractor answers the contact clause of "What is the
phone number of Human Resources and what is the capital of France?" and ignores the off-topic
clause (legacy behaviour, the number is right); misspelled or fragmentary inputs mostly reach the
legacy path because no domain cue is recognised (safe, a coverage item); two fragments were served
correctly ("maternity leave how many days", "sick leave full pay how many days").

## Fixes made in I-8 and regression

`handbook_bot/plan_f_verification.py` (the I-4 verification layer, not frozen): the widened
function-ask pattern and the yes/no predicate rule. `tests/test_plan_f_verification.py`: two
classification rows and two real-knowledge-base tests. No file under `handbook_bot/agents/`
changed; the six frozen specialist hashes match before and after. The whole evaluation was rerun
on the fixed tree; `compare_runs.py` shows exactly two questions changed between the runs (the two
defects above, both now fallbacks), every other answer, page, outcome, routing code, handoff code
and specialist count byte-identical, SHADOW still 192 of 192.

Targeted gate (22 files: adapter, routing and handoff, aggregation, findings verifier, pipeline,
imports, dispatcher, Coordinator, contracts, registry, legacy verifier, compatibility, orchestrator,
router, sources, the frozen specialists' integration and acceptance suites, UI presentation): 1,534
collected, 1,534 passed, 0 failed, 0 skipped, 0 xfail (85 s). Full suite: 6,674 passed, 0 failed,
0 skipped, 0 xfail (219 s; baseline 6,670 plus the 4 new tests); `pip check` clean.

## Readiness

READY FOR I-9 WITH MINOR KNOWN LIMITATIONS: no unresolved CRITICAL defect, no wrong supported
answer, 0 false acceptances, SHADOW invariant, both known unsafe cases blocked, multipart
completeness preserved, routing ambiguity falls back, handoff depth 1 with no cycle, every served
answer cited to its page, 0 Plan F model calls, diagnostics hygienic, tests and frozen hashes
clean. The known limitations are coverage items (D3 to D7), not safety items.
