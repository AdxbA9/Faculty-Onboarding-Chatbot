# Teaching & Learning Specialist

Module: `handbook_bot/agents/specialists/teaching.py`. Class:
`TeachingLearningSpecialist`. Canonical id: `teaching`.

THE SPECIALIST IS NOT WIRED INTO THE PRODUCTION RUNTIME. Nothing in
`answer_question`, the orchestrator, the UI or the evaluation runtime imports
it; the multi-agent switch stays off (`PLAN_F_ENABLED` False) and the
Milestone 1 pipeline is unchanged. The specialist is built and exercised in
isolation by `tests/test_teaching_specialist.py`.

## What it decides

Whether the approved, source-scoped evidence answers the teaching part of a
faculty question, returned as `SpecialistFindings`. It does not route (the
Coordinator did), does not write prose (Synthesis will), does not verify
(the Verifier will) and never calls another specialist.

## Version 1: deterministic and extractive

- No LLM call anywhere in the module; `llm_used` is always False.
- Each finding is extractive: `claim` is the quoted handbook text itself and
  `evidence_quote` is an exact slice of a retrieved chunk, validated against
  the chunk before the finding is built.
- `source_id` and `page` come from the chunk's metadata, never from
  generated values. Source and page are the authoritative citation.
- The cross-encoder rerank score is recorded in finding metadata as a
  heuristic, uncalibrated value. `Finding.confidence` stays at its contract
  default; `SpecialistFindings.confidence` is an ordinal status signal
  (1.0 supported, 0.5 partial, 0.0 otherwise), not a probability.

## Six gates before a chunk supports a clause

1. **Source and page scope.** The shared `RetrievalScope` compiled from
   `knowledge/handbook_sections.json` for the owned sections (page-granular,
   the first boundary). Unchanged by the corrections below.
2. **Section eligibility (`SectionGuard`).** Page ranges admit whole pages,
   and fifteen in-scope pages also carry a non-owned section. On such a page
   each chunk is attributed to a section from the headings printed on the
   page: every section starting there is located as a printed line (its
   number plus the first two title words, e.g. "3.5 Responsibility to"), a
   line belongs to the last section whose heading precedes it, a line window
   is checked line by line, and a paragraph window is split at the headings
   it contains. Inside a window, text after a heading belongs to that
   heading's section until the next heading, and text before the first
   heading belongs to the section that precedes that heading in the page's
   section order, never to a state remembered from an overlapping window. A
   window without any heading lies in the section the previous window ended
   in (windows overlap by more words than a heading marker, so a heading is
   never missed). Overlapping windows therefore attribute the same characters
   the same way. Text after a non-owned heading is never quoted. A shared
   page whose headings cannot be located is rejected whole
   (`metadata["boundary"]["unresolved_pages"]`): a false `not_found` is
   preferred to a supported claim from an excluded section. The guard
   verifies at construction that chunk metadata ids match list order and
   raises otherwise. Scope membership alone never implies ownership.
3. **Relevance (`is_relevant`).** The quoted unit must share one ANCHOR
   concept with the focus clause, or two VOCABULARY concepts. Concepts come
   from a small, explicit Teaching vocabulary with aliases (syllabus and
   syllabi, exam and examinations, grade and grading, class and classes,
   teach and taught, attendance and absences, office hours, Blackboard,
   Banner, LMS, teaching load and workload, add/drop, final exams and so
   on). Anchors are concepts specific enough to stand alone
   (`ANCHOR_CONCEPTS`); generic words (`GENERIC_TERMS`: policy, process,
   require, contact, fee, assignment, information, faculty, student,
   university, semester and the like) carry no concept and never count. Bare
   "attend" is plain vocabulary; only "attendance", "absence" or "attend"
   applied to classes, lectures, sessions or exams is the classroom
   attendance anchor, so "attend international conferences" is not
   attendance evidence. Professional, faculty, teaching or instructional
   development, development training, training modules, workshops or
   programmes, training for faculty and "new faculty" are anchors; bare
   "training" is plain, so security or safety training is not
   faculty-development evidence. A "classes end" line is never evidence for
   a "classes begin" question. Relevance establishes the TOPIC only.
   Two topic anchors are built on a base word: "grading policy / rules /
   scheme / system / criteria" is the grading anchor and "exam(ination)
   policy / rules / regulations / procedures" the exam-policy anchor; each
   also carries its base word (grade, exam) as plain vocabulary, and a
   text that uses the base word is on the anchor's topic, so "What is the
   grading policy?" is answered by sentences about grades and "What are
   the exam rules?" by the 12.16 breach-of-exam-rules item. Bare "policy",
   "rules" and "regulations" stay generic: research policy, HR policy,
   parking rules, travel rules and conference regulations carry no concept.
   "e-learning" is read as the LMS topic (the 10.6 policy). Hyphen-like
   separators between words are read as spaces before concept and cue
   matching ("office-hours", "add/drop", "part-time"); quotes are never
   rewritten.
4. **Reranker gate.** `MIN_RERANK_SCORE` applied to every kept candidate, as
   in production. The production `FINAL_K` cut is not applied (it limits
   answer context, not evidence eligibility).
5. **Quote quality (`quote_problem`, `select_quote`).** Rejected: section
   and chapter headings, numbered list titles ("4. Office hours"), any unit
   ending in a colon, units starting in lower case, prose units under five
   words, a truncated last sentence of a paragraph window (unless it ends
   with a table cell), and printed lines that are not table rows. Table
   rows (weekday-led calendar lines, grade-table lines including textual
   ranges such as "F Below 60 0.00", lines with three or more numeric cells
   and a label) are quoted whole. A sentence that opens with a
   context-dependent word (Such, They, These, This, Those, It), after any
   list marker such as "2." or "b." is stripped, is quoted together with
   its antecedent: the list's lead-in or the preceding plain sentence for a
   list item, the item's own earlier sentence for a sentence inside an
   item. A sibling list item, a heading or a unit outside the allowed span
   is never an antecedent; without one the sentence is not quoted at all.
   Lettered items ("a.", "b.", "A.", "B." after punctuation and before a
   capital) are list items like numbered ones: the marker stays with its
   item, the item is quoted whole with its marker ("A. Breach of Exam
   Rules: ..."), a lowercase marker is not a dangling fragment, and "b.
   These ..." never borrows "a. ..." as its antecedent. "a grade of A. The
   ..." still ends a sentence.
6. **Completeness (`requested_details`, `unit_coverage`).** Relevance is
   separated from answerability. When the clause explicitly asks for
   details, the clause is fully `supported` only when ONE local evidence
   unit (a list item, a table row or a sentence) of ONE finding contains
   every requested detail together with a concept of the clause. Details
   are never combined across findings, sentences, list items or faculty
   categories: "part-time" in one item and "five hours" in another do not
   answer a part-time quantity. Detail families: fee or cost (a fee word
   with a monetary value: a currency next to a number such as "AED 100",
   "100 AED" or "75,000 Dirhams", an explicit "amount / price / fee of N",
   or "free of charge"; "fee applies to examination 101" has no value),
   penalty or consequence, deadline or date (a date, weekday, "within N
   hours/days", "at the beginning of the semester" and similar timing; the
   word "deadline" itself is not a date), number or amount (a number
   followed by a count noun such as hours, credit hours, days, students,
   courses, sessions, percent or points whose phrase names the counted
   subject of the question, or whose "of" phrase does: "48 hours of office
   hours" answers an office-hours count and "48 hours of training" does
   not; room, page and section numbers, dates, list markers and counts of
   something else do not count), frequency (weekly, per week, N times),
   minimum and maximum (the limit word with a value-shaped number: one
   followed by a count noun, a percent sign or a currency, or attached to
   the limit phrase as in "not exceed 40"; a year, a dotted section number
   and an identifier such as "room 204", "course 101" or "page 12" are not
   values; a table row's numeric cells are its values), approving
   authority (an approval verb and an authority noun in one unit),
   responsible party or unit (for "who manages / supports / maintains /
   administers / is responsible for / is in charge of ...": a
   responsibility verb form and a named party such as the IT department,
   the Registrar, a dean, chair, council, committee, coordinator, faculty
   members or instructors in one unit; "Learning Management System" and
   the noun "support" name nobody), percentage or range, part-time and
   full-time distinction, location or system, an explicit requirement
   statement (for "what must ... contain", "what is required", "what are
   the requirements", "requirements for ..." and "what are the <subject>
   requirements" with a subject of up to four words: the unit must express
   an obligation with must, shall, required, responsible for, expected to
   or should), and the faculty category named in the clause
   (`CATEGORY_GROUPS`). "Should" satisfies a requirement question by design:
   the handbook states many of its requirements with "should", the quote
   keeps the original wording, and no deontic distinction is made.
   Requirement detection never changes ownership: "What are the research
   requirements?" is a requirement question that Teaching hands off.
   Uncovered details are named in `missing`, either "not found in the
   evidence" or "not stated together with the other requested details in
   one place", and the clause is `partial`. Among relevant units of a
   chunk, and among candidates of a clause, evidence whose single unit
   covers more requested details is chosen first.

## Quote units

A row chunk is one unit; a row window's units are its rows; a paragraph's
units are sentences. A sentence inside a bullet or numbered item is quoted
together with the item's opening so that "The teaching load for this
category is 12 credit hours" keeps its category. A prose unit is extended by
the following unit when that unit is allowed, quotable, shares a concept and
does not open a new list item; the combined quote stays under
`MAX_QUOTE_WORDS`. Printed lines are never joined. Prose lines are not
evidence units at all: the paragraph chunk carries the same text as
sentences. A heading fused to its first sentence without punctuation is
quoted with that sentence.

## Deduplication and cap

After quote selection, a quote that repeats or contains another finding's
quote on the same page is merged into it (the finding lists both clauses),
so a row and the row window that carries it yield one finding. At most
`MAX_FINDINGS_PER_CLAUSE` (two) findings per clause, chosen from at most
`MAX_CANDIDATES_PER_CLAUSE` relevant candidates: those covering more of the
requested details first, then reranker order.

## Calendar context

For lines on the Academic Calendar pages, the term and academic year are read
from the line itself ("Classes begin for Fall 2026-2027") or from the nearest
preceding semester header ("Fall Semester 2025/ 2026"), looking back to the
previous calendar page when needed, and recorded in
`metadata["calendar_context"]`. A line from another academic year than the
one asked for (default: the handbook's own year from the registry version) or
from another term than the one named in the clause is not evidence. The
paragraph chunks of calendar pages are not evidence units: tables are quoted
from rows. The date extractor remains unnecessary.

## Approved source and scope

The only approved source is `uos_faculty_handbook_2025_26`. Owned sections
(chapter and printed number; both printed 5.2 sections included): 1.14; 3.1,
3.2, 3.3, 3.4, 3.7, 3.10, 3.11, 3.12; 5.1, 5.2, 5.3, 5.5; 10.6; 12.7, 12.12
to 12.18; 16.6. Compiled ranges on the current map: 49-51, 75-80, 83-84,
88-98, 112-117, 119-121, 192-194, 220-232, 268-272. Compilation fails
loudly if an owned section is absent from the map or the map belongs to
another source. Excluded areas are listed in `EXCLUDED_SECTIONS` (2.9, 3.5,
3.6, 5.4, 6.2, 6.3, 12.8 to 12.11, 12.19, 12.20); only their boundary pages
remain reachable, and the section guard removes their text there.

## Ownership and handoffs

A clause is Teaching's work only when a positive teaching cue fires. The
cue set is a superset of the Coordinator's teaching cues (teaching,
instructional, courses, lectures, classes, syllabus, curriculum, exams,
grades, gradebook, grade center, marks, attendance, assignments, quizzes,
rubrics, assessments, office, credit and contact hours, workload, WLAM,
Blackboard, Collaborate, SafeAssign, Turnitin, LMS, Banner, thesis
supervision, professional and faculty development, pedagogy, peer
observation, mentoring, teaching awards and workshops, lessons, training,
workshops, students, academic integrity, plagiarism) plus natural
phrasings: consultation time and student consultations, class sections and
a section being cancelled or closed, the first week of classes, timetables
and course timetables, learners, AI and generative tools, summer teaching.
A clause with no teaching cue is not searched: with another domain's cue it
is returned as a `HandoffRequest` (misrouted); with no cue at all it is
reported as unowned. A clause with both a teaching cue and another domain's
cue is answered from Teaching's evidence and also handed off (shared).
Internship requirements, research grant deadlines, salary payment, visa
renewal, the IT help desk and parking permits remain unowned.

Other-domain cues are phrase-level. Faculty services: annual or sick leave,
leave of absence, vacation, sabbatical, employment contract, contract
renewal, probation, payroll, salary, employment benefits, allowances,
housing, health insurance, visa, passport, promotion, tenure, HR. Research:
research, grants, seed or external funding, research ethics, publications,
journals, patents, intellectual property. Institutional: "who do I contact
about", "where is / where can I find the ... office, department, unit,
centre, desk, library, clinic" or a named service (the Registrar, the
registration, admissions or finance office, HR, IT services, reception,
security, bookstore, cafeteria, parking), "who manages / supports /
maintains / administers / runs / provides ... support, help desk, service,
system, portal, office, unit, department, centre" (so "Who manages
Blackboard support?" and "Who supports Blackboard?" are shared with
Institutional and handed off, while "Who manages course grading?" is
not), phone number, email address, "which office handles", help desk,
governance. "Office hours" is never a place: "Where are office hours
held?" stays with Teaching. Bare "contract", "appointment", "benefits",
"funding", "extension" and "who approves" are not cues: the handbook's own
curricula-approval policy answers approval questions, so "Who approves a
course modification?" stays with Teaching. Cues are matched on the clause
with hyphen-like separators read as spaces ("office-hours" is owned). Only
the Coordinator-provided focus clauses are classified; the full question is
never re-scanned.

## Task handling

- `task.question` is the full original question; the assigned work is
  `task.context["focus_clauses"]`, or the whole question when absent. The
  full question is never re-scanned for other specialists' clauses.
- At most `PLAN_F_MAX_SUBTASKS` clauses per task; further clauses are
  reported in `missing` and `metadata["unprocessed_clauses"]`.
- `task.intent` selects the retrieval route when it is an existing route;
  otherwise `policy`. One intent applies to every clause of a task; a
  per-clause intent is a Coordinator and orchestration matter and is
  deferred (D-046). `task.system` supplies the system for the procedure
  rule when the clause does not name one.
- A task assigned to another specialist raises `ValueError`. A
  `source_scope` naming any source other than the handbook is refused with
  status `error` before any retrieval.

## Systems: Blackboard, Banner, MyUOS

The handbook contains policy-level references only. A clause that asks for a
procedure (steps, clicks, screens, menus, "how do I create, upload,
submit ...") about a system is never `supported`. It is `partial` only when
the retained evidence names the system asked about or shares an anchor
concept with the request (the LMS attendance rule for "which Blackboard
menu records attendance"); tangential teaching sentences are rejected and
the clause is `not_found`, with the absent procedure named in `missing`.
"How do grades reach Banner?" is a fact question and can be `supported`.
Instructions are never generated.

## Status semantics

- `supported`: every assigned clause is Teaching's, each has at least one
  finding that passed all six gates, every explicitly requested detail is
  covered, no clause asked for an unavailable procedure, no clause was
  unowned or unprocessed; `missing` is empty.
- `partial`: at least one valid finding, and at least one gap in `missing`
  (a clause without evidence, a requested detail not covered, a procedural
  request, an unowned clause or an unprocessed clause).
- `not_found`: the clauses are Teaching's but no chunk passed the gates.
- `out_of_scope`: no assigned clause carries a teaching cue; misrouted
  clauses are returned as handoffs, unowned clauses are listed in `missing`.
- `error`: an execution failure or an unapproved source scope; the message
  in `metadata["error"]` has key-shaped secrets and filesystem paths
  replaced. Missing content is never an error.

## Finding metadata

`chunk_id`, `chunk_type`, `chapter`, `section_no`, `section_title`,
`section_label_page_level` (always True: the label is the page's primary
section, not necessarily the quoted text's section), `page_section_nos`,
`shared_page`, `boundary_guard`, `clauses`, `route`, `rerank_score`,
`score_semantics`, `relevance` (shared anchors and vocabulary),
`details_covered`, `calendar_context` when applicable, `extractive`. Each
clause report in `metadata["clauses"]` lists `requested_details` and
`uncovered_details`. No more precise section number is fabricated; source
and page are authoritative.

## Dependency injection

`TeachingResources` holds an embed callable, a rerank callable, the FAISS
index, chunks, metadata, the registered source record and the section map.
`TeachingResources.from_knowledge_base(kb)` builds it from a loaded
knowledge base and refuses an unregistered document or one whose section map
was not applied. Importing the module loads no model and opens no
connection; there is no module-level registry.

## Integration with the Faculty Onboarding Coordinator

`handbook_bot/agents/specialists/dispatch.py` runs the specialist on the
tasks the Coordinator assigns to `teaching` and returns the Coordinator's
decision together with the specialist's findings; tasks for the other
specialists stay pending, handoff requests are preserved but not executed,
and no Synthesis or Verifier step is added. The seam is exercised by
`tests/test_teaching_integration.py` and is not wired into the production
runtime. The specialist keeps a single approved source today; the seam
carries no assumption about how many sources a specialist may use.

## Known limitations

- Ranking has only been exercised with stand-ins: the real embedding model
  and cross-encoder were not available locally. Real-model validation is a
  separate gate.
- A numeric table body without concept words (the grade table rows alone)
  is reachable only when it shares a unit with a matching sentence.
- The detail families are a small fixed set; a question that asks for a
  detail outside them (for example a room number) is judged by relevance
  only.
- One intent per task (deferred, D-046).
- Row boundary inside a paragraph window (cosmetic, deferred). A printed
  bullet that spans two lines without terminal punctuation, followed on the
  next line by a new capitalised sentence (page 224: "o Number of students
  receiving incompletes ... thereof" then "Faculty members shall directly
  enter the grades ..."), is joined into one unit because the paragraph
  chunk carries no line boundaries. A capital-letter rule would split
  legitimate mid-sentence proper nouns (the Dean, Blackboard, the
  Registrar), and the row boundary is only known to the chunker, so this is
  not corrected here. The quote is still verbatim and the status is still
  at most `partial`.
- `section_no` is page-primary metadata (`section_label_page_level` is
  always True). On a shared page the quoted text may belong to the
  preceding owned section (page 80's office-hours items carry the label
  3.5); `page_section_nos`, `boundary_guard`, `source_id` and `page` carry
  the more useful truth. No exact section provenance is fabricated.
- "Blackboard" and "LMS" are distinct concepts: a "Blackboard support"
  clause is answered from sentences that name Blackboard, and the LMS
  policy's "The IT department administers the LMS" answers "who manages the
  LMS"; the institutional handoff carries the question either way.
