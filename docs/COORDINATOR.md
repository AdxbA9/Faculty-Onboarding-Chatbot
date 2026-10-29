# Plan F Coordinator

Module: `handbook_bot/agents/coordinator.py`. Entry point:
`coordinate(question) -> CoordinatorDecision`; `analyze(question)` returns
the measurements behind a decision for traces and tests.

THE COORDINATOR IS NOT ACTIVE IN PRODUCTION. Nothing in `answer_question`,
the orchestrator, the UI or the evaluation runtime imports it, and
`PLAN_F_ENABLED` is False. This document describes the rules as implemented
and tested in isolation.

## What it decides

Which specialists a question needs, at which organizational level, which
university systems it names, whether it is simple, multipart, cross-domain
or an orientation journey, how it splits into specialist tasks, and whether
Synthesis will be needed. It never answers, retrieves, calls a specialist or
verifies.

## Order of work (deterministic, no LLM call)

1. Technical intent, multipart flag and route scores come from the existing
   Router with its arbitration switched off (`llm_fallback=False`).
2. Domains are scored from cue lexicons (routing vocabulary only).
3. Level and systems are detected from their own cue lists.
4. Specialists are selected and capped by the Plan F budgets.
5. One task per selected specialist is created from the question's clauses.
6. `requires_synthesis` is set.

## Domain cues (routing concepts, not policy content)

| Specialist | Strong cues (1.0 each) | Hints (0.5 each) |
|---|---|---|
| `teaching` | teach/teaching/instructor; teaching load, workload, WLAM; course, lecture, class, syllabus, curriculum; Blackboard, LMS, e-learning, gradebook, SafeAssign, Collaborate; add/drop; grades, marks, attendance, assignments, quizzes, exams, rubrics, assessment; office hours, credit hours, contact hours; thesis or dissertation supervision; professional development, pedagogy, peer observation, mentoring | students, semester, lesson, academic integrity, plagiarism, Turnitin, training, workshop, Banner |
| `research` | research; grant, funded, funding, principal investigator; research ethics, ethics approval or committee, REC, IRB, IACUC, Institutional Review Board, animal care and use committee, human subjects, animal use; publication, publishing a paper or article, journal, Scopus, h-index, citations; intellectual property, patent, technology transfer, TTO, commercialisation, royalty, inventor, invention disclosure, licensing of revenue, inventions, patents, technology or IP; research institute, group, center, project, committee, board; research priority areas, misconduct, assistant, advance, budget, portal, office, support unit, funding department, publishing, regulations, agreements, leave; University Research Board, VCRGS, scientific publishing or research, UoS journal; consultancy; a cap or count of theses, dissertations, students or candidates supervised, and supervision of research projects, postgraduate research or PhD candidates | conference, innovation, laboratory |
| `faculty_services` | HR, human resources, employee services / information / record / portal / self-service, employment; leave, sabbatical, vacation, absence, maternity; contract, resignation, probation, appointment, hiring, termination, renewal; benefits, salary, payroll, payslip, allowance, gratuity, pension, incentives; housing, accommodation, insurance, visa, residency, Emirates ID, passport, relocation, air tickets; promotion, rank, tenure, performance evaluation, FIS | library, parking, child care, clinic, health services, facilities, ID card, onboarding, joining |
| `institutional` | "who ... approves / authorizes / decides / is responsible / handles / reports to"; "who manages / supports / maintains / administers ... a support, help desk, service, system, portal, office, unit, department or centre" (the object must be a service or unit: "who manages course grading" stays with the subject domain); "who do I contact / ask / see"; "which office / unit / committee handles"; "where do I go / find / get / submit / apply / register"; "where is the office / department / unit / center / building" or a named service (Registrar, registration office, admissions office, finance office, HR, IT services, reception, security, bookstore, cafeteria, parking; "office hours" is never a place); phone, telephone, fax, e-mail address, extension, contact details; approval chain, organizational chart, governance, university structure; colleges, departments, campuses, degree programs, programs offered, accreditation, ranking, mission, vision, core values, councils, standing committees | dean, chair, chancellor, director, head of; department, college, university, office, unit |

A domain's score is the sum of its matched cues (each pattern counts once).
A domain is a candidate at a score of 1.0 or more: one strong cue, or two
hints. Scores and matched cues are recorded in `metadata["scores"]` and
`metadata["cues"]`. Bare "who" and "where" are never cues: "Who teaches the
course?", "Who manages course grading?" and "Where are office hours held?"
select `teaching` alone.

### Research routing (Milestone 4)

- **WLAM demotion.** The research cues score a view of the question in which
  the WLAM faculty categories ("research-intensive faculty", "active research
  faculty", "research active faculty") are blanked, same length; every other
  domain scores the normal text, and those category phrases (with
  "teaching-track faculty") are themselves a Teaching cue. "What is the
  teaching load for research-intensive faculty?" and "How many credits does
  research-intensive faculty teach?" select `teaching` alone; "What research
  output is expected of research-intensive faculty?" still selects
  `research` (with `teaching`), because "research" also appears outside the
  category phrase. "research" is never stripped globally.
- **Supervision split by detail cue.** A cap or count ("How many theses can I
  supervise at once?", "the maximum number of dissertations", "How many
  students can a supervisor take?", "How many PhD candidates may I
  supervise?") and the supervision of research projects or postgraduate
  research select `research` (the 3.5 cap). Supervision credit, workload and
  eligibility ("How is thesis supervision calculated?", "How many thesis
  supervision credits count toward workload?", "Am I eligible to supervise
  theses?") stay with `teaching` (3.1 and 3.11). There is no broad
  "supervision" cue.
- **Consultancy** is a strong research cue (it was a hint), and so is
  "consulting".
- **Conference policy in context (Step 4.5A).** Per diem; conference
  registration, fees, funding, travel, support, attendance, allowances,
  expenses, requests, documents, deadlines, budget, participation, papers,
  proceedings or approval; attending, participating in, presenting at,
  travelling to, registering for or funding a conference; local,
  international, external, overseas, scientific, academic or
  university-supported conferences; presenting a paper; "how many
  conferences". A bare "conference" stays a hint, and video conferences,
  conference rooms and conference calls never select `research`.
- **Other research-owned phrasings (Step 4.5A).** Chief editor,
  editor-in-chief, editorial board or secretary; petty cash; seed projects,
  grants, research or funding; inventions; copyright; Materials Transfer
  Agreements (MTA); licence agreements and signing a licence; conflict of
  interest or commitment; informed consent; co-investigators; final or
  progress project reports; publishing costs, fees or charges; open access;
  "tech transfer". Hyphenated forms are read with spaces ("per-diem",
  "tech-transfer", "co-investigator", "editor-in-chief").
- **Development funding is not research funding.** Funding or grants for
  professional, faculty, teaching or staff development are blanked from the
  research view, so "What funding is available for professional development
  workshops?" selects `teaching` alone. "Publication of course materials"
  and library journals are not scientific publishing.
- **Location and contact (Step 4.5A).** "Where ... located", "location of
  the office / department / unit / institute", "how do I contact / reach / get
  in touch with" and "contact person" are Institutional cues, so "How do I
  contact the IRB?" and "Where is the Research Institute for Medical and
  Health Sciences located?" also select `institutional`.
- **Round 2 (Step 4.5A).** Floors, buildings, addresses, "how do I find /
  get to the <place>", "is the office open during", "point of contact" and
  "someone I can contact" are Institutional cues. A channel lookup (phone,
  e-mail, fax, extension) goes to `institutional` alone only when no other
  clause names a subject domain; otherwise the subject specialist keeps the
  substantive clause. Research conference cues add symposium, congress,
  "take part in", "join", conference details within a short window on either
  side, "before the conference" and "conferences per year", while conference
  calls, rooms, halls and centres stay excluded; the verb "grant" (who grants
  tenure, grant access), "professional ethics committee", the rec center and
  an IP address are not research. RFD, RIMHS, RIHSS and OVCRGS, open access
  and page charges, book awards, tangible research property, disclosure
  evaluation, project closure, paid consultants and "who owns software"
  reach `research`.

- **Shared Research lexicon (Step 4.5A-R).** The Research strong cues are
  built from `specialists/domain_cues.py`, the lexicon the Research and
  Teaching specialists also read, so the Coordinator never routes to Research
  a phrase that Research would reject. A channel clause that also asks a
  subject question (role, responsibilities, policy, requirements, deadlines,
  per diem, fees, approvals, "what does X do") keeps its subject specialist
  beside Institutional Navigation.

### Clause splitting (Step 4.5A round 2)

A piece joined by "and", a comma or "as well as" that does not open a
question of its own (no interrogative, auxiliary or pronoun head) continues
the previous clause, so "What is the due date for course syllabi and the
conference travel form?" keeps its shared head. Every clause is still a slice
of the question. A greeting or thanks fragment ("Good morning", "Thanks")
is not a clause; a question made only of a greeting keeps it.

### Cue normalisation

Before cue matching, hyphen-like separators between word characters
(`-`, the Unicode hyphens and dashes, `/`) are read as spaces
(`cue_text`), so "office-hours", "peer-observation", "add-drop",
"add/drop" and "e-learning" score exactly like "office hours", "peer
observation", "add drop" and "e learning". Each separator is one
character, so cue positions still refer to the original text. This is
classification only: the task's `question`, its `focus_clauses` and every
recorded clause stay verbatim. The matched cue recorded in
`metadata["cues"]` shows the spaced form.

## Contact questions (conditional)

- **Direct channel lookup.** When the Router's route is `contact` and the
  question asks for a channel (phone, telephone, fax, e-mail address,
  extension, "number of the ..."), Institutional Navigation is selected
  alone; other domains that scored are listed in `metadata["dropped_domains"]`.
  "What is the HR phone number?" goes to `institutional` only.
  "What is the number of degree programs?" is not a channel lookup (route
  `count`, cue "degree programs") and also goes to `institutional`.
- **Contact request about a subject.** When the route is `contact` because
  of a contact verb ("who do I contact about ...") and a subject domain
  scored, the subject specialist(s) are kept and Institutional Navigation is
  added last; Institutional always keeps one slot within the budget.
  "Who do I contact about annual leave?" gives `faculty_services` then
  `institutional`. Other navigation phrasings ("who is responsible", "who
  handles", "who do I ask", "where is the HR department") are not Router
  contact routes; they reach Institutional through its own cues alongside
  the subject domain.

## Bare system questions

When no domain reaches the threshold but a system is named:

- `blackboard` or `banner` selects `teaching`, which owns those guides in the
  source-to-agent map, with the reason "system access question".
- `myuos` selects nobody. MyUOS is the general university portal (personal,
  HR, academic, research and student services alike), so a bare mention
  implies no domain; the decision is `unknown`, the system is recorded in
  `systems` and named in `reason`, and the handbook path handles it.
  Explicit context routes normally: "payslip in MyUOS" is
  `faculty_services`; "teaching schedule in MyUOS" is `teaching`.

## Level

Most specific level mentioned wins: `department` (department, chair, head of
department) over `college` (college, dean, deanship) over `university`
(university, UoS, chancellor, university council, board of trustees).
`None` when no level word appears. A unit name containing one of these words
sets the level too; this is a heuristic, not organizational knowledge.

## Systems

`blackboard` (Blackboard, Ultra, SafeAssign, Collaborate), `banner` (Banner,
Faculty Self Service, My UDC), `myuos` (MyUOS, employee portal). Reported in
order of first mention; each task carries the first system named in its own
text. Systems are never specialists.

## Selection and budgets

Candidates are ordered by score, then by the position of their first cue,
then by canonical order (`teaching`, `research`, `faculty_services`,
`institutional`), and capped at `min(PLAN_F_MAX_SPECIALISTS,
PLAN_F_MAX_SUBTASKS)` from `config.py` (3 and 3). Dropped domains are
recorded. `coordinate(..., max_specialists=, max_subtasks=)` overrides the
budgets for tests only.

## Decomposition

Every task's `question` is the complete original question, so a task always
stands alone. The specialist's own part travels separately in
`context["focus_clauses"]`, a list of verbatim clauses of the question
(never rewritten text); `context["full_question"]` repeats the question.

Clauses are split with the Router's own separator, so multipart detection
and decomposition agree. For a single specialist the focus is the whole
question. For several specialists each receives the clauses in which its
domain has a cue; a clause may be shared; a clause whose only cues belong to
a dropped domain is reported in `metadata["uncovered_clauses"]` instead of
being attached to another task; a clause with no cue continues the previous
clause; a specialist left without a clause has the whole question as focus.
`intent` (from `router.classify`) and `system` are derived from the focus
text when it is a real subset, otherwise from the whole question. The intent
is one per task; a specialist with several focus clauses resolves the
retrieval route of each clause from that clause (`retrieval_intent.py`,
D-078), so a contact clause does not set the route of a substantive one.
Each task has `task_id` `task-1..3`, `specialist_id`, `domain`, `intent`,
`level`, `system`, `requested_by="coordinator"` and `metadata["scoped"]`
(True when the focus is a real subset of the question).

Example: "Can I reduce my teaching load if I have a funded research project,
and who approves it?" selects `research`, `teaching`, `institutional`; all
three tasks carry the full question; research and teaching share the focus
"Can I reduce my teaching load if I have a funded research project",
institutional has the focus "who approves it"; `requires_synthesis` is True.

## Complexity and synthesis

`metadata["complexity"]` is one of `greeting`, `unknown`, `simple`,
`multipart`, `cross_domain`, `journey`. `requires_synthesis` is True when
two or more specialists are selected, or when one specialist receives a
multipart question. The `journey` classification by itself never forces
Synthesis: a single-specialist question that merely starts with "as a new
faculty member" is rendered like any other single-specialist question.

## Confidence

Coordinator confidence is a HEURISTIC ORDINAL SIGNAL for ordering and
traces. It is not statistically calibrated, not a measured probability, not
model certainty and not an evaluated routing accuracy. The rule: 1.0 for a
greeting; 0.0 when no specialist is selected; 0.75 when an override selected
the specialist; otherwise `min(1.0, 0.5 + 0.25 * best domain score)`, so one
strong cue gives 0.75 and two give 1.0. It ignores how close a tie is.

## Metadata schema

Every decision, greeting and empty question included, carries the same
`metadata` keys (`METADATA_KEYS` in the module): `complexity`, `analysis`
(the measurements behind the decision), `scores`, `cues`,
`dropped_domains`, `uncovered_clauses`, `limits`, `router` (`route`,
`is_multipart`, `llm_attempted`) and `journey`. Values are empty when
nothing applies; nothing is invented.

## Orientation journey

Journey cues (first week, new faculty, onboarding, orientation, induction,
get started, checklist, before arrival) set `complexity="journey"` and
`metadata["journey"]=True`. Specialists are still selected from domain cues
only; a journey question without a domain cue selects nobody and says so in
`reason`; a journey question with one specialist is not forced through
Synthesis. The orientation workflow itself (stage order, checklist
rendering) is not implemented.

## Failure policy

No domain cue and no owned system: no specialist, confidence 0.0, complexity
`unknown`, reason "no domain cue matched; fall back to the general handbook
path" (with the system named when one was mentioned, as for bare MyUOS).
The future orchestrator sends such questions through the Milestone 1
handbook pipeline. Nonsense never receives a fabricated domain. A greeting
returns intent `greeting` and no specialist. An empty question returns
`unknown`.

## Dispatch to the specialists (inactive)

`handbook_bot/agents/specialists/dispatch.py` is the deterministic seam
between the Faculty Onboarding Coordinator and the domain specialists.
`run_question(question, registry)` coordinates the question and calls
`dispatch(decision, registry)`, which walks the Coordinator's tasks in order
and, for each task, either runs the registered specialist or records the
task as pending. All four specialists are executable (`EXECUTABLE_SPECIALISTS`:
Teaching & Learning and Research & Innovation from the start, the Faculty
Services & HR Specialist since Step 4.11, Institutional Navigation since
Step 4.16-B6, D-105) and each runs only when it is registered ("executable
but not registered" otherwise); a task whose specialist is not registered,
or that a specialist's handoff produced, stays pending with the
Coordinator's task intact, so the caller still sees the full decision. Tests that describe the Teaching-only
phase pin `executable=(SpecialistId.TEACHING,)`. Tasks are handed over exactly as the Coordinator
built them (task id, full question, intent, level, system, focus clauses);
nothing is reinterpreted, reordered or reconstructed, and there is no
default-to-Teaching fallback. Handoff requests a specialist returns are
collected and preserved but not executed. A specialist that raises is
reported as an `error` finding for its task, with secrets and filesystem
paths removed; the decision and the other tasks are unaffected. No LLM call
is made in the seam, Synthesis and the independent Verifier are not part of
it, and nothing in the production runtime imports it (`PLAN_F_ENABLED`
stays False). Handoff rounds, Synthesis and Verifier collaboration are
later milestones. Real-model validation of the Teaching path is still
outstanding.

A decision is validated when it is built; the seam does not re-validate
it. Two cheap checks catch a decision altered afterwards: a subtask that is
not a `SpecialistTask` raises `TypeError`, and a repeated task id raises
`ValueError`. Any other post-construction change (for example a selected
specialist left without a task) is outside the seam's responsibility and is
a documented defensive-programming limitation.

For a compound question such as "Who manages Blackboard support and what
is my teaching load?" the Coordinator selects `teaching` and
`institutional`; the institutional task's focus is "Who manages Blackboard
support", and the teaching task's focus holds both clauses because the
first names Blackboard (a shared clause). The Teaching specialist then
answers the teaching-load clause, marks the support clause `partial`
unless the evidence names the responsible party, and returns an
institutional handoff, so no part of the question is silently absorbed.

## LLM arbitration: deferred

Not implemented in this phase (decision D-PLANF-018). The Router must choose
exactly one route, so a tie needs a tie-breaker; the Coordinator may select
several specialists when domains tie, and the orchestrator, Synthesis and the
Verifier reconcile their findings. An arbitration call would only avoid the
cost of an extra specialist run, which is measured at the Teaching Agent
gate. `used_llm` is always False today.
