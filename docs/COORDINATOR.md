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
| `teaching` | teach/teaching/instructor; teaching load, workload, WLAM; course, lecture, class, syllabus, curriculum; Blackboard, LMS, gradebook, SafeAssign, Collaborate; grades, marks, attendance, assignments, quizzes, exams, rubrics, assessment; office hours, credit hours, contact hours; thesis or dissertation supervision; professional development, pedagogy, peer observation, mentoring | students, semester, lesson, academic integrity, plagiarism, Turnitin, training, workshop, Banner |
| `research` | research; grant, funded, funding, principal investigator; research ethics, ethics approval or committee, REC, IRB, ACUC, human subjects, animal use; publication, publishing a paper or article, journal, Scopus, h-index, citations; intellectual property, patent, technology transfer, TTO, commercialization; research institute, group, center, project, committee, board | conference, consultancy, innovation, laboratory |
| `faculty_services` | HR, human resources, employee services / information / record / portal / self-service, employment; leave, sabbatical, vacation, absence, maternity; contract, resignation, probation, appointment, hiring, termination, renewal; benefits, salary, payroll, payslip, allowance, gratuity, pension, incentives; housing, accommodation, insurance, visa, residency, Emirates ID, passport, relocation, air tickets; promotion, rank, tenure, performance evaluation, FIS | library, parking, child care, clinic, health services, facilities, ID card, onboarding, joining |
| `institutional` | "who ... approves / authorizes / decides / is responsible / handles / reports to"; "who do I contact / ask / see"; "which office / unit / committee handles"; "where do I go / find / get / submit / apply / register"; "where is the office / department / unit / center / building"; phone, telephone, fax, e-mail address, extension, contact details; approval chain, organizational chart, governance, university structure; colleges, departments, campuses, degree programs, programs offered, accreditation, ranking, mission, vision, core values, councils, standing committees | dean, chair, chancellor, director, head of; department, college, university, office, unit |

A domain's score is the sum of its matched cues (each pattern counts once).
A domain is a candidate at a score of 1.0 or more: one strong cue, or two
hints. Scores and matched cues are recorded in `metadata["scores"]` and
`metadata["cues"]`.

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
text when it is a real subset, otherwise from the whole question.
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

## LLM arbitration: deferred

Not implemented in this phase (decision D-PLANF-018). The Router must choose
exactly one route, so a tie needs a tie-breaker; the Coordinator may select
several specialists when domains tie, and the orchestrator, Synthesis and the
Verifier reconcile their findings. An arbitration call would only avoid the
cost of an extra specialist run, which is measured at the Teaching Agent
gate. `used_llm` is always False today.
