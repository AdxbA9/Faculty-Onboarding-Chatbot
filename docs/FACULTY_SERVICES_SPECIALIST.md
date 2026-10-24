# Faculty Services & HR Specialist

Canonical id `faculty_services`. Module
`handbook_bot/agents/specialists/faculty_services.py` (Milestone 4, Steps
4.11 to 4.15, decisions D-085 to D-103). **Frozen for Milestone 4 (D-103,
2026-10-24).** Deterministic version 1: no LLM call, no
network, extractive findings only. Built on the shared helpers (`scoping`,
`evidence`, `details`, `grounding`, `retrieval_intent`, `domain_cues`) and its
own structured claim model (`claim_model`, D-097 and D-098, standard library only and
imported by no other module), never on the Teaching or Research modules,
which are frozen.

## Freeze status (D-103)

Frozen for Milestone 4 on 2026-10-24 (Step 4.15). Baseline: `faculty_services.py`
SHA-256 `4c158ee3…0a871`, `claim_model.py` SHA-256 `cb9304c9…901ac` (full
hashes of the module, the tests and `qa/hr_4_14_a/` in the Step 4.15 freeze
manifest). Accepted evidence on the frozen code: HR suite 1,309 passed, full
suite 3,793 passed, 0 failures and 0 expected failures; the 194 reconstructed
cases under both stand-in rankings unchanged; 17 of 19 rule switches killed by
at least one end-to-end case. Accepted limitations: the conservative misses
listed under "Known conservative misses" and "Strict scope and reader limits"
(wrapped true claims, out-of-scope routings, strict modality, condition,
comparison and locality rules, an omitted "when" event condition giving
partial, a negative question asking its own "when" condition giving partial);
"once", "upon" and "after" triggers still accepted; crafted-only evidence for
the D-100 to D-102 shapes and about 45 verbatim handbook pages; stand-in
rankings and fake embeddings only; no real-model evaluation; not wired into
the runtime. Reopening needs a new decision record (reason, affected files,
regression tests, impact, approval, new baseline hashes after the full gates);
ordinary feature growth is not a reason.

## Decision owned

Whether the approved, source-scoped evidence answers the employment,
benefits, leave, evaluation-process, promotion-process, discipline and
faculty-services part of a faculty question, expressed as a
`SpecialistFindings`. The specialist does not route (the Coordinator did),
does not write prose (Synthesis will), does not verify (the Verifier will)
and never calls another specialist.

## Ownership (D-085)

Section identities only, never a chapter: 2.1 to 2.13 (appointment classes,
recruitment and hiring, contract types, probation and resignation notice,
confirmation, dual career, administrators' appointments, termination for
cause, the Emirati track, clinical, emeritus and administrative
appointments, nepotism); 3.9, 3.13, 3.14; 4.1 to 4.7 (salary, monetary
benefits, leaves, duplication of benefits, administrative compensation,
annual incentives, the demise policy); 6.1 and 6.6 to 6.13 (the evaluation
process, the Faculty Information System, the contract-renewal rubric); 7.1,
7.5, 7.6, both printed 7.8 sections, 7.9 to 7.12 (promotion procedures,
dossier, provisions, appeals); 15.4, 15.6, 16.5. The chapter 2 preamble
(p56, the appointment definitions) is admitted as the one extra page; the
chapter 4 lead-in is covered by 4.1.

Shared sections, quoted for a bounded purpose only: 6.4 and 7.2 (the weight
and process of the research component; research output is Research's),
7.3 (the teaching-evidence requirement; the evidence itself is Teaching's)
and 7.7 (the journal-classification request only). A clause naming
publications, journals, Scopus, course files or syllabi is handed off.

Excluded and listed: 1.16 (directory rows), 3.1 to 3.8, 3.10 to 3.12, 5.1 to
5.5, 6.2, 6.3, 6.5, 7.4, chapters 8 to 12, 13, 14, 15.1 to 15.3, 15.5, 15.7,
15.8, 16.1 to 16.4, 16.6.

## Clause ownership and the shared cue lexicon

`domain_cues.py` holds `HR_OWNERSHIP_PATTERNS` (strong), `HR_CONTEXT_PATTERNS`
(weak, counted only beside another cue) and `HR_COLLISION_PATTERNS` (classes,
attendance, travel, evaluation, workload, chair, dean, semester, credit hours,
a bare "leave", grants: never a route on their own). The Coordinator scores
the same tables, so a question routed to HR is never rejected here. A clause
with a Teaching cue (loads, office hours, grades, syllabi, course files,
student attendance, examinations, Blackboard, Banner) or a Research cue
(grants, publications, conference travel, research projects) is handed off
as a `HandoffRequest`; a pure place or contact ask ("where is HR", "HR's
extension") is Institutional Navigation's alone and is never answered from
HR text; a mixed clause keeps the policy half and hands the contact half off.

## Gates before a unit supports a clause

The same six gates as the frozen specialists, read with HR vocabulary:
scope, the section guard on shared pages (p122, p124, p255, p257, p266 and
p268 carry non-HR sections), relevance by HR concepts, the production
reranker gate, quote quality (the salary-table row is the one HR table
shape), and completeness. Completeness is the requested details (amount,
penalty, deadline, quantity, duration, frequency, limits, approving
authority, responsible party, percentage, eligibility, part- or full-time,
location, requirement, a named faculty category), the shared ask gate
(D-079 to D-084: concepts, terms, roles, relation direction, numeric facts,
periods) and the HR fact gate below.

## The HR fact gate (D-086, D-087)

A clause that asks a number, an amount, a leave type, a pay status, a
faculty category, an administrative role or an academic rank is one fact.
The evidence must state it in ONE local item (a sentence, a bullet, a
lettered point), with:

- the leave type in the item (sick, maternity, paternity, special, Hajj,
  vacation, sabbatical): "15 working days" of sick leave never settles 15
  days of maternity leave; "sixty calendar days" of maternity leave never
  settles sixty days of Hajj leave;
- the pay status in the fact's local window (full pay, half pay, without
  pay): "full pay for up to 15 working days and half pay for an additional
  15 working days" binds each status to its own number;
- the category on each axis (part-time / full-time; regular, visiting,
  lecturer, instructor, clinical, emeritus, Emirati; new faculty; the
  administrative roles dean, associate dean, assistant dean, chairperson,
  director, program coordinator; the academic ranks): a fact for one member
  settles nothing about another, and the asked member must lead the fact
  when the item names several;
- the ask's anchor concepts in the fact's lead ("the faculty probation
  period is one year", not "a contract of up to two years, including a
  probation year");
- extension, renewal and stage wording agreeing with the fact ("extendable
  to a maximum of two years" states no two-year probation; "subsequently
  renewed for four years" is no first renewal); a rule stated for the
  probation stage ("resign during probation ... a six-month notice") settles
  no ask that does not name probation, whatever the general rule says;
- the number's subject (D-088): the fact's own object must measure what the
  ask asks ("a six-month notice" is no six-month probation period), and the
  ask's subject anchors, notice, contract and appointment included, must
  stand in the fact's lead inside its item ("the probation period is one
  year" states no one-year notice period);
- the number's tranche (D-088): a base ask, one with no pay status and no
  additional, limit or extension wording, is never settled by a number
  marked additional, up to / maximum / not exceeding, minimum / at least or
  without / half pay ("additional maternity leave without pay, up to a
  maximum of thirty days" states no thirty-day maternity leave; "up to twenty
  calendar days" of Hajj leave answers only an ask that says up to); an ask
  that says additional needs the word at the fact, and a fact that says it
  answers only an ask that says it or names the tranche by a non-full pay
  status the fact also states;
- the organisational level of a promotion-procedure ask (D-088): department,
  college or university, settled only by an item whose first-named body acts
  at that level; a heading or list lead-in carries no level and is never a
  quote; "what is the ... procedure" is a process request, at most partial;
- money and percentages as one fact: a range ("AED 2,000 to AED 3,000",
  "10% to 20%") is one fact with two ends; a single amount never settles a
  range unless the ask says "up to" (the high end) or "at least" (the low
  end); the qualifiers per child / total and annual / monthly / one-time
  must match and must not conflict ("25,000 Dirhams per child" is not
  "75,000 Dirhams for all children"; a one-time allowance is never annual).

The salary table: a row "Associate Professor 25,318 – 37,926" is a table
line; the header "Academic Rank Basic Monthly Salary (AED)" names the columns
and is never a row, never a quote and never evidence; a row settles only its
rank's amounts and currency, never a proposition such as "based on years of
service" or "negotiated individually" (D-090); a qualified rank ("senior
lecturer", "research professor", "assistant lecturer") has no row and never
inherits the base rank's; "full professor" is the one accepted alias of the
Professor row (D-091); a salary ask that names a rank needs the row of that rank, with the
asked amounts equal to its bounds; another rank's range is partial; an
unknown rank never inherits the nearest row; the general "what is the
salary scale" is answered by the lead sentence and rows. The administrative
compensation list is split by its role bullets, so the dean's amount never
meets the chairperson's reduction; a pure teaching-load question is
Teaching's and is handed off.

Before the shared parser reads the ask, number words become digits ("sixty
calendar days" is "60 calendar days"), a currency after its amount is
dropped (the HR money gate keeps it), and "definite term contract" loses the
word that the shared period reader would take for a term. A relation the
parser builds from a number phrase (an "action" that is no verb, or a target
that is the number after "up to"), from a nominalised HR concept ("the first
appointment ... for three years") or from "pay" inside a pay-status phrase is
dropped: the ask is copular, and its concepts, terms, the roles the ask
names and its facts still bind; HR verbs the shared inventory lacks (chair,
head, form, nominate, endorse, convene, constitute, deduct) keep their
relation, so "chaired by the Chancellor" is checked against the actor. When
the parser builds no relation at all or a degenerate one, the roles the
ask's policy segments name stay demanded, and an HR-built relation binds
the named actor to the stated action: it is stated only by an occurrence
whose own subject or agent is that actor ("the College Council shall form
the College Academic Promotions Committee" never states that the Chancellor
or the department chair forms it). The three promotion committees are
distinct bodies: an ask naming one is settled only by a sentence naming that
committee in the place the ask gives it (D-089).

## Procedures and source gaps

"How do I apply / resign / obtain" is a policy-process request: the handbook
states stages only (offer acceptance within 10 days, the employee file,
document attestation, the entry visa through the Passport Unit, travel and
accommodation arranged by HR, resignation to the dean and through the HR
portal, promotion applications in the first week of January, April or
October with a dossier), so such a clause is at most `partial` and the
documented stages are quoted; forms, portal steps, approval statuses and
timelines are never invented. A click, menu or screen request about the FIS,
MyUOS or the HR portal is a procedure request and is never supported.

A question for the contents of a source the handbook only mentions, the
Employee Handbook or the Benefits Chart ("what does the Employee Handbook
say about settling in the UAE"), carries a `named_source` detail that no
unit covers: at most `partial`, the mention quoted, the missing source named
in the limitations; a yes / no question about the source's existence or
coverage is answered by the mention (D-089). A strong frame always marks a
contents ask: the source as the speaker of a claim ("does the Employee
Handbook say / state / require that ...", "the Employee Handbook says ...")
or as its frame ("according to / under / in / per the ..."), the full name
"Faculty and Staff Pay Grade and Benefits Chart" included (D-090).

A claim of sameness or independence ("the same for every rank", "uniform
across colleges", "regardless of", "independent of") is settled only by a
unit that states sameness or independence itself; topic overlap never
settles it ("based on rank and specialization" is no "same for every
rank"). A question about the asker's own record (leave used or taken, leave
left, "my balance", "my current ...") fails closed: the handbook states
policy, not a person's record; a hypothetical or a policy question stays
answerable (D-091).

Sameness is bound to what the question compares (D-092): the sentence must
state that the asked subject is the same for the asked group ("Clinical
faculty ... evaluation follows the same timeline and process as that of
regular faculty"). "A rank equal to or higher than", "the same field of
specialization", "for the same purpose" and "the same procedure mentioned
above" (a conditional fallback) never settle "the same requirements /
procedure / membership". A question about a decision on the asker's own case
("have I been granted", "was my promotion approved", "did HR accept my
request", "did the committee recommend my promotion") fails closed like the
leave record; "who approves ...", "may ... be granted" and "can ... be
granted to me" stay policy questions. A passive question without an agent
("Can the probation period be extended?") needs the verb stated of the asked
subject, not of a neighbouring one ("the visiting contract may be extended").

A condition the question states as absent ("without Chancellor approval",
"with no notice", "does not require the Chancellor's approval") is answered
only by a sentence that states the same absence; "with Chancellor approval",
"requires the approval of the Chancellor" or "must provide a six-month
notice" contradict it and never support it. A leave state keeps its
polarity ("has unused vacation leave" against "has no unused vacation
leave"). "Without pay" stays a pay status (D-086). The passive binding also
runs when the shared parser read a relation from a noun: "Employment
confirmation includes the probation period" never answers "Can the
probation period be confirmed?", while "Termination of faculty contracts"
and "UoS may terminate faculty contracts" still answer "Can the contract be
terminated?" (D-093).

A negator governs only its own clause: in "not exceeding thirty days, may be
granted upon approval" it governs "exceeding thirty days", never "approval".
"Need no", "require no", "no ... required", "unnecessary" and "in the
absence of" state an absent condition like "without". An affirmative
question is never settled by a clause that negates its predicate for the
asked subject ("not entitled", "shall not impact", "not based on", "not
involved", "should not be part of"); "part of", "involved", "based on",
"impact" and "included" must be stated of the subject, not inferred from
the topic. "Vacation leave is unused / remains / has not been used" and
"vacation leave is exhausted" keep their polarity. An explicit verb is kept:
"Is the probation period terminated by the Dean?" and "Does employment
confirmation extend the probation period?" need that action of that subject
(and agent), and "includes" never stands for "extends" (D-094).

A negative question claims the negative proposition: "Is the probation
period not one year?", "Is the College Academic Promotions Committee not
formed by the College Council?" and "Is the basic salary not based on
rank?" are answered only by a sentence that negates the same words of the
same subject, and the affirmative rule contradicts them; "Is there no
furniture allowance?" and "Do faculty get no return tickets?" are negative
questions too. "Qualified", "receive", "get", "given", "provided" and
"awarded" belong to the entitlement family, so "not entitled" also answers
"Do faculty who resign during probation get annual leave?" (partial) and
"... not receive ticket allowances?" (supported). Each "without" frame is
read on its own ("without pay" stays a pay status beside "without special
circumstances"). A role keeps an explicit verb ("Does the Chancellor extend
probation?"), while "the Chancellor policy", "the HR section" and
"committee guidelines" are labels, not actors (D-095).

Semantic consolidation (D-096). A specific yes/no question is read once as
a claim (subject, predicate or action, object, polarity, scope, conditions,
population modifiers, value and qualifier) and each evidence clause once as
statements (its negations and what they govern, its conditional antecedents
and the consequents they condition, the restrictions of its subject). A
unit supports the question only when the two are compatible on every
dimension; an uncertain match stays partial. Ten contracts (six with a
module switch, four in one reader each): a negation of a narrower population or object ("faculty members who
resign during probation", "part-time and visiting faculty members", "a
single year's performance") never settles a broader negative question;
what an if / when / unless antecedent says is no fact, so "if the employee
has no unused vacation leave" negates nothing and "If the Deans Council
approves a promotion, ..." does not answer "Does the Deans Council approve
the promotion?"; "Is not any furniture allowance provided?" is a negative
existence question; a compound or coordinated label ("the HR rules and
regulations", "HR policy and procedures") is never an actor; a role keeps
any explicit verb with its object ("Does HR draft the Security Clearance?"
is partial, "Does HR draft employment offers?" supported), and "in
consultation with the Chancellor" makes the Chancellor no subject; an
explicit negation of the asked predicate contradicts the affirmative
question whatever the predicate ("Should termination be the first
resort?"); a relative, participle or "with" modifier keeps its polarity
("faculty who are not terminated during probation", "members not meeting
UoS standards"); a condition stated as a clause keeps its polarity and must
be the evidence's own condition ("if the faculty member receives a
termination notification" is not "If neither occurs"); and a maximum or
minimum asked is the qualifier stated ("Is the maximum probation period one
year?" is partial). Contracted negation is read with either apostrophe.

Structured claims (D-097). The three claim dimensions the D-096 contracts
did not read are now read through a small standard-library claim model
(`claim_model.py`, imported only by this specialist): the question becomes one
structured claim (the asked action and its modality, the asked population
and its quantifier, exclusivity and its focus, conditions and their roles,
every recognised operator) and the evidence sentences become statements. A
unit supports the question only when the claim is compatible on each
dimension. Modality is bound to the asked action: "Must the Faculty
Grievance Committee contact witnesses?" is partial against "may contact
witnesses", "Must faculty members apply for the bonus award?" against "may
apply", while "should" or "shall" answer "must" (D-056) and an obligation
answers "may"; "Termination ... requires the approval of the Chancellor"
answers "Must the Chancellor approve the termination ...?", and "Is approval
required to terminate ...?" asks no modality. A universal question needs an
explicitly universal statement ("All faculty including those with
administrative assignment shall undergo an annual evaluation" answers "Must
every faculty member undergo an annual evaluation?"; "Faculty members may
apply for the bonus award" does not answer "Are all faculty members
eligible ...?"); a generic question needs a statement of the same or a
broader population ("Muslim faculty attending their first Hajj" does not
answer "Are faculty entitled to Hajj leave?"); "some" or "certain" accepts a
narrower population, so "Are some faculty members not eligible for the bonus
award?" is supported by "Parttime and visiting faculty members are not
eligible". "Only" must be stated as exclusive on its focus: "Is HR the only
unit that assists with entry visas?" and "Are only Muslim faculty eligible
for Hajj leave?" are partial; "Is the probation period only one year?" is
partial because the same sentence says "extendable to a maximum of two
years"; "Is the Dean's appointment renewable only once?" and "Can the
suspension period only be extended by the Disciplinary Council?" are
supported. "Only if" needs a necessary condition ("if the employee has no
unused vacation leave" does not answer "only if ..."), an approval phrase is
necessary, and a condition of "the probation period" is read only in a
sentence about it (not in "the visiting contract may be extended with
Chancellor approval"). Most / many / few / several, an ambiguous "just", an
unqualified comparative and an event order fail closed. Support stays local
to one unit; a bound contradiction elsewhere in the returned quote (a
negation of the asked predicate for the asked population or any part of a
universal one, an exception to a universal question, an extension of an
exclusive value, a negation for the population an asked condition names)
vetoes it: "If I resign during probation, am I entitled to annual leave?" is
partial against "new faculty members who resign during probation are not
entitled to benefits such as annual leave".

Statement-level composite claims (D-098). The D-097 dimensions are now
checked together on one statement. A returned unit is read clause by clause
(a clause ends at ";" or ":", at "while / whereas", and at "and / or / but /
yet" only when a verb follows on both sides; a list such as "Pharmacy,
Physiotherapy, Nursing, and Law" is never split; ", extendable to ..." stays
with its clause; an "if / when / unless" antecedent restricts its clause and
is never a statement of its own), and the unit supports the question only
when one clause states the whole claim: the same polarity, actor, predicate,
object, modality, scope, frequency, condition, exclusivity and value, with
every operator of the question consumed by the dimension that checked it.
"Does health insurance cover all dependent children of faculty members?" is
partial against "up to three dependent children under 18 years old" (a cap
is no universal); "Is the appointment of clinical faculty limited to
Medicine and Law?" against the six-discipline list (a list is stated only by
the same items); "Does suspension from duty always result in the suspension
of salary?" against "Unless otherwise specified by the Disciplinary
Council" ("always" needs an explicit universal frequency and no exception);
"Do all senior administrative appointments require Chancellor approval?"
against "except for Deans, Vice Chancellors, and the Chancellor"; "Is the
department chair appointment renewable only once?" against "renewable
normally only once" (a hedge). A statement whose subject names another role
and not the asked one ("The Dean assists ..." for "Does HR assist ...?")
cannot support. The other clauses of the returned quote can only take
support away (an exception, a negation for part of a universal, a hedge, a
repetition or another time for an exclusive value). A word such as
"strictly" or "immediately" that no dimension reads must be restated. The
composite decides support only: a partial answer keeps its quotes and its
D-097 explanation and adds the first dimension the closest clause fails,
with that clause.

Specific claims require structured proof (D-099). A yes/no question about a
specific claim is supported only through the composite above. When the claim
cannot be read (an unreadable wrapper, two frequency words such as "Usually,
can the committee always ...?", an unknown leading word such as
"Currently,", a question opened by "Aren't" or "Cannot"), the answer is
partial: it is never supported from the earlier details alone. A question
is specific whether it is asked directly or wrapped in a conversational
frame ("Hi, ...", "Can you tell me whether ...", "I want to know if ...",
"Do you know whether ...", "Is it true that ...", "..., right?"); the
wrapper is removed before the claim is read, never a word that changes the
claim. "In general," and "Typically," stay as the claim's frequency, so "In
general, must Deans hold the rank of full professor?" is partial until the
handbook says "generally" or "normally"; a lead the Coordinator splits off
("In general" / "must Deans ...") is joined back to its question.
Overviews, what / how / where questions, procedures and open requests ("Can
you tell me about the furniture allowance?") are answered exactly as before.
Every word the question asks with is checked on the one clause that answers
it. "Professors receive a housing allowance of 90,000 Dirhams, and lecturers
receive a housing allowance of 60,000 Dirhams" is two clauses, so "Do
professors receive a housing allowance of 60,000 Dirhams?" is partial; "Is
the one-time furniture allowance under 30,000 Dirhams?" is partial against
"30,000 Dirhams" (a comparison is answered only by its restatement or by a
stated value of the same unit that satisfies it); "Is a new faculty member's
employment considered confirmed?" is partial against "If neither occurs, ...
is considered confirmed" (an "if" or "unless" clause, special circumstances
or a "subject to" that is not an approval makes the evidence conditional,
while "when", "upon approval" and "subject to the Dean's approval" do not);
"college faculty members" is not "department faculty members", and "May
visiting faculty apply for the bonus award?" is not answered by a rule about
faculty members (the handbook excludes visiting faculty in the next
sentence); "Do faculty members receive only ten days of training leave?" is
partial when they "may request additional training leave".

Payroll and payslips, the leave application procedure, leave balances and
carry-over, the Employee Handbook's contents, visa and passport renewal, HR
forms, employee ID cards, parking, housing allocation, insurance claims and
the onboarding checklist are not in the indexed source. They are recognised
as HR topics and fail closed: `not_found`, or `partial` with the missing
detail named, never `supported` from topic-adjacent text.

## Statuses

`supported`: every owned clause has a quotable unit from an owned or shared
section that covers all requested details, the grounding detail and the HR
fact detail. `partial`: a finding exists but a detail, relation, category,
leave type, pay status, fact, row or procedure stage is not stated in one
item (the default for an ambiguous clause). `not_found`: no relevant
quotable owned unit. `out_of_scope`: no owned clause (navigation-only,
Teaching-only or Research-only questions), with the handoffs recorded.
`error`: an unapproved source scope or an execution failure, redacted. A
`supported` finding settles the question by its quote. A rule stated in the
negative ("not entitled to annual leave") is quoted as it stands but never
supports the affirmative question: the result is `partial` and the missing
message names the negated predicate, so the answer layer can say "no"
(D-094, superseding the earlier reading). Symmetrically, a negative question
is supported only by the same negation of the same subject; against the
affirmative rule it is `partial` and the missing message names the
affirmative (D-095). A predicate the evidence negates for the asked subject
is `partial` with the message "the evidence negates it for the asked
subject", and a claim the evidence states only inside a condition is
`partial` with "the evidence states it only inside a condition (if / when /
unless)" (D-096). A modality, scope, exclusivity, condition-role or operator
dimension the evidence does not meet leaves the clause `partial` with that
dimension named in the missing message (D-097). A claim that no single
clause of the evidence states whole, or that the rest of the returned quote
vetoes, leaves the clause `partial` with the first failed dimension of the
closest clause, and that clause, named (D-098).

## Known conservative misses

A negative question whose subject is paraphrased ("probationary faculty who
resign" against "new faculty members who resign during probation") or whose
negated object is implied rather than listed ("not entitled to return
tickets" against "benefits such as annual leave and ticket allowances") is
partial; "Does the Chancellor extend the visiting contract?" is partial
because the handbook states "with Chancellor approval" (D-095). D-096 keeps
true negative questions partial when the evidence negates a narrower object
or population: "Does the award not impact salary?" (the handbook says "the
monthly salary"), "Do disciplinary measures not affect criminal
proceedings?" ("concurrent ... related to the same event") and "Will
individuals who report sexual harassment not face disciplinary action?"
(good-faith complaints). A true
synonym question whose verb the text lacks ("Do new faculty terminated during
probation receive return tickets?" against "are entitled to return
tickets") stays partial through the term check.

An approval ask whose object stands more than nine tokens before the passive
verb ("Who approves additional maternity leave without pay?") is partial
(the shared D-081 patient window); an ask whose verb the text lacks
("receive" against a verbless compensation list, "carry" against "workload
equal to") is partial; a deontic ask against an informational sentence
("Must the offer acceptance be submitted within 10 days?" against "HR informs
the candidate to ... submit an offer acceptance within 10 days") is partial;
a possessive role before the number ("the Committee's report ... within one
month") is read as the clause's subject; the unscoped resignation-notice ask
("Must I give six months' notice when I resign?") is partial, because the
probation sentence is scoped and the general rule states "at least six
months before the effective date" without the word notice; a percentage range written with a
hyphen in the ask ("10-20 percent") is two numbers for the shared reader;
"term" in "definite term contract" is handled, other "term" phrasings may be
read as a period; a base ask against an entitlement stated with "up to" ("Is
Hajj leave twenty days?", "Is sick leave 15 working days?") is partial until
the ask says up to or names the pay status (D-088); the vacation-day entitlement depends on a chart that is not
in the source.

The Step 4.14-A reconstructed campaign (194 labelled cases under both
stand-in rankings, `qa/hr_4_14_a/`) found two wrong supported answers, both
closed by D-100, and 27 conservative misses that stay as designed: ten true
claims in conversational wrappers the D-099 opener and wrapper lexicon does
not list ("Hi there, quick question: ...", "Sorry to bother you, but ...",
"I was wondering whether ...", "Could you please confirm that ...", "Good
evening team, would you happen to know whether ...", "Apologies for the
basic question, but ...", "One more thing - ...", "Thanks! Also, ...",
"Right, so ...", "Would it be correct to say that ..." - the last two are
listed openers whose C1 to C10 details are still computed from the wrapped
clause); four true claims routed out of HR scope by the shared ownership
lexicon (a committee electing its chair, the investigation committee's
recommendations, the Board of Trustees declaring a financial emergency,
Scopus-indexed publications); and thirteen true claims kept partial by the
strict modality, condition, comparison and locality rules above ("Must the
offer acceptance be submitted within 10 days?", "Is the Dean's monthly
compensation at least AED 5,000?", "Does the contract renew for four (4)
years after the first renewal?"). An opener-lexicon extension (F1) and
computing the details from the normalised wrapper text (F2) are backlog items
that need their own design confirmation; they are not part of D-100.

## Strict scope and reader limits (D-097, D-098, D-099, D-100, D-101, D-102)

The three claim dimensions the D-096 contracts did not read (modal strength,
universal quantification and affirmative scope, exclusivity) are read by the
structured claim model (D-097). Its scope rules are strict by design: a
universal question needs an explicitly universal statement, so "May all
emeritus faculty teach courses as needed?" and "Are all faculty entitled to
sick leave with full pay?" are partial where the handbook states the rule
generically; a generic question needs a statement of the same or a broader
population, so "Are female faculty eligible for sixty days of maternity leave
with full pay?" (stated for female faculty employed on a full-time basis) and
"Are terminated faculty entitled to return tickets?" (stated for new faculty
terminated during probation) are partial. The asked action is read from the
question's words: a relative clause with an object in the subject ("Do
applicants who meet the requirements receive ...?"), a compound subject whose
second noun can be a verb ("the HR checklist") or an "-ent" / "-ment" verb
outside the HR verb list leaves it unread; the clause then keeps the
conservative result of the other details. Comparatives ("more than",
"exceed"), event orders ("before the Deans Council approves"), proportional
quantifiers and an ambiguous "just" are recognised and fail closed.

The statement-level composite (D-098) is strict in the same direction. Five
true or defensible answers of the replay sets are partial: "Is the
end-of-service gratuity always one month's salary per year of service?" (the
handbook states no explicit "always"), "Is the maximum children's education
allowance 75,000 Dirhams for all children?" (stated for all qualifying
children), "Is the visiting contract extended with Chancellor approval?"
(stated for exceptional cases), "Is the department chair's appointment
renewable only once?" ("normally") and "At the department level, must the
promotions committee comprise at least three department faculty members
holding an academic rank equal to or higher than the rank applied for?"
(once the level frame is set aside, the comparison is read and fails
closed). A stated universal with a number in it never answers a universal
question, even one asking the same number. The clause reader is a bounded
pattern reader, not a parser: a clause whose verb has no auxiliary is not
split from the clause before it ("The HR checklist lists the documents, and
faculty members must attest the documents" is one statement), and a
statement's subject is read from the clause start, so a role in a leading
antecedent or in a modifier of the subject is tolerated rather than read as
the actor.

The D-099 contract is strict in the same direction. Fifteen answers of the
replay sets that the earlier contract supported are now partial although the
handbook supports them or makes them defensible: questions with a leading
hedge ("In general, must Deans hold the rank of full professor?",
"Typically, must faculty members on probation attend the induction
workshop?"); questions whose evidence holds only under a condition the
question does not ask ("Is additional sick leave without pay up to 30
days?", stated in special circumstances; "Is the end of service deduction 50
percent?", unless exceptions are granted; "Is approval required for unpaid
special leave?" and "Is unpaid special leave granted upon approval?", if the
employee has no unused vacation leave; "Is the furniture allowance paid in
the first month?", unless the faculty member defers it); a claim whose time
belongs to another clause ("May assistant professors request the housing
advance before the end of the first semester?"); "Is employment considered
confirmed if no written notification is received six months before the end
of the contract?" (the handbook states the condition as "If neither occurs"
in the next sentence); "In general, is the department chair appointment
renewable only once?" (a stated hedge such as "normally" vetoes an exclusive
claim even when the question asks one, because the exemption would let a
hedged statement about another appointment support it); and "Unless the
Disciplinary Council specifies otherwise, does suspension from duty also
suspend the salary?" (the handbook says "results in the suspension of
salaries", a noun the clause reader does not anchor). A comparison is never
answered beyond the number the C9 value reader asks ("Is the housing
allowance over 4,000 Dirhams?" is partial against 5,000 Dirhams). A clause
that only mentions the asked object under another verb still answers a claim
whose verb the returned unit never states (a D-098 paraphrase), and a
question in a wrapper can lose its retrieval (the wrapper words are part of
the query).

The D-100 bounded corrections (Milestone 4, Step 4.14) close the two
reproducible wrong answers of the Step 4.14-A audit without changing any
other answer of the reconstructed campaign. (1) Entity-subject binding. A
specific claim that names no role actor and no object - a copular or passive
ask about an entity, "Is the housing allowance paid once?" - reads its asked
entity as the noun phrase between the leading auxiliary and the anchor,
number, comparison operator, quantifier or verb that ends it, without
determiners, prepositions, currency and person words ("the housing
allowance"). In a sentence of several clauses that entity must be stated in
the supporting statement's own clause, its list label or its pronoun
antecedent whenever the sentence names it in another clause, so "Faculty
members receive the housing allowance monthly and the furniture allowance is
paid once" never answers the housing question (the subject dimension fails
with the clause named). An asked word the sentence states nowhere (a verb
the claim reader did not recognise, "Does health insurance cover the spouse
of faculty members?" against "is offered for ... their spouse") is not a
subject mismatch and is left to the predicate and value readers, so such a
true claim stays supported. A single-clause sentence is left to the
unit-level C9 entity check, and a claim with a role actor or an asked
object keeps the D-098 subject reader and the D-099 object binding. (2)
Parenthesised numerals. A bracketed numeral that repeats the number words
right before it ("two (2) years", "thirty (30) days", "five thousand (5,000)
Dirhams") is dropped from the text the claim and fact readers read, so the
C9 fact reader binds the number to its period or unit instead of a bare
digit matching any "2" of a statement ("Is the probation period two (2)
years?" is partial against "one year, extendable to a maximum of two
years"); a numeral that says something else ("Article (3)", "one (2)
years") and an ordinal ("the first (1) month") are kept. Each rule has its
switch (`_C12_ENTITY_SUBJECT`, `_C12_PAREN_NUMERAL`). Limits: the entity
phrase reader is a bounded token walk, not a parser, and the numeral rule
reads number words up to "thousand"; neither recovers the conservative
misses listed above.

Condition ownership (D-101, Milestone 4, Step 4.14-N1). A restriction that
mostly repeats the question's own words is the question's own content, not a
restriction ("Is renewal subject to successful evaluations?"). That test is
now also applied to the words that distinguish the restriction from its
statement's main clause, because a condition can restate the asked entity
without being asked: "Faculty members are not entitled to the housing
allowance if they receive university housing" shares "housing" with the
question "Are faculty members not entitled to the housing allowance?", but
its distinguishing word is "university", which the question does not ask, so
the condition stays a condition and the answer is partial ("the generic
claim is stated only under a restriction (sufficient)"); the same holds for
an affirmative claim ("Do faculty members receive a housing allowance?"
against "... if housing is not provided"). A condition the question states
itself ("Do faculty members receive a housing allowance if housing is not
provided?", "Is unpaid special leave granted if the employee has no unused
vacation leave?") keeps its exemption, and the condition dimension still
checks its role. The rule only withdraws exemptions, never grants one, so it
cannot make an answer more supported. Switch `_C12_OWN_CONTENT`.

Conditional language (D-102, Milestone 4, Step 4.14-W1). A condition
introduced by "when" or "whenever" is now a condition exactly like one
introduced by "if": "Faculty members are not entitled to the housing
allowance when they receive university housing" does not answer "Are
faculty members not entitled to the housing allowance?" (partial: "the
generic claim is stated only under a restriction (trigger)"), and neither
does "... whenever housing is not provided" or "HR must pay the housing
allowance when university housing is not available". The rule is uniform:
an event condition is a condition too, so "Does the basic salary increase
from the following month?" is partial against "When a faculty member is
promoted, the basic salary increases from the following month", and "Does
confirmation of employment include the probation period?" is partial
against page 63 ("... when calculating the employee's End of Service
Benefit"); asking with the condition ("When a faculty member is promoted,
does ...?", "... when calculating the End of Service Benefit?") is
supported. A concessive "even when" is not a condition; "may" questions
(possibilities) are not restricted; "once", "upon" and "after" are
unchanged. The rule can only withdraw support. Switch `_C12_WHEN_CONDITION`.

## Activation

`EXECUTABLE_SPECIALISTS` is Teaching, Research and Faculty Services & HR.
Institutional Navigation stays pending. `PLAN_F_ENABLED` stays False; the
production runtime does not import the package. No Verifier, no Synthesis,
no orchestrator, no specialist-to-specialist execution.

## Tests

`tests/test_faculty_services_scope.py` (ownership, exclusions, scope pages,
extra page, approved source, the guard on the shared pages),
`tests/test_faculty_services_specialist.py` (clause ownership, handoffs,
navigation split, statuses, procedures, source gaps, contract),
`tests/test_faculty_services_grounding.py` (roles, concepts, ask shapes,
approval direction, category, leave-type and pay-status binding, money and
ranges, contract and probation values, the D-096 claim-compatibility
invariants with metamorphic pairs), `tests/test_faculty_services_tables.py`
(salary rows, the compensation list, close numbers),
`tests/test_faculty_services_alignment.py` (one lexicon with the
Coordinator, collisions, handoffs of the frozen specialists),
`tests/test_faculty_services_integration.py` (dispatcher activation, mixed
questions, handoffs as data, no LLM call),
`tests/test_faculty_services_claims.py` (the D-097 claim model: its isolation,
the operator readers, the modal, scope, exclusivity and condition-role tables,
the HR claim and statement readers, metamorphic matrices per dimension, the
window veto, and the known failures, their controls and the campaign answers
on the fixture pages under both rankings),
`tests/test_faculty_services_statements.py` (the D-098 clause reader: clause
boundaries, secondary predications, elided subjects, antecedents, hedges,
anchors, subjects, restrictions, safe leading frames and the fallback),
`tests/test_faculty_services_composite.py` (the force, restriction,
frequency, universal-slot, enumeration, residue and ledger tables, the joint
comparison and its order, the subject rule, the window vetoes, the
first-failure message, the switches, the support-only composite, and the R10
and DR2 cases, the conservative losses and the guards on the fixture pages
under both rankings, with the synthetic classes on crafted pages) and
`tests/test_faculty_services_support_gate.py` (the D-099 specific / general
boundary and wrapper normalisation, the gate with no claim, frame or
statement, the hardened ledger, the clause, comparison, condition,
modifier, alternative-filler and fallback-polarity rules, a lead split off
by the Coordinator, the R11 wrong answers on the fixture pages and on
crafted pages under both rankings, wrapped and prefixed adversarial
transformations, general questions unchanged with every D-099 switch off,
the switches) and `tests/test_faculty_services_corrections.py` (the D-100
bounded corrections: the two Step 4.14-A defects and their controls on the
fixture pages and a crafted page, the entity and numeral readers, both
switches turned off end to end, the false-claim and true-claim guard lists
under both rankings, and the D-101 condition-ownership cases: seven claims
stated only under a condition that restates their entity, the switch turned
off end to end, asked conditions and unconditional claims unchanged, the
condition's subject and roots, and the rule's monotonicity; and the D-102
conditional-language cases: omitted state, event, deontic and positive
"when" / "whenever" conditions, the switch turned off end to end, asked
conditions, possibilities, a concession and controls unchanged, and the
never-creates-support property). Shared fixtures live in
`tests/faculty_services_support.py`. The reconstructed 194-case campaign,
its deterministic runner and the reversible switch-mutation harness live in
`qa/hr_4_14_a/` (commands and expected counts in its README).
