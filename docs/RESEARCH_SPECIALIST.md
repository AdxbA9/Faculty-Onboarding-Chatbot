# Research & Innovation Specialist

Module: `handbook_bot/agents/specialists/research.py`. Class:
`ResearchInnovationSpecialist`. Canonical id: `research`.

THE SPECIALIST IS NOT WIRED INTO THE PRODUCTION RUNTIME. Nothing in
`answer_question`, the orchestrator, the UI or the evaluation runtime imports
it; the multi-agent switch stays off (`PLAN_F_ENABLED` False) and the
Milestone 1 pipeline is unchanged. The specialist is exercised in isolation by
`tests/test_research_specialist.py` and through the Faculty Onboarding
Coordinator by `tests/test_research_integration.py`.

## What it decides

Whether the approved, source-scoped evidence answers the research part of a
faculty question, returned as `SpecialistFindings`. It does not route (the
Coordinator did), does not write prose (Synthesis will), does not verify (the
Verifier will) and never calls another specialist. It never imports the
Teaching & Learning Specialist: both use the shared helpers `scoping.py`,
`evidence.py`, `details.py` and `base.redact_error`.

## Version 1: deterministic and extractive

- No LLM call anywhere in the module; `llm_used` is always False.
- Each finding is extractive: `claim` is the quoted handbook text itself and
  `evidence_quote` is an exact slice of a retrieved chunk.
- `source_id` and `page` come from the chunk's metadata. Source and page are
  the authoritative citation.
- The rerank score is recorded as a heuristic, uncalibrated value;
  `SpecialistFindings.confidence` is an ordinal status signal (1.0 supported,
  0.5 partial, 0.0 otherwise), not a probability.

## Approved source and scope

The only approved source is `uos_faculty_handbook_2025_26`. Sections are
identified by chapter AND printed number, because the handbook repeats some
numbers and prints the copyright policy inside chapter 9 as 12.3.

| Kind | Sections |
|---|---|
| Owned | 3.5; 5.4; 8.1 to 8.11 (8.9 and 8.10 included); 9.1 to 9.5, the printed 12.3 in chapter 9, 9.7 to 9.10; 16.1 to 16.4 |
| Shared (research content only) | 6.4 research evaluation criteria; 7.2 research for promotion; 7.7 journal accreditation |
| Extra page | 175, the chapter 9 preamble (Technology Transfer Office, definitions of Inventor, Intellectual Property, Gross and Net Income) |
| Excluded | 3.1, 3.4, 3.6, 3.11 WLAM, 3.13; 4.3 leave, 4.6 incentives; 5.1 to 5.3, 5.5; the rest of chapters 6 and 7; 13.8; 15.1; 16.5 |

Compiled page ranges: 80-82, 117-119, 123-124, 144-145, 147-148, 153-182
(175 included), 259-266.

Page 175 lies outside every section record. `scoping.compile_scope` admits it
through its `extra_pages` argument, which refuses any page that carries a
section the specialist does not own, so an extra page can never widen the
scope into excluded text. Its metadata stays honest: chapter 9,
`section_no` "9", `page_section_nos` ["9"], and `boundary_guard` "chapter
preamble: page carries no section record". No section number is invented.

## Section guard and the page-level label

The shared `SectionGuard` attributes the text of a shared page to the section
whose printed heading precedes it. It guards pages 80 (3.4 before 3.5), 82
(3.5 before 3.6), 117 (5.3 before 5.4), 119 (5.4 before 5.5, whose
"two months' basic salary" is never quoted), 123 and 124 (6.4 among the
evaluation process sections), 144, 145, 147 and 148 (7.2 and 7.7 among the
promotion sections) and 266 (16.4 before 16.5). Pages 163 (8.8 to 8.11) and
180 (9.5, the printed 12.3, 9.7) carry owned sections only and are admitted
whole.

`section_no` is page-primary metadata: 5.4 amounts on page 119 carry the
label 5.5, 6.4 text on page 123 carries 6.3, 8.11 items on page 163 carry 8.9.
`page_section_nos`, `boundary_guard`, `source_id` and `page` carry the
truthful context; no chunk-level section precision is fabricated.

## Ownership and handoffs

A clause is the specialist's work only when a research cue fires on the
clause with the WLAM faculty categories blanked ("research-intensive
faculty", "active research faculty"): a teaching-load question about a
faculty category is Teaching's even when it says "research".

| Kind | Topics |
|---|---|
| Research | grants and funding, seed projects, research ethics, IRB, informed consent, IACUC, intellectual property, patents, copyright, commercialisation and licensing, Materials Transfer Agreements, inventors, institutes, centres, groups, the University Research Board and research committees, research projects and their progress and final reports, principal investigators, petty cash and research advances, scientific publishing, journals and editorial roles, consultancy (including "consulting"), conflict of interest, misconduct, human and animal research, priority areas, conference policy (5.4), the thesis supervision cap (3.5), research supervision |
| Shared (answered, and handed off) | research release and teaching load, research leave and sabbatical, research counted for promotion or evaluation, conference travel, "who handles" research matters, a substantive research question that also asks for a contact or location |
| Not Research (out of scope, handed off) | WLAM, teaching load, credit hours, supervision credit and eligibility, advising, teaching development; annual leave entitlement, salary, benefits, promotion procedure, evaluation process, incentives; navigation and contact: where, located, how do I contact, who should I contact, contact person, phone, email, which office handles, where do I submit or find, portal support |

A clause with another domain's cue and no research cue is `out_of_scope` with
a `HandoffRequest` (kind "misrouted"). A clause with both is answered from
research evidence and also handed off (kind "shared"). A clause with no cue at
all is reported unowned and not searched. A clause that names several other
domains is handed to each of them, strongest first; the first handoff keeps
the id `<task>-handoff-<clause>` and the others are numbered
`<task>-handoff-<clause>-<n>`. Only the Coordinator-provided focus clauses are
classified; the full question is never re-scanned. Handoffs are data: the
dispatcher preserves them and never executes them.

Navigation and contact (Step 4.5A). A clause is cut into interrogative
segments ("What does the RFD do | and how do I contact it?"). A segment that
asks where something is, how to reach someone, for a contact channel, which
office handles a matter or for portal support is navigation. A clause made
only of navigation is `out_of_scope` with a single Institutional handoff
(kind "misrouted"), whatever evidence is ranked first: research text that
merely names an office never answers "where" or "how do I contact". A clause
that also asks something substantive is answered from research evidence,
handed off to Institutional (kind "shared") and requests a "contact or
location details" detail that only an e-mail address, a telephone number, a
room, building or floor, or a web address can cover; the handbook's research
chapters state none, so the clause stays `partial`. A policy action by a
named body ("Which committee reviews research protocols?", "Who approves
conference travel?", "Who handles research grants?") is not navigation, and a
system click procedure ("Where do I click in the portal?") is a procedure
request.

Round 2 (Step 4.5A). Navigation also covers floors, buildings, addresses,
"how do I find / get to" a place, opening times, points of contact and
channel phrases ("phone", "email address", "extension", "website of"); a
procedure exemption needs a click cue. Contact evidence must describe the
body the clause names. A shared clause also names the foreign subject it asks
about: a Teaching or Faculty Services & HR subject ("office hours", "sick
leave", "promotion", "harassment") or an Institutional subject that is not
navigation ("laboratory safety", "safety rules", "health and safety"). The
Research evidence must state it, or the clause stays `partial` with "the
<owner> subject asked about (...)" in `missing`.

## Relevance

The quoted unit must share one ANCHOR concept with the clause or two
vocabulary concepts. Anchors are specific research topics (grant, funding in a
research context, research ethics, IRB, IACUC, IP, patent, technology
transfer, commercialisation, inventor, copyright, institute, centre, group,
research board, research committee, research project, principal
investigator, publishing, consultancy, misconduct, conflict of interest,
priority areas, per diem, conference, research assistant, sabbatical,
supervision cap, research support units, research budget, research policy).
The per diem is its own anchor, so a registration-fee line never answers a
per diem question. False-positive controls: "funding for training", a
"training project", "publish your course materials", "video conference",
professional ethics, student advising, a strategic-plan "research" mention
and library journal circulation carry no research anchor. Generic words
(`GENERIC_TERMS`: policy, rules, regulations, university, faculty, support
and the like) carry no concept.

## Requested details (completeness)

A clause that asks for a detail is fully `supported` only when ONE local unit
(a sentence or a list item) of ONE finding contains every requested detail
together with a concept of the clause. Details are never combined across
units or findings.

| Family | Evidence that counts |
|---|---|
| amount | a currency next to a number ("AED 5000", "$750", "AED 1000-1500") or "amount of N"; "at no cost" or "free of charge" is not an amount |
| minimum, maximum | the limit word with a value-shaped number (count noun, percent, attached limit, or a currency right after the limit phrase: "up to $750") |
| percentage | a number with % or "percent"; the bare word "percentage" states no value |
| deadline or date | for "when ..." any stated timing (a date, a period boundary, a relative limit); for explicit deadline wording (deadline, due date, by when, last day, cut-off, closing date) an END point: until, end of, no later than, before the X, within N days, N weeks before, closes on, or a date tied to the word "deadline". A window start alone ("announced at the beginning of each academic year") is not a deadline. The modal verb "may" is never a date: "May" counts only in date context ("15 May", "May 2026", "by May"); month abbreviations only next to a day number; never a bare year |
| how far in advance | a lead time: a number of days, weeks or months before, prior to or in advance of something ("at least one month before the conference date") |
| ownership | own, owned by, ownership, assigned to, belongs to, vested in, retains the rights, claims ownership, property of |
| permission | a statement that permits, conditions or prohibits (allowed, permitted, may, can, must, not permitted, only, except, entitled, subject to, approval) |
| specific focus | the thing a question singles out that no research concept covers ("a grant for time travel", "pay for a new car", "cover computer software", "without presenting a paper", "part-time faculty"): the words must appear in the unit |
| contact or location | an e-mail address, a telephone number, a room, building or floor, or a web address (never produced by research text) |
| duration | a number with a time unit tied to the subject ("two years"); never a year, section or item number |
| quantity | a number with a count noun naming the counted subject ("four theses", "62 active research groups") |
| frequency | per year, once, N times |
| authority | an approval verb (approve, authorize, review, recommend, determine, decide, decision, endorse, grant, permit) and a NAMED body (VCRGS, VCAA, University Research Board, research committees, TTO, IRB, IACUC, College of Graduate Studies, deanship, Chancellor ...) |
| responsible party | a responsibility verb form (including "to manage / oversee / administer" and "falls under the purview of") and a named party |
| eligibility | eligible, entitled, may apply, applicants must, must be, must have or hold, qualified, conditions |
| penalty or consequence | the shared penalty words plus closing the project, reclaiming, repaying or refunding funds, referral to the Board, withdrawal of support, rejection, "not obligated to reimburse" |
| role | a named role AND a duty or action (responsible, must, shall, supports, coordinates, prepares ...); naming the role is not enough |
| required documents | a named artefact (form, report, acceptance, abstract, flier, invoice, copy, letter, CV); "submit an application" alone is not a document list |
| project type, funding source, role, publication type, requirement, named research category | the corresponding named term in the same unit |

The interrogative decides the requested value, not the nouns: "Who
determines the funding ceiling?" asks for the authority, not an amount, and
"the maximum research advance as a percentage" asks for a percentage, not a
money amount. "How much time" asks for a limit and "how early" for a lead
time, not money. The counted subject of "how many theses can I supervise" is
"theses"; the verb phrase is not part of it. When the counted subject carries
no specific research concept ("research drones"), its head noun must appear
in the unit, so "at least three researchers" never counts drones.

Object binding (Step 4.5A). A value of the right shape is not enough: the
detail must belong to what the clause asks about. Each detail is bound to the
interrogative segment that asks for it, and the unit that states the value
(or, for a list item, the unit together with the lead-in of its list) must
name that segment's objects. Every SPECIFIC object must be named (petty cash,
research advance, per diem, registration, revenue or royalty share, research
assistants, student projects, the University's name, final or progress
reports, theses, editorial roles, visitors, research groups, sabbatical,
patents, copyright, disclosure, proceedings, MTAs, equipment, time); when the
clause names none, one CONTEXT object must be (conference or travel, grant or
project, principal investigator, consultancy, inventor, IP, IRB, IACUC,
journal). For a timing question the submitted thing (request, application,
report, proposal) is specific, and a "from receipt of X" phrase never binds
the timing to X. An approval binds to what is approved: the object after an
active verb ("The VCAA approves travel requests"), the noun right before a
passive form ("invoices approved by the Dean" is about invoices), or the
sentence subject after a function word ("a petty cash advance ..., subject to
approval by the VCRGS"). Eligibility binds to what one is eligible for
("entitled to a petty cash advance", "may apply for a research advance") or,
for a condition, to the sentence subject ("The principal investigator ...
must hold a Ph.D."). A recipient ("goes to the inventor") must be named as
the receiver, not as a modifier ("the Inventors' College"). When the clause
names no research category, a unit that states the rule of exactly one
category (seed, competitive, collaborative, targeted, visiting researchers)
does not answer it; a category named only as an exception ("except for seed
projects") leaves the rule general.

Conference locality. A conference clause that names one locality binds its
details to it: a unit that names only "local" never answers an international
or external question, and a local question needs the word "local" in the
unit. The general per diem and registration rules name no locality and remain
usable for an international question (they are the rules that apply outside
the local exception). A candidate that names only the other locality is
rejected ("names a different conference locality"). "Regional" is a
non-local locality.

Round 2 (Step 4.5A). Timing binds inside one sentence: the deadline or lead
time, the asked object, the plain subject words of the question ("flier",
"abstract", "acceptance letter", "adverse events") and a specific action
("announced", "published", "reported") must meet in the same sentence, the
action in the coordinate part that states the time. A relative end point
needs a time-bearing head ("before the conference date", "N days ahead of",
"upon project completion"); "before the Board", "until further notice" and a
window start ("opens at the end of ...", "after the end of ...") are not end
points. A recommending body is never the approver; "approved research
projects" is an adjective, not an approval; a passive approval binds to the
subject of its own clause, never across ", and"; appointing, nominating,
designating and signing are approval acts; "responsible for approving" needs
the approving act. Specific objects added: chief editor, editorial secretary,
editorial board, moderator, consultancy income, tangible research property,
project closure, licence agreements, payment, journals. A recipient is the
head noun of its phrase. A lettered item quoted without its lead-in credits a
bound value only when it names the clause's specific object or its own
context words. "Royalty share" and "share for inventors" ask for a
percentage; "who is the chief editor" asks for a named person, which the
research chapters never give.

Yes/no questions (round 2). With no slot or object focus, the words of a
yes/no question that no research anchor covers are required ("Is the seed
grant renewable?" requires "renewable"); the question's own verbs, pronouns
and topic words are not. A pronoun-led slot keeps its verb ("after I leave"),
"made with" and "using" are slots, and agent nouns built on a research verb
("publishers", "memberships") are kept. A value family triggered by a noun is
dropped when the question asks whether, not how much ("Is the per diem
taxable?"). Words match only in singular, plural, possessive, -ing, -ed and
British or American forms.

## Ask shape: completeness fails safe (Step 4.5A-R)

Root cause. Until Step 4.5A-R a clause was complete as soon as every
requested detail the specialist RECOGNISED was covered. The detail families,
categories and focus terms are lexical; an ask they did not recognise ("What
is the timeframe for ...?", "Who stamps a research proposal?", "What colour
is the form?") requested nothing, `uncovered` was empty, and topic relevance
alone made the clause `supported`. A partly recognised ask failed the same
way: the family was checked, the unrecognised action, subject or purpose was
not ("Who approves EXTENDING the advance?", "Can a CO-INVESTIGATOR apply?",
"for RESEARCH purposes"). Three adversarial rounds each closed the listed
phrasings and each found fresh ones, because the rule itself failed open.

The rule. Every substantive segment of a clause (navigation segments are
Institutional's) is assessed from its question structure, not from a phrase
table:

| Shape | When | Effect |
|---|---|---|
| `general` | an overview: "what is / are X", "what does X do", "the role / responsibilities / functions / types of X", "tell me about / explain / describe X", with no embedded interrogative, whose head noun is a research topic (an anchor concept, a named body, a proper name) or a narrow overview noun of one (role, functions, types, policy ...); a property of a topic ("the TTO's office number", "the application dates", "the procedure to ...") is never general | topic evidence may answer it (unchanged) |
| `specific_recognised` | the interrogative kind has a detail family that can answer it: who needs a party family (authority, responsibility, ownership, eligibility, identity, role, funding source, permission); when a timing family; how much, how many, how long, how often, how early the value, count, duration, frequency or lead-time families; where a publication type; a yes/no question any family, a focus property, an existence form ("is there", "does the University offer") or a requirement ("do I need", "is X required"); what / which any family or a selection form ("what research counts") | the family's evidence is required, and a grounding detail requires, in the statement that states the answer: the words of the ask the specialist cannot map, the head noun of the asked object for party questions, the yes/no subject ("co-investigator"), the same kind of act ("approves" is never answered by "reviews and recommends"), no other stated purpose ("for commercial purposes" never answers "for research purposes") and no other restricted beneficiary ("awards for coordinators of research groups" never answers a question about faculty members) |
| `specific_unknown` | any other specific ask, including why questions, "how do I ..." outside the process cues, and unrecognised timing wording | a requested detail that no evidence can cover is added: the clause is at most `partial` (or `not_found`), with "the specific ask of clause N is not recognised by the Research specialist" in `missing` |

An empty detail list is therefore not a blanket `partial`: a general overview
with no requested detail is still `supported` by the unit that describes the
topic. A clause made only of a bare noun phrase ("... and my conference travel
request", "what about my final report?") inherits the ask of the question
head ("What is the deadline for my conference travel request?"); when the head
asks something specific that cannot be carried over, the clause is
`specific_unknown`. An unrecognised "where" question is also handed to
Institutional Navigation. Each clause report carries `ask_shape`,
`ask_kinds`, `ask_residue` and `inherited_ask`.

Why unknown asks fail safe. The specialist is deterministic and extractive;
it can only confirm an answer whose kind it recognises. Fresh language will
always contain asks outside any word list, so the safe default for a specific
ask it does not understand is to quote the related evidence as `partial` and
say that the ask was not recognised, never to present topic text as the
answer.

## Relation contract: conservative `supported` (Step 4.5A-D)

Decision (D-076). The Research specialist stays a deterministic evidence
specialist. After Step 4.5A-R the remaining wrong `supported` results were
relation misbindings: the answer statement contained the asked words but
stated another relation (another action on the same object, the same rule
for another role, another purpose, another receiver). Word co-occurrence
cannot establish who does what to whom, and each round of paraphrase
patterns only moved the boundary. The specialist therefore no longer
settles relations it cannot prove:

- A `specific_recognised` clause is `supported` only when its relation is
  STRONG: one unit covers the grounding detail and the rules below prove,
  in the statement that states the answer, the relation the question asks.
- Anything the rules cannot prove is AMBIGUOUS: the clause is at most
  `partial`, the evidence is kept as a finding, and `missing` says "the
  asked relation (...) is not established unambiguously by the evidence for
  clause N; semantic relation verification belongs to the Milestone 5
  Verifier".
- A false negative is preferred to a false-positive `supported`.
- `general` asks keep their own rule (topic evidence may answer an
  overview); `specific_unknown` asks are never `supported`.

Semantic relation verification belongs to the Milestone 5 Verifier. Milestone 4 specialists fail conservatively when deterministic relation evidence is ambiguous.

The relation frame (`RelationFrame`) is read from the clause, or from the
restated ask an elliptical clause inherits:

| Part | Read from the question | Proved in the answering statement when |
|---|---|---|
| roles | one closed role lexicon: principal investigator, co-investigator, research assistant, inventor, student, staff, visiting researcher, postdoc, coordinator, dean, chair, VCRGS, VCAA, Chancellor, TTO, editorial roles, research boards and committees, consultant, referee, moderator | every asked role is named in the statement, or the statement refers to it with a pronoun and the unit names it |
| actions | one action-class lexicon (approve, review, recommend, submit, apply, register, extend, close, transfer, publish, disclose, sign, appoint, pay, distribute, attend, supervise, own, manage, teach, ...), from a verb or, in a relational ask, a nominal; classes the detail family already implies are skipped (an amount implies paying), except that a timing ask always states the action it times | the class occurs as a verb or nominal (never a noun compound such as "the travel budget") predicated of the asked head noun: the head of an active verb's object noun phrase, or of its one of/for complement when the question names that phrase's head ("covers copyright protection for ... software"); a passive's subject (a printed heading or an introductory adverbial is not the subject); the noun a reduced participle follows; anywhere in the clause a nominal is stated of ("X requires the approval of"). A participle after a comma or a conjunction has no provable noun. A bare definite object ("reviews THE application") or a clause-level nominal proves a modified target ("my research GRANT application", "the application FOR A RESEARCH GRANT") only when the statement states the modifiers. For a timed family the value is in the same coordinate part as the action |
| purpose | "for commercial / research / teaching / personal ... purposes" | the statement states that purpose |
| receivers | the receiver of an asked percentage ("to the inventor") | named as the receiver in that statement ("5% to the Inventors' College" does not name the inventor) |
| share owner | "of the University's share" | the statement states that share, or the lead-in of its list does |
| required thing | "Is X required", "Do I need X", "Should researchers get X" | X is the object of the statement's obligation ("must obtain IRB approval"), the subject of "is required", or stated by the verb of its action class ("must be approved") |
| permission | a permission family | the permission governs the asked action ("not allowed to combine teaching ..." never answers "can faculty teach?") |
| proposition | the content words of a yes/no ask, its actions as classes | contained in the unit |
| competing objects | "research advance" | a unit about the petty cash advance never answers it unless it names the research advance |
| position (subject, agent, patient) | the party before the auxiliary's predicate ("Does THE VCRGS submit ...", "Is THE VCRGS entitled ...", "how much can THE VCRGS receive"), the agent of a passive ("determined BY THE HEAD ..."), the patient ("Can COMPENSATED CONFERENCE TRAVEL be authorized ...") | the party is the subject of the same predicate (or, for a passive, its agent; for a nominal, its possessor or "of"/"by" phrase); a passive asked subject heads the stated passive subject with no contrasting restriction ("UNCOMPENSATED conference travel"); a generic asked party ("researchers", "I") is not answered by a statement whose subject is an authority or another population ("The IACUC ensures ..."); a subject stated as a capacity counts ("may not act as principal investigators") |
| arguments (to, from, by, among, between; for, at) | "report ... TO THE IACUC", "consent FROM THE IRB", "deducted FROM THE COLLEGE BUDGET", "shared BETWEEN the inventor and the university", "FOR STUDENTS" | for a yes/no ask, stated with the same preposition and the same words ("to the IRB" is not "to the IACUC"); a "for"/"at" phrase may be stated without its preposition when the statement has none; a value ask binds only a phrase that names a role; time adverbials ("at the same time") are not arguments |
| object | the noun phrase after the predicate ("offer VISITING RESEARCHERS opportunities", "have their RESEARCH load reduced"), or a fronted one ("how many REGIONAL CONFERENCES can faculty attend") | its head is in the stated object phrase and the stated phrase restricts the head with no word the question lacks ("TEACHING loads"); an asked modifier that is not the topic is stated in the unit; a quantity wrapper ("a maximum of four theses") is transparent |
| copular and value complements | "Is the Chancellor the general supervisor OF THE RESEARCH GROUPS", "the sabbatical leave OF A FACULTY MEMBER" | the stated head's own complement shares a word with the asked one ("of the journal", "for visiting researchers" do not) |
| negation and content | "commercial use", "at the University's cost", "viewed as research", "30%", "AFTER the study", "WITHOUT the project budget", "ONLY to the faculty member" | an occurrence preceded by no / not / non / without / never / except states nothing unless the question negates it too; numbers, "only", "without", "both" and temporal words (with their synonyms: before / prior) are content the unit must state |
| period | "per month", "per semester", "in two years", "per trip" | the statement states the same period |
| conditioned role | "in the event the Dean is planning ..., a written approval from the VCAA is required" | a rule conditioned on a role the question does not name answers nothing |
| clause-level nominal | "..., upon the Chancellor's approval", "..., subject to approval by the VCRGS" | the asked object heads the object of the clause's verb ("may approve AWARDS for coordinators": an approval of awards, not of coordinators) or, without an active verb, its subject |
| percentage base | "what percentage of TOTAL licensing revenue" | an item of an owner's share ("The University's share ... as follows: 40% to the VCRGS") never answers for a base with a scope word (total, net, gross) |
| counted object | "the minimum number of EXTERNAL researchers" | the counted object's restricting words are stated in the statement that states the count |

Each clause report carries `relation` (`strong`, `ambiguous`, or null for
general, unknown and process asks); each finding of a recognised ask carries
`metadata["relation"]`. A `supported` clause may list, after the strong
finding, a weaker finding marked `ambiguous`; only a strong finding states
the answer.

The rules are structural (the form of a word in its sentence, the position
of a party relative to the predicate, noun-phrase heads, complements,
coordinate parts, negators) over one role lexicon (now including named
bodies: IRB, IACUC, the ethics committee, the Library, the Purchasing
Department, "Head of ...") and one action-class lexicon (nominate is not
appoint); they are not question phrase patterns, and the specialist has no
semantic parser, classifier or model. The position rules were added after
the fresh adversarial relation check of this step, whose probes showed that
naming a party is not stating its relation: the VCRGS named as the approver
of an advance answered "is the VCRGS entitled to the advance"; the Head
named as the recommender answered "are prices determined by the Head".
A subject-less list item whose only mention of the asked party is an agent
phrase ("Reviewing applications submitted BY FACULTY MEMBERS") does not
answer "do faculty members review"; a recipient after the verb ("disclose
TO the TTO") is not its object; a list lead-in supplies the predicate of
its items ("shall be distributed as follows: a. 40% to ..."); a duty item
that opens with a gerund ("4. Assessing the colleges' needs ...") has its
subject in a lead-in the quote does not carry and so answers no question
about a party. What they cannot
decide (pronoun and anaphor resolution, participle attachment, synonyms,
inference from a heading or a neighbouring sentence) is the Verifier's.

Fresh adversarial relation check (read-only, 1069 fresh questions from
seven probe dimensions on the real handbook fixture, each under the
every-chunk-first and the lexical-overlap rankings): 29 `supported`
results, every one inspected; 0 MAJOR relation false positives. The
`supported` results state the asked relation (a list lead-in may name the
party: "The responsibilities of these committees include: 1. Reviewing
research grant applications ..."); four are MINOR, a stated relation
whose scope is looser than the ask: the item on authors from outside the
University (which itself says they follow the same procedures as faculty
members) for "Who approves book funding for university faculty authors?"
and "Who approves authored books by faculty members?"; "invoices approved
by the Dean until the advance term's closure" for "When must petty cash
invoices be approved by the Dean?"; the per-day local per diem and its
three-day maximum for "What is the total per diem for a whole local
conference trip?". The 178 questions of the Step 4.5A-R check replay to 4
`supported` results, all correct answers.

Recorded false negatives (each `partial`, with the evidence quoted):
"Who nominates the Editorial Secretary?" (", nominated by the
Editor-In-Chief": attachment after a comma); "Who reviews the research grant
forms?" ("have them reviewed by research committees": a pronoun object);
"Who reviews the application for a research advance?" and "Who approves a
research advance?" (the Purchasing Committee sentence says "the
application"; the advance is named in the previous sentence); "Is IRB
clearance required ...?" (the handbook says "IRB approval"); "Who approves
the annual research progress report?" ("an annual progress report for the
project, approved by the Dean"); "Who approves the registration of research
groups?" (Board approval is one item of a list of conditions); "What must an
inventor disclose?" (the obligation is stated for "individuals subject to
this policy"); "Do research committees review research grant
applications?" (the list lead-in says "these committees"); "Does the TTO
choose the method to transfer IP?" ("The TTO, in consultation with the
inventor(s), shall choose": a parenthetical between subject and verb);
"Does the Dean approve payment procedures?" (the earlier approval-target
layer reads "student research projects" before the passive); "Can the Dean
approve the closure of a project?" (the permission scope of "closure");
"Is my petty cash advance subject to VCRGS approval?" ("subject to" binds no
predicate); "Who approves conference travel?" stays `supported` on "approves
travel requests" because "conference" is the topic, not a restriction the
rule requires.

## Quote units

The shared rules apply (a list item whole, else a sentence; lettered items;
antecedents for pronoun-opening sentences; headings, lead-ins and fragments
are never evidence). Research also uses the shared opt-in lead-in join: a
list item that directly follows the unit introducing its list ("For promotion
purposes, research considered admissible includes:", "Requests must be
submitted at least one month before the conference date, including:") is
quoted together with that lead-in when that makes the extract more relevant.
Only the first item can be joined, a sibling item never is, and a bare
heading ending in a colon is not a lead-in. The lead-in and the item remain
separate local units for the detail rules.

A quote that opens a numbered or lettered item is extended over the rest of
that item ("15. The principal investigator must hold a Ph.D. ...
Additionally, they must be full-time employees ... and not visitors."): the
item is one evidence unit. A sibling item is never added, and the extract
stays within the allowed spans and 90 words. For a required-documents
question, a quote that holds an item of a list is extended backwards to the
list's lead-in and forwards to its last item; the documents detail counts only
when the whole list is quoted, its end is visible in the chunk, and it fits
120 words. Otherwise the original quote is kept and `missing` says that only
part of the handbook's list is in the evidence. Nothing is ever added that the
chunk does not print.

## Procedural safety

- A system or click procedure (portal, screen, button, menu, upload field,
  "where do I click") is never `supported`: at most `partial` when the
  evidence names the system or shares an anchor, otherwise `not_found`, with
  the absent procedure named in `missing`.
- A policy-process request ("How do I apply for a research grant?", "Walk me
  through applying for a seed grant", "How do I fill in / fill out / complete
  the invention disclosure form?") is at most `partial`: the handbook states
  stages (forms reviewed by research committees, forwarded to the VCRGS) but
  not the complete submission procedure, and `missing` says so.
- A review-chain question ("Who reviews research grant applications?") can
  be `supported` when the handbook states the chain.
- Click, screen and step requests are process requests whether or not a
  system is named ("What link do I use to apply ...?", "Which button do I
  click ...?", "List the steps needed to ...", "What are the instructions
  for ...?").
- Instructions are never generated or inferred.

## Status semantics

- `supported`: every owned clause has evidence that passed all gates and
  every requested detail is covered in one unit; `missing` is empty.
- `partial`: useful evidence exists, but a requested detail, a procedure or a
  complete policy process is missing, or a clause was unowned or not
  processed.
- `not_found`: the clauses are Research's but no evidence passed the gates.
- `out_of_scope`: no clause is Research's; handoffs name the owner.
- `error`: an execution failure or an unapproved `source_scope`, with
  secrets and paths redacted. A reranker that returns a different number of
  scores than candidates is an execution failure, never a silent `not_found`.

Topic relevance makes a chunk eligible to be searched; it never makes a
clause complete. A yes/no, ownership, permission, lead-time or
specific-focus question is `supported` only by a unit that states the asked
thing; related text alone yields `partial`. Since Step 4.5A-D a recognised
specific ask is `supported` only with a STRONG relation; an AMBIGUOUS
relation is `partial` (see "Relation contract").

## Dependency injection

`ResearchResources` (frozen dataclass): `embed`, `rerank`, `index`, `chunks`,
`metadata`, `source`, `section_map`. `from_knowledge_base(kb)` refuses an
unregistered document, another source id or an unapplied section map.
Importing the module loads no model and opens no connection; there is no
module-level registry.

## Integration with the Faculty Onboarding Coordinator

Research ownership terms are one lexicon (`specialists/domain_cues.py`,
Step 4.5A-R): the Coordinator builds its Research cues from it, the Research
specialist accepts any clause it marks as Research's (besides its own cue
table), and the Teaching specialist hands such a clause off. A phrase the
Coordinator routes to Research can no longer be rejected here as carrying no
research cue.

The dispatcher's `EXECUTABLE_SPECIALISTS` is Teaching and Research. A
research task runs only when a `ResearchInnovationSpecialist` is registered;
otherwise it stays pending ("executable but not registered"). Faculty
Services & HR and Institutional Navigation tasks stay pending. The Coordinator
routes research questions with the cues listed in `docs/COORDINATOR.md`,
including the thesis supervision split and the WLAM demotion.

### Retrieval route per focus clause (Step 4.7, D-078)

`task.intent` is the Coordinator's classification of the joined focus
clauses, so a contact clause beside a substantive one ("Who approves the
petty cash advance and what is the VCRGS phone number?") makes it `contact`.
The specialist no longer applies that intent to every clause. Each searched
clause takes its route from `retrieval_intent.resolve_clause_retrieval_intent`
(shared with Teaching): a task with one focus clause keeps the task intent;
a clause of a multi-clause task is classified on its own with the same
deterministic router; `contact` is kept only by a clause that is itself a
contact ask, and a clause whose navigation analysis found a substantive half
retrieves under `policy`, since the specialist answers that half and hands
the contact half to Institutional Navigation. The contact route ranks
directory rows and phone-bearing lines first and, applied to a substantive
clause, crowded the answering paragraph out of the candidate window: an
answered question came back `not_found` (the Step 4.6 MAJOR). The clause
report and each finding carry `route`; `metadata["route"]` remains the task
intent as given. Ownership, handoffs, the ask shape, the relation contract,
the section guard and the source scope are untouched.

## Known limitations

- Ranking has only been exercised with stand-ins: the real embedding model
  and cross-encoder are not available locally. Real-model validation of
  retrieval and answer quality is outstanding; no accuracy is claimed.
- Row boundary inside a paragraph window (Milestone 3 R-1, carried into the
  shared evidence module). A bullet ending without punctuation followed by a
  capitalised sentence forms one unit; in a synthetic analogue ("o Costs of
  research assistants and forfeiture of unused funds" followed by the
  research advance sentence) the word "forfeiture" satisfies a penalty
  question. The Step 4.5 audit found no such shape in the real Research
  scope; the synthetic regression test is kept and no row break is
  attempted.
- A list lead-in whose FIRST item shares nothing with the question is never
  quoted by the shared quote selector (page 119: "Requests must be submitted
  at least one month before the conference date, including: o Approval from
  the concerned college."), so "When must conference requests be submitted?"
  is `partial` on the real page, never answered from another unit. Bounded
  false negative in the shared evidence module (Teaching is frozen).
- A lettered item whose list lead-in is on the previous page carries no
  context ("d. The funding ceiling for each project is AED 5000." on page 168
  belongs to student research projects on page 167). Since round 2 it credits
  a bound value only when it names the clause's specific object or context
  words, so general questions stay `partial`; the student-projects question
  itself is a bounded false negative for the same reason.
- Sentence-level timing binding is conservative: a unit whose subject is
  named only in its section heading ("h. The TTO shall inform the
  Inventor(s) ... no later than 90 calendar days", under 9.4.2 Disclosure
  Evaluation) does not answer the disclosure-outcome deadline.
- The yes/no focus reads words, not meaning: when the handbook uses a
  synonym of the asked property, a legitimate yes/no question is `partial`.
- `section_no` stays the page-primary label (5.5 on the page 119 per diem,
  16.5 on page 266 IRB text). A quote-level attributed section is deferred to
  Step 4.7 or before Synthesis: the section guard records only an owned or
  not-owned flag per span, and deriving the section would mean changing the
  shared guard or duplicating its window logic.
- The deterministic stand-in reranker can rank a tangential item above the
  procedure page; tests that depend on ranking script it explicitly.
- Completeness fails safe since Step 4.5A-R (see "Ask shape"). The price is
  bounded false negatives: a legitimate specific question whose wording the
  specialist does not map ("What is the timeframe for research grant
  applications?", "How is the research grant ceiling calculated?") is
  `partial` even when the handbook answers it; a recognised ask whose answer
  sentence uses a synonym of an unmapped word ("insure" against
  "insurance") is `partial`; an item whose answer needs its list lead-in for
  the grounding words is `partial`; an overview whose subject has no anchor
  concept ("Explain the research and scholarly activities evaluation") is
  `partial`. Real-model validation may justify a semantic ask classifier
  later; the deterministic rule never trades safety for coverage.
- Relation contract (Step 4.5A-D, D-076). The Step 4.5A-R open classes are
  closed by contract, not by patterns: a telegraphic yes/no ("per diem paid
  in full?") is a yes/no ask with its proposition, and a relation the rules
  cannot prove is `partial`. The price is the false negatives listed under
  "Relation contract"; the Verifier (Milestone 5) is where a semantic check
  may upgrade them. The rules read one statement; a relation stated across
  two sentences or only in a heading stays `partial`.

## Future sources (not ingested)

HIGH: Research Funding Department internal grants procedure; REC, IRB and
IACUC application guide and forms. MEDIUM: external grants procedure; TTO
invention disclosure procedure; conference travel procedure and forms.
LOW to MEDIUM: research portal guide; consultancy approval form. Ingestion
is decided in Milestone 4.5.
