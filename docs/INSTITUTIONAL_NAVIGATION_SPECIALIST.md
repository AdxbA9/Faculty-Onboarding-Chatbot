# Institutional Navigation Specialist

Canonical id `institutional`. Module `handbook_bot/agents/specialists/institutional.py` with its standard-library cue,
tier, office-lexicon and directory-line module `handbook_bot/agents/specialists/navigation_cues.py` (Milestone 4, Step
4.16, decisions D-104 to D-107). **Frozen for Milestone 4 (D-107, 2026-10-29).** Deterministic version 1: no LLM call,
no network, extractive findings only. Phases B2 (telephone and fax numbers from the contact directory), B3 (office
responsibilities, organisational structure and documented library locations, Step 4.16-B3), B4 (ownership checks and
handoffs, Step 4.16-B4) and C1 (natural phrasing through a closed allowlist, Step 4.16-C1): procedures and approvals
are not answered; a Faculty Services & HR or Research & Innovation subject is handed to its owner as data (never
executed); a Teaching & Learning subject is left to the Coordinator's routing. Accepted in Step 4.16-B5 with documented
limitations; since Step 4.16-B6 (D-105) it is in `dispatch.EXECUTABLE_SPECIALISTS`, so the default dispatch executes it
whenever it is registered. The live chatbot does not use Plan F (`PLAN_F_ENABLED` is False). Built on the shared
helpers (`scoping`), never on the Teaching, Research or Faculty Services & HR modules, which are frozen.

## Freeze status (D-107)

Frozen for Milestone 4 on 2026-10-29 after the independent freeze review of the Step 4.16-C1 code. Baseline:
`institutional.py` SHA-256 `e37a5122…2e805`, `navigation_cues.py` SHA-256 `d136cd2e…74124` (full hashes in D-107). Accepted evidence: full
suite 6,302 passed before and after the review, 0 failures, 0 skipped, 0 expected failures; Institutional suite 2,293
and four-specialist dispatch 216 passed; contact accuracy 226/226 and evidence binding 690/690 under both stand-in
rankings with every `_B2_*` and `_B3_*` switch killed; the 62-question benchmark through the default dispatch
byte-identical to the stored C1 baseline (12/48, 0 wrong); the frozen 36-question holdout 5/24, 0 wrong; 346 fresh
adversarial questions (171 blind, 175 white-box) through the specialist alone and through the default dispatch, under
both rankings and repeated: 43 supported answers, all verified correct, 0 wrong supported answers, every refusal,
handoff and partial on target; determinism identical across repeats, fresh instances and rankings. Accepted
limitations: the conservative misses under "Known limits" (duty words shared with the office's own name, synonyms and
referral wordings, context outside the closed shapes, lowercase acronyms and typos, split and sub-row directory
entries, page 52 entries outside the lexicon), the Coordinator routing gap (DEP-2, a separate step), stand-in
rankings and fake embeddings only, no real-model evaluation, not wired into the runtime. Reopening needs a new
decision record (reason, affected files, regression tests, impact on the other specialists, approval, new baseline
hashes after the full gates); ordinary feature growth and recall improvements are not reasons to reopen.

## Decision owned

Whether the approved, source-scoped evidence identifies the office, unit, body or contact point a faculty question
asks for, and what that evidence establishes about it: that the office exists, that it is responsible for the asked
function, how to reach it, where it is. Expressed as a `SpecialistFindings`. The specialist does not route (the
Coordinator did), does not write prose (Synthesis will), does not verify (the Verifier will) and never calls another
specialist.

## Phase B2 behaviour: directory lookups

A channel clause (telephone, fax, "contact details", "how can I contact / reach X") that names a lexicon office is
answered from the contact directory, section 1.16. The directory is a three-column table (office, telephone number,
fax number) printed on pages 52 to 55; the index keeps each extracted line as one `row` chunk. The specialist reads
those lines once, in order, inside its compiled scope and the section guard (95 lines), so the answer does not depend
on retrieval ranking.

A number is returned only when the whole row binding holds:

1. the clause names a lexicon office, and nothing in it makes the asked unit something else: a library, clinic,
   hospital, laboratory, section for men or women, campus, a person (dean, manager, Vice Chancellor ...) or one office
   named inside another ("the Finance Department of the College of Engineering");
2. exactly one directory line holding numbers starts with the office's printed name or attested alias, followed by a
   value, a label or the end of the line ("Directorate of Human Resources ..." or "College of Medicine Library ..."
   is not the office's line);
3. that line holds exactly two complete numbers (`+971 6` and seven digits, kept exactly as printed) and nothing else;
4. neither neighbouring line runs into it (a trailing "/", an unfinished "+971" or "+971 6", a "Men:" / "Women:"
   sub-row) or completes a number broken on it, and neither neighbour is missing from the extracted lines;
5. the page's header line ("... Telephone Number Fax Number") names the two columns, which fixes which value is the
   telephone and which the fax.

The finding quotes that line verbatim (`claim == evidence_quote`), with the page, chunk id and row id; its metadata
names the field of each value (`contact_fields`, `contact_values`), the whole row (`row_values`), the column order and
the header line. Only the asked field is in `contact_values`: a telephone ask never returns the fax, and the reverse.

| Directory line | Result |
|---|---|
| office plus exactly two complete numbers, binding holds | `supported`: the asked field |
| office plus one number ("College of Sciences +971 6 5050489", whose single number is the printed **fax**) | `partial`: the finding quotes only the office name; no number |
| office plus a multi-line or Men / Women entry, a sub-entry ("Men's Library: ..."), or no number | `partial`: office name only, the reason in `missing` |
| labelled line ("Dental Hospital Call Center +971 6 5057666 (External) Ext 7666 (Internal)") | `partial`: quoted with its printed labels, never given a telephone or fax field; a fax ask is `not_found` |
| office name split over extracted lines (College of Arts, Humanities and Social Sciences; College of Graduate Studies and Research; the CCE; Business Services and Purchases Department) | `not_found`: lines are never joined |
| lexicon office the directory does not list (Main Library, Health and Safety Office, FMPD ...) | `not_found` |
| no lexicon office named ("IT Center", "IT Department", "registration" without an office noun, "Admissions") | `not_found`: nothing is guessed |
| e-mail or website ask | `not_found`: the directory prints neither, and no address is composed |

`supported` needs every asked field of every channel clause bound; anything less with at least one finding is
`partial`; no finding is `not_found`. Every result with a number carries the limitation that numbers are quoted as
printed in the 2025/26 handbook and are not verified as active. A number the directory prints for more than one entry
carries a further limitation naming the other entries (Registration, Admissions and the Information Technology Center
share +971 6 5050000; Public Relations and the Media Center share both numbers). Non-channel asks keep the phase B1
behaviour: `not_found` with no finding and one `missing` entry per clause.

Six row-binding protections (`_B2_ROW_START`, `_B2_ROW_COMPLETE`, `_B2_NEIGHBOR_CHECK`, `_B2_COLUMN_HEADER`,
`_B2_SINGLE_ENTRY`, `_B2_REFERENT_CHECK`) are module switches, all True; the negative-control tests switch each off and
show that a wrong office or a wrong number then comes out.

### Phase B3 change to the referent check (for team review)

The B3 adversarial reviews showed that the B2 referent check (a word list of sub-units and persons, and two offices
joined by anything but a conjunction) let wrong-office numbers out for unusual phrasings ("the accountant in the Finance
Department", "HR, University of Khorfakkan", "the Sharjah Police HR phone number", "Who decides the fax number of the
Finance Department?", "What is the alternate phone number of HR?"). Inside the same switch (`_B2_REFERENT_CHECK`), a
contact clause is now also answered only when the whole clause, courtesy words aside, has a plain contact form
(`navigation_cues.channel_form_problem`): "What is the phone / fax number (contact details, e-mail ...) of <office>(s)?",
"What is <office>'s phone number?", "How can I contact / reach <office>?", "What fax number does <office> use?", "Which
phone number should I use for <office>?", "... of <office> and what is its fax number?", "For <office>, what is the fax
number?", with an optional plain lower-case purpose ("for new faculty", "to send documents") and an office noun after the
name ("HR dept", "HR's main office", "... at the University of Sharjah"). Every other contact clause fails closed. All
113 B2 contact cases give the same results as in B2 (the same 118 numbers); a few natural phrasings answered in B2 now
fail closed (a context sentence before the ask, a capitalised name in the purpose such as "calling from Dubai", a
contact ask mixed with another subject). The lexicon and the directory reader are unchanged.

## Phase B3 behaviour: responsibilities, structure and locations

Every clause that is not a channel ask is answered from **statements** the specialist reads once, at construction, from
the in-scope paragraph passages of the approved handbook (inside the compiled scope and the section guard, never from
the contact-directory pages 52 to 55). Each passage is cut into units (sentences, list items, sub-items) and each unit
is read for its grammatical subject and relation, so the answer does not depend on retrieval ranking. **Evidence
levels:** a mention (E1) never answers a responsibility; a responsibility (E2) never answers a procedure; the handbook
prints no procedure (E3) for any university process, so no step is ever given.

### The question forms answered (fail closed on everything else)

The adversarial reviews showed that every extra word a question carries (a place, a person, a time, an exclusion, a
second ask, a beneficiary put in front, a qualifier of a member or a chair) can change what is asked, and that closed
lists of such words never cover them all. Phase B3 therefore answers a focus clause only when the **whole clause** has
one of these forms (courtesy words such as "please" and "thanks" aside), after the guards below have passed:

| Intent | Forms (⟦O⟧ = exactly one lexicon office; *duty* = a plain noun phrase) |
|---|---|
| responsibility | who / which office (unit, department, section, center, body) is responsible for / is in charge of / handles, manages, oversees, provides, maintains, administers, coordinates, conducts, organises, delivers, offers, supports, issues, operates, runs, supervises, facilitates, arranges, promotes, implements / deals with, takes care of, looks after *duty*; who should I contact (ask, talk to ...) about *duty* |
| office role | what does ⟦O⟧ do; what is ⟦O⟧ responsible for; what are the role / responsibilities / functions / duties / mandate / mission / services of ⟦O⟧; what services does ⟦O⟧ provide (to faculty); tell me about ⟦O⟧; is ⟦O⟧ responsible for / in charge of *duty*; does ⟦O⟧ provide (offer, handle, manage ...) *duty* |
| chair | who chairs ⟦O⟧; who is the chair(person) of ⟦O⟧; who is ⟦O⟧ chaired by |
| composition | who are the members of ⟦O⟧; what is the composition / membership of ⟦O⟧; what does ⟦O⟧ consist of / comprise / include; what is ⟦O⟧ composed of; what (are the) subgroups / centers / sections / departments / divisions / units / programs of ⟦O⟧, or does ⟦O⟧ have / comprise |
| organisation | what is / describe the organisational (governance) structure (chart) (of the University); how is the University organised / structured / governed; what are the governance bodies (of the University) |
| count | how many *units* does the University have / offer / ...; how many *units* are there (at the University) |
| colleges | which colleges does the University have; what are the colleges of the University; list the colleges |
| location | where is (where's, where can I find, whereabouts is, what is the location of) ⟦O⟧ [selectors]; which campus is ⟦O⟧ on; which building is ⟦O⟧ in; tell me where ⟦O⟧ is |

A *duty* may hold letters only, at most one "of", one "with" and one "and" / "or", at most seven content words, no
office name, no other institution or place, no pronoun, no time, condition, comparison or exclusion word, and no
preposition other than "of" / "with"; a trailing beneficiary ("for faculty") or setting ("on campus") is allowed. Any
other clause is answered with nothing and the missing entry names the forms. Guards that come first, whatever the
form: procedures and system access (no finding), approvals and authority (approve, authorise, authority, mandate to,
empowered, permitted, competent ... : no finding, never an office; see phase B4), and, decided first since phase B4,
the subjects of Faculty Services & HR, Research & Innovation or Teaching & Learning (shared `hr_owned` /
`research_owned` plus the local lists: no finding, handed off or left as phase B4 describes).

### What binds an office to a duty (E2)

A statement is bound to a lexicon office only in one of three ways:

1. **Named subject.** The office opens the statement proper (after a section number and heading, a list marker and
   item title, or one introductory clause) and is followed by a verb of a closed list (is responsible for, is in charge
   of, is dedicated / committed to, was established to, serves as, plays a role in, holds the mandate to, oversees,
   manages, maintains, administers, provides, offers, delivers, supports, coordinates, operates, conducts, organises,
   issues, extends, assists, arranges, performs the following functions). An office named later in the sentence is not
   its subject. Verbs coordinated with the relation verb bind their objects to the same subject ("DRC provide support
   and counseling, evaluate ..., and handle complaints ...").
2. **Strict anaphor.** "This Council", "The office", "The Center", "It", "Its mission is to ..." bind only to the office
   named as the subject of the unit just before it in the same passage, when the head noun fits that office.
3. **List lead-in.** A lead-in that binds the office ("The Health and Safety Office is responsible for the following:")
   binds the items of its own list, never beyond the passage; when the list runs on, the answer says so and is
   `partial`. The quote runs from the lead-in to the bound item; `statement_text` holds the item alone.

A statement whose subject is a unit outside the lexicon, or a list item without its lead-in, is **unbound**: it only
stops a single office from being named. "The University of Sharjah" / "UoS" / "the University" bind to the university
as a whole (tier `stated_fact`); "University" followed by another capitalised word starts another name; "we" and
"There are" bind to the university only when the same sentence names it. "DRC", printed for the Disability Resource
Center on page 255, is read as that office's name **only as the subject of a statement**, never in a question or on a
directory line, so no contact answer changes.

### Matching the asked duty

The duty must match a **duty phrase** of a bound statement, a list-item title ("Risk Assessment and Management") or the
object of its verb ("maintains a cutting-edge Data Center"), conjunct by conjunct: in a conjunct that holds asked
words, the last word (the head) is asked; words before the first asked word only grade it (cutting-edge, high-quality,
mandatory ...); at most one unasked word stands between asked words, never after an asked gerund or verb ("promote
safety awareness" is no "promoting awareness") and never after a preposition; asked words inside a prepositional
complement do not count ("infrastructure for ... academic activities", "assists faculty in preparing agreements").
Shared heads and modifiers are read across "X and Y" titles ("Health and Safety Training", "Risk Assessment and
Management"). A list-item title is no wider than its own body: "Policy Development and Implementation: Establishing and
maintaining health and safety policies ..." names health-and-safety policy duties, so a general "policy development" ask
gets the office only as a narrower candidate. An object cut at a preposition keeps its complement ("instruction in using the library resources",
"handle complaints from individuals with disabilities"), so the duty is never read wider than printed. Unit nouns head
their phrase, and an ask keeps a unit noun after a content word ("the Data Center" -> data, center): "Data Center"
answers "Who maintains the Data Center?", never "Who is responsible for data?". The office's own name never counts as
its duty, and a duty made only of beneficiary words ("faculty") is no duty.

| Statements that match the asked duty | Result |
|---|---|
| bound to exactly one office, and no other statement (another office, a unit outside the lexicon, the university; in the scope, or an approved-handbook sentence outside the scope that states a duty) contains all the duty's words by exact stem | `supported`: that office, up to three quotes |
| bound to one office, but another statement contains all the duty's words | `partial`: the office's quotes, "other passages also use the words of this duty ... (page ...)" |
| bound to more than one office, or one office and an unbound unit or the university | `partial`: every candidate quoted, "no single office is chosen" |
| unbound only, or the university only | `partial`: the quote, "no office is named as the answer" |
| none, but bound statements name a narrower duty with two or more of the asked words as modifiers ("Health and Safety Training" for "health and safety"), or, for a named office, the asked duty with a restricting complement | `partial`: those offices as candidates, "narrower duties named with these words", never the answer |
| none | `not_found` |

### Structure, counts and colleges

Chair and composition answers quote the office's chair and composition statements (tier `stated_fact`); a
composition naming no members is `partial`; an asked part type must be listed in the statement ("five specialised
centers", "comprises a diverse range of departments"); a part asked as the unit ("the members of the FMPD's subgroups")
and a vice / co- / acting chair are not answered. The organisational structure is answered only by the bare form
(the 1.3 composition statement, with the limitation that the chart, a figure, is not read; reporting lines are never
inferred from layout). Counts are answered for the university as a whole only, from a university statement that prints
the number before the asked noun (fifteen academic colleges, nine libraries, eight Vice Chancellors); a printed
modifier that changes the office ("eight Vice Chancellors" for "chancellors") rejects the count. Officeholders are
never named.

### Documented locations

Only the four library location entries of section 11.6 (page 199) are locations.

| Entry as printed | Code | Place |
|---|---|---|
| Main Library (W16) | W16 | the University's Women Campus (W16) |
| Main Library (M16) | M16 | the University's Men Campus |
| Medical Library | none | the Medical and Health Campus of the University |
| Fine Arts and Design Library (M22) | M22 | the Fine Arts and Design College |

A selector picks an entry only when it is printed right after the library's name (or a gender / code right before it):
a gender serving frame ("for women", "for female faculty"; "for women's studies" is no frame), a place ("on the Men
Campus", "at the Fine Arts and Design College"), which must be the entry's printed place, or a building code (W16,
w16, M-16). "Where is the Main Library?" and "... for men and women?" give both Main Library entries (`partial`); a
negation, or a gender word or code not attached to the name, and any other words after the name are not answered. Floors, rooms, hours,
maps, GPS, directions, branches and other institutions are `not_found`; "which campus" for the M22 entry and "which
building" for the Medical Library are `partial` (none is printed).

### Evidence-binding protections

Eleven module switches, all True, reported in `metadata["statements"]["evidence_binding_protections"]`. The
negative-control tests switch each off alone and show that a wrong office, a wrong location or an invented answer then
comes out (`qa/institutional_4_16/evidence_binding_runner.py --mutate`).

| Switch | Protects against | Killing cases |
|---|---|---|
| `_B3_SUBJECT_BINDING` | an office named anywhere in a sentence taken as its subject | B20, B23 |
| `_B3_RESPONSIBILITY_VERB` | a named office without a responsibility verb taken as responsible | F05 |
| `_B3_ANAPHORA_STRICT` | "The Center" resolved to a unit other than the one just named | B23 |
| `_B3_LIST_EXTENT` | list items beyond the passage bound to the lead-in office | A16 |
| `_B3_DUTY_ALL_TERMS` | one shared keyword taken as the duty | A07, B13, C09 |
| `_B3_AMBIGUITY_CHECK` | one office named when another unit is stated to do the same | S01 |
| `_B3_OWNERSHIP_GUARD` | another specialist's subject answered here | H08, H16 |
| `_B3_PROCEDURE_GUARD` | a system-access or procedure ask answered as a responsibility | G08 |
| `_B3_LOCATION_ENTRY` | any sentence naming a unit and a place taken as a location | S02 |
| `_B3_SAME_NAME_CHECK` | one of two same-name entries chosen silently | E01 |
| `_B3_OFFICE_WORDS_EXCLUDED` | an office's own name taken as its duty | B22 |


## Phase B4 behaviour: ownership checks and handoffs

Institutional Navigation works only on the focus clauses of its own task (`context["focus_clauses"]`, the Coordinator's
cut of the question). It cannot see which other specialists were selected, never calls one, and creates no new message
type: a clause it leaves to another specialist is returned as a `HandoffRequest` in its `SpecialistFindings`, which the
dispatcher preserves and never executes (one handoff round at most when the orchestrator runs them, D-PLANF-005).

Before a non-contact clause is answered, `_boundary` reads whose subject it is, with the shared `domain_cues.hr_owned` /
`research_owned` (unchanged) and the local HR and Teaching subject lists of phase B3:

| Clause | Result for the clause | Handoff |
|---|---|---|
| a Faculty Services & HR or Research & Innovation subject the SHARED cues name ("Which office handles faculty housing?", "Who approves a faculty research request?", "How do I apply for annual leave?") | no finding, no `missing` entry (as in the other specialists); limitation "handed off, not answered; Institutional Navigation names no office for them" | one per owner the shared cues name, research before HR ("Who approves research leave for staff on annual leave?" goes to both; no choice is made between them) |
| a clause only the local HR word list claims ("Who is responsible for environmental health?", "Who should I contact about my health?") | no finding; `missing`: "may belong to Faculty Services & HR ... the shared ownership cues do not claim it" | none: the local list only stops an answer; it never names a target |
| a clause the shared cues give to HR or Research that also has a Teaching subject ("Who handles exam appointments?", "Who handles the publication of exam results?") | no finding; `missing`: "its words point to more than one specialist ... ownership is ambiguous" | none: a handoff would guess |
| the same, in a handoff task from that owner (`task.requested_by`) | no finding; `missing`: "not handed back, and Institutional Navigation does not answer another specialist's subject (no statement is searched for it)" | none back to the requester (another named owner still gets one) |
| a Teaching & Learning subject ("Who approves grade changes?", "Who is responsible for office hours?") | no finding; `missing`: "its subject belongs to Teaching & Learning ... no handoff is created" | none: the shared cues hold no Teaching ownership lexicon, so the target would be a guess (DEP-3); the Coordinator routes Teaching subjects |
| a contact ask (any office) | phase B2 directory answer | never handed off |
| a location ask (any unit, "Where is the HR office?", "Where is the research office?") | phase B3 location answer: only the four library entries; another unit `not_found` ("no documented location entry" or "names no office of the lexicon, so its unit cannot be verified") | never handed off |
| a form whose object is a named office ("What does Human Resources do?", "Who chairs the College of Graduate Studies and Research?") | checked with the office names masked, so an office's own name is never another specialist's subject; answered by phase B3 | only if the rest of the clause names an HR or Research subject ("Is Human Resources responsible for leave requests?") |
| an approval or authority ask with an Institutional object ("Who approves purchase requests?") | `not_found`: approval and authority asks are not answered in Milestone 4, and an office that manages or handles a matter ("OIR ... managing approvals") is not shown to approve it. The owned chapters do print governance-level authority (the College Council "holds the mandate and authority to oversee ..." and its functions include "Approving curricula proposed by department councils", page 25; the Board of Trustees is "the University's highest authority", page 21); reading it is future work (C8) | none |
| anything else | phases B2 and B3 unchanged | none |

A task none of whose clauses is Institutional Navigation's is `out_of_scope` (with its handoffs, or with none for a
Teaching subject); otherwise the status comes from the clauses it owns. The handoff is built like the other specialists'
handoffs: `requested_by="institutional"`, the full question, exactly one focus clause, the task's intent, level and
system, `metadata` with the origin task and clause index, kind `misrouted`; no finding, quote or value travels with it.
Every result records `metadata["boundary"]` (the requester, the clauses owned, the handoffs made and the four
switches).

**Mixed questions.** The Coordinator cuts a question into clauses and gives each selected specialist its own; each
specialist answers only the clauses it owns, so a part is never answered twice ("What is the phone number of HR and how
much annual leave do I get?": HR answers the leave part and hands the phone part over, Institutional Navigation gives
the number). Nothing merges the results in this milestone; Institutional Navigation's findings stand alone. A clause that
itself holds two asks is read as one and fails closed (it is not split again here, because splitting re-opened the
bypasses of the B3 reviews).

**Known effect of separate tasks.** For "which office handles / who should I contact about <HR or Research subject>"
the Coordinator can give the same clause to both specialists; the subject specialist hands the navigation part to
Institutional Navigation (D-052, D-063, D-072) and Institutional Navigation hands the subject back (requirement R9), so
in the first round neither answers and both handoffs are kept as data. When the orchestrator runs handoffs (one round),
Institutional Navigation does not hand the clause back to its requester and does not answer it (it searches no statement
for another specialist's subject, so it does not claim that its chapters name no office).

**Plan-internal conflict (for team review).** Three rows of the Step 4.16-A routing document predate requirement R9
and expect Institutional Navigation to report on the office part of an HR subject: "Which office handles faculty
housing?" (`not_found` / `partial`, never Student Housing), "Which department handles employment-related questions?"
(`partial` from the HR directory line) and "Who do I contact about my leave balance?" (`partial`). R9 says a clause the
shared HR or Research cues claim is `out_of_scope` with a handoff. Phase B4 follows R9: the three are `out_of_scope`,
with no finding (Student Housing is never named) and an HR handoff. With the ownership check switched off they would be
`not_found` with no office anyway (no statement binds these duties and the directory line is E1 only), so only the
status and the note differ. The same document's "Which office handles purchasing requests?" (`partial`, two business
services names) is `not_found` since phase B3 (the duty slot is not a plain form); it fails closed.

### Phase B4 boundary protections

| Switch | Protects against | Negative control (`tests/test_institutional_integration.py`) |
|---|---|---|
| `_B4_OWNER_HANDOFF` | an HR or Research subject left with nobody to hand it to | "Which office handles faculty housing?", "Who approves a faculty research request?" |
| `_B4_NO_HAND_BACK` | a clause handed back to the specialist that handed it over (a loop) | "Who should I contact about research ethics?" from Research; "Which office handles faculty housing?" from HR |
| `_B4_NAVIGATION_KEPT` | a navigation ask handed to the owner of the unit it names | "Where is the HR office?", "What does Human Resources do?", "Where is the research office?" |
| `_B4_SHARED_OWNER_ONLY` | a handoff target taken from Institutional Navigation's own word list, or a Teaching subject handed to HR or Research because one word matched a shared cue | "Who is responsible for environmental health?", "Who handles exam appointments?" |

The phase-B3 `_B3_OWNERSHIP_GUARD` still decides whether ownership is checked at all (off: every clause is answered and
nothing is handed off); it stays killed by H08 and H16.

## Step 4.16-B5 acceptance (QA only; no production change)

Institutional Navigation was verified against its approved requirements (C1 to C12, U1 to U4, R1 to R11, the status
semantics; traceability in `SENIOR_PROJECT_HANDOVER/INSTITUTIONAL_STEP_4_16_B5/`). Results:

- **Safety:** 0 wrong answers on 62 independent questions written by blind authors from the handbook pages
  (`qa/institutional_4_16/coverage_cases.jsonl`), on the 690 evidence-binding runs and on the 226 contact runs. Every
  refusal of an unanswerable or foreign question was kept (14/14), and every citation was on its expected page.
- **Usefulness:** the narrow question forms answer few natural phrasings. 6 of the 48 source-answerable independent
  questions were answered correctly through the Coordinator and the dispatcher, 4 of 48 directly.
  - Causes: a greeting, a request verb ("Could you please give me ...") or a context sentence in the clause (12, and
    more among the Coordinator misses); the Coordinator not selecting Institutional Navigation (18 of the misses; DEP-2); units outside the lexicon
    (Security Services Unit, Transport Section, Mail Office, Central Laboratories Directorate, Facilities Management
    Section, Deans Council, UAAC, Board of Trustees); ownership word cues (6).
  - None gave a wrong answer.
- **Coordinator routing on the independent set:** exact selection 34/62; Institutional Navigation selected when
  expected 38/62.
- **Evidence conflicts (test group G) and determinism (group H):** `tests/test_institutional_conflicts.py`. Two
  directory lines for one office give no number; two units for one duty give both; a chunk from another source is
  never read; answering never calls the reranker, the embedder or the index.

The findings are classified in the B5 acceptance matrix (no P0 or P1; the coverage gap and the Coordinator routing gap
are P2, non-blocking for B6 and blocking for live integration). Recommended next: B6, then a bounded coverage correction
before the freeze review.

## Step 4.16-B6: default dispatch enablement (D-105)

`dispatch.EXECUTABLE_SPECIALISTS` now holds all four specialists (the only production change of the step, in
`dispatch.py`). The default `dispatch()` / `run_question()` executes Institutional Navigation's Coordinator tasks when
it is registered; an unregistered specialist and a task a handoff produced stay pending; handoffs remain data. Results
are unchanged: the default dispatch meets every recorded B4 matrix expectation, and the 62-question blind benchmark
through the default dispatcher reproduces the Step 4.16-B5 result question by question (routing 34/62 exact,
Institutional recall 25/46, coverage 6/48, 0 wrong answers, citations 6/6;
`qa/institutional_4_16/coverage_baseline_b5.jsonl`, `coverage_runner.py --default-dispatch`). The frozen specialists'
results are identical whether Institutional Navigation runs or is not registered. Only phase-state test assertions
moved (two in frozen specialists' test files). New tests: `tests/test_four_specialist_dispatch.py`.

The 12.5% end-to-end coverage, the Coordinator routing gap (21 of 46 Institutional-owned questions not routed), the
mutual HR / Research decline and the missing Teaching handoff remain integration risks; B6 does not change them.
Freezing this specialist needs a bounded coverage correction and a freeze review first. Known stale text recorded for
the freeze review: the module docstring of `institutional.py` (it still says the specialist is not in the default
dispatch; B6 was limited to `dispatch.py`), `docs/COORDINATOR.md` (it still calls Faculty Services & HR and
Institutional tasks pending), and two frozen test names that describe the earlier phase state.

## Step 4.16-C1: natural phrasing (coverage correction; Coordinator unchanged)

Faculty ask politely, add a sentence of context and ask indirectly; the plain forms of phases B2 and B3 refused most
such questions. After the B4 ownership check (which still reads the whole clause, context included), each owned focus
clause is now read as the plain question it asks (`navigation_cues.natural_ask`), and that plain question goes through
every unchanged B2 and B3 check. The reader uses closed lists only, no model:

- a greeting or courtesy word, a request frame ("Could you please tell me ...", "May I have the ...", "I'd like to
  know ...", "Do you know ...", "Can you list ...") and filler words ("actually", "really", "exactly", "basically",
  "please") are set aside; contractions are expanded ("What's" to "what is");
- after a request frame only, an embedded question is put in direct order ("where the Medical Library is located" to
  "where is the Medical Library located"; "the number of libraries the University runs" to "how many libraries does
  the University have"; "whether X is responsible for Y" to "is X responsible for Y");
- context is set aside only in closed shapes (a closed allowlist, the result of four adversarial review rounds): before
  the question, a greeting or thanks, a self-introduction ("I'm a new lecturer.", "I'm new here.", "I just joined the
  university this semester.") or the asker's own unit ("I'm joining the College of Medicine next month.", "I teach in the
  College of Law.": exactly one lexicon office, not the asked one, no place or gender word); after the question, only a
  greeting or thanks. Any other sentence (a situation, a correction, a sub-unit, a person, a time, a contact field, another
  institution, a qualifier), a trailing ", I need ..." clause, a question that refers back to its context ("its",
  "there", "this", "he", ...) and a clause naming another institution keep the clause exactly as written, and it fails
  closed as before. Word lists were tried first and broken twice by the review: only closed shapes are safe;
- a filler word is removed only between a word and a verb or predicate ("Who actually chairs", "What does X really do");
  elsewhere it may mark a correction ("X or actually Y", "phone or actually fax") and the clause is read as written;
- "my", "our", "we", "the latter" and the other references back to the context keep the clause as written;
- a clause with no question or with two questions is read exactly as written.

The result records `read_as` and `read_notes` on the clause report and `natural_phrasing` (count and switches) in the
metadata. Four switches, all on, each with a negative-control test: `_C1_NATURAL_ASK` (off: the B6 reading returns),
`_C1_CONTEXT_GUARD` (off: any context is set aside; a campus restriction is then lost), `_C1_KIND_CHANGE_CHECK` (a plain
ask of another ask kind is used only when the B4 ownership check, run again on the whole clause for the new kind, leaves
it to Institutional Navigation: a contact clause skips the B4 check, so a rewrite into a responsibility ask would
otherwise answer another specialist's subject unchecked) and `_C1_COMPOSITION_CONTINUES` (a composition
whose passage continues with "Additionally, ..." or "... also participate ..." is partial: "Who are the members of the
UAAC?").

Plain forms gained a few general wordings: "chairman/chairwoman", "who presides over (the meetings of)", "who sits/serves
on", "who is on", "run" as a count verb, "what is the building code of/for X" (a location ask; a missing code is stated),
"colleges at/in the university". Eight units printed in the owned chapters with their own statements were added to the
lexicon, all `contact=False` (never directory entries; a contact ask for them states that they are not in the
directory): Central Laboratories Directorate ("CLD", 250), Security Services Unit ("SSU", 251), Transport Section
(252), Mail Office (253), Facilities Management Section and Projects Management Section (254), Deans Council (24) and
University Academic Advisory Council ("UAAC", 27). They are never equated with the Central Laboratories, the Facilities
Management and Planning Department, the Physical Plant or the College Council. A glued heading that repeats its own
office name ("Projects Management Section The Projects Management Section is responsible for ...") is now read without
an adverbial clause, for the same office only.

Results (benchmark questions, expectations and judge unchanged; `SENIOR_PROJECT_HANDOVER/INSTITUTIONAL_STEP_4_16_C1/`):
the 62-question benchmark through the default dispatcher answers 12/48 source-answerable questions correctly (was
6/48), Institutional alone 10/48 (was 4/48); a 36-question holdout written blind and frozen before the change answers
5/24 through the system (was 2/24) and 5/24 alone (was 0/24); four of those holdout gains are not independent evidence
(disclosed in the C1 coverage comparison), so the conservative reading is 4/24 and 1/24. No wrong answer, no answer lost, every refusal kept, every
citation on its expected page, Coordinator selections identical. The first C1 version measured higher (14/48 and 15/48)
but a four-round adversarial review found that its open context handling could answer corrected, restricted or
redirected questions with the wrong office's value; the closed allowlist gives up those readings. Evidence binding
690/690 (three expectations that rested on a unit being outside the lexicon changed with it: D09, U17, S01), contact
accuracy 226/226, every B2 and B3 protection still killed. The Coordinator still does not select Institutional
Navigation for 18 of the 48 benchmark questions (DEP-2); 2 of those Institutional would answer alone.

## Decision recorded in Step 4.16-B4: D-106

The team provisionally accepts the phase-B3 design under four conditions (`docs/DECISIONS.md`, D-106): the dedicated
statement reader where the shared gate cannot bind safely; direct reading of approved evidence while source scope,
citations and deployment portability are preserved; the narrow question forms for Milestone 4, with natural-language
coverage to be measured in a later real-model evaluation; the documented coverage gaps non-blocking unless one prevents
an agreed acceptance case. Step 4.16-B4 found no condition violated (no shared module changed, handoffs carry no
evidence, no agreed B4 acceptance case blocked by a B3 gap); condition 3 stays open. D-105 stays reserved for the
default Dispatcher enablement (B6).

## Decisions recorded in Step 4.16-B3 (team decisions, no new D-number; D-105 stays reserved for default dispatch)

**Decision A: bounded direct directory reading is accepted.** The B2 directory reader reads the directory lines
directly instead of through relevance-ranked retrieval. The team accepted this provided six properties hold; a
read-only check on resources built exactly as the application builds them (`load_pdf` -> `build_chunks` ->
`annotate_metadata` with the registered source and its 206 section records; no model, no FAISS) found:

| Property | Result |
|---|---|
| 1. evidence from the approved Faculty Handbook | verified: all 95 directory lines carry `source_id` `uos_faculty_handbook_2025_26`, registered |
| 2. source-scope restrictions preserved | verified: every line inside the compiled scope (pages 52 to 55); an unapproved `source_scope` fails closed (`not_found`) |
| 3. correct document and page references | verified: findings carry the source id, the title "UOS Faculty Handbook 2025/26", the page and the chunk id |
| 4. numbers bound to the correct offices and columns | verified for the 17 bound offices: their values equal the cells of the printed directory rebuilt from the PDF's geometry, and the production-built directory and all 46 lookups equal the test fixture's. The B3 reviews showed that the B2 referent check could still bind an unusual phrasing to the wrong office; the plain contact forms above close the cases found (see the limitation below) |
| 5. no dependence on uncontrolled local paths | verified: `institutional.py` and `navigation_cues.py` open no file and use no path; the resources come from the knowledge base object |
| 6. deployment can use the approved source resources available to the application | verified on the application's build path (`InstitutionalResources.from_knowledge_base` reads `kb.source`, `kb.chunks`, `kb.metadata`, which `build_knowledge_base` builds with the same functions). Limitation: not exercised in a deployed instance, because Plan F is not wired into the live chatbot |

Phase B3 reads its statements the same bounded, direct way (in-scope paragraph passages of the registered handbook,
never the directory pages); this extends Decision A to statement reading and is recorded for team review.

**Decision B: page-52 contact expansion is deferred.** The directory begins on page 52 (Chancellor's Office, six Vice
Chancellors, the Call Center for New Students, Admissions). The 17-office answerable subset of B2 is accepted; no
telephone or fax answer was added in B3. Remaining directory coverage (the page-52 entries and the 19 lexicon offices
whose lines are one-number, labelled, multi-line or split) is future work. Contact coverage is not organisational
coverage: an office with no answerable number can still have its documented responsibilities answered (the Health and
Safety Office and the FMPD, which the directory does not list), and an office's directory line is never evidence of a
responsibility.

## Deviation from the approved plan (for team review)

The Step 4.16-A plan said B3 binds offices to duties with the shared `grounding.AskGate` / `RelationChecker` injected
with the office lexicon, and to stop if the shared gate needed a change. A read-only survey found that the unmodified
gate cannot bind the handbook's E2 statements and binds wrong offices (it treats "is responsible for" as a modal chain,
resolves no "The office" / "It" anaphor, reads a lead-in and its bullets as separate units, takes office-name words as
key terms, and gave cross-office supports on real pages 237, 250 and 251). Changing the shared gate is forbidden, so B3
implements its own statement reader in the two permitted files; no shared module was changed. The B2 handoff also
expected B3 responsibility evidence to come through retrieval; B3 reads statements directly (Decision A). Both are
recorded here for acceptance.

## Scope

Owned (no frozen specialist owns any of them): 1.1, 1.3 to 1.13 and 1.16 (structure, governance, colleges, institutional
facts and the contact directory), 11.4 to 11.6 (library staff, facilities, locations), 13.1 and 13.6 (the Health and
Safety Office, emergency communications), 15.1, 15.2, 15.3, 15.5, 15.7 and 15.8 (campus service units), and the
chapter 10 lead-in page 183 (the Information Technology Center's services and user support), admitted because no
section record covers it. Compiled page ranges: 15-16, 19-49, 52-55, 183, 198-199, 237-238, 240, 250-258. Excluded on
purpose: 1.2, 1.14 (Teaching), 1.15, the IT, library and health and safety policy sections, 10.6 (Teaching), 15.4 and
15.6 (Faculty Services & HR), and every section another specialist owns.

## Evidence tiers

| Tier | Meaning | Can support |
|---|---|---|
| E1 identification | the office is named (a directory row, a heading, a mention) | that the office exists; a printed contact (phase B2, with the row binding) |
| E2 responsibility | one statement binds the office to the asked function | which office handles something; a referral |
| E3 procedure | the steps are stated | a step-by-step answer (none exists in the handbook for university systems) |

A mention never proves a responsibility and a responsibility never proves a procedure (`navigation_cues.tier_satisfies`).
Locations, structure facts and counts need a statement of that fact. A directory line is E1 evidence only.

## Office lexicon

57 entries, each printed verbatim on the pages it lists (the tests read the handbook PDF): the 46 of phase B2 and, from
phase B3, "Library Staff" (198), "Medical Library" and "Fine Arts and Design Library" (199), marked `contact=False` so
the directory reader never reads them (the contact lexicon stays the 46 B2 entries). Step 4.16-C1 adds eight
`contact=False` chapter units with their own statements (see "Step 4.16-C1") and the aliases "CLD", "SSU" and "UAAC"
printed for them. Phase B3 adds the alias "OIR" for
the Office of International Relations (47; it has no directory line, so no contact answer changes) and the statement-only
alias "DRC" for the Disability Resource Center (255, 256), read only as the subject of a statement, never in a question
or on a directory line. Aliases only where the
handbook prints them for the same unit: "HR" (pages 57 and 72), "ITC" (183), "CCE" (54), "FMPD" and "EHSM" (254),
"Business Services Department" beside "Business Service Department" (251). Units the handbook never equates stay
distinct ("Physical Plant" and the "Facilities Management and Planning Department"; the "Business Service Department"
and the "Business Services and Purchases Department"; the "Health and Safety Office" and EHSM). "Registration" is
matched in a question only beside an office noun (the directory's office column needs none). A partial or shared word
names no office ("faculty housing" is not "Student Housing"). "IT Center", "IT Centre" and "IT Department" are not
printed in the handbook and are not aliases, so "What is the IT Center phone number?" is `not_found`; "ITC" and
"Information Technology Center" are answered.

## Boundaries

Navigation and contact are Institutional's (D-052, D-063, D-072); subject policy is the subject specialist's. The task
carries only the full question and its own focus clauses, not the other selected specialists. Since phase B4 a Research
or HR subject is handed to its owner (shared `research_owned` / `hr_owned`) and a Teaching subject is left without a
handoff (see "Phase B4 behaviour").

## Known limits

- The directory begins on page 52 (section 1.16 starts there), not page 53: page 52 prints the Chancellor's Office, six
  Vice Chancellors, the Call Center for New Students and Admissions. None of them is in the lexicon, so their numbers
  are not answered (`not_found`); their lines are still read for the neighbour and shared-number checks. "Vice
  Chancellor for Public Relations" contains the lexicon name "Public Relations"; the referent check keeps that ask
  from being answered with the Public Relations office's number.
- One-number lines are never answered: the College of Sciences line holds only its fax number (its telephone numbers
  are on other lines), the College of Computing & Informatics line holds only its telephone, and the College of Public
  Policy line holds one number (printed "+971 6 505 3212") with no fax printed. One extracted line cannot show which
  column a single number is in.
- Multi-line entries (Student Housing, the colleges with Men / Women numbers, the Deanship of Student Affairs, the
  Medical Service Department, the libraries, University City Security) and the four entries whose names are split over
  lines are never stitched, so their numbers are not given.
- The handbook contains no e-mail address; it prints two URLs (the main site, p. 207, outside the scope, and an IT
  Center page, p. 183, printed with an extraction defect). Website asks are not answered in phase B2.
- The Coordinator selects no specialist for "Who is the Chancellor?" and "What does the Business Services Department
  do?" (cue coverage, recorded, not changed).
- Phase B3 is a rule-based reader of English. Five adversarial reviews (about 1,350 questions replayed) found defects in
  every round; every confirmed wrong-office, wrong-location and wrong-number answer was fixed and kept as a regression
  case, and the design now fails closed on any question outside the plain forms. New phrasings may still find gaps:
  mostly fail-closed coverage gaps, possibly a wrong answer.
- Coverage gaps, all failing closed: department lists of colleges ("The departments within the College of X are as
  follows:") are not read; the OIR's list beyond item 1; British spellings ("centre", "counselling") do not meet the
  American ones; a duty asked by one word when the printed duty has a restricting complement ("Is the DRC responsible
  for handling complaints?"); purpose clauses do not restrict a duty ("conducts drills and training sessions to prepare
  ... for emergency situations" answers "Who conducts training sessions?"); "health and safety" asks are partial (no
  sentence names the duty itself); a focus clause holding two asks is read as one and fails closed (the Coordinator
  cuts questions into clauses; B4 does not split them again); a negated location ask is not answered.
- "College of Fine Arts and Design" (with "and") and other spelling variants of directory names are not matched (B2
  lexicon, unchanged without approval).
- Phase B4 (measured on the unchanged Coordinator): a Teaching subject is never handed off, so when the Coordinator
  selects only Institutional Navigation for it ("Who do I contact about academic integrity?", "Who is responsible for
  academic supervision?") nobody answers the subject in this milestone (DEP-3: a shared Teaching ownership cue, or a
  Coordinator cue, would be needed; neither is changed). "Which office provides IT support?" selects no specialist
  (DEP-2), although Institutional Navigation answers it when it is given the task. "Who should I contact about <HR or
  Research subject>" is declined by both specialists in the first round (see "Known effect of separate tasks").
  Institutional Navigation never searches its statements for another specialist's subject, even when its chapters
  happen to name an office for it ("forging strategic research partnerships" is printed for the OIR, page 47): such a
  clause goes to Research. Ownership is read with word cues: a clause whose words the shared cues claim is handed off
  even when the asked object is an institutional one ("How many research centers does the university have?" goes to
  Research; "Who is responsible for improving employee safety?" goes to HR through the shared cue "employee"; "Which
  office is responsible for forging strategic research partnerships?" goes to Research although page 47 prints it for the
  OIR); a subject the
  cues miss is answered or failed closed by phase B3, and the B3 local word lists stay broad ("environmental impact
  assessments" is refused as a Teaching subject through "assessments"). A location ask for a document ("Where can I find
  the annual leave policy?") is navigation to Institutional Navigation and HR hands it over as navigation, so neither
  answers it. Governance-level authority (C8: what a
  council or the Chancellor may decide) is not answered; approval asks are never answered with an office.

## Planned phases

B3 offices, structure and locations (Step 4.16-B3, done; accepted under D-106), B4 boundaries, handoffs and mixed
questions (Step 4.16-B4, done), B5 QA, acceptance, mutation and determinism (Step 4.16-B5, done: passed with documented
non-blocking limitations), B6 default dispatch enablement (Step 4.16-B6, done, D-105), C1 natural-phrasing coverage
correction (Step 4.16-C1, done), independent freeze review (done, 2026-10-29; frozen under D-107). Next, outside these
phases: four-agent integration (handoff rounds, Synthesis, Coordinator
cues); real-model evaluation; live integration. Design:
`SENIOR_PROJECT_HANDOVER/INSTITUTIONAL_STEP_4_16_A/`.

## Tests

`tests/test_institutional_scope.py` (scope ranges, foreign pages, no overlap with the frozen specialists, the extra page,
the section guard, the office lexicon against the handbook PDF, distinctness, identification and wrong-office cases,
ask kinds, evidence tiers), `tests/test_institutional_contract.py` (identity, resources, fail-closed results for the asks
not answered, the two directory questions answered since phase B2, clause reports, task validation, determinism, the
dispatcher override, the unchanged default dispatch, task boundaries of mixed questions) and
`tests/test_institutional_directory.py` (phase B2: the extracted directory lines against the printed directory rebuilt
from the PDF's geometry, the hazardous entries, the 113 contact-accuracy cases under both stand-in rankings,
determinism, a property test over every lexicon office and field, the cue readers, the six negative controls, the
dispatcher) and `tests/test_institutional_offices.py` (phase B3: the evidence-binding cases under both rankings and their
determinism, the eleven negative controls, every statement verbatim, in scope, never from the directory and printing
its office, the four location entries, the readers, the question forms, the Coordinator smoke),
`tests/test_institutional_integration.py` (phase B4: the 48-question boundary matrix through the unchanged Coordinator
and the dispatcher with all four specialists, task ownership, handoffs as data, no clause answered twice, no answer for
another specialist's subject, the boundary rules on the specialist alone, the handoff shape, system instructions, the
four negative controls, determinism, the default dispatch) and `tests/test_institutional_alignment.py` (phase B4: the
Coordinator's selections unchanged; every Teaching, Research and HR result identical whether Institutional Navigation
runs or stays pending), `tests/test_institutional_conflicts.py` (phase B5: evidence conflicts and determinism
independent of retrieval) and `tests/test_institutional_acceptance.py` (phase B5: the independent blind evaluation with
strict safety assertions and the measured coverage, the known limitations A to I, the redacted error path), and
`tests/test_four_specialist_dispatch.py` (Step 4.16-B6: all four specialists in the default dispatch, valid contracts,
pending only when unregistered or handoff-produced, handoffs never executed, findings from their own corpus, fail-closed
tasks, the recorded B4 matrix and the stored benchmark result reproduced through the default dispatch; since C1 the
stored C1 result, which differs from B5 only by newly correct answers) and `tests/test_institutional_natural_phrasing.py`
(Step 4.16-C1: the natural-phrasing reader, context never set aside, polite and context asks for contacts,
responsibilities and locations, office identity, procedures and logins still unanswered, the ownership check reading
the context, the added units, negative controls for the two C1 switches and for the subject-binding protection on an
added unit, determinism; its questions are not taken from the benchmark or the holdout). Shared
fixtures: `tests/institutional_support.py` (real handbook pages through the project's PDF
loader and chunker, fake embedder, the printed-directory oracle, both case evaluators with their handoff checks, the
four-specialist registry built with the frozen specialists' own test helpers). Reproducible evaluation:
`qa/institutional_4_16/` (the contact and evidence-binding cases, their builders, runners with mutation mode).
