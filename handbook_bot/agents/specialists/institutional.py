"""
Institutional Navigation Specialist (Plan F, Milestone 4, Step 4.16, D-104; phase B2: directory lookups; phase B3:
office responsibilities, structure and documented locations; phase B4: ownership checks and handoffs; phase C1: natural
phrasing). Frozen for Milestone 4 under D-107 (2026-10-29) after the independent freeze review of the Step 4.16-C1 code.

DECISION OWNED
    Whether the approved, source-scoped evidence identifies the office, unit, body or contact point a faculty question
    asks for, and what that evidence establishes about it (that the office exists, that it is responsible for the asked
    function, how to reach it, where it is), expressed as a ``SpecialistFindings``. The specialist does not route (the
    Coordinator did), does not write prose (Synthesis will), does not verify (the Verifier will) and never calls another
    specialist.

PHASE B2: TELEPHONE AND FAX NUMBERS FROM THE CONTACT DIRECTORY
    A channel clause (``navigation_cues.classify_ask``) that names a lexicon office is answered from the handbook's
    contact directory (section 1.16, a three-column table: office, telephone number, fax number). The directory's
    extracted lines are read once, in order, from the ``row`` chunks of section 1.16 inside the compiled scope and the
    section guard; nothing depends on retrieval ranking. A number is returned only when every link of the row binding
    holds:

    1. the clause names a lexicon office, and nothing in it makes the asked unit something else (a library, a clinic,
       a person, an office of another office);
    2. exactly one directory line that holds numbers starts with that office's printed name or attested alias;
    3. that line holds exactly two complete numbers and nothing else;
    4. neither neighbouring line runs into it (a trailing "/", an unfinished number, a "Men:" / "Women:" sub-row) or
       completes a number broken on it, and neither neighbour is missing from the extracted lines;
    5. the page's header line names the two columns, which fixes which number is the telephone and which the fax.

    The finding quotes that line verbatim and its metadata names the field of each value. When the office's line
    exists but the binding fails (one number, whose column a single line cannot show; a multi-line or Men / Women
    entry; a labelled sub-entry; a missing or running-on neighbour), the result is ``partial``: the finding quotes only
    the office name as the line prints it, so no unbound number is ever output, and ``missing`` says why. A labelled
    line ("+971 6 5057666 (External) Ext 7666 (Internal)") is quoted with its own labels and never assigned to the
    telephone or fax field (``partial``). An office whose printed name is split over several extracted lines, or that
    the directory does not list, is ``not_found`` for that ask. Lines are never joined, numbers never completed,
    reformatted or invented.

    E-mail and website asks are not answered (the directory prints no e-mail address and no website); no address is
    ever composed. Numbers are reported as printed in the 2025/26 handbook, not as verified to be active, and a number
    the directory prints for more than one entry carries a limitation saying so.

PHASE B3: OFFICE RESPONSIBILITIES, STRUCTURE AND DOCUMENTED LOCATIONS
    The in-scope paragraph chunks of the owned chapters (never the directory pages) are read once, in order, into
    statements (``navigation_cues.split_units`` / ``read_unit``). A statement binds an office to a duty (tier E2) only
    when the office's printed name or attested alias is the grammatical subject that opens it and a listed
    responsibility verb follows ("The ITC also maintains a cutting-edge Data Center ..."), or when it opens with an
    anaphor ("This Council", "The office", "It") whose antecedent is the sentence just before in the same passage,
    naming that one office with a matching head noun, or when it is an item of a list whose lead-in names the office
    ("The Health and Safety Office is responsible for the following: ..."), lead-in and item in one passage. A
    statement whose subject is a unit outside the lexicon ("The CLD", "The Mail Office"), or a list item whose lead-in
    is in another passage, binds no office; it only makes an answer partial. A chair ("chaired by") or a composition
    ("consists of", "operates through ... subgroups") is read the same way; the university as a whole is the subject
    of organisation and count statements. Library-style location entries ("Main Library (W16): ... on the University's
    Women Campus (W16).") give documented locations with their building codes as printed.

    Answers: which office is responsible for a duty (supported only when exactly one office is bound and no other
    unit is stated to do it), what a named office does, whether it does a named duty, who chairs or makes up a body,
    the university's organisation, the colleges and university-level counts as stated, and library locations (two
    entries printed under one name are given together, partial, unless the question picks one). Procedures and system
    access (no E3 evidence), approvals and the subjects of the other specialists are not answered (phase B4 below).
    Words of an office's own name are no evidence of a duty; a mention, a directory line or a neighbouring paragraph
    never is.

    Question forms: five adversarial reviews showed that any extra word of a question (a place, a person, a time, an
    exclusion, a second ask, a beneficiary put in front) can change what is asked. A focus clause is therefore answered
    only when the whole clause, courtesy words aside, has one of the plain forms of ``_FORMS`` with a plain duty slot
    (``_duty_ok``); any other clause is not answered. A clause holding two asks is read as one and fails closed: the
    Coordinator cuts a question into focus clauses, and splitting a clause again here re-opened the bypasses the reviews
    found. Contact clauses likewise need a plain contact form (``navigation_cues.channel_form_problem``, inside the
    phase-B2 referent check).

PHASE B4: OWNERSHIP CHECKS AND HANDOFFS
    The specialist works only on the focus clauses of its own task; it cannot see which other specialists the
    Coordinator selected and never calls one. Before a non-contact clause is answered, its subject is checked with the
    shared ownership cues (``domain_cues.hr_owned`` / ``research_owned``, unchanged) and the local HR and Teaching
    subject lists (``_boundary``):

    * a Faculty Services & HR or Research & Innovation subject that the SHARED cues name is not answered here; the clause
      is returned as a ``HandoffRequest`` to that specialist (one per owner the cues name, in canonical order), never
      executed. A clause is never handed back to the specialist whose handoff task it came in (``task.requested_by``);
    * a clause only the local HR list claims ("Who is responsible for environmental health?"), or one the shared cues
      give to HR or Research while it also has a Teaching subject ("Who handles exam appointments?"), is not answered
      and not handed off: whose it is stays unsettled, and no target is guessed;
    * a Teaching & Learning subject is not answered and not handed off: the shared cues hold no Teaching ownership
      lexicon (dependency DEP-3), so the target would be a guess; the Coordinator routes Teaching subjects itself;
    * contact and location asks are navigation and are never handed off for ownership; a form whose object is a named
      office ("What does HR do?", "Who chairs ...", "Where is ...") is checked with the office names masked, so an
      office's own name ("Human Resources", "College of Graduate Studies and Research") never makes it another
      specialist's subject;
    * an approval or authority ask is never answered with an office in Milestone 4: governance-level authority (what a
      council may approve, page 25) is future work, and an office that manages or handles a matter is not shown to
      approve it.

    A task none of whose clauses is Institutional Navigation's is ``out_of_scope`` (with its handoffs, or none for a
    Teaching subject). A handed-off clause adds no ``missing`` entry (as in the other specialists); a clause left
    unanswered without a handoff does.

PHASE C1 (THIS VERSION): NATURAL PHRASING
    A focus clause is read as the plain question it asks (``navigation_cues.natural_ask``, Step 4.16-C1): a request
    frame ("Could you please tell me ..."), a greeting, filler words in one closed position, an embedded question order
    and a count phrasing are set aside, and a context sentence is set aside only in closed shapes (before the question
    a greeting or thanks, a self-introduction or the asker's own unit; after it only thanks) and only when nothing in it
    can change the asked unit, field, place or condition. Every B2, B3 and B4 check still reads the result; a clause
    ``natural_ask`` cannot read safely is returned unchanged and fails closed as before. Four switches
    (``_C1_NATURAL_ASK``, ``_C1_CONTEXT_GUARD``, ``_C1_KIND_CHANGE_CHECK``, ``_C1_COMPOSITION_CONTINUES``), all on.

EVIDENCE TIERS (``navigation_cues``)
    E1 identification (the office is named), E2 responsibility (one statement binds the office to the asked function),
    E3 procedure (the steps are stated). A directory line is E1 evidence; it never proves a responsibility. B3 reaches
    E2 for responsibilities and "stated fact" for chairs, compositions, organisation, counts and locations; it never
    reaches E3.

SCOPE
    Sections no frozen specialist owns: the university chapter (structure, governance, colleges, the contact directory
    1.16), the IT Center lead-in (page 183), library staff, facilities and locations, the Health and Safety Office and
    emergency communications, and the campus service units. Every other specialist's section is excluded.

DEFAULT DISPATCH, FREEZE AND RUNTIME
    Since Step 4.16-B6 (D-105) ``dispatch.EXECUTABLE_SPECIALISTS`` contains Institutional Navigation, so the default
    ``dispatch()`` / ``run_question()`` executes its tasks whenever the specialist is registered. Frozen for Milestone 4
    under D-107 (2026-10-29): reopened only by a new decision record, as the other three specialists are. Not wired
    into the live chatbot: nothing in ``handbook_bot.orchestrator`` or the UI imports Plan F, and ``PLAN_F_ENABLED``
    stays False. Deterministic, no LLM call, no network.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from ...sources import SectionMap, SourceRecord, section_map_for
from ...text_utils import normalize_text
from . import domain_cues as _domain
from . import navigation_cues as _cues
from .base import Specialist
from .contracts import Finding, FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask
from .scoping import SectionGuard, compile_scope

APPROVED_SOURCE_ID = "uos_faculty_handbook_2025_26"
#: The last development phase this version implements (Step 4.16-C1, natural phrasing); frozen for Milestone 4 (D-107).
PHASE = "C1"

#: Handbook sections the specialist owns: (chapter, printed number, label). None is owned by Teaching, Research or HR.
OWNED_SECTIONS: Tuple[Tuple[int, str, str], ...] = (
    (1, "1.1", "Vision, Mission, and Core Values"),
    (1, "1.3", "Organizational Chart"),
    (1, "1.4", "University Governance"),
    (1, "1.5", "The President"),
    (1, "1.6", "The Board of Trustees"),
    (1, "1.7", "The University Administration"),
    (1, "1.8", "Shared Governance"),
    (1, "1.9", "Campuses, Colleges, Departments, Programs and Degrees"),
    (1, "1.10", "Language Institute"),
    (1, "1.11", "Accreditation"),
    (1, "1.12", "Office of International Relations"),
    (1, "1.13", "University Ranking"),
    (1, "1.16", "Contact Information"),
    (11, "11.4", "Library Staff"),
    (11, "11.5", "Facilities"),
    (11, "11.6", "Locations"),
    (13, "13.1", "Role of the Health and Safety Office"),
    (13, "13.6", "Emergency Communications"),
    (15, "15.1", "Central Laboratories"),
    (15, "15.2", "Business Service Department"),
    (15, "15.3", "Facilities Management"),
    (15, "15.5", "Disability Resource Center"),
    (15, "15.7", "Sport Facilities"),
    (15, "15.8", "Food Courts"),
)
SCOPE_SECTIONS: Tuple[Tuple[int, str, str], ...] = OWNED_SECTIONS
#: The chapter 10 lead-in (page 183: the Information Technology Center's services and user support) lies before the
#: chapter's first section record (10.1 starts on page 184); ``compile_scope`` admits it as an extra page.
EXTRA_PAGES: Tuple[Tuple[int, int], ...] = ((183, 183),)
#: Sections deliberately kept out. Listed so the decision is visible.
EXCLUDED_SECTIONS: Tuple[Tuple[int, str, str], ...] = (
    (1, "1.2", "2024-2030 University Strategic Priorities (no navigation content)"),
    (1, "1.14", "Academic Calendar (Teaching)"),
    (1, "1.15", "Statement on Diversity, Equity, and Inclusion (no office named)"),
    (10, "10.1", "Acceptable Use Policy (policy, not navigation)"),
    (10, "10.6", "Learning Management System (LMS) Policy (Teaching)"),
    (13, "13.2", "Health and Safety Policy (policy, not navigation)"),
    (15, "15.4", "Child Care Center (Faculty Services & HR)"),
    (15, "15.6", "Medical Clinics (Faculty Services & HR)"),
)
#: The section whose table is the contact directory.
DIRECTORY_SECTION: Tuple[int, str, str] = (1, "1.16", "Contact Information")

# ---------------------------------------------------------------------------
# Row-binding protections (phase B2). Each is True in production; the negative-control tests switch one off at a
# time and show that a wrong office or a wrong number then comes out. Never switch one off outside those tests.
# ---------------------------------------------------------------------------
#: A directory line belongs to an office only when it STARTS with the office's printed name or attested alias,
#: followed by a value, a label, a colon or the end of the line (off: the name anywhere in the line).
_B2_ROW_START = True
#: A number is bound only from a line holding exactly two complete numbers and nothing else (off: any line holding a
#: number, its numbers taken in column order).
_B2_ROW_COMPLETE = True
#: ... and only when neither neighbouring line runs into it or completes a number broken on it, and neither is
#: missing from the extracted lines (off: neighbours ignored).
_B2_NEIGHBOR_CHECK = True
#: ... and only on a page whose header line names the telephone and fax columns; the header fixes their order
#: (off: telephone first, fax second assumed without reading the header).
_B2_COLUMN_HEADER = True
#: ... and only when no other directory line starting with the same office holds a number (off: the first one used).
_B2_SINGLE_ENTRY = True
#: A clause whose office mention may not be the unit asked about (a library, clinic, person or sub-office of it, or
#: one office named within another) is not answered from the office's line (off: answered).
_B2_REFERENT_CHECK = True


def binding_protections() -> Dict[str, bool]:
    """The current state of the six row-binding protections (recorded in every result)."""
    return {"row_start": _B2_ROW_START, "row_complete": _B2_ROW_COMPLETE, "neighbor_check": _B2_NEIGHBOR_CHECK,
            "column_header": _B2_COLUMN_HEADER, "single_entry": _B2_SINGLE_ENTRY, "referent_check": _B2_REFERENT_CHECK}


# ---------------------------------------------------------------------------
# Phase B3 evidence-binding protections. Each is True in production; the negative-control tests switch one off at a
# time and show that a wrong office, an invented responsibility, a procedure or a wrong location then comes out.
# ---------------------------------------------------------------------------
#: The office must be the grammatical subject that opens the statement (off: the first office named anywhere in it).
_B3_SUBJECT_BINDING = True
#: A listed responsibility, chair or composition verb must follow the subject (off: naming the office is enough).
_B3_RESPONSIBILITY_VERB = True
#: "This Council", "The office", "It" refer only to the office of the sentence just before, with a matching head noun
#: (off: to the last office named anywhere earlier in the passage).
_B3_ANAPHORA_STRICT = True
#: List items belong to their lead-in only up to the end of the list (off: the rest of the passage belongs to it).
_B3_LIST_EXTENT = True
#: A statement answers an asked duty only when it states every content word of it (off: any one word).
_B3_DUTY_ALL_TERMS = True
#: Several offices for one duty, or a unit outside the lexicon stated to do it, make the answer partial (off: the first).
_B3_AMBIGUITY_CHECK = True
#: Clauses the shared HR / Research ownership cues or the local HR / Teaching subject lists claim are not answered
#: (off: answered, and no ownership check is made, so nothing is handed off either).
_B3_OWNERSHIP_GUARD = True
#: Procedure and system-access asks are never answered from responsibility statements (off: they are).
_B3_PROCEDURE_GUARD = True
#: A location comes only from an entry that opens with the unit's printed name (off: any sentence naming it and a place).
_B3_LOCATION_ENTRY = True
#: Entries printed under one name (the two "Main Library" entries) are given together, not chosen (off: the first).
_B3_SAME_NAME_CHECK = True
#: Words of an office's own name are no evidence that it performs a duty (off: they count).
_B3_OFFICE_WORDS_EXCLUDED = True


# ---------------------------------------------------------------------------
# Phase B4 boundary protections. Each is True in production; the negative-control tests switch one off at a time and
# show that a clause is then handed to no one, handed back to the specialist that handed it over, or handed off when it
# is Institutional Navigation's own navigation ask.
# ---------------------------------------------------------------------------
#: A clause whose subject the shared HR / Research cues (or the local HR subject list) claim is handed to that
#: specialist as a ``HandoffRequest`` (off: it is only left unanswered).
_B4_OWNER_HANDOFF = True
#: A clause is never handed back to the specialist whose handoff task it came in (off: it is).
_B4_NO_HAND_BACK = True
#: Contact and location asks are never handed off for ownership, and a form whose object is a named office is checked
#: with the office names masked (off: every non-contact clause is checked whole, office names included).
_B4_NAVIGATION_KEPT = True
#: A handoff goes only to an owner the SHARED cues name, and never when the clause also has a Teaching subject; the local
#: HR and Teaching word lists only stop an answer (off: the local HR list names the HR owner, and a Teaching subject
#: does not stop a handoff).
_B4_SHARED_OWNER_ONLY = True


# ---------------------------------------------------------------------------
# Step 4.16-C1: natural phrasing. A focus clause is read as the plain question it asks (``navigation_cues.natural_ask``):
# a request frame, a greeting, filler words and a context sentence that cannot change the ask are set aside; every check
# of the plain forms still reads the result.
# ---------------------------------------------------------------------------
#: Natural phrasing is read (off: the clause is read exactly as written, as in B1 to B6).
_C1_NATURAL_ASK = True
#: A context sentence is set aside only when it names no place, campus, gender, building, condition, exclusion or other
#: institution, negates no unit or field, names an office only when the question names one, and the question does not
#: refer back to it (off: any context sentence is set aside).
_C1_CONTEXT_GUARD = True
#: A plain ask of another kind than the clause is used only when the B4 ownership check, run on the whole clause for the
#: new kind, leaves it to Institutional Navigation (off: a contact clause, which skips the ownership check, can become an
#: unchecked responsibility ask for another specialist's subject).
_C1_KIND_CHANGE_CHECK = True
#: A composition statement followed in its passage by "Additionally, ..." or "... also participate ..." is not the whole
#: composition: the answer says so (off: the quoted sentences are given as the whole composition).
_C1_COMPOSITION_CONTINUES = True


def c1_protections() -> Dict[str, bool]:
    """The current state of the Step 4.16-C1 natural-phrasing switches (recorded in every result)."""
    return {"natural_ask": _C1_NATURAL_ASK, "context_guard": _C1_CONTEXT_GUARD, "kind_change_check": _C1_KIND_CHANGE_CHECK,
            "composition_continues": _C1_COMPOSITION_CONTINUES}


def b4_protections() -> Dict[str, bool]:
    """The current state of the four phase-B4 boundary protections (recorded in every result)."""
    return {"owner_handoff": _B4_OWNER_HANDOFF, "no_hand_back": _B4_NO_HAND_BACK, "navigation_kept": _B4_NAVIGATION_KEPT,
            "shared_owner_only": _B4_SHARED_OWNER_ONLY}


def b3_protections() -> Dict[str, bool]:
    """The current state of the eleven phase-B3 evidence-binding protections (recorded in every result)."""
    return {"subject_binding": _B3_SUBJECT_BINDING, "responsibility_verb": _B3_RESPONSIBILITY_VERB, "anaphora_strict": _B3_ANAPHORA_STRICT,
            "list_extent": _B3_LIST_EXTENT, "duty_all_terms": _B3_DUTY_ALL_TERMS, "ambiguity_check": _B3_AMBIGUITY_CHECK,
            "ownership_guard": _B3_OWNERSHIP_GUARD, "procedure_guard": _B3_PROCEDURE_GUARD, "location_entry": _B3_LOCATION_ENTRY,
            "same_name_check": _B3_SAME_NAME_CHECK, "office_words_excluded": _B3_OFFICE_WORDS_EXCLUDED}


_PHASE_LIMITATION = ("Institutional Navigation (phase C1, frozen for Milestone 4 under D-107): it answers telephone and fax asks from the contact "
                     "directory, office responsibilities, structure and documented locations stated in the handbook's owned chapters; "
                     "procedures, approvals, e-mail addresses, websites, opening hours and the subjects of the other specialists are not "
                     "answered (Faculty Services & HR and Research & Innovation subjects are handed off as data, never executed).")
_HANDED_OFF_LIMITATION = ("%d clause(s) belong to another specialist and were handed off, not answered; Institutional Navigation "
                          "names no office for them (an office is named only from a statement of its own chapters).")
_TEACHING_LIMITATION = ("%d clause(s) have a Teaching & Learning subject; they are not answered here and no handoff is created, because "
                        "the shared ownership cues hold no Teaching lexicon to name the target from (the Coordinator routes Teaching "
                        "subjects itself).")
_UNSETTLED_LIMITATION = ("%d clause(s) may belong to another specialist, but the shared ownership cues do not settle whose they are; they "
                         "are not answered and no handoff is guessed.")
_HAND_BACK_LIMITATION = ("%d clause(s) came in a handoff task from the specialist that owns their subject; they are not handed back "
                         "and not answered.")
_CHART_LIMITATION = ("The organisational chart (Figure 1, page 19) is a figure; reporting lines shown only in the chart are not "
                     "stated in the text and are not given.")
_MAX_OFFICE_STATEMENTS = 8
_AS_PRINTED_LIMITATION = ("Numbers are quoted as printed in the Faculty Handbook 2025/26 contact directory; the handbook does not state "
                          "that they are current, and they are not verified as active.")

_FIELD_LABEL = {_cues.TELEPHONE: "telephone number", _cues.FAX: "fax number", _cues.EXTENSION: "extension"}
_DIRECTORY_FIELDS = (_cues.TELEPHONE, _cues.FAX, _cues.EXTENSION)

# Lookup outcomes
BOUND = "bound"            # the row binding holds: values with their fields
LABELLED = "labelled"      # one line with values carrying their own printed labels; no telephone / fax field assigned
UNBOUND = "unbound"        # the office's line exists, the binding fails
SPLIT = "split"            # a directory office whose printed name is on no single extracted line
ABSENT = "absent"          # an office the directory does not list

_UNBOUND_REASON = {
    _cues.ROW_SINGLE: "its directory line holds one number, and one extracted line cannot show whether a number is in the telephone or "
                      "the fax column",
    _cues.ROW_FRAGMENT: "its directory entry runs over several extracted lines (a multi-number or Men / Women entry), and lines are never "
                        "joined",
    _cues.ROW_NAME_ONLY: "its directory line holds no number; the entry's numbers are on other extracted lines, which are never joined",
    _cues.ROW_SUB_ENTRY: "it is printed as a labelled sub-entry inside another directory entry, and its line cannot show the column of its "
                         "number",
    "neighbour": "a neighbouring directory line runs into its line, so the line may not hold the whole entry",
    "gap": "a neighbouring directory line is missing from the extracted lines, so the line cannot be checked against it",
    "header": "the column header of its directory page could not be read, so the order of the telephone and fax columns is unknown",
    "multiple": "the directory prints it on more than one line holding numbers, so no single line can be chosen",
}


@dataclass(frozen=True)
class InstitutionalResources:
    """What the specialist needs to retrieve evidence. Built from a ``KnowledgeBase`` in production and from fakes in tests."""

    embed: Callable[[str], np.ndarray]                             # text -> (1, dim) float32
    rerank: Callable[[Sequence[Sequence[str]]], Sequence[float]]   # [[query, text], ...] -> scores
    index: Any
    chunks: List[str]
    metadata: List[Dict[str, Any]]
    source: SourceRecord
    section_map: SectionMap

    @classmethod
    def from_knowledge_base(cls, kb) -> "InstitutionalResources":
        source = getattr(kb, "source", None)
        if source is None or not source.registered or source.source_id != APPROVED_SOURCE_ID:
            raise ValueError("the Institutional Navigation specialist requires the registered handbook %r" % APPROVED_SOURCE_ID)
        applied = ((getattr(kb, "stats", None) or {}).get("source") or {}).get("section_records", 0)
        if not applied:
            raise ValueError("the section map was not applied to the loaded document; Institutional scope cannot be compiled")
        embedder, reranker = kb.embedder, kb.reranker

        def embed(text: str) -> np.ndarray:
            return embedder.encode([normalize_text(text)], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        return cls(embed=embed, rerank=reranker.predict, index=kb.index, chunks=list(kb.chunks), metadata=list(kb.metadata), source=source,
                   section_map=section_map_for(source))


@dataclass(frozen=True)
class DirectoryRow:
    """One extracted line of the contact directory, read once at construction."""

    index: int                      # position in the resources' chunk list
    chunk_id: Any
    page: int
    row_id: int
    text: str                       # the chunk text, verbatim
    office_id: Optional[str]        # the office the line starts with (None: none)
    name_end: int                   # end of the office name in ``text`` (0 when no office)
    kind: Optional[str]             # navigation_cues.ROW_* (None when no office)
    values: Tuple[str, ...] = ()    # the line's values after the name, verbatim
    labels: Tuple[str, ...] = ()    # printed labels of a labelled line


@dataclass(frozen=True)
class DirectoryLookup:
    """What the directory establishes for one office."""

    office_id: str
    outcome: str                                         # BOUND | LABELLED | UNBOUND | SPLIT | ABSENT
    reason: Optional[str] = None                         # for UNBOUND: a key of _UNBOUND_REASON
    row: Optional[DirectoryRow] = None
    columns: Tuple[str, ...] = ()                        # for BOUND: the page header's column order
    values: Dict[str, str] = field(default_factory=dict)  # for BOUND: field -> value, verbatim


BIND_SUBJECT = "named subject"
BIND_ANAPHOR = "anaphor of the sentence before"
BIND_LIST = "list item under its lead-in"
BIND_OUTSIDE = "unit outside the office lexicon"
BIND_ORPHAN = "statement without its unit in the passage"


@dataclass(frozen=True)
class Statement:
    """One statement of the owned chapters, read once at construction. ``office_id`` is a lexicon office,
    ``navigation_cues.UNIVERSITY`` or None (a unit outside the lexicon, or no unit in the passage)."""

    office_id: Optional[str]
    relation: Optional[str]          # navigation_cues.REL_* or None (a university sentence without a recognised relation)
    binding: str                     # BIND_*
    subject_text: str                # the subject as printed
    index: int                       # chunk index
    page: int
    quote_start: int                 # the quote: chunk text [quote_start:quote_end], verbatim
    quote_end: int
    key_start: int                   # the unit that states the duty (the item for a list item)
    key_end: int
    list_continues: bool = False     # the list runs to the end of the passage and may continue in the next one
    item: bool = False               # the stating unit is a list item
    phrases: Tuple[str, ...] = ()    # the duty phrases it names (navigation_cues.duty_phrases): titles, objects of its verb
    #: for an unbound statement whose anaphor resolves to a lexicon office but whose verb binds nothing ("The Center
    #: integrates ..." after "DRC are dedicated to ..."): that office. It is never answered from, and it is no other unit.
    subject_office: Optional[str] = None


@dataclass(frozen=True)
class Placement:
    """A documented location entry (section 11.6 style), read once at construction."""

    office_id: str
    label: str                       # the entry title as printed, e.g. "Main Library (W16)"
    code: Optional[str]              # the building code printed with it
    place: Optional[str]             # the place as printed
    index: int
    page: int
    start: int
    end: int


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("’", "'")).strip().lower()


def _also_printed(others: Sequence[Optional[str]]) -> str:
    """"for X and Y and on 2 other directory lines" from the result of ``_same_number_elsewhere``."""
    names = list(dict.fromkeys(n for n in others if n))
    unnamed = sum(1 for n in others if not n)
    parts = (["for " + " and ".join(names)] if names else []) + (["on %s other directory line%s" % ("one" if unnamed == 1 else "%d" % unnamed,
                                                                                                   "" if unnamed == 1 else "s")] if unnamed else [])
    return " and ".join(parts)


class InstitutionalNavigationSpecialist(Specialist):
    """Institutional Navigation Specialist, deterministic version 1, phase B2."""

    specialist_id = SpecialistId.INSTITUTIONAL

    def __init__(self, resources: InstitutionalResources, *, owned_sections: Sequence[Tuple[int, str, str]] = SCOPE_SECTIONS,
                 extra_pages: Sequence[Tuple[int, int]] = EXTRA_PAGES) -> None:
        if resources.source.source_id != APPROVED_SOURCE_ID or not resources.source.registered:
            raise ValueError("the Institutional Navigation specialist may only use the registered handbook %r" % APPROVED_SOURCE_ID)
        self.resources = resources
        self.owned_sections = tuple(owned_sections)
        self.scope = compile_scope(resources.section_map, APPROVED_SOURCE_ID, self.owned_sections, extra_pages)
        self.guard = SectionGuard(resources.chunks, resources.metadata, resources.section_map, self.owned_sections, self.scope)
        self.directory_pages = self._directory_pages()
        self.directory = self._read_directory()
        self.header_columns = self._header_columns()
        self.statements, self.placements = self._read_statements()
        self.outside_statements = self._read_outside_statements()

    @classmethod
    def from_knowledge_base(cls, kb) -> "InstitutionalNavigationSpecialist":
        return cls(InstitutionalResources.from_knowledge_base(kb))

    # ---- the contact directory ----------------------------------------------
    def _directory_pages(self) -> FrozenSet[int]:
        if DIRECTORY_SECTION not in self.owned_sections:
            return frozenset()
        ranges = compile_scope(self.resources.section_map, APPROVED_SOURCE_ID, (DIRECTORY_SECTION,), ()).page_ranges or ()
        return frozenset(p for first, last in ranges for p in range(first, last + 1))

    def _read_directory(self) -> Dict[int, Tuple[DirectoryRow, ...]]:
        """Every in-scope ``row`` chunk of the directory pages, by page, in row order. A line on a shared page is kept only
        when the section guard attributes the whole line to an owned section."""
        by_page: Dict[int, List[DirectoryRow]] = {}
        for i, meta in enumerate(self.resources.metadata):
            page, rid = meta.get("page"), meta.get("row_id")
            if meta.get("chunk_type") != "row" or page not in self.directory_pages:
                continue
            if meta.get("source_id") != APPROVED_SOURCE_ID or not self.scope.allows(meta):
                continue
            if not isinstance(rid, int) or isinstance(rid, bool):
                continue
            text = self.resources.chunks[i]
            spans = self.guard.owned_spans(meta.get("chunk_id"), text, meta)
            if spans is not None and spans != [(0, len(text))]:
                continue
            hit = _cues.row_office(text, anchored=_B2_ROW_START)
            if hit is None:
                row = DirectoryRow(i, meta.get("chunk_id"), page, rid, text, None, 0, None)
            else:
                office_id, _start, end = hit
                kind, values, labels = _cues.read_row_values(text[end:])
                row = DirectoryRow(i, meta.get("chunk_id"), page, rid, text, office_id, end, kind, values, labels)
            by_page.setdefault(page, []).append(row)
        return {page: tuple(sorted(rows, key=lambda r: r.row_id)) for page, rows in sorted(by_page.items())}

    def _header_columns(self) -> Dict[int, Optional[Tuple[str, str]]]:
        out: Dict[int, Optional[Tuple[str, str]]] = {}
        for page, rows in self.directory.items():
            found = [cols for cols in (_cues.header_columns(r.text) for r in rows if r.office_id is None) if cols]
            out[page] = found[0] if len(found) == 1 else None
        return out

    def directory_rows(self) -> Tuple[DirectoryRow, ...]:
        return tuple(r for page in sorted(self.directory) for r in self.directory[page])

    def _neighbour_problem(self, row: DirectoryRow) -> Optional[str]:
        on_page = {r.row_id: r for r in self.directory.get(row.page, ())}
        before, after = on_page.get(row.row_id - 1), on_page.get(row.row_id + 1)
        if before is None or after is None:
            return "gap"
        if _cues.header_columns(before.text) is None and _cues.continues_into_next(before.text):
            return "neighbour"
        if _cues.continues_from_previous(after.text):
            return "neighbour"
        return None

    def lookup(self, office_id: str) -> DirectoryLookup:
        """What the directory establishes for one lexicon office, under the row binding."""
        candidates = [r for r in self.directory_rows() if r.office_id == office_id]
        if not candidates:
            office = _cues.OFFICES_BY_ID[office_id]
            return DirectoryLookup(office_id, SPLIT if set(office.pages) & set(self.directory_pages) else ABSENT)
        numbered = [r for r in candidates if r.values or re.search(r"\d", r.text[r.name_end:])]
        if _B2_SINGLE_ENTRY and len(numbered) > 1:
            return DirectoryLookup(office_id, UNBOUND, "multiple", numbered[0])
        row = numbered[0] if numbered else candidates[0]
        kind = row.kind
        if not _B2_ROW_COMPLETE and kind != _cues.ROW_COMPLETE and row.values:
            kind = _cues.ROW_COMPLETE                     # protection off: whatever numbers the line holds, in column order
        if kind not in (_cues.ROW_COMPLETE, _cues.ROW_LABELLED):
            return DirectoryLookup(office_id, UNBOUND, kind, row)
        if _B2_NEIGHBOR_CHECK:
            problem = self._neighbour_problem(row)
            if problem is not None:
                return DirectoryLookup(office_id, UNBOUND, problem, row)
        if kind == _cues.ROW_LABELLED:
            return DirectoryLookup(office_id, LABELLED, None, row)
        columns = self.header_columns.get(row.page) if _B2_COLUMN_HEADER else (_cues.TELEPHONE, _cues.FAX)
        if columns is None:
            return DirectoryLookup(office_id, UNBOUND, "header", row)
        return DirectoryLookup(office_id, BOUND, None, row, tuple(columns), dict(zip(columns, row.values)))

    def _same_number_elsewhere(self, row: DirectoryRow, value: str) -> List[Optional[str]]:
        """The other directory lines printing the same number: the lexicon name of the office each starts with, or None
        for a line that starts with no lexicon office."""
        return [(_cues.OFFICES_BY_ID[other.office_id].name if other.office_id else None) for other in self.directory_rows()
                if other.index != row.index and value in other.text]

    # ---- input checks -------------------------------------------------------
    def _check_task(self, task: SpecialistTask) -> None:
        if not isinstance(task, SpecialistTask):
            raise TypeError("run() expects a SpecialistTask, got %s" % type(task).__name__)
        if task.specialist_id is not None and task.specialist_id != self.specialist_id:
            raise ValueError("task %s is assigned to %s, not to %s" % (task.task_id, task.specialist_id.value, self.specialist_id.value))

    @staticmethod
    def _source_scope_problem(task: SpecialistTask) -> Optional[str]:
        unapproved = [s for s in task.source_scope if s != APPROVED_SOURCE_ID]
        if unapproved:
            return ("task source_scope names sources the Institutional Navigation specialist is not approved to use: %s"
                    % ", ".join(unapproved))
        return None

    @staticmethod
    def _focus_clauses(task: SpecialistTask) -> List[str]:
        raw = task.context.get("focus_clauses") if isinstance(task.context, dict) else None
        clauses = [c.strip() for c in raw if isinstance(c, str) and c.strip()] if isinstance(raw, (list, tuple)) else []
        return clauses or [task.question.strip()]

    # ---- findings -----------------------------------------------------------
    def _finding(self, task: SpecialistTask, number: int, lk: DirectoryLookup, clause_index: int, *, quote: str, entry: str,
                 fields: Sequence[str] = (), values: Optional[Dict[str, str]] = None) -> Finding:
        row = lk.row
        meta = self.resources.metadata[row.index]
        office = _cues.OFFICES_BY_ID[lk.office_id]
        header = next((r for r in self.directory.get(row.page, ()) if r.office_id is None and _cues.header_columns(r.text)), None)
        metadata: Dict[str, Any] = {
            "chunk_id": meta.get("chunk_id"), "chunk_type": meta.get("chunk_type"), "row_id": row.row_id, "chapter": meta.get("chapter"),
            "section_no": meta.get("section_no"), "section_title": meta.get("section_title"),
            "page_section_nos": list(meta.get("page_section_nos") or []),
            "boundary_guard": "attributed to an owned section" if self.guard.is_shared(meta) else "page carries owned sections only",
            "clauses": [clause_index],
            "office_id": office.office_id, "office_name": office.name,
            "directory_entry": entry,                      # complete_row | labelled_row | unbound_row
            "row_complete": entry == "complete_row",
            "contact_fields": list(fields),
            "contact_values": dict(values or {}),
            "row_values": dict(lk.values) if lk.outcome == BOUND else {},
            "labelled_values": [{"value": v, "label": l} for v, l in zip(row.values, row.labels)] if lk.outcome == LABELLED else [],
            "column_order": list(lk.columns) if lk.outcome == BOUND else [],
            "header_chunk_id": header.chunk_id if header is not None else None,
            "unbound_reason": lk.reason if lk.outcome == UNBOUND else None,
            "quote_scope": "office name only" if entry == "unbound_row" else "whole directory line",
            "evidence_tier": _cues.E1,
            "as_printed": True, "verified_active": False,
            "extractive": True,
        }
        page = meta.get("page")
        return Finding(finding_id="%s-f%d" % (task.task_id, number), claim=quote, evidence_quote=quote, source_id=meta["source_id"],
                       source_title=str(meta.get("source_title") or self.resources.source.title),
                       page=page if isinstance(page, int) and not isinstance(page, bool) and page >= 1 else None,
                       confidence=1.0 if lk.outcome == BOUND else 0.5, metadata=metadata)

    # ---- one channel clause -------------------------------------------------
    def _answer_channel(self, task: SpecialistTask, report: Dict[str, Any], findings: List[Finding], missing: List[str],
                        summary: List[str], shared: List[str]) -> None:
        i, clause = report["index"], report["clause"]
        fields = _cues.contact_fields(clause)
        report["contact_fields"] = list(fields)
        report["lookups"] = []
        if not report["offices"]:
            missing.append("clause %d (channel ask): not answered; the clause names no office of the Institutional Navigation lexicon, so "
                           "no directory entry can be selected: %s" % (i, clause))
            return
        if _B2_REFERENT_CHECK:
            why = _cues.referent_problem(clause)
            if why is not None:
                report["referent_problem"] = why
                missing.append("clause %d (channel ask): not answered; %s: %s" % (i, why, clause))
                return
        before_missing, before_findings = len(missing), len(findings)
        for office_id in report["offices"]:
            name = _cues.OFFICES_BY_ID[office_id].name
            lk = self.lookup(office_id)
            page = lk.row.page if lk.row is not None else None
            where = "contact directory, page %d" % page if page is not None else "contact directory"
            bound: Dict[str, str] = {}
            quote_kind: Optional[str] = None
            for f in fields:
                if f == _cues.EMAIL:
                    missing.append("clause %d (channel ask): no e-mail address is printed for %s in the handbook contact directory, and "
                                   "Institutional Navigation never composes one" % (i, name))
                    continue
                if f == _cues.WEBSITE:
                    missing.append("clause %d (channel ask): website asks are not answered in phase %s; the contact directory prints no "
                                   "website for %s, and none is composed" % (i, PHASE, name))
                    continue
                label = _FIELD_LABEL[f]
                if lk.outcome == BOUND:
                    if f in lk.values:
                        bound[f] = lk.values[f]
                    else:
                        missing.append("clause %d (channel ask): the directory line of %s prints no %s (%s)" % (i, name, label, where))
                elif lk.outcome == LABELLED:
                    if f == _cues.FAX:
                        missing.append("clause %d (channel ask): no fax number is printed for %s; its directory line holds an external "
                                       "number and an internal extension (%s)" % (i, name, where))
                    else:
                        quote_kind = "labelled_row"
                        missing.append("clause %d (channel ask): the directory line of %s labels its values External and Internal "
                                       "instead of using the telephone and fax columns; they are quoted with their printed labels and "
                                       "not given as its %s (%s)" % (i, name, label, where))
                elif lk.outcome == UNBOUND:
                    quote_kind = quote_kind or "unbound_row"
                    missing.append("clause %d (channel ask): the %s of %s is not given: %s (%s)" % (i, label, name, _UNBOUND_REASON[lk.reason], where))
                elif lk.outcome == SPLIT:
                    missing.append("clause %d (channel ask): the %s of %s is not given: its directory entry is split over several "
                                   "extracted lines (its printed name is on no single line), and lines are never joined" % (i, label, name))
                else:
                    missing.append("clause %d (channel ask): %s is not listed in the handbook contact directory (section 1.16); no %s "
                                   "is stated" % (i, name, label))
            finding_id = None
            if bound:
                findings.append(self._finding(task, len(findings) + 1, lk, i, quote=lk.row.text, entry="complete_row",
                                              fields=list(bound), values=bound))
                finding_id = findings[-1].finding_id
                summary.append("%s: %s (%s)" % (name, ", ".join("%s %s" % (f, v) for f, v in bound.items()), where))
                for f, value in bound.items():
                    others = self._same_number_elsewhere(lk.row, value)
                    if others:
                        note = "The %s printed for %s is also printed in the contact directory %s." % (_FIELD_LABEL[f], name, _also_printed(others))
                        if note not in shared:
                            shared.append(note)
            elif quote_kind == "labelled_row":
                findings.append(self._finding(task, len(findings) + 1, lk, i, quote=lk.row.text, entry="labelled_row"))
                finding_id = findings[-1].finding_id
                summary.append("%s: %s, as labelled in the contact directory (page %d); no telephone or fax field is assigned"
                               % (name, ", ".join("%s (%s)" % (v, l) for v, l in zip(lk.row.values, lk.row.labels)), page))
            elif quote_kind == "unbound_row":
                quote = lk.row.text[:lk.row.name_end].strip()
                findings.append(self._finding(task, len(findings) + 1, lk, i, quote=quote, entry="unbound_row"))
                finding_id = findings[-1].finding_id
                summary.append("%s is listed in the contact directory (page %d), but no number is bound to it" % (name, page))
            report["lookups"].append({"office_id": office_id, "outcome": lk.outcome, "reason": lk.reason,
                                      "chunk_id": lk.row.chunk_id if lk.row is not None else None, "page": page, "finding_id": finding_id})
        if len(findings) > before_findings:
            report["evidence_tier_reached"] = _cues.E1
        report["answered"] = len(missing) == before_missing and len(findings) > before_findings


    # ---- phase B3: reading the owned chapters --------------------------------
    def _page_starts_clean(self) -> Dict[int, bool]:
        """For each page: whether the page before it ends a sentence, so the page's first window starts with a new one.
        A page whose previous page is not in the corpus counts as not clean (its first unit may be a sentence tail)."""
        last: Dict[int, str] = {}
        for i, meta in enumerate(self.resources.metadata):
            if meta.get("chunk_type") == "paragraph" and isinstance(meta.get("page"), int):
                last[meta["page"]] = self.resources.chunks[i]
        return {page: (page - 1) in last and bool(re.search(r"[.!?:][\'’\")]?$", last[page - 1].rstrip())) for page in last}

    def _antecedent(self, units, valid, own, listed, k: int, head: str) -> Optional[Tuple[Optional[str], int]]:
        """What an anaphor at unit ``k`` refers to: (lexicon office, quote start), (None, start) for a unit outside the
        lexicon, or None when it cannot be resolved."""
        if not _B3_ANAPHORA_STRICT:
            for j in range(k - 1, -1, -1):
                if valid[j]:
                    spans = _cues.office_spans(units[j].text)
                    if spans:
                        return spans[-1][2], units[j].start
            return None
        j = k - 1
        if j < 0 or not valid[j]:
            return None
        if own[j] is not None:                     # the sentence before names its own subject
            office_id, start = own[j]
            if office_id is None:
                return None, start
        elif listed[j] is not None:                # an item of a list whose lead-in names the office
            office_id, start, _rel = listed[j]
        else:
            if head == "it":
                return None
            named = set(_cues.resolve_offices(units[j].text))
            if len(named) != 1:
                return None
            office_id, start = named.pop(), units[j].start
        if not _cues.head_matches(head, office_id, units[j].text):
            return None
        return office_id, start

    @staticmethod
    def _marker(text: str) -> Optional[str]:
        m = re.match(r"^(•|§|o(?=\s)|\d{1,2}\.(?=\s)|[a-h]\.(?=\s))", text or "")
        if not m:
            return None
        t = m.group(1)
        return "number" if t[0].isdigit() else "letter" if t.endswith(".") else t

    def _list_region(self, units, valid, k: int) -> Tuple[List[Tuple[int, int]], bool]:
        """The items of the list a lead-in at unit ``k`` opens, one (first unit, last unit) per item, and whether the list
        may continue in the next passage. Items keep the marker of the first item; a deeper marker is part of the current
        item; the lead-in's own marker (a nested lead-in) ends the list. An item runs over continuation sentences only
        up to the next item."""
        n = len(units)
        j = k + 1
        if not _B3_LIST_EXTENT:
            return ([(j, n - 1)] if j < n else []), True
        outer = self._marker(units[k].text) if units[k].item else None
        level = None
        out: List[Tuple[int, int]] = []
        while j < n and valid[j] and units[j].item:
            mk = self._marker(units[j].text)
            if level is None:
                level = mk
            if mk != level:
                if outer is not None and mk == outer:
                    break
                if out:
                    out[-1] = (out[-1][0], j)
                    j += 1
                    continue
                break
            a = b = j
            j += 1
            while j < n and valid[j] and not (units[j].lead_in and not units[j].item):
                if units[j].item:
                    mk2 = self._marker(units[j].text)
                    if mk2 == level or (outer is not None and mk2 == outer):
                        break
                    b = j
                    j += 1
                    continue
                nxt = next((x for x in range(j, n) if units[x].item or units[x].lead_in), None)
                if nxt is not None and units[nxt].item and self._marker(units[nxt].text) == level and all(valid[y] for y in range(j, nxt + 1)):
                    b = j
                    j += 1
                else:
                    break
            out.append((a, b))
        if not out:
            return out, False
        tail = out[-1][1] + 1
        continues = (tail >= n - 1 or not all(valid[tail:]) or
                     any(re.fullmatch(r"(?:\d{1,2}|[a-h])\.", u.text.strip()) for u in units[tail:]))
        return out, bool(continues)

    def _read_statements(self) -> Tuple[List[Statement], List[Placement]]:
        statements: List[Statement] = []
        placements: List[Placement] = []
        self.enumerations: List[Statement] = []
        seen, seen_place = set(), set()
        chunks, metas = self.resources.chunks, self.resources.metadata
        clean = self._page_starts_clean()
        for i, meta in enumerate(metas):
            page = meta.get("page")
            if meta.get("chunk_type") != "paragraph" or meta.get("source_id") != APPROVED_SOURCE_ID or not self.scope.allows(meta):
                continue
            if page in self.directory_pages or not isinstance(page, int):
                continue                      # the contact directory is never responsibility evidence
            text = chunks[i]
            spans = self.guard.owned_spans(meta.get("chunk_id"), text, meta)
            if spans == []:
                continue
            units = _cues.split_units(text)
            n = len(units)
            first_window = meta.get("chunk_id_on_page") in (0, None)
            valid = []
            for k, u in enumerate(units):
                ok = spans is None or any(a <= u.start and u.end <= b for a, b in spans)
                if k == 0 and not (first_window and (clean.get(page, False) or re.match(r"^(?:\d+(?:\.\d+)+\s|Chapter\s+\d+:)", u.text))):
                    ok = False                # a window, or a page, that starts inside a sentence
                if k == n - 1 and not re.search(r"[.!?:][\'’\")]?$", u.text.rstrip()):
                    ok = False                # a unit cut at the window end
                valid.append(ok)
            readings = [_cues.read_unit(u.text, u.item, subject_anywhere=not _B3_SUBJECT_BINDING) if valid[k] else None
                        for k, u in enumerate(units)]
            own: List[Optional[Tuple[Optional[str], int]]] = [None] * n       # the subject a unit names itself
            listed: List[Optional[Tuple[str, int, str]]] = [None] * n         # (office, quote start, relation) of a bound list
            in_list = [False] * n

            def add(st: Statement) -> None:
                key = (st.office_id, st.relation, page, _norm_key(text[st.key_start:st.key_end]))
                if key not in seen:
                    seen.add(key)
                    statements.append(st)

            for k, u in enumerate(units):
                if not valid[k]:
                    continue
                r = readings[k]
                office, rel, binding, qstart = None, r.relation, None, u.start
                if r.subject_kind == _cues.SUBJECT_OFFICE:
                    if rel is None and not _B3_RESPONSIBILITY_VERB:
                        rel = _cues.REL_RESPONSIBILITY
                    own[k] = (r.office_id, u.start)
                    if rel is not None:
                        office, binding = r.office_id, BIND_SUBJECT
                elif r.subject_kind == _cues.SUBJECT_OTHER:
                    own[k] = (None, u.start)
                elif r.subject_kind == _cues.SUBJECT_ANAPHOR:
                    ante = self._antecedent(units, valid, own, listed, k, r.anaphor_head)
                    if ante is not None:
                        own[k] = ante if ante[0] is not None else (None, u.start)
                        if ante[0] is not None and rel is not None:
                            office, qstart = ante
                            binding = BIND_ANAPHOR
                elif r.subject_kind == _cues.SUBJECT_UNIVERSITY:
                    office, binding = _cues.UNIVERSITY, BIND_SUBJECT
                if office is None and u.lead_in and r.subject_kind is None:
                    framed, _name = _cues.lead_in_office(u.text)
                    if framed is not None:
                        office, rel, binding = framed, _cues.REL_RESPONSIBILITY, BIND_SUBJECT
                        own[k] = (framed, u.start)
                    elif listed[k] is not None and listed[k][2] == _cues.REL_RESPONSIBILITY:
                        office, qstart, _r = listed[k]      # a sub-list lead-in inside a responsibility list, never a composition list
                        rel, binding = _cues.REL_RESPONSIBILITY, BIND_LIST
                phrases = _cues.duty_phrases(u.text, u.item, r)
                if office is not None and (rel is not None or office == _cues.UNIVERSITY) and binding != BIND_LIST:
                    add(Statement(office, rel, binding, r.subject_text, i, page, qstart, u.end, u.start, u.end, False, u.item, phrases))
                elif office is None and r.any_duty and r.subject_kind in (_cues.SUBJECT_OTHER, _cues.SUBJECT_ANAPHOR):
                    add(Statement(None, rel or _cues.REL_RESPONSIBILITY, BIND_OUTSIDE if own[k] is not None else BIND_ORPHAN,
                                  r.subject_text, i, page, u.start, u.end, u.start, u.end, False, u.item, phrases,
                                  own[k][0] if r.subject_kind == _cues.SUBJECT_ANAPHOR and own[k] is not None else None))
                if u.lead_in:
                    lead_office = office if office not in (None, _cues.UNIVERSITY) and rel in (_cues.REL_RESPONSIBILITY,
                                                                                               _cues.REL_COMPOSITION) else None
                    region, continues = self._list_region(units, valid, k)
                    for a, b in region:
                        for x in range(a, b + 1):
                            in_list[x] = True
                        item_reading = readings[a]
                        if item_reading is not None and item_reading.subject_kind in (_cues.SUBJECT_OFFICE, _cues.SUBJECT_OTHER):
                            continue                # an item that names its own subject stands on its own
                        item_phrases = _cues.duty_phrases(units[a].text, True, item_reading)
                        if lead_office is not None:
                            for x in range(a, b + 1):
                                if listed[x] is None:
                                    listed[x] = (lead_office, qstart, rel)
                            add(Statement(lead_office, rel, BIND_LIST, r.subject_text, i, page, qstart, units[b].end, units[a].start,
                                          units[a].end, continues, True, item_phrases))
                        elif item_phrases:
                            add(Statement(None, _cues.REL_RESPONSIBILITY, BIND_OUTSIDE if r.subject_kind == _cues.SUBJECT_OTHER else BIND_ORPHAN,
                                          r.subject_text, i, page, u.start, units[b].end, units[a].start, units[a].end, continues, True,
                                          item_phrases))
            for k, u in enumerate(units):          # "Ten colleges report to ... and include ...": enumerations of colleges
                if valid[k] and re.search(r"\bcolleges?\b[^.]{0,60}?\breports?\s+to\b", u.text, re.I):
                    if not any(e.page == page and _norm_key(self.resources.chunks[e.index][e.key_start:e.key_end]) == _norm_key(u.text)
                               for e in self.enumerations):
                        self.enumerations.append(Statement(None, None, BIND_OUTSIDE, "", i, page, u.start, u.end, u.start, u.end))
            for k, u in enumerate(units):          # a titled list item whose lead-in is not in this passage
                if valid[k] and u.item and not in_list[k] and listed[k] is None and readings[k].subject_kind is None:
                    if _cues.item_title(u.text):
                        add(Statement(None, _cues.REL_RESPONSIBILITY, BIND_ORPHAN, "", i, page, u.start, u.end, u.start, u.end, False, True,
                                      _cues.duty_phrases(u.text, True, readings[k])))
            for k, u in enumerate(units):
                if not valid[k]:
                    continue
                entry = _cues.location_entry(u.text) if _B3_LOCATION_ENTRY else self._loose_location(u.text)
                if entry is None:
                    continue
                key = (entry.office_id, entry.label, _norm_key(u.text))
                if key not in seen_place:
                    seen_place.add(key)
                    placements.append(Placement(entry.office_id, entry.label, entry.code, entry.place, i, page, u.start, u.end))
        bound_texts = {(st.page, _norm_key(self.resources.chunks[st.index][st.key_start:st.key_end])) for st in statements
                       if st.office_id not in (None, _cues.UNIVERSITY)}
        statements = [st for st in statements if st.office_id is not None
                      or (st.page, _norm_key(self.resources.chunks[st.index][st.key_start:st.key_end])) not in bound_texts]
        return statements, placements

    _OUTSIDE_DUTY = re.compile(r"\b(?:offers?|provides?|handles?|manages?|responsible\s+for|conducts?|oversees?|maintains?|supports?|"
                               r"delivers?|coordinates?|organi[sz]es?|administers?|operates?|supervises?)\b", re.I)

    def _read_outside_statements(self) -> List[Tuple[str, int, frozenset]]:
        """(office id, page, text without office names) of sentences of the approved handbook outside the Institutional
        scope whose subject is a lexicon office stated to do something ("The Deanship of Student Affairs ... offers
        comprehensive Counseling services", section 13.10). Never evidence: they only keep another office from being
        given as the only one."""
        out, seen = [], set()
        for i, meta in enumerate(self.resources.metadata):
            if meta.get("chunk_type") != "paragraph" or meta.get("source_id") != APPROVED_SOURCE_ID or self.scope.allows(meta):
                continue
            page = meta.get("page")
            if page in self.directory_pages or not isinstance(page, int):
                continue
            for u in _cues.split_units(self.resources.chunks[i]):
                r = _cues.read_unit(u.text, u.item)
                if r.any_duty or self._OUTSIDE_DUTY.search(u.text):      # any stated duty: ambiguity only, never evidence
                    key = (r.office_id, page, _norm_key(u.text))
                    if key not in seen:
                        seen.add(key)
                        out.append((r.office_id or "", page, _cues.stem_set(_cues.without_offices(u.text[r.core_start:]))))
        return out

    @staticmethod
    def _loose_location(text: str) -> Optional["_cues.LocationEntry"]:
        """Protection off only: any sentence naming a unit and a place ("... at the ... Campus")."""
        spans = _cues.office_spans(text)
        place = re.search(r"\b(?:on|at)\s+(the\s+[^.]+?(?:Campus|College|Complex)[^.]*?)\s*(?:\.|$)", text)
        if not spans or not place:
            return None
        return _cues.LocationEntry(spans[0][2], _cues.OFFICES_BY_ID[spans[0][2]].name, None, place.group(1).strip())

    def statement_quote(self, st: Statement) -> str:
        return self.resources.chunks[st.index][st.quote_start:st.quote_end].strip()

    def _matches(self, terms: Sequence[str], st: Statement, verb: Optional[str] = None) -> bool:
        """Whether a statement names the asked duty: one of its duty phrases (list-item title, object of its verb)
        matches the asked terms (``navigation_cues.phrase_matches``), and a matched list-item title is not narrowed by
        its own body (``_title_restricted``)."""
        phrases = list(st.phrases)
        if not _B3_OFFICE_WORDS_EXCLUDED and st.office_id not in (None, _cues.UNIVERSITY):
            office = _cues.OFFICES_BY_ID[st.office_id]
            phrases += [office.name] + [alias for alias, _pages in office.aliases]     # protection off: the name counts
        return any(_cues.phrase_matches(terms, p, all_terms=_B3_DUTY_ALL_TERMS, mask=_B3_OFFICE_WORDS_EXCLUDED, verb=verb)
                   and not self._title_restricted(st, p, terms) for p in phrases)

    def _title_restricted(self, st: Statement, phrase: str, terms: Sequence[str]) -> bool:
        """Whether a list item's body narrows the title the ask matched: the first time a title word comes back in the
        body, a word before it that is neither in the title, nor asked, nor a grading word narrows it ("Policy
        Development and Implementation: Establishing and maintaining health and safety policies" is a health-and-safety
        policy duty, not every policy development)."""
        if not st.item:
            return False
        text = self._statement_text(st)
        title = _cues.item_title(text)
        if not title or phrase not in (title, re.sub(r"\s*\([^)]*\)", "", title).strip()):
            return False
        body = text[text.index(title) + len(title):]
        tokens = re.findall(r"[A-Za-z][A-Za-z'’]*|[,.;:()]", body)
        title_stems = {_cues._stem(w) for w in re.findall(r"[A-Za-z][A-Za-z'’]*", title) if w.lower() not in _cues._STOP}
        asked = {_cues._stem(t) for t in terms}
        for i, tok in enumerate(tokens):
            if not tok[0].isalpha() or _cues._stem(tok) not in title_stems:
                continue
            j = i - 1
            while (j >= 0 and i - j <= 4 and tokens[j][0].isalpha() and not tokens[j].lower().endswith("ing")
                   and tokens[j].lower() not in ("the", "a", "an", "of", "to", "for", "in", "on", "with", "by", "from")
                   and not _cues._ANY_DUTY_RX.fullmatch(tokens[j])):        # a verb ("extends", "provides") ends the noun phrase
                w = tokens[j].lower()
                if w not in ("and", "or") and _cues._stem(w) not in title_stems and _cues._stem(w) not in asked \
                        and w not in _cues._PLAIN_MODIFIERS:
                    return True
                j -= 1
            return False                     # only the first time a title word comes back
        return False

    def _collapse(self, statements: Sequence[Statement]) -> List[Statement]:
        """For display: one statement per quoted passage (the longest quote from one start), and no quote that another kept
        quote of the same page already contains."""
        longest: Dict[Tuple[int, int], Statement] = {}
        for st in statements:
            key = (st.index, st.quote_start)
            if key not in longest or st.quote_end > longest[key].quote_end:
                longest[key] = st
        kept = [st for st in statements if longest[(st.index, st.quote_start)] is st]
        out: List[Statement] = []
        for pos, st in enumerate(kept):
            q = _norm_key(self.statement_quote(st))
            if any(o is not st and o.page == st.page and q in _norm_key(self.statement_quote(o)) and
                   (len(_norm_key(self.statement_quote(o))) > len(q) or other < pos) for other, o in enumerate(kept)):
                continue
            out.append(st)
        return out

    # ---- phase B3: findings ----------------------------------------------------
    def _b3_finding(self, task: SpecialistTask, number: int, st: Statement, clause_index: int, *, terms: Sequence[str] = (),
                    extra: Optional[Dict[str, Any]] = None) -> Finding:
        meta = self.resources.metadata[st.index]
        quote = self.statement_quote(st)
        if st.office_id == _cues.UNIVERSITY:
            office_name, tier = _cues.UNIVERSITY_NAME, _cues.STATED_FACT
        elif st.office_id is not None:              # a chair or a composition is a structure fact, not a responsibility
            office_name = _cues.OFFICES_BY_ID[st.office_id].name
            tier = _cues.E2 if st.relation == _cues.REL_RESPONSIBILITY else _cues.STATED_FACT
        else:
            office_name, tier = None, None
        metadata: Dict[str, Any] = {
            "chunk_id": meta.get("chunk_id"), "chunk_type": meta.get("chunk_type"), "chapter": meta.get("chapter"),
            "section_no": meta.get("section_no"), "section_title": meta.get("section_title"),
            "page_section_nos": list(meta.get("page_section_nos") or []),
            "boundary_guard": "attributed to an owned section" if self.guard.is_shared(meta) else "page carries owned sections only",
            "clauses": [clause_index], "office_id": st.office_id, "office_name": office_name, "relation": st.relation,
            "binding": st.binding, "subject_as_printed": st.subject_text, "evidence_tier": tier, "duty_terms": list(terms),
            "duty_phrases": list(st.phrases), "list_continues": st.list_continues, "extractive": True,
            "statement_text": self.resources.chunks[st.index][st.key_start:st.key_end].strip(),
            "quote_scope": {BIND_LIST: "lead-in and list item", BIND_ANAPHOR: "antecedent sentence and statement"}.get(st.binding, "statement"),
        }
        if extra:
            metadata.update(extra)
        page = meta.get("page")
        return Finding(finding_id="%s-f%d" % (task.task_id, number), claim=quote, evidence_quote=quote, source_id=meta["source_id"],
                       source_title=str(meta.get("source_title") or self.resources.source.title),
                       page=page if isinstance(page, int) and not isinstance(page, bool) and page >= 1 else None,
                       confidence=1.0 if tier is not None else 0.5, metadata=metadata)

    def _placement_finding(self, task: SpecialistTask, number: int, pl: Placement, clause_index: int) -> Finding:
        meta = self.resources.metadata[pl.index]
        quote = self.resources.chunks[pl.index][pl.start:pl.end].strip()
        metadata: Dict[str, Any] = {
            "chunk_id": meta.get("chunk_id"), "chunk_type": meta.get("chunk_type"), "chapter": meta.get("chapter"),
            "section_no": meta.get("section_no"), "section_title": meta.get("section_title"),
            "page_section_nos": list(meta.get("page_section_nos") or []),
            "boundary_guard": "attributed to an owned section" if self.guard.is_shared(meta) else "page carries owned sections only",
            "clauses": [clause_index], "office_id": pl.office_id, "office_name": _cues.OFFICES_BY_ID[pl.office_id].name,
            "relation": "location", "binding": "location entry", "entry_label": pl.label, "building_code": pl.code, "place": pl.place,
            "evidence_tier": _cues.STATED_FACT, "extractive": True, "quote_scope": "location entry",
        }
        page = meta.get("page")
        return Finding(finding_id="%s-f%d" % (task.task_id, number), claim=quote, evidence_quote=quote, source_id=meta["source_id"],
                       source_title=str(meta.get("source_title") or self.resources.source.title),
                       page=page if isinstance(page, int) and not isinstance(page, bool) and page >= 1 else None, metadata=metadata)

    def _placement_text(self, pl: Placement) -> str:
        return self.resources.chunks[pl.index][pl.start:pl.end]

    # ---- phase B3: answering ---------------------------------------------------
    _COLLEGES_ASK = re.compile(r"^\s*(?:which|what)\s+(?:are\s+the\s+)?colleges\s+(?:does|do)\s+(?:the\s+university(?:\s+of\s+sharjah)?|uos)\s+"
                               r"(?:have|has|include|comprise)\s*\??\s*$|"
                               r"^\s*(?:which|what)\s+colleges\s+(?:are\s+there|exist)(?:\s+(?:at|in)\s+the\s+university(?:\s+of\s+sharjah)?)?\s*\??\s*$|"
                               r"^\s*what\s+are\s+the\s+colleges(?:\s+(?:of|at|in)\s+the\s+university(?:\s+of\s+sharjah)?)?\s*\??\s*$|"
                               r"^\s*(?:list|name)\s+(?:all\s+)?(?:of\s+)?the\s+colleges(?:\s+(?:of|at|in)\s+the\s+university(?:\s+of\s+sharjah)?)?\s*[.?]?\s*$",
                               re.I)
    _FACET_ASK = re.compile(r"\b(?:floors?|rooms?|hours?|open(?:ing)?|clos(?:e|es|ing)|times?|timings?|schedules?|maps?|gps|coordinates|directions?|"
                            r"address(?:es)?|phone|telephone|levels?|stor(?:e)?ys?|stories|upstairs|downstairs|basement|ground\s+level|wings?|"
                            r"blocks?|gates?|entrances?|pins?|streets?|bus(?:\s+stops?)?|nearest|near(?:by)?|next\s+to|landmarks?|parking|"
                            r"building\s+numbers?)\b", re.I)
    #: Teaching & Learning and Faculty Services & HR subjects that the shared ownership cues do not cover; such clauses
    #: are not answered here (fail closed; the handoff itself is phase B4).
    _TEACHING_SUBJECT = re.compile(r"\b(?:exam(?:s|ination|inations)?|grad(?:e|es|ing)|curricul(?:a|um|ar)|syllab(?:us|i)|study\s+plans?|courses?|"
                                   r"teaching|lectures?|classes|class\s+schedules?|office\s+hours?|learning\s+outcomes?|thes(?:is|es)|dissertations?|"
                                   r"academic\s+(?:integrity|advising|calendar|honesty|misconduct|probation|supervision)|plagiarism|attendance|"
                                   r"student\s+assessments?|assessments?\s+of\s+students|course\s+management|"
                                   r"(?:professional|faculty|teaching|staff|academic|career)\s+development|"
                                   r"(?:program|programme|curriculum|course)\s+development)\b|(?<!risk\s)\bassessments?\b",
                                   re.I)
    _HR_SUBJECT = re.compile(r"\b(?:health\s*care|medical\s+(?:insurance|care|services?|clinics?|leave)|holidays?|leaves?|vacations?|"
                             r"salar(?:y|ies)|allowances?|payroll|contracts?|promotions?|appraisals?|faculty\s+evaluations?|housing|accommodation|"
                             r"recruitment|hiring|appointments?|resignations?|retirement|end\s+of\s+service|visas?|residency|insurance|"
                             r"well-?being|wellness)\b|\bhealth\b(?!\s*(?:and|&)\s*safety)(?!\s+(?:campus|sciences?))", re.I)
    _DUTY_FRAMES = (
        r"\b(?:responsible\s+for|in\s+charge\s+of|deals?\s+with|takes?\s+care\s+of|handles?|manages?|oversees?|provides?|offers?|supports?|"
        r"administers?|maintains?|runs?|coordinates?|issues?|organi[sz]es?|conducts?|delivers?|supervises?|operates?|helps?\s+with|"
        r"facilitates?|arranges?|looks?\s+after|promotes?|implements?)\s+(?P<obj>[^?]+)",
        r"\b(?:contact|ask|talk\s+to|speak\s+to|reach|call|go\s+to|see|turn\s+to)\b[^?]*?\b(?:about|regarding|for|concerning|with)\s+(?P<obj>[^?]+)",
        r"\breport\s+(?P<obj>[^?]+)",
    )
    _ROLE_WORDS = frozenset("""what which does do did is are the a an role roles responsibility responsibilities responsible function functions duty
duties task tasks mandate mission job work purpose service services provide provides offer offers handle handles manage manages oversee oversees
support supports administer administers maintain maintains deliver delivers charge for of in to at on with tell me about main key""".split())
    _NUMBER_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
                     "nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred")

    _AUTHORITY_WORDS = re.compile(r"\b(?:approv\w*|authori[sz]\w*|authorit(?:y|ies)|sign(?:s|ing)?\s+off|decid\w*|grant(?:s|ing|ed)?|confer\w*|"
                                  r"entitled\s+to|power\s+to|mandate\s+(?:and\s+authority\s+)?to|empower\w*|mandated|(?:permitted|allowed)\s+to|"
                                  r"permission\s+to|competen(?:ce|t)\s+to|jurisdiction)\b|"
                                  r"\b(?:has|have|holds?|with|given)\s+the\s+(?:mandate|authority|power|right)\b", re.I)
    _YES_NO = re.compile(r"^\s*(?:is|are|was|were|does|do|did|can|could|has|have)\b", re.I)

    # ---- the question forms phase B3 answers (after the fourth adversarial review) ---------------------------------------------
    # Every additional word a question may carry (a place, a person, a time, an exclusion, a second ask, a beneficiary put in
    # front, a qualifier of a member or a chair) can change what is asked, and closed lists of such words never covered them
    # all. So phase B3 answers a clause only when the WHOLE clause has one of these forms, with one lexicon office (⟦O⟧) where
    # the form names one and a duty that is a plain noun phrase; any other clause is not answered (fail closed).
    _WHO = r"(?:who|which\s+(?:office|unit|department|section|cent(?:er|re)|body))"
    _DUTY_VERBS = (r"(?:handles|manages|oversees|provides|maintains|administers|coordinates|conducts|organi[sz]es|delivers|offers|supports|issues|"
                   r"operates|runs|supervises|facilitates|arranges|promotes|implements)")
    _ROLE_VERBS = (r"(?:provide|offer|handle|manage|oversee|support|administer|maintain|deliver|issue|coordinate|operate|conduct|organi[sz]e|"
                   r"facilitate|supervise|run|promote|implement)")
    _UNI = r"(?:the\s+university(?:\s+of\s+sharjah)?|uos)"
    _PARTS = r"(?:sub-?groups|cent(?:er|re)s|sections|departments|divisions|units|sub-?units|programs|programmes)"
    _FORMS = tuple((intent, re.compile(p, re.I)) for intent, p in (
        ("responsible_office", r"%s\s+(?:is|are)\s+(?:responsible\s+for|in\s+charge\s+of)\s+(?P<duty>.+)" % _WHO),
        ("responsible_office", r"%s\s+(?P<verb>%s)\s+(?P<duty>.+)" % (_WHO, _DUTY_VERBS)),
        ("responsible_office", r"%s\s+(?:deals\s+with|takes\s+care\s+of|looks\s+after)\s+(?P<duty>.+)" % _WHO),
        ("responsible_office", r"(?:who|which\s+(?:office|unit|department))\s+(?:should|do|can|could|must|would)\s+i\s+(?:contact|ask|talk\s+to|"
                               r"speak\s+to|reach|call|go\s+to|see|turn\s+to)\s+(?:about|regarding|for|concerning|with)\s+(?P<duty>.+)"),
        ("office_role", r"what\s+(?:does|do)\s+⟦O⟧\s+do"),
        ("office_role", r"what\s+(?:is|are)\s+⟦O⟧\s+responsible\s+for"),
        ("office_role", r"what\s+(?:is|are)\s+the\s+(?:main\s+|key\s+)?(?:role|roles|responsibilit(?:y|ies)|functions?|duties|mandate|mission|"
                        r"services)\s+of\s+⟦O⟧"),
        ("office_role", r"what\s+services\s+(?:does|do)\s+⟦O⟧\s+(?:provide|offer)(?:\s+(?:to|for)\s+(?:faculty|staff|students)(?:\s+members)?)?"),
        ("office_role", r"tell\s+me\s+about\s+⟦O⟧"),
        ("office_role", r"(?:is|are)\s+⟦O⟧\s+(?:responsible\s+for|in\s+charge\s+of)\s+(?P<duty>.+)"),
        ("office_role", r"(?:does|do)\s+⟦O⟧\s+(?P<verb>%s)\s+(?P<duty>.+)" % _ROLE_VERBS),
        ("chair", r"who\s+chairs\s+⟦O⟧|who\s+is\s+the\s+chair(?:person|man|woman)?\s+of\s+⟦O⟧|who\s+is\s+⟦O⟧\s+chaired\s+by|"
                  r"who\s+presides\s+over\s+(?:the\s+meetings\s+of\s+)?⟦O⟧"),
        ("composition", r"who\s+(?:sits|serves)\s+on\s+⟦O⟧|who\s+is\s+on\s+⟦O⟧|"
                        r"who\s+are\s+the\s+members\s+of\s+⟦O⟧|what\s+is\s+the\s+(?:composition|membership)\s+of\s+⟦O⟧|"
                        r"what\s+(?:does|do)\s+⟦O⟧\s+(?:consist\s+of|comprise|include)|what\s+is\s+⟦O⟧\s+(?:composed|made\s+up)\s+of"),
        ("composition", r"(?:what|which)\s+(?:are\s+the\s+)?%s\s+of\s+⟦O⟧|(?:what|which)\s+%s\s+(?:does|do)\s+⟦O⟧\s+(?:have|comprise|include|"
                        r"consist\s+of)|(?:what|which)\s+%s\s+(?:make\s+up|are\s+in)\s+⟦O⟧" % (_PARTS, _PARTS, _PARTS)),
        ("organisation", r"(?:what\s+is|describe|explain)\s+the\s+(?:organi[sz]ational|governance)\s+(?:structure|chart)(?:\s+of\s+%s)?|"
                         r"how\s+is\s+%s\s+(?:organi[sz]ed|structured|governed)|what\s+are\s+the\s+governance\s+bodies(?:\s+of\s+%s)?|"
                         r"what\s+is\s+the\s+structure\s+of\s+%s" % (_UNI, _UNI, _UNI, _UNI)),
        ("count", r"how\s+many\s+(?P<noun>[a-z][a-z\- ]{1,40}?)\s+(?:does|do)\s+%s\s+(?:have|offer|manage|operate|run|include|comprise)|"
                  r"how\s+many\s+(?P<noun2>[a-z][a-z\- ]{1,40}?)\s+(?:are\s+there|exist)(?:\s+(?:at|in)\s+%s)?" % (_UNI, _UNI)),
        ("location", r"(?:where\s+(?:is|are)|where's|where\s+(?:can|do)\s+i\s+find|whereabouts\s+is|what\s+is\s+the\s+location\s+of)\s+"
                     r"(?:(?:women|men)(?:'s|s)?\s+)?⟦O⟧(?P<sel>[^?]*)"),
        ("location", r"what\s+is\s+the\s+building\s+code\s+(?:for|of)\s+(?:the\s+)?(?:(?:women|men)(?:'s|s)?\s+)?⟦O⟧(?P<sel3>[^?]*)"),
        ("location", r"(?:which|what)\s+campus\s+is\s+⟦O⟧\s+(?:on|in|at)|(?:which|what)\s+building\s+is\s+⟦O⟧\s+in|"
                     r"(?:can\s+you\s+)?tell\s+me\s+where\s+⟦O⟧(?P<sel2>[^?]*?)\s+is"),
    ))
    #: Words a duty slot may not hold: they give the duty a place, a time, a person, a condition, an exclusion or a second ask.
    _SLOT_FORBIDDEN = frozenset("""in on at to from by into onto towards toward via through within across among between under over about regarding
concerning against without except excluding besides beyond outside instead rather than not no never nor but if when whenever while because since
so although though my our your their his her its it this that these those us me i we you they them who whom whose which what where why how
other else another any some each every own same such following above below before after during until since last next first second former previous
current new old only also too there here""".split())
    _COURTESY_START = re.compile(r"^\s*(?:(?:please|kindly|hi|hello|dear\s+(?:sir|madam|all|team|colleagues?))\s*,?\s+)+", re.I)
    _COURTESY_END = re.compile(r"\s*,?\s*(?:please|thanks|thank\s+you(?:\s+very\s+much)?)?\s*[?.!]*\s*$", re.I)

    def _clean(self, clause: str) -> str:
        """The clause without its courtesy words, office names kept."""
        return self._COURTESY_END.sub("", self._COURTESY_START.sub("", _cues.normalise_text(clause))).strip()

    def _shape(self, clause: str) -> Tuple[str, List[str]]:
        """The clause as its form is read: courtesy words off, every office name replaced by ⟦O⟧ (with its "the")."""
        norm = _cues.normalise_text(clause)
        spans = _cues.office_spans(norm)
        out, last = [], 0
        for start, end, o in spans:
            before = norm[last:start]
            if _cues.OFFICES_BY_ID[o].kind in ("council", "board"):
                before = re.sub(r"\b(?:a|an|each|every)\s+$", "", before, flags=re.I)
            out.append(before)
            out.append("⟦O⟧")
            last = end
        out.append(norm[last:])
        shape = re.sub(r"\bthe\s+⟦O⟧", "⟦O⟧", "".join(out), flags=re.I)
        shape = re.sub(r"⟦O⟧\s+(?:section|office|department|unit|cent(?:er|re))\b", "⟦O⟧", shape, flags=re.I)   # "the EHSM Section"
        shape = self._COURTESY_END.sub("", self._COURTESY_START.sub("", shape))
        return re.sub(r"\s+", " ", shape).strip(), [o for _s, _e, o in spans]

    def _duty_ok(self, duty: str) -> bool:
        """Whether a duty slot is a plain noun phrase: letters only, at most one "of", one "with" and one "and" / "or", no
        office, place, pronoun, time, condition or exclusion word, at most seven content words; a trailing beneficiary ("for
        faculty") or setting ("on campus") is allowed."""
        d = re.sub(r"\s+(?:for|to)\s+(?:the\s+|all\s+|new\s+)?(?:faculty|staff|students|members|employees|everyone)(?:\s+members)?\s*$", "", duty.strip(), flags=re.I)
        d = re.sub(r"\s+(?:on|at|in|across|within)\s+(?:the\s+)?(?:campus|university(?:\s+of\s+sharjah)?|uos)\s*$", "", d, flags=re.I)
        if not d or "⟦O⟧" in d or _cues.elsewhere(d):
            return False
        words = re.findall(r"\S+", d)
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z'’\-]*", w) for w in words):
            return False
        low = [w.lower() for w in words]
        if any(w.lower() in self._SLOT_FORBIDDEN and w != "IT" for w in words):       # "IT" is the printed abbreviation, not "it"
            return False
        if low.count("of") > 1 or low.count("with") > 1 or low.count("and") + low.count("or") > 1:
            return False
        if any(w in _cues._FRAME and w not in _cues._UNIT_WORDS for w in low):
            return False                      # "the IT support charge", "reporting of risk assessments": a frame word is no duty word
        return len([w for w in low if w not in ("the", "a", "an", "of", "with", "and", "or")]) <= 7

    def _question_form(self, clause: str) -> Optional[str]:
        """The phase-B3 intent of a clause whose whole text has one of ``_FORMS``, or None."""
        if _cues.elsewhere(clause):
            return None                       # another institution or place: never the handbook's own units
        shape, offices = self._shape(clause)
        for intent, form in self._FORMS:
            m = form.fullmatch(shape)
            if not m:
                continue
            needs_office = "⟦O⟧" in form.pattern
            if needs_office and shape.count("⟦O⟧") != 1:
                continue
            if not needs_office and offices:
                continue
            duty = m.groupdict().get("duty")
            if duty is not None and not self._duty_ok(duty):
                continue
            return intent
        return None

    def _b3_intent(self, clause: str, kind: str) -> str:
        """The answering intent of a clause that ``_boundary`` left to Institutional Navigation."""
        masked = _cues.without_offices(clause)
        if kind in (_cues.PROCEDURE, _cues.SYSTEM_ACCESS):
            return "procedure" if _B3_PROCEDURE_GUARD else (self._question_form(clause) or "unrecognised")
        if kind == _cues.AUTHORITY or self._AUTHORITY_WORDS.search(masked):
            return "authority"
        if self._COLLEGES_ASK.search(clause):
            return "colleges"
        return self._question_form(clause) or "unrecognised"

    # ---- phase B4: whose subject the clause is ----------------------------------------------------------------------
    #: Forms whose object is the named office itself: its own name never makes the clause another specialist's subject.
    _OFFICE_REFERENT_FORMS = frozenset({"office_role", "chair", "composition", "location"})
    _OWNER_LABEL = {SpecialistId.RESEARCH: "Research & Innovation", SpecialistId.FACULTY_SERVICES: "Faculty Services & HR",
                    SpecialistId.TEACHING: "Teaching & Learning"}

    @staticmethod
    def _requester(task: SpecialistTask) -> Optional[SpecialistId]:
        """The specialist whose handoff request produced the task (None for a Coordinator task)."""
        try:
            return SpecialistId.parse(task.requested_by)
        except ValueError:
            return None

    def _boundary(self, task: SpecialistTask, clause: str, kind: str) -> Optional[Dict[str, Any]]:
        """Phase B4 ownership check of one non-contact clause: None when its subject is Institutional Navigation's;
        otherwise the owner and the handoff targets. Only the shared cues and the local subject lists are read; nothing
        about the other specialists' tasks is known or assumed."""
        if not _B3_OWNERSHIP_GUARD:
            return None
        if _B4_NAVIGATION_KEPT and kind in (_cues.CHANNEL, _cues.LOCATION):
            return None                       # a contact or location ask is navigation, whoever's unit it names
        masked = _cues.without_offices(clause)
        referent = _B4_NAVIGATION_KEPT and self._question_form(clause) in self._OFFICE_REFERENT_FORMS
        text = masked if referent else clause
        owners: List[SpecialistId] = []          # named by the SHARED cues: the only owners a handoff may go to (R9)
        cues: List[str] = []
        if _domain.research_owned(text):
            owners.append(SpecialistId.RESEARCH)
            cues.append("research_owned")
        if _domain.hr_owned(text):
            owners.append(SpecialistId.FACULTY_SERVICES)
            cues.append("hr_owned")
        local_hr = bool(self._HR_SUBJECT.search(masked))
        teaching = bool(self._TEACHING_SUBJECT.search(masked))
        if local_hr:
            cues.append("local HR subject list")
        if teaching:
            cues.append("local Teaching subject list")
        if not owners and not local_hr and not teaching:
            return None
        if owners and teaching and _B4_SHARED_OWNER_ONLY:
            # the shared cues name an owner, but the clause also has a Teaching subject ("Who handles exam appointments?"):
            # ownership is ambiguous, and a handoff would guess
            return {"owners": owners + [SpecialistId.TEACHING], "handoff": [], "cues": cues, "handed_back": [], "kind": "ambiguous"}
        if not owners and (teaching or _B4_SHARED_OWNER_ONLY):
            # only Institutional Navigation's own word lists claim the clause: it is not answered, and no target is guessed
            local = ([SpecialistId.FACULTY_SERVICES] if local_hr else []) + ([SpecialistId.TEACHING] if teaching else [])
            return {"owners": local, "handoff": [], "cues": cues, "handed_back": [],
                    "kind": "teaching" if local == [SpecialistId.TEACHING] else "unconfirmed"}
        if not owners:                            # _B4_SHARED_OWNER_ONLY off: the local HR list names the owner too
            owners = [SpecialistId.FACULTY_SERVICES]
        requester = self._requester(task)
        handed_back = [o for o in owners if o == requester and _B4_NO_HAND_BACK]
        targets = [o for o in owners if o not in handed_back] if _B4_OWNER_HANDOFF else []
        return {"owners": owners, "handoff": targets, "cues": cues, "handed_back": handed_back, "kind": "shared"}

    def _handoff(self, task: SpecialistTask, target: SpecialistId, clause: str, index: int, ordinal: int = 0) -> HandoffRequest:
        """The clause as data for its owner, in the shape the other specialists use: the full question and this one
        clause as focus, no evidence, no answer."""
        suffix = "%d" % index if ordinal == 0 else "%d-%d" % (index, ordinal)
        return HandoffRequest(
            from_specialist=self.specialist_id, requested_specialist=target,
            reason="misrouted clause %d belongs to %s: %s" % (index, target.value, clause[:120]),
            task=SpecialistTask(task_id="%s-handoff-%s" % (task.task_id, suffix), question=task.question, specialist_id=target,
                                domain=target.value, intent=task.intent, level=task.level, system=task.system,
                                context={"full_question": task.question, "focus_clauses": [clause]}, requested_by=self.specialist_id.value,
                                metadata={"origin_task_id": task.task_id, "clause_index": index, "kind": "misrouted"}),
            metadata={"clause_index": index, "kind": "misrouted"})

    def _asked_duty(self, clause: str, after: int = 0) -> Tuple[str, ...]:
        return self._asked_duty_and_verb(clause, after)[0]

    def _asked_duty_and_verb(self, clause: str, after: int = 0) -> Tuple[Tuple[str, ...], Optional[str], str]:
        """(duty terms, the asked verb, the duty object as printed in the clause)."""
        text = _cues.without_offices(clause)[after:]
        for frame in self._DUTY_FRAMES:
            m = re.search(frame, text, re.I)
            if m:
                verb = re.match(r"\s*([A-Za-z]+)", text[m.start():m.start("obj")])
                terms = _cues.ask_terms(" " + m.group("obj"))
                if terms and all(t.lower() in _cues.BENEFICIARIES for t in terms):
                    terms = ()                      # who is served is no duty ("Who is responsible for faculty?")
                return terms, (verb.group(1).lower() if verb else None), m.group("obj")
        return (), None, ""

    def _role_duty(self, clause: str) -> Tuple[Tuple[str, ...], Optional[str], str]:
        """The duty a named-office role ask adds ("Is the HSO responsible for risk assessments?" -> risk, assessments), the
        asked verb and the duty object as printed; no terms for "What does X do?", "What services does X provide to
        faculty?"."""
        spans = _cues.office_spans(clause)
        terms, verb, obj = self._asked_duty_and_verb(clause, spans[-1][1] if spans else 0)
        if terms:
            return terms, verb, obj
        residual = [t for t in _cues.ask_terms(" " + _cues.without_offices(clause)) if t.lower() not in self._ROLE_WORDS]
        return tuple(residual), None, ""

    #: What may follow an office's possessive in an ask about the office itself ("the College Council's members").
    _POSSESSIVE_OK = frozenset("""members membership chair chairperson composition role roles responsibility responsibilities function functions duty
duties mandate mission services service location locations structure subgroups sub-groups sections centers centres departments divisions units
main own""".split())

    def _one_office(self, report: Dict[str, Any], missing: List[str], intent: str, *, allow_men_women: bool = False,
                    allow_parts: bool = False, ignore_text: str = "") -> Optional[str]:
        i, clause, offices = report["index"], report["clause"], report["offices"]
        label = intent.replace("_", " ")
        if not offices:
            missing.append("clause %d (%s ask): not answered; the clause names no unit of the Institutional Navigation lexicon: %s" % (i, label, clause))
            return None
        if len(offices) > 1:
            missing.append("clause %d (%s ask): not answered; the clause names several units (%s); ask about one at a time: %s"
                           % (i, label, ", ".join(_cues.OFFICES_BY_ID[o].name for o in offices), clause))
            return None
        name = _cues.OFFICES_BY_ID[offices[0]].name
        other_place = _cues.elsewhere(clause)
        if other_place:
            missing.append("clause %d (%s ask): not answered; the clause asks about %s at or of %s, which the handbook does not document: %s"
                           % (i, label, name, other_place, clause))
            return None
        norm = _cues.normalise_text(clause)
        for _s, e, _o in _cues.office_spans(norm):
            pm = re.match(r"(?:'s|s')\s+([A-Za-z][A-Za-z\-]*)", norm[e:])
            if pm and pm.group(1).lower() not in self._POSSESSIVE_OK:
                missing.append("clause %d (%s ask): not answered; the clause asks about %s's %s, which is not %s itself: %s"
                               % (i, label, name, pm.group(1), name, clause))
                return None
        subject_part = clause.replace(ignore_text, " ") if ignore_text and ignore_text in clause else clause
        others = list(_cues.other_unit_mentions(subject_part))      # the asked duty object ("the Data Center") is no other unit
        if allow_parts:                      # a composition ask names the parts it asks for ("What centers does ... comprise?")
            others = [w for w in others if not re.fullmatch(r"cent(?:er|re)s?|sections?|departments?|units?|divisions?|programs?|programmes?|"
                                                            r"offices?|labs?|laborator(?:y|ies)|librar(?:y|ies)|schools?", w, re.I)]
        if others:
            missing.append("clause %d (%s ask): not answered; the clause asks about a unit other than %s (%s), which is not answered "
                           "from %s's statements: %s" % (i, label, name, ", ".join(others), name, clause))
            return None
        exempt = ("men", "women", "male", "female", "ladies", "boys", "girls", "campus")
        chair_noun = intent == "chair" and re.search(r"\bchair(?:person)?\s+of\b|['’]s\s+chair(?:person)?\b", norm, re.I)
        nouns = [w for w in _cues.subunit_nouns(clause) if not (allow_men_women and w in exempt)
                 and not (chair_noun and w in ("chair", "chairperson"))]
        if nouns:
            missing.append("clause %d (%s ask): not answered; the clause asks about a %s, which is not %s itself: %s" % (i, label, nouns[0], name, clause))
            return None
        return offices[0]

    def _answer_b3(self, task: SpecialistTask, report: Dict[str, Any], findings: List[Finding], missing: List[str], summary: List[str],
                   limitations: List[str]) -> None:
        i, clause, kind = report["index"], report["clause"], report["ask_kind"]
        intent = self._b3_intent(clause, kind)
        report["b3_intent"] = intent
        before_missing, before_findings = len(missing), len(findings)

        def emit(items) -> List[str]:
            ids = []
            for item in items:
                if isinstance(item, Placement):
                    f = self._placement_finding(task, len(findings) + 1, item, i)
                else:
                    st, terms = item
                    f = self._b3_finding(task, len(findings) + 1, st, i, terms=terms)
                findings.append(f)
                ids.append(f.finding_id)
            return ids

        if intent == "procedure":
            missing.append("clause %d (%s ask): not answered; the approved handbook gives no step-by-step procedure (E3 evidence) for "
                           "this, and Institutional Navigation never composes steps from a responsibility statement: %s" % (i, kind, clause))
        elif intent == "authority":
            missing.append("clause %d (approval ask): not answered; approval and authority asks are not answered by Institutional "
                           "Navigation in Milestone 4 (governance-level authority is future work), and it never infers an approver (an "
                           "office that manages or handles a matter is not shown to approve it): %s" % (i, clause))
        elif intent == "unrecognised" and kind == _cues.LOCATION and not report["offices"]:
            missing.append("clause %d (location ask): not answered; the clause names no office of the Institutional Navigation lexicon, "
                           "so its unit cannot be verified and no location is given (an office the handbook does not name is never "
                           "given): %s" % (i, clause))
        elif intent == "unrecognised":
            missing.append("clause %d (%s ask): not answered; Institutional Navigation answers only plain question forms "
                           "(who is responsible for / which office handles <duty>; what does <office> do; is <office> responsible for "
                           "<duty>; who chairs / who are the members of <office>; what is the organisational structure of the university; "
                           "how many <units> does the university have; where is <library>), and this clause has other words that could "
                           "change what is asked: %s" % (i, kind, clause))
        elif intent == "other":
            missing.append("clause %d (%s ask): not answered; in phase %s Institutional Navigation answers telephone and fax asks, office "
                           "responsibilities, structure and documented locations only: %s" % (i, kind, PHASE, clause))
        elif intent == "location":
            self._answer_location(report, missing, summary, emit)
        elif intent == "count":
            self._answer_count(dict(report, clause=self._clean(clause)), missing, summary, emit)
        elif intent == "colleges":
            self._answer_colleges(report, missing, summary, emit)
        elif intent == "organisation":
            self._answer_organisation(dict(report, clause=self._clean(clause)), missing, summary, emit, limitations)
        elif intent in ("chair", "composition"):
            self._answer_structure(report, missing, summary, emit, intent)
        elif intent == "office_role":
            self._answer_office_role(report, missing, summary, emit)
        else:
            self._answer_responsible_office(report, missing, summary, emit)
        if len(findings) > before_findings:
            tiers = [f.metadata.get("evidence_tier") for f in findings[before_findings:]]
            report["evidence_tier_reached"] = _cues.E2 if _cues.E2 in tiers else (_cues.STATED_FACT if _cues.STATED_FACT in tiers else None)
        report["answered"] = len(missing) == before_missing and len(findings) > before_findings
        report["findings"] = [f.finding_id for f in findings[before_findings:]]

    _UNI_SCOPE = re.compile(r"\b(?:of|across|throughout|for)\s+(?:the\s+(?:whole\s+|entire\s+)?)?(?:university(?:\s+of\s+sharjah)?|uos)\b|"
                            r"\buniversity(?:'s|’s)|\buniversity-wide\b|\b(?:whole|entire)\s+university\b", re.I)
    _COLLEGE_SCOPE = re.compile(r"\b(?:the|its|each|every|a|respective)\s+(?:respective\s+)?college(?:'s|’s)?\b|\bcollege(?:'s|’s)|"
                                r"\bwithin\s+(?:the\s+|each\s+|its\s+)?(?:respective\s+)?(?:college|department)\b", re.I)

    def _scope_ok(self, st: Statement, university_scope: bool) -> bool:
        """An ask about the University as a whole ("... of the University") is not answered from a college-level body's
        statement ("... within the respective college")."""
        if not university_scope or st.office_id in (None, _cues.UNIVERSITY):
            return True
        if _cues.OFFICES_BY_ID[st.office_id].kind == "college" or st.office_id == "college_council":
            return False
        return not self._COLLEGE_SCOPE.search(self.statement_quote(st))

    def _statement_text(self, st: Statement) -> str:
        return self.resources.chunks[st.index][st.key_start:st.key_end]

    def _core_stems(self, st: Statement) -> frozenset:
        key = (st.index, st.key_start, st.key_end)
        cache = self.__dict__.setdefault("_core_stem_cache", {})
        if key not in cache:
            cache[key] = _cues.stem_set(self._core_text(st))
        return cache[key]

    def _core_text(self, st: Statement) -> str:
        """The statement without its section heading and without office names, for the word check."""
        text = self._statement_text(st)
        return _cues.without_offices(text[_cues.read_unit(text).core_start:])

    _SERVED_BEFORE = re.compile(r"(?:\b(?:for|to|towards|regarding|about|concerning|serving|supporting)\s+(?:the\s+)?|\b(?:if|when|whenever|once)\s+"
                                r"(?:the\s+)?|\b(?:my|our)\s+(?:office|department|unit|cent(?:er|re)|team)\s+is\s+(?:the\s+)?)$", re.I)
    _SERVED_UNIT = re.compile(r"\b(?:to|for|towards)\s+(?:the\s+|our\s+|my\s+)?(?:[\w&'’-]+\s+){0,5}?(?:cent(?:er|re)|department|unit|team|office|"
                              r"section|division|directorate|deanship|committee|council)s?\b", re.I)

    def _office_in_object(self, clause: str) -> Optional[str]:
        """The unit a responsibility ask names as the one served or asked about, not the one that does the duty: a lexicon
        office inside the asked duty ("Who provides support to the Business Service Department?"), any unit noun after
        "to" / "for" in it ("... to the IT Center"), or an office fronted as the beneficiary ("For the ITC, who ...?", "If
        the BSD needs help, ...", "My department is the BSD. ..."). Returns its name as printed."""
        norm = _cues.normalise_text(clause)
        spans = _cues.office_spans(norm)
        for frame in self._DUTY_FRAMES:
            m = re.search(frame, norm, re.I)
            if m:
                for start, _end, office_id in spans:
                    if start >= m.start("obj"):
                        return _cues.OFFICES_BY_ID[office_id].name
                unit = self._SERVED_UNIT.search(_cues.without_offices(norm)[m.start("obj"):])
                if unit:
                    return unit.group(0).strip()
                break
        for start, _end, office_id in spans:
            if self._SERVED_BEFORE.search(norm[max(0, start - 40):start]):
                return _cues.OFFICES_BY_ID[office_id].name
        return None

    def _answer_responsible_office(self, report, missing, summary, emit) -> None:
        i, clause = report["index"], report["clause"]
        served = self._office_in_object(clause)
        if served is not None:
            missing.append("clause %d (responsibility ask): not answered; the clause names %s as the unit the duty is done for or about, "
                           "not as the unit that does it, so no statement answers it: %s" % (i, served, clause))
            return
        other_place = _cues.elsewhere(clause)
        if other_place:
            missing.append("clause %d (responsibility ask): not answered; the clause asks about %s, whose units the handbook does not "
                           "document: %s" % (i, other_place, clause))
            return
        university_scope = bool(self._UNI_SCOPE.search(clause))
        if report["offices"]:                      # "Who in the X handles Y?": only the named office's own statements decide
            terms, verb, obj = self._asked_duty_and_verb(clause)
            office_id = self._one_office(report, missing, "responsibility", ignore_text=obj)
            if office_id is not None:
                self._office_duty(report, missing, summary, emit, office_id, terms, verb, university_scope)
            return
        terms, verb, _obj = self._asked_duty_and_verb(clause)
        report["duty_terms"] = list(terms)
        if not terms:
            missing.append("clause %d (responsibility ask): not answered; the clause names no duty to look up: %s" % (i, clause))
            return
        bound = [st for st in self.statements if st.relation == _cues.REL_RESPONSIBILITY and st.office_id not in (None, _cues.UNIVERSITY)
                 and self._matches(terms, st, verb) and self._scope_ok(st, university_scope)]
        university = [st for st in self.statements if st.office_id == _cues.UNIVERSITY and st.relation == _cues.REL_RESPONSIBILITY
                      and self._matches(terms, st, verb)]
        unbound = [st for st in self.statements if st.office_id is None and self._matches(terms, st, verb)]
        offices = list(dict.fromkeys(st.office_id for st in bound))
        unbound = [st for st in unbound if st.subject_office is None or st.subject_office not in offices]   # the same office, not another unit
        report["candidates"] = {"offices": offices, "university": len(university), "units_outside_lexicon": len(unbound)}
        duty = " ".join(terms)
        if not offices and not unbound and not university:
            narrower = self._collapse([st for st in self.statements if st.relation == _cues.REL_RESPONSIBILITY
                                       and st.office_id not in (None, _cues.UNIVERSITY) and self._scope_ok(st, university_scope)
                                       and any(_cues.phrase_narrows(terms, p) or (
                                           _cues.phrase_matches(terms, p, verb=verb) and self._title_restricted(st, p, terms)) for p in st.phrases)])
            if narrower and _B3_DUTY_ALL_TERMS:
                self._narrower(report, missing, summary, emit, narrower, terms)
                return
            missing.append("clause %d (responsibility ask): no statement read from the approved handbook names this duty (%s) for an "
                           "office; no office is named: %s" % (i, duty, clause))
            return
        bound = self._collapse(bound)
        if len(offices) == 1 and ((not unbound and not university) or not _B3_AMBIGUITY_CHECK):
            chosen = [st for st in bound if st.office_id == offices[0]][:3]
            emit([(st, terms) for st in chosen])
            name = _cues.OFFICES_BY_ID[offices[0]].name
            pages = ", ".join(str(p) for p in sorted({st.page for st in chosen}))
            # Any other statement whose words cover the duty (another unit, a unit outside the lexicon, the university)
            # keeps the office from being the only answer, even when its phrase is worded otherwise.
            continuing = {st.page for st in self.statements if st.office_id == offices[0] and st.list_continues}
            loose = [st.page for st in self.statements if st.office_id != offices[0] and st.subject_office != offices[0]
                     and not (st.binding == BIND_ORPHAN and st.page in continuing)       # the office's own list, beyond its passage
                     and _cues.covers_stems(terms, self._core_stems(st))] if _B3_AMBIGUITY_CHECK else []
            loose += [page for office_id, page, stems in self.outside_statements
                      if office_id != offices[0] and _cues.covers_stems(terms, stems)] if _B3_AMBIGUITY_CHECK else []
            report["candidates"]["also_worded"] = len(loose)
            if loose:
                others = ", ".join(str(p) for p in sorted(set(loose)))
                missing.append("clause %d (responsibility ask): other passages also use the words of this duty (%s) for another unit "
                               "(page %s); %s is not given as the only unit: %s" % (i, duty, others, name, clause))
                summary.append("%s: responsibility stated on page %s; other units are named with these words on page %s" % (name, pages, others))
            else:
                summary.append("%s: responsibility stated on page %s" % (name, pages))
            return
        if not _B3_AMBIGUITY_CHECK and offices:
            chosen = [st for st in bound if st.office_id == offices[0]][:3]
            emit([(st, terms) for st in chosen])
            summary.append("%s: responsibility stated on page %s" % (_cues.OFFICES_BY_ID[offices[0]].name, chosen[0].page))
            return
        chosen = []
        for office_id in offices:
            chosen.extend([st for st in bound if st.office_id == office_id][:2])
        chosen.extend(university[:1])
        chosen.extend(unbound[:3])
        emit([(st, terms) for st in chosen])
        names = [_cues.OFFICES_BY_ID[o].name for o in offices]
        others = [x for x in ("the University of Sharjah as a whole" if university else "",
                              "a unit outside the office lexicon or a list item without its unit" if unbound else "") if x]
        if offices:
            missing.append("clause %d (responsibility ask): the handbook names this duty (%s) for more than one unit (%s); no single "
                           "office is chosen: %s" % (i, duty, ", ".join(names + others), clause))
            summary.append("several units are stated to do this: %s" % ", ".join(names + others))
        else:
            missing.append("clause %d (responsibility ask): the statements that name this duty (%s) give it to %s, so no office is named "
                           "as the answer: %s" % (i, duty, " and ".join(others), clause))
            summary.append("the duty is stated for %s, not for an office of the lexicon" % " and ".join(others))

    def _narrower(self, report, missing, summary, emit, narrower: Sequence[Statement], terms: Sequence[str]) -> None:
        """No statement names the asked duty itself, but bound statements name narrower duties with all its words ("Health and
        Safety Training" for "health and safety"): their offices are quoted as candidates, never as the answer."""
        i, clause = report["index"], report["clause"]
        offices = list(dict.fromkeys(st.office_id for st in narrower))
        chosen = [st for o in offices for st in [x for x in narrower if x.office_id == o][:2]]
        emit([(st, terms) for st in chosen])
        names = ", ".join(_cues.OFFICES_BY_ID[o].name for o in offices)
        missing.append("clause %d (responsibility ask): no statement read from the approved handbook names this duty (%s) itself; %s %s "
                       "stated to do narrower duties named with these words, so no office is given as the answer: %s"
                       % (i, " ".join(terms), names, "is" if len(offices) == 1 else "are", clause))
        summary.append("narrower duties named with these words are stated for %s (page %s)" % (names, ", ".join(str(p) for p in sorted({st.page for st in chosen}))))

    def _office_duty(self, report, missing, summary, emit, office_id: str, terms: Sequence[str], verb: Optional[str] = None,
                     university_scope: bool = False) -> None:
        i, clause = report["index"], report["clause"]
        name = _cues.OFFICES_BY_ID[office_id].name
        report["duty_terms"] = list(terms)
        if not terms:
            missing.append("clause %d (responsibility ask): not answered; the clause names no duty of %s to look up: %s" % (i, name, clause))
            return
        own = [st for st in self.statements if st.office_id == office_id and st.relation == _cues.REL_RESPONSIBILITY]
        matching = self._collapse([st for st in own if self._matches(terms, st, verb) and self._scope_ok(st, university_scope)])
        narrower = self._collapse([st for st in own if self._scope_ok(st, university_scope)
                                   and any(_cues.phrase_narrows(terms, p, complements=True, verb=verb) for p in st.phrases)])
        if matching:
            emit([(st, terms) for st in matching[:3]])
            summary.append("%s: responsibility stated on page %s" % (name, ", ".join(str(p) for p in sorted({st.page for st in matching[:3]}))))
        elif narrower and _B3_DUTY_ALL_TERMS:
            self._narrower(report, missing, summary, emit, narrower, terms)
        else:
            missing.append("clause %d (office role ask): no statement read from the approved handbook names this duty (%s) for %s; "
                           "nothing else about %s is taken as evidence of it: %s" % (i, " ".join(terms), name, name, clause))

    def _answer_office_role(self, report, missing, summary, emit) -> None:
        i, clause = report["index"], report["clause"]
        terms, verb, obj = self._role_duty(clause)
        office_id = self._one_office(report, missing, "office role", ignore_text=obj)
        if office_id is None:
            return
        university_scope = bool(self._UNI_SCOPE.search(clause))
        if terms:
            self._office_duty(report, missing, summary, emit, office_id, terms, verb, university_scope)
            return
        name = _cues.OFFICES_BY_ID[office_id].name
        if _cues.ask_residue(obj) or self._YES_NO.match(clause):
            # "Does the FMPD maintain the centers?": a yes / no ask, or an object made only of frame words, names no duty to
            # look up, and the office's other statements do not answer it
            report["duty_terms"] = []
            missing.append("clause %d (office role ask): not answered; the clause names no duty of %s that can be looked up: %s" % (i, name, clause))
            return
        own = [st for st in self.statements if st.office_id == office_id and st.relation == _cues.REL_RESPONSIBILITY]
        if not own:
            missing.append("clause %d (office role ask): the handbook names %s, but no statement read under the B3 binding rules binds it "
                           "to a responsibility (E1 only): %s" % (i, name, clause))
            return
        own = self._collapse(own)
        shown = own[:_MAX_OFFICE_STATEMENTS]
        emit([(st, ()) for st in shown])
        pages = ", ".join(str(p) for p in sorted({st.page for st in shown}))
        summary.append("%s: %d responsibility statement%s, page %s" % (name, len(shown), "" if len(shown) == 1 else "s", pages))
        if len(own) > len(shown):
            missing.append("clause %d (office role ask): %d further statements about %s are not quoted: %s" % (i, len(own) - len(shown), name, clause))
        if any(st.list_continues for st in shown):
            missing.append("clause %d (office role ask): the list of %s's responsibilities continues beyond the quoted passage (page %s); "
                           "its later items are not bound to %s in one passage and are not given: %s" % (
                               i, name, ", ".join(sorted({str(st.page) for st in shown if st.list_continues})), name, clause))

    def _enumerated(self, st: Statement) -> bool:
        """Whether a composition statement lists its members: a list under its lead-in, "including" / "such as", or at
        least three capitalised items after the verb."""
        if st.binding == BIND_LIST:
            return True
        text = self.resources.chunks[st.index][st.key_start:st.key_end]
        r = _cues.read_unit(text)
        obj = text[r.object_start:] if r.object_start >= 0 else text
        if re.search(r"\bincluding\b|\bsuch\s+as\b", obj):
            return True
        parts = [p for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", obj) if p.strip()]
        return len([p for p in parts if re.match(r"^\s*(?:the\s+)?[A-Z]", p)]) >= 3

    #: Restricted members ("the non-voting members"): the composition statements never say which members these are.
    _MEMBER_QUALIFIER = re.compile(r"\b(?:non-?voting|voting|elected|appointed|permanent|ex[\s-]?officio|invited|external|internal|honorary|"
                                   r"current|former|student|faculty|senior|regular|core|full|alternate|observer|nominated|temporary|new|"
                                   r"academic|administrative)\s+members?\b|\belected\b|\bappointed\b|\bmembers?\b[^?]*\b(?:can|may|could)\s+vote\b|"
                                   r"\bvot(?:e|es|ing)\b|\bex[\s-]?officio\b|\bby\s+invitation\b|\binvited\b|"
                                   r"\b(?:which|what)\s+(?:of\s+the\s+)?(?:[\w'’-]+\s+){0,4}?members?\b", re.I)
    _CHAIR_OTHER = re.compile(r"\b(?:vice|co|deputy|acting|interim|assistant|associate)[\s-]*chair\w*|\btemporar\w*|\bacting\s+capacity\b|"
                              r"\bin\s+the\s+absence\b|\bsubstitut\w*|\bstand(?:s|-)?\s*in\b", re.I)
    _PART_UNIT = re.compile(r"\b(?:each|every)\s+(?:sub-?group|cent(?:er|re)|section|department|division|unit|sub-?unit|college)\b|"
                            r"\b(?:sub-?groups?|cent(?:er|re)s?|sections?|departments?|divisions?|units?)(?:'s|s'|')?\s+(?:members?|responsibilit\w*|"
                            r"duties|tasks|roles?|consist\w*|compos\w*|made)\b|\bmembers?\s+of\s+(?:the\s+|each\s+)?[^?]{0,40}?\b(?:sub-?groups?|"
                            r"cent(?:er|re)s?|sections?|departments?|divisions?|units?)\b|\b(?:sub-?group|cent(?:er|re)|section|department|division|unit)\s+"
                            r"of\s+the\b", re.I)
    #: The kind of part a composition ask asks for ("What centers does the Language Institute comprise?").
    _PART_ASKED = re.compile(r"\b(?:cent(?:er|re)s|sections|departments|divisions|programs|programmes|units|sub-?groups|sub-?units|offices|labs|"
                             r"laboratories|schools|libraries|committees|colleges|faculties|institutes|branches|campuses)\b", re.I)

    def _answer_structure(self, report, missing, summary, emit, intent: str) -> None:
        i, clause = report["index"], report["clause"]
        office_id = self._one_office(report, missing, intent, allow_parts=(intent == "composition"))
        if office_id is None:
            return
        name = _cues.OFFICES_BY_ID[office_id].name
        relation = _cues.REL_CHAIR if intent == "chair" else _cues.REL_COMPOSITION
        what = "chair" if intent == "chair" else "composition"
        own = self._collapse([st for st in self.statements if st.office_id == office_id and st.relation == relation])
        if not own:
            missing.append("clause %d (structure ask): no statement read under the B3 binding rules gives the %s of %s: %s" % (i, what, name, clause))
            return
        shown = own[:4]
        masked = _cues.without_offices(clause)
        if intent == "chair" and self._CHAIR_OTHER.search(clause):
            missing.append("clause %d (structure ask): not answered; the handbook prints who chairs %s, and no vice, co-, acting or "
                           "stand-in chair: %s" % (i, name, clause))
            return
        if intent == "composition":
            if re.search(r"\bmembers?(?:hip)?\b", masked, re.I) and not any(re.search(r"\bmembers?\b", self.statement_quote(st), re.I) for st in shown):
                missing.append("clause %d (structure ask): not answered; no statement read gives the members of %s (its composition "
                               "statement lists parts, not members): %s" % (i, name, clause))
                return
            if self._PART_UNIT.search(masked):
                missing.append("clause %d (structure ask): not answered; the clause asks about the members or make-up of a part of %s, "
                               "and no statement read gives them: %s" % (i, name, clause))
                return
            part = self._PART_ASKED.search(masked)
            if part:
                stem = re.escape(_cues._stem(part.group(0)))
                listed = (r"\b(?:\d+|%s|several|various|range\s+of|following)\s+(?:[\w-]+\s+){0,2}%s\w*|"
                          r"\b(?:consists\s+of|composed\s+of|comprised\s+of|comprises|encompasses|includes|operates\s+through)\s+(?:[\w-]+\s+){0,4}?%s\w*"
                          % (self._NUMBER_WORDS, stem, stem))
                if not any(re.search(listed, self.statement_quote(st), re.I) for st in shown):
                    missing.append("clause %d (structure ask): not answered; no statement read lists %s of %s: %s"
                                   % (i, part.group(0).lower(), name, clause))
                    return
        emit([(st, ()) for st in shown])
        summary.append("%s: %s stated on page %s" % (name, what, ", ".join(str(p) for p in sorted({st.page for st in shown}))))
        if intent == "composition":
            qualifier = self._MEMBER_QUALIFIER.search(masked)
            if qualifier and not any(qualifier.group(0).lower() in self.statement_quote(st).lower() for st in shown):
                missing.append("clause %d (structure ask): the statement gives the composition of %s but does not say which members are %s: %s"
                               % (i, name, qualifier.group(0).lower(), clause))
            if not any(self._enumerated(st) for st in shown):
                missing.append("clause %d (structure ask): the statement names the composition of %s but does not list its members or parts: %s"
                               % (i, name, clause))
            if _C1_COMPOSITION_CONTINUES and any(self._MORE_MEMBERS.match(self.resources.chunks[st.index][st.quote_end:]) for st in shown):
                missing.append("clause %d (structure ask): the handbook adds more members of %s after the quoted sentences; they are not "
                               "given: %s" % (i, name, clause))
        if any(st.list_continues for st in shown):
            missing.append("clause %d (structure ask): the list continues beyond the quoted passage; its later items are not given: %s" % (i, clause))

    #: Step 4.16-C1: the next sentence of the passage adds members ("Additionally, seven students are elected ...", "... also
    #: participate in the Council")
    _MORE_MEMBERS = re.compile(r"\s*(?:(?:Additionally|In\s+addition|Also|Furthermore|Moreover|Besides)\b|[^.!?]{0,200}\balso\s+"
                               r"(?:participates?|serves?|sits?|attends?|includes?|joins?)\b)", re.I)

    _OTHER_BODY = re.compile(r"\b(?:colleges?|departments?|councils?|deanships?|offices?|cent(?:er|re)s?|units?|committees?|boards?|schools?|"
                             r"branch(?:es)?|campus(?:es)?|faculty|faculties|sections?|divisions?|institutes?|libraries|library)\b", re.I)

    #: The words of a bare structure ask; any other word asks for more than the composition statements state (a rank, a
    #: time, a person, a part, a yes / no fact ...).
    _ORG_FRAME = frozenset("""what is are the a an of describe explain show tell me about give outline please can could you current overall
university universitys uos sharjah organization organisation organizational organisational structure structures chart governance bodies how
organized organised structured governed its""".split())

    def _answer_organisation(self, report, missing, summary, emit, limitations) -> None:
        i, clause = report["index"], report["clause"]
        masked = _cues.without_offices(clause)
        other = _cues.elsewhere(clause) or next((w for w in _cues.other_unit_mentions(clause) if w not in ("body", "bodies")), None)
        body = self._OTHER_BODY.search(masked)
        if other or body:
            missing.append("clause %d (structure ask): not answered; the clause asks about the structure of %s, and only the University of "
                           "Sharjah's own composition statements are read: %s" % (i, other or body.group(0), clause))
            return
        rest = [w for w in (re.sub(r"['’]s$", "", w.lower()) for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*", masked)) if w not in self._ORG_FRAME]
        if rest:
            missing.append("clause %d (structure ask): not answered; the clause asks more about the structure (%s) than the composition "
                           "statements state, and nothing is inferred from them or from the chart: %s" % (i, " ".join(rest), clause))
            return
        own = [st for st in self.statements if st.office_id == _cues.UNIVERSITY and st.relation == _cues.REL_COMPOSITION]
        if not own:
            missing.append("clause %d (structure ask): no statement read under the B3 binding rules gives the university's composition: %s" % (i, clause))
            return
        emit([(st, ()) for st in own[:3]])
        summary.append("the university's composition is stated on page %s" % ", ".join(str(p) for p in sorted({st.page for st in own[:3]})))
        if _CHART_LIMITATION not in limitations:
            limitations.append(_CHART_LIMITATION)
        if re.search(r"\bgovernance\b|\bgoverned\b", clause, re.I):
            missing.append("clause %d (structure ask): no single statement lists the university's governance bodies; sections 1.4 to 1.8 "
                           "describe them one by one, and they are not assembled here: %s" % (i, clause))

    def _answer_colleges(self, report, missing, summary, emit) -> None:
        i, clause = report["index"], report["clause"]
        lists = list(self.enumerations)
        lists += [st for st in self.statements if st.office_id == _cues.UNIVERSITY and re.search(r"\bcolleges\b", self.statement_quote(st))
                  and st.relation == _cues.REL_COMPOSITION]
        lists = list(dict.fromkeys(lists))
        if not lists:
            missing.append("clause %d (structure ask): no statement read from the approved handbook lists the colleges: %s" % (i, clause))
            return
        emit([(st, ()) for st in lists[:5]])
        summary.append("the colleges are listed on page %s by the Vice Chancellor they report to" % ", ".join(str(p) for p in sorted({st.page for st in lists})))
        missing.append("clause %d (structure ask): the colleges are listed across several statements, grouped by the Vice Chancellor "
                       "they report to, and one list also names units that are not colleges; no single list is assembled: %s" % (i, clause))

    #: Printed words before a counted noun that make it another office ("Vice", "Associate", "Assistant" ...).
    _OFFICE_MODIFIERS = frozenset("vice associate assistant deputy acting sub co executive honorary former".split())

    def _answer_count(self, report, missing, summary, emit) -> None:
        i, clause = report["index"], report["clause"]
        m = re.search(r"\bhow\s+many\s+(?P<noun>[A-Za-z][A-Za-z\s\-’']{0,40}?)\s+(?P<verb>does|do|did|are|is|exist|has|have|offered|can|will)\b"
                      r"(?P<rest>[^?]*)", clause, re.I)
        if not m:
            missing.append("clause %d (count ask): not answered; the counted thing could not be read from the clause: %s" % (i, clause))
            return
        if report["offices"]:
            missing.append("clause %d (count ask): not answered; in phase %s counts are answered for the university as a whole only: %s"
                           % (i, PHASE, clause))
            return
        rest = re.sub(r"\b(?:the\s+university\s+of\s+sharjah|the\s+university|uos)\b", " ", m.group("rest").lower())
        rest = re.sub(r"\s+", " ", rest).strip()
        if rest not in ("", "have", "has", "offer", "manage", "there", "exist", "operate", "include", "comprise", "in", "at", "there in", "there at"):
            missing.append("clause %d (count ask): not answered; the count asked is limited to a part, place or relation of the university "
                           "(%s), and the handbook's counts are stated for the university as a whole only: %s" % (i, m.group("rest").strip(), clause))
            return
        np_words = [w for w in re.findall(r"[A-Za-z’'\-]+", m.group("noun")) if w.lower() not in ("the", "of", "a", "an")]
        head = np_words[-1]
        number = r"(?P<num>\d[\d,]*|(?:%s))" % self._NUMBER_WORDS
        verb = r"(?:is\s+composed\s+of|comprises|manages\s+(?:all\s+functions\s+of\s+)?|offers?\s+(?:a\s+total\s+of\s+)?|has|have|are|is)?\s*"
        found = []
        for st in self.statements:
            if st.office_id != _cues.UNIVERSITY:
                continue
            text = self.statement_quote(st)
            r = _cues.read_unit(text)
            rest_text = text[r.core_start:]
            lead = re.match(r"(?:(?:the|The)\s+)?(?:(?:organi[sz]ation|governance\s+structures?)\s+of\s+the\s+)?(?:University(?:\s+of\s+Sharjah)?|UoS)\s+|"
                            r"(?:we|We|There\s+(?:are|is))\s+", rest_text)
            if not lead:
                continue
            cm = re.match(verb + number + r"\b", rest_text[lead.end():], re.I)
            if not cm:
                continue
            following = re.findall(r"[A-Za-z’'\-]+", rest_text[lead.end() + cm.end():])[:4]
            stems = [_cues._stem(w) for w in following]
            if _cues._stem(head) not in stems:
                continue
            np_end = len(following)
            for x, w in enumerate(following):
                if w.lower() in _cues._STOP or w.lower() in _cues._PREPOSITIONS or w.lower().endswith("ing") or w.lower().endswith("ed"):
                    np_end = x
                    break
            if np_end == 0 or _cues._stem(following[np_end - 1]) != _cues._stem(head):
                continue                     # the printed noun phrase ("academic colleges") has another head than the asked noun
            k = np_end - 1
            counted = following[: k + 1]       # the counted phrase as printed, up to its head noun
            asked = {w.lower() for w in np_words}
            if any(c.lower() in self._OFFICE_MODIFIERS and c.lower() not in asked for c in counted):
                continue                     # "eight Vice Chancellors" counts no chancellors
            if all(_cues._stem(w) in [_cues._stem(c) for c in counted] for w in np_words):
                found.append((st, cm.group("num"), " ".join(counted)))
        if not found:
            missing.append("clause %d (count ask): no statement read from the approved handbook gives this count for the university: %s" % (i, clause))
            return
        emit([(st, ()) for st, _n, _c in found[:3]])
        numbers = list(dict.fromkeys(n.lower() for _st, n, _c in found))
        summary.append("the handbook states %s (page %s)" % ("; ".join("%s %s" % (n, c) for _st, n, c in found[:3]),
                                                            ", ".join(str(st.page) for st, _n, _c in found[:3])))
        if len(numbers) > 1:
            missing.append("clause %d (count ask): the handbook states different numbers (%s); none is chosen: %s" % (i, ", ".join(numbers), clause))

    _BRANCH = re.compile(r"\b(?:khorfakkan|kalba|al\s+dhaid|dhaid|saasst|branch(?:es)?)\b", re.I)
    _NEGATION = re.compile(r"\b(?:not|never|no|non|other\s+than|except|excluding|excluded|exclude|instead\s+of|rather\s+than|besides|apart\s+from|"
                           r"without|unless|but|outside|away\s+from|far\s+from|off|beyond|unlike|opposite|other\s+one|another)\b|"
                           r"\b(?:is|are|was|were|do|does|did|has|have|had|ca|can|could|wo|would|should|must|need)n'?t\b|\bnon-", re.I)
    _GENDER_WORD = re.compile(r"\b(?:women|woman|female|ladies|lady|girls?|men|man|male|boys?)(?:'s|s'|s)?(?![\w-])", re.I)
    _WOMEN = re.compile(r"(?:women|woman|female|ladies|lady|girl)", re.I)
    _G = r"(?:women|woman|female|ladies|lady|girls|men|man|male|boys)"
    #: One selector printed right after the library's name (in a row): "for (the) women", "for men and women", "on the Men
    #: Campus", "located at the X College", "(W16)", "M-16". The gender frame ends the noun phrase, so "for women's
    #: studies", "for female authors" pick nothing.
    _SELECTOR = re.compile(r"\s*,?\s*(?:(?:which|that)\s+is\s+|(?:located|situated|found)\s+)?(?:"
                           r"for\s+(?:the\s+)?(?P<g>%s)(?:'s|s)?(?:\s+(?:and|or|&)\s+(?:the\s+)?(?P<g2>%s)(?:'s|s)?)?"
                           r"(?:\s+(?:students|faculty|staff|members|section|campus|only))?(?=\s*(?:$|[?.,!;]|located\b|situated\b|on\b|at\b|in\b))|"
                           r"(?:on|at|in|of|inside|within)\s+(?:the\s+)?(?P<place>(?:(?!(?:on|at|in|of|for|near|inside|within|by)\b)[\w'’&-]+\s+){0,6}?"
                           r"(?:campus|college|colleges|complex|academy|hospital|building|branch|area|zone|centre|center|city|site|university))\b|"
                           r"\(?(?P<code>[A-Za-z]-?\d{1,3})\)?(?![\w-]))" % (_G, _G), re.I)
    _PREFIX_SELECTOR = re.compile(r"(?:(?P<g>women|men|ladies|girls|boys)(?:'s|s'|s)?|(?P<code>[A-Za-z]-?\d{1,3}))\s+(?:main\s+)?$", re.I)
    #: What may follow the library's name and its selectors: nothing else is read, so anything else fails closed.
    _TAIL_OK = re.compile(r"^\s*,?\s*(?:(?:located|situated|found|is)\s*)?(?:(?:on|at|in)\s+(?:the\s+)?(?:campus|university|uos)\s*)?"
                          r"(?:for\s+(?:the\s+)?(?:faculty|staff|students|members|everyone|visitors)(?:\s+members)?\s*)?(?:on|in|at)?\s*"
                          r"(?:exactly|now|please)?\s*,?\s*(?:please|thanks|thank\s+you)?\s*[?.!]*\s*$", re.I)
    _CODE_TOKEN = re.compile(r"(?<![\w'’])([A-Za-z])-?(\d{1,3})(?![\w-])")
    #: A place or body the question puts the library at or in ("on the Women Campus", "at the Fine Arts and Design
    #: College", "of the Engineering College", "Faculty of Medicine"): it must be the entry's printed place.
    _PLACE_QUALIFIER = re.compile(r"\b(?:on|at|in|of|for|near|inside|within|by)\s+(?:the\s+)?(?P<q>(?:(?!(?:on|at|in|of|for|near|inside|within|by|which|what)\b)"
                                  r"[\w'’&-]+\s+){0,6}?(?:campus|campuses|college|colleges|complex|academy|hospital|building|branch|area|zone|"
                                  r"centre|center|city|site))\b|\b(?P<f>faculty\s+of\s+[\w'’&-]+(?:\s+[\w'’&-]+)?)", re.I)
    _PLACE_STOP = frozenset("the a an of and & university universitys uos sharjah main its their".split())
    _GENDER_MAP = {"womens": "women", "woman": "women", "female": "women", "ladies": "women", "lady": "women", "girls": "women", "girl": "women",
                   "mens": "men", "man": "men", "male": "men", "boys": "men", "boy": "men"}

    def _place_words(self, phrase: str) -> List[str]:
        words = []
        for w in re.findall(r"[\w'’&-]+", phrase.lower()):
            w = re.sub(r"['’]s$|s['’]$", "", w.replace("’", "'"))
            w = self._GENDER_MAP.get(w, w)
            if w and w not in self._PLACE_STOP:
                words.append(w)
        return words

    def _place_hits(self, entries: Sequence[Placement], phrase: str) -> Optional[List[Placement]]:
        """The entries whose printed place holds every word of a place phrase; None when the phrase names no place."""
        words = self._place_words(phrase)
        if not words or (len(words) == 1 and words[0] in ("campus", "campuses")):
            return None                      # "on campus", "of the university" name no place
        return [pl for pl in entries if all(re.search(r"\b%s(?![a-z])" % re.escape(w), (pl.place or "").lower()) for w in words)]

    def _answer_location(self, report, missing, summary, emit) -> None:
        i, clause = report["index"], report["clause"]
        office_id = self._one_office(report, missing, "location", allow_men_women=True)
        if office_id is None:
            return
        name = _cues.OFFICES_BY_ID[office_id].name
        norm = _cues.normalise_text(clause)
        masked = _cues.without_offices(clause)
        facet = self._FACET_ASK.search(masked)
        if facet:
            missing.append("clause %d (location ask): not answered; the approved handbook states no %s for %s, and none is inferred: %s"
                           % (i, facet.group(0).lower(), name, clause))
            return
        entries = [pl for pl in self.placements if pl.office_id == office_id]
        if not entries:
            missing.append("clause %d (location ask): no documented location entry for %s is read from the approved handbook: %s" % (i, name, clause))
            return
        branch = self._BRANCH.search(clause) or (re.search(re.escape(_cues.elsewhere(clause)), clause) if _cues.elsewhere(clause) else None)
        if branch:
            missing.append("clause %d (location ask): not answered; the handbook documents no location of %s at %s: %s" % (i, name, branch.group(0), clause))
            return
        known = {pl.code for pl in entries if pl.code}
        codes = {(m.group(1) + m.group(2)).upper() for m in self._CODE_TOKEN.finditer(masked)}
        if codes - known:
            missing.append("clause %d (location ask): no documented entry of %s carries the building code %s: %s"
                           % (i, name, ", ".join(sorted(codes - known)), clause))
            return
        span = next(((a, b) for a, b, o in _cues.office_spans(norm) if o == office_id), (0, 0))
        # the selectors printed right after the name, in a row, and one right before it ("the women's Main Library")
        genders, places, sel_codes, anchored = set(), [], set(), []
        pos = span[1]
        while True:
            m = self._SELECTOR.match(norm, pos)
            if not m or m.end() == pos:
                break
            for g in ("g", "g2"):
                if m.group(g):
                    genders.add("women" if self._WOMEN.match(m.group(g)) else "men")
            if m.group("place"):
                places.append(m.group("place"))
            if m.group("code"):
                sel_codes.add(re.sub(r"-", "", m.group("code")).upper())
            anchored.append((m.start(), m.end()))
            pos = m.end()
        tail = norm[pos:]
        pre = self._PREFIX_SELECTOR.search(norm[:span[0]])
        if pre:
            anchored.append((pre.start(), pre.end()))
            if pre.group("g"):
                genders.add("women" if self._WOMEN.match(pre.group("g")) else "men")
            if pre.group("code"):
                sel_codes.add(re.sub(r"-", "", pre.group("code")).upper())

        def inside(k: int) -> bool:
            return any(a <= k < b for a, b in anchored)

        unanchored = [g.group(0) for g in self._GENDER_WORD.finditer(norm) if not inside(g.start()) and not (span[0] <= g.start() < span[1])]
        unanchored += [m.group(0) for m in self._CODE_TOKEN.finditer(norm) if not inside(m.start())]
        # a place put before the name ("At the Fine Arts and Design College, where is ...") must still be the entry's place
        for m in self._PLACE_QUALIFIER.finditer(_cues.without_offices(norm[:span[0]])):
            places.append(m.group("q") or m.group("f") or "")
        by_place = None
        for phrase in places:
            hits = self._place_hits(entries, phrase)
            if hits is None:
                continue
            if not hits:
                missing.append("clause %d (location ask): no documented entry of %s is at or of %s: %s" % (i, name, phrase.strip(), clause))
                return
            by_place = hits if by_place is None else [pl for pl in by_place if pl in hits]
        notes: List[str] = []
        chosen = list(entries)
        negated = bool(self._NEGATION.search(norm))
        if negated or unanchored:
            missing.append("clause %d (location ask): not answered; the clause excludes something, or uses a word for women, men or a "
                           "building code that does not belong to the library it names, so no entry is given: %s" % (i, clause))
            return
        if len(genders) > 1:
            notes.append("the question names both entries, so neither is chosen")
        elif not self._TAIL_OK.match(tail):
            missing.append("clause %d (location ask): not answered; the clause adds to the location ask (%s), and only the printed entries "
                           "are given: %s" % (i, tail.strip(" ,?.!"), clause))
            return
        else:
            by_code = [pl for pl in entries if pl.code in sel_codes] if sel_codes else None
            gender = next(iter(genders)) if genders else None
            by_gender = [pl for pl in entries if re.search(r"\b%s\b" % gender, self._placement_text(pl).lower())] if gender else None
            picks = [p for p in (by_code, by_place, by_gender) if p is not None]
            if picks:
                common = [pl for pl in entries if all(pl in p for p in picks)]
                if common:
                    chosen = common
                elif by_gender is not None and not by_gender and len(picks) == 1:
                    notes.append("the entry does not say it is for %s" % gender)
                else:
                    notes.append("the question's building code, campus or section do not point to one entry")
        if len(chosen) > 1 and not _B3_SAME_NAME_CHECK:
            chosen = chosen[:1]
        emit(chosen)
        summary.append("; ".join("%s: %s (page %d)" % (pl.label, pl.place, pl.page) for pl in chosen))
        if len(chosen) > 1:
            missing.append("clause %d (location ask): the handbook prints %d entries named %s (%s); the question does not pick one, so both "
                           "are given: %s" % (i, len(chosen), name, ", ".join(pl.label for pl in chosen), clause))
        for note in notes:
            missing.append("clause %d (location ask): %s: %s" % (i, note, clause))
        low = norm.lower()
        if re.search(r"\b(?:which|what)\s+campus\b", low) and any("campus" not in (pl.place or "").lower() for pl in chosen):
            missing.append("clause %d (location ask): no campus is printed for %s; the entry gives the place as printed: %s" % (i, name, clause))
        if re.search(r"\b(?:which|what)\s+building\b|\bbuilding\s+code\b", low) and any(not pl.code for pl in chosen):
            missing.append("clause %d (location ask): no building code is printed for %s; the entry gives the place as printed: %s" % (i, name, clause))

    def _leave_to_owner(self, task: SpecialistTask, report: Dict[str, Any], boundary: Dict[str, Any], handoffs: List[HandoffRequest],
                        missing: List[str]) -> None:
        """A clause whose subject another specialist owns: no finding; a handoff to each owner the cues name (never back
        to the requester), or a missing entry when no handoff is made."""
        i, clause, kind = report["index"], report["clause"], report["ask_kind"]
        owners = boundary["owners"]
        report["owner"] = [o.value for o in owners]
        report["ownership_cues"] = list(boundary["cues"])
        for n, target in enumerate(boundary["handoff"]):
            handoffs.append(self._handoff(task, target, clause, i, n))
        report["handoff_to"] = [t.value for t in boundary["handoff"]]
        labels = ", ".join(self._OWNER_LABEL[o] for o in owners)
        if boundary["kind"] == "teaching":
            report["outcome"] = "left_to_teaching"
            missing.append("clause %d (%s ask): not answered; its subject belongs to Teaching & Learning, so Institutional Navigation "
                           "does not answer it, and no handoff is created (the shared ownership cues name no Teaching target): %s"
                           % (i, kind, clause))
        elif boundary["kind"] == "ambiguous":
            report["outcome"] = "ownership_ambiguous"
            missing.append("clause %d (%s ask): not answered; its words point to more than one specialist (%s), so ownership is "
                           "ambiguous and no handoff is created: %s" % (i, kind, labels, clause))
        elif boundary["kind"] == "unconfirmed":
            report["outcome"] = "left_unconfirmed"
            missing.append("clause %d (%s ask): not answered; its subject may belong to %s (Institutional Navigation's own word "
                           "list), but the shared ownership cues do not claim it, so no handoff is created: %s" % (i, kind, labels, clause))
        elif boundary["handoff"]:
            report["outcome"] = "handed_off"
        elif boundary["handed_back"]:
            report["outcome"] = "not_handed_back"
            missing.append("clause %d (%s ask): not answered; its subject belongs to %s, which handed this clause over, so it is not "
                           "handed back, and Institutional Navigation does not answer another specialist's subject (no statement is "
                           "searched for it): %s" % (i, kind, labels, clause))
        else:
            report["outcome"] = "not_answered"
            missing.append("clause %d (%s ask): not answered; its subject belongs to %s, so Institutional Navigation does not answer "
                           "it: %s" % (i, kind, labels, clause))

    # ---- the run ------------------------------------------------------------
    def run(self, task: SpecialistTask) -> SpecialistFindings:
        started = time.perf_counter()
        self._check_task(task)
        problem = self._source_scope_problem(task)
        clauses = self._focus_clauses(task)
        reports = [_cues.clause_report(i, clause) for i, clause in enumerate(clauses)]
        findings: List[Finding] = []
        missing: List[str] = []
        summary: List[str] = []
        shared: List[str] = []
        extra_limitations: List[str] = []
        handoffs: List[HandoffRequest] = []
        owned = 0
        if problem is not None:
            missing.append(problem)
        else:
            for report in reports:
                # One focus clause is one ask, read as a whole, as in phase B2: every check reads every word of it. A
                # clause that holds two asks is answered by its first-priority ask kind (fail closed; never split here).
                boundary = None if report["ask_kind"] == _cues.CHANNEL else self._boundary(task, report["clause"], report["ask_kind"])
                if boundary is not None:
                    self._leave_to_owner(task, report, boundary, handoffs, missing)
                    continue
                owned += 1
                report["outcome"] = "read"
                work = report
                if _C1_NATURAL_ASK:
                    ask, notes = _cues.natural_ask(report["clause"], guard=_C1_CONTEXT_GUARD)
                    if notes:
                        # the ownership check above read the whole clause, context included; the answer reads the plain ask.
                        # A plain ask of another kind is checked again for that kind: a contact clause skips the ownership
                        # check, so a rewrite into a responsibility ask would otherwise answer another specialist's subject
                        # unchecked (the clause is then read as written)
                        plain = _cues.clause_report(report["index"], ask)
                        if (plain["ask_kind"] == report["ask_kind"] or not _C1_KIND_CHANGE_CHECK
                                or self._boundary(task, report["clause"], plain["ask_kind"]) is None):
                            work = plain
                            report["read_as"], report["read_notes"] = ask, list(notes)
                if work["ask_kind"] == _cues.CHANNEL:
                    self._answer_channel(task, work, findings, missing, summary, shared)
                else:
                    self._answer_b3(task, work, findings, missing, summary, extra_limitations)
                if work is not report:
                    report.update({k: v for k, v in work.items() if k not in ("index", "clause")})
        if problem is None and owned == 0:
            status = FindingStatus.OUT_OF_SCOPE
        elif not findings:
            status = FindingStatus.NOT_FOUND
        elif missing:
            status = FindingStatus.PARTIAL
        else:
            status = FindingStatus.SUPPORTED
        limitations = [_PHASE_LIMITATION]
        if any(f.metadata.get("directory_entry") in ("complete_row", "labelled_row") for f in findings):
            limitations.append(_AS_PRINTED_LIMITATION)
        limitations.extend(shared)
        limitations.extend(extra_limitations)
        outcomes = [r.get("outcome") for r in reports]
        if outcomes.count("handed_off"):
            limitations.append(_HANDED_OFF_LIMITATION % outcomes.count("handed_off"))
        if outcomes.count("left_to_teaching"):
            limitations.append(_TEACHING_LIMITATION % outcomes.count("left_to_teaching"))
        if outcomes.count("not_handed_back"):
            limitations.append(_HAND_BACK_LIMITATION % outcomes.count("not_handed_back"))
        unsettled = outcomes.count("ownership_ambiguous") + outcomes.count("left_unconfirmed")
        if unsettled:
            limitations.append(_UNSETTLED_LIMITATION % unsettled)
        if status is FindingStatus.OUT_OF_SCOPE:
            text = ("Institutional Navigation (phase %s): no assigned clause is Institutional Navigation's%s" %
                    (PHASE, "; handed off to %s" % ", ".join(sorted({h.requested_specialist.value for h in handoffs})) if handoffs else ""))
        elif summary:
            text = "Institutional Navigation (phase %s): %s" % (PHASE, "; ".join(summary))
        elif problem is None and any(r["ask_kind"] == _cues.CHANNEL for r in reports):
            text = "Institutional Navigation (phase %s): the contact directory gives no value for the ask (see missing)" % PHASE
        elif problem is None and any(r.get("b3_intent") not in (None, "other") for r in reports):
            text = "Institutional Navigation (phase %s): the approved handbook gives no usable evidence for the ask (see missing)" % PHASE
        else:
            text = "Institutional Navigation (phase %s): the ask was read; it is not answered in this phase" % PHASE
        return SpecialistFindings(
            specialist_id=self.specialist_id,
            task_id=task.task_id,
            status=status,
            findings=findings,
            summary=text,
            missing=missing,
            handoff_requests=handoffs,
            confidence={FindingStatus.SUPPORTED: 1.0, FindingStatus.PARTIAL: 0.5}.get(status, 0.0),
            limitations=limitations,
            llm_used=False,
            ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "phase": PHASE,
                "source_id": APPROVED_SOURCE_ID,
                "scope_page_ranges": [list(r) for r in (self.scope.page_ranges or ())],
                "clauses": reports,
                "directory": {"pages": sorted(self.directory), "lines_read": len(self.directory_rows()),
                              "header_columns": {str(p): list(c) if c else None for p, c in self.header_columns.items()},
                              "binding_protections": binding_protections()},
                "statements": {"read": len(self.statements), "location_entries": len(self.placements),
                               "evidence_binding_protections": b3_protections()},
                "boundary": {"requested_by": task.requested_by, "owned_clauses": owned, "handoffs": len(handoffs),
                             "boundary_protections": b4_protections()},
                "natural_phrasing": {"read_as": sum(1 for r in reports if r.get("read_as")), "protections": c1_protections()},
                "confidence_semantics": "ordinal by status: 1.0 supported, 0.5 partial, 0.0 otherwise; not a probability",
            },
        )


__all__ = ["ABSENT", "APPROVED_SOURCE_ID", "BOUND", "DIRECTORY_SECTION", "DirectoryLookup", "DirectoryRow", "EXCLUDED_SECTIONS", "EXTRA_PAGES",
           "InstitutionalNavigationSpecialist", "InstitutionalResources", "LABELLED", "OWNED_SECTIONS", "PHASE", "Placement", "SCOPE_SECTIONS",
           "SPLIT", "Statement", "UNBOUND", "b3_protections", "b4_protections", "binding_protections", "c1_protections"]
