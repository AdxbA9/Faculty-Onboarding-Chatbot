"""
Institutional Navigation cues, evidence tiers, office lexicon and directory-row readers (Plan F, Milestone 4, Step 4.16,
D-104; directory readers added in phase B2).

Standard library only, deterministic, imported by the Institutional Navigation specialist. Nothing here reads the
index or answers anything; it holds data and pure readers of text:

    classify_ask(clause)       the navigation ask kind of one focus clause
    resolve_offices(text)      the lexicon offices a text names (full printed name or a handbook-attested alias)
    office_spans(text)         the same matches with their character spans
    tier_satisfies(reached, required)
                               whether evidence of one tier is enough for an ask that needs another
    contact_fields(clause)     the contact fields a channel ask asks for (telephone, fax, extension, e-mail, website)
    referent_problem(clause)   why a clause's office mention may not be the unit asked about (a library, a clinic, a
                               person, an office of another office), or None
    row_office(text)           the lexicon office a contact-directory line starts with
    read_row_values(rest)      what a directory line holds after the office name (two numbers, one, labelled values ...)
    header_columns(text)       the column order a directory header line prints
    continues_into_next(text), continues_from_previous(text)
                               whether a directory line is part of a multi-line entry

EVIDENCE TIERS
    E1  identification   the handbook names the office (a directory row, a heading, a mention)
    E2  responsibility   one statement binds the office to the asked function ("The Health and Safety Office is
                         responsible for ...")
    E3  procedure        the handbook states the steps of the asked process
    A mention never proves a responsibility, and a responsibility never proves a procedure. An E2 statement names its
    office, so it also identifies it (E2 satisfies E1); nothing else satisfies a tier it is not.

OFFICE LEXICON
    Every entry is printed verbatim on the pages it lists in the Faculty Handbook 2025/26 (the tests read the PDF to
    check it). An alias is kept only when the handbook itself prints it for the same unit ("Human Resources (HR)",
    "Information Technology Center (ITC)"). Different printed names stay different entries unless the handbook
    states they are one unit: "Physical Plant" and the "Facilities Management and Planning Department" are two
    entries, and so are the "Business Service Department" (section 15.2) and the "Business Services and Purchases
    Department" (the directory). A one-word generic name ("Registration") is matched in a question only beside an
    office noun, so "registration of a complaint" names no office. An office the lexicon does not contain is never
    named by the specialist. Names the handbook does not print ("IT Center", "IT Department") are not aliases.

CONTACT DIRECTORY LINES (phase B2)
    The directory (section 1.16) is a three-column table: office, telephone number, fax number. The index keeps each
    extracted line as one ``row`` chunk. A line can be bound to an office only when it starts with the office's
    printed name (or attested alias) followed by a value or the end of the line; what follows the name is read here
    and classified, never repaired: two complete numbers, one number, labelled values ("(External)", "Ext ...
    (Internal)"), no number, a labelled sub-entry, or a fragment of a multi-line entry. Numbers are kept exactly as
    printed; nothing is joined, completed or reformatted. Only the entries marked ``contact`` take part: the units
    added in phase B3 for the chapters (library staff, two library names) never match a directory line, so the
    contact directory's coverage is unchanged.

STATEMENT READING (phase B3)
    split_units(text)          the sentences, list items, lead-ins and entry titles of a paragraph chunk, with spans
    read_unit(text, unit)      what one unit states: its subject (a lexicon office, an anaphor such as "This Council"
                               or "The office", the university, or a unit outside the lexicon) and the relation that
                               follows it (a responsibility, a chair, a composition)
    duty_terms(text), covers(terms, text)
                               the content words of an asked duty, and whether a statement states every one of them
    location_entry(text)       a library-style location entry ("Main Library (W16): ... on the University's Women
                               Campus (W16).") with its printed name, building code and place
    The readers only recognise; the specialist decides what a recognised statement may support.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Evidence tiers
# ---------------------------------------------------------------------------
E1 = "E1"   # identification
E2 = "E2"   # responsibility
E3 = "E3"   # procedure
TIERS: Tuple[str, ...] = (E1, E2, E3)
#: For each required tier, the tiers whose evidence satisfies it.
SATISFIES: Dict[str, FrozenSet[str]] = {E1: frozenset({E1, E2}), E2: frozenset({E2}), E3: frozenset({E3})}
#: Evidence a stated-fact ask needs (a location, a structure fact, a count): a statement of that fact, not an office tier.
STATED_FACT = "stated_fact"


def tier_satisfies(reached: Optional[str], required: str) -> bool:
    """Whether evidence of tier ``reached`` (None: no evidence) is enough for an ask that needs ``required``."""
    if required not in SATISFIES:
        raise ValueError("unknown evidence tier %r" % (required,))
    return reached in SATISFIES[required]


# ---------------------------------------------------------------------------
# Ask kinds
# ---------------------------------------------------------------------------
CHANNEL = "channel"                       # telephone, fax, e-mail, extension, website
OFFICE_ROLE = "office_role"               # what does X do / what is X responsible for / is X responsible for Y (phase B3)
REFERRAL = "referral"                     # who should I contact about X
AUTHORITY = "authority"                   # who approves / decides
RESPONSIBLE_OFFICE = "responsible_office" # which office handles / who is responsible for
LOCATION = "location"                     # where is / located / building
SYSTEM_ACCESS = "system_access"           # log in, portal, account access
PROCEDURE = "procedure"                   # how do I / steps / submit / apply
STRUCTURE = "structure"                   # organisational chart, governance, councils
FACT = "fact_or_count"                    # colleges, departments, campuses, programmes, accreditation, ranking, how many
OTHER = "other"

_ASK_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple((kind, re.compile(p, re.I)) for kind, p in (
    # Phase B2 adds the "how do I contact / reach X" and "number for X" phrasings (the Coordinator's own contact cues).
    (CHANNEL, r"\b(?:phone|telephone|fax|e-?mail(?:\s+address)?|extension|ext\.|website|web\s?page|url|hotline|contact\s+(?:numbers?|details|information))\b|"
              r"\bhow\s+(?:do|does|can|could|should)\s+(?:i|we|one)\s+(?:contact|reach|get\s+in\s+touch\s+with|call)\b|\bnumbers?\s+for\b"),
    (REFERRAL, r"\b(?:who|whom)\s+(?:should|do|can|could|must|would)\s+i\s+(?:contact|ask|talk\s+to|speak\s+to|reach|call|go\s+to|see)\b|"
               r"\bwho\s+(?:is|are)\s+the\s+contact\b|\bpoint\s+of\s+contact\b|\bcontact\s+person\b|"
               r"\bwhere\s+(?:should|do|can|could|must)\s+i\s+(?:report|go|turn)\b"),                           # phase B3
    (AUTHORITY, r"\bwho\b.{0,40}\b(?:approves?|authori[sz]es?|signs?\s+off|decides?)\b|\bapproval\s+(?:chain|authority|process)\b|"
                r"\b(?:approv(?:e|es|ed|ing|al|als)|authori[sz](?:e|es|ed|ing|ation)|sign(?:s|ing)?\s+off|decid(?:e|es|ing)|"
                r"grant(?:s|ing|ed)?|confer(?:s|ring|ral|red)?)\b|"                                               # phase B3: any approval word
                r"\bauthorit(?:y|ies)\s+(?:to|over)\b|\b(?:has|have|holds?|with|given)\s+the\s+(?:mandate|authority|power|right)\b|"
                r"\bmandate\s+(?:and\s+authority\s+)?to\b|\bentitled\s+to\b|\bpower\s+to\b|\bempower\w*|\bmandated\b|"
                r"\b(?:permitted|allowed)\s+to\b|\bpermission\s+to\b|\bcompeten(?:ce|t)\s+to\b|\bjurisdiction\b"),  # phase B3 review: authority nouns
    # Phase B3: system access and procedures are recognised before any office ask, so "What do I need to do to ...",
    # "What is the role of the ITC in password resets?" are never answered from a responsibility statement.
    (SYSTEM_ACCESS, r"\b(?:log\s?in|log-in|sign\s?in|passwords?|accounts?|portal|my\s?uos|blackboard|banner|docusign|resets?)\b"),
    (PROCEDURE, r"\bhow\s+(?:do|can|should|must|would|could)\s+(?:i|we|one|faculty)\b|\bhow\s+to\b|\bsteps?\b|\bprocedures?\b|\bsubmit\b|"
                r"\bapply\b|\bregister\b|\bwhat\s+(?:do|should|must)\s+i\s+(?:need\s+to\s+|have\s+to\s+)?do\b|\bprocess\s+(?:for|of|to)\b"),
    # Phase B3: an ask about a NAMED unit's function ("What does X do?", "What is X responsible for?", "What services
    # does X provide?", "Is X responsible for Y?", "Does X provide Y?"). "Who is responsible for Y?" stays responsible_office.
    (OFFICE_ROLE, r"\bwhat\s+(?:does|do)\s+(?:the\s+)?[^?]{2,90}?\s+do\b|\bwhat\s+(?:is|are)\s+(?:the\s+)?[^?]{2,90}?\s+responsible\s+for\b|"
                  r"\b(?:role|roles|responsibilit(?:y|ies)|functions?|duties|mandate|tasks)\s+of\b|\bwhat\s+services?\s+(?:does|do|is|are)\b|"
                  r"\bservices?\s+(?:offered|provided)\s+by\b|"
                  r"\bwhat\s+(?:does|do)\s+(?:the\s+)?[^?]{2,90}?\s+(?:provide|offer|handle|manage|oversee|support|administer|maintain|deliver)\b|"
                  r"^\s*(?:is|are)\s+(?:the\s+)?[^?]{2,90}?\s+(?:responsible\s+for|in\s+charge\s+of)\b|"
                  r"^\s*(?:does|do)\s+(?:the\s+)?[^?]{2,90}?\s+(?:provide|offer|handle|manage|oversee|support|administer|maintain|deliver|issue|"
                  r"coordinate|operate)\b"),
    (RESPONSIBLE_OFFICE, r"\b(?:which|what)\s+(?:office|unit|department|section|cent(?:er|re)|body)\b.{0,40}\b(?:handles?|responsible|deals?|manages?|"
                         r"provides?|takes\s+care|in\s+charge|supports?|issues?|maintains?|administers?|coordinates?|operates?|oversees?|runs?|"
                         r"organi[sz]es?|conducts?|delivers?|offers?|facilitates?|supervises?|arranges?|promotes?|establish(?:es)?|develops?|"
                         r"implements?)\b|\bwho\s+(?:is\s+responsible|handles|manages|supports|maintains|administers|oversees|"
                         r"provides|runs|is\s+in\s+charge|coordinates|conducts|offers|delivers|issues|organi[sz]es|operates|supervises|facilitates|"
                         r"arranges|takes\s+care\s+of|deals\s+with|looks\s+after|promotes|establishes|develops|implements)\b"),
    (LOCATION, r"\bwhere(?:'s|s|'re|\s+(?:is|are|was|were|can|could|do|does|would|will|should|might)\b)|\bwhereabouts\b|\btell\s+me\s+where\b|\blocated\b|\blocations?\b|\b(?:which|what)\s+(?:building|floor|room|campus)\b|\baddress\b"),
    (STRUCTURE, r"\borgani[sz]ational\s+(?:chart|structure)\b|\bgovernance\b|\bstructure\s+of\s+the\s+university\b|\bchain\s+of\s+command\b|"
                r"\bboard\s+of\s+trustees\b|\bcouncils?\b|"
                r"\bsub-?groups?\b|\bsub-?units?\b|\bchair(?:s|ed|person)?\b|\bmembers?\s+of\b|\bmembership\b|\bconsists?\s+of\b|\bcomposed\s+of\b|"
                r"\bcomposition\b"),                                                                                  # phase B3
    (FACT, r"\bhow\s+many\b|\bcolleges\b|\bdepartments\b|\bcampus(?:es)?\b|\bdegree\s+programs?\b|\bprograms?\s+offered\b|\baccreditation\b|"
           r"\branking\b|\bmission\b|\bvision\b|\bcore\s+values\b"),
))

#: The evidence each ask kind needs before it could ever be ``supported``.
REQUIRED_EVIDENCE: Dict[str, Optional[str]] = {
    CHANNEL: E1,             # plus the directory-row binding of phase B2
    OFFICE_ROLE: E2,
    REFERRAL: E2,
    AUTHORITY: E2,
    RESPONSIBLE_OFFICE: E2,
    SYSTEM_ACCESS: E3,
    PROCEDURE: E3,
    LOCATION: STATED_FACT,
    STRUCTURE: STATED_FACT,
    FACT: STATED_FACT,
    OTHER: None,
}
ASK_KINDS: Tuple[str, ...] = tuple(REQUIRED_EVIDENCE)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("’", "'").replace("‘", "'")).strip()


def classify_ask(clause: str) -> str:
    """The first ask kind whose pattern matches the clause, in the fixed priority order; ``other`` when none does."""
    text = _normalise(clause)
    for kind, pattern in _ASK_PATTERNS:
        if pattern.search(text):
            return kind
    return OTHER


# ---------------------------------------------------------------------------
# Office lexicon
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Office:
    """One unit as the handbook prints it. ``aliases`` pairs each attested alias with the pages that print it."""

    office_id: str
    name: str
    kind: str                                  # office | college | deanship | council | board | facility | library | section | service
    pages: Tuple[int, ...]
    aliases: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()
    needs_office_noun: bool = False            # a one-word generic name is matched only beside an office noun
    contact: bool = True                       # False: a chapter unit added in phase B3; never matched to a directory line
    #: Phase B3 review: acronyms the handbook prints for the unit that are read only as the subject of a statement, never
    #: in a question or on a directory line (so no contact answer changes).
    subject_aliases: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()


OFFICES: Tuple[Office, ...] = (
    # ---- the contact directory, section 1.16 (pages 53-55) ---------------------------------------------------------
    Office("registration", "Registration", "office", (53,), needs_office_noun=True),
    Office("disability_resource_center", "Disability Resource Center", "service", (53,), subject_aliases=(("DRC", (255, 256)),)),
    Office("dental_hospital_call_center", "Dental Hospital Call Center", "service", (53,)),
    Office("student_housing", "Student Housing", "service", (53,)),
    Office("passport_unit", "Passport Unit", "office", (53,)),
    Office("college_sharia_islamic_studies", "College of Shari'a & Islamic Studies", "college", (53,)),
    Office("college_sciences", "College of Sciences", "college", (53,)),
    Office("college_arts_humanities_social_sciences", "College of Arts, Humanities and Social Sciences", "college", (53,)),
    Office("college_business_administration", "College of Business Administration", "college", (53,)),
    Office("college_engineering", "College of Engineering", "college", (53,)),
    Office("college_health_sciences", "College of Health Sciences", "college", (53,)),
    Office("college_law", "College of Law", "college", (53,)),
    Office("college_fine_arts_design", "College of Fine Arts & Design", "college", (53,)),
    Office("college_communication", "College of Communication", "college", (53,)),
    Office("college_medicine", "College of Medicine", "college", (53,)),
    Office("college_dental_medicine", "College of Dental Medicine", "college", (53,)),
    Office("college_pharmacy", "College of Pharmacy", "college", (53,)),
    Office("college_computing_informatics", "College of Computing & Informatics", "college", (53,)),
    Office("college_public_policy", "College of Public Policy", "college", (53,)),
    Office("college_graduate_studies_research", "College of Graduate Studies and Research", "college", (54,)),
    Office("deanship_student_affairs", "Deanship of Student Affairs", "deanship", (54,)),
    Office("information_technology_center", "Information Technology Center", "office", (54, 183), aliases=(("ITC", (183,)),)),
    Office("physical_plant", "Physical Plant", "office", (54,)),
    Office("human_resources", "Human Resources", "office", (54,), aliases=(("HR", (57, 72)),)),
    Office("public_relations", "Public Relations", "office", (54,)),
    Office("media_center", "Media Center", "office", (54,)),
    Office("finance_department", "Finance Department", "office", (54,)),
    Office("alumni_association", "Alumni Association", "office", (54,)),
    Office("continuing_education_professional_development", "Center for Continuing Education and Professional Development", "office", (54,),
           aliases=(("CCE", (54,)),)),
    Office("central_laboratories", "Central Laboratories", "facility", (54,)),
    Office("medical_service_department", "Medical Service Department", "service", (54,)),
    Office("medical_colleges_library", "Medical Colleges Library", "library", (54,)),
    Office("mens_library", "Men's Library", "library", (54,)),
    Office("womens_library", "Women's Library", "library", (54,)),
    Office("business_services_purchases_department", "Business Services and Purchases Department", "office", (55,)),
    Office("university_city_security", "University City Security - Sharjah", "service", (55,)),
    # ---- units named in the chapters ---------------------------------------------------------------------------------
    Office("health_and_safety_office", "Health and Safety Office", "office", (237, 240)),
    Office("business_service_department", "Business Service Department", "office", (251,), aliases=(("Business Services Department", (251,)),)),
    Office("facilities_management_planning_department", "Facilities Management and Planning Department", "office", (254,), aliases=(("FMPD", (254,)),)),
    Office("environmental_health_safety_management", "Environmental, Health, and Safety Management", "section", (254,), aliases=(("EHSM", (254,)),)),
    Office("office_international_relations", "Office of International Relations", "office", (47,), aliases=(("OIR", (47,)),)),
    Office("language_institute", "Language Institute", "office", (44,)),
    Office("office_quality_assurance", "Office of Quality Assurance, Institutional Effectiveness and Accreditation", "office", (28,)),
    Office("college_council", "College Council", "council", (25,)),
    Office("board_of_trustees", "Board of Trustees", "board", (21,)),
    Office("main_library", "Main Library", "library", (199,)),
    # ---- phase B3: chapter units, each printed on its page; never matched to a directory line (contact=False) -------
    Office("library_staff", "Library Staff", "staff", (198,), contact=False),
    Office("medical_library", "Medical Library", "library", (199,), contact=False),
    Office("fine_arts_design_library", "Fine Arts and Design Library", "library", (199,), contact=False),
    # ---- Step 4.16-C1: units printed in the owned chapters with their own statements; never matched to a directory line
    Office("central_laboratories_directorate", "Central Laboratories Directorate", "facility", (250,), (("CLD", (250,)),), contact=False),
    Office("security_services_unit", "Security Services Unit", "service", (251,), (("SSU", (251,)),), contact=False),
    Office("transport_section", "Transport Section", "section", (252, 253), contact=False),
    Office("mail_office", "Mail Office", "office", (253,), contact=False),
    Office("facilities_management_section", "Facilities Management Section", "section", (254,), contact=False),
    Office("projects_management_section", "Projects Management Section", "section", (254,), contact=False),
    Office("deans_council", "Deans Council", "council", (24,), contact=False),
    Office("university_academic_advisory_council", "University Academic Advisory Council", "council", (27,), (("UAAC", (27,)),),
           contact=False),
)
OFFICES_BY_ID: Dict[str, Office] = {o.office_id: o for o in OFFICES}

#: Pairs of entries the handbook never states to be one unit; they must never resolve to each other.
NOT_EQUIVALENT: Tuple[Tuple[str, str], ...] = (
    ("physical_plant", "facilities_management_planning_department"),
    ("business_service_department", "business_services_purchases_department"),
    ("health_and_safety_office", "environmental_health_safety_management"),
    ("college_medicine", "college_dental_medicine"),
    # Step 4.16-C1: the handbook prints these as separate units (a directory line and a section; a department and its subgroups)
    ("central_laboratories", "central_laboratories_directorate"),
    ("facilities_management_section", "facilities_management_planning_department"),
    ("facilities_management_section", "physical_plant"),
    ("projects_management_section", "facilities_management_planning_department"),
    ("deans_council", "college_council"),
)

_OFFICE_NOUN = r"(?:office|department|unit|section|desk|cent(?:er|re))"


def _is_acronym(alias: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{2,6}", alias))


def _names(office: Office) -> List[str]:
    return [office.name] + [alias for alias, _pages in office.aliases]


def _patterns() -> List[Tuple[int, str, "re.Pattern[str]"]]:
    """(length, office id, pattern), longest name first, so a longer printed name wins over a name inside it."""
    out: List[Tuple[int, str, "re.Pattern[str]"]] = []
    for office in OFFICES:
        for name in _names(office):
            if _is_acronym(name):
                pattern = re.compile(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])")          # case-sensitive acronym
            elif office.needs_office_noun and name == office.name:
                body = re.escape(_normalise(name))
                pattern = re.compile(r"\b%s\s+%s\b|\b%s\s+of\s+%s\b" % (body, _OFFICE_NOUN, _OFFICE_NOUN, body), re.I)
            else:
                pattern = re.compile(r"(?<![\w])" + re.escape(_normalise(name)) + r"(?![\w])", re.I)
            out.append((len(name), office.office_id, pattern))
    out.sort(key=lambda t: (-t[0], t[1]))
    return out


_OFFICE_PATTERNS = _patterns()


@lru_cache(maxsize=65536)
def office_spans(text: str) -> Tuple[Tuple[int, int, str], ...]:
    """(start, end, office id) of every office mention in the normalised text, in order of appearance. A matched span
    is consumed, so a shorter printed name inside a longer one is not matched again; nothing is matched by a partial
    name or by a shared word."""
    norm = _normalise(text)
    taken: List[Tuple[int, int, str]] = []
    for _length, office_id, pattern in _OFFICE_PATTERNS:
        for m in pattern.finditer(norm):
            if any(m.start() < end and start < m.end() for start, end, _o in taken):
                continue
            taken.append((m.start(), m.end(), office_id))
    return tuple(sorted(taken))


def resolve_offices(text: str) -> Tuple[str, ...]:
    """The office ids a text names, in order of first appearance (see ``office_spans``)."""
    ordered: List[str] = []
    for _start, _end, office_id in office_spans(text):
        if office_id not in ordered:
            ordered.append(office_id)
    return tuple(ordered)


@lru_cache(maxsize=65536)
def _masked(text: str) -> str:
    """The normalised text with every office mention blanked out, so words inside an office name ("Call" in "Dental
    Hospital Call Center", "Library" in "Men's Library") are not read as words of the ask."""
    norm = _normalise(text)
    chars = list(norm)
    for start, end, _office_id in office_spans(norm):
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def clause_report(index: int, clause: str) -> Dict[str, object]:
    """What is recorded for one focus clause: its ask kind, the evidence the ask needs and the offices it names."""
    kind = classify_ask(clause)
    offices = resolve_offices(clause)
    return {
        "index": index,
        "clause": clause,
        "ask_kind": kind,
        "required_evidence": REQUIRED_EVIDENCE[kind],
        "offices": list(offices),
        "office_names": [OFFICES_BY_ID[o].name for o in offices],
        "evidence_tier_reached": None,
        "answered": False,
    }


# ---------------------------------------------------------------------------
# Contact fields of a channel ask (phase B2)
# ---------------------------------------------------------------------------
TELEPHONE = "telephone"
FAX = "fax"
EXTENSION = "extension"
EMAIL = "email"
WEBSITE = "website"
CONTACT_FIELDS: Tuple[str, ...] = (TELEPHONE, FAX, EXTENSION, EMAIL, WEBSITE)
#: The fields a generic contact ask ("contact details", "how can I contact X") asks for: the directory's two columns.
GENERIC_FIELDS: Tuple[str, ...] = (TELEPHONE, FAX)

_FIELD_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple((f, re.compile(p, re.I)) for f, p in (
    (TELEPHONE, r"\b(?:tele)?phones?\b|\bhotlines?\b|\bcall\b|\bcontact\s+number\b"),
    (FAX, r"\bfax(?:es)?\b"),
    (EXTENSION, r"\bextensions?\b|\bext\b"),
    (EMAIL, r"\be-?mails?\b"),
    (WEBSITE, r"\bweb\s?sites?\b|\bweb\s?pages?\b|\burls?\b|\bhome\s?pages?\b"),
))


def contact_fields(clause: str) -> Tuple[str, ...]:
    """The contact fields a channel clause asks for, in the fixed order of ``CONTACT_FIELDS``. Words inside an office
    name do not count. A clause naming no field ("contact details", "how can I reach X") asks for both directory
    columns, telephone and fax."""
    text = _masked(clause)
    found = tuple(f for f, pattern in _FIELD_PATTERNS if pattern.search(text))
    return found or GENERIC_FIELDS


# ---------------------------------------------------------------------------
# Referent check (phase B2): is the named office the unit the clause asks about?
# ---------------------------------------------------------------------------
#: Nouns that, outside an office name, make the asked unit something other than the named office: a library, clinic,
#: hospital, laboratory, housing or security unit, a section for men or women, a campus or branch, or a person.
_SUBUNIT = re.compile(
    r"\b(?:librar(?:y|ies)|clinics?|hospitals?|laborator(?:y|ies)|labs?|housing|security|stores?|bookstores?|cafeterias?|canteens?|gyms?|parking|"
    r"pharmac(?:y|ies)|branch(?:es)?|campus(?:es)?|men|women|male|female|ladies|boys|girls|deans?|deanship|manager|director|head|chair(?:person)?|"
    r"coordinator|advis[eo]r|officer|secretary|assistant|president|vice\s+chancellor|chancellor|provost|registrar|supervisor)\b", re.I)
#: What may stand between two office mentions of one clause when the clause asks about each of them ("HR and Finance").
_CONJUNCTION = re.compile(r"^(?:'s)?(?:\s+%s)?\s*(?:,\s*(?:(?:and|or|&)\s+)?|\s+(?:and|or|&|as\s+well\s+as|plus)\s+)(?:the\s+)?$" % _OFFICE_NOUN,
                          re.I)                                     # phase B3: "X's Y" and "X Y" are one referent, not two offices


#: Institutions and places other than the University of Sharjah's own units: the former branches (now universities of
#: their own), other universities and cities. An office asked about there is not the handbook's office (phase B3 review).
_ELSEWHERE_WORDS = (r"(?i:\b(?:university\s+of\s+(?!sharjah\b)[a-z]+|american\s+university(?:\s+of\s+sharjah)?|higher\s+colleges\s+of\s+technology|"
                    r"khor\s*fakk?an|kalba|(?:al\s*)?dhaid|saasst|dubai|abu\s+dhabi|al\s+ain|ajman|fujairah|ras\s+al\s+khaimah|rak|"
                    r"umm\s+al\s+quwain|uaq)\b)")
#: "Zayed University", "Khalifa University" (case-sensitive: a capitalised name before "University").
_OTHER_UNIVERSITY = r"\b(?!(?:The|This|That|Our|Which|What|Whose|Of|At|In|For|And)\b)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+University\b"
_ELSEWHERE = re.compile(r"%s|%s" % (_ELSEWHERE_WORDS, _OTHER_UNIVERSITY))
def elsewhere(clause: str) -> Optional[str]:
    """The other institution or place a clause names ("the University of Khorfakkan", "Kalba", "Zayed University"), or None."""
    m = _ELSEWHERE.search(_normalise(clause))
    return m.group(0) if m else None


# ---- plain contact forms (phase B3 review) ----------------------------------------------------------------------------
# Five reviews showed that any extra word of a contact ask can name another unit, a person, a place or a field the
# directory does not print, and that closed lists of such words never covered them all while they also refused plain
# asks. A contact clause is therefore answered only when the whole clause, courtesy words aside, has one of these forms.
_F_CORE = r"(?:(?:tele)?phone|fax|e-?mail|extension|ext\.?|website|web\s?page|url|hotline)"
_F_ONE = r"(?:%s(?:\s+(?:numbers?|no\.?|address(?:es)?|line))?)" % _F_CORE
_FIELD = r"(?:contact\s+(?:numbers?|details|information|info)|(?:both\s+(?:the\s+)?)?%s(?:\s*(?:,|and|&|or)\s*(?:the\s+)?%s)*)" % (_F_ONE, _F_ONE)
_OL = r"⟦O⟧(?:'s|s')?(?:\s*(?:,|and|&|or|as\s+well\s+as)\s*⟦O⟧(?:'s|s')?)*"
_PRE = (r"(?:what\s+(?:is|are)|what's|whats|(?:can|could)\s+(?:i|you)\s+(?:have|get|give\s+me|tell\s+me|share|send\s+me)|may\s+i\s+(?:have|get)|"
        r"(?:give|tell|send|share)\s+me|i\s+(?:need|want|would\s+like)|i'd\s+like|do\s+you\s+(?:have|know))")
_TAIL = r"(?:\s+(?:again|now|today|please|exactly|quickly|here|directly|(?:as\s+)?(?:listed|printed)(?:\s+in\s+the\s+(?:handbook|directory))?))*"
_PURPOSE = r"(?:\s+(?:for|to|so|if|when|because|since)\s+(?P<purpose>[^?]+?))?"
_CHANNEL_FORMS = tuple(re.compile(p, re.I) for p in (
    r"(?:%s\s+)?(?:the\s+)?%s\s+(?:of|for)\s+%s%s%s" % (_PRE, _FIELD, _OL, _TAIL, _PURPOSE),
    r"(?:%s\s+)?(?:the\s+)?%s\s+%s%s%s" % (_PRE, _OL, _FIELD, _TAIL, _PURPOSE),
    r"how\s+(?:can|do|could|should|would|may)\s+(?:i|we|one)\s+(?:contact|reach|call|phone|fax|e-?mail|get\s+in\s+touch\s+with)\s+%s%s%s"
    % (_OL, _TAIL, _PURPOSE),
    r"(?:what|which)\s+%s\s+(?:does|do)\s+%s\s+(?:use|have)%s" % (_FIELD, _OL, _TAIL),
    r"which\s+%s\s+should\s+i\s+(?:use|call|dial)\s+for\s+%s%s" % (_FIELD, _OL, _TAIL),
    # a second field of the same office: "... of the Finance Department and what is its fax number?", "... of HR? And the fax?"
    r"(?:%s\s+)?(?:the\s+)?%s\s+(?:of|for)\s+⟦O⟧%s(?:\s*\?)?\s*,?\s*(?:and|also|plus)\s+(?:what\s+(?:is|are)\s+|what's\s+)?(?:the|its|their)\s+%s"
    % (_PRE, _FIELD, _TAIL, _FIELD),
    # the office named first: "For the ITC, what is the fax number?"
    r"(?:for|regarding|about)\s+⟦O⟧\s*,\s*(?:%s\s+)?(?:the|its|their)\s+%s" % (_PRE, _FIELD),
))
_CHANNEL_LEAD = re.compile(r"^\s*(?:(?:hi|hello|dear\s+(?:sir|madam|all|team))\s*,?\s+)?(?:(?:i\s+am|i'm)\s+(?:a\s+)?new\s+[a-z ]{1,30}[.,;]\s*)?"
                           r"(?:(?:please|kindly)\s*,?\s+)?", re.I)
_CHANNEL_END = re.compile(r"\s*,?\s*(?:please|thanks|thank\s+you(?:\s+very\s+much)?)?\s*[?.!]*\s*$", re.I)
#: A purpose says who the number is for or why ("for new faculty", "to send documents") in plain lower-case words: a
#: capitalised name (another institution, a place, an office), a person or a branch in it fails closed.
_PURPOSE_BAD = re.compile(r"(?!I\b)\b[A-Z]|\b(?:cashiers?|payroll|accountants?|clerks?|receptionists?|technicians?|photographers?|boss|doctors?|"
                          r"nurses?|auditors?|deans?|managers?|directors?|heads?|branch(?:es)?|campus(?:es)?|other|another|else)\b")


def _channel_shape(clause: str) -> str:
    norm = _normalise(clause)
    out, last = [], 0
    for start, end, _o in office_spans(norm):
        out.append(norm[last:start])
        out.append("⟦O⟧")
        last = end
    out.append(norm[last:])
    shape = re.sub(r"(?i)\bthe\s+⟦O⟧", "⟦O⟧", "".join(out))
    shape = re.sub(r"(?i)⟦O⟧(?:'s)?\s+(?:main\s+)?(?:office|department|dept\.?|unit|section|team|desk|cent(?:er|re))\b", "⟦O⟧", shape)
    shape = re.sub(r"(?i)⟦O⟧\s+(?:at|of|in)\s+(?:the\s+)?(?:university(?:\s+of\s+sharjah)?|uos)\b", "⟦O⟧", shape)
    for _ in range(2):
        shape = _CHANNEL_END.sub("", _CHANNEL_LEAD.sub("", shape))
    return re.sub(r"\s+", " ", shape).strip()


def channel_form_problem(clause: str) -> Optional[str]:
    """Why a contact clause is not one of the plain contact forms, or None."""
    shape = _channel_shape(clause)
    for form in _CHANNEL_FORMS:
        m = form.fullmatch(shape)
        if m and not (m.groupdict().get("purpose") and _PURPOSE_BAD.search(m.group("purpose"))):
            return None
    return ("the clause is not a plain contact ask for a named office (such as \"What is the phone number of X?\"), so no "
            "directory entry is selected")


# ---- natural phrasing (Step 4.16-C1) ---------------------------------------------------------------------------------------
# Faculty write politely, add a sentence of context and ask indirectly. ``natural_ask`` reads such a clause as the plain
# question it asks, with closed lists only: it sets aside a context sentence when nothing in it can change the asked unit,
# field, place or condition, removes a request frame ("Could you please tell me ...", "May I have ..."), reorders an
# embedded question ("where X is" -> "where is X") and drops filler words. Every check the plain forms apply still reads
# the result; a clause it cannot read safely is returned unchanged and fails closed as before. ``_sentences`` below uses
# the module's one ``_ABBREVIATION`` pattern, defined with the unit reader (a shadowed C1 copy was removed at the freeze).
_ASK_START = re.compile(r"^\s*(?:(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|dear\s+\w+)\s*[,!.]?\s*)?(?:(?:please|kindly)\s*,?\s*)?"
                        r"(?:who|whom|which|what|what's|whats|where|where's|wheres|how|is|are|does|do|can|could|would|will|may|"
                        r"tell\s+me|give\s+me|share|send\s+me|list|name|describe|explain|"
                        r"i\s+(?:would|'d)\s+like\s+(?:to\s+know|the|both)|i'd\s+like\s+(?:to\s+know|the|both)|i\s+(?:want|need)\s+(?:to\s+know|the|both))\b",
                        re.I)
#: Words that let a context sentence change what is asked: a place, campus, gender or building; a condition or exclusion.
_CONTEXT_PLACE = re.compile(r"\b(?:men|men's|mens|women|women's|womens|male|female|ladies|gentlemen|boys|girls|campus(?:es)?|branch(?:es)?|"
                            r"building|buildings|floor|room|wing|block)\b|\d", re.I)
_CONTEXT_CONDITION = re.compile(r"\b(?:if|unless|except|only|besides|instead|excluding|without|whether)\b|\bother\s+than\b|"
                                r"\brather\s+than\b|\bapart\s+from\b", re.I)
_CONTEXT_NEGATION = re.compile(r"\b(?:not|no|never|none|neither|nor)\b|n't\b", re.I)
_FIELD_WORD = re.compile(r"\b(?:phone|telephone|fax|e-?mail|extension|number|numbers|website|url|address|line)\b", re.I)
#: Words of the question that can refer back to the context: it is then not read alone.
_ASK_ANAPHOR = re.compile(r"\b(?:its|they|them|their|theirs|this|that|these|those|such|same|here|he|she|him|his|her|hers|one|ones|"
                          r"my|mine|our|ours|we|us|latter|former)\b|"
                          r"\b(?:it)\b|\bthere\b(?!\s+(?:is|are|was|were)\b)", re.I)
_EXISTENTIAL = re.compile(r"\b(?:is|are|was|were)\s+there\b", re.I)
#: Step 4.16-C1 review (two adversarial rounds): a context sentence is set aside only when it has one of three closed
#: shapes, and only BEFORE the question: a greeting or thanks ("Hello.", "Thank you."), a self-introduction ("I'm a new
#: lecturer.", "I just joined the university this semester.") or the asker's own unit ("I'm joining the College of Medicine
#: next month."). After the question only a greeting or thanks is set aside: a sentence there may correct or redirect it
#: ("... I was thinking of the dental college."). Any other sentence keeps the clause as written (it fails closed as before).
_GREET = r"(?:(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))\s*[,!.]?\s*)?"
_WHEN = r"(?:\s+(?:this|last|next)\s+(?:week|month|semester|term|year)|\s+(?:today|recently))?"
_INTRO_ROLE = (r"(?:faculty\s+member|member\s+of\s+(?:the\s+)?faculty|lecturer|instructor|teacher|researcher|(?:assistant\s+|associate\s+)?professor|"
               r"postdoc(?:toral\s+(?:fellow|researcher))?|staff\s+member)")
_INTRO = re.compile(r"^%s(?:(?:i\s+am|i'm)\s+(?:(?:a|an)\s+(?:new\s+)?%s(?:\s+here|\s+at\s+(?:the\s+)?(?:university|uos))?|new(?:\s+here|\s+to\s+"
                    r"(?:the\s+)?(?:university|uos))?)|(?:i\s+(?:have\s+)?|i've\s+)?(?:just\s+)?(?:joined|started\s+at)\s+(?:the\s+)?(?:university|uos))"
                    r"%s\s*[.!]?$" % (_GREET, _INTRO_ROLE, _WHEN), re.I)
#: the asker's own unit, the whole sentence matching: "I'm joining the College of Medicine next month.", "just started in the
#: College of Sciences this week.", "I teach in the College of Law." (the office is named only as where the asker works)
_INTRO_UNIT = re.compile(r"^%s(?:(?:i\s+am|i'm|i've|i\s+have|i)\s+(?:just\s+)?|just\s+)(?:joined|joining|started|starting)(?:\s+(?:at|in))?\s+"
                         r"⟦O⟧%s\s*[.!]?$|^%s(?:i|we)\s+(?:teach|work)\s+(?:at|in)\s+⟦O⟧\s*[.!]?$" % (_GREET, _WHEN, _GREET), re.I)
#: a unit word of the question outside its office names: after an own-unit introduction it may refer back to that unit
#: ("I work in the College of Law. ... for the college?")
_UNIT_WORD = re.compile(r"\b(?:colleges?|facult(?:y|ies)|schools?|campus(?:es)?|staff|departments?|units?|sections?|offices?|"
                        r"cent(?:er|re)s?|programs?|programmes?|division|divisions|institutes?)\b", re.I)
_COURTESY_ONLY = re.compile(r"^(?:(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|dear\s+(?:sir|madam|all|team|colleagues?))|"
                            r"(?:many\s+)?thanks?(?:\s+you)?(?:\s+(?:very\s+much|so\s+much|in\s+advance))?|"
                            r"thank\s+you(?:\s+(?:very\s+much|so\s+much|in\s+advance))?)\s*[,!.]*$", re.I)
_GREETING = re.compile(r"^\s*(?:(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|dear\s+(?:sir|madam|all|team|colleagues?))\s*[,!.]?\s+)+",
                       re.I)
_COURTESY = re.compile(r"^\s*(?:please|kindly)\s*,?\s+", re.I)
#: Frames followed by an embedded question or a noun phrase.
_EMBED_FRAMES = tuple(re.compile(p, re.I) for p in (
    r"^(?:could|can|would|will)\s+you\s+(?:please\s+|kindly\s+)?(?:tell|let)\s+me(?:\s+know)?\s+(?!about\b)",
    r"^(?:please\s+)?(?:tell|let)\s+me(?:\s+know)?\s+(?!about\b)",                # "tell me about X" is itself a plain form
    r"^(?:i\s+would|i'd)\s+like\s+to\s+know\s+",
    r"^i\s+(?:want|need)\s+to\s+know\s+",
    r"^i\s+(?:was|am)\s+wondering\s+",
    r"^(?:do|would)\s+you\s+(?:happen\s+to\s+)?know\s+",
))
_NP_FRAMES = tuple(re.compile(p, re.I) for p in (
    r"^(?:could|can|would|will)\s+you\s+(?:please\s+|kindly\s+)?(?:give|send|share|provide)\s+(?:me|us)(?:\s+with)?\s+(?=(?:the|both)\b)",
    r"^(?:may|can|could)\s+i\s+(?:please\s+)?(?:have|get|ask\s+for)\s+(?=(?:the|both)\b)",
    r"^(?:please\s+)?(?:give|send|share|provide)\s+(?:me|us)(?:\s+with)?\s+(?=(?:the|both)\b)",
    r"^(?:i\s+would|i'd)\s+like\s+(?=(?:the|both)\b)",
    r"^i\s+(?:want|need)\s+(?=(?:the|both)\b)",
))
_VERB_FRAMES = re.compile(r"^(?:could|can|would|will)\s+you\s+(?:please\s+|kindly\s+)?(?=(?:list|name|describe|explain)\b)", re.I)
#: A filler word is removed only in one closed position: between a word and a verb or predicate ("Who actually chairs",
#: "What does X really do", "X actually responsible for"). After "or", "and", "as" or punctuation, or before a noun, it may
#: mark a correction ("X or actually Y", "phone or actually fax") and is kept, so the clause fails closed.
_ASK_FILLER = re.compile(r"\b(?P<prev>(?!(?:or|and|nor|but|than|plus|as|not)\b)[A-Za-z][\w']*)\s+(?:actually|really|exactly|basically|please|kindly)"
                         r"(?=\s+(?:do|does|did|chairs?|handles?|provides?|manages?|runs?|oversees?|maintains?|offers?|sits?|serves?|has|have|is|are|"
                         r"responsible|located|situated|based|in\s+charge|means?|covers?|consists?|includes?)\b)", re.I)
_CONTRACTION = ((re.compile(r"\bwhat's\b", re.I), "what is"), (re.compile(r"\bwhere's\b", re.I), "where is"),
                (re.compile(r"\bwho's\b", re.I), "who is"), (re.compile(r"\bwhats\b", re.I), "what is"))
_EMBEDDED = (
    # "where the Medical Library is (located)" -> "where is the Medical Library (located)"
    (re.compile(r"^(where|what)\s+(.+?)\s+(is|are)(\s+(?:located|situated|based))?\s*$", re.I), r"\1 \3 \2\4"),
    # "how many libraries the University (of Sharjah) has / runs" -> "how many libraries does the University have"
    (re.compile(r"^how\s+many\s+(.+?)\s+((?:the\s+)?university(?:\s+of\s+sharjah)?|uos)\s+(?:has|have|runs?|operates?|offers?|includes?)$", re.I),
     r"how many \1 does \2 have"),
    # "whether the DRC is responsible for X" -> "is the DRC responsible for X"
    (re.compile(r"^(?:if|whether)\s+(.+?)\s+(is|are)\s+((?:responsible\s+for|in\s+charge\s+of)\s+.+)$", re.I), r"\2 \1 \3"),
)
#: "the number of libraries the University of Sharjah runs" -> "how many libraries does the University of Sharjah have"
_NUMBER_OF = re.compile(r"^the\s+(?:total\s+)?(?:number|count)\s+of\s+(.+?)\s+(?:that\s+)?((?:the\s+)?university(?:\s+of\s+sharjah)?|uos)\s+"
                        r"(?:has|have|runs?|operates?|offers?|includes?)$", re.I)
_TRAILING_COUNT = re.compile(r"\s+(?:in\s+total|altogether|overall|in\s+all)\s*$", re.I)
#: ", I need to send them a claim" after a complete question: a trailing context clause
_TRAILING_CONTEXT = re.compile(r"^(?P<ask>.+?[^,\s])\s*,\s*(?P<ctx>(?:i|we)\s+(?:need|want|have|am|'m|would|'d|will|plan|hope)\b.+)$", re.I)


def _sentences(text: str) -> List[str]:
    parts: List[str] = []
    start = 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=\S)", text):
        piece = text[start:m.start()]
        if _ABBREVIATION.search(piece):
            continue
        parts.append(piece.strip())
        start = m.end()
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _is_ask(sentence: str) -> bool:
    return sentence.rstrip().endswith("?") or bool(_ASK_START.match(sentence))


def _office_shape(text: str) -> str:
    """The normalised text with each office mention (and a "the" before it) replaced by one marker."""
    norm = _normalise(text).strip()
    out, last = [], 0
    for start, end, _o in office_spans(norm):
        out.append(norm[last:start])
        out.append("⟦O⟧")
        last = end
    out.append(norm[last:])
    return re.sub(r"\s+", " ", re.sub(r"(?i)\bthe\s+⟦O⟧", "⟦O⟧", "".join(out))).strip()


def _context_sentence_problem(sentence: str, ask: str, after: bool = False) -> Optional[str]:
    """Why one context sentence may change the ask, or None when it has one of the allowed shapes (see ``_INTRO``)."""
    norm = _normalise(sentence).strip()
    if _COURTESY_ONLY.match(norm):
        return None                                          # "Hello.", "Thank you."
    if after:
        return "a sentence after the question may correct or redirect it"
    masked = _masked(sentence)
    offices = resolve_offices(sentence)
    if elsewhere(sentence):
        return "the context names another institution or place"
    if _CONTEXT_PLACE.search(masked):
        return "the context names a campus, gender, building or code"
    if _CONTEXT_CONDITION.search(masked):
        return "the context states a condition or an exclusion"
    if _CONTEXT_NEGATION.search(masked) and (_FIELD_WORD.search(masked) or offices):
        return "the context negates a unit or a contact field"
    if _INTRO.match(norm):
        return None                                          # "I'm a new lecturer.", "I just joined the university."
    ask_offices = resolve_offices(ask)
    if (len(offices) == 1 and ask_offices and offices[0] not in ask_offices and _INTRO_UNIT.match(_office_shape(sentence))
            and not _CONTEXT_PLACE.search(norm) and not _UNIT_WORD.search(_masked(ask))):            # the office name itself may carry a gender ("Women's Library")
        return None                                          # "I'm joining the College of Medicine next month."
    return "the context is not a greeting, a self-introduction or the asker's own unit"


def context_problem(context: Sequence[str], ask: str, after: Sequence[str] = ()) -> Optional[str]:
    """Why the context sentences before (``context``) or after (``after``) the ask may change what it asks (so they cannot
    be set aside), or None."""
    for sentence, later in [(c, False) for c in context] + [(c, True) for c in after]:
        why = _context_sentence_problem(sentence, ask, later)
        if why is not None:
            return why
    if _ASK_ANAPHOR.search(_EXISTENTIAL.sub(" ", re.sub(r"\bIT\b", " ", _masked(ask)))):
        return "the question refers back to the context"
    return None


def _plain_question(ask: str) -> Tuple[str, List[str]]:
    """The ask with its greeting, request frame, embedded order and filler words normalised."""
    notes: List[str] = []
    text = _normalise(ask).strip()
    end = re.search(r"\s*([?.!]*)\s*$", text)
    text = text[:end.start()].strip()
    for _ in range(2):
        g = _GREETING.match(text) or _COURTESY.match(text)
        if g:
            notes.append("greeting or courtesy word")
            text = text[g.end():]
    for rx, repl in _CONTRACTION:
        new = rx.sub(repl, text)
        if new != text:
            notes.append("contraction")
            text = new
    framed = False
    for frame in _EMBED_FRAMES:
        m = frame.match(text)
        if m:
            notes.append("request frame %r" % m.group(0).strip())
            text = text[m.end():]
            framed = True
            if not re.match(r"(?:who|whom|which|what|where|how|whether|if)\b", text, re.I):
                text = "what is " + text                     # "tell me the fax number of X" -> "what is the fax number of X"
            break
    else:
        for frame in _NP_FRAMES:
            m = frame.match(text)
            if m:
                notes.append("request frame %r" % m.group(0).strip())
                text = "what is " + text[m.end():]
                framed = True
                break
        else:
            m = _VERB_FRAMES.match(text)
            if m:
                notes.append("request frame %r" % m.group(0).strip())
                text = text[m.end():]
    filler = _ASK_FILLER.sub(lambda m: m.group("prev"), text)
    if filler != text:
        notes.append("filler word")
        text = re.sub(r"\s+", " ", filler).strip()
    m = _NUMBER_OF.match(re.sub(r"^what\s+is\s+", "", text, flags=re.I)) if framed else None
    if m:
        text = "how many %s does %s have" % (m.group(1), m.group(2))
        notes.append("count phrasing")
    elif framed:
        for rx, repl in _EMBEDDED:
            new = rx.sub(repl, text)
            if new != text:
                notes.append("embedded question order")
                text = new
                break
    if re.match(r"how\s+many\b", text, re.I):
        new = _TRAILING_COUNT.sub("", text)
        if new != text:
            notes.append("count qualifier")
            text = new
    return text.strip() + "?", notes


@lru_cache(maxsize=16384)
def natural_ask(clause: str, guard: bool = True) -> Tuple[str, Tuple[str, ...]]:
    """(the plain question a natural clause asks, what was set aside or normalised). The clause itself and an empty tuple
    when nothing is normalised or when the clause cannot be read safely (context that may restrict the ask, two
    questions, a question referring back to its context). ``guard=False`` (negative-control tests only) sets any context
    aside unchecked."""
    text = _normalise(clause).strip()
    if not text or (guard and elsewhere(text)):
        return clause, ()                         # another institution or place: read as written (no rewrite may split its name)
    sentences = _sentences(text)
    asks = [s for s in sentences if _is_ask(s)]
    if len(asks) != 1:
        return clause, ()
    ask = asks[0]
    at = next(k for k, s in enumerate(sentences) if s is ask)
    before, after = sentences[:at], sentences[at + 1:]
    trailing = _TRAILING_CONTEXT.match(ask.rstrip("?.! "))
    if trailing and not _is_ask(trailing.group("ctx")):
        after = [trailing.group("ctx")] + after
        ask = trailing.group("ask") + "?"
    notes: List[str] = []
    if before or after:
        why = context_problem(before, ask, after) if guard else None
        if why is not None:
            return clause, ()
        notes.append("context set aside: %s" % " ".join(before + after))
    plain, more = _plain_question(ask)
    notes.extend(more)
    if not notes:
        return clause, ()
    return plain, tuple(notes)


def normalise_text(text: str) -> str:
    """The text as every reader of this module sees it: straight apostrophes, single spaces."""
    return _normalise(text)


def subunit_nouns(clause: str) -> Tuple[str, ...]:
    """The sub-unit or person nouns of ``_SUBUNIT`` a clause uses outside every office name, lowercased, in order."""
    return tuple(dict.fromkeys(m.group(0).lower() for m in _SUBUNIT.finditer(_masked(clause))))


def nested_offices(clause: str) -> Optional[str]:
    """The words joining two office mentions when they are not a plain conjunction ("of the"), else None."""
    norm = _normalise(clause)
    spans = office_spans(norm)
    for (_s1, e1, _o1), (s2, _e2, _o2) in zip(spans, spans[1:]):
        if not _CONJUNCTION.match(norm[e1:s2]):
            return norm[e1:s2].strip()
    return None


def referent_problem(clause: str) -> Optional[str]:
    """Why the office a channel clause names may not be the unit it asks about, or None. Two cases: a sub-unit or
    person noun outside every office name ("the College of Medicine library", "the dean of the College of Law"), and
    two office mentions joined by anything but a conjunction ("the Finance Department of the College of Law")."""
    masked = _masked(clause)
    m = _SUBUNIT.search(masked)
    if m:
        return "the clause asks about a %s, which is not the directory entry of the office it names" % m.group(0).lower()
    norm = _normalise(clause)
    spans = office_spans(norm)
    problem = channel_form_problem(clause)
    if problem:
        return problem
    for (_s1, e1, _o1), (s2, _e2, _o2) in zip(spans, spans[1:]):
        if not _CONJUNCTION.match(norm[e1:s2]):
            return "the clause names one office within another (%r), so the unit asked about is not one directory entry" % norm[e1:s2].strip()
    return None


# ---------------------------------------------------------------------------
# Contact-directory lines (phase B2)
# ---------------------------------------------------------------------------
#: One complete UAE number as the directory prints it: country code, area code, seven digits (once printed 3 + 4).
NUMBER = r"\+971 6 (?:\d{7}|\d{3} \d{4})"
_COMPLETE = re.compile(r"^(%s) (%s)$" % (NUMBER, NUMBER))
_SINGLE = re.compile(r"^(%s)$" % NUMBER)
_LABELLED_VALUE = r"(%s|Ext \d{2,6}) \((External|Internal)\)" % NUMBER
_LABELLED = re.compile(r"^(?:%s ?)+$" % _LABELLED_VALUE)
_ANY_NUMBER = re.compile(NUMBER)
#: A printed label at the start of a line or after a name: "Men:", "Women:", "Men's Clinic:".
_LABEL = r"[A-Z][A-Za-z'’]*(?:\s[A-Z][A-Za-z'’]*)?:"
#: What may follow an office name at the start of a directory line: the end of the line, a colon (a labelled
#: sub-entry), or a space and a value or a label. Anything else ("Human Resources Records") is a different name.
_AFTER_NAME = r"(?=$|:|\s(?:\+|\d|\(|%s))" % _LABEL

ROW_COMPLETE = "complete"      # exactly two complete numbers, nothing else
ROW_SINGLE = "single"          # exactly one complete number
ROW_LABELLED = "labelled"      # values carrying their own printed labels, "(External)", "Ext ... (Internal)"
ROW_NAME_ONLY = "name_only"    # the office name and nothing else
ROW_SUB_ENTRY = "sub_entry"    # "Name: ...", a labelled entry inside another entry's cell
ROW_FRAGMENT = "fragment"      # part of a multi-line entry ("/", "Men:", an unfinished number)
ROW_KINDS: Tuple[str, ...] = (ROW_COMPLETE, ROW_SINGLE, ROW_LABELLED, ROW_NAME_ONLY, ROW_SUB_ENTRY, ROW_FRAGMENT)


def _raw_name_pattern(name: str) -> str:
    """A pattern for a printed name in raw extracted text: either apostrophe, any run of whitespace."""
    return r"\s+".join(re.escape(part).replace("'", "['’]") for part in name.split())


def _row_patterns(anchored: bool) -> List[Tuple[int, str, "re.Pattern[str]"]]:
    out: List[Tuple[int, str, "re.Pattern[str]"]] = []
    for office in OFFICES:
        if not office.contact:                 # phase B3 chapter units never match a directory line
            continue
        for name in _names(office):
            body = _raw_name_pattern(name)
            out.append((len(name), office.office_id, re.compile(r"^" + body + _AFTER_NAME if anchored else body)))
    out.sort(key=lambda t: (-t[0], t[1]))
    return out


_ROW_ANCHORED = _row_patterns(True)
_ROW_ANYWHERE = _row_patterns(False)


def row_office(text: str, *, anchored: bool = True) -> Optional[Tuple[str, int, int]]:
    """(office id, start, end of the name) when a directory line starts with an office's printed name or attested
    alias followed by a value, a label, a colon or the end of the line; None otherwise. The directory's office column
    is itself the office name, so "Registration" needs no office noun here. ``anchored=False`` (used only to show
    what the protection prevents) accepts the name anywhere in the line."""
    raw = (text or "").strip()
    for _length, office_id, pattern in (_ROW_ANCHORED if anchored else _ROW_ANYWHERE):
        m = pattern.match(raw) if anchored else pattern.search(raw)
        if m:
            return office_id, m.start(), m.end()
    return None


def read_row_values(rest: str) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """(kind, values, labels) of what a directory line holds after the office name. Values are verbatim substrings."""
    rest = (rest or "").strip()
    if not rest:
        return ROW_NAME_ONLY, (), ()
    if rest.startswith(":"):
        return ROW_SUB_ENTRY, tuple(_ANY_NUMBER.findall(rest)), ()
    m = _COMPLETE.match(rest)
    if m:
        return ROW_COMPLETE, (m.group(1), m.group(2)), ()
    m = _SINGLE.match(rest)
    if m:
        return ROW_SINGLE, (m.group(1),), ()
    if _LABELLED.match(rest):
        pairs = re.findall(_LABELLED_VALUE, rest)
        return ROW_LABELLED, tuple(v for v, _label in pairs), tuple(label for _v, label in pairs)
    return ROW_FRAGMENT, tuple(_ANY_NUMBER.findall(rest)), ()


def numbers_in(text: str) -> Tuple[str, ...]:
    """Every complete number printed in a line, in order, verbatim."""
    return tuple(_ANY_NUMBER.findall(text or ""))


_HEADER = re.compile(r"\b(Telephone|Fax) Number\s+(Telephone|Fax) Number\s*$")


def header_columns(text: str) -> Optional[Tuple[str, str]]:
    """The two number columns a directory header line prints, in order ("telephone", "fax"); None when the line is not
    a header naming both columns once."""
    m = _HEADER.search(_normalise(text))
    if not m or m.group(1) == m.group(2):
        return None
    return m.group(1).lower(), m.group(2).lower()


def continues_into_next(text: str) -> bool:
    """A line that runs on into the following line: it ends with "/" or an unfinished number ("+971", "+971 6"), or
    it is a labelled sub-row ("Men: ...", "Women: ...") of a multi-row entry."""
    t = _normalise(text)
    return bool(re.search(r"(?:/|\+971|\+971 6)$", t) or re.match(_LABEL, t))


def continues_from_previous(text: str) -> bool:
    """A line that completes the line before it: it starts with a digit or "/" (the rest of a number broken there)."""
    return bool(re.match(r"[\d/]", _normalise(text)))


# ---------------------------------------------------------------------------
# Statement reading (phase B3)
# ---------------------------------------------------------------------------
#: The university as a whole: the subject of structure and count statements ("The organization of the University of
#: Sharjah is composed of ...", "The University of Sharjah manages all functions of nine libraries."). It is not an
#: office and never answers a responsibility ask.
UNIVERSITY = "university"
UNIVERSITY_NAME = "University of Sharjah"

REL_RESPONSIBILITY = "responsibility"
REL_CHAIR = "chair"
REL_COMPOSITION = "composition"

SUBJECT_OFFICE = "office"          # a lexicon office named as the subject
SUBJECT_ANAPHOR = "anaphor"        # "This Council", "The office", "It", "Its mission"
SUBJECT_UNIVERSITY = "university"  # the University, UoS; "we" / "There are" only in a sentence that names the University
SUBJECT_OTHER = "other"            # a unit outside the lexicon ("The CLD", "The Mail Office", "Library staff" ...)

#: Verbs that bind a named subject to a responsibility (closed list; anything else binds nothing).
_RESPONSIBILITY = (
    r"(?:is|are)\s+(?:also\s+)?responsible\s+for", r"(?:is|are)\s+in\s+charge\s+of",
    r"(?:is|are)\s+(?:dedicated|committed)\s+to", r"remains?\s+committed\s+to", r"was\s+established\s+to",
    r"upholds\s+its\s+mission\s+to", r"places\s+(?:\w+\s+)?emphasis\s+on", r"holds\s+the\s+mandate(?:\s+and\s+authority)?\s+to",
    r"performs\s+the\s+following\s+functions", r"(?:is|are)\s+(?:actively\s+)?involved\s+in", r"plays\s+an?\s+(?:\w+\s+){0,2}role\s+in",
    r"assumes\s+the\s+(?:\w+\s+)?role\s+of", r"serves\s+as", r"(?:primary\s+)?mission\s+is\s+to",
    r"(?:oversees|manages|maintains|administers|supervises|handles|provides|provide|offers|offer|delivers|supports|coordinates|operates|"
    r"conducts|organi[sz]es|issues|extends|assists|furnishes|arranges)",
)
_RESPONSIBILITY_RX = re.compile(r"(?:also\s+)?(?:%s)\b" % "|".join(_RESPONSIBILITY), re.I)
#: A wider list, used only to notice that some other unit is stated to do something (it never binds an office).
_ANY_DUTY_RX = re.compile(r"(?:also\s+)?(?:(?:%s)|consolidates|focuses\s+on|performs|ensures|enforces|identifies|investigates|takes|helps|"
                          r"is\s+tasked\s+with|receives|monitors|implements|develops|promotes|facilitates|integrates|represents|utili[sz]e|"
                          r"partner|achieve|plays|includes?|may\s+\w+|have\s+the\s+right|reports?\s+to|report\s+to)\b" % "|".join(_RESPONSIBILITY), re.I)
_APPOSITIVE_DUTY = re.compile(r"(?:is|are)\s+[^.,;:]{0,140},\s+(?:dedicated|committed)\s+to\b", re.I)
_CHAIR_RX = re.compile(r"(?:is|are)\s+(?:formed\s+by\s+[^.]{1,80}?\s+and\s+)?chaired\s+by\b", re.I)
_COMPOSITION_RX = re.compile(r"(?:consists\s+of|(?:is|are)\s+(?:composed|comprised|made\s+up)\s+of|comprises|operates\s+through|encompasses)\b", re.I)

_PAGE_PREFIX = re.compile(r"^Page\s+\d+\s+")
_ITEM_MARK = re.compile(r"^(?:[•§]|o(?=\s)|\d{1,2}\.(?=\s)|[a-h]\.(?=\s))\s*")
_ITEM_TITLE = re.compile(r"^[A-Z][^:.]{0,90}:\s+")
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)+|Chapter\s+\d+:)\s+[^.]{0,120}?\s(?=(?:The|There)\s)")
#: One leading introductory clause ("To achieve this, ", "Established by Decision No. 299 of 2013-2014, ", "Meanwhile, "),
#: recognised by its first word only, so a subject with an inner comma ("Environmental, Health, ...") is never cut.
_ADVERBIAL = re.compile(r"^(?:Additionally|Furthermore|Moreover|Meanwhile|However|Also|Academically|Administratively|Established|As|In|To|From|"
                        r"During|Through|With|By|Since|Within|For|Beyond|Under|Upon|After|Before|Following|Overall|Currently|Today|Recently|"
                        r"Finally|Similarly|Likewise|Specifically|Accordingly|Consequently|Therefore|Thus|Each|At|On)\b[^,;:]{0,118},\s+"
                        r"(?=(?:the|The|we|We|it|It)\b|[A-Z])")
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)+\s+")
#: A list lead-in that names its office without making it the grammatical subject.
_LEAD_IN_FRAMES = (r"^(?:Key\s+)?(?:Functions|Responsibilities|Services|Duties|Roles)\s+(?:of|[Pp]rovided\s+by|[Oo]ffered\s+by)\s+(?:the\s+)?(?P<name>.+?)\s*:$",
                   r"\b(?:services|support)\s+(?:offered|provided)\s+by\s+(?:the\s+)?(?P<name>.+?),?\s+including\s*:$")
_ANAPHOR = re.compile(r"^(?:(?:This|The)\s+(?P<head>Council|council|Office|office|Department|department|Center|center|Centre|centre|Section|section|Unit|"
                      r"unit|Committee|committee|College|college|Institute|institute|Library|library)(?:’s|'s)?|(?P<it>It|Its))\s+")
#: "University" followed by another capitalised word is the start of another name ("University Dental Hospital", "University
#: Research Board", "the University Academic Advisory Council"), never the University itself.
_UNIVERSITY_SUBJECT = re.compile(r"^(?:(?:the|The)\s+)?(?:(?:organi[sz]ation|governance\s+structures?)\s+of\s+the\s+)?(?:University(?:\s+of\s+Sharjah)?|UoS)"
                                 r"(?:’s|'s)?\s+(?![A-Z][a-z])|^(?P<voice>we|We|There\s+(?:are|is))\s+")
#: "we" and "There are" stand for the University only when the same sentence names it; otherwise no subject is read.
_UNIVERSITY_NAMED = re.compile(r"\bUniversity\b(?!\s+[A-Z][a-z])|\bUniversity\s+of\s+Sharjah\b|\bUoS\b")
_OTHER_SUBJECT = re.compile(r"^(?:(?:the|The)\s+)?(?:[A-Z][\w’'&\-]*\s+){0,6}(?:[A-Z]{2,6}|Office|Section|Unit|Center|Centre|Department|Directorate|"
                            r"Institute|Committee|staff|team)(?:\s+\([A-Z]{2,6}\))?\s+")
_SUBJECT_SUFFIX = {"section": r"\s+Section"}
#: A subject outside the lexicon counts as a unit only when it names one ("Office", "Section", "Directorate", an acronym ...).
_UNIT_NOUN = re.compile(r"\b(?:Office|Section|Unit|Center|Centre|Department|Directorate|Council|Committee|Institute|Deanship|Board|"
                        r"Chancellor|President|Dean|staff|team|[A-Z]{2,6})\b")
_ABBREVIATION = re.compile(r"(?:\b(?:No|Nos|Dr|Mr|Mrs|Ms|Prof|St|vs|Vol|Fig)|\be\.g|\bi\.e|\bU\.S)\.$")
_BARE_MARK = re.compile(r"^(?:\d{1,2}|[a-h])\.$")
_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9•§“\"(])"                                                       # a sentence end before a capital, digit or bullet
    r"|(?<=[.!?])\s+(?=[a-h]\.\s+[A-Z])"                                                       # a lettered item after a sentence end
    r"|\s+(?=[•§]\s)"                                                                          # a bullet
    r"|\s+(?=\d+\.\d+(?:\.\d+)*\s+[A-Z])"                                                      # a section heading ("1.3 Organizational Chart")
    r"|(?<=[^\d.,\s])\s+(?=\d{1,2}\.\s+[A-Z])"                                                   # a numbered item after a word ("(EHSM) 2. Facilities")
    r"|(?<=[:;.])\s+(?=o\s)"                                                                   # an "o" sub-item
    r"|(?<=:)\s+(?=\d{1,2}\.\s|[a-h]\.\s)"                                                     # a numbered or lettered item after a colon
    r"|(?<=:)\s+(?=(?:[A-Z][\w’'&,\-]*\s+){0,6}[A-Z][\w’'&\-]*(?:\s*\([A-Z]{1,5}\d{0,3}\))?:\s)")  # an entry title ("Main Library (W16): ")


@dataclass(frozen=True)
class Unit:
    """One sentence, list item, lead-in or entry of a paragraph chunk. ``start``/``end`` index the chunk text."""

    start: int
    end: int
    text: str
    item: bool          # starts with a list marker (•, o, §, "1.", "a.")
    lead_in: bool       # ends with ":"


@lru_cache(maxsize=16384)
def split_units(text: str) -> Tuple[Unit, ...]:
    """The units of a paragraph chunk, in order, with their exact spans. A "Page N" header is left out of the span."""
    text = text or ""
    cuts = [0] + [m.end() for m in _BOUNDARY.finditer(text)] + [len(text)]
    raw: List[Tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        seg = text[a:b]
        lead = len(seg) - len(seg.lstrip())
        seg_end = a + len(seg.rstrip())
        if seg_end > a + lead:
            raw.append((a + lead, seg_end))
    merged: List[Tuple[int, int]] = []
    for s, e in raw:                       # "Decision No. 299", "e.g. administering", a bare "1.": not a unit end
        prev = text[merged[-1][0]:merged[-1][1]] if merged else ""
        if merged and (_ABBREVIATION.search(prev) or _BARE_MARK.match(prev)):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    units: List[Unit] = []
    for s, e in merged:
        m = _PAGE_PREFIX.match(text[s:e])
        if m:
            s += m.end()
        if s >= e:
            continue
        seg = text[s:e]
        units.append(Unit(s, e, seg, bool(_ITEM_MARK.match(seg)), seg.rstrip().endswith(":")))
    return tuple(units)


def _subject_patterns() -> List[Tuple[int, str, "re.Pattern[str]"]]:
    out: List[Tuple[int, str, "re.Pattern[str]"]] = []
    for office in OFFICES:
        for name in _names(office) + [alias for alias, _pages in office.subject_aliases]:
            body = _raw_name_pattern(name)
            if office.needs_office_noun and name == office.name:
                body = body + r"\s+" + _OFFICE_NOUN
            flags = 0 if _is_acronym(name) else re.I
            out.append((len(name), office.office_id, re.compile(r"(?:(?:the|The)\s+)?" + body + r"(?![\w])", flags)))
    out.sort(key=lambda t: (-t[0], t[1]))
    return out


_SUBJECT_PATTERNS = _subject_patterns()


def _core_offset(text: str, item: bool) -> int:
    """Where the statement proper starts inside a unit: after a list marker and its title ("5. Service Help Desk: "), a
    glued section heading ("13.1 Role of the Health and Safety Office "), and one leading adverbial clause ("To achieve
    this, ", "In line with its mandate, ")."""
    pos = 0
    m = _ITEM_MARK.match(text)
    if m:
        pos = m.end()
        t = _ITEM_TITLE.match(text[pos:])
        if t:
            pos += t.end()
    elif _ITEM_TITLE.match(text) and re.match(r"^[A-Z][^:.]{0,90}:\s+(?:The|the)\s", text):
        pos = _ITEM_TITLE.match(text).end()
    n = _NUMBERED_HEADING.match(text[pos:])
    if n:                                  # "11.4 Library Staff Library staff ...": the heading is a lexicon name
        after = pos + n.end()
        for _length, _office_id, pattern in _SUBJECT_PATTERNS:
            m = pattern.match(text[after:])
            if m and text[after + m.end():after + m.end() + 1] == " ":
                pos = after + m.end() + 1
                break
        else:
            h = _HEADING.match(text[pos:])
            if h:
                pos += h.end()
    else:
        h = _HEADING.match(text[pos:])
        if h:
            pos += h.end()
    a = _ADVERBIAL.match(text[pos:])
    if a:
        pos += a.end()
    return pos


@dataclass(frozen=True)
class Reading:
    """What one unit states. ``relation`` is None when no recognised relation follows the subject."""

    subject_kind: Optional[str]          # SUBJECT_OFFICE | SUBJECT_ANAPHOR | SUBJECT_UNIVERSITY | SUBJECT_OTHER | None
    office_id: Optional[str]             # for SUBJECT_OFFICE
    anaphor_head: Optional[str]          # for SUBJECT_ANAPHOR: the head noun, lowercased, or "it"
    subject_text: str                    # the subject as printed
    relation: Optional[str]              # REL_RESPONSIBILITY | REL_CHAIR | REL_COMPOSITION | None
    any_duty: bool                       # some duty verb follows the subject (wider list)
    core_start: int                      # offset of the statement proper inside the unit
    object_start: int = -1               # offset inside the unit where the object of the relation starts (-1: none)


def _relation_at(rest: str) -> Tuple[Optional[str], bool, int]:
    """(relation, any duty verb, offset in ``rest`` where the relation's object starts or -1)."""
    lead = re.match(r"^\s*(?:(?:also|has\s+also|has|have)\s+)?", rest)
    pos = lead.end()
    body = rest[pos:]
    for relation, rx in ((REL_CHAIR, _CHAIR_RX), (REL_COMPOSITION, _COMPOSITION_RX), (REL_RESPONSIBILITY, _RESPONSIBILITY_RX),
                         (REL_RESPONSIBILITY, _APPOSITIVE_DUTY)):    # appositive: "is one of the largest units ..., dedicated to ..."
        m = rx.match(body)
        if m:
            return relation, True, pos + m.end()
    m = _ANY_DUTY_RX.match(body)
    return None, bool(m), (pos + m.end()) if m else -1


@lru_cache(maxsize=65536)
def read_unit(text: str, item: bool = False, *, subject_anywhere: bool = False) -> Reading:
    """The subject and relation of one unit. The subject must open the statement proper (see ``_core_offset``); an
    office named later in the sentence ("... and the Deanship of Student Affairs to deliver ...", "the Vice Chancellor
    for Public Relations is ...") is not its subject. ``subject_anywhere`` (used only to show what the rule prevents)
    accepts the first office named anywhere in the unit."""
    core = _core_offset(text, item)
    rest = text[core:]
    if subject_anywhere:
        spans = office_spans(text)
        if spans:
            _s, _e, office_id = spans[0]
            found = [(m.end(), rel) for rel, rx in ((REL_CHAIR, _CHAIR_RX), (REL_COMPOSITION, _COMPOSITION_RX), (REL_RESPONSIBILITY, _RESPONSIBILITY_RX))
                     for m in [rx.search(text)] if m]
            relation, obj = (found[0][1], found[0][0]) if found else (None, -1)
            return Reading(SUBJECT_OFFICE, office_id, None, _normalise(text)[_s:_e], relation, relation is not None, core, obj)
    office_reading = _office_subject(rest, core)
    if office_reading is not None and office_reading.relation is None:
        # A glued heading that is itself the office name ("... Section (EHSM) Established in 2022, the EHSM Section is
        # dedicated to ..."): read the statement after one adverbial clause, for the same office only. Step 4.16-C1: the
        # adverbial may be absent ("Projects Management Section The Projects Management Section is responsible for ...").
        tail = rest[len(office_reading.subject_text):].lstrip()
        offset = len(rest) - len(tail)
        a = _ADVERBIAL.match(tail)
        skip = a.end() if a else 0
        if a or re.match(r"(?:the|The)\s", tail):
            again = _office_subject(tail[skip:], core + offset + skip)
            if again is not None and again.office_id == office_reading.office_id and again.relation is not None:
                return again
    if office_reading is not None:
        return office_reading
    a = _ANAPHOR.match(rest)
    if a:
        head = (a.group("head") or "it").lower()
        after = rest[a.end():]
        if a.group("it") == "Its" and not re.match(r"(?:primary\s+)?(?:mission|role|mandate|purpose)\s+is\s+to\b", after):
            return Reading(None, None, None, "", None, False, core)
        relation, duty, obj = _relation_at(after)
        return Reading(SUBJECT_ANAPHOR, None, head, rest[:a.end()].strip(), relation, duty, core, core + a.end() + obj if obj >= 0 else -1)
    u = _UNIVERSITY_SUBJECT.match(rest)
    if u and u.group("voice") and not _UNIVERSITY_NAMED.search(rest[u.end():]):
        return Reading(None, None, None, "", None, False, core)
    if u:
        relation, duty, obj = _relation_at(rest[u.end():])
        return Reading(SUBJECT_UNIVERSITY, UNIVERSITY, None, rest[:u.end()].strip(), relation, duty, core, core + u.end() + obj if obj >= 0 else -1)
    o = _OTHER_SUBJECT.match(rest)
    if o:
        relation, duty, obj = _relation_at(rest[o.end():])
        if relation is not None or duty:
            return Reading(SUBJECT_OTHER, None, None, rest[:o.end()].strip(), relation, True, core, core + o.end() + obj if obj >= 0 else -1)
    v = _ANY_DUTY_RX.search(rest)           # "The Vice Chancellor Office for Institutional Effectiveness ... supervises ..."
    if v and 0 < v.start() <= 140:
        subject = rest[:v.start()].strip()
        if re.match(r"^(?:(?:the|The)\s+\w|[A-Z])", subject) and "," not in subject and len(subject.split()) <= 14 \
                and _UNIT_NOUN.search(subject):
            relation, _duty, obj = _relation_at(rest[v.start():])
            return Reading(SUBJECT_OTHER, None, None, subject, relation, True, core, core + v.start() + obj if obj >= 0 else -1)
    return Reading(None, None, None, "", None, False, core)


def _office_subject(rest: str, core: int) -> Optional[Reading]:
    """A lexicon office opening ``rest`` (with an optional "(ACRONYM)", a printed "Section" for a section, and "at the
    University of Sharjah"), and the relation that follows it."""
    for _length, office_id, pattern in _SUBJECT_PATTERNS:
        m = pattern.match(rest)
        if not m:
            continue
        end = m.end()
        acr = re.match(r"\s*\([A-Z]{2,6}\)", rest[end:])
        if acr:
            end += acr.end()
        office = OFFICES_BY_ID[office_id]
        suffix = _SUBJECT_SUFFIX.get(office.kind)
        if suffix:
            s = re.match(suffix + r"(?:\s*\([A-Z]{2,6}\))?", rest[end:])
            if s:
                end += s.end()
        loc = re.match(r"\s+at\s+the\s+University\s+of\s+Sharjah", rest[end:])
        if loc:
            end += loc.end()
        relation, duty, obj = _relation_at(rest[end:])
        return Reading(SUBJECT_OFFICE, office_id, None, rest[:end].strip(), relation, duty, core, core + end + obj if obj >= 0 else -1)
    return None


def lead_in_office(text: str) -> Tuple[Optional[str], Optional[str]]:
    """For a lead-in that names its office in a frame ("Key Functions of the Office of International Relations:",
    "... services offered by the ITC, including:"): (office id or None, the name as printed)."""
    for frame in _LEAD_IN_FRAMES:
        m = re.search(frame, text.strip())
        if not m:
            continue
        name = m.group("name")
        for _length, office_id, pattern in _SUBJECT_PATTERNS:
            if pattern.fullmatch(name.strip()):
                return office_id, name
        return None, name
    return None, None


def head_matches(head: str, office_id: str, mention_text: str = "") -> bool:
    """Whether an anaphor's head noun ("council", "office", "department" ...) can refer to the office: the noun is part
    of the office's printed name, or printed right after its mention ("the EHSM Section"). "It" matches any office."""
    if head == "it":
        return True
    office = OFFICES_BY_ID[office_id]
    if head in office.name.lower().split():
        return True
    if office.kind == head:
        return True
    for name in _names(office) + [alias for alias, _pages in office.subject_aliases]:
        if re.search(_raw_name_pattern(name) + r"(?:\s*\([A-Z]{2,6}\))?\s+" + head + r"\b", mention_text, re.I):
            return True
    return False


# ---- duty terms -----------------------------------------------------------
#: Who a duty serves, never the duty itself ("Who is responsible for faculty?" names no duty).
BENEFICIARIES = frozenset("""faculty faculties staff student students member members employee employees people individual individuals researcher
researchers user users visitor visitors everyone community""".split())
_STOP = frozenset("""a an the of for to in on at by with and or about regarding concerning into from as is are was were be been being do does did
can could should would will shall may might must who whom whose which what where when how why i me my we our us you your it its this that these
those there their they them his her he she any all some each every please tell know need want get give one ones also other such""".split())
#: Words of the question frame and fillers that never name the asked duty.
_FRAME = frozenset("""office offices department departments unit units section sections center centre centers centres body bodies responsible
responsibility responsibilities handle handles handling handled manage manages managing managed oversee oversees overseeing provide provides
providing provided deal deals dealing take takes care charge contact contacts ask asking talk speak reach call go see report
reporting turn person people team university uos sharjah campus campuses matters matter issues issue questions question requests request
problems problem things thing related following above aforementioned assist assists assisting assistance help helps helping serve serves
serving""".split())
_TERM = re.compile(r"[A-Za-z][A-Za-z’'\-]*|\d[\d,]*")


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    """A crude stem: one plural, -ing or -ed ending, then an -ation ending, a final silent "e" and the "-at" of "-ate", so
    that "coordinate", "coordinating" and "coordination", "response" and "responses", "implementing" and
    "implementation" meet (phase B3 review). "management" and "manages" stay apart."""
    w = word.lower().replace("’", "'")
    w = re.sub(r"'s$", "", w)
    if w.endswith("ees") or w.endswith("ee"):
        return w[:-1] if w.endswith("ees") else w        # "committees" / "committee" never meet "committed"; "trainees" not "training"
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and not (suffix == "s" and w.endswith("ss")):
            w = w[: len(w) - len(suffix)] + repl
            break
    for tail in ("ation", "ition"):
        if w.endswith(tail) and len(w) - len(tail) >= 4:
            w = w[: len(w) - len(tail)]
            break
    if w.endswith("e") and len(w) >= 4:
        w = w[:-1]
    if w.endswith("at") and len(w) >= 6:
        w = w[:-2]
    return w


#: Unit nouns: a frame word alone ("which office"), but part of the duty right after a content word ("the Data Center").
_UNIT_WORDS = frozenset("center centre centers centres office offices department departments unit units section sections desk desks".split())


def duty_terms(text: str) -> Tuple[str, ...]:
    """The content words of an asked duty, in order, without question-frame words, fillers or office names. "IT" (the
    printed abbreviation) is kept as the term "IT"; the pronoun "it" is a stop word. A unit noun right after a content
    word stays ("the Data Center" -> data, center)."""
    masked = _masked(text)
    out: List[str] = []
    prev_content = False
    for tok in _TERM.findall(masked):
        if tok == "IT":
            if "IT" not in out:
                out.append("IT")
            prev_content = True
            continue
        for p in (p for p in re.split(r"-", tok) if p):
            low = re.sub(r"'s$", "", p.lower().replace("’", "'"))
            keep = low not in _STOP and len(low) >= 2 and (low not in _FRAME or (low in _UNIT_WORDS and prev_content))
            if keep and low not in out:
                out.append(low)
            prev_content = keep
    return tuple(out)


def _tokens(text: str) -> List[str]:
    return [t for tok in _TERM.findall(text or "") for t in re.split(r"[-/]", tok) if t]


def covers(terms: Sequence[str], text: str, *, all_terms: bool = True, prefix: bool = True) -> bool:
    """Whether a statement states the asked duty: every term (or, with ``all_terms=False``, any term) matches a word
    of the text by stem, or (unless ``prefix=False``) by a shared prefix of at least five letters. "IT" matches only
    the printed "IT" or "Information Technology"."""
    if not terms:
        return False
    toks = _tokens(text)
    stems = {_stem(t) for t in toks}
    upper = set(toks)
    low_text = (text or "").lower()

    def one(term: str) -> bool:
        if term == "IT":
            return "IT" in upper or "information technology" in low_text
        st = _stem(term)
        if st in stems:
            return True
        return prefix and any(len(s) >= 5 and len(st) >= 5 and (s.startswith(st) or st.startswith(s)) for s in stems)

    results = [one(t) for t in terms]
    return all(results) if all_terms else any(results)


def stem_set(text: str) -> frozenset:
    """The stems of a text's words, for repeated exact-stem checks ("IT" when printed, or for "Information Technology")."""
    toks = _tokens(text)
    out = {_stem(t) for t in toks}
    if "IT" in toks or "information technology" in (text or "").lower():
        out.add("IT")
    return frozenset(out)


def covers_stems(terms: Sequence[str], stems: frozenset) -> bool:
    """``covers(terms, text, prefix=False)`` on the precomputed ``stem_set(text)``."""
    return bool(terms) and all(("IT" if t == "IT" else _stem(t)) in stems for t in terms)


def without_offices(text: str) -> str:
    """The text with every office name blanked out (office-name words are no evidence of a duty)."""
    return _masked(text)


# ---- location entries -------------------------------------------------------
_PLACE = re.compile(r"\b(?:located\s+)?(?:on|at)\s+(the\s+[^.]+?)\s*(?:\.|$)")
_CODE = re.compile(r"\(([A-Z]{1,2}\d{1,3})\)")


@dataclass(frozen=True)
class LocationEntry:
    """A location entry as printed: "<Name> (<CODE>): <description> ... on|at <place>."."""

    office_id: str
    label: str              # the entry title as printed, e.g. "Main Library (W16)"
    code: Optional[str]     # the building code printed after the name, if any
    place: Optional[str]    # the place phrase as printed, e.g. "the University's Women Campus (W16)"


_PLACE_NOUN = re.compile(r"\b(?:Campus|College|Complex|Building)\b")


def location_entry(text: str) -> Optional[LocationEntry]:
    """A unit that opens with a library's printed name (the 11.6 entries), an optional building code and a colon, and
    states where it is with a place noun ("on the ... Campus", "at the ... College"). Anything else is no location
    entry: another kind of unit, a missing place, or an "at the ..." phrase that names no place."""
    for _length, office_id, pattern in _SUBJECT_PATTERNS:
        if OFFICES_BY_ID[office_id].kind != "library":
            continue
        m = re.match(r"(?:" + pattern.pattern + r")(?P<code>\s*\([A-Z]{1,2}\d{1,3}\))?:\s+", text, pattern.flags)
        if not m:
            continue
        place = _PLACE.search(text[m.end():])
        if not place or not _PLACE_NOUN.search(place.group(1)):
            return None
        code = _CODE.search(m.group("code") or "")
        return LocationEntry(office_id, text[: m.end()].rstrip().rstrip(":").strip(), code.group(1) if code else None, place.group(1).strip())
    return None


# ---- duty phrases (phase B3, after the adversarial review) -----------------------------------------------------------
#: Where the object of a relation ends: punctuation, or a word that starts a complement, purpose or circumstance.
_OBJECT_CUT = re.compile(r"\s*[,;:.(]|\s+(?:to|that|which|who|whose|where|while|so|ensuring|including|across|through|by|from|within|via|using|"
                         r"in\s+order|in\s+coordination|in\s+accordance|in\s+line|with\s+a|for\s+both|for\s+all|ranking|"
                         r"accommodating|as\s+well\s+as|and\s+(?:to|is|are|was|has|have|will|can|continuously)|enhancing|providing|"
                         r"offering|delivering|contributing|helping|supporting|preparing)\b", re.I)
#: A leading gerund that only introduces its object ("providing mandatory training ..."): it is not a duty word itself.
_INTRO_GERUNDS = frozenset({"providing", "offering", "delivering", "extending", "furnishing"})
#: Words that never name a duty in a phrase or an ask (places, the university itself).
_FILLER = frozenset("university uos sharjah campus campuses community all various following above".split())
#: Modifiers that only grade a duty ("a cutting-edge Data Center", "mandatory training"); any other word before a
#: one-word duty narrows it ("infrastructure needs", "safety awareness", "several programs").
_PLAIN_MODIFIERS = frozenset("""cutting edge high quality comprehensive essential advanced modern regular mandatory robust efficient effective
specialized specialised necessary appropriate relevant continuous ongoing wide broad overall general full timely reliable proper dedicated new
functions aspects operations activities one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen""".split())


_PREPOSITION_START = re.compile(r"^(?:with|to|for|in|on|at|from|across|through|between|among|into|onto|towards?|via|by|of|about|after|"
                                r"before|during|under|over|within|without)\b", re.I)


def item_title(text: str) -> Optional[str]:
    """The title of a list item ("• Risk Assessment and Management: Conducting ..." -> "Risk Assessment and Management")."""
    m = _ITEM_MARK.match(text or "")
    if not m:
        return None
    t = re.match(r"([A-Z][^:.]{0,90}):\s", text[m.end():])
    return t.group(1).strip() if t else None


def _cut(phrase: str) -> str:
    m = _OBJECT_CUT.search(phrase)
    if not m:
        return phrase.strip()
    head = phrase[: m.start()]
    if re.search(r"\b(?:in|for|of|on|to|with|at|about|into|from)\s*$", head, re.I) or \
            re.match(r"\s*(?:from|by|within|across|through|via)\b", phrase[m.start():], re.I):
        # the cut word opens the complement of a preposition ("instruction in using the library resources"): the
        # complement limits the duty, so it is kept, up to the next punctuation
        p = re.search(r"[,;:.(]", phrase[m.end():])
        return (phrase[: m.end() + p.start()] if p else phrase).strip()
    return head.strip()


#: A verb coordinated with the relation verb of a statement (", and handle ...", ", evaluate ...").
_COORDINATED_VERB = re.compile(r",\s*(?:and\s+)?(?:also\s+)?(?=(?:handles?|evaluates?|provides?|manages?|maintains?|oversees?|coordinates?|conducts?|"
                               r"organi[sz]es?|delivers?|supports?|offers?|issues?|operates?|administers?|supervises?|implements?|"
                               r"develops?|monitors?|investigates?|arranges?)\s+)", re.I)


@lru_cache(maxsize=65536)
def duty_phrases(text: str, item: bool, reading: "Reading") -> Tuple[str, ...]:
    """The duty phrases a unit names, verbatim: a list item's title and the name in brackets in it (or, without a title,
    the start of its body), and the object right after the relation verb of a statement. Nothing else of the sentence
    names the duty: its circumstances, purpose, beneficiaries and examples do not."""
    out: List[str] = []
    if item:
        title = item_title(text)
        if title:
            out.append(title)
            for inner in re.findall(r"\(([^)]+)\)", title):
                out.append(inner)
            plain = re.sub(r"\s*\([^)]*\)", "", title).strip()
            if plain and plain != title:
                out.append(plain)
        elif reading is None or reading.subject_kind is None:   # "o OIR is committed to ...": the subject and verb are no duty
            m = _ITEM_MARK.match(text)
            body = _cut(text[m.end():]) if m else ""
            if body:
                out.append(body)
    if reading is not None and reading.object_start >= 0:
        phrase = _cut(text[reading.object_start:].lstrip())
        if phrase:
            out.append(phrase)
        if reading.relation is not None:
            # the objects of verbs coordinated with the relation verb ("DRC provide support and counseling, evaluate the
            # quality of services, and handle complaints ...") are duties of the same subject
            for m in _COORDINATED_VERB.finditer(text, reading.object_start):
                more = _cut(text[m.end():])
                if more:
                    out.append(more)
    # a phrase that opens with a preposition ("with colleges and departments", "From the female dormitory to the
    # colleges") names a partner, a place or a route, not a duty; one made only of who is served ("faculty and staff
    # in") or of a placeholder ("the following") names none
    def names_duty(p: str) -> bool:
        words = [w for w in _phrase_terms(p) if w.lower() not in _FRAME]
        return bool(words) and not all(w.lower() in BENEFICIARIES for w in words)

    return tuple(dict.fromkeys(p for p in out if p and not _PREPOSITION_START.match(p) and names_duty(p)))


def _phrase_terms(phrase: str, *, mask: bool = True) -> List[str]:
    text = _masked(phrase) if mask else _normalise(phrase)
    out: List[str] = []
    for tok in _TERM.findall(text):
        for part in re.split(r"[-/]", tok):
            if not part:
                continue
            if part == "IT":
                out.append("IT")
                continue
            low = re.sub(r"'s$", "", part.lower().replace("’", "'"))
            if low in _STOP or low in _FILLER or len(low) < 2:
                continue
            out.append(low)
    return out


def _stems(words: Sequence[str]) -> List[str]:
    return ["IT" if w == "IT" else _stem(w) for w in words]


_PREPOSITIONS = frozenset("in of for on to with at about into from by via through within across among between under over".split())
#: Words that only introduce a duty object ("providing mandatory training", "deliver high-quality services").
_LEAD_WORDS = frozenset("""providing offering delivering extending furnishing provide provides offer offers deliver delivers extend extends
furnish furnishes""".split())


def _conjunct_words(conjunct: str, asked: set, after_prep: bool = False) -> List[Tuple[str, bool]]:
    """(word, after a preposition) for the content words of one conjunct, in order: stop words, fillers, frame words
    (unit nouns aside) and unasked beneficiary words are dropped. ``after_prep`` carries a preposition of an earlier
    conjunct ("... for University administration and academic activities": both are inside the complement)."""
    out: List[Tuple[str, bool]] = []
    for tok in _TERM.findall(conjunct):
        for part in (p for p in re.split(r"[-/]", tok) if p):
            if part == "IT":
                out.append(("IT", after_prep))
                continue
            low = re.sub(r"'s$", "", part.lower().replace("’", "'"))
            if low in _PREPOSITIONS:
                after_prep = True
                continue
            if low in _STOP or low in _FILLER or len(low) < 2 or (low in _FRAME and low not in _UNIT_WORDS):
                continue
            if low in BENEFICIARIES and _stem(low) not in asked and after_prep:
                continue                   # "training for faculty": who is served; "user support" keeps "user"
            out.append((low, after_prep))
    return out


def _effective_conjuncts(phrase: str, asked: set, mask: bool) -> Tuple[Tuple[Tuple[str, bool], ...], ...]:
    return _effective_conjuncts_cached(phrase, frozenset(asked), mask)


_PREPOSITION_RX = re.compile(r"\b(?:%s)\b" % "|".join(sorted(_PREPOSITIONS)), re.I)


@lru_cache(maxsize=65536)
def _effective_conjuncts_cached(phrase: str, asked: frozenset, mask: bool) -> Tuple[Tuple[Tuple[str, bool], ...], ...]:
    """The conjuncts of a duty phrase with their shared words: in "Health and Safety Training" the one-word "Health"
    shares the head "Training"; in "Risk Assessment and Management" the one-word "Management" shares the modifier "Risk"."""
    text = _masked(phrase) if mask else _normalise(phrase)
    raw, after_prep = [], False
    for c in re.split(r",|\band\b|\bor\b|&", text):
        cw = _conjunct_words(c, asked, after_prep)
        after_prep = after_prep or any(ap for _w, ap in cw) or bool(_PREPOSITION_RX.search(c))
        if cw:
            raw.append(cw)
    out = []
    for k, cw in enumerate(raw):
        if len(cw) == 1 and k < len(raw) - 1 and len(raw[-1]) > 1:
            out.append(cw + [raw[-1][-1]])
        elif len(cw) == 1 and k == len(raw) - 1 and len(raw) > 1:
            multi = next((c for c in raw[:-1] if len(c) > 1), None)
            out.append((multi[:-1] if multi else []) + cw)
        else:
            out.append(cw)
    return tuple(tuple(c) for c in out)


def phrase_matches(terms: Sequence[str], phrase: str, *, extra: int = 1, all_terms: bool = True, mask: bool = True,
                   verb: Optional[str] = None) -> bool:
    """Whether a duty phrase names the asked duty. Every asked term matches a word of the phrase by stem (no prefix
    guessing), and in every conjunct that holds asked words: the conjunct's head (last word) is asked, so no unasked head
    or complement narrows the duty ("the University's infrastructure needs" is no "needs", "instruction in using the
    library resources" no "library resources", "a cutting-edge Data Center" no "data"); every word before the first asked
    one only grades it (cutting-edge, high-quality, mandatory ...); at most ``extra`` unasked word stands between asked
    ones, never after an asked gerund or verb ("promote safety awareness" is no "promoting awareness") and never after a
    preposition. A leading word that only introduces the object (providing, deliver ...) or that is the asked verb is
    skipped, and a phrase that opens with another gerund ("aligning with ...") is matched only when the ask uses it.
    ``all_terms=False`` (only to show what the rule prevents) accepts any one shared word."""
    if not terms:
        return False
    tst = set(_stems([t if t == "IT" else t.lower() for t in terms]))
    verb_like = {_stem(t) for t in terms if t.lower().endswith("ing")}
    conj = _effective_conjuncts(phrase, tst, mask)

    def lead(w: str) -> bool:
        return (w in _LEAD_WORDS or w in _INTRO_GERUNDS or (verb is not None and _stem(w) == _stem(verb))) and _stem(w) not in tst

    conj = [cw[1:] if cw and lead(cw[0][0]) else cw for cw in conj]
    conj = [cw for cw in conj if cw]
    stems_all = {("IT" if w == "IT" else _stem(w)) for cw in conj for w, _ in cw}
    if not conj:
        return False
    if not all_terms:
        return bool(tst & stems_all)
    if not tst <= stems_all:
        return False
    printed_first = (re.findall(r"[A-Za-z’'\-]+", phrase) or [""])[0].lower()
    if printed_first.endswith("ing") and "-" not in printed_first and len(printed_first) > 5 and printed_first not in _INTRO_GERUNDS \
            and _stem(printed_first) not in tst and not (verb and _stem(printed_first) == _stem(verb)):
        return False                       # "aligning with the goals" names an aligning, not the goals ("cutting-edge" is no gerund)
    covered = set()
    for cw in conj:
        st = ["IT" if w == "IT" else _stem(w) for w, _ in cw]
        idx = [i for i, x in enumerate(st) if x in tst]
        if not idx or idx[-1] != len(cw) - 1:
            continue
        first, last = idx[0], idx[-1]
        if cw[first][1]:
            continue                       # the asked words sit inside a complement ("infrastructure for ... academic activities")
        if any(w not in _PLAIN_MODIFIERS for w, _ in cw[:first]):
            continue
        interior = [cw[i] for i in range(first + 1, last) if st[i] not in tst]
        if interior and (st[first] in verb_like or cw[first][0].endswith("ing") or len(interior) > extra
                         or any(ap for _w, ap in interior)):
            continue
        covered |= {st[i] for i in idx}
    return tst <= covered


def phrase_narrows(terms: Sequence[str], phrase: str, *, mask: bool = True, complements: bool = False, verb: Optional[str] = None) -> bool:
    """Whether a duty phrase names a narrower duty of the asked one: every asked word (two or more) modifies an unasked
    head, with nothing before them that governs them ("Health and Safety Training" for "health and safety"); with
    ``complements``, also the asked duty followed only by a complement that restricts it ("handle complaints from
    individuals with disabilities" for "handling complaints"). Used only to give the office as a partial candidate,
    never as a supported answer."""
    tst = set(_stems([t if t == "IT" else t.lower() for t in terms]))
    if len(tst) < 2:
        return False                       # one generic word ("maintenance") may be another sense of the printed duty
    covered = set()
    for cw in _effective_conjuncts(phrase, tst, mask):
        if cw and (cw[0][0] in _LEAD_WORDS or cw[0][0] in _INTRO_GERUNDS or (verb is not None and _stem(cw[0][0]) == _stem(verb))) \
                and _stem(cw[0][0]) not in tst:
            cw = cw[1:]
        st = ["IT" if w == "IT" else _stem(w) for w, _ in cw]
        idx = [i for i, x in enumerate(st) if x in tst]
        if not idx or cw[idx[0]][1] or any(not (st[i] in tst or cw[i][0] in _PLAIN_MODIFIERS) for i in range(idx[-1] + 1)):
            continue
        tail = cw[idx[-1] + 1:]
        if not tail:
            continue
        if complements and all(ap for _w, ap in tail):
            covered |= {st[i] for i in idx}
        elif len(tst) >= 2 and not any(ap for _w, ap in cw[:idx[-1] + 1]):
            covered |= {st[i] for i in idx}
    return tst <= covered


_BENEFICIARY_NP = (r"(?:(?:the|all|our|my|its|their|both|new)\s+)*(?:faculty|staff|students?|members?|employees?|everyone|researchers?|users?|"
                   r"visitors?|people|individuals)(?:\s+members?)?")
#: A beneficiary phrase of an ask ("... to faculty", "... for students and staff"): cut, and only it, so what follows it
#: ("... with disabilities", "... in English", "... for laboratory equipment") still names the duty (phase B3 review).
_ASK_QUALIFIER = re.compile(r"\s+(?:for|to|of)\s+%s(?:\s*(?:,\s*(?:and|or|&)?|and|or|&)\s*%s)*(?![\w-])" % (_BENEFICIARY_NP, _BENEFICIARY_NP),
                            re.I)
#: The setting of an ask ("... on campus", "... at the university"): no part of the duty.
_ASK_SETTING = re.compile(r"\s+(?:on|at|in|within|across|throughout|around)\s+(?:the\s+)?(?:campus(?:es)?|university(?:\s+of\s+sharjah)?|uos)(?![\w-])",
                          re.I)


def ask_terms(obj: str) -> Tuple[str, ...]:
    """The asked duty's content words, after cutting the beneficiary phrases and the setting."""
    return duty_terms(_ASK_QUALIFIER.sub(" ", _ASK_SETTING.sub(" ", " " + (obj or ""))))


def ask_residue(obj: str) -> Tuple[str, ...]:
    """The words of an asked object after its beneficiaries and setting, frame words included ("the centers" ->
    "centers"): an object with a residue but no duty term names nothing that can be looked up."""
    text = _ASK_QUALIFIER.sub(" ", _ASK_SETTING.sub(" ", " " + (obj or "")))
    return tuple(w for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", _masked(text)) if w.lower() not in _STOP and w.lower() not in BENEFICIARIES)


# ---- other units a clause names (phase B3) ----------------------------------------------------------------------------
_UNIT_HEAD = re.compile(r"\b(?:cent(?:er|re)s?|sections?|departments?|councils?|committees?|units?|offices?|divisions?|institutes?|"
                        r"librar(?:y|ies)|staff|teams?|boards?|schools?|programs?|programmes?|deanships?|directorates?|clinics?|labs?|"
                        r"laborator(?:y|ies)|hospitals?|sub-?committees?|panels?|working\s+groups?|task\s*forces?|groups?|"
                        r"representatives?|secretariats?|forums?|commissions?|jur(?:y|ies)|assembl(?:y|ies)|bod(?:y|ies)|caucus(?:es)?|"
                        r"bureaus?)\b", re.I)
_SAME_UNIT = re.compile(r"^(?:office|offices|department|unit|section|desk|cent(?:er|re))$", re.I)
_KNOWN_ACRONYMS = frozenset({"IT", "UOS", "UoS", "UAE", "PHD", "PhD", "MSC", "BSC", "HR"})


def other_unit_mentions(clause: str) -> Tuple[str, ...]:
    """Unit nouns and capitalised acronyms a clause uses outside every lexicon office name ("the Writing Center of the
    Language Institute", "the UAAC and the College Council", "the College of Medicine library"). A plain office noun
    printed right after an office name ("the HR department", "the Registration office") is the same unit."""
    norm = _normalise(clause)
    spans = office_spans(norm)
    chars = list(norm)
    for s, e, _o in spans:
        chars[s:e] = " " * (e - s)
    masked = "".join(chars)
    out: List[str] = []
    for m in _UNIT_HEAD.finditer(masked):
        if _SAME_UNIT.match(m.group(0)) and any(e <= m.start() and not norm[e:m.start()].strip() for _s, e, _o in spans):
            continue
        out.append(m.group(0).lower())
    for m in re.finditer(r"\b[A-Z]{2,6}\b", masked):
        if m.group(0) not in _KNOWN_ACRONYMS:
            out.append(m.group(0))
    return tuple(dict.fromkeys(out))



__all__ = ["ASK_KINDS", "AUTHORITY", "CHANNEL", "CONTACT_FIELDS", "E1", "E2", "E3", "EMAIL", "EXTENSION", "FACT", "FAX", "GENERIC_FIELDS",
           "LOCATION", "LocationEntry", "NOT_EQUIVALENT", "NUMBER", "OFFICES", "OFFICES_BY_ID", "OFFICE_ROLE", "OTHER", "Office", "PROCEDURE",
           "REFERRAL", "REL_CHAIR", "REL_COMPOSITION", "REL_RESPONSIBILITY", "REQUIRED_EVIDENCE", "RESPONSIBLE_OFFICE", "ROW_COMPLETE",
           "ROW_FRAGMENT", "ROW_KINDS", "ROW_LABELLED", "ROW_NAME_ONLY", "ROW_SINGLE", "ROW_SUB_ENTRY", "Reading", "SATISFIES", "STATED_FACT",
           "STRUCTURE", "SUBJECT_ANAPHOR", "SUBJECT_OFFICE", "SUBJECT_OTHER", "SUBJECT_UNIVERSITY", "SYSTEM_ACCESS", "TELEPHONE", "TIERS",
           "context_problem", "natural_ask",
           "UNIVERSITY", "UNIVERSITY_NAME", "Unit", "WEBSITE", "classify_ask", "clause_report", "contact_fields", "continues_from_previous",
           "continues_into_next", "covers", "duty_terms", "head_matches", "header_columns", "lead_in_office", "location_entry", "numbers_in",
           "BENEFICIARIES", "ask_residue", "ask_terms", "channel_form_problem", "covers_stems", "duty_phrases", "elsewhere", "normalise_text", "phrase_narrows",
           "stem_set", "item_title", "nested_offices", "office_spans", "other_unit_mentions", "phrase_matches", "read_row_values",
           "read_unit", "referent_problem", "resolve_offices", "row_office", "split_units", "subunit_nouns", "tier_satisfies", "without_offices"]
