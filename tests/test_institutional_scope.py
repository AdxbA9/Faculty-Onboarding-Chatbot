"""
Institutional Navigation scope, office lexicon and evidence tiers (Milestone 4, Step 4.16-B1, D-104).

The scope owns only sections no frozen specialist owns and compiles to the planned page ranges; foreign pages are out;
the extra page is admitted only because no section record covers it. Every lexicon name and alias is printed verbatim on
its pages of the real handbook; distinct printed units are never merged; a shared or partial word names no office.
The evidence tiers never let a mention prove a responsibility or a responsibility prove a procedure.
"""
from __future__ import annotations

import pytest

from handbook_bot.agents.specialists import faculty_services as HR
from handbook_bot.agents.specialists import institutional as I
from handbook_bot.agents.specialists import navigation_cues as N
from handbook_bot.agents.specialists import research as R
from handbook_bot.agents.specialists import teaching as T
from handbook_bot.agents.specialists.scoping import compile_scope

from faculty_services_support import registry_and_map
from institutional_support import make_institutional, page_text

PLANNED_RANGES = ((15, 16), (19, 49), (52, 55), (183, 183), (198, 199), (237, 238), (240, 240), (250, 258))


def _scope_pages():
    scope = compile_scope(registry_and_map()[1], I.APPROVED_SOURCE_ID, I.OWNED_SECTIONS, I.EXTRA_PAGES)
    return scope, {p for first, last in scope.page_ranges for p in range(first, last + 1)}


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
def test_the_scope_compiles_to_the_planned_page_ranges():
    scope, _pages = _scope_pages()
    assert tuple(tuple(r) for r in scope.page_ranges) == PLANNED_RANGES
    assert set(scope.source_ids) == {I.APPROVED_SOURCE_ID}                       # RetrievalScope stores a frozenset


@pytest.mark.parametrize("page", [17, 18, 50, 51, 56, 57, 63, 75, 80, 106, 153, 184, 186, 192, 194, 200, 202, 239, 241, 243, 259])
def test_foreign_pages_are_outside_the_scope(page):
    _scope, pages = _scope_pages()
    assert page not in pages


def test_no_owned_section_belongs_to_a_frozen_specialist():
    ours = {(c, n) for c, n, _ in I.OWNED_SECTIONS}
    for module in (T, R, HR):
        theirs = {(c, n) for name in ("OWNED_SECTIONS", "SHARED_SECTIONS") for c, n, _ in getattr(module, name, ())}
        assert not ours & theirs, (module.__name__, sorted(ours & theirs))


def test_the_contact_directory_is_the_section_hr_leaves_to_institutional_navigation():
    hr_excluded = {(c, n): label for c, n, label in HR.EXCLUDED_SECTIONS}
    assert "Institutional Navigation" in hr_excluded[(1, "1.16")]
    assert (1, "1.16", "Contact Information") in I.OWNED_SECTIONS


def test_the_extra_page_is_admitted_only_because_no_section_record_covers_it():
    smap = registry_and_map()[1]
    assert not [r for r in smap.sections() if r.start_page <= 183 <= r.end_page]
    with pytest.raises(ValueError):
        compile_scope(smap, I.APPROVED_SOURCE_ID, I.OWNED_SECTIONS, ((184, 184),))      # 10.1 is not owned: never silently wider


def test_the_section_guard_resolves_every_shared_page_of_the_scope():
    sp = make_institutional()
    assert sp.guard.unresolved_pages == []
    assert {16, 19, 49, 52, 198, 199, 238, 240, 255, 257} <= set(sp.guard.shared_pages)


# ---------------------------------------------------------------------------
# Office lexicon
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("office", N.OFFICES, ids=lambda o: o.office_id)
def test_every_office_name_is_printed_verbatim_on_its_pages(office):
    for page in office.pages:
        assert office.name in page_text(page), (office.name, page)


@pytest.mark.parametrize("office", [o for o in N.OFFICES if o.aliases], ids=lambda o: o.office_id)
def test_every_alias_is_printed_by_the_handbook_for_the_same_unit(office):
    for alias, pages in office.aliases:
        for page in pages:
            text = page_text(page)
            assert alias in text, (alias, page)
            if N._is_acronym(alias):
                assert "(%s)" % alias in text, (alias, page)                    # printed as "<full name> (ACRONYM)"


def test_office_ids_names_and_aliases_are_unique():
    ids = [o.office_id for o in N.OFFICES]
    names = [n.lower() for o in N.OFFICES for n in [o.name] + [a for a, _ in o.aliases]]
    assert len(ids) == len(set(ids)) and len(names) == len(set(names))


@pytest.mark.parametrize("a,b", N.NOT_EQUIVALENT)
def test_units_the_handbook_never_equates_stay_distinct(a, b):
    assert a in N.OFFICES_BY_ID and b in N.OFFICES_BY_ID and a != b
    assert N.resolve_offices(N.OFFICES_BY_ID[a].name) == (a,)
    assert N.resolve_offices(N.OFFICES_BY_ID[b].name) == (b,)


@pytest.mark.parametrize("text,offices", [
    ("What is the phone number of Human Resources?", ("human_resources",)),
    ("What is HR's email?", ("human_resources",)),
    ("Where is the ITC?", ("information_technology_center",)),
    ("What is the fax number of the Finance Department?", ("finance_department",)),
    ("Where is the Registration office?", ("registration",)),
    ("What is the phone of the College of Dental Medicine?", ("college_dental_medicine",)),
    ("What does the FMPD do?", ("facilities_management_planning_department",)),
    ("Where is the Main Library located?", ("main_library",)),
    ("Is Physical Plant the Facilities Management and Planning Department?", ("physical_plant", "facilities_management_planning_department")),
])
def test_offices_are_identified_by_their_printed_name_or_an_attested_alias(text, offices):
    assert N.resolve_offices(text) == offices


@pytest.mark.parametrize("text", [
    "Which office handles faculty housing?",                 # "Student Housing" is not faculty housing
    "How do I handle registration of a complaint?",          # a generic word is no office
    "Which office handles business services?",               # a partial name names no unit
    "Who is responsible for health and safety on campus?",   # a topic is not an office
    "Who approves a faculty research request?",
    "Who should I contact about Blackboard?",
    "What is the hr policy on leave?",                       # a lower-case "hr" is not the attested acronym
    "Where is the medical college?",
])
def test_a_shared_or_partial_word_names_no_office(text):
    assert N.resolve_offices(text) == ()


def test_a_shorter_printed_name_inside_a_longer_one_is_not_counted_twice():
    assert N.resolve_offices("Human Resources (HR)") == ("human_resources",)
    assert N.resolve_offices("Information Technology Center (ITC)") == ("information_technology_center",)
    assert N.resolve_offices("College of Fine Arts & Design") == ("college_fine_arts_design",)


# ---------------------------------------------------------------------------
# Ask kinds and evidence tiers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,kind", [
    ("What is the phone number of Human Resources?", N.CHANNEL),
    ("What is the fax number of the Finance Department?", N.CHANNEL),
    ("What is the email of the IT service desk?", N.CHANNEL),
    ("Who should I contact about Blackboard?", N.REFERRAL),
    ("Who approves a faculty research request?", N.AUTHORITY),
    ("Which office handles purchasing requests?", N.RESPONSIBLE_OFFICE),
    ("Who is responsible for health and safety on campus?", N.RESPONSIBLE_OFFICE),
    ("Where is the Main Library located?", N.LOCATION),
    ("How do I log in to MyUOS?", N.SYSTEM_ACCESS),
    ("How do I submit an IT service request?", N.PROCEDURE),
    ("What is the organizational structure of the university?", N.STRUCTURE),
    ("How many colleges does the university have?", N.FACT),
    ("Tell me something.", N.OTHER),
])
def test_ask_kinds(clause, kind):
    assert N.classify_ask(clause) == kind


def test_every_ask_kind_states_the_evidence_it_needs():
    assert set(N.REQUIRED_EVIDENCE) == set(N.ASK_KINDS)
    assert N.REQUIRED_EVIDENCE[N.CHANNEL] == N.E1
    assert {N.REQUIRED_EVIDENCE[k] for k in (N.REFERRAL, N.AUTHORITY, N.RESPONSIBLE_OFFICE)} == {N.E2}
    assert {N.REQUIRED_EVIDENCE[k] for k in (N.PROCEDURE, N.SYSTEM_ACCESS)} == {N.E3}


@pytest.mark.parametrize("reached,required,ok", [
    (N.E1, N.E1, True), (N.E2, N.E1, True), (N.E3, N.E1, False),
    (N.E1, N.E2, False), (N.E2, N.E2, True), (N.E3, N.E2, False),     # a mention never proves a responsibility
    (N.E1, N.E3, False), (N.E2, N.E3, False), (N.E3, N.E3, True),     # a responsibility never proves a procedure
    (None, N.E1, False), (None, N.E2, False), (None, N.E3, False),
])
def test_a_tier_is_never_proved_by_weaker_evidence(reached, required, ok):
    assert N.tier_satisfies(reached, required) is ok


def test_an_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        N.tier_satisfies(N.E1, "E4")
