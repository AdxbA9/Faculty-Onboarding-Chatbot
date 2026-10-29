"""
Institutional Navigation phase B3: office responsibilities, organisational structure and documented locations
(Milestone 4, Step 4.16-B3).

A responsibility is answered only from a statement that binds the office to the duty in the approved handbook (E2): the
office named as the subject of a closed list of responsibility verbs, a strict anaphor ("This Council", "The office")
resolved to the office named just before it, or a list item under a lead-in that binds it ("The Health and Safety Office
is responsible for the following:"). The asked duty must match a duty phrase of that statement (a list-item title or the
object of its verb) in all its terms; the office's own name, a shared keyword, a neighbouring paragraph or a directory
entry never count. Structure answers quote chair and composition statements; locations quote the library location
entries of section 11.6 exactly, and fail closed when the question does not pick one entry.

The tests replay the evidence-binding cases of ``qa/institutional_4_16/evidence_cases.jsonl`` under both stand-in
rankings (judged by ``institutional_support.evaluate_evidence_case``), check determinism, check every statement the
specialist reads against the printed text, test the readers directly, and switch each B3 protection off to show that
a wrong office, a wrong location or an invented answer then comes out (negative controls).
"""
from __future__ import annotations

import json

import pytest

from handbook_bot.agents.coordinator import coordinate
from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists import institutional as I
from handbook_bot.agents.specialists import navigation_cues as N
from handbook_bot.agents.specialists.contracts import Finding, FindingStatus, SpecialistFindings, SpecialistId
from handbook_bot.agents.specialists.registry import SpecialistRegistry
from handbook_bot.text_utils import EMAIL_PATTERN, PHONE_PATTERN

from institutional_support import (DIRECTORY_PDF_PAGES, RawOverlapReranker, ScriptedReranker, evaluate_evidence_case, evidence_cases,
                                   make_institutional, office_named_in, page_text, run, run_case, specialist_for_evidence_case)

SUPPORTED, PARTIAL, NOT_FOUND = FindingStatus.SUPPORTED, FindingStatus.PARTIAL, FindingStatus.NOT_FOUND
OVERRIDE = D.EXECUTABLE_SPECIALISTS + (SpecialistId.INSTITUTIONAL,)
CASES = evidence_cases()
CASES_BY_ID = {c["case_id"]: c for c in CASES}
RANKINGS = {"raw": lambda: RawOverlapReranker(), "high": lambda: ScriptedReranker([], default=9.0)}
SWITCHES = ("_B3_SUBJECT_BINDING", "_B3_RESPONSIBILITY_VERB", "_B3_ANAPHORA_STRICT", "_B3_LIST_EXTENT", "_B3_DUTY_ALL_TERMS",
            "_B3_AMBIGUITY_CHECK", "_B3_OWNERSHIP_GUARD", "_B3_PROCEDURE_GUARD", "_B3_LOCATION_ENTRY", "_B3_SAME_NAME_CHECK",
            "_B3_OFFICE_WORDS_EXCLUDED")
_BUILT = {}


def _specialist(case, ranking="raw"):
    key = (ranking, json.dumps({"a": case.get("append_lines") or {}, "r": case.get("replace_lines") or {}}, sort_keys=True))
    if key not in _BUILT:
        _BUILT[key] = specialist_for_evidence_case(case, reranker=RANKINGS[ranking]())
    return _BUILT[key]


def _comparable(result: SpecialistFindings) -> dict:
    out = result.to_dict()
    out.pop("ms", None)
    return out


@pytest.fixture(scope="module")
def sp():
    return make_institutional()


# ---------------------------------------------------------------------------
# The evidence-binding cases
# ---------------------------------------------------------------------------
def test_the_case_set_covers_every_category_and_every_protection():
    categories = {c["category"] for c in CASES}
    assert categories == {"A_responsibility", "B_wrong_office", "C_similar_names", "D_structure", "E_location", "F_missing", "G_procedure",
                          "H_boundary", "I_b2_regression", "M_mixed", "R_review", "S_synthetic", "T_review", "U_review", "V_review"}
    assert len({c["case_id"] for c in CASES}) == len(CASES) >= 130
    assert {s for c in CASES for s in c["kills"]} == set(SWITCHES)
    for c in CASES:
        # out_of_scope since Step 4.16-B4: another specialist's subject, never with a finding
        assert c["evidence"] and set(c["expected_status"]) <= {"supported", "partial", "not_found", "out_of_scope"}
        assert ("out_of_scope" in c["expected_status"]) == ("expected_handoffs" in c)


@pytest.mark.parametrize("ranking", sorted(RANKINGS))
@pytest.mark.parametrize("case_id", [c["case_id"] for c in CASES])
def test_evidence_case(case_id, ranking):
    case = CASES_BY_ID[case_id]
    sp_ = _specialist(case, ranking)
    verdict = evaluate_evidence_case(case, run_case(sp_, case), sp_)
    assert verdict["outcome"] == "as_expected", verdict


@pytest.mark.parametrize("case_id", [c["case_id"] for c in CASES])
def test_evidence_case_is_deterministic_across_runs_instances_and_rankings(case_id):
    case = CASES_BY_ID[case_id]
    first = _comparable(run_case(_specialist(case), case))
    assert _comparable(run_case(_specialist(case), case)) == first
    assert _comparable(run_case(specialist_for_evidence_case(case), case)) == first
    assert _comparable(run_case(_specialist(case, "high"), case)) == first


# ---------------------------------------------------------------------------
# Negative controls: each protection, switched off alone, lets a wrong answer out
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("switch", SWITCHES)
def test_each_protection_is_needed(monkeypatch, switch):
    assert getattr(I, switch) is True
    intended = [c for c in CASES if switch in c["kills"]]
    assert intended
    monkeypatch.setattr(I, switch, False)
    assert I.b3_protections()[switch[4:].lower()] is False
    outcomes = []
    for case in intended:
        fresh = specialist_for_evidence_case(case)           # the statements are read when the specialist is built
        outcomes.append(evaluate_evidence_case(case, run_case(fresh, case), fresh))
    assert any(v["outcome"] == "blocker" for v in outcomes), outcomes


def test_every_protection_is_on_and_reported(sp):
    assert I.b3_protections() == {k: True for k in I.b3_protections()} and len(I.b3_protections()) == len(SWITCHES)
    r = run(sp, "Who is responsible for health and safety on campus?")
    assert r.metadata["statements"]["evidence_binding_protections"] == I.b3_protections()
    assert r.metadata["statements"]["read"] == len(sp.statements) and r.metadata["statements"]["location_entries"] == len(sp.placements) == 4


# ---------------------------------------------------------------------------
# What the specialist reads from the handbook
# ---------------------------------------------------------------------------
def test_every_statement_is_verbatim_in_scope_and_prints_its_office(sp):
    assert len(sp.statements) > 200
    for st in sp.statements:
        meta = sp.resources.metadata[st.index]
        quote = sp.statement_quote(st)
        assert quote and quote in sp.resources.chunks[st.index]
        assert sp.scope.allows(meta) and meta["chunk_type"] == "paragraph" and meta["source_id"] == I.APPROVED_SOURCE_ID
        assert st.page not in DIRECTORY_PDF_PAGES                     # the contact directory is never responsibility evidence
        assert st.binding in (I.BIND_SUBJECT, I.BIND_ANAPHOR, I.BIND_LIST, I.BIND_OUTSIDE, I.BIND_ORPHAN)
        if st.office_id is not None:
            assert office_named_in(st.office_id, quote), (st, quote)
            assert st.binding in (I.BIND_SUBJECT, I.BIND_ANAPHOR, I.BIND_LIST)
        else:
            assert st.binding in (I.BIND_OUTSIDE, I.BIND_ORPHAN)


def test_the_location_entries_are_the_four_printed_library_entries(sp):
    entries = sorted((pl.label, pl.code, pl.place, pl.page) for pl in sp.placements)
    assert entries == [("Fine Arts and Design Library (M22)", "M22", "the Fine Arts and Design College", 199),
                       ("Main Library (M16)", "M16", "the University's Men Campus", 199),
                       ("Main Library (W16)", "W16", "the University's Women Campus (W16)", 199),
                       ("Medical Library", None, "the Medical and Health Campus of the University", 199)]
    printed = page_text(199)
    for pl in sp.placements:
        quote = sp.resources.chunks[pl.index][pl.start:pl.end].strip()
        assert " ".join(quote.split()) in printed and pl.place in quote and (pl.code is None or pl.code in quote)


#: chapter units that are never directory entries: the three libraries of phase B3 and the eight units with their own
#: statements added in Step 4.16-C1
NON_DIRECTORY = ("library_staff", "medical_library", "fine_arts_design_library",
                 "central_laboratories_directorate", "security_services_unit", "transport_section", "mail_office",
                 "facilities_management_section", "projects_management_section", "deans_council", "university_academic_advisory_council")


def test_the_libraries_added_for_b3_are_never_directory_offices(sp):
    for office_id in NON_DIRECTORY:
        office = N.OFFICES_BY_ID[office_id]
        assert office.contact is False
        assert sp.lookup(office_id).outcome == I.ABSENT
    assert N.row_office("Medical Library +971 6 5051111 +971 6 5052222") is None
    assert N.row_office("Mail Office +971 6 5051111 +971 6 5052222") is None
    assert {o.office_id for o in N.OFFICES if o.contact} == {o.office_id for o in N.OFFICES} - set(NON_DIRECTORY)


# ---------------------------------------------------------------------------
# Answers: shape, tiers and the existing contract
# ---------------------------------------------------------------------------
def test_a_responsibility_answer_uses_the_existing_contract_only(sp):
    r = run(sp, "Who is responsible for risk assessments?")
    reference = SpecialistFindings(specialist_id=SpecialistId.INSTITUTIONAL, task_id="x", status=NOT_FOUND).to_dict()
    assert set(r.to_dict()) == set(reference)
    assert set(r.findings[0].to_dict()) == set(Finding(finding_id="x", claim="c", evidence_quote="c", source_id="s").to_dict())
    assert r.status is SUPPORTED and r.metadata["phase"] == I.PHASE == "C1"
    f = r.findings[0]
    assert f.page == 237 and f.source_id == I.APPROVED_SOURCE_ID and f.claim == f.evidence_quote
    assert f.metadata["office_id"] == "health_and_safety_office" and f.metadata["evidence_tier"] == N.E2
    assert f.metadata["binding"] == I.BIND_LIST and f.metadata["quote_scope"] == "lead-in and list item"
    assert f.evidence_quote.startswith("The Health and Safety Office is responsible for the following:")
    assert "Risk Assessment and Management" in f.evidence_quote
    assert r.metadata["clauses"][0]["evidence_tier_reached"] == N.E2 and r.handoff_requests == []


def test_a_list_item_is_quoted_from_its_lead_in_and_names_the_item_that_binds(sp):
    # The quote must be one verbatim passage, so it runs from the lead-in through the items printed before the bound item;
    # ``statement_text`` is the bound item alone, and only its title and verb object are matched.
    r = run(sp, "Who is responsible for emergency preparedness?")
    f = r.findings[0]
    assert f.evidence_quote.startswith("The Health and Safety Office is responsible for the following:")
    assert f.metadata["statement_text"].lstrip("•o ").startswith("Emergency Preparedness and Response")
    assert "Policy Development" not in f.metadata["statement_text"] and f.metadata["statement_text"] in f.evidence_quote
    assert any("Emergency Preparedness" in p for p in f.metadata["duty_phrases"])


def test_an_office_role_answer_quotes_only_the_offices_own_statements(sp):
    r = run(sp, "What does the Disability Resource Center do?")
    assert r.status is SUPPORTED
    assert all(f.metadata["office_id"] == "disability_resource_center" and office_named_in("disability_resource_center", f.evidence_quote)
               for f in r.findings)
    assert any("The DRC is" in f.evidence_quote for f in r.findings)          # the acronym printed on page 255 binds its statements


def test_a_duty_stated_for_two_units_names_no_single_office():
    case = CASES_BY_ID["S01"]
    sp_ = _specialist(case)
    r = run_case(sp_, case)
    assert r.status is PARTIAL and any("more than one unit" in m for m in r.missing)
    # the crafted sentence names the Mail Office, a lexicon unit since Step 4.16-C1 (before: a unit outside the lexicon, None)
    assert {f.metadata["office_id"] for f in r.findings} == {"information_technology_center", "mail_office"}


def test_an_answer_never_carries_a_contact_detail_or_url(sp):
    for question in ("What does the Information Technology Center do?", "Where is the Medical Library?", "Who chairs the College Council?",
                     "What are the subgroups of the Facilities Management and Planning Department?", "How many libraries does the university have?"):
        text = json.dumps(_comparable(run(sp, question)), ensure_ascii=False)
        assert not PHONE_PATTERN.search(text) and not EMAIL_PATTERN.search(text) and "http" not in text, question


@pytest.mark.parametrize("question,phrase,status", [
    ("How do I submit an IT service request?", "step-by-step procedure", NOT_FOUND),
    ("Who approves purchase requests?", "never infers an approver", NOT_FOUND),
    # Step 4.16-B4: another specialist's subject is out_of_scope; an HR subject is handed off and adds no missing entry
    # (test_institutional_integration.py), a Teaching subject is not handed off and says why
    ("What is the grading policy?", "belongs to Teaching & Learning", FindingStatus.OUT_OF_SCOPE),
    ("Where is the Main Library on the third floor?", "states no floor", NOT_FOUND),
    ("What does the Main Library do?", "E1 only", NOT_FOUND),
])
def test_unanswered_asks_say_why(sp, question, phrase, status):
    r = run(sp, question)
    assert r.status is status and r.findings == [] and any(phrase in m for m in r.missing), r.missing


def test_a_location_answer_quotes_the_entry_and_adds_nothing(sp):
    r = run(sp, "Where is the Main Library for women?")
    assert r.status is SUPPORTED and len(r.findings) == 1
    f = r.findings[0]
    assert f.metadata["relation"] == "location" and f.metadata["building_code"] == "W16" and f.metadata["evidence_tier"] == N.STATED_FACT
    assert "Women Campus" in f.evidence_quote and not any(w in r.summary.lower() for w in ("floor", "room", "hours", "gps", "map"))


def test_a_count_quotes_the_statement_that_states_it(sp):
    r = run(sp, "How many libraries does the university have?")
    assert r.status is SUPPORTED and r.findings[0].page == 199 and "nine libraries" in r.findings[0].evidence_quote
    assert "states nine libraries" in r.summary


def test_one_focus_clause_is_one_ask_read_as_a_whole(sp):
    # The third review showed that splitting a focus clause lets a dropped part (a place, a person, a correction, a
    # fronted beneficiary) escape every check, so a clause is read as a whole, by its first-priority ask kind; the
    # Coordinator cuts a question into clauses and phase B4 does not split them again. A clause holding a procedure ask
    # is a procedure ask.
    r = run(sp, "Who provides IT support? How do I request it?")
    report = r.metadata["clauses"][0]
    assert report["ask_kind"] == N.PROCEDURE and "sub_asks" not in report and r.status is NOT_FOUND and r.findings == []
    r = run(sp, "In Kalba, where is the Medical Library?")
    assert r.status is NOT_FOUND and r.findings == [] and "Kalba" in r.missing[0]


# ---------------------------------------------------------------------------
# The readers
# ---------------------------------------------------------------------------
def test_split_units_cuts_sentences_and_list_items():
    text = ("The Health and Safety Office is responsible for the following: 1. Policy Development: Developing policies. "
            "2. Risk Assessment: Conducting risk assessments, e.g. of labs. The office investigates incidents.")
    units = [u.text.strip() for u in N.split_units(text)]
    assert units[0] == "The Health and Safety Office is responsible for the following:"
    assert any(u.startswith("1. Policy Development:") for u in units) and any(u.startswith("2. Risk Assessment:") for u in units)
    assert any("e.g. of labs." in u for u in units)                    # an abbreviation is not a sentence end
    assert units[-1] == "The office investigates incidents."
    for u in N.split_units(text):
        assert text[u.start:u.end].strip() == u.text.strip()


@pytest.mark.parametrize("text,kind,office,relation", [
    ("The Health and Safety Office is responsible for the following:", N.SUBJECT_OFFICE, "health_and_safety_office", N.REL_RESPONSIBILITY),
    ("The College Council is chaired by the Dean of the College.", N.SUBJECT_OFFICE, "college_council", N.REL_CHAIR),
    ("The ITC also maintains a cutting-edge Data Center.", N.SUBJECT_OFFICE, "information_technology_center", N.REL_RESPONSIBILITY),
    ("This Council is responsible for coordinating graduate studies.", N.SUBJECT_ANAPHOR, None, N.REL_RESPONSIBILITY),
    ("The Mail Office also maintains a cutting-edge Data Center.", N.SUBJECT_OFFICE, "mail_office", N.REL_RESPONSIBILITY),   # C1 unit
    ("The Purchasing Section also maintains a cutting-edge Data Center.", N.SUBJECT_OTHER, None, N.REL_RESPONSIBILITY),
    ("There are eight Vice Chancellors that support the Chancellor in the daily operation of the University of Sharjah.",
     N.SUBJECT_UNIVERSITY, N.UNIVERSITY, None),
    ("There are five specialized centers in the institute.", None, None, None),          # "There are" without the University
    ("We offer a total of 149 academic degree programs.", None, None, None),           # "we" is not bound by assumption
    ("University Research Board", None, None, None),                                  # a committee name, not the University
    ("The Vice Chancellor for Public Relations is a member of the Board.", None, None, None),
])
def test_read_unit_binds_only_the_grammatical_subject(text, kind, office, relation):
    r = N.read_unit(text)
    assert (r.subject_kind, r.office_id if kind != N.SUBJECT_ANAPHOR else None, r.relation) == (kind, office, relation)


def test_a_responsibility_needs_a_verb_of_the_closed_list():
    assert N.read_unit("The Health and Safety Office team meets at the Fine Arts and Design College.").relation is None
    assert N.read_unit("Main Library (W16) is located on the University's Women Campus (W16).").relation is None


@pytest.mark.parametrize("terms,phrase,expected", [
    (("risk", "assessments"), "Risk Assessment and Management", True),
    (("emergency", "preparedness"), "Emergency Preparedness and Response", True),
    (("facilities", "management"), "Risk Assessment and Management", False),        # every asked term must be in the phrase
    (("data", "center"), "a cutting-edge Data Center", True),
    (("maintenance",), "the maintenance of international alliances", False),        # a one-term ask must head a conjunct
    (("business", "services"), "Business Service Department", False),              # an office name is masked
])
def test_phrase_matches(terms, phrase, expected):
    assert N.phrase_matches(terms, phrase) is expected


@pytest.mark.parametrize("clause,expected", [
    ("Who maintains the Data Center?", ("Center",)),
    ("What does the Writing Center of the Language Institute do?", ("Center",)),       # a unit noun printed before the office
    ("What is the phone number of the HR department?", ()),                             # the office noun right after its name
    ("What does the Language Institute do?", ()),
    ("What is the phone number of the College of Sciences office?", ()),
])
def test_other_unit_mentions(clause, expected):
    assert tuple(w.title() for w in N.other_unit_mentions(clause)) == expected


@pytest.mark.parametrize("text,expected", [
    ("Medical Library: a specialized library in medicine and health sciences located on the Medical and Health Campus of the University.",
     ("medical_library", None)),
    ("Main Library (W16): a central library for women on the University's Women Campus (W16).", ("main_library", "W16")),
    ("Medical Library: a specialized library in medicine and health sciences.", None),                           # no place
    ("The Health and Safety Office team meets at the Fine Arts and Design College.", None),     # not a library
    ("The Main Library is a modern building.", None),                                           # no place
])
def test_location_entry_reads_library_entries_only(text, expected):
    entry = N.location_entry(text)
    assert (None if entry is None else (entry.office_id, entry.code)) == expected


# ---------------------------------------------------------------------------
# Rules added after the second adversarial review
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("obj,expected", [
    (" the needs of students with disabilities", ("needs", "disabilities")),          # only the beneficiary phrase is cut
    (" technical support to faculty for laboratory equipment", ("technical", "support", "laboratory", "equipment")),
    (" instruction to students in English", ("instruction", "english")),
    (" mandatory training for faculty, staff, and students", ("mandatory", "training")),
    (" health and safety on campus", ("health", "safety")),
    (" the following", ()),                                                            # a placeholder names no duty
])
def test_ask_terms_cut_only_the_beneficiaries_and_the_setting(obj, expected):
    assert N.ask_terms(obj) == expected


@pytest.mark.parametrize("a,b", [("coordinate", "coordinating"), ("coordination", "coordinating"), ("response", "responses"),
                                 ("implementing", "implementation"), ("service", "services"), ("college", "colleges")])
def test_stems_meet_across_inflections(a, b):
    assert N._stem(a) == N._stem(b)


def test_stems_keep_management_and_manages_apart():
    assert N._stem("management") != N._stem("manages")


@pytest.mark.parametrize("terms,phrase,expected", [
    (("needs",), "supporting the University's infrastructure needs", False),          # a narrower duty
    (("awareness",), "promote safety awareness", False),
    (("programs",), "several programs", False),
    (("health",), "Health and Safety Training", False),                                 # "Health" shares the head "Training"
    (("instruction",), "assistance and instruction in using the library resources", False),
    (("data",), "a cutting-edge Data Center", False),                                   # "Data Center" names a centre, not data
    (("data", "center"), "a cutting-edge Data Center", True),                           # a grading modifier is no narrowing
    (("library", "resources"), "assistance and instruction in using the library resources", False),   # governed by "instruction in using"
    (("management",), "Risk Assessment and Management", False),                          # "Risk" is shared by "Management"
    (("promoting", "awareness"), "promote safety awareness", False),                    # "safety" narrows the object of the verb
    (("IT", "support"), "IT Technical Support and Services", True),
    (("libraries",), "all functions of nine libraries", False),                       # inside the "of" complement of "functions"
    (("academic", "activities"), "infrastructure for University administration and academic activities", False),
])
def test_a_one_word_duty_is_never_matched_to_a_narrower_one(terms, phrase, expected):
    assert N.phrase_matches(terms, phrase) is expected


def test_the_drc_acronym_binds_statements_but_never_a_question_or_a_directory_line(sp):
    office = N.OFFICES_BY_ID["disability_resource_center"]
    assert office.subject_aliases == (("DRC", (255, 256)),) and office.aliases == ()
    for alias, pages in office.subject_aliases:
        assert all(alias in page_text(p) for p in pages)
    assert "the Disability Resource Center (DRC)" in page_text(255)
    assert N.resolve_offices("What is the phone number of the DRC?") == ()
    assert N.row_office("DRC +971 6 5051111 +971 6 5052222") is None
    assert run(sp, "What is the phone number of the DRC?").status is NOT_FOUND
    assert any(st.office_id == "disability_resource_center" and st.subject_text.endswith("DRC") for st in sp.statements)


@pytest.mark.parametrize("clause,office", [
    ("Who provides support to the Business Service Department?", "Business Service Department"),
    ("Who provides training to the ITC?", "Information Technology Center"),
    ("Who provides support to the IT Center?", "to the IT Center"),                       # an unprinted unit served
    ("For the Business Service Department, who provides essential support?", "Business Service Department"),
    ("Who in the ITC maintains the Data Center?", None),
])
def test_an_office_inside_the_asked_duty_is_the_unit_served(sp, clause, office):
    assert sp._office_in_object(clause) == office


@pytest.mark.parametrize("question", [
    "What is the phone number of the Finance Department of the University of Khorfakkan?",
    "What is the phone number of HR in Kalba?",
    "What is the phone number of the Finance Department cashier?",
    "What is the phone number of the accountant in the Finance Department?",
    "What is the Sharjah Police HR phone number?",
    "What is the alternate phone number of HR?",
    "What are the HR and PSU phone numbers?",                # a second unit outside the lexicon (the CLD is one since C1)
    "Who decides the fax number of the Finance Department?",
])
def test_a_contact_ask_that_is_not_a_plain_form_fails_closed(question):
    assert N.referent_problem(question) is not None


@pytest.mark.parametrize("question", [
    "What is the phone number of the HR team?", "What is the phone number of the Registration office, please?",
    "What is the phone number of HR at the University of Sharjah?", "What is the HR phone number for new faculty?",
    "Which phone number should I use for the Finance Department?", "What fax number does the ITC use?",
    "What is the phone number of HR dept?", "What is the phone number of HR's main office?",
    "What is the phone number of the Finance Department and what is its fax number?", "For the ITC, what is the fax number?",
])
def test_plain_contact_forms_are_answered(question):
    assert N.referent_problem(question) is None


def test_courtesy_words_and_channel_follow_ups_keep_the_phase_b2_answer(sp):
    r = run(sp, "What is the phone number of Human Resources? Thanks.")
    assert r.status is SUPPORTED and r.missing == []
    r = run(sp, "What is the phone number of Human Resources? And the fax number?")
    assert r.status is SUPPORTED and r.findings[0].metadata["contact_values"] == {"telephone": "+971 6 5050023", "fax": "+971 6 5585200"}
    r = run(sp, "What does the ITC do and what is its phone number?")             # a mixed clause: no plain contact form
    assert r.status is NOT_FOUND and r.findings == []
    for question in ("What is the phone number of the Media Center? And the phone number of the Translation Center?",
                     "At the University of Khorfakkan, what is the phone number of Human Resources?"):
        assert run(sp, question).status is NOT_FOUND, question


def test_a_duty_named_only_inside_narrower_duties_gives_candidates_never_the_answer(sp):
    r = run(sp, "Who is responsible for health and safety on campus?")
    assert r.status is PARTIAL and {f.metadata["office_id"] for f in r.findings} == {"health_and_safety_office"}
    assert any("narrower duties" in m for m in r.missing)
    assert N.phrase_narrows(("health", "safety"), "Health and Safety Training") is True
    assert N.phrase_narrows(("needs",), "the University's infrastructure needs") is False


@pytest.mark.parametrize("question,intent", [
    ("Who is the current Chancellor according to the organizational chart?", "unrecognised"),     # no answered form
    ("Who reports to the Chancellor in the organizational chart?", "unrecognised"),
    ("In Kalba, where is the Medical Library?", "unrecognised"),
    ("Besides the ITC, who provides IT support?", "unrecognised"),
    ("Who provides support to the IT Center?", "unrecognised"),                                   # "to" in the duty
    ("Who is in charge of the organizational structure?", "responsible_office"),                  # no statement binds it
    ("Which office provides IT support?", "responsible_office"),
    ("What does the EHSM Section do?", "office_role"),
    ("Where is the Main Library for women?", "location"),
    ("How many libraries does the university have?", "count"),
    ("Which office has the authority to manage the Data Center?", "authority"),
    # whose subject a clause is (HR, Research, Teaching) is decided before the intent since phase B4 (_boundary, tested in
    # test_institutional_integration.py)
    ("What is the organizational structure of the university?", "organisation"),
    ("Who chairs the College Council?", "chair"),
    ("Who are the members of the College Council?", "composition"),
    ("Does the FMPD maintain the centers?", "office_role"),
])
def test_intents_and_question_forms(sp, question, intent):
    assert sp._b3_intent(question, N.classify_ask(question)) == intent


# ---------------------------------------------------------------------------
# Through the existing Coordinator and dispatcher (routing unchanged)
# ---------------------------------------------------------------------------
def _registry(sp):
    reg = SpecialistRegistry()
    reg.register(sp)
    return reg


@pytest.mark.parametrize("question", ["Who is responsible for health and safety on campus?", "Where is the Medical Library?",
                                      "Who chairs the College Council?"])
def test_b3_answers_run_through_the_override_and_the_default_dispatch(sp, question):
    decision = coordinate(question)
    out = D.dispatch(decision, _registry(sp), executable=OVERRIDE, question=question)
    runs = [r for r in out.runs if r.specialist_id is SpecialistId.INSTITUTIONAL]
    if SpecialistId.INSTITUTIONAL in decision.selected_specialists:
        assert runs and runs[0].state == D.STATE_EXECUTED and runs[0].findings.handoff_requests == []
    # Step 4.16-B6 (D-105): the default dispatch executes it too, with the same result
    default = D.run_question(question, _registry(sp))
    assert default.executed_specialists == [SpecialistId.INSTITUTIONAL]
    assert _comparable(default.results[0]) == _comparable(runs[0].findings)
