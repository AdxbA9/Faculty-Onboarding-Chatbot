"""
Institutional Navigation phase B2: telephone and fax numbers from the contact directory (Milestone 4, Step 4.16-B2).

A number is returned only when the row binding holds: a lexicon office named as the unit asked about, exactly one
directory line with numbers starting with its printed name, that line holding exactly two complete numbers, no
neighbouring line running into it, and a page header fixing the telephone and fax columns. The tests check the lookups
against an independent oracle (the printed directory rebuilt from the PDF's ruled cells and word positions), replay the
contact-accuracy cases of ``qa/institutional_4_16/contact_cases.jsonl`` under both stand-in rankings, check that every
number output is verbatim, bound and printed in that cell, and switch each binding protection off to show that a wrong
office or a wrong number then comes out (negative controls).
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

from institutional_support import (RawOverlapReranker, ScriptedReranker, contact_cases, evaluate_contact_case, make_institutional,
                                   make_institutional_with_lines, normalise, printed_directory, run, run_case, specialist_for_case)

SUPPORTED, PARTIAL, NOT_FOUND = FindingStatus.SUPPORTED, FindingStatus.PARTIAL, FindingStatus.NOT_FOUND
OVERRIDE = D.EXECUTABLE_SPECIALISTS + (SpecialistId.INSTITUTIONAL,)
CASES = contact_cases()
CASES_BY_ID = {c["case_id"]: c for c in CASES}
RANKINGS = {"raw": lambda: RawOverlapReranker(), "high": lambda: ScriptedReranker([], default=9.0)}
#: Entries whose printed cell holds one value per column and whose extracted line holds the whole row.
CLEAN = ("registration", "disability_resource_center", "passport_unit", "college_sharia_islamic_studies", "college_fine_arts_design",
         "college_communication", "college_medicine", "college_dental_medicine", "college_pharmacy", "information_technology_center",
         "physical_plant", "human_resources", "public_relations", "media_center", "finance_department", "alumni_association",
         "central_laboratories")
_BUILT = {}


@pytest.fixture(scope="module")
def sp():
    return make_institutional()


def _specialist(case, ranking="raw"):
    key = (ranking, json.dumps(case.get("replace_lines") or {}, sort_keys=True))
    if key not in _BUILT:
        _BUILT[key] = specialist_for_case(case, reranker=RANKINGS[ranking]())
    return _BUILT[key]


def _comparable(result: SpecialistFindings) -> dict:
    out = result.to_dict()
    out.pop("ms", None)
    return out


def _oracle():
    return {e["office_id"]: e for e in printed_directory() if e["office_id"]}


# ---------------------------------------------------------------------------
# The extracted lines against the printed directory (the stop condition of the plan)
# ---------------------------------------------------------------------------
def test_the_directory_is_read_from_section_1_16_with_one_column_header_per_page(sp):
    assert sp.directory_pages == frozenset({52, 53, 54, 55})
    assert sorted(sp.directory) == [52, 53, 54, 55]
    assert sp.header_columns == {p: (N.TELEPHONE, N.FAX) for p in (52, 53, 54, 55)}
    headers = [e for e in printed_directory() if e["office"] == "Office / College / Department"]
    assert [(e["page"], e["telephone"], e["fax"]) for e in headers] == [(p, ["Telephone Number"], ["Fax Number"]) for p in (52, 53, 54, 55)]


def test_every_clean_entry_line_holds_exactly_its_printed_cells(sp):
    oracle = _oracle()
    for office_id in CLEAN:
        lk = sp.lookup(office_id)
        printed = oracle[office_id]
        assert len(printed["telephone"]) == 1 and len(printed["fax"]) == 1, office_id
        assert lk.outcome == I.BOUND, office_id
        assert lk.values == {N.TELEPHONE: printed["telephone"][0], N.FAX: printed["fax"][0]}, office_id
        assert normalise(lk.row.text) == normalise("%s %s %s" % (printed["office"], printed["telephone"][0], printed["fax"][0])), office_id
        assert lk.row.page == printed["page"]


def test_only_the_clean_entries_are_bound_and_every_bound_value_is_its_printed_cell(sp):
    oracle = _oracle()
    bound = [o.office_id for o in N.OFFICES if sp.lookup(o.office_id).outcome == I.BOUND]
    assert sorted(bound) == sorted(CLEAN)
    for office_id in bound:
        for field, value in sp.lookup(office_id).values.items():
            assert oracle[office_id][field] == [value], (office_id, field)


@pytest.mark.parametrize("office_id,outcome,reason", [
    ("college_sciences", I.UNBOUND, N.ROW_SINGLE),               # its line holds only the FAX number
    ("college_computing_informatics", I.UNBOUND, N.ROW_SINGLE),  # its line holds only the telephone; the fax is on two other lines
    ("college_public_policy", I.UNBOUND, N.ROW_SINGLE),          # one number printed with an internal space; no fax printed
    ("dental_hospital_call_center", I.LABELLED, None),           # External / Ext ... Internal
    ("student_housing", I.UNBOUND, N.ROW_FRAGMENT),
    ("college_business_administration", I.UNBOUND, N.ROW_NAME_ONLY),
    ("college_engineering", I.UNBOUND, N.ROW_FRAGMENT),
    ("college_health_sciences", I.UNBOUND, N.ROW_FRAGMENT),
    ("college_law", I.UNBOUND, N.ROW_FRAGMENT),
    ("deanship_student_affairs", I.UNBOUND, N.ROW_FRAGMENT),
    ("medical_service_department", I.UNBOUND, N.ROW_NAME_ONLY),
    ("university_city_security", I.UNBOUND, N.ROW_NAME_ONLY),
    ("mens_library", I.UNBOUND, N.ROW_SUB_ENTRY),
    ("womens_library", I.UNBOUND, N.ROW_SUB_ENTRY),
    ("medical_colleges_library", I.UNBOUND, N.ROW_SUB_ENTRY),
    ("college_arts_humanities_social_sciences", I.SPLIT, None),
    ("college_graduate_studies_research", I.SPLIT, None),
    ("continuing_education_professional_development", I.SPLIT, None),
    ("business_services_purchases_department", I.SPLIT, None),
    ("main_library", I.ABSENT, None),
    ("health_and_safety_office", I.ABSENT, None),
])
def test_hazardous_entries_are_never_bound(sp, office_id, outcome, reason):
    lk = sp.lookup(office_id)
    assert (lk.outcome, lk.reason) == (outcome, reason) and lk.values == {}


def test_the_college_of_sciences_line_holds_its_fax_number_not_its_telephone(sp):
    printed = _oracle()["college_sciences"]
    line = sp.lookup("college_sciences").row.text
    assert N.numbers_in(line) == tuple(printed["fax"])
    assert printed["telephone"][0] not in line


def test_the_directory_prints_no_email_address_or_website(sp):
    texts = [r.text for r in sp.directory_rows()] + [line for e in printed_directory() for line in e["telephone"] + e["fax"] + [e["office"]]]
    for text in texts:
        assert not EMAIL_PATTERN.search(text) and "http" not in text.lower() and "www" not in text.lower(), text


# ---------------------------------------------------------------------------
# The contact-accuracy cases, both rankings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ranking", sorted(RANKINGS))
@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_contact_case(case, ranking):
    sp_ = _specialist(case, ranking)
    verdict = evaluate_contact_case(case, run_case(sp_, case), sp_)
    assert verdict["outcome"] == "as_expected", verdict


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_contact_case_is_deterministic_across_runs_instances_and_rankings(case):
    first = _comparable(run_case(_specialist(case, "raw"), case))
    assert _comparable(run_case(_specialist(case, "raw"), case)) == first
    assert _comparable(run_case(specialist_for_case(case), case)) == first
    assert _comparable(run_case(_specialist(case, "high"), case)) == first


def test_the_case_set_covers_every_category_and_gives_no_number_for_unbound_entries():
    categories = {c["category"] for c in CASES}
    assert {"clean_telephone", "clean_fax", "single_number_line", "labelled_line", "fragmented_entry", "split_entry", "not_in_directory",
            "unattested_name", "email_website_extension", "referent_trap", "shared_number", "synthetic_neighbour", "synthetic_row_start",
            "synthetic_header", "synthetic_single_entry"} <= categories
    for case in CASES:
        if case["category"] in ("single_number_line", "labelled_line", "fragmented_entry", "split_entry", "not_in_directory", "unattested_name",
                                "referent_trap"):
            assert case["expected_values"] == {}, case["case_id"]


# ---------------------------------------------------------------------------
# Properties over every lexicon office and every field
# ---------------------------------------------------------------------------
PHRASINGS = ("What is the phone number of the {}?", "What is the fax number of the {}?", "How can I contact the {}?",
             "What are the telephone and fax numbers of the {}?", "What is the extension of the {}?", "What is the email of the {}?",
             "What is the website of the {}?")


@pytest.mark.parametrize("office", N.OFFICES, ids=[o.office_id for o in N.OFFICES])
def test_every_number_output_is_verbatim_bound_asked_and_printed_in_that_cell(sp, office):
    name = office.name + (" office" if office.needs_office_noun else "")
    for phrasing in PHRASINGS:
        question = phrasing.format(name)
        result = run(sp, question)
        produced = {}
        for f in result.findings:
            assert f.metadata["office_id"] == office.office_id, question
            if f.metadata["directory_entry"] == "complete_row":
                assert set(f.metadata["contact_fields"]) <= set(N.contact_fields(question)), question
                produced.setdefault(office.office_id, {}).update(f.metadata["contact_values"])
        pseudo = {"case_id": "grid", "category": "grid", "question": question, "expected_status": result.status.value, "expected_values": produced}
        verdict = evaluate_contact_case(pseudo, result, sp)
        assert verdict["outcome"] == "as_expected", verdict
        if office.office_id not in CLEAN:
            assert produced == {}, question


def test_status_follows_the_binding(sp):
    assert run(sp, "What is the phone number of Human Resources?").status is SUPPORTED
    assert run(sp, "What is the phone number of the College of Sciences?").status is PARTIAL
    assert run(sp, "What is the phone number of the College of Arts, Humanities and Social Sciences?").status is NOT_FOUND
    r = run(sp, "What is the phone number and email of Human Resources?")
    assert r.status is PARTIAL and [f.metadata["contact_values"] for f in r.findings] == [{N.TELEPHONE: "+971 6 5050023"}]
    assert len(r.missing) == 1 and "e-mail" in r.missing[0]


def test_unbound_findings_quote_only_the_office_name_as_printed(sp):
    r = run(sp, "What is the phone number of the College of Sciences?")
    assert [f.claim for f in r.findings] == ["College of Sciences"] and r.findings[0].metadata["quote_scope"] == "office name only"
    assert not PHONE_PATTERN.search(json.dumps([r.summary, r.missing, [f.to_dict() for f in r.findings]], ensure_ascii=False))
    r = run(sp, "What is the fax number of the College of Shari'a & Islamic Studies?")
    assert r.status is SUPPORTED and r.findings[0].claim == "College of Shari’a & Islamic Studies +971 6 5050181 +971 6 5050334"


def test_telephone_and_fax_are_kept_apart(sp):
    tel = run(sp, "What is the phone number of the Finance Department?").findings[0].metadata
    fax = run(sp, "What is the fax number of the Finance Department?").findings[0].metadata
    assert tel["contact_values"] == {N.TELEPHONE: "+971 6 5050040"} and fax["contact_values"] == {N.FAX: "+971 6 5585183"}
    assert tel["row_values"] == fax["row_values"] == {N.TELEPHONE: "+971 6 5050040", N.FAX: "+971 6 5585183"}
    assert tel["column_order"] == [N.TELEPHONE, N.FAX] and tel["header_chunk_id"] is not None


def test_the_labelled_line_is_never_given_a_telephone_or_fax_field(sp):
    r = run(sp, "What is the phone number of the Dental Hospital Call Center?")
    meta = r.findings[0].metadata
    assert r.status is PARTIAL and meta["directory_entry"] == "labelled_row" and meta["contact_values"] == {}
    assert meta["labelled_values"] == [{"value": "+971 6 5057666", "label": "External"}, {"value": "Ext 7666", "label": "Internal"}]
    assert run(sp, "What is the fax number of the Dental Hospital Call Center?").findings == []


def test_numbers_are_reported_as_printed_not_as_verified_active(sp):
    r = run(sp, "What is the phone number of Human Resources?")
    assert any("not verified as active" in lim for lim in r.limitations)
    assert r.findings[0].metadata["as_printed"] is True and r.findings[0].metadata["verified_active"] is False
    assert not any("not verified as active" in lim for lim in run(sp, "What is the phone number of the College of Sciences?").limitations)


@pytest.mark.parametrize("question,expected", [
    ("What is the phone number of the Registration office?", "also printed in the contact directory for Information Technology Center and on one other directory line"),
    ("What is the phone number of the Information Technology Center?", "also printed in the contact directory for Registration and on one other directory line"),
    ("What is the phone number of Public Relations?", "also printed in the contact directory for Media Center"),
    ("What is the fax number of the Media Center?", "also printed in the contact directory for Public Relations"),
    ("What is the fax number of the Alumni Association?", "on 2 other directory lines"),
])
def test_a_number_printed_for_several_entries_carries_a_limitation(sp, question, expected):
    r = run(sp, question)
    assert r.status is SUPPORTED and any(expected in lim for lim in r.limitations), r.limitations


def test_results_use_the_existing_contract_only(sp):
    r = run(sp, "What is the phone number of Human Resources?")
    reference = SpecialistFindings(specialist_id=SpecialistId.INSTITUTIONAL, task_id="x", status=NOT_FOUND).to_dict()
    assert set(r.to_dict()) == set(reference)
    assert set(r.findings[0].to_dict()) == set(Finding(finding_id="x", claim="c", evidence_quote="c", source_id="s").to_dict())
    f = r.findings[0]
    assert f.source_id == I.APPROVED_SOURCE_ID and f.source_title == "UOS Faculty Handbook 2025/26" and f.page == 54
    assert f.metadata["chunk_type"] == "row" and f.metadata["section_no"] == "1.16" and f.metadata["evidence_tier"] == N.E1
    assert r.metadata["phase"] == I.PHASE == "C1" and r.metadata["clauses"][0]["evidence_tier_reached"] == N.E1
    assert r.metadata["clauses"][0]["answered"] is True and r.metadata["directory"]["binding_protections"] == {k: True for k in I.binding_protections()}


def test_asks_neither_phase_b2_nor_b3_answers_keep_the_phase_b1_behaviour(sp):
    # Step 4.16-B3: the Main Library and health-and-safety questions this test held until phase B2 are answered from
    # bound statements now (see test_the_directory_is_never_responsibility_or_location_evidence); these are not. (Step
    # 4.16-B4: "Which office handles faculty housing?" is an HR subject handed off since phase B4, so it left this list.)
    for question in ("How do I log in to MyUOS?", "Who handles complaints?", "What is the email of the IT service desk?"):
        r = run(sp, question)
        assert r.status is NOT_FOUND and r.findings == [] and len(r.missing) == 1 and question in r.missing[0]
        assert r.metadata["clauses"][0]["answered"] is False and r.metadata["clauses"][0]["evidence_tier_reached"] is None


def test_the_directory_is_never_responsibility_or_location_evidence(sp):
    for question in ("Where is the Main Library located?", "Who is responsible for health and safety on campus?",
                     "What does the Information Technology Center do?", "What does the Medical Service Department do?"):
        r = run(sp, question)
        assert all("directory_entry" not in f.metadata and f.page not in (52, 53, 54, 55) for f in r.findings)
    assert not any(st.page in (52, 53, 54, 55) for st in sp.statements)
    assert not any(pl.page in (52, 53, 54, 55) for pl in sp.placements)


# ---------------------------------------------------------------------------
# The cue readers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("Human Resources +971 6 5050023 +971 6 5585200", ("human_resources", N.ROW_COMPLETE)),
    ("Registration +971 6 5050000 +971 6 5050717", ("registration", N.ROW_COMPLETE)),          # the office column needs no office noun
    ("College of Shari’a & Islamic Studies +971 6 5050181 +971 6 5050334", ("college_sharia_islamic_studies", N.ROW_COMPLETE)),
    ("College of Sciences +971 6 5050489", ("college_sciences", N.ROW_SINGLE)),
    ("College of Public Policy +971 6 505 3212", ("college_public_policy", N.ROW_SINGLE)),
    ("Dental Hospital Call Center +971 6 5057666 (External) Ext 7666 (Internal)", ("dental_hospital_call_center", N.ROW_LABELLED)),
    ("College of Law Men: +971 6 5050195", ("college_law", N.ROW_FRAGMENT)),
    ("Student Housing +971 6 5585487 /", ("student_housing", N.ROW_FRAGMENT)),
    ("Men’s Library: +971 6 5050245", ("mens_library", N.ROW_SUB_ENTRY)),
    ("Medical Service Department", ("medical_service_department", N.ROW_NAME_ONLY)),
    ("College of Medicine Library +971 6 5051111 +971 6 5052222", None),       # a different unit whose name starts like the college's
    ("Directorate of Human Resources +971 6 5051111 +971 6 5052222", None),    # the office named, but not at the start
    ("Vice Chancellor for Public Relations +971 505 0055", None),
    ("College of Arts, Humanities and Social", None),
    ("Business Services and Purchases", None),
    ("Professional Development (CCE) 5057087", None),
    ("Office / College / Department Telephone Number Fax Number", None),
])
def test_row_office_and_row_values(text, expected):
    hit = N.row_office(text)
    if expected is None:
        assert hit is None
    else:
        assert hit is not None and (hit[0], N.read_row_values(text[hit[2]:])[0]) == expected


def test_row_values_are_verbatim_and_never_repaired():
    assert N.read_row_values(" +971 6 505 3212") == (N.ROW_SINGLE, ("+971 6 505 3212",), ())
    assert N.read_row_values(" +971 6 5050225 / +971 6") == (N.ROW_FRAGMENT, ("+971 6 5050225",), ())
    assert N.read_row_values(" +971 505 0055") == (N.ROW_FRAGMENT, (), ())                   # not a complete number: not completed
    assert N.read_row_values(" +971 6 5057666 (External) Ext 7666 (Internal)") == (N.ROW_LABELLED, ("+971 6 5057666", "Ext 7666"),
                                                                                   ("External", "Internal"))


@pytest.mark.parametrize("text,columns", [
    ("Office / College / Department Telephone Number Fax Number", (N.TELEPHONE, N.FAX)),
    ("Office / College / Department Fax Number Telephone Number", (N.FAX, N.TELEPHONE)),
    ("Office / College / Department", None),
    ("Office / College / Department Telephone Number Telephone Number", None),
])
def test_header_columns(text, columns):
    assert N.header_columns(text) == columns


@pytest.mark.parametrize("text,into_next,from_previous", [
    ("+971 6 5050225 / +971 6", True, False), ("Women: +971 6 5051406 / +971", True, False), ("Men: +971 6 5052778", True, False),
    ("Men’s Clinic: +971 6 5050207", True, False), ("6 5051401 / +971 6 5051443 /", True, True), ("5057087", False, True),
    ("+971 6 5585160", False, False), ("+971 6 5050523 /", True, False), ("Physical Plant +971 6 5050058 +971 6 5050051", False, False),
])
def test_continuation_lines(text, into_next, from_previous):
    assert (N.continues_into_next(text), N.continues_from_previous(text)) == (into_next, from_previous)


@pytest.mark.parametrize("clause,fields", [
    ("What is the phone number of Human Resources?", (N.TELEPHONE,)),
    ("What is the telephone number of HR?", (N.TELEPHONE,)),
    ("What is the fax number of the Finance Department?", (N.FAX,)),
    ("What is the fax number of the Dental Hospital Call Center?", (N.FAX,)),           # "Call" inside the office name is no ask
    ("What are the telephone and fax numbers of the College of Pharmacy?", (N.TELEPHONE, N.FAX)),
    ("How can I contact Human Resources?", N.GENERIC_FIELDS),
    ("What are the contact details of the Finance Department?", N.GENERIC_FIELDS),
    ("What is the extension of the Dental Hospital Call Center?", (N.EXTENSION,)),
    ("What is the phone number and email of HR?", (N.TELEPHONE, N.EMAIL)),
    ("What is the website of the ITC?", (N.WEBSITE,)),
])
def test_contact_fields(clause, fields):
    assert N.contact_fields(clause) == fields


@pytest.mark.parametrize("clause,problem", [
    ("What is the phone number of Human Resources?", False),
    ("What are the phone numbers of HR and the Finance Department?", False),
    ("What is the fax number of the HR office and the Finance Department?", False),
    ("What is the phone number of the Men's Library?", False),                         # "Library" inside the office name
    ("What is the phone number of the Vice Chancellor for Public Relations?", True),
    ("What is the phone number of the College of Medicine library?", True),
    ("What is the phone number of the dean of the College of Law?", True),
    ("What is the phone number of the Finance Department of the College of Engineering?", True),
    ("What is the phone number of the women's section of the College of Medicine?", True),
])
def test_referent_problem(clause, problem):
    assert (N.referent_problem(clause) is not None) is problem


@pytest.mark.parametrize("clause,kind", [
    ("How can I contact Human Resources?", N.CHANNEL),
    ("How do I reach the College of Medicine?", N.CHANNEL),
    ("What is the number for the Passport Unit?", N.CHANNEL),
    ("What are the contact numbers of the Passport Unit?", N.CHANNEL),
    ("Who should I contact about Blackboard?", N.REFERRAL),
    ("What is the number of colleges?", N.FACT),                                    # a count, not a channel ask
    ("How many colleges does the university have?", N.FACT),
    ("How do I submit an IT service request?", N.PROCEDURE),
])
def test_phase_b2_channel_phrasings(clause, kind):
    assert N.classify_ask(clause) == kind


def test_office_spans_and_resolve_offices_agree():
    for text in ("Human Resources (HR) and the Finance Department", "College of Fine Arts & Design", "the Registration office of the ITC"):
        assert N.resolve_offices(text) == tuple(dict.fromkeys(o for _s, _e, o in N.office_spans(text)))


# ---------------------------------------------------------------------------
# Negative controls: each protection off lets a wrong office or a wrong number out
# ---------------------------------------------------------------------------
KILLERS = {
    "_B2_ROW_START": ("SYN-03",),
    "_B2_ROW_COMPLETE": ("C6-07", "C6-09", "C7-05"),
    "_B2_NEIGHBOR_CHECK": ("SYN-01", "SYN-02"),
    "_B2_COLUMN_HEADER": ("SYN-04", "SYN-05"),
    "_B2_SINGLE_ENTRY": ("SYN-06",),
    "_B2_REFERENT_CHECK": ("C12-01", "C12-02", "C12-04", "C12-05", "C12-06", "C12-07"),
}


@pytest.mark.parametrize("switch", sorted(KILLERS))
def test_switching_a_protection_off_lets_a_wrong_answer_out(monkeypatch, switch):
    assert getattr(I, switch) is True
    for case_id in KILLERS[switch]:                                  # every killer passes with the protection on
        case = CASES_BY_ID[case_id]
        assert evaluate_contact_case(case, run_case(_specialist(case), case), _specialist(case))["outcome"] == "as_expected"
    monkeypatch.setattr(I, switch, False)
    for case_id in KILLERS[switch]:
        case = CASES_BY_ID[case_id]
        fresh = specialist_for_case(case)                            # the row-start rule is applied when the directory is read
        verdict = evaluate_contact_case(case, run_case(fresh, case), fresh)
        assert verdict["outcome"] == "blocker", (switch, verdict)


def test_without_row_completeness_and_neighbour_checks_the_sciences_fax_comes_out_as_its_telephone(monkeypatch):
    case = CASES_BY_ID["C6-01"]
    assert case["question"] == "What is the phone number of the College of Sciences?"
    monkeypatch.setattr(I, "_B2_ROW_COMPLETE", False)
    fresh = make_institutional()
    assert evaluate_contact_case(case, run_case(fresh, case), fresh)["outcome"] == "as_expected"     # the neighbour check still holds
    monkeypatch.setattr(I, "_B2_NEIGHBOR_CHECK", False)
    result = run_case(fresh, case)
    assert [f.metadata["contact_values"] for f in result.findings] == [{N.TELEPHONE: "+971 6 5050489"}]
    verdict = evaluate_contact_case(case, result, fresh)
    assert verdict["outcome"] == "blocker" and any("differs from the printed cell" in b for b in verdict["blockers"])


def test_all_protections_are_on():
    assert I.binding_protections() == {"row_start": True, "row_complete": True, "neighbor_check": True, "column_header": True,
                                       "single_entry": True, "referent_check": True}


# ---------------------------------------------------------------------------
# Plan F: the override runs it, and since Step 4.16-B6 (D-105) the default dispatch runs it when it is registered
# ---------------------------------------------------------------------------
def _registry(sp_):
    reg = SpecialistRegistry()
    reg.register(sp_)
    return reg


def test_the_override_answers_a_directory_question_through_the_dispatcher(sp):
    out = D.run_question("What is the phone number of Human Resources?", _registry(sp), executable=OVERRIDE)
    assert out.executed_specialists == [SpecialistId.INSTITUTIONAL] and out.llm_calls == 0
    result = out.results[0]
    assert result.status is SUPPORTED and [f.metadata["contact_values"] for f in result.findings] == [{N.TELEPHONE: "+971 6 5050023"}]


def test_the_default_dispatch_answers_a_directory_question_since_b6(sp):
    # Step 4.16-B6 (D-105): the default dispatch executes Institutional Navigation; the answer is the override's
    assert SpecialistId.INSTITUTIONAL in D.EXECUTABLE_SPECIALISTS
    out = D.run_question("What is the phone number of Human Resources?", _registry(sp))
    assert out.executed_specialists == [SpecialistId.INSTITUTIONAL] and out.pending_specialists == []
    result = out.results[0]
    assert result.status is SUPPORTED and [f.metadata["contact_values"] for f in result.findings] == [{N.TELEPHONE: "+971 6 5050023"}]


def test_a_mixed_question_gives_institutional_only_its_own_clause(sp):
    question = "What is the phone number of HR and how much annual leave do I get?"
    decision = coordinate(question)
    out = D.dispatch(decision, _registry(sp), executable=OVERRIDE, question=question)
    runs = [r for r in out.runs if r.specialist_id is SpecialistId.INSTITUTIONAL]
    assert len(runs) == 1 and runs[0].state == D.STATE_EXECUTED
    values = [f.metadata["contact_values"] for f in runs[0].findings.findings if f.metadata["directory_entry"] == "complete_row"]
    assert values == [{N.TELEPHONE: "+971 6 5050023"}]
    assert runs[0].findings.handoff_requests == []
