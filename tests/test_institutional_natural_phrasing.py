"""
Institutional Navigation natural phrasing (Milestone 4, Step 4.16-C1).

A focus clause is read as the plain question it asks (``navigation_cues.natural_ask``): a request frame ("Could you please
tell me ...", "May I have ..."), a greeting, filler words and an allowed context sentence are set aside, an embedded
question is put in direct order, and the result goes through every check of the plain forms. After two adversarial review
rounds the allowed context is closed: before the question, a greeting or thanks, a self-introduction ("I'm a new
lecturer.", "I just joined the university.") or the asker's own unit ("I'm joining the College of Medicine next month.");
after it, only a greeting or thanks. Any other sentence (a situation, a correction, a sub-unit, a person, a time, a field,
another institution) keeps the clause as written, and so does a question that refers back to its context. The rewrite must
keep the ask kind, the ownership check of phase B4 still reads the whole clause, and a composition the handbook continues
is partial. Units printed in the owned chapters with their own statements were added to the lexicon (``contact=False``:
never directory entries).

The questions here are written for these tests; none is taken from the 62-question benchmark or the C1 holdout.
No model, no LLM, no network.
"""
from __future__ import annotations

import json

import pytest

from handbook_bot.agents.specialists import institutional as I
from handbook_bot.agents.specialists import navigation_cues as N
from handbook_bot.agents.specialists.contracts import FindingStatus

from institutional_support import make_institutional, run, task

SUPPORTED, PARTIAL, NOT_FOUND, OUT_OF_SCOPE = (FindingStatus.SUPPORTED, FindingStatus.PARTIAL, FindingStatus.NOT_FOUND,
                                               FindingStatus.OUT_OF_SCOPE)
HR_TEL, HR_FAX = "+971 6 5050023", "+971 6 5585200"
FIN_TEL, FIN_FAX = "+971 6 5050040", "+971 6 5585183"


def _without_ms(value):
    if isinstance(value, dict):
        return {k: _without_ms(v) for k, v in value.items() if k != "ms"}
    if isinstance(value, list):
        return [_without_ms(v) for v in value]
    return value


@pytest.fixture(scope="module")
def sp():
    return make_institutional()


def _values(r):
    return [f.metadata.get("contact_values") for f in r.findings]


# ---------------------------------------------------------------------------
# The reader: what is read and what is never set aside
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,plain", [
    ("Could you please tell me the HR Department's phone number?", "what is the HR Department's phone number?"),
    ("Can you give me the Finance Department's fax number?", "what is the Finance Department's fax number?"),
    ("Please tell me where the Medical Library is located.", "where is the Medical Library located?"),
    ("May I have the telephone number for Human Resources?", "what is the telephone number for Human Resources?"),
    ("I'd like to know which office provides IT support.", "which office provides IT support?"),
    ("Do you know who chairs the College Council?", "who chairs the College Council?"),
    ("Hello, I'm a new lecturer. Which department handles risk assessments?", "Which department handles risk assessments?"),
    ("What does the Health and Safety Office really do?", "What does the Health and Safety Office do?"),
    ("I need to know the total number of colleges the University has.", "how many colleges does the University have?"),
    ("Where's the Fine Arts and Design Library?", "where is the Fine Arts and Design Library?"),
])
def test_natural_phrasing_is_read_as_the_plain_question(clause, plain):
    ask, notes = N.natural_ask(clause)
    assert ask == plain and notes


@pytest.mark.parametrize("clause,why", [
    ("I'm on the Men Campus. Where is the Main Library?", "campus"),
    ("I teach female students. Where is the Main Library?", "gender"),
    ("Only if it is open late. Where is the Medical Library?", "condition"),
    ("We are not from the ITC. Who provides IT support?", "negation of a unit"),
    ("I work at the Finance Department. What is the fax number?", "office in context, none in the question"),
    ("I'm in the Finance Department. What is its fax number?", "the question refers back"),
    ("I visit the University of Khorfakkan often. Where is the Medical Library?", "another institution"),
    ("My office is in building M12. Where is the Main Library?", "building code"),
    ("Who provides IT support? How do I log in?", "two questions"),
])
def test_context_that_may_restrict_the_ask_is_never_set_aside(clause, why):
    assert N.natural_ask(clause) == (clause, ()), why


def test_tell_me_about_an_office_keeps_its_plain_form(sp):
    # "tell me about X" is a phase-B3 form: the request frame is not stripped before "about" (fourth review round)
    question = "Tell me about the Disability Resource Center."
    assert N.natural_ask(question) == (question, ())
    r = run(sp, question)
    assert r.status is SUPPORTED and {f.metadata["office_id"] for f in r.findings} == {"disability_resource_center"}


def test_a_plain_question_is_unchanged():
    for clause in ("What is the phone number of Human Resources?", "Who chairs the College Council?", "Where is the Medical Library?"):
        assert N.natural_ask(clause) == (clause, ())


# ---------------------------------------------------------------------------
# Answers through the specialist
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,values", [
    ("Could you please tell me the HR Department's phone number?", [{"telephone": HR_TEL}]),
    ("May I have the fax number of Human Resources, please?", [{"fax": HR_FAX}]),
    ("Hi, I'm a new lecturer. Can you give me the Finance Department's telephone number?", [{"telephone": FIN_TEL}]),
    ("Hello. Could you send me the Finance Department's fax number? Thank you.", [{"fax": FIN_FAX}]),
    ("I'd like both the phone and the fax number of the Finance Department.", [{"telephone": FIN_TEL, "fax": FIN_FAX}]),
])
def test_polite_and_context_contact_asks_keep_the_asked_field(sp, question, values):
    r = run(sp, question)
    assert r.status is SUPPORTED and _values(r) == values, (r.status, r.missing)
    assert r.metadata["clauses"][0]["read_as"] and r.metadata["clauses"][0]["clause"] == question


@pytest.mark.parametrize("question,office", [
    ("I'm a new faculty member. Could you tell me which office handles IT services?", "information_technology_center"),
    ("Please tell me which department provides IT support.", "information_technology_center"),
    ("Hello. What does the Disability Resource Center do?", "disability_resource_center"),
])
def test_polite_and_context_responsibility_asks(sp, question, office):
    r = run(sp, question)
    assert r.status is SUPPORTED and {f.metadata["office_id"] for f in r.findings} == {office}, (r.status, r.missing)


@pytest.mark.parametrize("question,code", [
    ("Please tell me where the Medical Library is located.", None),
    ("I'm new here. Where can I find the Medical Library?", None),
    ("Where's the Fine Arts and Design Library?", "M22"),
    ("Could you tell me the building code for the women's Main Library?", "W16"),
])
def test_polite_and_context_location_asks(sp, question, code):
    r = run(sp, question)
    assert r.status is SUPPORTED and len(r.findings) == 1, (r.status, r.missing)
    assert r.findings[0].metadata["relation"] == "location" and r.findings[0].metadata["building_code"] == code


def test_a_campus_or_gender_restriction_in_context_fails_closed(sp):
    for question in ("I'm on the Women Campus. Where is the Main Library?", "I teach male students. Where is the Main Library?"):
        r = run(sp, question)
        assert r.findings == [] and r.status is NOT_FOUND and "read_as" not in r.metadata["clauses"][0], question


def test_office_identity_is_preserved(sp):
    # the context names the asker's own unit, the question another: the answer is the question's office, never the context's
    r = run(sp, "I teach in the College of Medicine. What is the phone number of Human Resources?")
    assert r.status is SUPPORTED and [f.metadata["office_id"] for f in r.findings] == ["human_resources"]
    # "I'm in X" may be a place as well as a unit: not on the allowlist, so the clause is read as written
    question = "I'm in the College of Medicine. What is the phone number of Human Resources?"
    assert N.natural_ask(question) == (question, ())
    # a question that names no office never takes the context's office
    r = run(sp, "I'm in the College of Medicine. What is the phone number?")
    assert r.findings == []


def test_unsupported_procedures_logins_and_approvals_stay_unanswered(sp):
    for question in ("Could you please tell me how to log in to MyUOS?", "I'm new. Can you tell me the steps to get a car sticker?",
                     "Please tell me who approves purchase requests.", "Could you give me the e-mail address of Human Resources?"):
        r = run(sp, question)
        assert r.findings == [] and r.status in (NOT_FOUND, OUT_OF_SCOPE), question
        text = json.dumps(_without_ms(r.to_dict()))
        assert "@" not in text.replace("@pytest", "") and "http" not in text


def test_the_ownership_check_still_reads_the_context(sp):
    # the HR subject is in the context sentence: the clause is handed to HR, not answered from Institutional text
    r = run(sp, "I want to ask about my annual leave. Who should I contact about it?")
    assert r.findings == [] and r.status is OUT_OF_SCOPE


# ---------------------------------------------------------------------------
# The added units (printed on their pages; never directory entries)
# ---------------------------------------------------------------------------
NEW_UNITS = ("central_laboratories_directorate", "security_services_unit", "transport_section", "mail_office", "facilities_management_section",
             "projects_management_section", "deans_council", "university_academic_advisory_council")


def test_the_added_units_are_chapter_units_not_directory_entries():
    for office_id in NEW_UNITS:
        office = N.OFFICES_BY_ID[office_id]
        assert office.contact is False and office.pages, office_id
    assert ("central_laboratories", "central_laboratories_directorate") in N.NOT_EQUIVALENT
    assert ("facilities_management_section", "physical_plant") in N.NOT_EQUIVALENT


def test_a_contact_ask_for_an_added_unit_gives_no_number(sp):
    for question in ("What is the phone number of the Mail Office?", "What is the fax number of the Transport Section?"):
        r = run(sp, question)
        assert not any(f.metadata.get("contact_values") for f in r.findings), question


@pytest.mark.parametrize("question,office", [
    ("Who issues permits on campus?", "security_services_unit"),
    ("What does the Mail Office do?", "mail_office"),
    ("What does the Central Laboratories Directorate do?", "central_laboratories_directorate"),
])
def test_an_added_unit_is_named_only_from_its_own_statements(sp, question, office):
    r = run(sp, question)
    assert r.findings and {f.metadata["office_id"] for f in r.findings} <= {office, None}, (r.status, r.missing)
    for f in r.findings:
        assert f.claim == f.evidence_quote and f.page in N.OFFICES_BY_ID[office].pages + (250, 251, 252, 253, 254)


# ---------------------------------------------------------------------------
# Negative controls and determinism
# ---------------------------------------------------------------------------
def test_without_the_context_guard_a_campus_restriction_is_dropped(sp, monkeypatch):
    question = "I'm on the Women Campus. Where is the Main Library?"
    assert run(sp, question).findings == []
    monkeypatch.setattr(I, "_C1_CONTEXT_GUARD", False)
    r = run(sp, question)
    assert r.metadata["clauses"][0].get("read_as") == "Where is the Main Library?" and r.findings   # the restriction is lost
    monkeypatch.setattr(I, "_C1_CONTEXT_GUARD", True)
    assert run(sp, question).findings == []


def test_without_natural_phrasing_the_b6_reading_returns(sp, monkeypatch):
    question = "Could you please tell me the HR Department's phone number?"
    assert run(sp, question).status is SUPPORTED
    monkeypatch.setattr(I, "_C1_NATURAL_ASK", False)
    assert run(sp, question).status is NOT_FOUND
    monkeypatch.setattr(I, "_C1_NATURAL_ASK", True)
    assert I.c1_protections() == {"natural_ask": True, "context_guard": True, "kind_change_check": True, "composition_continues": True}


# ---------------------------------------------------------------------------
# The adversarial review of Step 4.16-C1 (three lenses): every confirmed finding stays closed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause", [
    # a correction moves the ask to another unit
    "What is the fax number of the College of Medicine? Sorry, I meant the College of Dental Medicine.",
    "What is the phone number of HR? Actually I need Finance.",
    "Who chairs the Deans Council? Oops, I meant the UAAC.",
    "What is the phone number of Physical Plant? I mean the Facilities Management Section.",
    "What is the phone number of the Information Technology Center? I am asking about the SSU.",
    "Who chairs the Deans Council, I am asking about the UAAC",
    # a sub-unit or a person
    "What is the phone number of the College of Medicine? I mean the library.",
    "I'm trying to reach the dean. What is the phone number of the College of Medicine?",
    "What is the phone number of the College of Dental Medicine, I am trying to reach the clinic.",
    "What is the phone number of the Finance Department? For the cashier.",
    "What is the phone number of the College of Medicine? For Dr. Ali.",
    "I am looking for the security desk. What is the phone number of the Physical Plant?",
    # a unit inside another unit
    "The College of Law has its own Finance Department. What is the phone number of the Finance Department?",
    "I work in the College of Law's finance office. What is the phone number of the Finance Department?",
    "Ref. College of Law. What is the fax number of Human Resources?",
    # a time or absence condition
    "Who chairs the College Council? When the Dean is away.",
    "Who chairs the Deans Council? In the absence of the Chancellor.",
    # another institution or place
    "Where is the Medical Library? I'm asking about the one at the University Hospital Sharjah.",
    "What is the phone number of Human Resources? I work at Sharjah Municipality.",
    # a contact field: asked, ruled out or added
    "What does the Deans Council do? I need their phone number.",
    "I am deaf and cannot use voice calls. What is the contact number of the Disability Resource Center?",
    "I need to fax them. What is the contact number of the ITC?",
    "I'd also like the fax. What is the phone number of Human Resources?",
    "I have their phone number. Who is responsible for proposing study plans?",
    # a situation sentence: not on the allowlist, before or after the question
    "Hi, I started last week. Can you give me the Finance Department's telephone number?",
    "I need to send a form. Could you send me the Finance Department's fax number?",
    "My printer is broken. What is the phone number of the Information Technology Center?",
    # an informal correction or redirect after the question (second review round)
    "What is the fax number of the College of Medicine? i was thinking of the dental college.",
    "What is the fax number of the College of Medicine? The dental one.",
    "What is the fax number of the Central Laboratories? i was asking about the cld.",
    "Who chairs the College Council? I'm asking about the deputy.",
    "How many colleges does the University have? I'm asking about the medical ones.",
    "What is the phone number of the Finance Department? i teach at zayed university.",
    "What is the phone number of the College of Medicine, I want to talk to the chairman.",
    # an own-unit introduction whose office name carries a gender
    "I work in the Women's Library. Where is the Main Library?",
    # the question refers back to the asker's own unit (third review round)
    "I work in the College of Law. What is the phone number of the Finance Department for my college?",
    "I work in the College of Law. What is the fax number of the Finance Department for the latter?",
    "I teach in the College of Engineering. What is the phone number of the Finance Department for our faculty?",
    # an own-unit introduction without its subject
    "Joined with the College of Pharmacy. What is the phone number of HR?",
    # the question refers back to the own unit with a unit word (fourth review round)
    "I work in the College of Law. What is the phone number of Registration office for the college?",
    "I work in the College of Law. What is the fax number of the Finance Department for the college I work in?",
    "I teach in the College of Medicine. What is the phone number of Human Resources for the college's staff?",
    # gender; another institution split by a rewrite
    "I am a woman. What is the phone number of the College of Law?",
    "Please tell me how many colleges the American University of Sharjah has.",
])
def test_the_review_contexts_are_never_set_aside(clause):
    assert N.natural_ask(clause) == (clause, ())


@pytest.mark.parametrize("clause,plain", [
    ("What is the fax number of the College of Medicine, actually the College of Dental Medicine?", None),
    ("What is the phone number of Human Resources or actually the Finance Department?", None),
    # third review round: a correction without a comma or a determiner
    ("What is the phone number of Physical Plant or actually Facilities Management Section?", None),
    ("What is the fax number of the College of Medicine or actually College of Dental Medicine?", None),
    ("What is the phone or actually fax number of HR?", None),
    ("Where is the Main Library for men or actually women?", None),
    ("What is the phone number of the Finance Department as well as actually HR?", None),
    ("Who actually chairs the College Council?", "Who chairs the College Council?"),
    ("What is the Transport Section actually responsible for?", "What is the Transport Section responsible for?"),
    ("Where exactly is the Medical Library?", "Where is the Medical Library?"),
])
def test_a_filler_word_that_marks_a_correction_is_kept(clause, plain):
    # "X (or) actually Y" corrects the office, field or gender: removing "actually" would turn it into a list of two; a filler
    # word is removed only between a word and a verb or predicate
    assert N.natural_ask(clause) == ((clause, ()) if plain is None else (plain, ("filler word",)))


def test_a_rewrite_never_skips_the_ownership_check(sp, monkeypatch):
    # a contact word makes the whole clause a contact ask, which the B4 ownership check does not read; the plain ask
    # would be a Teaching subject. With the context guard off (defence in depth), the kind-change check still reads the
    # clause as written; with both off, the Teaching subject is answered from Institutional text.
    question = "I have their phone number. Who is responsible for proposing study plans?"
    assert run(sp, "Who is responsible for proposing study plans?").status is OUT_OF_SCOPE
    assert run(sp, question).findings == []
    monkeypatch.setattr(I, "_C1_CONTEXT_GUARD", False)
    assert run(sp, question).findings == []
    monkeypatch.setattr(I, "_C1_KIND_CHANGE_CHECK", False)
    assert run(sp, question).findings                                           # the protection is needed


def test_a_composition_the_handbook_continues_is_partial(sp, monkeypatch):
    # p27: "The Council consists of faculty members ... Additionally, seven students are elected ..."
    r = run(sp, "Who are the members of the UAAC?")
    assert r.status is PARTIAL and any("adds more members" in m for m in r.missing)
    assert run(sp, "Who are the members of the College Council?").status is SUPPORTED      # p25: nothing added after the list
    monkeypatch.setattr(I, "_C1_COMPOSITION_CONTINUES", False)
    assert run(sp, "Who are the members of the UAAC?").status is SUPPORTED     # the protection is needed


def test_an_added_unit_keeps_the_subject_binding_protection(sp, monkeypatch):
    # p28 "It provides recommendations and consultations to the Deans Council ..." is the UAAC's duty: the Deans Council is
    # its object, not its subject. With the B3 subject rule switched off it is read as a Deans Council duty.
    wrong = "provides recommendations and consultations to the Deans Council"
    r = run(sp, "What does the Deans Council do?")
    assert r.status is SUPPORTED and {f.page for f in r.findings} == {24}
    assert not any(wrong in f.evidence_quote for f in r.findings)
    monkeypatch.setattr(I, "_B3_SUBJECT_BINDING", False)
    mutated = run(make_institutional(), "What does the Deans Council do?")     # statements are read when the specialist is built
    assert any(wrong in f.evidence_quote and f.metadata["office_id"] == "deans_council" for f in mutated.findings)


def test_natural_phrasing_is_deterministic(sp):
    other = make_institutional()
    for question in ("Could you please tell me the HR Department's phone number?", "I'm new here. Where can I find the Medical Library?",
                     "I'm on the Women Campus. Where is the Main Library?"):
        assert _without_ms(run(sp, question).to_dict()) == _without_ms(run(other, question).to_dict())
        assert N.natural_ask(question) == N.natural_ask(question)
