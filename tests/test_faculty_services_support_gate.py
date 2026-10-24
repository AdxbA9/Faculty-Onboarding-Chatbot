"""
Faculty Services & HR specific-claim support gate (Milestone 4, Step 4.12-C12, D-099): SPECIFIC HR CLAIM + NO COMPLETE
STRUCTURED PROOF = NEVER SUPPORTED. The specific / general boundary and the wrapper normalisation (class F); the gate with no
claim, no frame or no statement; the hardened ledger (no static operator); the lexical clause boundary and object binding
(class A), comparisons (class B), material conditions with no copular bypass (class C), important modifiers (class D) and
alternative fillers (class E); the R11 wrong answers on the fixture pages and on crafted pages under both rankings; wrapped
and prefixed adversarial transformations; general questions unchanged; the switches.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from handbook_bot.agents.specialists import claim_model as C
from handbook_bot.agents.specialists import faculty_services as F
from handbook_bot.agents.specialists.contracts import FindingStatus

from faculty_services_support import C12_CORPUS_PAGES, ScriptedReranker, corpus_from, make_specialist, page, registry_and_map, run

SUPPORTED = FindingStatus.SUPPORTED
COMPATIBLE, CONTRADICTED = C.Compatibility.COMPATIBLE, C.Compatibility.CONTRADICTED
COMPOSITE = "the asked claim stated in one statement"
GATE = "the specific claim read as a structured claim"
C12_SWITCHES = sorted(n for n in dir(F) if n.startswith("_C12_") and isinstance(getattr(F, n), bool))


def gate(question):
    return F._specific_claim_support_gate(question)


def local(question, unit):
    return gate(question)[2].check(unit)


def window(question, quote):
    return gate(question)[2].check.window(quote)


def frame_of(question):
    text = F.specific_claim_text(question)
    return F.read_claim_frame(text, F._c12_read_claim(text))


# ---------------------------------------------------------------------------
# The specific / general boundary and wrapper normalisation (class F)
# ---------------------------------------------------------------------------
SPECIFIC = [   # (question, the text its claim is read from: "=" unchanged, None unreadable)
    ("Is the faculty probation period one year?", "="),
    ("Hi, is the faculty probation period one year?", "is the faculty probation period one year?"),
    ("Hello! Must Deans hold the rank of full professor?", "Must Deans hold the rank of full professor?"),
    ("Please, can the probation period be extended with Chancellor approval?", "can the probation period be extended with Chancellor approval?"),
    ("Can you tell me whether faculty members must apply for the bonus award?", "must faculty members apply for the bonus award?"),
    ("Can you tell me if faculty members receive a housing allowance?", "do faculty members receive a housing allowance?"),
    ("I want to know whether unpaid special leave can exceed thirty days.", "can unpaid special leave exceed thirty days?"),
    ("I want to know if the Faculty Grievance Committee must contact witnesses.", "must the Faculty Grievance Committee contact witnesses?"),
    ("Do you know whether HR drafts employment offers?", "does HR draft employment offers?"),
    ("Do you know if suspension from duty always results in the suspension of salary?", "does suspension from duty always result in the suspension of salary?"),
    ("Could you tell me whether the Dean must hold the rank of full professor?", "must the Dean hold the rank of full professor?"),
    ("Would you tell me whether the Chancellor appoints the Central Hiring Committee?", "does the Chancellor appoint the Central Hiring Committee?"),
    ("I'd like to know whether faculty members may apply for the bonus award in consecutive years.",
     "may faculty members apply for the bonus award in consecutive years?"),
    ("Is it true that all senior administrative appointments require Chancellor approval?", "do all senior administrative appointments require Chancellor approval?"),
    ("Termination during probation doesn't require the Dean's approval, right?", "does Termination during probation not require the Dean's approval?"),
    ("I heard that HR is the only unit that assists with entry visas. Is that right?", "is HR the only unit that assists with entry visas?"),
    ("Please confirm: must faculty members apply for the bonus award?", "must faculty members apply for the bonus award?"),
    ("Quick question - does health insurance cover all dependent children of faculty members?",
     "does health insurance cover all dependent children of faculty members?"),
    ("Hi, can you tell me whether HR only drafts employment offers?", "does HR only draft employment offers?"),
    ("Do faculty members receive a housing allowance, please?", "Do faculty members receive a housing allowance?"),
    ("Just wondering whether faculty members must apply for the bonus award?", "must faculty members apply for the bonus award?"),
    ("Any idea if the furniture allowance is paid in the first month?", "is the furniture allowance paid in the first month?"),
    ("I'm not sure whether HR drafts employment offers.", "does HR draft employment offers?"),
    ("Can you confirm that the probation period is one year?", "is the probation period one year?"),
    ("Just to check, is it true that all senior administrative appointments require Chancellor approval?",
     "do all senior administrative appointments require Chancellor approval?"),
    ("Faculty members must apply for the bonus award?", "must Faculty members apply for the bonus award?"),
    ("Is it necessary to check if the form is signed?", "="),
    ("Does the Dean decide whether the probation is extended?", "="),
    ("For new faculty, can you tell me whether the probation period is one year?", "For new faculty, is the probation period one year?"),
    ("When the department has fewer than three eligible faculty, is the same procedure used at every college?", "="),
    ("If a faculty member resigns during probation, are they entitled to benefits?", "="),
    ("In general, must Deans hold the rank of full professor?", "="),
    ("Typically, is the faculty probation period only one year?", "="),
    ("Hi, in general, must Deans hold the rank of full professor?", "in general, must Deans hold the rank of full professor?"),
    ("Is the probation period two (2) years?", "Is the probation period two years?"),                      # D-100 (Step 4.14, P2)
    ("Is maternity leave with full pay sixty (60) calendar days?", "Is maternity leave with full pay sixty calendar days?"),
    ("Is the Dean's monthly compensation five thousand (5,000) AED?", "Is the Dean's monthly compensation five thousand AED?"),
    ("Is the notice period described in Article (3) six months?", "="),                                      # not a repeated number
    ("Aren't terminated faculty entitled to return tickets?", None),
    ("Cannot faculty who resign during probation receive ticket allowances?", None),
    ("The Employee Handbook says settling in the UAE is handled by HR?", None),
]
GENERAL = [
    "Tell me about maternity leave.", "What is the probation policy?", "Can you tell me about the furniture allowance?", "Could you explain the promotion process?",
    "Do you know what the probation period is?", "Can you tell me how to resign?", "Are there any rules about employment of relatives?",
    "Hi, what is the probation policy?", "How do I apply for leave?", "Where can I find the HR form?", "Please explain the end-of-service gratuity.",
    "What happens if I resign during probation?", "Who decides whether the probation is extended?", "Can you tell me what happens if I resign?", "Sick leave?",
    "On which day of the month is the salary paid?",
    "At the department level, within how many months must the chair forward the promotion committee's report to the dean?",
]


@pytest.mark.parametrize("question,reading", SPECIFIC)
def test_a_specific_claim_and_the_text_its_claim_is_read_from(question, reading):
    assert F.is_specific_claim(question)
    assert F.specific_claim_text(question) == (question if reading == "=" else reading)


@pytest.mark.parametrize("question", GENERAL)
def test_an_overview_a_wh_question_a_procedure_or_an_open_request_is_general(question):
    assert not F.is_specific_claim(question)


@pytest.mark.parametrize("question,operators", [
    ("Can you tell me whether all faculty members must always attend the induction workshop?", ("all", "must", "always")),
    ("Hi, is it true that only Muslim faculty may take Hajj leave?", ("only", "may")),
    ("I want to know if the allowance is paid under 5,000 Dirhams unless the Dean objects.", ("under", "unless")),
    ("Do you know whether every faculty member may request additional training leave?", ("every", "may", "additional")),
    ("Please confirm: is the allowance over 5,000 Dirhams if the faculty member is new?", ("over", "if")),
])
def test_wrapper_normalisation_never_strips_a_semantic_operator(question, operators):
    text = F.specific_claim_text(question)
    assert all(re.search(r"\b%s\b" % op, text) for op in operators), text


# ---------------------------------------------------------------------------
# The gate: no claim, no frame or no statement never supports
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "Currently, must faculty members apply for the bonus award?",                              # an unknown leading word
    "Normally, is sick leave without pay granted in all circumstances?",                      # two frequency operators
    "Usually, can the Faculty Grievance Committee always contact witnesses upon request?",
    "Aren't terminated faculty entitled to return tickets?",                                  # a negative-contraction lead
    "The Employee Handbook says settling in the UAE is handled by HR?",                       # a reported clause
])
def test_a_specific_claim_with_no_structured_proof_carries_a_detail_no_unit_covers(question):
    _claim, _details, detail = gate(question)
    assert F.is_specific_claim(question) and detail is not None and detail.label == GATE
    assert detail.check("Faculty members must apply for the bonus award; sick leave without pay is granted in all circumstances.") is False


def test_two_frequency_operators_leave_no_usable_frame():
    q = "Usually, can the Faculty Grievance Committee always contact witnesses upon request?"
    claim = F._c12_read_claim(F.specific_claim_text(q))
    assert claim is not None and F.read_claim_frame(F.specific_claim_text(q), claim) is None


@pytest.mark.parametrize("question", ["Hi, must faculty members submit the annual leave request?",
                                      "Can you tell me whether faculty members must submit the annual leave request?"])
@pytest.mark.parametrize("unit", ["The library opens at eight.", "Probation is explained in the handbook."])
def test_a_unit_with_no_statement_never_supports_a_wrapped_claim(question, unit):
    assert gate(question)[2].label == COMPOSITE and local(question, unit) is False


@pytest.mark.parametrize("question", ["Tell me about maternity leave.", "Can you tell me about the furniture allowance?", "How do I apply for leave?",
                                      "What is the probation policy?", "Hi, what is the probation policy?"])
def test_a_general_question_keeps_its_c10_c11_reading(question):
    claim, details, detail = gate(question)
    expected = F.read_claim(question)
    assert claim == expected and [d.label for d in details] == [d.label for d in F.structured_details(question, expected)]
    assert detail is None or detail.label != GATE


@pytest.mark.parametrize("question", ["Hi, is the faculty probation period one year?", "In general, must Deans hold the rank of full professor?",
                                      "Does health insurance cover all children of faculty members?"])
def test_a_specific_claim_has_no_static_operator_in_its_ledger(question):
    frame = frame_of(question)
    assert frame is not None and all(e.handler != "static" for e in frame.ledger)


@pytest.mark.parametrize("question,cls,handler", [
    ("Does health insurance cover all children of faculty members?", "scope", "unverified"),      # a universal no slot reader holds
    ("Do all University-level academic administrators receive regular academic appointments?", "scope", "scope"),   # a modifier, no time
    ("Is the allowance paid each year?", "scope", "value_time"),                                  # a time distributive
    ("Is the housing allowance under 5,000 Dirhams?", "comparison", "value_time"),
    ("In general, must Deans hold the rank of full professor?", "frequency", "frequency"),
])
def test_every_operator_is_owned_by_a_dimension_that_evaluates_it(question, cls, handler):
    assert any(e.cls == cls and e.handler == handler for e in frame_of(question).ledger)


# ---------------------------------------------------------------------------
# Class B: comparisons
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phrase,relation", [
    ("under", "<"), ("over", ">"), ("below", "<"), ("above", ">"), ("less than", "<"), ("more than", ">"), ("at least", ">="), ("at most", "<="),
    ("no more than", "<="), ("no less than", ">="), ("equal to", "=="), ("higher than", ">"), ("lower than", "<"), ("up to", "<="),
    ("not exceeding", "<="), ("exceed", ">"),
])
def test_every_comparison_operator_is_read_and_never_disappears(phrase, relation):
    q = "Is the housing allowance %s 5,000 Dirhams?" % phrase
    assert [(c.relation, c.number, c.unit) for c in F._c12_comparisons(F.cue_text(q.lower()))] == [(relation, 5000.0, "currency")]
    assert any(e.cls == "comparison" and e.handler == "value_time" for e in frame_of(q).ledger)


@pytest.mark.parametrize("question,unit,ok", [
    ("Is the housing allowance over 5,000 Dirhams?", "The housing allowance is 5,000 Dirhams.", False),
    ("Is the housing allowance under 5,000 Dirhams?", "The housing allowance is 5,000 Dirhams.", False),
    ("Is the housing allowance 5,000 Dirhams?", "The housing allowance is 5,000 Dirhams.", True),
    ("Is the housing allowance over 4,000 Dirhams?", "The housing allowance is 5,000 Dirhams.", False),   # the C9 value contract asks its number
    ("Are faculty members entitled to sick leave with full pay for at least 15 working days?",
     "Faculty members are entitled to sick leave with full pay for 15 working days.", True),
    ("Are faculty members entitled to sick leave with full pay for more than 15 working days?",
     "Faculty members are entitled to sick leave with full pay for 15 working days.", False),
    ("Is the monthly compensation of a dean under AED 10,000?", "Dean: Monthly compensation of AED 10,000 and a reduction in teaching load of nine credit hours per semester.", False),
    ("Is the one-time furniture allowance under 30,000 Dirhams?", "A one-time furniture allowance of 30,000 Dirhams is provided, which can be utilized over four years.", False),
    ("Is the Faculty Grievance Committee composed of over seven faculty members?",
     "The Faculty Grievance Committee, composed of seven faculty members elected annually from the faculty at-large, shall be responsible for addressing grievances.", False),
    ("Is the Faculty Grievance Committee composed of seven faculty members?",
     "The Faculty Grievance Committee, composed of seven faculty members elected annually from the faculty at-large, shall be responsible for addressing grievances.", True),
])
def test_a_comparison_is_proved_only_by_its_restatement_or_a_safe_deterministic_value(question, unit, ok):
    assert local(question, unit) is ok


@pytest.mark.parametrize("question,statement,holds", [
    ("is the monthly compensation of a director up to aed 3000", "Monthly compensation of AED 2,000 to AED 3,000 and a reduction in teaching load.", True),
    ("is the monthly compensation of a director under aed 3000", "Monthly compensation of AED 2,000 to AED 3,000 and a reduction in teaching load.", False),
    ("is the clinical allowance up to 20 percent of the basic salary", "Clinical allowance ranging from 10% to 20% of the basic salary.", True),
    ("can up to 30 days of sick leave without pay be granted", "In special circumstances, up to an additional 30 days of sick leave without pay may be granted.", True),
    ("can unpaid special leave exceed thirty days", "Unpaid special leave, not exceeding thirty days, may be granted upon approval.", False),
    ("is the probation period under 4 months", "The probation period is 2 years and 3 months.", False),               # another time unit
    ("do faculty members receive more than 30 days of leave", "Faculty members receive 30 days of annual leave and 10 days of sick leave.", False),
    ("are faculty members over the age of 60 offered renewable contracts", "Faculty members over the age of 65 will be offered renewable one-year contracts.", False),
    ("will faculty members over the age of 65 be offered renewable contracts", "Faculty members over the age of 65 will be offered renewable one-year contracts.", True),
    ("is health insurance offered for children under 18", "Health insurance covers dependent children under 16 years old.", False),  # a population: exact only
])
def test_the_comparison_contract(question, statement, holds):
    frame = F.read_claim_frame(question + "?", F.read_claim(question + "?"))
    c = F._c12_comparisons(question, frame.anchor_roots if frame is not None else frozenset())[0]
    assert F._c12_comparison_holds(c, SimpleNamespace(text=statement)) is holds          # the comparison reads the statement text only


# ---------------------------------------------------------------------------
# Class A: the lexical clause boundary and object binding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sentence,clauses", [
    ("Professors receive a housing allowance of 90,000 Dirhams, and lecturers receive a housing allowance of 60,000 Dirhams.", 2),
    ("Faculty members submit the leave request through the HR portal or they email the department chair.", 2),
    ("The probation period lasts one year, yet the Chancellor may shorten it to six months.", 2),
    ("Faculty members must submit the resignation letter to the Dean and HR calculates the end-of-service gratuity.", 2),
    ("The appointment of clinical faculty is limited to Dental Medicine and Veterinary Medicine.", 1),        # an enumeration never splits
    ("Faculty members, staff members and visitors must register at the front desk.", 1),
])
def test_a_coordinated_clause_with_a_new_subject_and_a_finite_verb_is_its_own_clause(sentence, clauses):
    assert len(F._c12_clause_spans(sentence)) == clauses


@pytest.mark.parametrize("question,unit,ok", [
    ("Do professors receive a housing allowance of 60,000 Dirhams?",
     "Professors receive a housing allowance of 90,000 Dirhams, and lecturers receive a housing allowance of 60,000 Dirhams.", False),
    ("Do lecturers receive a housing allowance of 60,000 Dirhams?",
     "Professors receive a housing allowance of 90,000 Dirhams, and lecturers receive a housing allowance of 60,000 Dirhams.", True),
    ("Do faculty members email the leave request to HR?", "Faculty members submit the leave request through the HR portal or they email the department chair.", False),
    ("May the Chancellor shorten the probation period to six months?", "The probation period lasts one year, yet the Chancellor may shorten it to six months.", True),
    ("Is a new faculty member's employment considered confirmed?",                            # the asked verb is stated: the object
     "New faculty members must receive written notification of employment confirmation or termination six months before the end of the contract. "
     "If neither occurs, the new faculty member's employment is considered confirmed.", False),                                                                 # mention with its own subject is no statement
])
def test_a_claim_binds_inside_one_clause(question, unit, ok):
    assert local(question, unit) is ok


# ---------------------------------------------------------------------------
# Class C: material conditions, no copular bypass
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,unit,ok", [
    ("Is the probation period three months?", "If the visiting appointment is for one semester, the probation period is three months.", False),
    ("Is a new faculty member's employment considered confirmed?", "New faculty members must receive written notification of employment confirmation or termination six months before the end of the contract. "
     "If neither occurs, the new faculty member's employment is considered confirmed.", False),
    ("Is the probation period of visiting faculty six months?", "In special circumstances, the probation period of visiting faculty is six months.", False),
    ("Is the furniture allowance paid in the first month?", "The furniture allowance is paid in the first month unless the faculty member defers it.", False),
    ("Unless the Disciplinary Council specifies otherwise, does suspension from duty result in the suspension of salary?",
     "Unless otherwise specified by the Disciplinary Council, suspension from duty also results in the suspension of salaries.", True),   # asked
    ("Is the probation period of visiting faculty six months in special circumstances?",
     "In special circumstances, the probation period of visiting faculty is six months.", True),
    ("Is the housing advance paid?", "The housing advance is paid subject to the Dean's approval.", True),                     # an approval gate
    # D-102 (Step 4.14-W1, Option A): a "when" condition the question omits is a material condition, as "if" is. This row was True under
    # D-098 / D-099 (an accepted "when" trigger); the salary increases from the following month only when the faculty member is promoted.
    ("Does the basic salary increase from the following month?",
     "When a faculty member is promoted, the basic salary increases from the following month.", False),                     # the event omitted
    ("When a faculty member is promoted, does the basic salary increase from the following month?",
     "When a faculty member is promoted, the basic salary increases from the following month.", True),                      # the event asked
    ("Does the basic salary increase from the following month when a faculty member is promoted?",
     "When a faculty member is promoted, the basic salary increases from the following month.", True),                      # the event asked
    ("Does a faculty member's primary and continuous appointment remain their faculty role?",
     "A faculty member's primary and continuous appointment remains their faculty role, even when holding an administrative title.", True),  # concessive
    ("Does the College Academic Promotions Committee review promotion applications from various college departments?",
     "The College Academic Promotions Committee shall review promotion applications from various college departments to determine if candidates "
     "meet the requirements for promotion.", True),                                                                           # a complement "if"
    ("Does the College Council form the College Academic Promotions Committee?",
     "In September of each year, the College Council shall form the College Academic Promotions Committee, chaired by the college dean if a full "
     "professor or by another professor within the college if the dean is not a full professor.", True),                     # a secondary predication
])
def test_an_unconditional_claim_is_never_proved_by_a_statement_that_holds_only_under_a_condition(question, unit, ok):
    assert local(question, unit) is ok


def test_a_negative_claim_meets_the_unconditional_check_too():
    unit = "The furniture allowance is not paid in the first month unless the faculty member asks for it."
    assert local("Is the furniture allowance not paid in the first month?", unit) is False


# ---------------------------------------------------------------------------
# Class D: important modifiers; class E: alternative fillers
# ---------------------------------------------------------------------------
DAPC = ("The Department Academic Promotions Committee should comprise at least three department faculty members holding an academic rank "
        "equivalent to or higher than the rank applied for.")


@pytest.mark.parametrize("question,unit,ok", [
    ("Does promotion require a recommendation from the college committee?", "Promotion requires a recommendation from the department committee.", False),
    ("Does promotion require a recommendation from the department committee?", "Promotion requires a recommendation from the department committee.", True),
    ("Does promotion require a recommendation from the college committee?",
     "Promotion requires a recommendation from the department committee and the college committee.", True),
    ("Must the Department Academic Promotions Committee comprise at least three college faculty members?", DAPC, False),
    ("Is a visiting appointment a definite term contract of up to two years?",
     "Visiting Appointment: Involves a definite term contract of up to two years, including a probation year.", True),   # the list label
    ("May senior faculty members apply for the bonus award?", "Faculty members may apply for the bonus award in consecutive years.", False),
    ("May faculty members apply for the bonus award?", "Faculty members may apply for the bonus award in consecutive years.", True),
])
def test_an_important_modifier_is_stated_and_another_of_its_family_vetoes(question, unit, ok):
    assert local(question, unit) is ok


@pytest.mark.parametrize("question,unit,ok", [
    ("May visiting faculty apply for the bonus award?", "Parttime and visiting faculty members are not eligible for the bonus award.", False),
    ("Are visiting faculty members not eligible for the bonus award?", "Parttime and visiting faculty members are not eligible for the bonus award.", True),
    ("Is there equal opportunity in employment at UoS?", "Prohibited Nepotism and Employment of Relatives The University of Sharjah is committed to "
     "maintaining a fair and equal opportunity employment policy, emphasizing competence and merit.", True),           # a heading word negates nothing
])
def test_a_fallback_statement_meets_the_polarity_of_the_claim(question, unit, ok):
    # No anchor of the asked predicate in the unit: the C11 polarity readers skip it; D-099 reads its negator.
    assert local(question, unit) is ok


@pytest.mark.parametrize("quote,veto", [
    ("Faculty members receive ten days of training leave and may request additional training leave.", True),
    ("Faculty members receive ten days of training leave; additional days may be granted for training leave.", True),
    ("Faculty members receive ten days of training leave. Additional documents may be requested.", False),
    ("Faculty members receive ten days of training leave. Additional sick leave may be granted.", False),
])
def test_an_exclusive_value_is_vetoed_only_by_another_filler_of_the_same_slot(quote, veto):
    assert window("Do faculty members receive only ten days of training leave?", quote) is (CONTRADICTED if veto else COMPATIBLE)


# ---------------------------------------------------------------------------
# Fixture pages and crafted pages, both rankings
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def specialists():
    corpus = corpus_from(registry_and_map(), C12_CORPUS_PAGES)
    return {"raw": make_specialist(corpus), "high": make_specialist(corpus, ScriptedReranker([], default=9.0))}


R11_REAL = [
    "In general, does health insurance cover all dependent children of faculty members?", "Typically, is the appointment of clinical faculty limited to Medicine and Law?",
    "As a rule, do all senior administrative appointments require Chancellor approval?",
    "Usually, can the Faculty Grievance Committee always contact witnesses upon request?", "Normally, is sick leave without pay granted in all circumstances?",
    "Currently, must faculty members apply for the bonus award?", "In general, must the Faculty Grievance Committee contact witnesses?",
    "Can you tell me whether faculty members must apply for the bonus award?", "I want to know if the Faculty Grievance Committee must contact witnesses.",
    "Hi, do all senior administrative appointments require Chancellor approval?", "Do you know if suspension from duty always results in the suspension of salary?",
    "Does health insurance cover all children of faculty members?", "Is the one-time furniture allowance under 30,000 Dirhams?",
    "Is the monthly compensation of a dean under AED 10,000?", "Is a new faculty member's employment considered confirmed?",
]


@pytest.mark.parametrize("question", R11_REAL)
def test_the_r11_wrong_answers_are_never_supported(specialists, question):
    for ranking in ("raw", "high"):
        r = run(specialists[ranking], question)
        assert r.status is not SUPPORTED, (ranking, question)
        assert any("the asked claim" in m or "vetoes the asked claim" in m or "could not be read as a structured claim" in m for m in r.missing), (ranking, r.missing)


@pytest.mark.parametrize("question", ["Hi, is the faculty probation period one year?", "Can you tell me if the furniture allowance is 30,000 Dirhams?",
                                      "Must Deans hold the rank of full professor?", "Do you know whether Deans must hold the rank of full professor?"])
def test_a_wrapped_specific_claim_is_still_supported_through_its_structured_proof(specialists, question):
    for ranking in ("raw", "high"):
        r = run(specialists[ranking], question)
        assert r.status is SUPPORTED, (ranking, question, r.missing)
        assert any(COMPOSITE in f.metadata["details_covered"] for f in r.findings), (ranking, question)


@pytest.mark.parametrize("clauses,supported", [
    (["In general", "must Deans hold the rank of full professor"], False),
    (["Typically", "must Deans hold the rank of full professor"], False),
    (["Hi", "is the faculty probation period one year"], True),
])
def test_a_lead_split_off_by_the_coordinator_rejoins_its_yes_no_clause(specialists, clauses, supported):
    # The coordinator splits "In general, must Deans ...?" at its comma; the hedge must stay the claim's operator.
    for ranking in ("raw", "high"):
        r = run(specialists[ranking], ", ".join(clauses) + "?", clauses=clauses)
        assert (r.status is SUPPORTED) is supported, (ranking, clauses, r.missing)


def test_a_hedged_claim_is_partial_until_its_hedge_is_stated(specialists):
    for ranking in ("raw", "high"):
        assert run(specialists[ranking], "Must Deans hold the rank of full professor?").status is SUPPORTED
        r = run(specialists[ranking], "In general, must Deans hold the rank of full professor?")
        assert r.status is not SUPPORTED and any("'in general' asked" in m for m in r.missing), (ranking, r.missing)


@pytest.mark.parametrize("question", GENERAL[:8] + ["Is there information about health insurance?", "What does the handbook say about Hajj leave?"])
def test_a_general_question_is_answered_as_before(specialists, monkeypatch, question):
    # D-099 touches no general question: the answer is the same with every C12 switch off.
    for ranking in ("raw", "high"):
        on = run(specialists[ranking], question)
        for name in C12_SWITCHES:
            monkeypatch.setattr(F, name, False)
        off = run(specialists[ranking], question)
        monkeypatch.undo()
        assert (on.status, [f.evidence_quote for f in on.findings], list(on.missing)) == (off.status, [f.evidence_quote for f in off.findings], list(off.missing))


R11_SYNTHETIC = [
    (["Professors receive a housing allowance of 90,000 Dirhams, and lecturers receive a housing allowance of 60,000 Dirhams."],
     [("Do professors receive a housing allowance of 60,000 Dirhams?", False), ("Do lecturers receive a housing allowance of 60,000 Dirhams?", True)]),
    (["Faculty members must submit the resignation letter to the Dean and HR calculates the end-of-service gratuity."],
     [("Must faculty members calculate the end-of-service gratuity?", False)]),
    (["Faculty members submit the leave request through the HR portal or they email the department chair."],
     [("Do faculty members email the leave request to HR?", False)]),
    (["In special circumstances, the probation period of visiting faculty is six months."], [("Is the probation period of visiting faculty six months?", False)]),
    (["If the visiting appointment is for one semester, the probation period is three months."], [("Is the probation period three months?", False)]),
    (["Faculty members receive the housing allowance every month.", "The housing allowance is 5,000 Dirhams."],
     [("Is the housing allowance under 5,000 Dirhams?", False), ("Is the housing allowance over 5,000 Dirhams?", False)]),
    (["Faculty members receive ten days of training leave and may request additional training leave."],
     [("Do faculty members receive only ten days of training leave?", False), ("Do faculty members receive ten days of training leave?", True)]),
    (["Promotion requires a recommendation from the department committee."],
     [("Does promotion require a recommendation from the college committee?", False), ("Does promotion require a recommendation from the department committee?", True)]),
]


@pytest.mark.parametrize("lines,questions", R11_SYNTHETIC)
def test_the_r11_synthetic_wrong_answers_on_crafted_pages(lines, questions):
    corpus = corpus_from(registry_and_map(), [page(107, *lines)])
    for reranker in (None, ScriptedReranker([], default=9.0)):
        sp = make_specialist(corpus, reranker)
        for question, supported in questions:
            r = run(sp, question)
            assert (r.status is SUPPORTED) is supported, (question, r.status, r.missing)


@pytest.mark.parametrize("question,supported", [
    ("May visiting faculty apply for the bonus award?", False), ("Hi, may visiting faculty apply for the bonus award?", False),
    ("Are visiting faculty members eligible for the bonus award?", False), ("Are visiting faculty members not eligible for the bonus award?", True),
    ("May faculty members apply for the bonus award in consecutive years?", True),
])
def test_a_population_excluded_in_the_next_sentence_is_not_covered_by_the_general_rule(question, supported):
    corpus = corpus_from(registry_and_map(), [page(107, "Faculty members may apply for the bonus award in consecutive years.",
                                                   "Parttime and visiting faculty members are not eligible for the bonus award.")])
    for reranker in (None, ScriptedReranker([], default=9.0)):
        r = run(make_specialist(corpus, reranker), question)
        assert (r.status is SUPPORTED) is supported, (question, r.status, r.missing)


ADVERSARIAL = [   # wrappers and semantic prefixes combined with operators, on one crafted page
    ("Hi, must faculty members on probation attend the induction workshop?", True),
    ("Can you tell me whether faculty members on probation must attend the induction workshop?", True),
    ("Hi, must all faculty members on probation always attend the induction workshop?", False),
    ("In general, must faculty members on probation attend the induction workshop?", False),
    ("Typically, must faculty members on probation attend the induction workshop?", False),
    ("Can you tell me whether only faculty members on probation must attend the induction workshop?", False),
    ("Do you know whether every faculty member on probation must attend the induction workshop?", False),
    ("Is it true that faculty members on probation must attend the induction workshop unless the Dean exempts them?", False),
    ("Hi, must faculty members on probation attend the induction workshop if they joined in the spring?", False),
    ("I want to know if faculty members on probation must attend the additional induction workshop.", False),
    ("Please confirm: must faculty members on probation attend over three induction workshops?", False),
    ("Quick question - must faculty members on probation attend the induction workshop under two times?", False),
]


@pytest.mark.parametrize("question,supported", ADVERSARIAL)
def test_wrapped_and_prefixed_operators_never_add_support(question, supported):
    corpus = corpus_from(registry_and_map(), [page(107, "Faculty members on probation must attend the induction workshop.")])
    for reranker in (None, ScriptedReranker([], default=9.0)):
        r = run(make_specialist(corpus, reranker), question)
        assert (r.status is SUPPORTED) is supported, (question, r.status, r.missing)


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("switch,question,unit", [
    ("_C12_CLAUSE", "Do professors receive a housing allowance of 60,000 Dirhams?",
     "Professors receive a housing allowance of 90,000 Dirhams, and lecturers receive a housing allowance of 60,000 Dirhams."),
    ("_C12_OBJECT_ANCHOR", "Is a new faculty member's employment considered confirmed?",
     "New faculty members must receive written notification of employment confirmation or termination six months before the end of the contract. "
     "If neither occurs, the new faculty member's employment is considered confirmed."),
    ("_C12_OBJECT_BINDING", "Do faculty members email the leave request to HR?",
     "Faculty members submit the leave request through the HR portal or they email the department chair."),
    ("_C12_COMPARE", "Is the housing allowance under 5,000 Dirhams?", "The housing allowance is 5,000 Dirhams."),
    ("_C12_CONDITION", "Is the probation period three months?", "If the visiting appointment is for one semester, the probation period is three months."),
    ("_C12_COPULAR", "Is the probation period of visiting faculty six months?", "In special circumstances, the probation period of visiting faculty is six months."),
    ("_C12_NEGATIVE_CONDITION", "Is the furniture allowance not paid in the first month?",
     "The furniture allowance is not paid in the first month unless the faculty member asks for it."),
    ("_C12_MODIFIER", "Does promotion require a recommendation from the college committee?", "Promotion requires a recommendation from the department committee."),
    ("_C12_FALLBACK_POLARITY", "May visiting faculty apply for the bonus award?", "Parttime and visiting faculty members are not eligible for the bonus award."),
    ("_C12_LEDGER", "Does health insurance cover all children of faculty members?",
     "Comprehensive health insurance is offered for faculty members, their spouse, and up to three dependent children under 18 years old."),
    ("_C12_PAREN_NUMERAL", "Is the probation period two (2) years?",                                        # D-100 (Step 4.14, P2)
     "The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval."),
    ("_C12_ENTITY_SUBJECT", "Is the housing allowance paid once?",                                          # D-100 (Step 4.14, P1)
     "Faculty members receive the housing allowance monthly and the furniture allowance is paid once."),
    ("_C12_OWN_CONTENT", "Are faculty members not entitled to the housing allowance?",                            # D-101 (Step 4.14-N1)
     "Faculty members are not entitled to the housing allowance if they receive university housing."),
    ("_C12_WHEN_CONDITION", "Are faculty members not entitled to the housing allowance?",                         # D-102 (Step 4.14-W1)
     "Faculty members are not entitled to the housing allowance when they receive university housing."),
])
def test_each_c12_switch_turns_its_own_rule_off(monkeypatch, switch, question, unit):
    assert local(question, unit) is False
    monkeypatch.setattr(F, switch, False)
    assert local(question, unit) is True


def test_the_alternative_filler_switch_turns_its_veto_off(monkeypatch):
    quote = "Faculty members receive ten days of training leave and may request additional training leave."
    assert window("Do faculty members receive only ten days of training leave?", quote) is CONTRADICTED
    monkeypatch.setattr(F, "_C12_ALTERNATIVE_FILLER", False)
    assert window("Do faculty members receive only ten days of training leave?", quote) is COMPATIBLE


def test_the_gate_and_wrapper_switches(specialists, monkeypatch):
    for ranking in ("raw", "high"):
        assert run(specialists[ranking], "Currently, must faculty members apply for the bonus award?").status is not SUPPORTED
        assert run(specialists[ranking], "Hi, is the faculty probation period one year?").status is SUPPORTED
    monkeypatch.setattr(F, "_C12_WRAPPERS", False)        # an unnormalised wrapper fails closed, never open
    assert gate("Hi, is the faculty probation period one year?")[2].label == GATE
    monkeypatch.setattr(F, "_C12_WRAPPERS", True)
    monkeypatch.setattr(F, "_C12_GATE", False)            # with no gate a specific claim falls back to the C1-C10 details
    assert run(specialists["raw"], "Currently, must faculty members apply for the bonus award?").status is SUPPORTED
