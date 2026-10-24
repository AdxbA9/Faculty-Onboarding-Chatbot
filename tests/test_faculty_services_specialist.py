"""
Faculty Services & HR specialist: clause ownership, handoffs, the navigation split, procedure and source-gap behaviour,
status rules and the findings contract (Milestone 4, Step 4.11; D-085).
"""
from __future__ import annotations

import pytest

from handbook_bot.agents.specialists import faculty_services as F
from handbook_bot.agents.specialists.contracts import FindingStatus, HandoffRequest, SpecialistId

from faculty_services_support import P61, P63, P107, PAGES, ScriptedReranker, assert_traceable, corpus_from, make_specialist, registry_and_map, run, task


@pytest.fixture(scope="module")
def rm():
    return registry_and_map()


@pytest.fixture(scope="module")
def corpus(rm):
    return corpus_from(rm, PAGES)


@pytest.fixture(scope="module")
def specialist(corpus):
    return make_specialist(corpus)


@pytest.fixture(scope="module")
def high(corpus):
    return make_specialist(corpus, ScriptedReranker([], default=9.0))


def _clause(result, index=0):
    return result.metadata["clauses"][index]


# ---------------------------------------------------------------------------
# 1. Clause ownership: the shared HR lexicon, never a collision word alone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,owned,other", [
    ("What is the probation period for new faculty?", True, None),
    ("How many days of sick leave am I entitled to?", True, None),
    ("Is the furniture allowance AED 30,000?", True, None),
    ("Does the Chancellor approve termination for cause?", True, None),
    ("Can I register my child at the University nursery?", True, None),
    ("What is my teaching load?", False, SpecialistId.TEACHING),
    ("How many theses may I supervise?", False, SpecialistId.RESEARCH),
    ("What classes do I teach this semester?", False, SpecialistId.TEACHING),
    ("Is student attendance mandatory?", False, SpecialistId.TEACHING),
    ("Can I travel abroad during the semester?", False, None),
    ("Who approves the workload plan?", False, SpecialistId.TEACHING),
    ("Where is the HR office?", True, SpecialistId.INSTITUTIONAL),
])
def test_clause_ownership_and_other_domains(clause, owned, other):
    mine, strongest = F.classify_clause(clause)
    assert mine is owned
    assert strongest is other


def test_collision_words_alone_never_make_a_clause_hr():
    for clause in ("What classes do I teach?", "Is attendance mandatory?", "Can I travel?", "What is the evaluation?", "What is my workload?",
                   "Who is the dean?", "When does the semester start?", "How many credit hours?", "Leave the campus early."):
        assert F.classify_clause(clause)[0] is False, clause


# ---------------------------------------------------------------------------
# 2. Handoffs and the navigation split
# ---------------------------------------------------------------------------
def test_teaching_and_research_clauses_are_handed_off_not_answered(specialist):
    r = run(specialist, "What is my teaching load and can I apply for a research advance?",
            clauses=["What is my teaching load", "can I apply for a research advance"])
    assert r.status is FindingStatus.OUT_OF_SCOPE and not r.findings
    targets = [h.requested_specialist for h in r.handoff_requests]
    assert targets == [SpecialistId.TEACHING, SpecialistId.RESEARCH]
    assert all(isinstance(h, HandoffRequest) and h.task.requested_by == "faculty_services" for h in r.handoff_requests)
    assert [c["outcome"] for c in r.metadata["clauses"]] == ["handed_off", "handed_off"]


def test_pure_navigation_clause_is_institutional_alone(specialist):
    for q in ("Where is the HR office?", "What is the HR department's extension?", "Which building is the Passport Unit in?"):
        r = run(specialist, q)
        assert r.status is FindingStatus.OUT_OF_SCOPE and not r.findings, q
        assert [h.requested_specialist for h in r.handoff_requests] == [SpecialistId.INSTITUTIONAL]


def test_mixed_policy_and_navigation_clause_keeps_the_policy_half(specialist):
    r = run(specialist, "What is the probation period and where is the HR office?")
    assert r.status is FindingStatus.PARTIAL and r.findings
    assert all("probation" in f.evidence_quote.lower() for f in r.findings)
    assert any("Institutional" in m for m in r.missing)
    assert SpecialistId.INSTITUTIONAL in [h.requested_specialist for h in r.handoff_requests]


def test_two_clauses_split_by_the_coordinator_are_answered_separately(specialist):
    r = run(specialist, "What is the probation period and where is the HR office?",
            clauses=["What is the probation period", "where is the HR office"])
    outcomes = [c["outcome"] for c in r.metadata["clauses"]]
    assert outcomes == ["supported", "handed_off"]


def test_unowned_clause_is_reported_not_searched(specialist):
    r = run(specialist, "Can I travel abroad during the semester?")
    assert r.status is FindingStatus.OUT_OF_SCOPE and _clause(r)["outcome"] == "unowned"
    assert any("no faculty services or HR cue" in m for m in r.missing)


# ---------------------------------------------------------------------------
# 3. Status rules
# ---------------------------------------------------------------------------
def test_supported_when_the_unit_states_the_asked_fact(specialist, corpus):
    r = run(specialist, "Is the probation period one year?")
    assert r.status is FindingStatus.SUPPORTED and r.confidence == 1.0
    assert any("probation period is one year" in f.evidence_quote for f in r.findings)
    assert_traceable(r, corpus)


def test_partial_when_the_topic_is_stated_but_not_the_fact(specialist):
    r = run(specialist, "Is the probation period two years?")
    assert r.status is FindingStatus.PARTIAL and r.findings and r.confidence == 0.5
    assert any("does not settle" in m or "not this fact" in m for m in r.missing)


def test_not_found_when_no_hr_unit_is_relevant(rm):
    sp = make_specialist(corpus_from(rm, [PAGES[0]]))      # the chapter 2 preamble only
    r = run(sp, "How many days of sick leave am I entitled to?")
    assert r.status is FindingStatus.NOT_FOUND and not r.findings and r.confidence == 0.0


def test_a_negative_rule_never_supports_the_affirmative_ask(specialist):
    # D-094 (supersedes the 4.11 reading): the quote answers "no", so it does not support the affirmative ask; the negative
    # rule stays quoted as a finding and the missing message names the negated predicate for the answer layer
    r = run(specialist, "Are new faculty who resign during probation entitled to annual leave?")
    assert r.status is FindingStatus.PARTIAL
    assert any("not entitled to benefits such as annual leave" in f.evidence_quote for f in r.findings)
    assert any("the asked predicate (entitled)" in m and "negates it" in m for m in r.missing)


def test_a_negative_ask_is_never_supported_by_affirmative_evidence(specialist):
    # D-095: the negative claim is the proposition; the affirmative rule stays quoted and the missing message names it
    r = run(specialist, "Is the probation period not one year?")
    assert r.status is FindingStatus.PARTIAL
    assert any("probation period is one year" in f.evidence_quote for f in r.findings)
    assert any("the asked negative value" in m and "states the affirmative" in m for m in r.missing)
    r = run(specialist, "Are new faculty who resign during probation not entitled to annual leave?")
    assert r.status is FindingStatus.SUPPORTED
    assert any("not entitled to benefits such as annual leave" in f.evidence_quote for f in r.findings)


def test_specific_unknown_ask_is_never_supported(specialist):
    r = run(specialist, "Which of the leave policies is the most generous one for me?")
    assert r.status is not FindingStatus.SUPPORTED


# ---------------------------------------------------------------------------
# 4. Procedures: documented stages only
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", ["How do I resign from the University?", "How do I apply for sick leave?", "What is the procedure for getting an entry visa?"])
def test_process_requests_are_at_most_partial(specialist, q):
    r = run(specialist, q)
    assert r.status is not FindingStatus.SUPPORTED
    assert _clause(r)["process"] is True
    if r.findings:
        assert any("documented stages only" in m for m in r.missing)
        assert any("stages only" in lim for lim in r.limitations)


def test_documented_stages_are_quoted_for_a_resignation_process(specialist):
    r = run(specialist, "How do I resign from the University?")
    assert r.status is FindingStatus.PARTIAL
    assert all("resign" in f.evidence_quote.lower() for f in r.findings)


def test_system_procedure_requests_are_never_supported(specialist):
    r = run(specialist, "Which menu do I click in the Faculty Information System to update my employee details?", system="fis")
    assert r.status is not FindingStatus.SUPPORTED
    assert _clause(r)["procedural"] is True


# ---------------------------------------------------------------------------
# 5. Source gaps fail closed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", [
    "How many days of vacation leave do faculty get?",
    "How do I get my employee ID card?",
    "How is my payroll processed?",
    "What is my leave balance and does it carry over?",
    "How do I renew my residence visa?",
    "Where do I get a parking permit?",
    "What does the Employee Handbook advise about renting a flat in the UAE?",
])
def test_source_gap_asks_are_never_supported(specialist, high, q):
    for sp in (specialist, high):
        r = run(sp, q)
        assert r.status is not FindingStatus.SUPPORTED, (q, r.status)


# ---------------------------------------------------------------------------
# 6. Contract and metadata
# ---------------------------------------------------------------------------
def test_findings_are_extractive_and_carry_scope_metadata(specialist, corpus):
    r = run(specialist, "Is sick leave with full pay 15 working days?")
    assert r.status is FindingStatus.SUPPORTED and r.llm_used is False
    for f in r.findings:
        assert f.metadata["extractive"] is True and f.metadata["route"] == "policy"
        assert f.metadata["chunk_id"] is not None and f.source_id == F.APPROVED_SOURCE_ID
    assert r.metadata["owned_clauses"] == 1 and r.metadata["route"] == "policy"
    assert_traceable(r, corpus)


def test_clause_limit_reports_unprocessed_clauses(specialist):
    clauses = ["What is the probation period", "How many sick days do I get", "Is the furniture allowance AED 30,000",
               "What is the salary scale", "Who approves termination for cause"]
    r = run(specialist, " and ".join(clauses), clauses=clauses)
    assert len(r.metadata["clauses"]) == F.MAX_CLAUSES
    assert r.metadata["unprocessed_clauses"] == clauses[F.MAX_CLAUSES:]


def test_malformed_task_is_rejected(specialist):
    with pytest.raises(TypeError):
        specialist.run("not a task")


# ---------------------------------------------------------------------------
# D-089 (Step 4.12-C2): a named source that is only mentioned never settles a question for its contents
# ---------------------------------------------------------------------------
def test_named_missing_source_detection():
    assert F.named_missing_source("What does the Employee Handbook say about settling in the UAE?") == "the Employee Handbook"
    assert F.named_missing_source("What are the Employee Handbook rules for housing?") == "the Employee Handbook"
    assert F.named_missing_source("What does the Benefits Chart say about vacation days?") == "the Faculty and Staff Pay Grade and Benefits Chart"
    assert F.named_missing_source("How many vacation days are provided according to the Benefits Chart?") == "the Faculty and Staff Pay Grade and Benefits Chart"
    assert F.named_missing_source("Does the Faculty Handbook say HR provides an Employee Handbook?") is None
    assert F.named_missing_source("Does the Employee Handbook cover living in the UAE?") is None
    assert F.named_missing_source("What topics does the Employee Handbook cover?") is None
    assert F.named_missing_source("What does the Faculty Handbook say about probation?") is None
    assert F.named_missing_source("Is the probation period one year?") is None


def test_employee_handbook_contents_fail_closed_but_the_mention_is_quotable():
    rm = registry_and_map()
    sp = make_specialist(corpus_from(rm, [P61, P63, P107]), ScriptedReranker([], default=9.0))
    for q in ("What does the Employee Handbook say about settling in the UAE?", "What does the Employee Handbook say about HR policies?",
              "What are the Employee Handbook rules for housing?", "What does the Employee Handbook require for onboarding?",
              "What does the Benefits Chart say about vacation days?"):
        r = run(sp, q)
        assert r.status is not FindingStatus.SUPPORTED, q
        assert any("not its contents" in m or "no evidence" in m for m in r.missing), (q, r.missing)
    # the mention itself, and the indexed handbook's own contents, stay supported
    assert run(sp, "Does HR provide an Employee Handbook to new faculty members?").status is FindingStatus.SUPPORTED
    assert run(sp, "Does the Employee Handbook cover living in the UAE?").status is FindingStatus.SUPPORTED
    assert run(sp, "Is the probation period one year?").status is FindingStatus.SUPPORTED
    r = run(sp, "What does the Employee Handbook say about settling in the UAE?")
    assert any("Employee Handbook is not in the indexed sources" in l for l in r.limitations)


# ---------------------------------------------------------------------------
# D-090 (Step 4.12-C3): strong named-source frames and the full Benefits Chart name
# ---------------------------------------------------------------------------
EH, CHART = "the Employee Handbook", "the Faculty and Staff Pay Grade and Benefits Chart"


def test_the_employee_handbook_say_that_frame_is_a_contents_ask():
    assert F.named_missing_source("Does the Employee Handbook say that the UoS overview includes the salary scale?") == EH
    assert F.named_missing_source("The Employee Handbook says settling in the UAE is handled by HR?") == EH


def test_the_employee_handbook_state_that_frame_is_a_contents_ask():
    assert F.named_missing_source("Does the Employee Handbook state that the UoS overview includes the salary scale?") == EH
    assert F.named_missing_source("The Employee Handbook states that housing is free?") == EH


def test_require_and_provide_that_frames_are_contents_asks():
    assert F.named_missing_source("Does the Employee Handbook require that new faculty settle in the UAE?") == EH
    assert F.named_missing_source("Does the Employee Handbook provide that HR policies apply to new faculty?") == EH


def test_prepositional_frames_are_contents_asks_even_with_a_coverage_word():
    assert F.named_missing_source("Per the Employee Handbook, are HR policies covered?") == EH
    assert F.named_missing_source("According to the Employee Handbook, is housing free?") == EH
    assert F.named_missing_source("In the Employee Handbook, is settling in the UAE explained?") == EH


def test_the_full_benefits_chart_name_in_prepositional_frames():
    assert F.named_missing_source("Under the Faculty and Staff Pay Grade and Benefits Chart, is summer vacation a vacation day?") == CHART
    assert F.named_missing_source("According to the Faculty and Staff Pay Grade and Benefits Chart, are religious holidays vacation days?") == CHART
    assert F.named_missing_source("Per the Faculty and Staff Pay Grade and Benefits Chart, how long is the summer vacation?") == CHART


def test_mentions_and_scope_statements_stay_mention_asks():
    for q in ("Does HR provide an Employee Handbook to new faculty members?", "Does the Employee Handbook cover living in the UAE?",
              "Does the Employee Handbook cover HR policies?", "Are vacation days provided as per the Faculty and Staff Pay Grade and Benefits Chart?",
              "What topics does the Employee Handbook cover?", "Does the Faculty Handbook state that the Employee Handbook covers living in the UAE?"):
        assert F.named_missing_source(q) is None, q


def test_strong_frames_fail_closed_and_mentions_stay_answerable():
    rm = registry_and_map()
    from faculty_services_support import P106
    sp = make_specialist(corpus_from(rm, [P61, P106, P107]), ScriptedReranker([], default=9.0))
    for q in ("Does the Employee Handbook say that the UoS overview includes the salary scale?", "Does the Employee Handbook say that the salary scale is based on faculty rank?",
              "Does the Employee Handbook state that the UoS overview includes the salary scale?", "Per the Employee Handbook, are HR policies covered?",
              "Under the Faculty and Staff Pay Grade and Benefits Chart, is summer vacation a vacation day?"):
        assert run(sp, q).status is not FindingStatus.SUPPORTED, q
    assert run(sp, "Does the Employee Handbook cover HR policies?").status is FindingStatus.SUPPORTED
    assert run(sp, "Are vacation days provided as per the Faculty and Staff Pay Grade and Benefits Chart?").status is FindingStatus.SUPPORTED


# ---------------------------------------------------------------------------
# D-091 (Step 4.12-C4): the asker's own leave record is not in the handbook
# ---------------------------------------------------------------------------
def test_personal_usage_asks_are_detected():
    for q in ("How many leave days have I used?", "How many leave days have I taken?", "How much sick leave have I already used?"):
        assert F.personal_record(q) == "usage", q


def test_personal_remaining_asks_are_detected():
    for q in ("How many leave days do I still have?", "How many leave days do I have left?", "How many days are remaining for me?",
              "Do I still have 10 sick-leave days left?", "How many vacation days are available to me now?", "How many Hajj leave days do I personally have left?"):
        assert F.personal_record(q) == "remaining", q


def test_personal_balance_asks_are_detected_and_policy_asks_are_not():
    for q in ("What is my leave balance?", "What is my annual leave balance?", "How many leave days remain in my account?"):
        assert F.personal_record(q) is not None, q
    for q in ("Must I give a six-month notice when I resign during probation?", "How many days of Hajj leave may faculty receive?",
              "What is the sick leave entitlement?", "If I have used all my sick leave days, can I use annual vacation leave?", "Can I take Hajj leave?",
              "What can faculty view on the My Account page of the Faculty Information System?"):
        assert F.personal_record(q) is None, q


def test_personal_record_asks_fail_closed_and_policy_stays_answerable():
    rm = registry_and_map()
    sp = make_specialist(corpus_from(rm, [P63, P107]), ScriptedReranker([], default=9.0))
    for q in ("How many leave days have I used?", "How many leave days do I still have?", "What is my leave balance?", "How much sick leave have I already used?"):
        r = run(sp, q)
        assert r.status is not FindingStatus.SUPPORTED, q
    r = run(sp, "How many leave days have I used?")
    assert any("personal leave usage or balance is not available" in m for m in r.missing)
    assert run(sp, "Is maternity leave sixty calendar days with full pay?").status is FindingStatus.SUPPORTED
    assert run(sp, "Must I give a six-month notice when I resign during probation?").status is FindingStatus.SUPPORTED


# ---------------------------------------------------------------------------
# D-092 (Step 4.12-C5): a decision on the asker's own case is not in the handbook
# ---------------------------------------------------------------------------
def test_personal_grant_status_asks_are_detected():
    for q in ("How many days of special leave have I been granted?", "How many days of maternity leave was I granted?",
              "Were thirty days of special leave granted to me?", "Have I been granted Hajj leave?", "Was I given unpaid special leave?",
              "Was my unpaid special leave granted?", "Did HR grant me unpaid special leave?"):
        assert F.personal_record(q) == "status", q


def test_personal_approval_status_asks_are_detected():
    for q in ("How many leave days were approved for me?", "Has my maternity leave been approved?", "Has my promotion been approved?",
              "Was my promotion approved?", "Did the Chancellor approve my promotion?", "Has my contract renewal been approved?",
              "Was my visa renewal approved?", "Did HR approve my salary adjustment?", "Has my termination request been approved?",
              "Was my administrative appointment approved?", "What is my leave approval status?", "When was my leave approved?",
              "Has my probation been extended?", "Was I promoted?", "Has my application been reviewed?", "Is my leave request still pending?",
              "What is the outcome of my promotion application?", "Did the committee extend my probation?", "Have I been confirmed?"):
        assert F.personal_record(q) == "status", q


def test_personal_acceptance_status_asks_are_detected():
    for q in ("Has my housing request been accepted?", "Has my leave request been accepted?", "Was my resignation accepted?",
              "Did HR accept my leave request?", "Is my resignation accepted?"):
        assert F.personal_record(q) == "status", q


def test_any_past_decision_on_the_askers_own_case_is_detected():
    for q in ("Did the committee recommend my promotion?", "Has my dossier been forwarded to the dean?", "Was I nominated for the bonus award?",
              "Has my evaluation been submitted?", "Was my case investigated?", "Did the dean endorse my application?", "Have I been nominated for the award?",
              "Has HR received my documents?", "Did I receive the furniture allowance?", "Have I received my end-of-service compensation?",
              "Did my salary increase this year?", "Am I approved for unpaid leave?", "Has my resignation gone through?", "Was my leave cut short?",
              "What was decided on my appeal?", "Who approved my leave?", "Has the dean signed off my leave?"):
        assert F.personal_record(q) == "status", q
    for q in ("Was I supposed to give six months notice?", "Was I required to submit an annual report?", "Who recommends my promotion?",
              "Does the committee recommend promotions?", "Who approves my leave?", "Am I entitled to sixty calendar days of maternity leave with full pay?",
              "Are my dependants covered by health insurance?", "Will my contract be renewed if my evaluations are satisfactory?"):
        assert F.personal_record(q) is None, q


def test_policy_grant_and_approval_asks_are_not_personal():
    for q in ("May unpaid special leave be granted for up to thirty days?", "Who approves termination for cause?", "Can maternity leave be granted?",
              "How many days of unpaid special leave may be granted?", "Who approves additional maternity leave without pay?",
              "Is additional maternity leave approved by the Vice Chancellor for Academic Affairs?", "Can unpaid special leave be granted to me?",
              "Who will approve my leave?", "Does the dean have to approve my leave?", "If my leave is approved, when does it start?",
              "If the dean has already approved my leave, who notifies HR?", "What can faculty view on the My Account page of the Faculty Information System?",
              "Can the probation period be extended?", "Who reviews my promotion application?", "Will unpaid leave be deducted from my salary?"):
        assert F.personal_record(q) is None, q


def test_personal_status_asks_fail_closed_and_policy_stays_answerable():
    rm = registry_and_map()
    from faculty_services_support import P64, P152
    for reranker in (None, ScriptedReranker([], default=9.0)):
        sp = make_specialist(corpus_from(rm, [P63, P64, P107, P152]), reranker)
        for q in ("How many days of special leave have I been granted?", "How many leave days were approved for me?", "Has my maternity leave been approved?",
                  "How many days of maternity leave was I granted?", "Has my promotion been approved?", "Was unpaid special leave approved for me?",
                  "Was my unpaid special leave granted?", "Did the Chancellor approve my promotion?", "Has my leave request been accepted?",
                  "Did the committee recommend my promotion?"):
            r = run(sp, q)
            assert r.status is not FindingStatus.SUPPORTED, q
            assert not r.findings or any("grant, approval, acceptance or decision status is not available" in m for m in r.missing), q
        r = run(sp, "Has my maternity leave been approved?")
        assert any("decision on the asker's own case" in x for x in r.limitations)
    sp = make_specialist(corpus_from(rm, [P63, P64, P107, P152]), ScriptedReranker([], default=9.0))
    assert run(sp, "May unpaid special leave be granted for up to thirty days?").status is FindingStatus.SUPPORTED
    assert run(sp, "Does the Chancellor approve termination for cause?").status is FindingStatus.SUPPORTED
    assert run(sp, "Is maternity leave sixty calendar days with full pay?").status is FindingStatus.SUPPORTED
