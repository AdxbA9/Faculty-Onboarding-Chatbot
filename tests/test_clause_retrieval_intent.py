"""Per-clause retrieval intent (Milestone 4, Step 4.7; D-078): the Step 4.6 MAJOR and its controls.

A mixed question ("Who approves the petty cash advance and what is the VCRGS phone number?") gives the Research task
the Coordinator's joined intent "contact". The specialist used to retrieve every focus clause under that task-level
route, whose ranking favours directory rows and phone-bearing lines; the paragraph that answers the substantive clause
fell out of the candidate window and an answered question came back NOT_FOUND. The fixture below rebuilds that window
in miniature: a few handbook policy lines beside a thirty-line directory of phone numbers inside the Research scope.
Under the old task-level route the paragraph is crowded out; under the per-clause route it is not. Teaching shares the
helper and gets the same test shape. No model, no LLM, no network; the dispatch seam is not wired into the runtime and
handoffs are data, never executed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists import research as R
from handbook_bot.agents.specialists import teaching as T
from handbook_bot.agents.specialists.contracts import FindingStatus, HandoffRequest, SpecialistId
from handbook_bot.agents.specialists.research import APPROVED_SOURCE_ID
from handbook_bot.agents.specialists.retrieval_intent import resolve_clause_retrieval_intent
from handbook_bot.sources import load_section_map, load_source_registry

import test_research_integration as RI
import test_research_specialist as RT
import test_teaching_integration as TI
import test_teaching_specialist as TS

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fixture: policy lines beside a directory that the contact route ranks first
# ---------------------------------------------------------------------------
UNITS = ["Research Funding Department", "Technology Transfer Office", "Research Groups Department", "Institutional Review Board",
         "Research Support Office", "Innovation Centre", "Grants Office", "Research Ethics Office", "Sponsored Projects Office",
         "Research Institute Secretariat"]


def directory(page_no, n=30):
    """``n`` printed directory lines (room, telephone, fax, extension) on one page: row chunks with phone numbers,
    the shape the contact route boosts and the specialists never quote as evidence for a policy clause."""
    rows = ["%s, Room M%d-%03d, Tel: +971 6 505 %04d, Fax: +971 6 558 %04d, ext. %d"
            % (UNITS[i % len(UNITS)], 11 + i % 3, 100 + i, 1000 + i, 2000 + i, 3000 + i) for i in range(n)]
    return RT.page(page_no, *rows)


def research_page(number):
    return next(p for p in RT.RESEARCH_PAGES if p["page"] == number)


CROWD_PAGES = [
    RT.page(166,
            "20. The principal investigator is entitled to a petty cash advance of up to AED 5000, subject to approval by the VCRGS.",
            "21. The principal investigator may apply for a research advance of up to 10% of the basic research project grant.",
            "22. The advance may be raised to a maximum of 30% of the basic research project grant in justified cases."),
    research_page(170),      # research group awards (the Step 4.6 campaign case)
    research_page(176),      # intellectual property ownership (the TTO mixed case)
    directory(174),
]

PETTY = "Who approves the petty cash advance"
PETTY_MAX = "What is the maximum petty cash advance"
PHONE = "what is the VCRGS phone number"
IP = "Who owns intellectual property created by faculty"
TTO_PHONE = "what is the TTO phone number"
OFFICE_HOURS = "What is the minimum number of office hours per week?"
SYLLABUS = "Who hands the syllabus to the department office"
OFFICE_COUNT = "How many office hours must full-time faculty schedule per week"
BB_PHONE = "what is the phone number of the Blackboard support office"


@pytest.fixture(scope="module")
def registry_and_map():
    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    return registry.get_source(APPROVED_SOURCE_ID), load_section_map(str(REPO / "knowledge" / "handbook_sections.json"))


@pytest.fixture(scope="module")
def research(registry_and_map):
    return RT.make_specialist(RT.corpus_from(registry_and_map, CROWD_PAGES))


@pytest.fixture(scope="module")
def teaching(registry_and_map):
    return TS.make_specialist(TS.corpus_from(registry_and_map, list(TS.PAGES) + [directory(79)]))


@pytest.fixture(scope="module")
def registry(research):
    return RI.build_registry(TI.teaching_for(RI._teaching_corpus()), research)


def report(result, index=0):
    return result.metadata["clauses"][index]


def reports_by_clause(result):
    return {r["clause"].lower(): r for r in result.metadata["clauses"]}


def quotes(result):
    return sorted(f.evidence_quote for f in result.findings)


def joined(clauses):
    return " and ".join(clauses) + "?"


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("intent", ["policy", "policy_yesno", "date", "count", "list"])
def test_a_sole_clause_keeps_a_non_contact_task_intent(intent):
    assert resolve_clause_retrieval_intent("When do Fall classes begin?", intent, sole_clause=True) == intent


@pytest.mark.parametrize("intent", [None, "", "greeting", "no-such-route"])
def test_a_greeting_or_unknown_intent_falls_back_to_policy(intent):
    assert resolve_clause_retrieval_intent(PETTY, intent, sole_clause=True) == "policy"
    assert resolve_clause_retrieval_intent(PETTY, intent) == "policy"


def test_a_sole_pure_contact_clause_keeps_the_contact_route():
    assert resolve_clause_retrieval_intent("What is the VCRGS phone number?", "contact", sole_clause=True) == "contact"
    assert resolve_clause_retrieval_intent("What is the VCRGS phone number?", "contact", sole_clause=True, substantive=False) == "contact"


def test_a_sole_substantive_clause_never_retrieves_as_contact():
    assert resolve_clause_retrieval_intent(PETTY, "contact", sole_clause=True) == "policy"
    assert resolve_clause_retrieval_intent(PETTY + "?", "contact", sole_clause=True, substantive=True) == "policy"


def test_the_callers_substantive_finding_overrides_a_contact_label():
    # the router labels the whole mixed clause "contact"; a caller that found a substantive half beside the contact
    # half (Research's navigation analysis) retrieves it as policy, since it answers that half and hands the other off
    mixed = "Who approves the petty cash advance and what is the VCRGS phone number?"
    assert resolve_clause_retrieval_intent(mixed, "contact", sole_clause=True) == "contact"
    assert resolve_clause_retrieval_intent(mixed, "contact", sole_clause=True, substantive=True) == "policy"
    assert resolve_clause_retrieval_intent(mixed, "contact", substantive=True) == "policy"


@pytest.mark.parametrize("task_intent", ["contact", "policy", "date", "count"])
def test_clauses_of_a_multi_clause_task_are_classified_on_their_own(task_intent):
    assert resolve_clause_retrieval_intent(PETTY, task_intent) == "policy"
    assert resolve_clause_retrieval_intent(PHONE, task_intent) == "contact"
    assert resolve_clause_retrieval_intent("When must conference requests be submitted", task_intent) == "date"
    assert resolve_clause_retrieval_intent(OFFICE_COUNT, task_intent) == "count"
    assert resolve_clause_retrieval_intent("Can researchers participate in more than one research group", task_intent) == "policy_yesno"


# ---------------------------------------------------------------------------
# A. The exact minimised Step 4.6 failure (permanent regression)
# ---------------------------------------------------------------------------
def test_step_4_6_reproducer_substantive_clause_under_a_contact_task_intent(research):
    contaminated = research.run(RT.task(PETTY, intent="contact"))
    clean = research.run(RT.task(PETTY, intent="policy"))
    assert clean.status is FindingStatus.SUPPORTED and report(clean)["route"] == "policy"
    assert contaminated.status is FindingStatus.SUPPORTED
    assert report(contaminated)["route"] == "policy" and report(contaminated)["relation"] == "strong"
    assert quotes(contaminated) == quotes(clean)
    assert all(f.metadata["route"] == "policy" for f in contaminated.findings)
    assert contaminated.metadata["route"] == "contact"            # the task-level intent is still reported as given


def test_the_fixture_reproduces_the_defect_under_the_old_task_level_route(research, monkeypatch):
    # guards the test above against a fixture that no longer crowds the paragraph out: with the retired rule (every
    # clause retrieves under the task intent) the Step 4.6 failure must still appear here
    monkeypatch.setattr(R, "resolve_clause_retrieval_intent", lambda clause, task_intent, **kw: task_intent)
    result = research.run(RT.task(PETTY, intent="contact"))
    assert result.status is FindingStatus.NOT_FOUND and not result.findings
    assert report(result)["route"] == "contact"


# ---------------------------------------------------------------------------
# B, C, F, G. Mixed tasks in the specialist: order, TTO, two substantive clauses
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clauses", [[PETTY, PHONE], [PHONE, PETTY]], ids=["policy+contact", "contact+policy"])
def test_a_mixed_task_keeps_its_policy_clause_in_either_order(research, clauses):
    result = research.run(RT.task(joined(clauses), clauses=clauses, intent="contact"))
    reports = reports_by_clause(result)
    assert len(reports) == 2                                        # no clause disappears
    petty, phone = reports[PETTY.lower()], reports[PHONE.lower()]
    assert petty["outcome"] == "supported" and petty["route"] == "policy" and petty["relation"] == "strong"
    assert phone["outcome"] == "handed_off" and phone["target"] == "institutional"
    assert {h.requested_specialist for h in result.handoff_requests} == {SpecialistId.INSTITUTIONAL}
    assert quotes(result) == quotes(research.run(RT.task(PETTY, intent="policy")))


def test_an_ip_clause_beside_a_tto_contact_clause(research):
    clauses = [IP, TTO_PHONE]
    result = research.run(RT.task(joined(clauses), clauses=clauses, intent="contact"))
    reports = reports_by_clause(result)
    assert reports[IP.lower()]["outcome"] == "supported" and reports[IP.lower()]["route"] == "policy"
    assert reports[TTO_PHONE.lower()]["outcome"] == "handed_off" and reports[TTO_PHONE.lower()]["target"] == "institutional"
    assert quotes(result) == quotes(research.run(RT.task(IP, intent="policy")))


def test_one_contact_clause_does_not_contaminate_two_substantive_clauses(research):
    clauses = [PETTY, PETTY_MAX, PHONE]
    result = research.run(RT.task(joined(clauses), clauses=clauses, intent="contact"))
    reports = reports_by_clause(result)
    assert len(reports) == 3
    for clause in (PETTY, PETTY_MAX):
        assert reports[clause.lower()]["outcome"] == "supported" and reports[clause.lower()]["route"] == "policy"
    assert reports[PHONE.lower()]["outcome"] == "handed_off"


def test_each_research_clause_reports_its_own_route():
    # a date clause beside a contact clause retrieves as a date clause, whatever the task intent says
    clauses = ["When must conference requests be submitted", "what is the VCAA phone number"]
    for task_intent in ("contact", "policy"):
        assert resolve_clause_retrieval_intent(clauses[0], task_intent) == "date"
        assert resolve_clause_retrieval_intent(clauses[1], task_intent) == "contact"


# ---------------------------------------------------------------------------
# B, G. Through the real Coordinator and the dispatcher
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "Who approves the petty cash advance and what is the VCRGS phone number?",
    "What is the VCRGS phone number and who approves the petty cash advance?",
], ids=["policy+contact", "contact+policy"])
def test_a_mixed_question_through_the_coordinator_keeps_the_research_half(registry, question):
    result = D.run_question(question, registry)
    assert result.executed_specialists == [SpecialistId.RESEARCH]
    assert result.pending_specialists == [SpecialistId.INSTITUTIONAL]
    run = next(r for r in result.runs if r.specialist_id is SpecialistId.RESEARCH)
    assert run.task.intent == "contact"                             # the Coordinator's joined intent: the source of the contamination
    reports = reports_by_clause(run.findings)
    assert len(reports) == 2                                        # no clause disappears
    petty, phone = reports[PETTY.lower()], reports[PHONE.lower()]
    assert petty["outcome"] == "supported" and petty["route"] == "policy" and petty["relation"] == "strong"
    assert phone["outcome"] == "handed_off" and phone["target"] == "institutional"
    assert run.findings.status is not FindingStatus.NOT_FOUND
    assert all(isinstance(h, HandoffRequest) for h in run.findings.handoff_requests)
    pending = next(r for r in result.runs if r.specialist_id is SpecialistId.INSTITUTIONAL)
    assert pending.state == D.STATE_PENDING and pending.findings is None   # handed off as data, never executed


def test_the_step_4_6_campaign_question_through_the_coordinator(registry):
    result = D.run_question("Who approves the research group awards and how do I contact the VCRGS office?", registry)
    run = next(r for r in result.runs if r.specialist_id is SpecialistId.RESEARCH)
    assert run.task.intent == "contact"
    awards = reports_by_clause(run.findings)["who approves the research group awards"]
    assert awards["outcome"] == "supported" and awards["route"] == "policy"


# ---------------------------------------------------------------------------
# D, E. Controls: a true contact clause keeps contact retrieval; policy clauses are unchanged
# ---------------------------------------------------------------------------
def test_a_pure_contact_clause_keeps_the_contact_route_and_is_institutionals(research):
    assert resolve_clause_retrieval_intent("What is the VCRGS phone number?", "contact", sole_clause=True) == "contact"
    result = research.run(RT.task("What is the VCRGS phone number?", intent="contact"))
    assert report(result)["outcome"] == "handed_off" and report(result)["target"] == "institutional"
    assert {h.requested_specialist for h in result.handoff_requests} == {SpecialistId.INSTITUTIONAL}


@pytest.mark.parametrize("question,intent", [
    (PETTY + "?", "policy"),
    ("Does the VCRGS approve the petty cash advance?", "policy_yesno"),
    ("How much is the petty cash advance?", "policy"),
    ("Who approves the research group awards?", "policy"),
    (IP + "?", "policy"),
])
def test_sole_policy_clauses_keep_their_route_and_their_evidence(research, question, intent):
    result = research.run(RT.task(question, intent=intent))
    assert result.status is FindingStatus.SUPPORTED
    assert report(result)["route"] == intent and result.metadata["route"] == intent


# ---------------------------------------------------------------------------
# H. Teaching shares the helper
# ---------------------------------------------------------------------------
def test_teaching_substantive_clause_under_a_contact_task_intent(teaching):
    contaminated = teaching.run(TS.task(OFFICE_HOURS, intent="contact"))
    clean = teaching.run(TS.task(OFFICE_HOURS, intent="policy"))
    assert clean.status is FindingStatus.SUPPORTED and report(clean)["route"] == "policy"
    assert contaminated.status is FindingStatus.SUPPORTED and report(contaminated)["route"] == "policy"
    assert quotes(contaminated) == quotes(clean)
    assert contaminated.metadata["route"] == "contact"


def test_the_teaching_fixture_reproduces_the_defect_under_the_old_task_level_route(teaching, monkeypatch):
    monkeypatch.setattr(T, "resolve_clause_retrieval_intent", lambda clause, task_intent, **kw: task_intent)
    result = teaching.run(TS.task(OFFICE_HOURS, intent="contact"))
    assert result.status is FindingStatus.NOT_FOUND and not result.findings


@pytest.mark.parametrize("clauses", [[SYLLABUS, BB_PHONE], [BB_PHONE, SYLLABUS]], ids=["policy+contact", "contact+policy"])
def test_a_mixed_teaching_task_keeps_its_policy_clause_in_either_order(teaching, clauses):
    result = teaching.run(TS.task(joined(clauses), clauses=clauses, intent="contact"))
    reports = reports_by_clause(result)
    assert len(reports) == 2
    assert reports[SYLLABUS.lower()]["outcome"] == "supported" and reports[SYLLABUS.lower()]["route"] == "policy"
    # a pure contact clause is Institutional Navigation's alone (D-079): handed off as data, never searched
    assert reports[BB_PHONE.lower()]["outcome"] == "handed_off" and reports[BB_PHONE.lower()]["target"] == "institutional"
    assert quotes(result) == quotes(teaching.run(TS.task(SYLLABUS, intent="policy")))


def test_teaching_clauses_are_classified_on_their_own(teaching):
    clauses = [OFFICE_COUNT, BB_PHONE]
    result = teaching.run(TS.task(joined(clauses), clauses=clauses, intent="contact"))
    reports = reports_by_clause(result)
    assert reports[OFFICE_COUNT.lower()]["route"] == "count" and reports[OFFICE_COUNT.lower()]["outcome"] == "supported"
    assert reports[BB_PHONE.lower()]["outcome"] == "handed_off"
    assert resolve_clause_retrieval_intent(BB_PHONE, "contact") == "contact"     # the route rule itself is unchanged (D-078)


def test_teaching_true_contact_and_date_behaviour_is_unchanged(teaching):
    contact = teaching.run(TS.task(BB_PHONE + "?", intent="contact"))
    # a pure contact clause is handed off to Institutional Navigation (D-079); the task intent is reported as given
    assert report(contact)["outcome"] == "handed_off" and report(contact)["target"] == "institutional"
    assert contact.metadata["route"] == "contact"
    assert {h.requested_specialist for h in contact.handoff_requests} == {SpecialistId.INSTITUTIONAL}
    dated = teaching.run(TS.task("When do Fall classes begin?", intent="date"))
    assert report(dated)["route"] == "date" and dated.metadata["route"] == "date"
