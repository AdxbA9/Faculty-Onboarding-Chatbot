"""
Faculty Services & HR through the Coordinator and the dispatcher: activation, mixed-domain questions, handoffs as data,
no LLM call, Institutional pending (Milestone 4, Step 4.11; D-085).
"""
from __future__ import annotations

import pytest

from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists import faculty_services as F
from handbook_bot.agents.specialists.contracts import FindingStatus, HandoffRequest, SpecialistId
from handbook_bot.agents.specialists.registry import SpecialistRegistry

from faculty_services_support import PAGES, corpus_from, make_specialist, registry_and_map

import test_teaching_specialist as TS


@pytest.fixture(scope="module")
def rm():
    return registry_and_map()


@pytest.fixture(scope="module")
def registry(rm):
    reg = SpecialistRegistry()
    reg.register(make_specialist(corpus_from(rm, PAGES)))
    reg.register(TS.make_specialist(TS.corpus_from(rm, TS.PAGES)))
    return reg


def test_faculty_services_is_executable_institutional_pending():
    assert D.EXECUTABLE_SPECIALISTS == (SpecialistId.TEACHING, SpecialistId.RESEARCH, SpecialistId.FACULTY_SERVICES)
    assert SpecialistId.INSTITUTIONAL not in D.EXECUTABLE_SPECIALISTS


def test_hr_only_question_runs_the_hr_specialist(registry):
    out = D.run_question("What is the probation period for new faculty?", registry)
    assert out.executed_specialists == [SpecialistId.FACULTY_SERVICES] and out.pending_specialists == []
    run = out.runs[0]
    assert run.findings.status is FindingStatus.SUPPORTED and run.findings.llm_used is False and out.llm_calls == 0


def test_hr_and_teaching_question_runs_both_and_hands_nothing_across(registry):
    out = D.run_question("How many days of sick leave do I get and how many office hours must I hold?", registry)
    assert set(out.executed_specialists) == {SpecialistId.FACULTY_SERVICES, SpecialistId.TEACHING}
    for run in out.runs:
        assert run.findings is not None and run.findings.llm_used is False
        for c in run.findings.metadata["clauses"]:
            if run.specialist_id is SpecialistId.FACULTY_SERVICES:
                assert "office hours" not in c["clause"].lower() or c["outcome"] == "handed_off"
            else:
                assert "sick leave" not in c["clause"].lower() or c["outcome"] == "handed_off"
    assert all(isinstance(h, HandoffRequest) for h in out.handoffs) and out.llm_calls == 0


def test_hr_and_institutional_question_keeps_institutional_pending(registry):
    out = D.run_question("What is the probation period and where is the HR office?", registry)
    assert SpecialistId.FACULTY_SERVICES in out.executed_specialists
    assert out.pending_specialists == [SpecialistId.INSTITUTIONAL]
    hr = next(r for r in out.runs if r.specialist_id is SpecialistId.FACULTY_SERVICES)
    assert hr.findings.status in (FindingStatus.SUPPORTED, FindingStatus.PARTIAL)
    assert hr.findings.findings and all("probation" in f.evidence_quote.lower() for f in hr.findings.findings)


def test_hr_research_and_institutional_question(registry):
    out = D.run_question("Is the furniture allowance AED 30,000, who approves research grants, and where is the research office?", registry)
    assert SpecialistId.FACULTY_SERVICES in out.executed_specialists
    assert SpecialistId.INSTITUTIONAL in out.pending_specialists or SpecialistId.RESEARCH in out.pending_specialists
    hr = next(r for r in out.runs if r.specialist_id is SpecialistId.FACULTY_SERVICES)
    assert any("30,000 Dirhams" in f.evidence_quote for f in hr.findings.findings)
    assert out.llm_calls == 0


def test_an_unregistered_hr_specialist_stays_pending(rm):
    reg = SpecialistRegistry()
    reg.register(TS.make_specialist(TS.corpus_from(rm, TS.PAGES)))
    out = D.run_question("How many days of sick leave do I get?", reg)
    assert out.executed_specialists == [] and out.pending_specialists == [SpecialistId.FACULTY_SERVICES]


def test_handoff_tasks_are_never_executed(registry):
    out = D.run_question("What is my teaching load and how many days of sick leave do I get?", registry)
    for run in out.runs:
        assert run.task.requested_by == "coordinator"
    for h in out.handoffs:
        assert h.task.requested_by in ("faculty_services", "teaching")


def test_hr_results_are_valid_contracts(registry):
    out = D.run_question("Is the clinical allowance 10 to 20 percent of the basic salary?", registry)
    hr = next(r for r in out.runs if r.specialist_id is SpecialistId.FACULTY_SERVICES)
    payload = hr.findings.to_dict()
    assert payload["specialist_id"] == "faculty_services" and payload["llm_used"] is False
    assert payload["status"] == "supported"
    assert F.APPROVED_SOURCE_ID == payload["findings"][0]["source_id"]
