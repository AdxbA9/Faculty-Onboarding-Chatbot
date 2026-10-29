"""
Four specialists in the default Plan F dispatch (Milestone 4, Step 4.16-B6, D-105).

Since D-105 ``dispatch.EXECUTABLE_SPECIALISTS`` holds Teaching & Learning, Research & Innovation, Faculty Services & HR
and Institutional Navigation. These tests use the real Coordinator and the default dispatcher (no ``executable=``
override) with all four specialists registered on their test fixtures:

* each specialist executes by default for a question routed to it and returns a valid contract result;
* the dispatcher runs exactly the Coordinator's tasks, in order, and nothing else;
* an unregistered specialist and a task a handoff produced stay pending; handoffs are data and are never executed, so
  no handoff round can loop;
* invalid or unsupported tasks fail closed;
* every finding is quoted from its own specialist's corpus (no evidence crosses between specialists);
* the default dispatch (no ``executable=`` argument) meets every recorded B4 matrix expectation (selection,
  Institutional status, handoffs, findings) and gives, question by question, the judged B5 benchmark result stored in
  ``qa/institutional_4_16/coverage_baseline_b5.jsonl`` (a copy of the Step 4.16-B5 run): enablement changes the default
  path, not the behaviour. Since Step 4.16-C1 (natural phrasing) the judged result is the stored C1 run
  (``coverage_baseline_c1.jsonl``); it differs from the B5 one only by newly correct answers, with the same Coordinator
  selections.
No model, no LLM, no network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from handbook_bot.agents.coordinator import REQUESTED_BY, coordinate
from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists.base import check_findings
from handbook_bot.agents.specialists.contracts import FindingStatus, SpecialistId
from handbook_bot.agents.specialists.registry import SpecialistRegistry

import importlib.util

from handbook_bot.agents.specialists import navigation_cues as N

from institutional_support import four_specialist_registry, make_institutional, run, task
from test_institutional_integration import MATRIX

REPO = Path(__file__).resolve().parents[1]
OVERRIDE = (SpecialistId.TEACHING, SpecialistId.RESEARCH, SpecialistId.FACULTY_SERVICES, SpecialistId.INSTITUTIONAL)
BLIND = [json.loads(line) for line in open(REPO / "qa" / "institutional_4_16" / "coverage_cases.jsonl", encoding="utf-8")]
BASELINE = {r["id"]: r for r in (json.loads(line) for line in open(REPO / "qa" / "institutional_4_16" / "coverage_baseline_b5.jsonl",
                                                                   encoding="utf-8"))}
#: Step 4.16-C1: the same benchmark after the natural-phrasing correction (a copy of the C1 run through the default dispatch)
BASELINE_C1 = {r["id"]: r for r in (json.loads(line) for line in open(REPO / "qa" / "institutional_4_16" / "coverage_baseline_c1.jsonl",
                                                                      encoding="utf-8"))}
_spec = importlib.util.spec_from_file_location("coverage_runner", REPO / "qa" / "institutional_4_16" / "coverage_runner.py")
CR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CR)


def _without_ms(value):
    if isinstance(value, dict):
        return {k: _without_ms(v) for k, v in value.items() if k != "ms"}
    if isinstance(value, list):
        return [_without_ms(v) for v in value]
    return value


@pytest.fixture(scope="module")
def registry():
    return four_specialist_registry()


def test_all_four_specialists_are_executable_by_default():
    assert D.EXECUTABLE_SPECIALISTS == OVERRIDE
    assert set(D.EXECUTABLE_SPECIALISTS) == set(SpecialistId) and len(D.EXECUTABLE_SPECIALISTS) == 4


@pytest.mark.parametrize("question,owner", [
    ("How many office hours must I hold?", SpecialistId.TEACHING),
    ("What percentage of licensing revenue goes to the inventor?", SpecialistId.RESEARCH),
    ("What is the probation period for new faculty?", SpecialistId.FACULTY_SERVICES),
    ("What is the phone number of Human Resources?", SpecialistId.INSTITUTIONAL),
])
def test_each_specialist_executes_by_default_and_returns_a_valid_result(registry, question, owner):
    out = D.run_question(question, registry)
    assert out.executed_specialists == [owner] and out.pending_specialists == [] and out.llm_calls == 0
    run = out.runs[0]
    assert run.state == D.STATE_EXECUTED and run.task.requested_by == REQUESTED_BY
    result = check_findings(owner, run.task, run.findings)
    assert result.status is FindingStatus.SUPPORTED and result.findings and result.llm_used is False
    assert all(f.source_id == "uos_faculty_handbook_2025_26" and f.page and f.evidence_quote for f in result.findings)


def test_a_three_specialist_question_runs_each_selected_specialist_once(registry):
    question = ("How many office hours must I hold, what is the probation period, what percentage of licensing revenue goes "
                "to the inventor, and what is the phone number of Human Resources?")
    decision = coordinate(question)
    out = D.dispatch(decision, registry, question=question)
    assert [r.specialist_id for r in out.runs] == [t.specialist_id for t in decision.subtasks]
    assert [r.task for r in out.runs] == list(decision.subtasks)
    assert out.executed_specialists == list(decision.selected_specialists) and out.pending_specialists == []
    assert len(decision.selected_specialists) <= 3                          # PLAN_F_MAX_SPECIALISTS, unchanged


def test_only_registered_specialists_execute_and_none_is_substituted(registry):
    three = SpecialistRegistry()
    for sid in (SpecialistId.TEACHING, SpecialistId.RESEARCH, SpecialistId.FACULTY_SERVICES):
        three.register(registry.resolve(sid))
    out = D.run_question("What is the phone number of Human Resources?", three)
    assert out.executed_specialists == [] and out.pending_specialists == [SpecialistId.INSTITUTIONAL]
    assert "executable but not registered" in out.runs[0].reason and out.runs[0].findings is None


def test_a_handoff_task_placed_in_a_decision_is_never_executed(registry):
    """No second round: the dispatcher keeps a task a specialist's handoff produced pending, so handoffs cannot loop."""
    question = "Which office handles faculty housing?"
    out = D.run_question(question, registry)
    handoffs = out.handoffs
    assert {(h.from_specialist, h.requested_specialist) for h in handoffs} == {(SpecialistId.INSTITUTIONAL, SpecialistId.FACULTY_SERVICES),
                                                                             (SpecialistId.FACULTY_SERVICES, SpecialistId.INSTITUTIONAL)}
    decision = coordinate(question)
    decision.subtasks[0].requested_by = handoffs[0].from_specialist.value     # as if a handoff task were placed in a decision
    again = D.dispatch(decision, registry, question=question)
    assert again.runs[0].state == D.STATE_PENDING and "not issued by the Faculty Onboarding Coordinator" in again.runs[0].reason
    assert again.runs[0].findings is None


@pytest.mark.parametrize("qid", [row[0] for row in MATRIX])
def test_runs_are_exactly_the_coordinator_tasks_and_handoffs_stay_data(registry, qid):
    question = next(row[1] for row in MATRIX if row[0] == qid)
    out = D.run_question(question, registry)
    assert [r.task.task_id for r in out.runs] == [t.task_id for t in out.decision.subtasks]
    assert all(r.state == D.STATE_EXECUTED and r.task.requested_by == REQUESTED_BY for r in out.runs)
    executed = {r.task.task_id for r in out.runs}
    assert all(h.task.task_id not in executed and h.task.requested_by == h.from_specialist.value for h in out.handoffs)


@pytest.mark.parametrize("qid", [row[0] for row in MATRIX])
def test_every_finding_is_quoted_from_its_own_specialists_corpus(registry, qid):
    question = next(row[1] for row in MATRIX if row[0] == qid)
    out = D.run_question(question, registry)
    for run in out.runs:
        res = registry.resolve(run.specialist_id).resources
        texts = {m.get("chunk_id"): t for t, m in zip(res.chunks, res.metadata)}
        for f in run.findings.findings:
            assert f.evidence_quote in texts.get(f.metadata.get("chunk_id"), ""), (run.specialist_id, f.finding_id)
            assert f.source_id == "uos_faculty_handbook_2025_26"


def test_invalid_or_unsupported_tasks_fail_closed(registry):
    nothing = D.run_question("asdkjh qwe zzz", registry)
    assert nothing.runs == [] and nothing.executed_specialists == []
    decision = coordinate("What is the phone number of Human Resources?")
    decision.subtasks[0].source_scope = ["unapproved_source"]
    out = D.dispatch(decision, registry, question="What is the phone number of Human Resources?")
    result = out.runs[0].findings
    assert out.runs[0].state == D.STATE_EXECUTED and result.status is FindingStatus.NOT_FOUND and result.findings == []
    # a login or password ask that the Coordinator gives to Institutional Navigation gets no finding and no invented step
    routed = D.run_question("Who do I contact to reset my university email password?", registry)
    inst = next(r.findings for r in routed.runs if r.specialist_id is SpecialistId.INSTITUTIONAL)
    assert inst.status is FindingStatus.NOT_FOUND and inst.findings == []
    for question in ("How do I reset my Blackboard password?", "How do I log in to MyUOS?"):   # given to it directly
        direct = make_institutional().run(task(question))
        assert direct.status is FindingStatus.NOT_FOUND and direct.findings == [] and direct.handoff_requests == []


@pytest.mark.parametrize("qid", [row[0] for row in MATRIX])
def test_the_default_dispatch_meets_the_recorded_b4_matrix_expectations(registry, qid):
    _id, question, selected, status, handoffs, findings = next(row for row in MATRIX if row[0] == qid)
    out = D.run_question(question, registry)                                  # default dispatch: no executable= argument
    assert [s.value for s in out.decision.selected_specialists] == selected and out.pending_specialists == []
    inst = next((r.findings for r in out.runs if r.specialist_id is SpecialistId.INSTITUTIONAL), None)
    if status is None:
        assert inst is None
        return
    assert inst.status.value == status
    assert [h.requested_specialist.value for h in inst.handoff_requests] == handoffs
    assert [(f.page, f.metadata.get("office_id")) for f in inst.findings] == findings


@pytest.mark.parametrize("cid", [c["id"] for c in BLIND])
def test_the_default_dispatch_reproduces_the_recorded_benchmark_result(registry, cid):
    # the Coordinator's selection is still the B5 one (the Coordinator is unchanged); the judged Institutional result is the
    # stored Step 4.16-C1 one
    case = next(c for c in BLIND if c["id"] == cid)
    sp = registry.resolve(SpecialistId.INSTITUTIONAL)
    texts = {m.get("chunk_id"): t for t, m in zip(sp.resources.chunks, sp.resources.metadata)}
    out = D.run_question(case["question"], registry)                          # default dispatch: no executable= argument
    inst = next((r.findings for r in out.runs if r.specialist_id is SpecialistId.INSTITUTIONAL), None)
    assert sorted(s.value for s in out.decision.selected_specialists) == BASELINE[cid]["selected"] == BASELINE_C1[cid]["selected"]
    assert CR.judge(N, case, inst, texts) == BASELINE_C1[cid]["system"]


def test_the_c1_benchmark_result_differs_from_b5_only_by_new_correct_answers():
    good = ("correct", "correct_partial")
    assert list(BASELINE) == list(BASELINE_C1) == [c["id"] for c in BLIND]
    for cid, before in BASELINE.items():
        after = BASELINE_C1[cid]
        assert after["selected"] == before["selected"], cid
        for path in ("system", "direct"):
            assert after[path]["outcome"] not in ("wrong", "unexpected_answer"), (cid, path)
            if before[path]["outcome"] in good:
                assert after[path] == before[path], (cid, path)              # nothing answered before is lost or changed
            elif after[path] != before[path]:
                assert after[path]["outcome"] in good and after[path]["citations_ok"], (cid, path)
