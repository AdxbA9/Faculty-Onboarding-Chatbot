"""
Institutional Navigation alignment (Milestone 4, Step 4.16-B4, D-104): adding Institutional Navigation changes neither the
Coordinator's routing nor the frozen specialists' results.

* The Coordinator's selections for the B4 matrix and for questions of the frozen specialists' own domains are the
  selections of the unchanged Coordinator (recorded here; ``coordinator.py`` and ``domain_cues.py`` are not modified).
* Running Institutional Navigation (executed by default since Step 4.16-B6, D-105) leaves the decision, the tasks and
  every Teaching, Research and Faculty Services & HR result identical to a dispatch in which Institutional Navigation
  is not registered and so stays pending: no evidence, task or handoff crosses between specialists.
No model, no LLM, no network.
"""
from __future__ import annotations

import pytest

from handbook_bot.agents.coordinator import coordinate
from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists.contracts import SpecialistId

from institutional_support import four_specialist_registry

from test_institutional_integration import HR, INST, MATRIX, OVERRIDE, RESEARCH, TEACHING

#: Questions of the frozen specialists' own domains (from their test files) and the recorded Coordinator cue gaps: the
#: selection of the unchanged Coordinator.
DOMAIN_QUESTIONS = [
    ("What is the probation period for new faculty?", [HR]),
    ("What percentage of licensing revenue goes to the inventor?", [RESEARCH]),
    ("How many office hours must I hold?", [TEACHING]),
    ("How many days of sick leave do I get and how many office hours must I hold?", [HR, TEACHING]),
    ("What does Human Resources do?", [HR]),
    ("Who is the Chancellor?", []),                                       # cue gap DEP-2: no specialist selected
    ("What does the Business Services Department do?", []),               # cue gap DEP-2
]
ALL_QUESTIONS = [(row[1], row[2]) for row in MATRIX] + DOMAIN_QUESTIONS


def _without_ms(value):
    if isinstance(value, dict):
        return {k: _without_ms(v) for k, v in value.items() if k != "ms"}
    if isinstance(value, list):
        return [_without_ms(v) for v in value]
    return value


@pytest.fixture(scope="module")
def registry():
    return four_specialist_registry()


@pytest.mark.parametrize("question,selected", ALL_QUESTIONS)
def test_the_coordinator_selections_are_unchanged(question, selected):
    decision = coordinate(question)
    assert [s.value for s in decision.selected_specialists] == selected
    assert decision.to_dict() == coordinate(question).to_dict()


@pytest.fixture(scope="module")
def registry_without_institutional(registry):
    from handbook_bot.agents.specialists.registry import SpecialistRegistry

    reg = SpecialistRegistry()
    for sid in (SpecialistId.TEACHING, SpecialistId.RESEARCH, SpecialistId.FACULTY_SERVICES):
        reg.register(registry.resolve(sid))
    return reg


@pytest.mark.parametrize("question,selected", ALL_QUESTIONS)
def test_executing_institutional_changes_no_other_result(registry, registry_without_institutional, question, selected):
    with_inst = D.run_question(question, registry)                        # default dispatch: all four execute (D-105)
    default = D.run_question(question, registry_without_institutional)   # Institutional not registered: pending
    assert with_inst.decision.to_dict() == default.decision.to_dict() == coordinate(question).to_dict()
    assert [r.task.task_id for r in with_inst.runs] == [r.task.task_id for r in default.runs] == [t.task_id for t in with_inst.decision.subtasks]
    for a, b in zip(with_inst.runs, default.runs):
        assert a.specialist_id is b.specialist_id and a.task is not None and a.task.to_dict() == b.task.to_dict()
        if a.specialist_id is SpecialistId.INSTITUTIONAL:
            assert b.state == D.STATE_PENDING and b.findings is None and a.state == D.STATE_EXECUTED
        else:
            assert a.state == b.state == D.STATE_EXECUTED
            assert _without_ms(a.findings.to_dict()) == _without_ms(b.findings.to_dict())
    # the frozen specialists' handoffs are the same; only Institutional Navigation's own are added
    others = [h.to_dict() for h in with_inst.handoffs if h.from_specialist is not SpecialistId.INSTITUTIONAL]
    assert others == [h.to_dict() for h in default.handoffs]


def test_institutional_tasks_reach_it_exactly_as_the_coordinator_built_them(registry):
    for question, selected in ALL_QUESTIONS:
        if INST not in selected:
            continue
        out = D.run_question(question, registry, executable=OVERRIDE)
        for run_, task in zip(out.runs, out.decision.subtasks):
            assert run_.task is task
            if run_.specialist_id is SpecialistId.INSTITUTIONAL:
                assert [c["clause"] for c in run_.findings.metadata["clauses"]] == task.context["focus_clauses"]
