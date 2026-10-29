"""
Institutional Navigation acceptance evaluation (Milestone 4, Step 4.16-B5).

1. Independent questions. ``qa/institutional_4_16/coverage_cases.jsonl`` holds 62 faculty-style questions whose expected
   answers were written by blind authors from the handbook pages alone (every expected quote checked verbatim on its
   page). They run through the unchanged Coordinator and the dispatcher with all four specialists (system path) and
   through Institutional Navigation alone (direct path), judged by ``coverage_runner.judge``. The safety results are
   asserted strictly (no wrong answer, every refusal of an unanswerable or foreign question kept, every citation on its
   expected page); the usefulness results (coverage) are recorded exactly, so any change is deliberate and reviewed.
2. The known B4 limitations A to I of the Step 4.16-B5 brief, each with its Coordinator selection, Institutional output,
   other specialists' output and handoffs, recorded as they are (unanswered questions are not counted as answers).
No model, no LLM, no network.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists import navigation_cues as N
from handbook_bot.agents.specialists.contracts import SpecialistId

from institutional_support import four_specialist_registry, make_institutional, run

REPO = Path(__file__).resolve().parents[1]
OVERRIDE = D.EXECUTABLE_SPECIALISTS + (SpecialistId.INSTITUTIONAL,)
_spec = importlib.util.spec_from_file_location("coverage_runner", REPO / "qa" / "institutional_4_16" / "coverage_runner.py")
CR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CR)
CASES = [json.loads(line) for line in open(REPO / "qa" / "institutional_4_16" / "coverage_cases.jsonl", encoding="utf-8")]


@pytest.fixture(scope="module")
def registry():
    return four_specialist_registry()


@pytest.fixture(scope="module")
def sp():
    return make_institutional()


@pytest.fixture(scope="module")
def rows(registry, sp):
    texts = {m.get("chunk_id"): t for t, m in zip(sp.resources.chunks, sp.resources.metadata)}
    out = []
    for case in CASES:
        result = D.run_question(case["question"], registry, executable=OVERRIDE)
        inst = next((r.findings for r in result.runs if r.specialist_id is SpecialistId.INSTITUTIONAL), None)
        out.append(dict(case, selected=sorted(s.value for s in result.decision.selected_specialists),
                        system=CR.judge(N, case, inst, texts), direct=CR.judge(N, case, run(sp, case["question"]), texts)))
    return out


def test_the_case_set_is_the_frozen_blind_set():
    assert len(CASES) == 62 and len({c["id"] for c in CASES}) == 62
    assert sum(c["answerable_from_handbook"] and "institutional" in c["expected_owner"] for c in CASES) == 48
    for c in CASES:
        assert c["source"].startswith("Step 4.16-B5 blind author")
        assert bool(c["answerable_from_handbook"]) == bool(c["evidence_quote"]) == bool(c["expected_pages"]), c["id"]


@pytest.mark.parametrize("path", ["system", "direct"])
def test_safety_no_wrong_answer_and_every_refusal_kept(rows, path):
    m = CR.metrics(rows, path)
    assert m["unsafe_supported"] == [] and m["wrong_any_status"] == []
    left, total = m["correct_refusals_on_unanswerable_or_foreign"].split("/")
    assert left == total == "14"
    right, answered = m["answer_correctness_on_answered"].split("/")
    assert right == answered
    ok, cited = m["citation_accuracy"].split("/")
    assert ok == cited


def test_usefulness_is_recorded_exactly(rows):
    """The measured coverage. A change here must be deliberate: update the numbers and the step report.
    Step 4.16-B5: system 6/48 (strict 3/48), direct 4/48 (strict 3/48). Step 4.16-C1 (natural phrasing with the closed
    context allowlist of the four review rounds; the questions, expectations and judge unchanged): system 12/48 (strict
    8/48), direct 10/48 (strict 9/48)."""
    system, direct = CR.metrics(rows, "system"), CR.metrics(rows, "direct")
    assert system["coverage_strict_supported"] == "8/48" and system["coverage_correct_supported_or_partial"] == "12/48"
    assert direct["coverage_strict_supported"] == "9/48" and direct["coverage_correct_supported_or_partial"] == "10/48"
    assert system["outcomes"] == {"abstained": 28, "correct": 8, "not_selected": 22, "correct_partial": 4}
    answered = {r["id"]: r["system"]["outcome"] for r in rows if r["system"]["outcome"] in ("correct", "correct_partial")}
    assert answered == {"CT01": "correct", "CT02": "correct", "CT04": "correct", "CT05": "correct", "CT07": "correct",
                        "RS02": "correct_partial", "SL02": "correct", "SL05": "correct", "SL12": "correct",
                        "SL14": "correct_partial", "BD01": "correct_partial", "BD02": "correct_partial"}


def test_coordinator_routing_is_recorded_exactly(rows):
    exact = sum(r["selected"] == sorted(o for o in r["expected_owner"] if o != "none") for r in rows)
    inclusion = sum(("institutional" in r["selected"]) == ("institutional" in r["expected_owner"]) for r in rows)
    assert (exact, inclusion) == (34, 38)


# ---------------------------------------------------------------------------
# The known B4 limitations (brief section 7)
# ---------------------------------------------------------------------------
LIMITS = [
    # id, question, Coordinator selection, Institutional status, Institutional handoffs, another specialist answers with findings
    ("A", "Which office handles faculty housing?", ["faculty_services", "institutional"], "out_of_scope", ["faculty_services"], False),
    ("B", "Who should I contact about my leave balance?", ["faculty_services", "institutional"], "out_of_scope", ["faculty_services"], False),
    ("C", "Who do I contact about academic integrity?", ["institutional"], "out_of_scope", [], False),
    ("D", "Which office provides IT support?", [], None, [], False),
    ("E", "Where is the research office?", ["institutional", "research"], "not_found", [], False),
    ("F", "Who approves a faculty research request?", ["institutional", "research"], "out_of_scope", ["research"], False),
    ("G1", "What is the phone number of HR and how much annual leave do I get?", ["faculty_services", "institutional"], "supported", [], True),
    ("G2", "What is the fax number of Human Resources and how long is the probation period?", ["faculty_services", "institutional"],
     "supported", [], True),
    ("H1", "What is the phone number of the Finance Department and who approves research grants?", ["institutional", "research"],
     "supported", ["research"], True),
    ("H2", "Where is the Medical Library and what percentage of licensing revenue goes to the inventor?", ["institutional", "research"],
     "supported", [], True),
    ("I1", "Who approves research leave?", ["institutional", "research"], "out_of_scope", ["research"], False),
    ("I2", "Who handles exam appointments?", ["faculty_services", "institutional", "teaching"], "out_of_scope", [], True),
    ("I3", "Who is responsible for environmental health?", ["institutional"], "out_of_scope", [], False),
]


@pytest.mark.parametrize("cid,question,selected,status,handoffs,other_answers", LIMITS, ids=[x[0] for x in LIMITS])
def test_known_limitation_is_recorded_as_it_is(registry, sp, cid, question, selected, status, handoffs, other_answers):
    out = D.run_question(question, registry, executable=OVERRIDE)
    assert sorted(s.value for s in out.decision.selected_specialists) == selected
    inst = next((r.findings for r in out.runs if r.specialist_id is SpecialistId.INSTITUTIONAL), None)
    assert (inst.status.value if inst else None) == status
    assert ([h.requested_specialist.value for h in inst.handoff_requests] if inst else []) == handoffs
    assert any(r.findings.findings for r in out.runs if r.specialist_id is not SpecialistId.INSTITUTIONAL) == other_answers
    if inst is not None and status == "out_of_scope":
        assert inst.findings == []                     # an unanswered clause is never presented as an answer
    if cid == "D":                                     # the Coordinator gap: Institutional answers it when given the task
        direct = run(sp, question)
        assert direct.status.value == "supported" and direct.findings[0].metadata["office_id"] == "information_technology_center"


# ---------------------------------------------------------------------------
# Safe failure (test strategy section 3)
# ---------------------------------------------------------------------------
def test_an_exception_inside_institutional_becomes_a_redacted_error_result(registry, monkeypatch):
    inst = registry.resolve(SpecialistId.INSTITUTIONAL)

    def boom(task):
        raise RuntimeError("index missing at C:/Users/someone/private/institutional.faiss gsk_ABCDEFGHIJKLMNOPQRST")

    monkeypatch.setattr(inst, "run", boom)
    out = D.run_question("What is the phone number of Human Resources?", registry, executable=OVERRIDE)
    result = next(r.findings for r in out.runs if r.specialist_id is SpecialistId.INSTITUTIONAL)
    assert result.status.value == "error" and result.findings == [] and result.llm_used is False
    text = json.dumps(result.to_dict())
    assert "someone/private" not in text and "gsk_ABCDEFGHIJKLMNOPQRST" not in text
