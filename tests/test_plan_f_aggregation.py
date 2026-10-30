"""
Runtime integration phase I-5: deterministic multi-finding aggregation (``handbook_bot/plan_f_aggregation.py``), its seam in
the adapter (``plan_f_runtime.PlanFRuntime._aggregate``) and in the pipeline (``ui/pipeline.py``), and the provision slice
of an excess quote (``plan_f_verification.verify_findings_sliced``).

The aggregator is exercised on structured specialist results built from the real handbook pages (production chunking and
metadata, fake embedder, stand-in reranker, no model, no network) through the real Coordinator and the real dispatcher,
with stub specialists that return controlled per-clause results: two specialists covering two clauses, one specialist whose
single report holds the findings of two clauses, same-page and multi-page evidence, a duplicate finding, and one-field
mutations of a valid aggregate (missing clause, rejected child, partial run, wrong entity, contradicted modifier, missing
condition, conflicting value, conflicting contact, handoff, shared clause, general ask, malformed finding, invalid
evidence reference, raising aggregator, reordered findings). The four frozen specialists are not modified; the legacy
Verifier is reused unchanged on every rendering.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

import test_plan_f_pipeline as P
import test_plan_f_runtime as T
import test_plan_f_verification as VT
from handbook_bot import config
from handbook_bot import plan_f_aggregation as A
from handbook_bot import plan_f_runtime as PF
from handbook_bot import plan_f_verification as V
from handbook_bot.agents.coordinator import coordinate
from handbook_bot.agents.specialists.contracts import CoordinatorDecision, Finding, FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask
from handbook_bot.agents.specialists.dispatch import STATE_PENDING, DispatchResult, SpecialistRun, dispatch
from handbook_bot.agents.types import VerifyResult
from handbook_bot.plan_f_aggregation import AggregationCode as G
from handbook_bot.plan_f_verification import ReasonCode as R
from ui import pipeline

REPO = Path(__file__).resolve().parents[1]
HR_PHONE = T.HR_PHONE

# ---- questions and the shapes the real Coordinator gives them (checked by test_question_shapes) -------------------------
CROSS = "How many credit hours is a full teaching load and how many days of sick leave with full pay are faculty entitled to?"
CROSS_REVERSED = "How many days of sick leave with full pay are faculty entitled to and how many credit hours is a full teaching load?"
CROSS_CONTACT = "How many credit hours is a full teaching load and what is the phone number of the Finance Department?"
CROSS_RESEARCH = "How many credit hours is a full teaching load and how many theses can a faculty member supervise at once?"
CROSS_CONDITION = "How many office hours per semester must faculty hold and how many days of sick leave with full pay are faculty entitled to?"
CROSS_COLLEGES = "Does HR assist new faculty members in obtaining entry visas, and how many colleges does the University of Sharjah have?"
SAME_HR = "Does HR arrange travel and hotel accommodations for new faculty members, and does HR assist new faculty members in obtaining entry visas?"
SAME_TEACHING = "How many office hours must faculty hold, and how many credit hours is a full teaching load?"
OFFICE_TWICE = "How many office hours must faculty hold, and how many office hours per week must faculty hold?"
ENUM_MULTI = "What does HR provide to new faculty members, and who drafts employment offers?"
SHARED = "Who approves research leave?"
VISAS = "Who handles faculty visas and where is that office?"
SYNTHESIS = "Where is the Research Institute and what research funding is available?"
LOAD_CLAUSE = "How many credit hours is a full teaching load"
SICK_CLAUSE = "how many days of sick leave with full pay are faculty entitled to"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def kb():
    return T.make_kb()


@pytest.fixture(scope="module")
def runtime(kb):
    runtime, build = PF.build_runtime(kb)
    assert build.ok, build.errors
    return runtime


@pytest.fixture(scope="module")
def Q(kb):
    """Real quotes, located in their production chunks."""
    quotes = {
        "load": VT.located(kb, "o Ranked full time regular faculty, visiting faculty or senior lecturer workload", "per academic year."),
        "lecturer": VT.located(kb, 'o "E" Lecturer is mainly devoted to teaching and services.', "per semester."),
        "sick": VT.located(kb, "• Faculty are entitled to sick leave with full pay", "per contract year."),
        "maternity": VT.located(kb, "• Female faculty employed on a full-time basis", "Academic Affairs."),
        "office": VT.located(kb, "1. Full-Time Faculty: Full-time faculty members", "receive academic support."),
        "evening": VT.located(kb, "3. Consideration for Evening Courses:", "during the evening."),
        "theses": VT.located(kb, "• A faculty member is allowed to supervise a maximum of four theses", "thesis process."),
        "travel": VT.located(kb, "• HR arranges travel and hotel accommodations", "new faculty members."),
        "visas": VT.located(kb, "• HR assists new faculty members in obtaining entry visas", "Passport Unit."),
        "offers": VT.located(kb, "• Employment offers are drafted by HR", "Security Clearance."),
        "colleges": VT.located(kb, "Academically, the University is composed of fifteen academic colleges", "Vice Chancellors."),
        "consultancy": VT.located(kb, "Consultancy agreements must include a clause", "professional consultancy activities."),
        "hr_provides": VT.located(kb, "• HR provides information to new faculty members", "settling in the UAE)."),
        "hr_row": (T.chunk_index(kb, "Human Resources +971 6 5050023"), "Human Resources +971 6 5050023 +971 6 5585200"),
        "finance_row": (T.chunk_index(kb, "Finance Department +971 6 5050040"), "Finance Department +971 6 5050040 +971 6 5585183"),
    }
    return quotes


@pytest.fixture(autouse=True)
def _legacy_mode(monkeypatch):
    monkeypatch.setattr(config, "PLAN_F_ENABLED", False)
    monkeypatch.setattr(config, "PLAN_F_SHADOW", False)
    pipeline.reset_plan_f_state()
    yield
    pipeline.reset_plan_f_state()


def finding(kb, cid, quote, task, *, number=1, clauses=(0,), **metadata):
    """A finding of ``task`` whose metadata names the clause report(s) it answers, as the frozen specialists' findings do."""
    return T.finding_for(kb, cid, quote, task, number=number, clauses=list(clauses), **metadata)


def retrieval(kb, task, located, *, requested=("number or amount",), covered=None, relation=None, ask_shape="specific_recognised",
              status=FindingStatus.SUPPORTED, missing=(), handoffs=(), extra=(), outcome=None, llm_used=False, number=1):
    """A one-report result of a retrieval specialist for ``task`` (its first focus clause) whose finding covers every requested detail."""
    cid, quote = located
    clause = task.context["focus_clauses"][0]
    covered = list(requested if covered is None else covered)
    findings = [finding(kb, cid, quote, task, number=number, details_covered=covered)] + list(extra)
    report = {"index": 0, "clause": clause, "outcome": outcome or status.value, "ask_shape": ask_shape, "requested_details": list(requested),
              "uncovered_details": [], "route": "policy", "findings": [f.finding_id for f in findings]}
    if relation is not None:
        report["relation"] = relation
    return SpecialistFindings(specialist_id=task.specialist_id, task_id=task.task_id, status=status, findings=findings, summary="stub",
                              missing=list(missing), handoff_requests=list(handoffs), llm_used=llm_used,
                              metadata={"clauses": [report], "unprocessed_clauses": []})


def directory(kb, task, located, *, office_id, value, field="telephone", offices=None):
    """A complete directory row (E1) of Institutional Navigation for ``task``."""
    cid, quote = located
    meta = {"office_id": office_id, "office_name": office_id.replace("_", " ").title(), "evidence_tier": "E1", "directory_entry": "complete_row",
            "row_complete": True, "contact_fields": [field], "contact_values": {field: value}}
    f = finding(kb, cid, quote, task, **meta)
    report = {"index": 0, "clause": task.context["focus_clauses"][0], "outcome": "read", "ask_kind": "channel", "evidence_tier_reached": "E1",
              "offices": list(offices if offices is not None else [office_id]), "findings": [f.finding_id]}
    return SpecialistFindings(specialist_id=SpecialistId.INSTITUTIONAL, task_id=task.task_id, status=FindingStatus.SUPPORTED, findings=[f],
                              summary="stub", metadata={"clauses": [report]})


def stated_fact(kb, task, located):
    """A stated fact about the University (Institutional Navigation, tier stated_fact)."""
    cid, quote = located
    f = finding(kb, cid, quote, task, office_id="university", office_name="University of Sharjah", evidence_tier="stated_fact",
                binding="named subject", relation="composition")
    report = {"index": 0, "clause": task.context["focus_clauses"][0], "outcome": "read", "ask_kind": "fact_or_count",
              "evidence_tier_reached": "stated_fact", "offices": [], "findings": [f.finding_id]}
    return SpecialistFindings(specialist_id=SpecialistId.INSTITUTIONAL, task_id=task.task_id, status=FindingStatus.SUPPORTED, findings=[f],
                              summary="stub", metadata={"clauses": [report]})


def not_found(task):
    return SpecialistFindings(specialist_id=task.specialist_id, task_id=task.task_id, status=FindingStatus.NOT_FOUND, summary="stub",
                              missing=["nothing"], metadata={"clauses": [{"index": 0, "clause": task.context["focus_clauses"][0], "outcome": "not_found"}]})


def runtime_with(kb, **behaviours):
    return PF.PlanFRuntime(T.stub_registry(**behaviours), kb.chunks, kb.metadata)


def attempt(kb, question, **behaviours):
    return runtime_with(kb, **behaviours).attempt(question)


def cross_stubs(kb, Q, *, teaching=None, hr=None):
    """The valid two-specialist aggregate: the page-91 workload sentence for the teaching clause, the page-107 sick-leave sentence
    for the HR clause (overridable)."""
    def teach(task):
        return retrieval(kb, task, Q["load"]) if teaching is None else teaching(task)

    def hr_(task):
        return retrieval(kb, task, Q["sick"]) if hr is None else hr(task)

    return {"teaching": teach, "faculty_services": hr_}


def decision_for(question, clauses, tasks, *, uncovered=(), complexity="cross_domain", route="count"):
    """A hand-built Coordinator decision: ``tasks`` are (specialist id, focus clauses) in task order."""
    subtasks = [SpecialistTask("task-%d" % (n + 1), question, specialist_id=sid, intent=route,
                               context={"full_question": question, "focus_clauses": list(focus)}, requested_by="coordinator",
                               metadata={"scoped": list(focus) != [question]}) for n, (sid, focus) in enumerate(tasks)]
    selected = list(dict.fromkeys(sid for sid, _f in tasks))
    return CoordinatorDecision(selected_specialists=selected, subtasks=subtasks, requires_synthesis=True, confidence=1.0, reason="test",
                               metadata={"complexity": complexity, "analysis": {"clauses": list(clauses)},
                                         "router": {"route": route, "is_multipart": True}, "uncovered_clauses": list(uncovered),
                                         "dropped_domains": []})


def aggregate_direct(kb, decision, **behaviours):
    dispatched = dispatch(decision, T.stub_registry(**behaviours), question=decision.subtasks[0].question)
    return A.aggregate(decision, dispatched, kb.chunks, kb.metadata)


def strip(attempt_):
    """The attempt without wall times, for determinism comparisons."""
    d = attempt_.to_dict()
    d.pop("timings")
    for e in d["trace"]:
        e["ms"] = 0.0
        e["detail"].pop("specialist_ms", None)
    if d["result"]:
        d["result"]["timings"] = {}
        for e in d["result"]["agent_trace"]:
            e["ms"] = 0.0
            e["detail"].pop("specialist_ms", None)
    return d


# ---------------------------------------------------------------------------
# Contract: codes, result shape, hygiene, determinism, no model
# ---------------------------------------------------------------------------
def test_codes_are_the_project_vocabulary():
    assert {c.value for c in G} == {"AGGREGATED_ACCEPTED", "NO_AGGREGATION_REQUIRED", "MULTIPART_UNSUPPORTED", "UNRESOLVED_CLAUSE",
                                    "CROSS_DOMAIN_UNRESOLVED", "HANDOFF_REQUIRED", "UNSAFE_RESULT", "MISSING_REQUIRED_CLAUSE",
                                    "CHILD_FINDING_REJECTED", "VALUE_CONFLICT", "AGGREGATION_ERROR"}
    assert A.PHASE == "I-5" and A.VERSION == "plan-f-aggregation-1"
    assert PF.PlanFOutcome.REJECTED_BY_AGGREGATOR.value == "PLAN_F_REJECTED_BY_AGGREGATOR"
    assert A.PATH_PASSTHROUGH == PF.PATH_PASSTHROUGH == "single_finding_passthrough" and A.PATH_AGGREGATE == PF.PATH_AGGREGATE == "multi_finding_aggregate"


def test_question_shapes_from_the_real_coordinator():
    """The fixtures rely on these decisions; the Coordinator is unchanged in I-5."""
    def shape(question):
        d = coordinate(question)
        return d.metadata["analysis"]["clauses"], [(t.specialist_id.value, t.context["focus_clauses"]) for t in d.subtasks]

    clauses, tasks = shape(CROSS)
    assert clauses == [LOAD_CLAUSE, SICK_CLAUSE] and tasks == [("teaching", [LOAD_CLAUSE]), ("faculty_services", [SICK_CLAUSE])]
    clauses, tasks = shape(CROSS_REVERSED)
    assert clauses[0].lower() == SICK_CLAUSE.lower() and clauses[1].lower() == LOAD_CLAUSE.lower() and [t for t, _ in tasks] == ["teaching", "faculty_services"]
    assert [t for t, _ in shape(CROSS_CONTACT)[1]] == ["teaching", "institutional"]
    assert [t for t, _ in shape(CROSS_RESEARCH)[1]] == ["teaching", "research"]
    assert [t for t, _ in shape(CROSS_CONDITION)[1]] == ["faculty_services", "teaching"]
    assert [t for t, _ in shape(CROSS_COLLEGES)[1]] == ["faculty_services", "institutional"]
    for question in (SAME_HR, SAME_TEACHING, OFFICE_TWICE, ENUM_MULTI):
        clauses, tasks = shape(question)
        assert len(clauses) == 2 and len(tasks) == 1 and tasks[0][1] == [question]          # one specialist, the whole question as its clause
    assert len(shape(HR_PHONE)[0]) == 1 and len(shape(HR_PHONE)[1]) == 1


def test_result_is_immutable_serialisable_deterministic_and_text_free(kb, Q):
    first = attempt(kb, CROSS, **cross_stubs(kb, Q))
    second = attempt(kb, CROSS, **cross_stubs(kb, Q))
    record = first.aggregation
    assert isinstance(record, A.AggregationResult) and record.accepted and record.safe_to_render
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.accepted = False
    text = json.dumps(record.to_dict())
    assert record.to_dict() == second.aggregation.to_dict()                  # no wall time inside the record
    assert CROSS not in text and LOAD_CLAUSE not in text and "12 credit hours" not in text and "15 working days" not in text
    assert "ms" not in record.to_dict() and record.ms >= 0.0 and set(record.timings) == {"map_ms", "verify_ms", "conflict_ms", "dedup_ms"}
    assert json.dumps(first.to_dict(), default=str)


def test_aggregation_module_has_no_model_network_or_randomness_dependency():
    source = (REPO / "handbook_bot" / "plan_f_aggregation.py").read_text(encoding="utf-8")
    for forbidden in ("ask_groq", "synthesize", "Groq(", "chat.completions", "build_prompt", "import random", "import requests", "import httpx",
                      "import openai", "from .orchestrator", "import orchestrator", "from .retrieval", "from .agents.router"):
        assert forbidden not in source, forbidden


# ---------------------------------------------------------------------------
# Reading the decision: clauses, ownership, pre-dispatch problems
# ---------------------------------------------------------------------------
def test_request_clauses_ownership_and_applicability():
    decision = coordinate(CROSS)
    assert A.request_clauses(decision) == [LOAD_CLAUSE, SICK_CLAUSE] and A.applicable(decision)
    owners = A.clause_owners(decision)
    assert [[(t.task_id, i, s) for t, i, s in lst] for lst in owners] == [[("task-1", 0, False)], [("task-2", 0, False)]]
    assert A.request_problem(decision) is None
    single = coordinate(HR_PHONE)
    assert len(A.request_clauses(single)) == 1 and not A.applicable(single) and not A.applicable(coordinate("Hello"))
    shared = coordinate(OFFICE_TWICE)
    assert A.applicable(shared) and all(lst == [(shared.subtasks[0], 0, True)] for lst in A.clause_owners(shared))
    assert A.request_problem(shared) is None


@pytest.mark.parametrize("question,code,part", [
    (SHARED, G.CROSS_DOMAIN_UNRESOLVED, "research, institutional"),
    (VISAS, G.CROSS_DOMAIN_UNRESOLVED, "institutional, faculty_services"),
    ("Which office handles faculty housing?", G.CROSS_DOMAIN_UNRESOLVED, "assigned to"),
])
def test_shared_clauses_are_rejected_before_any_specialist_runs(kb, question, code, part):
    calls = []

    def spy(task):
        calls.append(task.task_id)
        return not_found(task)

    problem = A.request_problem(coordinate(question))
    assert problem is not None and problem[0] is code and part in problem[1]
    result = attempt(kb, question, teaching=spy, research=spy, faculty_services=spy, institutional=spy)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason.startswith(code.value) and result.requires_fallback
    assert calls == [] and "dispatcher" not in T.trace_agents(result) and result.aggregation.dispatched is False
    assert result.path == PF.PATH_AGGREGATE and result.verified is False and result.result is None


def test_decision_level_problems_are_coded():
    decision = decision_for(CROSS, [LOAD_CLAUSE, SICK_CLAUSE], [("teaching", [LOAD_CLAUSE]), ("faculty_services", [SICK_CLAUSE])], uncovered=["where is it"])
    assert A.request_problem(decision) == (G.UNRESOLVED_CLAUSE, "1 request clause(s) belong to no selected specialist")
    decision = decision_for(CROSS, [LOAD_CLAUSE, SICK_CLAUSE], [("teaching", [LOAD_CLAUSE]), ("faculty_services", [SICK_CLAUSE])])
    decision.subtasks[1].requested_by = "teaching"
    assert A.request_problem(decision)[0] is G.UNSAFE_RESULT
    decision = decision_for(CROSS, [LOAD_CLAUSE, SICK_CLAUSE], [("teaching", [LOAD_CLAUSE]), ("faculty_services", ["another clause entirely"])])
    assert A.request_problem(decision) == (G.UNRESOLVED_CLAUSE, "request clause 1 is owned by no task")     # neither a request clause nor the whole question
    decision = decision_for(CROSS, [LOAD_CLAUSE, SICK_CLAUSE], [("teaching", [LOAD_CLAUSE, SICK_CLAUSE]), ("faculty_services", [CROSS])])
    assert A.request_problem(decision) == (G.CROSS_DOMAIN_UNRESOLVED, "task task-2 (faculty_services) answers no request clause")
    decision = decision_for(CROSS, [], [("teaching", [CROSS])])
    assert A.request_problem(decision)[0] is G.MULTIPART_UNSUPPORTED


# ---------------------------------------------------------------------------
# Positive: two specialists, two clauses; ordering; pages; duplicates; pass-through
# ---------------------------------------------------------------------------
def test_two_specialists_cover_two_clauses(kb, Q):
    result = attempt(kb, CROSS, **cross_stubs(kb, Q))
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and not result.requires_fallback and result.verified, result.reason
    assert result.path == PF.PATH_AGGREGATE and result.verification is None and result.specialist_id == "teaching,faculty_services"
    answer = result.result
    assert answer.answer == Q["load"][1] + "\n\n" + Q["sick"][1]                          # verbatim quotes, request order, nothing added
    assert answer.pages == [91, 107] and answer.llm_calls == 0 and answer.used_llm is False and answer.retried is False
    assert [(i["idx"], i["meta"]["page"], i["meta"]["plan_f"]["clause_index"], i["meta"]["plan_f"]["specialist_id"]) for i in answer.items] == [
        (Q["load"][0], 91, 0, "teaching"), (Q["sick"][0], 107, 1, "faculty_services")]
    assert [(s.question, s.route) for s in answer.sub_questions] == [(LOAD_CLAUSE, "count"), (SICK_CLAUSE, "count")]
    assert answer.num_candidates == 2 and answer.num_reranked == 2 and answer.evidence.startswith(Q["load"][1][:40])
    record = result.aggregation.to_dict()
    assert record["reason"] == "AGGREGATED_ACCEPTED" and record["clause_count"] == record["covered_count"] == 2 and record["completeness"] == "complete"
    assert [c["finding_ids"] for c in record["clauses"]] == [["task-1-f1"], ["task-2-f1"]] and all(c["covered"] for c in record["clauses"])
    assert [(e["finding_id"], e["page"], e["clause_indexes"]) for e in record["evidence"]] == [("task-1-f1", 91, [0]), ("task-2-f1", 107, [1])]
    assert record["duplicates"] == [] and record["conflicts"] == [] and record["specialists"] == ["teaching", "faculty_services"]
    assert record["pages"] == [91, 107] and record["findings_considered"] == 2 and record["findings_accepted"] == 2 and record["dispatched"] is True
    assert T.trace_agents(result) == ["plan_f_adapter", "plan_f_registry", "coordinator", "aggregator", "dispatcher", "specialist:teaching",
                                      "specialist:faculty_services", "aggregator", "verifier", "plan_f_outcome"]
    decisions = [e.decision for e in result.trace if e.agent == "aggregator"]
    assert decisions == ["shape:cross_domain", "accepted"] and all(e.used_llm is False for e in result.trace)
    assert {"aggregation_ms", "aggregation_map_ms", "aggregation_verify_ms", "aggregation_dedup_ms", "aggregation_conflict_ms", "verify_ms"} <= set(result.timings)


def test_output_is_deterministic_across_runs(kb, Q):
    runs = [attempt(kb, CROSS, **cross_stubs(kb, Q)) for _ in range(3)]
    assert strip(runs[0]) == strip(runs[1]) == strip(runs[2])
    assert len({r.result.answer for r in runs}) == 1 and len({tuple(r.result.pages) for r in runs}) == 1
    assert len({tuple(i["meta"]["plan_f"]["finding_id"] for i in r.result.items) for r in runs}) == 1
    assert len({json.dumps(r.aggregation.to_dict()) for r in runs}) == 1
    decision = coordinate(CROSS)
    dispatched = dispatch(decision, T.stub_registry(**cross_stubs(kb, Q)), question=CROSS)
    a, b = A.aggregate(decision, dispatched, kb.chunks, kb.metadata), A.aggregate(decision, dispatched, kb.chunks, kb.metadata)
    assert a.result.to_dict() == b.result.to_dict() and [p.finding.finding_id for p in a.parts] == [p.finding.finding_id for p in b.parts]
    assert A.render_parts(a.parts) == A.render_parts(b.parts)


def test_rendering_follows_request_order_not_task_order(kb, Q):
    """Clause 1 is the HR clause here although the teaching task is task-1."""
    result = attempt(kb, CROSS_REVERSED, **cross_stubs(kb, Q))
    assert result.outcome is PF.PlanFOutcome.ACCEPTED, result.reason
    assert result.result.answer == Q["sick"][1] + "\n\n" + Q["load"][1] and result.result.pages == [107, 91]
    assert [i["meta"]["plan_f"]["clause_index"] for i in result.result.items] == [0, 1]
    assert [e["task_id"] for e in result.aggregation.to_dict()["evidence"]] == ["task-2", "task-1"]


def test_multi_page_evidence_keeps_every_page(kb, Q):
    def research(task):
        return retrieval(kb, task, Q["theses"], relation="strong")

    result = attempt(kb, CROSS_RESEARCH, teaching=lambda task: retrieval(kb, task, Q["load"]), research=research)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED, result.reason
    assert result.result.pages == [91, 81] and {i["meta"]["page"] for i in result.result.items} == {91, 81}
    assert result.aggregation.pages == (91, 81)


def test_same_page_evidence_from_one_shared_report(kb, Q):
    """One specialist got the whole question as its clause; its two findings are mapped to the two clauses by verification."""
    def hr(task):
        first = retrieval(kb, task, Q["travel"], requested=("the asked specifics stated with the answer",))
        second = finding(kb, *Q["visas"], task, number=2, details_covered=["the asked specifics stated with the answer"])
        first.findings.append(second)
        first.metadata["clauses"][0]["findings"].append(second.finding_id)
        return first

    result = attempt(kb, SAME_HR, faculty_services=hr)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED, result.reason
    assert result.result.answer == Q["travel"][1] + "\n\n" + Q["visas"][1] and result.result.pages == [61]
    record = result.aggregation.to_dict()
    assert [c["shared_report"] for c in record["clauses"]] == [True, True] and [c["finding_ids"] for c in record["clauses"]] == [["task-1-f1"], ["task-1-f2"]]
    assert [(e["finding_id"], e["page"], e["clause_indexes"]) for e in record["evidence"]] == [("task-1-f1", 61, [0]), ("task-1-f2", 61, [1])]


def test_one_verified_finding_may_cover_two_clauses_and_is_rendered_once(kb, Q):
    result = attempt(kb, OFFICE_TWICE, teaching=lambda task: retrieval(kb, task, Q["office"]))
    assert result.outcome is PF.PlanFOutcome.ACCEPTED, result.reason
    assert result.result.answer == Q["office"][1] and result.result.pages == [79] and len(result.result.items) == 1
    record = result.aggregation.to_dict()
    assert record["covered_count"] == 2 and record["duplicates"] == [["task-1-f1", "task-1-f1", 1]]
    assert record["evidence"] == [{"finding_id": "task-1-f1", "task_id": "task-1", "specialist_id": "teaching", "source_id": result.result.items[0]["meta"]["source_id"],
                                   "page": 79, "chunk_id": Q["office"][0], "clause_indexes": [0, 1], "derived_from": None}]


def test_the_same_quote_returned_twice_is_rendered_once(kb, Q):
    def hr(task):
        twin = finding(kb, *Q["sick"], task, number=2, details_covered=["number or amount"])
        return retrieval(kb, task, Q["sick"], extra=[twin])

    result = attempt(kb, CROSS, **cross_stubs(kb, Q, hr=hr))
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and result.result.answer.count(Q["sick"][1]) == 1
    assert result.aggregation.duplicates == (("task-2-f2", "task-2-f1", 1),) and len(result.result.items) == 2


def test_single_finding_pass_through_is_unchanged(kb, runtime, monkeypatch):
    result = runtime.attempt(HR_PHONE)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and result.path == PF.PATH_PASSTHROUGH and result.aggregation is None and result.verified
    assert result.result.answer == "Human Resources +971 6 5050023 +971 6 5585200" and result.result.sub_questions == []
    assert T.trace_agents(result) == ["plan_f_adapter", "plan_f_registry", "coordinator", "dispatcher", "specialist:institutional", "quote_guard",
                                      "subject_guard", "findings_verifier", "verifier", "plan_f_outcome"]
    assert next(e for e in result.trace if e.agent == "plan_f_outcome").detail["path"] == PF.PATH_PASSTHROUGH
    called = []
    monkeypatch.setattr(A, "aggregate", lambda *a, **k: called.append(1))
    assert runtime.attempt(HR_PHONE).outcome is PF.PlanFOutcome.ACCEPTED and called == []      # the aggregator is never consulted


def test_direct_aggregate_on_a_single_clause_decision_says_so(kb, runtime):
    decision = coordinate(HR_PHONE)
    dispatched = dispatch(decision, runtime.registry, question=HR_PHONE)
    outcome = A.aggregate(decision, dispatched, kb.chunks, kb.metadata)
    assert outcome.result.reason is G.NO_AGGREGATION_REQUIRED and not outcome.accepted and outcome.parts == ()


# ---------------------------------------------------------------------------
# Enumeration: still never proven complete
# ---------------------------------------------------------------------------
def test_enumerative_clause_inside_a_multipart_request_is_not_covered(kb, Q):
    def hr(task):
        first = retrieval(kb, task, Q["hr_provides"], requested=("the asked specifics stated with the answer",))
        second = finding(kb, *Q["offers"], task, number=2, details_covered=["the asked specifics stated with the answer"])
        first.findings.append(second)
        return first

    result = attempt(kb, ENUM_MULTI, faculty_services=hr)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason.startswith("CHILD_FINDING_REJECTED: clause 0: ENUMERATION_INCOMPLETE")
    record = result.aggregation
    assert record.clauses[0].covered is False and record.clauses[0].verification.reason is R.ENUMERATION_INCOMPLETE and record.covered_count == 0


# ---------------------------------------------------------------------------
# The provision slice (the I-4 consultancy case)
# ---------------------------------------------------------------------------
def test_provision_slice_keeps_the_answering_provision_and_its_lead_in():
    quote = "Leave entitlements are as follows: • Sick leave with full pay is 15 working days. • Maternity leave is sixty calendar days."
    base = Finding("f1", quote, quote, "src", "Source", 7, metadata={"chunk_id": 3, "clauses": [0]})
    units = V.quote_units(quote)
    assert [p for _t, p in units] == [0, 1, 2]
    sliced = V.provision_slice(base, 1)
    expected = "Leave entitlements are as follows: • Sick leave with full pay is 15 working days."
    assert sliced.evidence_quote == expected
    assert sliced.finding_id == "f1/slice" and sliced.claim == sliced.evidence_quote and sliced.page == 7 and sliced.source_id == "src"
    assert sliced.metadata == {"chunk_id": 3, "clauses": [0], "derived_from": "f1", "slice": [0, len(expected)], "slice_provision": 1}
    start, end = sliced.metadata["slice"]
    assert quote[start:end] == sliced.evidence_quote                                      # an exact substring, by offsets
    later = V.provision_slice(base, 2)
    assert later.evidence_quote == "• Maternity leave is sixty calendar days." and later.metadata["slice_provision"] == 2
    assert V.provision_slice(base, 9) is None and V.provision_slice(base, -1) is None
    single = Finding("f2", "One sentence only.", "One sentence only.", "src", "Source", 7, metadata={"chunk_id": 3})
    assert V.provision_slice(single, 0) is None                                            # nothing to cut


def test_consultancy_quote_is_served_as_its_provision_slice(kb, Q):
    def research(task):
        return retrieval(kb, task, Q["consultancy"], relation="strong")

    result = attempt(kb, VT.CONSULTANCY, research=research)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and result.path == PF.PATH_PASSTHROUGH, result.reason
    expected = "16.1.7 Time Allocation for Consultancy: Full-time faculty members are permitted to spend up to 20% of their time or an equivalent of one working day per week on professional consultancy activities."
    assert result.result.answer == expected and result.result.pages == [260]
    item = result.result.items[0]
    assert item["chunk"] == expected and expected in kb.chunks[item["idx"]] and Q["consultancy"][1] not in result.result.answer
    plan_f = item["meta"]["plan_f"]
    assert plan_f["finding_id"] == "task-1-f1/slice" and plan_f["derived_from"] == "task-1-f1" and plan_f["slice"] == [117, 314] and plan_f["slice_provision"] == 1
    assert Q["consultancy"][1][117:314] == expected
    entry = next(e for e in result.trace if e.agent == "findings_verifier")
    assert entry.detail["sliced"] == [["task-1-f1/slice", "task-1-f1", 117, 314]] and entry.detail["accepted_finding_ids"] == ["task-1-f1/slice"]
    assert result.verification.accepted and result.verification.completeness.value == "complete"


def test_slice_is_not_used_when_it_fails_on_its_own_or_when_the_objection_is_not_excess(kb, Q):
    task = VT.task_for(VT.CONSULTANCY, "research")
    findings = retrieval(kb, task, Q["consultancy"], relation="strong")
    sliced = V.verify_findings_sliced(VT.CONSULTANCY, findings, kb.chunks, kb.metadata)
    assert sliced.verification.accepted and sliced.sliced == (("task-1-f1/slice", "task-1-f1", 117, 314),)
    assert sliced.findings.findings[0].finding_id == "task-1-f1/slice" and sliced.findings.findings[0].metadata["derived_from"] == "task-1-f1"
    # the slice must pass every check itself: asked per semester, the sliced unit (per week) fails the condition check
    other = "How many days per semester can faculty spend on consultancy?"
    sliced = V.verify_findings_sliced(other, retrieval(kb, VT.task_for(other, "research"), Q["consultancy"], relation="strong"), kb.chunks, kb.metadata)
    assert not sliced.verification.accepted and sliced.sliced == () and sliced.findings is not None
    assert sliced.verification.reason in (R.CONDITION_MISMATCH, R.EXCESS_EVIDENCE)
    # a quote the verifier accepts is returned untouched
    plain = retrieval(kb, VT.task_for(VT.SICK, "faculty_services"), Q["sick"])
    sliced = V.verify_findings_sliced(VT.SICK, plain, kb.chunks, kb.metadata)
    assert sliced.verification.accepted and sliced.sliced == () and sliced.findings is plain


def test_clause_view_projects_one_report_and_keeps_the_run_level_facts(kb, Q):
    task = VT.task_for(CROSS, "teaching")
    first = retrieval(kb, task, Q["load"])
    second = finding(kb, *Q["sick"], task, number=2, clauses=(1,), details_covered=["number or amount"])
    first.findings.append(second)
    first.metadata["clauses"].append({"index": 1, "clause": SICK_CLAUSE, "outcome": "supported", "ask_shape": "specific_recognised",
                                      "requested_details": ["number or amount"], "uncovered_details": [], "findings": [second.finding_id]})
    view = V.clause_view(first, 1)
    assert [f.finding_id for f in view.findings] == ["task-1-f2"] and view.metadata["clauses"][0]["index"] == 1 and view.metadata["view_index"] == 1
    assert view.status is first.status and view.task_id == first.task_id and view.specialist_id is first.specialist_id
    assert [f.finding_id for f in V.clause_view(first, 0, ["task-1-f2", "task-1-f1"]).findings] == ["task-1-f1", "task-1-f2"]   # the result's order
    with pytest.raises(ValueError):
        V.clause_view(first, 2)
    with pytest.raises(ValueError):
        V.clause_view(first, 0, ["nobody"])
    assert V.status_code(first) == (R.MULTIPART_UNSUPPORTED, "2 clause reports") and V.status_code(view) == (None, "")
    assert V.run_code(first) == (None, "") and V.report_code(SpecialistId.TEACHING, first.metadata["clauses"][0]) == (None, "")


# ---------------------------------------------------------------------------
# Negative: one wrong component, no aggregate (all or nothing)
# ---------------------------------------------------------------------------
def test_missing_clause_falls_back_instead_of_serving_the_other_clause(kb, Q):
    result = attempt(kb, CROSS, **cross_stubs(kb, Q, hr=not_found))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.result is None and result.requires_fallback
    assert result.reason == "MISSING_REQUIRED_CLAUSE: task task-2: status not_found" and result.aggregation.covered_count == 0
    assert result.aggregation.completeness == "not_assessed" and result.status == "not_found" and "verifier" not in T.trace_agents(result)


def test_partial_run_falls_back(kb, Q):
    def hr(task):
        return retrieval(kb, task, Q["sick"], status=FindingStatus.PARTIAL, missing=["half pay not stated"])

    result = attempt(kb, CROSS, **cross_stubs(kb, Q, hr=hr))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason == "MISSING_REQUIRED_CLAUSE: task task-2: status partial"


@pytest.mark.parametrize("label,hr_quote,code", [
    ("contradicted modifier: maternity sentence for the sick-leave clause", "maternity", R.MODIFIER_MISMATCH),
    ("wrong subject: lecturer teaching load for the sick-leave clause", "lecturer", R.SUBJECT_MISMATCH),
])
def test_child_rejected_by_the_findings_verifier_falls_back(kb, Q, label, hr_quote, code):
    result = attempt(kb, CROSS, **cross_stubs(kb, Q, hr=lambda task: retrieval(kb, task, Q[hr_quote])))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.result is None, label
    assert result.reason.startswith("CHILD_FINDING_REJECTED: clause 1: %s" % code.value), (label, result.reason)
    record = result.aggregation
    assert record.covered_count == 1 and record.clauses[0].covered is True and record.clauses[1].covered is False
    assert record.clauses[1].verification.reason is code and record.completeness == "partial" and record.evidence == ()
    assert "verifier" not in T.trace_agents(result)                                       # the first clause is never rendered alone


def test_missing_condition_falls_back(kb, Q):
    result = attempt(kb, CROSS_CONDITION, teaching=lambda task: retrieval(kb, task, Q["office"]), faculty_services=lambda task: retrieval(kb, task, Q["sick"]))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR
    assert result.reason.startswith("CHILD_FINDING_REJECTED: clause 0: CONDITION_MISMATCH") and result.aggregation.clauses[0].verification.reason is R.CONDITION_MISMATCH


def test_wrong_entity_in_a_contact_clause_falls_back(kb, Q):
    def institutional(task):
        return directory(kb, task, Q["hr_row"], office_id="human_resources", value="+971 6 5050023", offices=["finance_department"])

    result = attempt(kb, CROSS_CONTACT, teaching=lambda task: retrieval(kb, task, Q["load"]), institutional=institutional)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason.startswith("CHILD_FINDING_REJECTED: clause 1: SUBJECT_MISMATCH")
    good = attempt(kb, CROSS_CONTACT, teaching=lambda task: retrieval(kb, task, Q["load"]),
                   institutional=lambda task: directory(kb, task, Q["finance_row"], office_id="finance_department", value="+971 6 5050040"))
    assert good.outcome is PF.PlanFOutcome.ACCEPTED and good.result.answer == Q["load"][1] + "\n\n" + Q["finance_row"][1] and good.result.pages == [91, 54]


def test_wrong_contact_field_falls_back(kb, Q):
    def fax_only(task):
        return directory(kb, task, Q["finance_row"], office_id="finance_department", value="+971 6 5585183", field="fax")

    result = attempt(kb, CROSS_CONTACT, teaching=lambda task: retrieval(kb, task, Q["load"]), institutional=fax_only)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason.startswith("CHILD_FINDING_REJECTED: clause 1: REQUESTED_DETAIL_MISSING")


def test_stated_fact_and_yes_no_clauses_aggregate_across_specialists(kb, Q):
    result = attempt(kb, CROSS_COLLEGES, faculty_services=lambda task: retrieval(kb, task, Q["visas"], requested=("the asked specifics stated with the answer",)),
                     institutional=lambda task: stated_fact(kb, task, Q["colleges"]))
    assert result.outcome is PF.PlanFOutcome.ACCEPTED, result.reason
    assert result.result.answer == Q["visas"][1] + "\n\n" + Q["colleges"][1] and result.result.pages == [61, 24]


def test_conflicting_numeric_values_for_the_same_ask_fall_back(kb, Q):
    c1, c2 = "How many office hours must faculty hold", "how many office hours should faculty hold"
    assert V.read_request(c1) == V.read_request(c2)
    decision = decision_for("%s, and %s?" % (c1, c2), [c1, c2], [("teaching", [c1]), ("teaching", [c2])])

    def teaching(task):
        return retrieval(kb, task, Q["office"] if task.context["focus_clauses"] == [c1] else Q["evening"])

    outcome = aggregate_direct(kb, decision, teaching=teaching)
    assert outcome.result.reason is G.VALUE_CONFLICT and not outcome.accepted and outcome.parts == ()
    assert outcome.result.conflicts == ((0, 1, "quantity"),) and outcome.result.covered_count == 2 and outcome.result.evidence == ()
    agreeing = aggregate_direct(kb, decision, teaching=lambda task: retrieval(kb, task, Q["office"]))
    assert agreeing.accepted and agreeing.result.duplicates == (("task-2-f1", "task-1-f1", 1),)


def test_conflicting_contact_values_for_the_same_office_and_field_fall_back(kb, Q):
    c1, c2 = "What is the phone number of the Human Resources Department", "What is the phone number of the Human Resources Department?"
    assert V.read_request(c1) == V.read_request(c2)
    decision = decision_for(c1 + " and " + c2, [c1, c2], [("institutional", [c1]), ("institutional", [c2])], route="contact")

    def institutional(task):
        value = "+971 6 5050023" if task.context["focus_clauses"] == [c1] else "+971 6 5585200"
        return directory(kb, task, Q["hr_row"], office_id="human_resources", value=value)

    outcome = aggregate_direct(kb, decision, institutional=institutional)
    assert outcome.result.reason is G.VALUE_CONFLICT and outcome.result.conflicts == ((0, 1, "contact"),)


def test_a_duplicate_never_covers_a_clause_it_did_not_verify_for(kb, Q):
    result = attempt(kb, SAME_TEACHING, teaching=lambda task: retrieval(kb, task, Q["office"]))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason.startswith("CHILD_FINDING_REJECTED: clause 1:")
    assert result.aggregation.clauses[0].covered is True and result.aggregation.clauses[1].covered is False and result.aggregation.evidence == ()


def test_handoff_present_falls_back_and_never_runs_the_target(kb, Q):
    calls = []

    def hr(task):
        calls.append(task.specialist_id.value)
        return retrieval(kb, task, Q["sick"], handoffs=[HandoffRequest(from_specialist=task.specialist_id, requested_specialist=SpecialistId.INSTITUTIONAL, reason="stub")])

    def institutional(task):
        raise AssertionError("the handoff target must never run")

    result = attempt(kb, CROSS, **dict(cross_stubs(kb, Q, hr=hr), institutional=institutional))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason == "HANDOFF_REQUIRED: task task-2 requests a handoff to institutional"
    assert calls == ["faculty_services"] and result.dispatch is not None and len(result.dispatch.handoffs) == 1       # preserved, not executed


def test_general_ask_report_is_rejected_as_a_child(kb, Q):
    result = attempt(kb, CROSS, **cross_stubs(kb, Q, teaching=lambda task: retrieval(kb, task, Q["load"], ask_shape="general")))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason.startswith("CHILD_FINDING_REJECTED: clause 0: GENERAL_ASK_MISMATCH")


def test_synthesis_question_falls_back_with_a_named_reason(kb):
    result = attempt(kb, SYNTHESIS, research=lambda task: retrieval(kb, task, (0, "x"), status=FindingStatus.PARTIAL, missing=["location"]) if False else
                     SpecialistFindings(specialist_id=task.specialist_id, task_id=task.task_id, status=FindingStatus.PARTIAL,
                                        findings=[Finding("task-1-f1", "x", "x", "src", page=1, metadata={"chunk_id": 0, "clauses": [0]})],
                                        summary="stub", missing=["location"], metadata={"clauses": [{"index": 0, "clause": task.context["focus_clauses"][0], "outcome": "partial"}]}))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason == "MISSING_REQUIRED_CLAUSE: task task-1: status partial"


@pytest.mark.parametrize("label,mutate,code,child", [
    ("wrong page", lambda f: setattr(f, "page", f.page + 1), G.CHILD_FINDING_REJECTED, R.INVALID_EVIDENCE_REFERENCE),
    ("wrong chunk", lambda f: f.metadata.__setitem__("chunk_id", f.metadata["chunk_id"] + 1), G.CHILD_FINDING_REJECTED, R.INVALID_EVIDENCE_REFERENCE),
    ("tampered quote", lambda f: setattr(f, "evidence_quote", f.evidence_quote.replace("15", "25")), G.CHILD_FINDING_REJECTED, R.INVALID_EVIDENCE_REFERENCE),
    ("finding without its clause index", lambda f: f.metadata.__setitem__("clauses", []), G.MISSING_REQUIRED_CLAUSE, None),
])
def test_malformed_or_invalid_evidence_falls_back(kb, Q, label, mutate, code, child):
    def hr(task):
        result = retrieval(kb, task, Q["sick"])
        mutate(result.findings[0])
        return result

    result = attempt(kb, CROSS, **cross_stubs(kb, Q, hr=hr))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason.startswith(code.value + ": clause 1"), (label, result.reason)
    if child is not None:
        assert result.aggregation.clauses[1].verification.reason is child, label


def test_aggregation_exception_is_redacted_and_falls_back(kb, Q, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError(r"defect at C:\Users\someone\private\agg.json with key gsk_ABCDEFGHIJKLMNOPQRST")

    monkeypatch.setattr(A, "EvidenceRef", boom)                                   # raised inside aggregate(): after coverage, while building the evidence list
    result = attempt(kb, CROSS, **cross_stubs(kb, Q))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason == "AGGREGATION_ERROR: aggregation raised: RuntimeError"
    text = json.dumps(result.to_dict(), default=str)
    assert "gsk_ABCDEFGHIJKLMNOPQRST" not in text and "someone" not in text and "defect at" not in text
    assert result.aggregation.reason is G.AGGREGATION_ERROR and result.verified is False and result.aggregation.covered_count == 2
    monkeypatch.setattr(V, "read_request", boom)                                  # raised inside a clause's verification: a rejected child
    result = attempt(kb, CROSS, **cross_stubs(kb, Q))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason.startswith("CHILD_FINDING_REJECTED: clause 0: VERIFICATION_ERROR")
    assert "someone" not in json.dumps(result.to_dict(), default=str)
    monkeypatch.setattr(A, "request_problem", boom)                               # raised before dispatch: a redacted adapter error, never a raise
    result = attempt(kb, CROSS, **cross_stubs(kb, Q))
    assert result.outcome is PF.PlanFOutcome.ERROR_REDACTED and result.reason.startswith("aggregator raised: RuntimeError") and result.result is None
    assert "someone" not in json.dumps(result.to_dict(), default=str) and "dispatcher" not in T.trace_agents(result)


def test_pending_task_and_llm_report_fall_back(kb, Q, monkeypatch):
    def pending(decision, registry, *, executable=None, question=None):
        runs = [SpecialistRun(task, task.specialist_id, STATE_PENDING, None, "not registered") for task in decision.subtasks]
        return DispatchResult(question=question or "", decision=decision, runs=runs)

    monkeypatch.setattr(PF, "dispatch", pending)
    result = attempt(kb, CROSS, **cross_stubs(kb, Q))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason == "UNRESOLVED_CLAUSE: task task-1 was not executed"
    monkeypatch.undo()
    result = attempt(kb, CROSS, **cross_stubs(kb, Q, hr=lambda task: retrieval(kb, task, Q["sick"], llm_used=True)))
    assert result.outcome is PF.PlanFOutcome.UNSAFE_RESULT and "LLM call" in result.reason and result.aggregation is None
    decision = coordinate(CROSS)
    dispatched = dispatch(decision, T.stub_registry(**cross_stubs(kb, Q, hr=lambda task: retrieval(kb, task, Q["sick"], llm_used=True))), question=CROSS)
    assert A.aggregate(decision, dispatched, kb.chunks, kb.metadata).result.reason is G.UNSAFE_RESULT


def test_verifier_must_cite_every_evidence_page(kb, Q, monkeypatch):
    real = PF.verify

    def partial_pages(question, synthesis, evidence, sub_questions, *, is_retry=False):
        verdict = real(question, synthesis, evidence, sub_questions, is_retry=is_retry)
        return verdict if is_retry else VerifyResult(True, [91], [], [], None)                 # the final rendering: one page cited only

    monkeypatch.setattr(PF, "verify", partial_pages)
    result = attempt(kb, CROSS, **cross_stubs(kb, Q))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_VERIFIER and "did not cite every evidence page (107)" in result.reason and result.result is None

    def rejecting(question, synthesis, evidence, sub_questions, *, is_retry=False):
        return real(question, synthesis, evidence, sub_questions, is_retry=is_retry) if is_retry else VerifyResult(False, [], ["stub"], [], None)

    monkeypatch.setattr(PF, "verify", rejecting)
    result = attempt(kb, CROSS, **cross_stubs(kb, Q))
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_VERIFIER and result.result is None and result.aggregation.accepted   # the aggregate was fine; the rendering was not


def test_reordered_findings_keep_the_specialists_order_inside_a_clause(kb, Q):
    def teaching(task):
        second = finding(kb, *Q["lecturer"], task, number=2, details_covered=["number or amount"])
        return retrieval(kb, task, Q["load"], extra=[second])

    def reordered(task):
        result = teaching(task)
        result.findings.reverse()
        return result

    straight = attempt(kb, CROSS, **cross_stubs(kb, Q, teaching=teaching))
    assert straight.outcome is PF.PlanFOutcome.ACCEPTED and straight.result.answer.startswith(Q["load"][1])
    assert straight.aggregation.clauses[0].verification.dropped == (("task-1-f2", "VALUE_CONFLICT"),)       # 15 versus 12 credit hours
    swapped = attempt(kb, CROSS, **cross_stubs(kb, Q, teaching=reordered))
    assert swapped.outcome is PF.PlanFOutcome.ACCEPTED and swapped.result.answer.startswith(Q["lecturer"][1])
    assert swapped.aggregation.clauses[0].verification.dropped == (("task-1-f1", "VALUE_CONFLICT"),)
    assert strip(straight) != strip(swapped)                                               # the specialist's order is followed, never re-ranked


# ---------------------------------------------------------------------------
# Pipeline: LEGACY untouched, SHADOW non-invasive, ACTIVE requires the aggregate and every clause record
# ---------------------------------------------------------------------------
def test_shadow_keeps_the_legacy_answer_and_records_the_aggregation(kb, Q, monkeypatch):
    P.set_mode(monkeypatch, pipeline.MODE_SHADOW)
    P.install_runtime(monkeypatch, P.stub_runtime(kb, **cross_stubs(kb, Q)))
    result, d = pipeline.ask_with_diagnostics(kb, CROSS)
    direct = P.legacy_direct(kb, CROSS)
    assert P.visible(result) == P.visible(direct) and P.trace_of(result) == P.trace_of(direct)
    assert d["mode"] == "shadow" and d["served"] == "legacy" and d["accepted"] is True and d["outcome"] == "PLAN_F_ACCEPTED"
    assert d["plan_f_path"] == "multi_finding_aggregate" and d["aggregation_attempted"] is True and d["aggregation_accepted"] is True
    assert d["aggregation_reason"] == "AGGREGATED_ACCEPTED" and d["aggregation_clauses"] == 2 and d["aggregation_covered"] == 2
    assert d["aggregation_findings_considered"] == 2 and d["aggregation_findings_accepted"] == 2 and d["aggregation_duplicates"] == 0
    assert d["aggregation_conflict"] is False and d["aggregation_pages"] == [91, 107] and d["aggregation_specialists"] == ["teaching", "faculty_services"]
    assert d["aggregation_dispatched"] is True and d["aggregation_ms"] >= 0.0 and d["plan_f_pages"] == [91, 107]
    assert d["findings_verified"] is True and d["findings_accepted"] is True and d["findings_reason"] == "ACCEPTED" and d["findings_completeness"] == "complete"
    text = json.dumps(d)
    assert CROSS not in text and LOAD_CLAUSE not in text and Q["load"][1][:30] not in text and "5050023" not in text


def test_active_serves_the_aggregate_and_falls_back_on_a_rejected_child(kb, Q, monkeypatch):
    P.set_mode(monkeypatch, pipeline.MODE_ACTIVE)
    P.install_runtime(monkeypatch, P.stub_runtime(kb, **cross_stubs(kb, Q)))
    result, d = pipeline.ask_with_diagnostics(kb, CROSS)
    assert d["served"] == "plan_f" and d["accepted"] is True and result.answer == Q["load"][1] + "\n\n" + Q["sick"][1] and result.pages == [91, 107]
    pipeline.reset_plan_f_state()
    P.install_runtime(monkeypatch, P.stub_runtime(kb, **cross_stubs(kb, Q, hr=lambda task: retrieval(kb, task, Q["maternity"]))))
    result, d = pipeline.ask_with_diagnostics(kb, CROSS)
    direct = P.legacy_direct(kb, CROSS)
    assert d["served"] == "legacy" and d["accepted"] is False and P.visible(result) == P.visible(direct)
    assert d["outcome"] == "PLAN_F_REJECTED_BY_AGGREGATOR" and d["aggregation_reason"] == "CHILD_FINDING_REJECTED" and d["aggregation_covered"] == 1
    assert d["findings_reason"] == "MODIFIER_MISMATCH" and d["findings_accepted"] is False and d["fallback_reason"].startswith("CHILD_FINDING_REJECTED")


def test_active_requires_the_aggregate_and_every_clause_record(kb, Q):
    real = attempt(kb, CROSS, **cross_stubs(kb, Q))
    assert real.verified and pipeline._verified(real)
    record = real.aggregation
    unaccepted = dataclasses.replace(real, aggregation=dataclasses.replace(record, accepted=False, safe_to_render=False))
    assert unaccepted.verified is False and pipeline._verified(unaccepted) is False
    clauses = list(record.clauses)
    clauses[1] = dataclasses.replace(clauses[1], verification=None)
    missing = dataclasses.replace(real, aggregation=dataclasses.replace(record, clauses=tuple(clauses)))
    assert missing.verified is False and pipeline._verified(missing) is False
    clauses[1] = dataclasses.replace(record.clauses[1], verification=dataclasses.replace(record.clauses[1].verification, accepted=False, safe_to_render=False))
    rejected = dataclasses.replace(real, aggregation=dataclasses.replace(record, clauses=tuple(clauses)))
    assert rejected.verified is False and pipeline._verified(rejected) is False
    assert dataclasses.replace(real, aggregation=dataclasses.replace(record, clauses=())).verified is False


def test_legacy_mode_is_untouched_by_the_aggregation_layer(kb):
    assert pipeline.plan_f_mode() == pipeline.MODE_LEGACY
    result, d = pipeline.ask_with_diagnostics(kb, CROSS)
    assert d == {"mode": "legacy", "served": "legacy", "attempted": False} and result.answer == pipeline._legacy_answer(kb, CROSS).answer
    source = (REPO / "ui" / "pipeline.py").read_text(encoding="utf-8")
    assert "plan_f_aggregation" not in source and "plan_f_verification" not in source      # the seam reads the adapter's record only
    for name in ("app.py", "ui/chat.py", "handbook_bot/qa.py", "handbook_bot/orchestrator.py", "handbook_bot/agents/router.py",
                 "handbook_bot/retrieval.py", "handbook_bot/agents/synthesis.py", "handbook_bot/agents/verifier.py", "handbook_bot/knowledge_base.py"):
        assert "plan_f_aggregation" not in (REPO / name).read_text(encoding="utf-8"), name


def test_real_specialists_on_the_test_corpus_aggregate_the_cross_domain_question(runtime):
    """No stub: the frozen Teaching and HR specialists on the fake-embedder corpus, through the real Coordinator and dispatcher."""
    result = runtime.attempt(CROSS)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and result.path == PF.PATH_AGGREGATE and result.verified, result.reason
    assert result.result.pages[-1] == 107 and "sick leave with full pay" in result.result.answer and "credit hours" in result.result.answer
    assert [i["meta"]["plan_f"]["clause_index"] for i in result.result.items] == sorted(i["meta"]["plan_f"]["clause_index"] for i in result.result.items)
    assert result.result.llm_calls == 0 and result.dispatch.llm_calls == 0 and all(e.used_llm is False for e in result.trace)
    assert strip(runtime.attempt(CROSS)) == strip(result)
