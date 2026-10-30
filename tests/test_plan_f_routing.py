"""
Runtime integration phase I-6: deterministic routing refinement (``handbook_bot/plan_f_routing.py``), the bounded handoff
round (``handbook_bot/plan_f_handoff.py``) and their seams in the adapter and the pipeline.

Routing is exercised on the real Coordinator's decisions for the Category C questions of the I-5 corpus and on hand-built
decisions; handoffs on stub specialists that return the frozen specialists' own handoff shape (a task for one clause, the
source as requester, the clause report ``handed_off``) and on crafted runs for the control-flow edges: self loop, visited
target, repeated target, unknown or unregistered target, malformed task, whole-question handoff, a clause the source
answered, an unsupported source, a second hop, a raising target, a model-reporting target. Every handoff answer goes through
the I-4 verifier and the I-5 aggregator; the four frozen specialists and the Coordinator's decisions are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_plan_f_aggregation as AT
import test_plan_f_pipeline as P
import test_plan_f_runtime as T
import test_plan_f_verification as VT
from handbook_bot import plan_f_aggregation as A
from handbook_bot import plan_f_handoff as H
from handbook_bot import plan_f_routing as R
from handbook_bot import plan_f_runtime as PF
from handbook_bot.agents.coordinator import coordinate, cue_scores
from handbook_bot.agents.specialists.contracts import FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask
from handbook_bot.agents.specialists.dispatch import STATE_EXECUTED, DispatchResult, SpecialistRun, dispatch
from handbook_bot.agents.specialists.registry import SpecialistRegistry
from handbook_bot.plan_f_aggregation import AggregationCode as G
from handbook_bot.plan_f_handoff import HandoffCode as HC
from handbook_bot.plan_f_routing import RoutingCode as RC
from ui import pipeline

REPO = Path(__file__).resolve().parents[1]
kb = AT.kb                      # the shared real-page knowledge base (fake embedder, stand-in reranker, no model)
runtime = AT.runtime
Q = AT.Q
_legacy_mode = AT._legacy_mode
HR_PHONE = T.HR_PHONE
CROSS = AT.CROSS
LOAD_CLAUSE, SICK_CLAUSE = AT.LOAD_CLAUSE, AT.SICK_CLAUSE
CONFERENCE = "Who approves conference travel?"
CEILING_APPROVAL = "Who approves the funding ceiling for research grants?"
THESES_HR_PHONE = "How many theses can a faculty member supervise at once and what is the phone number of the Human Resources Department?"
HOUSING = "Which office handles faculty housing?"
REDUCE = "Can I reduce my teaching load if I have a funded research project, and who approves it?"
RESEARCH_LEAVE = "Who approves research leave?"
HR_LOCATED = "Where is the Human Resources Department located?"
SAME_TEACHING = "How many credit hours is a full teaching load, and how many office hours must faculty hold?"
OFFERS = "Who drafts employment offers?"
OFFICE_CLAUSE = "how many office hours must faculty hold"


# ---------------------------------------------------------------------------
# Helpers: stub results in the frozen specialists' own shapes
# ---------------------------------------------------------------------------
def handoff_task(task, clause, target, *, index=0, kind="misrouted", requested_by=None, focus=None):
    """The handoff task the frozen specialists build: the full question, ONE focus clause, the source as requester."""
    return SpecialistTask("%s-handoff-%d" % (task.task_id, index), task.question, specialist_id=target, domain=target, intent=task.intent,
                          context={"full_question": task.question, "focus_clauses": list(focus) if focus is not None else [clause]},
                          requested_by=requested_by or task.specialist_id.value,
                          metadata={"origin_task_id": task.task_id, "clause_index": index, "kind": kind})


def stub_result(kb_, task, *, answers=None, declines=None, requested=("number or amount",), handoff_task_fn=None, shared_outcome=False, llm_used=False):
    """One result in the retrieval specialists' shape for every focus clause of ``task``: ``answers`` maps a clause to its
    located quote (supported), ``declines`` maps a clause to the specialist it is handed to (report ``handed_off`` and a
    handoff request carrying the clause task); any other clause is ``not_found``."""
    answers, declines = dict(answers or {}), dict(declines or {})
    findings, reports, missing, handoffs = [], [], [], []
    for index, clause in enumerate(task.context["focus_clauses"]):
        if clause in answers:
            cid, quote = answers[clause]
            f = AT.finding(kb_, cid, quote, task, number=len(findings) + 1, clauses=(index,), details_covered=list(requested))
            findings.append(f)
            reports.append({"index": index, "clause": clause, "outcome": "supported", "ask_shape": "specific_recognised",
                            "requested_details": list(requested), "uncovered_details": [], "route": "policy", "findings": [f.finding_id]})
        elif clause in declines:
            target = declines[clause]
            built = (handoff_task_fn or handoff_task)(task, clause, target, index=index)
            handoffs.append(HandoffRequest(from_specialist=task.specialist_id, requested_specialist=target, reason="misrouted clause %d" % index,
                                           task=built, metadata={"clause_index": index, "kind": "misrouted"}))
            reports.append({"index": index, "clause": clause, "outcome": "supported" if shared_outcome else "handed_off", "target": target})
        else:
            reports.append({"index": index, "clause": clause, "outcome": "not_found", "ask_shape": "specific_recognised",
                            "requested_details": list(requested), "uncovered_details": list(requested), "findings": []})
            missing.append("no evidence for clause %d" % index)
    if findings and not missing:
        status = FindingStatus.SUPPORTED
    elif findings:
        status = FindingStatus.PARTIAL
    elif handoffs and not missing:
        status = FindingStatus.OUT_OF_SCOPE
    else:
        status = FindingStatus.NOT_FOUND
    return SpecialistFindings(specialist_id=task.specialist_id, task_id=task.task_id, status=status, findings=findings, summary="stub", missing=missing,
                              handoff_requests=handoffs, llm_used=llm_used, metadata={"clauses": reports, "unprocessed_clauses": []})


def behaviour(kb_, **kwargs):
    return lambda task: stub_result(kb_, task, **kwargs)


def never(task):
    raise AssertionError("this specialist must not run")


def trace(attempt):
    return [(e.agent, e.decision) for e in attempt.trace]


# ---------------------------------------------------------------------------
# Routing refinement
# ---------------------------------------------------------------------------
def test_routing_codes_versions_and_outcomes():
    assert {c.value for c in RC} == {"ROUTE_UNCHANGED", "ROUTE_CLAUSES_SCOPED", "ROUTE_REFINED_BY_REQUEST_KIND", "ROUTE_AMBIGUOUS", "ROUTE_NO_OWNER", "ROUTE_ERROR"}
    assert {c.value for c in HC} == {"HANDOFF_NONE", "HANDOFF_EXECUTED", "HANDOFF_TARGET_INVALID", "HANDOFF_SELF_LOOP", "HANDOFF_TARGET_ALREADY_VISITED",
                                     "HANDOFF_CLAUSE_UNRESOLVED", "HANDOFF_REMAINDER_UNSUPPORTED", "HANDOFF_SECOND_HOP_BLOCKED", "HANDOFF_UNSAFE_RESULT",
                                     "HANDOFF_EXCEPTION"}
    assert R.PHASE == H.PHASE == "I-6" and H.MAX_DEPTH == 1 and H.APPROVED_TARGETS == {"teaching", "research", "faculty_services", "institutional"}
    assert PF.PlanFOutcome.REJECTED_BY_ROUTING.value == "PLAN_F_REJECTED_BY_ROUTING" and PF.PlanFOutcome.REJECTED_BY_HANDOFF.value == "PLAN_F_REJECTED_BY_HANDOFF"
    assert cue_scores("How many credit hours is a full teaching load") == coordinate("How many credit hours is a full teaching load").metadata["scores"]


def test_single_owner_decisions_are_returned_as_the_same_object(runtime):
    for question in (HR_PHONE, CROSS, "How many theses can a faculty member supervise at once?"):
        decision = coordinate(question)
        refined, routing = R.refine(decision)
        assert refined is decision and routing.reason is RC.ROUTE_UNCHANGED and routing.refined is False and routing.executable
        assert routing.initial_specialists == routing.final_specialists and routing.dropped_tasks == () and routing.rules_applied == ()
    result = runtime.attempt(HR_PHONE)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and result.path == PF.PATH_PASSTHROUGH and result.routing.reason is RC.ROUTE_UNCHANGED
    assert "routing" not in T.trace_agents(result) and result.handoff is None and result.handoff_outcome is None and result.specialist_calls == 1
    assert T.trace_agents(result) == ["plan_f_adapter", "plan_f_registry", "coordinator", "dispatcher", "specialist:institutional", "quote_guard",
                                      "subject_guard", "findings_verifier", "verifier", "plan_f_outcome"]
    assert "routing_ms" in result.timings and result.to_dict()["routing"]["reason"] == "ROUTE_UNCHANGED"


@pytest.mark.parametrize("question,owner,dropped", [
    (CONFERENCE, "research", "institutional"),
    (CEILING_APPROVAL, "research", "institutional"),
    (RESEARCH_LEAVE, "research", "institutional"),
    ("Who approves Hajj leave?", "faculty_services", "institutional"),
])
def test_party_ask_about_a_subject_is_owned_by_the_subject_specialist(question, owner, dropped):
    decision = coordinate(question)
    assert {s.value for s in decision.selected_specialists} == {owner, dropped}
    refined, routing = R.refine(decision)
    assert routing.reason is RC.ROUTE_REFINED_BY_REQUEST_KIND and routing.refined and routing.executable
    assert routing.final_specialists == (owner,) and [t.specialist_id.value for t in refined.subtasks] == [owner]
    assert routing.owners_before == ((owner, dropped),) and routing.owners_after == ((owner,),) and len(routing.dropped_tasks) == 1
    assert routing.rules_applied[0][0] == 0 and "party ask" in routing.rules_applied[0][1]
    assert refined.metadata["complexity"] == "simple" and refined.requires_synthesis is False and PF.decision_problem(refined) is None
    assert refined.metadata["routing"]["reason"] == "ROUTE_REFINED_BY_REQUEST_KIND" and refined.subtasks[0].context["focus_clauses"] == [question]


def test_refined_party_ask_is_served_from_the_subject_specialist_without_running_the_other(kb, Q):
    cid, quote = VT.located(kb, "• The Vice Chancellor for Academic Affairs approves travel requests", "interprets the policy.")
    research = behaviour(kb, answers={CONFERENCE: (cid, quote)}, requested=("approving authority",))
    result = AT.attempt(kb, CONFERENCE, research=research, institutional=never)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and result.path == PF.PATH_PASSTHROUGH, result.reason
    assert result.result.answer == quote and result.result.pages == [117] and result.specialist_calls == 1
    assert trace(result)[2:5] == [("coordinator", "research,institutional"), ("routing", "ROUTE_REFINED_BY_REQUEST_KIND"), ("dispatcher", "executed:1,pending:0")]
    assert result.routing.final_specialists == ("research",) and result.dispatch.executed_specialists == [SpecialistId.RESEARCH]


def test_contact_and_location_clauses_are_owned_by_institutional_navigation(kb, Q):
    decision = coordinate(THESES_HR_PHONE)
    assert [t.specialist_id.value for t in decision.subtasks] == ["institutional", "research", "faculty_services"]
    refined, routing = R.refine(decision)
    assert routing.reason is RC.ROUTE_REFINED_BY_REQUEST_KIND and routing.final_specialists == ("institutional", "research") and routing.dropped_tasks == ("task-3",)
    assert routing.owners_after == (("research",), ("institutional",)) and "navigation ask" in routing.rules_applied[0][1]
    research = behaviour(kb, answers={"How many theses can a faculty member supervise at once": Q["theses"]})
    institutional = lambda task: AT.directory(kb, task, Q["hr_row"], office_id="human_resources", value="+971 6 5050023")
    result = AT.attempt(kb, THESES_HR_PHONE, research=research, institutional=institutional, faculty_services=never)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and result.path == PF.PATH_AGGREGATE, result.reason
    assert result.result.answer == Q["theses"][1] + "\n\n" + Q["hr_row"][1] and result.result.pages == [81, 54] and result.specialist_calls == 2
    located_decision = coordinate(HR_LOCATED)
    refined, routing = R.refine(located_decision)
    assert routing.final_specialists == ("institutional",) and routing.reason is RC.ROUTE_REFINED_BY_REQUEST_KIND


def test_office_selection_ask_with_a_subject_claim_stays_ambiguous_and_runs_nobody():
    decision = coordinate(HOUSING)
    refined, routing = R.refine(decision)
    assert refined is decision and routing.reason is RC.ROUTE_AMBIGUOUS and not routing.executable and routing.ambiguous_clauses == (0,)
    assert "office-selection" in routing.rules_applied[0][1] and routing.owners_after == (("institutional", "faculty_services"),)
    result = AT.attempt(AT.kb.__wrapped__(None) if False else None, HOUSING) if False else None       # placeholder keeps the lambda-free style
    owner, why = R.settle_owner("Which office handles faculty housing", ["institutional", "faculty_services"])
    assert owner is None and "hand it to each other" in why


def test_ambiguous_routing_falls_back_before_any_specialist_runs(kb):
    for question in (HOUSING, REDUCE):
        result = AT.attempt(kb, question, teaching=never, research=never, faculty_services=never, institutional=never)
        assert result.outcome is PF.PlanFOutcome.REJECTED_BY_ROUTING and result.reason.startswith("ROUTE_AMBIGUOUS: request clause 0 is claimed by")
        assert result.result is None and result.requires_fallback and result.specialist_calls == 0 and result.dispatch is None
        assert trace(result)[-2][0] == "routing" and "dispatcher" not in T.trace_agents(result) and result.path is None
        assert result.routing.reason is RC.ROUTE_AMBIGUOUS and result.verified is False


def test_two_subject_specialists_stay_ambiguous_without_score_or_order():
    owner, why = R.settle_owner("Can I reduce my teaching load if I have a funded research project", ["research", "teaching"])
    assert owner is None and "no ownership rule" in why
    owner, why = R.settle_owner("Who approves conference travel", ["teaching", "research", "institutional"])
    assert owner is None and "several subject specialists" in why
    assert R.settle_owner("Who approves conference travel", ["research"]) == ("research", None)
    assert R.settle_owner("what is the phone number of the Human Resources Department", ["faculty_services", "teaching"])[0] is None


def test_same_domain_multipart_task_is_scoped_to_the_coordinators_clauses(kb, Q):
    decision = coordinate(SAME_TEACHING)
    assert decision.subtasks[0].context["focus_clauses"] == [SAME_TEACHING]
    refined, routing = R.refine(decision)
    assert routing.reason is RC.ROUTE_CLAUSES_SCOPED and routing.scoped_tasks == ("task-1",) and routing.refined
    assert refined.subtasks[0].context["focus_clauses"] == [LOAD_CLAUSE, OFFICE_CLAUSE] and refined.subtasks[0].metadata["scoped"] is True
    assert refined.metadata["complexity"] == "multipart" and refined.requires_synthesis is True
    teaching = behaviour(kb, answers={LOAD_CLAUSE: Q["load"], OFFICE_CLAUSE: Q["office"]})
    result = AT.attempt(kb, SAME_TEACHING, teaching=teaching)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and result.path == PF.PATH_AGGREGATE, result.reason
    assert result.result.answer == Q["load"][1] + "\n\n" + Q["office"][1] and result.result.pages == [91, 79] and result.specialist_calls == 1
    assert [s.question for s in result.result.sub_questions] == [LOAD_CLAUSE, OFFICE_CLAUSE]
    partial = behaviour(kb, answers={LOAD_CLAUSE: Q["load"]})                                   # the second clause not found: nothing served
    result = AT.attempt(kb, SAME_TEACHING, teaching=partial)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason == "MISSING_REQUIRED_CLAUSE: task task-1: status partial"


@pytest.mark.parametrize("question,expected", [
    ("What is the due date for course syllabi and the conference travel form?", 1),                 # a continuation, not a split (three specialists share it: ambiguous)
    ("Does HR arrange travel and hotel accommodations for new faculty members?", 1),
    ("How many office hours must faculty hold, and how many credit hours is a full teaching load?", 2),
    ("How many days of sick leave with full pay are faculty entitled to, and how many days of annual leave do faculty members get?", 2),
])
def test_clause_scoping_follows_the_coordinators_split_exactly(question, expected):
    decision = coordinate(question)
    clauses = A.request_clauses(decision)
    assert len(clauses) == expected
    refined, routing = R.refine(decision)
    if expected == 1:
        assert refined is decision and routing.scoped_tasks == () and routing.reason in (RC.ROUTE_UNCHANGED, RC.ROUTE_AMBIGUOUS)
    else:
        focus = refined.subtasks[0].context["focus_clauses"]
        assert focus == clauses                                                                    # request order and verbatim text
        assert all(c in question for c in focus) and question.index(focus[0]) < question.index(focus[1])
        if "with full pay" in question:
            assert focus[0].endswith("with full pay are faculty entitled to")                       # the condition travels with its clause


def test_clause_with_only_foreign_cues_has_no_owner(kb):
    clauses = [LOAD_CLAUSE, "what is the phone number of the Finance Department"]
    question = "%s, and %s?" % (clauses[0], clauses[1])
    decision = AT.decision_for(question, clauses, [("teaching", [question])], complexity="multipart")
    refined, routing = R.refine(decision)
    assert routing.reason is RC.ROUTE_NO_OWNER and routing.unowned_clauses == (1,) and not routing.executable and refined is decision
    assigned, unowned = R.assign_by_cue(clauses, ["teaching"])
    assert assigned == {"teaching": [LOAD_CLAUSE]} and unowned == [1]
    assigned, unowned = R.assign_by_cue(["How many credit hours is a full teaching load", "and what about part-time staff"], ["teaching"])
    assert assigned["teaching"] == ["How many credit hours is a full teaching load", "and what about part-time staff"] and unowned == []   # no cue: continues


def test_routing_error_is_redacted_and_falls_back(kb, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError(r"defect at C:\Users\someone\private\route.json gsk_ABCDEFGHIJKLMNOPQRST")

    monkeypatch.setattr(R, "settle_owner", boom)
    decision = coordinate(CONFERENCE)
    refined, routing = R.refine(decision)
    assert refined is decision and routing.reason is RC.ROUTE_ERROR and routing.detail == "routing raised: RuntimeError"
    result = AT.attempt(kb, CONFERENCE, research=never, institutional=never)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_ROUTING and result.reason == "ROUTE_ERROR: routing raised: RuntimeError" and result.result is None
    text = json.dumps(result.to_dict(), default=str)
    assert "someone" not in text and "gsk_ABCDEFGHIJKLMNOPQRST" not in text


def test_routing_is_deterministic_and_text_free():
    for question in (CONFERENCE, THESES_HR_PHONE, SAME_TEACHING, HOUSING, HR_PHONE):
        a, b = R.refine(coordinate(question))[1], R.refine(coordinate(question))[1]
        assert a.to_dict() == b.to_dict()
        text = json.dumps(a.to_dict())
        assert question not in text and "credit hours" not in text and "ms" not in a.to_dict()


# ---------------------------------------------------------------------------
# The bounded handoff round
# ---------------------------------------------------------------------------
def offers_fixture(kb):
    cid, quote = VT.located(kb, "• Employment offers are drafted by HR", "Security Clearance.")
    hr = behaviour(kb, declines={OFFERS: "teaching"})
    teaching = behaviour(kb, answers={OFFERS: (cid, quote)}, requested=("responsible party or unit",))
    return quote, hr, teaching


def test_valid_handoff_executes_once_and_the_answer_is_verified(kb):
    quote, hr, teaching = offers_fixture(kb)
    calls = []

    def counted(task):
        calls.append(task.task_id)
        return teaching(task)

    result = AT.attempt(kb, OFFERS, faculty_services=hr, teaching=counted)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and result.path == PF.PATH_AGGREGATE and result.verified, result.reason
    assert result.result.answer == quote and result.result.pages == [61] and calls == ["task-1-handoff-0"] and result.specialist_calls == 2
    assert result.handoff.code is HC.HANDOFF_EXECUTED and result.handoff.depth == 1 and result.handoff.visited == ("faculty_services", "teaching")
    assert result.handoff_outcome == "HANDOFF_ACCEPTED" and result.handoff.sources == ("faculty_services",) and result.handoff.targets == ("teaching",)
    record = result.handoff.records[0]
    assert (record.source, record.target, record.task_id, record.request_clause, record.report_index, record.kind, record.executed, record.status) == \
        ("faculty_services", "teaching", "task-1-handoff-0", 0, 0, "misrouted", True, "supported")
    assert trace(result)[2:] == [("coordinator", "faculty_services"), ("dispatcher", "executed:1,pending:0"), ("specialist:faculty_services", "out_of_scope"),
                                 ("handoff", "HANDOFF_EXECUTED"), ("specialist:teaching", "supported"), ("aggregator", "accepted"), ("verifier", "accepted"),
                                 ("plan_f_outcome", "PLAN_F_ACCEPTED")]
    item = result.result.items[0]["meta"]["plan_f"]
    assert item["specialist_id"] == "teaching" and item["clause_index"] == 0 and item["finding_id"] == "task-1-handoff-0-f1"
    agg = result.aggregation
    assert agg.clauses[0].task_id == "task-1-handoff-0" and agg.clauses[0].specialist_id == "teaching" and agg.specialists == ("teaching",)
    assert [r.task.task_id for r in result.dispatch.runs] == ["task-1", "task-1-handoff-0"] and result.dispatch.runs[1].reason == "handoff round, depth 1"
    assert result.dispatch.runs[0].findings.handoff_requests == [] and result.dispatch.handoffs == []    # settled: the executed request is no longer pending
    assert {"handoff_ms", "handoff_specialist_ms", "aggregation_ms"} <= set(result.timings)
    d = result.to_dict()
    assert d["handoff"]["code"] == "HANDOFF_EXECUTED" and d["handoff_outcome"] == "HANDOFF_ACCEPTED" and d["specialist_calls"] == 2
    assert OFFERS not in json.dumps(d["handoff"]) and quote not in json.dumps(d["handoff"])


def test_handoff_target_rejected_by_the_verifier_falls_back(kb, Q):
    quote, hr, _teaching = offers_fixture(kb)
    wrong = behaviour(kb, answers={OFFERS: Q["sick"]}, requested=("responsible party or unit",))
    result = AT.attempt(kb, OFFERS, faculty_services=hr, teaching=wrong)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.result is None
    assert result.reason.startswith("CHILD_FINDING_REJECTED: clause 0:") and result.handoff_outcome == "HANDOFF_TARGET_REJECTED"
    assert result.handoff.code is HC.HANDOFF_EXECUTED and result.aggregation.clauses[0].task_id == "task-1-handoff-0"


def test_handoff_completes_a_cross_domain_aggregate_only_when_the_target_answers(kb, Q):
    """Teaching answers clause 1; HR declines clause 2 to Research. With Research's answer: an aggregate. Without it: nothing."""
    teaching = behaviour(kb, answers={LOAD_CLAUSE: Q["load"]})
    hr = behaviour(kb, declines={SICK_CLAUSE: "research"})
    research = behaviour(kb, answers={SICK_CLAUSE: Q["sick"]})
    result = AT.attempt(kb, CROSS, teaching=teaching, faculty_services=hr, research=research)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and result.result.answer == Q["load"][1] + "\n\n" + Q["sick"][1] and result.result.pages == [91, 107]
    assert result.specialist_calls == 3 and result.handoff_outcome == "HANDOFF_ACCEPTED" and result.aggregation.specialists == ("teaching", "research")
    assert [c.task_id for c in result.aggregation.clauses] == ["task-1", "task-2-handoff-0"] and result.specialist_id == "teaching,research"
    missing = AT.attempt(kb, CROSS, teaching=teaching, faculty_services=hr, research=behaviour(kb))
    assert missing.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and missing.result is None and missing.handoff_outcome == "MISSING_REQUIRED_CLAUSE"
    assert missing.reason == "MISSING_REQUIRED_CLAUSE: task task-2-handoff-0: status not_found"      # clause 1 is never served alone


def test_handoff_conflict_falls_back(kb, Q, monkeypatch):
    c1, c2 = "How many office hours must faculty hold", "how many office hours should faculty hold"
    question = "%s, and %s?" % (c1, c2)
    decision = AT.decision_for(question, [c1, c2], [("teaching", [c1]), ("teaching", [c2])])
    monkeypatch.setattr(PF, "coordinate", lambda text: decision)

    def teaching(task):
        focus = task.context["focus_clauses"]
        return stub_result(kb, task, answers={c1: Q["office"]}) if focus == [c1] else stub_result(kb, task, declines={c2: "research"})

    research = behaviour(kb, answers={c2: Q["evening"]})
    result = AT.attempt(kb, question, teaching=teaching, research=research)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_AGGREGATOR and result.reason.startswith("VALUE_CONFLICT")
    assert result.handoff_outcome == "HANDOFF_CONFLICT" and result.aggregation.conflicts == ((0, 1, "quantity"),) and result.result is None


def test_handoff_to_an_already_executed_specialist_is_blocked(kb, Q):
    teaching = behaviour(kb, answers={LOAD_CLAUSE: Q["load"]})
    hr = behaviour(kb, declines={SICK_CLAUSE: "teaching"})
    result = AT.attempt(kb, CROSS, teaching=teaching, faculty_services=hr)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_HANDOFF and result.reason == "HANDOFF_TARGET_ALREADY_VISITED: faculty_services -> teaching: target already executed for this request"
    assert result.handoff.code is HC.HANDOFF_TARGET_ALREADY_VISITED and result.handoff.executed == () and result.specialist_calls == 2 and result.result is None
    assert result.handoff.visited == ("teaching", "faculty_services") and result.handoff.depth == 0


def test_repeated_target_from_two_sources_is_blocked(kb, Q, monkeypatch):
    c1, c2 = LOAD_CLAUSE, SICK_CLAUSE
    decision = AT.decision_for(CROSS, [c1, c2], [("teaching", [c1]), ("faculty_services", [c2])])
    monkeypatch.setattr(PF, "coordinate", lambda text: decision)
    calls = []

    def research(task):
        calls.append(task.task_id)
        return stub_result(kb, task, answers={c1: Q["load"], c2: Q["sick"]})

    result = AT.attempt(kb, CROSS, teaching=behaviour(kb, declines={c1: "research"}), faculty_services=behaviour(kb, declines={c2: "research"}), research=research)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_HANDOFF and result.handoff.code is HC.HANDOFF_TARGET_ALREADY_VISITED and calls == []
    codes = [r.code for r in result.handoff.records]
    assert HC.HANDOFF_TARGET_ALREADY_VISITED in codes and len(codes) == 2 and result.handoff.executed == ()


def test_second_hop_is_blocked_and_the_targets_request_is_recorded_not_followed(kb):
    quote, hr, teaching = offers_fixture(kb)
    research_calls = []

    def teaching_again(task):
        f = teaching(task)
        f.handoff_requests.append(HandoffRequest(from_specialist=SpecialistId.TEACHING, requested_specialist=SpecialistId.RESEARCH, reason="again",
                                                 task=handoff_task(task, OFFERS, "research", requested_by="teaching")))
        return f

    def research(task):
        research_calls.append(task.task_id)
        return stub_result(kb, task)

    result = AT.attempt(kb, OFFERS, faculty_services=hr, teaching=teaching_again, research=research)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_HANDOFF and result.reason == "HANDOFF_SECOND_HOP_BLOCKED: faculty_services -> teaching: the target asked for a handoff to research"
    assert research_calls == [] and result.handoff.depth == 1 and result.handoff.records[0].second_hop == ("research",) and result.specialist_calls == 2
    assert result.result is None and result.handoff_outcome == "HANDOFF_SECOND_HOP_BLOCKED"


def test_cycle_back_to_the_source_is_a_blocked_second_hop(kb):
    quote, hr, _teaching = offers_fixture(kb)

    def teaching_hands_back(task):
        return stub_result(kb, task, declines={OFFERS: "faculty_services"})

    result = AT.attempt(kb, OFFERS, faculty_services=hr, teaching=teaching_hands_back)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_HANDOFF and result.handoff.code is HC.HANDOFF_SECOND_HOP_BLOCKED
    assert result.handoff.records[0].second_hop == ("faculty_services",) and result.specialist_calls == 2 and result.handoff.visited == ("faculty_services", "teaching")


def crafted_round(kb, registry=None, *, source=SpecialistId.FACULTY_SERVICES, target=SpecialistId.TEACHING, focus=None, requested_by=None, task=True,
                  outcome="handed_off", run_specialist=None, clauses=None, llm_used=False):
    """A first-round execution of one task whose result carries one handoff request, for the validation edges."""
    decision = coordinate(OFFERS)
    source_task = decision.subtasks[0]
    assert source_task.specialist_id is source
    built = SpecialistTask("task-1-handoff-0", OFFERS, specialist_id=target.value if isinstance(target, SpecialistId) else "teaching",
                           context={"full_question": OFFERS, "focus_clauses": list(focus) if focus is not None else [OFFERS]},
                           requested_by=requested_by or source.value, metadata={"origin_task_id": "task-1", "clause_index": 0, "kind": "misrouted"}) if task else None
    findings = SpecialistFindings(specialist_id=source, task_id="task-1", status=FindingStatus.OUT_OF_SCOPE, summary="stub", llm_used=llm_used,
                                  handoff_requests=[HandoffRequest(from_specialist=source, requested_specialist=target, reason="stub", task=built)],
                                  metadata={"clauses": [{"index": 0, "clause": OFFERS, "outcome": outcome}]})
    run = SpecialistRun(source_task, run_specialist or source, STATE_EXECUTED, findings, "executed")
    dispatched = DispatchResult(question=OFFERS, decision=decision, runs=[run])
    registry = registry or T.stub_registry(teaching=lambda t: stub_result(kb, t, answers={OFFERS: VT.located(kb, "• Employment offers are drafted by HR", "Security Clearance.")},
                                                                         requested=("responsible party or unit",)))
    return H.run_handoffs(decision, dispatched, registry, clauses if clauses is not None else A.request_clauses(decision))


def test_handoff_validation_edges_block_without_running_anything(kb):
    round_, d2, r2 = crafted_round(kb, task=False)
    assert round_.code is HC.HANDOFF_TARGET_INVALID and d2 is None and r2 is None and round_.specialist_calls == 0 and "no well-formed handoff task" in round_.detail
    round_, _d, _r = crafted_round(kb, focus=[OFFERS, "and more"])
    assert round_.code is HC.HANDOFF_TARGET_INVALID
    round_, _d, _r = crafted_round(kb, requested_by="research")
    assert round_.code is HC.HANDOFF_TARGET_INVALID
    round_, _d, _r = crafted_round(kb, run_specialist=SpecialistId.TEACHING)                       # the run's specialist equals the target
    assert round_.code is HC.HANDOFF_SELF_LOOP
    three = SpecialistRegistry()
    for sid in (SpecialistId.RESEARCH, SpecialistId.FACULTY_SERVICES, SpecialistId.INSTITUTIONAL):
        three.register(T.stub_class(sid)(never))
    round_, _d, _r = crafted_round(kb, registry=three)
    assert round_.code is HC.HANDOFF_TARGET_INVALID and "not registered" in round_.detail
    round_, _d, _r = crafted_round(kb, focus=["Who drafts the employment contracts"])
    assert round_.code is HC.HANDOFF_CLAUSE_UNRESOLVED and "names no single request clause" in round_.detail
    round_, _d, _r = crafted_round(kb, outcome="supported")                                        # a shared-ownership note, not a decline
    assert round_.code is HC.HANDOFF_CLAUSE_UNRESOLVED and "did not decline" in round_.detail
    round_, _d, _r = crafted_round(kb, llm_used=False, clauses=["Who drafts employment offers", "and who signs them"])   # whole question of a two-clause request
    assert round_.code is HC.HANDOFF_CLAUSE_UNRESOLVED
    assert all(not r.executed for r in round_.records)


def test_handoff_binding_accepts_the_whole_question_of_a_one_clause_request(kb):
    round_, d2, r2 = crafted_round(kb)
    assert round_.code is HC.HANDOFF_EXECUTED and d2 is not None and r2 is not None and round_.specialist_calls == 1
    assert [t.task_id for t in d2.subtasks] == ["task-1-handoff-0"] and d2.subtasks[0].metadata["handoff"]["validated"] is True
    assert d2.metadata["handoff_round"]["code"] == "HANDOFF_EXECUTED" and [r.task.task_id for r in r2.runs] == ["task-1", "task-1-handoff-0"]
    assert r2.runs[0].findings.handoff_requests == [] and len(r2.handoffs) == 0                      # settled: no longer pending
    assert H.validated_handoff_task(d2.subtasks[0], d2) and not H.validated_handoff_task(d2.subtasks[0], coordinate(OFFERS))
    outcome = A.aggregate(d2, r2, kb.chunks, kb.metadata)
    assert outcome.accepted and outcome.result.clauses[0].task_id == "task-1-handoff-0"


def test_unsettled_handoff_task_is_never_aggregated(kb):
    decision = coordinate(OFFERS)
    fake = SpecialistTask("task-1-handoff-0", OFFERS, specialist_id="teaching", context={"focus_clauses": [OFFERS]}, requested_by="faculty_services",
                          metadata={"handoff": {"source": "faculty_services", "target": "teaching", "depth": 1, "validated": True}})
    decision.subtasks.append(fake)
    decision.selected_specialists.append(SpecialistId.TEACHING)
    assert not H.validated_handoff_task(fake, decision) and A.request_problem(decision)[0] is G.UNSAFE_RESULT and not A.applicable(decision)
    deep = SpecialistTask("x", OFFERS, specialist_id="teaching", context={"focus_clauses": [OFFERS]}, requested_by="faculty_services",
                          metadata={"handoff": {"source": "faculty_services", "target": "teaching", "depth": 2, "validated": True}})
    assert not H.validated_handoff_task(deep, decision)


def test_source_with_an_unsupported_remaining_clause_blocks_the_handoff_without_running_it(kb, Q):
    calls = []

    def research(task):
        calls.append(task.task_id)
        return stub_result(kb, task, answers={SICK_CLAUSE: Q["sick"]})

    teaching = behaviour(kb, answers={})                                                             # clause 1 not found
    hr = behaviour(kb, declines={SICK_CLAUSE: "research"})
    result = AT.attempt(kb, CROSS, teaching=teaching, faculty_services=hr, research=research)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_HANDOFF and result.handoff.code is HC.HANDOFF_REMAINDER_UNSUPPORTED
    assert calls == [] and result.result is None and result.specialist_calls == 2 and "teaching keeps clause(s)" in result.reason
    two = "%s, and %s?" % (LOAD_CLAUSE, SICK_CLAUSE)
    decision = AT.decision_for(two, [LOAD_CLAUSE, SICK_CLAUSE], [("teaching", [LOAD_CLAUSE, SICK_CLAUSE])], complexity="multipart")
    dispatched = dispatch(decision, T.stub_registry(teaching=behaviour(kb, declines={SICK_CLAUSE: "research"})), question=two)   # clause 1 not found, clause 2 declined
    round_, d2, r2 = H.run_handoffs(decision, dispatched, T.stub_registry(research=research), [LOAD_CLAUSE, SICK_CLAUSE])
    assert round_.code is HC.HANDOFF_REMAINDER_UNSUPPORTED and round_.specialist_calls == 0 and calls == [] and d2 is None


def test_handoff_exception_and_model_call_are_rejected_and_redacted(kb, monkeypatch):
    quote, hr, teaching = offers_fixture(kb)

    def boom(task):
        raise RuntimeError(r"boom at C:\Users\someone\private\teaching.faiss gsk_ABCDEFGHIJKLMNOPQRST")

    result = AT.attempt(kb, OFFERS, faculty_services=hr, teaching=boom)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_HANDOFF and result.handoff.code is HC.HANDOFF_EXCEPTION
    assert result.handoff.records[0].detail == "target raised: RuntimeError" and result.specialist_calls == 2
    text = json.dumps(result.to_dict(), default=str)
    assert "someone" not in text and "gsk_ABCDEFGHIJKLMNOPQRST" not in text
    talkative = lambda task: stub_result(kb, task, answers={OFFERS: VT.located(kb, "• Employment offers are drafted by HR", "Security Clearance.")},
                                         requested=("responsible party or unit",), llm_used=True)
    result = AT.attempt(kb, OFFERS, faculty_services=hr, teaching=talkative)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_HANDOFF and result.handoff.code is HC.HANDOFF_UNSAFE_RESULT and result.result is None
    monkeypatch.setattr(H, "check_findings", boom)
    result = AT.attempt(kb, OFFERS, faculty_services=hr, teaching=teaching)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_HANDOFF and result.handoff.code is HC.HANDOFF_EXCEPTION and "someone" not in json.dumps(result.to_dict(), default=str)


def test_handoff_without_a_task_in_the_i1_shape_never_runs_the_target(kb):
    """The I-1 stub: a supported result plus a bare handoff request (no task, no clause). Still never executed."""
    calls = []

    def hr(task):
        calls.append(task.task_id)
        return T._hr_phone_task_stub(kb, FindingStatus.SUPPORTED)(task)

    result = AT.attempt(kb, HR_PHONE, institutional=T._hr_phone_task_stub(kb, FindingStatus.SUPPORTED, handoff=True), faculty_services=hr)
    assert result.outcome is PF.PlanFOutcome.REJECTED_BY_HANDOFF and result.handoff.code is HC.HANDOFF_TARGET_INVALID and calls == []
    assert result.reason.startswith("HANDOFF_TARGET_INVALID: institutional -> faculty_services") and result.result is None and result.specialist_calls == 1


def test_handoff_attempts_are_deterministic(kb):
    quote, hr, teaching = offers_fixture(kb)
    first, second = AT.attempt(kb, OFFERS, faculty_services=hr, teaching=teaching), AT.attempt(kb, OFFERS, faculty_services=hr, teaching=teaching)
    assert AT.strip(first) == AT.strip(second) and first.handoff.to_dict() == second.handoff.to_dict()
    assert "ms" not in first.handoff.to_dict() and all("ms" not in r for r in first.handoff.to_dict()["records"])


# ---------------------------------------------------------------------------
# Pipeline: LEGACY untouched, SHADOW non-invasive, ACTIVE serves a routed or handed-off answer only when everything held
# ---------------------------------------------------------------------------
def test_shadow_records_routing_and_handoff_without_touching_the_answer(kb, monkeypatch):
    quote, hr, teaching = offers_fixture(kb)
    P.set_mode(monkeypatch, pipeline.MODE_SHADOW)
    P.install_runtime(monkeypatch, P.stub_runtime(kb, faculty_services=hr, teaching=teaching))
    result, d = pipeline.ask_with_diagnostics(kb, OFFERS)
    direct = P.legacy_direct(kb, OFFERS)
    assert P.visible(result) == P.visible(direct) and P.trace_of(result) == P.trace_of(direct)
    assert d["served"] == "legacy" and d["accepted"] is True and d["outcome"] == "PLAN_F_ACCEPTED" and d["plan_f_path"] == "multi_finding_aggregate"
    assert d["handoff_attempted"] is True and d["handoff_source"] == ["faculty_services"] and d["handoff_target"] == ["teaching"] and d["handoff_depth"] == 1
    assert d["handoff_visited"] == ["faculty_services", "teaching"] and d["handoff_outcome"] == "HANDOFF_ACCEPTED" and d["handoff_ms"] >= 0.0
    assert d["specialist_calls"] == 2 and d["routing_reason"] == "ROUTE_UNCHANGED" and d["routing_refined"] is False and d["routing_ms"] >= 0.0
    assert OFFERS not in json.dumps(d) and quote not in json.dumps(d)
    cid, cquote = VT.located(kb, "• The Vice Chancellor for Academic Affairs approves travel requests", "interprets the policy.")
    pipeline.reset_plan_f_state()
    P.install_runtime(monkeypatch, P.stub_runtime(kb, research=behaviour(kb, answers={CONFERENCE: (cid, cquote)}, requested=("approving authority",)), institutional=never))
    result, d = pipeline.ask_with_diagnostics(kb, CONFERENCE)
    assert d["served"] == "legacy" and d["routing_refined"] is True and d["routing_reason"] == "ROUTE_REFINED_BY_REQUEST_KIND"
    assert d["routing_initial"] == ["research", "institutional"] and d["routing_final"] == ["research"] and d["handoff_attempted"] is False and d["specialist_calls"] == 1


def test_active_serves_routed_and_handed_off_answers_and_falls_back_on_blocked_handoffs(kb, monkeypatch):
    quote, hr, teaching = offers_fixture(kb)
    P.set_mode(monkeypatch, pipeline.MODE_ACTIVE)
    P.install_runtime(monkeypatch, P.stub_runtime(kb, faculty_services=hr, teaching=teaching))
    result, d = pipeline.ask_with_diagnostics(kb, OFFERS)
    assert d["served"] == "plan_f" and result.answer == quote and result.pages == [61] and d["handoff_outcome"] == "HANDOFF_ACCEPTED"
    pipeline.reset_plan_f_state()
    P.install_runtime(monkeypatch, P.stub_runtime(kb, faculty_services=hr, teaching=lambda task: stub_result(kb, task, declines={OFFERS: "faculty_services"})))
    result, d = pipeline.ask_with_diagnostics(kb, OFFERS)
    direct = P.legacy_direct(kb, OFFERS)
    assert d["served"] == "legacy" and P.visible(result) == P.visible(direct) and d["outcome"] == "PLAN_F_REJECTED_BY_HANDOFF"
    assert d["handoff_outcome"] == "HANDOFF_SECOND_HOP_BLOCKED" and d["fallback_reason"].startswith("HANDOFF_SECOND_HOP_BLOCKED")
    pipeline.reset_plan_f_state()
    P.install_runtime(monkeypatch, P.stub_runtime(kb, institutional=never, faculty_services=never))
    result, d = pipeline.ask_with_diagnostics(kb, HOUSING)
    assert d["served"] == "legacy" and d["outcome"] == "PLAN_F_REJECTED_BY_ROUTING" and d["routing_reason"] == "ROUTE_AMBIGUOUS" and d["specialist_calls"] == 0


def test_legacy_mode_is_untouched_by_routing_and_handoffs(kb):
    assert pipeline.plan_f_mode() == pipeline.MODE_LEGACY
    result, d = pipeline.ask_with_diagnostics(kb, OFFERS)
    assert d == {"mode": "legacy", "served": "legacy", "attempted": False} and result.answer == pipeline._legacy_answer(kb, OFFERS).answer
    source = (REPO / "ui" / "pipeline.py").read_text(encoding="utf-8")
    assert "plan_f_routing" not in source and "plan_f_handoff" not in source
    for name in ("app.py", "ui/chat.py", "handbook_bot/qa.py", "handbook_bot/orchestrator.py", "handbook_bot/agents/router.py", "handbook_bot/retrieval.py",
                 "handbook_bot/agents/synthesis.py", "handbook_bot/agents/verifier.py", "handbook_bot/knowledge_base.py",
                 "handbook_bot/agents/specialists/dispatch.py", "handbook_bot/agents/coordinator.py"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert "plan_f_routing" not in text and "plan_f_handoff" not in text, name
    for module in ("plan_f_routing.py", "plan_f_handoff.py"):
        text = (REPO / "handbook_bot" / module).read_text(encoding="utf-8")
        for forbidden in ("ask_groq", "Groq(", "import random", "import requests", "importlib", "__import__", "from .orchestrator", "from .retrieval"):
            assert forbidden not in text, (module, forbidden)


def test_shared_finding_covers_a_second_scoped_clause_only_on_the_specialists_own_assertion(kb, Q):
    """The frozen specialists attach one finding to several clauses but keep the first clause's covered details on it; the
    second clause's view carries the coverage its supported report asserts (first-listed finding, no uncovered detail)."""
    office_week = "how many office hours per week must faculty hold"
    first_clause = "How many office hours must faculty hold"
    question = "%s, and %s?" % (first_clause, office_week)
    cid, quote = Q["office"]

    def teaching(task):
        focus = task.context["focus_clauses"]
        assert focus == [first_clause, office_week]
        shared = AT.finding(kb, cid, quote, task, number=1, clauses=(0, 1), details_covered=["number or amount", "the asked specifics stated with the answer (office_hours, faculty)"])
        reports = [{"index": 0, "clause": focus[0], "outcome": "supported", "ask_shape": "specific_recognised", "findings": [shared.finding_id],
                    "requested_details": ["number or amount", "the asked specifics stated with the answer (office_hours, faculty)"], "uncovered_details": []},
                   {"index": 1, "clause": focus[1], "outcome": "supported", "ask_shape": "specific_recognised", "findings": [shared.finding_id],
                    "requested_details": ["number or amount", "the asked specifics stated with the answer (office_hours, week, faculty)"], "uncovered_details": []}]
        return SpecialistFindings(specialist_id=task.specialist_id, task_id=task.task_id, status=FindingStatus.SUPPORTED, findings=[shared], summary="stub",
                                  metadata={"clauses": reports, "unprocessed_clauses": []})

    result = AT.attempt(kb, question, teaching=teaching)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED, result.reason
    assert result.result.answer == quote and len(result.result.items) == 1 and "projected_coverage" not in result.result.items[0]["meta"]["plan_f"]
    assert result.aggregation.duplicates == (("task-1-f1", "task-1-f1", 1),) and result.aggregation.evidence[0].clause_indexes == (0, 1)
    assert result.aggregation.clauses[1].covered is True and result.aggregation.clauses[1].verification.accepted      # the projected view answered clause 1
    second = coordinate(question)
    refined, _routing = R.refine(second)
    run = teaching(refined.subtasks[0])
    assert run.findings[0].metadata["details_covered"] == ["number or amount", "the asked specifics stated with the answer (office_hours, faculty)"]   # the original is untouched
    from handbook_bot import plan_f_verification as V
    view = V.clause_view(run, 1)
    assert view.findings[0].metadata["projected_coverage"] is True and "the asked specifics stated with the answer (office_hours, week, faculty)" in view.findings[0].metadata["details_covered"]
    run.metadata["clauses"][1]["uncovered_details"] = ["the asked specifics stated with the answer (office_hours, week, faculty)"]
    assert "projected_coverage" not in V.clause_view(run, 1).findings[0].metadata                  # not asserted by the report: no projection
    run.metadata["clauses"][1]["uncovered_details"] = []
    run.metadata["clauses"][1]["findings"] = ["task-1-f9", shared_id] if (shared_id := run.findings[0].finding_id) else []
    assert "projected_coverage" not in V.clause_view(run, 1).findings[0].metadata                  # not the first-listed finding: no projection
    run.findings[0].metadata["clauses"] = [0]
    run.metadata["clauses"][1]["findings"] = [shared_id]
    with pytest.raises(ValueError):
        V.clause_view(run, 1)                                                                      # not shared with clause 1 at all


def test_real_specialists_resolve_a_category_c_case_on_the_test_corpus(runtime):
    """No stub: the frozen Research specialist alone answers the approval ask the Coordinator shared with Institutional Navigation."""
    result = runtime.attempt(CONFERENCE)
    assert result.routing.reason is RC.ROUTE_REFINED_BY_REQUEST_KIND and result.routing.final_specialists == ("research",)
    assert result.outcome is PF.PlanFOutcome.ACCEPTED and result.path == PF.PATH_PASSTHROUGH and result.result.pages == [117], result.reason
    assert "approves travel requests" in result.result.answer and result.specialist_calls == 1 and result.result.llm_calls == 0
    assert AT.strip(runtime.attempt(CONFERENCE)) == AT.strip(result)
