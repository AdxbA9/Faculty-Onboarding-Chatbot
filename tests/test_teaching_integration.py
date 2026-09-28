"""Faculty Onboarding Coordinator to Teaching & Learning Specialist integration (Step 3.10).

The Coordinator is the real one; the Teaching specialist runs over the synthetic
handbook pages of the specialist test module with the real registry record and
section map; ranking is a raw token-overlap stand-in or a scripted reranker. No
model, no LLM, no network. The dispatch seam is not wired into the runtime.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import faiss
import pytest

from handbook_bot.agents import coordinator
from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists.base import Specialist, check_findings
from handbook_bot.agents.specialists.contracts import (
    ALL_FINDING_STATUSES,
    CoordinatorDecision,
    FindingStatus,
    HandoffRequest,
    SpecialistFindings,
    SpecialistId,
    SpecialistTask,
)
from handbook_bot.agents.specialists.registry import SpecialistRegistry
from handbook_bot.agents.specialists.teaching import APPROVED_SOURCE_ID, TeachingLearningSpecialist, TeachingResources
from handbook_bot.chunking import build_chunks
from handbook_bot.sources import annotate_metadata, load_section_map, load_source_registry

from conftest import FakeEmbedder
from test_teaching_specialist import GRADING_AND_EXAM_PAGES, PAGES, RawOverlapReranker, ScriptedReranker, page

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    source = registry.get_source(APPROVED_SOURCE_ID)
    section_map = load_section_map(str(REPO / "knowledge" / "handbook_sections.json"))
    chunks, metadata = build_chunks(PAGES)
    annotate_metadata(metadata, source, section_map)
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    return SimpleNamespace(chunks=chunks, metadata=metadata, index=index, embedder=embedder, source=source, section_map=section_map)


def teaching_for(corpus, reranker=None, index=None):
    res = TeachingResources(embed=lambda t: corpus.embedder.encode([t]), rerank=(reranker or RawOverlapReranker()).predict,
                            index=index or corpus.index, chunks=corpus.chunks, metadata=corpus.metadata, source=corpus.source,
                            section_map=corpus.section_map)
    return TeachingLearningSpecialist(res)


def registry_with(*specialists):
    registry = SpecialistRegistry()
    for s in specialists:
        registry.register(s)
    return registry


@pytest.fixture(scope="module")
def registry(corpus):
    return registry_with(teaching_for(corpus))


class CountingTeaching(TeachingLearningSpecialist):
    """Records every task it is given, to prove non-Teaching tasks never reach it."""

    def __init__(self, resources):
        super().__init__(resources)
        self.calls = []

    def run(self, task):
        self.calls.append(task)
        return super().run(task)


class RaisingTeaching(Specialist):
    specialist_id = SpecialistId.TEACHING

    def run(self, task):
        raise RuntimeError(r"specialist crashed at C:\Users\PC\secret\index gsk_ABCDEFGHIJKLMNOP")


def assert_routing_preserved(result, question):
    direct = coordinator.coordinate(question)
    assert [s.value for s in result.decision.selected_specialists] == [s.value for s in direct.selected_specialists]
    assert [t.task_id for t in result.decision.subtasks] == [t.task_id for t in direct.subtasks]
    assert [r.specialist_id for r in result.runs] == [t.specialist_id for t in result.decision.subtasks]
    assert [r.task.task_id for r in result.runs] == [t.task_id for t in result.decision.subtasks]
    for run, task in zip(result.runs, result.decision.subtasks):
        assert run.task is task                                        # the Coordinator's own object, unchanged


def assert_teaching_result(result, corpus):
    for run in result.runs:
        if run.state != D.STATE_EXECUTED:
            assert run.findings is None
            continue
        f = run.findings
        assert f.specialist_id is SpecialistId.TEACHING and f.task_id == run.task.task_id
        assert f.status.value in ALL_FINDING_STATUSES and f.llm_used is False
        check_findings(SpecialistId.TEACHING, run.task, f)
        for finding in f.findings:
            chunk, meta = corpus.chunks[finding.metadata["chunk_id"]], corpus.metadata[finding.metadata["chunk_id"]]
            assert finding.evidence_quote in chunk and finding.claim == finding.evidence_quote
            assert finding.source_id == meta["source_id"] == APPROVED_SOURCE_ID and finding.page == meta["page"]
        for h in f.handoff_requests:
            assert isinstance(h, HandoffRequest) and h.from_specialist is SpecialistId.TEACHING
        assert f.to_dict()["missing"] == list(f.missing)


# ---------------------------------------------------------------------------
# A, K: Teaching-only, supported
# ---------------------------------------------------------------------------
def test_teaching_only_question_is_executed_and_supported(registry, corpus):
    q = "What is the teaching load for Regular Faculty?"
    result = D.run_question(q, registry)
    assert_routing_preserved(result, q)
    assert result.executed_specialists == [SpecialistId.TEACHING] and result.pending_specialists == []
    f = result.results[0]
    assert f.status is FindingStatus.SUPPORTED and f.missing == [] and f.confidence == 1.0
    assert any('"A" Regular Faculty' in x.evidence_quote and x.page == 96 for x in f.findings)
    assert f.findings[0].metadata["details_covered"] == ["the requested faculty category (regular faculty)"]
    assert result.llm_calls == 0 and result.question == q
    assert_teaching_result(result, corpus)
    assert result.to_dict()["executed_specialists"] == ["teaching"]


def test_run_question_equals_coordinate_then_dispatch(registry):
    q = "What is the teaching load for Regular Faculty?"
    a = D.run_question(q, registry).to_dict()
    b = D.dispatch(coordinator.coordinate(q), registry, question=q).to_dict()
    for d in (a, b):
        for run in d["runs"]:
            if run["findings"]:
                run["findings"].pop("ms", None)
    assert a == b


# ---------------------------------------------------------------------------
# B, C: focus clauses and multi-clause tasks
# ---------------------------------------------------------------------------
def test_focus_clauses_are_passed_untouched(registry, corpus):
    q = "How many office hours must faculty hold and how do I apply for annual leave?"
    result = D.run_question(q, registry)
    assert_routing_preserved(result, q)
    teaching_run = next(r for r in result.runs if r.specialist_id is SpecialistId.TEACHING)
    assert teaching_run.task.question == q
    assert teaching_run.task.context["focus_clauses"] == ["How many office hours must faculty hold"]
    report = teaching_run.findings.metadata["clauses"]
    assert [c["clause"] for c in report] == ["How many office hours must faculty hold"]
    assert teaching_run.findings.handoff_requests == []                 # the leave clause was never seen by Teaching
    assert_teaching_result(result, corpus)


def test_multi_clause_teaching_task_reports_each_clause(registry, corpus):
    q = "What is my teaching load and when do classes begin?"
    result = D.run_question(q, registry)
    assert_routing_preserved(result, q)
    assert result.executed_specialists == [SpecialistId.TEACHING]
    f = result.results[0]
    assert f.status in (FindingStatus.SUPPORTED, FindingStatus.PARTIAL)
    assert f.metadata["clauses"] and all(c["outcome"] in ("supported", "partial", "not_found") for c in f.metadata["clauses"])
    assert_teaching_result(result, corpus)


# ---------------------------------------------------------------------------
# D, E, F: mixed-domain questions keep the other task pending
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,other", [
    ("Can I reduce my teaching load if I have a funded research project, and how do I apply for a research grant?", SpecialistId.RESEARCH),
    ("How many office hours must faculty hold and how do I apply for annual leave?", SpecialistId.FACULTY_SERVICES),
    ("Where is the Registrar's office and what is my teaching load?", SpecialistId.INSTITUTIONAL),
])
def test_mixed_domain_question_executes_teaching_and_keeps_the_other_pending(registry, corpus, question, other):
    result = D.run_question(question, registry)
    assert_routing_preserved(result, question)
    assert set(result.decision.selected_specialists) == {SpecialistId.TEACHING, other}
    assert result.executed_specialists == [SpecialistId.TEACHING] and result.pending_specialists == [other]
    pending = next(r for r in result.runs if r.specialist_id is other)
    assert pending.state == D.STATE_PENDING and pending.findings is None and "not executable in this phase" in pending.reason
    assert pending.task in result.decision.subtasks                      # the Coordinator's full decision stays visible
    teaching = next(r for r in result.runs if r.specialist_id is SpecialistId.TEACHING)
    assert teaching.findings.status is not FindingStatus.ERROR
    assert_teaching_result(result, corpus)


def test_other_domain_clause_is_never_answered_by_teaching(corpus):
    spy = CountingTeaching(teaching_for(corpus).resources)
    registry = registry_with(spy)
    q = "How many office hours must faculty hold and how do I apply for annual leave?"
    result = D.run_question(q, registry)
    assert [t.task_id for t in spy.calls] == [r.task.task_id for r in result.runs if r.specialist_id is SpecialistId.TEACHING]
    assert all("leave" not in c for t in spy.calls for c in t.context["focus_clauses"])
    assert all(x.page != 107 for f in result.results for x in f.findings)


# ---------------------------------------------------------------------------
# G: non-Teaching question
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", ["How do I apply for annual leave?", "What is the parking fee?", "Hello"])
def test_non_teaching_question_never_executes_teaching(corpus, question):
    spy = CountingTeaching(teaching_for(corpus).resources)
    result = D.run_question(question, registry_with(spy))
    assert_routing_preserved(result, question)
    assert result.executed_specialists == [] and result.results == [] and spy.calls == []
    assert SpecialistId.TEACHING not in result.decision.selected_specialists
    assert result.pending_specialists == [t.specialist_id for t in result.decision.subtasks]


# ---------------------------------------------------------------------------
# H, I, J: procedure, not_found, partial semantics survive integration
# ---------------------------------------------------------------------------
def test_procedure_question_is_never_upgraded(registry, corpus):
    q = "Where exactly do I click in Blackboard to create an assignment?"
    result = D.run_question(q, registry)
    assert_routing_preserved(result, q)
    f = result.results[0]
    assert f.status in (FindingStatus.NOT_FOUND, FindingStatus.PARTIAL)
    assert any("blackboard" in m for m in f.missing)
    assert all(not any(w in x.evidence_quote.lower() for w in ("click", "menu", "button", "screen")) for x in f.findings)
    assert_teaching_result(result, corpus)


def test_not_found_question(registry, corpus):
    q = "What is the fee for a make-up exam?"
    result = D.run_question(q, registry)
    f = result.results[0]
    assert f.status is FindingStatus.NOT_FOUND and f.findings == [] and f.missing
    assert_teaching_result(result, corpus)


def test_partial_question_keeps_missing(registry, corpus):
    q = "How many office hours must part-time faculty hold?"
    result = D.run_question(q, registry)
    f = result.results[0]
    assert f.status is FindingStatus.PARTIAL and f.confidence == 0.5
    assert any("part-time" in m.lower() or "number or amount" in m for m in f.missing)
    assert_teaching_result(result, corpus)


# ---------------------------------------------------------------------------
# L: errors are isolated
# ---------------------------------------------------------------------------
def test_teaching_execution_error_is_isolated_and_redacted(corpus):
    class Broken:
        ntotal = corpus.index.ntotal

        def search(self, *a, **k):
            raise RuntimeError(r"index failure at C:\Users\PC\secret\path gsk_ABCDEFGHIJKLMNOP")

    q = "How many office hours must faculty hold and how do I apply for annual leave?"
    result = D.run_question(q, registry_with(teaching_for(corpus, index=Broken())))
    assert_routing_preserved(result, q)
    f = result.results[0]
    assert f.status is FindingStatus.ERROR and f.findings == []
    assert "Users" not in f.metadata["error"] and "gsk_ABC" not in f.metadata["error"]
    assert result.pending_specialists == [SpecialistId.FACULTY_SERVICES]


def test_specialist_that_raises_becomes_an_error_result(corpus):
    q = "How many office hours must faculty hold and how do I apply for annual leave?"
    result = D.run_question(q, registry_with(RaisingTeaching()))
    f = result.results[0]
    assert f.status is FindingStatus.ERROR and f.task_id == "task-1" and f.specialist_id is SpecialistId.TEACHING
    assert f.metadata["error"].startswith("RuntimeError:") and "<path>" in f.metadata["error"] and "gsk_***" in f.metadata["error"]
    assert "Users" not in f.metadata["error"] and "gsk_ABC" not in f.metadata["error"]
    assert result.pending_specialists == [SpecialistId.FACULTY_SERVICES] and result.decision.subtasks[1].specialist_id is SpecialistId.FACULTY_SERVICES
    assert D.redact_error(RuntimeError("cannot open /home/user/private/file")) == "RuntimeError: cannot open <path>"


# ---------------------------------------------------------------------------
# M: handoffs are preserved, never executed
# ---------------------------------------------------------------------------
def test_handoff_is_preserved_but_not_executed(corpus):
    spy = CountingTeaching(teaching_for(corpus).resources)
    registry = registry_with(spy)
    q = "Can my teaching load be reduced if I have a funded research project?"
    result = D.run_question(q, registry)
    assert_routing_preserved(result, q)
    teaching = next(r for r in result.runs if r.specialist_id is SpecialistId.TEACHING)
    assert teaching.findings.requested_agents == [SpecialistId.RESEARCH]
    assert [h.requested_specialist for h in result.handoffs] == [SpecialistId.RESEARCH]
    assert result.handoffs[0].task.requested_by == "teaching"
    assert len(spy.calls) == 1                                          # the handoff task was not run by anyone
    assert result.pending_specialists == [SpecialistId.RESEARCH]         # the Coordinator's own research task stays pending
    assert [r.state for r in result.runs] == [D.STATE_PENDING, D.STATE_EXECUTED]   # Coordinator order: research first


# ---------------------------------------------------------------------------
# N, O: limits belong to the Coordinator and the specialist
# ---------------------------------------------------------------------------
def test_coordinator_specialist_limit_is_respected_not_reinterpreted(registry):
    q = "What is my teaching load, how do I apply for a research grant, and when is my salary paid?"
    full = D.run_question(q, registry)
    assert [s.value for s in full.decision.selected_specialists] == ["teaching", "research", "faculty_services"]
    assert full.executed_specialists == [SpecialistId.TEACHING] and full.pending_specialists == [SpecialistId.RESEARCH, SpecialistId.FACULTY_SERVICES]
    limited = D.run_question(q, registry, max_specialists=1)
    assert [s.value for s in limited.decision.selected_specialists] == ["teaching"] and len(limited.runs) == 1
    assert limited.decision.metadata["limits"]["specialists"] == 1
    assert len(full.runs) == len(full.decision.subtasks)                 # never more tasks than the Coordinator produced


def test_teaching_subtask_limit_is_reported_not_silenced(registry):
    clauses = ["What is my teaching load", "when do classes begin", "what are my office hour obligations", "what is the grading system", "what does the LMS policy cover"]
    task = SpecialistTask(task_id="task-1", question=" and ".join(clauses), specialist_id="teaching", domain="teaching", intent="policy",
                          context={"full_question": " and ".join(clauses), "focus_clauses": clauses}, requested_by="coordinator")
    decision = CoordinatorDecision(selected_specialists=[SpecialistId.TEACHING], subtasks=[task], reason="test")
    result = D.dispatch(decision, registry, question=task.question)
    f = result.results[0]
    assert f.metadata["unprocessed_clauses"] == clauses[3:] and f.status is FindingStatus.PARTIAL
    assert sum("not processed" in m for m in f.missing) == 2


# ---------------------------------------------------------------------------
# Pending specialists, ordering, task identity
# ---------------------------------------------------------------------------
def test_unregistered_but_executable_specialist_is_pending_not_a_failure(corpus):
    q = "What is the teaching load for Regular Faculty?"
    result = D.run_question(q, SpecialistRegistry())
    assert result.executed_specialists == [] and result.pending_specialists == [SpecialistId.TEACHING]
    assert "not registered" in result.runs[0].reason


def test_no_default_to_teaching_for_other_specialists(registry):
    task = SpecialistTask(task_id="task-1", question="How do I apply for annual leave?", specialist_id="faculty_services",
                          domain="faculty_services", context={"focus_clauses": ["How do I apply for annual leave?"]}, requested_by="coordinator")
    decision = CoordinatorDecision(selected_specialists=[SpecialistId.FACULTY_SERVICES], subtasks=[task], reason="test")
    result = D.dispatch(decision, registry)
    assert result.executed_specialists == [] and result.pending_specialists == [SpecialistId.FACULTY_SERVICES]
    assert result.result_for("task-1") is None


def test_task_identity_and_order_are_preserved(registry):
    q = "Can I reduce my teaching load if I have a funded research project, and how do I apply for a research grant?"
    result = D.run_question(q, registry)
    ids = [t.task_id for t in result.decision.subtasks]
    assert [r.task.task_id for r in result.runs] == ids == ["task-1", "task-2"]
    assert result.decision.subtasks[0].specialist_id is SpecialistId.RESEARCH and result.runs[0].state == D.STATE_PENDING
    assert result.result_for("task-2").task_id == "task-2" and result.result_for("task-2").specialist_id is SpecialistId.TEACHING


def test_dispatch_rejects_wrong_inputs(registry):
    with pytest.raises(TypeError):
        D.dispatch("not a decision", registry)
    with pytest.raises(TypeError):
        D.dispatch(coordinator.coordinate("What is my teaching load?"), object())


# ---------------------------------------------------------------------------
# P: determinism; production safety
# ---------------------------------------------------------------------------
def test_repeated_integration_runs_are_identical(registry):
    for q in ("What is the teaching load for Regular Faculty?", "How many office hours must faculty hold and how do I apply for annual leave?",
              "Can my teaching load be reduced if I have a funded research project?", "Where exactly do I click in Blackboard to create an assignment?"):
        outs = []
        for _ in range(5):
            d = copy.deepcopy(D.run_question(q, registry).to_dict())
            for run in d["runs"]:
                if run["findings"]:
                    run["findings"].pop("ms", None)
            outs.append(d)
        assert all(o == outs[0] for o in outs), q


def test_architecture_stays_inactive_and_free_of_llm_calls():
    from handbook_bot import config

    assert config.PLAN_F_ENABLED is False and config.MAX_LLM_CALLS == 2
    for name in ("orchestrator.py", "qa.py", "retrieval.py", "agents/router.py", "agents/synthesis.py", "agents/verifier.py"):
        text = (REPO / "handbook_bot" / name).read_text(encoding="utf-8")
        assert "dispatch" not in text and "specialists" not in text
    source = (REPO / "handbook_bot" / "agents" / "specialists" / "dispatch.py").read_text(encoding="utf-8")
    assert "groq" not in source.lower() and "openai" not in source.lower()
    assert "from ..synthesis" not in source and "from ..verifier" not in source and "llm" not in source.lower().replace("llm_calls", "").replace("llm_used", "").replace("no llm", "")


# ---------------------------------------------------------------------------
# Step 3.12A: compound questions, topic anchors, hyphens, value semantics, defensive dispatch
# ---------------------------------------------------------------------------
def corpus_from(pages):
    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    source = registry.get_source(APPROVED_SOURCE_ID)
    section_map = load_section_map(str(REPO / "knowledge" / "handbook_sections.json"))
    chunks, metadata = build_chunks(pages)
    annotate_metadata(metadata, source, section_map)
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    return SimpleNamespace(chunks=chunks, metadata=metadata, index=index, embedder=embedder, source=source, section_map=section_map)


def test_who_manages_system_support_keeps_institutional_pending_and_is_not_absorbed(corpus):
    q = "Who manages Blackboard support and what is my teaching load?"
    spy = CountingTeaching(teaching_for(corpus).resources)
    result = D.run_question(q, registry_with(spy))
    assert_routing_preserved(result, q)
    assert set(result.decision.selected_specialists) == {SpecialistId.TEACHING, SpecialistId.INSTITUTIONAL}
    assert result.pending_specialists == [SpecialistId.INSTITUTIONAL] and result.executed_specialists == [SpecialistId.TEACHING]
    institutional = next(t for t in result.decision.subtasks if t.specialist_id is SpecialistId.INSTITUTIONAL)
    assert institutional.context["focus_clauses"] == ["Who manages Blackboard support"]
    f = result.results[0]
    outcomes = [c["outcome"] for c in f.metadata["clauses"]]
    assert outcomes[0] in ("partial", "not_found") and outcomes[1] == "supported"        # the support half is never absorbed
    assert f.status is FindingStatus.PARTIAL
    assert [h.requested_specialist for h in result.handoffs] == [SpecialistId.INSTITUTIONAL]
    assert any("teaching load" in x.evidence_quote.lower() for x in f.findings)
    assert len(spy.calls) == 1
    assert_teaching_result(result, corpus)


def test_where_is_a_unit_and_a_grading_question_split_cleanly(corpus):
    q = "Where is the Registrar and what is the grading policy?"
    spy = CountingTeaching(teaching_for(corpus).resources)
    result = D.run_question(q, registry_with(spy))
    assert_routing_preserved(result, q)
    assert [s.value for s in result.decision.selected_specialists] == ["institutional", "teaching"]
    assert result.pending_specialists == [SpecialistId.INSTITUTIONAL]
    assert spy.calls[0].context["focus_clauses"] == ["what is the grading policy"]
    f = result.results[0]
    assert f.handoff_requests == [] and f.status is not FindingStatus.OUT_OF_SCOPE
    assert all("Registrar" not in x.evidence_quote for x in f.findings)
    assert_teaching_result(result, corpus)


@pytest.mark.parametrize("question,must_start", [
    ("What is the grading policy?", ("12.14 Grading System", "Instructors may establish grading policies")),
    ("What are the exam rules?", ("A. Breach of Exam Rules",)),
])
def test_grading_and_exam_questions_are_supported_through_the_integrated_path(question, must_start):
    corpus = corpus_from(GRADING_AND_EXAM_PAGES)
    result = D.run_question(question, registry_with(teaching_for(corpus, ScriptedReranker([], default=9.0))))
    assert_routing_preserved(result, question)
    f = result.results[0]
    assert f.status is FindingStatus.SUPPORTED, (f.status, f.missing)
    assert any(x.evidence_quote.startswith(must_start) for x in f.findings)
    assert all("travel rules" not in x.evidence_quote for x in f.findings)
    assert_teaching_result(result, corpus)


@pytest.mark.parametrize("question,must_contain", [
    ("What is the e-learning policy?", "Learning Management System"),
    ("What is the add/drop deadline?", "Add/Drop"),
])
def test_e_learning_and_add_drop_are_routed_and_answered(registry, corpus, question, must_contain):
    result = D.run_question(question, registry)
    assert_routing_preserved(result, question)
    assert result.executed_specialists == [SpecialistId.TEACHING]
    f = result.results[0]
    assert f.status is FindingStatus.SUPPORTED, (f.status, f.missing)
    assert any(must_contain in x.evidence_quote for x in f.findings)
    assert_teaching_result(result, corpus)


@pytest.mark.parametrize("question", ["What are the office-hours requirements?", "What are the office hours requirements?"])
def test_hyphenated_teaching_phrase_is_routed_and_owned(registry, question):
    result = D.run_question(question, registry)
    assert result.executed_specialists == [SpecialistId.TEACHING]
    f = result.results[0]
    report = f.metadata["clauses"][0]
    assert f.status is not FindingStatus.OUT_OF_SCOPE and report["outcome"] != "unowned"
    assert "an explicit requirement statement about the subject" in report["requested_details"]
    assert result.decision.subtasks[0].question == question


@pytest.mark.parametrize("question,fixture,label", [
    ("What is the minimum number of office hours per week?", page(79, "3.4 Responsibility to Office Hours", "The minimum is described in section 3.4 for office hours."), "minimum"),
    ("What is the maximum class size?", page(229, "12.18 Class Size Policy: Class sizes reached a maximum in 2024 for every course."), "maximum or limit"),
    ("What is the fee for re-marking an exam?", page(224, "12.12 Examinations Policy:", "A re-marking fee applies to examination 101 for every course."), "fee or cost"),
])
def test_bare_numbers_stay_partial_through_the_integrated_path(question, fixture, label):
    corpus = corpus_from([fixture])
    result = D.run_question(question, registry_with(teaching_for(corpus, ScriptedReranker([], default=9.0))))
    f = result.results[0]
    assert f.status is FindingStatus.PARTIAL and any(m.startswith(label + " not found") for m in f.missing), (f.status, f.missing)
    assert_teaching_result(result, corpus)


def test_dispatch_rejects_a_decision_altered_after_validation(registry):
    q = "What is my teaching load, how do I apply for a research grant, and when is my salary paid?"
    decision = coordinator.coordinate(q)
    altered = copy.deepcopy(decision)
    altered.subtasks[1].task_id = altered.subtasks[0].task_id                # a repeated id
    with pytest.raises(ValueError):
        D.dispatch(altered, registry, question=q)
    altered = copy.deepcopy(decision)
    altered.subtasks.append("not a task")                                     # a non-task
    with pytest.raises(TypeError):
        D.dispatch(altered, registry, question=q)
    assert D.dispatch(decision, registry, question=q).to_dict()["executed_specialists"] == ["teaching"]   # the normal path is unchanged
