"""Faculty Onboarding Coordinator to Research & Innovation Specialist integration (Milestone 4, Step 4.4).

The Coordinator is the real one; each specialist runs on its own injected resources over synthetic handbook pages
with the real registry record and section map (Teaching on the Teaching fixture pages, Research on the Research
fixture pages), which is how dependency injection works in production too: the dispatcher never sees resources.
Ranking is a raw token-overlap stand-in or a scripted reranker. No model, no LLM, no network. The dispatch seam is
not wired into the runtime. Handoffs are preserved, never executed.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from handbook_bot.agents import coordinator
from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists.base import Specialist, check_findings
from handbook_bot.agents.specialists.contracts import (
    ALL_FINDING_STATUSES,
    CoordinatorDecision,
    FindingStatus,
    HandoffRequest,
    SpecialistId,
    SpecialistTask,
)
from handbook_bot.agents.specialists.registry import SpecialistRegistry
from handbook_bot.agents.specialists.research import APPROVED_SOURCE_ID, ResearchInnovationSpecialist
from handbook_bot.sources import load_section_map, load_source_registry

import test_research_specialist as RT
import test_teaching_integration as TI

REPO = Path(__file__).resolve().parents[1]
TEACHING_AND_RESEARCH = (SpecialistId.TEACHING, SpecialistId.RESEARCH)


@pytest.fixture(scope="module")
def registry_and_map():
    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    return registry.get_source(APPROVED_SOURCE_ID), load_section_map(str(REPO / "knowledge" / "handbook_sections.json"))


@pytest.fixture(scope="module")
def research_corpus(registry_and_map):
    return RT.corpus_from(registry_and_map, RT.RESEARCH_PAGES)


@pytest.fixture(scope="module")
def teaching_corpus():
    return _teaching_corpus()


def _teaching_corpus():
    import faiss
    from types import SimpleNamespace

    from handbook_bot.chunking import build_chunks
    from handbook_bot.sources import annotate_metadata
    from conftest import FakeEmbedder

    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    source = registry.get_source(APPROVED_SOURCE_ID)
    section_map = load_section_map(str(REPO / "knowledge" / "handbook_sections.json"))
    chunks, metadata = build_chunks(TI.PAGES)
    annotate_metadata(metadata, source, section_map)
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    return SimpleNamespace(chunks=chunks, metadata=metadata, index=index, embedder=embedder, source=source, section_map=section_map)


class CountingResearch(ResearchInnovationSpecialist):
    """Records every task it is given."""

    def __init__(self, resources):
        super().__init__(resources)
        self.calls = []

    def run(self, task):
        self.calls.append(task)
        return super().run(task)


class RaisingResearch(Specialist):
    specialist_id = SpecialistId.RESEARCH

    def run(self, task):
        raise RuntimeError(r"research index missing at C:\Users\PC\secret\research.faiss gsk_ABCDEFGHIJKLMNOP")


def research_for(corpus, reranker=None):
    return RT.make_specialist(corpus, reranker)


def build_registry(*specialists):
    registry = SpecialistRegistry()
    for s in specialists:
        registry.register(s)
    return registry


@pytest.fixture(scope="module")
def teaching(teaching_corpus):
    return TI.teaching_for(teaching_corpus)


@pytest.fixture(scope="module")
def research(research_corpus):
    return research_for(research_corpus)


@pytest.fixture(scope="module")
def registry(teaching, research):
    return build_registry(teaching, research)


def assert_routing_preserved(result, question):
    direct = coordinator.coordinate(question)
    assert result.decision.to_dict() == direct.to_dict()
    assert [r.task.task_id for r in result.runs] == [t.task_id for t in result.decision.subtasks]
    assert [r.specialist_id for r in result.runs] == [t.specialist_id for t in result.decision.subtasks]
    for run, task in zip(result.runs, result.decision.subtasks):
        assert run.task is task                                        # the Coordinator's own object, unchanged


def assert_results_valid(result, corpora):
    for run in result.runs:
        if run.state != D.STATE_EXECUTED:
            assert run.findings is None and run.reason
            continue
        f = run.findings
        assert f.specialist_id is run.specialist_id and f.task_id == run.task.task_id
        assert f.status.value in ALL_FINDING_STATUSES and f.llm_used is False
        check_findings(run.specialist_id, run.task, f)
        corpus = corpora[run.specialist_id]
        for finding in f.findings:
            chunk, meta = corpus.chunks[finding.metadata["chunk_id"]], corpus.metadata[finding.metadata["chunk_id"]]
            assert finding.evidence_quote in chunk and finding.claim == finding.evidence_quote
            assert finding.source_id == meta["source_id"] == APPROVED_SOURCE_ID and finding.page == meta["page"]
            assert "page_section_nos" in finding.metadata and "boundary_guard" in finding.metadata
        for h in f.handoff_requests:
            assert isinstance(h, HandoffRequest) and h.from_specialist is run.specialist_id


def run_question(question, registry, **kw):
    return D.run_question(question, registry, executable=kw.pop("executable", TEACHING_AND_RESEARCH), **kw)


@pytest.fixture(scope="module")
def corpora(teaching_corpus, research_corpus):
    return {SpecialistId.TEACHING: teaching_corpus, SpecialistId.RESEARCH: research_corpus}


# ---------------------------------------------------------------------------
# Research only, supported
# ---------------------------------------------------------------------------
def test_research_only_question_is_executed_and_supported(registry, corpora):
    q = "What percentage of licensing revenue goes to the inventor?"
    result = run_question(q, registry)
    assert_routing_preserved(result, q)
    assert result.executed_specialists == [SpecialistId.RESEARCH] and result.pending_specialists == []
    f = result.results[0]
    assert f.status is FindingStatus.SUPPORTED and f.missing == [] and f.confidence == 1.0
    assert any("50% to the inventor(s)" in x.evidence_quote for x in f.findings)
    assert result.llm_calls == 0 and result.question == q
    assert_results_valid(result, corpora)


def test_run_question_equals_coordinate_then_dispatch(registry):
    q = "Who approves conference travel requests?"
    a = run_question(q, registry).to_dict()
    b = D.dispatch(coordinator.coordinate(q), registry, executable=TEACHING_AND_RESEARCH, question=q).to_dict()
    for d in (a, b):
        for run in d["runs"]:
            if run["findings"]:
                run["findings"].pop("ms", None)
    assert a == b


# ---------------------------------------------------------------------------
# Research with the other specialists
# ---------------------------------------------------------------------------
def test_research_and_teaching_both_execute_and_research_hands_the_load_part_to_teaching(registry, corpora):
    q = "Can research reduce my teaching load?"
    result = run_question(q, registry)
    assert_routing_preserved(result, q)
    assert result.executed_specialists == [SpecialistId.TEACHING, SpecialistId.RESEARCH] and result.pending_specialists == []
    research_result = result.result_for(next(t.task_id for t in result.decision.subtasks if t.specialist_id is SpecialistId.RESEARCH))
    assert SpecialistId.TEACHING in research_result.requested_agents
    assert all("credit hours" not in x.evidence_quote for x in research_result.findings)     # WLAM rows are Teaching's evidence
    assert_results_valid(result, corpora)


@pytest.mark.parametrize("question,pending", [
    ("Can I take a sabbatical for research?", SpecialistId.FACULTY_SERVICES),
    ("Who should I contact about research ethics?", SpecialistId.INSTITUTIONAL),
])
def test_research_executes_and_the_other_specialist_stays_pending(registry, corpora, question, pending):
    result = run_question(question, registry)
    assert_routing_preserved(result, question)
    assert result.executed_specialists == [SpecialistId.RESEARCH] and result.pending_specialists == [pending]
    run = next(r for r in result.runs if r.specialist_id is pending)
    assert run.state == D.STATE_PENDING and run.findings is None and "not executable in this phase" in run.reason
    assert run.task in result.decision.subtasks
    assert_results_valid(result, corpora)


def test_three_domain_question_keeps_hr_pending_and_runs_teaching_and_research(registry, corpora):
    q = "What is my teaching load, how do I apply for a research grant, and when is my salary paid?"
    result = run_question(q, registry)
    assert_routing_preserved(result, q)
    assert result.executed_specialists == [SpecialistId.TEACHING, SpecialistId.RESEARCH]
    assert result.pending_specialists == [SpecialistId.FACULTY_SERVICES]
    research_task = next(t for t in result.decision.subtasks if t.specialist_id is SpecialistId.RESEARCH)
    assert research_task.context["focus_clauses"] == ["how do I apply for a research grant"]
    research_result = result.result_for(research_task.task_id)
    assert research_result.status is FindingStatus.PARTIAL                      # policy stages only, never a procedure
    assert [c["clause"] for c in research_result.metadata["clauses"]] == ["how do I apply for a research grant"]
    assert_results_valid(result, corpora)


# ---------------------------------------------------------------------------
# Status semantics through dispatch
# ---------------------------------------------------------------------------
def test_partial_not_found_and_out_of_scope_survive_dispatch(research_corpus, teaching):
    silent = build_registry(teaching, research_for(research_corpus, RT.ScriptedReranker([], default=-5.0)))
    not_found = run_question("What are the research priority areas?", silent)
    assert not_found.results[0].status is FindingStatus.NOT_FOUND and not_found.results[0].findings == []
    decision = CoordinatorDecision(
        selected_specialists=[SpecialistId.RESEARCH],
        subtasks=[SpecialistTask(task_id="task-1", question="What is the teaching load for research-intensive faculty?", specialist_id="research",
                                 domain="research", context={"focus_clauses": ["What is the teaching load for research-intensive faculty?"]},
                                 requested_by="coordinator")],
        reason="a misrouted WLAM clause")
    foreign = D.dispatch(decision, build_registry(research_for(research_corpus)), executable=TEACHING_AND_RESEARCH)
    f = foreign.results[0]
    assert f.status is FindingStatus.OUT_OF_SCOPE and f.requested_agents == [SpecialistId.TEACHING]
    assert [h.requested_specialist for h in foreign.handoffs] == [SpecialistId.TEACHING] and foreign.executed_specialists == [SpecialistId.RESEARCH]


def test_dispatch_never_changes_the_research_status(registry, research):
    for q in ("What percentage of licensing revenue goes to the inventor?", "How do I apply for a research grant?", "Who funds research externally?"):
        result = run_question(q, registry)
        run = next(r for r in result.runs if r.specialist_id is SpecialistId.RESEARCH)
        direct = research.run(run.task)
        assert direct.status == run.findings.status and [x.evidence_quote for x in direct.findings] == [x.evidence_quote for x in run.findings.findings]
        assert direct.missing == run.findings.missing


def test_research_execution_error_is_isolated_and_redacted(teaching, corpora):
    q = "What is my teaching load, how do I apply for a research grant, and when is my salary paid?"
    result = run_question(q, build_registry(teaching, RaisingResearch()))
    assert_routing_preserved(result, q)
    research_run = next(r for r in result.runs if r.specialist_id is SpecialistId.RESEARCH)
    f = research_run.findings
    assert research_run.state == D.STATE_EXECUTED and f.status is FindingStatus.ERROR and f.findings == []
    assert "<path>" in f.metadata["error"] and "gsk_***" in f.metadata["error"] and "Users" not in f.metadata["error"]
    teaching_run = next(r for r in result.runs if r.specialist_id is SpecialistId.TEACHING)
    assert teaching_run.findings.status is not FindingStatus.ERROR                 # the other task is unaffected
    assert result.pending_specialists == [SpecialistId.FACULTY_SERVICES]


# ---------------------------------------------------------------------------
# Handoffs are preserved, never executed
# ---------------------------------------------------------------------------
def test_research_handoffs_are_preserved_but_not_executed(teaching_corpus, research_corpus):
    teaching_spy = TI.CountingTeaching(TI.teaching_for(teaching_corpus).resources)
    research_spy = CountingResearch(research_for(research_corpus).resources)
    registry = build_registry(teaching_spy, research_spy)
    q = "Can I take a sabbatical for research?"
    result = run_question(q, registry)
    research_result = result.results[0]
    assert research_result.requested_agents == [SpecialistId.FACULTY_SERVICES]
    assert [h.requested_specialist for h in result.handoffs] == [SpecialistId.FACULTY_SERVICES]
    handoff = result.handoffs[0]
    assert handoff.task.requested_by == "research" and handoff.task.context["focus_clauses"] == research_spy.calls[0].context["focus_clauses"]
    assert len(research_spy.calls) == 1 and teaching_spy.calls == []                 # no second round, no handoff execution
    assert result.pending_specialists == [SpecialistId.FACULTY_SERVICES]
    assert result.to_dict()["handoffs"][0]["requested_specialist"] == "faculty_services"


def test_research_never_receives_another_specialists_clause(teaching_corpus, research_corpus):
    research_spy = CountingResearch(research_for(research_corpus).resources)
    registry = build_registry(TI.teaching_for(teaching_corpus), research_spy)
    q = "What is my teaching load, how do I apply for a research grant, and when is my salary paid?"
    run_question(q, registry)
    assert len(research_spy.calls) == 1
    assert research_spy.calls[0].context["focus_clauses"] == ["how do I apply for a research grant"]
    assert research_spy.calls[0].question == q                                      # the full question is carried, not rescanned


@pytest.mark.parametrize("question", ["What is the teaching load for research-intensive faculty?", "How many days of annual leave do I get?",
                                      "Where is the HR department?", "Hello", "What is the parking fee?"])
def test_research_is_never_executed_without_a_research_task(teaching_corpus, research_corpus, question):
    research_spy = CountingResearch(research_for(research_corpus).resources)
    result = run_question(question, build_registry(TI.teaching_for(teaching_corpus), research_spy))
    assert_routing_preserved(result, question)
    assert SpecialistId.RESEARCH not in result.decision.selected_specialists and research_spy.calls == []


# ---------------------------------------------------------------------------
# Registration, executable set, identity, determinism, safety
# ---------------------------------------------------------------------------
def test_research_is_executable_in_this_phase_and_institutional_is_not():
    # Step 4.11 added Faculty Services & HR to the executable set; Institutional Navigation stays pending
    assert D.EXECUTABLE_SPECIALISTS == (SpecialistId.TEACHING, SpecialistId.RESEARCH, SpecialistId.FACULTY_SERVICES)
    assert SpecialistId.INSTITUTIONAL not in D.EXECUTABLE_SPECIALISTS


def test_default_dispatch_runs_a_registered_research_specialist(registry):
    result = D.run_question("What percentage of licensing revenue goes to the inventor?", registry)
    assert result.executed_specialists == [SpecialistId.RESEARCH]


def test_unregistered_research_is_pending_not_a_failure(teaching):
    result = run_question("What percentage of licensing revenue goes to the inventor?", build_registry(teaching))
    assert result.executed_specialists == [] and result.pending_specialists == [SpecialistId.RESEARCH]
    assert "executable but not registered" in result.runs[0].reason and result.results == []


def test_registered_research_stays_pending_when_the_phase_does_not_execute_it(registry):
    result = run_question("What percentage of licensing revenue goes to the inventor?", registry, executable=(SpecialistId.TEACHING,))
    assert result.pending_specialists == [SpecialistId.RESEARCH] and "not executable in this phase" in result.runs[0].reason


def test_task_identity_order_and_focus_clauses_are_preserved(registry):
    q = "Can I reduce my teaching load if I have a funded research project, and how do I apply for a research grant?"
    result = run_question(q, registry)
    assert_routing_preserved(result, q)
    ids = [t.task_id for t in result.decision.subtasks]
    assert [r.task.task_id for r in result.runs] == ids and len(set(ids)) == len(ids)
    for run in result.runs:
        assert result.result_for(run.task.task_id) is run.findings
        assert run.findings.metadata["clauses"][0]["clause"] == run.task.context["focus_clauses"][0]


def test_repeated_integration_runs_are_identical(registry):
    for q in ("What percentage of licensing revenue goes to the inventor?", "Can research reduce my teaching load?",
              "Can I take a sabbatical for research?", "What is my teaching load, how do I apply for a research grant, and when is my salary paid?"):
        outs = []
        for _ in range(5):
            d = copy.deepcopy(run_question(q, registry).to_dict())
            for run in d["runs"]:
                if run["findings"]:
                    run["findings"].pop("ms", None)
            outs.append(d)
        assert all(o == outs[0] for o in outs), q


def test_research_integration_makes_no_llm_call_and_stays_inactive(registry):
    from handbook_bot import config

    result = run_question("What is my teaching load, how do I apply for a research grant, and when is my salary paid?", registry)
    assert result.llm_calls == 0 and all(f.llm_used is False for f in result.results)
    assert config.PLAN_F_ENABLED is False and config.MAX_LLM_CALLS == 2
    for name in ("orchestrator.py", "qa.py", "retrieval.py", "agents/router.py", "agents/synthesis.py", "agents/verifier.py"):
        text = (REPO / "handbook_bot" / name).read_text(encoding="utf-8")
        assert "dispatch" not in text and "specialists" not in text              # the runtime imports neither the seam nor a specialist
