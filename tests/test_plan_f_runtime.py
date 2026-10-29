"""
Runtime integration phase I-1: the Plan F runtime adapter (``handbook_bot/plan_f_runtime.py``).

The adapter is exercised on a knowledge base built from the full real handbook, read with the project's PDF loader and
chunked with the project's chunker exactly as production does, embedded with the fake embedder and reranked with the
stand-in reranker (no model, no network). The four frozen specialists are built from it with their own
``from_knowledge_base`` builders. Nothing here touches the live application: ``ui/pipeline.ask`` still calls
``answer_question`` directly and the adapter is wired into nothing.

Required I-1 regression cases: the known wrong supported answer of the frozen HR specialist ("What does Human
Resources do?") is never rendered, and the misleading partial result ("How many days of annual leave do faculty
members get?", whose evidence is the sixty-day maternity sentence) is never rendered.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

import faiss
import pytest

import institutional_support as IS
from conftest import FakeEmbedder
from faculty_services_support import RawOverlapReranker, registry_and_map
from handbook_bot import plan_f_runtime as PF
from handbook_bot.agents.specialists.base import Specialist
from handbook_bot.agents.specialists.contracts import Finding, FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask
from handbook_bot.agents.specialists.dispatch import STATE_PENDING, DispatchResult, SpecialistRun
from handbook_bot.agents.specialists.registry import DuplicateSpecialistError, SpecialistRegistry
from handbook_bot.agents.types import EvidenceResult, SynthesisResult, VerifyResult
from handbook_bot.agents.verifier import verify
from handbook_bot.chunking import build_chunks
from handbook_bot.knowledge_base import KnowledgeBase
from handbook_bot.qa import QAResult
from handbook_bot.sources import annotate_metadata, unregistered_source

REPO = Path(__file__).resolve().parents[1]
QARESULT_FIELDS = ["answer", "pages", "best_section", "evidence", "query_type", "items", "used_llm",
                   "timings", "num_candidates", "num_reranked", "agent_trace", "llm_calls", "retried", "sub_questions"]

HR_PHONE = "What is the phone number of the Human Resources Department?"
HR_KNOWN_WRONG = "What does Human Resources do?"
ANNUAL_LEAVE = "How many days of annual leave do faculty members get?"


# ---------------------------------------------------------------------------
# Fixtures: the shared real-page knowledge base and the adapter on it
# ---------------------------------------------------------------------------
def make_kb(pages=None, *, reranker=None, groq_client=None, source=None, stats=None, metadata_override=None):
    """A ``KnowledgeBase`` from real handbook pages (all of them by default), production chunking and metadata."""
    pages = list(pages) if pages is not None else list(IS.handbook_pages())
    registered, section_map = registry_and_map()
    chunks, metadata = build_chunks(pages)
    annotate_metadata(metadata, registered, section_map)
    embedder = FakeEmbedder()
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embeddings)
    if stats is None:
        stats = {"total_chunks": len(chunks), "total_pages": len(pages),
                 "source": {"source_id": registered.source_id, "registered": True, "section_records": len(section_map.records)}}
    return KnowledgeBase(pdf_file=str(IS.HANDBOOK_PDF), embedder=embedder, reranker=reranker or RawOverlapReranker(), groq_client=groq_client,
                         pages=pages, chunks=chunks, metadata=metadata if metadata_override is None else metadata_override,
                         embeddings=embeddings, index=index, stats=stats, source=registered if source is None else source)


@pytest.fixture(scope="module")
def kb():
    return make_kb()


@pytest.fixture(scope="module")
def runtime(kb):
    runtime, build = PF.build_runtime(kb)
    assert build.ok and runtime is not None, build.errors
    return runtime


def chunk_index(kb, needle, chunk_type=None):
    """The first chunk holding ``needle``; a printed ``row`` chunk is preferred over a paragraph window that repeats it."""
    hits = [i for i, text in enumerate(kb.chunks) if needle in text]
    assert hits, needle
    wanted = [i for i in hits if kb.metadata[i].get("chunk_type") == (chunk_type or "row")]
    return (wanted or hits)[0]


def trace_agents(result_or_attempt):
    trace = result_or_attempt.trace if isinstance(result_or_attempt, PF.AdapterResult) else result_or_attempt.agent_trace
    return [e.agent for e in trace]


def trace_json(attempt):
    return json.dumps(attempt.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Stub specialists (controlled results for the gating tests)
# ---------------------------------------------------------------------------
def stub_class(sid):
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def run(self, task):
        return self.behaviour(task)

    return type("Stub_%s" % sid.value, (Specialist,), {"specialist_id": sid, "__init__": __init__, "run": run})


def stub_registry(**behaviours):
    """All four specialists as stubs; a specialist without a behaviour returns ``not_found``."""
    def not_found(task):
        return SpecialistFindings(specialist_id=task.specialist_id, task_id=task.task_id, status=FindingStatus.NOT_FOUND,
                                  summary="stub: nothing", metadata={"clauses": [{"index": 0, "outcome": "not_found"}]})

    registry = SpecialistRegistry()
    for sid in SpecialistId:
        registry.register(stub_class(sid)(behaviours.get(sid.value, not_found)))
    return registry


def finding_for(kb, cid, quote, task, *, number=1, **metadata):
    meta = kb.metadata[cid]
    md = {"chunk_id": cid, "extractive": True}
    md.update(metadata)
    return Finding(finding_id="%s-f%d" % (task.task_id, number), claim=quote, evidence_quote=quote, source_id=meta["source_id"],
                   source_title=meta["source_title"], page=meta["page"], metadata=md)


def retrieval_result(kb, task, cid, quote, *, ask_shape="specific_recognised", requested=("number or amount",),
                     covered=("number or amount",), relation=None, handoffs=(), status=FindingStatus.SUPPORTED, missing=(),
                     extra_findings=()):
    clause = task.context["focus_clauses"][0]
    report = {"index": 0, "clause": clause, "outcome": status.value, "ask_shape": ask_shape, "requested_details": list(requested),
              "uncovered_details": [], "route": "policy"}
    if relation is not None:
        report["relation"] = relation
    findings = [finding_for(kb, cid, quote, task, details_covered=list(covered))] + list(extra_findings)
    return SpecialistFindings(specialist_id=task.specialist_id, task_id=task.task_id, status=status, findings=findings,
                              summary="stub", missing=list(missing), handoff_requests=list(handoffs),
                              metadata={"clauses": [report], "unprocessed_clauses": []})


# ---------------------------------------------------------------------------
# Registry factory (section 24)
# ---------------------------------------------------------------------------
def test_registry_builds_all_four_specialists_from_the_knowledge_base(kb):
    build = PF.build_specialist_registry(kb)
    assert build.ok and build.errors == {} and build.ms >= 0.0
    assert build.registry.ids() == list(PF.REQUIRED_SPECIALISTS) == [SpecialistId.TEACHING, SpecialistId.RESEARCH,
                                                                        SpecialistId.FACULTY_SERVICES, SpecialistId.INSTITUTIONAL]
    for sid, builder in PF.DEFAULT_BUILDERS.items():
        assert type(build.registry.resolve(sid)).from_knowledge_base == builder     # the specialists' own builders, nothing re-created
    json.dumps(build.to_dict())


def test_two_builds_are_independent_and_duplicate_registration_is_refused(kb):
    first, second = PF.build_specialist_registry(kb), PF.build_specialist_registry(kb)
    assert first.registry is not second.registry
    assert first.registry.resolve("teaching") is not second.registry.resolve("teaching")
    with pytest.raises(DuplicateSpecialistError):
        first.registry.register(second.registry.resolve("teaching"))
    assert not any(isinstance(v, SpecialistRegistry) for v in vars(PF).values())      # no module-level registry
    with pytest.raises(TypeError):
        PF.DEFAULT_BUILDERS[SpecialistId.TEACHING] = None                              # the builder table is read-only


@pytest.mark.parametrize("label,make", [
    ("malformed kb", lambda kb: object()),
    ("missing source", lambda kb: SimpleNamespace(embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks,
                                                  metadata=kb.metadata, source=None, stats=kb.stats)),
    ("unregistered source", lambda kb: SimpleNamespace(embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks,
                                                       metadata=kb.metadata, source=unregistered_source("data/other.pdf"), stats=kb.stats)),
    ("missing registry information", lambda kb: SimpleNamespace(embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks,
                                                                metadata=kb.metadata, source=kb.source, stats={})),
    ("missing section records", lambda kb: SimpleNamespace(embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks,
                                                           metadata=kb.metadata, source=kb.source, stats={"source": {"section_records": 0}})),
    ("missing metadata", lambda kb: SimpleNamespace(embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks,
                                                    metadata=None, source=kb.source, stats=kb.stats)),
    ("malformed metadata", lambda kb: SimpleNamespace(embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks,
                                                      metadata=kb.metadata[:10], source=kb.source, stats=kb.stats)),
])
def test_defective_knowledge_bases_fail_closed(kb, label, make):
    build = PF.build_specialist_registry(make(kb))
    assert build.registry is None and not build.ok, label
    assert set(build.errors) == {s.value for s in PF.REQUIRED_SPECIALISTS}, label
    assert all(isinstance(v, str) and v for v in build.errors.values()), label
    runtime, build2 = PF.build_runtime(make(kb))
    assert runtime is None and not build2.ok


def test_one_failing_builder_rejects_the_whole_registry_and_redacts_the_error(kb):
    def broken(_kb):
        raise RuntimeError(r"index missing at C:\Users\someone\private\teaching.faiss gsk_ABCDEFGHIJKLMNOPQRST")

    builders = dict(PF.DEFAULT_BUILDERS)
    builders[SpecialistId.FACULTY_SERVICES] = broken
    build = PF.build_specialist_registry(kb, builders=builders)
    assert build.registry is None and list(build.errors) == ["faculty_services"]
    assert "gsk_ABCDEFGHIJKLMNOPQRST" not in build.errors["faculty_services"] and "someone" not in build.errors["faculty_services"]
    assert build.errors["faculty_services"].startswith("RuntimeError")
    runtime, _ = PF.build_runtime(kb, builders=builders)
    assert runtime is None


def test_one_specialist_unavailable_or_misbuilt_means_no_partial_registry(kb):
    builders = {sid: b for sid, b in PF.DEFAULT_BUILDERS.items() if sid is not SpecialistId.RESEARCH}
    build = PF.build_specialist_registry(kb, builders=builders)
    assert build.registry is None and list(build.errors) == ["research"]
    wrong = dict(PF.DEFAULT_BUILDERS)
    wrong[SpecialistId.TEACHING] = PF.DEFAULT_BUILDERS[SpecialistId.RESEARCH]       # produces the Research specialist
    build = PF.build_specialist_registry(kb, builders=wrong)
    assert build.registry is None and "teaching" in build.errors and "produced the research specialist" in build.errors["teaching"]


def test_missing_specialists_and_an_incomplete_registry_make_plan_f_unavailable(kb, runtime):
    assert PF.missing_specialists(None) == [s.value for s in PF.REQUIRED_SPECIALISTS]
    three = SpecialistRegistry()
    for sid in (SpecialistId.TEACHING, SpecialistId.RESEARCH, SpecialistId.FACULTY_SERVICES):
        three.register(runtime.registry.resolve(sid))
    assert PF.missing_specialists(three) == ["institutional"]
    attempt = PF.PlanFRuntime(three, kb.chunks, kb.metadata).attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.REGISTRY_UNAVAILABLE and attempt.requires_fallback and attempt.result is None
    assert trace_agents(attempt) == ["plan_f_adapter", "plan_f_registry", "plan_f_outcome"]
    with pytest.raises(TypeError):
        PF.PlanFRuntime(object(), kb.chunks, kb.metadata)
    with pytest.raises(ValueError):
        PF.PlanFRuntime(three, kb.chunks, kb.metadata[:5])


# ---------------------------------------------------------------------------
# Finding -> legacy evidence item (section 25)
# ---------------------------------------------------------------------------
def test_adapt_finding_preserves_the_provenance(kb):
    cid = chunk_index(kb, "Human Resources +971 6 5050023")
    task = SpecialistTask("task-1", HR_PHONE, specialist_id="institutional", context={"focus_clauses": [HR_PHONE]})
    finding = finding_for(kb, cid, "Human Resources +971 6 5050023 +971 6 5585200", task, office_id="human_resources",
                          evidence_tier="E1", contact_values={"telephone": "+971 6 5050023"})
    item = PF.adapt_finding(finding, SpecialistId.INSTITUTIONAL, HR_PHONE, kb.chunks, kb.metadata)
    assert item["idx"] == cid and item["chunk"] == finding.evidence_quote
    meta = item["meta"]
    assert meta["page"] == 54 == finding.page and meta["source_id"] == finding.source_id and meta["source_title"] == finding.source_title
    assert meta["chunk_id"] == cid and meta["section_no"] == "1.16" and meta["section_title"] and meta["chunk_type"] == "row"
    plan_f = meta["plan_f"]
    assert plan_f["specialist_id"] == "institutional" and plan_f["finding_id"] == "task-1-f1" and plan_f["focus_clause"] == HR_PHONE
    assert plan_f["claim"] == plan_f["evidence_quote"] == finding.evidence_quote and plan_f["chunk_id"] == cid
    assert plan_f["office_id"] == "human_resources" and plan_f["contact_values"] == {"telephone": "+971 6 5050023"}
    assert "rerank_score" not in item                                     # nothing invented for the legacy shape
    assert kb.metadata[cid] is not meta and "plan_f" not in kb.metadata[cid]   # the corpus metadata is untouched
    json.dumps(item)


def test_adapt_finding_rejects_a_malformed_finding(kb):
    cid = chunk_index(kb, "Human Resources +971 6 5050023")
    task = SpecialistTask("task-1", HR_PHONE, specialist_id="institutional", context={"focus_clauses": [HR_PHONE]})
    bad = finding_for(kb, cid, "Human Resources +971 6 5050023 +971 6 5585200", task)
    bad.metadata["chunk_id"] = cid + 1
    with pytest.raises(ValueError):
        PF.adapt_finding(bad, SpecialistId.INSTITUTIONAL, HR_PHONE, kb.chunks, kb.metadata)


# ---------------------------------------------------------------------------
# Guard 1: the quote must exist in its source chunk (section 26)
# ---------------------------------------------------------------------------
def _row_finding(kb, **overrides):
    cid = chunk_index(kb, "Human Resources +971 6 5050023")
    base = dict(finding_id="task-1-f1", claim="Human Resources +971 6 5050023 +971 6 5585200",
                evidence_quote="Human Resources +971 6 5050023 +971 6 5585200", source_id=kb.metadata[cid]["source_id"],
                source_title=kb.metadata[cid]["source_title"], page=54, metadata={"chunk_id": cid})
    base.update(overrides)
    return SimpleNamespace(**base), cid


def test_quote_guard_accepts_an_exact_quote(kb):
    finding, _ = _row_finding(kb)
    assert PF.quote_problem(finding, kb.chunks, kb.metadata) is None


@pytest.mark.parametrize("label,overrides,expected", [
    ("missing quote", {"evidence_quote": "Human Resources +971 6 5050023 +971 6 5585299"}, "not found"),
    ("quote from another chunk", {"metadata": {"chunk_id": "OTHER"}}, "not found"),
    ("bad chunk id", {"metadata": {"chunk_id": "54"}}, "not an integer"),
    ("boolean chunk id", {"metadata": {"chunk_id": True}}, "not an integer"),
    ("out-of-range chunk id", {"metadata": {"chunk_id": 10 ** 9}}, "out of range"),
    ("negative chunk id", {"metadata": {"chunk_id": -1}}, "out of range"),
    ("empty quote", {"evidence_quote": ""}, "empty quote"),
    ("whitespace quote", {"evidence_quote": "   "}, "empty quote"),
    ("malformed metadata", {"metadata": None}, "no metadata"),
    ("metadata without chunk id", {"metadata": {}}, "missing"),
    ("page mismatch", {"page": 53}, "page mismatch"),
    ("missing page", {"page": None}, "no page"),
    ("source mismatch", {"source_id": "another_source"}, "source mismatch"),
])
def test_quote_guard_rejects_every_unsafe_shape(kb, label, overrides, expected):
    finding, cid = _row_finding(kb, **overrides)
    if overrides.get("metadata") == {"chunk_id": "OTHER"}:
        finding.metadata = {"chunk_id": chunk_index(kb, "Passport Unit +971 6 5050066")}   # a real chunk that does not hold the quote
    problem = PF.quote_problem(finding, kb.chunks, kb.metadata)
    assert problem is not None and expected in problem, (label, problem)


def test_quote_guard_uses_no_normalisation(kb):
    finding, _ = _row_finding(kb, evidence_quote="Human  Resources +971 6 5050023 +971 6 5585200")   # one extra space
    assert PF.quote_problem(finding, kb.chunks, kb.metadata) is not None
    finding, _ = _row_finding(kb, evidence_quote="human resources +971 6 5050023 +971 6 5585200")    # case
    assert PF.quote_problem(finding, kb.chunks, kb.metadata) is not None


def test_quote_guard_rejection_ends_the_attempt_before_the_verifier(kb):
    cid = chunk_index(kb, "Employment offers are drafted by HR", "paragraph")

    def tampered(task):
        return retrieval_result(kb, task, cid, "Employment offers are drafted by the Dean upon receiving the selected candidate")   # not in the chunk

    attempt = PF.PlanFRuntime(stub_registry(faculty_services=tampered), kb.chunks, kb.metadata).attempt("Who drafts employment offers?")
    assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_QUOTE_GUARD and attempt.result is None and attempt.requires_fallback
    assert "verifier" not in trace_agents(attempt) and "subject_guard" not in trace_agents(attempt)
    assert [e.decision for e in attempt.trace if e.agent == "quote_guard"] == ["rejected"]


# ---------------------------------------------------------------------------
# Guard 2: the asked subject must be bound (section 27)
# ---------------------------------------------------------------------------
def test_named_entities_are_read_deterministically():
    assert PF.named_entities(HR_KNOWN_WRONG) == ["Human Resources"]
    assert PF.named_entities(HR_PHONE) == ["Human Resources Department"]
    assert PF.named_entities("Does HR arrange travel for new faculty members?") == ["HR"]
    assert PF.named_entities("What does the Research Funding Department do?") == ["Research Funding Department"]
    assert PF.named_entities("What is the mission of the University of Sharjah?") == []        # the University is no entity
    assert PF.named_entities("How many office hours must faculty hold?") == []
    assert PF.named_entities("Who chairs the College of Graduate Studies and Research?") == ["College of Graduate Studies and Research"]
    assert PF.named_entities("Is the TTO part of the VCRGS office?") == ["TTO", "VCRGS"]
    assert PF.named_entities("") == []


def test_entity_in_text_rules():
    row = "Human Resources +971 6 5050023 +971 6 5585200"
    assert PF.entity_in_text("Human Resources Department", row)        # the closing unit noun may be omitted
    assert PF.entity_in_text("Human Resources", row)
    assert not PF.entity_in_text("Human Resources", "Human resourcesful staff")      # whole phrase only
    assert not PF.entity_in_text("Finance Department", row)
    assert PF.entity_in_text("HR", "HR arranges travel") and not PF.entity_in_text("HR", "hr arranges travel")   # acronyms are case-sensitive
    assert not PF.entity_in_text("HR", "CHR committee")
    assert not PF.entity_in_text("Main Library", "the library")           # one remaining word is too generic to count


def _report(clause, **extra):
    base = {"index": 0, "clause": clause, "outcome": "read", "ask_kind": "channel", "evidence_tier_reached": "E1", "offices": []}
    base.update(extra)
    return base


def test_subject_guard_accepts_an_explicit_subject_in_the_evidence(kb):
    cid = chunk_index(kb, "Human Resources +971 6 5050023")
    task = SpecialistTask("task-1", HR_PHONE, specialist_id="institutional", context={"focus_clauses": [HR_PHONE]})
    finding = finding_for(kb, cid, "Human Resources +971 6 5050023 +971 6 5585200", task, office_id="human_resources", evidence_tier="E1",
                          directory_entry="complete_row", row_complete=True, contact_values={"telephone": "+971 6 5050023"})
    item = PF.adapt_finding(finding, SpecialistId.INSTITUTIONAL, HR_PHONE, kb.chunks, kb.metadata)
    bound, why = PF.subject_binding(HR_PHONE, _report(HR_PHONE, offices=["human_resources"]), finding, item, SpecialistId.INSTITUTIONAL)
    assert bound, why


def test_subject_guard_accepts_a_metadata_bound_subject_in_a_lowercase_question(runtime):
    attempt = runtime.attempt("what is the phone number of human resources")
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED and attempt.result.answer == "Human Resources +971 6 5050023 +971 6 5585200"


@pytest.mark.parametrize("clause,label", [
    ("What is the phone number of the Finance Department?", "unrelated entity"),
    ("What is the telephone number of the Registration office?", "neighbouring office"),
])
def test_subject_guard_rejects_another_office_row(kb, clause, label):
    cid = chunk_index(kb, "Human Resources +971 6 5050023")
    task = SpecialistTask("task-1", clause, specialist_id="institutional", context={"focus_clauses": [clause]})
    finding = finding_for(kb, cid, "Human Resources +971 6 5050023 +971 6 5585200", task, office_id="human_resources", evidence_tier="E1",
                          directory_entry="complete_row", row_complete=True, contact_values={"telephone": "+971 6 5050023"})
    item = PF.adapt_finding(finding, SpecialistId.INSTITUTIONAL, clause, kb.chunks, kb.metadata)
    bound, why = PF.subject_binding(clause, _report(clause), finding, item, SpecialistId.INSTITUTIONAL)
    assert not bound and "not the office the clause asks about" in why, label


def test_subject_guard_rejects_a_directory_row_that_is_not_complete(kb):
    cid = chunk_index(kb, "Human Resources +971 6 5050023")
    task = SpecialistTask("task-1", HR_PHONE, specialist_id="institutional", context={"focus_clauses": [HR_PHONE]})
    finding = finding_for(kb, cid, "Human Resources +971 6 5050023 +971 6 5585200", task, office_id="human_resources", evidence_tier="E1",
                          directory_entry="labelled_row", row_complete=False, contact_values={})
    item = PF.adapt_finding(finding, SpecialistId.INSTITUTIONAL, HR_PHONE, kb.chunks, kb.metadata)
    assert PF.subject_binding(HR_PHONE, _report(HR_PHONE, offices=["human_resources"]), finding, item, SpecialistId.INSTITUTIONAL)[0] is False
    finding.metadata.pop("office_id")
    assert "names no office" in PF.subject_binding(HR_PHONE, _report(HR_PHONE), finding, item, SpecialistId.INSTITUTIONAL)[1]


def test_subject_guard_rejects_a_generic_statement_not_bound_to_the_asked_entity(kb):
    """The HR fixture-corpus shape of the known wrong answer: a supported quote that never names Human Resources."""
    cid = chunk_index(kb, "may terminate faculty contracts")
    quote = kb.chunks[cid]
    task = SpecialistTask("task-1", HR_KNOWN_WRONG, specialist_id="faculty_services", context={"focus_clauses": [HR_KNOWN_WRONG]})
    finding = finding_for(kb, cid, quote, task, details_covered=["the asked specifics stated with the answer (hr)"])
    item = PF.adapt_finding(finding, SpecialistId.FACULTY_SERVICES, HR_KNOWN_WRONG, kb.chunks, kb.metadata)
    report = {"index": 0, "clause": HR_KNOWN_WRONG, "outcome": "supported", "ask_shape": "specific_recognised",
              "requested_details": ["the asked specifics stated with the answer (hr)"], "uncovered_details": []}
    bound, why = PF.subject_binding(HR_KNOWN_WRONG, report, finding, item, SpecialistId.FACULTY_SERVICES)
    assert not bound and "Human Resources" in why

    def forced(task_):
        return retrieval_result(kb, task_, cid, quote, requested=("the asked specifics stated with the answer (hr)",),
                                covered=("the asked specifics stated with the answer (hr)",))

    attempt = PF.PlanFRuntime(stub_registry(faculty_services=forced), kb.chunks, kb.metadata).attempt(HR_KNOWN_WRONG)
    assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_SUBJECT_GUARD and attempt.result is None


def test_subject_guard_requires_every_requested_detail_and_the_clause_content(kb):
    cid = chunk_index(kb, "required to schedule a minimum of five office hours per week", "paragraph")
    quote = "Full-time faculty members in academic and career/technical programs are required to schedule a minimum of five office hours per week"
    assert quote in kb.chunks[cid]
    question = "How many office hours must faculty hold?"
    task = SpecialistTask("task-1", question, specialist_id="teaching", context={"focus_clauses": [question]})
    finding = finding_for(kb, cid, quote, task, details_covered=["number or amount"])
    item = PF.adapt_finding(finding, SpecialistId.TEACHING, question, kb.chunks, kb.metadata)
    report = {"index": 0, "clause": question, "outcome": "supported", "ask_shape": "specific_recognised",
              "requested_details": ["number or amount", "the asked specifics stated with the answer (office_hours)"], "uncovered_details": []}
    bound, why = PF.subject_binding(question, report, finding, item, SpecialistId.TEACHING)
    assert not bound and "requested detail not covered" in why
    report["requested_details"] = ["number or amount"]
    assert PF.subject_binding(question, report, finding, item, SpecialistId.TEACHING)[0] is True
    unrelated = "How many days of maternity leave are female faculty entitled to?"
    assert PF.subject_binding(unrelated, dict(report, clause=unrelated), finding, item, SpecialistId.TEACHING)[0] is False
    assert "lists no requested detail" in PF.subject_binding(question, dict(report, requested_details=[]), finding, item, SpecialistId.TEACHING)[1]


# ---------------------------------------------------------------------------
# Required regression cases (sections 10 and 11)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [HR_KNOWN_WRONG, "What does HR do?"])
def test_known_hr_false_support_is_blocked(runtime, question):
    attempt = runtime.attempt(question)
    assert attempt.specialist_id == "faculty_services" and attempt.status == "supported"      # the frozen specialist still says so
    assert attempt.outcome is PF.PlanFOutcome.UNSAFE_RESULT and attempt.result is None and attempt.requires_fallback
    assert "ask shape general" in attempt.reason
    assert "verifier" not in trace_agents(attempt) and trace_agents(attempt)[-1] == "plan_f_outcome"


def test_misleading_partial_annual_leave_result_is_blocked(runtime):
    attempt = runtime.attempt(ANNUAL_LEAVE)
    assert attempt.specialist_id == "faculty_services" and attempt.status == "partial"
    assert attempt.outcome is PF.PlanFOutcome.UNSAFE_RESULT and attempt.result is None and attempt.requires_fallback
    assert attempt.reason == "status partial"
    assert "sixty" not in trace_json(attempt).lower() or attempt.result is None
    assert all("maternity" not in (e.detail.get("summary") or "") for e in attempt.trace) or attempt.result is None


def test_partial_results_can_never_render_even_when_the_evidence_looks_right(kb):
    cid = chunk_index(kb, "sixty calendar days of maternity leave")
    quote = "Female faculty employed on a full-time basis are entitled to sixty calendar days of maternity leave with full pay."

    def partial(task):
        return retrieval_result(kb, task, cid, quote, status=FindingStatus.PARTIAL, missing=["the asked leave type"])

    attempt = PF.PlanFRuntime(stub_registry(faculty_services=partial), kb.chunks, kb.metadata).attempt(ANNUAL_LEAVE)
    assert attempt.outcome is PF.PlanFOutcome.UNSAFE_RESULT and attempt.reason == "status partial" and attempt.result is None


# ---------------------------------------------------------------------------
# Status gating (section 28)
# ---------------------------------------------------------------------------
def _hr_phone_task_stub(kb, status, *, handoff=False, general=False, raise_error=False):
    cid = chunk_index(kb, "Human Resources +971 6 5050023")
    quote = "Human Resources +971 6 5050023 +971 6 5585200"

    def behaviour(task):
        if raise_error:
            raise RuntimeError(r"boom at C:\Users\someone\private\institutional.faiss gsk_ABCDEFGHIJKLMNOPQRST")
        if status in (FindingStatus.NOT_FOUND, FindingStatus.OUT_OF_SCOPE):
            return SpecialistFindings(specialist_id=task.specialist_id, task_id=task.task_id, status=status, summary="stub",
                                      missing=["x"] if status is FindingStatus.NOT_FOUND else [], metadata={"clauses": [{"index": 0}]})
        handoffs = [HandoffRequest(from_specialist=task.specialist_id, requested_specialist=SpecialistId.FACULTY_SERVICES,
                                   reason="stub handoff")] if handoff else []
        report = {"index": 0, "clause": task.context["focus_clauses"][0], "outcome": "read", "ask_kind": "channel",
                  "evidence_tier_reached": "E1", "offices": ["human_resources"]}
        if general:
            report = {"index": 0, "clause": task.context["focus_clauses"][0], "outcome": "supported", "ask_shape": "general",
                      "requested_details": ["the asked specifics stated with the answer (hr)"], "uncovered_details": []}
        finding = finding_for(kb, cid, quote, task, office_id="human_resources", evidence_tier="E1", directory_entry="complete_row",
                              row_complete=True, contact_values={"telephone": "+971 6 5050023"}, details_covered=[])
        return SpecialistFindings(specialist_id=task.specialist_id, task_id=task.task_id, status=status, findings=[finding],
                                  summary="stub", missing=["y"] if status is FindingStatus.PARTIAL else [], handoff_requests=handoffs,
                                  metadata={"clauses": [report]})

    return behaviour


@pytest.mark.parametrize("label,kwargs,expected_reason", [
    ("partial", dict(status=FindingStatus.PARTIAL), "status partial"),
    ("not found", dict(status=FindingStatus.NOT_FOUND), "status not_found"),
    ("out of scope", dict(status=FindingStatus.OUT_OF_SCOPE), "status out_of_scope"),
    ("error", dict(status=FindingStatus.SUPPORTED, raise_error=True), "status error"),
    ("supported with handoff", dict(status=FindingStatus.SUPPORTED, handoff=True), "handoff pending to faculty_services"),
])
def test_only_an_eligible_supported_result_progresses(kb, label, kwargs, expected_reason):
    attempt = PF.PlanFRuntime(stub_registry(institutional=_hr_phone_task_stub(kb, **kwargs)), kb.chunks, kb.metadata).attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.UNSAFE_RESULT and attempt.result is None, label
    assert attempt.reason.startswith(expected_reason), (label, attempt.reason)
    assert "quote_guard" not in trace_agents(attempt) and "verifier" not in trace_agents(attempt)
    if label == "supported with handoff":
        assert [e.decision for e in attempt.trace if e.agent == "plan_f_gate"] == ["HANDOFF_PENDING"]
    if label == "error":
        text = trace_json(attempt)
        assert "gsk_ABCDEFGHIJKLMNOPQRST" not in text and "someone" not in text and "gsk_***" in text


def test_an_eligible_supported_stub_result_is_rendered(kb):
    attempt = PF.PlanFRuntime(stub_registry(institutional=_hr_phone_task_stub(kb, FindingStatus.SUPPORTED)), kb.chunks, kb.metadata).attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED and attempt.result.answer == "Human Resources +971 6 5050023 +971 6 5585200"


def test_a_general_ask_is_not_eligible_even_when_the_entity_is_quoted(kb):
    attempt = PF.PlanFRuntime(stub_registry(faculty_services=_hr_phone_task_stub(kb, FindingStatus.SUPPORTED, general=True)),
                              kb.chunks, kb.metadata).attempt("Who drafts employment offers?")
    assert attempt.outcome is PF.PlanFOutcome.UNSAFE_RESULT and "ask shape general" in attempt.reason


def test_a_pending_task_requests_fallback(kb, runtime, monkeypatch):
    def pending(decision, registry, *, executable=None, question=None):
        task = decision.subtasks[0]
        return DispatchResult(question=question or "", decision=decision,
                              runs=[SpecialistRun(task, task.specialist_id, STATE_PENDING, None, "institutional is executable but not registered")])

    monkeypatch.setattr(PF, "dispatch", pending)
    attempt = runtime.attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.REQUIRES_FALLBACK and "task pending" in attempt.reason and attempt.result is None


# ---------------------------------------------------------------------------
# One valid supported example from each frozen specialist (section 29)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,specialist,pages,starts_with", [
    (HR_PHONE, "institutional", [54], "Human Resources +971 6 5050023 +971 6 5585200"),
    ("Does HR arrange travel and hotel accommodations for new faculty members?", "faculty_services", [61],
     "• HR arranges travel and hotel accommodations for new faculty members."),
    ("How many credit hours is a full teaching load?", "teaching", [96], 'o "A" Regular Faculty'),
    ("How many theses can a faculty member supervise at once?", "research", [81],
     "• A faculty member is allowed to supervise a maximum of four theses/dissertations simultaneously"),
])
def test_each_specialist_has_a_supported_example_the_adapter_renders(kb, runtime, question, specialist, pages, starts_with):
    attempt = runtime.attempt(question)
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED and not attempt.requires_fallback, attempt.reason
    assert attempt.specialist_id == specialist and attempt.status == "supported"
    result = attempt.result
    assert isinstance(result, QAResult) and result.answer.startswith(starts_with) and result.pages == pages
    assert result.llm_calls == 0 and result.used_llm is False and result.retried is False
    for item in result.items:
        assert item["chunk"] in kb.chunks[item["idx"]] and item["meta"]["page"] in result.pages
    assert trace_agents(attempt) == ["plan_f_adapter", "plan_f_registry", "coordinator", "dispatcher", "specialist:%s" % specialist,
                                     "quote_guard", "subject_guard", "verifier", "plan_f_outcome"]


def test_secondary_findings_that_do_not_cover_the_ask_are_dropped_not_rendered(runtime):
    attempt = runtime.attempt("How many days of maternity leave are female faculty entitled to?")
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED
    assert attempt.result.pages == [107] and "sick leave" not in attempt.result.answer and "maternity leave" in attempt.result.answer
    guard = next(e for e in attempt.trace if e.agent == "subject_guard")
    assert guard.detail["accepted"] == ["task-1-f1"] and [d["finding"] for d in guard.detail["dropped"]] == ["task-1-f2"]
    attempt = runtime.attempt("How many office hours must faculty hold?")
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED and attempt.result.pages == [79, 80]       # two findings, both bound


# ---------------------------------------------------------------------------
# Complex questions all fall back (section 30)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,outcome,reason_part", [
    ("Who handles faculty visas and where is that office?", PF.PlanFOutcome.UNSAFE_RESULT, "multiple specialists"),
    ("Where is the Research Institute and what research funding is available?", PF.PlanFOutcome.UNSAFE_RESULT, "synthesis required"),
    ("Which office handles faculty housing?", PF.PlanFOutcome.UNSAFE_RESULT, "multiple specialists"),
    ("Can I reduce my teaching load if I have a funded research project, and who approves it?", PF.PlanFOutcome.UNSAFE_RESULT, "multiple specialists"),
    ("asdfgh qwerty zxcvb", PF.PlanFOutcome.NOT_APPLICABLE, "no specialist selected"),
    ("Hello", PF.PlanFOutcome.NOT_APPLICABLE, "greeting"),
    ("", PF.PlanFOutcome.NOT_APPLICABLE, "empty question"),
    ("   ", PF.PlanFOutcome.NOT_APPLICABLE, "empty question"),
])
def test_complex_or_empty_questions_request_fallback(runtime, question, outcome, reason_part):
    attempt = runtime.attempt(question)
    assert attempt.outcome is outcome and attempt.result is None and attempt.requires_fallback
    assert reason_part in attempt.reason
    assert "dispatcher" not in trace_agents(attempt)                      # nothing is executed for a question outside the safe case


def test_decision_problem_covers_every_gate():
    from handbook_bot.agents.coordinator import coordinate

    assert PF.decision_problem(coordinate("Hello")) == "greeting: no specialist needed"
    assert PF.decision_problem(coordinate("asdfgh")).startswith("no specialist selected")
    decision = coordinate(HR_PHONE)
    assert PF.decision_problem(decision) is None
    decision.subtasks[0].context["focus_clauses"] = ["a", "b"]
    assert "focus clauses" in PF.decision_problem(decision)
    decision = coordinate(HR_PHONE)
    decision.metadata["uncovered_clauses"] = ["where is it"]
    assert PF.decision_problem(decision).startswith("uncovered clause")
    decision = coordinate(HR_PHONE)
    decision.metadata["router"]["is_multipart"] = True
    assert PF.decision_problem(decision) == "multipart question"
    decision = coordinate(HR_PHONE)
    decision.requires_synthesis = True
    assert PF.decision_problem(decision) == "synthesis required by the Coordinator"
    decision = coordinate(HR_PHONE)
    decision.subtasks[0].requested_by = "research"
    assert "not issued by the Coordinator" in PF.decision_problem(decision)


def test_a_raising_specialist_is_reported_as_a_redacted_error(kb):
    attempt = PF.PlanFRuntime(stub_registry(institutional=_hr_phone_task_stub(kb, FindingStatus.SUPPORTED, raise_error=True)),
                              kb.chunks, kb.metadata).attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.UNSAFE_RESULT and attempt.status == "error" and attempt.result is None
    assert "gsk_ABCDEFGHIJKLMNOPQRST" not in trace_json(attempt) and "someone" not in trace_json(attempt)


def test_coordinator_and_dispatcher_exceptions_are_redacted(kb, runtime, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError(r"failed reading C:\Users\someone\private\cues.json with key gsk_ABCDEFGHIJKLMNOPQRST")

    monkeypatch.setattr(PF, "coordinate", boom)
    attempt = runtime.attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.ERROR_REDACTED and attempt.result is None and attempt.requires_fallback
    assert "gsk_ABCDEFGHIJKLMNOPQRST" not in trace_json(attempt) and "someone" not in trace_json(attempt)
    monkeypatch.undo()
    monkeypatch.setattr(PF, "dispatch", boom)
    attempt = runtime.attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.ERROR_REDACTED and "dispatcher raised" in attempt.reason
    assert "gsk_ABCDEFGHIJKLMNOPQRST" not in trace_json(attempt)


def test_handoffs_are_never_executed(kb):
    calls = []

    def institutional(task):
        calls.append(("institutional", task.task_id))
        return _hr_phone_task_stub(kb, FindingStatus.SUPPORTED, handoff=True)(task)

    def hr(task):
        calls.append(("faculty_services", task.task_id))
        raise AssertionError("the handoff target must never run")

    attempt = PF.PlanFRuntime(stub_registry(institutional=institutional, faculty_services=hr), kb.chunks, kb.metadata).attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.UNSAFE_RESULT and calls == [("institutional", "task-1")]
    assert attempt.dispatch is not None and len(attempt.dispatch.runs) == 1 and attempt.dispatch.handoffs


# ---------------------------------------------------------------------------
# The unchanged legacy Verifier on adapted evidence (section 31)
# ---------------------------------------------------------------------------
def test_safe_adapted_finding_is_accepted_by_the_legacy_verifier(runtime):
    attempt = runtime.attempt(HR_PHONE)
    verifier = next(e for e in attempt.trace if e.agent == "verifier")
    assert verifier.decision == "accepted" and verifier.detail["impl"] == "agents.verifier" and verifier.detail["pages"] == [54]
    assert verifier.used_llm is False


def test_an_unsupported_rendering_is_rejected_by_the_real_verifier(kb):
    cid = chunk_index(kb, "Human Resources +971 6 5050023")
    task = SpecialistTask("task-1", HR_PHONE, specialist_id="institutional", context={"focus_clauses": [HR_PHONE]})
    finding = finding_for(kb, cid, "Human Resources +971 6 5050023 +971 6 5585200", task)
    item = PF.adapt_finding(finding, SpecialistId.INSTITUTIONAL, HR_PHONE, kb.chunks, kb.metadata)
    good = SynthesisResult(answer=PF.render_answer([item]), claimed_pages=[54], prompt_version=PF.PROMPT_VERSION)
    bad = SynthesisResult(answer="The phone number of Human Resources is +971 6 9999999.", claimed_pages=[54], prompt_version=PF.PROMPT_VERSION)
    evidence = EvidenceResult(HR_PHONE, "contact", [item], None, True, None, False, None)
    assert verify(HR_PHONE, good, [evidence], [], is_retry=False).accepted is True
    assert verify(HR_PHONE, bad, [evidence], [], is_retry=False).accepted is False


def test_verifier_rejection_triggers_fallback(runtime, monkeypatch):
    monkeypatch.setattr(PF, "verify", lambda *a, **k: VerifyResult(False, [], ["stub rejection"], [], None))
    attempt = runtime.attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_VERIFIER and attempt.result is None and attempt.requires_fallback
    assert [e.decision for e in attempt.trace if e.agent == "verifier"] == ["rejected"]
    monkeypatch.setattr(PF, "verify", lambda *a, **k: VerifyResult(True, [999], [], [], None))     # a page outside the evidence
    attempt = runtime.attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_VERIFIER and "cited no accepted page" in attempt.reason
    monkeypatch.setattr(PF, "verify", lambda *a, **k: "not a verdict")
    attempt = runtime.attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.ERROR_REDACTED


def test_pages_are_taken_from_accepted_evidence_only(runtime):
    for question in (HR_PHONE, "How many office hours must faculty hold?", "How many theses can a faculty member supervise at once?"):
        attempt = runtime.attempt(question)
        result = attempt.result
        assert result.pages and set(result.pages) <= {item["meta"]["page"] for item in result.items}
        assert result.pages == PF.accepted_pages(result.items) or set(result.pages) <= set(PF.accepted_pages(result.items))


# ---------------------------------------------------------------------------
# QAResult contract (section 32)
# ---------------------------------------------------------------------------
def test_qaresult_type_is_unchanged():
    assert [f.name for f in dataclasses.fields(QAResult)] == QARESULT_FIELDS


def test_accepted_result_fills_every_field_consistently(kb, runtime):
    attempt = runtime.attempt(HR_PHONE)
    result = attempt.result
    assert isinstance(result, QAResult) and set(vars(result)) == set(QARESULT_FIELDS)
    assert result.answer == "Human Resources +971 6 5050023 +971 6 5585200" == result.items[0]["chunk"]
    assert result.pages == [54] and result.best_section == "Contact Information" and result.query_type == "contact"
    assert result.evidence == result.answer and len(result.items) == 1
    assert result.used_llm is False and result.llm_calls == 0 and result.retried is False and result.sub_questions == []
    assert result.num_candidates == 1 and result.num_reranked == 1
    assert set(result.timings) >= {"coordinator_ms", "dispatch_ms", "specialist_ms", "guards_ms", "verify_ms", "generation_ms", "total_ms"}
    assert all(isinstance(v, float) and v >= 0.0 for v in result.timings.values()) and result.timings["generation_ms"] == 0.0
    assert result.agent_trace == attempt.trace and all(e.used_llm is False for e in result.agent_trace)
    payload = result.to_dict()
    json.dumps(payload)
    assert set(payload) == set(QARESULT_FIELDS)
    assert "plan_f" in result.items[0]["meta"] and result.items[0]["meta"]["chunk_type"] == "row"
    assert kb.metadata[result.items[0]["idx"]].get("plan_f") is None


def test_long_quotes_are_previewed_like_the_legacy_result(runtime):
    attempt = runtime.attempt("How many days of maternity leave are female faculty entitled to?")
    result = attempt.result
    assert result.evidence == result.answer[:len(result.evidence)] or result.evidence.endswith("...")
    assert len(result.evidence) <= 523


def test_plan_f_makes_no_model_call(kb):
    class ExplodingClient:
        def __getattr__(self, name):
            raise AssertionError("the adapter must never touch the model client")

    spy = make_kb(kb.pages, groq_client=ExplodingClient())
    runtime, build = PF.build_runtime(spy)
    assert build.ok
    attempt = runtime.attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED and attempt.result.llm_calls == 0
    for question in (HR_KNOWN_WRONG, ANNUAL_LEAVE, "Who handles faculty visas and where is that office?"):
        assert runtime.attempt(question).result is None
    source = (REPO / "handbook_bot" / "plan_f_runtime.py").read_text(encoding="utf-8")
    for forbidden in ("ask_groq", "synthesize", "Groq(", "chat.completions", "build_prompt"):
        assert forbidden not in source, forbidden
    assert "from .orchestrator" not in source and "import orchestrator" not in source


def test_attempts_are_deterministic(runtime):
    def strip(attempt):
        d = attempt.to_dict()
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

    for question in (HR_PHONE, HR_KNOWN_WRONG, "How many theses can a faculty member supervise at once?", "Hello"):
        assert strip(runtime.attempt(question)) == strip(runtime.attempt(question))


# ---------------------------------------------------------------------------
# Test-only fallback wrapper (section 19) and isolation from the live runtime (section 23)
# ---------------------------------------------------------------------------
def test_answer_with_fallback_prefers_plan_f_and_otherwise_returns_the_legacy_result(kb, runtime):
    accepted = PF.answer_with_fallback(HR_PHONE, kb, runtime)
    assert accepted.answer == "Human Resources +971 6 5050023 +971 6 5585200" and accepted.llm_calls == 0
    legacy = PF.answer_with_fallback("Which office handles faculty housing?", kb, runtime)
    assert isinstance(legacy, QAResult) and legacy.answer != accepted.answer
    agents = trace_agents(legacy)
    assert agents[:3] == ["plan_f_adapter", "plan_f_registry", "coordinator"] and "router" in agents       # Plan F decisions, then the legacy trace
    assert "plan_f_outcome" in agents and agents.index("plan_f_outcome") < agents.index("router")
    without = PF.answer_with_fallback(HR_PHONE, kb, None)
    assert isinstance(without, QAResult) and trace_agents(without)[0] == "router"


def test_live_runtime_modules_do_not_import_the_adapter():
    """Since runtime integration phase I-2, ``ui/pipeline.py`` is the one approved seam that may reference the adapter
    (imported lazily, see ``tests/test_plan_f_pipeline.py``); the legacy core, the UI and the evaluation runner stay
    free of it."""
    live = ["app.py", "ui/chat.py", "handbook_bot/qa.py", "handbook_bot/orchestrator.py", "handbook_bot/agents/router.py",
            "handbook_bot/retrieval.py", "handbook_bot/agents/synthesis.py", "handbook_bot/agents/verifier.py", "handbook_bot/agents/types.py",
            "handbook_bot/__init__.py", "handbook_bot/knowledge_base.py", "eval/run_eval.py"]
    for name in live:
        text = (REPO / name).read_text(encoding="utf-8")
        assert "plan_f_runtime" not in text, name
    pipeline = (REPO / "ui" / "pipeline.py").read_text(encoding="utf-8")
    assert "answer_question(" in pipeline and "plan_f_runtime" in pipeline
    config_text = (REPO / "handbook_bot" / "config.py").read_text(encoding="utf-8")
    assert re.search(r"^PLAN_F_ENABLED: bool = _env_bool\(\"PLAN_F_ENABLED\", False\)", config_text, re.M)
    assert re.search(r"^PLAN_F_SHADOW: bool = _env_bool\(\"PLAN_F_SHADOW\", False\)", config_text, re.M)
