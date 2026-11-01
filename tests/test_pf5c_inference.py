"""
Post-freeze PF-5C (decision record D-115): safe guide-intent inference.

Offline. The real eight-source corpus without a model, synthetic clauses written for these tests (none is a question of an
evaluation set), and the Plan F adapter end to end with the fake embedder and the raw-overlap reranker. Inference only
decides which guide is a CANDIDATE for a clause that names no system; every test of an accepted inference also shows that
the D-111 evidence rules still decide the answer:
    A  strong action/object asks are inferred; single ambiguous words, weak asks and ties are not (threshold and margin)
    B  policy, authority, organizational, contact and other-platform asks are never inferred; an explicit system is never inferred
    C  the signal table is printed in the official guides; the held sources are in no family
    D  inference is not evidence: bound with every asked word as a term, re-verified by the adapter from the clause
    E  conflicts, wrong sources and unsupported asks
    F  the runtime: registry selection, the owner of an inferred question, the decline rule, handoffs, typos, diagnostics
    G  two grounding refusals added after the development probes (a procedure heading naming another action, a location
       named only by a heading)
and the frozen PF-5C fresh held-out set.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import faiss
import pytest

from conftest import FakeEmbedder
from faculty_services_support import RawOverlapReranker
from handbook_bot import guide_grounding as GG
from handbook_bot import guide_inference as GI
from handbook_bot import plan_f_runtime as PF
from handbook_bot import source_scopes as SS
from handbook_bot.agents.specialists.contracts import FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask
from handbook_bot.agents.specialists.guide_evidence import GuideCapableSpecialist
from handbook_bot.knowledge_base import KnowledgeBase, assemble_corpus
from handbook_bot.sources import load_source_registry

REPO = Path(__file__).resolve().parents[1]
HANDBOOK = "uos_faculty_handbook_2025_26"
SHORT_GUIDE = GI.SHORT
GRADEBOOK = GI.GRADEBOOK
ATTENDANCE = GI.ATTENDANCE
COLLABORATE = GI.COLLABORATE
COMMUNICATION = GI.COMMUNICATION
GRADE_TRANSFER = GI.GRADE_TRANSFER
ITHENTICATE = GI.ITHENTICATE


@pytest.fixture(scope="module")
def parts():
    return assemble_corpus(load_source_registry(str(REPO / "knowledge" / "sources.json")).enabled_sources())


@pytest.fixture(scope="module")
def real_index(parts):
    return GG.GuideIndex(parts.chunks, parts.metadata, parts.source_pages, parts.records)


def _answer(index, clause, source_id):
    binding, problem = index.bind(clause, source_id)
    if binding is None:
        return None, binding, problem
    ranks = {s.key: 0 for s in index.structures_of(source_id)}
    procs = {p.key: 0 for p in index.procedures_of(source_id)}
    units = []
    for cid, meta in enumerate(index.metadata):
        if meta.get("source_id") == source_id:
            units.extend(GG.statement_units(cid, index.chunks[cid], meta, 0, binding))
    chosen, _tried = GG.select_answer(binding, units, index, procs, ranks)
    return chosen, binding, None


# ---------------------------------------------------------------------------
# A. Strong asks, weak asks, ties
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause, family, candidate", [
    ("How do I make my course private while I prepare it?", "blackboard", SHORT_GUIDE),
    ("How do I create an assignment?", "blackboard", SHORT_GUIDE),
    ("How do I add content to my course?", "blackboard", SHORT_GUIDE),
    ("How do I assign automatic zeros to missing work?", "blackboard", GRADEBOOK),
    ("How do I post grades for an assignment?", "blackboard", GRADEBOOK),
    ("How do I take attendance?", "blackboard", ATTENDANCE),
    ("How do I schedule an online class?", "blackboard", COLLABORATE),
    ("How do I create a discussion board?", "blackboard", COMMUNICATION),
    ("How do I send my final grades?", "grade_transfer", GRADE_TRANSFER),
    ("How do I view the similarity report for my paper?", "ithenticate", ITHENTICATE),
    ("How do I check my paper for plagiarism?", "ithenticate", ITHENTICATE),
])
def test_a_strong_action_object_ask_is_inferred(clause, family, candidate):
    inference = GI.infer(clause)
    assert inference.accepted and inference.reason == "accepted", inference
    assert inference.family == family and candidate in inference.candidates
    assert inference.score >= GI.ACCEPT_SCORE and inference.margin >= GI.MARGIN
    assert all(GI._FAMILY_OF[c] == family for c in inference.candidates)


@pytest.mark.parametrize("clause", [
    "I need help with grades.", "How do I use the system?", "Where is the report?", "How do I submit it?",
    "Can I collaborate with someone?", "Where do I record this?", "How do I submit grades?", "How do I send a message to my students?",
    "How do I teach online?", "How do I create a session?", "How do I create a new course?", "How do I upload a document?",
    "Can you tell me about submissions?", "What are my reports?",
])
def test_a_single_ambiguous_word_or_a_weak_ask_infers_nothing(clause):
    inference = GI.infer(clause)
    assert not inference.accepted and not inference.candidates, inference
    assert inference.reason in ("no_signal", "weak", "ambiguous", "no_object", "support") or inference.reason.startswith("intent:"), inference


def test_grades_alone_never_imply_grade_transfer_but_sending_final_grades_does():
    assert not GI.infer("How do I submit grades?").accepted
    assert not GI.infer("How do I send grades?").accepted
    assert GI.infer("How do I submit final grades?").candidates == (GRADE_TRANSFER,)


def test_two_systems_that_fit_equally_well_are_refused_by_the_margin():
    inference = GI.infer("How do I upload final grades?")              # Gradebook upload versus the Banner transfer
    assert not inference.accepted and inference.reason == "ambiguous"
    assert inference.margin is not None and inference.margin < GI.MARGIN
    assert {f for f, _s in inference.scores} == {"blackboard", "grade_transfer"}


def test_the_score_is_the_sum_of_printed_objects_plus_one_action_bonus():
    inference = GI.infer("How do I take attendance?")
    assert inference.signals == ("blackboard.attendance",) and inference.governed
    assert inference.score == 1.0 + GI.ACTION_BONUS
    alone = GI.infer("How do I use attendance?")                      # the object without a governing task verb
    assert not alone.accepted and alone.reason == "weak" and alone.score == 1.0


def test_weak_context_alone_never_infers_even_with_an_action():
    inference = GI.infer("How do I add a course?")                    # "course" is context (0.5), never a distinctive object
    assert not inference.accepted and inference.reason == "weak"


def test_thresholds_are_the_documented_values():
    assert (GI.ACCEPT_SCORE, GI.MARGIN, GI.ACTION_BONUS, GI.DISTINCTIVE) == (2.0, 1.5, 1.0, 1.0)


# ---------------------------------------------------------------------------
# B. Boundaries: policy, organizational, contact, other platforms, explicit systems
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause", [
    "What is the attendance policy?", "How many absences are allowed before a student fails?", "Do I have to take attendance?",
    "Is it mandatory to post grades within a week?", "What is the deadline to submit final grades?",
    "Am I required to use the discussion board?", "What is the penalty for plagiarism?", "How many office hours must I post?",
])
def test_a_policy_ask_is_never_inferred_into_a_tool_answer(clause):
    inference = GI.infer(clause)
    assert not inference.accepted and inference.reason in ("policy", "organizational"), inference      # "office hours": both


@pytest.mark.parametrize("clause", [
    "Who approves final grades?", "Which office checks the similarity report?", "Where is the e-learning support office?",
    "Which team helps with online classes?", "Who do I contact about the attendance tool?",
    "What is the email of the support desk for attendance?", "Where is the department office to submit final grades?",
    "How do I approve final grades as chair?",
])
def test_an_organizational_or_contact_ask_never_routes_to_a_software_guide(clause):
    inference = GI.infer(clause)
    assert not inference.accepted and inference.reason in ("organizational", "support", "policy") or inference.reason.startswith("intent:"), inference


@pytest.mark.parametrize("clause", [
    "How do I schedule a Zoom meeting for my online class?", "How do I check the similarity report in Turnitin?",
    "How do I create a discussion board in Google Classroom?", "Can SafeAssign check my paper for plagiarism?",
    "How do I share my screen in Microsoft Teams during an online session?", "How do I post grades in Moodle?",
])
def test_another_platform_named_is_never_inferred_to_be_an_enabled_guide(clause):
    inference = GI.infer(clause)
    assert not inference.accepted and inference.reason == "other_platform", inference


@pytest.mark.parametrize("clause, named", [
    ("How do I create a discussion board in Blackboard?", COMMUNICATION),
    ("Where is the similarity report in iThenticate?", ITHENTICATE),
    ("How do I send grades to Banner?", GRADE_TRANSFER),
    ("How do I open the gradebook?", GRADEBOOK),
])
def test_an_explicit_system_keeps_the_d111_path_and_is_never_inferred(real_index, clause, named):
    assert GI.infer(clause).reason == "explicit" and not GI.inferred_sources(clause)
    assert named in SS.guide_sources_named(clause, real_index.records.values())
    binding, problem = real_index.bind(clause, named)
    if binding is not None:
        assert binding.inferred is False and binding.names_bound


@pytest.mark.parametrize("clause, blocked", [
    ("How do I check my students' assignments for plagiarism?", "ithenticate"),     # iThenticate is for researchers' work
    ("Where is the similarity report for my course essays?", "ithenticate"),
    ("How do I add a journal to my research publications?", "blackboard"),          # a research journal is not the tool
    ("How do I create a journal for my paper submission?", "blackboard"),
    ("How do I send final grades to my students?", "grade_transfer"),              # grades go to students in the Gradebook
    ("How do I create a teaching assignment?", "blackboard"),
])
def test_a_collision_context_removes_the_colliding_signal(clause, blocked):
    inference = GI.infer(clause)
    assert not inference.accepted or inference.family != blocked, inference


# ---------------------------------------------------------------------------
# C. The signal table is official vocabulary; held sources are in no family
# ---------------------------------------------------------------------------
def test_every_signal_word_is_printed_in_each_guide_it_points_to(real_index):
    for signal in GI.SIGNALS:
        for guide in signal.guides:
            record = real_index.record(guide)
            declared = GG.stems(" ".join(str(getattr(record, k, "") or "") for k in ("title", "notes", "system")))
            vocabulary = real_index.vocabulary(guide) | declared         # the registry's declared words ("final grades", D-114)
            for word in signal.words:
                word = GG.stem(word) if signal.phrase else word        # a phrase keeps its printed words
                assert ({word} | GG._SYNONYMS.get(word, frozenset())) & vocabulary, (signal.key, guide, word)
        assert signal.provenance, signal.key


def test_the_families_are_the_seven_enabled_faculty_guides_and_nothing_else(real_index):
    guides = [g for ids in GI.FAMILIES.values() for g in ids]
    assert len(guides) == len(set(guides)) == 7
    for guide in guides:
        record = real_index.record(guide)
        assert record is not None and SS.usable_record(record) and SS.faculty_audience(record), guide
    assert "uos_performance_appraisal_phase2_guide" not in guides


@pytest.mark.parametrize("clause", [
    "How do I start the Phase 2 progress review?", "Where do I submit my performance appraisal?",
    "How do I create grade categories?", "How do I create a test with question pools?",
])
def test_held_and_absent_guides_are_never_inferred(clause):
    inference = GI.infer(clause)
    assert "uos_performance_appraisal_phase2_guide" not in inference.candidates
    assert not any("categor" in c or "assignments_tests" in c for c in inference.candidates)


def test_the_inference_record_holds_no_question_text():
    clause = "Where can I find the similarity report for my zebra manuscript?"
    record = json.dumps(GI.infer(clause).to_dict()) + json.dumps(GI.infer_question(clause, [clause]).to_dict())
    assert "zebra" not in record and "manuscript" not in record


def test_inference_is_deterministic_and_pure():
    clause = "How do I export the attendance sheet?"
    assert GI.infer(clause) == GI.infer(clause)
    GI.infer.cache_clear()
    assert GI.infer(clause).to_dict() == GI.infer(clause).to_dict()


# ---------------------------------------------------------------------------
# D. Inference is not evidence
# ---------------------------------------------------------------------------
def test_an_inferred_binding_skips_only_the_system_check_every_asked_word_stays_a_term(real_index):
    clause = "How do I check the similarity report?"
    binding, problem = real_index.bind(clause, ITHENTICATE)
    assert binding is not None and binding.inferred and not binding.names_bound, problem
    assert {GG.stem(w) for w in ("check", "similarity", "report")} <= set(binding.terms)
    direct, refusal = GG.bind(clause, real_index.record(ITHENTICATE), real_index.vocabulary(ITHENTICATE))
    assert direct is None and "does not name the system" in refusal              # bind alone never infers


def test_an_inferred_guide_still_needs_every_asked_word_in_its_text(real_index):
    chosen, binding, _ = _answer(real_index, "How do I check the similarity report of my zebra manuscript?", ITHENTICATE)
    assert binding is not None and binding.inferred and chosen is None


def test_a_non_candidate_guide_is_never_bound_by_inference(real_index):
    clause = "How do I check the similarity report?"
    for guide in (SHORT_GUIDE, GRADEBOOK, GRADE_TRANSFER, COLLABORATE):
        binding, problem = real_index.bind(clause, guide)
        assert binding is None and "does not name the system" in problem, guide


def test_the_adapter_recomputes_the_inference_and_refuses_a_forged_guide_answer(real_index, parts):
    cid = next(i for i, m in enumerate(parts.metadata) if m.get("source_id") == ITHENTICATE and "similarity report" in parts.chunks[i]
               and not m.get("source_conflict"))
    quote = parts.chunks[cid]
    for clause in ("Where is the report?", "How do I check the report?"):           # no confident inference
        problem = GG.verify_answer(clause, GG.FORM_STATEMENT, [(cid, quote, parts.metadata[cid].get("page"))], None, real_index)
        assert problem and "does not name the system" in problem, clause


def test_the_authority_guard_accepts_an_inferred_guide_only_for_its_own_clause(parts):
    cid = next(i for i, m in enumerate(parts.metadata) if m.get("source_id") == ITHENTICATE and "similarity report" in parts.chunks[i])
    meta = parts.metadata[cid]
    assert SS.source_authority_problem("How do I check the similarity report?", meta, GI.inferred_sources("How do I check the similarity report?")) is None
    assert "does not name the system" in SS.source_authority_problem("Where is the report?", meta, GI.inferred_sources("Where is the report?"))
    assert "does not name the system" in SS.source_authority_problem("How do I check the similarity report?", meta)    # nothing passed

    class _F:
        metadata = {"chunk_id": cid}
    problem = PF.guide_evidence_problem("Where is the report?", "x", _F(), parts.chunks, parts.metadata)
    assert problem and "does not name the system" in problem


# ---------------------------------------------------------------------------
# E. Conflicts, wrong sources, unsupported asks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause", [
    "How many documents can I check for similarity per year?", "What is the maximum number of documents I can check for plagiarism?",
])
def test_the_quota_conflict_stays_blocked_for_an_inferred_clause(real_index, clause):
    chosen, binding, problem = _answer(real_index, clause, ITHENTICATE)
    assert chosen is None or not any(p.chunk_id for p in chosen.parts if "30 Documents" in p.quote), (clause, chosen)


def test_an_inferred_guide_not_approved_for_the_specialist_is_refused(real_index, parts):
    from handbook_bot.agents.specialists.guide_evidence import GuideEvidence
    evidence = GuideEvidence(lambda t: None, lambda pairs: [0.0] * len(pairs), None, parts.chunks, parts.metadata, real_index)
    ok, refused = evidence.bindings("How do I create a discussion board?", {HANDBOOK, ITHENTICATE})       # Research's approved set
    assert not ok and refused == [{"source_id": COMMUNICATION, "reason": "not approved for this specialist (D-110)"}]
    ok, refused = evidence.bindings("How do I create a discussion board?", {HANDBOOK, COMMUNICATION})
    assert [b.source_id for b in ok] == [COMMUNICATION] and ok[0].inferred


@pytest.mark.parametrize("clause", [
    "What is the capital of France?", "How do I book a meeting room?", "How do I renew my residence visa?",
    "Where do I park on campus?", "How do I apply for annual leave?",
])
def test_an_unsupported_ask_infers_nothing(clause):
    assert not GI.infer(clause).accepted


# ---------------------------------------------------------------------------
# F. The runtime
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def kb(parts):
    embedder = FakeEmbedder()
    embeddings = embedder.encode(parts.chunks)
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embeddings)
    primary = parts.records[0]
    stats = {"total_chunks": len(parts.chunks), "total_pages": sum(len(p) for p in parts.source_pages.values()),
             "source": {"source_id": primary.source_id, "registered": True, "section_records": parts.primary_section_records},
             "sources": parts.per_source}
    return KnowledgeBase(pdf_file="corpus", embedder=embedder, reranker=RawOverlapReranker(), groq_client=None,
                         pages=parts.source_pages[primary.source_id], chunks=parts.chunks, metadata=parts.metadata,
                         embeddings=embeddings, index=index, stats=stats, source=primary, sources=parts.records,
                         source_pages=parts.source_pages, corpus_key=parts.key)


@pytest.fixture(scope="module")
def runtime(kb):
    rt, build = PF.build_runtime(kb, guide_scopes=True)
    assert build.ok
    return rt


def test_the_guide_registry_is_selected_by_an_accepted_inference_only(runtime):
    registry, record = runtime.select_registry("How do I take attendance?")
    assert registry is runtime.guide_registry and record["selected"] and record["named"] == []
    assert record["inferred"] == [ATTENDANCE, COLLABORATE] and record["inference"]["accepted"]
    registry, record = runtime.select_registry("How do I submit grades?")
    assert registry is runtime.registry and not record["selected"] and record["inferred"] == []
    assert record["inference"]["lead"]["reason"] == "weak"


def test_a_clause_of_a_multi_part_question_can_be_inferred(runtime):
    _registry, record = runtime.select_registry("What is the attendance policy? Also, how do I check the similarity report?")
    assert record["selected"] and record["inferred"] == [ITHENTICATE]
    accepted = [p["accepted"] for p in record["inference"]["parts"]]
    assert accepted[0] is False and accepted.count(True) == 1          # the whole question asks a policy; one clause infers


def test_the_trace_records_the_inference_with_signal_keys_and_scores_only(runtime):
    attempt = runtime.attempt("How do I export the attendance sheet for my zebra class?")
    entries = [e for e in attempt.trace if e.agent == "guide_inference"]
    assert entries and entries[0].decision in ("accepted", "refused:weak", "refused:ambiguous")
    detail = entries[0].detail
    assert {"family", "candidates", "score", "runner_up", "margin", "signals", "reason"} <= set(detail)
    assert "zebra" not in json.dumps([e.detail for e in attempt.trace], default=str)


def test_an_inferred_question_with_no_specialist_goes_to_the_guides_owner_and_serves_only_a_guide_answer(runtime):
    for question in ("How do I allow recording downloads?", "How do I exclude a resource from the similarity report?",
                     "How do I create a learning module?"):
        attempt = runtime.attempt(question)
        decisions = [e.decision for e in attempt.trace if e.agent == "guide_inference"]
        if "owner_assigned" not in decisions:
            continue
        if attempt.outcome is PF.PlanFOutcome.ACCEPTED:
            records = [((i.get("meta") or {}).get("plan_f") or {}).get("guide") for i in attempt.result.items]
            assert records and all(isinstance(r, dict) for r in records), question
        else:
            assert "declined" in decisions and attempt.outcome is PF.PlanFOutcome.NOT_APPLICABLE, (question, attempt.reason)


def test_an_unnamed_question_the_coordinator_routes_keeps_its_owner(runtime):
    attempt = runtime.attempt("How do I create a journal for my students?")       # routed to Research by the frozen Coordinator
    assert not any(e.agent == "software_navigation" for e in attempt.trace)
    assert not any(e.decision == "owner_assigned" for e in attempt.trace if e.agent == "guide_inference")


class _Base:
    def __init__(self, sid):
        self.specialist_id = sid

    def run(self, task):
        clause = task.context["focus_clauses"][0]
        handoff = HandoffRequest(from_specialist=self.specialist_id, requested_specialist=SpecialistId.INSTITUTIONAL, reason="location clause 0",
                                 task=None, metadata={"clause_index": 0, "kind": "location"})
        return SpecialistFindings(specialist_id=self.specialist_id, task_id=task.task_id, status=FindingStatus.OUT_OF_SCOPE, findings=[],
                                  summary="handed off", handoff_requests=[handoff],
                                  metadata={"clauses": [{"index": 0, "clause": clause, "outcome": "handed_off", "target": "institutional"}]})


class _InferredNoAnswer:
    def answer(self, clause, route, approved):
        return None, {"applies": True, "refused": [], "bindings": [{"source_id": SHORT_GUIDE, "intent": "location", "inferred": True}]}


def test_an_inferred_binding_never_removes_a_handoff_to_institutional_navigation():
    wrapper = GuideCapableSpecialist.__new__(GuideCapableSpecialist.wrap.__globals__["GuideCapableTeaching"])
    wrapper.base, wrapper.evidence = _Base(SpecialistId.TEACHING), _InferredNoAnswer()
    wrapper.approved_sources = frozenset({HANDBOOK, SHORT_GUIDE})
    wrapper.approved_guides = frozenset({SHORT_GUIDE})
    clause = "Where is the course image?"
    result = wrapper.run(SpecialistTask(task_id="t-1", question=clause, specialist_id=SpecialistId.TEACHING, domain="teaching", intent="policy",
                                        context={"full_question": clause, "focus_clauses": [clause]}))
    assert [h.requested_specialist for h in result.handoff_requests] == [SpecialistId.INSTITUTIONAL]


def test_a_misspelled_system_name_is_corrected_and_takes_the_explicit_path(runtime):
    attempt = runtime.attempt("How do I check the similarity report in iThentikate?")
    assert any(e.agent == "system_name_spelling" for e in attempt.trace)
    scope = next(e for e in attempt.trace if e.agent == "source_scope")
    assert scope.detail["named"] == [ITHENTICATE] and not any(e.agent == "guide_inference" for e in attempt.trace)
    assert attempt.question == "How do I check the similarity report in iThentikate?"


def test_a_misspelled_ordinary_word_does_not_stop_a_strong_inference():
    inference = GI.infer("How do I chek the similarty report?")       # neither word is a system name: no correction
    assert not inference.accepted                                       # "similarty" is no printed signal: refused, never guessed
    assert GI.infer("How do I chek the similarity report?").accepted   # the printed object still carries the inference


def test_guide_scopes_off_never_infers(kb):
    rt, _ = PF.build_runtime(kb, guide_scopes=False)
    registry, record = rt.select_registry("How do I take attendance?")
    assert registry is rt.registry and record == {"enabled": False, "selected": False, "named": []}
    attempt = rt.attempt("How do I take attendance?")
    assert not any(e.agent in ("guide_inference", "source_scope") for e in attempt.trace)


def test_guide_inference_is_imported_lazily():
    code = ("import sys; import handbook_bot.plan_f_runtime, handbook_bot.guide_grounding, handbook_bot.source_scopes, "
            "handbook_bot.agents.specialists.guide_evidence; print('handbook_bot.guide_inference' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().splitlines()[-1] == "False", out.stderr[-2000:]


def test_the_diagnostics_carry_the_inference_and_the_developer_panel_shows_it(kb, monkeypatch):
    from handbook_bot import config
    from ui import pipeline, presentation
    monkeypatch.setattr(config, "PLAN_F_ENABLED", True)
    monkeypatch.setattr(config, "PLAN_F_SHADOW", False)
    monkeypatch.setattr(config, "PLAN_F_GUIDE_SCOPES", True)
    pipeline.reset_plan_f_state()
    try:
        _res, d = pipeline.ask_with_diagnostics(kb, "How do I take attendance?")
    finally:
        pipeline.reset_plan_f_state()
    assert d["guide_sources_named"] == [] and d["guide_sources_inferred"] == [ATTENDANCE, COLLABORATE]
    assert d["guide_inference"]["accepted"] and d["guide_inference"]["family"] == "blackboard"
    assert presentation._source_scope_label(d).startswith("handbook + inferred guides: ")
    assert presentation._inference_label(d).startswith("accepted: blackboard 2.0")
    refused = dict(d, guide_inference={"accepted": False, "reason": "weak", "family": "grade_transfer", "score": 1.0})
    assert presentation._inference_label(refused) == "refused: weak (grade_transfer 1.0)"


# ---------------------------------------------------------------------------
# G. Grounding refusals added in PF-5C (each only refuses)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("heading, verb", [
    ("Follow these steps to set your attendance reports", "set"), ("D- Checking Session Attendance", "check"),
    ("How to Send Grades to Banner as instructor?", "send"), ("A. Create New Scheduled Sessions", "creat"),
    ("Session Details", None), ("3) Student Grid", None),
])
def test_the_purpose_verb_of_a_heading(heading, verb):
    assert GG.purpose_verb(heading) == verb


def test_a_procedure_whose_heading_names_another_action_is_refused(real_index):
    chosen, binding, _ = _answer(real_index, "How do I mark attendance for my students in Blackboard?", ATTENDANCE)
    assert binding is not None and (chosen is None or "set your attendance reports" not in (chosen.heading or ""))
    chosen, binding, _ = _answer(real_index, "How do I mark attendance in Blackboard Ultra?", COLLABORATE)
    assert chosen is None or chosen.heading != "D- Checking Session Attendance"


@pytest.mark.parametrize("clause, head", [
    ("Where is the student grid in Blackboard?", "grid"), ("Where can I find the similarity report in iThenticate for my paper?", "report"),
    ("Where in Blackboard do I see the attendance overview of my Collaborate sessions?", "overview"),
    ("Where are the recordings of my Collaborate sessions?", "record"), ("Where do I go after logging in to Blackboard?", None),
])
def test_the_object_a_where_ask_places(real_index, clause, head):
    source = ITHENTICATE if "similarity" in clause else COLLABORATE if "Collaborate" in clause else GRADEBOOK
    binding, problem = real_index.bind(clause, source)
    assert binding is not None, problem
    assert binding.info.get("location_head") == head


NOT_PLACED = "the statement does not place the asked object (it is named only by a heading or as the place of something else)"


@pytest.mark.parametrize("quote", [
    "3) Student Grid: 3. If a submission exists, select View in the menu to view the student's work.",      # only the heading names it
    "In the student grid view, select Filter to open the Filter Criteria panel.",                           # the place of something else
])
def test_a_location_that_does_not_place_the_asked_object_is_refused(real_index, quote):
    binding, _ = real_index.bind("Where is the student grid in Blackboard?", GRADEBOOK)
    assert GG.statement_problem(binding, quote) == NOT_PLACED
    chosen, _binding, _ = _answer(real_index, "Where is the student grid in Blackboard?", GRADEBOOK)
    assert chosen is None or not any(q in chosen.parts[0].quote for q in ("If a submission exists", "select Filter"))


@pytest.mark.parametrize("clause, quote, source", [
    ("Where is the gradebook icon in Blackboard?", "2. Course gradebook: - Select the Gradebook icon on the navigation bar on the top right corner.",
     GRADEBOOK),
    ("Where in Blackboard do I see the attendance overview of my Collaborate sessions?",
     "D- Checking Session Attendance: - On the attendance panel, select overview at the top right.", COLLABORATE),     # a heading qualifier
    ("Where do I find Collaborate sessions in Blackboard Ultra?",
     "From The Details & Actions Menu, click on “View All Sessions” under Blackboard Collaborate.", COLLABORATE),     # a quoted control
])
def test_a_location_that_places_the_asked_object_is_kept(real_index, clause, quote, source):
    binding, problem = real_index.bind(clause, source)
    assert binding is not None and GG.statement_problem(binding, quote) is None, problem


# ---------------------------------------------------------------------------
# H. Adversarial review findings (each reproduced by an independent skeptic, each fixed by refusing more)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause, reason", [
    # the asker's or a colleague's own attendance or accommodation (HR, housing) is not a course's
    ("How do I check faculty attendance?", "role_object"), ("How do I check my attendance?", "role_object"),
    ("How do I export instructor attendance?", "role_object"), ("How do I check my employee attendance record?", "role_object"),
    ("How do I create accommodations for faculty members?", "role_object"),
    # someone other than the asker does the asked thing
    ("How do students check their attendance?", "other_actor"), ("How do students see posted grades?", "other_actor"),
    ("Can a TA take attendance for me?", "other_actor"), ("Can my secretary send announcements for me?", "other_actor"),
    # permission, obligation, advice, misconduct, appeal
    ("Must attendance be part of the overall grade?", "policy"), ("Is it ok to change the default grading schema?", "policy"),
    ("May I delete a student's discussion response?", "policy"), ("Do I need to take attendance in every class?", "policy"),
    ("Is the instructor responsible for taking attendance?", "policy"), ("Is recording online classes required?", "policy"),
    ("What is the maximum similarity index allowed?", "policy"), ("Where do I report plagiarism?", "policy"),
    ("How do I submit a final grade appeal?", "policy"), ("What content should a course syllabus include?", "policy"),
    # help, support, troubleshooting and information requests
    ("Where can I get help with the student grid?", "support"), ("Where do I go to get help with the roster?", "support"),
    ("Where can I find out more about the similarity index?", "support"),
    # other platforms
    ("How do I take attendance using Google Forms?", "other_platform"), ("How do I check similarity using Unicheck?", "other_platform"),
])
def test_review_boundaries_refuse_before_scoring(clause, reason):
    inference = GI.infer(clause)
    assert not inference.accepted and inference.reason == reason, inference


@pytest.mark.parametrize("clause", [
    "Can students see the roster?", "Can students preview the course?", "Can I delete students' discussion responses?",
    "Is uploading the syllabus required?",
])
def test_yes_no_and_definition_asks_are_never_inferred(clause):
    inference = GI.infer(clause)
    assert not inference.accepted, inference


@pytest.mark.parametrize("clause", [
    "Where is the content?", "Where is my content?", "Where is the roster?", "Where are the conversations?", "Where is attendance?",
])
def test_a_where_ask_with_one_generic_word_infers_nothing(clause):
    assert not GI.infer(clause).accepted


@pytest.mark.parametrize("clause", [
    "How do I learn the modules?", "Where can I see my child's courses?", "Finally, how do I send grades?",
    "How do I submit final exam grades?", "How do I merge my courses?", "How do I organize a panel discussion?",
    "How do I record attendance at a faculty meeting?", "How do I register for the online training sessions?",
    "How do I add content to my course file?", "How do I set up my accommodation?", "How do I approve my final grades?",
    "How do I send students their final grades?", "Where can I find a list of approved journals?", "How do I create a journal club?",
    "How do I make an assignment submission for a student?",
])
def test_a_phrase_only_matches_its_printed_words_and_a_colliding_context_blocks_its_signal(clause):
    assert not GI.infer(clause).accepted, GI.infer(clause)


@pytest.mark.parametrize("clause, family", [
    ("How do I upload final grades?", "ambiguous"), ("How do I post final grades?", "ambiguous"),
])
def test_gradebook_and_grade_transfer_verbs_on_final_grades_tie(clause, family):
    assert GI.infer(clause).reason == family


@pytest.mark.parametrize("clause, quote_head, source", [
    ("How do I see my attendance report in Blackboard?", "Follow these steps to set your attendance reports", ATTENDANCE),
    ("How do I create content in Blackboard?", "Create an Assignment", SHORT_GUIDE),
    ("How do I create a group in Blackboard?", "Create an Assignment", SHORT_GUIDE),
    ("How do I assign a grade in Blackboard?", "Assign automatic zeros", GRADEBOOK),
])
def test_a_heading_that_changes_what_is_only_seen_or_acts_on_another_object_is_refused(real_index, clause, quote_head, source):
    chosen, binding, problem = _answer(real_index, clause, source)
    assert binding is not None, problem
    assert chosen is None or chosen.heading != quote_head, (clause, chosen.heading)


def test_a_heading_naming_the_object_inside_a_compound_is_kept(real_index):
    chosen, binding, problem = _answer(real_index, "How do I upload grades to Blackboard?", GRADEBOOK)
    assert chosen is not None and chosen.heading == "Upload a file to the gradebook", problem


@pytest.mark.parametrize("clause", [
    "How do I send an announcement to all faculty in Blackboard?", "How do I send an announcement to other faculty members in Blackboard?",
])
def test_a_faculty_group_named_as_recipient_stays_a_term(real_index, clause):
    binding, problem = real_index.bind(clause, SHORT_GUIDE)
    assert binding is not None and GG.stem("faculty") in binding.terms, problem
    chosen, _binding, _ = _answer(real_index, clause, SHORT_GUIDE)
    assert chosen is None


def test_the_faculty_asker_is_still_bound_by_the_audience(real_index):
    binding, _ = real_index.bind("How do I, as an instructor, send an announcement in Blackboard?", SHORT_GUIDE)
    assert binding is not None and binding.actor_bound and GG.stem("instructor") not in binding.terms


def test_a_statement_about_what_students_can_do_does_not_answer_what_the_asker_may_do(real_index):
    binding, _ = real_index.bind("Can I delete students' discussion responses in Blackboard?", COMMUNICATION)
    quote = "Students can delete only their own discussions, responses, and replies."
    assert binding is not None and GG.statement_problem(binding, quote) == "the statement is about what students can do, not about the asker"
    student, _ = real_index.bind("Can students delete their own discussion responses in Blackboard?", COMMUNICATION)
    assert student is not None and GG.statement_problem(student, quote) is None


# ---------------------------------------------------------------------------
# I. Second adversarial round: the strict evidence profile of an inferred clause and the named-path refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause, source, heading", [
    ("How do I create an assignment?", SHORT_GUIDE, "Create an Assignment"),
    ("How do I add a learning module?", SHORT_GUIDE, "Add a Learning Module or Folder"),
    ("How do I assign automatic zeros to missing work?", GRADEBOOK, "Assign automatic zeros"),
    ("How do I send my final grades?", GRADE_TRANSFER, "How to Send Grades to Banner as instructor?"),
])
def test_an_inferred_clause_is_answered_under_a_heading_that_names_the_asked_task(real_index, clause, source, heading):
    chosen, binding, problem = _answer(real_index, clause, source)
    assert binding is not None and binding.inferred, problem
    assert chosen is not None and chosen.heading == heading and chosen.form in (GG.FORM_PROCEDURE, GG.FORM_SECTION)


@pytest.mark.parametrize("clause, source", [
    ("How do I add students to a learning module?", SHORT_GUIDE),          # a person the heading does not name
    ("How do I create a discussion topic?", COMMUNICATION),                 # "Delete discussion topics ..." names another action
    ("How do I share attendance with the LMS?", ATTENDANCE),                # "... to set your attendance reports"
    ("How do I view my class roster?", SHORT_GUIDE),                        # "Course View" names no action
])
def test_an_inferred_clause_is_never_answered_under_another_task_or_by_a_statement(real_index, clause, source):
    chosen, binding, _problem = _answer(real_index, clause, source)
    assert chosen is None or not binding.inferred, (clause, chosen and chosen.heading)


def test_an_inferred_binding_never_accepts_a_statement(real_index):
    binding, _ = real_index.bind("How do I check the similarity report?", ITHENTICATE)
    assert binding.inferred and GG.statement_problem(binding, "The similarity index and originality report can be found under the “Documents”")
    value, problem = GG.bind("What is the maximum file size for a similarity check?", real_index.record(ITHENTICATE),
                             real_index.vocabulary(ITHENTICATE), inferred=True)       # never produced by inference; the rule holds anyway
    assert value is not None, problem
    assert GG.statement_problem(value, "Files must be less than 100MB.") == (
        "an inferred clause is answered only by a complete procedure or list headed by the asked task, never by a statement")
    assert GG.inferred_problem(binding, ["Checking Similarity Report"]) is None
    assert GG.inferred_problem(binding, ["How to upload a document"]) is not None


@pytest.mark.parametrize("clause", [
    "What can I add to a learning module?", "Where can I find the similarity report?", "Which option do I select to post grades?",
    "How do I send an announcement to one student?", "How do I assign zeros to one student?", "How do I check similarity for a professor?",
    "How do I send final grades to the instructor?", "How do I add a learning module to the curriculum?",
    "How do I assign automatic zeros to a group?",
])
def test_lists_places_controls_and_scoped_or_delegated_tasks_are_not_inferred(clause):
    assert not GI.infer(clause).accepted, GI.infer(clause)


def test_a_destructive_heading_is_not_overridden_by_a_passing_mention_of_the_asked_action(real_index):
    chosen, _binding, _ = _answer(real_index, "How do I create a discussion for instructors in Blackboard?", COMMUNICATION)
    assert chosen is None or not chosen.heading.startswith("Delete discussion topics")


def test_a_set_action_under_an_add_heading_keeps_the_text_override(real_index):
    chosen, _binding, problem = _answer(real_index, "How do I set accommodations for a student in the Blackboard gradebook?", GRADEBOOK)
    assert chosen is not None and chosen.heading.startswith("Add accommodations"), problem


def test_a_role_specific_procedure_needs_that_role(real_index):
    for clause in ("How do I revise grades in the Grade Transfer Dashboard?", "How do I revise my grades in Banner as instructor?"):
        chosen, _binding, _ = _answer(real_index, clause, GRADE_TRANSFER)
        assert chosen is None or "Chair/Dean" not in (chosen.heading or ""), clause
    chosen, _binding, problem = _answer(real_index, "How does a chair approve grades in the Grade Transfer Dashboard?", GRADE_TRANSFER)
    assert chosen is not None and "Chair/Dean" in chosen.heading, problem


def test_a_negated_ask_needs_negated_evidence(real_index):
    binding, _ = real_index.bind("Which option do I select to not allow recording downloads in Blackboard Collaborate?", COLLABORATE)
    assert GG.statement_problem(binding, "To allow downloading the recordings select “Allow recording downloads” option.") \
        == "the clause is negated; the evidence states no negation"


def test_a_see_ask_needs_a_heading_that_opens_or_shows(real_index):
    for clause in ("So, how do I see the attendance report in Blackboard?", "How do I see the student grid in Blackboard?"):
        binding, _ = real_index.bind(clause, GRADEBOOK if "grid" in clause else ATTENDANCE)
        assert binding.info.get("frame_verb") == "see"
        assert GG.task_problem(binding, ["Student Grid"], "1. Select a cell to begin grading.") is not None
        assert GG.task_problem(binding, ["Checking Session Attendance"], "x") is None


def test_a_bound_in_another_unit_does_not_answer_a_value_ask(real_index):
    binding, _ = real_index.bind("What is the maximum number of pixels for the course image in Blackboard?", SHORT_GUIDE)
    quote = ("Select Edit display settings under Course Image from the Details & Actions pane. Your image must be at least 1200 x 240 "
             "pixels and it can't be larger than 25 KB.")
    assert binding is not None and binding.maximum and GG.statement_problem(binding, quote) == "the stated bound governs a quantity in another unit"


def test_a_yes_no_statement_only_about_students_does_not_answer_the_asker(real_index):
    binding, _ = real_index.bind("Can I access a private course in Blackboard?", SHORT_GUIDE)
    quote = "Students see private courses in their course lists, but they cannot access them."
    assert GG.statement_problem(binding, quote) == "the statement is about what students can do, not about the asker"
    kept = "If a student was absent with an acceptable excuse, you may change their attendance record to excused."
    binding, _ = real_index.bind("Can I change a student's attendance record to excused in Blackboard?", ATTENDANCE)
    assert GG.statement_problem(binding, kept) is None


def test_the_object_of_where_do_x_appear(real_index):
    binding, _ = real_index.bind("Where do due dates appear in Blackboard?", SHORT_GUIDE)
    assert binding.info.get("location_head") == "dat"


# ---------------------------------------------------------------------------
# The frozen PF-5C fresh held-out set
# ---------------------------------------------------------------------------
def test_the_pf5c_fresh_held_out_set_is_frozen_and_never_edited():
    path = REPO / "qa" / "post_freeze_pf5c" / "fresh_held_out_questions.jsonl"
    labels = json.loads((REPO / "qa" / "post_freeze_pf5c" / "fresh_held_out_labels.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == labels["frozen"]["sha256"] == (
        "3c8325a7e306b4d300d09d581185160e02db655d579317c6136033d95bcde4ad")
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(questions) == labels["frozen"]["questions"] == len(labels["labels"]) == 60
    assert [q["id"] for q in questions] == sorted(labels["labels"])
    summary = labels["summary"]
    assert summary["yes"] + summary["partial"] + summary["no"] == 60
