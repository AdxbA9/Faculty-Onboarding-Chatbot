"""
Runtime integration phase I-4: findings-aware verification (``handbook_bot/plan_f_verification.py``) and its seam
in the adapter (``plan_f_runtime.PlanFRuntime.attempt``) and the pipeline (``ui/pipeline.py``).

The verifier is exercised on structured specialist results built from the real handbook pages (production chunking
and metadata, fake embedder, stand-in reranker, no model, no network), mostly as single-field mutations of a correct
finding: wrong contact field, wrong office, wrong page, wrong unit noun, contradicted category modifier, missing
condition, wrong relation, negated predicate, enumerative ask, quote spanning two provisions, conflicting value in a
second finding, partial or handoff status, invalid evidence reference, raising verifier. The four frozen specialists
are not modified; the legacy Verifier is reused unchanged through the I-1 guards.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

import test_plan_f_runtime as T
from handbook_bot import config
from handbook_bot import plan_f_runtime as PF
from handbook_bot import plan_f_verification as V
from handbook_bot.agents.specialists.contracts import Finding, FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask
from handbook_bot.plan_f_verification import Completeness as C
from handbook_bot.plan_f_verification import ReasonCode as R
from handbook_bot.plan_f_verification import RequestKind as K
from ui import pipeline

REPO = Path(__file__).resolve().parents[1]
HR_PHONE = T.HR_PHONE
HR_KNOWN_WRONG = T.HR_KNOWN_WRONG
ANNUAL_LEAVE = T.ANNUAL_LEAVE
SICK = "How many days of sick leave with full pay are faculty entitled to?"
MATERNITY = "How many days of maternity leave are female faculty entitled to?"
OFFICE_HOURS = "How many office hours must faculty hold?"
THESES = "How many theses can a faculty member supervise at once?"
PETTY = "How much petty cash advance can a principal investigator receive?"
CEILING = "Who determines the funding ceiling for research grants?"
WLAM = "What is the Workload Allocation Model?"
CONSULTANCY = "How many days per week can faculty spend on consultancy?"
HR_PROVIDES = "What does HR provide to new faculty members?"


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


@pytest.fixture(autouse=True)
def _legacy_mode(monkeypatch):
    monkeypatch.setattr(config, "PLAN_F_ENABLED", False)
    monkeypatch.setattr(config, "PLAN_F_SHADOW", False)
    pipeline.reset_plan_f_state()
    yield
    pipeline.reset_plan_f_state()


def quote_of(kb, cid, start, end=None):
    """The exact substring of chunk ``cid`` from ``start`` to the end of ``end`` (or to the end of the chunk)."""
    chunk = kb.chunks[cid]
    i = chunk.index(start)
    j = len(chunk) if end is None else chunk.index(end, i) + len(end)
    return chunk[i:j]


def located(kb, start, end=None, chunk_type="paragraph"):
    """The first chunk (of the preferred type) that holds ``start`` and, after it, ``end``, with the exact quote between them."""
    hits = [i for i, text in enumerate(kb.chunks) if start in text and (end is None or end in text[text.index(start):])]
    assert hits, (start, end)
    wanted = [i for i in hits if kb.metadata[i].get("chunk_type") == chunk_type]
    cid = (wanted or hits)[0]
    return cid, quote_of(kb, cid, start, end)


def task_for(question, specialist):
    return SpecialistTask("task-1", question, specialist_id=specialist, context={"focus_clauses": [question]})


def retrieval(kb, question, specialist, cid, quote, *, requested=("the asked specifics stated with the answer",), relation=None, **kwargs):
    """A supported single-clause result of a retrieval specialist whose finding covers every requested detail."""
    task = task_for(question, specialist)
    return task, T.retrieval_result(kb, task, cid, quote, requested=requested, covered=requested, relation=relation, **kwargs)


def institutional(kb, question, cid, quote, *, office_id, ask_kind="channel", tier="E1", offices=None, field="telephone", value=None,
                  entry="complete_row", row_complete=True, binding=None, status=FindingStatus.SUPPORTED, handoffs=(), extra_meta=None,
                  reports=None, missing=()):
    """A supported Institutional result: a directory row (E1) or a statement (E2 / stated fact)."""
    task = task_for(question, "institutional")
    meta = {"office_id": office_id, "office_name": office_id.replace("_", " ").title(), "evidence_tier": tier}
    if tier == "E1":
        values = {field: value} if value else {}
        meta.update({"directory_entry": entry, "row_complete": row_complete, "contact_fields": [field], "contact_values": values})
    else:
        meta.update({"binding": binding or "named subject"})
    meta.update(extra_meta or {})
    finding = T.finding_for(kb, cid, quote, task, **meta)
    report = {"index": 0, "clause": question, "outcome": "read", "ask_kind": ask_kind, "evidence_tier_reached": tier,
              "offices": list(offices if offices is not None else [office_id])}
    return task, SpecialistFindings(specialist_id=SpecialistId.INSTITUTIONAL, task_id=task.task_id, status=status, findings=[finding],
                                    summary="stub", missing=list(missing), handoff_requests=list(handoffs),
                                    metadata={"clauses": reports if reports is not None else [report]})


def verify(kb, task, findings):
    return V.verify_findings(task.context["focus_clauses"][0], findings, kb.chunks, kb.metadata)


def stub_attempt(kb, specialist, result_fn, question):
    """Run the adapter on a stub registry whose ``specialist`` returns ``result_fn(task)``."""
    registry = T.stub_registry(**{specialist: result_fn})
    return PF.PlanFRuntime(registry, kb.chunks, kb.metadata).attempt(question)


# ---------------------------------------------------------------------------
# Contract: result shape, codes, determinism, hygiene
# ---------------------------------------------------------------------------
def test_reason_codes_are_the_project_vocabulary():
    assert {c.value for c in R} == {
        "ACCEPTED", "UNSUPPORTED_STATUS", "PARTIAL_FINDING", "NOT_FOUND", "OUT_OF_SCOPE", "HANDOFF_REQUIRED", "MULTIPART_UNSUPPORTED",
        "GENERAL_ASK_MISMATCH", "UNRECOGNISED_ASK", "SUBJECT_MISMATCH", "RELATION_MISMATCH", "REQUESTED_DETAIL_MISSING", "CONDITION_MISMATCH",
        "MODIFIER_MISMATCH", "ENUMERATION_INCOMPLETE", "EXCESS_EVIDENCE", "VALUE_CONFLICT", "INVALID_EVIDENCE_REFERENCE", "VERIFICATION_ERROR",
    }
    assert V.PHASE == "I-4" and V.VERSION == "plan-f-findings-verification-1"


def test_verification_result_is_immutable_serialisable_and_text_free(kb):
    cid = T.chunk_index(kb, "Human Resources +971 6 5050023")
    task, findings = institutional(kb, HR_PHONE, cid, "Human Resources +971 6 5050023 +971 6 5585200", office_id="human_resources", value="+971 6 5050023")
    verdict = verify(kb, task, findings)
    assert isinstance(verdict, V.FindingsVerification) and verdict.accepted and verdict.safe_to_render
    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.accepted = False                                                # type: ignore[misc]
    text = json.dumps(verdict.to_dict())
    assert HR_PHONE not in text and "5050023" not in text and "Human Resources" not in text
    assert verdict.to_dict()["reason"] == "ACCEPTED" and verdict.to_dict()["request_kind"] == "contact"
    assert set(verdict.to_dict()) >= {"accepted", "reason", "detail", "status", "specialist_id", "request_kind", "subject_match", "relation_match",
                                      "detail_coverage", "evidence_integrity", "completeness", "safe_to_render", "accepted_finding_ids", "dropped",
                                      "verdicts", "version"}
    assert "ms" not in verdict.to_dict() and verdict.ms >= 0.0                     # the wall time stays off the deterministic record


def test_verification_is_deterministic_and_never_mutates_the_findings(kb):
    cid, quote = located(kb, "• Faculty are entitled to sick leave with full pay", "per contract year.")
    task, findings = retrieval(kb, SICK, "faculty_services", cid, quote, requested=("number or amount", "eligibility condition"))
    before = json.dumps(findings.to_dict(), default=str)
    a, b = verify(kb, task, findings), verify(kb, task, findings)
    strip = lambda d: {k: v for k, v in d.items() if k != "ms"}
    assert strip(a.to_dict()) == strip(b.to_dict()) and a.accepted
    assert json.dumps(findings.to_dict(), default=str) == before


def test_verification_module_is_deterministic_code_with_no_model_or_network_dependency():
    source = (REPO / "handbook_bot" / "plan_f_verification.py").read_text(encoding="utf-8")
    for forbidden in ("groq", "openai", "requests", "httpx", "urllib", "random", "sentence_transformers", "faiss", "torch"):
        assert not re.search(r"^\s*(?:import|from)\s+" + forbidden + r"\b", source, re.M), forbidden
    assert "answer_question" not in source and "orchestrator" not in source


# ---------------------------------------------------------------------------
# Reading the request
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,kind,extra", [
    (HR_PHONE, K.CONTACT, {"field": "telephone"}),
    ("What is the fax number of the Passport Unit?", K.CONTACT, {"field": "fax"}),
    ("What is the email address of the IT Center?", K.CONTACT, {"field": "email"}),
    ("Where is the Main Library?", K.LOCATION, {}),
    ("How many credit hours is a full teaching load?", K.QUANTITY, {"unit": "hour", "unit_modifiers": ("credit",)}),
    (SICK, K.QUANTITY, {"unit": "day"}),
    (THESES, K.QUANTITY, {"unit": "thes", "conditions": (("at", "once"),)}),
    (CONSULTANCY, K.QUANTITY, {"unit": "day", "conditions": (("week",),)}),
    (PETTY, K.AMOUNT, {}),
    ("What percentage of absences bars a student from the final examination?", K.PERCENTAGE, {}),
    ("How long is the probation period?", K.DURATION, {}),
    ("How often are faculty evaluated?", K.FREQUENCY, {}),
    ("When must final grades be submitted after the examination?", K.TIMING, {}),
    ("What is the teaching load for lecturers?", K.VALUE, {}),
    (WLAM, K.DEFINITION, {"entities": ("Workload Allocation Model",)}),
    (CEILING, K.AUTHORITY, {"verb": "determin"}),
    ("Who drafts employment offers?", K.AUTHORITY, {"verb": "draft"}),
    ("Who is responsible for health and safety on campus?", K.RESPONSIBILITY, {}),
    ("Which office handles faculty housing?", K.RESPONSIBILITY, {"body_noun": "office"}),
    ("Which committee reviews research on human subjects?", K.SELECTION, {"body_noun": "committee", "verb": "review"}),
    ("What information does HR provide to new faculty members?", K.SELECTION, {"object_head": "information", "verb": "provid"}),
    ("Who is eligible for sabbatical leave?", K.ELIGIBILITY, {}),
    ("What must instructors do with final examination answer sheets?", K.OBLIGATION, {"actor": "instructor"}),
    ("Does HR arrange travel and hotel accommodations for new faculty members?", K.YES_NO, {"verb": "arrang", "entities": ("HR",)}),
    ("Is maternity leave paid?", K.YES_NO, {"verb": "pay"}),
    (HR_PROVIDES, K.ENUMERATION, {}),
    ("What does the Technology Transfer Office do?", K.ENUMERATION, {}),
    ("What are the research priority areas?", K.ENUMERATION, {}),
    ("What are the requirements for promotion?", K.ENUMERATION, {}),
    ("What services does the library provide?", K.ENUMERATION, {}),
    ("What is the policy on make-up examinations?", K.GENERAL, {}),
    ("Tell me about research grants", K.GENERAL, {}),
    ("How are final grades entered into Blackboard?", K.UNRECOGNISED, {}),
])
def test_read_request_reads_the_kind_and_its_fields(clause, kind, extra):
    shape = V.read_request(clause)
    assert shape.kind is kind, shape
    for key, value in extra.items():
        assert getattr(shape, key) == value, (key, getattr(shape, key))


def test_read_request_reads_phrases_modifiers_and_conditions():
    shape = V.read_request(SICK)
    assert (("sick",), "leav") in shape.phrases and (("full",), "pay") in shape.phrases and ("full", "pay") in shape.conditions
    assert V.read_request(MATERNITY).phrases[-1] == (("female",), "faculty")
    shape = V.read_request("How many days of maternity leave are part-time female faculty entitled to?")
    assert (("part-time", "female"), "faculty") in shape.phrases
    assert V.read_request("Is X not required?").negated is True and V.read_request(SICK).negated is False


def test_quote_units_split_provisions_and_sentences():
    units = V.quote_units("Consultancy agreements must include a clause stating that the University is not liable for the consultancy activity. "
                          "16.1.7 Time Allocation for Consultancy: Full-time faculty members are permitted to spend up to 20% of their time.")
    assert [p for _t, p in units] == [0, 1]
    units = V.quote_units("3.11 Workload Allocation Model (WLAM) Articles 20 and 21 of the University President Decision No. (3) of 2019 establishes it.")
    assert len(units) == 1 and units[0][1] == 0                                 # "No. (3)" is not a sentence end; the opening number is provision 0
    units = V.quote_units("• Female faculty are entitled to sixty calendar days of maternity leave with full pay. Additional leave may be approved. "
                          "4.3.4 Unpaid Special Leave Unpaid special leave may be granted.")
    assert [p for _t, p in units] == [0, 0, 1]
    assert V.quote_units("Human Resources +971 6 5050023 +971 6 5585200") == [("Human Resources +971 6 5050023 +971 6 5585200", 0)]
    assert [p for _t, p in V.quote_units('o "E" Lecturer is mainly devoted to teaching and services. The teaching load for lecturers is 15 credit hours per semester.')] == [0, 0]


# ---------------------------------------------------------------------------
# Status and structure (one code per non-renderable shape; consistent with the I-1 gate)
# ---------------------------------------------------------------------------
def _shapes(kb):
    cid = T.chunk_index(kb, "Human Resources +971 6 5050023")
    row = "Human Resources +971 6 5050023 +971 6 5585200"
    out = {}
    out["supported"] = institutional(kb, HR_PHONE, cid, row, office_id="human_resources", value="+971 6 5050023")[1]
    out["partial"] = institutional(kb, HR_PHONE, cid, row, office_id="human_resources", value="+971 6 5050023", status=FindingStatus.PARTIAL, missing=["x"])[1]
    out["handoff"] = institutional(kb, HR_PHONE, cid, row, office_id="human_resources", value="+971 6 5050023",
                                   handoffs=[HandoffRequest(SpecialistId.INSTITUTIONAL, SpecialistId.FACULTY_SERVICES, "stub")])[1]
    out["two reports"] = institutional(kb, HR_PHONE, cid, row, office_id="human_resources", value="+971 6 5050023",
                                       reports=[{"index": 0, "outcome": "read", "ask_kind": "channel", "evidence_tier_reached": "E1", "offices": []},
                                                {"index": 1, "outcome": "read", "ask_kind": "channel", "evidence_tier_reached": "E1", "offices": []}])[1]
    out["not read"] = institutional(kb, HR_PHONE, cid, row, office_id="human_resources", value="+971 6 5050023", reports=[
        {"index": 0, "outcome": "handed_off", "ask_kind": "channel", "evidence_tier_reached": "E1", "offices": []}])[1]
    out["tier"] = institutional(kb, HR_PHONE, cid, row, office_id="human_resources", value="+971 6 5050023", reports=[
        {"index": 0, "outcome": "read", "ask_kind": "channel", "evidence_tier_reached": "E3", "offices": []}])[1]
    for status in (FindingStatus.NOT_FOUND, FindingStatus.OUT_OF_SCOPE):
        out[status.value] = SpecialistFindings(specialist_id=SpecialistId.INSTITUTIONAL, task_id="task-1", status=status, summary="stub",
                                               missing=["x"] if status is FindingStatus.NOT_FOUND else [], metadata={"clauses": [{"index": 0}]})
    out["error"] = SpecialistFindings(specialist_id=SpecialistId.INSTITUTIONAL, task_id="task-1", status=FindingStatus.ERROR, summary="boom",
                                      metadata={"error": "redacted", "clauses": []})
    scid, squote = located(kb, "• Faculty are entitled to sick leave with full pay", "per contract year.")
    task = task_for(SICK, "faculty_services")
    out["general"] = T.retrieval_result(kb, task, scid, squote, ask_shape="general", requested=("x",), covered=("x",))
    out["unknown"] = T.retrieval_result(kb, task, scid, squote, ask_shape="specific_unknown", requested=("x",), covered=("x",))
    out["no details"] = T.retrieval_result(kb, task, scid, squote, requested=(), covered=())
    out["ambiguous"] = T.retrieval_result(kb, task, scid, squote, requested=("x",), covered=("x",), relation="ambiguous")
    out["llm"] = T.retrieval_result(kb, task, scid, squote, requested=("x",), covered=("x",))
    out["llm"].llm_used = True
    out["retrieval ok"] = T.retrieval_result(kb, task, scid, squote, requested=("number or amount",), covered=("number or amount",), relation="strong")
    return out


@pytest.mark.parametrize("label,code", [
    ("supported", None), ("partial", R.PARTIAL_FINDING), ("handoff", R.HANDOFF_REQUIRED), ("two reports", R.MULTIPART_UNSUPPORTED),
    ("not read", R.UNSUPPORTED_STATUS), ("tier", R.UNSUPPORTED_STATUS), ("not_found", R.NOT_FOUND), ("out_of_scope", R.OUT_OF_SCOPE),
    ("error", R.UNSUPPORTED_STATUS), ("general", R.GENERAL_ASK_MISMATCH), ("unknown", R.UNRECOGNISED_ASK), ("no details", R.REQUESTED_DETAIL_MISSING),
    ("ambiguous", R.RELATION_MISMATCH), ("llm", R.UNSUPPORTED_STATUS), ("retrieval ok", None),
])
def test_status_code_names_every_non_renderable_shape_and_agrees_with_the_i1_gate(kb, label, code):
    findings = _shapes(kb)[label]
    got, _detail = V.status_code(findings)
    assert got is code, (label, got)
    assert (got is None) == (PF.result_problem(findings) is None), label          # the I-1 gate and the verifier never disagree
    if code is not None:
        verdict = V.verify_findings(HR_PHONE if findings.specialist_id is SpecialistId.INSTITUTIONAL else SICK, findings, kb.chunks, kb.metadata)
        assert verdict.accepted is False and verdict.safe_to_render is False and verdict.reason is code and verdict.accepted_finding_ids == ()


def test_handoff_information_is_preserved_not_consumed(kb):
    findings = _shapes(kb)["handoff"]
    before = [h.to_dict() for h in findings.handoff_requests]
    verdict = V.verify_findings(HR_PHONE, findings, kb.chunks, kb.metadata)
    assert verdict.reason is R.HANDOFF_REQUIRED and [h.to_dict() for h in findings.handoff_requests] == before and len(before) == 1


# ---------------------------------------------------------------------------
# Evidence integrity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label,mutate", [
    ("tampered quote", lambda f: setattr(f, "evidence_quote", f.evidence_quote.replace("5050023", "5050024"))),
    ("wrong page", lambda f: setattr(f, "page", f.page + 1)),
    ("chunk out of range", lambda f: f.metadata.__setitem__("chunk_id", 10 ** 9)),
    ("chunk id missing", lambda f: f.metadata.pop("chunk_id")),
])
def test_invalid_evidence_reference_is_rejected(kb, label, mutate):
    cid = T.chunk_index(kb, "Human Resources +971 6 5050023")
    task, findings = institutional(kb, HR_PHONE, cid, "Human Resources +971 6 5050023 +971 6 5585200", office_id="human_resources", value="+971 6 5050023")
    mutate(findings.findings[0])
    verdict = verify(kb, task, findings)
    assert verdict.reason is R.INVALID_EVIDENCE_REFERENCE and verdict.evidence_integrity is False and not verdict.safe_to_render, label


# ---------------------------------------------------------------------------
# Contact and entity binding (Institutional): correct value + wrong entity never passes
# ---------------------------------------------------------------------------
def test_contact_row_of_the_asked_office_with_the_asked_field_is_accepted(kb):
    cid = T.chunk_index(kb, "Human Resources +971 6 5050023")
    task, findings = institutional(kb, HR_PHONE, cid, "Human Resources +971 6 5050023 +971 6 5585200", office_id="human_resources", value="+971 6 5050023")
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.request_kind is K.CONTACT and verdict.completeness is C.COMPLETE
    assert verdict.subject_match is True and verdict.detail_coverage is True and verdict.evidence_integrity is True
    assert verdict.accepted_finding_ids == ("task-1-f1",) and verdict.dropped == ()
    task, findings = institutional(kb, "What is the fax number of the Human Resources Department?", cid, "Human Resources +971 6 5050023 +971 6 5585200",
                                   office_id="human_resources", field="fax", value="+971 6 5585200")
    assert verify(kb, task, findings).accepted


@pytest.mark.parametrize("label,question,kwargs,code", [
    ("fax asked, telephone bound", "What is the fax number of the Human Resources Department?", dict(office_id="human_resources", value="+971 6 5050023"), R.REQUESTED_DETAIL_MISSING),
    ("bound value not in the row", HR_PHONE, dict(office_id="human_resources", value="+971 6 5050040"), R.REQUESTED_DETAIL_MISSING),
    ("another office asked (Finance)", "What is the phone number of the Finance Department?", dict(office_id="human_resources", value="+971 6 5050023", offices=["finance_department"]), R.SUBJECT_MISMATCH),
    ("another office asked, report mislabelled", "What is the phone number of the Finance Department?", dict(office_id="human_resources", value="+971 6 5050023"), R.SUBJECT_MISMATCH),
    ("row labelled as another office", HR_PHONE, dict(office_id="finance_department", value="+971 6 5050023"), R.SUBJECT_MISMATCH),
    ("labelled row, not complete", HR_PHONE, dict(office_id="human_resources", value="+971 6 5050023", entry="labelled_row", row_complete=False), R.SUBJECT_MISMATCH),
    ("location asked of a directory row", "Where is the Human Resources Department?", dict(office_id="human_resources", value="+971 6 5050023", ask_kind="location"), R.REQUESTED_DETAIL_MISSING),
])
def test_contact_mutations_are_rejected_for_the_right_reason(kb, label, question, kwargs, code):
    cid = T.chunk_index(kb, "Human Resources +971 6 5050023")
    task, findings = institutional(kb, question, cid, "Human Resources +971 6 5050023 +971 6 5585200", **kwargs)
    verdict = verify(kb, task, findings)
    assert verdict.accepted is False and verdict.safe_to_render is False and verdict.reason is code, (label, verdict.reason, verdict.detail)


def test_correct_number_in_another_offices_row_never_passes(kb):
    cid = T.chunk_index(kb, "Finance Department +971 6 5050040")
    for offices in (["human_resources"], ["finance_department"]):              # whatever the clause report claims was asked
        task, findings = institutional(kb, HR_PHONE, cid, "Finance Department +971 6 5050040 +971 6 5585183", office_id="finance_department",
                                       value="+971 6 5050040", offices=offices)
        verdict = verify(kb, task, findings)
        assert verdict.reason is R.SUBJECT_MISMATCH and verdict.subject_match is False


def test_stated_fact_count_bound_to_the_university_is_accepted(kb):
    cid, quote = located(kb, "Academically, the University is composed of fifteen academic colleges", "Vice Chancellors.")
    task, findings = institutional(kb, "How many colleges does the University of Sharjah have?", cid, quote, office_id="university", ask_kind="fact_or_count",
                                   tier="stated_fact", offices=[], binding="named subject")
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.request_kind is K.QUANTITY and verdict.verdicts[0].value == 15.0
    task, findings = institutional(kb, "How many campuses does the University of Sharjah have?", cid, quote, office_id="university", ask_kind="fact_or_count",
                                   tier="stated_fact", offices=[], binding="named subject")
    assert verify(kb, task, findings).reason is R.REQUESTED_DETAIL_MISSING             # fifteen colleges is not a number of campuses


# ---------------------------------------------------------------------------
# Numeric and policy binding (HR, Teaching, Research): a correct-looking number from the wrong policy never passes
# ---------------------------------------------------------------------------
def test_sick_leave_quantity_is_accepted_with_its_value(kb):
    cid, quote = located(kb, "• Faculty are entitled to sick leave with full pay", "per contract year.")
    task, findings = retrieval(kb, SICK, "faculty_services", cid, quote, requested=("number or amount", "eligibility condition"))
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.request_kind is K.QUANTITY and verdict.completeness is C.COMPLETE and verdict.verdicts[0].value == 15.0
    assert verdict.relation_match is None and verdict.detail_coverage is True


@pytest.mark.parametrize("label,question,code", [
    ("maternity days asked; sick leave quoted", "How many days of maternity leave with full pay are faculty entitled to?", R.MODIFIER_MISMATCH),
    ("calendar days asked; working days quoted", "How many calendar days of sick leave with full pay are faculty entitled to?", R.MODIFIER_MISMATCH),
    ("hours asked; days quoted", "How many hours of sick leave with full pay are faculty entitled to?", R.REQUESTED_DETAIL_MISSING),
    ("per semester asked; per contract year quoted", "How many days of sick leave with full pay per semester are faculty entitled to?", R.CONDITION_MISMATCH),
    ("another population asked", "How many days of sick leave with full pay are research assistants entitled to?", R.SUBJECT_MISMATCH),
])
def test_sick_leave_mutations_are_rejected_for_the_right_reason(kb, label, question, code):
    cid, quote = located(kb, "• Faculty are entitled to sick leave with full pay", "per contract year.")
    task, findings = retrieval(kb, question, "faculty_services", cid, quote, requested=("number or amount", "eligibility condition"))
    verdict = verify(kb, task, findings)
    assert verdict.accepted is False and verdict.reason is code, (label, verdict.reason, verdict.detail)


def test_a_modifier_of_another_noun_is_not_a_contrast(kb):
    """I-3 case 1 on the real pages: "full teaching load" asked, "workload ... 12 credit hours" stated. The asked modifier
    "teaching" (of "load") must not be contrasted with "credit" (of "hours"); the sentence answers the question."""
    cid, quote = located(kb, "o Ranked full time regular faculty", "per academic year.")
    task, findings = retrieval(kb, "How many credit hours is a full teaching load?", "teaching", cid, quote, requested=("number or amount",))
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.request_kind is K.QUANTITY and verdict.verdicts[0].value == 12.0, (verdict.reason, verdict.detail)
    task, findings = retrieval(kb, "How many office hours is a full teaching load?", "teaching", cid, quote, requested=("number or amount",))
    assert verify(kb, task, findings).reason in (R.MODIFIER_MISMATCH, R.SUBJECT_MISMATCH)      # the unit noun itself is contrasted


def test_maternity_sentence_never_answers_another_leave_type(kb):
    cid, quote = located(kb, "• Female faculty employed on a full-time basis", "Academic Affairs.")
    task, findings = retrieval(kb, MATERNITY, "faculty_services", cid, quote, requested=("number or amount", "eligibility condition"))
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.completeness is C.ELABORATED and verdict.verdicts[0].value == 60.0
    for question in ("How many days of sick leave with full pay are female faculty entitled to?",
                     "How many days of annual leave with full pay are female faculty entitled to?",
                     "How many days of maternity leave are part-time female faculty entitled to?"):
        task, findings = retrieval(kb, question, "faculty_services", cid, quote, requested=("number or amount", "eligibility condition"))
        verdict = verify(kb, task, findings)
        assert verdict.accepted is False and verdict.reason in (R.MODIFIER_MISMATCH, R.SUBJECT_MISMATCH), (question, verdict.reason, verdict.detail)


def test_office_hours_unit_noun_condition_and_value_conflict(kb):
    cid, quote = located(kb, "1. Full-Time Faculty: Full-time faculty members", "receive academic support.")
    task, findings = retrieval(kb, OFFICE_HOURS, "teaching", cid, quote, requested=("number or amount",))
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.completeness is C.ELABORATED and verdict.verdicts[0].value == 5.0 and verdict.verdicts[0].unit_index == 0
    task, findings = retrieval(kb, "How many credit hours must faculty hold?", "teaching", cid, quote, requested=("number or amount",))
    assert verify(kb, task, findings).reason in (R.MODIFIER_MISMATCH, R.SUBJECT_MISMATCH)    # the I-1 coverage rule may fire first
    task, findings = retrieval(kb, "How many office hours per semester must faculty hold?", "teaching", cid, quote, requested=("number or amount",))
    assert verify(kb, task, findings).reason is R.CONDITION_MISMATCH
    # a second finding stating another value for the asked quantity is dropped, the primary is kept
    cid2, quote2 = located(kb, "3. Consideration for Evening Courses:", "during the evening.")
    task = task_for(OFFICE_HOURS, "teaching")
    second = T.finding_for(kb, cid2, quote2, task, number=2, details_covered=["number or amount"])
    findings = T.retrieval_result(kb, task, cid, quote, requested=("number or amount",), covered=("number or amount",), extra_findings=[second])
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.accepted_finding_ids == ("task-1-f1",) and verdict.dropped == (("task-1-f2", "VALUE_CONFLICT"),)
    # a second finding with the same value is kept
    same = T.finding_for(kb, cid, quote, task, number=2, details_covered=["number or amount"])
    same.finding_id = "task-1-f2"
    findings = T.retrieval_result(kb, task, cid, quote, requested=("number or amount",), covered=("number or amount",), extra_findings=[same])
    assert verify(kb, task, findings).accepted_finding_ids == ("task-1-f1", "task-1-f2")


def test_research_quantity_condition_idiom_and_population(kb):
    cid, quote = located(kb, "• A faculty member is allowed to supervise a maximum of four theses", "thesis process.")
    task, findings = retrieval(kb, THESES, "research", cid, quote, requested=("number or amount",), relation="strong")
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.verdicts[0].value == 4.0 and verdict.completeness is C.ELABORATED     # "at once" is stated as "simultaneously"
    task, findings = retrieval(kb, "How many theses can a faculty member supervise per year?", "research", cid, quote, requested=("number or amount",), relation="strong")
    assert verify(kb, task, findings).reason is R.CONDITION_MISMATCH
    cid, quote = located(kb, "20. The principal investigator is entitled to a petty cash advance", "approved completion period.")
    task, findings = retrieval(kb, PETTY, "research", cid, quote, requested=("amount or cost",), relation="strong")
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.request_kind is K.AMOUNT and verdict.verdicts[0].value == 5000.0 and verdict.completeness is C.ELABORATED
    task, findings = retrieval(kb, "How much petty cash advance can a research assistant receive?", "research", cid, quote, requested=("amount or cost",), relation="strong")
    assert verify(kb, task, findings).reason is R.SUBJECT_MISMATCH


def test_percentage_ask_needs_a_percentage_in_the_answering_unit(kb):
    cid, quote = located(kb, "3. Absences reaching 20% will prohibit", "final examination(s).")
    question = "What percentage of absences bars a student from the final examination?"
    task, findings = retrieval(kb, question, "teaching", cid, quote, requested=("percentage or range",))
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.request_kind is K.PERCENTAGE and verdict.verdicts[0].value == 20.0
    cid, quote = located(kb, "Instructors must correct final examination answer sheets", "examination date.")
    task, findings = retrieval(kb, question, "teaching", cid, quote, requested=("percentage or range",))
    assert verify(kb, task, findings).reason in (R.REQUESTED_DETAIL_MISSING, R.SUBJECT_MISMATCH)


# ---------------------------------------------------------------------------
# Relation and request kind
# ---------------------------------------------------------------------------
def test_definition_ask_needs_the_subject_defined_not_only_named(kb):
    cid, quote = located(kb, "3.11 Workload Allocation Model (WLAM) Articles 20 and 21", "teaching loads.")
    task, findings = retrieval(kb, WLAM, "teaching", cid, quote)
    verdict = verify(kb, task, findings)
    assert verdict.accepted is False and verdict.reason is R.RELATION_MISMATCH and verdict.relation_match is False and verdict.request_kind is K.DEFINITION
    cid, quote = located(kb, "The Technology Transfer Office (TTO) is entrusted", "research excellence and innovation.")
    task, findings = retrieval(kb, "What is the Technology Transfer Office?", "research", cid, quote, relation="strong")
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.relation_match is True and verdict.completeness is C.ELABORATED


def test_party_asks_bind_the_party_to_the_asked_action_in_the_asked_position(kb):
    cid, quote = located(kb, "c. The VCRGS determines the funding ceiling", "approval.")
    task, findings = retrieval(kb, CEILING, "research", cid, quote, requested=("approving authority",), relation="strong")
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.request_kind is K.AUTHORITY and verdict.relation_match is True and verdict.completeness is C.COMPLETE
    task, findings = retrieval(kb, "Who publishes the funding ceiling for research grants?", "research", cid, quote, requested=("approving authority",), relation="strong")
    assert verify(kb, task, findings).reason is R.RELATION_MISMATCH
    # the same agent named after another verb ("prepared by the VCRGS") does not state who determines
    cid, quote = located(kb, "17. The funding ceiling for such research projects is determined", "approval.")
    task, findings = retrieval(kb, CEILING, "research", cid, quote, requested=("approving authority",), relation="strong")
    verdict = verify(kb, task, findings)
    assert verdict.accepted is False and verdict.reason is R.RELATION_MISMATCH
    # a passive statement with its agent
    cid, quote = located(kb, "• Employment offers are drafted by HR", "Security Clearance.")
    task, findings = retrieval(kb, "Who drafts employment offers?", "faculty_services", cid, quote)
    assert verify(kb, task, findings).accepted


def test_selection_ask_needs_the_asked_kind_of_body(kb):
    cid, quote = located(kb, "• Institutional Review Board (IRB):", "regulatory compliance.")
    question = "Which committee reviews research on human subjects?"
    task, findings = retrieval(kb, question, "research", cid, quote, requested=("approving authority",), relation="strong")
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.request_kind is K.SELECTION and verdict.completeness is C.COMPLETE
    task, findings = retrieval(kb, "Which office reviews research on human subjects?", "research", cid, quote, requested=("approving authority",), relation="strong")
    assert verify(kb, task, findings).reason is R.RELATION_MISMATCH


def test_obligation_ask_needs_a_requirement_for_the_asked_actor(kb):
    cid, quote = located(kb, "Instructors must correct final examination answer sheets", "examination date.")
    task, findings = retrieval(kb, "What must instructors do with final examination answer sheets?", "teaching", cid, quote,
                               requested=("an explicit requirement statement about the subject",))
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.request_kind is K.OBLIGATION and verdict.relation_match is True
    task, findings = retrieval(kb, "What must students do with final examination answer sheets?", "teaching", cid, quote,
                               requested=("an explicit requirement statement about the subject",))
    assert verify(kb, task, findings).reason is R.SUBJECT_MISMATCH


def test_yes_no_ask_needs_the_asked_predicate_with_the_asked_polarity(kb):
    cid, quote = located(kb, "• HR arranges travel and hotel accommodations", "new faculty members.")
    question = "Does HR arrange travel and hotel accommodations for new faculty members?"
    task, findings = retrieval(kb, question, "faculty_services", cid, quote)
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.request_kind is K.YES_NO and verdict.relation_match is True and verdict.completeness is C.COMPLETE
    task, findings = retrieval(kb, "Does HR arrange travel and hotel accommodations for visiting faculty members?", "faculty_services", cid, quote)
    assert verify(kb, task, findings).reason is R.MODIFIER_MISMATCH
    cid, quote = located(kb, "f. UoS does not claim ownership of copyright material", "by the university.")
    task, findings = retrieval(kb, "Does UoS claim ownership of copyright material created by faculty?", "research", cid, quote, relation="strong")
    verdict = verify(kb, task, findings)
    assert verdict.accepted is False and verdict.reason is R.RELATION_MISMATCH and "negates" in verdict.detail


def test_enumerative_ask_is_never_answered_by_one_finding_but_a_selection_ask_is(kb):
    cid, quote = located(kb, "• HR provides information to new faculty members", "settling in the UAE).")
    task, findings = retrieval(kb, HR_PROVIDES, "faculty_services", cid, quote)
    verdict = verify(kb, task, findings)
    assert verdict.accepted is False and verdict.reason is R.ENUMERATION_INCOMPLETE and verdict.completeness is C.ENUMERATION_UNPROVEN
    assert verdict.request_kind is K.ENUMERATION and verdict.subject_match is True
    task, findings = retrieval(kb, "What information does HR provide to new faculty members?", "faculty_services", cid, quote)
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.request_kind is K.SELECTION and verdict.relation_match is True


def test_general_ask_is_rejected_by_the_verifier_even_when_the_report_calls_it_specific(kb):
    cid, quote = located(kb, "• Faculty are entitled to sick leave with full pay", "per contract year.")
    task, findings = retrieval(kb, "What is the policy on sick leave?", "faculty_services", cid, quote)
    assert verify(kb, task, findings).reason is R.GENERAL_ASK_MISMATCH


def test_unrecognised_request_kind_falls_back_to_the_specialist_contract(kb):
    cid, quote = located(kb, "• Faculty are entitled to sick leave with full pay", "per contract year.")
    task, findings = retrieval(kb, "Summarise the sick leave entitlement with full pay for faculty", "faculty_services", cid, quote)
    verdict = verify(kb, task, findings)
    assert verdict.request_kind is K.UNRECOGNISED and verdict.accepted and verdict.completeness is C.CONTRACT and verdict.relation_match is None


# ---------------------------------------------------------------------------
# Provision boundary and completeness
# ---------------------------------------------------------------------------
def test_quote_spanning_two_provisions_is_excess_evidence_not_trimmed(kb):
    cid, quote = located(kb, "Consultancy agreements must include a clause", "professional consultancy activities.")
    task, findings = retrieval(kb, CONSULTANCY, "research", cid, quote, requested=("number or amount", "duration"), relation="strong")
    verdict = verify(kb, task, findings)
    assert verdict.accepted is False and verdict.reason is R.EXCESS_EVIDENCE and verdict.completeness is C.EXCESS
    assert verdict.verdicts[0].unit_index == 1 and verdict.verdicts[0].value == 1.0        # the answering unit was found, in the second provision
    # the same rule, the other way round: a correct first provision followed by the next numbered section
    cid, quote = located(kb, "• Female faculty employed on a full-time basis", "unused vacation leave.")
    task, findings = retrieval(kb, MATERNITY, "faculty_services", cid, quote, requested=("number or amount", "eligibility condition"))
    verdict = verify(kb, task, findings)
    assert verdict.reason is R.EXCESS_EVIDENCE and "continues" in verdict.detail
    # trimming is not performed: the rendered answer would have to be the specialist's own quote
    assert findings.findings[0].evidence_quote == quote


def test_same_provision_elaboration_is_kept_whole(kb):
    cid, quote = located(kb, 'o "E" Lecturer is mainly devoted', "per semester.")
    task, findings = retrieval(kb, "What is the teaching load for lecturers?", "teaching", cid, quote, requested=("the requested faculty category (lecturers)",))
    verdict = verify(kb, task, findings)
    assert verdict.accepted and verdict.completeness is C.ELABORATED and verdict.verdicts[0].unit_index == 1 and verdict.verdicts[0].value == 15.0


# ---------------------------------------------------------------------------
# Verifier failure safety
# ---------------------------------------------------------------------------
def test_a_raising_verifier_rejects_with_a_redacted_error(kb, monkeypatch):
    cid = T.chunk_index(kb, "Human Resources +971 6 5050023")
    task, findings = institutional(kb, HR_PHONE, cid, "Human Resources +971 6 5050023 +971 6 5585200", office_id="human_resources", value="+971 6 5050023")

    def boom(_clause):
        raise RuntimeError(r"failed at C:\Users\someone\private\rules.json with key gsk_ABCDEFGHIJKLMNOPQRST")

    monkeypatch.setattr(V, "read_request", boom)
    verdict = verify(kb, task, findings)
    assert verdict.accepted is False and verdict.safe_to_render is False and verdict.reason is R.VERIFICATION_ERROR
    text = json.dumps(verdict.to_dict())
    assert "gsk_ABCDEFGHIJKLMNOPQRST" not in text and "someone" not in text


def test_adapter_falls_back_when_the_verifier_raises_or_rejects(kb, runtime, monkeypatch):
    def boom(_clause):
        raise RuntimeError(r"C:\Users\someone\private gsk_ABCDEFGHIJKLMNOPQRST")

    monkeypatch.setattr(V, "read_request", boom)
    attempt = runtime.attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_FINDINGS_VERIFIER and attempt.result is None and attempt.requires_fallback
    assert attempt.verification is not None and attempt.verification.reason is R.VERIFICATION_ERROR and attempt.verified is False
    assert "gsk_ABCDEFGHIJKLMNOPQRST" not in T.trace_json(attempt) and "someone" not in T.trace_json(attempt)
    assert "verifier" not in T.trace_agents(attempt)                                       # nothing is rendered or verified after a rejection
    monkeypatch.undo()

    def explode(*args, **kwargs):
        raise RuntimeError("verifier module failure")

    monkeypatch.setattr(V, "verify_findings", explode)
    attempt = runtime.attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.ERROR_REDACTED and attempt.result is None


# ---------------------------------------------------------------------------
# Adapter integration
# ---------------------------------------------------------------------------
def test_adapter_runs_the_verifier_between_the_i1_guards_and_rendering(runtime):
    attempt = runtime.attempt(HR_PHONE)
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED and attempt.verified and attempt.verification.request_kind is K.CONTACT
    agents = T.trace_agents(attempt)
    assert agents.index("subject_guard") < agents.index("findings_verifier") < agents.index("verifier") < agents.index("plan_f_outcome")
    entry = next(e for e in attempt.trace if e.agent == "findings_verifier")
    assert entry.decision == "accepted" and entry.used_llm is False and entry.detail["reason"] == "ACCEPTED"
    assert "findings_verify_ms" in attempt.timings and 0.0 <= attempt.timings["findings_verify_ms"] < 5000.0
    json.dumps(attempt.to_dict(), default=str)
    assert attempt.to_dict()["verification"]["accepted"] is True and HR_PHONE not in json.dumps(attempt.to_dict()["verification"])


@pytest.mark.parametrize("question,specialist,kind", [
    (HR_PHONE, "institutional", K.CONTACT),
    ("Does HR arrange travel and hotel accommodations for new faculty members?", "faculty_services", K.YES_NO),
    ("How many credit hours is a full teaching load?", "teaching", K.QUANTITY),
    (THESES, "research", K.QUANTITY),
])
def test_each_frozen_specialists_i1_example_is_still_accepted(runtime, question, specialist, kind):
    attempt = runtime.attempt(question)
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED and attempt.specialist_id == specialist, attempt.reason
    assert attempt.verified and attempt.verification.request_kind is kind and attempt.result.llm_calls == 0 and attempt.result.used_llm is False
    assert all(e.used_llm is False for e in attempt.trace)


def test_known_hr_cases_stay_blocked_and_the_verifier_classifies_them(kb, runtime):
    from handbook_bot.agents.coordinator import coordinate
    from handbook_bot.agents.specialists.dispatch import dispatch

    for question, code in ((HR_KNOWN_WRONG, R.GENERAL_ASK_MISMATCH), ("What does HR do?", R.GENERAL_ASK_MISMATCH), (ANNUAL_LEAVE, R.PARTIAL_FINDING)):
        attempt = runtime.attempt(question)
        assert attempt.outcome is PF.PlanFOutcome.UNSAFE_RESULT and attempt.result is None          # the I-1 gate still fires first
        assert "findings_verifier" not in T.trace_agents(attempt)
        decision = coordinate(question)
        run = dispatch(decision, runtime.registry, question=question).runs[0]
        verdict = V.verify_findings(decision.subtasks[0].context["focus_clauses"][0], run.findings, kb.chunks, kb.metadata)
        assert verdict.accepted is False and verdict.reason is code, (question, verdict.reason)
        if question == ANNUAL_LEAVE:
            assert verdict.completeness is C.PARTIAL and "sixty" not in json.dumps(verdict.to_dict())


def test_misleading_supported_shapes_through_the_adapter_are_rejected_by_the_findings_verifier(kb):
    """Forced supported results that pass the I-1 gates: the findings-aware layer is the one that stops them."""
    cases = []
    cid, quote = located(kb, "3.11 Workload Allocation Model (WLAM) Articles 20 and 21", "teaching loads.")
    cases.append((WLAM, "teaching", cid, quote, {}, R.RELATION_MISMATCH))
    cid, quote = located(kb, "Consultancy agreements must include a clause", "professional consultancy activities.")
    cases.append((CONSULTANCY, "research", cid, quote, {"requested": ("number or amount",), "relation": "strong"}, R.EXCESS_EVIDENCE))
    cid, quote = located(kb, "• HR provides information to new faculty members", "settling in the UAE).")
    cases.append((HR_PROVIDES, "faculty_services", cid, quote, {}, R.ENUMERATION_INCOMPLETE))
    cid, quote = located(kb, "• Female faculty employed on a full-time basis", "Academic Affairs.")
    cases.append(("How many days of sick leave with full pay are female faculty entitled to?", "faculty_services", cid, quote,
                  {"requested": ("number or amount",)}, R.MODIFIER_MISMATCH))
    for question, specialist, cid, quote, kwargs, code in cases:
        def forced(task, _cid=cid, _quote=quote, _kwargs=kwargs):
            requested = _kwargs.get("requested", ("the asked specifics stated with the answer",))
            return T.retrieval_result(kb, task, _cid, _quote, requested=requested, covered=requested, relation=_kwargs.get("relation"))

        attempt = stub_attempt(kb, specialist, forced, question)
        assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_FINDINGS_VERIFIER and attempt.result is None and attempt.requires_fallback, (question, attempt.reason)
        assert attempt.verification.reason is code and attempt.reason.startswith(code.value)
        assert "verifier" not in T.trace_agents(attempt) and T.trace_agents(attempt)[-1] == "plan_f_outcome"


def test_accepted_result_renders_only_the_verified_findings(kb):
    cid, quote = located(kb, "1. Full-Time Faculty: Full-time faculty members", "receive academic support.")
    cid2, quote2 = located(kb, "3. Consideration for Evening Courses:", "during the evening.")

    def two(task):
        second = T.finding_for(kb, cid2, quote2, task, number=2, details_covered=["number or amount"])
        return T.retrieval_result(kb, task, cid, quote, requested=("number or amount",), covered=("number or amount",), extra_findings=[second])

    attempt = stub_attempt(kb, "teaching", two, OFFICE_HOURS)
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED and attempt.result.answer == quote and attempt.result.pages == [79]
    assert attempt.verification.dropped == (("task-1-f2", "VALUE_CONFLICT"),) and len(attempt.result.items) == 1


def test_adapter_result_without_verification_is_never_served_by_the_pipeline(kb, monkeypatch):
    cid = T.chunk_index(kb, "Human Resources +971 6 5050023")
    task, findings = institutional(kb, HR_PHONE, cid, "Human Resources +971 6 5050023 +971 6 5585200", office_id="human_resources", value="+971 6 5050023")
    real = PF.PlanFRuntime(T.stub_registry(institutional=lambda t: findings), kb.chunks, kb.metadata).attempt(HR_PHONE)
    assert real.outcome is PF.PlanFOutcome.ACCEPTED and real.verified
    unverified = dataclasses.replace(real, verification=None)
    assert unverified.verified is False and pipeline._verified(unverified) is False and pipeline._verified(real) is True
    rejected = dataclasses.replace(real, verification=dataclasses.replace(real.verification, accepted=False, safe_to_render=False))
    assert pipeline._verified(rejected) is False


# ---------------------------------------------------------------------------
# Pipeline: SHADOW stays non-invasive, ACTIVE requires the findings-aware acceptance, diagnostics carry codes only
# ---------------------------------------------------------------------------
def test_pipeline_diagnostics_carry_the_verification_codes_without_text(kb, monkeypatch):
    monkeypatch.setattr(config, "PLAN_F_SHADOW", True)
    legacy = pipeline._legacy_answer(kb, HR_PHONE)
    result, d = pipeline.ask_with_diagnostics(kb, HR_PHONE)
    assert result.answer == legacy.answer and d["served"] == "legacy" and d["mode"] == "shadow"
    assert d["findings_verified"] is True and d["findings_accepted"] is True and d["findings_reason"] == "ACCEPTED"
    assert d["findings_completeness"] == "complete" and d["findings_request_kind"] == "contact" and d["findings_ms"] >= 0.0
    text = json.dumps(d)
    assert HR_PHONE not in text and "5050023" not in text
    monkeypatch.setattr(config, "PLAN_F_SHADOW", False)
    monkeypatch.setattr(config, "PLAN_F_ENABLED", True)
    pipeline.reset_plan_f_state()
    result, d = pipeline.ask_with_diagnostics(kb, HR_PHONE)
    assert d["served"] == "plan_f" and d["accepted"] is True and d["findings_accepted"] is True and result.answer.startswith("Human Resources +971 6 5050023")
    result, d = pipeline.ask_with_diagnostics(kb, HR_KNOWN_WRONG)
    assert d["served"] == "legacy" and d["findings_verified"] is False and d["findings_reason"] is None and d["outcome"] == "PLAN_F_UNSAFE_RESULT"


def test_active_mode_serves_legacy_when_the_findings_verifier_rejects_or_raises(kb, monkeypatch):
    monkeypatch.setattr(config, "PLAN_F_ENABLED", True)
    legacy = pipeline._legacy_answer(kb, HR_PHONE)

    def boom(_clause):
        raise RuntimeError("verifier defect")

    monkeypatch.setattr(V, "read_request", boom)
    result, d = pipeline.ask_with_diagnostics(kb, HR_PHONE)
    assert result.answer == legacy.answer and d["served"] == "legacy" and d["accepted"] is False
    assert d["outcome"] == "PLAN_F_REJECTED_BY_FINDINGS_VERIFIER" and d["findings_reason"] == "VERIFICATION_ERROR" and d["findings_accepted"] is False
    assert d["fallback_reason"].startswith("VERIFICATION_ERROR") and "verifier defect" not in json.dumps(d)


def test_legacy_mode_is_untouched_by_the_verification_layer(kb):
    assert pipeline.plan_f_mode() == pipeline.MODE_LEGACY
    result, d = pipeline.ask_with_diagnostics(kb, HR_PHONE)
    assert d == {"mode": "legacy", "served": "legacy", "attempted": False}
    assert result.answer == pipeline._legacy_answer(kb, HR_PHONE).answer
    source = (REPO / "ui" / "pipeline.py").read_text(encoding="utf-8")
    assert "plan_f_verification" not in source                                         # the seam reads the adapter's record only
