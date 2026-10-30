"""
Runtime integration phase I-7: the chat interface's presentation helpers (``ui/presentation.py``) and the way ``ui/chat.py``
consumes the pipeline's per-turn diagnostics.

The helpers are pure: they read a finished ``QAResult`` and the diagnostics record of the same turn. The tests build those
two inputs by hand in the shapes the pipeline produces (LEGACY, SHADOW and ACTIVE records; single-finding, aggregate,
routing-refined, handoff and fallback outcomes) and check the labels, the source chips, the Markdown escaping of technical
identifiers, the N/A rule for timings the runtime did not record, the Developer-panel sections and the per-turn isolation
of diagnostics. No browser, no NiceGUI import.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from handbook_bot.orchestrator import NO_KEY_REPLY
from handbook_bot.qa import QAResult, REFUSAL
from ui import presentation as P

REPO = Path(__file__).resolve().parents[1]
HR_ROW = "Human Resources +971 6 5050023 +971 6 5585200"
SICK = "• Faculty are entitled to sick leave with full pay for up to 15 working days and half pay for an additional 15 working days per contract year."
LOAD = "o Ranked full time regular faculty, visiting faculty or senior lecturer workload equal to 12 credit hours per regular semester or 24 credit hours per academic year."


def item(page, chunk, section="Contact Information", source_title="UOS Faculty Handbook 2025/26", chunk_type="row", plan_f=None):
    meta = {"page": page, "section_title": section, "source_title": source_title, "chunk_type": chunk_type, "source_id": "uos_faculty_handbook_2025_26"}
    if plan_f:
        meta["plan_f"] = plan_f
    return {"idx": 1, "chunk": chunk, "meta": meta}


def legacy_diag():
    return {"mode": "legacy", "served": "legacy", "attempted": False}


def plan_f_diag(**over):
    d = {"mode": "active", "served": "plan_f", "attempted": True, "outcome": "PLAN_F_ACCEPTED", "reason": "rendered 1 supported finding(s) of institutional",
         "specialist_id": "institutional", "status": "supported", "accepted": True, "fallback_reason": None, "plan_f_pages": [54], "legacy_pages": None,
         "plan_f_ms": 612.4, "plan_f_timings": {}, "registry_ok": True, "registry_reused": True, "registry_build_ms": 678.0, "plan_f_llm_calls": 0,
         "plan_f_used_llm": False, "legacy_llm_calls": None, "findings_verified": True, "findings_accepted": True, "findings_reason": "ACCEPTED",
         "findings_completeness": "complete", "findings_request_kind": "contact", "findings_ms": 0.6, "plan_f_path": P.PATH_SINGLE,
         "aggregation_attempted": False, "aggregation_accepted": None, "aggregation_reason": None, "aggregation_clauses": None, "aggregation_covered": None,
         "aggregation_findings_considered": None, "aggregation_findings_accepted": None, "aggregation_duplicates": None, "aggregation_conflict": None,
         "aggregation_pages": [], "aggregation_specialists": [], "aggregation_dispatched": None, "aggregation_ms": None,
         "routing_refined": False, "routing_reason": "ROUTE_UNCHANGED", "routing_initial": ["institutional"], "routing_final": ["institutional"], "routing_ms": 0.1,
         "handoff_attempted": False, "handoff_source": [], "handoff_target": [], "handoff_depth": 0, "handoff_visited": [], "handoff_outcome": None,
         "handoff_ms": None, "specialist_calls": 1, "error": None}
    d.update(over)
    return d


def plan_f_result(answer=HR_ROW, pages=(54,), items=None, timings=None, query_type="contact"):
    return QAResult(answer=answer, pages=list(pages), best_section="Contact Information", evidence=answer[:60], query_type=query_type,
                    items=items if items is not None else [item(54, HR_ROW)], used_llm=False,
                    timings=timings if timings is not None else {"coordinator_ms": 3.1, "routing_ms": 0.1, "dispatch_ms": 590.0, "specialist_ms": 588.0,
                                                                  "guards_ms": 0.4, "findings_verify_ms": 0.6, "verify_ms": 0.3, "generation_ms": 0.0, "total_ms": 612.4},
                    num_candidates=1, num_reranked=1, llm_calls=0)


def legacy_result(answer="The phone number is +971 6 5050023.", query_type="contact", used_llm=False, pages=(54,), items=None, timings=None, best_section="Page 54"):
    return QAResult(answer=answer, pages=list(pages), best_section=best_section, evidence="", query_type=query_type,
                    items=items if items is not None else [item(54, HR_ROW)], used_llm=used_llm,
                    timings=timings if timings is not None else {"retrieval_ms": 207.5, "rerank_ms": 419.1, "generation_ms": 0.0, "total_ms": 626.7},
                    num_candidates=24, num_reranked=5, llm_calls=1 if used_llm else 0)


def rows_of(sections, title):
    for name, rows in sections:
        if name == title:
            return dict(rows)
    return None


# ---------------------------------------------------------------------------
# Markdown safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    (NO_KEY_REPLY, "GROQ\\_API\\_KEY is not set. Please set it, then reload the app."),
    ("Use add_drop and e_learning forms.", "Use add\\_drop and e\\_learning forms."),
    ("Set `GROQ_API_KEY` in the environment.", "Set `GROQ_API_KEY` in the environment."),          # code spans untouched
    ("_emphasis_ stays, snake_case changes", "_emphasis_ stays, snake\\_case changes"),
    ("", ""),
    (SICK, SICK),
])
def test_technical_identifiers_keep_their_underscores(text, expected):
    assert P.escape_identifiers(text) == expected


def test_escaping_is_idempotent_on_plain_prose():
    for text in (SICK, LOAD, HR_ROW, "Page 54 · Contact Information"):
        assert P.escape_identifiers(text) == text


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def test_legacy_kinds_are_read_from_the_result_itself():
    assert P.legacy_kind(legacy_result()) == "Legacy deterministic extractor"
    assert P.legacy_kind(legacy_result(answer=NO_KEY_REPLY, query_type="policy")) == "Legacy model path (no model client configured)"
    assert P.legacy_kind(legacy_result(answer="Faculty may ...", query_type="policy", used_llm=True)).startswith("Legacy model synthesis")
    assert P.legacy_kind(legacy_result(answer=REFUSAL, query_type="policy")) == "Refusal (no supporting evidence)"
    assert P.legacy_kind(legacy_result(answer="Some text", query_type="policy")) == "Legacy answer"


def test_answer_kind_never_claims_a_specialist_answer_the_user_did_not_get():
    assert P.answer_kind(legacy_result(), legacy_diag()) == "Legacy deterministic extractor"
    assert P.answer_kind(legacy_result(), None) == "Legacy deterministic extractor"
    assert P.answer_kind(plan_f_result(), plan_f_diag()) == "Verified specialist answer"
    assert P.answer_kind(plan_f_result(), plan_f_diag(plan_f_path=P.PATH_AGGREGATE, aggregation_attempted=True)) == "Verified multi-part answer"
    assert P.answer_kind(plan_f_result(), plan_f_diag(routing_refined=True)) == "Verified specialist answer (routing refined)"
    assert P.answer_kind(plan_f_result(), plan_f_diag(handoff_attempted=True, handoff_outcome="HANDOFF_ACCEPTED")) == "Verified specialist answer (handoff completed)"
    fallback = plan_f_diag(served="legacy", outcome="PLAN_F_UNSAFE_RESULT", accepted=False, fallback_reason="ask shape general: only a recognised specific ask is eligible")
    assert P.answer_kind(legacy_result(answer=NO_KEY_REPLY, query_type="policy"), fallback) == "Legacy fallback after Plan F attempt: Legacy model path (no model client configured)"
    shadow = plan_f_diag(mode="shadow", served="legacy")
    assert P.answer_kind(legacy_result(), shadow) == "Legacy deterministic extractor · Plan F ran in shadow"
    greeting = QAResult(answer="Hello! Ask me a question about the University of Sharjah Faculty Handbook.", query_type="greeting", timings={"total_ms": 0.0})
    not_applicable = plan_f_diag(served="legacy", outcome=P.NOT_APPLICABLE, accepted=False, specialist_id=None, plan_f_path=None, routing_initial=[], routing_final=[])
    assert P.answer_kind(greeting, not_applicable) == "Legacy answer (Plan F not applicable)"
    assert P.served_label(not_applicable) == "Legacy (Plan F not applicable)"
    assert "Specialist registry" not in dict(P.performance_rows(greeting, not_applicable, 0.8))
    assert P.answer_kind(QAResult(answer=P.ERROR_TEXT), P.error_diagnostics(RuntimeError("x"))) == "Error (safe reply)"
    for label in (P.answer_kind(legacy_result(), legacy_diag()), P.answer_kind(legacy_result(), shadow)):
        assert "Verified" not in label and "specialist answer" not in label


def test_served_and_path_labels():
    assert P.served_label(legacy_diag()) == "Legacy" and P.served_label(None) == "Legacy"
    assert P.served_label(plan_f_diag()) == "Plan F"
    assert P.served_label(plan_f_diag(served="legacy")) == "Legacy (fallback after Plan F attempt)"
    assert P.served_label(plan_f_diag(mode="shadow", served="legacy")) == "Legacy (Plan F in shadow)"
    assert P.path_label(plan_f_diag()) == "single finding" and P.path_label(plan_f_diag(plan_f_path=P.PATH_AGGREGATE)) == "multi-finding aggregate"
    assert P.path_label(legacy_diag()) == P.NA and P.mode_label(legacy_diag()) == "LEGACY" and P.mode_label(None) == P.NA


def test_domain_labels_only_for_a_served_specialist_answer():
    assert P.domain_labels(legacy_diag()) == [] and P.domain_labels(plan_f_diag(served="legacy")) == [] and P.domain_labels(plan_f_diag(mode="shadow", served="legacy")) == []
    assert P.domain_labels(plan_f_diag()) == ["Institutional Navigation"]
    assert P.domain_labels(plan_f_diag(plan_f_path=P.PATH_AGGREGATE, aggregation_specialists=["teaching", "faculty_services"])) == ["Teaching & Learning", "Faculty Services & HR"]
    assert P.domain_labels(plan_f_diag(routing_final=["research"], specialist_id="research,institutional")) == ["Research & Innovation"]
    assert P.domain_labels(plan_f_diag(routing_final=[], specialist_id="research,institutional")) == ["Research & Innovation", "Institutional Navigation"]


# ---------------------------------------------------------------------------
# Source chips
# ---------------------------------------------------------------------------
def test_chips_show_the_document_once_and_one_chip_per_page_with_its_section():
    result = plan_f_result()
    assert P.source_chips(result) == [("doc", "UOS Faculty Handbook 2025/26"), ("page", "Page 54 · Contact Information")]


def test_a_page_placeholder_section_never_becomes_a_duplicate_looking_chip():
    result = legacy_result(best_section="Page 54", items=[item(54, HR_ROW, section=None)])
    assert P.source_chips(result) == [("doc", "UOS Faculty Handbook 2025/26"), ("page", "Page 54")]
    result = legacy_result(best_section="Page 54", items=[item(54, HR_ROW, section="Page 54")], pages=[54, 54])
    assert [label for _c, label in P.source_chips(result)] == ["UOS Faculty Handbook 2025/26", "Page 54"]


def test_multi_page_evidence_keeps_every_page_in_the_answers_order():
    items = [item(96, "o \"A\" Regular Faculty ...", section="Workload Allocation Model (WLAM)", chunk_type="paragraph"),
             item(91, LOAD, section="Workload Allocation Model (WLAM)", chunk_type="paragraph"),
             item(107, SICK, section="Leaves and Absences", chunk_type="paragraph")]
    result = plan_f_result(answer="a\n\nb\n\nc", pages=[96, 91, 107], items=items)
    assert [label for _c, label in P.source_chips(result)] == ["UOS Faculty Handbook 2025/26", "Page 96 · Workload Allocation Model (WLAM)",
                                                               "Page 91 · Workload Allocation Model (WLAM)", "Page 107 · Leaves and Absences"]
    assert P.ordered_pages([96, 91, 107, 91, "x", True]) == [96, 91, 107]


def test_same_page_evidence_with_the_same_section_gives_one_chip_and_different_sections_two():
    same = plan_f_result(answer="a\n\nb", pages=[61], items=[item(61, "• HR arranges travel ...", section="Recruitment and Hiring", chunk_type="paragraph"),
                                                            item(61, "• HR assists ... visas ...", section="Recruitment and Hiring", chunk_type="paragraph")])
    assert [label for _c, label in P.source_chips(same)] == ["UOS Faculty Handbook 2025/26", "Page 61 · Recruitment and Hiring"]
    different = plan_f_result(answer="a\n\nb", pages=[61], items=[item(61, "x", section="Recruitment and Hiring", chunk_type="paragraph"),
                                                                 item(61, "y", section="Pre-joining Formalities", chunk_type="paragraph")])
    assert [label for _c, label in P.source_chips(different)] == ["UOS Faculty Handbook 2025/26", "Page 61 · Recruitment and Hiring", "Page 61 · Pre-joining Formalities"]


def test_chips_without_pages_fall_back_to_a_real_section_and_mark_ocr_evidence():
    assert P.source_chips(QAResult(answer="Hello!", pages=[], best_section="Greeting")) == []
    assert P.source_chips(QAResult(answer="x", pages=[], best_section="Leaves and Absences", items=[])) == [("section", "Leaves and Absences")]
    refusal = legacy_result(answer=REFUSAL, query_type="policy", pages=[], best_section="No matching evidence")   # retrieved items, nothing cited
    assert P.source_chips(refusal) == []
    assert rows_of(P.dev_sections(refusal, legacy_diag(), {}, 1.0), "Evidence")["Document"] == P.NA
    ocr = legacy_result(items=[item(54, HR_ROW, section=None, chunk_type="image_ocr")])
    assert P.source_chips(ocr)[-1] == ("ocr", "image-OCR evidence")
    long_section = plan_f_result(items=[item(54, HR_ROW, section="A very long section title that goes on and on beyond the chip width limit")])
    assert P.source_chips(long_section)[1][1].endswith("...") and len(P.source_chips(long_section)[1][1]) <= len("Page 54 · ") + 44


# ---------------------------------------------------------------------------
# Timings: never a fake zero
# ---------------------------------------------------------------------------
def test_plan_f_timings_come_from_the_runtime_and_unrecorded_stages_are_na():
    rows = dict(P.performance_rows(plan_f_result(), plan_f_diag(), wall_ms=640.2))
    assert rows["Specialist execution"] == "588.0 ms" and rows["Findings verification"] == "0.6 ms" and rows["Legacy Verifier"] == "0.3 ms"
    assert rows["Plan F total"] == "612.4 ms" and rows["Legacy total"] == P.NA and rows["End-to-end"] == "640.2 ms" and rows["Specialist registry"] == "reused"
    assert rows["Retrieval"] == P.NA and rows["Rerank"] == P.NA                       # the specialists do not time these separately: never a zero
    bare = plan_f_result(timings={"total_ms": 5.0})
    rows = dict(P.performance_rows(bare, plan_f_diag(plan_f_ms=None, registry_reused=False, registry_build_ms=678.0), wall_ms=None))
    assert rows["Specialist execution"] == P.NA and rows["Routing refinement"] == P.NA and rows["Legacy Verifier"] == P.NA
    assert rows["Plan F total"] == "5.0 ms" and rows["End-to-end"] == P.NA and rows["Specialist registry"] == "built (678.0 ms)"
    assert P.fmt_ms(None) == P.NA and P.fmt_ms(True) == P.NA and P.fmt_ms(-1) == P.NA and P.fmt_ms(0) == "0.0 ms" and P.fmt_ms(12.34) == "12.3 ms"
    assert P.fmt_ms(0.0225) == "<0.1 ms" and P.fmt_ms(0.05) == "0.1 ms"


def test_fallback_timing_separates_the_plan_f_attempt_from_the_legacy_time():
    fallback = plan_f_diag(served="legacy", outcome="PLAN_F_UNSAFE_RESULT", accepted=False, plan_f_ms=425.7)
    rows = dict(P.performance_rows(legacy_result(answer=NO_KEY_REPLY, query_type="policy"), fallback, wall_ms=1061.9))
    assert rows["Plan F attempt"] == "425.7 ms" and rows["Retrieval"] == "207.5 ms" and rows["Rerank"] == "419.1 ms" and rows["Legacy total"] == "626.7 ms"
    assert rows["End-to-end"] == "1061.9 ms" and rows["Specialist registry"] == "reused"
    legacy = dict(P.performance_rows(legacy_result(), legacy_diag(), wall_ms=630.0))
    assert "Plan F attempt" not in legacy and legacy["Legacy total"] == "626.7 ms" and "Specialist registry" not in legacy
    shadow = dict(P.performance_rows(legacy_result(), plan_f_diag(mode="shadow", served="legacy", plan_f_ms=612.4), wall_ms=1250.0))
    assert shadow["Plan F attempt (shadow)"] == "612.4 ms" and shadow["Legacy total"] == "626.7 ms"
    unrecorded = dict(P.performance_rows(QAResult(answer="x", timings={}), legacy_diag(), wall_ms=None))
    assert unrecorded["Retrieval"] == P.NA and unrecorded["Legacy total"] == P.NA and unrecorded["End-to-end"] == P.NA


# ---------------------------------------------------------------------------
# The Developer panel
# ---------------------------------------------------------------------------
def test_legacy_mode_panel_has_no_plan_f_rows():
    sections = P.dev_sections(legacy_result(), legacy_diag(), {"total_chunks": 13688, "total_pages": 272, "ocr_chunks": 0}, 630.0)
    titles = [t for t, _r in sections]
    assert titles == ["Runtime", "Evidence", "Performance", "LLM", "Knowledge base"]
    runtime = rows_of(sections, "Runtime")
    assert runtime == {"Mode": "LEGACY", "Served path": "Legacy", "Answered by": "Legacy deterministic extractor", "Query type": "contact"}
    assert "Plan F" not in " ".join(k for _t, rows in sections for k, _v in rows)
    evidence = rows_of(sections, "Evidence")
    assert evidence["Pages cited"] == "54" and evidence["Candidates retrieved"] == "24" and evidence["Chunks reranked"] == "5" and "Domain" not in evidence
    assert rows_of(sections, "LLM") == {"Legacy LLM calls": "0"} and rows_of(sections, "Knowledge base")["Total chunks"] == "13688"


def test_active_single_answer_panel_describes_the_runtime():
    d = plan_f_diag(routing_refined=True, routing_initial=["research", "institutional"], routing_final=["research"], routing_reason="ROUTE_REFINED_BY_REQUEST_KIND",
                    specialist_id="research", findings_request_kind="authority")
    result = plan_f_result(answer="• The Vice Chancellor for Academic Affairs approves travel requests and interprets the policy.", pages=[117],
                           items=[item(117, "• The Vice Chancellor ...", section="Participation in Conferences", chunk_type="paragraph")], query_type="policy")
    sections = P.dev_sections(result, d, {}, 700.0)
    assert [t for t, _r in sections] == ["Runtime", "Routing", "Verification", "Evidence", "Performance", "LLM"]
    runtime = rows_of(sections, "Runtime")
    assert runtime["Mode"] == "ACTIVE" and runtime["Served path"] == "Plan F" and runtime["Answered by"] == "Verified specialist answer (routing refined)"
    assert runtime["Plan F path"] == "single finding" and runtime["Plan F outcome"] == "PLAN_F_ACCEPTED" and "Fallback reason" not in runtime
    routing = rows_of(sections, "Routing")
    assert routing == {"Coordinator route": "Research & Innovation, Institutional Navigation", "Routing refined": "yes", "Final specialist(s)": "Research & Innovation",
                       "Routing reason": "ROUTE_REFINED_BY_REQUEST_KIND", "Specialist calls": "1"}
    verification = rows_of(sections, "Verification")
    assert verification == {"Findings verifier": "accepted", "Reason": "ACCEPTED", "Completeness": "complete", "Request kind": "authority", "Timing": "0.6 ms"}
    evidence = rows_of(sections, "Evidence")
    assert evidence["Domain"] == "Research & Innovation" and evidence["Pages cited"] == "117" and evidence["Sections"] == "Participation in Conferences" and evidence["Answer parts"] == "1"
    assert rows_of(sections, "LLM") == {"Plan F LLM calls": "0"}


def test_aggregate_and_handoff_panels_show_their_records():
    d = plan_f_diag(plan_f_path=P.PATH_AGGREGATE, specialist_id="teaching,faculty_services", aggregation_attempted=True, aggregation_accepted=True,
                    aggregation_reason="AGGREGATED_ACCEPTED", aggregation_clauses=2, aggregation_covered=2, aggregation_findings_considered=4,
                    aggregation_findings_accepted=3, aggregation_duplicates=0, aggregation_conflict=False, aggregation_pages=[96, 91, 107],
                    aggregation_specialists=["teaching", "faculty_services"], aggregation_ms=2.6, routing_initial=["teaching", "faculty_services"],
                    routing_final=["teaching", "faculty_services"], specialist_calls=2,
                    handoff_attempted=True, handoff_source=["faculty_services"], handoff_target=["research"], handoff_depth=1,
                    handoff_visited=["teaching", "faculty_services", "research"], handoff_outcome="HANDOFF_ACCEPTED", handoff_ms=0.2)
    items = [item(96, "a", section="Workload Allocation Model (WLAM)", chunk_type="paragraph"), item(91, LOAD, section="Workload Allocation Model (WLAM)", chunk_type="paragraph"),
             item(107, SICK, section="Leaves and Absences", chunk_type="paragraph")]
    result = plan_f_result(answer="a\n\n" + LOAD + "\n\n" + SICK, pages=[96, 91, 107], items=items, query_type="count",
                           timings={"specialist_ms": 1284.0, "handoff_ms": 0.2, "handoff_specialist_ms": 610.0, "aggregation_ms": 2.6, "verify_ms": 0.4, "generation_ms": 0.0, "total_ms": 1900.0})
    sections = P.dev_sections(result, d, {}, 1950.0)
    assert [t for t, _r in sections] == ["Runtime", "Routing", "Verification", "Aggregation", "Handoff", "Evidence", "Performance", "LLM"]
    assert rows_of(sections, "Runtime")["Answered by"] == "Verified multi-part answer (handoff completed)" and rows_of(sections, "Runtime")["Plan F path"] == "multi-finding aggregate"
    aggregation = rows_of(sections, "Aggregation")
    assert aggregation == {"Attempted": "yes", "Clauses covered": "2 of 2", "Aggregate accepted": "yes", "Reason": "AGGREGATED_ACCEPTED",
                           "Findings": "4 considered, 3 rendered", "Duplicates removed": "0", "Timing": "2.6 ms"}
    handoff = rows_of(sections, "Handoff")
    assert handoff == {"Attempted": "yes", "Source": "Faculty Services & HR", "Target": "Research & Innovation", "Depth": "1", "Outcome": "HANDOFF_ACCEPTED",
                       "Visited": "Teaching & Learning, Faculty Services & HR, Research & Innovation", "Timing": "0.2 ms"}
    evidence = rows_of(sections, "Evidence")
    assert evidence["Domain"] == "Teaching & Learning, Faculty Services & HR" and evidence["Pages cited"] == "96, 91, 107" and evidence["Answer parts"] == "3"
    assert evidence["Sections"] == "Workload Allocation Model (WLAM); Leaves and Absences"          # one entry per distinct section
    performance = rows_of(sections, "Performance")
    assert performance["Specialist execution"] == "1894.0 ms" and performance["Aggregation"] == "2.6 ms" and performance["Handoff round"] == "0.2 ms"


def test_fallback_and_error_panels():
    fallback = plan_f_diag(served="legacy", outcome="PLAN_F_UNSAFE_RESULT", accepted=False, fallback_reason="status partial", specialist_id="faculty_services",
                           findings_verified=False, findings_accepted=None, findings_reason=None, findings_completeness=None, findings_request_kind=None, findings_ms=None)
    sections = P.dev_sections(legacy_result(answer=NO_KEY_REPLY, query_type="policy"), fallback, {}, 1100.0)
    runtime = rows_of(sections, "Runtime")
    assert runtime["Served path"] == "Legacy (fallback after Plan F attempt)" and runtime["Fallback reason"] == "status partial" and runtime["Plan F outcome"] == "PLAN_F_UNSAFE_RESULT"
    assert rows_of(sections, "Verification")["Findings verifier"] == P.NA and rows_of(sections, "LLM") == {"Plan F LLM calls": "0", "Legacy LLM calls": "0"}
    assert "Domain" not in rows_of(sections, "Evidence")
    leaky = RuntimeError("boom at /home/someone/.secrets token=abcdefghijklmnopqrstuvwxyz0123")    # a path and a key-shaped token
    error = P.dev_sections(QAResult(answer=P.ERROR_TEXT), P.error_diagnostics(leaky), {}, None)
    runtime = rows_of(error, "Runtime")
    assert runtime["Error category"] == "RuntimeError" and runtime["Served path"] == "Error"
    assert "someone" not in str(error) and "abcdefghijklmnopqrstuvwxyz0123" not in str(error) and "boom" not in str(error)


def test_shadow_panel_shows_the_internal_attempt_while_the_answer_stays_legacy():
    shadow = plan_f_diag(mode="shadow", served="legacy", answer_equal=False, pages_equal=True, page_overlap=1, legacy_pages=[54], legacy_llm_calls=0)
    sections = P.dev_sections(legacy_result(), shadow, {}, 1300.0)
    runtime = rows_of(sections, "Runtime")
    assert runtime["Mode"] == "SHADOW" and runtime["Served path"] == "Legacy (Plan F in shadow)" and runtime["Answered by"] == "Legacy deterministic extractor · Plan F ran in shadow"
    assert runtime["Plan F outcome"] == "PLAN_F_ACCEPTED" and rows_of(sections, "Routing")["Specialist calls"] == "1"
    assert P.domain_labels(shadow) == []                                                        # no specialist label leaks into the visible answer


# ---------------------------------------------------------------------------
# Per-turn records
# ---------------------------------------------------------------------------
def test_each_turn_owns_a_copy_of_its_diagnostics():
    d = plan_f_diag()
    first = P.assistant_message(plan_f_result(), d, 640.0)
    d["served"] = "legacy"
    d["outcome"] = "PLAN_F_UNSAFE_RESULT"
    second = P.assistant_message(legacy_result(), d, 1100.0)
    assert first["diagnostics"]["served"] == "plan_f" and first["diagnostics"]["outcome"] == "PLAN_F_ACCEPTED" and first["wall_ms"] == 640.0
    assert second["diagnostics"]["served"] == "legacy" and second["wall_ms"] == 1100.0 and first["diagnostics"] is not second["diagnostics"]
    assert P.answer_kind(first["result"], first["diagnostics"]) == "Verified specialist answer"
    assert P.answer_kind(second["result"], second["diagnostics"]).startswith("Legacy fallback after Plan F attempt")
    none = P.assistant_message(legacy_result(), None, None)
    assert none["diagnostics"] == {} and none["wall_ms"] is None and none["content"] == legacy_result().answer
    assert P.error_diagnostics(ValueError("x")) == {"mode": None, "served": "error", "attempted": False, "error": "ValueError"}


def test_chat_module_consumes_the_turns_diagnostics_and_stays_free_of_runtime_names():
    source = (REPO / "ui" / "chat.py").read_text(encoding="utf-8")
    assert "ask_with_diagnostics" in source and "presentation.assistant_message(" in source and "presentation.escape_identifiers(" in source
    assert "presentation.dev_sections(" in source and "presentation.source_chips(" in source and "presentation.SEARCHING_TEXT" in source
    assert "ran into an error" not in source and "{exc}" not in source and "presentation.ERROR_TEXT" in source
    for forbidden in ("PLAN_F", "plan_f_runtime", "plan_f_aggregation", "plan_f_routing", "plan_f_handoff", "coordinator", "Coordinator"):
        assert forbidden not in source, forbidden
    assert 'aria-label="Send message"' in source and 'aria-label="Message the Onboarding Chatbot"' in source
    tree = ast.parse((REPO / "ui" / "presentation.py").read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    assert "nicegui" not in imported and not any(name.startswith("handbook_bot.plan_f") for name in imported if name)   # pure presentation, no runtime decision
