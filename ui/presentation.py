"""
Presentation helpers for the chat interface (runtime integration, phase I-7).

Pure functions, no NiceGUI import, no decision of their own: they read a finished ``QAResult`` and the diagnostics record
``ui.pipeline.ask_with_diagnostics`` returned for the SAME turn, and produce the labels, source chips, timing rows and
Developer-panel sections the chat page renders. Everything they state comes from those two inputs; a value the runtime did
not record is shown as ``N/A`` (never a made-up zero) or its row is left out.

What normal users see: the answer text (rendered as Markdown with technical identifiers kept intact), the source chips
(document, printed page and section) and, for an answer the verified specialist path served, a small domain indicator.
Reason codes, specialist ids, routing records, handoff records and error categories appear only in Developer mode.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from handbook_bot.agents.types import EXTRACTOR_ROUTES
from handbook_bot.config import GROQ_MODEL
from handbook_bot.orchestrator import NO_KEY_REPLY
from handbook_bot.qa import REFUSAL

#: The runtime's path names (``AdapterResult.path``), as the pipeline diagnostics report them.
PATH_SINGLE = "single_finding_passthrough"
PATH_AGGREGATE = "multi_finding_aggregate"
#: The adapter's outcome when no specialist applies (a greeting, an empty question, no domain cue): nothing was attempted.
NOT_APPLICABLE = "PLAN_F_NOT_APPLICABLE"
#: Specialist ids as the runtime names them, with the product names shown to users.
DOMAIN_NAMES: Mapping[str, str] = {
    "teaching": "Teaching & Learning",
    "research": "Research & Innovation",
    "faculty_services": "Faculty Services & HR",
    "institutional": "Institutional Navigation",
}
#: Shown while an answer is being produced (knowledge-base lookup, specialists, verification, or the legacy path).
SEARCHING_TEXT = "Searching official university sources…"
#: Shown instead of any exception text.
ERROR_TEXT = "Sorry, something went wrong while answering that. Please try again."
NA = "N/A"
#: ``best_section`` values that are placeholders, not handbook sections.
_PLACEHOLDER_SECTIONS = frozenset({"", "-", "Greeting", "Low confidence retrieval", "No matching evidence"})
_PAGE_PLACEHOLDER = re.compile(r"^\s*page\s+\d+\s*$", re.I)
_INTRAWORD_UNDERSCORE = re.compile(r"(?<=\w)_(?=\w)")
_CODE_SPAN = re.compile(r"(`+)[^`]*?\1")


# ---------------------------------------------------------------------------
# Markdown safety
# ---------------------------------------------------------------------------
def escape_identifiers(text: str) -> str:
    """Escape the underscores INSIDE words so that identifiers such as ``GROQ_API_KEY`` or ``add_drop`` render intact
    instead of as Markdown emphasis. Code spans are left untouched; the text is otherwise unchanged."""
    if not text:
        return text or ""
    out: List[str] = []
    last = 0
    for m in _CODE_SPAN.finditer(text):
        out.append(_INTRAWORD_UNDERSCORE.sub(r"\\_", text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_INTRAWORD_UNDERSCORE.sub(r"\\_", text[last:]))
    return "".join(out)


# ---------------------------------------------------------------------------
# Reading the turn
# ---------------------------------------------------------------------------
def _d(diagnostics: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(diagnostics) if isinstance(diagnostics, Mapping) else {}


def plan_f_attempted(diagnostics: Optional[Mapping[str, Any]]) -> bool:
    return bool(_d(diagnostics).get("attempted"))


def served_by_plan_f(diagnostics: Optional[Mapping[str, Any]]) -> bool:
    return _d(diagnostics).get("served") == "plan_f"


def mode_label(diagnostics: Optional[Mapping[str, Any]]) -> str:
    mode = _d(diagnostics).get("mode")
    return str(mode).upper() if mode else NA


def legacy_kind(result: Any) -> str:
    """How the legacy pipeline produced ``result``, as far as the result itself says."""
    answer = (getattr(result, "answer", "") or "").strip()
    if answer == REFUSAL or answer.rstrip(".") == REFUSAL.rstrip("."):
        return "Refusal (no supporting evidence)"
    if answer == NO_KEY_REPLY:
        return "Legacy model path (no model client configured)"
    if getattr(result, "used_llm", False):
        return "Legacy model synthesis (%s)" % GROQ_MODEL
    if getattr(result, "query_type", None) in EXTRACTOR_ROUTES:
        return "Legacy deterministic extractor"
    return "Legacy answer"


def answer_kind(result: Any, diagnostics: Optional[Mapping[str, Any]]) -> str:
    """The one line the Developer panel shows as "Answered by": never names a specialist path the user did not get."""
    d = _d(diagnostics)
    if d.get("served") == "error":
        return "Error (safe reply)"
    if served_by_plan_f(d):
        base = "Verified multi-part answer" if d.get("plan_f_path") == PATH_AGGREGATE else "Verified specialist answer"
        extras = []
        if d.get("routing_refined"):
            extras.append("routing refined")
        if d.get("handoff_attempted") and d.get("handoff_outcome") == "HANDOFF_ACCEPTED":
            extras.append("handoff completed")
        return base + (" (%s)" % ", ".join(extras) if extras else "")
    kind = legacy_kind(result)
    if plan_f_attempted(d) and d.get("outcome") == NOT_APPLICABLE:    # nothing ran: greeting, empty question, no domain cue
        return kind + " (Plan F not applicable)"
    if d.get("mode") == "active" and plan_f_attempted(d):
        return "Legacy fallback after Plan F attempt: " + kind
    if d.get("mode") == "shadow" and plan_f_attempted(d):
        return kind + " · Plan F ran in shadow"
    return kind


def path_label(diagnostics: Optional[Mapping[str, Any]]) -> str:
    path = _d(diagnostics).get("plan_f_path")
    return {PATH_SINGLE: "single finding", PATH_AGGREGATE: "multi-finding aggregate"}.get(path, NA if not path else str(path))


def served_label(diagnostics: Optional[Mapping[str, Any]]) -> str:
    d = _d(diagnostics)
    served = d.get("served")
    if served == "plan_f":
        return "Plan F"
    if served == "error":
        return "Error"
    if plan_f_attempted(d) and d.get("outcome") == NOT_APPLICABLE:
        return "Legacy (Plan F not applicable)"
    if d.get("mode") == "active" and plan_f_attempted(d):
        return "Legacy (fallback after Plan F attempt)"
    if d.get("mode") == "shadow":
        return "Legacy (Plan F in shadow)"
    return "Legacy"


def _names(ids: Any) -> List[str]:
    if isinstance(ids, str):
        ids = [s for s in re.split(r"[,+]", ids) if s]
    return [DOMAIN_NAMES.get(str(s), str(s)) for s in (ids or [])]


def domain_labels(diagnostics: Optional[Mapping[str, Any]]) -> List[str]:
    """Product names of the specialists whose findings the user is reading; empty unless the specialist path served."""
    d = _d(diagnostics)
    if not served_by_plan_f(d):
        return []
    ids = d.get("aggregation_specialists") if d.get("plan_f_path") == PATH_AGGREGATE else None
    if not ids:
        ids = d.get("routing_final") or d.get("specialist_id")
    return list(dict.fromkeys(_names(ids)))


# ---------------------------------------------------------------------------
# Source chips
# ---------------------------------------------------------------------------
def _section_of(meta: Mapping[str, Any]) -> Optional[str]:
    for key in ("section_title", "section"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in _PLACEHOLDER_SECTIONS and not _PAGE_PLACEHOLDER.match(value):
            return value.strip()
    return None


def ordered_pages(pages: Any) -> List[int]:
    """Distinct cited pages in the result's own order (first occurrence)."""
    out: List[int] = []
    for p in pages or []:
        if isinstance(p, bool):
            continue
        try:
            n = int(p)
        except (TypeError, ValueError):
            continue
        if n not in out:
            out.append(n)
    return out


def _title_of(meta: Mapping[str, Any]) -> str:
    title = meta.get("source_title")
    return title.strip() if isinstance(title, str) and title.strip() else ""


def source_ids(result: Any) -> List[str]:
    """Distinct source ids behind the evidence items, in citation order (Developer panel only; never a user-facing label)."""
    out: List[str] = []
    for item in (getattr(result, "items", None) or []):
        meta = item.get("meta") if isinstance(item, dict) and isinstance(item.get("meta"), dict) else {}
        sid = meta.get("source_id")
        if isinstance(sid, str) and sid and sid not in out:
            out.append(sid)
    return out


def source_chips(result: Any) -> List[Tuple[str, str]]:
    """(css class, label) chips: one chip per distinct cited page with the page's section when the evidence names one,
    the document once in front of them, an OCR marker when image text was used. Two evidence items on the same page with
    the same section give one chip; a page without a section is "Page N"; a refusal that cites no page gets no chip at
    all, so the handbook is never shown as the source of an answer it did not give; nothing is invented.

    Post-freeze PF-2: when the evidence comes from several documents, each document chip is followed by its own page
    chips (canonical document title, then ``Page N``), in the order the result cites them; a one-document result is
    rendered exactly as before. Raw source ids never appear in a chip."""
    items = [i for i in (getattr(result, "items", None) or []) if isinstance(i, dict)]
    pages = ordered_pages(getattr(result, "pages", None))
    chips: List[Tuple[str, str]] = []
    documents: List[str] = []
    sections: Dict[int, List[str]] = {}
    cited: List[Tuple[str, int]] = []                      # (document title, page) in evidence order
    doc_sections: Dict[Tuple[str, int], List[str]] = {}
    for item in items:
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        title = _title_of(meta)
        if title and title not in documents:
            documents.append(title)
        page = meta.get("page")
        section = _section_of(meta)
        if isinstance(page, int) and not isinstance(page, bool):
            if (title, page) not in cited:
                cited.append((title, page))
            if section:
                sections.setdefault(page, [])
                if section not in sections[page]:
                    sections[page].append(section)
                doc_sections.setdefault((title, page), [])
                if section not in doc_sections[(title, page)]:
                    doc_sections[(title, page)].append(section)

    def page_chip_rows(names: List[str], page: int) -> List[Tuple[str, str]]:
        if names:
            return [("page", "Page %d · %s" % (page, _shorten(name, 44))) for name in names]
        return [("page", "Page %d" % page)]

    if len(documents) <= 1:                                # the pre-PF-2 rendering, unchanged
        for page in pages:
            chips.extend(page_chip_rows(sections.get(page) or [], page))
        if pages and documents:                           # the document only when something in it is actually cited
            chips.insert(0, ("doc", _shorten(documents[0], 40)))
    else:
        page_set = set(pages)
        shown: set = set()
        for title in documents:
            own = [p for p in pages if (title, p) in cited and p in page_set]
            if not own:
                continue
            chips.append(("doc", _shorten(title, 40)))
            for page in own:
                chips.extend(page_chip_rows(doc_sections.get((title, page)) or [], page))
                shown.add(page)
        for page in pages:                                 # a cited page no document claims: shown without a document
            if page not in shown:
                chips.extend(page_chip_rows(sections.get(page) or [], page))
    if not pages:
        best = getattr(result, "best_section", None)
        if isinstance(best, str) and best.strip() and best.strip() not in _PLACEHOLDER_SECTIONS and not _PAGE_PLACEHOLDER.match(best):
            chips.append(("section", _shorten(best.strip(), 48)))
    if any((i.get("meta") or {}).get("chunk_type") == "image_ocr" for i in items):
        chips.append(("ocr", "image-OCR evidence"))
    return chips

def _shorten(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Timings and the Developer panel
# ---------------------------------------------------------------------------
def fmt_ms(value: Any) -> str:
    """A recorded duration; N/A for anything the runtime did not record. A recorded but tiny value is shown as
    "<0.1 ms" so it cannot be mistaken for an invented zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return NA
    if 0 < value < 0.05:
        return "<0.1 ms"
    return "%.1f ms" % float(value)


def _present(mapping: Mapping[str, Any], key: str) -> bool:
    return key in mapping and mapping[key] is not None


def performance_rows(result: Any, diagnostics: Optional[Mapping[str, Any]], wall_ms: Optional[float]) -> List[Tuple[str, str]]:
    """What each stage cost, as the runtime recorded it: the specialist path's own stages for a Plan F answer (the
    specialists time their execution as a whole, so retrieval and rerank are N/A there, never a zero), the legacy stages
    for a legacy answer, the Plan F attempt separately from the legacy time on a fallback, and the end-to-end wall time
    measured around the whole call. A stage the runtime did not time is N/A."""
    d = _d(diagnostics)
    t = dict(getattr(result, "timings", None) or {})
    rows: List[Tuple[str, str]] = []
    if served_by_plan_f(d):
        specialists = t.get("specialist_ms")
        if _present(t, "handoff_specialist_ms"):
            specialists = (specialists or 0.0) + float(t["handoff_specialist_ms"])
        rows.append(("Specialist execution", fmt_ms(specialists) if specialists is not None else NA))
        rows.append(("Retrieval", NA))                                   # not timed separately inside the specialists
        rows.append(("Rerank", NA))
        rows.append(("Routing refinement", fmt_ms(t.get("routing_ms")) if _present(t, "routing_ms") else NA))
        if _present(t, "findings_verify_ms"):
            rows.append(("Findings verification", fmt_ms(t["findings_verify_ms"])))
        if _present(t, "aggregation_ms"):
            rows.append(("Aggregation", fmt_ms(t["aggregation_ms"])))
        if _present(t, "handoff_ms"):
            rows.append(("Handoff round", fmt_ms(t["handoff_ms"])))
        rows.append(("Legacy Verifier", fmt_ms(t.get("verify_ms")) if _present(t, "verify_ms") else NA))
        rows.append(("Generation", fmt_ms(t.get("generation_ms")) if _present(t, "generation_ms") else NA))
        rows.append(("Plan F total", fmt_ms(d.get("plan_f_ms") if _present(d, "plan_f_ms") else t.get("total_ms"))))
        rows.append(("Legacy total", NA))
    else:
        if plan_f_attempted(d):
            rows.append(("Plan F attempt" + (" (shadow)" if d.get("mode") == "shadow" else ""), fmt_ms(d.get("plan_f_ms"))))
        rows.append(("Retrieval", fmt_ms(t.get("retrieval_ms")) if _present(t, "retrieval_ms") else NA))
        rows.append(("Rerank", fmt_ms(t.get("rerank_ms")) if _present(t, "rerank_ms") else NA))
        rows.append(("Generation", fmt_ms(t.get("generation_ms")) if _present(t, "generation_ms") else NA))
        rows.append(("Legacy total", fmt_ms(t.get("total_ms")) if _present(t, "total_ms") else NA))
    rows.append(("End-to-end", fmt_ms(wall_ms)))
    if plan_f_attempted(d) and _present(d, "registry_ok") and d.get("outcome") != NOT_APPLICABLE:
        if not d.get("registry_ok"):
            registry = "unavailable"
        elif d.get("registry_reused"):
            registry = "reused"
        else:
            registry = "built (%s)" % fmt_ms(d.get("registry_build_ms"))
        rows.append(("Specialist registry", registry))
    return rows


def _yes_no(value: Any) -> str:
    return "yes" if value else "no"


def dev_sections(result: Any, diagnostics: Optional[Mapping[str, Any]], kb_stats: Optional[Mapping[str, Any]] = None,
                 wall_ms: Optional[float] = None) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """The Developer panel: (section title, rows). Sections that do not apply to the turn are left out; a value the
    runtime did not record is N/A."""
    d = _d(diagnostics)
    attempted = plan_f_attempted(d)
    sections: List[Tuple[str, List[Tuple[str, str]]]] = []

    runtime = [("Mode", mode_label(d)), ("Served path", served_label(d)), ("Answered by", answer_kind(result, d))]
    if attempted:
        runtime.append(("Plan F path", path_label(d)))
        if d.get("outcome"):
            runtime.append(("Plan F outcome", str(d["outcome"])))
        if d.get("served") != "plan_f" and d.get("fallback_reason"):
            runtime.append(("Fallback reason", _shorten(str(d["fallback_reason"]), 120)))
    runtime.append(("Query type", str(getattr(result, "query_type", None) or NA).replace("_", " ")))
    if d.get("error"):
        runtime.append(("Error category", _shorten(str(d["error"]), 80)))
    sections.append(("Runtime", runtime))

    if attempted:
        routing = [("Coordinator route", ", ".join(_names(d.get("routing_initial"))) or NA),
                   ("Routing refined", _yes_no(d.get("routing_refined")) if _present(d, "routing_refined") else NA),
                   ("Final specialist(s)", ", ".join(_names(d.get("routing_final"))) or NA),
                   ("Routing reason", str(d.get("routing_reason") or NA)),
                   ("Specialist calls", str(d["specialist_calls"]) if _present(d, "specialist_calls") else NA)]
        sections.append(("Routing", routing))
        verification = [("Findings verifier", ("accepted" if d.get("findings_accepted") else "rejected") if d.get("findings_verified") else NA),
                        ("Reason", str(d.get("findings_reason") or NA)),
                        ("Completeness", str(d.get("findings_completeness") or NA)),
                        ("Request kind", str(d.get("findings_request_kind") or NA)),
                        ("Timing", fmt_ms(d.get("findings_ms")))]
        sections.append(("Verification", verification))
        if d.get("aggregation_attempted"):
            aggregation = [("Attempted", "yes"),
                           ("Clauses covered", "%s of %s" % (d.get("aggregation_covered", NA), d.get("aggregation_clauses", NA))),
                           ("Aggregate accepted", _yes_no(d.get("aggregation_accepted"))),
                           ("Reason", str(d.get("aggregation_reason") or NA)),
                           ("Findings", "%s considered, %s rendered" % (d.get("aggregation_findings_considered", NA), d.get("aggregation_findings_accepted", NA))),
                           ("Duplicates removed", str(d.get("aggregation_duplicates", NA))),
                           ("Timing", fmt_ms(d.get("aggregation_ms")))]
            sections.append(("Aggregation", aggregation))
        if d.get("handoff_attempted"):
            handoff = [("Attempted", "yes"),
                       ("Source", ", ".join(_names(d.get("handoff_source"))) or NA),
                       ("Target", ", ".join(_names(d.get("handoff_target"))) or NA),
                       ("Depth", str(d.get("handoff_depth", NA))),
                       ("Outcome", str(d.get("handoff_outcome") or NA)),
                       ("Visited", ", ".join(_names(d.get("handoff_visited"))) or NA),
                       ("Timing", fmt_ms(d.get("handoff_ms")))]
            sections.append(("Handoff", handoff))

    chips = source_chips(result)
    pages = ordered_pages(getattr(result, "pages", None))
    evidence = [("Pages cited", ", ".join(str(p) for p in pages) or NA),
                ("Document", "; ".join(label for cls, label in chips if cls == "doc") or NA),
                ("Source ids", ", ".join(source_ids(result)) or NA),
                ("Sections", "; ".join(dict.fromkeys(label.split(" · ", 1)[1] for cls, label in chips if cls == "page" and " · " in label)) or NA),
                ("Evidence items", str(len(getattr(result, "items", None) or []))),
                ("Image-OCR evidence", _yes_no(any(((i.get("meta") or {}).get("chunk_type") == "image_ocr") for i in (getattr(result, "items", None) or []) if isinstance(i, dict))))]
    if served_by_plan_f(d):
        evidence.insert(0, ("Domain", ", ".join(domain_labels(d)) or NA))
        parts = str(getattr(result, "answer", "") or "").count("\n\n") + 1 if getattr(result, "answer", "") else 0
        evidence.append(("Answer parts", str(parts)))
    else:
        evidence.append(("Candidates retrieved", str(getattr(result, "num_candidates", NA))))
        evidence.append(("Chunks reranked", str(getattr(result, "num_reranked", NA))))
    sections.append(("Evidence", evidence))

    sections.append(("Performance", performance_rows(result, d, wall_ms)))

    llm: List[Tuple[str, str]] = []
    if attempted:
        llm.append(("Plan F LLM calls", str(d.get("plan_f_llm_calls", NA))))
    if d.get("served") != "plan_f":
        llm.append(("Legacy LLM calls", str(getattr(result, "llm_calls", NA))))
    elif _present(d, "legacy_llm_calls"):
        llm.append(("Legacy LLM calls", str(d["legacy_llm_calls"])))
    sections.append(("LLM", llm))

    stats = dict(kb_stats or {})
    if stats:
        sections.append(("Knowledge base", [("Total chunks", str(stats.get("total_chunks", NA))), ("Total pages", str(stats.get("total_pages", NA))),
                                            ("OCR chunks", str(stats.get("ocr_chunks", NA)))]))
    return sections


# ---------------------------------------------------------------------------
# The per-turn record
# ---------------------------------------------------------------------------
def assistant_message(result: Any, diagnostics: Optional[Mapping[str, Any]], wall_ms: Optional[float]) -> Dict[str, Any]:
    """The chat record of one answered turn. It owns a COPY of the diagnostics of that call, so a later turn can never
    change what an earlier bubble's Developer panel shows."""
    return {"role": "assistant", "content": getattr(result, "answer", "") or "", "result": result, "diagnostics": _d(diagnostics),
            "wall_ms": float(wall_ms) if isinstance(wall_ms, (int, float)) and not isinstance(wall_ms, bool) else None}


def error_diagnostics(exc: BaseException) -> Dict[str, Any]:
    """What Developer mode may know about a failed turn: the exception category, never its text."""
    return {"mode": None, "served": "error", "attempted": False, "error": type(exc).__name__}


__all__ = ["DOMAIN_NAMES", "ERROR_TEXT", "NA", "NOT_APPLICABLE", "PATH_AGGREGATE", "PATH_SINGLE", "SEARCHING_TEXT", "answer_kind", "assistant_message",
           "dev_sections", "domain_labels", "error_diagnostics", "escape_identifiers", "fmt_ms", "legacy_kind", "mode_label", "ordered_pages",
           "path_label", "performance_rows", "plan_f_attempted", "served_by_plan_f", "served_label", "source_chips"]
