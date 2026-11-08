"""
Between a verified answer and the database (post-freeze PF-13).

``answer_record`` turns the page's turn (``QAResult`` + the seam's diagnostics + wall time) into plain JSON-able values:
the answer row, the evidence rows (one per evidence item, with the item's text and metadata exactly as retrieved) and
the user-facing fields. ``restore_turn`` rebuilds a ``QAResult`` and a diagnostics dict from those rows so the page's
unchanged renderers (``_render_message``, ``presentation.source_chips``, ``evidence_rows``, ``dev_sections``) show the
saved answer as it was shown: same text, same pages, same evidence, same translation state, same Developer details.
Nothing is re-verified or re-fetched; a restored turn is marked ``stored`` with its save time and the corpus key it was
answered against, so the page can say it is a saved answer and whether the documents have changed since.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from handbook_bot.qa import QAResult

#: How much of an evidence item's text is kept (the page shows at most 360 characters; the full chunk is rarely longer).
EXCERPT_LIMIT = 4000
#: Diagnostics keys worth keeping (codes, counts and pages; never the question or the answer text).
DIAGNOSTIC_KEYS = ("mode", "served", "attempted", "outcome", "reason", "specialist_id", "status", "accepted", "fallback_reason",
                   "plan_f_pages", "legacy_pages", "plan_f_sources", "legacy_sources", "guide_scope_enabled", "guide_scope_selected",
                   "guide_sources_named", "plan_f_guide_served", "guide_answer_forms", "guide_answer_parts", "guide_sources_inferred", "guide_inference",
                   "answer_equal", "pages_equal", "page_overlap", "plan_f_ms", "plan_f_timings", "registry_ok", "registry_reused", "registry_build_ms",
                   "plan_f_llm_calls", "plan_f_used_llm", "legacy_llm_calls", "legacy_used_llm", "findings_verified", "findings_accepted", "findings_reason",
                   "findings_completeness", "findings_request_kind", "findings_ms", "plan_f_path", "aggregation_attempted", "aggregation_accepted",
                   "aggregation_reason", "aggregation_clauses", "aggregation_covered", "aggregation_findings_considered", "aggregation_findings_accepted",
                   "aggregation_duplicates", "aggregation_conflict", "aggregation_pages", "aggregation_specialists", "aggregation_dispatched", "aggregation_ms",
                   "routing_refined", "routing_reason", "routing_initial", "routing_final", "routing_ms", "handoff_attempted", "handoff_source", "handoff_target",
                   "handoff_depth", "handoff_visited", "handoff_outcome", "handoff_ms", "specialist_calls", "error", "language")


def jsonable(value: Any, depth: int = 0) -> Any:
    """A JSON-serialisable copy: numpy scalars to Python numbers, sets and tuples to lists, non-finite floats to None,
    unknown objects to their string."""
    if depth > 12:
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and callable(getattr(value, "item")):      # numpy scalar
        try:
            return jsonable(value.item(), depth + 1)
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v, depth + 1) for v in value]
    if hasattr(value, "value") and hasattr(type(value), "__members__"):     # Enum
        return jsonable(value.value, depth + 1)
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return jsonable(value.to_dict(), depth + 1)
        except Exception:
            return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {k: jsonable(getattr(value, k), depth + 1) for k in value.__dataclass_fields__}
    return str(value)


def dumps(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))


def loads(text: Optional[str], default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def answer_record(result: Any, diagnostics: Optional[Dict[str, Any]], wall_ms: Optional[float], corpus_key: Optional[str] = None) -> Dict[str, Any]:
    """The rows for one assistant turn: ``answer`` (one row), ``evidence`` (a list of rows) and ``message`` fields."""
    diag = dict(diagnostics or {})
    kept = {k: jsonable(diag.get(k)) for k in DIAGNOSTIC_KEYS if k in diag}
    pages = [int(p) for p in (getattr(result, "pages", None) or []) if isinstance(p, int) and not isinstance(p, bool)]
    evidence: List[Dict[str, Any]] = []
    for item in getattr(result, "items", None) or []:
        if not isinstance(item, dict):
            continue
        position = len(evidence)
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        text = item.get("chunk") if isinstance(item.get("chunk"), str) else (item.get("text") if isinstance(item.get("text"), str) else "")
        page = meta.get("page")
        score = item.get("rerank_score")
        evidence.append({
            "position": position,
            "source_id": meta.get("source_id") if isinstance(meta.get("source_id"), str) else None,
            "page": int(page) if isinstance(page, int) and not isinstance(page, bool) else None,
            "section": (meta.get("section_title") or meta.get("section")) if isinstance(meta.get("section_title") or meta.get("section"), str) else None,
            "chunk_type": meta.get("chunk_type") if isinstance(meta.get("chunk_type"), str) else None,
            "excerpt": (text or "")[:EXCERPT_LIMIT],
            "meta": dumps(meta),
            "rerank_score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)) else None,
        })
    answer_language = str(getattr(result, "answer_language", "en") or "en")
    translation = getattr(result, "translation", None) or {}
    record = {
        "message": {"content": str(getattr(result, "answer", "") or ""), "answer_language": answer_language,
                    "status": "notice" if _is_notice(getattr(result, "answer", "")) else "complete"},
        "answer": {
            "served": diag.get("served"), "outcome": diag.get("outcome"), "specialist": diag.get("specialist_id"), "plan_f_path": diag.get("plan_f_path"),
            "query_type": getattr(result, "query_type", None), "pages": dumps(pages), "best_section": getattr(result, "best_section", None) or None,
            "evidence_preview": getattr(result, "evidence", None) or None, "used_llm": 1 if getattr(result, "used_llm", False) else 0,
            "llm_calls": int(getattr(result, "llm_calls", 0) or 0), "retried": 1 if getattr(result, "retried", False) else 0,
            "original_answer": getattr(result, "original_answer", None) or None, "translation": dumps(translation),
            "query_transform": dumps(getattr(result, "query_transform", None) or {}), "diagnostics": dumps(kept),
            "corpus_key": corpus_key, "wall_ms": float(wall_ms) if isinstance(wall_ms, (int, float)) and not isinstance(wall_ms, bool) else None,
        },
        "evidence": evidence,
    }
    return record


def _is_notice(answer: Any) -> bool:
    try:
        from ui.presentation import notice_kind
        return notice_kind(answer) is not None
    except Exception:
        return False


def restore_turn(message: Dict[str, Any], answer: Optional[Dict[str, Any]], evidence: Sequence[Dict[str, Any]]) -> Tuple[QAResult, Dict[str, Any], Dict[str, Any]]:
    """``(QAResult, diagnostics, extras)`` rebuilt from rows. ``extras`` carries ``stored_at`` and ``corpus_key`` for the page."""
    answer = answer or {}
    items: List[Dict[str, Any]] = []
    for row in sorted(evidence, key=lambda r: int(r.get("position") or 0)):
        meta = loads(row.get("meta"), {})
        if not isinstance(meta, dict):
            meta = {}
        for key, column in (("source_id", "source_id"), ("page", "page"), ("chunk_type", "chunk_type")):
            if row.get(column) is not None and key not in meta:
                meta[key] = row[column]
        item: Dict[str, Any] = {"chunk": row.get("excerpt") or "", "meta": meta}
        if row.get("rerank_score") is not None:
            item["rerank_score"] = row["rerank_score"]
        items.append(item)
    pages = [int(p) for p in loads(answer.get("pages"), []) if isinstance(p, int)]
    result = QAResult(
        answer=str(message.get("content") or ""), pages=pages, best_section=answer.get("best_section") or "",
        evidence=answer.get("evidence_preview") or "", query_type=answer.get("query_type") or "policy", items=items,
        used_llm=bool(answer.get("used_llm")), llm_calls=int(answer.get("llm_calls") or 0), retried=bool(answer.get("retried")),
        answer_language=str(message.get("answer_language") or "en"), original_answer=answer.get("original_answer") or "",
        translation=loads(answer.get("translation"), {}) or {}, query_transform=loads(answer.get("query_transform"), {}) or {},
    )
    diagnostics = loads(answer.get("diagnostics"), {}) or {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    diagnostics.setdefault("served", answer.get("served"))
    diagnostics.setdefault("specialist_id", answer.get("specialist"))
    diagnostics.setdefault("plan_f_path", answer.get("plan_f_path"))
    diagnostics.setdefault("outcome", answer.get("outcome"))
    extras = {"stored_at": message.get("created_at"), "corpus_key": answer.get("corpus_key"), "wall_ms": answer.get("wall_ms")}
    return result, diagnostics, extras


__all__ = ["DIAGNOSTIC_KEYS", "EXCERPT_LIMIT", "answer_record", "dumps", "jsonable", "loads", "restore_turn"]
