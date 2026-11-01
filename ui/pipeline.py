"""
Thin wrapper around the handbook_bot pipeline.

Keeps the UI layer decoupled from backend internals. Since runtime integration
phase I-2 this module is also the ONE approved seam between the legacy
pipeline and the Plan F runtime adapter (``handbook_bot/plan_f_runtime.py``):

    PLAN_F_ENABLED off, PLAN_F_SHADOW off   LEGACY   ``answer_question`` exactly as before;
                                                     Plan F is neither imported nor built
    PLAN_F_SHADOW on                        SHADOW   the legacy answer is returned unchanged;
                                                     Plan F also runs, for diagnostics only
    PLAN_F_ENABLED on, PLAN_F_SHADOW off    ACTIVE   Plan F first; its answer is returned only
                                                     when the adapter accepted it AND the
                                                     findings-aware verification layer (phase
                                                     I-4) accepted it (on the aggregation path
                                                     of phase I-5: the aggregate and every
                                                     clause's own record), else legacy

Both switches default to off (``handbook_bot.config``); SHADOW takes
precedence when both are on. ``ask(kb, question) -> QAResult`` is unchanged
for the UI. ``ask_with_diagnostics`` returns the Plan F diagnostics beside the
result for tests and measurement: outcomes, pages, timings and redacted
errors, never the question or the answer text. No Plan F failure can raise
through ``ask``; legacy failures keep their existing behaviour. The Plan F
runtime is built lazily, once per knowledge-base object, by
``plan_f_runtime.RuntimeCache``, and never in LEGACY mode.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from handbook_bot import KnowledgeBase, QAResult, answer_question, build_knowledge_base
from handbook_bot import config as _config

MODE_LEGACY = "legacy"
MODE_SHADOW = "shadow"
MODE_ACTIVE = "active"
#: The only adapter outcome that may put a Plan F answer in front of a user (``PlanFOutcome.ACCEPTED``).
_ACCEPTED = "PLAN_F_ACCEPTED"

_log = logging.getLogger(__name__)
_PLAN_F_LOCK = threading.Lock()
_PLAN_F_CACHE: Any = None                              # plan_f_runtime.RuntimeCache, created on the first SHADOW or ACTIVE question
_RECENT: Deque[Dict[str, Any]] = deque(maxlen=200)     # recent Plan F diagnostics for measurement; no question or answer text


@dataclass
class StartupCheck:
    ok: bool
    message: str = ""


def preflight() -> StartupCheck:
    """Cheap sanity checks before the slow KB build."""
    found = False
    for directory in ("data", "."):
        if not os.path.isdir(directory):
            continue
        for fname in os.listdir(directory):
            if fname.lower().endswith(".pdf"):
                found = True
                break
        if found:
            break
    if not found:
        return StartupCheck(
            ok=False,
            message=(
                "No handbook PDF was found. Place the UOS Faculty Handbook "
                "PDF inside the 'data/' folder and restart the app."
            ),
        )
    return StartupCheck(ok=True)


def load_knowledge_base(verbose: bool = True) -> KnowledgeBase:
    """Build the knowledge base (embeddings + index). Called once at startup."""
    return build_knowledge_base(verbose=verbose)


# ---------------------------------------------------------------------------
# Operating mode
# ---------------------------------------------------------------------------
def plan_f_mode(enabled: Optional[bool] = None, shadow: Optional[bool] = None) -> str:
    """The operating mode for the configured switches (or the given ones): SHADOW whenever ``PLAN_F_SHADOW`` is on,
    ACTIVE when only ``PLAN_F_ENABLED`` is on, LEGACY otherwise."""
    enabled = _config.PLAN_F_ENABLED if enabled is None else bool(enabled)
    shadow = _config.PLAN_F_SHADOW if shadow is None else bool(shadow)
    if shadow:
        return MODE_SHADOW
    if enabled:
        return MODE_ACTIVE
    return MODE_LEGACY


# ---------------------------------------------------------------------------
# The legacy path (unchanged call)
# ---------------------------------------------------------------------------
def _legacy_answer(kb: KnowledgeBase, question: str) -> QAResult:
    """The Milestone 1 path, called exactly as this module has always called it."""
    return answer_question(
        question,
        embedder=kb.embedder,
        reranker=kb.reranker,
        index=kb.index,
        chunks=kb.chunks,
        metadata=kb.metadata,
        pages=kb.pages,
        groq_client=kb.groq_client,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def ask(kb: KnowledgeBase, question: str) -> QAResult:
    """Answer a single question against the loaded knowledge base."""
    return ask_with_diagnostics(kb, question)[0]


def ask_with_diagnostics(kb: KnowledgeBase, question: str) -> Tuple[QAResult, Dict[str, Any]]:
    """``ask`` plus the Plan F diagnostics of this question (a JSON-serialisable dict).

    LEGACY: the legacy result and a minimal record; Plan F is not imported. SHADOW: the legacy result, produced first
    and returned unchanged, with the Plan F attempt recorded beside it. ACTIVE: the Plan F result when the adapter
    accepted it, else the legacy result with the fallback reason.
    """
    mode = plan_f_mode()
    if mode == MODE_LEGACY:
        return _legacy_answer(kb, question), {"mode": MODE_LEGACY, "served": "legacy", "attempted": False}
    if mode == MODE_SHADOW:
        result = _legacy_answer(kb, question)                      # first and unchanged: nothing below can touch it
        attempt, entry, error = _attempt_plan_f(kb, question)
        diagnostics = _diagnose(mode, "legacy", attempt, entry, error, legacy=result)
        _record(diagnostics)
        return result, diagnostics
    attempt, entry, error = _attempt_plan_f(kb, question)
    if attempt is not None and _outcome_of(attempt) == _ACCEPTED and getattr(attempt, "result", None) is not None and _verified(attempt):
        diagnostics = _diagnose(mode, "plan_f", attempt, entry, error, legacy=None)
        _record(diagnostics)
        return attempt.result, diagnostics
    result = _legacy_answer(kb, question)
    diagnostics = _diagnose(mode, "legacy", attempt, entry, error, legacy=result)
    _record(diagnostics)
    return result, diagnostics


def recent_plan_f_diagnostics(limit: int = 50) -> List[Dict[str, Any]]:
    """The most recent diagnostics (newest last), for measurement in SHADOW or ACTIVE mode."""
    with _PLAN_F_LOCK:
        items = list(_RECENT)
    return items[-max(0, int(limit)):] if limit else []


def reset_plan_f_state() -> None:
    """Forget the runtime cache and the recent diagnostics (tests, or after a knowledge base is rebuilt)."""
    global _PLAN_F_CACHE
    with _PLAN_F_LOCK:
        _PLAN_F_CACHE = None
        _RECENT.clear()


def groq_is_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Plan F attempt (lazy import; never raises)
# ---------------------------------------------------------------------------
def _plan_f_cache() -> Any:
    """The process-wide runtime cache, created on first use and therefore never in LEGACY mode."""
    global _PLAN_F_CACHE
    with _PLAN_F_LOCK:
        if _PLAN_F_CACHE is None:
            from handbook_bot import plan_f_runtime as plan_f       # lazy on purpose: LEGACY mode never imports Plan F
            _PLAN_F_CACHE = plan_f.RuntimeCache()
        return _PLAN_F_CACHE


def _attempt_plan_f(kb: KnowledgeBase, question: str) -> Tuple[Any, Any, Optional[str]]:
    """(adapter result or None, cache entry or None, redacted error or None). Never raises."""
    entry = None
    try:
        entry = _plan_f_cache().get_or_build(kb)
        if entry.runtime is None:
            if not entry.reused:
                _log.warning("Plan F registry unavailable for this knowledge base; legacy answers only: %s", entry.build.errors)
            return None, entry, None
        return entry.runtime.attempt(question), entry, None
    except Exception as exc:                                       # the seam never raises for Plan F
        return None, entry, _redact(exc)


def _redact(exc: BaseException) -> str:
    try:
        from handbook_bot.agents.specialists.base import redact_error
        return redact_error(exc)
    except Exception:
        return type(exc).__name__


def _outcome_of(attempt: Any) -> Optional[str]:
    return getattr(getattr(attempt, "outcome", None), "value", None)


def _accepted_record(record: Any) -> bool:
    return record is not None and bool(getattr(record, "accepted", False)) and bool(getattr(record, "safe_to_render", False))


def _verified(attempt: Any) -> bool:
    """Whether the findings-aware verification layer (phase I-4) accepted the attempt's result for rendering. ACTIVE mode
    serves a Plan F answer only when the adapter accepted it AND this layer did: an attempt without a verification
    record is never served. On the aggregation path (phase I-5) the aggregate itself and EVERY clause's own
    verification record must be accepted."""
    aggregation = getattr(attempt, "aggregation", None)
    if aggregation is not None:
        clauses = list(getattr(aggregation, "clauses", ()) or ())
        return _accepted_record(aggregation) and bool(clauses) and all(_accepted_record(getattr(c, "verification", None)) for c in clauses)
    return _accepted_record(getattr(attempt, "verification", None))


_HANDBOOK_SOURCE_ID = "uos_faculty_handbook_2025_26"


def _source_ids_of(result: Any) -> List[str]:
    """Distinct source ids behind a result's evidence items, in citation order (post-freeze PF-2; ids only)."""
    out: List[str] = []
    for item in (getattr(result, "items", None) or []):
        meta = item.get("meta") if isinstance(item, dict) and isinstance(item.get("meta"), dict) else {}
        sid = meta.get("source_id")
        if isinstance(sid, str) and sid and sid not in out:
            out.append(sid)
    return out


def _diagnose(mode: str, served: str, attempt: Any, entry: Any, error: Optional[str], *, legacy: Optional[QAResult]) -> Dict[str, Any]:
    """The diagnostic record of one question: outcomes, pages, timings and redacted errors only."""
    d: Dict[str, Any] = {
        "mode": mode, "served": served, "attempted": attempt is not None, "outcome": None, "reason": "",
        "specialist_id": None, "status": None, "accepted": False, "fallback_reason": None,
        "plan_f_pages": [], "legacy_pages": list(legacy.pages) if legacy is not None else None,
        "plan_f_sources": [], "legacy_sources": _source_ids_of(legacy) if legacy is not None else None,
        "guide_scope_enabled": None, "guide_scope_selected": None, "guide_sources_named": [], "plan_f_guide_served": False,
        "guide_answer_forms": [], "guide_answer_parts": 0, "guide_sources_inferred": [], "guide_inference": None,
        "answer_equal": None, "pages_equal": None, "page_overlap": 0,
        "plan_f_ms": None, "plan_f_timings": {},
        "registry_ok": None, "registry_reused": None, "registry_build_ms": None, "registry_errors": {},
        "plan_f_llm_calls": 0, "plan_f_used_llm": False,
        "legacy_llm_calls": legacy.llm_calls if legacy is not None else None,
        "legacy_used_llm": legacy.used_llm if legacy is not None else None,
        "findings_verified": False, "findings_accepted": None, "findings_reason": None, "findings_completeness": None,
        "findings_request_kind": None, "findings_ms": None,
        "plan_f_path": None, "aggregation_attempted": False, "aggregation_accepted": None, "aggregation_reason": None,
        "aggregation_clauses": None, "aggregation_covered": None, "aggregation_findings_considered": None,
        "aggregation_findings_accepted": None, "aggregation_duplicates": None, "aggregation_conflict": None, "aggregation_pages": [],
        "aggregation_specialists": [], "aggregation_dispatched": None, "aggregation_ms": None,
        "routing_refined": None, "routing_reason": None, "routing_initial": [], "routing_final": [], "routing_ms": None,
        "handoff_attempted": False, "handoff_source": [], "handoff_target": [], "handoff_depth": 0, "handoff_visited": [],
        "handoff_outcome": None, "handoff_ms": None, "specialist_calls": None,
        "error": error,
    }
    try:
        if entry is not None:
            d["registry_ok"] = entry.runtime is not None
            d["registry_reused"] = bool(entry.reused)
            d["registry_build_ms"] = float(entry.build.ms)
            d["registry_errors"] = dict(entry.build.errors)
            if entry.runtime is None:
                d["fallback_reason"] = "registry unavailable"
        if attempt is not None:
            outcome = _outcome_of(attempt)
            d["outcome"] = outcome
            d["reason"] = str(getattr(attempt, "reason", ""))[:200]
            d["specialist_id"] = getattr(attempt, "specialist_id", None)
            d["status"] = getattr(attempt, "status", None)
            d["accepted"] = outcome == _ACCEPTED and getattr(attempt, "result", None) is not None
            d["plan_f_sources"] = _source_ids_of(getattr(attempt, "result", None)) if d["accepted"] else []
            guide = getattr(attempt, "guide_scope", None) or {}        # post-freeze PF-3B: source ids and flags only
            d["guide_scope_enabled"] = bool(guide.get("enabled"))
            d["guide_scope_selected"] = bool(guide.get("selected"))
            d["guide_sources_named"] = [str(s) for s in (guide.get("named") or [])]
            d["guide_sources_inferred"] = [str(s) for s in (guide.get("inferred") or [])]     # post-freeze PF-5C (D-115)
            inference = guide.get("inference")
            if isinstance(inference, dict) and isinstance(inference.get("lead"), dict):
                lead = inference["lead"]
                d["guide_inference"] = {k: lead.get(k) for k in ("accepted", "reason", "family", "candidates", "score", "runner_up",
                                                                 "runner_up_score", "margin", "signals")}
            d["plan_f_guide_served"] = bool(d["accepted"]) and any(s != _HANDBOOK_SOURCE_ID for s in d["plan_f_sources"])
            if d["accepted"]:                                       # post-freeze PF-3C (D-111): the form of each guide answer, no text
                records = [((i.get("meta") or {}).get("plan_f") or {}).get("guide") for i in (attempt.result.items or [])]
                records = [r for r in records if isinstance(r, dict)]
                d["guide_answer_forms"] = sorted({str(r.get("form")) for r in records})
                d["guide_answer_parts"] = len(records)
            timings = dict(getattr(attempt, "timings", {}) or {})
            d["plan_f_ms"] = float(timings.get("total_ms", 0.0))
            d["plan_f_timings"] = {k: float(v) for k, v in timings.items()}
            dispatched = getattr(attempt, "dispatch", None)
            d["plan_f_llm_calls"] = int(getattr(dispatched, "llm_calls", 0) or 0)
            d["plan_f_used_llm"] = any(bool(getattr(e, "used_llm", False)) for e in getattr(attempt, "trace", []) or [])
            d["plan_f_path"] = getattr(attempt, "path", None)
            verification = getattr(attempt, "verification", None)
            if verification is not None:                           # phase I-4: codes only, never the question or the answer
                d["findings_verified"] = True
                d["findings_accepted"] = _accepted_record(verification)
                d["findings_reason"] = getattr(getattr(verification, "reason", None), "value", None)
                d["findings_completeness"] = getattr(getattr(verification, "completeness", None), "value", None)
                d["findings_request_kind"] = getattr(getattr(verification, "request_kind", None), "value", None)
                d["findings_ms"] = float(getattr(verification, "ms", 0.0) or 0.0)
            aggregation = getattr(attempt, "aggregation", None)
            if aggregation is not None:                            # phase I-5: counts, codes and pages only
                d["aggregation_attempted"] = True
                d["aggregation_accepted"] = _accepted_record(aggregation)
                d["aggregation_reason"] = getattr(getattr(aggregation, "reason", None), "value", None)
                d["aggregation_clauses"] = int(getattr(aggregation, "clause_count", 0) or 0)
                d["aggregation_covered"] = int(getattr(aggregation, "covered_count", 0) or 0)
                d["aggregation_findings_considered"] = int(getattr(aggregation, "findings_considered", 0) or 0)
                d["aggregation_findings_accepted"] = int(getattr(aggregation, "findings_accepted", 0) or 0)
                d["aggregation_duplicates"] = len(getattr(aggregation, "duplicates", ()) or ())
                d["aggregation_conflict"] = bool(getattr(aggregation, "conflicts", ()) or ())
                d["aggregation_pages"] = [int(p) for p in (getattr(aggregation, "pages", ()) or ())]
                d["aggregation_specialists"] = [str(s) for s in (getattr(aggregation, "specialists", ()) or ())]
                d["aggregation_dispatched"] = bool(getattr(aggregation, "dispatched", False))
                d["aggregation_ms"] = float(getattr(aggregation, "ms", 0.0) or 0.0)
                records = [getattr(c, "verification", None) for c in (getattr(aggregation, "clauses", ()) or ())]
                if records:                                        # the clause records stand in for the single I-4 record
                    d["findings_verified"] = all(r is not None for r in records)
                    d["findings_accepted"] = all(_accepted_record(r) for r in records)
                    failing = next((r for r in records if not _accepted_record(r)), None)
                    d["findings_reason"] = getattr(getattr(failing, "reason", None), "value", None) if failing is not None else "ACCEPTED"
                    classes = [getattr(getattr(r, "completeness", None), "value", None) for r in records if r is not None]
                    d["findings_completeness"] = next((c for c in classes if c not in ("complete", "elaborated")), classes[0] if classes else None)
                    d["findings_ms"] = sum(float(getattr(r, "ms", 0.0) or 0.0) for r in records if r is not None)
            routing = getattr(attempt, "routing", None)
            if routing is not None:                                # phase I-6: codes and specialist ids only
                d["routing_refined"] = bool(getattr(routing, "refined", False))
                d["routing_reason"] = getattr(getattr(routing, "reason", None), "value", None)
                d["routing_initial"] = [str(s) for s in (getattr(routing, "initial_specialists", ()) or ())]
                d["routing_final"] = [str(s) for s in (getattr(routing, "final_specialists", ()) or ())]
                d["routing_ms"] = float(getattr(routing, "ms", 0.0) or 0.0)
            handoff = getattr(attempt, "handoff", None)
            if handoff is not None:
                d["handoff_attempted"] = bool(getattr(handoff, "attempted", False))
                d["handoff_source"] = [str(s) for s in (getattr(handoff, "sources", ()) or ())]
                d["handoff_target"] = [str(s) for s in (getattr(handoff, "targets", ()) or ())]
                d["handoff_depth"] = int(getattr(handoff, "depth", 0) or 0)
                d["handoff_visited"] = [str(s) for s in (getattr(handoff, "visited", ()) or ())]
                d["handoff_outcome"] = getattr(attempt, "handoff_outcome", None) or getattr(getattr(handoff, "code", None), "value", None)
                d["handoff_ms"] = float(timings.get("handoff_ms", 0.0) or 0.0)
            d["specialist_calls"] = int(getattr(attempt, "specialist_calls", 0) or 0)
            d["accepted"] = d["accepted"] and _verified(attempt)   # accepted means: the adapter AND the findings-aware layer accepted
            if getattr(attempt, "result", None) is not None:
                d["plan_f_pages"] = list(attempt.result.pages)
                if legacy is not None:
                    d["answer_equal"] = attempt.result.answer == legacy.answer
                    d["pages_equal"] = list(attempt.result.pages) == list(legacy.pages)
                    d["page_overlap"] = len(set(attempt.result.pages) & set(legacy.pages))
            if not d["accepted"]:
                d["fallback_reason"] = d["reason"] or d["fallback_reason"]
        elif error is not None:
            d["fallback_reason"] = "plan_f error: " + error
        elif d["fallback_reason"] is None:
            d["fallback_reason"] = "plan_f not attempted"
    except Exception as exc:                                       # diagnostics must never fail a question
        d["error"] = ((d.get("error") or "") + " diagnostics: " + type(exc).__name__).strip()
    return d


def _record(diagnostics: Dict[str, Any]) -> None:
    with _PLAN_F_LOCK:
        _RECENT.append(diagnostics)
    _log.debug("plan_f %s", json.dumps(diagnostics, default=str))
