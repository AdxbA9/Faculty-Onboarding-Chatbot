"""
Orchestrator: deterministic control flow for one question.

THIS IS NOT AN AGENT AND IT NEVER CALLS AN LLM TO DECIDE ANYTHING.
The order of work is fixed and known in advance, so it is ordinary Python:

    route -> gather evidence -> extractor or synthesis -> verify -> (one retry) -> result

An LLM "manager" would add a call to every question, make the control flow
non-deterministic and make it untestable. Agents own decisions *inside* a step
(which route, how to word the answer, whether it is supported). The
orchestrator owns only the sequence, the LLM-call budget and the retry cap.

RULES ENFORCED HERE
    * Agents never call each other. Each is called from here and returns a
      structured result from ``agents.types``; this module decides what is next.
    * Every LLM call is counted. A call that would exceed ``MAX_LLM_CALLS`` is
      not made. A Verifier-requested retry draws from a separate allowance of
      ``MAX_VERIFY_RETRIES``, which is clamped to 1: a draft is retried once or
      not at all. Hard ceiling per question: MAX_LLM_CALLS + 1.
    * The contact/count/date extractors stay plain tools with their zero-LLM
      fast path. There is no "contact agent".
    * A failure of any LLM call never raises to the caller and is never
      retried silently.

MILESTONE 1 SCOPE
    ``RouteDecision.is_multipart`` is recorded in the trace and otherwise
    ignored: the Planner and the Evidence agent are Milestone 2. With
    ``PLANNER_ENABLED`` off (or no planner module present) this is the
    Option A pipeline.

VERIFIER INTEGRATION (agents/verifier.py is owned by another engineer)
    If ``handbook_bot.agents.verifier`` exposes a callable ``verify`` it is
    used, with this exact call:

        verify(question: str,
               synthesis: SynthesisResult,
               evidence: list[EvidenceResult],      # one entry in Milestone 1
               sub_questions: list[SubQuestion],    # empty in Milestone 1
               *, is_retry: bool) -> VerifyResult

    ``evidence[0].used_extractor`` tells it the draft came from a deterministic
    extractor. Until that module exists the Part 1 check (``qa.verify_answer``
    and the Part 1 page rule) is wrapped in the same ``VerifyResult`` shape. It
    is reproduced unchanged, known defects included, and it never asks for a
    retry: fixing it is the Verifier owner's task, not this module's.
"""
from __future__ import annotations

import importlib
import importlib.util
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .agents import router as router_agent
from .agents import synthesis as synthesis_agent
from .agents.types import (
    EXTRACTOR_ROUTES,
    EvidenceResult,
    RouteDecision,
    SubQuestion,
    SynthesisResult,
    TraceEntry,
    VerifyResult,
)
from .config import (
    EVIDENCE_PREVIEW_CHARS,
    FINAL_K,
    MAX_LLM_CALLS,
    MAX_VERIFY_RETRIES,
    MIN_RERANK_SCORE,
    PLANNER_ENABLED,
    VERIFY_ANSWERS,
)
from .extractors import (
    extract_contact_answer,
    extract_count_answer,
    extract_date_answer,
)
from .qa import REFUSAL, QAResult, legacy_fallback_pages, verify_answer
from .retrieval import deduplicate_by_text, gather_candidates
from .text_utils import normalize_text

GREETING_REPLY = "Hello! Ask me a question about the University of Sharjah Faculty Handbook."
NO_KEY_REPLY = "GROQ_API_KEY is not set. Please set it, then reload the app."
#: Shown when the language model could not be reached. Deliberately NOT the
#: refusal sentence: "the handbook does not say" and "the service is down" are
#: different facts, and an outage must not be scored as a correct refusal.
LLM_UNAVAILABLE_REPLY = (
    "I could not generate an answer right now because the language-model "
    "service did not respond. Please try again."
)

VerifierFn = Callable[..., VerifyResult]

#: What the count extractor is able to count. ``extract_count_answer`` knows
#: only the handbook's degree-program totals and returns them for ANY question
#: it is handed ("What is the total teaching load?" -> "UoS offers 149 total
#: degree programs."). The Router now sends every "number of ..." question to
#: ``count`` (Part 1 mis-routed many of them to ``contact``), so without this
#: guard better routing would produce more confidently wrong answers. A count
#: question outside this scope skips the tool and goes to the LLM path, where
#: the grounding check requires its numbers to appear in the evidence.
#: Temporary: the real fix is in extractors.py (week 7, feature/tool-fixes).
_COUNT_TOOL_SCOPE = re.compile(
    r"\b(?:programs?|programmes?|degrees?|bachelor'?s?|master'?s?|ph\.?d|doctoral|doctorate|"
    r"diplomas?|post\s*graduate)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# LLM-call budget
# ---------------------------------------------------------------------------
class LLMBudget:
    """Counts the LLM calls of ONE question and refuses calls beyond the limits.

    One instance per question, created in :func:`run`; there is no shared or
    module-level state, so concurrent questions cannot affect each other.
    A call is counted when it is issued, whether or not it succeeds: a failed
    call still went to the API.
    """

    def __init__(self, max_calls: int, retry_allowance: int) -> None:
        self.max_calls = max(0, int(max_calls))
        self.retry_allowance = max(0, min(int(retry_allowance), 1))   # never more than one retry
        self.normal = 0
        self.retries = 0
        self.issued: List[str] = []
        self.refused: List[str] = []

    @property
    def used(self) -> int:
        return self.normal + self.retries

    def try_spend(self, purpose: str, *, reserve: int = 0, is_retry: bool = False) -> bool:
        """Permit and count one call, or refuse it.

        ``reserve`` keeps that many normal calls free for a later step: the
        router's optional arbitration must not use up the call the answer needs.
        """
        if is_retry:
            allowed = self.retries < self.retry_allowance
        else:
            allowed = self.normal + 1 + max(0, reserve) <= self.max_calls
        if not allowed:
            self.refused.append(purpose)
            return False
        if is_retry:
            self.retries += 1
        else:
            self.normal += 1
        self.issued.append(purpose)
        return True


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _safe_error(exc: BaseException) -> str:
    """Exception class plus a short message with anything key-shaped removed."""
    text = re.sub(r"gsk_[A-Za-z0-9]+", "gsk_***", str(exc))
    text = re.sub(r"\s+", " ", text)[:120]
    return "%s: %s" % (type(exc).__name__, text) if text else type(exc).__name__


def _item_pages(items: List[Dict]) -> List[int]:
    return sorted({int(item["meta"]["page"]) for item in items})


# ---------------------------------------------------------------------------
# Step: route
# ---------------------------------------------------------------------------
def _route(question: str, groq_client, budget: LLMBudget,
           trace: List[TraceEntry]) -> RouteDecision:
    started = time.perf_counter()
    try:
        report = router_agent.route_with_report(
            question,
            groq_client=groq_client,
            # Arbitration is optional; keep one call free for the answer itself.
            llm_gate=lambda: budget.try_spend("router_fallback", reserve=1),
        )
    except Exception as exc:       # the Router is written never to raise; this is a seat belt
        decision = RouteDecision("policy", ["policy"], False, 0.0, False, {})
        trace.append(TraceEntry("router", "policy", False, _ms(started),
                                {"error": _safe_error(exc), "fallback": "policy"}))
        return decision

    decision = report.decision
    detail: Dict = {
        "scores": decision.scores,
        "routes": decision.routes,
        "confidence": decision.confidence,
        "margin": report.margin,
        "ambiguous": report.ambiguous,
        "is_multipart": decision.is_multipart,
        "mode": "llm" if decision.used_llm else "deterministic",
    }
    if report.cues:
        detail["cues"] = report.cues
    if report.ambiguous:
        detail["candidates"] = report.candidates
    if report.llm_skipped:
        detail["llm_skipped"] = report.llm_skipped
    if report.llm_error:
        detail["llm_error"] = report.llm_error
        detail["fallback"] = "deterministic"
    # used_llm on the trace entry means "an LLM call was issued by this step".
    trace.append(TraceEntry("router", decision.primary_route, report.llm_attempted, report.ms, detail))
    return decision


# ---------------------------------------------------------------------------
# Step: evidence (retrieval -> rerank -> gate -> extractor). Inline in Milestone 1;
# it moves to agents/evidence.py, unchanged in behaviour, in Milestone 2.
# ---------------------------------------------------------------------------
def _gather_evidence(
    question: str,
    route: str,
    *,
    embedder,
    reranker,
    index,
    chunks: List[str],
    metadata: List[Dict],
    pages: List[Dict],
    timings: Dict[str, float],
) -> Tuple[EvidenceResult, int, Optional[Tuple[List[int], str]]]:
    """Returns (evidence, number of candidates, extractor (pages, snippet) or None)."""
    # ---- Dense retrieval + lexical fusion ---------------------------------
    started = time.perf_counter()
    query_embedding = embedder.encode(
        [normalize_text(question)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    candidates = gather_candidates(question, route, query_embedding, index, chunks, metadata)
    timings["retrieval_ms"] = _ms(started)

    if not candidates:
        return (EvidenceResult(question, route, [], None, False, "no_candidates", False, None),
                0, None)

    # ---- Cross-encoder rerank ---------------------------------------------
    started = time.perf_counter()
    rerank_scores = reranker.predict([[question, cand["chunk"]] for cand in candidates])
    reranked: List[Dict] = []
    for cand, score in zip(candidates, rerank_scores):
        item = dict(cand)
        item["rerank_score"] = float(score)
        reranked.append(item)
    reranked.sort(
        key=lambda x: (
            x["rerank_score"],
            x.get("routing_boost", 0.0),
            x.get("lexical_score", 0.0),
            x.get("dense_score", 0.0),
        ),
        reverse=True,
    )
    final_items = deduplicate_by_text(reranked)[:FINAL_K]
    timings["rerank_ms"] = _ms(started)

    best_score = final_items[0]["rerank_score"] if final_items else None
    if not final_items or best_score < MIN_RERANK_SCORE:
        return (EvidenceResult(question, route, final_items, best_score, False,
                               "rerank_score_below_threshold", False, None),
                len(candidates), None)

    # ---- Deterministic extractor fast path (tools, zero LLM calls) ---------
    extracted = None
    tool_applicable = True
    if route == "contact":
        extracted = extract_contact_answer(question, final_items)
    elif route == "count":
        tool_applicable = bool(_COUNT_TOOL_SCOPE.search(question))
        if tool_applicable:
            extracted = extract_count_answer(question, final_items, pages)
    elif route == "date":
        extracted = extract_date_answer(question, final_items)

    if extracted:
        answer, source_pages, snippet = extracted
        return (EvidenceResult(question, route, final_items, best_score, True, None, True, answer),
                len(candidates), ([int(p) for p in source_pages], snippet))

    reason = None
    if route in EXTRACTOR_ROUTES:
        reason = "extractor_no_match" if tool_applicable else "extractor_not_applicable"
    return (EvidenceResult(question, route, final_items, best_score, True, reason, False, None),
            len(candidates), None)


# ---------------------------------------------------------------------------
# Step: verify
# ---------------------------------------------------------------------------
def _legacy_verify(question: str, synthesis: SynthesisResult, evidence: List[EvidenceResult],
                   sub_questions: List[SubQuestion], *, is_retry: bool = False) -> VerifyResult:
    """The Part 1 (MCBV9) check in the VerifyResult shape. See the module docstring."""
    ev = evidence[0]
    if ev.used_extractor:
        # Part 1 never checked extractor answers: they are built from handbook rows.
        return VerifyResult(True, list(synthesis.claimed_pages), [], [], None)
    pages = list(synthesis.claimed_pages) or legacy_fallback_pages(ev.items)
    accepted = bool(synthesis.answer) and verify_answer(synthesis.answer, ev.items, ev.route)
    # Part 1 kept the pages even when it replaced the answer with a refusal, and
    # it had no retry. Both are reproduced; neither is this module's to change.
    return VerifyResult(accepted, pages, [], [], None)


def _resolve_verifier(injected: Optional[VerifierFn]) -> Tuple[VerifierFn, str]:
    """Pick the verifier: injected (tests) > agents/verifier.py > Part 1 check."""
    if injected is not None:
        return injected, "injected"
    try:
        if importlib.util.find_spec("handbook_bot.agents.verifier") is None:
            return _legacy_verify, "legacy-mcbv9"
        module = importlib.import_module("handbook_bot.agents.verifier")
    except Exception as exc:       # a broken teammate module must not take the app down
        return _legacy_verify, "legacy-mcbv9 (agents.verifier failed to import: %s)" % _safe_error(exc)
    fn = getattr(module, "verify", None)
    if not callable(fn):
        return _legacy_verify, "legacy-mcbv9 (agents.verifier has no verify())"
    return fn, "agents.verifier"


def _verify(verifier: VerifierFn, impl: str, question: str, synthesis: SynthesisResult,
            evidence: EvidenceResult, allowed_pages: List[int], *, is_retry: bool,
            trace: List[TraceEntry]) -> VerifyResult:
    started = time.perf_counter()
    note: Dict = {"impl": impl, "is_retry": is_retry}
    try:
        result = verifier(question, synthesis, [evidence], [], is_retry=is_retry)
        if not isinstance(result, VerifyResult):
            raise TypeError("verify() returned %s, expected VerifyResult" % type(result).__name__)
    except Exception as exc:
        if verifier is _legacy_verify:
            raise
        # Fall back to the Part 1 check rather than fail open (unverified answer)
        # or fail closed (a bug in the verifier would look like a refusal).
        note["error"] = _safe_error(exc)
        note["impl"] = "legacy-mcbv9 (after error in %s)" % impl
        result = _legacy_verify(question, synthesis, [evidence], [], is_retry=is_retry)

    # Invariant: a page that was never retrieved can never be cited.
    clean = [int(p) for p in result.pages if isinstance(p, int) and p in allowed_pages]
    if len(clean) != len(result.pages):
        note["dropped_pages"] = [p for p in result.pages if p not in clean][:5]
        result.pages = clean

    if result.accepted:
        decision = "accepted"
    elif result.retry_feedback and not is_retry:
        decision = "retry_requested"
    else:
        decision = "rejected"
    note["pages"] = list(result.pages)
    if result.unsupported_claims:
        note["unsupported_claims"] = [str(c)[:80] for c in result.unsupported_claims[:3]]
    if result.uncovered_subquestions:
        note["uncovered_subquestions"] = list(result.uncovered_subquestions)
    trace.append(TraceEntry("verifier", decision, False, _ms(started), note))
    return result


# ---------------------------------------------------------------------------
# Step: synthesis (one LLM call, counted, never raises)
# ---------------------------------------------------------------------------
def _synthesize(question: str, evidence: EvidenceResult, groq_client, budget: LLMBudget,
                trace: List[TraceEntry], *, feedback: Optional[str]) -> Optional[SynthesisResult]:
    """One draft. Returns None when no draft could be produced; the reason is in the trace."""
    is_retry = feedback is not None
    label = "llm_retry" if is_retry else "llm"
    purpose = "synthesis_retry" if is_retry else "synthesis"
    started = time.perf_counter()
    if not budget.try_spend(purpose, is_retry=is_retry):
        trace.append(TraceEntry("synthesis", "skipped:llm_budget", False, _ms(started),
                                {"max_llm_calls": budget.max_calls, "issued": list(budget.issued)}))
        return None
    try:
        draft = synthesis_agent.synthesize(question, evidence.items, evidence.route,
                                           groq_client, feedback=feedback)
    except Exception as exc:
        trace.append(TraceEntry("synthesis", "llm_error", True, _ms(started),
                                {"error": _safe_error(exc)}))
        return None
    trace.append(TraceEntry("synthesis", label, True, _ms(started), {
        "path": "llm",
        "prompt_version": draft.prompt_version,
        "claimed_pages": list(draft.claimed_pages),
        "answer_chars": len(draft.answer),
        "refused": draft.answer.strip() == REFUSAL,
    }))
    return draft


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(
    question: str,
    *,
    embedder,
    reranker,
    index,
    chunks: List[str],
    metadata: List[Dict],
    pages: List[Dict],
    groq_client=None,
    verifier: Optional[VerifierFn] = None,
) -> QAResult:
    """Answer one question. ``verifier`` is for tests; production resolves it itself."""
    t_total = time.perf_counter()
    timings: Dict[str, float] = {}
    trace: List[TraceEntry] = []
    budget = LLMBudget(MAX_LLM_CALLS, MAX_VERIFY_RETRIES)
    question = (question or "").strip()

    def _finish(result: QAResult) -> QAResult:
        timings["total_ms"] = _ms(t_total)
        result.timings = timings
        result.agent_trace = trace
        result.llm_calls = budget.used
        return result

    # ---- 1. Route ----------------------------------------------------------
    started = time.perf_counter()
    decision = _route(question, groq_client, budget, trace)
    timings["router_ms"] = _ms(started)
    route = decision.primary_route

    if route == "greeting":
        return _finish(QAResult(answer=GREETING_REPLY, query_type=route, best_section="Greeting"))

    if PLANNER_ENABLED and decision.is_multipart:
        # Milestone 2 plugs the Planner in here. Until agents/planner.py exists
        # the question continues as a single question (the Option A path).
        if importlib.util.find_spec("handbook_bot.agents.planner") is None:
            trace.append(TraceEntry("planner", "unavailable", False, 0.0,
                                    {"reason": "agents/planner.py is Milestone 2; handled as one question"}))

    # ---- 2. Evidence ---------------------------------------------------------
    started = time.perf_counter()
    evidence, num_candidates, extractor_extra = _gather_evidence(
        question, route, embedder=embedder, reranker=reranker, index=index,
        chunks=chunks, metadata=metadata, pages=pages, timings=timings)
    items = evidence.items
    trace.append(TraceEntry(
        "evidence",
        ("extractor:" + route) if evidence.used_extractor
        else ("sufficient" if evidence.sufficient else "insufficient:" + str(evidence.reason)),
        False, _ms(started),
        {"candidates": num_candidates, "kept": len(items), "best_score": evidence.best_score,
         "gate": MIN_RERANK_SCORE, "pages": _item_pages(items), "reason": evidence.reason,
         "stage": "inline (orchestrator) - becomes agents/evidence.py in Milestone 2"}))

    if not evidence.sufficient:
        # A structured "insufficient" signal, turned into the controlled refusal. No LLM is called.
        section = "No matching evidence" if evidence.reason == "no_candidates" else "Low confidence retrieval"
        return _finish(QAResult(answer=REFUSAL, query_type=route, best_section=section, items=items,
                                num_candidates=num_candidates, num_reranked=len(items)))

    allowed_pages = sorted(set(_item_pages(items)) | set(extractor_extra[0] if extractor_extra else []))
    common = dict(query_type=route, items=items, num_candidates=num_candidates, num_reranked=len(items))

    # ---- 3. First draft: extractor (no LLM) or Synthesis ---------------------
    used_llm = False
    if evidence.used_extractor:
        ext_pages, snippet = extractor_extra
        draft: Optional[SynthesisResult] = synthesis_agent.from_extractor(evidence.extractor_answer, ext_pages)
        display_evidence = snippet
        timings["generation_ms"] = 0.0
        trace.append(TraceEntry("synthesis", "extractor", False, 0.0,
                                {"path": "extractor", "tool": route, "prompt_version": draft.prompt_version}))
    else:
        display_evidence = items[0]["chunk"]
        if groq_client is None:
            trace.append(TraceEntry("synthesis", "skipped:no_client", False, 0.0, {}))
            return _finish(QAResult(answer=NO_KEY_REPLY, pages=_item_pages(items[:3]),
                                    evidence=display_evidence, **common))
        started = time.perf_counter()
        draft = _synthesize(question, evidence, groq_client, budget, trace, feedback=None)
        timings["generation_ms"] = _ms(started)
        used_llm = draft is not None or trace[-1].decision == "llm_error"
        if draft is None:
            reply = REFUSAL if trace[-1].decision == "skipped:llm_budget" else LLM_UNAVAILABLE_REPLY
            return _finish(QAResult(answer=reply, evidence=display_evidence, used_llm=used_llm, **common))

    # ---- 4. Verify, with at most one retry -----------------------------------
    retried = False
    if not VERIFY_ANSWERS:
        trace.append(TraceEntry("verifier", "skipped", False, 0.0, {"reason": "VERIFY_ANSWERS is off"}))
        answer = draft.answer or REFUSAL
        cited = list(draft.claimed_pages) or legacy_fallback_pages(items)
    else:
        check, impl = _resolve_verifier(verifier)
        verdict = _verify(check, impl, question, draft, evidence, allowed_pages, is_retry=False, trace=trace)
        if (not verdict.accepted and verdict.retry_feedback and groq_client is not None
                and budget.retry_allowance > 0):
            t_retry = time.perf_counter()
            second = _synthesize(question, evidence, groq_client, budget, trace,
                                 feedback=verdict.retry_feedback)
            timings["generation_ms"] = timings.get("generation_ms", 0.0) + _ms(t_retry)
            if second is not None:
                retried = True
                used_llm = True
                draft = second
                display_evidence = items[0]["chunk"]
                verdict = _verify(check, impl, question, draft, evidence, allowed_pages,
                                  is_retry=True, trace=trace)
        answer = draft.answer if verdict.accepted else REFUSAL
        cited = list(verdict.pages)
    timings["verify_ms"] = sum(entry.ms for entry in trace if entry.agent == "verifier")

    # ---- 5. Result -------------------------------------------------------------
    if display_evidence and len(display_evidence) > EVIDENCE_PREVIEW_CHARS:
        display_evidence = display_evidence[:EVIDENCE_PREVIEW_CHARS].rstrip() + "..."
    best_section = items[0]["meta"].get("section") or f"Page {items[0]['meta']['page']}"
    return _finish(QAResult(answer=answer, pages=cited, best_section=best_section,
                            evidence=display_evidence, used_llm=used_llm, retried=retried, **common))
