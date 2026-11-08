"""
Bilingual answering (post-freeze PF-11): the Arabic layer around the unchanged English pipeline.

    question --detect--> Arabic? --transform (arabic_query)--> English query --core pipeline (unchanged)--> English result
                                                                                  |
                                             answer language Arabic? --localise (arabic_answers)--> Arabic result

``answer`` is called by the approved seam (``ui.pipeline.ask_with_diagnostics``) and receives the seam's own core as a
callable, so LEGACY / SHADOW / ACTIVE, the Plan F adapter, the findings verification, the legacy Verifier and the
fallback all run exactly as for an English question. This module decides only (1) which language the answer should be
in, (2) which English query the core sees, and (3) how the verified English answer is shown in Arabic.

Answer-language semantics (``answer_language``):
    "auto"  the language of the question: Arabic for an Arabic or Arabic-framed mixed question, else English
    "ar"    Arabic, whatever the question's language (an English question is answered in English and then localised)
    "en"    English, whatever the question's language (the Arabic question is still transformed; the English answer stands)

Model requests added by this layer, both bounded and both recorded: at most ONE query translation (only when the
lexicon understood too little of the question, see ``arabic_query``) and at most ONE answer translation (only for a
real answer, never for a refusal or a service notice). Both are counted in ``QAResult.llm_calls``.
The diagnostics gain a ``language`` record of codes and counts only; the question and the answer text never enter it.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Tuple

from . import arabic_answers
from .arabic_query import transform_query
from .arabic_text import detect_language, has_arabic
from .qa import QAResult

CoreFn = Callable[[str], Tuple[QAResult, Dict[str, Any]]]
ANSWER_LANGUAGES = ("auto", "en", "ar")
_KEY_SHAPED = re.compile(r"gsk_[A-Za-z0-9]+")


def resolve_answer_language(preference: Optional[str], question: str) -> str:
    """``"ar"`` or ``"en"`` for the given preference and question (see the module docstring)."""
    pref = (preference or "auto").strip().lower()
    if pref in ("ar", "en"):
        return pref
    return "ar" if detect_language(question or "").is_arabic else "en"


def _redact(exc: BaseException) -> str:
    text = _KEY_SHAPED.sub("gsk_***", str(exc))
    return ("%s: %s" % (type(exc).__name__, re.sub(r"\s+", " ", text)[:120])).strip(": ")


def localize(result: QAResult, client, *, allow_translation: Optional[bool] = None) -> Dict[str, Any]:
    """Turn the verified English ``result`` into its Arabic presentation IN PLACE and return the translation record.

    A fixed reply (refusal, greeting, service notice) keeps its canonical English sentence in ``answer`` so every reader
    of that sentence (the page's notice labels, the evaluations) keeps working; ``answer_language`` says the page must
    show its Arabic counterpart (``arabic_answers.localize_reply``). A real answer is translated by one model request
    and accepted only when ``check_translation`` passes; otherwise it is framed (Arabic lead-in + verified English)."""
    from . import config
    allow = config.ARABIC_ANSWER_TRANSLATION if allow_translation is None else bool(allow_translation)
    english = result.answer or ""
    result.answer_language = "ar"
    if arabic_answers.localize_reply(english) is not None or not english.strip():
        result.translation = {"status": "fixed", "llm_calls": 0, "problems": []}
        return result.translation
    record: Dict[str, Any] = {"status": "framed", "llm_calls": 0, "problems": [], "reason": None}
    if client is None:
        record["reason"] = "no_client"
    elif not allow:
        record["reason"] = "disabled"
    else:
        record["llm_calls"] = 1
        try:
            arabic = arabic_answers.translate_answer(client, english)
            check = arabic_answers.check_translation(english, arabic)
            record["check"] = check.to_dict()
            if check.ok:
                result.original_answer = english
                result.answer = arabic
                record["status"] = "translated"
                result.translation = record
                result.llm_calls = int(result.llm_calls or 0) + 1
                return record
            record["status"] = "framed_after_check"
            record["problems"] = list(check.problems)
        except Exception as exc:
            record["status"] = "framed_after_error"
            record["error"] = _redact(exc)
        result.llm_calls = int(result.llm_calls or 0) + 1
    result.original_answer = english
    result.answer = arabic_answers.frame_answer(english, result.items or [], result.pages or [])
    result.translation = record
    return record


def answer(question: str, *, core: CoreFn, client=None, answer_language: Optional[str] = "auto",
           allow_query_model: Optional[bool] = None, allow_translation: Optional[bool] = None) -> Tuple[QAResult, Dict[str, Any]]:
    """Answer ``question`` through ``core`` with the Arabic layer applied where it applies. Never raises for the layer's
    own reasons: a failure in transformation or localisation leaves the English result standing."""
    text = (question or "").strip()
    profile = detect_language(text)
    target = resolve_answer_language(answer_language, text)
    language: Dict[str, Any] = {"detected": profile.code, "primary": profile.primary, "answer_language": target, "query_method": "passthrough",
                                "query_coverage": None, "query_llm_calls": 0, "query_unmapped": 0, "translation_status": None,
                                "translation_llm_calls": 0, "translation_problems": []}
    transform = None
    core_question = text
    if has_arabic(text):
        try:
            from . import config
            allow_q = config.ARABIC_QUERY_MODEL_FALLBACK if allow_query_model is None else bool(allow_query_model)
            transform = transform_query(text, client=client if allow_q else None, allow_model=allow_q)
            core_question = transform.english_query if transform.english_query.strip() else text
            language.update({"query_method": transform.method, "query_coverage": round(transform.coverage, 3),
                             "query_llm_calls": transform.llm_calls, "query_unmapped": len(transform.unmapped)})
            if transform.model_error:
                language["query_model_error"] = transform.model_error
        except Exception as exc:                                   # the untransformed question goes to the core
            language["query_error"] = _redact(exc)
            transform = None
    result, diagnostics = core(core_question)
    if transform is not None:
        result.query_transform = transform.to_dict()
        result.llm_calls = int(result.llm_calls or 0) + int(transform.llm_calls)
    if target == "ar":
        try:
            record = localize(result, client, allow_translation=allow_translation)
            language["translation_status"] = record.get("status")
            language["translation_llm_calls"] = int(record.get("llm_calls") or 0)
            language["translation_problems"] = list(record.get("problems") or [])
            if record.get("error"):
                language["translation_error"] = record["error"]
        except Exception as exc:                                   # the English answer stands
            result.answer_language = "en"
            language["translation_status"] = "error"
            language["translation_error"] = _redact(exc)
    else:
        result.answer_language = "en"
        language["translation_status"] = "not_requested"
    diagnostics = dict(diagnostics or {})
    diagnostics["language"] = language
    return result, diagnostics


__all__ = ["ANSWER_LANGUAGES", "answer", "localize", "resolve_answer_language"]
