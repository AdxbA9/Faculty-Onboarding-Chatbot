"""
Arabic answers (post-freeze PF-11): fixed Arabic replies, the bilingual grounded frame, model translation and the
deterministic translation-faithfulness check.

Architecture (see docs/DECISIONS.md, D-122): the chatbot answers an Arabic question by running its unchanged English
pipeline on the transformed query, so every English guard (extractors, specialist verification, the legacy Verifier,
the conflict guard, citation selection) keeps its full effect on an ENGLISH intermediate answer. The Arabic text the
user sees is produced from that verified intermediate in one of three ways:

    fixed      the refusal, greeting and service notices are fixed Arabic sentences (``REPLIES_AR``); never translated
    translated ONE model request translates the verified English answer; the result is accepted only when
               ``check_translation`` finds every number, telephone, e-mail, URL, acronym and list item preserved, the
               negation not dropped, the length within bounds and the text actually Arabic
    framed     otherwise (no client, translation disabled, check failed) the answer is an Arabic lead-in naming the
               source and page followed by the verified English text verbatim (``frame_answer``)

Verifying the English intermediate does NOT prove the Arabic translation; the check below is what stands between the
two, and it is deterministic and limited (a meaning change that keeps every number, name and negation marker passes
it). The status is reported with the answer (``translation.status``), never hidden.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .arabic_text import detect_language, has_arabic, western_digits
from .orchestrator import GREETING_REPLY, LLM_BUDGET_REPLY, LLM_UNAVAILABLE_REPLY, MODEL_UNAVAILABLE_REPLY, NO_KEY_REPLY
from .qa import REFUSAL

#: The page's own safe-error sentence (ui.presentation.ERROR_TEXT) is localised by the page; the backend ones are here.
REPLIES_AR: Dict[str, str] = {
    REFUSAL: "لا تتوفر لديّ هذه المعلومة في الوثائق المعتمدة.",
    GREETING_REPLY: "مرحبًا! اسألني عن دليل أعضاء هيئة التدريس في جامعة الشارقة.",
    NO_KEY_REPLY: "لم يتم ضبط مفتاح خدمة النموذج اللغوي (GROQ_API_KEY). يرجى ضبطه ثم إعادة تحميل التطبيق.",
    LLM_UNAVAILABLE_REPLY: "تعذّر توليد الإجابة الآن لأن خدمة النموذج اللغوي لم تستجب. يرجى المحاولة مرة أخرى.",
    MODEL_UNAVAILABLE_REPLY: "تعذّر توليد الإجابة لأن النموذج اللغوي المضبوط غير متاح لهذا الحساب. يرجى الطلب من مسؤول النظام ضبط GROQ_MODEL في ملف .env على نموذج متاح ثم إعادة تشغيل التطبيق.",
    LLM_BUDGET_REPLY: "تعذّرت الإجابة عن هذا السؤال ضمن حد طلبات النموذج اللغوي المضبوط (MAX_LLM_CALLS).",
}

#: Lead-in of a framed answer. {source} is the document title as the knowledge base recorded it, {pages} the printed pages.
FRAME_LEAD = "وفقًا للوثيقة المعتمدة «{source}» ({pages})، ينص المصدر على ما يلي:"
FRAME_LEAD_NO_PAGES = "وفقًا للوثيقة المعتمدة «{source}»، ينص المصدر على ما يلي:"
FRAME_NOTE = "النص معروض بلغته الأصلية (الإنجليزية) كما ورد في المصدر المعتمد؛ لم تُعدَّل أرقامه ولا شروطه."
DEFAULT_SOURCE_AR = "الوثائق المعتمدة"
PAGE_WORD_AR = "صفحة"
PAGES_WORD_AR = "الصفحات"

#: Document titles in Arabic for the registered sources (the page shows the recorded English title next to them).
SOURCE_TITLES_AR: Dict[str, str] = {
    "uos_faculty_handbook_2025_26": "دليل أعضاء هيئة التدريس 2025/2026",
    "uos_blackboard_faculty_short_guide": "الدليل المختصر لأعضاء هيئة التدريس في Blackboard Ultra",
    "uos_grade_transfer_dashboard_guide": "دليل لوحة نقل الدرجات",
    "uos_blackboard_gradebook_guide": "دليل Blackboard Ultra: سجل الدرجات",
    "uos_ithenticate_guidelines": "إرشادات iThenticate",
    "uos_blackboard_attendance_guide": "أداة الحضور في Blackboard Ultra",
    "uos_blackboard_collaborate_guide": "دليل Blackboard Collaborate Ultra",
    "uos_blackboard_communication_guide": "Blackboard Learn Ultra: أدوات التواصل",
}

TRANSLATION_SYSTEM = (
    "You are a professional translator for a university in the United Arab Emirates. Translate the given English text "
    "into clear, formal Modern Standard Arabic for faculty members. Rules: translate faithfully and completely; do not "
    "add, omit, summarise or explain anything; keep every number exactly as written using Western digits (0-9); keep "
    "telephone numbers, e-mail addresses, web addresses, currency codes such as AED, acronyms such as VCAA, HR, IRB, "
    "WLAM, PhD, and system names such as Blackboard exactly as written in Latin letters; keep every negation, condition, "
    "exception and approval requirement; keep the line structure (one line per list item or step); do not translate the "
    "word 'Pages' or add citations. Reply with the Arabic translation only."
)

_KEY_SHAPED = re.compile(r"gsk_[A-Za-z0-9]+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_PHONE = re.compile(r"\+?\d[\d\-\s()/]{6,}\d")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL = re.compile(r"https?://\S+|www\.\S+")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9&]{1,}(?:/[A-Z]+)?\b")
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[a-z][.)])\s+", re.M)
_NEGATION_EN = re.compile(r"\b(?:not|no|never|cannot|can't|don't|doesn't|isn't|aren't|won't|without|neither|nor|none|unable|prohibited|forbidden|"
                          r"not allowed|non-refundable|unless|except)\b", re.I)
_NEGATION_AR = re.compile(r"(?:^|[\s(«])(?:لا|لم|لن|ليس|ليست|ليسوا|غير|عدم|بدون|دون|ممنوع|محظور|يحظر|يُحظر|لا يجوز|لا يُسمح|لا يسمح|باستثناء|ما لم|إلا|الا|مستثنى|كلا)(?=[\s،؛,;:.!؟)»]|$)")   # a marker may be followed by punctuation ("لا؛ يحق ...")
_PAGES_LINE = re.compile(r"^\s*pages?\s*:.*$", re.I | re.M)
_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")


def localize_reply(answer: str) -> Optional[str]:
    """The fixed Arabic sentence for one of the backend's own replies, or None when ``answer`` is a real answer."""
    text = (answer or "").strip()
    for english, arabic in REPLIES_AR.items():
        if text == english or text.rstrip(".") == english.rstrip("."):
            return arabic
    return None


def is_fixed_reply(answer: str) -> bool:
    return localize_reply(answer) is not None


# ---------------------------------------------------------------------------
# Framed (bilingual) answer: Arabic lead-in + verified English source text
# ---------------------------------------------------------------------------
def _source_title(items: Sequence[Dict[str, Any]]) -> str:
    for item in items or []:
        meta = item.get("meta") if isinstance(item, dict) and isinstance(item.get("meta"), dict) else {}
        sid = meta.get("source_id")
        if isinstance(sid, str) and sid in SOURCE_TITLES_AR:
            return SOURCE_TITLES_AR[sid]
        title = meta.get("source_title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return DEFAULT_SOURCE_AR


def _pages_phrase(pages: Sequence[int]) -> str:
    pages = [int(p) for p in pages if isinstance(p, int) and not isinstance(p, bool)]
    if not pages:
        return ""
    if len(pages) == 1:
        return "%s %d" % (PAGE_WORD_AR, pages[0])
    return "%s %s" % (PAGES_WORD_AR, " و".join(str(p) for p in pages))


def frame_answer(english: str, items: Sequence[Dict[str, Any]], pages: Sequence[int]) -> str:
    """The Arabic lead-in plus the verified English text as a Markdown quotation. Nothing of ``english`` is changed."""
    pages_phrase = _pages_phrase(pages)
    lead = (FRAME_LEAD.format(source=_source_title(items), pages=pages_phrase) if pages_phrase
            else FRAME_LEAD_NO_PAGES.format(source=_source_title(items)))
    quoted = "\n".join("> " + line if line.strip() else ">" for line in (english or "").strip().splitlines())
    return "%s\n\n%s\n\n%s" % (lead, quoted, FRAME_NOTE)


# ---------------------------------------------------------------------------
# Model translation (one request; the caller counts it)
# ---------------------------------------------------------------------------
def _redact(exc: BaseException) -> str:
    text = _KEY_SHAPED.sub("gsk_***", str(exc))
    return ("%s: %s" % (type(exc).__name__, re.sub(r"\s+", " ", text)[:120])).strip(": ")


def translate_answer(client, english: str) -> str:
    """Translate ``english`` with ONE chat completion. Raises on transport failure or an empty reply."""
    from .config import GROQ_MODEL, LLM_MAX_TOKENS, LLM_REASONING_TOKENS
    caller = client.with_options(max_retries=0, timeout=40.0) if hasattr(client, "with_options") else client
    resp = caller.chat.completions.create(
        model=GROQ_MODEL, temperature=0.0, max_tokens=2 * LLM_MAX_TOKENS + LLM_REASONING_TOKENS,
        messages=[{"role": "system", "content": TRANSLATION_SYSTEM}, {"role": "user", "content": (english or "").strip()[:6000]}])
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("empty translation")
    return text


# ---------------------------------------------------------------------------
# Deterministic translation-faithfulness check
# ---------------------------------------------------------------------------
@dataclass
class TranslationCheck:
    ok: bool
    problems: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "problems": list(self.problems), "detail": dict(self.detail)}


def _numbers(text: str) -> List[str]:
    text = _PHONE.sub(" ", _EMAIL.sub(" ", _URL.sub(" ", text)))
    return sorted(n.replace(",", "") for n in _NUMBER.findall(text))


def _structured(text: str) -> Dict[str, List[str]]:
    return {"phones": sorted(re.sub(r"\s+", " ", p).strip() for p in _PHONE.findall(text)),
            "emails": sorted(_EMAIL.findall(text)), "urls": sorted(u.rstrip(".,)") for u in _URL.findall(text)),
            "acronyms": sorted(set(a for a in _ACRONYM.findall(_URL.sub(" ", _EMAIL.sub(" ", text))) if a not in ("I", "A"))),
            "list_items": [m.group(0) for m in _LIST_LINE.finditer(text)]}


def check_translation(english: str, arabic: str) -> TranslationCheck:
    """Does ``arabic`` preserve what can be checked without understanding either text? Every finding is a reason code."""
    problems: List[str] = []
    english = (english or "").strip()
    arabic_raw = (arabic or "").strip()
    arabic = western_digits(arabic_raw)
    detail: Dict[str, Any] = {}
    if not arabic:
        return TranslationCheck(False, ["empty"], detail)
    # the script share is measured on the prose: contacts, addresses and Latin tokens carried over from the English
    # source (acronyms, system names) are what a faithful translation MUST keep, so they do not count against it
    prose = _URL.sub(" ", _EMAIL.sub(" ", _PHONE.sub(" ", arabic)))
    prose = " ".join(w for w in prose.split() if not (_LATIN_WORD.search(w) and w.strip(".,;:()") in english))
    profile = detect_language(prose)
    detail["arabic_share"] = round(profile.arabic_share, 3)
    if not has_arabic(arabic) or profile.arabic_share < 0.5:
        problems.append("not_arabic")
    if _PAGES_LINE.search(arabic):
        problems.append("pages_line_added")
    en_numbers, ar_numbers = _numbers(english), _numbers(arabic)
    detail["numbers_english"], detail["numbers_arabic"] = en_numbers, ar_numbers
    if en_numbers != ar_numbers:
        problems.append("numbers_differ")
    s_en, s_ar = _structured(english), _structured(arabic)
    for key in ("phones", "emails", "urls"):
        if s_en[key] != s_ar[key]:
            problems.append("%s_differ" % key)
    missing_acronyms = [a for a in s_en["acronyms"] if a not in s_ar["acronyms"]]
    detail["acronyms_missing"] = missing_acronyms
    if missing_acronyms:
        problems.append("acronyms_missing")
    if len(s_en["list_items"]) != len(s_ar["list_items"]):
        problems.append("list_structure_differs")
    detail["list_items"] = [len(s_en["list_items"]), len(s_ar["list_items"])]
    en_neg, ar_neg = len(_NEGATION_EN.findall(english)), len(_NEGATION_AR.findall(arabic))
    detail["negation"] = [en_neg, ar_neg]
    if en_neg and not ar_neg:
        problems.append("negation_dropped")
    letters_en = len(re.findall(r"[A-Za-z]", english))
    letters_ar = profile.arabic_letters + profile.latin_letters
    ratio = (letters_ar / letters_en) if letters_en else 1.0
    detail["length_ratio"] = round(ratio, 2)
    if letters_en >= 40 and not (0.45 <= ratio <= 2.2):
        problems.append("length_out_of_bounds")
    latin_left = [w for w in _LATIN_WORD.findall(arabic) if w not in s_ar["acronyms"] and w not in english]
    detail["latin_words_not_in_source"] = latin_left[:5]
    if len(latin_left) > 3:
        problems.append("foreign_words_added")
    if localize_reply(arabic_raw) is None and arabic_raw.strip() == REPLIES_AR[REFUSAL]:
        problems.append("became_refusal")
    return TranslationCheck(not problems, problems, detail)


__all__ = ["DEFAULT_SOURCE_AR", "FRAME_LEAD", "FRAME_NOTE", "REPLIES_AR", "SOURCE_TITLES_AR", "TRANSLATION_SYSTEM", "TranslationCheck",
           "check_translation", "frame_answer", "is_fixed_reply", "localize_reply", "translate_answer"]
