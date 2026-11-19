"""
Conversational layer: small talk and friendly "not found" wording, in English and Arabic.

Two jobs, both deliberately free of any fact about the university:

1. ``detect(text)`` recognises a message that is ONLY small talk (hi, thanks, bye, who are you, what can you do, ok...)
   and ``reply(intent, lang)`` answers it from a pool of varied replies, so the bot never shows the "no evidence"
   notice for "thanks!" and never repeats the same sentence twice in a row. A message that carries a real question
   ("hi, how many days of sick leave?") is NOT small talk and goes to the normal pipeline.
2. ``friendly_refusal(lang, seed)`` gives a rotating, human sentence + tip to show instead of the fixed backend refusal
   sentence. The backend's decision is unchanged (it still refused); only the wording on the page varies.

Matching is typo-tolerant: tokens are compared with a small edit-distance check (``thnaks``, ``helo``, ``goodbey``,
``شكراا``) and repeated letters are squeezed (``heyyyy``, ``thanksss``). Everything here is pure Python.
"""
from __future__ import annotations

import difflib
import itertools
import random
import re
import threading
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

try:                                                     # the project's Arabic normaliser (alef/yaa/taa folding, diacritics)
    from .arabic_text import normalize_arabic
except Exception:                                        # pragma: no cover - keeps the module importable on its own
    def normalize_arabic(text: str) -> str:
        return text

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
#: intent -> phrases (each phrase is matched token by token, typo-tolerant). Longer phrases are tried first.
PHRASES: Dict[str, Sequence[str]] = {
    "greeting": (
        "hi", "hello", "hey", "hiya", "heya", "howdy", "yo", "salam", "salaam", "hola", "greetings", "hi there", "hello there",
        "hey there", "good morning", "good afternoon", "good evening", "morning", "evening", "assalamu alaikum",
        "assalamualaikum", "as salam alaikum", "salam alaikum", "marhaba",
        "مرحبا", "مرحبتين", "اهلا", "اهلين", "اهلا وسهلا", "اهلا بك", "اهلا بكم", "السلام عليكم", "سلام عليكم",
        "السلام عليكم ورحمه الله", "السلام عليكم ورحمه الله وبركاته", "سلام", "صباح الخير", "مساء الخير", "صباح النور",
        "مساء النور", "هاي", "هلا", "هلا والله", "يا هلا", "هلو", "تحيه طيبه", "تحياتي",
    ),
    "how_are_you": (
        "how are you", "how r u", "how are u", "how r you", "hows it going", "how is it going", "how are things",
        "how are you doing", "how you doing", "whats up", "sup", "wassup", "what is up",
        "كيف حالك", "كيف الحال", "كيفك", "شلونك", "شخبارك", "اخبارك", "ايش اخبارك", "كيف حالكم", "عامل ايه",
    ),
    "thanks": (
        "thanks", "thank", "thank you", "thankyou", "thanx", "thnx", "thx", "tnx", "ty", "tysm", "thanks a lot",
        "thank you so much", "thanks so much", "many thanks", "much appreciated", "appreciate it", "i appreciate it",
        "appreciated", "cheers", "thats helpful", "that was helpful", "very helpful", "helpful",
        "شكرا", "شكرا لك", "شكرا جزيلا", "الف شكر", "مشكور", "مشكوره", "يعطيك العافيه", "تسلم", "تسلمي", "تسلم ايدك",
        "جزاك الله خير", "جزاك الله خيرا", "بارك الله فيك", "ممنون", "شكرا كثير",
    ),
    "bye": (
        "bye", "goodbye", "good bye", "bye bye", "see you", "see ya", "cya", "see you later", "later", "take care",
        "good night", "have a nice day", "have a good day", "thats all", "that is all", "im done", "i am done",
        "مع السلامه", "الى اللقاء", "باي", "في امان الله", "تصبح على خير", "وداعا", "الله معك", "يلا باي",
    ),
    "identity": (
        "who are you", "what are you", "who r u", "what is this", "what is this bot", "are you a bot", "are you human",
        "are you real", "whats your name", "what is your name", "your name", "introduce yourself",
        "من انت", "مين انت", "ما اسمك", "شو اسمك", "ايش اسمك", "من تكون", "عرف بنفسك", "هل انت روبوت", "هل انت انسان",
    ),
    "help": (
        "help", "help me", "i need help", "need help", "can you help", "can you help me", "please help", "help please",
        "what can you do", "what do you do", "how can you help", "how can you help me", "what can i ask",
        "what can i ask you", "what should i ask", "what do you know", "how does this work", "how do i use this",
        "how to use", "options", "menu", "start", "get started", "i am new", "im new", "new here", "i am a new faculty",
        "where do i start", "where should i start",
        "مساعده", "ساعدني", "ابي مساعده", "ابغى مساعده", "احتاج مساعده", "محتاج مساعده", "ممكن تساعدني", "كيف تساعدني",
        "ماذا تستطيع", "ماذا يمكنك", "ماذا تفعل", "شو تقدر تسوي", "ايش تقدر تسوي", "ماذا اسال", "كيف استخدمك", "من وين ابدا",
        "انا جديد", "انا جديده", "عضو جديد",
    ),
    "ack": (
        "ok", "okay", "okey", "k", "kk", "cool", "great", "nice", "perfect", "awesome", "good", "very good",
        "got it", "alright", "all right", "sure", "fine", "noted", "understood", "i see", "makes sense", "wow", "amazing",
        "excellent", "lol", "haha",
        "تمام", "حلو", "ممتاز", "طيب", "اوكي", "اوك", "زين", "فهمت", "واضح", "رائع", "جميل", "ماشي",
    ),
}

#: Words that may accompany small talk without changing it ("thanks a lot bot", "hi there assistant").
FILLERS = frozenset({
    "please", "pls", "plz", "so", "very", "really", "much", "a", "lot", "the", "dear", "sir", "madam", "bot", "chatbot",
    "assistant", "again", "guys", "man", "bro", "buddy", "friend", "there", "and", "oh", "ah", "um", "uh", "well", "just",
    "for", "it", "that", "this", "you", "u", "ya", "for the help", "all", "everyone", "again", "dr", "doctor", "prof",
    "يا", "و", "جدا", "كثير", "مره", "ثانيه", "اخي", "اختي", "دكتور", "استاذ", "حبيبي", "لك", "عليك",
})

_INTENT_PRIORITY = ("help", "identity", "how_are_you", "thanks", "bye", "greeting", "ack")
_MAX_TOKENS = 9

_TOKEN = re.compile(r"[A-Za-z]+|[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]+|\d+")
_REPEAT = re.compile(r"(.)\1{2,}")
_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def _norm_token(tok: str) -> str:
    tok = _REPEAT.sub(r"\1\1", tok.lower())               # heyyyy -> heyy, شكراااا -> شكراا
    if _ARABIC.search(tok):
        tok = normalize_arabic(tok)
        tok = tok.replace("ة", "ه").replace("ى", "ي")
        tok = re.sub(r"ا{2,}$", "ا", tok)                  # شكراا -> شكرا
    return tok


def tokens(text: str) -> List[str]:
    """Lower-cased, typo-squeezed word tokens of ``text`` (Latin, Arabic and digits; punctuation and emoji dropped)."""
    text = unicodedata.normalize("NFKC", text or "").replace("'", "").replace("’", "")
    return [_norm_token(t) for t in _TOKEN.findall(text)]


def _close(a: str, b: str) -> bool:
    """Typo-tolerant token equality: exact for short words (bar stretched letters), ~one edit for longer ones."""
    if a == b:
        return True
    if len(b) <= 3 or len(a) <= 3:
        # stretched short words only: heyy ~ hey, hii ~ hi, okk ~ ok, thxx ~ thx
        return bool(re.fullmatch(re.escape(b) + re.escape(b[-1]) + "*", a))
    if a[0] != b[0] or abs(len(a) - len(b)) > 2:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= (0.8 if len(b) <= 5 else 0.75)


_PHRASE_TOKENS: List[Tuple[Tuple[str, ...], str]] = sorted(
    ((tuple(tokens(p)), intent) for intent, ps in PHRASES.items() for p in ps if tokens(p)),
    key=lambda pi: len(pi[0]), reverse=True)
_FILLER_TOKENS = frozenset(_norm_token(f) for f in FILLERS if " " not in f)


def detect(text: str) -> Optional[str]:
    """The small-talk intent of ``text`` (greeting, how_are_you, thanks, bye, identity, help, ack) or None.

    None whenever the message carries anything that is not small talk, so real questions are never intercepted."""
    toks = tokens(text)
    if not toks or len(toks) > _MAX_TOKENS:
        return None
    found: List[str] = []
    i = 0
    while i < len(toks):
        for phrase, intent in _PHRASE_TOKENS:
            n = len(phrase)
            if i + n <= len(toks) and all(_close(toks[i + k], phrase[k]) for k in range(n)):
                found.append(intent)
                i += n
                break
        else:
            if toks[i] in _FILLER_TOKENS:
                i += 1
                continue
            return None
    if not found:
        return None
    for intent in _INTENT_PRIORITY:
        if intent in found:
            return intent
    return None


def is_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text or ""))


# ---------------------------------------------------------------------------
# Replies (no university facts: only what the assistant itself can do)
# ---------------------------------------------------------------------------
REPLIES: Dict[str, Dict[str, Sequence[str]]] = {
    "en": {
        "greeting": (
            "Hello! I'm the UOS Faculty Onboarding Assistant. What would you like to know today?",
            "Hi there! Ask me anything about faculty policies, leave, teaching, Blackboard or university contacts.",
            "Hey! Good to see you. How can I help with your onboarding today?",
            "Hello and welcome! I can look things up in the Faculty Handbook and the official university guides for you.",
            "Hi! Ready when you are. Try something like *\"How many days of sick leave do I get?\"*",
        ),
        "how_are_you": (
            "I'm doing great, thanks for asking! What can I help you with?",
            "All good here and ready to help. What's on your mind?",
            "Doing well, thank you! Is there a policy or procedure you'd like me to check?",
            "I'm fine, thanks! Ask me anything about your onboarding at UOS.",
        ),
        "thanks": (
            "You're welcome! Let me know if there's anything else.",
            "Happy to help! Feel free to ask another question anytime.",
            "Anytime! Good luck with your onboarding.",
            "Glad that was useful. Anything else you'd like to check?",
            "My pleasure! I'm here whenever you need me.",
            "No problem at all. Ask away if something else comes up.",
        ),
        "bye": (
            "Goodbye! Your conversation is saved, so you can come back to it anytime.",
            "Take care! Come back whenever you have another question.",
            "See you soon, and good luck with the semester!",
            "Bye for now! Your onboarding checklist will be waiting for you.",
        ),
        "identity": (
            "I'm the **UOS Faculty Onboarding Assistant**, a chatbot that answers faculty questions using only the approved "
            "University of Sharjah documents: the Faculty Handbook 2025/26 and the official guides. Every answer shows "
            "its source and page so you can check it.",
            "I'm a virtual assistant for new faculty at the University of Sharjah. I search the Faculty Handbook and the "
            "official university guides and answer with the page I found it on.",
        ),
        "help": (
            "Sure! Here's what I can help with:\n\n"
            "- **Policies & HR:** leave, contracts, probation, benefits\n"
            "- **Teaching:** teaching loads, office hours, assessment rules\n"
            "- **Blackboard & tools:** attendance, gradebook, announcements, iThenticate\n"
            "- **Contacts:** phone numbers and offices from the handbook\n\n"
            "You can also use the **Onboarding checklist** and **Guided procedures** in the menu. "
            "Try asking: *\"What is the teaching load for a lecturer?\"*",
            "Of course. Ask me in plain words, in English or Arabic, for example:\n\n"
            "- *How many days of maternity leave do female faculty get?*\n"
            "- *How do I take attendance in Blackboard?*\n"
            "- *Who approves conference travel requests?*\n"
            "- *What is the phone number of Human Resources?*\n\n"
            "I'll answer from the approved documents and show you the page.",
            "Happy to help! I answer questions about faculty policies, procedures, teaching tools and university contacts. "
            "If you're new, the **Onboarding checklist** in the menu is a great place to start.",
        ),
        "ack": (
            "Great! Anything else I can look up for you?",
            "Perfect. Just ask if you need anything else.",
            "Got it! I'm here if another question comes up.",
            "Sounds good. What would you like to check next?",
        ),
    },
    "ar": {
        "greeting": (
            "مرحبًا! أنا مساعد تهيئة أعضاء هيئة التدريس في جامعة الشارقة. بماذا أستطيع مساعدتك اليوم؟",
            "أهلًا وسهلًا! اسألني عن السياسات والإجازات والتدريس وبلاك بورد وجهات الاتصال في الجامعة.",
            "مرحبًا بك! كيف يمكنني مساعدتك في التهيئة اليوم؟",
            "أهلًا بك! أبحث لك في دليل أعضاء هيئة التدريس والأدلة الرسمية للجامعة.",
            "أهلًا! جرّب أن تسأل مثلًا: *كم يوم إجازة مرضية يحق لي؟*",
        ),
        "how_are_you": (
            "أنا بخير، شكرًا لسؤالك! كيف أستطيع مساعدتك؟",
            "الحمد لله، وجاهز لمساعدتك. ما الذي تود معرفته؟",
            "بخير، شكرًا! هل هناك سياسة أو إجراء تريد أن أتحقق منه؟",
        ),
        "thanks": (
            "العفو! يسعدني مساعدتك في أي وقت.",
            "على الرحب والسعة! اسألني إن احتجت أي شيء آخر.",
            "لا شكر على واجب، بالتوفيق في التهيئة!",
            "يسعدني أن الإجابة كانت مفيدة. هل هناك شيء آخر؟",
            "في الخدمة دائمًا!",
        ),
        "bye": (
            "مع السلامة! محادثتك محفوظة ويمكنك العودة إليها في أي وقت.",
            "في أمان الله! عد متى ما كان لديك سؤال آخر.",
            "إلى اللقاء، وبالتوفيق في الفصل الدراسي!",
        ),
        "identity": (
            "أنا **مساعد تهيئة أعضاء هيئة التدريس** في جامعة الشارقة، أجيب عن أسئلتك من الوثائق المعتمدة فقط: "
            "دليل أعضاء هيئة التدريس 2025/26 والأدلة الرسمية للجامعة، مع ذكر المصدر ورقم الصفحة.",
            "أنا مساعد افتراضي لأعضاء هيئة التدريس الجدد في جامعة الشارقة، أبحث في الدليل والأدلة الرسمية وأذكر لك الصفحة.",
        ),
        "help": (
            "بالتأكيد! هذا ما أستطيع مساعدتك فيه:\n\n"
            "- **السياسات والموارد البشرية:** الإجازات والعقود وفترة التجربة والمزايا\n"
            "- **التدريس:** العبء التدريسي والساعات المكتبية وقواعد التقييم\n"
            "- **بلاك بورد والأدوات:** الحضور وسجل الدرجات والإعلانات\n"
            "- **جهات الاتصال:** أرقام الهواتف والمكاتب من الدليل\n\n"
            "ويمكنك أيضًا استخدام **قائمة التهيئة** و**الإجراءات الموجّهة** من القائمة.",
            "طبعًا. اسألني بكلماتك بالعربية أو الإنجليزية، مثلًا:\n\n"
            "- *كم يوم إجازة أمومة تستحق عضوة هيئة التدريس؟*\n"
            "- *كيف أسجل الحضور في بلاك بورد؟*\n"
            "- *ما رقم هاتف الموارد البشرية؟*",
        ),
        "ack": (
            "رائع! هل هناك شيء آخر أبحث عنه لك؟",
            "تمام. اسألني متى احتجت.",
            "ممتاز! أنا هنا إن كان لديك سؤال آخر.",
        ),
    },
}

#: Shown instead of the fixed refusal sentence. Never states a fact, never says something does not exist.
REFUSALS: Dict[str, Sequence[str]] = {
    "en": (
        "I couldn't find that in the approved university documents I have access to.",
        "Hmm, that one doesn't seem to be covered in the Faculty Handbook or the official guides.",
        "I looked through the handbook and the guides, but nothing there answers this, so I'd rather not guess.",
        "I don't have a reliable source for that in the approved documents.",
        "That isn't something I can find in the documents I'm allowed to use.",
        "Sorry, I couldn't find a clear answer to that in the handbook or the university guides.",
    ),
    "ar": (
        "لم أجد ذلك في الوثائق الجامعية المعتمدة المتاحة لي.",
        "يبدو أن هذا غير مذكور في دليل أعضاء هيئة التدريس أو الأدلة الرسمية.",
        "بحثت في الدليل والأدلة الرسمية ولم أجد ما يجيب عن هذا، ولا أريد أن أخمّن.",
        "ليس لدي مصدر موثوق لهذا في الوثائق المعتمدة.",
        "عذرًا، لم أجد إجابة واضحة عن ذلك في الدليل أو أدلة الجامعة.",
    ),
}
TIPS: Dict[str, Sequence[str]] = {
    "en": (
        "Try rephrasing, or name the policy, office or tool you mean, for example \"sick leave\" or \"Blackboard gradebook\".",
        "Shorter questions with one main topic usually work best.",
        "You could ask about a specific policy, procedure, office or date and I'll check again.",
        "For things outside the handbook, your department or Human Resources is the best place to ask.",
        "Tip: the Resources & documents page lists everything I can search.",
    ),
    "ar": (
        "جرّب إعادة الصياغة، أو اذكر السياسة أو المكتب أو الأداة التي تقصدها، مثل \"الإجازة المرضية\" أو \"سجل الدرجات في بلاك بورد\".",
        "الأسئلة القصيرة ذات الموضوع الواحد تعطي أفضل النتائج عادةً.",
        "اسأل عن سياسة أو إجراء أو مكتب أو تاريخ محدد وسأبحث مجددًا.",
        "للأمور غير المذكورة في الدليل، قسمك الأكاديمي أو الموارد البشرية هم الجهة الأنسب.",
        "نصيحة: صفحة الموارد والمستندات تعرض كل ما أستطيع البحث فيه.",
    ),
}

# ---------------------------------------------------------------------------
# Rotation: each pool loops from a random start, so replies vary and never repeat back to back.
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_CYCLES: Dict[Tuple[str, ...], "itertools.count[int]"] = {}


def _next_index(key: Tuple[str, ...], size: int) -> int:
    with _LOCK:
        counter = _CYCLES.get(key)
        if counter is None:
            counter = _CYCLES[key] = itertools.count(random.randrange(max(size, 1)))
        return next(counter) % max(size, 1)


def _lang(lang: Optional[str]) -> str:
    return "ar" if lang == "ar" else "en"


def reply(intent: str, lang: Optional[str] = "en", seed: Optional[int] = None) -> str:
    """A reply for a small-talk ``intent``. ``seed`` picks a fixed variant (tests, re-renders); else the pool rotates."""
    pool = REPLIES[_lang(lang)].get(intent) or REPLIES[_lang(lang)]["greeting"]
    i = (seed % len(pool)) if seed is not None else _next_index(("reply", _lang(lang), intent), len(pool))
    return pool[i]


def friendly_refusal(lang: Optional[str] = "en", seed: Optional[int] = None) -> Tuple[str, str]:
    """``(sentence, tip)`` to show for a refusal. Same ``seed`` -> same wording, so a re-rendered bubble does not change."""
    lg = _lang(lang)
    texts, tips = REFUSALS[lg], TIPS[lg]
    if seed is None:
        seed = _next_index(("refusal", lg), len(texts) * len(tips))
    return texts[seed % len(texts)], tips[(seed // len(texts) + seed) % len(tips)]


ALL_REPLIES = frozenset(r for lang in REPLIES.values() for pool in lang.values() for r in pool)

__all__ = ["ALL_REPLIES", "PHRASES", "REFUSALS", "REPLIES", "TIPS", "detect", "friendly_refusal", "is_arabic", "reply", "tokens"]
