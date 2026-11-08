"""
Interface strings of the chat page in English and Arabic (post-freeze PF-11).

Three languages are distinguished on the page and must not be confused:
    interface language   the labels, notices and first-use screen (``state["lang"]``, this module)
    question language    detected by the backend from the text the user typed (never chosen here)
    answer language      the user's preference, ``auto`` (the question's language), ``en`` or ``ar`` (``state["answer_lang"]``)

Defaults: the interface opens in the browser's preferred language when that is Arabic (``Accept-Language``), else in
English; ``?lang=ar`` or ``?lang=en`` overrides it; the answer language is ``auto``. Switching the interface language
re-renders the shell and the conversation from the page state; no message is lost.

Every key exists in both languages (pinned by tests). Backend sentences (refusal, service notices) are NOT here: the
backend owns them and ``handbook_bot.arabic_answers.REPLIES_AR`` holds their Arabic counterparts.
"""
from __future__ import annotations

from typing import Dict, List

LANGUAGES = ("en", "ar")
DIRECTION = {"en": "ltr", "ar": "rtl"}
ANSWER_LANGUAGES = ("auto", "en", "ar")
VOICE_LANGUAGES = ("auto", "en", "ar")

STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "org": "University of Sharjah",
        "app_name": "Faculty Onboarding Assistant",
        "app_subtitle": ("Answers for faculty members about university policies, procedures, services and teaching tools, "
                         "grounded in approved University of Sharjah documents and cited to the page they come from."),
        "about_lead": "About this assistant.",
        "about": ("Answers come from the UOS Faculty Handbook 2025/26 and approved official guides. "
                  "When the documents do not state something, the assistant says so instead of guessing."),
        "session_note": "Conversations are kept for this browser session only.",
        "new_chat": "New chat",
        "conversations": "Conversations",
        "conversations_aria": "Conversations in this session",
        "new_conversation": "New conversation",
        "skip": "Skip to the message box",
        "welcome": "Welcome to the {app}",
        "what_to_ask": "What you can ask",
        "status_setup": "Setup needed",
        "status_starting": "Starting up",
        "status_limited": "Limited service",
        "status_ready": "Ready",
        "status_starting_title": "The document index is loading. First start can take a minute.",
        "status_limited_title": "Contact, date and count lookups work; open questions need the language-model service, which is not configured.",
        "status_ready_title": "Documents indexed and the answer service configured.",
        "banner_setup": "Setup required.",
        "banner_starting": "Starting up.",
        "banner_starting_text": "The document index and language models are loading; the first start can take a minute.",
        "banner_limited": "Limited service.",
        "banner_limited_text": "Contact, date and count lookups work. Open questions need the language-model service, which is not configured on this installation.",
        "note_grounded": "Grounded in approved documents: the Faculty Handbook 2025/26 and official university guides.",
        "note_cited": "Every answer names its source and the printed page, so you can check it.",
        "note_arabic": "Ask in English or Arabic; answers follow the language of your question unless you choose otherwise.",
        "theme_to_dark": "Switch to dark mode",
        "theme_to_light": "Switch to light mode",
        "lang_switch": "العربية",
        "lang_switch_aria": "Switch the interface to Arabic",
        "developer": "Developer",
        "developer_aria": "Developer mode",
        "developer_details": "Developer details",
        "composer_placeholder": "Ask about a policy, procedure, office or teaching tool…",
        "composer_aria": "Message the Onboarding Chatbot",
        "send": "Send",
        "send_aria": "Send message",
        "hint_enter": "Enter",
        "hint_shift": "Shift",
        "hint_text": "to send · {shift}+{enter} for a new line",
        "help_empty": "Type a question to send.",
        "help_busy": "Please wait for the current answer to finish.",
        "help_loading": "The document index is still loading. Please try again in a moment.",
        "searching": "Searching official university sources…",
        "answer_ready": "Answer ready.",
        "your_message": "Your message",
        "assistant_answer": "Assistant answer",
        "you": "YOU",
        "sources": "Sources",
        "verified_by": "Verified · {domains}",
        "verified_title": "Verified from the approved documents by the {domains} specialist",
        "page": "Page {page}",
        "ocr": "image text (OCR)",
        "source_document": "Source document",
        "evidence": "Supporting evidence ({count})",
        "copy": "Copy answer",
        "copied": "Copied",
        "show_original": "Show the verified English source answer",
        "translation_checked": "Translated from the verified English answer; numbers, contacts and conditions checked",
        "translation_framed": "Source text shown in English (translation not available or not verified)",
        "answer_language": "Answer language",
        "answer_auto": "Same as the question",
        "answer_en": "English",
        "answer_ar": "Arabic",
        "notice_empty": "No supporting evidence found",
        "notice_config": "Language-model service not configured",
        "notice_outage": "Language-model service unavailable",
        "notice_config_problem": "Language-model configuration problem",
        "notice_limit": "Answer limit reached",
        "notice_error": "Something went wrong",
        "hint_empty": "The approved documents do not state this, or the question needs rewording. Try naming the policy, procedure, office or date you mean.",
        "hint_outage": "The answer service did not respond this time. Please send the question again in a moment.",
        "hint_config": "Deterministic lookups (contacts, dates, counts) still work. An administrator needs to complete the model configuration for open questions.",
        "hint_limit": "Please ask a narrower question.",
        "hint_error": "Please try again. If it keeps happening, tell the administrator what you asked.",
        "error_text": "Sorry, something went wrong while answering that. Please try again.",
        "logo_alt": "University of Sharjah logo",
        "voice_start": "Ask by voice",
        "voice_start_aria": "Ask by voice ({voice})",
        "voice_stop": "Stop listening",
        "voice_cancel": "Cancel voice input",
        "voice_language": "Voice language",
        "voice_auto": "Interface language",
        "voice_en": "English",
        "voice_ar": "Arabic",
        "voice_starting": "Starting the microphone…",
        "voice_listening": "Listening… speak your question, then press Stop or pause.",
        "voice_ready": "Transcript added to the message box. Check it, edit if needed, then press Send.",
        "voice_cancelled": "Voice input cancelled.",
        "voice_unsupported": "Voice input is not available in this browser. You can type your question.",
        "voice_err_permission": "Microphone access was not allowed. Allow the microphone in the browser's site settings, or type your question.",
        "voice_err_no_mic": "No microphone was found. Connect one or type your question.",
        "voice_err_no_speech": "No speech was heard. Press the microphone and try again, or type your question.",
        "voice_err_network": "The speech service could not be reached. Check the connection and try again, or type your question.",
        "voice_err_timeout": "Nothing was recognised within the time limit. Try again, or type your question.",
        "voice_err_language": "This language is not available in the browser's speech service. Choose the other voice language or type your question.",
        "voice_err_unrecognised": "The speech could not be recognised. Try again, or type your question.",
        "voice_busy": "The microphone is already listening.",
        "voice_privacy": "Speech is recognised by your browser's speech service; the audio may be processed on that provider's servers. The page keeps only the text you see in the message box.",
        "voice_listening_label": "Listening",
    },
    "ar": {
        "org": "جامعة الشارقة",
        "app_name": "مساعد تهيئة أعضاء هيئة التدريس",
        "app_subtitle": ("إجابات لأعضاء هيئة التدريس عن سياسات الجامعة وإجراءاتها وخدماتها وأدوات التدريس، "
                         "مستندة إلى وثائق جامعة الشارقة المعتمدة ومُوثَّقة بالصفحة التي وردت فيها."),
        "about_lead": "عن هذا المساعد.",
        "about": ("تأتي الإجابات من دليل أعضاء هيئة التدريس 2025/2026 والأدلة الرسمية المعتمدة. "
                  "وعندما لا تنص الوثائق على أمر ما، يقول المساعد ذلك بدلًا من التخمين."),
        "session_note": "تُحفظ المحادثات لجلسة المتصفح الحالية فقط.",
        "new_chat": "محادثة جديدة",
        "conversations": "المحادثات",
        "conversations_aria": "محادثات هذه الجلسة",
        "new_conversation": "محادثة جديدة",
        "skip": "الانتقال إلى مربع الرسالة",
        "welcome": "مرحبًا بك في {app}",
        "what_to_ask": "ما الذي يمكنك السؤال عنه",
        "status_setup": "يلزم الإعداد",
        "status_starting": "جارٍ التشغيل",
        "status_limited": "خدمة محدودة",
        "status_ready": "جاهز",
        "status_starting_title": "جارٍ تحميل فهرس الوثائق. قد يستغرق التشغيل الأول دقيقة.",
        "status_limited_title": "تعمل عمليات البحث عن جهات الاتصال والتواريخ والأعداد؛ أما الأسئلة المفتوحة فتحتاج إلى خدمة النموذج اللغوي غير المضبوطة حاليًا.",
        "status_ready_title": "تمت فهرسة الوثائق وضبط خدمة الإجابة.",
        "banner_setup": "يلزم الإعداد.",
        "banner_starting": "جارٍ التشغيل.",
        "banner_starting_text": "جارٍ تحميل فهرس الوثائق والنماذج اللغوية؛ قد يستغرق التشغيل الأول دقيقة.",
        "banner_limited": "خدمة محدودة.",
        "banner_limited_text": "تعمل عمليات البحث عن جهات الاتصال والتواريخ والأعداد. أما الأسئلة المفتوحة فتحتاج إلى خدمة النموذج اللغوي غير المضبوطة في هذا التثبيت.",
        "note_grounded": "مستند إلى الوثائق المعتمدة: دليل أعضاء هيئة التدريس 2025/2026 والأدلة الجامعية الرسمية.",
        "note_cited": "كل إجابة تذكر مصدرها ورقم الصفحة المطبوعة لتتمكن من التحقق منها.",
        "note_arabic": "اسأل بالعربية أو بالإنجليزية؛ تتبع الإجابة لغة سؤالك ما لم تختر غير ذلك.",
        "theme_to_dark": "التبديل إلى الوضع الداكن",
        "theme_to_light": "التبديل إلى الوضع الفاتح",
        "lang_switch": "English",
        "lang_switch_aria": "تبديل الواجهة إلى الإنجليزية",
        "developer": "المطوّر",
        "developer_aria": "وضع المطوّر",
        "developer_details": "تفاصيل المطوّر",
        "composer_placeholder": "اسأل عن سياسة أو إجراء أو مكتب أو أداة تدريس…",
        "composer_aria": "مراسلة مساعد التهيئة",
        "send": "إرسال",
        "send_aria": "إرسال الرسالة",
        "hint_enter": "Enter",
        "hint_shift": "Shift",
        "hint_text": "للإرسال · {shift}+{enter} لسطر جديد",
        "help_empty": "اكتب سؤالًا للإرسال.",
        "help_busy": "يرجى انتظار اكتمال الإجابة الحالية.",
        "help_loading": "ما زال فهرس الوثائق قيد التحميل. يرجى المحاولة بعد قليل.",
        "searching": "جارٍ البحث في مصادر الجامعة الرسمية…",
        "answer_ready": "الإجابة جاهزة.",
        "your_message": "رسالتك",
        "assistant_answer": "إجابة المساعد",
        "you": "أنت",
        "sources": "المصادر",
        "verified_by": "مُتحقَّق منه · {domains}",
        "verified_title": "تم التحقق من الوثائق المعتمدة بواسطة مختص {domains}",
        "page": "صفحة {page}",
        "ocr": "نص من صورة (OCR)",
        "source_document": "وثيقة المصدر",
        "evidence": "الأدلة الداعمة ({count})",
        "copy": "نسخ الإجابة",
        "copied": "تم النسخ",
        "show_original": "عرض الإجابة الإنجليزية المتحقَّق منها",
        "translation_checked": "مُترجمة من الإجابة الإنجليزية المتحقَّق منها؛ تم التحقق من الأرقام وجهات الاتصال والشروط",
        "translation_framed": "النص المصدر معروض بالإنجليزية (الترجمة غير متاحة أو لم يتم التحقق منها)",
        "answer_language": "لغة الإجابة",
        "answer_auto": "بلغة السؤال",
        "answer_en": "الإنجليزية",
        "answer_ar": "العربية",
        "notice_empty": "لا توجد أدلة داعمة",
        "notice_config": "خدمة النموذج اللغوي غير مضبوطة",
        "notice_outage": "خدمة النموذج اللغوي غير متاحة",
        "notice_config_problem": "مشكلة في ضبط النموذج اللغوي",
        "notice_limit": "تم بلوغ حد الإجابة",
        "notice_error": "حدث خطأ ما",
        "hint_empty": "لا تنص الوثائق المعتمدة على ذلك، أو يحتاج السؤال إلى إعادة صياغة. جرّب ذكر السياسة أو الإجراء أو المكتب أو التاريخ الذي تقصده.",
        "hint_outage": "لم تستجب خدمة الإجابة هذه المرة. يرجى إرسال السؤال مرة أخرى بعد قليل.",
        "hint_config": "ما زالت عمليات البحث المحددة (جهات الاتصال والتواريخ والأعداد) تعمل. يحتاج مسؤول النظام إلى إكمال ضبط النموذج للأسئلة المفتوحة.",
        "hint_limit": "يرجى طرح سؤال أضيق.",
        "hint_error": "يرجى المحاولة مرة أخرى. وإذا تكرر الأمر، أخبر مسؤول النظام بما سألت عنه.",
        "error_text": "عذرًا، حدث خطأ أثناء الإجابة عن ذلك. يرجى المحاولة مرة أخرى.",
        "logo_alt": "شعار جامعة الشارقة",
        "voice_start": "اسأل بالصوت",
        "voice_start_aria": "اسأل بالصوت ({voice})",
        "voice_stop": "إيقاف الاستماع",
        "voice_cancel": "إلغاء الإدخال الصوتي",
        "voice_language": "لغة الصوت",
        "voice_auto": "لغة الواجهة",
        "voice_en": "الإنجليزية",
        "voice_ar": "العربية",
        "voice_starting": "جارٍ تشغيل الميكروفون…",
        "voice_listening": "جارٍ الاستماع… انطق سؤالك ثم اضغط إيقاف أو توقّف عن الكلام.",
        "voice_ready": "أُضيف النص المنطوق إلى مربع الرسالة. راجعه وعدّله إن لزم، ثم اضغط إرسال.",
        "voice_cancelled": "تم إلغاء الإدخال الصوتي.",
        "voice_unsupported": "الإدخال الصوتي غير متاح في هذا المتصفح. يمكنك كتابة سؤالك.",
        "voice_err_permission": "لم يُسمح بالوصول إلى الميكروفون. اسمح بالميكروفون من إعدادات الموقع في المتصفح، أو اكتب سؤالك.",
        "voice_err_no_mic": "لم يُعثر على ميكروفون. صِل ميكروفونًا أو اكتب سؤالك.",
        "voice_err_no_speech": "لم يُسمع أي كلام. اضغط الميكروفون وحاول مرة أخرى، أو اكتب سؤالك.",
        "voice_err_network": "تعذّر الوصول إلى خدمة التعرف على الكلام. تحقق من الاتصال وحاول مرة أخرى، أو اكتب سؤالك.",
        "voice_err_timeout": "لم يُتعرّف على شيء ضمن المهلة. حاول مرة أخرى، أو اكتب سؤالك.",
        "voice_err_language": "هذه اللغة غير متاحة في خدمة التعرف على الكلام في المتصفح. اختر لغة الصوت الأخرى أو اكتب سؤالك.",
        "voice_err_unrecognised": "تعذّر التعرف على الكلام. حاول مرة أخرى، أو اكتب سؤالك.",
        "voice_busy": "الميكروفون يستمع بالفعل.",
        "voice_privacy": "يتم التعرف على الكلام بواسطة خدمة الكلام في متصفحك؛ وقد تُعالج التسجيلات على خوادم مزوّد تلك الخدمة. لا تحتفظ الصفحة إلا بالنص الذي تراه في مربع الرسالة.",
        "voice_listening_label": "جارٍ الاستماع",
    },
}

#: Specialist display names.
DOMAIN_NAMES: Dict[str, Dict[str, str]] = {
    "en": {"teaching": "Teaching & Learning", "research": "Research & Innovation", "faculty_services": "Faculty Services & HR", "institutional": "Institutional Navigation"},
    "ar": {"teaching": "التعليم والتعلم", "research": "البحث والابتكار", "faculty_services": "خدمات أعضاء هيئة التدريس والموارد البشرية", "institutional": "التوجيه المؤسسي"},
}

#: The first-use screen in Arabic. Every Arabic example is checked (qa/pf11_arabic/check_examples.py) to reach the same
#: deterministic answer path as its English counterpart, so the shortcuts work without a model key.
TOPICS_AR: List[Dict[str, object]] = [
    {"id": "teaching", "icon": "teaching", "title": "التعليم والتعلم",
     "blurb": "الأعباء التدريسية، ومنصة Blackboard وأدوات القاعة الدراسية، وقواعد التقييم.",
     "examples": ["ما هو العبء التدريسي للمحاضرين؟", "كيف أرسل إعلانًا للطلبة عبر Blackboard؟"]},
    {"id": "research", "icon": "research", "title": "البحث والابتكار",
     "blurb": "دعم البحث، وسفر المؤتمرات، والإشراف، وقواعد النشر.",
     "examples": ["من يوافق على طلبات السفر لحضور المؤتمرات؟"]},
    {"id": "faculty_services", "icon": "services", "title": "خدمات أعضاء هيئة التدريس والموارد البشرية",
     "blurb": "الإجازات، والمزايا، والتعيينات، وإجراءات الموارد البشرية.",
     "examples": ["كم يوماً من إجازة الأمومة تستحقها عضوات هيئة التدريس؟"]},
    {"id": "institutional", "icon": "institutional", "title": "التوجيه المؤسسي",
     "blurb": "المكاتب، وأرقام الاتصال، والحوكمة، وخدمات الحرم الجامعي.",
     "examples": ["ما هو رقم هاتف قسم الموارد البشرية؟", "ما هو رقم فاكس وحدة الجوازات؟"]},
]


def t(lang: str, key: str, **fmt: object) -> str:
    """The string ``key`` in ``lang`` (English when the key is missing in that language), formatted."""
    table = STRINGS.get(lang) or STRINGS["en"]
    text = table.get(key)
    if text is None:
        text = STRINGS["en"].get(key, key)
    return text.format(**fmt) if fmt else text


def domain_name(lang: str, domain_id: str) -> str:
    return (DOMAIN_NAMES.get(lang) or DOMAIN_NAMES["en"]).get(domain_id, (DOMAIN_NAMES["en"].get(domain_id, domain_id)))


def normalize_lang(value: object) -> str:
    code = str(value or "").strip().lower()[:2]
    return code if code in LANGUAGES else "en"


def lang_from_request(query_lang: object, accept_language: object) -> str:
    """The initial interface language: ``?lang=`` wins; else the first language of ``Accept-Language`` when it is Arabic."""
    code = str(query_lang or "").strip().lower()[:2]
    if code in LANGUAGES:
        return code
    first = str(accept_language or "").split(",")[0].strip().lower()
    return "ar" if first.startswith("ar") else "en"


__all__ = ["ANSWER_LANGUAGES", "VOICE_LANGUAGES", "DIRECTION", "DOMAIN_NAMES", "LANGUAGES", "STRINGS", "TOPICS_AR", "domain_name", "lang_from_request", "normalize_lang", "t"]
