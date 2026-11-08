"""
Deterministic answer extractors.

For a handful of common, high-value question types we prefer to build the
answer directly from the retrieved evidence instead of asking the LLM:

* **contact** – phone / fax / email for a named office
* **count**   – "how many <x> programs does UoS offer?"
* **date**    – academic-calendar style "when do classes begin?"

When these extractors fire we get 100%-grounded answers with zero LLM cost
and no hallucination risk. When they can't find a high-confidence match we
fall back to the LLM path.

An extractor answer skips the model and the Verifier's prose check, so a loose
match would read as a certain fact. A value is therefore stated only when the
printed line binds it (D-120): a contact number needs a complete directory row
for exactly the asked office, and a date needs a row stating the asked event,
a date and any term the question names. Anything less declines, and the model
and the Verifier handle the question.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .retrieval import classify_query
from .text_utils import (
    CONTACT_QUESTION_PATTERN,
    COUNT_PATTERN,
    DAY_PATTERN,
    EMAIL_PATTERN,
    MONTH_PATTERN,
    normalize_line,
    normalize_text,
    tokenize,
)


# Type alias: (answer_text, list_of_page_numbers, evidence_snippet)
ExtractorResult = Tuple[str, List[int], str]
# Returns the printed rows of the page an item comes from, in printed order.
PageLines = Callable[[Dict], List[str]]


# ---------------------------------------------------------------------------
# Contact extractor
# ---------------------------------------------------------------------------
def _office_query_target(question: str) -> str:
    """Strip the question down to the office name being asked about.

    e.g. "What is the phone number for the Information Technology Center?"
         -> "information technology center"
    """
    q = question.strip().rstrip("?")
    m = re.search(r"(?:for|of)\s+(.+)$", q, re.I)
    target = m.group(1).strip() if m else q
    target = re.sub(
        r"\b(phone|telephone|fax|contact|number|email|what|is|the)\b",
        " ",
        target,
        flags=re.I,
    )
    return normalize_text(target)


_FULL_NUMBER = r"\+\d{3} \d \d{3} ?\d{4}"
#: A directory row that holds a name, then exactly two complete numbers, then nothing else.
_COMPLETE_ROW = re.compile(rf"^(?P<office>\S.*?)\s+(?P<telephone>{_FULL_NUMBER})\s+(?P<fax>{_FULL_NUMBER})$")
_COLUMN_HEADER = re.compile(r"\btelephone\b.*\bfax\b", re.I)
#: A line that carries on into the next one: a trailing "/" or an unfinished number.
_RUNS_ON = re.compile(r"(?:/|\+\d{0,3}(?: \d{0,3})?)$")
#: A line that continues the previous one: a broken number, or a Men / Women sub-row.
_CONTINUES = re.compile(r"^(?:\d|(?:men|women)(?:['’]s)?\b)", re.I)
#: A word a question adds after an office name that the directory may not print.
_GENERIC_TAIL = re.compile(r"\s+(?:department|dept|office|unit|section)$")


def _office_key(name: str) -> str:
    """An office name for comparison: case, apostrophe style and outer punctuation do not matter."""
    return normalize_text(name).replace("’", "'").strip(" .,:;").lower()


def page_row_lines(chunks: Sequence[str], metadata: Sequence[Dict], item: Dict) -> List[str]:
    """The printed rows of the document page ``item`` comes from, in printed order."""
    page, source = item["meta"].get("page"), item["meta"].get("source_id")
    return [
        normalize_line(chunk) for chunk, meta in zip(chunks, metadata)
        if meta.get("chunk_type") == "row" and meta.get("page") == page and meta.get("source_id") == source
    ]


def _rows_naming(office: str, lines: Sequence[str]) -> List[int]:
    """Indexes of the lines that start with ``office`` and hold a digit."""
    return [i for i, line in enumerate(lines)
            if (_office_key(line) + " ").startswith(office + " ") and re.search(r"\d", line)]


def _bind_directory_row(office: str, lines: Sequence[str]) -> Optional[Tuple[str, str, str]]:
    """(telephone, fax, printed row) of ``office``, or None unless every link of the binding holds.

    The same links the Institutional Specialist requires (D-104), written here because the legacy path imports no
    Plan F module: the page's header fixes the columns, one row starts with exactly the asked office, that row holds
    two complete numbers and nothing else, and neither neighbouring line runs into it. Lines are never joined. A
    trailing generic word is dropped from the asked name only when the page prints no row under the full name."""
    headers = [line for line in lines if re.search(r"\bfax\b", line, re.I) and re.search(r"\btelephone\b", line, re.I)]
    if not headers or not all(_COLUMN_HEADER.search(line) for line in headers):
        return None
    named = _rows_naming(office, lines)
    if not named:
        office = _GENERIC_TAIL.sub("", office)
        named = _rows_naming(office, lines)
    if len(named) != 1:
        return None
    at = named[0]
    row = _COMPLETE_ROW.match(lines[at])
    if not row or _office_key(row.group("office")) != office:
        return None
    previous = lines[at - 1] if at else ""
    following = lines[at + 1] if at + 1 < len(lines) else ""
    if _RUNS_ON.search(previous) or _CONTINUES.match(previous) or _CONTINUES.match(following):
        return None
    return normalize_text(row.group("telephone")), normalize_text(row.group("fax")), lines[at]


def _email_answer(target: str, items: List[Dict]) -> Optional[ExtractorResult]:
    """An address from a row that names every word of the asked target."""
    wanted = set(tokenize(target))
    if not wanted:
        return None
    for item in items:
        if item["meta"].get("chunk_type") not in {"row", "row_window"}:
            continue
        for line in item["chunk"].split(" | "):
            emails = EMAIL_PATTERN.findall(line)
            if emails and wanted <= set(tokenize(line)):
                return f"The email address is {emails[0]}.", [item["meta"]["page"]], line
    return None


def extract_contact_answer(
    question: str, items: List[Dict], page_lines: Optional[PageLines] = None
) -> Optional[ExtractorResult]:
    """Return (answer, pages, evidence) for a contact question or None.

    ``page_lines`` gives the printed rows of an item's page. A number needs its neighbours and the column header to
    be bound, so without it no number is stated."""
    if not CONTACT_QUESTION_PATTERN.search(question):
        return None

    target = _office_query_target(question)
    ql = question.lower()
    want_fax = "fax" in ql
    want_phone = "phone" in ql
    want_email = "email" in ql

    if want_email:
        return _email_answer(target, items)
    if page_lines is None:
        return None

    office = _office_key(target)
    seen = set()
    for item in items:
        if item["meta"].get("chunk_type") not in {"row", "row_window"}:
            continue
        key = (item["meta"].get("source_id"), item["meta"].get("page"))
        if key in seen:
            continue
        seen.add(key)
        bound = _bind_directory_row(office, page_lines(item))
        if not bound:
            continue
        telephone, fax, line = bound
        pages = [item["meta"]["page"]]
        if want_fax and want_phone:
            return f"The phone number is {telephone}. The fax number is {fax}.", pages, line
        if want_fax:
            return f"The fax number is {fax}.", pages, line
        return f"The phone number is {telephone}.", pages, line
    return None


# ---------------------------------------------------------------------------
# Count extractor
# ---------------------------------------------------------------------------
# Each key maps to a regex pulled from the handbook's program-count section.
_COUNT_PATTERNS = {
    "total": (
        r"(?:total of|total number of|offers? a total of|offer a total of|"
        r"we offer a total of)\s+(\d{1,4}(?:,\d{3})?)\s+"
        r"(?:academic\s+)?(?:degree\s+)?programs"
    ),
    "bachelor": r"(\d{1,4}(?:,\d{3})?)\s+Bachelor(?:'s)?\s+(?:degrees?|programs?)",
    "master": r"(\d{1,4}(?:,\d{3})?)\s+Master(?:'s)?\s+(?:degrees?|programs?)",
    "phd": r"(\d{1,4}(?:,\d{3})?)\s+PhD\s+(?:degrees?|programs?)",
    "postgraduate": (
        r"(\d{1,4}(?:,\d{3})?)\s+(?:Post\s*Graduate|Postgraduate|Professional Diploma)"
        r"(?:\s+and\s+Professional\s+Diploma)?\s+(?:degrees?|programs?)"
    ),
}


def _find_counts(text: str) -> Dict[str, int]:
    found: Dict[str, int] = {}
    for label, pattern in _COUNT_PATTERNS.items():
        m = re.search(pattern, text, re.I)
        if m:
            found[label] = int(m.group(1).replace(",", ""))
    return found


def extract_count_answer(
    question: str, items: List[Dict], pages: List[Dict]
) -> Optional[ExtractorResult]:
    """Return a deterministic count answer (degrees/programs) or None."""
    if not COUNT_PATTERN.search(question):
        return None

    # Look at the top retrieved chunks *plus* the first ~45 pages of prose –
    # program counts tend to live in the "About UoS" section.
    combined = " \n ".join(
        [x["chunk"] for x in items] +
        [p["text"] for p in pages[: min(45, len(pages))]]
    )
    found = _find_counts(combined)
    if not found:
        return None

    q = question.lower()

    def _has(*keys: str) -> bool:
        return all(k in found for k in keys)

    if (all(k in q for k in ("bachelor", "master", "phd"))
            or "post" in q or "diploma" in q):
        parts = []
        if "bachelor" in found:
            parts.append(f"{found['bachelor']} Bachelor's")
        if "master" in found:
            parts.append(f"{found['master']} Master's")
        if "phd" in found:
            parts.append(f"{found['phd']} PhD")
        if "postgraduate" in found:
            parts.append(f"{found['postgraduate']} postgraduate/professional diploma")
        answer = "UoS offers " + ", ".join(parts) + " degree programs."
    elif "bachelor" in q and "bachelor" in found:
        answer = f"UoS offers {found['bachelor']} Bachelor's degree programs."
    elif "master" in q and "master" in found:
        answer = f"UoS offers {found['master']} Master's degree programs."
    elif "phd" in q and "phd" in found:
        answer = f"UoS offers {found['phd']} PhD degree programs."
    elif ("post" in q or "diploma" in q) and "postgraduate" in found:
        answer = (
            f"UoS offers {found['postgraduate']} postgraduate/professional "
            "diploma programs."
        )
    elif "total" in q and "total" in found:
        answer = f"UoS offers {found['total']} total degree programs."
    else:
        parts = []
        if "total" in found:
            parts.append(f"{found['total']} total degree programs")
        if "bachelor" in found:
            parts.append(f"{found['bachelor']} Bachelor's")
        if "master" in found:
            parts.append(f"{found['master']} Master's")
        if "phd" in found:
            parts.append(f"{found['phd']} PhD")
        if "postgraduate" in found:
            parts.append(f"{found['postgraduate']} postgraduate/professional diploma")
        answer = "UoS offers " + ", ".join(parts) + "."

    pages_used = sorted({item["meta"]["page"] for item in items[:3]})
    evidence = next(
        (
            item["chunk"]
            for item in items
            if re.search(r"\b149\b|Bachelor|Master|PhD|Diploma", item["chunk"], re.I)
        ),
        items[0]["chunk"],
    )
    return answer, pages_used, evidence


# ---------------------------------------------------------------------------
# Date / calendar extractor
# ---------------------------------------------------------------------------
_DATE_LABELS = {
    "classes begin":        ("classes begin",),
    "last day for add/drop": ("add/drop", "last day for add/drop"),
    "final exams":          ("final exam",),
    "classes end":          ("classes end",),
}


#: A term or year named in a question. Rows of different terms read alike and most do not print their term, so a row is
#: bound to the asked term only when it prints it. Relative wording is never printed, so it always goes to the model.
_ASKED_TERM = re.compile(
    r"\b(?:fall|autumn|spring|summer|20\d{2}"
    r"|(?:first|second|next|this|current|coming|upcoming|last|previous)\s+(?:semester|term|year))\b",
    re.I,
)


def _resolve_target_label(q: str) -> Optional[str]:
    for label, keywords in _DATE_LABELS.items():
        if any(k in q for k in keywords):
            return label
    return None


def _states_event_and_date(text: str, label: str) -> bool:
    """True when a lowercased row names the asked event and prints a day of a month."""
    names_event = any(re.search(re.escape(k) + r"s?\b", text) for k in _DATE_LABELS[label])
    return names_event and bool(MONTH_PATTERN.search(text)) and bool(re.search(r"\b\d{1,2}\b", text))


def extract_date_answer(
    question: str, items: List[Dict]
) -> Optional[ExtractorResult]:
    """Return a calendar-date answer built from the best matching row."""
    if classify_query(question) != "date":
        return None

    target_label = _resolve_target_label(question.lower())
    if not target_label:
        return None

    asked_terms = [term.lower() for term in _ASKED_TERM.findall(question)]

    # Expand row_window chunks back into their constituent rows so we can
    # score each row individually.
    rows: List[Dict] = []
    for item in items:
        ctype = item["meta"].get("chunk_type")
        if ctype == "row":
            rows.append(item)
        elif ctype == "row_window":
            for part in item["chunk"].split(" | "):
                rows.append({"chunk": normalize_line(part), "meta": item["meta"]})

    best: Optional[Dict] = None
    best_score = -1.0
    for row in rows:
        text = row["chunk"].lower()
        if not _states_event_and_date(text, target_label) or not all(term in text for term in asked_terms):
            continue
        score = 0.0
        if target_label in text:
            score += 3.0
        for tok in tokenize(question):
            if tok in text:
                score += 0.35
        if (MONTH_PATTERN.search(text)
                or DAY_PATTERN.search(text)
                or re.search(r"\b\d{1,2}\b", text)):
            score += 0.35
        if score > best_score:
            best = row
            best_score = score

    if not best or best_score < 1.1:
        return None

    row_text = best["chunk"]
    page = best["meta"]["page"]
    answer = re.sub(r"\s*\|\s*", " ", row_text).rstrip(".") + "."
    return answer, [page], row_text
