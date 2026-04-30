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
"""
from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple

from .retrieval import classify_query
from .text_utils import (
    CONTACT_QUESTION_PATTERN,
    COUNT_PATTERN,
    DAY_PATTERN,
    EMAIL_PATTERN,
    MONTH_PATTERN,
    PHONE_PATTERN,
    normalize_line,
    normalize_text,
    tokenize,
)


# Type alias: (answer_text, list_of_page_numbers, evidence_snippet)
ExtractorResult = Tuple[str, List[int], str]


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


def _score_row_against_target(target: str, row: str) -> float:
    """Combine token-overlap with fuzzy-ratio to find the best matching row."""
    target_tokens = tokenize(target)
    if not target_tokens:
        return 0.0
    row_tokens = set(tokenize(row))
    overlap = sum(1 for t in target_tokens if t in row_tokens)
    seq_ratio = difflib.SequenceMatcher(None, target.lower(), row.lower()).ratio()
    return overlap + seq_ratio


def extract_contact_answer(
    question: str, items: List[Dict]
) -> Optional[ExtractorResult]:
    """Return (answer, pages, evidence) for a contact question or None."""
    if not CONTACT_QUESTION_PATTERN.search(question):
        return None

    target = _office_query_target(question)
    ql = question.lower()
    want_fax = "fax" in ql
    want_email = "email" in ql

    # Prefer row-shaped chunks – that's where phone tables live.
    row_items = [
        item for item in items
        if item["meta"].get("chunk_type") in {"row", "row_window"}
    ] or items

    best: Optional[Dict] = None
    best_score = -1.0
    for item in row_items:
        score = _score_row_against_target(target, item["chunk"])
        if score > best_score:
            best = item
            best_score = score

    if not best or best_score < 0.35:
        return None

    text = best["chunk"]
    pages = [best["meta"]["page"]]

    if want_email:
        emails = EMAIL_PATTERN.findall(text)
        if emails:
            return f"The email address is {emails[0]}.", pages, text
        return None

    numbers = [normalize_text(x) for x in PHONE_PATTERN.findall(text)]
    if not numbers:
        return None

    if want_fax:
        # Tables typically list "Office | Telephone | Fax", so the fax number
        # tends to be the last phone-shaped token on the row.
        fax = numbers[-1] if len(numbers) >= 2 else numbers[0]
        return f"The fax number is {fax}.", pages, text

    return f"The phone number is {numbers[0]}.", pages, text


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


def _resolve_target_label(q: str) -> Optional[str]:
    for label, keywords in _DATE_LABELS.items():
        if any(k in q for k in keywords):
            return label
    return None


def extract_date_answer(
    question: str, items: List[Dict]
) -> Optional[ExtractorResult]:
    """Return a calendar-date answer built from the best matching row."""
    if classify_query(question) != "date":
        return None

    target_label = _resolve_target_label(question.lower())

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
        score = 0.0
        if target_label and target_label in text:
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

    if not best or best_score < 1.1 or not target_label:
        return None

    row_text = best["chunk"]
    page = best["meta"]["page"]
    answer = re.sub(r"\s*\|\s*", " ", row_text).rstrip(".") + "."
    return answer, [page], row_text
