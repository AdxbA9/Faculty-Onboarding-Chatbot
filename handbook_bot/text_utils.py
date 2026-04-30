"""
Low-level text helpers: regex patterns, normalisation and tokenisation.

These are dependency-free so they can be used from any layer.
"""
from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Regex patterns (compiled once, reused everywhere)
# ---------------------------------------------------------------------------
PHONE_PATTERN = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

MONTH_PATTERN = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"january|february|march|april|june|july|august|"
    r"september|october|november|december)\b",
    re.I,
)
DAY_PATTERN = re.compile(
    r"\b(mon|tue|wed|thu|fri|sat|sun|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)
YEAR_TERM_PATTERN = re.compile(r"\b(fall|spring|summer)\b", re.I)

# Query intent patterns
COUNT_PATTERN = re.compile(r"\b(how many|number of|total number|count of|total)\b", re.I)
DATE_QUESTION_PATTERN = re.compile(
    r"\b(when|date|day|begin|starts?|begins?|held|deadline|"
    r"last day|add/drop|final exams?)\b",
    re.I,
)
CONTACT_QUESTION_PATTERN = re.compile(r"\b(phone|telephone|fax|contact|email|number)\b", re.I)
TABLE_HINT_PATTERN = re.compile(
    r"\b(phone|telephone|fax|contact|email|office|number|"
    r"date|when|calendar|schedule|committee|list|programs?)\b",
    re.I,
)
YESNO_PATTERN = re.compile(r"^(does|do|did|can|is|are|was|were)\b", re.I)
GREETING_PATTERN = re.compile(r"^(hi|hello|hey|salam|hola)\b\s*[!.?]?\s*$", re.I)

# Stopwords for lexical scoring. Handbook-specific terms like "university",
# "sharjah", "uos", "handbook" are stopworded because nearly every question
# repeats them and they add noise to overlap scores.
STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "for", "to", "in", "on", "at", "and",
    "or", "by", "with", "what", "which", "who", "when", "where", "how", "many",
    "does", "do", "did", "can", "could", "should", "would", "be", "it", "this",
    "that", "from", "as", "about", "into", "under", "their", "them", "they",
    "university", "sharjah", "uos", "faculty", "handbook", "please", "tell", "me",
}


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Collapse whitespace and non-breaking spaces into single spaces."""
    text = text.replace("\u00a0", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_line(line: str) -> str:
    """Like normalize_text, but also strips leading/trailing pipe noise."""
    line = line.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", line).strip(" |")


def tokenize(text: str) -> List[str]:
    """Tokenise for lexical scoring. Lowercases, drops stopwords and very
    short tokens. Keeps simple symbols that appear in handbook content."""
    tokens = re.findall(r"[a-zA-Z0-9+&\-/']+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
