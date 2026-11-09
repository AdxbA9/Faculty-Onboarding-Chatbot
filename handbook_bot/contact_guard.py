"""
Contact-column guard for model-written answers (post-freeze PF-16, decision D-127).

The handbook's contact directory (section 1.16) prints, per office, a TELEPHONE column and a FAX column under the header
"Office / College / Department Telephone Number Fax Number". Several entries run over several lines and label their
telephones ("Men: ... Women: ..."), with the unlabelled fax numbers after them:

    College of Engineering
    Men: +971 6 5050901
    Women: +971 6 5050952
    +971 6 5585173 /
    +971 6 5585191

The legacy Verifier checks that every number in an answer is printed in the evidence; it cannot see which column a number
sits in. The PF-16 live run (case I14) showed the consequence: "What is the telephone number of the College of
Engineering?" was answered with the FAX number +971 6 5585173. This guard reads the directory pages, assigns every printed
number to its column, and rejects a model-written answer that gives a number only from the OTHER column than the one the
question asks for (a question naming one of telephone or fax). The rejected answer becomes the controlled refusal, so the
guard can only remove a wrong answer, never add a new one; extractor and specialist answers are not touched.

Column rules, per entry (an entry starts at a line that begins with a name, not with a label, a number or "/"):
    numbers joined by "/" form one group; a group right after a printed label ("Men:", "Women's Clinic:") or carrying
    "(External)"/"(Internal)" is labelled; in an entry with labelled groups, labelled groups are telephones and unlabelled
    groups after them are fax numbers; in an entry without labels the first group is the telephone column and the second
    the fax column, as the header prints them. Pages without a header, or without line structure, give no columns.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set

from .agents.specialists import navigation_cues as _cues

TELEPHONE, FAX = _cues.TELEPHONE, _cues.FAX
_LABEL_AT_START = re.compile(r"^[A-Z][A-Za-z'’]*(?:\s[A-Z][A-Za-z'’]*)?:")
_TOKEN = re.compile(r"(?P<label>[A-Z][A-Za-z'’]*(?:\s[A-Z][A-Za-z'’]*)?:)|(?P<num>%s)|(?P<slash>/)|(?P<ext>\((?:External|Internal)\))"
                    % _cues.NUMBER)
_CACHE: Dict[str, Any] = {}                  # the last pages object and its table (held, so its id cannot be reused)
_SPACES = re.compile("[    ]")


def _digits(number: str) -> str:
    return re.sub(r"\D", "", number or "")


def _entries(lines: Sequence[str]) -> List[List[str]]:
    """The directory entries of one page (the lines after its column header), each a list of lines."""
    out: List[List[str]] = []
    started = False
    prev_line = ""
    for raw in lines:
        line = (raw or "").strip()
        if not line:
            continue
        if _cues.header_columns(line) or (line == "Fax Number" and prev_line == "Telephone Number"):
            started = True                            # the header on one line, or split over "Telephone Number" / "Fax Number"
            prev_line = line
            continue
        prev_line = line
        if not started:
            continue
        continues = bool(_LABEL_AT_START.match(line)) or line[0] in "+/(" or line[0].isdigit()
        if continues and out:
            out[-1].append(line)
        else:
            out.append([line])
    return out


def _entry_fields(entry: List[str]) -> Dict[str, str]:
    # One text per entry: a line that ends with "/" or with an unfinished number ("+971", "+971 6") runs on into the next
    # line; any other line break separates two value groups (marked "|").
    text = ""
    for line in entry:
        if text:
            text += " " if re.search(r"(?:/|\+971|\+971 6)$", text) else " | "
        text += line
    groups: List[Dict[str, Any]] = []
    prev = None
    for m in re.finditer(_TOKEN.pattern + r"|(?P<brk>\|)", text):
        kind = m.lastgroup
        if kind == "num":
            if prev == "slash" and groups:
                groups[-1]["numbers"].append(m.group("num"))
            else:
                groups.append({"numbers": [m.group("num")], "labelled": prev == "label"})
        elif kind == "ext" and groups:
            groups[-1]["labelled"] = True
        prev = kind
    fields: Dict[str, str] = {}
    labelled_any = any(g["labelled"] for g in groups)
    for i, g in enumerate(groups):
        if labelled_any:
            field = TELEPHONE if g["labelled"] else (FAX if any(h["labelled"] for h in groups[:i]) else TELEPHONE)
        else:
            field = TELEPHONE if i == 0 else (FAX if i == 1 else None)
        if field:
            for n in g["numbers"]:
                fields[_digits(n)] = field
    return fields


def directory_columns(pages: Sequence[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Every directory number (digits only) -> the columns it is printed in (a number shared by two offices may be a
    telephone for one and a fax for another). Cached per pages object."""
    if _CACHE.get("pages") is pages:
        return _CACHE["table"]
    table: Dict[str, Set[str]] = {}
    for page in pages or []:
        lines = page.get("lines") if isinstance(page, dict) else None   # the printed reading order (rows reorder multi-line cells)
        if not lines:
            continue
        for entry in _entries(lines):
            for digits, field in _entry_fields(entry).items():
                table.setdefault(digits, set()).add(field)
    _CACHE.update(pages=pages, table=table)
    return table


def answer_problem(question: str, answer: str, pages: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Why a model-written answer states a number from the wrong directory column, or None."""
    asked = [f for f in _cues.contact_fields(question or "") if f in (TELEPHONE, FAX)]
    if len(asked) != 1:
        return None                                   # both columns or neither: nothing to check
    want = asked[0]
    # model output separates digit groups with no-break spaces (U+202F seen live, U+00A0, U+2007, U+2009)
    numbers = _cues.numbers_in(_SPACES.sub(" ", answer or ""))
    if not numbers:
        return None
    table = directory_columns(pages)
    known = [(n, table[_digits(n)]) for n in numbers if _digits(n) in table]
    # An answer that gives the asked number and also mentions the other column ("phone X, fax Y") is fine; an answer whose
    # directory numbers all come from the other column answers a different question.
    if known and not any(want in fields for _n, fields in known):
        n, fields = known[0]
        return "%s is printed as a %s number; the question asks for the %s number" % (n, "/".join(sorted(fields)), want)
    return None


__all__ = ["answer_problem", "directory_columns"]
