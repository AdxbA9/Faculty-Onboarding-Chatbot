"""
Deterministic answers from the handbook's contact directory (post-freeze PF-16.5, decision D-128).

The legacy contact extractor (D-120) binds only a single printed line that holds an office name and exactly two numbers.
Every entry printed over several lines - labelled telephones ("Men: ..." / "Women: ..."), sub-units of a group entry
("Libraries" -> "Men's Library: ..."), names or numbers wrapped onto the next line - went to the model, which in the PF-16
live run gave the College of Engineering's FAX number as its telephone and one library's number as "the Library".

This module reads the directory pages (the lines after the "Telephone Number" / "Fax Number" header) in printed order and
binds every complete number to its UNIT and its COLUMN:

    entry      the office name, which may run over two lines ("College of Graduate Studies and" / "Research")
    group      the numbers of one cell, joined across "/" and across lines that end in "/" or an unfinished number
    label      a printed label before a group ("Men:", "Women's Clinic:", "Medical Colleges Library:"); a one-word label
               after a name line inside a labelled group is that sub-unit's wrapped name ("College of Fine Arts & Design"
               / "Library:")
    column     in an entry with labels, labelled groups are telephones and unlabelled groups after them are fax numbers;
               in an entry without labels the first group is the telephone column and the second the fax column

``answer(question, pages)`` states the asked column of exactly the asked unit, with the page, or - when the asked name is a
generic noun that several sub-units share ("the Library") - lists every one of them. It answers nothing when the unit is
not found or not certain; the existing pipeline then continues unchanged. Numbers that are not complete "+971 6 ..."
numbers (mobile emergency lines, a local-format security number, a malformed row) are never stated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agents.specialists import navigation_cues as _cues

TELEPHONE, FAX = _cues.TELEPHONE, _cues.FAX
_NUMBER = re.compile(_cues.NUMBER)
_LABEL_LINE = re.compile(r"^(?P<label>[A-Z][A-Za-z'’&().\- ]{0,40}?):\s*(?P<rest>.*)$")
_VALUE_START = re.compile(r"^(?:\+|\d|/|Ext\b)")
_RUNS_ON = re.compile(r"(?:/|\+971|\+971 6)\s*$")
_SECTION_LABELS = {"men", "women"}                   # labels that split one unit's telephones, not sub-units
_SPACES = re.compile("[    ]")
_CACHE: Dict[str, Any] = {}


@dataclass
class Group:
    label: str = ""
    text: str = ""
    labelled: bool = False
    column: str = ""

    @property
    def numbers(self) -> List[str]:
        return _NUMBER.findall(re.sub(r"\s+", " ", self.text))


@dataclass
class Entry:
    name: str
    page: int
    groups: List[Group] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)

    def column(self, which: str) -> List[Group]:
        return [g for g in self.groups if g.column == which and g.numbers]


def _norm(text: str) -> str:
    t = _SPACES.sub(" ", text or "").lower().replace("’", "'").replace("&", " and ")
    t = re.sub(r"\([^)]*\)", " ", t)                  # "(CCE)" is an alias, matched separately
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\b(?:the|a|an)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _assign_columns(entry: Entry) -> None:
    labelled_seen = False
    unlabelled = 0
    has_labels = any(g.labelled for g in entry.groups)
    for g in entry.groups:
        if g.labelled:
            g.column = TELEPHONE
            labelled_seen = True
        elif has_labels:
            g.column = FAX if labelled_seen else TELEPHONE
        else:
            g.column = TELEPHONE if unlabelled == 0 else (FAX if unlabelled == 1 else "")
            unlabelled += 1


def parse(pages: Sequence[Dict[str, Any]]) -> List[Entry]:
    """Every directory entry of the pages, in printed order. Cached per pages object."""
    if _CACHE.get("pages") is pages:
        return _CACHE["entries"]
    entries: List[Entry] = []
    for page in pages or []:
        lines = page.get("lines") if isinstance(page, dict) else None
        if not lines:
            continue
        stripped = [(l or "").strip() for l in lines]
        start = next((i + 1 for i in range(1, len(stripped)) if stripped[i] == "Fax Number" and stripped[i - 1] == "Telephone Number"), None)
        if start is None:
            continue
        body: List[str] = []
        for l in stripped[start:]:
            if not l:
                continue
            # a name line that also prints the first number ("Vice Chancellor for Academic Affairs +971 6 5053032")
            if not _VALUE_START.match(l) and not _LABEL_LINE.match(l) and "+" in l:
                cut = l.index("+")
                body.extend([l[:cut].strip(), l[cut:].strip()])
            else:
                body.append(l)
        current: Optional[Entry] = None
        pending_prefix = ""
        i = 0
        while i < len(body):
            line = body[i]
            nxt = body[i + 1] if i + 1 < len(body) else ""
            label = _LABEL_LINE.match(line)
            if _VALUE_START.match(line) and current is not None:
                last = current.groups[-1] if current.groups else None
                if last is not None and _RUNS_ON.search(last.text):
                    last.text += " " + line                     # a cell continued on this line
                else:
                    current.groups.append(Group(text=line))
                current.lines.append(line)
            elif label and current is not None:
                name = (pending_prefix + " " + label.group("label")).strip() if pending_prefix else label.group("label").strip()
                pending_prefix = ""
                current.groups.append(Group(label=name, text=label.group("rest"), labelled=True))
                current.lines.append(line)
            else:
                # a name line: the rest of a wrapped office name, the wrapped name of the next sub-unit, or a new entry
                open_group = current is not None and current.groups and all(g.labelled for g in current.groups)
                next_label = _LABEL_LINE.match(nxt)
                if current is not None and not current.groups:
                    current.name = (current.name + " " + line).strip()
                    current.lines.append(line)
                elif open_group and next_label and len(next_label.group("label").split()) == 1:
                    pending_prefix = line
                    current.lines.append(line)
                else:
                    current = Entry(name=line, page=int(page.get("page")), lines=[line])
                    entries.append(current)
            i += 1
    for e in entries:
        _assign_columns(e)
    _CACHE.update(pages=pages, entries=entries)
    return entries


@dataclass
class Unit:
    """An answerable unit: a whole entry, or one labelled sub-unit of a group entry."""
    name: str
    entry: Entry
    telephone: List[Tuple[str, str]]          # (label or "", number)
    fax: List[str]
    aliases: Tuple[str, ...] = ()


def units(pages: Sequence[Dict[str, Any]]) -> List[Unit]:
    out: List[Unit] = []
    for e in parse(pages):
        tel = [(g.label if g.label.lower() in _SECTION_LABELS else "", n) for g in e.column(TELEPHONE) for n in g.numbers
               if not g.label or g.label.lower() in _SECTION_LABELS]
        fax = [n for g in e.column(FAX) for n in g.numbers]
        if not tel:
            # an entry that prints only labelled sub-units ("Medical Service Department": Men's / Women's Clinic) answers
            # with those sub-units, each named
            tel = [(g.label, n) for g in e.column(TELEPHONE) for n in g.numbers if g.label]
        acronyms = tuple(a.lower() for a in re.findall(r"\(([A-Z]{2,6})\)", e.name))
        if tel or fax:
            out.append(Unit(e.name, e, tel, fax, acronyms))
        for g in e.groups:                                   # labelled sub-units ("Men's Library:")
            if g.labelled and g.label.lower() not in _SECTION_LABELS and g.numbers:
                out.append(Unit(g.label, e, [("", n) for n in g.numbers], []))
    return out


def _asked_target(question: str) -> str:
    q = _SPACES.sub(" ", question or "").strip().rstrip("?").strip()
    q = re.sub(r"\s+on\s+page\s+\d+\b", "", q, flags=re.I)
    m = re.search(r"\b(?:of|for)\s+(?:the\s+)?(.+)$", q, re.I)
    target = m.group(1) if m else ""
    target = re.sub(r"\b(?:telephone|phone|fax|contact|number|numbers)\b", " ", target, flags=re.I)
    return _norm(target)


def _tail_free(name: str) -> str:
    return re.sub(r"\s+(?:department|dept|office|unit|section)$", "", name)


_HEADER_TEXT = "Contact directory: Office / College / Department | Telephone Number | Fax Number |"


def _evidence(lines: List[str]) -> str:
    """The printed lines an answer rests on, with the page's column header, as one evidence text (numbers that wrap onto
    the next line are joined, exactly as they are printed)."""
    return _HEADER_TEXT + " " + re.sub(r"\s+", " ", " ".join(lines)).strip()


def _format(numbers: List[Tuple[str, str]]) -> str:
    return ", ".join(("%s: %s" % (label, n)) if label else n for label, n in numbers)


def answer(question: str, pages: Sequence[Dict[str, Any]]) -> Optional[Tuple[str, List[int], str]]:
    """(answer, [page], evidence lines) for a telephone / fax question about one directory unit, or about a generic name
    several sub-units share; None when the asked unit is not certain."""
    fields = [f for f in _cues.contact_fields(question or "") if f in (TELEPHONE, FAX)]
    if not fields:
        return None
    target = _asked_target(question)
    if not target:
        return None
    pool = units(pages)
    exact = [u for u in pool if target in {_norm(u.name), _tail_free(_norm(u.name))} | set(u.aliases)
             or _tail_free(target) in {_norm(u.name), _tail_free(_norm(u.name))}]
    if len(exact) == 1:
        u = exact[0]
        have = {TELEPHONE: bool(u.telephone), FAX: bool(u.fax)}
        if not any(have[f] for f in fields):
            # nothing of what was asked is printed: say so, and give what is printed for this unit
            printed = (["telephone %s" % _format(u.telephone)] if u.telephone else []) + (["fax %s" % ", ".join(u.fax)] if u.fax else [])
            what = " or ".join("%s number" % f for f in fields)
            text = "The contact directory lists no %s for %s%s." % (what, u.name, ("; it prints " + "; ".join(printed)) if printed else "")
        else:
            parts = []
            if TELEPHONE in fields:
                parts.append("telephone %s" % _format(u.telephone) if u.telephone else "no telephone number is listed")
            if FAX in fields:
                parts.append("fax %s" % ", ".join(u.fax) if u.fax else "no fax number is listed")
            text = "%s (contact directory): %s." % (u.name, "; ".join(parts))
        return text, [u.entry.page], _evidence(u.entry.lines)
    if exact:
        return None                                         # several units with exactly this name: not certain
    words = target.split()
    if len(words) == 1 and len(words[0]) >= 4 and TELEPHONE in fields and FAX not in fields:
        shared = [u for u in pool if u.entry.groups and re.search(r"\b%s\b" % re.escape(words[0]), _norm(u.name))
                  and u.name != u.entry.name]
        if len(shared) >= 2:
            # every sub-unit named with its own number, under the printed entry name ("Libraries (contact directory):
            # telephone Medical Colleges Library: ..., Men's Library: ..."): only printed words, so the Verifier can check it
            entries = list(dict.fromkeys(u.entry.name for u in shared))
            text = " ".join("%s (contact directory): telephone %s." % (
                name, ", ".join("%s: %s" % (u.name, n) for u in shared if u.entry.name == name for _l, n in u.telephone))
                for name in entries)
            pages_ = sorted({u.entry.page for u in shared})
            return text, pages_, _evidence(list(dict.fromkeys(l for u in shared for l in u.entry.lines)))
    return None


__all__ = ["Entry", "Group", "Unit", "answer", "parse", "units"]
