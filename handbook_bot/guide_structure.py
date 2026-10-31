"""
Deterministic guide structure recovery (post-freeze PF-3D, decision record D-112).

WHY
    PF-3C (D-111) serves a guide procedure only when the PF-2 chunker read it as one marked list under a reliable
    heading. Measured on the 27 core guide questions PF-3C did not serve, 10 are false fallbacks caused by how the PDF
    structure was read, not by the evidence: the guides bullet many items with a private-use glyph on its own row (which
    the chunker does not treat as a list marker), title sub-sections with a number or letter ("1. Download Worflow",
    "C. Reorder content:") that the chunker takes for steps, use short headings without a colon ("Filter by child
    courses"), split a list's lead-in sentence across rows, and continue a section on the next page.

WHAT (formatting and text evidence only: no model, no invented step, no guessed sequence)
    * ``read_blocks``: the rows of one page become blocks. A block is a HEADING (a row ending with ":" or "?"; a "How
      to ..." row; a short title row between two sentences; a numbered or lettered sub-section title followed by
      prose), an ITEM (opened by a printed marker: a glyph bullet on its own row or at the row start, "-", "o ", a
      number or a letter) or PROSE. A row that is neither continues the open block, exactly as the chunker continues a
      step; a row opening in lower case is never a heading; rows without a three-letter word (figure callouts) are
      ignored.
    * ``recover``: per source, SECTIONS (a heading and its body until the next heading of the same or a higher rank)
      and LISTS (a run of consecutive items with its lead-in) are built across pages. A section reaching the end of a
      page continues on the next page only when that page opens with no heading; when the next page is not indexed
      while later pages are, the end is unproven and recorded as a problem.
    * Every block is mapped to EXACT quotes of indexed chunks of its own page (procedure steps, prose windows, rows),
      sentence by sentence when one chunk does not hold the whole block. A block that cannot be quoted exactly makes its
      structure unusable.
    * ``classify``: a list is an ordered procedure, unordered options, requirements, settings, features, contacts,
      numeric limits, approval stages or an ordered list, from its labels, its context and its items. Bullets are an
      ordered workflow only under an action heading with at least two instruction items. Nothing is renumbered: an
      item is rendered with its printed label, a bullet as a bullet.

Whether a recovered structure may answer a question is decided by ``guide_grounding.structure_problem`` (D-111 rules plus
the D-112 completeness rules).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .text_utils import normalize_text

VERSION = "guide-structure-1"

#: Bullet glyphs the guides print (private-use Symbol and Wingdings code points included).
GLYPHS = "•▪■●‣⁃◦∙·"
_GLYPH_ROW = re.compile(r"^[\s" + GLYPHS + r"]+$")
_GLYPH_START = re.compile(r"^[" + GLYPHS + r"]\s*(?=\S)")
_DASH_BULLET = re.compile(r"^[-–]\s+(?=\S)")
_O_BULLET = re.compile(r"^o\s+(?=[A-Z])")
_NUMBERED = re.compile(r"^(?:step\s+)?(\d{1,2})\s*[.)]\s+(?=\S)", re.I)
_LETTERED = re.compile(r"^([A-Za-z])\s*[.)]\s+(?=\S)")
_WORD3 = re.compile(r"[A-Za-z]{3,}")
_TERMINAL = (".", "!", "?", ";", ",", ":")
_SENTENCE_END = re.compile(r"(?<=[.!?:])\s+(?=[A-Z\"“(\d])")
_HOWTO = re.compile(r"^\s*how\s+(?:to|do|can|does)\b", re.I)
#: Every word capitalised ("Checking Similarity Report", "Gradebook Settings Panel"); short connectors allowed in lower case.
_TITLE_CASE = re.compile(r"^(?:[A-Z][\w&’'/-]*)(?:\s+(?:[A-Z][\w&’'/-]*|of|and|or|the|to|for|in|on|&))*$")

#: Action verbs that open an instruction ("Click ...", "In the menu, select ...", "Next, select ...").
INSTRUCTION_VERBS = (
    "click select choose enter type open go navigate upload download submit save create add review confirm post name provide "
    "use press tab drag search check set edit delete copy log sign look follow insert filter exempt show hide allow collect "
    "randomize point mark access find browse toggle revise approve transfer make view").split()
_VERB = r"(?:%s)" % "|".join(INSTRUCTION_VERBS)
_INSTRUCTION = re.compile(
    r"^(?:(?:next|then|finally|first|now)\s*,?\s+|(?:to|from|in|on|once|after|if|when|inside|within|by)\b[^.!?]{0,110}?,\s*|"
    r"from\b[^.!?,]{0,60}\s(?=(?:click|select|choose|open)\b))?(?:please\s+)?" + _VERB +
    r"\b(?!\s+(?:only|appears?|is|are|was|were|will|can|may|has|have|displays?|shows?)\b)", re.I)
#: An item that opens with a short title ("D. Set your view. Use the Items per page menu ...", "Roster: View your class list
#: ..."): a named option or callout, not a step of a sequence.
_TITLED_ITEM = re.compile(r"^(?:(?:\d{1,2}|[A-Za-z])\s*[.)]\s+)?(?P<title>[^.:!?]{2,48}?)[.:]\s+(?P<rest>\S.*)$")


def titled(block: "Block") -> bool:
    m = _TITLED_ITEM.match(block.text)
    return bool(m) and _words(m.group("title")) <= 5 and _words(m.group("rest")) >= 3

LIST_TYPES = ("ordered_procedure", "ordered_list", "unordered_options", "requirements", "settings", "features", "contacts",
              "numeric_limits", "approval_stages")


@dataclass(frozen=True)
class Block:
    kind: str                    # "heading" | "item" | "prose"
    marker: str                  # "bullet" | "numbered" | "lettered" | ""
    label: str                   # the printed label ("1.", "a)", "C.") or "" for bullets and prose
    text: str                    # normalized printed text (label included; glyph excluded)
    page: int
    rank: int = 0                # headings: 1 (section title) or 2 (colon lead-in heading); others 0
    rows: Tuple[str, ...] = ()

    @property
    def title(self) -> str:
        """A heading without its label and closing colon ("C. Reorder content:" -> "Reorder content")."""
        text = self.text
        if self.label and text.startswith(self.label):
            text = text[len(self.label):]
        return text.strip().rstrip(":").strip()

    def sentences(self) -> List[str]:
        body = self.text[len(self.label):].strip() if self.label and self.text.startswith(self.label) else self.text
        return [s.strip() for s in _SENTENCE_END.split(body) if s.strip()]

    @property
    def instruction(self) -> bool:
        """At least one sentence of the block opens with an instruction."""
        return any(_INSTRUCTION.match(s) for s in self.sentences())


def _strip_marker(text: str) -> Tuple[str, str, str]:
    """(marker, label, text) of a row that opens an item; ("", "", text) otherwise."""
    m = _GLYPH_START.match(text)
    if m:
        return "bullet", "", text[m.end():].strip()
    m = _DASH_BULLET.match(text)
    if m:
        return "bullet", "", text[m.end():].strip()
    m = _O_BULLET.match(text)
    if m:
        return "bullet", "", text[m.end():].strip()
    m = _NUMBERED.match(text)
    if m:
        return "numbered", text[: m.end()].strip(), text
    m = _LETTERED.match(text)
    if m:
        return "lettered", text[: m.end()].strip(), text
    return "", "", text


def _words(text: str) -> int:
    return len(text.split())


def read_blocks(rows: Sequence[str], page: int) -> List[Block]:
    """The blocks of one page (see the module docstring). Deterministic."""
    lines: List[Tuple[str, bool]] = []                     # (text, opened by a glyph row above)
    pending = False
    for row in rows:
        text = (row or "").strip()
        if not text:
            continue
        if _GLYPH_ROW.match(text):
            pending = True
            continue
        if not _WORD3.search(text):                         # figure callout letters ("A E", "C")
            continue
        lines.append((text, pending))
        pending = False

    def next_content(i: int) -> Optional[Tuple[str, bool]]:
        return lines[i + 1] if i + 1 < len(lines) else None

    blocks: List[Dict[str, Any]] = []
    for i, (text, glyph) in enumerate(lines):
        marker, label, body = _strip_marker(text)
        if glyph and not marker:
            marker, label, body = "bullet", "", text
        prev = blocks[-1] if blocks else None
        nxt = next_content(i)
        heading_rank = 0
        if marker in ("numbered", "lettered"):
            title = body[len(label):].strip()
            following = _strip_marker(nxt[0]) if nxt is not None else ("", "", "")
            if (title and _words(title) <= 6 and not title.endswith((".", "!", "?", ";", ",")) and title[:1].isupper()
                    and nxt is not None and following[0] not in ("numbered", "lettered") and following[2][:1].isupper()):
                heading_rank = 1                            # "1. Download Worflow", "C. Reorder content:", "3) Student Grid"
        elif marker == "bullet" and body.endswith(":") and _words(body) <= 8 and body[:1].isupper() and nxt is not None:
            heading_rank = 1 if _HOWTO.match(body) else 2   # a bulleted title: "Send an Announcement:", "Create an Assignment:"
        elif not marker and text[:1].isupper():
            if text.endswith((":", "?")) and _words(text) <= 12:
                heading_rank = 1 if (_HOWTO.match(text) or text.rstrip(": ").endswith("?")) else 2
            elif _HOWTO.match(text) and _words(text) <= 10 and not text.endswith((".", ",", ";")) and nxt is not None:
                heading_rank = 1                            # "How to upload a document"
            elif (_words(text) <= 5 and not text.endswith(_TERMINAL) and nxt is not None and _TITLE_CASE.match(text)
                  and (nxt[1] or nxt[0][:1].isupper() or _strip_marker(nxt[0])[0])):
                heading_rank = 1                            # a Title Case label: "Checking Similarity Report"
            elif (_words(text) <= 6 and not text.endswith(_TERMINAL) and nxt is not None
                  and (nxt[1] or nxt[0][:1].isupper() or _strip_marker(nxt[0])[0])
                  and (prev is None or prev["kind"] == "heading" or prev["text"].endswith((".", "!", "?", ":")))):
                heading_rank = 1                            # a short title line between two sentences
        if heading_rank:
            blocks.append({"kind": "heading", "marker": marker, "label": label, "text": normalize_text(body), "rank": heading_rank,
                           "rows": [text]})
            continue
        if marker:
            blocks.append({"kind": "item", "marker": marker, "label": label, "text": normalize_text(body), "rank": 0, "rows": [text]})
            continue
        if prev is not None and prev["kind"] in ("item", "prose"):
            prev["text"] = normalize_text(prev["text"] + " " + text)
            prev["rows"].append(text)
            continue
        blocks.append({"kind": "prose", "marker": "", "label": "", "text": normalize_text(text), "rank": 0, "rows": [text]})
    return [Block(b["kind"], b["marker"], b["label"], b["text"], page, b["rank"], tuple(b["rows"])) for b in blocks]


# ---------------------------------------------------------------------------
# Exact quotes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StructurePart:
    chunk_id: int
    page: int
    quote: str
    marker: str                  # "heading" | "lead_in" | "bullet" | "numbered" | "lettered" | "prose"


_CHUNK_ORDER = {"procedure_step": 0, "paragraph": 1, "row_window": 2, "row": 3, "procedure_window": 4}


class PageChunks:
    """The indexed chunks of each guide page, in quoting preference order."""

    def __init__(self, chunks: Sequence[str], metadata: Sequence[Mapping[str, Any]], source_ids: Iterable[str]) -> None:
        wanted = set(source_ids)
        self.chunks = chunks
        self.by_page: Dict[Tuple[str, int], List[int]] = {}
        for i, meta in enumerate(metadata):
            sid = meta.get("source_id")
            if sid in wanted and meta.get("chunk_type") in _CHUNK_ORDER:
                self.by_page.setdefault((sid, meta.get("page")), []).append(i)
        for ids in self.by_page.values():
            ids.sort(key=lambda i: (_CHUNK_ORDER[metadata[i].get("chunk_type")], i))

    def find(self, source_id: str, page: int, text: str) -> Optional[int]:
        for cid in self.by_page.get((source_id, page), ()):
            if text in self.chunks[cid]:
                return cid
        return None

    def quote(self, source_id: str, block: Block, marker: str) -> Optional[List[StructurePart]]:
        """Exact quotes of ``block``: one part when one chunk holds it, else its sentences merged greedily per chunk."""
        cid = self.find(source_id, block.page, block.text)
        if cid is not None:
            return [StructurePart(cid, block.page, block.text, marker)]
        sentences = [s.strip() for s in _SENTENCE_END.split(block.text) if s.strip()]
        parts: List[StructurePart] = []
        i = 0
        while i < len(sentences):
            cid = self.find(source_id, block.page, sentences[i])
            if cid is None:
                return None
            j, text = i + 1, sentences[i]
            while j < len(sentences) and (text + " " + sentences[j]) in self.chunks[cid]:
                text = text + " " + sentences[j]
                j += 1
            parts.append(StructurePart(cid, block.page, text, marker if not parts else "prose"))
            i = j
        return parts


# ---------------------------------------------------------------------------
# Sections and lists across pages
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Structure:
    key: str
    kind: str                    # "section" | "list"
    source_id: str
    heading: str                 # what names the structure: the lead-in of a list, else its section's title
    section_title: Optional[str]
    lead_in: Optional[str]
    context: Tuple[Block, ...]   # the heading block (section) or lead-in source block (list)
    blocks: Tuple[Block, ...]    # body blocks (section) or items (list)
    section_blocks: Tuple[Block, ...]
    list_type: str
    pages: Tuple[int, ...]
    parts: Tuple[StructurePart, ...]
    start: str
    end: str
    problems: Tuple[str, ...] = ()

    def text(self) -> str:
        """What the structure states: its section title, its lead-in (a list) and every unit."""
        lead = [self.section_title] if self.section_title and self.section_title != self.heading else []
        return " ".join(lead + [self.heading] + [b.text for b in self.blocks])

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "kind": self.kind, "source_id": self.source_id, "heading": self.heading, "lead_in": self.lead_in,
                "section_title": self.section_title, "list_type": self.list_type, "pages": list(self.pages), "start": self.start,
                "end": self.end, "units": [{"marker": b.marker or b.kind, "label": b.label, "page": b.page} for b in self.blocks],
                "chunk_ids": [p.chunk_id for p in self.parts], "problems": list(self.problems)}


_FEATURES = re.compile(r"\byou\s+can\s*:?\s*$", re.I)
_EMAIL_OR_PHONE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s()-]{6,}\d")
_QUANTITY = re.compile(r"\d[\d,.]*\s*(?:%|mb|kb|gb|pages?|words?|files?|characters?|pixels?|documents?|days?|hours?)\b", re.I)


def classify(context: str, items: Sequence[Block], action_heading: bool) -> str:
    """The list type (LIST_TYPES) from the context and the items: deterministic and conservative (an unordered reading
    whenever order is not shown by the printed labels or by an action heading over instructions)."""
    ctx = (context or "").strip().rstrip(":").strip()
    if _FEATURES.search(ctx):
        return "features"
    low = ctx.lower()
    texts = [b.text for b in items]
    labelled = bool(items) and all(b.marker in ("numbered", "lettered") for b in items)
    if items and sum(1 for b in items if titled(b)) * 2 > len(items):
        return "unordered_options"                          # named options or callouts, whatever their labels
    if labelled and any(b.instruction for b in items):
        return "ordered_procedure"
    if "requirement" in low or (texts and sum(1 for t in texts if re.search(r"\bmust\b", t, re.I)) * 2 > len(texts)):
        return "requirements"
    if "setting" in low:
        return "settings"
    if texts and sum(1 for t in texts if _EMAIL_OR_PHONE.search(t)) * 2 > len(texts):
        return "contacts"
    if texts and sum(1 for t in texts if _QUANTITY.search(t)) * 2 > len(texts):
        return "numeric_limits"
    if "approv" in low:
        return "approval_stages"
    instructions = sum(1 for b in items if b.instruction)
    if labelled:
        return "ordered_list"
    if action_heading and instructions >= 2:
        return "ordered_procedure"
    return "unordered_options"


def recover(source_id: str, pages: Sequence[Mapping[str, Any]], page_chunks: PageChunks,
            is_action_heading: Any = None) -> List[Structure]:
    """Every section and list of one source, in page order. ``is_action_heading(text) -> bool`` says whether a heading
    names an action (supplied by the grounding module; None treats no heading as an action heading)."""
    ordered = sorted(pages, key=lambda p: p.get("page", 0))
    indexed = [p.get("page") for p in ordered]
    last = max(indexed) if indexed else 0
    per_page: Dict[int, List[Block]] = {p.get("page"): read_blocks(p.get("rows") or p.get("lines") or [], p.get("page")) for p in ordered}
    stream: List[Block] = []
    page_start: Dict[int, int] = {}
    for number in indexed:
        page_start[number] = len(stream)
        stream.extend(per_page[number])
    action = is_action_heading or (lambda _text: False)
    structures: List[Structure] = []

    def end_of(start: int, rank: int) -> Tuple[int, str, List[str]]:
        """The exclusive end of the section opened at ``start`` and why it ends there."""
        problems: List[str] = []
        i = start + 1
        while i < len(stream):
            block = stream[i]
            if block.page != stream[i - 1].page:
                prev_page, page = stream[i - 1].page, block.page
                if page != prev_page + 1:
                    problems.append("page %d is not indexed, so the section's end on page %d is not proven" % (prev_page + 1, prev_page))
                    return i, "page %d end (gap)" % prev_page, problems
                if block.kind == "heading":
                    return i, "page %d opens with a heading" % page, problems
            if block.kind == "heading" and block.rank <= rank:
                return i, "next heading on page %d" % block.page, problems
            i += 1
        tail = stream[-1].page if stream else 0
        if tail != last:
            problems.append("the document continues after page %d without indexed text" % tail)
        return i, "end of document (page %d)" % tail, problems

    def parts_of(context_marker: str, context_text: Optional[Block], body: Sequence[Block]) -> Tuple[List[StructurePart], List[str]]:
        parts: List[StructurePart] = []
        problems: List[str] = []
        units = ([(context_text, context_marker)] if context_text is not None else []) + [(b, b.marker or "prose") for b in body]
        for block, marker in units:
            quoted = page_chunks.quote(source_id, block, marker)
            if quoted is None:
                problems.append("a block on page %d is not held verbatim by any indexed chunk" % block.page)
                return [], problems
            parts.extend(quoted)
        return parts, problems

    for s, block in enumerate(stream):
        if block.kind != "heading":
            continue
        e, why, gap = end_of(s, block.rank)
        body = stream[s + 1:e]
        pages_used = tuple(dict.fromkeys([block.page] + [b.page for b in body]))
        if body:
            parts, problems = parts_of("heading", block, body)
            structures.append(Structure("%s:S:%d:%d" % (source_id, block.page, s), "section", source_id, block.title, block.title, None,
                                        (block,), tuple(body), tuple(body), "section", pages_used, tuple(parts),
                                        "heading on page %d" % block.page, why, tuple(gap + problems)))
        # the lists of this section: runs of consecutive items with their lead-in
        j = 0
        while j < len(body):
            if body[j].kind != "item":
                j += 1
                continue
            family = "bullet" if body[j].marker == "bullet" else "labelled"
            k = j
            while k < len(body) and body[k].kind == "item" and ("bullet" if body[k].marker == "bullet" else "labelled") == family:
                k += 1                                      # a run keeps one marker family: bullets never run into numbers
            run = body[j:k]
            before = body[j - 1] if j > 0 else None
            lead_block: Optional[Block] = None
            lead_in: Optional[str] = None
            if before is not None and before.kind in ("prose", "item"):
                last_sentence = before.sentences()[-1] if before.sentences() else ""
                if last_sentence.endswith(":") or (before.kind == "prose" and run[0].marker in ("numbered", "lettered")):
                    lead_in = last_sentence
                    lead_block = Block("prose", "", "", last_sentence, before.page, 0, ())
            run_end = "next heading or section end" if k == len(body) else "a block that is not an item"
            run_problems = list(gap) if k == len(body) else []
            context = lead_in or block.title
            list_type = classify(context, run, action(block.title) or action(lead_in or ""))
            parts, problems = parts_of("lead_in" if lead_block is not None else "heading", lead_block or block, run)
            structures.append(Structure("%s:L:%d:%d:%d" % (source_id, run[0].page, s, j), "list", source_id, context, block.title, lead_in,
                                        (lead_block or block,), tuple(run), tuple(body), list_type,
                                        tuple(dict.fromkeys([(lead_block or block).page] + [b.page for b in run])), tuple(parts),
                                        "lead-in sentence" if lead_in else "section heading", run_end, tuple(run_problems + problems)))
            j = k
    # one section key per heading: a heading nested inside a longer section keeps both (each answers its own scope)
    return structures


__all__ = ["Block", "GLYPHS", "INSTRUCTION_VERBS", "LIST_TYPES", "PageChunks", "Structure", "StructurePart", "VERSION", "classify",
           "read_blocks", "recover"]
