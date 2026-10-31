"""
Specialist source policy for approved official guides (post-freeze PF-2 seam, activated in PF-3B under D-110).

WHY
    The four frozen specialists accepted evidence from the Faculty Handbook
    only. PF-3B reopens exactly one contract boundary, source acceptance: a
    specialist may now accept evidence from the official guides approved for
    its own domain. Everything else in the specialists stays as frozen at
    `runtime-freeze-i9`. This module is the single place that decides which
    sources a specialist may accept and when a guide may answer at all.

WHAT
    * ``GUIDE_ALLOWLIST``: the explicit, code-level map of the guide sources
      each specialist may ever accept (the PF-2 domain mapping). No wildcard.
    * ``approved_sources_for`` / ``source_policy``: the approved set of one
      specialist = the primary handbook + every guide that is (a) in the
      allow-list for that specialist, (b) registered AND enabled, (c) not
      marked internal use, and (d) lists that specialist in its registry
      ``domains``. A disabled, held, unknown or unlisted source is never
      approved.
    * ``SYSTEM_TERMS`` / ``guide_sources_named``: the deterministic system
      vocabulary of each guide. The runtime uses guide-enabled specialists
      only for a question that names a guide's system; every other question
      runs on the specialists exactly as frozen at I-9.
    * Three checks the adapter applies to every finding that rests on a
      guide (they never apply to handbook evidence):
        - ``source_authority_problem``: the clause must name the system the
          guide documents (domain-source binding);
        - the same function refuses a policy, entitlement or obligation ask
          (the handbook stays authoritative for policy: system/policy
          boundary);
        - ``procedure_problem``: a "how do I / steps" clause is answered by a
          complete procedure or not at all.
    * ``page_collisions``: a guide whose page numbers fall inside a
      specialist's handbook page ranges refuses the guide-enabled registry
      (some frozen indexes are keyed by handbook page number).
    * ``conflict_topic``: the registered source conflict a chunk carries.

This module imports only the standard library and the source record type.
No model call, no file read.
"""
from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from .sources import SourceRecord

#: The four frozen specialists, by id, in the registry's build order.
SPECIALIST_IDS = ("teaching", "research", "faculty_services", "institutional")

#: The primary, section-mapped source every specialist accepts.
HANDBOOK_SOURCE_ID = "uos_faculty_handbook_2025_26"

#: Guide sources each specialist may ever accept (D-110). The registry must agree (enabled, domains); this table can only
#: narrow what the registry says, never widen it.
GUIDE_ALLOWLIST: Mapping[str, FrozenSet[str]] = {
    "teaching": frozenset({"uos_blackboard_faculty_short_guide", "uos_blackboard_gradebook_guide", "uos_grade_transfer_dashboard_guide"}),
    "research": frozenset({"uos_ithenticate_guidelines"}),
    "faculty_services": frozenset(),          # handbook only; the Performance Appraisal guide is on INGESTION HOLD
    "institutional": frozenset({"uos_blackboard_faculty_short_guide", "uos_grade_transfer_dashboard_guide"}),
}

#: The system each guide documents, as the words a question uses to name it. A guide answers only a clause that names
#: its system; a clause that names none is answered from the handbook or not at all.
SYSTEM_TERMS: Mapping[str, "re.Pattern[str]"] = {
    "uos_blackboard_faculty_short_guide": re.compile(r"\b(?:blackboard|bb\s+ultra)\b", re.I),
    "uos_blackboard_gradebook_guide": re.compile(r"\b(?:blackboard|bb\s+ultra|grade\s?book)\b", re.I),
    # "Banner" the student information system, never a course banner image ("course banner", "banner image")
    "uos_grade_transfer_dashboard_guide": re.compile(
        r"\b(?:grade\s+transfer|transfer(?:ring|red|s)?\s+(?:the\s+|my\s+|our\s+|final\s+|course\s+|student\s+)?(?:grades?|marks?))\b|"
        r"(?<!course\s)\bbanner\b(?!\s+(?:images?|photos?|pictures?|graphics?|pixels?))", re.I),
    "uos_ithenticate_guidelines": re.compile(r"\bi\s?thenticate\b", re.I),
}

#: Post-freeze PF-3C (D-111): the names under which each guide's DECLARED system is written in a question. When a clause
#: names one of these and the evidence comes from that guide, the registry's declaration stands for the name: a guide does
#: not repeat "Blackboard" in every sentence. Only these exact names are bound (case-insensitive, whole words); every other
#: word of the clause, actions and objects included, must still be stated by the evidence. Longest names first.
SYSTEM_NAMES: Mapping[str, Tuple[str, ...]] = {
    "uos_blackboard_faculty_short_guide": ("Blackboard Ultra", "BB Ultra", "Blackboard"),
    # the gradebook guide documents one module of Blackboard Ultra: its title names it ("Blackboard Ultra Guide: Gradebook")
    "uos_blackboard_gradebook_guide": ("Blackboard Ultra Gradebook", "Blackboard Gradebook", "Blackboard Ultra", "BB Ultra", "Blackboard",
                                       "Gradebook", "Grade Book"),
    # the grade transfer guide documents the dashboard that moves grades from Blackboard to Banner (registry: "Grade Transfer
    # Dashboard (Blackboard to Banner)"); "transfer grades" is the action it documents and is NOT bound
    "uos_grade_transfer_dashboard_guide": ("Grade Transfer Dashboard", "Grade Transfer Portal", "Blackboard", "Banner"),
    "uos_ithenticate_guidelines": ("iThenticate",),
}

#: Audiences whose documents may let second-person procedural text stand for the faculty member asking (D-111).
FACULTY_AUDIENCES = frozenset({"faculty", "instructor"})

#: Document types that are system manuals rather than policy (their evidence is subject to the checks below).
GUIDE_DOCUMENT_TYPES = frozenset({"system_guide", "user_guide", "guideline"})

#: A policy, entitlement or obligation ask: the handbook (or another policy source) is authoritative, never a manual.
_POLICY_ASK = re.compile(
    r"\b(?:polic(?:y|ies)|regulations?|rules?|entitle\w*|eligib\w*|obligat\w*|mandatory|compulsory|required\s+to|"
    r"(?:am|are|is)\s+(?:i|we|faculty|instructors?)\s+(?:required|allowed|permitted|obliged)|must\s+(?:i|we|faculty|instructors?)|"
    r"(?:i|we|faculty|instructors?)\s+must|have\s+to|deadlines?|due\s+(?:date|by)|within\s+how\s+(?:many|long)|"
    r"how\s+many\s+(?:days|hours|weeks|months)|penalt(?:y|ies)|sanctions?|disciplin\w*|misconduct|approv(?:e|al)\s+policy)\b", re.I)

#: A request for a procedure ("how do I", "steps", "walk me through"): answered by the whole procedure or not at all.
_HOWTO_ASK = re.compile(
    r"\bhow\s+(?:do|can|could|should|would|does|to)\b|\bsteps?\b|\bprocedures?\b|\bprocess\b|\binstructions?\b|"
    r"\bwalk\s+(?:me|us)\s+through\b|\bguide\s+(?:me|us)\b", re.I)


# ---------------------------------------------------------------------------
# The approved source sets
# ---------------------------------------------------------------------------
def _usable(record: Any) -> bool:
    return (isinstance(record, SourceRecord) and record.registered and record.enabled
            and record.access != "internal_use" and "HOLD" not in (record.verification_status or "").upper())


def approved_sources_for(records: Iterable[SourceRecord], specialist_id: str,
                         primary_source_id: str = HANDBOOK_SOURCE_ID) -> FrozenSet[str]:
    """The source ids ``specialist_id`` may accept: the primary handbook plus the guides allowed for it in BOTH the
    code-level allow-list and the registry (enabled, not held, listing the specialist in ``domains``)."""
    if specialist_id not in GUIDE_ALLOWLIST:
        raise ValueError("unknown specialist %r" % (specialist_id,))
    allowed = GUIDE_ALLOWLIST[specialist_id]
    approved = {primary_source_id}
    for record in records:
        if record.source_id == primary_source_id or record.source_id not in allowed:
            continue
        if _usable(record) and specialist_id in record.domains:
            approved.add(record.source_id)
    return frozenset(approved)


def source_policy(records: Sequence[SourceRecord], primary_source_id: str = HANDBOOK_SOURCE_ID) -> Dict[str, FrozenSet[str]]:
    """``approved_sources_for`` for every specialist."""
    return {sid: approved_sources_for(records, sid, primary_source_id) for sid in SPECIALIST_IDS}


def domain_sources(records: Iterable[SourceRecord]) -> Dict[str, FrozenSet[str]]:
    """Which enabled sources the registry alone maps to each specialist (its ``domains``), before the allow-list."""
    table: Dict[str, set] = {sid: set() for sid in SPECIALIST_IDS}
    for record in records:
        if not getattr(record, "enabled", True):
            continue
        for domain in record.domains:
            if domain in table:
                table[domain].add(record.source_id)
    return {sid: frozenset(ids) for sid, ids in table.items()}


# ---------------------------------------------------------------------------
# Which guides a question names
# ---------------------------------------------------------------------------
def guide_sources_named(text: str, records: Iterable[SourceRecord]) -> List[str]:
    """Usable guide sources whose system the text names, in registry order."""
    out: List[str] = []
    for record in records:
        pattern = SYSTEM_TERMS.get(record.source_id)
        if pattern is not None and _usable(record) and pattern.search(text or ""):
            out.append(record.source_id)
    return out


def usable_record(record: Any) -> bool:
    """A registered, enabled source that is neither internal use nor held (the D-110 usability test)."""
    return _usable(record)


def faculty_audience(record: Any) -> bool:
    """True when the registry says the document is written for faculty or instructors (D-111 actor binding)."""
    return bool(set(getattr(record, "audience", ()) or ()) & FACULTY_AUDIENCES)


def is_guide_meta(meta: Optional[Mapping[str, Any]]) -> bool:
    """True when a chunk's metadata says it comes from a system manual (not the handbook or another policy source)."""
    if not isinstance(meta, Mapping):
        return False
    if meta.get("source_id") == HANDBOOK_SOURCE_ID:
        return False
    return meta.get("document_type") in GUIDE_DOCUMENT_TYPES or meta.get("authority_level") == "system_guide"


# ---------------------------------------------------------------------------
# The checks on findings that rest on a guide
# ---------------------------------------------------------------------------
def source_authority_problem(clause: str, meta: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Why a finding resting on ``meta`` must not answer ``clause``, or None. Handbook evidence is never refused here."""
    if not is_guide_meta(meta):
        return None
    source_id = meta.get("source_id")
    pattern = SYSTEM_TERMS.get(source_id)
    if pattern is None:
        return "source %s has no registered system vocabulary; a guide answers only for its own system" % source_id
    if not pattern.search(clause or ""):
        return "the clause does not name the system documented by %s; a system guide does not answer outside its system" % source_id
    if _POLICY_ASK.search(clause or ""):
        return "a policy, entitlement or obligation ask: the handbook is authoritative, a system guide is not"
    return None


def procedure_problem(clause: str, quote: str, chunk: str, meta: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Why a finding resting on a guide must not answer a procedure ask, or None. A "how do I / steps" clause is answered
    only by a quote that covers every step of one whole procedure (a procedure window or step spanning it); one step or
    a sentence of a longer procedure is never presented as the procedure."""
    if not is_guide_meta(meta) or not _HOWTO_ASK.search(clause or ""):
        return None
    kind = meta.get("chunk_type")
    count = meta.get("step_count")
    if kind not in ("procedure_window", "procedure_step") or not isinstance(count, int) or count < 1:
        return "a procedure ask answered from a %s chunk: the complete procedure is not established" % (kind or "unknown")
    if kind == "procedure_step":
        if count != 1:
            return "one step of a %d-step procedure is not the procedure" % count
    else:
        first, last = meta.get("step_start"), meta.get("step_end")
        if first != 0 or last != count - 1:
            return "steps %s-%s of a %d-step procedure are not the whole procedure" % (first, last, count)
    heading = meta.get("procedure_heading")
    body = chunk[len(heading) + 2:] if isinstance(heading, str) and chunk.startswith(heading + ": ") else chunk
    units = [u.strip() for u in body.split(" | ") if u.strip()]
    missing = [u for u in units if u not in (quote or "")]
    if missing:
        return "the quote leaves out %d of %d procedure steps" % (len(missing), len(units))
    return None


def page_collisions(records: Sequence[SourceRecord], page_counts: Mapping[str, int], registry: Any) -> List[str]:
    """Guides whose page numbers fall inside a specialist's handbook page ranges (some frozen indexes, such as the
    Teaching calendar rows and the Institutional directory, are keyed by handbook page number). Empty for the pilot."""
    problems: List[str] = []
    for sid in SPECIALIST_IDS:
        specialist = _lookup(registry, sid)
        scope = getattr(specialist, "scope", None) if specialist is not None else None
        ranges = tuple(getattr(scope, "page_ranges", None) or ())
        if not ranges:
            continue
        lowest = min(first for first, _last in ranges)
        for record in records:
            if not getattr(record, "enabled", True) or record.section_map:
                continue                                           # the handbook (section-mapped) is the scope itself
            count = page_counts.get(record.source_id, 0)
            hit = [p for p in range(1, count + 1) if any(first <= p <= last for first, last in ranges)]
            if hit:
                problems.append("%s: pages %d-%d of %s overlap the %s scope (lowest scoped page %d)"
                                % (sid, hit[0], hit[-1], record.source_id, sid, lowest))
    return problems


def conflict_topic(meta: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The registered conflict topic a chunk carries, or None (see ``SourceRecord.conflicts``)."""
    if not isinstance(meta, Mapping):
        return None
    topic = meta.get("source_conflict")
    return topic if isinstance(topic, str) and topic else None


def _lookup(registry: Any, sid: str) -> Any:
    for name in ("get", "resolve", "specialist", "lookup"):
        fn = getattr(registry, name, None)
        if callable(fn):
            try:
                found = fn(sid)
            except Exception:
                found = None
            if found is not None:
                return found
    return None


__all__ = ["FACULTY_AUDIENCES", "GUIDE_ALLOWLIST", "GUIDE_DOCUMENT_TYPES", "HANDBOOK_SOURCE_ID", "SPECIALIST_IDS", "SYSTEM_NAMES", "SYSTEM_TERMS",
           "approved_sources_for", "conflict_topic", "domain_sources", "faculty_audience", "guide_sources_named", "is_guide_meta",
           "page_collisions", "procedure_problem", "source_authority_problem", "source_policy", "usable_record"]
