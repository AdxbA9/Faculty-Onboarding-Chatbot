"""
Domain-to-source policy and the retrieval scope-extension seam (post-freeze PF-2).

WHY
    The four frozen specialists compile their retrieval scope from the Faculty
    Handbook's section map (``compile_scope``): source id and page ranges of
    the handbook only. Approved system guides are indexed in the same corpus,
    so a seam outside the frozen files is needed to let each specialist's
    retrieval see the guides of its own domain, deterministically, without
    letting any specialist see every document.

WHAT
    * The policy is data: every registered source names the specialists that
      may retrieve from it (``SourceRecord.domains``). The handbook lists all
      four; a guide lists the owners PF-1 mapped (Teaching for the Blackboard
      guides, Teaching and Institutional Navigation for the grade transfer
      guide, Research for iThenticate).
    * ``extend_specialist_scopes`` is called by ``plan_f_runtime`` after the
      four specialists are built: it replaces each specialist's ``scope`` with
      the same scope plus ``also_source_ids`` for its approved guides. The
      handbook constraints are untouched; a guide chunk is admitted to that
      specialist's retrieval only.
    * ``page_collisions`` checks the one hazard the frozen code cannot see: a
      guide whose page numbers fall inside a specialist's handbook page ranges
      would be mixed into the section guard's per-page attribution. The pilot
      guides have 3 to 11 pages and every specialist scope starts at page 14
      or later, and the check records that for each build.

LIMIT (recorded, not hidden)
    Each frozen specialist also rejects, inside its own file, any retrieved
    chunk whose ``source_id`` is not the handbook (one guard line per
    specialist). Until a decision record reopens those lines, the seam admits
    guide chunks to retrieval but no specialist renders guide evidence; the
    legacy path (unscoped retrieval) already sees the guides. This module
    therefore changes no served Plan F answer on its own.

This module imports only the standard library, the retrieval scope type and
the source record type. No model call, no file read.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence

from .retrieval import RetrievalScope
from .sources import SourceRecord

#: The four frozen specialists, by id, in the registry's build order.
SPECIALIST_IDS = ("teaching", "research", "faculty_services", "institutional")


def domain_sources(records: Iterable[SourceRecord]) -> Dict[str, FrozenSet[str]]:
    """Which enabled sources each specialist may retrieve from, read from the records' ``domains``."""
    table: Dict[str, set] = {sid: set() for sid in SPECIALIST_IDS}
    for record in records:
        if not getattr(record, "enabled", True):
            continue
        for domain in record.domains:
            if domain in table:
                table[domain].add(record.source_id)
    return {sid: frozenset(ids) for sid, ids in table.items()}


def extended_scope(scope: RetrievalScope, source_ids: Iterable[str]) -> RetrievalScope:
    """``scope`` plus ``source_ids`` as also-admitted sources. Sources the scope already names add nothing."""
    already = set(scope.source_ids or ()) | set(scope.also_source_ids or ())
    extra = {s for s in source_ids if isinstance(s, str) and s and s not in already}
    if not extra:
        return scope
    return replace(scope, also_source_ids=frozenset(set(scope.also_source_ids or ()) | extra))


def extend_specialist_scopes(registry: Any, records: Sequence[SourceRecord]) -> Dict[str, Dict[str, Any]]:
    """Apply the domain policy to the built specialists. Returns a report per specialist: the sources added
    (beyond the ones its own scope already names) and the resulting scope description. A specialist without a
    ``RetrievalScope`` attribute is left alone and reported as such."""
    policy = domain_sources(records)
    report: Dict[str, Dict[str, Any]] = {}
    for sid in SPECIALIST_IDS:
        specialist = _lookup(registry, sid)
        if specialist is None:
            report[sid] = {"applied": False, "reason": "not registered"}
            continue
        scope = getattr(specialist, "scope", None)
        if not isinstance(scope, RetrievalScope):
            report[sid] = {"applied": False, "reason": "no RetrievalScope attribute"}
            continue
        new_scope = extended_scope(scope, policy.get(sid, frozenset()))
        added = sorted(set(new_scope.also_source_ids or ()) - set(scope.also_source_ids or ()))
        if added:
            specialist.scope = new_scope
        report[sid] = {"applied": bool(added), "added": added, "allowed_sources": sorted(set(scope.source_ids or ()) | set(new_scope.also_source_ids or ()))}
    return report


def page_collisions(records: Sequence[SourceRecord], page_counts: Mapping[str, int], registry: Any) -> List[str]:
    """Guides whose page numbers fall inside a specialist's handbook page ranges (a hazard for the per-page
    section guard of the frozen specialists). Empty for the pilot set."""
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


__all__ = ["SPECIALIST_IDS", "conflict_topic", "domain_sources", "extend_specialist_scopes", "extended_scope", "page_collisions"]
