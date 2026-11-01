"""
The D-111 guide evidence path of a specialist (post-freeze PF-3C).

WHY
    PF-3B (D-110) let the frozen extraction of three specialists read approved guides; it served nothing and, measured,
    chose off-target guide quotes. PF-3C gives guide evidence ONE path, used only by the guide-enabled registry: the
    frozen specialist keeps its I-9 handbook-only extraction untouched, and a clause that names the system of a guide
    approved for that specialist (D-110) and asks a system question is first answered from that guide under the
    D-111 rules of ``handbook_bot.guide_grounding``. When the guide path finds no provably complete, fully covered
    evidence, the frozen specialist's own result for that clause stands, exactly as at I-9.

WHAT
    * ``GuideEvidence``: guide-scoped retrieval (the production hybrid search and cross-encoder, restricted to the named
      approved guide sources, the production reranker gate on every kept chunk) and the D-111 selection of a complete
      procedure, one item of an unordered list, or one statement.
    * ``GuideCapableSpecialist``: wraps one frozen specialist. Same id, same task contract; the frozen specialist is
      called for every clause the guide path does not answer, and its findings, reports, ``missing`` entries and
      handoffs for an answered clause are replaced by the guide clause report and findings. Every guide finding carries
      the D-111 record (ids, stems, labels) that the adapter re-verifies independently.

Post-freeze PF-5B (D-114): a clause the guide path binds to an approved guide (it names the guide's system and asks a
system question that is not a contact ask) is a SOFTWARE question owned by this specialist: when the guide path does
not answer it, the frozen specialist's request to hand it to Institutional Navigation ("Where do I find X in
Blackboard?") is dropped and the clause is reported not found, so the question falls back instead of being answered
from the handbook's offices. Physical and organizational asks ("Where is the Blackboard support office?") never bind to
a guide and keep their handoff.

No model call. No file read. The diagnostics carry ids, intents, forms and reasons, never question or answer text.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...config import MIN_RERANK_SCORE
from ...guide_grounding import (GUIDE_KEY, ORGANIZATIONAL, VERSION, Binding, GuideAnswer, GuideIndex, GuideIntent, select_answer,
                                statement_units)
from ...retrieval import RetrievalScope, deduplicate_by_text, gather_candidates
from ...source_scopes import HANDBOOK_SOURCE_ID, guide_sources_named
from ...text_utils import normalize_text
from ..types import ALL_ROUTES
from .base import Specialist
from .contracts import Finding, FindingStatus, SpecialistFindings, SpecialistId, SpecialistTask

#: Clauses per task the guide path reads (beyond it the frozen specialist decides alone).
MAX_GUIDE_CLAUSES = 4
#: Post-freeze PF-5B (D-114): the report marker of a software clause kept by its system's specialist.
SOFTWARE_NAVIGATION = "D-114 software navigation stays with the system's specialist"
#: The clause report fields the adapter's I-1 gate and the I-4 report check read (kept true for a guide report).
ASK_SPECIFIC_RECOGNISED = "specific_recognised"
RELATION_STRONG = "strong"
_SUPPORTED_OUTCOMES = ("supported", "read")
_CLAUSE_REF = re.compile(r"\bclause (\d+)\b")


class GuideEvidence:
    """Guide-scoped retrieval and D-111 selection over one knowledge base. Shared by the guide-capable specialists."""

    def __init__(self, embed: Callable[[str], np.ndarray], rerank: Callable[[Sequence[Sequence[str]]], Sequence[float]], index: Any,
                 chunks: Sequence[str], metadata: Sequence[Dict[str, Any]], guide_index: GuideIndex) -> None:
        self.embed = embed
        self.rerank = rerank
        self.index = index
        self.chunks = chunks
        self.metadata = metadata
        self.guide_index = guide_index
        self.records = list(guide_index.records.values())

    @classmethod
    def from_knowledge_base(cls, kb: Any, guide_index: Optional[GuideIndex] = None) -> "GuideEvidence":
        embedder, reranker = kb.embedder, kb.reranker

        def embed(text: str) -> np.ndarray:
            return embedder.encode([normalize_text(text)], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        if guide_index is None:
            guide_index = GuideIndex(kb.chunks, kb.metadata, getattr(kb, "source_pages", None) or {}, getattr(kb, "sources", None) or ())
        return cls(embed, reranker.predict, kb.index, kb.chunks, kb.metadata, guide_index)

    # ---- which guides may answer a clause -------------------------------------------------------------------------
    def bindings(self, clause: str, approved: Iterable[str]) -> Tuple[List[Binding], List[Dict[str, Any]]]:
        """The D-111 bindings of ``clause`` to every approved guide whose system it names, and the refusals. Post-freeze
        PF-5C (D-115): a clause that names no system is tried against the candidate guides ``guide_inference`` accepts for
        it (``GuideIndex.bind`` recomputes the inference and binds with ``inferred=True``)."""
        allowed = set(approved) - {HANDBOOK_SOURCE_ID}
        ok: List[Binding] = []
        refused: List[Dict[str, Any]] = []
        sources = guide_sources_named(clause, self.records)
        if not sources:
            from ...guide_inference import inferred_sources
            known = {r.source_id for r in self.records}
            sources = [s for s in inferred_sources(clause) if s in known]
        for source_id in sources:
            if source_id not in allowed:
                refused.append({"source_id": source_id, "reason": "not approved for this specialist (D-110)"})
                continue
            binding, problem = self.guide_index.bind(clause, source_id)     # D-114: with the guide's vocabulary
            if binding is None:
                refused.append({"source_id": source_id, "reason": problem})
            else:
                ok.append(binding)
        return ok, refused

    # ---- retrieval ---------------------------------------------------------------------------------------------------
    def retrieve(self, clause: str, route: str, sources: Iterable[str]) -> List[Dict[str, Any]]:
        """The production hybrid retrieval and cross-encoder over ``sources`` only, deduplicated and gated."""
        scope = RetrievalScope(source_ids=frozenset(sources))
        embedding = np.asarray(self.embed(clause), dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        candidates = gather_candidates(clause, route, embedding, self.index, list(self.chunks), list(self.metadata), scope=scope)
        if not candidates:
            return []
        scores = self.rerank([[clause, c["chunk"]] for c in candidates])
        items: List[Dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            items.append(item)
        items.sort(key=lambda x: (x["rerank_score"], x.get("routing_boost", 0.0), x.get("lexical_score", 0.0), x.get("dense_score", 0.0)),
                   reverse=True)
        return [item for item in deduplicate_by_text(items) if item["rerank_score"] >= MIN_RERANK_SCORE]

    # ---- selection ---------------------------------------------------------------------------------------------------
    def answer(self, clause: str, route: str, approved: Iterable[str]) -> Tuple[Optional[GuideAnswer], Dict[str, Any]]:
        """The D-111 answer of ``clause`` from the approved guides it names, or None, with a diagnostics record."""
        started = time.perf_counter()
        bindings, refused = self.bindings(clause, approved)
        diag: Dict[str, Any] = {"applies": bool(bindings), "refused": refused, "bindings": [b.to_dict() for b in bindings]}
        if not bindings:
            diag["ms"] = (time.perf_counter() - started) * 1000.0
            return None, diag
        t = time.perf_counter()
        pool = self.retrieve(clause, route, [b.source_id for b in bindings])
        diag["retrieval_ms"] = (time.perf_counter() - t) * 1000.0
        selection_started = time.perf_counter()
        diag["retrieved"] = [{"chunk_id": int(i["idx"]), "source_id": i["meta"].get("source_id"), "page": i["meta"].get("page"),
                              "chunk_type": i["meta"].get("chunk_type")} for i in pool[:10]]
        first_rank: Dict[str, int] = {}
        for rank, item in enumerate(pool):
            first_rank.setdefault(str(item["meta"].get("source_id")), rank)
        considered: List[Dict[str, Any]] = []
        for binding in sorted(bindings, key=lambda b: first_rank.get(b.source_id, 10 ** 6)):
            units = []
            procedure_ranks: Dict[str, int] = {}
            structure_ranks: Dict[str, int] = {}                 # post-freeze PF-3D (D-112): recovered sections and lists
            for rank, item in enumerate(pool):
                meta = item["meta"]
                if meta.get("source_id") != binding.source_id:
                    continue
                cid = int(item["idx"])
                proc = self.guide_index.procedure_of_chunk(cid)
                if proc is not None:
                    procedure_ranks.setdefault(proc.key, rank)
                for key in getattr(self.guide_index, "structures_by_chunk", {}).get(cid, ()):
                    structure_ranks.setdefault(key, rank)
                units.extend(statement_units(cid, self.chunks[cid], meta, rank, binding))
            chosen, tried = select_answer(binding, units, self.guide_index, procedure_ranks, structure_ranks)
            considered.append({"source_id": binding.source_id, "intent": binding.intent.value, "tried": tried[:12]})
            if chosen is not None:
                diag.update({"considered": considered, "form": chosen.form, "source_id": binding.source_id, "procedure_key": chosen.procedure_key,
                             "chunk_ids": [p.chunk_id for p in chosen.parts], "selection_ms": (time.perf_counter() - selection_started) * 1000.0,
                             "ms": (time.perf_counter() - started) * 1000.0})
                return chosen, diag
        diag.update({"considered": considered, "selection_ms": (time.perf_counter() - selection_started) * 1000.0,
                     "ms": (time.perf_counter() - started) * 1000.0})
        return None, diag

    def inspect(self, question: str, route: str = "policy") -> Dict[str, Any]:
        """Measurement view (``qa/post_freeze_pf3c/evidence_selection.py``): the D-111 choice for ``question`` from every
        usable guide it names, regardless of which specialist would own it, with the evidence in selection order."""
        approved = guide_sources_named(question, self.records)
        if not approved:                                          # post-freeze PF-5C (D-115)
            from ...guide_inference import inferred_sources
            approved = list(inferred_sources(question))
        chosen, diag = self.answer(question, route, approved)
        evidence: List[Dict[str, Any]] = []
        if chosen is not None:
            for part in chosen.parts:
                evidence.append({"chunk_id": part.chunk_id, "source_id": part.source_id, "page": part.page, "text": part.quote, "kind": chosen.form})
        return {"form": chosen.form if chosen else None, "intent": chosen.binding.intent.value if chosen else None, "evidence": evidence,
                "diagnostics": diag}


class GuideCapableSpecialist(Specialist):
    """A frozen specialist with the D-111 guide path in front of it (guide-enabled registry only). Build it with
    :meth:`wrap`, which picks the subclass that declares the wrapped specialist's id (the registry reads the id from the
    class)."""

    def __init__(self, base: Specialist, evidence: GuideEvidence, approved_sources: Iterable[str]) -> None:
        if not isinstance(base, Specialist):
            raise TypeError("GuideCapableSpecialist wraps a Specialist, got %s" % type(base).__name__)
        declared = type(self).__dict__.get("specialist_id")
        if declared is None or SpecialistId.parse(declared) is not base.specialist_id:
            raise ValueError("a guide-capable %s cannot wrap the %s specialist" % (declared, base.specialist_id.value))
        self.base = base
        self.evidence = evidence
        self.approved_sources: FrozenSet[str] = frozenset(approved_sources) | {HANDBOOK_SOURCE_ID}
        self.approved_guides: FrozenSet[str] = self.approved_sources - {HANDBOOK_SOURCE_ID}

    @staticmethod
    def wrap(base: Specialist, evidence: GuideEvidence, approved_sources: Iterable[str]) -> "GuideCapableSpecialist":
        cls = _WRAPPERS.get(base.specialist_id)
        if cls is None:
            raise ValueError("no guide path is defined for the %s specialist" % base.specialist_id.value)
        return cls(base, evidence, approved_sources)

    def __getattr__(self, name: str) -> Any:
        # scope, guard, resources ... of the frozen specialist (read by the registry checks and the tests)
        if name in ("base", "evidence", "approved_sources", "approved_guides"):
            raise AttributeError(name)
        return getattr(self.base, name)

    @staticmethod
    def _clauses(task: SpecialistTask) -> List[str]:
        raw = task.context.get("focus_clauses") if isinstance(task.context, dict) else None
        clauses = [c.strip() for c in raw if isinstance(c, str) and c.strip()] if isinstance(raw, (list, tuple)) else []
        return clauses or [task.question.strip()]

    @staticmethod
    def _route(task: SpecialistTask) -> str:
        return task.intent if task.intent in ALL_ROUTES and task.intent != "greeting" else "policy"

    def run(self, task: SpecialistTask) -> SpecialistFindings:
        started = time.perf_counter()
        clauses = self._clauses(task)
        if not self.approved_guides or len(clauses) > MAX_GUIDE_CLAUSES:
            return self.base.run(task)
        route = self._route(task)
        answers: Dict[int, GuideAnswer] = {}
        diags: Dict[str, Any] = {}
        for i, clause in enumerate(clauses):
            chosen, diag = self.evidence.answer(clause, route, self.approved_guides)
            if diag.get("applies"):
                diags[str(i)] = {k: v for k, v in diag.items() if k != "retrieved"}
            if chosen is not None:
                answers[i] = chosen
        guide_ms = (time.perf_counter() - started) * 1000.0
        if not answers:
            result = self._keep_software_clauses(clauses, diags, set(), self.base.run(task))
            if diags:
                result.metadata = dict(result.metadata or {})
                result.metadata["guide_path"] = {"version": VERSION, "answered": [], "clauses": diags, "ms": guide_ms}
            return result
        base = self.base.run(task) if len(answers) < len(clauses) else None
        if base is not None:
            base = self._keep_software_clauses(clauses, diags, set(answers), base)
        return self._merge(task, clauses, route, answers, diags, base, started, guide_ms)

    # ---- post-freeze PF-5B (D-114): software navigation stays with the system's specialist ------------------------------
    def _keep_software_clauses(self, clauses: List[str], diags: Mapping[str, Any], answered: Iterable[int],
                               result: SpecialistFindings) -> SpecialistFindings:
        """``result`` without its handoffs to Institutional Navigation for the clauses an approved guide BOUND (a software
        question about the guide's system that is not a contact ask and names no organizational unit, place or role) and the
        guide path did not answer; each such clause is reported not found and listed as missing. Any other result is returned
        unchanged."""
        if self.specialist_id is SpecialistId.INSTITUTIONAL or not result.handoff_requests:
            return result
        done = set(answered)
        owned = set()
        for key, diag in diags.items():
            # post-freeze PF-5C (D-115): only a clause that NAMES the guide's system is kept from Institutional Navigation; an
            # inferred binding never removes a handoff (inference decides candidates, not ownership of an unanswered clause)
            intents = {b.get("intent") for b in (diag.get("bindings") or []) if isinstance(b, dict) and not b.get("inferred")}
            if not (intents and GuideIntent.CONTACT.value not in intents and str(key).isdigit() and int(key) not in done):
                continue
            clause = clauses[int(key)] if int(key) < len(clauses) else ""
            if not ORGANIZATIONAL.search(clause):              # an office, team, role or contact keeps its Institutional handoff
                owned.add(int(key))
        dropped = [h for h in result.handoff_requests
                   if h.requested_specialist is SpecialistId.INSTITUTIONAL and (h.metadata or {}).get("clause_index") in owned]
        if not dropped:
            return result
        kept = [h for h in result.handoff_requests if not any(h is d for d in dropped)]
        metadata = dict(result.metadata or {})
        reports = list(metadata.get("clauses") or [])
        missing = list(result.missing)
        indices = sorted({int((h.metadata or {}).get("clause_index")) for h in dropped})
        for i in indices:
            clause = clauses[i] if i < len(clauses) else ""
            if 0 <= i < len(reports) and isinstance(reports[i], dict):
                clause = reports[i].get("clause", clause)
                reports[i] = {"index": i, "clause": clause, "outcome": "not_found", "kept": SOFTWARE_NAVIGATION}
            missing.append("clause %d: a software question about a guide's system that the approved guide does not answer (D-114)" % i)
        metadata["clauses"] = reports
        metadata["software_navigation_kept"] = indices
        status = result.status
        if status is FindingStatus.OUT_OF_SCOPE and not kept:
            status = FindingStatus.PARTIAL if result.findings else FindingStatus.NOT_FOUND
        elif status is FindingStatus.SUPPORTED:
            status = FindingStatus.PARTIAL
        confidence = {FindingStatus.SUPPORTED: 1.0, FindingStatus.PARTIAL: 0.5}.get(status, 0.0)
        return SpecialistFindings(
            specialist_id=result.specialist_id, task_id=result.task_id, status=status, findings=list(result.findings),
            summary="%s; %d software clause(s) kept, not handed to institutional navigation" % (result.summary, len(indices)),
            missing=missing, handoff_requests=kept, confidence=confidence, limitations=list(result.limitations),
            llm_used=result.llm_used, ms=result.ms, metadata=metadata)

    # ---- the merged result -------------------------------------------------------------------------------------------
    def _guide_parts(self, task: SpecialistTask, index: int, clause: str, route: str, answer: GuideAnswer) -> Tuple[Dict[str, Any], List[Finding]]:
        record = self.evidence.guide_index.record(answer.binding.source_id)
        title = (getattr(record, "label", None) or getattr(record, "title", "")) if record is not None else ""
        details = ["D-111 guide terms stated (%d) and system bound by %s" % (len(answer.binding.terms), answer.binding.source_id)]
        findings: List[Finding] = []
        for k, part in enumerate(answer.parts):
            findings.append(Finding(
                finding_id="%s-guide-%d-%d" % (self.specialist_id.value, index, k + 1), claim=part.quote, evidence_quote=part.quote,
                source_id=part.source_id, source_title=title, page=part.page, confidence=1.0,
                metadata={"chunk_id": part.chunk_id, "clauses": [index], "details_covered": list(details), "relation": RELATION_STRONG,
                          GUIDE_KEY: answer.record(k)}))
        report = {"index": index, "clause": clause, "outcome": "supported", "procedural": answer.binding.intent is GuideIntent.PROCEDURE,
                  "system": answer.binding.source_id, "findings": [f.finding_id for f in findings], "rejected": [],
                  "requested_details": list(details), "uncovered_details": [], "ask_shape": ASK_SPECIFIC_RECOGNISED, "relation": RELATION_STRONG,
                  "route": route, GUIDE_KEY: {k: v for k, v in answer.record(0).items() if k != "part"}}
        return report, findings

    def _merge(self, task: SpecialistTask, clauses: List[str], route: str, answers: Mapping[int, GuideAnswer], diags: Dict[str, Any],
               base: Optional[SpecialistFindings], started: float, guide_ms: float) -> SpecialistFindings:
        answered = set(answers)
        reports: List[Dict[str, Any]] = []
        findings: List[Finding] = []
        missing: List[str] = []
        handoffs = []
        limitations: List[str] = []
        metadata: Dict[str, Any] = {}
        if base is not None:
            base_reports = list((base.metadata or {}).get("clauses") or [])
            if len(base_reports) != len(clauses) or any(not isinstance(r, dict) or r.get("index", i) != i for i, r in enumerate(base_reports)):
                return base                                       # an unexpected report layout: the frozen result stands alone
            metadata = {k: v for k, v in (base.metadata or {}).items() if k != "clauses"}
            reports = base_reports
            for finding in base.findings:
                indices = [c for c in (finding.metadata.get("clauses") or []) if c not in answered]
                if not indices and (finding.metadata.get("clauses") or []):
                    continue
                if indices != list(finding.metadata.get("clauses") or []):
                    finding = Finding(finding.finding_id, finding.claim, finding.evidence_quote, finding.source_id, finding.source_title,
                                      finding.page, finding.confidence, dict(finding.metadata, clauses=indices))
                findings.append(finding)
            for entry in base.missing:
                m = _CLAUSE_REF.search(entry)
                if m is None or int(m.group(1)) not in answered:
                    missing.append(entry)
            handoffs = [h for h in base.handoff_requests if (h.metadata or {}).get("clause_index") not in answered]
            limitations = list(base.limitations)
        else:
            reports = [{"index": i, "clause": c, "outcome": "not_found"} for i, c in enumerate(clauses)]
            metadata = {"unprocessed_clauses": [], "route": route, "owned_clauses": len(clauses)}
        for i in sorted(answered):
            report, guide_findings = self._guide_parts(task, i, clauses[i], route, answers[i])
            reports[i] = report
            findings.extend(guide_findings)
        metadata["clauses"] = reports
        metadata.setdefault("unprocessed_clauses", [])
        metadata["guide_path"] = {"version": VERSION, "answered": sorted(answered), "clauses": diags, "ms": guide_ms}
        every = all(isinstance(r, dict) and r.get("outcome") in _SUPPORTED_OUTCOMES for r in reports)
        if findings and not missing and not handoffs and (base is None or base.status is FindingStatus.SUPPORTED or every):
            status, summary = FindingStatus.SUPPORTED, "%d clause(s) answered, %d from approved guides" % (len(clauses), len(answered))
        elif findings:
            status, summary = FindingStatus.PARTIAL, "%d clause(s) answered from approved guides; %d gap(s)" % (len(answered), len(missing))
        else:
            status, summary = FindingStatus.NOT_FOUND, "no evidence"
        confidence = {FindingStatus.SUPPORTED: 1.0, FindingStatus.PARTIAL: 0.5}.get(status, 0.0)
        return SpecialistFindings(
            specialist_id=self.specialist_id, task_id=task.task_id, status=status, findings=findings, summary=summary, missing=missing,
            handoff_requests=list(handoffs), confidence=confidence, limitations=limitations, llm_used=bool(base.llm_used) if base else False,
            ms=(time.perf_counter() - started) * 1000.0, metadata=metadata)


class GuideCapableTeaching(GuideCapableSpecialist):
    specialist_id = SpecialistId.TEACHING


class GuideCapableResearch(GuideCapableSpecialist):
    specialist_id = SpecialistId.RESEARCH


class GuideCapableInstitutional(GuideCapableSpecialist):
    specialist_id = SpecialistId.INSTITUTIONAL


#: The specialists a guide may be approved for (D-110): Faculty Services & HR has no guide path (the appraisal guide is held).
_WRAPPERS: Dict[SpecialistId, type] = {SpecialistId.TEACHING: GuideCapableTeaching, SpecialistId.RESEARCH: GuideCapableResearch,
                                       SpecialistId.INSTITUTIONAL: GuideCapableInstitutional}


__all__ = ["GuideCapableInstitutional", "GuideCapableResearch", "GuideCapableSpecialist", "GuideCapableTeaching", "GuideEvidence",
           "MAX_GUIDE_CLAUSES", "SOFTWARE_NAVIGATION"]
