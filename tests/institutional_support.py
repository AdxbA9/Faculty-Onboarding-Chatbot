"""
Shared fixtures of the Institutional Navigation tests (Milestone 4, Step 4.16-B1).

The corpus is built from REAL handbook pages, read with the project's own PDF loader and chunked with the project's own
chunker, exactly as the production index is built, but embedded with the fake embedder (no model). The page set covers
every page of the Institutional scope and the foreign pages around it (strategic priorities, the academic calendar,
the appointment chapter, the IT policies, library circulation, health and safety policies, the HR-owned child care and
medical clinic pages), so scope and section-guard tests see real boundaries.

The HR fixture helpers (``faculty_services_support``) are imported read-only for the source registry, the section map,
the corpus builder and the stand-in rerankers; nothing in them is changed.

Phase B2 (Step 4.16-B2) adds ``printed_directory()``, an independent oracle of the printed contact directory rebuilt
from the PDF's own geometry (the ruled cells of the office column and the words inside the telephone and fax columns),
not from the extracted lines the specialist reads, and ``make_institutional_with_lines()``, which builds the specialist
on a copy of the corpus with chosen directory lines replaced, for the negative controls.

Phase B3 (Step 4.16-B3) adds ``evidence_cases()`` (``qa/institutional_4_16/evidence_cases.jsonl``),
``make_institutional_with_text()``, which builds the specialist on a copy of the corpus with a crafted sentence appended
to chosen passages, and ``evaluate_evidence_case()``, the judge of the responsibility, structure and location answers.
``evaluate_contact_case()`` judges directory findings only; the phase-B3 statement findings a mixed clause may add are
checked for verbatim quotes and carry no number.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from handbook_bot.agents.specialists.base import check_findings
from handbook_bot.agents.specialists.contracts import SpecialistTask
from handbook_bot.agents.specialists.institutional import InstitutionalNavigationSpecialist, InstitutionalResources

from faculty_services_support import RawOverlapReranker, ScriptedReranker, corpus_from, registry_and_map

REPO = Path(__file__).resolve().parents[1]
HANDBOOK_PDF = REPO / "data" / "UOS Faculty Handbook 25-26.pdf"
#: Institutional scope pages plus the foreign pages that border them.
FIXTURE_PAGES = frozenset(set(range(14, 60)) | set(range(181, 186)) | set(range(196, 203)) | set(range(236, 260)))


@lru_cache(maxsize=1)
def handbook_pages() -> tuple:
    from handbook_bot.pdf_loader import load_pdf

    return tuple(load_pdf(str(HANDBOOK_PDF)))


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("’", "'").replace("‘", "'")).strip()


@lru_cache(maxsize=None)
def page_text(number: int) -> str:
    """The printed text of one handbook page, whitespace collapsed and curly apostrophes straightened."""
    page = next(p for p in handbook_pages() if p["page"] == number)
    return normalise(" ".join(page["lines"]))


@lru_cache(maxsize=1)
def institutional_corpus():
    pages: List[Dict] = [p for p in handbook_pages() if p["page"] in FIXTURE_PAGES]
    return corpus_from(registry_and_map(), pages)


def make_institutional(corpus=None, reranker=None, **kwargs) -> InstitutionalNavigationSpecialist:
    corpus = corpus or institutional_corpus()
    res = InstitutionalResources(embed=lambda t: corpus.embedder.encode([t]), rerank=(reranker or RawOverlapReranker()).predict,
                                 index=corpus.index, chunks=corpus.chunks, metadata=corpus.metadata, source=corpus.source,
                                 section_map=corpus.section_map)
    return InstitutionalNavigationSpecialist(res, **kwargs)


def task(question, *, clauses=None, intent="policy", task_id="task-1", specialist_id="institutional", requested_by="coordinator",
         **extra) -> SpecialistTask:
    """A task as the Coordinator builds it; ``requested_by`` names another specialist for a handoff task (Step 4.16-B4)."""
    context = {"full_question": question, "focus_clauses": list(clauses) if clauses else [question]}
    return SpecialistTask(task_id=task_id, question=question, specialist_id=specialist_id, domain="institutional", intent=intent,
                          context=context, requested_by=requested_by, **extra)


def four_specialist_registry(institutional=None):
    """The four Plan F specialists on their test fixtures (Step 4.16-B4): Teaching, Research and Faculty Services & HR built
    by their own test helpers, unchanged and only imported; Institutional Navigation on its fixture corpus."""
    import faculty_services_support as FS
    import test_research_integration as RI
    import test_research_specialist as RT
    import test_teaching_integration as TI
    from handbook_bot.agents.specialists.registry import SpecialistRegistry

    rm = FS.registry_and_map()
    registry = SpecialistRegistry()
    for specialist in (TI.teaching_for(RI._teaching_corpus()), RT.make_specialist(RT.corpus_from(rm, RT.RESEARCH_PAGES)),
                       FS.make_specialist(FS.corpus_from(rm, FS.PAGES)), institutional or make_institutional()):
        registry.register(specialist)
    return registry


def run(sp, question, **kw):
    t = task(question, **kw)
    return check_findings(sp, t, sp.run(t))


def make_institutional_with_lines(replacements: Dict[object, str], reranker=None, **kwargs) -> InstitutionalNavigationSpecialist:
    """The specialist on a copy of the fixture corpus in which the row chunks named by chunk id get new text."""
    return make_institutional_with_text(replacements=replacements, reranker=reranker, **kwargs)


def make_institutional_with_text(appends: Dict[object, str] = None, replacements: Dict[object, str] = None, reranker=None,
                                 **kwargs) -> InstitutionalNavigationSpecialist:
    """The specialist on a copy of the fixture corpus in which the chunks named by chunk id get new text
    (``replacements``) or a crafted sentence appended to their printed text (``appends``). The index is unchanged."""
    corpus = institutional_corpus()
    appends, replacements = appends or {}, replacements or {}
    chunks, metadata = list(corpus.chunks), [dict(m) for m in corpus.metadata]
    done = set()
    for i, meta in enumerate(metadata):
        cid = meta.get("chunk_id")
        if cid in replacements:
            chunks[i] = replacements[cid]
        if cid in appends:
            chunks[i] = chunks[i] + appends[cid]
        if cid in replacements or cid in appends:
            meta["text"] = chunks[i]
            done.add(cid)
    assert done == set(replacements) | set(appends), "unknown chunk ids: %r" % ((set(replacements) | set(appends)) - done)
    res = InstitutionalResources(embed=lambda t: corpus.embedder.encode([t]), rerank=(reranker or RawOverlapReranker()).predict,
                                 index=corpus.index, chunks=chunks, metadata=metadata, source=corpus.source, section_map=corpus.section_map)
    return InstitutionalNavigationSpecialist(res, **kwargs)


#: The printed directory's columns on pages 52-55 (x ranges in PDF points, from the table's vertical rules).
DIRECTORY_PDF_PAGES = (52, 53, 54, 55)
_OFFICE_COL, _TEL_COL, _FAX_COL = (72.0, 277.4), (277.4, 445.9), (445.9, 546.7)


@lru_cache(maxsize=1)
def printed_directory() -> tuple:
    """The contact directory as the PDF prints it: one entry per ruled cell of the office column, with the lines of
    the telephone and fax columns inside that cell. Cells without office words (sub-rows) are added to the entry above.
    Built from geometry only; it never reads the extracted lines the specialist reads."""
    import pymupdf

    from handbook_bot.agents.specialists import navigation_cues as N

    def rules(page):
        out = []
        for drawing in page.get_drawings():
            for item in drawing["items"]:
                if item[0] == "l" and abs(item[1].y - item[2].y) < 0.5:
                    out.append((round(item[1].y, 1), min(item[1].x, item[2].x), max(item[1].x, item[2].x)))
                elif item[0] == "re" and item[1].height < 2.5:
                    out.append((round((item[1].y0 + item[1].y1) / 2, 1), item[1].x0, item[1].x1))
        return out

    def lines(words, col, y0, y1):
        sel = sorted((w for w in words if col[0] <= (w[0] + w[2]) / 2 < col[1] and y0 <= (w[1] + w[3]) / 2 < y1),
                     key=lambda w: (round(w[1] / 3), w[0]))
        out, cur, last = [], [], None
        for w in sel:
            if last is not None and abs(w[1] - last) > 3:
                out.append(" ".join(cur))
                cur = []
            cur.append(w[4])
            last = w[1]
        if cur:
            out.append(" ".join(cur))
        return out

    # A cell is mapped to a lexicon office only when it prints that office's whole name, alone or followed by its
    # attested acronym in brackets ("... Professional Development (CCE)").
    printed_names = {}
    for office in N.OFFICES:
        printed_names[normalise(office.name)] = office.office_id
        for alias, _pages in office.aliases:
            printed_names[normalise("%s (%s)" % (office.name, alias))] = office.office_id
    doc = pymupdf.open(str(HANDBOOK_PDF))
    entries: List[Dict] = []
    for number in DIRECTORY_PDF_PAGES:
        page = doc[number - 1]
        words = page.get_text("words")
        ys = sorted({y for y, x0, x1 in rules(page) if x0 <= _OFFICE_COL[0] + 20 and x1 >= _OFFICE_COL[1] - 20})
        merged: List[float] = []
        for y in ys:
            if not merged or y - merged[-1] > 4:
                merged.append(y)
        for y0, y1 in zip(merged, merged[1:]):
            office = " ".join(lines(words, _OFFICE_COL, y0, y1))
            tel, fax = lines(words, _TEL_COL, y0, y1), lines(words, _FAX_COL, y0, y1)
            if not office:
                if entries and entries[-1]["page"] == number:
                    entries[-1]["telephone"] += tel
                    entries[-1]["fax"] += fax
                continue
            entries.append({"page": number, "office": office, "office_id": printed_names.get(normalise(office)), "telephone": tel, "fax": fax})
    return tuple(entries)


CASES_PATH = REPO / "qa" / "institutional_4_16" / "contact_cases.jsonl"


def contact_cases() -> List[Dict]:
    """The Step 4.16-B2 contact-accuracy cases (``qa/institutional_4_16/contact_cases.jsonl``)."""
    import json

    with open(CASES_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def specialist_for_case(case: Dict, reranker=None) -> InstitutionalNavigationSpecialist:
    lines = case.get("replace_lines")
    if lines:
        return make_institutional_with_lines({int(k): v for k, v in lines.items()}, reranker=reranker)
    return make_institutional(reranker=reranker)


def run_case(sp, case: Dict):
    return run(sp, case["question"], clauses=case.get("clauses"))


def evaluate_contact_case(case: Dict, result, sp) -> Dict:
    """Judge one result against its case and, on the real corpus, against the printed directory.

    blocker   a number given for an office or field the case does not expect, a number that differs from the printed
              cell, a number outside a bound finding, or a quote that is not verbatim in its line: wrong office or wrong
              number, the failures this phase must never produce
    mismatch  a status, an expected value, entry kind or limitation not met, with no blocker
    as_expected otherwise
    """
    from handbook_bot.agents.specialists import navigation_cues as N
    from handbook_bot.text_utils import PHONE_PATTERN

    blockers: List[str] = []
    problems: List[str] = []
    synthetic = bool(case.get("replace_lines"))
    oracle = {e["office_id"]: e for e in printed_directory() if e["office_id"]}
    texts = {meta.get("chunk_id"): text for text, meta in zip(sp.resources.chunks, sp.resources.metadata)}
    produced: Dict[str, Dict[str, str]] = {}
    allowed = set()
    for f in result.findings:
        m = f.metadata
        if f.claim != f.evidence_quote or f.evidence_quote not in texts.get(m["chunk_id"], ""):
            blockers.append("finding %s: quote not verbatim in chunk %r" % (f.finding_id, m["chunk_id"]))
        if "directory_entry" not in m:            # a phase-B3 statement finding of another ask: never a contact value
            if m.get("contact_values"):
                blockers.append("finding %s: a statement finding carries a contact value" % f.finding_id)
            continue
        if m["directory_entry"] == "complete_row":
            for field, value in m["contact_values"].items():
                if value not in f.evidence_quote:
                    blockers.append("finding %s: %s %r not in its quote" % (f.finding_id, field, value))
                produced.setdefault(m["office_id"], {})[field] = value
                allowed.add(value)
                if not synthetic and oracle.get(m["office_id"], {}).get(field) != [value]:
                    blockers.append("%s %s %r differs from the printed cell %r" % (m["office_id"], field, value, oracle.get(m["office_id"], {}).get(field)))
            for value in m["row_values"].values():
                allowed.add(value)
        elif m["directory_entry"] == "labelled_row":
            printed_lines = oracle.get(m["office_id"], {}).get("telephone", []) + oracle.get(m["office_id"], {}).get("fax", [])
            for item in m["labelled_values"]:
                allowed.add(item["value"])
                if not synthetic and "%s (%s)" % (item["value"], item["label"]) not in printed_lines:
                    blockers.append("%s labelled value %r not printed with label %r" % (m["office_id"], item["value"], item["label"]))
            if m["contact_values"]:
                blockers.append("finding %s: a labelled line was given a field" % f.finding_id)
        else:
            if PHONE_PATTERN.search(f.claim) or m["contact_values"]:
                blockers.append("finding %s: an unbound finding carries a number" % f.finding_id)
    expected = case.get("expected_values") or {}
    for office_id, fields in produced.items():
        for field, value in fields.items():
            if expected.get(office_id, {}).get(field) != value:
                blockers.append("unexpected %s %s %r" % (office_id, field, value))
    for office_id, fields in expected.items():
        for field, value in fields.items():
            if produced.get(office_id, {}).get(field) != value:
                problems.append("expected %s %s %r not given" % (office_id, field, value))
    for text in list(result.missing) + list(result.limitations):
        if PHONE_PATTERN.search(text):
            blockers.append("number outside a finding: %r" % text)
    for number in N.numbers_in(result.summary):
        if number not in allowed:
            blockers.append("summary number %r is not a bound value" % number)
    if result.status.value != case["expected_status"]:
        problems.append("status %s, expected %s" % (result.status.value, case["expected_status"]))
    blockers.extend(handoff_problems(case, result)["blockers"])           # Step 4.16-B4: no contact case expects a handoff
    if case.get("expected_entry") and not any(f.metadata.get("directory_entry") == case["expected_entry"] for f in result.findings):
        problems.append("no %s finding" % case["expected_entry"])
    if case.get("expected_limitation") and not any(case["expected_limitation"] in lim for lim in result.limitations):
        problems.append("limitation %r missing" % case["expected_limitation"])
    if produced or any(f.metadata.get("directory_entry") == "labelled_row" for f in result.findings):
        if not any("not verified as active" in lim for lim in result.limitations):
            blockers.append("numbers output without the as-printed limitation")
    outcome = "blocker" if blockers else ("mismatch" if problems else "as_expected")
    return {"case_id": case["case_id"], "category": case["category"], "outcome": outcome, "status": result.status.value,
            "expected_status": case["expected_status"], "produced": produced, "blockers": blockers, "problems": problems}


EVIDENCE_CASES_PATH = REPO / "qa" / "institutional_4_16" / "evidence_cases.jsonl"
#: Categories whose asks Institutional Navigation must leave without any finding (procedures, other specialists' subjects).
NO_FINDING_CATEGORIES = frozenset({"G_procedure", "H_boundary"})


def evidence_cases() -> List[Dict]:
    """The Step 4.16-B3 evidence-binding cases (``qa/institutional_4_16/evidence_cases.jsonl``)."""
    import json

    with open(EVIDENCE_CASES_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def specialist_for_evidence_case(case: Dict, reranker=None) -> InstitutionalNavigationSpecialist:
    appends, replacements = case.get("append_lines"), case.get("replace_lines")
    if appends or replacements:
        return make_institutional_with_text(appends={int(k): v for k, v in (appends or {}).items()},
                                            replacements={int(k): v for k, v in (replacements or {}).items()}, reranker=reranker)
    return make_institutional(reranker=reranker)


def office_named_in(office_id, quote: str) -> bool:
    """Whether a quote prints the office it is bound to: its lexicon name or an attested alias, word for word (case and
    whitespace aside), or "University" / "UoS" for the university as a whole. Independent of the specialist's reader."""
    from handbook_bot.agents.specialists import navigation_cues as N

    text = normalise(quote).lower()
    if office_id == N.UNIVERSITY:
        return bool(re.search(r"\buniversity\b|\buos\b", text))
    office = N.OFFICES_BY_ID[office_id]
    names = [office.name] + [alias for alias, _pages in office.aliases + office.subject_aliases]
    return any(re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(normalise(n).lower()), text) for n in names)


def _location_triple(f) -> list:
    m = f.metadata
    return [m.get("entry_label"), m.get("building_code"), m.get("place")]


def evaluate_evidence_case(case: Dict, result, sp) -> Dict:
    """Judge one result of a phase-B3 case.

    blocker     a quote that is not verbatim in its passage; a statement finding quoted from a contact-directory page; a
                finding bound to an office the case does not allow; a bound finding whose quote does not print its office;
                a forbidden quote; a supported answer the case does not allow; a location entry the case does not list; any
                finding for a procedure or another specialist's subject; a contact value the case does not expect or a
                number outside a bound finding; a supported count whose number is not the expected one. These are the
                wrong-office, wrong-location and invented-answer failures this phase must never produce.
    mismatch    a status, required office, required quote, expected location, count or contact value not met, with no blocker
    as_expected otherwise
    """
    from handbook_bot.agents.specialists import navigation_cues as N
    from handbook_bot.text_utils import PHONE_PATTERN

    blockers: List[str] = []
    problems: List[str] = []
    texts = {meta.get("chunk_id"): text for text, meta in zip(sp.resources.chunks, sp.resources.metadata)}
    allowed = set(case.get("allowed_offices") or [])
    expected_locations = [list(x) for x in case.get("expected_locations") or []]
    expected_contacts = case.get("expected_contacts") or {}
    quotes = [f.evidence_quote for f in result.findings]
    offices_found, locations_found, contacts_found = set(), [], {}
    for f in result.findings:
        m = f.metadata
        office = m.get("office_id")
        if f.claim != f.evidence_quote or f.evidence_quote not in texts.get(m.get("chunk_id"), ""):
            blockers.append("finding %s: quote not verbatim in chunk %r" % (f.finding_id, m.get("chunk_id")))
        if "directory_entry" in m:                # a phase-B2 contact finding
            if not expected_contacts and case["category"] != "I_b2_regression":
                blockers.append("finding %s: a contact-directory finding for a non-contact ask" % f.finding_id)
            for field, value in (m.get("contact_values") or {}).items():
                contacts_found.setdefault(office, {})[field] = value
                if expected_contacts.get(office, {}).get(field) != value:
                    blockers.append("unexpected contact value %s %s %r" % (office, field, value))
            if office is not None:
                offices_found.add(office)
                if office not in allowed:
                    blockers.append("finding %s: contact finding for %s, not allowed" % (f.finding_id, office))
            continue
        if f.page in DIRECTORY_PDF_PAGES:
            blockers.append("finding %s: a statement quoted from contact-directory page %s" % (f.finding_id, f.page))
        if m.get("contact_values"):
            blockers.append("finding %s: a statement finding carries a contact value" % f.finding_id)
        offices_found.add(office)
        if office not in allowed:
            blockers.append("finding %s: bound to %r, which the case does not allow (%s)" % (f.finding_id, office, sorted(map(str, allowed))))
        if office is not None and m.get("evidence_tier") is not None and not office_named_in(office, f.evidence_quote):
            blockers.append("finding %s: the quote does not print its office %s" % (f.finding_id, office))
        if office is None and m.get("evidence_tier") is not None:
            blockers.append("finding %s: an unbound finding carries an evidence tier" % f.finding_id)
        if m.get("relation") == "location":
            triple = _location_triple(f)
            locations_found.append(triple)
            if triple not in expected_locations:
                blockers.append("finding %s: location entry %r is not one the case lists" % (f.finding_id, triple))
    for bad in case.get("forbidden_quotes") or []:
        if any(normalise(bad).lower() in normalise(q).lower() for q in quotes):
            blockers.append("forbidden quote %r given" % bad)
    if result.status.value == "supported" and "supported" not in case["expected_status"]:
        blockers.append("supported, but the case allows only %s" % case["expected_status"])
    if case["category"] in NO_FINDING_CATEGORIES and result.findings:
        blockers.append("%d finding(s) for a %s ask" % (len(result.findings), case["category"]))
    allowed_numbers = {v for fields in contacts_found.values() for v in fields.values()}
    for text in list(result.missing) + list(result.limitations):
        if PHONE_PATTERN.search(text):
            blockers.append("number outside a finding: %r" % text)
    for number in N.numbers_in(result.summary):
        if number not in allowed_numbers:
            blockers.append("summary number %r is not a bound contact value" % number)
    handoffs = handoff_problems(case, result)
    blockers.extend(handoffs["blockers"])
    problems.extend(handoffs["problems"])
    count = case.get("expected_count")
    if count and result.status.value == "supported" and not re.search(r"\bstates %s\b" % re.escape(count), result.summary):
        blockers.append("supported count without the expected number %r: %s" % (count, result.summary))
    if result.status.value not in case["expected_status"]:
        problems.append("status %s, expected %s" % (result.status.value, "/".join(case["expected_status"])))
    for office in case.get("required_offices") or []:
        if office not in offices_found:
            problems.append("no finding for required office %s" % office)
    for q in case.get("required_quotes") or []:
        if not any(normalise(q).lower() in normalise(x).lower() for x in quotes):
            problems.append("required quote %r missing" % q)
    for triple in expected_locations:
        if triple not in locations_found:
            problems.append("expected location %r not given" % (triple,))
    if count and not re.search(r"\bstates %s\b" % re.escape(count), result.summary):
        problems.append("expected count %r not in the summary" % count)
    for office, fields in expected_contacts.items():
        for field, value in fields.items():
            if contacts_found.get(office, {}).get(field) != value:
                problems.append("expected %s %s %r not given" % (office, field, value))
    outcome = "blocker" if blockers else ("mismatch" if problems else "as_expected")
    return {"case_id": case["case_id"], "category": case["category"], "outcome": outcome, "status": result.status.value,
            "expected_status": case["expected_status"], "offices": sorted(map(str, offices_found)), "locations": locations_found,
            "handoffs": handoffs["made"], "blockers": blockers, "problems": problems}


def handoff_problems(case: Dict, result) -> Dict:
    """Phase B4 handoff checks of one result (Step 4.16-B4).

    blocker   a handoff to a specialist the case does not expect (a guessed or wrong owner), a handoff task that does not
              carry exactly one of the case's own clauses as its focus, comes from another specialist or carries anything but
              the full question and that clause, and an out_of_scope result with a finding
    mismatch  an expected handoff that is not made
    """
    expected = list(case.get("expected_handoffs") or [])
    clauses = [c.strip() for c in (case.get("clauses") or [case["question"]])]
    blockers: List[str] = []
    problems: List[str] = []
    made = [h.requested_specialist.value for h in result.handoff_requests]
    for h in result.handoff_requests:
        target = h.requested_specialist.value
        if target not in expected:
            blockers.append("handoff to %s, which the case does not expect (%s)" % (target, expected))
        t = h.task
        if (h.from_specialist.value != "institutional" or t is None or t.requested_by != "institutional" or t.specialist_id is not h.requested_specialist
                or t.question != case["question"] or set(t.context) != {"full_question", "focus_clauses"}
                or t.context["full_question"] != case["question"] or len(t.context["focus_clauses"]) != 1
                or t.context["focus_clauses"][0] not in clauses):
            blockers.append("handoff to %s does not carry exactly one of the case's own clauses" % target)
    if len(made) != len(set(zip(made, [h.metadata.get("clause_index") for h in result.handoff_requests]))):
        blockers.append("the same clause is handed to the same specialist twice: %s" % made)
    for target in expected:
        if target not in made:
            problems.append("expected handoff to %s not made" % target)
    if result.status.value == "out_of_scope" and result.findings:
        blockers.append("out_of_scope with %d finding(s)" % len(result.findings))
    return {"made": made, "blockers": blockers, "problems": problems}


__all__ = ["CASES_PATH", "DIRECTORY_PDF_PAGES", "EVIDENCE_CASES_PATH", "FIXTURE_PAGES", "HANDBOOK_PDF", "NO_FINDING_CATEGORIES", "RawOverlapReranker",
           "ScriptedReranker", "contact_cases", "evaluate_contact_case", "evaluate_evidence_case", "evidence_cases", "four_specialist_registry", "handoff_problems", "handbook_pages",
           "institutional_corpus", "make_institutional", "make_institutional_with_lines", "make_institutional_with_text", "normalise",
           "office_named_in", "page_text", "printed_directory", "run", "run_case", "specialist_for_case", "specialist_for_evidence_case", "task"]
