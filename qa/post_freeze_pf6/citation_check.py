"""PF-6 citation and procedure integrity: every Plan F answer the PF-6 evaluation served (guide scopes ON) and a sample of the
frozen 192-question handbook answers (production default, guide scopes OFF) are asked again through the real seam and every
cited item is re-verified independently of the specialist that produced it:

* the quote is inside its chunk and, normalized, inside the printed text of the cited page (a procedure step chunk: its
  heading and its step each printed on that page); the page and source are the chunk's own; the source is enabled and its
  title is the registry's;
* a guide answer: its D-111 record re-verified by ``guide_grounding.verify_answer`` (one source, the specialist's approved set,
  every asked term stated, the complete procedure or section in printed order); its citation heading recomputed;
* the answering specialist is approved for every cited guide (D-110);
* the UI source chips (``ui.presentation.source_chips``) name the registry titles and exactly the cited pages, no raw id;
* the answer is identical to the evaluation's (determinism).

    python -X utf8 qa/post_freeze_pf6/citation_check.py <repo> <pf6 results.json> <frozen i8_results.json> <out dir>

Writes ``citation_review.json`` and ``procedure_final.json``.
"""
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
REPO = os.path.abspath(sys.argv[1])
R = json.load(open(sys.argv[2], encoding="utf-8"))
FROZEN = json.load(open(sys.argv[3], encoding="utf-8"))
OUT = os.path.abspath(sys.argv[4])
os.makedirs(OUT, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, REPO)
for name in ("PLAN_F_ENABLED", "PLAN_F_SHADOW", "PLAN_F_GUIDE_SCOPES"):
    os.environ.pop(name, None)

from handbook_bot import config                                     # noqa: E402
from handbook_bot import guide_grounding as GG                      # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from handbook_bot.name_spelling import correct_system_names         # noqa: E402
from ui import pipeline, presentation                               # noqa: E402

HB = "uos_faculty_handbook_2025_26"
_WS = re.compile(r"\s+")
kb = build_knowledge_base(verbose=False)
TITLES = {r.source_id: r.title for r in kb.sources}
PAGE_TEXT = {(sid, p["page"]): p["text"] for sid, pages in kb.source_pages.items() for p in pages}
words = frozenset(re.findall(r"[a-z]+", "\n".join(kb.chunks).lower()))


def norm(s):
    return _WS.sub(" ", s or "").strip()


_WORD = re.compile(r"[A-Za-z0-9]+")


def words_on_page(quote, page_text):
    """The quote's words appear contiguously on the printed page. The PF-2 chunker stores a procedure step as "<heading>: N. <step>"
    (the printed label "1)" becomes "1.", the heading is joined with ": "), so one- and two-digit list labels are left out of both
    sides; every other word, number and its order must match."""
    def toks(s):
        return [t.lower() for t in _WORD.findall(s or "") if not (t.isdigit() and len(t) <= 2)]
    q, p = toks(quote), toks(page_text)
    if not q:
        return False
    n = len(q)
    return any(p[i:i + n] == q for i in range(len(p) - n + 1))


def step_on_page(quote, chunk_meta, page_text):
    """A procedure_step chunk is "<procedure heading>: <label> <step>"; on the printed page an introduction may stand between the
    heading and its steps, so the heading and the step are each looked up on the page (both must be printed there)."""
    heading = chunk_meta.get("procedure_heading")
    if chunk_meta.get("chunk_type") != "procedure_step" or not heading or not quote.startswith(heading + ":"):
        return False
    body = quote[len(heading) + 1:].strip()
    label = chunk_meta.get("step_label")
    if label and body.startswith(label):
        body = body[len(label):].strip()
    return words_on_page(heading, page_text) and words_on_page(body, page_text)


_MODE = {}


def ask(q, scopes):
    if _MODE.get("scopes") is not scopes:                          # a new runtime for each guide-scope setting
        config.PLAN_F_ENABLED, config.PLAN_F_SHADOW, config.PLAN_F_GUIDE_SCOPES = True, False, scopes
        pipeline.reset_plan_f_state()
        _MODE["scopes"] = scopes
    return pipeline.ask_with_diagnostics(kb, q)


def check(qid, q, expected_answer, scopes):
    res, d = ask(q, scopes)
    runtime = pipeline._plan_f_cache().get_or_build(kb).runtime
    index = runtime.guide_index
    row = {"id": qid, "question": q, "guide_scopes": scopes, "served": d.get("served"), "specialist": d.get("routing_final"),
           "same_answer_as_evaluation": res.answer == expected_answer, "items": [], "problems": []}
    if d.get("served") != "plan_f":
        row["problems"].append("not served in the re-run")
        return row, None
    guide_records = []
    for item in res.items:
        meta = item.get("meta") or {}
        idx, quote, page, sid = item.get("idx"), item.get("chunk") or "", meta.get("page"), meta.get("source_id")
        chunk_meta = kb.metadata[idx] if isinstance(idx, int) and 0 <= idx < len(kb.metadata) else {}
        rec = {"source_id": sid, "page": page, "chunk_id": idx,
               "quote_in_chunk": isinstance(idx, int) and quote in kb.chunks[idx],
               "quote_on_page_strict": bool(quote) and norm(quote) in norm(PAGE_TEXT.get((sid, page), "")),
               "quote_on_page": words_on_page(quote, PAGE_TEXT.get((sid, page), ""))
                                or step_on_page(quote, chunk_meta, PAGE_TEXT.get((sid, page), "")),
               "page_is_chunk_page": chunk_meta.get("page") == page, "source_is_chunk_source": chunk_meta.get("source_id") == sid,
               "source_enabled": sid in TITLES, "title_is_registry_title": meta.get("source_title") in (TITLES.get(sid), None)}
        guide = (meta.get("plan_f") or {}).get("guide")
        if guide:
            rec["citation_heading"] = meta.get("section")
            rec["heading_recomputed"] = GG.citation_heading(guide, index)
            rec["heading_matches"] = rec["citation_heading"] == rec["heading_recomputed"]
            guide_records.append((item, guide))
        row["items"].append(rec)
        for k in ("quote_in_chunk", "quote_on_page", "page_is_chunk_page", "source_is_chunk_source", "source_enabled", "title_is_registry_title"):
            if not rec[k]:
                row["problems"].append("%s: %s" % (k, idx))
        if guide and not rec["heading_matches"]:
            row["problems"].append("citation heading differs from the recomputed one")
    proc = None
    if guide_records:
        first = guide_records[0][1]
        specialist = (d.get("routing_final") or [None])[0]
        policy = index.policy.get(str(specialist), frozenset())
        clause = (guide_records[0][0].get("meta") or {}).get("plan_f", {}).get("focus_clause") or q
        clause, _ = correct_system_names(clause, words.__contains__)
        by_clause = {}
        for item, record in guide_records:
            by_clause.setdefault((item.get("meta") or {}).get("plan_f", {}).get("clause_index"), []).append((item, record))
        verifications = []
        for ci, group in by_clause.items():
            gclause = (group[0][0].get("meta") or {}).get("plan_f", {}).get("focus_clause") or clause
            gclause, _ = correct_system_names(gclause, words.__contains__)
            rec0 = group[0][1]
            parts = [(it.get("idx"), it.get("chunk"), (it.get("meta") or {}).get("page")) for it, _r in group]
            verdict = GG.verify_answer(gclause, rec0.get("form"), parts, rec0.get("procedure_key"), index, specialist_sources=policy)
            approved = all((it.get("meta") or {}).get("source_id") in policy for it, _r in group)
            verifications.append({"clause_index": ci, "form": rec0.get("form"), "procedure_key": rec0.get("procedure_key"),
                                  "parts": len(group), "verify_answer": verdict, "specialist_approved": approved})
            if verdict is not None:
                row["problems"].append("verify_answer: %s" % verdict)
            if not approved:
                row["problems"].append("specialist not approved for the cited guide")
        row["guide_verification"] = verifications
        proc = {"id": qid, "question": q, "verifications": verifications,
                "pages": sorted({(it.get("meta") or {}).get("page") for it, _r in guide_records}),
                "source": sorted({(it.get("meta") or {}).get("source_id") for it, _r in guide_records}),
                "first_unit": (guide_records[0][0].get("chunk") or "")[:160], "last_unit": (guide_records[-1][0].get("chunk") or "")[-160:]}
        if first.get("form") in (GG.FORM_PROCEDURE, GG.FORM_SECTION):
            key = first.get("procedure_key")
            if first.get("form") == GG.FORM_PROCEDURE:
                p = index.procedure(key)
                proc["index_steps"] = len(p.steps) if p else None
            else:
                s = index.structure(key)
                proc["index_units"] = len(s.parts) if s else None
            proc["complete"] = all(v["verify_answer"] is None for v in verifications)
    chips = presentation.source_chips(res)
    doc_chips = [label for cls, label in chips if cls == "doc"]
    page_chips = [label for cls, label in chips if cls == "page"]
    chip_pages = sorted({int(m.group(1)) for label in page_chips for m in [re.match(r"Pages? (\d+)", label)] if m})
    row["chips"] = chips
    row["chip_problems"] = []
    if any("uos_" in label for _c, label in chips):
        row["chip_problems"].append("raw source id in a chip")
    cited_titles = {TITLES.get((i.get("meta") or {}).get("source_id")) for i in res.items}
    if any(not any(label.rstrip("…") and t and t.startswith(label.rstrip("…").rstrip(".")) for t in cited_titles) for label in doc_chips):
        row["chip_problems"].append("a document chip names no cited source")
    if chip_pages and not set(chip_pages) <= set(res.pages):
        row["chip_problems"].append("a page chip cites a page the answer does not cite")
    row["problems"].extend(row["chip_problems"])
    row["pass"] = not row["problems"]
    return row, proc


rows, procs = [], []
active = {r["id"]: r for r in R["results"] if r["mode"] == "active"}
for qid, rec in active.items():
    if (rec.get("diagnostics") or {}).get("served") == "plan_f":
        row, proc = check(qid, rec["question"], rec.get("answer"), True)
        rows.append(dict(row, set=next((q["set"] for q in R["questions"] if q["id"] == qid), None)))
        if proc:
            procs.append(proc)
        print(qid, row["pass"], row["problems"][:2])
frozen_active = [r for r in FROZEN["results"] if r.get("mode") == "active" and (r.get("diagnostics") or {}).get("served") == "plan_f"]
for rec in frozen_active[::5]:                                                   # every fifth of the 55 handbook answers
    row, _ = check(rec["id"], rec.get("question") or rec.get("q"), rec.get("answer"), False)
    rows.append(dict(row, set="frozen192"))
    print(rec["id"], row["pass"], row["problems"][:2])

coverage = {}
for row in rows:
    for item in row["items"]:
        coverage.setdefault(item["source_id"], set()).add(row["id"])
summary = {"answers_checked": len(rows), "items_checked": sum(len(r["items"]) for r in rows), "passed": sum(r["pass"] for r in rows),
           "failed": [(r["id"], r["problems"]) for r in rows if not r["pass"]],
           "not_identical_to_evaluation": [r["id"] for r in rows if not r["same_answer_as_evaluation"]],
           "sources_covered": {k: len(v) for k, v in sorted(coverage.items())},
           "sources_not_covered": sorted(set(TITLES) - set(coverage))}
json.dump({"method": __doc__.strip(), "summary": summary, "answers": rows}, open(os.path.join(OUT, "citation_review.json"), "w",
          encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False, default=str)
psum = {"guide_answers": len(procs), "procedures_and_sections": sum(1 for p in procs if "complete" in p),
        "complete": sum(1 for p in procs if p.get("complete")), "incomplete_served": [p["id"] for p in procs if p.get("complete") is False]}
json.dump({"method": "Every served guide answer re-verified: one approved source, every asked term stated, the complete procedure or section "
                     "in printed order (guide_grounding.verify_answer), pages and first and last units recorded.", "summary": psum,
           "answers": procs}, open(os.path.join(OUT, "procedure_final.json"), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
print(json.dumps({"citation": summary, "procedure": psum}, default=str))
