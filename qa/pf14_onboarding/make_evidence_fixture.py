"""Freeze the corpus chunks the onboarding catalog quotes into tests/fixtures/pf14_evidence.json (post-freeze PF-14).

The fixture lets the provenance tests run without the 15,000-chunk cache; when the cache is present the tests also
re-check every quote against it. Run again only when the corpus or the catalog changes:

    python -X utf8 qa/pf14_onboarding/make_evidence_fixture.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from onboarding import catalog as C                                                # noqa: E402

CACHE = "cache/corpus_c8987b6079b11f39_v13"
chunks = json.load(open(CACHE + "_chunks.json", encoding="utf-8"))
meta = json.load(open(CACHE + "_meta.json", encoding="utf-8"))
by_page = {}
for i, m in enumerate(meta):
    by_page.setdefault((m["source_id"], m["page"]), []).append(i)


def find(src, page, quote):
    for i in by_page.get((src, page), []):
        if quote in chunks[i]:
            return i
    return None


needed = {}
missing = []
for t in C.TASKS.values():
    for e in t.evidence:
        cid = find(e.source_id, e.page, e.quote)
        (needed.__setitem__(cid, None) if cid is not None else missing.append(("task", t.id, e.quote)))
for p in C.PROCEDURES.values():
    for s in p.steps:
        if s.text in chunks[s.chunk_id] and meta[s.chunk_id]["source_id"] == p.source_id and meta[s.chunk_id]["page"] == s.page:
            needed[s.chunk_id] = None
        else:
            missing.append(("procedure", p.id, s.text))
for r in C.RESOURCES.values():
    for e in list(r.evidence) + [c.evidence for c in r.contacts]:
        cid = find(e.source_id, e.page, e.quote)
        (needed.__setitem__(cid, None) if cid is not None else missing.append(("resource", r.id, e.quote)))
if missing:
    raise SystemExit("quotes not found in the corpus: %r" % missing)
fixture = {"corpus_key": "c8987b6079b11f39", "catalog_version": C.CATALOG_VERSION,
           "chunks": {str(cid): {"source_id": meta[cid]["source_id"], "page": meta[cid]["page"], "section_title": meta[cid].get("section_title"),
                                 "chunk_type": meta[cid].get("chunk_type"), "text": chunks[cid]} for cid in sorted(needed)}}
out = os.path.join(ROOT, "tests", "fixtures", "pf14_evidence.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(fixture, f, ensure_ascii=False, indent=1)
print("wrote", out, "chunks:", len(needed))
