"""Merge the measured scan (inventory_scan.py) with the manual classification (classification.json) into the PF-1 master
document inventory: document_inventory.json, document_inventory.csv and the count summary printed at the end.

    python -X utf8 qa/post_freeze_pf1/build_inventory.py <scan json> <classification json> <out dir>

No absolute path is written; the scan records only the base name of the source folder.
"""
import csv
import json
import os
import sys
from collections import Counter

SCAN, CLASS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
scan = json.load(open(SCAN, encoding="utf-8"))
klass = {k: v for k, v in json.load(open(CLASS, encoding="utf-8")).items() if not k.startswith("_")}
HANDBOOK_PAGES = 272                                            # verified against cache/<handbook>_v12_pages.json
FIELDS = ["doc_id", "filename", "title", "format", "bytes", "pages", "page_basis", "unit", "system", "primary_topic", "secondary_topics", "audience",
          "language", "arabic_share", "date", "version", "officiality", "relevance", "extraction", "extraction_notes", "screenshots", "sparse_pages",
          "image_heavy_pages", "images_total", "words", "numbered_steps", "urls", "emails", "phones", "duplicate", "primary_specialist",
          "secondary_specialists", "cross_domain", "handbook_overlap", "status", "priority", "notes", "sha256"]

rows = []
by_name = {r["filename"]: r for r in scan["records"]}
for name, c in klass.items():
    r = by_name.get(name)
    if r is None:
        raise SystemExit("classified file not in the scan: %s" % name)
    f = r.get("features") or {}
    if r.get("format") == "pdf":
        pages, basis = r.get("pages"), "PDF page count"
    elif r.get("declared_pages"):
        pages, basis = r.get("declared_pages"), "DOCX declared page property (estimate by words: %s)" % r.get("estimated_pages_by_words")
    else:
        pages, basis = r.get("estimated_pages_by_words"), "estimate by words (350 per page)"
    rows.append({"doc_id": c["doc_id"], "filename": name, "title": c["title"], "format": r.get("format"), "bytes": r["bytes"], "pages": pages, "page_basis": basis,
                 "unit": c["unit"], "system": c["system"], "primary_topic": c["primary_topic"], "secondary_topics": "; ".join(c["secondary_topics"]),
                 "audience": c["audience"], "language": c["language"], "arabic_share": f.get("arabic_share"), "date": c["date"], "version": c["version"],
                 "officiality": c["officiality"], "relevance": c["relevance"], "extraction": c["extraction"], "extraction_notes": c["extraction_notes"],
                 "screenshots": c["screenshots"], "sparse_pages": r.get("sparse_pages"), "image_heavy_pages": r.get("image_heavy_pages"),
                 "images_total": r.get("images_total"), "words": f.get("words"), "numbered_steps": f.get("numbered_steps"), "urls": f.get("urls"),
                 "emails": f.get("emails"), "phones": f.get("phones"), "duplicate": c["duplicate"], "primary_specialist": c["primary_specialist"],
                 "secondary_specialists": "; ".join(c["secondary_specialists"]), "cross_domain": bool(c["secondary_specialists"]),
                 "handbook_overlap": c["handbook_overlap"], "status": c["status"], "priority": c["priority"], "notes": c["notes"], "sha256": r["sha256"]})
unclassified = [n for n in by_name if n not in klass]

approved = [x for x in rows if x["status"] == "APPROVED CANDIDATE"]
held = [x for x in rows if x["status"] == "HOLD FOR REVIEW"]
rejected = [x for x in rows if x["status"].startswith("REJECT")]
pdf_pages = sum(x["pages"] or 0 for x in rows if x["format"] == "pdf")
docx_pages = sum(x["pages"] or 0 for x in rows if x["format"] == "docx")
approved_pages = sum(x["pages"] or 0 for x in approved)
summary = {
    "source_folder": scan["source_folder"], "files_in_folder": scan["files"] + len(unclassified) if False else scan["files"],
    "candidate_documents": len(rows), "unclassified_or_container_files": unclassified,
    "by_format": dict(Counter(x["format"] for x in rows)),
    "approved": len(approved), "hold": len(held), "rejected_or_duplicate": len(rejected),
    "pdf_pages_all_candidates": pdf_pages, "docx_declared_pages_all_candidates": docx_pages,
    "approved_pages": approved_pages, "handbook_pages": HANDBOOK_PAGES,
    "potential_corpus_pages_if_all_approved_ingested": HANDBOOK_PAGES + approved_pages,
    "potential_corpus_pages_if_every_candidate_ingested": HANDBOOK_PAGES + pdf_pages + docx_pages,
    "by_priority": dict(Counter(x["priority"] for x in rows)),
    "by_extraction": dict(Counter(x["extraction"] for x in rows)),
    "by_language": dict(Counter(x["language"].split(" ")[0] for x in rows)),
    "by_primary_specialist": dict(Counter(x["primary_specialist"] for x in rows)),
    "multi_domain": [x["doc_id"] for x in rows if x["cross_domain"]],
    "duplicates": [x["doc_id"] for x in rows if x["duplicate"].startswith("DUPLICATE")],
    "exact_duplicate_groups": scan["exact_duplicate_groups"],
    "approved_ids": [x["doc_id"] for x in approved], "hold_ids": [x["doc_id"] for x in held], "rejected_ids": [x["doc_id"] for x in rejected],
    "p0": [x["doc_id"] for x in rows if x["priority"] == "P0"], "p1": [x["doc_id"] for x in rows if x["priority"] == "P1"],
    "p2": [x["doc_id"] for x in rows if x["priority"] == "P2"],
}
os.makedirs(OUT, exist_ok=True)
json.dump({"summary": summary, "documents": rows}, open(os.path.join(OUT, "document_inventory.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
with open(os.path.join(OUT, "document_inventory.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for x in rows:
        w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in x.items()})
print(json.dumps(summary, indent=1, ensure_ascii=False))
