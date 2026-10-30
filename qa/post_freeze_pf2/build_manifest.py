"""Write ``pilot_source_manifest.json``: every P0 pilot document of PF-1 with its ingestion decision, identity and what the
loader indexed from it. Repository-relative paths only; the held document is listed without a file.

    python -X utf8 qa/post_freeze_pf2/build_manifest.py <repo>
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
REPO = os.path.abspath(sys.argv[1])
os.chdir(REPO)
sys.path.insert(0, REPO)

from handbook_bot.knowledge_base import assemble_corpus          # noqa: E402
from handbook_bot.sources import load_source_registry             # noqa: E402

PF1 = {
    "uos_faculty_handbook_2025_26": {"pf1_doc_id": None, "pf1_priority": None, "original_file": "UOS Faculty Handbook 25-26.pdf"},
    "uos_blackboard_faculty_short_guide": {"pf1_doc_id": "guide_bb_faculty_short", "pf1_priority": "P0", "original_file": "Blackboard_Faculty_Short_Guide_En.pdf"},
    "uos_grade_transfer_dashboard_guide": {"pf1_doc_id": "guide_grade_transfer_dashboard", "pf1_priority": "P0", "original_file": "Grade_Transfer_Guide.pdf"},
    "uos_blackboard_gradebook_guide": {"pf1_doc_id": "guide_bb_gradebook", "pf1_priority": "P0", "original_file": "Bb_gradebook_En.pdf"},
    "uos_ithenticate_guidelines": {"pf1_doc_id": "guide_ithenticate", "pf1_priority": "P0", "original_file": "IThenticate_Guidelines_en_ar.pdf"},
    "uos_performance_appraisal_phase2_guide": {"pf1_doc_id": "guide_hr_appraisal_phase2", "pf1_priority": "P0", "original_file": "Performance_Appraisal_Phase_2_User_Guide.pdf"},
}
DECISIONS = {
    "uos_faculty_handbook_2025_26": "PRIMARY SOURCE (unchanged): loaded first, chunked exactly as at the I-9 freeze",
    "uos_blackboard_faculty_short_guide": "INGESTED: approved official ITC guide, English only",
    "uos_grade_transfer_dashboard_guide": "INGESTED: approved official ITC guide; Arabic pages 4-6 excluded by the english_only page filter",
    "uos_blackboard_gradebook_guide": "INGESTED: approved official ITC guide, English only",
    "uos_ithenticate_guidelines": "INGESTED WITH A REGISTERED CONFLICT: English page 1 indexed, Arabic page 2 excluded; the yearly submission limit (EN 30 / AR 15) is stamped source_conflict and blocked from direct Plan F service",
    "uos_performance_appraisal_phase2_guide": "INGESTION HOLD - INTERNAL USE REVIEW REQUIRED: every page carries the running header 'Internal Use'; not copied into the repository, not loaded, not served; the pilot proceeds with four documents",
}

registry = load_source_registry()
parts = assemble_corpus(registry.enabled_sources())
indexed = {s["source_id"]: s for s in parts.per_source}
documents = []
for record in registry:
    entry = {
        "source_id": record.source_id, "title": record.title, "citation_label": record.label, "document_type": record.document_type,
        "authority_level": record.authority_level, "department": record.department, "system": record.system,
        "publication_date": record.publication_date, "version": record.version, "language": record.language, "access": record.access,
        "enabled": record.enabled, "verification_status": record.verification_status, "relative_path": record.relative_path,
        "file_present": record.exists(), "bytes": os.path.getsize(record.absolute_path()) if record.exists() else None, "sha256": record.sha256 or None,
        "page_filter": record.page_filter, "domains": list(record.domains), "conflicts": [c.to_dict() for c in record.conflicts],
        "decision": DECISIONS[record.source_id], **PF1[record.source_id],
    }
    if record.source_id in indexed:
        s = indexed[record.source_id]
        entry["indexed"] = {k: s[k] for k in ("pages_loaded", "pages_indexed", "language_filter", "procedures", "chunks", "chunk_types",
                                               "chunk_id_range", "conflict_chunks", "section_records")}
    else:
        entry["indexed"] = None
    documents.append(entry)

manifest = {
    "phase": "PF-2", "generated_from": "knowledge/sources.json and the loader (assemble_corpus)", "corpus_key": parts.key,
    "cache_format": "v13", "total_chunks": len(parts.chunks), "pages_indexed_total": sum(len(p) for p in parts.source_pages.values()),
    "source_folder_note": "the original files were taken from the PF-1 inventory folder outside the repository (not recorded here); the four ingested copies live under data/guides/ with stable file names",
    "documents": documents,
}
out = os.path.join(REPO, "qa", "post_freeze_pf2", "pilot_source_manifest.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=1, ensure_ascii=False)
print("wrote", os.path.relpath(out, REPO), "documents", len(documents), "chunks", len(parts.chunks))
