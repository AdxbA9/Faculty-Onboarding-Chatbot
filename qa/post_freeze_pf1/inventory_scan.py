"""PF-1 inventory and extraction scan of the candidate official guides (post-freeze knowledge expansion, phase PF-1).

For every PDF and DOCX file in the source folder the scan records what can be MEASURED: size, SHA-256, page count (PDF
pages; the page count a DOCX declares in its own properties, plus a word-based estimate), document properties (title,
author, creation and modification dates as stored in the file), per-page text volume, embedded image counts, pages with
almost no text (screenshot-heavy or scanned), Arabic-script share, replacement characters, URLs, e-mail addresses, phone
numbers, numbered-step lines, menu-path arrows, and a short text sample from the first content page. PDF text comes from
PyMuPDF exactly as the production loader reads it; DOCX text comes from the document XML with the standard library (no
new dependency). Nothing is OCR'd and nothing is written anywhere but the output JSON. The source folder is given on the
command line and is recorded only by its base name, never by its absolute path.

    python -X utf8 qa/post_freeze_pf1/inventory_scan.py <source folder> <output json>
"""
import hashlib
import json
import os
import re
import sys
import zipfile
from xml.etree import ElementTree as ET

import fitz  # PyMuPDF, the production PDF reader

SRC, OUT = sys.argv[1], sys.argv[2]
ARABIC = re.compile(r"[؀-ۿ]")
URL = re.compile(r"https?://[^\s)>\]]+|www\.[^\s)>\]]+", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?:\+971|0[0-9])[\s\-]?\d[\d\s\-]{6,}")
STEP = re.compile(r"^\s*(?:step\s*)?\d{1,2}[.)]\s+\S", re.I | re.M)
MENU = re.compile(r"\s(?:>|→|>>|›)\s")
W = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_features(text):
    letters = [c for c in text if c.isalpha()]
    return {"chars": len(text), "words": len(text.split()), "arabic_share": round(sum(1 for c in letters if ARABIC.match(c)) / max(1, len(letters)), 3),
            "replacement_chars": text.count("�"), "urls": len(URL.findall(text)), "emails": len(set(EMAIL.findall(text))),
            "phones": len(PHONE.findall(text)), "numbered_steps": len(STEP.findall(text)), "menu_paths": len(MENU.findall(text))}


def scan_pdf(path):
    doc = fitz.open(path)
    pages = []
    all_text = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        words = page.get_text("words") or []
        images = page.get_images(full=True)
        big = 0
        for img in images:
            try:
                info = doc.extract_image(img[0])
                if info and info.get("width", 0) * info.get("height", 0) >= 40000:
                    big += 1
            except Exception:
                pass
        pages.append({"page": i + 1, "chars": len(text.strip()), "words": len(words), "images": len(images), "large_images": big})
        all_text.append(text)
    meta = doc.metadata or {}
    doc.close()
    text = "\n".join(all_text)
    sparse = [p["page"] for p in pages if p["chars"] < 80]
    image_heavy = [p["page"] for p in pages if p["large_images"] >= 1 and p["chars"] < 400]
    sample_page = next((p for p in pages if p["chars"] >= 200), pages[0] if pages else None)
    sample = ""
    if sample_page:
        sample = " ".join(all_text[sample_page["page"] - 1].split())[:420]
    return {"format": "pdf", "pages": len(pages), "declared_pages": None,
            "properties": {k: (meta.get(k) or "") for k in ("title", "author", "subject", "creator", "producer", "creationDate", "modDate")},
            "per_page": pages, "text_pages": sum(1 for p in pages if p["chars"] >= 80), "sparse_pages": sparse, "image_heavy_pages": image_heavy,
            "images_total": sum(p["images"] for p in pages), "large_images_total": sum(p["large_images"] for p in pages),
            "features": text_features(text), "sample": sample, "sample_page": sample_page["page"] if sample_page else None}


def scan_docx(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        xml = z.read("word/document.xml")
        core = z.read("docProps/core.xml") if "docProps/core.xml" in names else b""
        app = z.read("docProps/app.xml") if "docProps/app.xml" in names else b""
        media = [n for n in names if n.startswith("word/media/")]
    root = ET.fromstring(xml)
    paragraphs = []
    for p in root.iter("{%s}p" % W["w"]):
        t = "".join(x.text or "" for x in p.iter("{%s}t" % W["w"]))
        if t.strip():
            paragraphs.append(t.strip())
    tables = len(list(root.iter("{%s}tbl" % W["w"])))
    text = "\n".join(paragraphs)
    props = {}
    for blob, tags in ((core, ("title", "creator", "lastModifiedBy", "created", "modified", "revision")), (app, ("Pages", "Words", "Application"))):
        if blob:
            try:
                r = ET.fromstring(blob)
                for el in r.iter():
                    tag = el.tag.split("}")[-1]
                    if tag in tags:
                        props[tag] = (el.text or "").strip()
            except ET.ParseError:
                pass
    declared = int(props["Pages"]) if props.get("Pages", "").isdigit() else None
    words = len(text.split())
    return {"format": "docx", "pages": declared, "declared_pages": declared, "estimated_pages_by_words": max(1, round(words / 350)),
            "properties": props, "paragraphs": len(paragraphs), "tables": tables, "images_total": len(media), "large_images_total": None,
            "text_pages": None, "sparse_pages": [], "image_heavy_pages": [], "features": text_features(text),
            "sample": " ".join(text.split())[:420], "sample_page": 1}


records = []
for name in sorted(os.listdir(SRC)):
    path = os.path.join(SRC, name)
    if not os.path.isfile(path):
        continue
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    rec = {"filename": name, "extension": ext, "bytes": os.path.getsize(path), "sha256": sha256(path)}
    try:
        if ext == "pdf":
            rec.update(scan_pdf(path))
        elif ext == "docx":
            rec.update(scan_docx(path))
        else:
            rec.update({"format": ext, "pages": None, "note": "container or unsupported type: not extracted"})
    except Exception as exc:
        rec.update({"format": ext, "pages": None, "error": type(exc).__name__})
    records.append(rec)

# exact duplicates by content hash
by_hash = {}
for r in records:
    by_hash.setdefault(r["sha256"], []).append(r["filename"])
json.dump({"source_folder": os.path.basename(os.path.normpath(SRC)), "files": len(records), "records": records,
           "exact_duplicate_groups": [v for v in by_hash.values() if len(v) > 1]},
          open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("scanned %d files -> %s" % (len(records), OUT))
for r in records:
    f = r.get("features") or {}
    print("%-52s %-5s %9d B  pages=%-4s text_pages=%-4s sparse=%-3s imgheavy=%-3s imgs=%-4s ar=%-5s steps=%-3s urls=%-3s emails=%-2s phones=%-2s err=%s" % (
        r["filename"][:52], r["extension"], r["bytes"], r.get("pages"), r.get("text_pages"), len(r.get("sparse_pages") or []), len(r.get("image_heavy_pages") or []),
        r.get("images_total"), f.get("arabic_share"), f.get("numbered_steps"), f.get("urls"), f.get("emails"), f.get("phones"), r.get("error") or r.get("note", "")[:20]))
