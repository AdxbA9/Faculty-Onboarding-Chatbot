import os
import re
import json
import math
import difflib
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# ==========================================================
# SETTINGS
# ==========================================================
PDF_PATH = None  # None = auto-detect first PDF in current folder

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

PARA_CHUNK_SIZE_WORDS = 160
PARA_CHUNK_OVERLAP_WORDS = 50
ROW_WINDOW_SIZE = 4
ROW_WINDOW_STEP = 2
TOP_K_DENSE = 36
TOP_K_LEXICAL = 18
RERANK_CANDIDATES = 24
FINAL_K = 5
MIN_RERANK_SCORE = -1.4

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

PHONE_PATTERN = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
MONTH_PATTERN = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b", re.I)
DAY_PATTERN = re.compile(r"\b(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
YEAR_TERM_PATTERN = re.compile(r"\b(fall|spring|summer)\b", re.I)
COUNT_PATTERN = re.compile(r"\b(how many|number of|total number|count of|total)\b", re.I)
DATE_QUESTION_PATTERN = re.compile(r"\b(when|date|day|begin|starts?|begins?|held|deadline|last day|add/drop|final exams?)\b", re.I)
CONTACT_QUESTION_PATTERN = re.compile(r"\b(phone|telephone|fax|contact|email|number)\b", re.I)
TABLE_HINT_PATTERN = re.compile(r"\b(phone|telephone|fax|contact|email|office|number|date|when|calendar|schedule|committee|list|programs?)\b", re.I)
YESNO_PATTERN = re.compile(r"^(does|do|did|can|is|are|was|were)\b", re.I)
GREETING_PATTERN = re.compile(r"^(hi|hello|hey)\b", re.I)

STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "for", "to", "in", "on", "at", "and",
    "or", "by", "with", "what", "which", "who", "when", "where", "how", "many",
    "does", "do", "did", "can", "could", "should", "would", "be", "it", "this",
    "that", "from", "as", "about", "into", "under", "their", "them", "they",
    "university", "sharjah", "uos", "faculty", "handbook", "please", "tell", "me",
}

# ==========================================================
# HELPERS
# ==========================================================

def find_pdf_file() -> str:
    if PDF_PATH:
        if os.path.exists(PDF_PATH):
            return PDF_PATH
        raise FileNotFoundError(f"File '{PDF_PATH}' not found.")

    pdf_files = [f for f in os.listdir() if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError("No PDF file was found in the current folder.")
    pdf_files.sort()
    print(f"Detected PDF file: {pdf_files[0]}")
    return pdf_files[0]


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = re.sub(r"\s+", " ", line).strip(" |")
    return line


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9+&\-/']+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def detect_heading(line: str) -> bool:
    if not line or len(line) > 115 or PHONE_PATTERN.search(line):
        return False
    letters = re.sub(r"[^A-Za-z]", "", line)
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
    numbered = bool(re.match(r"^\d+(?:\.\d+)*\s+", line))
    title_like = line.istitle() and len(line.split()) <= 11
    return numbered or upper_ratio > 0.72 or title_like


def row_has_table_shape(row: str) -> bool:
    pipes = row.count(" | ")
    return pipes >= 2 or bool(PHONE_PATTERN.search(row)) or bool(EMAIL_PATTERN.search(row))


def parse_page_ranges(text: str) -> str:
    nums = sorted({int(x) for x in re.findall(r"\d+", text)})
    return ", ".join(f"Page {n}" for n in nums)


def classify_query(question: str) -> str:
    q = question.strip().lower()
    if GREETING_PATTERN.search(q):
        return "greeting"
    if CONTACT_QUESTION_PATTERN.search(q):
        return "contact"
    if COUNT_PATTERN.search(q):
        return "count"
    if DATE_QUESTION_PATTERN.search(q) or YEAR_TERM_PATTERN.search(q):
        return "date"
    if re.search(r"\b(name|list|which are|what are the|standing committee|categories|core values)\b", q):
        return "list"
    if YESNO_PATTERN.search(q):
        return "policy_yesno"
    return "policy"


def extract_rows_from_pymupdf_page(page) -> List[str]:
    words = page.get_text("words") or []
    if not words:
        return []
    # words tuple: x0,y0,x1,y1,text,block,line,word
    words = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
    rows: List[List[Tuple[float, str]]] = []
    current: List[Tuple[float, str]] = []
    current_y: Optional[float] = None
    tolerance = 2.8
    for w in words:
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], str(w[4])
        if current_y is None or abs(y0 - current_y) <= tolerance:
            current.append((x0, txt))
            current_y = y0 if current_y is None else (current_y + y0) / 2
        else:
            rows.append(current)
            current = [(x0, txt)]
            current_y = y0
    if current:
        rows.append(current)

    out = []
    for row in rows:
        row.sort(key=lambda z: z[0])
        text = " ".join(t for _, t in row)
        text = normalize_line(text)
        if text:
            out.append(text)
    return out


def remove_common_headers_footers(raw_pages: List[Dict]) -> List[Dict]:
    line_counts: Dict[str, int] = {}
    row_counts: Dict[str, int] = {}
    page_lines: List[List[str]] = []
    page_rows: List[List[str]] = []

    for page in raw_pages:
        lines = [normalize_line(x) for x in page["raw_lines"] if normalize_line(x)]
        rows = [normalize_line(x) for x in page.get("raw_rows", []) if normalize_line(x)]
        page_lines.append(lines)
        page_rows.append(rows)
        for line in set(lines[:3] + lines[-3:]):
            if len(line) >= 4:
                line_counts[line] = line_counts.get(line, 0) + 1
        for row in set(rows[:3] + rows[-3:]):
            if len(row) >= 4:
                row_counts[row] = row_counts.get(row, 0) + 1

    threshold = max(4, int(len(raw_pages) * 0.45))
    repeated = {
        line for line, count in {**line_counts, **row_counts}.items()
        if count >= threshold and (len(line) > 70 or re.search(r"faculty handbook|university of sharjah|page\s+\d+", line, re.I))
    }

    cleaned_pages = []
    current_section = ""
    for page, lines, rows in zip(raw_pages, page_lines, page_rows):
        filtered_lines = [line for line in lines if line not in repeated]
        filtered_rows = [row for row in rows if row not in repeated]
        heading_source = filtered_lines[:8] if filtered_lines else filtered_rows[:8]
        for line in heading_source:
            if detect_heading(line):
                current_section = line
                break
        cleaned_pages.append({
            "page": page["page"],
            "lines": filtered_lines,
            "rows": filtered_rows,
            "text": normalize_text(" ".join(filtered_lines or filtered_rows)),
            "section_hint": current_section,
        })
    return [p for p in cleaned_pages if p["text"]]


def extract_text_from_pdf(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found.")

    raw_pages = []
    if fitz is not None:
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            txt = page.get_text("text") or ""
            rows = extract_rows_from_pymupdf_page(page)
            raw_pages.append({"page": i + 1, "raw_lines": txt.splitlines(), "raw_rows": rows})
        doc.close()
    elif PdfReader is not None:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            raw_pages.append({"page": i + 1, "raw_lines": txt.splitlines(), "raw_rows": txt.splitlines()})
    else:
        raise RuntimeError("No PDF extraction library is available.")

    return remove_common_headers_footers(raw_pages)


def build_chunks(pages: List[Dict]) -> Tuple[List[str], List[Dict]]:
    chunks: List[str] = []
    metadata: List[Dict] = []

    def add_chunk(text: str, meta: Dict):
        clean = normalize_text(text)
        if len(clean) < 12:
            return
        meta = dict(meta)
        meta["chunk_id"] = len(chunks)
        meta["text"] = clean
        chunks.append(clean)
        metadata.append(meta)

    for page in pages:
        page_num = page["page"]
        section_hint = page.get("section_hint", "")
        lines = page["lines"]
        rows = page.get("rows") or lines
        text = page["text"]

        # Paragraph chunks
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        bucket: List[str] = []
        bucket_words = 0
        para_id = 0
        for sent in sentences:
            words = sent.split()
            if not words:
                continue
            if bucket_words + len(words) > PARA_CHUNK_SIZE_WORDS and bucket:
                chunk_text = " ".join(bucket)
                add_chunk(chunk_text, {
                    "page": page_num,
                    "section": section_hint,
                    "chunk_type": "paragraph",
                    "chunk_id_on_page": para_id,
                })
                para_id += 1
                overlap = " ".join(chunk_text.split()[-PARA_CHUNK_OVERLAP_WORDS:]) if PARA_CHUNK_OVERLAP_WORDS else ""
                bucket = [overlap] if overlap else []
                bucket_words = len(overlap.split()) if overlap else 0
            bucket.append(sent)
            bucket_words += len(words)
        if bucket:
            add_chunk(" ".join(bucket), {
                "page": page_num,
                "section": section_hint,
                "chunk_type": "paragraph",
                "chunk_id_on_page": para_id,
            })

        # Exact row chunks
        for i, row in enumerate(rows):
            if len(row) < 6:
                continue
            add_chunk(row, {
                "page": page_num,
                "section": section_hint,
                "chunk_type": "row",
                "row_id": i,
                "table_like": row_has_table_shape(row),
            })

        # Row windows for tables/calendars/contacts
        for start in range(0, max(1, len(rows)), ROW_WINDOW_STEP):
            window = rows[start:start + ROW_WINDOW_SIZE]
            if not window:
                continue
            add_chunk(" | ".join(window), {
                "page": page_num,
                "section": section_hint,
                "chunk_type": "row_window",
                "row_start": start,
                "row_end": min(start + ROW_WINDOW_SIZE - 1, len(rows) - 1),
                "table_like": any(row_has_table_shape(r) for r in window),
            })

    return chunks, metadata


def cache_paths(pdf_file: str):
    base = os.path.splitext(os.path.basename(pdf_file))[0]
    return (
        os.path.join(CACHE_DIR, f"{base}_mcbv9_embeddings.npy"),
        os.path.join(CACHE_DIR, f"{base}_mcbv9_chunks.json"),
        os.path.join(CACHE_DIR, f"{base}_mcbv9_meta.json"),
        os.path.join(CACHE_DIR, f"{base}_mcbv9_pages.json"),
    )


def build_or_load_embeddings(pdf_file: str, chunks: List[str], metadata: List[Dict], pages: List[Dict], embedder):
    emb_path, chunks_path, meta_path, pages_path = cache_paths(pdf_file)
    if all(os.path.exists(p) for p in [emb_path, chunks_path, meta_path, pages_path]):
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                cached_chunks = json.load(f)
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            with open(pages_path, "r", encoding="utf-8") as f:
                cached_pages = json.load(f)
            if cached_chunks == chunks and cached_meta == metadata and cached_pages == pages:
                print("Loading cached embeddings...")
                return np.load(emb_path)
        except Exception:
            pass

    print("Creating embeddings...")
    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(emb_path, embeddings)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(pages_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    return embeddings


def lexical_score(query: str, text: str) -> float:
    q = tokenize(query)
    if not q:
        return 0.0
    t = tokenize(text)
    if not t:
        return 0.0
    tset = set(t)
    overlap = sum(1 for token in q if token in tset)
    phrase_bonus = 0.0
    qnorm = normalize_text(query).lower()
    tnorm = normalize_text(text).lower()
    if len(qnorm) > 8 and qnorm in tnorm:
        phrase_bonus += 2.4
    elif len(q) >= 2 and sum(1 for token in q[:4] if token in tset) >= max(2, min(4, len(q)) - 1):
        phrase_bonus += 0.9
    return overlap / math.sqrt(len(tset) + 1) + phrase_bonus


def candidate_route_boost(query_type: str, question: str, meta: Dict, chunk: str) -> float:
    boost = 0.0
    ctype = meta.get("chunk_type")
    if query_type == "contact":
        if ctype == "row":
            boost += 0.9
        elif ctype == "row_window":
            boost += 0.55
        if PHONE_PATTERN.search(chunk) or EMAIL_PATTERN.search(chunk):
            boost += 0.55
    elif query_type == "date":
        if ctype in {"row", "row_window"}:
            boost += 0.7
        if MONTH_PATTERN.search(chunk) or DAY_PATTERN.search(chunk) or re.search(r"\b\d{1,2}\s+[A-Z][a-z]{2,}\b", chunk):
            boost += 0.35
    elif query_type == "count":
        if re.search(r"\b\d{1,4}(?:,\d{3})?\b", chunk):
            boost += 0.35
        if ctype == "paragraph":
            boost += 0.25
    elif query_type == "list":
        if ctype in {"row_window", "paragraph"}:
            boost += 0.25
        if "•" in chunk or " | " in chunk:
            boost += 0.25
    elif query_type == "policy_yesno":
        if ctype == "paragraph":
            boost += 0.4
    else:
        if ctype == "paragraph":
            boost += 0.2
    if TABLE_HINT_PATTERN.search(question) and meta.get("table_like"):
        boost += 0.2
    return boost


def gather_candidates(question: str, query_type: str, query_embedding: np.ndarray, index, chunks: List[str], metadata: List[Dict]) -> List[Dict]:
    D, I = index.search(query_embedding, TOP_K_DENSE)
    candidates: Dict[int, Dict] = {}

    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        candidates[idx] = {
            "idx": idx,
            "chunk": chunks[idx],
            "meta": metadata[idx],
            "dense_score": float(score),
            "lexical_score": 0.0,
        }

    lexical_ranked = sorted(
        ((i, lexical_score(question, text)) for i, text in enumerate(chunks)),
        key=lambda x: x[1],
        reverse=True,
    )[:TOP_K_LEXICAL]

    for idx, lex_score in lexical_ranked:
        if lex_score <= 0:
            continue
        if idx not in candidates:
            candidates[idx] = {
                "idx": idx,
                "chunk": chunks[idx],
                "meta": metadata[idx],
                "dense_score": 0.0,
                "lexical_score": float(lex_score),
            }
        else:
            candidates[idx]["lexical_score"] = float(lex_score)

    merged = list(candidates.values())
    for cand in merged:
        cand["routing_boost"] = candidate_route_boost(query_type, question, cand["meta"], cand["chunk"])
    merged.sort(key=lambda x: (x["dense_score"] + 0.28 * x["lexical_score"] + x["routing_boost"]), reverse=True)
    return merged[:RERANK_CANDIDATES]


def deduplicate_by_text(items: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for item in items:
        text = item["chunk"]
        if text not in seen:
            seen.add(text)
            out.append(item)
    return out


def office_query_target(question: str) -> str:
    q = question.strip().rstrip("?")
    m = re.search(r"(?:for|of)\s+(.+)$", q, re.I)
    target = m.group(1).strip() if m else q
    target = re.sub(r"\b(phone|telephone|fax|contact|number|email|what|is|the)\b", " ", target, flags=re.I)
    return normalize_text(target)


def score_row_target(target: str, row: str) -> float:
    target_tokens = tokenize(target)
    row_tokens = set(tokenize(row))
    if not target_tokens:
        return 0.0
    overlap = sum(1 for t in target_tokens if t in row_tokens)
    seq = difflib.SequenceMatcher(None, target.lower(), row.lower()).ratio()
    return overlap + seq


def extract_contact_answer(question: str, items: List[Dict]) -> Optional[Tuple[str, List[int], str]]:
    if not CONTACT_QUESTION_PATTERN.search(question):
        return None
    target = office_query_target(question)
    want_fax = "fax" in question.lower()
    want_email = "email" in question.lower()

    row_items = [item for item in items if item["meta"].get("chunk_type") in {"row", "row_window"}]
    if not row_items:
        row_items = items
    best = None
    best_score = -1.0
    for item in row_items:
        score = score_row_target(target, item["chunk"])
        if score > best_score:
            best = item
            best_score = score
    if not best or best_score < 0.35:
        return None

    text = best["chunk"]
    pages = [best["meta"]["page"]]
    if want_email:
        emails = EMAIL_PATTERN.findall(text)
        if emails:
            return (f"The email address is {emails[0]}.", pages, text)
        return None

    numbers = [normalize_text(x) for x in PHONE_PATTERN.findall(text)]
    if not numbers:
        return None

    if want_fax:
        if len(numbers) >= 2:
            return (f"The fax number is {numbers[-1]}.", pages, text)
        return (f"The fax number is {numbers[0]}.", pages, text)

    # Assume common table order Office | Telephone | Fax.
    return (f"The phone number is {numbers[0]}.", pages, text)


def extract_count_answer(question: str, items: List[Dict], pages: List[Dict]) -> Optional[Tuple[str, List[int], str]]:
    if not COUNT_PATTERN.search(question):
        return None
    combined = " \n ".join([x["chunk"] for x in items] + [p["text"] for p in pages[: min(45, len(pages))]])
    patterns = {
        "total": r"(?:total of|total number of|offers? a total of|offer a total of|we offer a total of)\s+(\d{1,4}(?:,\d{3})?)\s+(?:academic\s+)?(?:degree\s+)?programs",
        "bachelor": r"(\d{1,4}(?:,\d{3})?)\s+Bachelor(?:'s)?\s+(?:degrees?|programs?)",
        "master": r"(\d{1,4}(?:,\d{3})?)\s+Master(?:'s)?\s+(?:degrees?|programs?)",
        "phd": r"(\d{1,4}(?:,\d{3})?)\s+PhD\s+(?:degrees?|programs?)",
        "postgraduate": r"(\d{1,4}(?:,\d{3})?)\s+(?:Post\s*Graduate|Postgraduate|Professional Diploma)(?:\s+and\s+Professional\s+Diploma)?\s+(?:degrees?|programs?)",
    }
    found: Dict[str, int] = {}
    for label, pattern in patterns.items():
        m = re.search(pattern, combined, re.I)
        if m:
            found[label] = int(m.group(1).replace(",", ""))
    if not found:
        return None

    q = question.lower()
    if all(k in q for k in ["bachelor", "master", "phd"]) or "post" in q or "diploma" in q:
        parts = []
        if "bachelor" in found:
            parts.append(f"{found['bachelor']} Bachelor's")
        if "master" in found:
            parts.append(f"{found['master']} Master's")
        if "phd" in found:
            parts.append(f"{found['phd']} PhD")
        if "postgraduate" in found:
            parts.append(f"{found['postgraduate']} postgraduate/professional diploma")
        answer = "UoS offers " + ", ".join(parts) + " degree programs."
    elif "bachelor" in q and "bachelor" in found:
        answer = f"UoS offers {found['bachelor']} Bachelor's degree programs."
    elif "master" in q and "master" in found:
        answer = f"UoS offers {found['master']} Master's degree programs."
    elif "phd" in q and "phd" in found:
        answer = f"UoS offers {found['phd']} PhD degree programs."
    elif ("post" in q or "diploma" in q) and "postgraduate" in found:
        answer = f"UoS offers {found['postgraduate']} postgraduate/professional diploma programs."
    elif "total" in q and "total" in found:
        answer = f"UoS offers {found['total']} total degree programs."
    else:
        parts = []
        if "total" in found:
            parts.append(f"{found['total']} total degree programs")
        if "bachelor" in found:
            parts.append(f"{found['bachelor']} Bachelor's")
        if "master" in found:
            parts.append(f"{found['master']} Master's")
        if "phd" in found:
            parts.append(f"{found['phd']} PhD")
        if "postgraduate" in found:
            parts.append(f"{found['postgraduate']} postgraduate/professional diploma")
        answer = "UoS offers " + ", ".join(parts) + "."

    pages_used = sorted({item["meta"]["page"] for item in items[:3]})
    evidence = next((item["chunk"] for item in items if re.search(r"\b149\b|Bachelor|Master|PhD|Diploma", item["chunk"], re.I)), items[0]["chunk"])
    return answer, pages_used, evidence


def extract_date_answer(question: str, items: List[Dict]) -> Optional[Tuple[str, List[int], str]]:
    if classify_query(question) != "date":
        return None
    q = question.lower()
    target_label = None
    if "classes begin" in q or re.search(r"\bwhen do classes begin\b", q):
        target_label = "classes begin"
    elif "add/drop" in q or "last day for add/drop" in q:
        target_label = "last day for add/drop"
    elif "final exam" in q:
        target_label = "final exams"
    elif "classes end" in q:
        target_label = "classes end"

    rows = []
    for item in items:
        if item["meta"].get("chunk_type") == "row":
            rows.append(item)
        elif item["meta"].get("chunk_type") == "row_window":
            for part in item["chunk"].split(" | "):
                rows.append({"chunk": normalize_line(part), "meta": item["meta"]})

    best = None
    best_score = -1.0
    for row in rows:
        text = row["chunk"].lower()
        score = 0.0
        if target_label and target_label in text:
            score += 3.0
        for tok in tokenize(question):
            if tok in text:
                score += 0.35
        if MONTH_PATTERN.search(text) or DAY_PATTERN.search(text) or re.search(r"\b\d{1,2}\b", text):
            score += 0.35
        if score > best_score:
            best = row
            best_score = score
    if not best or best_score < 1.1:
        return None

    row_text = best["chunk"]
    page = best["meta"]["page"]
    if target_label:
        answer = row_text
        # Cleaner formatting if label exists.
        answer = re.sub(r"\s*\|\s*", " ", answer)
        return answer + ".", [page], row_text
    return None


def verify_answer(answer: str, final_items: List[Dict], query_type: str) -> bool:
    if answer.strip() == "I do not have this information.":
        return True
    context = " \n ".join(item["chunk"] for item in final_items)
    alower = answer.lower()
    if query_type == "contact":
        nums = PHONE_PATTERN.findall(answer) + EMAIL_PATTERN.findall(answer)
        return bool(nums) and any(n in context for n in nums)
    if query_type == "count":
        nums = re.findall(r"\d{1,4}(?:,\d{3})?", answer)
        return bool(nums) and all(n in context for n in nums)
    if query_type == "date":
        date_tokens = re.findall(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|\d{1,2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\b", answer, re.I)
        if not date_tokens:
            return False
        return sum(1 for t in date_tokens if re.search(rf"\b{re.escape(t)}\b", context, re.I)) >= max(1, len(date_tokens) // 2)
    # Generic: need meaningful overlap with best evidence.
    best = final_items[0]["chunk"].lower()
    tokens = [t for t in tokenize(answer) if len(t) > 2]
    if not tokens:
        return False
    overlap = sum(1 for t in tokens[:10] if t in best)
    return overlap >= 2 or any(answer_part.lower() in best for answer_part in [answer[:50], answer[:80]] if len(answer_part) > 20)


def build_prompt(question: str, final_items: List[Dict], query_type: str) -> str:
    context_parts = []
    for i, item in enumerate(final_items, 1):
        meta = item["meta"]
        label = f"Source {i} | Page {meta['page']} | Type {meta.get('chunk_type', 'paragraph')}"
        if meta.get("section"):
            label += f" | Section {meta['section']}"
        context_parts.append(f"{label}\n{item['chunk']}")
    context = "\n\n".join(context_parts)

    extra = {
        "contact": "Return the exact phone, fax, or email only if it is clearly tied to the requested office.",
        "date": "For calendar questions, return the exact event/date wording from the source. Do not guess the semester.",
        "count": "For count questions, return all requested categories exactly if they are present.",
        "policy_yesno": "For yes/no questions, answer only if the context directly addresses that exact policy point.",
    }.get(query_type, "")

    return f"""
You answer questions about the University of Sharjah Faculty Handbook.
Use ONLY the supplied context.

Rules:
1. Give a direct, specific answer.
2. If the answer is not clearly supported by the context, reply exactly: I do not have this information.
3. Do not guess. Do not combine unrelated rows or pages.
4. {extra}
5. After the answer, add a new line exactly like this: Pages: page_numbers_only
6. Only cite pages from the supplied context.

Context:
{context}

Question: {question}

Answer:
""".strip()


def ask_groq(client: Groq, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You answer accurately from supplied context only. Be conservative."},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def parse_answer_and_pages(raw: str, final_items: List[Dict]) -> Tuple[str, List[int]]:
    allowed_pages = sorted({int(item["meta"]["page"]) for item in final_items})
    m = re.search(r"Pages:\s*([0-9,\-\s]+)", raw, re.I)
    pages = []
    if m:
        nums = re.findall(r"\d+", m.group(1))
        for n in nums:
            val = int(n)
            if val in allowed_pages and val not in pages:
                pages.append(val)
    body = re.sub(r"\n?Pages:\s*[0-9,\-\s]+\s*$", "", raw, flags=re.I).strip()
    if not pages:
        pages = allowed_pages[:3]
    return body, pages


def main():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Please set it before running MCBV9.py.")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Loading reranker...")
    reranker = CrossEncoder(RERANK_MODEL)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    pdf_file = find_pdf_file()
    print(f"Reading PDF: {pdf_file}...")
    pages = extract_text_from_pdf(pdf_file)
    chunks, metadata = build_chunks(pages)
    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    embeddings = build_or_load_embeddings(pdf_file, chunks, metadata, pages, embedder)

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print("\n==========================================================")
    print("System Ready! Type your question (or type 'exit' to quit).")
    print(f"Generation model: {GROQ_MODEL}")
    print("Improved mode: query routing + table/date/contact verification")
    print("==========================================================\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            print("Please enter a question.\n")
            continue

        query_type = classify_query(question)
        if query_type == "greeting":
            print("\nBot: Hello. Ask me a question about the University of Sharjah Faculty Handbook.\n")
            continue

        query_embedding = embedder.encode(
            [normalize_text(question)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        candidates = gather_candidates(question, query_type, query_embedding, index, chunks, metadata)
        if not candidates:
            print("\nBot: I do not have this information.\n")
            continue

        pairs = [[question, cand["chunk"]] for cand in candidates]
        rerank_scores = reranker.predict(pairs)
        reranked = []
        for cand, score in zip(candidates, rerank_scores):
            cand = dict(cand)
            cand["rerank_score"] = float(score)
            reranked.append(cand)

        reranked.sort(
            key=lambda x: (
                x["rerank_score"],
                x.get("routing_boost", 0.0),
                x.get("lexical_score", 0.0),
                x.get("dense_score", 0.0),
            ),
            reverse=True,
        )
        reranked = deduplicate_by_text(reranked)
        final_items = reranked[:FINAL_K]

        if not final_items or final_items[0]["rerank_score"] < MIN_RERANK_SCORE:
            print("\nBot: I do not have this information.\n")
            continue

        deterministic = None
        if query_type == "contact":
            deterministic = extract_contact_answer(question, final_items)
        elif query_type == "count":
            deterministic = extract_count_answer(question, final_items, pages)
        elif query_type == "date":
            deterministic = extract_date_answer(question, final_items)

        if deterministic:
            answer, source_pages, evidence = deterministic
        else:
            prompt = build_prompt(question, final_items, query_type)
            raw = ask_groq(client, prompt)
            answer, source_pages = parse_answer_and_pages(raw, final_items)
            if not answer:
                answer = "I do not have this information."
            if not verify_answer(answer, final_items, query_type):
                answer = "I do not have this information."
            evidence = final_items[0]["chunk"]

        best = final_items[0]
        evidence = evidence[:520].strip() + ("..." if len(evidence) > 520 else "")
        source_pages_text = ", ".join(f"Page {p}" for p in source_pages)
        print(f"\nBot: {answer}\n")
        print(f"Sources: {source_pages_text}")
        if best["meta"].get("section"):
            print(f"Best Section: {best['meta']['section']}")
        print("\nTop Evidence Snippet:")
        print(evidence)
        print()


if __name__ == "__main__":
    main()
