import os
import re
import json
import math
import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# Try better PDF extraction first.
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

PARA_CHUNK_SIZE_WORDS = 180
PARA_CHUNK_OVERLAP_WORDS = 60
LINE_WINDOW_SIZE = 8
LINE_WINDOW_STEP = 4

TOP_K_DENSE = 30
TOP_K_LEXICAL = 12
RERANK_CANDIDATES = 20
FINAL_K = 6
MIN_RERANK_SCORE = -2.0

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

PHONE_PATTERN = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
YEAR_TERM_PATTERN = re.compile(r"\b(fall|spring|summer)\b", re.I)
COUNT_PATTERN = re.compile(r"\b(how many|number of|total number|count of|total)\b", re.I)
TABLE_HINT_PATTERN = re.compile(
    r"\b(phone|telephone|fax|contact|email|office|number|date|when|calendar|schedule|committee|list|programs?)\b",
    re.I,
)

STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "for", "to", "in", "on", "at", "and",
    "or", "by", "with", "what", "which", "who", "when", "where", "how", "many",
    "does", "do", "did", "can", "could", "should", "would", "be", "it", "this",
    "that", "from", "as", "about", "into", "under", "their", "them", "they",
    "university", "sharjah", "uos", "faculty", "handbook",
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
    line = re.sub(r"\s+", " ", line).strip()
    return line


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9+\-']+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def detect_heading(line: str) -> bool:
    if not line:
        return False
    if len(line) > 110:
        return False
    if PHONE_PATTERN.search(line):
        return False
    letters = re.sub(r"[^A-Za-z]", "", line)
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
    numbered = bool(re.match(r"^\d+(?:\.\d+)*\s+", line))
    title_like = line.istitle() and len(line.split()) <= 10
    return numbered or upper_ratio > 0.72 or title_like


def remove_common_headers_footers(raw_pages: List[Dict]) -> List[Dict]:
    line_counts: Dict[str, int] = {}
    page_lines: List[List[str]] = []

    for page in raw_pages:
        lines = [normalize_line(x) for x in page["raw_lines"] if normalize_line(x)]
        page_lines.append(lines)
        for line in set(lines):
            if len(line) >= 4:
                line_counts[line] = line_counts.get(line, 0) + 1

    threshold = max(4, int(len(raw_pages) * 0.45))
    repeated = {
        line for line, count in line_counts.items()
        if count >= threshold and (len(line) > 80 or re.search(r"faculty handbook|university of sharjah|page \d+", line, re.I))
    }

    cleaned_pages = []
    current_section = ""
    for page, lines in zip(raw_pages, page_lines):
        filtered = [line for line in lines if line not in repeated]
        for line in filtered[:8]:
            if detect_heading(line):
                current_section = line
                break
        cleaned_pages.append({
            "page": page["page"],
            "lines": filtered,
            "text": normalize_text(" ".join(filtered)),
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
            raw_pages.append({"page": i + 1, "raw_lines": txt.splitlines()})
        doc.close()
    elif PdfReader is not None:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            raw_pages.append({"page": i + 1, "raw_lines": txt.splitlines()})
    else:
        raise RuntimeError("No PDF extraction library is available.")

    return remove_common_headers_footers(raw_pages)


def build_chunks(pages: List[Dict]) -> Tuple[List[str], List[Dict]]:
    chunks: List[str] = []
    metadata: List[Dict] = []

    def add_chunk(text: str, meta: Dict):
        clean = normalize_text(text)
        if len(clean) < 20:
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
        text = page["text"]

        # Paragraph/sentence chunks
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
                    "page_start": page_num,
                    "page_end": page_num,
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
                "page_start": page_num,
                "page_end": page_num,
                "section": section_hint,
                "chunk_type": "paragraph",
                "chunk_id_on_page": para_id,
            })

        # Line-window chunks for tables / contacts / calendars
        line_id = 0
        for start in range(0, max(1, len(lines)), LINE_WINDOW_STEP):
            window = lines[start:start + LINE_WINDOW_SIZE]
            if not window:
                continue
            add_chunk(" | ".join(window), {
                "page": page_num,
                "page_start": page_num,
                "page_end": page_num,
                "section": section_hint,
                "chunk_type": "line_window",
                "line_start": start,
                "line_end": min(start + LINE_WINDOW_SIZE - 1, len(lines) - 1),
                "chunk_id_on_page": line_id,
            })
            line_id += 1

    return chunks, metadata


def cache_paths(pdf_file: str):
    base = os.path.splitext(os.path.basename(pdf_file))[0]
    return (
        os.path.join(CACHE_DIR, f"{base}_mcbv8_embeddings.npy"),
        os.path.join(CACHE_DIR, f"{base}_mcbv8_chunks.json"),
        os.path.join(CACHE_DIR, f"{base}_mcbv8_meta.json"),
        os.path.join(CACHE_DIR, f"{base}_mcbv8_pages.json"),
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
    if overlap == 0:
        return 0.0
    phrase_bonus = 0.0
    qnorm = normalize_text(query).lower()
    tnorm = normalize_text(text).lower()
    if len(qnorm) > 6 and qnorm in tnorm:
        phrase_bonus += 2.0
    # Boost table-like contexts for table-like questions.
    if TABLE_HINT_PATTERN.search(query) and ("|" in text or PHONE_PATTERN.search(text)):
        phrase_bonus += 1.5
    return overlap / math.sqrt(len(tset) + 1) + phrase_bonus


def gather_candidates(question: str, query_embedding: np.ndarray, index, chunks: List[str], metadata: List[Dict]) -> List[Dict]:
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
        [(i, lexical_score(question, text)) for i, text in enumerate(chunks)],
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

    # query-aware boost for table-like questions on line_window chunks
    for cand in candidates.values():
        cand["routing_boost"] = 0.0
        if TABLE_HINT_PATTERN.search(question) and cand["meta"].get("chunk_type") == "line_window":
            cand["routing_boost"] += 0.35
        if COUNT_PATTERN.search(question) and re.search(r"\b\d{1,4}\b", cand["chunk"]):
            cand["routing_boost"] += 0.25
        if YEAR_TERM_PATTERN.search(question) and re.search(r"\b(mon|tue|wed|thu|fri|sat|sun|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", cand["chunk"], re.I):
            cand["routing_boost"] += 0.2

    merged = list(candidates.values())
    merged.sort(key=lambda x: (x["dense_score"] + 0.25 * x["lexical_score"] + x["routing_boost"]), reverse=True)
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


def extract_phone_or_fax(question: str, items: List[Dict]) -> Optional[str]:
    q = question.lower()
    if not any(term in q for term in ["phone", "telephone", "fax", "contact"]):
        return None

    target_is_fax = "fax" in q
    office_match = re.search(r"for\s+(.+?)(?:\?|$)", question, re.I)
    office_name = office_match.group(1).strip() if office_match else ""

    for item in items:
        text = item["chunk"]
        if office_name and office_name.lower() not in text.lower():
            continue
        matches = PHONE_PATTERN.findall(text)
        if not matches:
            continue
        if target_is_fax:
            if re.search(r"fax", text, re.I):
                return f"The fax number is {matches[-1].strip()}."
        else:
            return f"The phone number is {matches[0].strip()}."
    return None


def extract_count_answer(question: str, items: List[Dict], pages: List[Dict]) -> Optional[str]:
    if not COUNT_PATTERN.search(question):
        return None

    combined = " ".join([x["chunk"] for x in items] + [p["text"] for p in pages[: min(40, len(pages))]])
    patterns = {
        "total": r"(?:total of|total number of|offers a total of|offer a total of|offers)\s+(\d{1,4}(?:,\d{3})?)\s+(?:degree programs|programs)",
        "bachelor": r"(\d{1,4}(?:,\d{3})?)\s+Bachelor(?:'s)?\s+(?:degrees?|programs?)",
        "master": r"(\d{1,4}(?:,\d{3})?)\s+Master(?:'s)?\s+(?:degrees?|programs?)",
        "phd": r"(\d{1,4}(?:,\d{3})?)\s+PhD\s+(?:degrees?|programs?)",
        "postgraduate": r"(\d{1,4}(?:,\d{3})?)\s+(?:postgraduate|professional diploma)\s+(?:degrees?|programs?)",
    }
    found = {}
    for label, pattern in patterns.items():
        m = re.search(pattern, combined, re.I)
        if m:
            found[label] = int(m.group(1).replace(",", ""))

    q = question.lower()
    if not found:
        return None
    if "bachelor" in q and "bachelor" in found:
        return f"UoS offers {found['bachelor']} Bachelor's degree programs."
    if "master" in q and "master" in found:
        return f"UoS offers {found['master']} Master's degree programs."
    if "phd" in q and "phd" in found:
        return f"UoS offers {found['phd']} PhD degree programs."
    if ("post" in q or "diploma" in q) and "postgraduate" in found:
        return f"UoS offers {found['postgraduate']} postgraduate/professional diploma programs."
    if "total" in q and "total" in found:
        return f"UoS offers {found['total']} total degree programs."

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
    return "UoS offers " + ", ".join(parts) + "."


def build_prompt(question: str, final_items: List[Dict]) -> str:
    context_parts = []
    for i, item in enumerate(final_items, 1):
        meta = item["meta"]
        label = f"Source {i} | Page {meta['page']}"
        if meta.get("section"):
            label += f" | Section {meta['section']}"
        label += f" | Type {meta.get('chunk_type', 'paragraph')}"
        context_parts.append(f"{label}\n{item['chunk']}")

    context = "\n\n".join(context_parts)
    return f"""
You are answering questions about the University of Sharjah Faculty Handbook.
Use ONLY the context below.

Rules:
1. Give a direct, specific answer.
2. If the answer is not clearly supported by the context, reply exactly: I do not have this information.
3. Prefer exact facts, names, dates, phone numbers, categories, and counts from the context.
4. For table-like information, read carefully across the provided line-based sources before answering.
5. After the answer, add a new line in this exact format: Pages: page_numbers_only
6. Do not mention any page that is not in the supplied sources.
7. Do not use outside knowledge.

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
            {"role": "system", "content": "You answer accurately from supplied context only."},
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
        raise RuntimeError("GROQ_API_KEY is not set. Please set it before running MCBV8.py.")

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
    print("==========================================================\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            print("Please enter a question.\n")
            continue

        query_embedding = embedder.encode(
            [normalize_text(question)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        candidates = gather_candidates(question, query_embedding, index, chunks, metadata)
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

        deterministic = extract_phone_or_fax(question, final_items)
        if not deterministic:
            deterministic = extract_count_answer(question, final_items, pages)

        if deterministic:
            answer = deterministic
            source_pages = sorted({item["meta"]["page"] for item in final_items[:3]})
        else:
            prompt = build_prompt(question, final_items)
            raw = ask_groq(client, prompt)
            answer, source_pages = parse_answer_and_pages(raw, final_items)
            if not answer:
                answer = "I do not have this information."

        best = final_items[0]
        evidence = best["chunk"][:420].strip()
        if len(best["chunk"]) > 420:
            evidence += "..."

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
