import os
import re
import json
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    from groq import Groq
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'groq' package is not installed. Run: pip install groq"
    ) from exc

# ==========================================================
# SETTINGS
# ==========================================================
PDF_PATH = None  # None = auto-detect first PDF in current folder

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Default model is configurable. Change via environment variable if needed.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

CHUNK_SIZE_WORDS = 160
CHUNK_OVERLAP_WORDS = 80
TOP_K = 30
FINAL_K = 5
MIN_RERANK_SCORE = 0.0
MAX_CONTEXT_CHARS = 5000
MAX_EVIDENCE_CHARS = 300

PHONE_PATTERN = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================================
# HELPERS
# ==========================================================

def find_pdf_file() -> str:
    if PDF_PATH is not None:
        if os.path.exists(PDF_PATH):
            return PDF_PATH
        raise FileNotFoundError(f"File '{PDF_PATH}' not found.")

    pdf_files = sorted([f for f in os.listdir() if f.lower().endswith(".pdf")])
    if not pdf_files:
        raise FileNotFoundError("No PDF file was found in the current folder.")

    print(f"Detected PDF file: {pdf_files[0]}")
    return pdf_files[0]


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_common_headers_footers(page_texts: List[Dict]) -> List[Dict]:
    """Remove long repeated lines that look like handbook headers/footers."""
    line_counts: Dict[str, int] = {}
    split_pages: List[List[str]] = []

    for page in page_texts:
        lines = [line.strip() for line in page["raw_lines"] if line.strip()]
        split_pages.append(lines)
        for line in set(lines):
            line_counts[line] = line_counts.get(line, 0) + 1

    threshold = max(3, int(len(page_texts) * 0.4))
    repeated = {
        line for line, count in line_counts.items() if count >= threshold and len(line) > 120
    }

    cleaned_pages = []
    for page, lines in zip(page_texts, split_pages):
        filtered = [line for line in lines if line not in repeated]
        cleaned_pages.append({
            "page": page["page"],
            "text": normalize_text(" ".join(filtered)),
        })

    return cleaned_pages


def extract_text_from_pdf(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found.")

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({
            "page": i + 1,
            "raw_lines": text.splitlines(),
        })

    cleaned_pages = remove_common_headers_footers(pages)
    return [p for p in cleaned_pages if p["text"]]


def chunk_text(pages: List[Dict]) -> Tuple[List[str], List[Dict]]:
    chunks: List[str] = []
    metadata: List[Dict] = []

    for page_data in pages:
        page_num = page_data["page"]
        text = normalize_text(page_data["text"])
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            continue

        chunk_words: List[str] = []
        chunk_id_on_page = 0

        def flush_chunk() -> None:
            nonlocal chunk_words, chunk_id_on_page
            if not chunk_words:
                return
            chunk_text_value = " ".join(chunk_words)
            chunk_id = len(chunks)
            chunks.append(chunk_text_value)
            metadata.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "page_start": page_num,
                "page_end": page_num,
                "chunk_id_on_page": chunk_id_on_page,
                "text": chunk_text_value,
            })
            chunk_id_on_page += 1

        for sentence in sentences:
            sentence_words = sentence.split()
            while sentence_words:
                space_left = CHUNK_SIZE_WORDS - len(chunk_words)
                if space_left <= 0:
                    flush_chunk()
                    chunk_words = (
                        chunk_words[-CHUNK_OVERLAP_WORDS:] if CHUNK_OVERLAP_WORDS else []
                    )
                    space_left = CHUNK_SIZE_WORDS - len(chunk_words)

                if len(sentence_words) <= space_left:
                    chunk_words.extend(sentence_words)
                    sentence_words = []
                else:
                    chunk_words.extend(sentence_words[:space_left])
                    sentence_words = sentence_words[space_left:]
                    flush_chunk()
                    chunk_words = (
                        chunk_words[-CHUNK_OVERLAP_WORDS:] if CHUNK_OVERLAP_WORDS else []
                    )

        if chunk_words:
            flush_chunk()

    return chunks, metadata


def cache_paths(pdf_file: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(pdf_file))[0]
    return (
        os.path.join(CACHE_DIR, f"{base}_embeddings.npy"),
        os.path.join(CACHE_DIR, f"{base}_chunks.json"),
    )


def build_or_load_embeddings(pdf_file: str, chunks: List[str], embedder: SentenceTransformer) -> np.ndarray:
    emb_path, chunk_path = cache_paths(pdf_file)

    if os.path.exists(emb_path) and os.path.exists(chunk_path):
        with open(chunk_path, "r", encoding="utf-8") as f:
            cached_chunks = json.load(f)
        if cached_chunks == chunks:
            print("Loading cached embeddings...")
            return np.load(emb_path)

    print("Creating embeddings...")
    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(emb_path, embeddings)
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return embeddings


def create_query_embedding(question: str, embedder: SentenceTransformer) -> np.ndarray:
    clean_question = normalize_text(question)
    return embedder.encode(
        [clean_question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def trim_context(final_items: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts: List[str] = []
    total = 0
    for item in final_items:
        part = f"[Page {item['meta']['page']}] {item['chunk']}"
        if total + len(part) > max_chars and parts:
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)


def build_messages(context: str, question: str) -> List[Dict[str, str]]:
    system_message = (
        "You are a careful University of Sharjah faculty onboarding assistant. "
        "Answer ONLY from the provided handbook context. "
        "Never use outside knowledge. "
        "If the answer is not clearly supported by the context, reply exactly: "
        "I do not have this information. "
        "Be concise, accurate, and student-project friendly. "
        "When possible, mention the relevant page number already included in the context tags."
    )
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Return a short direct answer."
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def extract_phone_fallback(question: str, chunks: List[Dict]) -> Optional[str]:
    query = question.lower()
    if not any(term in query for term in ["phone", "telephone", "tel", "contact", "fax"]):
        return None

    prefer_fax = "fax" in query
    for chunk_data in chunks:
        chunk = chunk_data["chunk"]
        if prefer_fax and re.search(r"fax", chunk, re.I):
            matches = PHONE_PATTERN.findall(chunk)
            if matches:
                return f"The fax number is {matches[0].strip()}."
        if not prefer_fax and re.search(r"phone|telephone|tel|contact", chunk, re.I):
            matches = PHONE_PATTERN.findall(chunk)
            if matches:
                return f"The phone number is {matches[0].strip()}."

    for chunk_data in chunks:
        matches = PHONE_PATTERN.findall(chunk_data["chunk"])
        if matches:
            label = "fax number" if prefer_fax else "phone number"
            return f"The {label} is {matches[0].strip()}."

    return None


def extract_count_fallback(question: str, candidate_chunks: List[Dict], pages: List[Dict]) -> Optional[str]:
    query = question.lower()
    if not re.search(r"\b(how many|number of|total number|count of|total|how much)\b", query):
        return None

    def find_counts(text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        patterns = {
            "total": r"\b(?:total of|total number of|a total of|offers a total of|offer a total of)\s+(\d{1,4}(?:,\d{3})?)\b",
            "bachelor": r"(\d{1,4}(?:,\d{3})?)\s+Bachelor(?:'s)?\s+degrees?",
            "master": r"(\d{1,4}(?:,\d{3})?)\s+Master(?:'s)?\s+degrees?",
            "phd": r"(\d{1,4}(?:,\d{3})?)\s+PhD\s+degrees?",
            "postgraduate": r"(\d{1,4}(?:,\d{3})?)\s+(?:Post\s*Graduate|Postgraduate|Professional\s+Diploma)s?",
        }
        for label, pattern in patterns.items():
            match = re.search(pattern, text, re.I)
            if match:
                counts[label] = int(match.group(1).replace(",", ""))
        return counts

    combined_text = " ".join([c["chunk"] for c in candidate_chunks] + [p["text"] for p in pages])
    counts = find_counts(combined_text)
    if not counts:
        return None

    parts = []
    if "total" in counts:
        parts.append(f"{counts['total']} total degree programs")
    if "bachelor" in counts:
        parts.append(f"{counts['bachelor']} Bachelor's degrees")
    if "master" in counts:
        parts.append(f"{counts['master']} Master's degrees")
    if "phd" in counts:
        parts.append(f"{counts['phd']} PhD degrees")
    if "postgraduate" in counts:
        parts.append(f"{counts['postgraduate']} postgraduate/professional diploma programs")

    if not parts:
        return None
    if "total" in counts:
        if len(parts) > 1:
            return f"As of Fall 2025/2026, UoS offers {parts[0]}, including " + ", ".join(parts[1:]) + "."
        return f"As of Fall 2025/2026, UoS offers {parts[0]}."
    return "As of Fall 2025/2026, UoS offers " + ", ".join(parts) + "."


def deduplicate_chunks(reranked_items: List[Dict]) -> List[Dict]:
    seen = set()
    result = []
    for item in reranked_items:
        text = item["chunk"]
        if text not in seen:
            seen.add(text)
            result.append(item)
    return result


def format_sources(final_items: List[Dict]) -> str:
    seen_pages = []
    for item in final_items:
        page = item["meta"]["page"]
        if page not in seen_pages:
            seen_pages.append(page)
    return ", ".join(f"Page {page}" for page in seen_pages)


def ask_groq(messages: List[Dict[str, str]], client: Groq) -> str:
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.1,
        max_completion_tokens=220,
    )
    return completion.choices[0].message.content.strip()


# ==========================================================
# INITIALIZE MODELS
# ==========================================================
if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Create a key at https://console.groq.com and set it in your environment."
    )

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading reranker...")
reranker = CrossEncoder(RERANK_MODEL)

print("Connecting to Groq...")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ==========================================================
# LOAD PDF + BUILD INDEX
# ==========================================================
pdf_file = find_pdf_file()
print(f"Reading PDF: {pdf_file}...")

pages = extract_text_from_pdf(pdf_file)
chunks, metadata = chunk_text(pages)
if not chunks:
    raise ValueError("No text chunks were created from the PDF.")

embeddings = build_or_load_embeddings(pdf_file, chunks, embedder)

print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("\n==========================================================")
print("System Ready! Type your question (or type 'exit' to quit).")
print(f"Generation model: {GROQ_MODEL}")
print("==========================================================\n")


# ==========================================================
# CHAT LOOP
# ==========================================================
while True:
    question = input("You: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.\n")
        continue

    query_embedding = create_query_embedding(question, embedder)
    distances, indices = index.search(query_embedding, TOP_K)

    candidates = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        candidates.append({
            "faiss_score": float(score),
            "chunk": chunks[idx],
            "meta": metadata[idx],
        })

    if not candidates:
        print("\nBot: I do not have this information.\n")
        continue

    pairs = [[question, c["chunk"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    reranked = []
    for c, r_score in zip(candidates, rerank_scores):
        reranked.append({
            "rerank_score": float(r_score),
            "faiss_score": c["faiss_score"],
            "chunk": c["chunk"],
            "meta": c["meta"],
        })

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    reranked = deduplicate_chunks(reranked)

    if not reranked or reranked[0]["rerank_score"] < MIN_RERANK_SCORE:
        print("\nBot: I do not have this information.\n")
        continue

    final_items = reranked[:FINAL_K]

    phone_answer = extract_phone_fallback(question, final_items)
    count_answer = extract_count_fallback(question, final_items, pages)
    if phone_answer:
        answer = phone_answer
    elif count_answer:
        answer = count_answer
    else:
        context = trim_context(final_items)
        messages = build_messages(context, question)
        try:
            answer = ask_groq(messages, groq_client)
        except Exception as exc:  # pragma: no cover
            print(f"\nBot: Groq API error: {exc}\n")
            continue

    best_item = final_items[0]
    evidence = best_item["chunk"][:MAX_EVIDENCE_CHARS].strip()

    print("\nBot:", answer)
    print("\nSources:", format_sources(final_items))
    print("\nTop Evidence Snippet:")
    print(evidence + ("..." if len(best_item["chunk"]) > MAX_EVIDENCE_CHARS else ""))
    print()
