import os
import re
import json
import numpy as np
import faiss
import torch
from typing import Dict, List, Optional

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================================
# SETTINGS
# ==========================================================
PDF_PATH = None  # None = auto-detect first PDF in current folder

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
LLM_MODEL = "google/flan-t5-xl"

CHUNK_SIZE_WORDS = 160
CHUNK_OVERLAP_WORDS = 80
TOP_K = 40
FINAL_K = 4

MIN_FAISS_SCORE = 0.0
MIN_RERANK_SCORE = 0.0
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

    pdf_files = [f for f in os.listdir() if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError("No PDF file was found in the current folder.")

    print(f"Detected PDF file {pdf_files[0]}")
    return pdf_files[0]


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_common_headers_footers(page_texts):
    """Very simple repeated-line remover.

    Good for repeated handbook headers/footers.
    """
    line_counts = {}
    split_pages = []

    for page in page_texts:
        lines = [line.strip() for line in page["raw_lines"] if line.strip()]
        split_pages.append(lines)
        unique_lines = set(lines)
        for line in unique_lines:
            line_counts[line] = line_counts.get(line, 0) + 1

    threshold = max(3, int(len(page_texts) * 0.4))
    repeated = {
        line
        for line, count in line_counts.items()
        if count >= threshold and len(line) > 120
    }

    cleaned_pages = []
    for page, lines in zip(page_texts, split_pages):
        filtered = [line for line in lines if line not in repeated]
        cleaned_pages.append({
            "page": page["page"],
            "text": normalize_text(" ".join(filtered)),
        })

    return cleaned_pages


def extract_text_from_pdf(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found.")

    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        raw_lines = text.splitlines()
        pages.append({
            "page": i + 1,
            "raw_lines": raw_lines,
        })

    cleaned_pages = remove_common_headers_footers(pages)
    cleaned_pages = [p for p in cleaned_pages if p["text"]]
    return cleaned_pages


def chunk_text(pages):
    chunks = []
    metadata = []

    for page_data in pages:
        page_num = page_data["page"]
        text = normalize_text(page_data["text"])

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            continue

        chunk_words = []
        chunk_id_on_page = 0

        def flush_chunk():
            nonlocal chunk_words, chunk_id_on_page
            if not chunk_words:
                return
            chunk_text = " ".join(chunk_words)
            chunk_id = len(chunks)
            chunks.append(chunk_text)
            metadata.append({
                "chunk_id": chunk_id,
                "page_start": page_num,
                "page_end": page_num,
                "page": page_num,
                "chunk_id_on_page": chunk_id_on_page,
                "text": chunk_text,
            })
            chunk_id_on_page += 1

        for sentence in sentences:
            sentence_words = sentence.split()
            if not sentence_words:
                continue

            while sentence_words:
                space_left = CHUNK_SIZE_WORDS - len(chunk_words)
                if space_left <= 0:
                    flush_chunk()
                    chunk_words = chunk_words[-CHUNK_OVERLAP_WORDS:] if CHUNK_OVERLAP_WORDS else []
                    space_left = CHUNK_SIZE_WORDS - len(chunk_words)

                if len(sentence_words) <= space_left:
                    chunk_words.extend(sentence_words)
                    sentence_words = []
                else:
                    chunk_words.extend(sentence_words[:space_left])
                    sentence_words = sentence_words[space_left:]
                    flush_chunk()
                    chunk_words = chunk_words[-CHUNK_OVERLAP_WORDS:] if CHUNK_OVERLAP_WORDS else []

        if chunk_words:
            flush_chunk()

    return chunks, metadata


def cache_paths(pdf_file: str):
    base = os.path.splitext(os.path.basename(pdf_file))[0]
    return (
        os.path.join(CACHE_DIR, f"{base}_embeddings.npy"),
        os.path.join(CACHE_DIR, f"{base}_meta.json"),
    )


def build_or_load_embeddings(pdf_file: str, chunks, embedder):
    emb_path, meta_path = cache_paths(pdf_file)

    if os.path.exists(emb_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            cached_chunks = json.load(f)
        if cached_chunks == chunks:
            print("Loading cached embeddings...")
            return np.load(emb_path)

    print("Creating embeddings...")
    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.save(emb_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return embeddings


def create_query_embedding(question: str, embedder) -> np.ndarray:
    clean_question = normalize_text(question)
    return embedder.encode(
        [clean_question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def build_prompt(context: str, question: str) -> str:
    return (
        "You are a helpful and concise question-answering assistant for the University of Sharjah handbook.\n"
        "Use ONLY the provided context to answer. Do not use any outside knowledge.\n"
        "If the answer is not clearly stated in the provided context, reply exactly:\n"
        "I do not have this information.\n"
        "Answer in 1-2 sentences.\n"
        "Do NOT repeat the context verbatim; instead, summarize or paraphrase the answer in your own words.\n"
        "If the question asks for a number, give the exact number found in the context.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )


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
        chunk = chunk_data["chunk"]
        matches = PHONE_PATTERN.findall(chunk)
        if matches:
            label = "fax number" if prefer_fax else "phone number"
            return f"The {label} is {matches[0].strip()}."

    return None


def extract_count_fallback(question: str, candidate_chunks: List[Dict], pages: List[Dict]) -> Optional[str]:
    query = question.lower()
    if not re.search(r"\b(how many|number of|total number|count of|total|how much)\b", query):
        return None

    def find_counts(text: str) -> Dict[str, int]:
        counts = {}
        count_patterns = {
            "total": r"\b(?:total of|total number of|a total of|offers a total of|offer a total of)\s+(\d{1,4}(?:,\d{3})?)\b",
            "bachelor": r"(\d{1,4}(?:,\d{3})?)\s+Bachelor(?:'s)?\s+degrees?",
            "master": r"(\d{1,4}(?:,\d{3})?)\s+Master(?:'s)?\s+degrees?",
            "phd": r"(\d{1,4}(?:,\d{3})?)\s+PhD\s+degrees?",
            "postgraduate": r"(\d{1,4}(?:,\d{3})?)\s+(?:Post\s*Graduate|Postgraduate|Professional\s+Diploma)s?",
        }

        for label, pattern in count_patterns.items():
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
            return (
                f"As of Fall 2025/2026, UoS offers {parts[0]}, including "
                + ", ".join(parts[1:])
                + "."
            )
        return f"As of Fall 2025/2026, UoS offers {parts[0]}."

    return "As of Fall 2025/2026, UoS offers " + ", ".join(parts) + "."


def deduplicate_chunks(reranked_items):
    seen = set()
    result = []

    for item in reranked_items:
        text = item["chunk"]
        if text not in seen:
            seen.add(text)
            result.append(item)
    return result


# ==========================================================
# INITIALIZE MODELS
# ==========================================================
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading reranker...")
reranker = CrossEncoder(RERANK_MODEL)

print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float32,
)
llm_model.eval()


# ==========================================================
# LOAD PDF + BUILD INDEX
# ==========================================================
pdf_file = find_pdf_file()
print(f"Reading PDF {pdf_file}...")

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

    D, I = index.search(query_embedding, TOP_K)

    candidates = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue

        candidates.append({
            "faiss_score": float(score),
            "chunk": chunks[idx],
            "meta": metadata[idx],
        })

    if not candidates:
        print("\nBot I do not have this information.\n")
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
        print("\nBot I do not have this information.\n")
        continue

    final_items = reranked[:FINAL_K]
    context = "\n\n".join(
        [f"[Page {item['meta']['page']}] {item['chunk']}" for item in final_items]
    )

    phone_answer = extract_phone_fallback(question, final_items)
    count_answer = extract_count_fallback(question, final_items, pages)
    if phone_answer:
        answer = phone_answer
    elif count_answer:
        answer = count_answer
    else:
        prompt = build_prompt(context, question)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=180,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.15,
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    best_item = final_items[0]
    best_page = best_item["meta"]["page"]
    evidence = best_item["chunk"][:300].strip()

    print("\nBot", answer)
    print("\nBest Source")
    print(f"- Page {best_page}")
    print("\nTop Evidence Snippet")
    print(evidence + ("..." if len(best_item["chunk"]) > 300 else ""))
    print()
