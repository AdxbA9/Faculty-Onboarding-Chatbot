import os
import numpy as np
import faiss
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================================
# CONFIGURATION
# ==========================================================

PDF_PATH = "handbook.pdf"   # Change if needed

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "google/flan-t5-base"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 8
FINAL_K = 3

# ==========================================================
# LOAD MODELS
# ==========================================================

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading reranker...")
reranker = CrossEncoder(RERANK_MODEL)

print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_MODEL,
    dtype=torch.float32
)
llm_model.eval()

# ==========================================================
# PDF PROCESSING
# ==========================================================

def extract_text_from_pdf(path):
    if not os.path.exists(path):
        print(f"\nERROR: File '{path}' not found.")
        print("Place your PDF in this folder or update PDF_PATH.\n")
        exit()

    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((i + 1, text))

    return pages


def chunk_text(pages):
    chunks = []
    metadata = []

    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            chunks.append(chunk)
            metadata.append({"page": page_num})
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks, metadata


# ==========================================================
# BUILD INDEX (COSINE SIMILARITY)
# ==========================================================

print("Reading PDF...")
pages = extract_text_from_pdf(PDF_PATH)

print("Chunking text...")
chunks, metadata = chunk_text(pages)

print("Building FAISS index...")

embeddings = embedder.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True
)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine (normalized)
index.add(embeddings)

print("\nSystem Ready! Type your question (or 'exit' to quit).\n")

# ==========================================================
# CHAT LOOP
# ==========================================================

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Embed question
    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )

    D, I = index.search(np.array(query_embedding), TOP_K)

    retrieved_chunks = [chunks[i] for i in I[0]]
    retrieved_meta = [metadata[i] for i in I[0]]

    # Rerank
    pairs = [[question, chunk] for chunk in retrieved_chunks]
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(scores, retrieved_chunks, retrieved_meta),
        key=lambda x: x[0],
        reverse=True
    )[:FINAL_K]

    final_chunks = [x[1] for x in reranked]
    final_meta = [x[2] for x in reranked]

    context = "\n\n".join(final_chunks)

    prompt = f"""
You are a university policy assistant.

Answer clearly and completely using ONLY the provided context.

If the answer is not found in the context, say:
"I don’t have enough information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=200
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nBot:", answer)
    print("\nSources:")
    for m in final_meta:
        print(f"- Page {m['page']}")
    print("\n")