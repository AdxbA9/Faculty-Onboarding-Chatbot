import os
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import CrossEncoder, SentenceTransformer


APP_TITLE = "UOS Faculty Onboarding Chatbot"
APP_SUBTITLE = "RAG-based assistant for the University of Sharjah Faculty Handbook"
DEFAULT_SAMPLE_QUESTIONS = [
    "What is the main purpose of the University of Sharjah Faculty Handbook?",
    "What should faculty consult if the handbook does not cover a topic fully?",
    "What is the University of Sharjah's mission?",
    "How many degree programs does UoS offer in Fall 2025/2026?",
    "What is the telephone number for Information Technology Center?",
    "What is the function of standing committees at UoS?",
]


def load_backend_module() -> Any:
    """Load MCBV9.py from the same folder as this app."""
    backend_path = Path(__file__).with_name("MCBV9.py")
    if not backend_path.exists():
        raise FileNotFoundError(
            "MCBV9.py was not found in the same folder as this Streamlit app. "
            "Place streamlit_mcbv9_app.py and MCBV9.py together inside your MiniV6 folder."
        )

    spec = importlib.util.spec_from_file_location("mcbv9_backend", backend_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def load_models():
    backend = load_backend_module()
    embedder = SentenceTransformer(backend.EMBED_MODEL)
    reranker = CrossEncoder(backend.RERANK_MODEL)
    api_key = os.getenv("GROQ_API_KEY", "")
    groq_client = Groq(api_key=api_key) if api_key else None
    return backend, embedder, reranker, groq_client


@st.cache_resource(show_spinner=True)
def prepare_knowledge_base(pdf_name_hint: str = ""):
    backend, embedder, reranker, groq_client = load_models()
    pdf_file = backend.find_pdf_file()
    pages = backend.extract_text_from_pdf(pdf_file)
    chunks, metadata = backend.build_chunks(pages)
    if not chunks:
        raise ValueError("No chunks were extracted from the PDF.")
    embeddings = backend.build_or_load_embeddings(pdf_file, chunks, metadata, pages, embedder)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return {
        "backend": backend,
        "embedder": embedder,
        "reranker": reranker,
        "groq_client": groq_client,
        "pdf_file": pdf_file,
        "pages": pages,
        "chunks": chunks,
        "metadata": metadata,
        "embeddings": embeddings,
        "index": index,
    }


def ask_bot(question: str, kb: Dict[str, Any]) -> Dict[str, Any]:
    backend = kb["backend"]
    embedder = kb["embedder"]
    reranker = kb["reranker"]
    groq_client = kb["groq_client"]
    pages = kb["pages"]
    chunks = kb["chunks"]
    metadata = kb["metadata"]
    index = kb["index"]

    query_type = backend.classify_query(question)
    if query_type == "greeting":
        return {
            "answer": "Hello. Ask me a question about the University of Sharjah Faculty Handbook.",
            "pages": [],
            "best_section": "Greeting",
            "evidence": "",
            "query_type": query_type,
            "raw_items": [],
        }

    query_embedding = embedder.encode(
        [backend.normalize_text(question)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    candidates = backend.gather_candidates(question, query_type, query_embedding, index, chunks, metadata)
    if not candidates:
        return {
            "answer": "I do not have this information.",
            "pages": [],
            "best_section": "No matching evidence",
            "evidence": "",
            "query_type": query_type,
            "raw_items": [],
        }

    pairs = [[question, cand["chunk"]] for cand in candidates]
    rerank_scores = reranker.predict(pairs)
    reranked = []
    for cand, score in zip(candidates, rerank_scores):
        item = dict(cand)
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda x: (
            x["rerank_score"],
            x.get("routing_boost", 0.0),
            x.get("lexical_score", 0.0),
            x.get("dense_score", 0.0),
        ),
        reverse=True,
    )
    reranked = backend.deduplicate_by_text(reranked)
    final_items = reranked[: backend.FINAL_K]

    if not final_items or final_items[0]["rerank_score"] < backend.MIN_RERANK_SCORE:
        return {
            "answer": "I do not have this information.",
            "pages": [],
            "best_section": "Low confidence retrieval",
            "evidence": "",
            "query_type": query_type,
            "raw_items": final_items,
        }

    deterministic = None
    if query_type == "contact":
        deterministic = backend.extract_contact_answer(question, final_items)
    elif query_type == "count":
        deterministic = backend.extract_count_answer(question, final_items, pages)
    elif query_type == "date":
        deterministic = backend.extract_date_answer(question, final_items)

    if deterministic:
        answer, source_pages, evidence = deterministic
    else:
        if groq_client is None:
            answer = "GROQ_API_KEY is not set. Please set it, then reload the app."
            source_pages = sorted({int(item["meta"]["page"]) for item in final_items[:3]})
            evidence = final_items[0]["chunk"]
        else:
            prompt = backend.build_prompt(question, final_items, query_type)
            raw = backend.ask_groq(groq_client, prompt)
            answer, source_pages = backend.parse_answer_and_pages(raw, final_items)
            if not answer:
                answer = "I do not have this information."
            if not backend.verify_answer(answer, final_items, query_type):
                answer = "I do not have this information."
            evidence = final_items[0]["chunk"]

    best_section = final_items[0]["meta"].get("section") or f"Page {final_items[0]['meta']['page']}"
    return {
        "answer": answer,
        "pages": source_pages,
        "best_section": best_section,
        "evidence": evidence,
        "query_type": query_type,
        "raw_items": final_items,
    }


def render_sources(result: Dict[str, Any]):
    pages = result.get("pages") or []
    best_section = result.get("best_section", "")
    evidence = result.get("evidence", "")
    final_items = result.get("raw_items") or []

    if pages:
        st.caption("Sources: " + ", ".join(f"Page {p}" for p in pages))
    if best_section:
        st.caption(f"Best Section: {best_section}")

    if evidence:
        with st.expander("Top evidence snippet"):
            st.write(evidence)

    if final_items:
        with st.expander("Retrieved evidence blocks"):
            for i, item in enumerate(final_items, 1):
                meta = item["meta"]
                st.markdown(
                    f"**{i}. Page {meta['page']} · {meta.get('chunk_type', 'chunk')}**  "
                    f"Rerank: `{item.get('rerank_score', 0):.3f}`"
                )
                if meta.get("section"):
                    st.caption(meta["section"])
                st.write(item["chunk"])
                st.divider()


st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Controls")
    st.write("Use the same folder for:")
    st.code("MCBV9.py\nstreamlit_mcbv9_app.py\nUOS Faculty Handbook 25-26.pdf")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Sample questions")
    for q in DEFAULT_SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.prefill_question = q

    st.subheader("System status")
    if os.getenv("GROQ_API_KEY"):
        st.success("GROQ_API_KEY found")
    else:
        st.warning("GROQ_API_KEY not set")
    if os.getenv("GROQ_MODEL"):
        st.info(f"Model: {os.getenv('GROQ_MODEL')}")

try:
    kb = prepare_knowledge_base()
    st.success(f"Loaded handbook: {Path(kb['pdf_file']).name}")
except Exception as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill_question" not in st.session_state:
    st.session_state.prefill_question = ""

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            render_sources(msg["meta"])

prompt = st.chat_input("Ask a question about the Faculty Handbook...", key="chatbox")
if not prompt and st.session_state.prefill_question:
    prompt = st.session_state.prefill_question
    st.session_state.prefill_question = ""

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_bot(prompt, kb)
        st.markdown(result["answer"])
        render_sources(result)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "meta": result}
    )
