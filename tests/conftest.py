"""
Shared fixtures for the Milestone 1 test suite.

Nothing here touches the network. The two Hugging Face models are replaced by
tiny local substitutes, the Groq client by a scripted fake, and the handbook by
a six-chunk corpus that covers every route the orchestrator can take.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List

import numpy as np
import pytest

CORPUS = [
    ("Faculty members are entitled to thirty days of annual leave each academic year "
     "subject to approval by the dean.", 3, "Leave", "paragraph"),
    ("Information Technology Center | +971 6 5050000 | itcenter@sharjah.ac.ae", 14, "Contacts", "row"),
    ("Mon | 25 Aug | 02 Rabi I | Classes begin", 49, "Calendar", "row"),
    ("The University Council is chaired by the Chancellor and meets twice each semester.",
     7, "Governance", "paragraph"),
    ("Promotion to associate professor requires a minimum of five years in rank and a "
     "record of research.", 40, "Promotion", "paragraph"),
    ("UoS offers a total of 149 degree programs: 62 Bachelor's, 60 Master's and 23 PhD programs.",
     15, "About", "paragraph"),
]

_STOP = {"the", "a", "an", "is", "are", "of", "for", "to", "in", "on", "at", "and", "or",
         "by", "with", "what", "which", "who", "when", "where", "how", "does", "do", "can"}


def _toks(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


class FakeEmbedder:
    """Hashed bag-of-words embedding: deterministic, no model download."""
    dim = 256

    def encode(self, texts, **_):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in _toks(text):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
            norm = np.linalg.norm(out[i]) or 1.0
            out[i] /= norm
        return out


class FakeReranker:
    """Token-overlap score; ``fixed`` forces every score (to test the refusal gate)."""

    def __init__(self, fixed=None):
        self.fixed = fixed

    def predict(self, pairs):
        if self.fixed is not None:
            return [self.fixed] * len(pairs)
        return [float(len(set(_toks(q)) & set(_toks(c)))) for q, c in pairs]


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.prompts: List[str] = []

    def create(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs["messages"][-1]["content"])
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return _Resp(reply)


class _Chat:
    def __init__(self, replies):
        self.completions = _Completions(replies)


class FakeGroqClient:
    """Scripted stand-in for ``groq.Groq``. Each call returns the next reply;
    the last reply repeats. An Exception in the list is raised by that call."""

    def __init__(self, *replies):
        self.chat = _Chat(replies or ["I do not have this information."])

    def with_options(self, **_):
        return self

    @property
    def calls(self) -> int:
        return self.chat.completions.calls

    @property
    def prompts(self) -> List[str]:
        return self.chat.completions.prompts


def make_items(*rows) -> List[Dict]:
    """Evidence items in the shape retrieval produces: (chunk, page[, score])."""
    items = []
    for row in rows:
        chunk, page = row[0], row[1]
        score = row[2] if len(row) > 2 else 1.0
        items.append({"chunk": chunk, "meta": {"page": page, "section": "", "chunk_type": "paragraph"},
                      "rerank_score": score})
    return items


@pytest.fixture(scope="session")
def corpus_items() -> List[Dict]:
    return [{"chunk": text, "meta": {"page": page, "section": section, "chunk_type": ctype},
             "rerank_score": 1.0}
            for text, page, section, ctype in CORPUS]


@pytest.fixture(scope="session")
def kb():
    """Everything ``answer_question`` needs, built once per test session."""
    import faiss

    chunks = [row[0] for row in CORPUS]
    metadata = [{"page": row[1], "section": row[2], "chunk_type": row[3]} for row in CORPUS]
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    pages = [{"page": i + 1, "text": ""} for i in range(50)]
    return dict(embedder=embedder, reranker=FakeReranker(), index=index,
                chunks=chunks, metadata=metadata, pages=pages)


@pytest.fixture
def ask(kb):
    """Call the public pipeline with the fake knowledge base."""
    from handbook_bot.qa import answer_question

    def _ask(question, client=None, **overrides):
        kwargs = dict(kb)
        kwargs.update(overrides)
        return answer_question(question, groq_client=client, **kwargs)

    return _ask


@pytest.fixture
def run(kb):
    """Call the orchestrator directly (needed to inject a verifier)."""
    from handbook_bot import orchestrator

    def _run(question, client=None, verifier=None, **overrides):
        kwargs = dict(kb)
        kwargs.update(overrides)
        return orchestrator.run(question, groq_client=client, verifier=verifier, **kwargs)

    return _run


GROUNDED_LEAVE = ("Faculty members are entitled to thirty days of annual leave each academic year."
                  "\nPages: 3")
FABRICATED_LEAVE = "Faculty members receive a housing allowance of 5000 dirhams each year.\nPages: 3"
LEAVE_QUESTION = "How much annual leave are faculty members entitled to?"
