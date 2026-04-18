\# MiniV1



\## Overview

MiniV1 is the first working version of the Faculty Onboarding Chatbot. It is a terminal-based RAG chatbot that answers questions from the University of Sharjah Faculty Handbook.



\## Files

\- `rag\_chatbot.py` — main chatbot script

\- `handbook.pdf` — source document used by the chatbot



\## Main features

\- PDF text extraction

\- text chunking with overlap

\- embeddings using `BAAI/bge-base-en-v1.5`

\- FAISS retrieval

\- Cross-Encoder reranking

\- answer generation using `google/flan-t5-base`

\- source page display



\## Improvements planned for later versions

\- improved accuracy

\- better handling of structured tables

\- better citation quality

\- UI integration

