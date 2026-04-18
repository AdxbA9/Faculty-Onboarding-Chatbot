\# MiniV4



\## Overview

MiniV4 is a standalone Python-script version of the Faculty Onboarding Chatbot. It is a more mature RAG prototype than MiniV3 and introduces caching, better preprocessing, and fallback handling for structured factual questions.



\## Files

\- `MCBV6.py` — main chatbot script

\- `UOS Faculty Handbook 25-26.pdf` — source handbook document

\- `README.md` — version notes



\## Main features

\- automatic PDF detection in the current folder

\- PDF text extraction using `pypdf`

\- repeated header/footer cleaning

\- chunking with overlap

\- embeddings using `sentence-transformers/all-mpnet-base-v2`

\- FAISS retrieval with cosine similarity

\- Cross-Encoder reranking using `ms-marco-MiniLM-L-12-v2`

\- answer generation using `google/flan-t5-xl`

\- embedding caching for faster repeated runs

\- fallback extraction for phone-number questions

\- fallback extraction for count/number questions

\- best source page and evidence snippet output



\## Improvements from MiniV3

\- moved from notebook format to a standalone Python script

\- added automatic PDF detection

\- added repeated header/footer removal

\- added cached embeddings to reduce recomputation

\- added deduplication of reranked chunks

\- added fallback extraction for phone/contact questions

\- added fallback extraction for count questions

\- improved prompt design and generation settings

\- improved evidence and source presentation



\## Notes

MiniV4 represents a more practical and optimized command-line chatbot version. It is a major transition from an experimental notebook prototype to a reusable standalone RAG script.

