\# MiniV3



\## Overview

MiniV3 is a notebook-based version of the Faculty Onboarding Chatbot. It represents a more structured and cleaner implementation of the RAG pipeline compared with the earlier terminal script versions.



\## Files

\- `junior.ipynb` — Jupyter Notebook version of the chatbot

\- `UOS Faculty Handbook 25-26.pdf` — source handbook document

\- `README.md` — version notes



\## Main features

\- PDF text extraction using `pypdf`

\- chunking with overlap

\- embeddings using `BAAI/bge-base-en-v1.5`

\- FAISS retrieval with cosine similarity

\- Cross-Encoder reranking using `ms-marco-MiniLM-L-6-v2`

\- answer generation using `google/flan-t5-base`

\- source page display



\## Improvements from MiniV2

\- moved from a plain script format to a notebook-based prototype

\- clearer step-by-step pipeline structure

\- improved validation for missing PDF file and empty chunks

\- improved handling of empty user input

\- explicit conversion of embeddings to `np.float32` before FAISS indexing

\- deduplicated source pages before output

\- updated source handbook file name to `UOS Faculty Handbook 25-26.pdf`



\## Notes

MiniV3 is still a terminal-style chatbot flow inside a notebook, but it is more organized and safer than the earlier versions. It serves as an intermediate development stage before later optimization and UI integration.

