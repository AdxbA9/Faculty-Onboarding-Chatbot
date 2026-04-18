# MiniV5

## Overview
MiniV5 represents the major refinement stage of the Faculty Onboarding Chatbot project. Instead of one single script, this stage contains multiple internal versions that show the evolution of the backend from a standalone RAG script to improved Groq-based versions and a Streamlit UI.

## Files
- `MCBV6.py` — standalone optimized RAG script
- `MCBV7.py` — Groq-based generation version
- `MCBV8.py` — stronger retrieval and structured-content handling
- `MCBV9.py` — strongest backend version with routing and verification
- `streamlit_mcbv9_app.py` — Streamlit UI for MCBV9
- `requirements_mcbv7.txt`
- `requirements_mcbv8.txt`
- `requirements_mcbv9.txt`
- `requirements_streamlit_ui.txt`
- `UOS Faculty Handbook 25-26.pdf`

## Purpose of this folder
This folder keeps the internal development history of the chatbot backend in one place without duplicating the handbook PDF or creating unnecessary separate top-level project folders.

## Internal progression
- `MCBV6` — optimized standalone RAG script
- `MCBV7` — moved generation to Groq
- `MCBV8` — improved retrieval and structured/table handling
- `MCBV9` — strongest backend with routing, exact lookup, and verification
- `streamlit_mcbv9_app.py` — user interface layer for MCBV9

## Notes
MiniV5 is kept as one development stage because all versions share the same handbook source and belong to the same backend refinement phase.