\# MCBV9



\## Overview

MCBV9 is the strongest backend version in the MiniV5 stage. It adds query routing, exact lookup logic, and a verification layer to improve answer reliability.



\## Main changes from MCBV8

\- query routing by question type

\- stronger exact lookup for contacts, dates, and counts

\- verification layer before final answer

\- cleaner source handling

\- safer fallback behavior



\## Improvement focus

MCBV9 aims to reduce unsupported answers and improve reliability for structured factual questions.



\## Related UI

The Streamlit interface for this version is provided in `streamlit\_mcbv9\_app.py`.

