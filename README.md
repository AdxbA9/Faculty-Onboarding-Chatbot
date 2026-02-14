

This project is a simple Python-based chatbot designed to assist faculty members during onboarding at the University of Sharjah.

The chatbot provides responses based on official university documents, primarily the Faculty Handbook (2025–2026).


1️ Clone the Repository

```bash
git clone https://github.com/R4shed-1/Faculty-Onboarding-Chatbot.git
cd Faculty-Onboarding-Chatbot
```
2️ Install Required Libraries

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist, install manually:

```bash
pip install fastapi uvicorn PyPDF2
```
How to Run (Command Line Version)
```bash
python chatbot_reader.py
```
You will see:

```
Custom Data Chatbot (type 'exit' to quit)
```

Type your question and press Enter.

To exit, type:

```
exit
```

---
How to Run (Web Version)
Run:
```bash
uvicorn app:app --reload
```
Then open your browser and go to:
```
http://127.0.0.1:8000
```
You can now interact with the chatbot in the browser.
---
Project Structure
```
Mini_Chatbot/
│
├── app.py                 # FastAPI web app
├── chatbot_reader.py      # Core chatbot logic
├── Faculty_Guide.pdf      # Data source
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```
---
 -Data Source
The chatbot uses:
Official University of Sharjah Faculty Handbook (2025–2026)
No personal or sensitive data is collected or processed.

-Data Privacy
* Only official university documents are used.
* No personal data storage.
* No external APIs used.

Then run:

```bash
git add README.md
git commit -m "Add README documentation"
git push
```
