#!/usr/bin/env bash
# One-shot setup + run script for macOS / Linux.
# Usage: ./start.sh   (run `chmod +x start.sh` first)

set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo
    echo "Created .env from template. Edit it and add your GROQ_API_KEY, then re-run this script."
    echo
    exit 0
fi

echo "Starting the app on http://localhost:8501 ..."
python app.py
