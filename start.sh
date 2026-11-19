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
pip install -r requirements.txt -c constraints.txt || pip install -r requirements.txt

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo
    echo "Created .env from template. Edit it and add your GROQ_API_KEY, then re-run this script."
    echo
    exit 0
fi

# The four-specialist answer path with the Blackboard / iThenticate guides (the mode every QA run used).
export PLAN_F_ENABLED="${PLAN_F_ENABLED:-1}"
export PLAN_F_GUIDE_SCOPES="${PLAN_F_GUIDE_SCOPES:-1}"
export REFUSAL_RESCUE="${REFUSAL_RESCUE:-1}"
export LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-low}"

echo "Starting the app on http://localhost:8501 ..."
python app.py
