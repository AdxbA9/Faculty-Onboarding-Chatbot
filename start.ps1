# One-shot setup + run script for Windows PowerShell.
# Usage: right-click -> Run with PowerShell, or: .\start.ps1
#
# If PowerShell blocks the script, run this once (in an admin PS):
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    py -m venv .venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install --upgrade pip | Out-Null
pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "Created .env from template. Edit it and add your GROQ_API_KEY, then re-run this script." -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

Write-Host "Starting the app on http://localhost:8501 ..." -ForegroundColor Green
python app.py
