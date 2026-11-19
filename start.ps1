# Setup + run for Windows PowerShell. Same as SETUP.bat followed by START_CHATBOT.bat.
# Usage: .\start.ps1
#
# The Python environment lives in %LOCALAPPDATA%\fobv, not inside the project: an environment in a deep project
# folder fails on Windows with "file path too long" (see SETUP.bat).
#
# If PowerShell blocks the script, run this once:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = Join-Path $env:LOCALAPPDATA "fobv\Scripts\python.exe"
if (-not (Test-Path $python) -or -not (Test-Path ".env")) {
    cmd /c "`"$PSScriptRoot\SETUP.bat`""
}
if (-not (Test-Path $python)) {
    Write-Host "Setup did not finish. Run SETUP.bat and read its messages." -ForegroundColor Yellow
    exit 1
}

$env:PLAN_F_ENABLED = "1"
$env:PLAN_F_GUIDE_SCOPES = "1"
$env:REFUSAL_RESCUE = "1"
if (-not $env:LLM_REASONING_EFFORT) { $env:LLM_REASONING_EFFORT = "low" }
Write-Host "Starting the app on http://localhost:8501 ..." -ForegroundColor Green
& $python -X utf8 app.py
