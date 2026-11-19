@echo off
rem Starts the chatbot. Run SETUP.bat once first (this file offers to do it when the environment is missing).
title UOS Faculty Onboarding Chatbot
cd /d "%~dp0"
set "PY=%LOCALAPPDATA%\fobv\Scripts\python.exe"
if not exist "%PY%" (
  echo The Python environment is not installed yet. Running SETUP.bat first...
  call "%~dp0SETUP.bat"
)
if not exist "%PY%" (
  echo Setup did not finish. Run SETUP.bat and read its messages.
  pause
  exit /b 1
)
if not exist ".env" (
  copy /y ".env.example" ".env" >nul
  echo Created ".env". Open it in Notepad, add your GROQ_API_KEY, then start again.
  pause
  exit /b 1
)
rem The four-specialist answer path with the Blackboard / iThenticate guides (the mode every QA run used).
set PLAN_F_ENABLED=1
set PLAN_F_GUIDE_SCOPES=1
set REFUSAL_RESCUE=1
if not defined LLM_REASONING_EFFORT set LLM_REASONING_EFFORT=low
echo Starting the UOS Faculty Onboarding Chatbot...
echo The first start downloads the language models (about 500 MB) and can take several minutes.
echo When it says the app is ready, open http://localhost:8501 in your browser.
echo Close this window to stop the chatbot.
"%PY%" -X utf8 app.py
pause
