@echo off
rem One-time setup for Windows. Double-click it, wait until it says "Setup finished".
rem
rem The Python environment is created in %LOCALAPPDATA%\fobv, NOT inside this folder. Windows refuses paths longer than
rem 260 characters and the libraries (torch, transformers) contain files that need 145 characters below the environment
rem folder, so an environment inside a deep project folder fails with "file path too long". A short, fixed location
rem works wherever this project folder is.
setlocal
title UOS Faculty Onboarding Chatbot - setup
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\fobv"
set "PY=%VENV%\Scripts\python.exe"

if exist "%PY%" goto have_env

set "BASEPY="
for %%V in (3.11 3.12 3.10) do (
  if not defined BASEPY (
    py -%%V -c "import sys" >nul 2>&1 && set "BASEPY=py -%%V"
  )
)
if not defined BASEPY (
  python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "BASEPY=python"
)
if not defined BASEPY (
  echo.
  echo Python 3.10, 3.11 or 3.12 was not found.
  echo Install Python 3.11 from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then run SETUP.bat again.
  echo.
  pause
  exit /b 1
)
echo Creating the Python environment in %VENV% ...
%BASEPY% -m venv "%VENV%"
if errorlevel 1 (
  echo Could not create the environment.
  pause
  exit /b 1
)

:have_env
echo Installing the libraries ^(the first time this downloads about 2 GB and takes several minutes^)...
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt -c constraints.txt
if errorlevel 1 (
  echo.
  echo The tested versions could not be installed; trying the newest compatible versions instead...
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Installation failed. Check the internet connection and run SETUP.bat again.
    pause
    exit /b 1
  )
)

if not exist ".env" (
  copy /y ".env.example" ".env" >nul
  echo.
  echo Created the settings file ".env".
  echo Open it in Notepad and replace "your_groq_api_key_here" with a key from https://console.groq.com/keys
)

echo.
echo Setup finished. Double-click START_CHATBOT.bat to start the chatbot.
echo.
pause
