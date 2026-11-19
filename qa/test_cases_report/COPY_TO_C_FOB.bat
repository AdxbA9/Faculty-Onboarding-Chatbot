@echo off
rem Copies the project to the short path C:\fob (avoids Windows long-path errors) and launches the chatbot from there.
set LOG=%~dp0copy_log.txt
pushd "%~dp0..\.."
set SRC=%CD%
popd
echo start %date% %time% > "%LOG%"
echo source %SRC% >> "%LOG%"
robocopy "%SRC%" "C:\fob" /E /XD .venv .venv_tests .venv_broken_backup __pycache__ .pytest_cache /R:1 /W:1 /NFL /NDL /NP >> "%LOG%" 2>&1
echo robocopy exit %ERRORLEVEL% >> "%LOG%"
if exist "C:\fob\app.py" (
  start "UOS Chatbot" /D "C:\fob" "C:\fob\START_CHATBOT.bat"
  echo launched >> "%LOG%"
) else (
  echo COPY FAILED >> "%LOG%"
)
