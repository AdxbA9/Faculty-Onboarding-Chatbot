@echo off
rem Installs Python 3.11 (official python.org installer, current user only) if missing, then runs the test cases.
cd /d "%~dp0..\.."
set R=qa\test_cases_report
del /q "%R%\DONE.txt" 2>nul
echo start %date% %time% > "%R%\console.txt"
set PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
if not exist "%PY311%" (
  echo downloading python 3.11.9 from python.org >> "%R%\console.txt"
  curl.exe -L -s -o "%TEMP%\python-3.11.9-amd64.exe" https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe >> "%R%\console.txt" 2>&1
  "%TEMP%\python-3.11.9-amd64.exe" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 >> "%R%\console.txt" 2>&1
)
if not exist "%PY311%" (
  echo PYTHON INSTALL FAILED >> "%R%\console.txt"
  echo failed install > "%R%\DONE.txt"
  exit /b 1
)
"%PY311%" --version >> "%R%\console.txt" 2>&1
if not exist ".venv_tests\Scripts\python.exe" "%PY311%" -m venv .venv_tests >> "%R%\console.txt" 2>&1
".venv_tests\Scripts\python.exe" -m pip install --upgrade pip >> "%R%\console.txt" 2>&1
".venv_tests\Scripts\python.exe" -m pip install -r requirements.txt >> "%R%\console.txt" 2>&1
echo deps done %time% >> "%R%\console.txt"
".venv_tests\Scripts\python.exe" -X utf8 "%R%\run_test_cases.py" >> "%R%\console.txt" 2>&1
if not exist "%R%\DONE.txt" echo failed %ERRORLEVEL% > "%R%\DONE.txt"
