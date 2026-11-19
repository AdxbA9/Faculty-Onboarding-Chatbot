@echo off
rem Runs the system test cases. The venv lives at a short path to avoid the Windows 260-character path limit.
cd /d "%~dp0..\.."
set R=qa\test_cases_report
set V=%LOCALAPPDATA%\fobv
del /q "%R%\DONE.txt" 2>nul
echo start %date% %time% > "%R%\console.txt"
set PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
if not exist "%PY311%" (
  echo PYTHON MISSING >> "%R%\console.txt"
  echo failed nopython > "%R%\DONE.txt"
  exit /b 1
)
if not exist "%V%\Scripts\python.exe" "%PY311%" -m venv "%V%" >> "%R%\console.txt" 2>&1
"%V%\Scripts\python.exe" -m pip install --upgrade pip >> "%R%\console.txt" 2>&1
"%V%\Scripts\python.exe" -m pip install -r requirements.txt >> "%R%\console.txt" 2>&1
echo deps done %time% >> "%R%\console.txt"
"%V%\Scripts\python.exe" -X utf8 "%R%\run_test_cases.py" >> "%R%\console.txt" 2>&1
if not exist "%R%\DONE.txt" echo failed %ERRORLEVEL% > "%R%\DONE.txt"
