@echo off
cd /d "%~dp0..\.."
set R=qa\test_cases_report
del /q "%R%\DONE.txt" 2>nul
echo start %date% %time% > "%R%\console.txt"
set PYEXE=
where py >> "%R%\console.txt" 2>&1 && set PYEXE=py -3
if "%PYEXE%"=="" where python >> "%R%\console.txt" 2>&1 && set PYEXE=python
if "%PYEXE%"=="" (
  for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if exist "%%D\python.exe" set PYEXE="%%D\python.exe"
)
if "%PYEXE%"=="" (
  echo NO PYTHON FOUND >> "%R%\console.txt"
  echo failed nopython > "%R%\DONE.txt"
  exit /b 1
)
echo using %PYEXE% >> "%R%\console.txt"
%PYEXE% --version >> "%R%\console.txt" 2>&1
if not exist ".venv_tests\Scripts\python.exe" (
  %PYEXE% -m venv .venv_tests >> "%R%\console.txt" 2>&1
)
".venv_tests\Scripts\python.exe" -m pip install --upgrade pip >> "%R%\console.txt" 2>&1
".venv_tests\Scripts\python.exe" -m pip install -r requirements.txt >> "%R%\console.txt" 2>&1
echo deps done %time% >> "%R%\console.txt"
".venv_tests\Scripts\python.exe" -X utf8 "%R%\run_test_cases.py" >> "%R%\console.txt" 2>&1
if not exist "%R%\DONE.txt" echo failed %ERRORLEVEL% > "%R%\DONE.txt"
