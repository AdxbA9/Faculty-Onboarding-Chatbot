@echo off
cd /d "%~dp0..\.."
del /q "qa\test_cases_report\DONE.txt" 2>nul
".venv\Scripts\python.exe" -X utf8 "qa\test_cases_report\run_test_cases.py" > "qa\test_cases_report\console.txt" 2>&1
if not exist "qa\test_cases_report\DONE.txt" echo failed %ERRORLEVEL% > "qa\test_cases_report\DONE.txt"
