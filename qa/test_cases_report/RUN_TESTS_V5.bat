@echo off
rem Same as V4 but pins the package versions used by the project's original .venv (older builds that Windows Smart App Control accepts).
cd /d "%~dp0..\.."
set R=qa\test_cases_report
set V=%LOCALAPPDATA%\fobv
del /q "%R%\DONE.txt" 2>nul
echo start %date% %time% > "%R%\console.txt"
"%V%\Scripts\python.exe" -m pip install -r "%R%\pins.txt" >> "%R%\console.txt" 2>&1
echo pins done %time% >> "%R%\console.txt"
"%V%\Scripts\python.exe" -c "import pydantic, fastapi, nicegui, numpy, torch, faiss, fitz, sentence_transformers, groq; print('IMPORTS OK')" >> "%R%\console.txt" 2>&1
"%V%\Scripts\python.exe" -X utf8 "%R%\run_test_cases.py" >> "%R%\console.txt" 2>&1
if not exist "%R%\DONE.txt" echo failed %ERRORLEVEL% > "%R%\DONE.txt"
