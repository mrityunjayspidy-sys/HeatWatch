@echo off
echo ====================================================
echo Starting HeatWatch AI - Urban Heat Intelligence Platform
echo ====================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Starting backend server on http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
"%PYTHON_EXE%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
