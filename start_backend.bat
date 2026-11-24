@echo off
cd /d "%~dp0"
echo ============================================================
echo   MedAssist-AI Agentic System -- Backend Startup
echo ============================================================
echo.

REM Set UTF-8 encoding to avoid crashes on Windows
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM .env is optional. Without it, the backend uses local Ollama.

set "PYTHON=python"
if exist "backend\venv\Scripts\python.exe" set "PYTHON=%CD%\backend\venv\Scripts\python.exe"

REM Check if indexes exist
if not exist "data\processed_documents\faiss_index.bin" (
    echo [INFO] No index found. Building index from raw PDFs...
    cd backend
    "%PYTHON%" -m rag.indexer
    if errorlevel 1 (
        echo [ERROR] Indexing failed.
        exit /b 1
    )
    cd ..
    echo.
)

echo [INFO] Starting FastAPI backend on http://127.0.0.1:8000
echo [INFO] Press Ctrl+C to stop
echo.
cd backend
"%PYTHON%" -m uvicorn main:app --reload --port 8000 --host 127.0.0.1
