@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [Setup] Creating virtual environment...
    python -m venv .venv || (echo Python not found. Please install Python 3.10+ first. & pause & exit /b 1)
)

if not exist ".venv\.installed" (
    echo [Setup] Installing packages...
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || (echo Package install failed. & pause & exit /b 1)
    type nul > ".venv\.installed"
)

echo Starting app (default port 8501, auto-increments if busy)
".venv\Scripts\python.exe" -m streamlit run app_main.py
pause
