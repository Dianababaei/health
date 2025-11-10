@echo off
REM Artemis Health Dashboard Startup Script for Windows

echo Starting Artemis Health Dashboard...
echo.

REM Check if streamlit is installed
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Streamlit is not installed.
    echo Please install dependencies: pip install -r requirements.txt
    exit /b 1
)

REM Run the dashboard
echo Launching dashboard at http://localhost:8501
echo.
streamlit run dashboard/app.py
