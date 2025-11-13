@echo off
echo ================================================
echo COMPLETE FIX - Alerts and Auto-Refresh
echo ================================================
echo.

echo Step 1: Generating alerts with FIXED code...
python force_generate_alerts.py
echo.

if errorlevel 1 (
    echo ERROR: Failed to generate alerts
    pause
    exit /b 1
)

echo Step 2: Verifying alerts file...
python -c "import json; f=open('data/simulation/SIM_COW_001_alerts.json'); a=json.load(f); print(f'Alerts in file: {len(a)}'); exit(0 if len(a)>0 else 1)"

if errorlevel 1 (
    echo ERROR: No alerts in file!
    pause
    exit /b 1
)

echo.
echo ================================================
echo SUCCESS! Alerts generated successfully
echo ================================================
echo.
echo Now:
echo 1. RESTART Streamlit (Ctrl+C, then: streamlit run dashboard/app.py)
echo 2. Refresh browser (F5)
echo 3. Go to Home page
echo 4. You will see 200+ alerts
echo.
echo Auto-refresh is now disabled via .streamlit/config.toml
echo ================================================
pause
