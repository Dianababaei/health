@echo off
echo ========================================
echo QA TEST SETUP - Livestock Health System
echo ========================================
echo.
echo This will:
echo 1. Generate fresh test data (21 days)
echo 2. Clear old database
echo 3. Start dashboard for testing
echo.
echo Test data includes:
echo   - Fever alert (Day 3)
echo   - Heat stress alert (Day 7)
echo   - Inactivity alert (Day 11)
echo   - Estrus detection (Day 15)
echo   - Sensor malfunction (Day 18)
echo   - All behavioral states
echo.
pause
echo.

echo Step 1: Generating test data...
python tools/generate_qa_test_data.py

if errorlevel 1 (
    echo ERROR: Failed to generate test data
    pause
    exit /b 1
)

echo.
echo Step 2: Cleaning old database...
if exist data\alert_state.db (
    del data\alert_state.db
    echo Database deleted
) else (
    echo No old database found (OK)
)

echo.
echo Step 3: Starting dashboard...
echo.
echo ========================================
echo READY FOR TESTING!
echo ========================================
echo.
echo Next steps:
echo 1. Dashboard will open in browser
echo 2. Go to Home page
echo 3. Upload: QA_TEST_001_sensor_data.csv
echo 4. Enter Cow ID: QA_TEST_001
echo 5. Click "Process Upload Data"
echo 6. Follow QA_TESTING_GUIDE.md checklist
echo.
echo See QA_TESTING_GUIDE.md for full test procedure
echo.
echo Press Ctrl+C to stop dashboard when done
echo ========================================
echo.
pause
streamlit run dashboard/app.py
