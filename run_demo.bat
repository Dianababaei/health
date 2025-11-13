@echo off
echo ========================================
echo ARTEMIS DEMO - Livestock Health Monitor
echo ========================================
echo.
echo Demo data is already loaded!
echo.
echo Opening dashboard with:
echo   - 14 days of sensor data
echo   - Fever scenario (Day 3)
echo   - Real-time alerts
echo.
echo Press Ctrl+C to stop
echo.
pause
echo.
echo Starting dashboard...
streamlit run dashboard/app.py
