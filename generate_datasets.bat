@echo off
REM Generate all synthetic datasets for Windows

echo ==========================================
echo Synthetic Dataset Generation
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import numpy, pandas" >nul 2>&1
if errorlevel 1 (
    echo Error: Required packages (numpy, pandas^) not installed
    echo Run: pip install -r requirements.txt
    exit /b 1
)

echo [OK] Dependencies installed
echo.

REM Create output directory
if not exist "data\synthetic" mkdir "data\synthetic"

REM Run the dataset generator
echo Running dataset generator...
cd src\data
python dataset_generator.py

if %errorlevel% equ 0 (
    echo.
    echo ==========================================
    echo [OK] Dataset generation complete!
    echo ==========================================
    echo.
    echo Generated files:
    echo   - data\synthetic\train.csv
    echo   - data\synthetic\val.csv
    echo   - data\synthetic\test.csv
    echo   - data\synthetic\multiday_*.csv
    echo   - data\synthetic\dataset_metadata.json
    echo.
) else (
    echo.
    echo [ERROR] Dataset generation failed
    exit /b 1
)
