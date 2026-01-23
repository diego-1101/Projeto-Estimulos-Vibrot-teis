@echo off
echo ========================================
echo EEG PSD Dashboard - Quick Start
echo ========================================
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting Dashboard...
echo ========================================
echo.
echo Dashboard will be available at:
echo http://localhost:8050
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
