@echo off
echo Installing PyTorch for SWTBot Fine-tuning Pipeline...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found, proceeding with PyTorch installation...
echo.

REM Run the Python installation script
python install_pytorch.py

echo.
echo Installation completed!
echo.
echo Next steps:
echo 1. Run: python scripts/setup_env.py
echo 2. Run: python scripts/collect_data.py  
echo 3. Run: python scripts/train.py
echo.
pause
