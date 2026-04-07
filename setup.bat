@echo off
chcp 65001 >nul
rem Setup Script
echo [INFO] checking python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] python not found, please install python 3.10 or higher and add it to the environment variables.
    pause
    exit /b 1
)

echo [INFO] installing python dependencies...
python -m pip install -r requirements.txt

echo [INFO] checking frontend dependencies...
if exist "frontend\package.json" (
    cd frontend
    echo [INFO] installing frontend dependencies...
    call npm install
    cd ..
) else (
    echo [WARN] 
)

echo [INFO] All Done.
pause
