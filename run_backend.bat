@echo off
chcp 65001 >nul
echo [INFO] Starting Backend...
set PYTHONPATH=%cd%
python app/api/main.py
pause
