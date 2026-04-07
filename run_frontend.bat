@echo off
chcp 65001 >nul
echo [INFO] Starting Frontend...
cd frontend
npm run dev
pause
