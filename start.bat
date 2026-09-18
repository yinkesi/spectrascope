@echo off
rem SpectraScope 启动脚本
cd /d %~dp0
start "SpectraScope" python -m uvicorn app.main:app --port 8765
timeout /t 2 >nul
start "" http://127.0.0.1:8765
