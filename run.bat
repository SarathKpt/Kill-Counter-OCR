@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install.bat first.
  pause
  exit /b 1
)
start "" "http://127.0.0.1:8765/control"
:run_server
.venv\Scripts\python.exe app.py
if %ERRORLEVEL%==23 goto run_server
pause
