@echo off
setlocal
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 py -m venv .venv
if errorlevel 1 (
  echo Could not create a virtual environment. Make sure Python is available from Command Prompt as "python" or "py".
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo.
echo Installation complete. Start the service with run.bat.
pause
