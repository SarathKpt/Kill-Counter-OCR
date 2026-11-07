@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install -r requirements-build.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onedir --windowed --name "FENIKZ Kill OCR CPU" --distpath releases\apps --workpath releases\build\cpu --specpath releases\build\specs --add-data "%CD%\packaging\config.json;." --collect-all rapidocr --collect-all onnxruntime --collect-all pystray --hidden-import pystray._win32 --hidden-import win32api --hidden-import win32con --hidden-import win32gui --collect-all windows_capture tray_cpu.py
if errorlevel 1 exit /b 1
echo.
echo CPU application is in releases\apps\FENIKZ Kill OCR CPU.
echo Copy the complete folder when distributing it; do not move only the EXE.
pause
