@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-gpu\Scripts\python.exe" (
  echo Run install_gpu.bat first.
  pause
  exit /b 1
)
.venv-gpu\Scripts\python.exe -m pip install -r requirements-build.txt
if errorlevel 1 exit /b 1
.venv-gpu\Scripts\python.exe -m PyInstaller --noconfirm --clean --onedir --windowed --name "FENIKZ Kill OCR GPU" --distpath releases\apps --workpath releases\build\gpu --specpath releases\build\specs --add-data "%CD%\packaging\config.json;." --add-data "%CD%\.venv-gpu\Lib\site-packages\nvidia;nvidia" --collect-all rapidocr --collect-all onnxruntime --collect-all pystray --hidden-import pystray._win32 --hidden-import win32api --hidden-import win32con --hidden-import win32gui --collect-all windows_capture --collect-all nvidia.cudnn tray_gpu.py
if errorlevel 1 exit /b 1
echo.
echo GPU application is in releases\apps\FENIKZ Kill OCR GPU.
echo Copy the complete folder when distributing it; do not move only the EXE.
echo GPU mode requires a compatible NVIDIA driver. The GPU build includes the CUDA/cuDNN runtime libraries.
pause
