@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-gpu\Scripts\python.exe" (
  python -m venv .venv-gpu
  if errorlevel 1 py -m venv .venv-gpu
)
if errorlevel 1 (
  echo Could not create the GPU environment. Make sure Python is available as "python" or "py".
  pause
  exit /b 1
)
.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
if errorlevel 1 exit /b 1
echo.
echo GPU environment installed. Start it with run_gpu.bat.
echo For CUDA, install the NVIDIA CUDA 13 toolkit first; the display driver alone is not enough.
echo The app will report CPU fallback if required CUDA libraries are missing.
pause
