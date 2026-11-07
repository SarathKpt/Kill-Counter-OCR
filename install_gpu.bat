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
echo The GPU environment installs the CUDA and cuDNN runtime packages used by ONNX Runtime.
echo Keep your NVIDIA display driver current; a separate CUDA toolkit is usually not required.
echo The app reports the active inference provider and any CPU fallback.
pause
