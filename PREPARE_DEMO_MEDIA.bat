@echo off
setlocal
cd /d "%~dp0"
title AUTONIQ Cafe Monitor - Prepare Low-CPU Demo Media

echo ============================================================
echo   Preparing cached demo videos and optimized CPU model
echo ============================================================
echo Original recordings in data\ are never edited or removed.
echo This preparation is needed only on first run or after a recording changes.
echo.

docker info >nul 2>&1
if errorlevel 1 (
  echo ERROR: Start Docker Desktop and wait until the engine is running.
  pause
  exit /b 1
)
if not exist ".env" copy /Y ".env.example" ".env" >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\apply_low_spec_env.ps1"
if errorlevel 1 goto fail

echo Building the backend image if required...
docker compose build backend
if errorlevel 1 goto fail

echo.
echo [1/2] Preparing reduced-FPS video proxies. This can take time on the first run.
docker compose run --rm backend python /workspace/scripts/prepare_demo_proxies.py
if errorlevel 1 goto fail

echo.
echo [2/2] Preparing an Intel OpenVINO model when supported. PyTorch fallback remains available.
docker compose run --rm backend python /workspace/scripts/prepare_openvino_model.py
if errorlevel 1 goto fail

echo.
echo Preparation complete. Run START_DEMO.bat when ready.
pause
exit /b 0

:fail
echo.
echo Preparation did not complete. Review the error above.
pause
exit /b 1
