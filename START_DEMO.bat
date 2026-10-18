@echo off
setlocal
cd /d "%~dp0"
title AUTONIQ Cafe Monitor - Start Demo

echo ============================================================
echo   AUTONIQ Cafe Monitor - CPU-Optimized Demo Launcher
echo ============================================================
echo.
echo This project update does not contain or replace the data folder.
echo Existing CCTV videos and cup images remain in data\.
echo.

docker --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker CLI was not found. Install Docker Desktop first.
  pause
  exit /b 1
)
docker compose version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker Compose v2 was not found. Update Docker Desktop.
  pause
  exit /b 1
)
docker info >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker Desktop is installed but its engine is not responding.
  echo Open Docker Desktop, wait until the engine says Running, then launch this file again.
  pause
  exit /b 1
)

if not exist ".env" (
  echo Creating local .env from .env.example...
  copy /Y ".env.example" ".env" >nul
  if errorlevel 1 (
    echo ERROR: Could not create .env. Check project folder permissions.
    pause
    exit /b 1
  )
)

echo [1/7] Applying CPU-friendly settings. Existing RTSP credentials are preserved...
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\apply_low_spec_env.ps1"
if errorlevel 1 (
  echo ERROR: Could not update performance settings in .env.
  pause
  exit /b 1
)

echo.
echo [2/7] Building backend and dashboard images. First build may take several minutes...
docker compose build backend frontend
if errorlevel 1 (
  echo ERROR: Image build failed. Review the output and README.md.
  pause
  exit /b 1
)

echo.
echo [3/7] Starting app services so the preparation tools can use their volumes...
docker compose up -d
if errorlevel 1 (
  echo ERROR: Docker Compose could not start. Run: docker compose logs --tail 120
  pause
  exit /b 1
)

echo.
echo [4/7] Preparing low-FPS preview copies from recordings.
echo This first-time step may take several minutes; it never changes original files in data\.
docker compose run --rm backend python /workspace/scripts/prepare_demo_proxies.py
if errorlevel 1 (
  echo WARNING: Preview conversion did not finish. Original recordings remain untouched.
  echo The app may fall back to original footage and use more CPU.
)

echo.
echo [5/7] Preparing an Intel OpenVINO model where supported.
echo This runs only once per model volume. If it fails, PyTorch remains the fallback.
docker compose run --rm backend python /workspace/scripts/prepare_openvino_model.py
if errorlevel 1 (
  echo WARNING: OpenVINO preparation returned an error. The app will attempt PyTorch fallback.
)

echo.
echo [6/7] Restarting services so the new caches and model are used...
docker compose restart backend diagnostics

echo.
echo [7/7] Checking backend health...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$deadline=(Get-Date).AddSeconds(120); do { try { $r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); exit 1"
if errorlevel 1 (
  echo WARNING: Backend did not become healthy within 120 seconds.
  echo Check: docker compose ps
  echo Check: docker compose logs --tail 120 backend
) else (
  echo Backend is healthy.
)

echo.
echo Main dashboard: http://localhost:8080
echo Diagnostics:    http://localhost:8090
echo API docs:       http://localhost:8000/docs
echo.
echo IMPORTANT: Start AI on one camera at a time for this CPU.
echo For seated people, select Camera 03 (Main Hall). It uses 2x2 overlapping image tiles.
echo For total inside occupancy, select Camera 07 and set the manually verified opening count first.
start "" "http://localhost:8080"
start "" "http://localhost:8090"
echo To stop services, double-click STOP_DEMO.bat.
echo.
pause
exit /b 0
