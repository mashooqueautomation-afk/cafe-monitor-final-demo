@echo off
setlocal
cd /d "%~dp0"
title AUTONIQ Cafe Monitor - Offline Recording Analysis

echo ============================================================
echo   OFFLINE RECORDING ANALYSIS
echo ============================================================
echo.
echo This job scans available recordings and writes annotated frames and CSV reports to runtime\offline_analysis.
echo It uses CPU and can take time on an older laptop.
echo IMPORTANT: Stop monitoring in the dashboard before continuing.
echo.
choice /C YN /N /M "Has monitoring been stopped? Press Y to continue or N to cancel: "
if errorlevel 2 exit /b 0

docker compose ps backend >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker Compose services are not available. Start the project first.
  pause
  exit /b 1
)

echo.
echo Running analysis at a two-second sampling interval...
docker compose exec -T backend python /workspace/scripts/analyze_recordings.py --interval-seconds 2
if errorlevel 1 (
  echo.
  echo Analysis exited with an error. Check the message above and backend model/download logs.
  pause
  exit /b 1
)
echo.
echo Open the printed index.html report path in a browser or navigate to runtime\offline_analysis.
pause
