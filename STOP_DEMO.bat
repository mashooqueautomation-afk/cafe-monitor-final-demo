@echo off
setlocal
cd /d "%~dp0"
title AUTONIQ Cafe Monitor - Stop Demo

docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is not running. Open it first if you need to stop the containers cleanly.
  pause
  exit /b 1
)

echo Stopping only the cafe-monitor-work Compose project...
docker compose down
if errorlevel 1 (
  echo Stop failed. Check whether Docker is still running and retry.
  pause
  exit /b 1
)
echo Done. The named model volume is retained; do not use -v unless you intend to delete it.
pause
