@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo ==========================================
echo   Audio Monitor - Frontend Panel
echo ==========================================
echo.

:: -- kill old server on port 8081 --
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8081.*LISTENING"') do (
    echo Killing old server PID %%a ...
    taskkill /F /PID %%a >nul 2>&1
)

:: -- find python --
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python"
    goto :found
)
py --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py"
    goto :found
)
echo [ERROR] Python 3.10+ is required.
echo Download: https://www.python.org/downloads/
pause
exit /b 1

:found
echo Python found: %PYTHON%
echo.
echo Starting server on port 8081 ...
echo.
echo Click the gear icon to set backend IP.
echo Example: http://192.168.1.100:8080
echo.
echo Press Ctrl+C to stop.
echo ==========================================

:: -- start server in background, wait a moment, then open browser --
start "" /B %PYTHON% -m http.server 8081
ping -n 2 127.0.0.1 >nul
start http://localhost:8081/frontend-panel.html?v=20260616

echo Server running. Close this window to stop.
pause
