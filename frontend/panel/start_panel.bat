@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Audio Monitor - Web Panel
color 0E

echo.
echo   ==============================================
echo     Audio Monitor - Web Panel
echo   ==============================================
echo.

set PORT=8081

:: --- Find Python ---
set PYTHON=
for %%p in (python python3 py) do (
    %%p --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON=%%p
        goto :found
    )
)
echo   [WARN] Python not found, trying to open directly...
start "" "frontend-panel.html"
goto :end

:found
echo   [OK] %PYTHON% found
echo.
echo   Backend API default: http://localhost:8080
echo   If backend is on another machine, click gear icon
echo   in the panel to set the IP.
echo.
echo   Panel:      http://localhost:%PORT%/frontend-panel.html
echo   Dashboard:  http://localhost:%PORT%/backend-dashboard.html
echo.
echo   Press Ctrl+C to stop
echo   ==============================================
echo.

start "" "http://localhost:%PORT%/frontend-panel.html"
%PYTHON% -m http.server %PORT%

:end
echo.
pause
