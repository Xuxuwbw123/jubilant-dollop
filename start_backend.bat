@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Backend - Audio Monitor
color 0A

:: ============================================
::  Backend launcher - double click to start
:: ============================================

:: --- Find Python ---
set PYTHON=
for %%p in (python python3 py) do (
    %%p --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON=%%p
        goto :found_python
    )
)
echo [ERROR] Python not found! Install Python and add to PATH.
echo Download: https://www.python.org/downloads/
pause
exit /b 1

:found_python
echo [OK] %PYTHON% detected

:: --- Check dependencies ---
%PYTHON% -c "import torch, numpy, soundfile, librosa, websockets, yaml" >nul 2>&1
if errorlevel 1 (
    echo [--] Installing missing dependencies...
    %PYTHON% -m pip install -r requirements.txt -q
    if errorlevel 1 (
        echo [WARN] pip install failed, some features may not work
    )
)

:: --- Ensure panel files exist (copy to design-demos for serving) ---
if not exist "design-demos" mkdir "design-demos"
if not exist "design-demos\frontend-panel.html" (
    copy /y "frontend\panel\frontend-panel.html" "design-demos\" >nul 2>&1
)
if not exist "design-demos\backend-dashboard.html" (
    copy /y "frontend\panel\backend-dashboard.html" "design-demos\" >nul 2>&1
)

:: --- Check model ---
if not exist "models\best_model_e2v.pt" (
    echo [WARN] Model not found: models\best_model_e2v.pt
    echo [WARN] Backend will fall back to simulate mode.
    echo.
)

:: --- Launch ---
echo.
echo   ==============================================
echo     Audio Monitor - Backend Server
echo   ==============================================
echo     GUI + HTTP(:8080) + WebSocket(:8765)
echo     Press Ctrl+C or close GUI window to stop
echo   ==============================================
echo.

%PYTHON% backend\main.py --host 0.0.0.0 %*
set _err=%errorlevel%

echo.
if %_err% neq 0 (
    echo [WARN] Backend exited with code %_err%
) else (
    echo [OK] Backend exited normally
)
pause
