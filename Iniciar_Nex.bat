@echo off
title Nex Vision AI - Sistema Unificado de Cocina
cd /d "%~dp0"

echo ========================================================
echo   NEX VISION AI - PANEL Y MONITOR DE COCINA (720p)
echo ========================================================
echo.
echo Iniciando servidor unificado (API + Base de Datos + Dashboard)...
echo En unos segundos se abrira automaticamente http://localhost:8000
echo.
echo (Para apagar el sistema cierra esta ventana o presiona Ctrl+C)
echo ========================================================

start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000"
"C:\PROYECTOSIA\nex-venv\Scripts\uvicorn.exe" src.api.main:app --host 0.0.0.0 --port 8000
pause
