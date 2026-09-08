@echo off
title Inventory Forecast Backend Server

echo Starting FastAPI Backend...
echo.

REM Change directory to your project folder
cd /d "C:\Users\Sanchit\OneDrive\Desktop\ML_Project"

echo Installing required dependencies (only if missing)...
py -m pip install fastapi uvicorn pandas numpy statsmodels prophet matplotlib --quiet

echo.
echo Launching server at:  http://127.0.0.1:8000
echo API Docs available at: http://127.0.0.1:8000/docs
echo.
echo Press CTRL + C to stop the server.
echo.

REM Start FastAPI backend using uvicorn
py -m uvicorn ml_backend:app --reload

pause
