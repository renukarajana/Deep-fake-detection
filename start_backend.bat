@echo off
echo Installing required packages...
python -m pip install uvicorn fastapi python-multipart pydantic --quiet
echo.
echo Starting Backend Server...
echo Backend will be available at http://localhost:8000
echo.
python backend\app.py
pause

