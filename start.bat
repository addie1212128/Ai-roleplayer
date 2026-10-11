@echo off
REM ---- AI Roleplayer: one-click launcher for Windows ----
cd /d "%~dp0"

echo Setting up AI Roleplayer...

REM Create a virtual environment the first time.
if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

REM Install/refresh dependencies.
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo.
echo Starting the app. Your browser will open shortly.
echo (Close this window to stop the app.)
echo.

python run.py

pause
