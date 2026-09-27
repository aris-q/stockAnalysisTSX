@echo off
cd /d C:\Code_Git\stockAnalysisTSX

REM Refuse to start a second TSX instance. Two app.py processes on the same
REM port run independent scheduler threads and write to the same JSON files
REM with no cross-process locking. TSX uses port 5051 (the US app uses 5050),
REM so both apps can run side by side.
netstat -ano | findstr "LISTENING" | findstr ":5051 " >nul
if %errorlevel%==0 (
    echo.
    echo ============================================================
    echo  Stock.AI TSX is already running on port 5051.
    echo  NOT starting a second instance.
    echo  If it's stuck or you need to restart it, close the existing
    echo  server window first, then run this again.
    echo ============================================================
    echo.
    pause
    exit /b 1
)

echo Stock.AI TSX starting on http://localhost:5051
call venv\Scripts\activate
python app.py
