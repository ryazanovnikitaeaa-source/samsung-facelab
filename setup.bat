@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.12 from python.org with the Python launcher, then retry.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
".venv\Scripts\python.exe" scripts\download_models.py
if errorlevel 1 goto error
echo Ready. Open start.bat to launch FaceLab.
pause
exit /b 0
:error
echo Setup failed. Check the message above and the internet connection.
pause
exit /b 1
