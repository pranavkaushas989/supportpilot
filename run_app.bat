@echo off
title SupportPilot - keep this window open while using the app
cd /d "%~dp0"
set PY=python
if exist "%USERPROFILE%\anaconda3\python.exe" set PY="%USERPROFILE%\anaconda3\python.exe"
echo Using Python: %PY% > run_log.txt
echo Installing packages (first run can take a few minutes)...
echo Installing packages... >> run_log.txt
%PY% -m pip install -r requirements.txt >> run_log.txt 2>&1
echo PIP_DONE >> run_log.txt
echo Starting SupportPilot at http://localhost:8501 ...
%PY% -m streamlit run app.py --server.port 8501 --server.headless true >> run_log.txt 2>&1
echo APP_STOPPED >> run_log.txt
pause
