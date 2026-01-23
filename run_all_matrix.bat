@echo off
setlocal enabledelayedexpansion

call "%~dp0_env.bat"

"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install -e ".[all]"
"%PYTHON%" -m playwright install chromium

"%PYTHON%" run_all.py %*
set "EXITCODE=%ERRORLEVEL%"

pause
exit /b %EXITCODE%
