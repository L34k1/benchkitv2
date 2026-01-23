@echo off
setlocal enabledelayedexpansion

call "%~dp0_env.bat"

"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install -e ".[all]"
"%PYTHON%" -m playwright install chromium

set "OUT_ROOT=%~dp0outputs\smoke_8ch_600s"

"%PYTHON%" run_all.py ^
  --out-root "%OUT_ROOT%" ^
  --data-dir data ^
  --runs 1 ^
  --windows 600 ^
  --channels 8 ^
  --sequences PAN ^
  --overlays OVL_OFF ^
  --modes IO TFFR A1 A2 ^
  --tools all ^
  --cadence-ms 16.666 ^
  --load-duration-multiplier 1

set "EXITCODE=%ERRORLEVEL%"
pause
exit /b %EXITCODE%
