@echo off
setlocal

call "%~dp0_env.bat"

if "%~1"=="" (
  echo Usage: gen_synth_one.bat EDF^|NWB [--no-clobber]
  exit /b 1
)

set "FORMAT=%~1"
shift

%PYTHON% -m pip install --upgrade pip
if exist "scripts\desktop\requirements-desktop.txt" (
  %PYTHON% -m pip install -r "scripts\desktop\requirements-desktop.txt"
)

%PYTHON% scripts\data\gen_synth_one.py --format %FORMAT% %*
if errorlevel 1 (
  echo.
  echo gen_synth_one failed with exit code %ERRORLEVEL%.
  echo If you double-clicked the batch file, this pause keeps the window open.
  pause
  exit /b %ERRORLEVEL%
)
echo.
echo gen_synth_one completed.
pause
exit /b 0
