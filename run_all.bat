@echo off
setlocal enabledelayedexpansion

call "%~dp0_env.bat"

set "BENCHKIT_COLLECT_HEADED=1"

"%PYTHON%" -m pip install --upgrade pip

if exist "scripts\desktop\requirements-desktop.txt" (
  "%PYTHON%" -m pip install -r "scripts\desktop\requirements-desktop.txt"
)

if exist "scripts\web\requirements-web.txt" (
  "%PYTHON%" -m pip install -r "scripts\web\requirements-web.txt"
  "%PYTHON%" -m playwright install chromium
)

"%PYTHON%" run_all.py --runs 1 %*
set "EXITCODE=%ERRORLEVEL%"

if "%EXITCODE%"=="0" (
  "%PYTHON%" scripts\analysis\unify_outputs_csv.py --out-root outputs --out-dir outputs\_aggregate\unified --benches IO TFFR A1_THROUGHPUT A2_CADENCED
  "%PYTHON%" scripts\analysis\aggregate_by_tool_phase.py --unified-dir outputs\_aggregate\unified --out-dir outputs\_aggregate\by_tool_phase
  "%PYTHON%" scripts\analysis\aggregate_by_tool_window_channels.py --unified-dir outputs\_aggregate\unified --out-dir outputs\_aggregate\by_tool_window_channels
)

if not "%EXITCODE%"=="0" (
  echo.
  echo run_all.py failed with exit code %EXITCODE%.
  if exist "outputs\_orchestration\orchestrator.log" (
    echo --- tail of outputs\_orchestration\orchestrator.log ---
    powershell -NoProfile -Command "Get-Content -Path 'outputs\\_orchestration\\orchestrator.log' -Tail 200"
  ) else (
    echo No orchestrator log found at outputs\_orchestration\orchestrator.log
  )
  echo.
  pause
)
exit /b %EXITCODE%
