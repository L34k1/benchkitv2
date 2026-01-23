@echo off
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV_DIR=.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating venv...
  py -3 -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
