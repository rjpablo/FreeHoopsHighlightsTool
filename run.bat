@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run: installing the libraries. This takes a few minutes...
  call "%~dp0install.bat" --quiet
  if errorlevel 1 exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0app\FreeHoopsHighlightsTool.py" %*
if errorlevel 1 (
  echo.
  pause
)
