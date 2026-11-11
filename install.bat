@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title FreeHoops Highlights Tool - install

set "PY="
where py >nul 2>&1
if errorlevel 1 goto try_python
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 goto try_python
set "PY=py -3"
goto have_python
:try_python
where python >nul 2>&1
if errorlevel 1 goto no_python
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 goto no_python
set "PY=python"
goto have_python
:no_python
echo.
echo Python 3.10 or later was not found. Install it from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH" and keep the tcl/tk option), then run install.bat again.
echo.
pause
exit /b 1

:have_python
echo Using: %PY%
%PY% --version
if exist ".venv\Scripts\python.exe" goto have_venv
echo Creating a private Python environment in .venv ...
%PY% -m venv .venv
if errorlevel 1 (
  echo Could not create the virtual environment.
  pause
  exit /b 1
)
:have_venv
set "VPY=%~dp0.venv\Scripts\python.exe"
"%VPY%" -m pip install --disable-pip-version-check --upgrade pip

"%VPY%" -c "import torch" >nul 2>&1
if not errorlevel 1 goto torch_done
where nvidia-smi >nul 2>&1
if errorlevel 1 goto torch_cpu
echo NVIDIA GPU found: installing PyTorch with CUDA (large download, about 2.5 GB)...
"%VPY%" -m pip install --disable-pip-version-check torch torchvision --index-url https://download.pytorch.org/whl/cu128
if not errorlevel 1 goto torch_done
echo The CUDA build did not install. Falling back to the CPU build.
:torch_cpu
echo Installing PyTorch (CPU build, about 200 MB)...
"%VPY%" -m pip install --disable-pip-version-check torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
  echo.
  echo PyTorch could not be installed for this Python version.
  echo If you are on a very new Python, install Python 3.12 or 3.13 as well and run:
  echo   rmdir /s /q .venv
  echo   py -3.13 -m venv .venv
  echo then run install.bat again.
  pause
  exit /b 1
)
:torch_done
echo Installing the other libraries (Ultralytics YOLO, OpenCV, NumPy, Pillow)...
"%VPY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo Library install failed. Check the internet connection and run install.bat again.
  pause
  exit /b 1
)
"%VPY%" -c "import torch, cv2, numpy, PIL, ultralytics, tkinter; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
  echo Something is still missing, see the messages above.
  pause
  exit /b 1
)
echo.
echo Install finished. Double-click run.bat to start.
echo.
if /i not "%~1"=="--quiet" pause
exit /b 0
