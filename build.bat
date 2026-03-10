@echo off
setlocal enabledelayedexpansion

:: Parse arguments
set GPU=0
if "%1"=="--gpu" set GPU=1

:: Resolve Python launcher (prefer py -3 on Windows)
set "PYTHON="
where py >nul 2>&1 && set "PYTHON=py -3"
if "%PYTHON%"=="" (
    where python >nul 2>&1 && set "PYTHON=python"
)
if "%PYTHON%"=="" (
    echo ERROR: Python 3 was not found in PATH.
    exit /b 1
)

:: Read version from version.py using Python (reliable across quote styles)
for /f %%a in ('%PYTHON% -c "from version import __version__; print(__version__)"') do set VERSION=%%a
if "%VERSION%"=="" (
    echo ERROR: Could not read version from version.py
    exit /b 1
)
echo Building TinyReadAloud v%VERSION% (GPU=%GPU%)

:: Step 1: Ensure runtime/build dependencies in the active Python env
%PYTHON% -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    exit /b 1
)

%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.txt.
    exit /b 1
)

if "%GPU%"=="1" (
    echo Installing GPU runtime dependencies...
    %PYTHON% -m pip install --upgrade onnxruntime-gpu nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12
    if errorlevel 1 (
        echo Failed to install GPU runtime dependencies.
        exit /b 1
    )

    %PYTHON% -c "import onnxruntime as ort; import sys; providers = ort.get_available_providers(); print('ONNX providers:', providers); sys.exit(0 if 'CUDAExecutionProvider' in providers else 1)"
    if errorlevel 1 (
        echo ERROR: CUDAExecutionProvider not available. GPU package build aborted.
        exit /b 1
    )
) else (
    %PYTHON% -m pip show onnxruntime >nul 2>&1
    if errorlevel 1 (
        %PYTHON% -m pip show onnxruntime-gpu >nul 2>&1
        if errorlevel 1 (
            echo Installing CPU ONNX Runtime ^(onnxruntime^)...
            %PYTHON% -m pip install onnxruntime
            if errorlevel 1 (
                echo Failed to install onnxruntime.
                exit /b 1
            )
        )
    )
)

%PYTHON% -m pip show pyinstaller >nul 2>&1 || (
    echo Installing PyInstaller...
    %PYTHON% -m pip install pyinstaller
    if errorlevel 1 (
        echo Failed to install PyInstaller.
        exit /b 1
    )
)

:: Step 2: Clean previous build
taskkill /F /IM TinyReadAloud.exe >nul 2>&1

call :remove_dir build
if errorlevel 1 exit /b 1

call :remove_dir dist
if errorlevel 1 exit /b 1

call :remove_dir installer_output
if errorlevel 1 exit /b 1

:: Step 3: Generate icon if not present
if not exist assets\app.ico (
    echo Generating app icon...
    %PYTHON% generate_icon.py
    if errorlevel 1 (
        echo Icon generation failed!
        exit /b 1
    )
)

:: Step 4: Run PyInstaller
echo.
echo === Running PyInstaller ===
set TINYREADALOUD_GPU=%GPU%
%PYTHON% -m PyInstaller tinyreadaloud.spec --noconfirm
if errorlevel 1 (
    echo PyInstaller failed!
    exit /b 1
)

if not exist dist\TinyReadAloud\TinyReadAloud.exe (
    echo ERROR: PyInstaller output missing: dist\TinyReadAloud\TinyReadAloud.exe
    exit /b 1
)

:: Step 5: Run Inno Setup
echo.
echo === Running Inno Setup ===
set ISCC=
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
) else if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" (
    set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
) else (
    for /f "delims=" %%i in ('where iscc 2^>nul') do (
        if not defined ISCC set "ISCC=%%i"
    )
)

if "%ISCC%"=="" (
    echo Inno Setup 6 not found.
    echo Download from: https://jrsoftware.org/issetup.exe
    echo.
    echo PyInstaller output is in: dist\TinyReadAloud\
    exit /b 1
) else (
    echo Using Inno Setup compiler: %ISCC%
)

"%ISCC%" /DMyAppVersion=%VERSION% installer.iss
if errorlevel 1 (
    echo Inno Setup failed!
    exit /b 1
)

if not exist installer_output\TinyReadAloud-%VERSION%-Setup.exe (
    echo ERROR: Expected installer not found.
    exit /b 1
)

copy /Y installer_output\TinyReadAloud-%VERSION%-Setup.exe installer_output\TinyReadAloud-Setup.exe >nul
copy /Y installer_output\TinyReadAloud-%VERSION%-Setup.exe installer_output\setup.exe >nul

echo.
echo === Build complete ===
echo Installer: installer_output\TinyReadAloud-%VERSION%-Setup.exe
echo Installer (latest alias): installer_output\TinyReadAloud-Setup.exe
echo Installer (generic alias): installer_output\setup.exe

exit /b 0

:remove_dir
set "TARGET_DIR=%~1"
if not exist "%TARGET_DIR%" exit /b 0

for /l %%I in (1,1,8) do (
    rmdir /s /q "%TARGET_DIR%" >nul 2>&1
    if not exist "%TARGET_DIR%" exit /b 0
    timeout /t 1 /nobreak >nul
)

echo ERROR: Could not remove "%TARGET_DIR%". Close processes that are using it and retry.
exit /b 1
