@echo off
setlocal enabledelayedexpansion

:: Parse arguments
set GPU=0
if "%1"=="--gpu" set GPU=1

:: Read version from version.py
for /f "tokens=2 delims='""'" %%a in ('findstr __version__ version.py') do set VERSION=%%a
if "%VERSION%"=="" (
    echo ERROR: Could not read version from version.py
    exit /b 1
)
echo Building TinyReadAloud v%VERSION% (GPU=%GPU%)

:: Step 1: Ensure build dependencies
pip show pyinstaller >nul 2>&1 || (
    echo Installing PyInstaller...
    pip install pyinstaller
)

:: Step 2: Clean previous build
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

:: Step 3: Generate icon if not present
if not exist assets\app.ico (
    echo Generating app icon...
    python generate_icon.py
    if errorlevel 1 (
        echo Icon generation failed!
        exit /b 1
    )
)

:: Step 4: Run PyInstaller
echo.
echo === Running PyInstaller ===
set TINYREADALOUD_GPU=%GPU%
pyinstaller tinyreadaloud.spec --noconfirm
if errorlevel 1 (
    echo PyInstaller failed!
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
) else (
    echo Inno Setup 6 not found.
    echo Download from: https://jrsoftware.org/issetup.exe
    echo.
    echo PyInstaller output is in: dist\TinyReadAloud\
    exit /b 1
)

"%ISCC%" /DMyAppVersion=%VERSION% installer.iss
if errorlevel 1 (
    echo Inno Setup failed!
    exit /b 1
)

echo.
echo === Build complete ===
echo Installer: installer_output\TinyReadAloud-%VERSION%-Setup.exe
