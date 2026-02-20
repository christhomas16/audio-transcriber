@echo off
setlocal EnableDelayedExpansion

:: Batch Transcribe and Summarize Runner Script for Windows
:: Processes all MP4 files in videos/ directory using Whisper + Ollama
:: This script checks for and installs all prerequisites automatically.

title Audio Transcriber - Batch Processing

:: Change to script directory
cd /d "%~dp0"

echo.
echo ============================================================
echo   Audio Transcriber - Windows Setup and Runner
echo ============================================================
echo.

:: ---------------------------------------------------------------
:: 1. Check Python
:: ---------------------------------------------------------------
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo   Install Python 3.10+ from: https://www.python.org/downloads/
    echo   IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

:: Verify Python version is 3.x
python --version 2>&1 | findstr /R "^Python 3\." >nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python 3.x is required. Found:
    python --version
    echo.
    echo   Install Python 3.10+ from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo [OK] %%i found

:: ---------------------------------------------------------------
:: 2. Check FFmpeg
:: ---------------------------------------------------------------
where ffmpeg >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] FFmpeg not found. Required for audio/video processing.
    echo.
    echo   Install FFmpeg using one of these methods:
    echo     Option A - winget:   winget install Gyan.FFmpeg
    echo     Option B - choco:    choco install ffmpeg
    echo     Option C - manual:   https://www.gyan.dev/ffmpeg/builds/
    echo                          Download, extract, add bin\ folder to PATH
    echo.
    echo   After installing, restart this terminal and try again.
    echo.

    set /p "CONTINUE_NO_FFMPEG=Continue without FFmpeg? (y/N): "
    if /i not "!CONTINUE_NO_FFMPEG!"=="y" (
        exit /b 1
    )
) else (
    echo [OK] FFmpeg found
)

:: ---------------------------------------------------------------
:: 3. Check Tesseract (optional, for contact extraction OCR)
:: ---------------------------------------------------------------
where tesseract >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Tesseract OCR not found. Contact extraction from video slides will be skipped.
    echo.
    echo   Install Tesseract ^(optional, not needed with --no-contacts^):
    echo     Option A - winget:  winget install UB-Mannheim.TesseractOCR
    echo     Option B - choco:   choco install tesseract
    echo     Option C - manual:  https://github.com/UB-Mannheim/tesseract/wiki
    echo.
) else (
    echo [OK] Tesseract OCR found
)

:: ---------------------------------------------------------------
:: 4. Create virtual environment if it doesn't exist
:: ---------------------------------------------------------------
if not exist "venv" (
    echo.
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

:: ---------------------------------------------------------------
:: 5. Activate virtual environment
:: ---------------------------------------------------------------
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: ---------------------------------------------------------------
:: 6. Install/upgrade pip and install Python dependencies
:: ---------------------------------------------------------------
echo [INFO] Checking Python dependencies...
python -m pip install --upgrade pip -q 2>nul
python -m pip install -q -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install Python dependencies.
    echo   If you see build errors for openai-whisper, you may need:
    echo     - Microsoft Visual C++ Build Tools
    echo       https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo     - Or install Rust: https://rustup.rs/
    echo.
    pause
    exit /b 1
)
echo [OK] All Python dependencies installed

:: ---------------------------------------------------------------
:: 7. Check for .env file
:: ---------------------------------------------------------------
if not exist ".env" (
    echo.
    echo [WARNING] .env file not found.
    echo.
    echo   Create a .env file with your Ollama configuration:
    echo     echo OLLAMA_URL=http://localhost:11434 ^> .env
    echo     echo OLLAMA_MODEL=qwen3:8b ^>^> .env
    echo.
    echo   Without .env, you can still run with --transcribe-only
    echo.
)

:: ---------------------------------------------------------------
:: 8. Check for videos directory
:: ---------------------------------------------------------------
if not exist "videos" (
    echo [WARNING] No 'videos\' directory found. Creating it...
    mkdir videos
    echo   Add your MP4 files to the 'videos\' folder and run again.
)

:: ---------------------------------------------------------------
:: 9. Run the batch processor, passing through all arguments
:: ---------------------------------------------------------------
echo.
echo [INFO] Starting batch transcription and summarization...
echo.
python batch_transcribe_summarize.py %*

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Batch processing completed!
) else (
    echo.
    echo [ERROR] Batch processing finished with errors.
    pause
    exit /b 1
)

pause
