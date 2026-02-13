#!/bin/bash

# Batch Transcribe and Summarize Runner Script
# Processes all MP4 files in videos/ directory using Whisper + Ollama

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed."
    echo ""
    echo "Install Python 3.7+ from: https://python.org"
    echo "  macOS:         brew install python3"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi
print_success "Python3 found"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    print_warning "FFmpeg not found. Required for audio/video processing."
    echo ""
    echo "Install FFmpeg:"
    echo "  macOS:         brew install ffmpeg"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo ""
    echo "Continuing without FFmpeg (transcription may fail)..."
else
    print_success "FFmpeg found"
fi

# Check Tesseract (for contact extraction OCR)
if ! command -v tesseract &> /dev/null; then
    print_warning "Tesseract OCR not found. Contact extraction from video slides will be skipped."
    echo ""
    echo "Install Tesseract (optional, for --no-contacts this isn't needed):"
    echo "  macOS:         brew install tesseract"
    echo "  Ubuntu/Debian: sudo apt install tesseract-ocr"
else
    print_success "Tesseract OCR found"
fi

# Create virtual environment if it doesn't exist
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install/check Python dependencies from requirements.txt
print_info "Checking Python dependencies..."
python -m pip install -q -r requirements.txt 2>&1 | grep -v "already satisfied" || true
print_success "All Python dependencies installed"

# Check for .env file
if [ ! -f ".env" ]; then
    print_warning ".env file not found."
    echo ""
    echo "Create a .env file with your Ollama configuration:"
    echo "  echo 'OLLAMA_URL=http://localhost:11434' >> .env"
    echo "  echo 'OLLAMA_MODEL=qwen3:8b' >> .env"
    echo ""
    echo "Without .env, you can still run with --transcribe-only"
fi

# Check for videos directory
if [ ! -d "videos" ]; then
    print_warning "No 'videos/' directory found. Create it and add MP4 files."
fi

print_info "Starting batch transcription and summarization..."
python batch_transcribe_summarize.py "$@"

if [ $? -eq 0 ]; then
    print_success "Batch processing completed!"
else
    print_error "Batch processing finished with errors."
    exit 1
fi
