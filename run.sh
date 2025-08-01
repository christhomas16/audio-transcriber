#!/bin/bash

# Audio Transcriber Runner Script
# This script provides an easy way to run the audio transcriber

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Function to check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed or not in PATH"
        echo ""
        echo "Please install Python 3.7+ from: https://python.org"
        echo ""
        echo "Or use your system package manager:"
        echo "  macOS: brew install python3"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
        echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_success "Python $PYTHON_VERSION found"
}

# Function to check if pip is available
check_pip() {
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        print_error "pip3 is not installed or not in PATH"
        echo ""
        echo "Please install pip:"
        echo "  python3 -m ensurepip --upgrade"
        echo "  or download from: https://pip.pypa.io/en/stable/installation/"
        exit 1
    fi
    
    print_success "pip found"
}

# Function to check if FFmpeg is installed
check_ffmpeg() {
    if ! command -v ffmpeg &> /dev/null; then
        print_warning "FFmpeg is not installed (required for audio processing)"
        echo ""
        echo "Please install FFmpeg:"
        echo ""
        echo "  macOS:"
        echo "    brew install ffmpeg"
        echo ""
        echo "  Ubuntu/Debian:"
        echo "    sudo apt update && sudo apt install ffmpeg"
        echo ""
        echo "  CentOS/RHEL/Fedora:"
        echo "    sudo yum install ffmpeg"
        echo "    or: sudo dnf install ffmpeg"
        echo ""
        echo "  Windows:"
        echo "    Download from: https://ffmpeg.org/download.html"
        echo ""
        echo "  Or install via conda:"
        echo "    conda install ffmpeg"
        echo ""
        read -p "Press Enter to continue anyway (may not work without FFmpeg) or Ctrl+C to exit: "
    else
        print_success "FFmpeg found"
    fi
}

# Function to setup Python virtual environment
setup_virtual_environment() {
    print_step "Setting up Python virtual environment..."
    
    VENV_DIR="venv"
    
    # Check if virtual environment already exists
    if [ -d "$VENV_DIR" ]; then
        print_info "Virtual environment already exists at $VENV_DIR"
        echo ""
        echo "Would you like to recreate it? (y/n)"
        read -p "Enter your choice: " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
        else
            print_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    # Create virtual environment
    print_info "Creating virtual environment in $VENV_DIR..."
    if python3 -m venv "$VENV_DIR"; then
        print_success "Virtual environment created successfully"
    else
        print_error "Failed to create virtual environment"
        echo ""
        echo "Make sure you have the venv module installed:"
        echo "  python3 -m ensurepip --upgrade"
        echo "  python3 -m pip install --upgrade pip"
        exit 1
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip in virtual environment
    print_info "Upgrading pip in virtual environment..."
    python -m pip install --upgrade pip
    
    print_success "Virtual environment setup complete!"
}

# Function to activate virtual environment
activate_virtual_environment() {
    VENV_DIR="venv"
    
    if [ -d "$VENV_DIR" ]; then
        if [ -f "$VENV_DIR/bin/activate" ]; then
            print_info "Activating virtual environment..."
            source "$VENV_DIR/bin/activate"
            return 0
        else
            print_warning "Virtual environment exists but activate script not found"
            return 1
        fi
    else
        print_warning "Virtual environment not found"
        return 1
    fi
}

# Function to install Python dependencies
install_dependencies() {
    print_step "Installing Python dependencies..."
    
    # Check if we're in a virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_info "Not in virtual environment, installing globally..."
    else
        print_info "Installing in virtual environment: $VIRTUAL_ENV"
    fi
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        print_info "Installing from requirements.txt..."
        if python -m pip install -r requirements.txt; then
            print_success "Dependencies installed from requirements.txt"
        else
            print_error "Failed to install from requirements.txt"
            exit 1
        fi
    else
        print_info "requirements.txt not found, installing openai-whisper directly..."
        if python -m pip install openai-whisper; then
            print_success "openai-whisper installed"
        else
            print_error "Failed to install openai-whisper"
            exit 1
        fi
    fi
}

# Function to check if required packages are installed
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check if whisper is installed
    if ! python -c "import whisper" &> /dev/null; then
        print_warning "OpenAI Whisper is not installed"
        echo ""
        echo "Would you like to install it automatically? (y/n)"
        read -p "Enter your choice: " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_dependencies
        else
            echo ""
            echo "Please install manually:"
            echo "  pip install openai-whisper"
            echo "  or: pip install -r requirements.txt"
            exit 1
        fi
    fi
    
    print_success "All dependencies are installed"
}

# Function to setup environment
setup_environment() {
    print_step "Setting up environment..."
    
    # Check Python
    check_python
    
    # Check pip
    check_pip
    
    # Check FFmpeg
    check_ffmpeg
    
    # Setup virtual environment
    setup_virtual_environment
    
    # Install dependencies in virtual environment
    install_dependencies
    
    print_success "Environment setup complete!"
    echo ""
    echo "To use the transcriber in the future, run:"
    echo "  ./run.sh your_audio_file.wav"
    echo ""
    echo "The script will automatically activate the virtual environment."
}

# Function to display usage
show_usage() {
    echo "Audio Transcriber - Usage Examples:"
    echo ""
    echo "Basic usage:"
    echo "  $0 audio.wav"
    echo ""
    echo "With custom model:"
    echo "  $0 audio.wav -m large"
    echo ""
    echo "With custom language:"
    echo "  $0 audio.wav -l es"
    echo ""
    echo "With custom output file:"
    echo "  $0 audio.wav -o my_transcription.txt"
    echo ""
    echo "Without timestamps:"
    echo "  $0 audio.wav --no-timestamps"
    echo ""
    echo "Setup environment (create virtual environment and install dependencies):"
    echo "  $0 --setup"
    echo ""
    echo "Available models: tiny, base, small, medium, large"
    echo "Language codes: en (English), es (Spanish), fr (French), etc."
    echo ""
    echo "For more options, run: python3 transcriber.py --help"
}

# Main execution
main() {
    print_info "Starting Audio Transcriber..."
    
    # Check if no arguments provided
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi
    
    # Check if help is requested
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_usage
        exit 0
    fi
    
    # Check if setup is requested
    if [ "$1" = "--setup" ]; then
        setup_environment
        exit 0
    fi
    
    # Quick environment check (without interactive prompts)
    print_info "Performing quick environment check..."
    
    # Check Python installation
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed. Run '$0 --setup' to install dependencies."
        exit 1
    fi
    
    # Try to activate virtual environment
    if ! activate_virtual_environment; then
        print_warning "Virtual environment not found. Run '$0 --setup' to create one."
        print_info "Continuing with system Python..."
    fi
    
    # Check if whisper is installed
    if ! python -c "import whisper" &> /dev/null; then
        print_error "OpenAI Whisper is not installed. Run '$0 --setup' to install dependencies."
        exit 1
    fi
    
    # Check FFmpeg (warning only)
    if ! command -v ffmpeg &> /dev/null; then
        print_warning "FFmpeg not found. Audio processing may not work properly."
        print_info "Run '$0 --setup' to get installation instructions."
    fi
    
    # Run the transcriber with all provided arguments
    print_info "Running transcriber with arguments: $@"
    python transcriber.py "$@"
    
    if [ $? -eq 0 ]; then
        print_success "Transcription completed successfully!"
    else
        print_error "Transcription failed!"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"