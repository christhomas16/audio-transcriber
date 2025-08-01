#!/bin/bash

# Audio Transcriber with Diarization Runner Script
# This script provides an easy way to run the audio transcriber with speaker identification

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
        print_error "FFmpeg is not installed (required for audio processing, especially M4A files)"
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
        echo "FFmpeg is especially important for M4A files and other formats that soundfile cannot read directly."
        read -p "Press Enter to continue anyway (may not work without FFmpeg) or Ctrl+C to exit: "
    else
        print_success "FFmpeg found"
        # Test FFmpeg functionality
        if ffmpeg -version &> /dev/null; then
            print_success "FFmpeg is working correctly"
        else
            print_warning "FFmpeg found but may not be working correctly"
        fi
    fi
}

# Function to check Hugging Face token
check_hf_token() {
    if [ -z "$HF_TOKEN" ] && [ ! -f ".env" ]; then
        print_warning "Hugging Face token not found"
        echo ""
        echo "For speaker diarization, you need a Hugging Face token."
        echo "Get one from: https://huggingface.co/settings/tokens"
        echo ""
        echo "Then either:"
        echo "  1. Set environment variable: export HF_TOKEN=your_token_here"
        echo "  2. Create .env file: echo 'HF_TOKEN=your_token_here' > .env"
        echo ""
        read -p "Press Enter to continue (diarization will fail without token) or Ctrl+C to exit: "
    elif [ -f ".env" ]; then
        print_success "Found .env file with HF_TOKEN"
    elif [ ! -z "$HF_TOKEN" ]; then
        print_success "HF_TOKEN environment variable found"
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

# Function to install Python dependencies for diarization
install_diarization_dependencies() {
    print_step "Installing Python dependencies for diarization..."
    
    # Check if we're in a virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_info "Not in virtual environment, installing globally..."
    else
        print_info "Installing in virtual environment: $VIRTUAL_ENV"
    fi
    
    # Install core dependencies
    print_info "Installing core dependencies..."
    if python -m pip install torch torchaudio transformers soundfile scipy numpy; then
        print_success "Core dependencies installed"
    else
        print_error "Failed to install core dependencies"
        exit 1
    fi
    
    # Install pyannote.audio (this might take a while)
    print_info "Installing pyannote.audio (this may take several minutes)..."
    if python -m pip install pyannote.audio; then
        print_success "pyannote.audio installed"
    else
        print_error "Failed to install pyannote.audio"
        exit 1
    fi
    
    # Install additional dependencies
    print_info "Installing additional dependencies..."
    if python -m pip install python-dotenv; then
        print_success "Additional dependencies installed"
    else
        print_error "Failed to install additional dependencies"
        exit 1
    fi
    
    print_success "All diarization dependencies installed!"
}

# Function to check if required packages are installed
check_diarization_dependencies() {
    print_info "Checking diarization dependencies..."
    
    # Check if pyannote.audio is installed
    if ! python -c "import pyannote.audio" &> /dev/null; then
        print_warning "pyannote.audio is not installed"
        echo ""
        echo "Would you like to install it automatically? (y/n)"
        read -p "Enter your choice: " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_diarization_dependencies
        else
            echo ""
            echo "Please install manually:"
            echo "  pip install pyannote.audio torch torchaudio transformers soundfile scipy numpy python-dotenv"
            exit 1
        fi
    fi
    
    print_success "All diarization dependencies are installed"
}

# Function to setup environment
setup_environment() {
    print_step "Setting up environment for diarization..."
    
    # Check Python
    check_python
    
    # Check pip
    check_pip
    
    # Check FFmpeg
    check_ffmpeg
    
    # Check HF token
    check_hf_token
    
    # Setup virtual environment
    setup_virtual_environment
    
    # Install dependencies in virtual environment
    install_diarization_dependencies
    
    print_success "Environment setup complete!"
    echo ""
    echo "To use the diarization transcriber in the future, run:"
    echo "  ./run_diarization.sh your_audio_file.wav"
    echo ""
    echo "The script will automatically activate the virtual environment."
}

# Function to display usage
show_usage() {
    echo "Audio Transcriber with Diarization - Usage Examples:"
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
    echo "With debug mode:"
    echo "  $0 audio.wav --debug"
    echo ""
    echo "Setup environment (install dependencies):"
    echo "  $0 --setup"
    echo ""
    echo "Available models: tiny, base, small, medium, large"
    echo "Language codes: en (English), es (Spanish), fr (French), etc."
    echo ""
    echo "For more options, run: python transcriber_with_diarization.py --help"
}

# Main execution
main() {
    print_info "Starting Audio Transcriber with Diarization..."
    
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
    
    # Check if pyannote.audio is installed
    if ! python -c "import pyannote.audio" &> /dev/null; then
        print_error "pyannote.audio is not installed. Run '$0 --setup' to install dependencies."
        exit 1
    fi
    
    # Check FFmpeg (warning only)
    if ! command -v ffmpeg &> /dev/null; then
        print_warning "FFmpeg not found. Audio processing may not work properly."
        print_info "Run '$0 --setup' to get installation instructions."
    fi
    
    # Run the transcriber with all provided arguments
    print_info "Running diarization transcriber with arguments: $@"
    python transcriber_with_diarization.py "$@"
    
    if [ $? -eq 0 ]; then
        print_success "Transcription with diarization completed successfully!"
    else
        print_error "Transcription failed!"
        exit 1
    fi
}

# Run main function with all arguments
main "$@" 