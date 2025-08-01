# Audio Transcriber

A Python-based audio transcription tool using OpenAI's Whisper model. This tool can transcribe audio files in various formats and languages with optional timestamp information.

## Features

- Transcribe audio files using OpenAI Whisper
- Support for multiple model sizes (tiny, base, small, medium, large)
- Multi-language support with auto-detection
- Optional timestamp generation
- Command-line interface with customizable options
- Easy-to-use shell script wrapper

## Requirements

- Python 3.7+
- OpenAI Whisper
- FFmpeg (for audio processing)

## Installation

### Quick Start (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/christhomas16/audio-transcriber.git
   cd audio-transcriber
   ```

2. Run the automatic setup:
   ```bash
   ./run.sh --setup
   ```
   
   This will:
   - Check Python and pip installation
   - Install FFmpeg (if needed)
   - Create a Python virtual environment
   - Install all required dependencies
   - Set up everything automatically

### Manual Installation

If you prefer manual installation:

1. Clone this repository:
   ```bash
   git clone https://github.com/christhomas16/audio-transcriber.git
   cd audio-transcriber
   ```

2. Create a Python virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install FFmpeg (required by Whisper):
   
   **macOS:**
   ```bash
   brew install ffmpeg
   ```
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```
   
   **Windows:**
   Download from https://ffmpeg.org/download.html

5. Make the run script executable:
   ```bash
   chmod +x run.sh
   ```

## Usage

### Using the shell script (recommended)

Basic usage:
```bash
./run.sh audio.wav
# Also works with M4A, MP3, and other formats:
./run.sh recording.m4a
```

With custom model size:
```bash
./run.sh audio.wav -m large
```

With specific language:
```bash
./run.sh audio.wav -l es  # Spanish
```

With custom output file:
```bash
./run.sh audio.wav -o my_transcription.txt
```

Without timestamps:
```bash
./run.sh audio.wav --no-timestamps
```

### Using Python directly

```bash
# If using virtual environment (recommended):
source venv/bin/activate  # On Windows: venv\Scripts\activate
python transcriber.py audio.wav

# Or with system Python:
python3 transcriber.py audio.wav
python3 transcriber.py audio.wav -m large -l en -o output.txt
```

### Command-line options

- `audio_file`: Path to the audio file to transcribe (required)
- `-m, --model`: Whisper model size (tiny, base, small, medium, large) - default: base
- `-l, --language`: Language code (e.g., en, es, fr) - default: en (auto-detect if empty)
- `-o, --output`: Output file path - default: `[audio_filename]_transcription.txt`
- `--no-timestamps`: Don't include timestamps in output file

## Supported Audio Formats

Whisper supports many audio formats including:
- WAV
- MP3
- M4A
- FLAC
- OGG
- AAC
- WMA

## Model Sizes and Performance

| Model  | Parameters | English-only | Multilingual | Required VRAM | Relative Speed |
|--------|------------|--------------|--------------|---------------|----------------|
| tiny   | 39 M       | ✓            | ✓            | ~1 GB         | ~32x           |
| base   | 74 M       | ✓            | ✓            | ~1 GB         | ~16x           |
| small  | 244 M      | ✓            | ✓            | ~2 GB         | ~6x            |
| medium | 769 M      | ✓            | ✓            | ~5 GB         | ~2x            |
| large  | 1550 M     | ✗            | ✓            | ~10 GB        | 1x             |

## Language Support

Whisper supports 99 languages. Some common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `ar` - Arabic

Leave the language parameter empty for auto-detection.

## Output Format

The tool generates two types of output:

1. **Console output**: Displays the transcribed text directly in the terminal
2. **File output**: Saves transcription to a text file

### With timestamps (default):
```
TRANSCRIPTION WITH TIMESTAMPS
==================================================

[0.00s - 2.50s] Hello, this is a sample audio file.
[2.50s - 5.10s] The transcription includes timestamps.
[5.10s - 7.80s] This helps with navigation and editing.
```

### Without timestamps:
```
TRANSCRIPTION
==================================================

Hello, this is a sample audio file. The transcription includes timestamps. This helps with navigation and editing.
```

## Examples

1. **Basic transcription:**
   ```bash
   ./run.sh meeting_recording.wav
   ```

2. **High-quality transcription for important content:**
   ```bash
   ./run.sh interview.mp3 -m large
   ```

3. **Spanish audio with custom output:**
   ```bash
   ./run.sh spanish_audio.wav -l es -o spanish_transcription.txt
   ```

4. **Quick transcription for notes (faster, less accurate):**
   ```bash
   ./run.sh voice_note.m4a -m tiny
   ```

5. **M4A files (common iPhone/iPad recordings):**
   ```bash
   ./run.sh recording.m4a -m base
   ```

6. **Auto-detect language:**
   ```bash
   ./run.sh multilingual_audio.wav -l ""
   ```

## Virtual Environment Management

The transcriber uses a Python virtual environment to isolate dependencies. Here's how to manage it:

### Automatic Management (Recommended)
The `run.sh` script automatically handles the virtual environment:
- Creates it during setup
- Activates it when running transcription
- No manual intervention needed

### Manual Virtual Environment Management
If you need to manage the virtual environment manually:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Deactivate virtual environment
deactivate

# Remove virtual environment (to recreate)
rm -rf venv
./run.sh --setup  # Recreates it
```

### Benefits of Virtual Environment
- **Isolation**: No conflicts with other Python projects
- **Clean Environment**: Fresh Python environment for the transcriber
- **Reproducible**: Same environment across different machines
- **Easy Management**: Automatic setup and activation

## Performance Tips

- Use `tiny` or `base` models for faster transcription if accuracy isn't critical
- Use `large` model for best accuracy but slower processing
- Ensure good audio quality for better transcription results
- For very long audio files, consider splitting them into smaller chunks

## Troubleshooting

### Common Issues

1. **"Module not found" error:**
   ```bash
   pip install openai-whisper
   ```

2. **FFmpeg not found:**
   Install FFmpeg as described in the installation section

3. **Out of memory errors:**
   Try using a smaller model (e.g., `tiny` or `base`)

4. **Poor transcription quality:**
   - Check audio quality and volume levels
   - Try a larger model
   - Specify the correct language if auto-detection fails

### Performance Issues

- First run may be slower as Whisper downloads the model
- Models are cached in `~/.cache/whisper/` for subsequent runs
- GPU acceleration is automatically used if available (CUDA/Metal)

## File Structure

```
audio-transcriber/
├── transcriber.py      # Main Python script
├── run.sh             # Shell script wrapper with auto-setup
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore patterns
├── README.md          # This file
└── venv/              # Virtual environment (created by setup)
    ├── bin/
    ├── lib/
    └── ...
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source. Please check the Whisper license for model usage terms.

## Acknowledgments

- OpenAI for the Whisper model
- The open-source community for FFmpeg and related tools