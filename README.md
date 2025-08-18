# Audio Transcriber

A Python-based audio transcription tool using OpenAI's Whisper model with optional speaker diarization. This tool can transcribe audio files in various formats and languages, with the ability to identify and distinguish between different speakers in the audio.

## Features

### Basic Transcription
- Transcribe audio files using OpenAI Whisper
- Support for multiple model sizes (tiny, base, small, medium, large)
- Multi-language support with auto-detection
- Optional timestamp generation
- Command-line interface with customizable options
- Easy-to-use shell script wrapper

### Speaker Diarization (Advanced)
- **Speaker identification**: Automatically identify and distinguish between different speakers
- **Voice fingerprinting**: Create consistent speaker profiles across the audio
- **Multi-speaker support**: Handle conversations with multiple participants
- **Confidence-based matching**: Intelligent speaker matching with confidence thresholds
- **Debug mode**: Detailed logging for troubleshooting speaker identification

## Requirements

### Basic Transcription
- Python 3.7+
- OpenAI Whisper
- FFmpeg (for audio processing)

### Speaker Diarization (Additional Requirements)
- PyTorch and TorchAudio
- Transformers library
- Pyannote.audio (speaker diarization)
- Hugging Face token (free account required)
- Additional dependencies (see requirements_diarization.txt)

## Installation

### Quick Start (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/christhomas16/audio-transcriber.git
   cd audio-transcriber
   ```

2. For basic transcription, run the automatic setup:
   ```bash
   ./run.sh --setup
   ```
   
   This will:
   - Check Python and pip installation
   - Install FFmpeg (if needed)
   - Create a Python virtual environment
   - Install all required dependencies
   - Set up everything automatically

3. For speaker diarization (advanced), run the diarization setup:
   ```bash
   ./run_diarization.sh --setup
   ```
   
   This will:
   - Set up everything from basic transcription
   - Install PyTorch and Pyannote.audio
   - Guide you through Hugging Face token setup
   - Install all diarization dependencies

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

### Basic Transcription

#### Using the shell script (recommended)

Basic usage:
```bash
./run.sh audio.wav
# Also works with M4A, MP3, MP4, and other formats:
./run.sh recording.m4a
./run.sh video.mp4
```

### Speaker Diarization (Advanced)

#### Using the diarization script

Basic usage with speaker identification:
```bash
./run_diarization.sh audio.wav
# Also works with M4A, MP3, MP4, and other formats:
./run_diarization.sh recording.m4a
./run_diarization.sh video.mp4
```

With debug mode to see speaker identification details:
```bash
./run_diarization.sh audio.wav --debug
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

#### Basic Transcription
```bash
# If using virtual environment (recommended):
source venv/bin/activate  # On Windows: venv\Scripts\activate
python transcriber.py audio.wav

# Or with system Python:
python3 transcriber.py audio.wav
python3 transcriber.py audio.wav -m large -l en -o output.txt
```

#### Speaker Diarization
```bash
# If using virtual environment (recommended):
source venv/bin/activate  # On Windows: venv\Scripts\activate
python transcriber_with_diarization.py audio.wav

# With debug mode:
python transcriber_with_diarization.py audio.wav --debug

# With custom model and language:
python transcriber_with_diarization.py audio.wav -m large -l en --debug
```

### Command-line options

#### Basic Transcription
- `audio_file`: Path to the audio file to transcribe (required)
- `-m, --model`: Whisper model size (tiny, base, small, medium, large) - default: base
- `-l, --language`: Language code (e.g., en, es, fr) - default: en (auto-detect if empty)
- `-o, --output`: Output file path - default: `[audio_filename]_transcription.txt`
- `--no-timestamps`: Don't include timestamps in output file

#### Speaker Diarization (Additional Options)
- `--debug`: Enable debug mode to see detailed speaker identification process
- All basic transcription options are also available

## Supported Audio Formats

Both basic transcription and speaker diarization support many audio formats including:
- **WAV** - Uncompressed audio
- **MP3** - Compressed audio
- **M4A** - Apple's audio format (common in iPhone recordings)
- **FLAC** - Lossless compressed audio
- **OGG** - Open source audio format
- **AAC** - Advanced Audio Coding
- **WMA** - Windows Media Audio
- **MP4** - Video files with audio (audio will be extracted)

**Note**: M4A files are particularly well-supported and work great for speaker diarization, making them ideal for meeting recordings and voice memos.

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

### Basic Transcription

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

6. **MP4 video files:**
   ```bash
   ./run.sh video.mp4 -m medium
   ./run.sh presentation.mp4 -o presentation_transcript.txt
   ```

7. **Auto-detect language:**
   ```bash
   ./run.sh multilingual_audio.wav -l ""
   ```

### Speaker Diarization

7. **Meeting with multiple speakers:**
   ```bash
   ./run_diarization.sh meeting_recording.m4a
   ```

8. **Interview with speaker identification:**
   ```bash
   ./run_diarization.sh interview.wav -m large --debug
   ```

9. **Multi-language conversation with speakers:**
   ```bash
   ./run_diarization.sh conversation.mp3 -l es --debug
   ```

10. **Podcast with speaker tracking:**
    ```bash
    ./run_diarization.sh podcast.m4a -o podcast_transcript.txt
    ```

11. **Video with multiple speakers:**
    ```bash
    ./run_diarization.sh meeting_video.mp4 -m large --debug
    ./run_diarization.sh interview_video.mp4 -o interview_transcript.txt
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

## Hugging Face Token Setup (for Diarization)

Speaker diarization requires a free Hugging Face account and token:

1. **Create a Hugging Face account** at [https://huggingface.co/join](https://huggingface.co/join)

2. **Get your access token**:
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Give it a name (e.g., "audio-transcriber")
   - Select "Read" role
   - Copy the token

3. **Set up the token** (choose one method):
   ```bash
   # Method 1: Environment variable
   export HF_TOKEN=your_token_here
   
   # Method 2: .env file (recommended)
   echo 'HF_TOKEN=your_token_here' > .env
   ```

4. **Accept the model licenses**:
   - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Click "Accept" to accept the license
   - Visit [pyannote/embedding](https://huggingface.co/pyannote/embedding)
   - Click "Accept" to accept the license

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

3. **"Format not recognised" error (especially for M4A files):**
   - Make sure FFmpeg is installed and working
   - Check if the audio file is corrupted
   - Some M4A files may require specific codecs

4. **Out of memory errors:**
   Try using a smaller model (e.g., `tiny` or `base`)

5. **Poor transcription quality:**
   - Check audio quality and volume levels
   - Try a larger model
   - Specify the correct language if auto-detection fails

6. **Hugging Face token issues (for diarization):**
   - Make sure you have a valid HF_TOKEN
   - Accept the model licenses on Hugging Face
   - Check if your token has the correct permissions

### Performance Issues

- First run may be slower as Whisper downloads the model
- Models are cached in `~/.cache/whisper/` for subsequent runs
- GPU acceleration is automatically used if available (CUDA/Metal)

## File Structure

```
audio-transcriber/
├── transcriber.py                    # Basic transcription script
├── transcriber_with_diarization.py   # Advanced transcription with speaker identification
├── run.sh                           # Basic transcription shell script wrapper
├── run_diarization.sh               # Diarization shell script wrapper
├── requirements.txt                 # Basic transcription dependencies
├── requirements_diarization.txt     # Diarization dependencies
├── .gitignore                      # Git ignore patterns
├── README.md                       # This file
└── venv/                           # Virtual environment (created by setup)
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