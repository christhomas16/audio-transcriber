# Audio Transcriber

A Python-based audio/video transcription tool using OpenAI's Whisper model with optional speaker diarization and batch processing with LLM summarization.

## Features

- **Basic transcription** — Whisper-powered transcription with multiple model sizes, multi-language support, and timestamps
- **Batch processing** — Transcribe all videos in a folder, summarize with Ollama, and extract contact info (emails, phones, URLs, social handles) via OCR
- **Speaker diarization** — Identify and distinguish speakers using pyannote.audio voice fingerprinting

## Requirements

- Python 3.7+
- FFmpeg (for audio/video processing)
- **Batch processing:** Ollama running locally or on your network, Tesseract OCR (optional, for contact extraction from video frames)
- **Speaker diarization:** Hugging Face token (free account) with accepted licenses for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/embedding](https://huggingface.co/pyannote/embedding)

## Installation

### Quick Start

```bash
git clone https://github.com/christhomas16/audio-transcriber.git
cd audio-transcriber

# Basic transcription setup (creates venv, installs deps, checks FFmpeg)
./run.sh --setup

# Speaker diarization setup (adds PyTorch + pyannote.audio)
./run_diarization.sh --setup
```

### Manual Installation

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt                # Basic + batch processing
pip install -r requirements_diarization.txt    # Speaker diarization
```

Install FFmpeg: `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Ubuntu/Debian)

## Usage

### Basic Transcription

```bash
./run.sh audio.wav                    # Default model (medium.en)
./run.sh recording.m4a -m large       # Larger model for better accuracy
./run.sh audio.wav -l es              # Spanish
./run.sh audio.wav -l ""              # Auto-detect language
./run.sh audio.wav -o output.txt      # Custom output file
./run.sh audio.wav --no-timestamps    # Plain text without timestamps
```

### Batch Transcription + Summarization

Place MP4 files in `videos/` and create a `.env` file:

```bash
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

Optionally install Tesseract for contact extraction from video slides: `brew install tesseract` (macOS) or `sudo apt install tesseract-ocr` (Ubuntu/Debian)

```bash
./run_batch.sh                    # Transcribe + summarize + extract contacts
./run_batch.sh --dry-run          # Preview what will be processed
./run_batch.sh --limit 1          # Process only first N videos
./run_batch.sh --transcribe-only  # Skip Ollama summarization
./run_batch.sh --summarize-only   # Only summarize existing transcriptions
./run_batch.sh --no-contacts      # Skip contact info extraction
./run_batch.sh -m large           # Use a different Whisper model
```

**Windows:** Use `run_batch.bat` instead (handles venv setup and dependency installation automatically).

Output files are written to `output/`:

| File | Contents |
|------|----------|
| `[video]_transcription.txt` | Timestamped transcription + full text |
| `[video]_summary.txt` | Structured summary (Overview, Key Points, Implementation, Results) |
| `[video]_contacts.txt` | Extracted emails, phones, URLs, and social handles |

Batch processing is **resume-safe** — it skips videos that already have output files.

### Speaker Diarization

```bash
./run_diarization.sh audio.wav            # Basic speaker identification
./run_diarization.sh audio.wav --debug    # Detailed speaker matching output
./run_diarization.sh audio.wav -m large   # Larger model
```

Set up your Hugging Face token:

```bash
# Option 1: Environment variable
export HF_TOKEN=your_token_here

# Option 2: .env file (recommended)
echo 'HF_TOKEN=your_token_here' > .env
```

### Running Python Directly

```bash
source venv/bin/activate
python transcriber.py audio.wav
python batch_transcribe_summarize.py --help
python transcriber_with_diarization.py audio.wav --debug
```

## Whisper Models

| Model | Parameters | Multilingual | VRAM | Relative Speed |
|--------|-----------|-------------|-------|---------------|
| tiny | 39 M | Yes | ~1 GB | ~32x |
| base | 74 M | Yes | ~1 GB | ~16x |
| small | 244 M | Yes | ~2 GB | ~6x |
| medium | 769 M | Yes | ~5 GB | ~2x |
| large | 1550 M | Yes | ~10 GB | 1x |

English-only variants (`tiny.en`, `base.en`, `small.en`, `medium.en`) are available and slightly faster for English content. The default model is `medium.en`.

## Supported Formats

Audio: WAV, MP3, M4A, FLAC, OGG, AAC, WMA, AIFF, OPUS
Video: MP4, AVI, MOV, MKV, WebM (audio is extracted automatically)

## Troubleshooting

- **"Module not found"** — Run `./run.sh --setup` or `pip install -r requirements.txt`
- **FFmpeg not found** — Install via your package manager (`brew install ffmpeg`, `apt install ffmpeg`)
- **Out of memory** — Use a smaller model (`-m tiny` or `-m base`)
- **Poor transcription quality** — Try a larger model, check audio quality, or specify the correct language
- **HF token issues (diarization)** — Verify HF_TOKEN is set and you've accepted the pyannote model licenses on Hugging Face
- **First run is slow** — Whisper downloads the model on first use; subsequent runs use the cache at `~/.cache/whisper/`

## License

This project is open source. Please check the Whisper license for model usage terms.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the transcription model
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [Ollama](https://ollama.ai) for local LLM inference
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction from video frames
