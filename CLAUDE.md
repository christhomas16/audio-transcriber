# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Audio transcription tool using OpenAI's Whisper with three modes of operation:
1. **Single-file transcription** (`transcriber.py`) - Basic Whisper transcription
2. **Batch transcription + summarization** (`batch_transcribe_summarize.py`) - Processes all MP4s in a folder, summarizes via Ollama, extracts contacts via OCR
3. **Speaker diarization** (`transcriber_with_diarization.py`) - Whisper + pyannote.audio for speaker identification with voice fingerprinting

## Commands

```bash
# Setup
./run.sh --setup                    # Create venv, install deps, check FFmpeg
./run_diarization.sh --setup        # Setup including PyTorch + pyannote.audio

# Simplest workflow: drop videos in videos/, get transcripts in output/
./transcribe.sh                     # Transcribe-only (no summary, no contacts), resume-safe
./transcribe.sh -m base             # Use a faster model
./transcribe.sh --dry-run           # Preview what will be processed

# Single file transcription
./run.sh audio.wav                  # Basic transcription (default model: medium.en)
./run.sh audio.wav -m large -l es   # Custom model and language

# Batch processing
./run_batch.sh                      # Transcribe + summarize + extract contacts
./run_batch.sh --dry-run            # Preview what will be processed
./run_batch.sh --transcribe-only    # Skip Ollama summarization
./run_batch.sh --summarize-only     # Summarize existing transcriptions only
./run_batch.sh --no-contacts        # Skip OCR/contact extraction
./run_batch.sh --limit 1            # Process only first N videos

# Diarization
./run_diarization.sh audio.wav --debug  # With speaker identification debug output

# Direct Python (after activating venv)
source venv/bin/activate
python transcriber.py audio.wav
python batch_transcribe_summarize.py --help
python transcriber_with_diarization.py audio.wav --debug
```

## Architecture

- **No shared library** - the three Python scripts are independent. `transcriber.py` and `batch_transcribe_summarize.py` both use `whisper` directly (not the HuggingFace transformers pipeline). `transcriber_with_diarization.py` uses the HuggingFace `transformers` pipeline for ASR and `pyannote.audio` for diarization.
- **Shell wrappers** (`run.sh`, `run_batch.sh`, `run_diarization.sh`) handle venv activation, dependency checking, and delegate to the Python scripts. `run.sh` also supports `--batch` mode which calls `batch_transcribe_summarize.py`.
- **Batch processing** is resume-safe: it skips videos that already have output files in the output directory.
- **Ollama config** is loaded from `.env` (OLLAMA_URL, OLLAMA_MODEL) via python-dotenv; CLI flags can override.
- **Diarization** requires a Hugging Face token (HF_TOKEN in .env or environment) and accepted model licenses for pyannote/speaker-diarization-3.1 and pyannote/embedding.
- **Contact extraction** in batch mode uses FFmpeg to extract frames at 30-second intervals, then Tesseract OCR to find emails/phones/URLs/social handles.

## Dependencies

Two separate requirements files:
- `requirements.txt` - core deps (whisper, requests, python-dotenv, pytesseract, Pillow)
- `requirements_diarization.txt` - adds torch, torchaudio, transformers, pyannote.audio, scipy, soundfile

System dependencies: FFmpeg (required), Tesseract OCR (optional, for contact extraction)

## Key Conventions

- Default Whisper model is `medium.en` (English-only, good speed/accuracy tradeoff) for `transcriber.py` and batch mode. **Diarization defaults to `base`** and maps model names to HuggingFace IDs (e.g. `large` → `openai/whisper-large-v3`); it accepts only `tiny/base/small/medium/large`, not `.en` variants.
- Output files follow the pattern `{stem}_transcription.txt`, `{stem}_summary.txt`, `{stem}_contacts.txt`. Diarization writes `{stem}_transcription_with_speakers.txt`.
- Transcription files have a "TRANSCRIPTION WITH TIMESTAMPS" header, followed by `[start - end] text` lines, optionally followed by a "FULL TEXT" section (batch mode adds this). `read_transcription()` in batch mode parses both the new (FULL TEXT) and legacy (timestamp-only) formats.
- Existing output files are backed up with `_backup_{timestamp}` suffix (single-file mode) or simply skipped (batch mode)
- Batch summarization truncates transcripts to 120k chars, uses a 600s Ollama timeout, and includes a `/no_think` directive in the prompt (tuned for qwen3).

## Diarization: voice fingerprinting

`transcriber_with_diarization.py` aligns Whisper ASR chunks (30s chunks, 5s stride) to pyannote diarization segments using speaker embeddings (cosine similarity), maintaining a running-average voiceprint per speaker. Thresholds (see `get_consistent_person_name_by_voice()`): strong match ≥0.85 updates the voiceprint, weak match ≥0.7 assigns without updating, segments <2s force-match to the nearest existing speaker, and a new speaker is created only when similarity <0.7 **and** duration ≥2s. Device is auto-detected: MPS (macOS) → CUDA → CPU. Use `--debug` to log matching decisions.
