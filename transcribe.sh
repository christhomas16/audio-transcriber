#!/bin/bash
#
# Simplest possible workflow:
#   1. Drop one or more video files into the videos/ folder
#   2. Run: ./transcribe.sh
#   3. Find the transcripts in the output/ folder as {name}_transcription.txt
#
# This only transcribes — no AI summary, no contact/OCR extraction.
# It is resume-safe: videos that already have a transcript in output/ are skipped.
#
# Pass-through options (optional):
#   ./transcribe.sh -m base          # use a faster/smaller model
#   ./transcribe.sh -m large-v3      # use a more accurate model
#   ./transcribe.sh --limit 1        # only do the first video
#   ./transcribe.sh --dry-run        # show what would be processed
#
# Supported model sizes: tiny, base, small, medium, medium.en (default), large, large-v3

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Delegate to the batch runner, forcing transcription-only mode
# (no Ollama summary, no Tesseract contact extraction) and passing
# through any extra flags the user provided.
exec ./run_batch.sh --transcribe-only --no-contacts "$@"
