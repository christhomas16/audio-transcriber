#!/usr/bin/env python3
"""
Batch transcribe and summarize medical education conference videos.
Uses OpenAI Whisper for transcription and Ollama (qwen3:8b) for summarization.
"""

import argparse
import re
import subprocess
import sys
import os
import tempfile
import time
import warnings
from pathlib import Path

import whisper
import requests
from dotenv import load_dotenv
import pytesseract
from PIL import Image

# Load environment variables from .env
load_dotenv()

# Suppress common warnings
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*multilingual Whisper.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Defaults (Ollama config loaded from .env, CLI flags can override)
DEFAULT_VIDEOS_DIR = "videos"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_WHISPER_MODEL = "medium.en"
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
OLLAMA_TIMEOUT = 600  # 10 minutes per summary request


def load_whisper_model(model_size="medium.en"):
    """Load Whisper model once for reuse across all files."""
    print(f"Loading Whisper model: {model_size}")
    print("  This may take a moment for first-time model download...")
    try:
        model = whisper.load_model(model_size)
        print("  Model loaded successfully!")
        return model
    except Exception as e:
        print(f"  ERROR: Failed to load model: {e}")
        sys.exit(1)


def find_mp4_files(videos_dir):
    """Find all MP4 files in the given directory, sorted alphabetically."""
    videos_path = Path(videos_dir)
    if not videos_path.is_dir():
        print(f"ERROR: Videos directory '{videos_dir}' not found.")
        sys.exit(1)
    mp4_files = sorted(videos_path.glob("*.mp4"))
    if not mp4_files:
        print(f"No MP4 files found in '{videos_dir}'.")
        sys.exit(1)
    return mp4_files


def get_output_paths(video_path, output_dir):
    """Derive transcription, summary, and contacts file paths from a video file path."""
    stem = video_path.stem
    out = Path(output_dir)
    transcription_path = out / f"{stem}_transcription.txt"
    summary_path = out / f"{stem}_summary.txt"
    contacts_path = out / f"{stem}_contacts.txt"
    return transcription_path, summary_path, contacts_path


def transcribe_video(model, video_path, output_path):
    """
    Transcribe a video file using a pre-loaded Whisper model.
    Saves timestamped transcription to output_path.
    Returns the plain text transcription.
    """
    print(f"  Transcribing: {video_path.name}")
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"    File size: {file_size_mb:.1f} MB")

    result = model.transcribe(str(video_path), language="en", fp16=False)

    # Save with timestamps (matching existing format from transcriber.py)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("TRANSCRIPTION WITH TIMESTAMPS\n")
        f.write("=" * 50 + "\n\n")
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            f.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")
        # Append plain text at the end for easy reading and extraction
        f.write("\n" + "=" * 50 + "\n")
        f.write("FULL TEXT\n")
        f.write("=" * 50 + "\n\n")
        f.write(result["text"])

    print(f"    Transcription saved: {output_path.name}")
    return result["text"]


def read_transcription(transcription_path):
    """
    Read an existing transcription file and extract the plain text.
    Handles both the new format (with FULL TEXT section) and the
    legacy format (timestamps only, from transcriber.py).
    """
    content = transcription_path.read_text(encoding="utf-8")

    # Try to find the FULL TEXT section first (our new format)
    marker = "FULL TEXT\n" + "=" * 50
    if marker in content:
        idx = content.index(marker) + len(marker)
        return content[idx:].strip()

    # Fallback: strip timestamp prefixes from each line (legacy format)
    lines = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("[") and "]" in line:
            text = line.split("]", 1)[1].strip()
            if text:
                lines.append(text)
        elif line and not line.startswith("=") and line != "TRANSCRIPTION WITH TIMESTAMPS":
            lines.append(line)
    return " ".join(lines)


def build_summary_prompt(transcription_text, video_name):
    """Build the summarization prompt for a medical education conference talk."""
    return f"""/no_think
You are summarizing a medical education conference presentation. The talk is titled:
"{video_name}"

Below is the full transcription of the presentation. Please provide a structured summary with the following sections:

## Title
Restate the presentation title.

## Presenters
List the presenter(s) and their affiliations if mentioned.

## Overview
A 2-3 sentence high-level summary of what this presentation is about and why it matters.

## Key Points
- Bullet point list of the main ideas, methods, tools, or innovations presented (4-8 bullets).

## Implementation Details
Briefly describe any specific tools, technologies, AI models, or platforms mentioned and how they were used.

## Results and Impact
Summarize any outcomes, results, evaluation data, or impact metrics shared.

## Relevance to Medical Education
1-2 sentences on how this work contributes to the broader field of medical education or could be applied at other institutions.

Keep the summary factual and grounded in the transcription content. Do not invent details not present in the talk. Use clear, professional language suitable for medical educators.

---

TRANSCRIPTION:
{transcription_text}"""


def summarize_with_ollama(text, video_name, ollama_url, model_name):
    """
    Send transcription text to Ollama for summarization.
    Returns the summary string.
    """
    # Truncate very long transcriptions to avoid overwhelming the LLM context
    max_chars = 120000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Transcription truncated due to length]"

    prompt = build_summary_prompt(text, video_name)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 2048,
        },
    }

    url = f"{ollama_url}/api/generate"
    print(f"    Sending to Ollama ({model_name}) for summarization...")

    response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()

    result = response.json()
    summary = result.get("response", "").strip()

    if not summary:
        raise ValueError("Ollama returned an empty response")

    return summary


def save_summary(summary_text, output_path):
    """Save a summary to a text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"    Summary saved: {output_path.name}")


def extract_frames(video_path, output_dir, interval=30):
    """
    Extract key frames from a video at a given interval (seconds).
    Returns a list of frame image paths.
    """
    frames_dir = Path(output_dir) / "_frames" / video_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_pattern = str(frames_dir / "frame_%04d.png")

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",
        "-y",
        frame_pattern,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed: {result.stderr[:200]}")

    frames = sorted(frames_dir.glob("frame_*.png"))
    return frames


def ocr_frames(frame_paths):
    """Run OCR on extracted frames and return combined text."""
    all_text = []
    for frame_path in frame_paths:
        try:
            img = Image.open(frame_path)
            text = pytesseract.image_to_string(img).strip()
            if text:
                all_text.append(text)
        except Exception:
            continue
    return "\n".join(all_text)


def extract_contacts(text):
    """
    Extract contact information from text using regex patterns.
    Returns a dict with lists of emails, phones, urls, and twitter handles.
    """
    contacts = {
        "emails": [],
        "phones": [],
        "urls": [],
        "twitter": [],
    }

    # Emails
    emails = re.findall(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
    )
    contacts["emails"] = sorted(set(e.lower() for e in emails))

    # Phone numbers (US formats)
    phones = re.findall(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text
    )
    contacts["phones"] = sorted(set(phones))

    # URLs (http/https/www)
    urls = re.findall(
        r"https?://[^\s<>\"'\)]+|www\.[^\s<>\"'\)]+", text
    )
    # Clean trailing punctuation
    cleaned_urls = []
    for u in urls:
        u = u.rstrip(".,;:!?)")
        cleaned_urls.append(u)
    contacts["urls"] = sorted(set(cleaned_urls))

    # Twitter/X handles — use lookbehind to skip email domains
    twitter = re.findall(r"(?<![a-zA-Z0-9._%+\-])@[a-zA-Z0-9_]{2,15}\b", text)
    contacts["twitter"] = sorted(set(twitter))

    return contacts


def save_contacts(source_labels, output_path):
    """
    Save extracted contacts to a file.
    source_labels is a dict mapping source name to its contacts dict.
    """
    has_any = False
    for src_contacts in source_labels.values():
        for vals in src_contacts.values():
            if vals:
                has_any = True
                break

    if not has_any:
        return False

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("EXTRACTED CONTACT INFORMATION\n")
        f.write("=" * 50 + "\n\n")

        for source_name, src_contacts in source_labels.items():
            # Check if this source has anything
            if not any(src_contacts.values()):
                continue
            f.write(f"Source: {source_name}\n")
            f.write("-" * 40 + "\n")
            if src_contacts["emails"]:
                f.write("Emails:\n")
                for e in src_contacts["emails"]:
                    f.write(f"  {e}\n")
            if src_contacts["phones"]:
                f.write("Phone Numbers:\n")
                for p in src_contacts["phones"]:
                    f.write(f"  {p}\n")
            if src_contacts["urls"]:
                f.write("URLs:\n")
                for u in src_contacts["urls"]:
                    f.write(f"  {u}\n")
            if src_contacts["twitter"]:
                f.write("Social Media:\n")
                for t in src_contacts["twitter"]:
                    f.write(f"  {t}\n")
            f.write("\n")

    return True


def extract_contacts_from_video(video_path, transcription_text, output_dir):
    """
    Extract contact info from both audio transcription and video frames (OCR).
    Returns True if any contacts were found and saved.
    """
    stem = video_path.stem
    contacts_path = Path(output_dir) / f"{stem}_contacts.txt"

    if contacts_path.exists():
        print(f"  Contacts file exists, skipping: {contacts_path.name}")
        return None  # signal skip

    source_contacts = {}

    # Extract from transcription text
    if transcription_text:
        audio_contacts = extract_contacts(transcription_text)
        source_contacts["Audio Transcription"] = audio_contacts

    # Extract from video frames via OCR
    print(f"    Extracting frames for OCR...")
    try:
        frames = extract_frames(video_path, output_dir, interval=30)
        print(f"    Extracted {len(frames)} frames, running OCR...")
        ocr_text = ocr_frames(frames)
        if ocr_text:
            slide_contacts = extract_contacts(ocr_text)
            source_contacts["Video Slides (OCR)"] = slide_contacts
    except Exception as e:
        print(f"    WARNING: Frame extraction/OCR failed: {e}")

    # Clean up frames
    frames_dir = Path(output_dir) / "_frames" / video_path.stem
    if frames_dir.exists():
        for f in frames_dir.glob("*"):
            f.unlink()
        frames_dir.rmdir()
    # Remove _frames parent if empty
    frames_parent = Path(output_dir) / "_frames"
    if frames_parent.exists() and not any(frames_parent.iterdir()):
        frames_parent.rmdir()

    # Save if anything was found
    saved = save_contacts(source_contacts, contacts_path)
    if saved:
        print(f"    Contacts saved: {contacts_path.name}")
        return True
    else:
        print(f"    No contact info found")
        return False


def process_one_video(model, video_path, output_dir, ollama_url, ollama_model,
                      extract_contacts_enabled=True):
    """
    Process a single video: transcribe (if needed) and summarize (if needed).
    Returns a result dict with status information.
    """
    result = {
        "video": video_path.name,
        "transcribed": False,
        "summarized": False,
        "contacts_extracted": False,
        "skipped_transcription": False,
        "skipped_summary": False,
        "skipped_contacts": False,
        "error": None,
    }

    transcription_path, summary_path, contacts_path = get_output_paths(video_path, output_dir)

    try:
        # Step 1: Transcription
        if transcription_path.exists():
            print(f"  Transcription exists, skipping: {transcription_path.name}")
            result["skipped_transcription"] = True
            plain_text = read_transcription(transcription_path)
        elif model is not None:
            start_time = time.time()
            plain_text = transcribe_video(model, video_path, transcription_path)
            elapsed = time.time() - start_time
            print(f"    Transcription took {elapsed / 60:.1f} minutes")
            result["transcribed"] = True
        else:
            raise ValueError(
                "No transcription file found and no model loaded (--summarize-only mode)"
            )

        # Step 2: Summarization
        if ollama_url is None:
            result["skipped_summary"] = True
        elif summary_path.exists():
            print(f"  Summary exists, skipping: {summary_path.name}")
            result["skipped_summary"] = True
        else:
            if not plain_text or len(plain_text.strip()) < 50:
                raise ValueError(
                    "Transcription text is too short or empty to summarize"
                )
            start_time = time.time()
            summary = summarize_with_ollama(
                plain_text, video_path.stem, ollama_url, ollama_model
            )
            save_summary(summary, summary_path)
            elapsed = time.time() - start_time
            print(f"    Summarization took {elapsed:.1f} seconds")
            result["summarized"] = True

        # Step 3: Contact extraction (audio + video frames OCR)
        if not extract_contacts_enabled:
            result["skipped_contacts"] = True
        else:
            contact_result = extract_contacts_from_video(
                video_path, plain_text, output_dir
            )
            if contact_result is None:
                result["skipped_contacts"] = True
            elif contact_result:
                result["contacts_extracted"] = True

    except Exception as e:
        result["error"] = str(e)
        print(f"    ERROR: {e}")

    return result


def print_report(results, total_time):
    """Print a final summary report of the batch run."""
    print("\n" + "=" * 70)
    print("BATCH PROCESSING REPORT")
    print("=" * 70)

    total = len(results)
    transcribed = sum(1 for r in results if r["transcribed"])
    summarized = sum(1 for r in results if r["summarized"])
    contacts = sum(1 for r in results if r["contacts_extracted"])
    skipped_t = sum(1 for r in results if r["skipped_transcription"])
    skipped_s = sum(1 for r in results if r["skipped_summary"])
    skipped_c = sum(1 for r in results if r["skipped_contacts"])
    errors = [r for r in results if r["error"]]

    print(f"\nTotal videos:              {total}")
    print(f"Newly transcribed:         {transcribed}")
    print(f"Newly summarized:          {summarized}")
    print(f"Contacts extracted:        {contacts}")
    print(f"Transcriptions skipped:    {skipped_t}")
    print(f"Summaries skipped:         {skipped_s}")
    print(f"Contacts skipped:          {skipped_c}")
    print(f"Errors:                    {len(errors)}")
    print(f"Total time:                {total_time / 60:.1f} minutes")

    if errors:
        print(f"\n{'=' * 70}")
        print("FAILURES:")
        print(f"{'=' * 70}")
        for r in errors:
            print(f"  {r['video']}")
            print(f"    Error: {r['error']}")
            print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe and summarize medical education conference videos"
    )
    parser.add_argument(
        "-v",
        "--videos-dir",
        default=DEFAULT_VIDEOS_DIR,
        help=f"Directory containing MP4 files (default: {DEFAULT_VIDEOS_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_WHISPER_MODEL,
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v2", "large-v3"],
        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama server URL (reads OLLAMA_URL from .env)",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name (reads OLLAMA_MODEL from .env)",
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only transcribe, skip summarization",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only summarize existing transcriptions, skip transcription",
    )
    parser.add_argument(
        "--no-contacts",
        action="store_true",
        help="Skip contact info extraction from audio and video frames",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N videos (default: 0 = all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually doing it",
    )

    args = parser.parse_args()

    # Validate Ollama config when summarization is needed
    if not args.transcribe_only:
        if not args.ollama_url:
            print("ERROR: OLLAMA_URL not set. Add it to .env or pass --ollama-url.")
            sys.exit(1)
        if not args.ollama_model:
            print("ERROR: OLLAMA_MODEL not set. Add it to .env or pass --ollama-model.")
            sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find videos
    mp4_files = find_mp4_files(args.videos_dir)
    if args.limit > 0:
        mp4_files = mp4_files[: args.limit]
        print(f"\nFound {len(mp4_files)} MP4 files (limited to {args.limit})")
    else:
        print(f"\nFound {len(mp4_files)} MP4 files in '{args.videos_dir}'")

    # Dry run mode
    if args.dry_run:
        for i, vf in enumerate(mp4_files, 1):
            t_path, s_path, c_path = get_output_paths(vf, args.output_dir)
            t_status = "EXISTS" if t_path.exists() else "PENDING"
            s_status = "EXISTS" if s_path.exists() else "PENDING"
            c_status = "EXISTS" if c_path.exists() else "PENDING"
            print(f"  {i:2d}. {vf.name}")
            print(f"      Transcription: {t_status}  |  Summary: {s_status}  |  Contacts: {c_status}")
        return

    # Load Whisper model (once) -- skip if summarize-only mode
    model = None
    if not args.summarize_only:
        model = load_whisper_model(args.model)

    # Check Ollama connectivity (unless transcribe-only)
    if not args.transcribe_only:
        try:
            resp = requests.get(f"{args.ollama_url}/api/tags", timeout=10)
            resp.raise_for_status()
            print(f"Ollama server at {args.ollama_url} is reachable")
        except Exception as e:
            print(f"WARNING: Cannot reach Ollama at {args.ollama_url}: {e}")
            print("Summarization will likely fail.")
            if not args.summarize_only:
                print("Continuing with transcription only...")
                args.transcribe_only = True
            else:
                print("Exiting.")
                sys.exit(1)

    # Process each video
    print(f"\nStarting batch processing...\n")
    results = []
    batch_start = time.time()

    try:
        for i, video_path in enumerate(mp4_files, 1):
            print(f"\n[{i}/{len(mp4_files)}] {video_path.name}")
            print("-" * 60)

            ollama_url = None if args.transcribe_only else args.ollama_url
            ollama_model = args.ollama_model

            r = process_one_video(
                model, video_path, args.output_dir, ollama_url, ollama_model,
                extract_contacts_enabled=not args.no_contacts,
            )
            results.append(r)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")

    total_time = time.time() - batch_start
    print_report(results, total_time)


if __name__ == "__main__":
    main()
