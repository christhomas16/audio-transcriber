#!/usr/bin/env python3
"""
Audio Transcriber using OpenAI's Whisper model
"""

import argparse
import sys
import os
import whisper
from pathlib import Path
from datetime import datetime


def backup_existing_file(output_file):
    """
    Create a backup of an existing file with timestamp.
    
    Args:
        output_file (str): Path to the output file
        
    Returns:
        str: Path to the backup file if created, None if no backup needed
    """
    if os.path.exists(output_file):
        # Generate timestamp for backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path(output_file)
        backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        backup_path = file_path.parent / backup_name
        
        try:
            os.rename(output_file, backup_path)
            print(f"📁 Backed up existing file to: {backup_path}")
            return str(backup_path)
        except Exception as e:
            print(f"⚠️ Warning: Could not backup existing file: {e}")
            return None
    return None


def transcribe_audio(audio_file, model_size="medium.en", language="en", output_file=None, with_timestamps=True):
    """
    Transcribe audio file using Whisper model
    
    Args:
        audio_file (str): Path to audio file
        model_size (str): Whisper model size (tiny, base, small, medium, large, medium.en)
        language (str): Language code (e.g., 'en', 'es', 'fr')
        output_file (str): Optional output file path
        with_timestamps (bool): Include timestamps in output
    
    Returns:
        str: Transcribed text
    """
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        return None
    
    # Auto-select appropriate model based on language
    if language and language != "en" and model_size.endswith(".en"):
        print(f"Warning: Using English-only model '{model_size}' for language '{language}'")
        print("Switching to multilingual 'medium' model for better results")
        model_size = "medium"
    elif not language and model_size.endswith(".en"):
        print("Note: Using English-only model for auto-detection. Consider 'medium' for multilingual audio.")
    
    print(f"Loading Whisper model: {model_size}")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    print(f"Transcribing audio file: {audio_file}")
    try:
        # Transcribe audio file
        result = model.transcribe(audio_file, language=language, fp16=False)
        
        # Print transcription to console
        print("\n" + "="*50)
        print("TRANSCRIPTION:")
        print("="*50)
        print(result["text"])
        print("="*50)
        
        # Save transcription to file if specified
        if output_file:
            save_transcription(result, output_file, with_timestamps)
        else:
            # Generate default output filename
            audio_path = Path(audio_file)
            default_output = audio_path.stem + "_transcription.txt"
            save_transcription(result, default_output, with_timestamps)
        
        return result["text"]
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


def save_transcription(result, output_file, with_timestamps=True):
    """
    Save transcription to file
    
    Args:
        result (dict): Whisper transcription result
        output_file (str): Output file path
        with_timestamps (bool): Include timestamps in output
    """
    try:
        # Backup existing file if it exists
        backup_existing_file(output_file)
        
        with open(output_file, "w", encoding="utf-8") as f:
            if with_timestamps:
                f.write("TRANSCRIPTION WITH TIMESTAMPS\n")
                f.write("="*50 + "\n\n")
                for segment in result["segments"]:
                    start, end, text = segment["start"], segment["end"], segment["text"]
                    f.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")
            else:
                f.write("TRANSCRIPTION\n")
                f.write("="*50 + "\n\n")
                f.write(result["text"])
        
        print(f"\nTranscription saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving transcription: {e}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI Whisper")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("-m", "--model", choices=["tiny", "base", "small", "medium", "large", "medium.en"], 
                       default="medium.en", help="Whisper model size (default: medium.en for English)")
    parser.add_argument("-l", "--language", default="en", 
                       help="Language code (e.g., en, es, fr) - leave empty for auto-detection")
    parser.add_argument("-o", "--output", help="Output file path (default: [audio_filename]_transcription.txt)")
    parser.add_argument("--no-timestamps", action="store_true", 
                       help="Don't include timestamps in output file")
    
    args = parser.parse_args()
    
    # Convert language to None if empty string for auto-detection
    language = args.language if args.language else None
    
    # Transcribe the audio
    result = transcribe_audio(
        audio_file=args.audio_file,
        model_size=args.model,
        language=language,
        output_file=args.output,
        with_timestamps=not args.no_timestamps
    )
    
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()