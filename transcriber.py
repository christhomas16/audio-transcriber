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
import wave
import contextlib


def get_audio_duration(audio_file):
    """
    Get audio file duration in seconds.
    
    Args:
        audio_file (str): Path to audio file
        
    Returns:
        float: Duration in seconds, or None if unable to determine
    """
    try:
        # Try to get duration for WAV files
        if audio_file.lower().endswith('.wav'):
            with contextlib.closing(wave.open(audio_file, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        else:
            # For other formats, we'll estimate based on file size
            # This is a rough approximation
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            # Rough estimate: 1MB ≈ 1 minute for compressed audio
            return file_size_mb * 60
    except:
        return None


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
        print(f"❌ Error: Audio file '{audio_file}' not found.")
        return None
    
    print(f"📂 Processing audio file: {audio_file}")
    print(f"🌍 Language: {language if language else 'Auto-detect'}")
    
    # Get audio duration and show time estimates
    duration = get_audio_duration(audio_file)
    if duration:
        print(f"⏱️ Audio duration: {duration:.1f} seconds")
        
        # Estimate processing time based on model size and duration
        if model_size in ["tiny"]:
            processing_factor = 0.1  # Very fast
        elif model_size in ["base"]:
            processing_factor = 0.2  # Fast
        elif model_size in ["small"]:
            processing_factor = 0.3  # Medium
        elif model_size in ["medium", "medium.en"]:
            processing_factor = 0.5  # Good balance
        elif model_size in ["large"]:
            processing_factor = 0.8  # Slower but accurate
        else:
            processing_factor = 0.4  # Default estimate
        
        estimated_time = duration * processing_factor
        if estimated_time < 60:
            print(f"⏰ Estimated processing time: {estimated_time:.0f} seconds")
        else:
            print(f"⏰ Estimated processing time: {estimated_time/60:.1f} minutes")
    else:
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.1f} MB")
    
    print("")
    
    # Auto-select appropriate model based on language
    if language and language != "en" and model_size.endswith(".en"):
        print(f"⚠️ Warning: Using English-only model '{model_size}' for language '{language}'")
        print("🔄 Switching to multilingual 'medium' model for better results")
        model_size = "medium"
    elif not language and model_size.endswith(".en"):
        print("💡 Note: Using English-only model for auto-detection. Consider 'medium' for multilingual audio.")
    
    print(f"🤖 Loading Whisper model: {model_size}")
    print("  ⏳ This may take a moment for first-time model download...")
    try:
        model = whisper.load_model(model_size)
        print("  ✅ Model loaded successfully!")
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return None
    
    print("")
    print(f"🎵 Starting transcription...")
    print("  📝 Processing audio and generating text...")
    try:
        # Transcribe audio file
        result = model.transcribe(audio_file, language=language, fp16=False)
        print("  ✅ Transcription completed!")
        
        # Print transcription to console
        print("\n" + "="*50)
        print("TRANSCRIPTION:")
        print("="*50)
        print(result["text"])
        print("="*50)
        
        # Save transcription to file
        print("")
        print("💾 Saving transcription to file...")
        if output_file:
            save_transcription(result, output_file, with_timestamps)
        else:
            # Generate default output filename
            audio_path = Path(audio_file)
            default_output = audio_path.stem + "_transcription.txt"
            print(f"  📝 Using default filename: {default_output}")
            save_transcription(result, default_output, with_timestamps)
        
        # Final completion summary
        print("")
        print("🎉 Transcription process completed successfully!")
        if duration:
            print(f"📊 Processed {duration:.1f} seconds of audio")
        segments_count = len(result.get("segments", []))
        if segments_count > 0:
            print(f"📝 Generated {segments_count} text segments")
        
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
        
        print("  📁 Writing transcription data...")
        with open(output_file, "w", encoding="utf-8") as f:
            if with_timestamps:
                print("  ⏰ Including timestamps...")
                f.write("TRANSCRIPTION WITH TIMESTAMPS\n")
                f.write("="*50 + "\n\n")
                segments = result["segments"]
                print(f"  📝 Writing {len(segments)} segments...")
                for i, segment in enumerate(segments, 1):
                    start, end, text = segment["start"], segment["end"], segment["text"]
                    f.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")
                    if i % 10 == 0:  # Progress update every 10 segments
                        print(f"    📊 Processed {i}/{len(segments)} segments...")
            else:
                print("  📄 Writing plain text format...")
                f.write("TRANSCRIPTION\n")
                f.write("="*50 + "\n\n")
                f.write(result["text"])
        
        print("  ✅ File write completed!")
        print(f"\n💾 Transcription saved to: {output_file}")
        
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