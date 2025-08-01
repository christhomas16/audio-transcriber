#!/usr/bin/env python3
"""
Audio Transcriber with Speaker Diarization
Combines OpenAI Whisper for transcription with Pyannote for speaker identification.
"""

import warnings
import os
import sys
import logging
import argparse
import tempfile
import subprocess
import numpy as np
import torch
import soundfile as sf
from datetime import datetime
from dotenv import load_dotenv
from transformers import pipeline
from pyannote.audio import Pipeline
from pyannote.audio import Model as PyannoteModel
from pyannote.audio import Inference
from scipy.spatial.distance import cdist

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.pipelines.speaker_verification")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.tasks.segmentation.mixins")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.model")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*multilingual Whisper.*")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class AudioTranscriberWithDiarization:
    def __init__(self, model_name="openai/whisper-base", debug=False):
        """
        Initialize the audio transcriber with diarization.
        
        Args:
            model_name (str): Whisper model to use
            debug (bool): Whether to enable debug mode
        """
        self.debug = debug
        self.whisper_model_name = model_name
        
        # Load environment variables and check for token
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("\nError: Hugging Face token not found.")
            print("Please set your HF_TOKEN environment variable.")
            print("You can get a token from: https://huggingface.co/settings/tokens")
            print("Then add it to your .env file or export it:")
            print("  echo 'HF_TOKEN=your_token_here' > .env")
            sys.exit(1)
        
        # Note about version warnings
        print("Note: You may see PyTorch version warnings. These are normal and won't affect functionality.")
        
        # Speaker identification components
        self.person_counter = 1
        self.speaker_voiceprints = {}
        
        # Initialize models
        print("Initializing ASR + Diarization pipeline...")
        try:
            # Initialize ASR pipeline
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            torch_dtype = torch.float32
            
            # Handle language-specific models
            if model_name.endswith(".en") and language and language != "en":
                print(f"Warning: Using English-only model '{model_name}' but language is set to '{language}'")
                print("Consider using a multilingual model for better results")
            
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch_dtype,
                device=device,
                return_timestamps=True,
                generate_kwargs={"max_new_tokens": 256}  # Reduced from 448 to avoid token limit issues
            )
            
            # Initialize Diarization pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            # Initialize Speaker Embedding model for voiceprinting
            self.embedding_model = PyannoteModel.from_pretrained(
                "pyannote/embedding",
                use_auth_token=hf_token
            )
            
            if device == "mps":
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("mps"))
                self.embedding_model.to(torch.device("mps"))
            elif device == "cuda":
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                self.embedding_model.to(torch.device("cuda"))
            
            self.device_info = device
            print(f"Models loaded successfully on device: {self.device_info}\n")
        except Exception as e:
            print(f"\nError during model initialization: {e}")
            sys.exit(1)

    def load_audio_with_ffmpeg(self, audio_file, target_sr=16000):
        """
        Load audio file using FFmpeg to handle various formats including M4A.
        
        Args:
            audio_file (str): Path to audio file
            target_sr (int): Target sample rate
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            print("  📁 Creating temporary file...")
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            print("  🔄 Converting audio with FFmpeg...")
            # Use FFmpeg to convert to WAV
            cmd = [
                'ffmpeg', '-i', audio_file,
                '-acodec', 'pcm_s16le',
                '-ar', str(target_sr),
                '-ac', '1',  # Convert to mono
                '-y',  # Overwrite output file
                temp_wav_path
            ]
            
            # Run FFmpeg command with progress
            print("  ⏳ Running FFmpeg conversion (this may take a moment)...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  ❌ FFmpeg error: {result.stderr}")
                return None, None
            
            print("  ✅ FFmpeg conversion completed!")
            print("  📖 Loading converted audio file...")
            
            # Load the converted WAV file
            audio_data, sample_rate = sf.read(temp_wav_path)
            
            print(f"  ✅ Audio loaded: {len(audio_data)} samples at {sample_rate} Hz")
            
            # Clean up temporary file
            os.unlink(temp_wav_path)
            
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"  ❌ Error loading audio with FFmpeg: {e}")
            return None, None

    def load_audio(self, audio_file):
        """
        Load audio file with fallback methods.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        # First try soundfile directly
        try:
            print("  🎵 Trying to load audio directly...")
            audio_data, sample_rate = sf.read(audio_file)
            print(f"  ✅ Direct loading successful: {len(audio_data)} samples at {sample_rate} Hz")
            return audio_data, sample_rate
        except Exception as e:
            if self.debug:
                print(f"  ⚠️ Soundfile failed: {e}")
        
        # Fallback to FFmpeg
        print("  🔄 Trying FFmpeg conversion...")
        return self.load_audio_with_ffmpeg(audio_file)
    
    def get_embedding(self, audio_path, segment):
        """Extracts an embedding for a given audio file path and speaker segment."""
        # Segments shorter than a certain duration can cause errors in the embedding model.
        MIN_DURATION = 0.05  # 50 milliseconds, a safe threshold for pyannote/embedding
        if segment.duration < MIN_DURATION:
            if self.debug:
                print(f"⏩ Skipping embedding for very short segment ({segment.duration:.3f}s)")
            return None
            
        try:
            inference = Inference(self.embedding_model, window="whole")
            embedding = inference(audio_path, segment)
            return embedding
        except Exception as e:
            if self.debug:
                print(f"❌ Error extracting embedding: {e}")
            return None
    
    def get_consistent_person_name_by_voice(self, embedding, duration, 
                                            strong_match_threshold=0.85, 
                                            weak_match_threshold=0.7, 
                                            min_duration_for_new_speaker=2.0):
        """
        Assigns a consistent person name using a tiered confidence system.
        """
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)

        if not self.speaker_voiceprints:
            person_name = f"Speaker {self.person_counter}"
            self.speaker_voiceprints[person_name] = {'embedding': embedding, 'count': 1}
            self.person_counter += 1
            if self.debug:
                print(f"\n🎤 New speaker identified: {person_name} (first voice)")
            return person_name

        distances = {
            name: cdist(embedding, vp['embedding'], metric='cosine')[0, 0]
            for name, vp in self.speaker_voiceprints.items()
        }

        min_distance = min(distances.values())
        best_match_person = min(distances, key=distances.get)
        similarity = 1 - min_distance

        if similarity >= strong_match_threshold:
            person_name = best_match_person
            self.update_voiceprint(person_name, embedding)
            if self.debug:
                print(f"✅ Strong match: {person_name} (similarity: {similarity:.2f}), voiceprint updated.")
            return person_name
        elif similarity >= weak_match_threshold:
            person_name = best_match_person
            if self.debug:
                print(f"👉 Weak match: {person_name} (similarity: {similarity:.2f}), assigning but not updating voiceprint.")
            return person_name
        elif duration < min_duration_for_new_speaker:
            person_name = best_match_person
            if self.debug:
                print(f"⚠️ Very weak match (sim: {similarity:.2f}), but short duration ({duration:.1f}s). Force-matching to {person_name} without updating.")
            return person_name
        else:
            person_name = f"Speaker {self.person_counter}"
            self.speaker_voiceprints[person_name] = {'embedding': embedding, 'count': 1}
            self.person_counter += 1
            if self.debug:
                print(f"\n🎤 New speaker identified: {person_name} (similarity to closest match {best_match_person}: {similarity:.2f})")
            return person_name

    def update_voiceprint(self, person_name, embedding):
        """Updates a person's voiceprint with a new embedding using a running average."""
        if person_name in self.speaker_voiceprints:
            current_vp = self.speaker_voiceprints[person_name]
            current_count = current_vp['count']
            
            # Update embedding using running average
            current_embedding = current_vp['embedding']
            new_embedding = (current_embedding * current_count + embedding) / (current_count + 1)
            
            self.speaker_voiceprints[person_name] = {
                'embedding': new_embedding,
                'count': current_count + 1
            }
        else:
            self.speaker_voiceprints[person_name] = {'embedding': embedding, 'count': 1}

    def find_best_speaker(self, chunk_start, chunk_end, speaker_timeline):
        """Find the speaker with the best overlap for a given time chunk"""
        best_speaker = None
        best_overlap = 0

        chunk_duration = chunk_end - chunk_start

        for speaker_segment in speaker_timeline:
            # Calculate overlap
            overlap_start = max(chunk_start, speaker_segment['start'])
            overlap_end = min(chunk_end, speaker_segment['end'])
            overlap_duration = max(0, overlap_end - overlap_start)

            # Calculate overlap ratio
            overlap_ratio = overlap_duration / chunk_duration if chunk_duration > 0 else 0

            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_speaker = speaker_segment['speaker']

        return best_speaker if best_overlap > 0.1 else None  # Require at least 10% overlap

    def align_asr_with_diarization(self, asr_result, diarization, temp_audio_file):
        """Align ASR chunks with speaker diarization using voice embeddings."""
        segments = []
        asr_chunks = asr_result.get("chunks", [])
        if not asr_chunks:
            return segments

        # Aggregate embeddings for each speaker label in the current chunk
        chunk_embeddings = {}
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if speaker_label not in chunk_embeddings:
                chunk_embeddings[speaker_label] = []
            
            embedding = self.get_embedding(temp_audio_file, turn)
            
            if embedding is not None:
                chunk_embeddings[speaker_label].append(embedding)

        # Create a mapping from chunk speaker labels to consistent person names
        chunk_speaker_map = {}
        for speaker_label, embeddings in chunk_embeddings.items():
            if not embeddings:
                continue
            avg_embedding = np.mean(embeddings, axis=0)
            duration = sum(turn.end - turn.start for turn, _, lbl in diarization.itertracks(yield_label=True) if lbl == speaker_label)
            person_name = self.get_consistent_person_name_by_voice(avg_embedding, duration)
            chunk_speaker_map[speaker_label] = person_name

        # Create a speaker timeline using the consistent names
        speaker_timeline = []
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if speaker_label in chunk_speaker_map:
                person_name = chunk_speaker_map[speaker_label]
                speaker_timeline.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': person_name
                })

        print(f"  📊 Found {len(speaker_timeline)} speaker segments")
        print(f"  📝 Found {len(asr_chunks)} transcription chunks")
        if self.debug:
            print(f"  🔍 Processing speaker assignments...")

        # Assign speakers to ASR chunks
        for chunk in asr_chunks:
            chunk_start = chunk['timestamp'][0] if chunk['timestamp'][0] is not None else 0
            chunk_end = chunk['timestamp'][1] if chunk['timestamp'][1] is not None else chunk_start + 1
            chunk_text = chunk['text'].strip()

            if not chunk_text:
                continue

            best_speaker = self.find_best_speaker(chunk_start, chunk_end, speaker_timeline)

            if best_speaker:
                segments.append({
                    'speaker': best_speaker,
                    'text': chunk_text,
                    'start_time': chunk_start,
                    'end_time': chunk_end
                })
                if self.debug:
                    print(f"🎯 [{chunk_start:.1f}s-{chunk_end:.1f}s] {best_speaker}: {chunk_text[:50]}...")

        return segments

    def transcribe_with_diarization(self, audio_file, language="en", output_file=None, with_timestamps=True):
        """
        Transcribe audio file with speaker diarization.
        
        Args:
            audio_file (str): Path to audio file
            language (str): Language code (e.g., 'en', 'es', 'fr')
            output_file (str): Optional output file path
            with_timestamps (bool): Include timestamps in output
        
        Returns:
            list: List of transcription segments with speaker information
        """
        
        # Check if audio file exists
        if not os.path.exists(audio_file):
            print(f"Error: Audio file '{audio_file}' not found.")
            return None
        
        print(f"📂 Processing audio file: {audio_file}")
        print(f"🌍 Language: {language if language else 'Auto-detect'}")
        
        try:
            # Check file extension and provide format info
            file_ext = os.path.splitext(audio_file)[1].lower()
            supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.mp4']
            
            if file_ext not in supported_formats:
                print(f"Warning: File extension '{file_ext}' may not be supported. Supported formats: {', '.join(supported_formats)}")
            
            # Load and normalize audio using the new load_audio method
            audio_data, sample_rate = self.load_audio(audio_file)
            
            if audio_data is None or sample_rate is None:
                print("Failed to load audio file. Please check if FFmpeg is installed.")
                return None
            
            # Show audio duration after successful loading
            duration_seconds = len(audio_data) / sample_rate
            print(f"⏱️ Audio duration: {duration_seconds:.1f} seconds")
            
            # Estimate processing time
            if duration_seconds < 60:
                print("⏰ Estimated processing time: 1-3 minutes")
            elif duration_seconds < 300:  # 5 minutes
                print("⏰ Estimated processing time: 3-8 minutes")
            elif duration_seconds < 900:  # 15 minutes
                print("⏰ Estimated processing time: 8-15 minutes")
            elif duration_seconds < 1800:  # 30 minutes
                print("⏰ Estimated processing time: 15-25 minutes")
            else:
                print("⏰ Estimated processing time: 25+ minutes (long audio file)")
            
            print("📋 Processing stages:")
            print("  1. 🎯 Speaker diarization (70% of total time)")
            print("  2. 🎤 Speech recognition (20% of total time)")
            print("  3. 🔗 Speaker alignment (10% of total time)")
            
            print("")
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data /= np.max(np.abs(audio_data))
            
            # Save audio to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, sample_rate)
                temp_audio_file = temp_file.name

            try:
                # Run diarization
                print("🎯 Running speaker diarization...")
                print("  ⏳ This is the most time-consuming step...")
                print("  📊 Processing audio for speaker identification...")
                diarization = self.diarization_pipeline(temp_audio_file)
                print("  ✅ Diarization completed!")
                print("  🎤 Speaker segments identified and voiceprints created")

                # Run ASR with chunking for long-form audio
                print("🎤 Running speech recognition...")
                print("  ⏳ Processing audio chunks (faster than diarization)...")
                
                # Run ASR with chunking for long-form audio
                # Note: language parameter is handled differently in transformers pipeline
                print("  📝 Processing audio in chunks...")
                asr_result = self.asr_pipeline(
                    audio_data, 
                    return_timestamps=True,
                    chunk_length_s=30,
                    stride_length_s=5,
                    ignore_warning=True  # Suppress chunking warning
                )
                print("  ✅ Speech recognition completed!")
                
                # Check if we got any results
                if not asr_result or not asr_result.get("chunks"):
                    print("  ⚠️ No transcription chunks found, trying alternative approach...")
                    # Try without chunking for very short audio
                    asr_result = self.asr_pipeline(
                        audio_data, 
                        return_timestamps=True,
                        ignore_warning=True
                    )

                # Combine ASR and diarization results
                print("🔗 Aligning transcription with speaker information...")
                segments = self.align_asr_with_diarization(asr_result, diarization, temp_audio_file)
                print("  ✅ Alignment completed!")
                
                # Display results
                if segments:
                    print("\n" + "="*50)
                    print("TRANSCRIPTION WITH SPEAKER IDENTIFICATION")
                    print("="*50)
                    
                    for segment in segments:
                        start_time = segment.get('start_time', 0)
                        end_time = segment.get('end_time', 0)
                        speaker = segment['speaker']
                        text = segment['text']
                        
                        if with_timestamps:
                            print(f"[{start_time:0>7.2f}s - {end_time:0>7.2f}s] 👤 {speaker}: {text}")
                        else:
                            print(f"👤 {speaker}: {text}")
                    
                    print("="*50)
                    
                    # Save to file if specified
                    if output_file:
                        self.save_transcription_with_speakers(segments, output_file, with_timestamps)
                    else:
                        # Generate default output filename
                        audio_path = os.path.splitext(audio_file)[0]
                        default_output = f"{audio_path}_transcription_with_speakers.txt"
                        self.save_transcription_with_speakers(segments, default_output, with_timestamps)
                
                return segments
                
            finally:
                os.unlink(temp_audio_file)

        except Exception as e:
            error_msg = str(e)
            print(f"Error processing audio: {error_msg}")
            
            # Provide specific guidance for common errors
            if "max_new_tokens" in error_msg and "max_target_positions" in error_msg:
                print("\n💡 This error is related to token limits. The script has been updated to handle this.")
                print("   If you continue to see this error, try using a smaller model (e.g., -m tiny or -m base)")
            elif "chunk_length_s" in error_msg:
                print("\n💡 This error is related to audio chunking. The script has been updated to handle this.")
            
            return None

    def save_transcription_with_speakers(self, segments, output_file, with_timestamps=True):
        """
        Save transcription with speaker information to file.
        
        Args:
            segments (list): List of transcription segments
            output_file (str): Output file path
            with_timestamps (bool): Include timestamps in output
        """
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("TRANSCRIPTION WITH SPEAKER IDENTIFICATION\n")
                f.write("="*50 + "\n\n")
                
                for segment in segments:
                    start_time = segment.get('start_time', 0)
                    end_time = segment.get('end_time', 0)
                    speaker = segment['speaker']
                    text = segment['text']
                    
                    if with_timestamps:
                        f.write(f"[{start_time:0>7.2f}s - {end_time:0>7.2f}s] {speaker}: {text}\n")
                    else:
                        f.write(f"{speaker}: {text}\n")
                
                f.write(f"\n\nTotal speakers identified: {len(self.speaker_voiceprints)}\n")
                f.write(f"Speakers: {', '.join(self.speaker_voiceprints.keys())}\n")
            
            print(f"\nTranscription with speaker identification saved to: {output_file}")
            
        except Exception as e:
            print(f"Error saving transcription: {e}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files with speaker diarization")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("-m", "--model", choices=["tiny", "base", "small", "medium", "large"], 
                       default="base", help="Whisper model size (default: base)")
    parser.add_argument("-l", "--language", default="en", 
                       help="Language code (e.g., en, es, fr) - leave empty for auto-detection")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--no-timestamps", action="store_true", 
                       help="Don't include timestamps in output file")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Convert model name to full path
    model_mapping = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base", 
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v3"
    }
    
    model_name = model_mapping.get(args.model, "openai/whisper-base")
    
    # Convert language to None if empty string for auto-detection
    language = args.language if args.language else None
    
    # Initialize transcriber
    transcriber = AudioTranscriberWithDiarization(
        model_name=model_name,
        debug=args.debug
    )
    
    # Transcribe with diarization
    segments = transcriber.transcribe_with_diarization(
        audio_file=args.audio_file,
        language=language,
        output_file=args.output,
        with_timestamps=not args.no_timestamps
    )
    
    if segments is None:
        sys.exit(1)


if __name__ == "__main__":
    main() 