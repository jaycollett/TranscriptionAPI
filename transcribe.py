import os
import logging
import warnings
import torch
import sqlite3
import math
import numpy as np
import json  # NEW: Ensure json is imported for serialization
from faster_whisper import WhisperModel
from pydub import AudioSegment

# Set audiofile location
upload_folder = os.getenv("UPLOAD_FOLDER", "/tmp/audio_files")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress Whisper's FP16 warning
warnings.filterwarnings("ignore", category=UserWarning, module="faster_whisper")

# Force GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"✅ Running Faster-Whisper on {device.upper()}.")

# Sentinel variable to ensure model loads only once
_whisper_model = None

def load_whisper_model():
    """Ensures Faster-Whisper model is loaded only once."""
    global _whisper_model
    if _whisper_model is None:
        logging.info("🔄 Loading Faster-Whisper model...")

        # Get the model name from environment variable or default to "large-v3-turbo"
        model_name = os.environ.get("MODEL", "large-v3-turbo")

        # Select the device (GPU if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model with optimized settings
        _whisper_model = WhisperModel(
            model_name,
            device=device,
            compute_type="float32" 
        )

        logging.info("✅ Faster-Whisper model loaded successfully.")

    return _whisper_model

def transcribe_audio(file_path, guid):
    """Performs multi-pass Faster-Whisper transcription with varied beam_size and temperature settings."""
    logging.info(f"🎙️ Starting transcription for: {file_path}")

    # Calculate and log estimated processing time based on audio duration
    try:
        audio = AudioSegment.from_file(file_path)
        duration_sec = len(audio) / 1000.0  # Convert milliseconds to seconds
        psf = 15.1  # Processing speed factor
        estimated_processing_time = math.ceil(duration_sec / psf) * 3
        logging.info(
            f"⏱️ Estimated processing time: {estimated_processing_time} seconds (~{estimated_processing_time/60:.2f} min)"
        )
    except Exception as e:
        logging.warning(f"⚠️ Could not calculate estimated processing time: {e}")

    # Load Whisper model
    model = load_whisper_model()

    # Perform multiple transcriptions with varied settings
    passes = [
        {"temperature": 0.0, "beam_size": 7},
        {"temperature": 0.2, "beam_size": 5},
        {"temperature": 0.0, "beam_size": 5}
    ]

    transcriptions = []
    confidence_scores = []
    all_segments = []  # Store segments for each pass

    for i, params in enumerate(passes):
        logging.info(f"🔄 Pass {i+1} with temperature={params['temperature']} and beam_size={params['beam_size']}")
        segments = list(model.transcribe(
            file_path,
            language="en",
            vad_filter=False,
            beam_size=params["beam_size"],
            temperature=params["temperature"],
            word_timestamps="all"
        )[0])  

        # Save segments for this pass
        all_segments.append(segments)

        # Clean and join text from segments
        transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())

        # Calculate average confidence
        log_probs = [
            word.probability
            for segment in segments if hasattr(segment, "words") and segment.words
            for word in segment.words if hasattr(word, "probability")
        ]
        avg_confidence = np.mean(log_probs) if log_probs else 0.0
        transcriptions.append(transcript)
        confidence_scores.append(avg_confidence)

        logging.info(f"📈 Pass {i+1} confidence score: {avg_confidence:.4f}")

    if not confidence_scores:
        logging.error(f"❌ Whisper transcription failed for {guid}: No valid transcription found")
        return {"transcription": "", "timings": []}  # Return empty transcription instead of crashing

    # Determine the best transcription based on confidence scores
    best_index = np.argmax(confidence_scores)
    best_segments = all_segments[best_index]  # ✅ Fix: Ensure `best_segments` is assigned before use

    # Create final transcript
    final_transcript = " ".join(segment.text.strip() for segment in best_segments if segment.text.strip())

    # Extract segment timings in structured format
    best_timings = [{"start": seg.start, "end": seg.end, "text": seg.text.strip()} for seg in best_segments if seg.text.strip()]

    logging.info(f"✅ Transcription completed for GUID: {guid}")

    # Save the transcription result to a file
    transcript_output_path = os.path.join(upload_folder , f"{guid}.txt")
    with open(transcript_output_path, "w") as transcript_file:
        transcript_file.write(final_transcript)

    # Return transcription and segment timings
    return {"transcription": final_transcript, "timings": best_timings}

