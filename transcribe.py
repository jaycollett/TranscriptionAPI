import os
import whisper
import logging
import warnings
import torch
import sqlite3
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress Whisper's FP16 warning
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

# Force GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"✅ Running Whisper on {device.upper()}.")

# Sentinel variable to ensure model loads only once
_whisper_model = None

def load_whisper_model():
    """Ensures the Whisper model is loaded only once using a sentinel."""
    global _whisper_model
    if _whisper_model is None:
        logging.info(f"🔄 Loading Whisper model on {device}...")
        _whisper_model = whisper.load_model(MODEL, device=device)
        logging.info("✅ Whisper model loaded successfully.")
    return _whisper_model

# Get environment variables for Whisper model & language
MODEL = os.environ.get("MODEL", "large-v3-turbo")  # Use large-v3-turbo for best accuracy
LANGUAGE = os.environ.get("LANGUAGE", "en")  # Default to English

def transcribe_audio(file_path, guid):
    """Performs multi-pass Whisper transcription and selects the best output without storing confidence scores."""
    logging.info(f"🎙️ Starting transcription for: {file_path}")
    model = load_whisper_model()

    # Perform multiple transcriptions with varied settings
    passes = [
        {"temperature": 0.0},   # Standard transcription (deterministic)
        {"temperature": 0.3}   # Slightly varied transcription
      ]

    transcriptions = []
    confidence_scores = []

    for i, params in enumerate(passes):
        logging.info(f"🔄 Pass {i+1} with temperature={params['temperature']}")
        result = model.transcribe(file_path, language=LANGUAGE, temperature=params["temperature"], word_timestamps=True)
        transcript = result["text"].strip()

        # Extract word-level confidence scores correctly
        log_probs = [
            word["probability"]
            for segment in result["segments"]
            if "words" in segment
            for word in segment["words"]
            if "probability" in word
        ]

        avg_confidence = np.mean(log_probs) if log_probs else 0.0
        transcriptions.append(transcript)
        confidence_scores.append(avg_confidence)

        logging.info(f"Pass {i+1} transcription: {transcript[:200]}")
        logging.info(f"Pass {i+1} confidence score: {avg_confidence:.4f}")

    # Select the best transcription based on highest confidence
    best_index = np.argmax(confidence_scores)
    final_transcript = transcriptions[best_index]

    # Save to database (NO CONFIDENCE SCORE STORED)
    with sqlite3.connect("transcriptions.db") as conn:
        cursor = conn.cursor()
        logging.info(f"📝 Saving transcription to database: {final_transcript[:200]}")
        cursor.execute("UPDATE transcriptions SET transcription = ?, status = 'processed', completed_at = CURRENT_TIMESTAMP WHERE guid = ?", 
                       (final_transcript, guid))
        conn.commit()

    logging.info(f"✅ Transcription completed and saved for GUID: {guid}")
    return final_transcript
