import os
import logging
import warnings
import threading
import torch # type: ignore
import math
import numpy as np  # type: ignore
import json
import re
from faster_whisper import WhisperModel # type: ignore
from pydub import AudioSegment # type: ignore
from pydub.utils import mediainfo # type: ignore
import time

# Set audio file location from environment variable or default to /tmp/audio_files
upload_folder = os.getenv("UPLOAD_FOLDER", "/tmp/audio_files")

# Configure logging to include timestamp, log level, logger name, and message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('transcribe')

# Suppress specific warning from Faster-Whisper about FP16 usage
warnings.filterwarnings("ignore", category=UserWarning, module="faster_whisper")

# Determine device type (GPU if available, else CPU) and set compute precision
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"

logger.info(f"Running Faster-Whisper on {device.upper()} with compute type {compute_type}")

# Sentinel variable to ensure the model is loaded only once and reused
_whisper_model = None
_model_lock = threading.Lock()

# Processing speed factor: seconds of audio per unit of the estimate formula.
# Calibrated to the five-pass decode; see docs/PUNCHLIST.md (pipeline item 16).
PROCESSING_SPEED_FACTOR = 15.1


def whisper_model_loaded():
    """True once load_whisper_model() has completed; used by the /health endpoint."""
    return _whisper_model is not None


def estimate_processing_seconds(duration_sec):
    """Estimated wall-clock seconds to transcribe duration_sec of audio.

    One formula shared by /upload (the ETA the client is given) and the
    transcription log line, so the two can never drift apart.
    """
    return math.ceil(float(duration_sec) / PROCESSING_SPEED_FACTOR) * 5 + 45

def load_whisper_model():
    """Load the Faster-Whisper model once and reuse it."""
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            start_time = time.time()
            logger.info("Loading Faster-Whisper model...")
            model_name = os.environ.get("MODEL", "large-v3-turbo")
            # Try to load from local cache first
            local_model_path = "/app/models/whisper"
            if os.path.exists(local_model_path):
                logger.info(f"Loading Whisper model from local cache: {local_model_path}")
                _whisper_model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=os.cpu_count(),
                    num_workers=2,
                    download_root=local_model_path
                )
            else:
                logger.info("Local Whisper model cache not found, downloading from internet")
                _whisper_model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=os.cpu_count(),
                    num_workers=2
                )
            elapsed = time.time() - start_time
            logger.info(f"Faster-Whisper model '{model_name}' loaded successfully in {elapsed:.2f} seconds")
    return _whisper_model

def normalize_timestamp(ts):
    """Convert timestamp to a consistent format (float)"""
    if isinstance(ts, (tuple, list)):
        return float(ts[0])
    return float(ts)

def clean_boundary_duplicates(text):
    """
    Remove duplicated phrases that might occur at segment boundaries.
    Finds word sequences (2+ words) that repeat with optional spacing/punctuation between.
    """
    # Find word sequences that repeat (with at least 2 words)
    pattern = r'\b(\w+\s+\w+(?:\s+\w+){0,3})[.,;!?\s]*\1\b'
    
    while True:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            break
        
        # Replace the second occurrence with empty string
        start, end = match.span()
        phrase = match.group(1)
        phrase_len = len(phrase)
        duplicate_pos = text[start:end].lower().find(phrase.lower(), phrase_len)
        if duplicate_pos > 0:
            duplicate_pos += start
            text = text[:duplicate_pos] + text[duplicate_pos + phrase_len:]
    
    return text


# NOTE (2026-09-07): preprocess_audio_for_transcription() was removed.
# It normalised to -20 dBFS, then measured "noise" as mean(abs(samples))/32768 on the
# NORMALISED signal. That is a crest-factor measurement of the speech itself, not a
# noise floor, so every file in the archive scored 0.046-0.060 against the 0.015
# threshold and the "only if noise is detected" branch was never skipped. It then ran
# noisereduce with no noise profile over material that is ~78% speech, so the profile
# it estimated was largely speech: measured effect was speech-active frames falling
# from 76% to 65% and dynamic range widening by 8 dB, plus a second lossy MP3 encode.
# faster-whisper's own feature extractor normalises the input, so feeding it the
# original file is both simpler and measurably better: the same model/config on the
# untouched 755 s file produces 2.77 words/sec against 0.15-1.24 through this stage.
# Deleted rather than repaired: a correct noise-floor estimate would need non-speech
# frame detection, and doing nothing is already the better answer.


def temperature_ladder(base, step=0.2):
    """Return the temperature fallback ladder faster-whisper expects.

    faster-whisper only performs temperature fallback when `temperature` is a
    sequence; a scalar is wrapped into a single-element list, which disables the
    fallback entirely. Fallback is the built-in escape from a decode that trips the
    compression-ratio or log-probability thresholds, i.e. exactly the mid-file
    collapse this service was producing. Above 0.5 faster-whisper also resets the
    previous-text prompt, which breaks any repetition loop already under way.

    The ladder runs from `base` up to 1.0 inclusive; `base` is clamped to [0.0, 1.0].
    """
    base = min(max(float(base), 0.0), 1.0)
    steps = []
    t = base
    while t < 1.0 + 1e-9:
        steps.append(round(t, 2))
        t += step
    if not steps or steps[-1] < 1.0:
        steps.append(1.0)
    return tuple(steps)


def get_audio_duration(file_path):
    """Return the duration of the audio file in seconds without modifying it.

    Reads the container header through ffprobe (pydub's mediainfo), which is
    instant and allocates nothing; decoding a 46 minute stereo MP3 to PCM just to
    measure it costs roughly 490 MB. The full decode is kept only as a fallback
    for files whose header carries no duration, and it raises on undecodable
    input so callers can reject the file.
    """
    try:
        info = mediainfo(file_path) or {}
        duration = info.get("duration")
        if duration not in (None, "", "N/A"):
            return float(duration)
        logger.warning(f"ffprobe reported no duration for {file_path}; decoding to measure it")
    except Exception as e:
        logger.warning(f"ffprobe duration lookup failed for {file_path}: {e}; decoding to measure it")
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # Convert milliseconds to seconds

def transcribe_audio(file_path, guid):
    """
    Performs multi-pass Faster-Whisper transcription, penalizing passes with a low word rate.
    Returns the best transcript (with timings) based on an adjusted confidence score.
    """
    start_time = time.time()
    logger.info(f"Starting transcription for: {file_path} (GUID: {guid})")

    try:
        duration_sec = get_audio_duration(file_path)
        estimated_processing_time = estimate_processing_seconds(duration_sec)
        logger.info(f"Audio duration: {duration_sec:.2f} seconds. Estimated processing time: ~{estimated_processing_time/60:.2f} min")
    except Exception as e:
        logger.warning(f"Could not calculate estimated processing time: {e}")
        duration_sec = 0

    # No preprocessing: faster-whisper normalises internally and the previous
    # denoise stage measurably degraded every file (see note above).
    processed_file_path = file_path
    logger.info(f"Using audio file for transcription: {processed_file_path}")

    model = load_whisper_model()  # Load the Whisper model (cached)

    # Define transcription passes with different parameters.
    # "temperature" is the BASE of each pass's fallback ladder, not a scalar.
    passes = [
        {"temperature": 0.2, "patience": 3.0, "beam_size": 5},
        {"temperature": 0.0, "patience": 2.8, "beam_size": 7},
        {"temperature": 0.0, "patience": 2.5, "beam_size": 1},  # Greedy
        {"temperature": 0.3, "patience": 3.2, "beam_size": 10},
        {"temperature": 0.2, "patience": 3.5, "beam_size": 15, "condition_on_previous_text": False}
    ]

    transcriptions = []  # Store raw transcripts from each pass
    confidence_scores = []
    all_segments = []   # List to hold segments from each pass
    word_counts = []    # List to store word count for each pass
    adjusted_confidences = []  # List to store adjusted confidence values

    def calculate_weighted_confidence(segments):
        """
        Calculate a weighted average confidence score for a list of transcription segments.
        Each word's probability is weighted by its duration.
        """
        total_duration = 0.0  # Total duration of all words
        weighted_sum = 0.0

        for segment in segments:
            words = getattr(segment, "words", None)
            if not words:
                continue
            for word in words:
                prob = getattr(word, "probability", None)
                if prob is None or prob <= 0:
                    continue
                word_start = getattr(word, "start", None)
                word_end = getattr(word, "end", None)
                if word_start is not None and word_end is not None:
                    duration = word_end - word_start
                else:
                    seg_start = getattr(segment, "start", 0)
                    seg_end = getattr(segment, "end", 0)
                    seg_duration = seg_end - seg_start
                    duration = seg_duration / len(words) if len(words) > 0 else 0
                weighted_sum += duration * prob
                total_duration += duration

        return weighted_sum / total_duration if total_duration > 0 else 0.0  # Avoid division by zero


    def run_transcription_pass(params, pass_index):
        """
        Run a single transcription pass using the given parameters.
        Returns a dict containing the segments, transcript, and confidence.
        """
        pass_start_time = time.time()  # Track how long this pass takes
        logger.info(f"Starting pass {pass_index+1} with patience={params['patience']}, temperature={params['temperature']}, and beam_size={params['beam_size']}")
        try:
            segments = list(model.transcribe(
                processed_file_path,
                language="en",
                vad_filter=True,
                vad_parameters={"threshold": 0.35, "min_speech_duration_ms": 250, "min_silence_duration_ms": 300},
                beam_size=params["beam_size"],
                temperature=temperature_ladder(params["temperature"]),
                word_timestamps="all",
                suppress_tokens=[-1],
                # No initial_prompt. The previous one asserted "a single speaker",
                # which is false for these multi-voice class and Q&A recordings, and
                # a prompt is prepended as previous-text context, so it can prime the
                # repetition loops it was meant to prevent. The configuration measured
                # at 2.77 words/sec on this material used no prompt at all.
                condition_on_previous_text=params.get("condition_on_previous_text", True),
                patience=params["patience"]
            )[0])
            transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
            words = len(transcript.split())
            wps = words / duration_sec if duration_sec > 0 else 0
            total_duration = duration_sec  # Define total_duration for logging and validation
            logger.info(f"Pass {pass_index+1} stats: {words} words in {total_duration:.1f}s audio ({wps:.2f} words/sec)")
            if total_duration > 5.0 and words < int(total_duration / 4):
                expected_words = int(total_duration / 4)
                logger.warning(f"Suspiciously short transcript detected ({words} words for {total_duration:.1f}s audio, expected at least {expected_words}).")
        except Exception as e:
            logger.error(f"Error during transcription in pass {pass_index+1}: {str(e)}")
            raise

        avg_confidence = calculate_weighted_confidence(segments)  # Compute confidence score
        pass_time = time.time() - pass_start_time
        logger.info(f"Pass {pass_index+1} completed in {pass_time:.2f}s with weighted confidence score: {avg_confidence:.4f}")
        
        # Normalize timestamps to ensure consistent format (float)
        normalized_segments = []
        for seg in segments:
            if seg.text.strip():  # Only process non-empty segments
                normalized_segments.append({
                    "start": normalize_timestamp(seg.start),
                    "end": normalize_timestamp(seg.end),
                    "text": seg.text.strip(),
                    "confidence": calculate_weighted_confidence([seg]),
                    "original_segment": seg  # Keep original segment for later processing
                })
        
        return {
            "segments": segments,
            "transcript": transcript,
            "confidence": avg_confidence,
            "segment_confidences": normalized_segments
        }

    # Run each transcription pass sequentially.
    transcriptions = []
    all_segments = []
    confidence_scores = []
    word_counts = []
    adjusted_confidences = []

    for i, params in enumerate(passes):
        result = run_transcription_pass(params, i)
        transcript = result["transcript"]
        segments = result["segments"]
        conf = result["confidence"]

        transcriptions.append(transcript)
        all_segments.append(segments)
        confidence_scores.append(conf)

        wc = len(transcript.split())
        word_counts.append(wc)
        # Compute words per second for this pass
        wps = wc / duration_sec if duration_sec > 0 else 0

        # If the pass has less than 1.4 words per second, penalize its confidence.
        if wps < 1.4:
            adjusted = conf * (wps / 1.4)
        else:
            adjusted = conf
        adjusted_confidences.append(adjusted)
        logger.info(f"Pass {i+1} adjusted confidence: {adjusted:.4f} (word rate: {wps:.2f} words/sec, word count: {wc})")

    if not adjusted_confidences:
        logger.error(f"Whisper transcription failed for {guid}: No valid transcription found")
        return {"transcription": "", "timings": [], "duration_sec": duration_sec}

    # Select the best pass based on the adjusted confidence scores.
    best_index = int(np.argmax(adjusted_confidences))  # Choose the best scoring pass
    best_result = {
        "segments": all_segments[best_index],
        "transcript": transcriptions[best_index],
        "confidence": confidence_scores[best_index],
        "segment_confidences": [
            {
                "start": normalize_timestamp(seg.start),
                "end": normalize_timestamp(seg.end),
                "text": seg.text.strip(),
                "confidence": calculate_weighted_confidence([seg]),
                "original_segment": seg
            }
            for seg in all_segments[best_index] if seg.text.strip()
        ]
    }

    # Use the best pass segments directly without reprocessing low-confidence segments.
    best_segments = best_result["segments"].copy()  # Create a copy to avoid modifying the original

    final_transcript = " ".join(segment.text.strip() for segment in best_segments if segment.text.strip())
    final_transcript = clean_boundary_duplicates(final_transcript)

    # Extract segment timings for storage using normalized timestamps
    final_timings = []
    for seg in best_segments:
        if seg.text.strip():
            final_timings.append({
                "start": normalize_timestamp(seg.start),
                "end": normalize_timestamp(seg.end),
                "text": seg.text.strip()
            })

    # The MFA transcript file is written by app.run_forced_alignment from the
    # returned timings; nothing here needs to touch the upload folder.
    total_time = time.time() - start_time
    logger.info(f"Transcription completed for GUID: {guid} in {total_time:.2f} seconds")

    return {"transcription": final_transcript, "timings": final_timings, "duration_sec": duration_sec}

if __name__ == "__main__":
    # For testing purposes
    test_file = os.path.join(upload_folder, "test_audio.mp3")
    test_guid = "00000000-0000-0000-0000-000000000000"
    result = transcribe_audio(test_file, test_guid)  # Run test transcription
    print(json.dumps(result, indent=2))
