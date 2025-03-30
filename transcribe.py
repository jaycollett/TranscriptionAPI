import os
import logging
import warnings
import torch # type: ignore
import math
import numpy as np # type: ignore # type: ignore
import json
import re
import string
from faster_whisper import WhisperModel # type: ignore
from pydub import AudioSegment # type: ignore
import time
from functools import lru_cache

# Set audio file location
upload_folder = os.getenv("UPLOAD_FOLDER", "/tmp/audio_files")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('transcribe')

# Suppress Whisper's FP16 warning
warnings.filterwarnings("ignore", category=UserWarning, module="faster_whisper")

# Configuration from environment variables
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float32"

logger.info(f"Running Faster-Whisper on {device.upper()} with compute type {compute_type}")

# Sentinel variable to ensure the model loads only once
_whisper_model = None

def load_whisper_model():
    """Load the Faster-Whisper model once and reuse it."""
    global _whisper_model
    if _whisper_model is None:
        start_time = time.time()
        logger.info("Loading Faster-Whisper model...")
        # Get the model name from environment variable or default to "large-v3-turbo"
        model_name = os.environ.get("MODEL", "large-v3-turbo")
        _whisper_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=os.cpu_count(),  # Use all available CPU cores
            num_workers=2  # Number of workers for the loader
        )
        elapsed = time.time() - start_time
        logger.info(f"Faster-Whisper model '{model_name}' loaded successfully in {elapsed:.2f} seconds")
    return _whisper_model

def clean_transcript_duplicates(text):
    """
    Clean up a transcript by removing accidental adjacent duplicate words,
    both within individual sentences and across sentence boundaries.

    This function splits the text into sentences (based on punctuation),
    then uses a regex to remove consecutive duplicate words within each sentence.
    Finally, it checks the boundary between sentences: if the last word of the previous
    sentence is identical to the first word of the next sentence—and the previous sentence
    ended with a period ('.')—the duplicate at the start of the next sentence is removed,
    including any following punctuation.
    
    Args:
        text (str): The input transcript text.
    
    Returns:
        str: The cleaned transcript.
    """
    # Step 1: Split the text into sentences.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text

    # Step 2: Clean duplicates within each sentence.
    cleaned_sentences = []
    for sentence in sentences:
        # Remove adjacent duplicate words within the sentence.
        sentence_cleaned = re.sub(r'\b(\w+)(\s+)\1\b', r'\1', sentence, flags=re.IGNORECASE)
        cleaned_sentences.append(sentence_cleaned)

    # Step 3: Clean duplicates at sentence boundaries.
    final_sentences = [cleaned_sentences[0]]
    for i in range(1, len(cleaned_sentences)):
        prev_sentence = final_sentences[-1].strip()
        current_sentence = cleaned_sentences[i].strip()
        if not prev_sentence or not current_sentence:
            final_sentences.append(current_sentence)
            continue

        # Extract the last word of the previous sentence and the first word of the current sentence.
        prev_words = re.findall(r'\w+', prev_sentence)
        current_words = re.findall(r'\w+', current_sentence)
        if prev_words and current_words:
            last_word_prev = prev_words[-1].strip(string.punctuation).lower()
            first_word_curr = current_words[0].strip(string.punctuation).lower()
            # If they're identical and the previous sentence ends with a period, remove the duplicate
            if last_word_prev == first_word_curr and prev_sentence[-1] == '.':
                # This regex matches the first word, any punctuation following it, and any extra whitespace.
                current_sentence = re.sub(r'^\b\w+\b[^\w\s]*\s*', '', current_sentence)
        final_sentences.append(current_sentence)

    return " ".join(final_sentences)


@lru_cache(maxsize=1)
def get_audio_duration(file_path):
    """Return the duration of the audio file in seconds."""
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
        psf = 15.1  # Processing speed factor
        estimated_processing_time = math.ceil(duration_sec / psf) * 5 + 45
        logger.info(f"Audio duration: {duration_sec:.2f} seconds. Estimated processing time: ~{estimated_processing_time/60:.2f} min")
    except Exception as e:
        logger.warning(f"Could not calculate estimated processing time: {e}")
        duration_sec = 0

    model = load_whisper_model()

    # Define transcription passes with different parameters.
    passes = [
        {"temperature": 0.4, "patience": 3.5, "beam_size": 10},
        {"temperature": 0.2, "patience": 3.4, "beam_size": 5},
        {"temperature": 0.0, "patience": 3.4, "beam_size": 7},
        {"temperature": 0.0, "patience": 3.0, "beam_size": 5},
        {"temperature": 0.2, "patience": 3.5, "beam_size": 26}
    ]

    transcriptions = []
    confidence_scores = []
    all_segments = []   # List to hold segments from each pass
    word_counts = []    # List to store word count for each pass
    adjusted_confidences = []  # List to store adjusted confidence values

    def calculate_weighted_confidence(segments):
        """
        Calculate a weighted average confidence score for a list of transcription segments.
        Each word's probability is weighted by its duration.
        """
        total_duration = 0.0
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

        return weighted_sum / total_duration if total_duration > 0 else 0.0

    def run_transcription_pass(params, pass_index):
        """
        Run a single transcription pass using the given parameters.
        Returns a dict containing the segments, transcript, and confidence.
        """
        pass_start_time = time.time()
        logger.info(f"Starting pass {pass_index+1} with patience={params['patience']}, temperature={params['temperature']}, and beam_size={params['beam_size']}")
        try:
            segments = list(model.transcribe(
                file_path,
                language="en",
                vad_filter=False,
                beam_size=params["beam_size"],
                temperature=params["temperature"],
                word_timestamps="all",
                suppress_tokens=[-1],
                initial_prompt=None,
                condition_on_previous_text=True,
                patience=params["patience"]
            )[0])
            transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
            total_duration = get_audio_duration(file_path)
            words = len(transcript.split())
            wps = words / total_duration if total_duration > 0 else 0
            logger.info(f"Pass {pass_index+1} stats: {words} words in {total_duration:.1f}s audio ({wps:.2f} words/sec)")
            if total_duration > 5.0 and words < int(total_duration / 4):
                expected_words = int(total_duration / 4)
                logger.warning(f"Suspiciously short transcript detected ({words} words for {total_duration:.1f}s audio, expected at least {expected_words}).")
        except Exception as e:
            logger.error(f"Error during transcription in pass {pass_index+1}: {str(e)}")
            raise

        avg_confidence = calculate_weighted_confidence(segments)
        pass_time = time.time() - pass_start_time
        logger.info(f"Pass {pass_index+1} completed in {pass_time:.2f}s with weighted confidence score: {avg_confidence:.4f}")
        return {
            "segments": segments,
            "transcript": transcript,
            "confidence": avg_confidence
        }

    # Run each transcription pass sequentially.
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
        return {"transcription": "", "timings": []}

    # Select the best pass based on the adjusted confidence scores.
    best_index = int(np.argmax(adjusted_confidences))
    best_segments = all_segments[best_index]
    final_transcript = " ".join(segment.text.strip() for segment in best_segments if segment.text.strip())
    final_transcript = clean_transcript_duplicates(final_transcript)

    # Extract segment timings for storage.
    final_timings = [{"start": seg.start, "end": seg.end, "text": seg.text.strip()} for seg in best_segments if seg.text.strip()]

    # Save the transcript file (a single line) for MFA.
    transcript_output_path = os.path.join(upload_folder, f"{guid}.txt")
    with open(transcript_output_path, "w") as transcript_file:
        transcript_file.write(final_transcript)

    total_time = time.time() - start_time
    logger.info(f"Transcription completed for GUID: {guid} in {total_time:.2f} seconds")

    return {"transcription": final_transcript, "timings": final_timings}

if __name__ == "__main__":
    # For testing purposes
    test_file = os.path.join(upload_folder, "test_audio.mp3")
    test_guid = "00000000-0000-0000-0000-000000000000"
    result = transcribe_audio(test_file, test_guid)
    print(json.dumps(result, indent=2))
