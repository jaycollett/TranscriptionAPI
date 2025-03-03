import os
import difflib
import whisper
import logging
import warnings
import torch
import sqlite3
from collections import Counter
from fuzzywuzzy import process  # Requires `pip install fuzzywuzzy`

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
MODEL = os.environ.get("MODEL", "turbo")  # Highest quality model
LANGUAGE = os.environ.get("LANGUAGE", None)  # Auto-detect language if not set

def align_two(seq1, seq2):
    """ Aligns two token sequences and returns equal-length lists, padding with None where necessary. """
    sm = difflib.SequenceMatcher(None, seq1, seq2)
    aligned1, aligned2 = [], []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            aligned1.extend(seq1[i1:i2])
            aligned2.extend(seq2[j1:j2])
        elif tag == "replace":
            tokens1, tokens2 = seq1[i1:i2], seq2[j1:j2]
            L = max(len(tokens1), len(tokens2))
            tokens1 += [None] * (L - len(tokens1))
            tokens2 += [None] * (L - len(tokens2))
            aligned1.extend(tokens1)
            aligned2.extend(tokens2)
        elif tag == "delete":
            aligned1.extend(seq1[i1:i2])
            aligned2.extend([None] * (i2 - i1))
        elif tag == "insert":
            aligned1.extend([None] * (j2 - j1))
            aligned2.extend(seq2[j1:j2])

    return aligned1, aligned2

def get_best_match(word, confidence_list):
    """Finds the closest word match in the confidence list using fuzzy matching."""
    if not confidence_list:
        return 0.5  # Default confidence when no data exists

    words, probs = zip(*confidence_list)  # Unpack into separate lists
    match = process.extractOne(word, words, score_cutoff=80)  # Fuzzy match

    if match:
        matched_word = match[0]
        return probs[words.index(matched_word)]
    
    return 0.5  # Default confidence if no match is found


def consensus_with_probabilities(token_lists, word_confidences):
    """ Computes consensus transcription using word probabilities. """
    aligned1, aligned2 = align_two(token_lists[0], token_lists[1])
    confidence1, confidence2 = word_confidences[0], word_confidences[1]
    consensus = []

    for idx, (t1, t2) in enumerate(zip(aligned1, aligned2)):
        if t1 is None:
            consensus.append(t2)
        elif t2 is None:
            consensus.append(t1)
        elif t1 == t2:
            consensus.append(t1)
        else:
            # Get probabilities with better matching
            prob1 = get_best_match(t1, confidence1)
            prob2 = get_best_match(t2, confidence2)

            # Choose the word with the higher probability
            chosen_word = t1 if prob1 > prob2 else t2
            logging.warning(f"Mismatch at column {idx}: '{t1}' ({prob1:.2f if prob1 else '??'}) vs '{t2}' ({prob2:.2f if prob2 else '??'}); choosing '{chosen_word}'")
            consensus.append(chosen_word)

    return consensus

def transcribe_audio(file_path, guid):
    """ Performs dual-pass Whisper transcription with consensus alignment, and updates the database."""
    logging.info(f"Starting transcription for: {file_path}")

    model = load_whisper_model()
    transcripts = []
    word_confidences = []

    for attempt in range(2):
        logging.info(f"Transcription attempt {attempt + 1}...")
        result = model.transcribe(file_path, language=LANGUAGE, beam_size=7, best_of=7, task="transcribe")

        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.float()

        transcript_text = result["text"].strip()
        logging.info(f"Transcription attempt {attempt + 1} completed: {transcript_text[:100]}...")
        transcripts.append(transcript_text)

        # Extract word-level probabilities safely
        word_probs = [(word.get("text", ""), word.get("probability", 0.5)) for segment in result.get("segments", []) for word in segment.get("words", [])]
        word_confidences.append(word_probs)

    token_lists = [t.split() for t in transcripts]
    logging.info("Computing final consensus transcription...")
    consensus_tokens = consensus_with_probabilities(token_lists, word_confidences)
    final_transcript = " ".join(consensus_tokens)

    # Save to database
    with sqlite3.connect("transcriptions.db") as conn:
        cursor = conn.cursor()
        logging.info(f"📝 Saving transcription to database: {final_transcript[:200]}")
        cursor.execute("UPDATE transcriptions SET transcription = ?, status = 'processed', completed_at = CURRENT_TIMESTAMP WHERE guid = ?", (final_transcript, guid))
        conn.commit()

    logging.info(f"✅ Transcription completed and saved for GUID: {guid}")
    return final_transcript
