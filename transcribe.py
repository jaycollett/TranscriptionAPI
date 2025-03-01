import os
import difflib
import whisper
import logging
import warnings
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress Whisper's FP16 warning
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

# Get environment variables for Whisper model & language
MODEL = os.environ.get("MODEL", "base")  # Default to "base" model to save memory
LANGUAGE = os.environ.get("LANGUAGE", None)  # Auto-detect language if not set

# Singleton Model Storage
_whisper_model = None

def load_whisper_model():
    """Ensures the Whisper model is loaded only once."""
    global _whisper_model
    if _whisper_model is None:
        logging.info("🔄 Loading Whisper model...")
        _whisper_model = whisper.load_model(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

        # Reduce memory usage on GPU
        if torch.cuda.is_available():
            _whisper_model.half()

        logging.info("✅ Whisper model loaded successfully.")
    return _whisper_model

# Load the model once at startup
model = load_whisper_model()

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

def consensus_two(seq1, seq2):
    """ Computes consensus between two aligned sequences. """
    aligned1, aligned2 = align_two(seq1, seq2)
    consensus = []

    for idx, (t1, t2) in enumerate(zip(aligned1, aligned2)):
        if t1 is None:
            consensus.append(t2)
        elif t2 is None:
            consensus.append(t1)
        elif t1 == t2:
            consensus.append(t1)
        else:
            logging.warning(f"Mismatch at column {idx}: '{t1}' vs '{t2}'; choosing '{t1}'")
            consensus.append(t1)

    return consensus

def consensus_three(seq1, seq2, seq3):
    """ Merges three transcriptions using sequential consensus voting. """
    consensus_result = consensus_two(consensus_two(seq1, seq2), seq3)
    
    # Prevent repetition artifacts at the end of the transcription
    if len(consensus_result) > 10:
        last_ten_words = consensus_result[-10:]
        if len(set(last_ten_words)) < 3:  # If last 10 words have low diversity, trim excess
            consensus_result = consensus_result[:-5]
            logging.warning("🚨 Detected excessive repetition at the end of transcription. Trimming output.")
    
    return consensus_result

def transcribe_audio(file_path):
    """ Performs triple-pass Whisper transcription with consensus alignment. """
    logging.info(f"Starting transcription for: {file_path}")
    
    transcripts = []
    
    # Perform three transcription attempts
    for attempt in range(3):
        logging.info(f"Transcription attempt {attempt + 1}...")
        result = model.transcribe(file_path, language=LANGUAGE, beam_size=7, best_of=7) if LANGUAGE else model.transcribe(file_path)
        transcript_text = result["text"].strip()
        logging.info(f"Transcription attempt {attempt + 1} completed: {transcript_text[:100]}...")  # Log partial transcription
        transcripts.append(transcript_text)
    
    # Tokenize transcripts into word sequences
    token_lists = [t.split() for t in transcripts]
    
    # Compute consensus transcription
    logging.info("Computing final consensus transcription...")
    consensus_tokens = consensus_three(token_lists[0], token_lists[1], token_lists[2])
    final_transcript = " ".join(consensus_tokens)

    logging.info(f"Transcription completed for {file_path}: {final_transcript[:200]}...")
    return final_transcript
