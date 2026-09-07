import os
import re
import uuid
import sqlite3
import threading  # For background processing and thread-local DB connections
import time
import json
import difflib
import shutil
import subprocess
from contextlib import closing
from flask import Flask, request, jsonify  # Web framework and request handling
from werkzeug.exceptions import HTTPException
from transcribe import (  # Custom transcription logic
    transcribe_audio,
    load_whisper_model,
    get_audio_duration,
    estimate_processing_seconds,
    whisper_model_loaded,
    whisper_span,
)
from textnorm import norm_words
from datetime import datetime, timedelta, timezone
import logging  # Logging for debugging and monitoring


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER", "/tmp/audio_files")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Upload limits. The cap is generous (a 46 minute stereo MP3 is under 100 MB) and
# exists so a runaway client cannot fill the container layer with one request.
# Oversize bodies get a 413 from Werkzeug before the handler runs.
UPLOAD_MAX_BYTES = 1024 * 1024 * 1024  # 1 GiB
app.config['MAX_CONTENT_LENGTH'] = UPLOAD_MAX_BYTES
ALLOWED_EXTENSIONS = frozenset({'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.mp4'})

# Serialises the check-then-save-then-insert sequence in /upload so two concurrent
# uploads of the same GUID cannot both pass the existence check.
upload_lock = threading.Lock()

# Database configuration

db_file = os.getenv("DB_FILE", "transcriptions.db")
db_connection_timeout = 30  # Connection timeout in seconds

# Maximum number of consecutive failed attempts (garbage results or exceptions)
# before a job is moved to a terminal state instead of being requeued. This keeps a
# single bad sermon from permanently blocking the head of the FIFO queue.
max_garbage_retries = int(os.getenv("MAX_GARBAGE_RETRIES", "3"))
# Words-per-second floor for the post-transcription quality gate. Normal sermon
# material runs 2.5-2.8 words/sec; the 2026-09 decode collapse produced 0.54. The
# alphanumeric-ratio check alone is blind to coherent prose at half length.
MIN_WORDS_PER_SEC = float(os.getenv("MIN_WORDS_PER_SEC", "1.0"))

# Anomaly gate (0.6.0). The decode hands back a count of segments that tripped one of
# faster-whisper's own quality signals and a count of 60 s speech windows under 1.2
# words/sec. On the six reference files a healthy decode scored zero on both, so
# these are set just above the noise floor rather than at a measured failure point;
# docs/QUALITY_PROPOSAL.md section 4 makes revisiting them a first-week task.
ANOMALY_WINDOWS_MAX = int(os.getenv("ANOMALY_WINDOWS_MAX", "2"))
ANOMALY_SEGMENTS_MAX = int(os.getenv("ANOMALY_SEGMENTS_MAX", "5"))
# Two decodes are "the same result" when the word counts match and the mean
# log-probabilities agree to this. The decode is deterministic on most material, so a
# job rejected by the anomaly gate usually re-decodes to the identical transcript and
# retrying it is pure cost with a guaranteed outcome.
ANOMALY_RETRY_EPSILON = float(os.getenv("ANOMALY_RETRY_EPSILON", "1e-6"))

# Where Montreal Forced Aligner keeps its per-corpus working directory. MFA names
# it after the corpus directory basename, so a job's tree is <root>/<guid>_mfa_input.
MFA_ROOT_DIR = os.getenv("MFA_ROOT_DIR", "/mfa")
MFA_DICTIONARY_PATH = "/mfa/pretrained_models/dictionary/english_mfa.dict"
MFA_ACOUSTIC_MODEL = "english_mfa"
# Utterance construction for the per-utterance TextGrid (harness I1). Whisper
# segments are merged into utterances split only at gaps of MFA_UTTERANCE_GAP_SEC or
# more, each padded MFA_UTTERANCE_PAD_SEC into the surrounding silence.
MFA_UTTERANCE_GAP_SEC = float(os.getenv("MFA_UTTERANCE_GAP_SEC", "0.4"))
MFA_UTTERANCE_PAD_SEC = float(os.getenv("MFA_UTTERANCE_PAD_SEC", "0.15"))
MFA_UTTERANCE_MIN_SEC = float(os.getenv("MFA_UTTERANCE_MIN_SEC", "8.0"))
MFA_UTTERANCE_MAX_SEC = float(os.getenv("MFA_UTTERANCE_MAX_SEC", "30.0"))
# One attempt at MFA's default beams. The 40/100 then 100/400 ladder cost 279 s on
# the 1369 s file and still failed; per-utterance alignment did it in 30 s at the
# defaults. The timeout scales with the file and never drops below the floor.
MFA_TIMEOUT_FLOOR_SEC = float(os.getenv("MFA_TIMEOUT_FLOOR_SEC", "120"))
MFA_TIMEOUT_BASE_SEC = float(os.getenv("MFA_TIMEOUT_BASE_SEC", "60"))
MFA_TIMEOUT_PER_SEC = float(os.getenv("MFA_TIMEOUT_PER_SEC", "0.5"))
# A refined edge this far from Whisper's own word timestamp counts as agreement.
ALIGNMENT_AGREE_SEC = 0.25
# A refined span shorter than this share of the Whisper span means the aligner
# covered only part of the segment, so its answer is discarded for that segment.
SPAN_RATIO_FLOOR = 0.5


def _percentile(values, pct):
    """Linear-interpolated percentile; None on an empty list."""
    if not values:
        return None
    data = sorted(values)
    k = (len(data) - 1) * pct / 100.0
    low = int(k)
    high = min(low + 1, len(data) - 1)
    return round(data[low] + (data[high] - data[low]) * (k - low), 4)

# Worker timing. The poll interval only applies when the queue is empty; a worker
# that just finished a job checks for the next one immediately.
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "30"))
CLEANUP_INTERVAL_SEC = 3600
# Rows in a terminal state older than this are deleted along with their files.
CLEANUP_ROW_AGE = '-1 day'
# Files in UPLOAD_FOLDER older than this with no live row are swept. The upload
# folder is a host bind mount, so it outlives the container-layer database.
ORPHAN_FILE_AGE_SEC = 2 * 24 * 3600
# A worker that is not mid-job and has not polled in this long is reported as dead.
WORKER_STALE_AFTER = timedelta(minutes=5)

TERMINAL_STATUSES = ('completed', 'error', 'quarantined')
UUID_PREFIX_RE = re.compile(
    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
)

# Ensure database file exists before initializing
if not os.path.exists(db_file):
    open(db_file, 'w').close()

# Thread-local storage for database connections
# This ensures each thread gets its own dedicated connection
local_storage = threading.local()


def utcnow():
    """Current time in UTC. A function so tests can freeze it."""
    return datetime.now(timezone.utc)


def format_utc(dt):
    """Format a datetime the way every timestamp in the API is formatted."""
    return dt.strftime('%Y-%m-%d %H:%M:%S UTC')


def get_db_connection():
    """
    Returns a SQLite database connection with proper timeout settings.
    Uses thread-local storage to ensure each thread gets its own connection; the
    connection is only ever used from the thread that created it, so SQLite's
    same-thread check stays on.
    """
    if not hasattr(local_storage, 'connection'):
        conn = sqlite3.connect(
            db_file,
            timeout=db_connection_timeout,
            isolation_level=None,  # Use autocommit mode
        )
        # journal_mode=WAL is persistent and set once in init_db(). These three are
        # per-connection, so they have to be applied on every connection to mean
        # anything outside the initialisation thread.
        conn.execute('PRAGMA synchronous = NORMAL')
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute(f'PRAGMA busy_timeout = {db_connection_timeout * 1000}')
        local_storage.connection = conn
        app.logger.debug(f"Created new database connection for thread {threading.current_thread().name}")

    return local_storage.connection

def close_db_connection():
    # Clean up thread-local connection if it exists
    """Close the database connection for the current thread if it exists."""
    if hasattr(local_storage, 'connection'):
        local_storage.connection.close()
        delattr(local_storage, 'connection')
        app.logger.debug(f"Closed database connection for thread {threading.current_thread().name}")

# Initialize SQLite database with timings column included in the CREATE TABLE statement
def column_exists(cursor, table, column):
    # Check whether a column already exists on a table (used for safe migrations)
    """Return True if the given column is present on the table."""
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())

def ensure_schema(cursor):
    # Create tables/indexes if missing and apply additive column migrations
    """Create the transcriptions table, its indexes, and apply additive migrations.

    Safe to call repeatedly: every statement is idempotent (CREATE ... IF NOT EXISTS
    and a guarded ALTER TABLE), so existing databases are migrated in place without
    touching existing rows.
    """
    # Create the main transcriptions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcriptions (
            guid TEXT PRIMARY KEY,
            filename TEXT,
            status TEXT DEFAULT 'pending',
            transcription TEXT DEFAULT NULL,
            timings TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP DEFAULT NULL,
            processing_time_est INTEGER DEFAULT 0,
            attempt_count INTEGER DEFAULT 0,
            processing_seconds REAL DEFAULT NULL,
            words_per_second REAL DEFAULT NULL,
            mfa_applied INTEGER DEFAULT NULL,
            anomaly_count INTEGER DEFAULT NULL,
            anomaly_windows INTEGER DEFAULT NULL,
            flagged_segments TEXT DEFAULT NULL,
            rescue_attempted INTEGER DEFAULT NULL,
            rescue_selected INTEGER DEFAULT NULL,
            speech_seconds REAL DEFAULT NULL,
            last_anomaly_fingerprint TEXT DEFAULT NULL
        )
    ''')

    # Create index for status to optimize queue queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON transcriptions(status)')

    # Create index for created_at to optimize cleanup queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON transcriptions(created_at)')

    # Additive migrations for pre-existing databases. Each is guarded so the
    # function can run against any earlier schema.
    migrations = [
        ('attempt_count', 'INTEGER DEFAULT 0'),
        ('processing_seconds', 'REAL DEFAULT NULL'),
        ('words_per_second', 'REAL DEFAULT NULL'),
        ('mfa_applied', 'INTEGER DEFAULT NULL'),
        # 0.6.0 decode diagnostics.
        ('anomaly_count', 'INTEGER DEFAULT NULL'),
        ('anomaly_windows', 'INTEGER DEFAULT NULL'),
        ('flagged_segments', 'TEXT DEFAULT NULL'),
        ('rescue_attempted', 'INTEGER DEFAULT NULL'),
        ('rescue_selected', 'INTEGER DEFAULT NULL'),
        ('speech_seconds', 'REAL DEFAULT NULL'),
        ('last_anomaly_fingerprint', 'TEXT DEFAULT NULL'),
    ]
    for column, definition in migrations:
        if not column_exists(cursor, 'transcriptions', column):
            cursor.execute(f'ALTER TABLE transcriptions ADD COLUMN {column} {definition}')
            app.logger.info(f"Migrated transcriptions table: added '{column}' column")

def init_db():
    # Initialize the SQLite database schema and performance settings
    """Initialize the database with necessary tables."""
    # A dedicated connection for initialisation, kept out of thread-local storage.
    # sqlite3's context manager only commits or rolls back; closing() is what
    # actually releases the connection.
    with closing(sqlite3.connect(db_file, timeout=db_connection_timeout)) as conn:
        cursor = conn.cursor()

        # Create/migrate schema (tables, indexes, additive columns)
        ensure_schema(cursor)

        # Write-Ahead Logging is a persistent database property, so it only needs
        # setting once here. The per-connection pragmas live in get_db_connection().
        cursor.execute('PRAGMA journal_mode = WAL')

        app.logger.info("Database initialized successfully")
init_db()

def is_garbage_transcription(text, threshold=0.2, min_length=50):
    """Determine if the transcription looks like garbage."""
    preview = text[:1000]
    total_chars = len(preview)
    alnum_chars = sum(c.isalnum() for c in preview)
    if alnum_chars < min_length:
        return True
    ratio = alnum_chars / total_chars if total_chars else 0
    return ratio < threshold

def anomaly_fingerprint(word_count, mean_logprob):
    """A compact identity for a decode result, for comparing attempt to attempt."""
    if mean_logprob is None:
        return f"{int(word_count)}:none"
    return f"{int(word_count)}:{float(mean_logprob):.12f}"


def same_anomaly_result(previous, current):
    """True when two fingerprints describe materially the same decode."""
    if not previous or not current:
        return False
    previous_words, _, previous_logprob = previous.partition(":")
    current_words, _, current_logprob = current.partition(":")
    if previous_words != current_words:
        return False
    if previous_logprob == "none" or current_logprob == "none":
        return previous_logprob == current_logprob
    try:
        return abs(float(previous_logprob) - float(current_logprob)) <= ANOMALY_RETRY_EPSILON
    except ValueError:
        return False


def parse_flagged_segments(raw):
    """Decode the stored flagged_segments JSON for an API response.

    Always a list: the column is null on every row written before 0.6.0 and on any
    job that has not completed, and a client should not have to tell those apart from
    a job with nothing flagged.
    """
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        app.logger.warning(f"Ignoring unparseable flagged_segments value: {raw!r}")
        return []
    return value if isinstance(value, list) else []


def _count_failed_attempt(cursor, guid, reason, terminal_status):
    """Increment the job's attempt counter, then requeue it or move it to terminal_status.

    Returns the resulting status string ('pending' or terminal_status).
    """
    cursor.execute(
        "UPDATE transcriptions SET attempt_count = attempt_count + 1 WHERE guid = ?",
        (guid,)
    )
    cursor.execute("SELECT attempt_count FROM transcriptions WHERE guid = ?", (guid,))
    row = cursor.fetchone()
    attempt_count = row[0] if row else 0

    if attempt_count >= max_garbage_retries:
        cursor.execute(
            "UPDATE transcriptions SET status = ?, completed_at = CURRENT_TIMESTAMP WHERE guid = ?",
            (terminal_status, guid)
        )
        app.logger.error(
            f"Giving up on transcription {guid} after {attempt_count} failed attempt(s) "
            f"({reason}). Moving to terminal '{terminal_status}' status so the queue can advance."
        )
        return terminal_status

    cursor.execute(
        "UPDATE transcriptions SET status = 'pending', transcription = NULL, timings = NULL, "
        "anomaly_count = NULL, anomaly_windows = NULL, flagged_segments = NULL, "
        "rescue_attempted = NULL, rescue_selected = NULL, speech_seconds = NULL "
        # last_anomaly_fingerprint is deliberately NOT cleared: it is how the next
        # attempt recognises that it has produced this exact result before.
        "WHERE guid = ?",
        (guid,)
    )
    # run_forced_alignment skips MFA when <guid>_aligned already holds output, so a
    # requeued job would otherwise refine its new transcript against the previous
    # attempt's word list.
    stale_alignment = os.path.join(app.config['UPLOAD_FOLDER'], f"{guid}_aligned")
    if os.path.isdir(stale_alignment):
        shutil.rmtree(stale_alignment, ignore_errors=True)
        app.logger.info(f"Removed stale alignment output for requeued job {guid}")
    app.logger.warning(
        f"Transcription {guid} failed ({reason}). Resetting to 'pending' "
        f"(attempt {attempt_count} of {max_garbage_retries})."
    )
    return 'pending'

def handle_garbage_result(cursor, guid, reason):
    # Count a garbage result and either requeue the job or quarantine it
    """Increment the job's attempt counter, then requeue or quarantine it.

    After max_garbage_retries consecutive garbage results the job is moved to the
    terminal 'quarantined' status so it stops being the oldest 'pending' row and the
    FIFO queue can advance to the next sermon. Returns the resulting status string
    ('pending' or 'quarantined').
    """
    return _count_failed_attempt(cursor, guid, reason, 'quarantined')

def handle_transient_failure(cursor, guid, reason):
    """Count an exception (CUDA OOM, I/O error, MFA prep failure) against the job.

    Uses the same attempt counter as garbage results: the job is requeued while it
    is below max_garbage_retries and marked 'error' once it reaches the cap, so a
    transient failure gets another run and a deterministic one still terminates.
    Returns the resulting status string ('pending' or 'error').
    """
    return _count_failed_attempt(cursor, guid, reason, 'error')

def recover_stuck_jobs(cursor):
    # Reset orphaned in-flight jobs left in 'processing' (e.g. after a container restart)
    """Reset any rows stuck in 'processing' back to 'pending' and return the count.

    The worker is single-threaded and only ever selects 'pending' rows, so any row in
    'processing' at the top of a worker cycle was left there by a run that died
    mid-transcription (a container restart, or a DB write that failed while marking
    the job 'error'). Recovery is NOT counted as a failed attempt (attempt_count is
    left untouched) since the job never produced a result.
    """
    cursor.execute("SELECT COUNT(*) FROM transcriptions WHERE status = 'processing'")
    stuck = cursor.fetchone()[0]
    if stuck:
        cursor.execute("UPDATE transcriptions SET status = 'pending' WHERE status = 'processing'")
        app.logger.warning(
            f"Recovery: reset {stuck} orphaned job(s) stuck in 'processing' back to 'pending'."
        )
    return stuck

def job_audio_path(guid, filename):
    """Path the uploaded audio was saved to: <UPLOAD_FOLDER>/<guid><lowercased ext>."""
    ext = os.path.splitext(filename)[-1].lower()
    return os.path.join(app.config['UPLOAD_FOLDER'], f"{guid}{ext}")

def remove_path(path):
    """Delete a file or directory tree if it exists. Returns True when something was removed."""
    try:
        if os.path.isfile(path):
            os.remove(path)
            app.logger.info(f"Deleted file: {path}")
            return True
        if os.path.isdir(path):
            shutil.rmtree(path)
            app.logger.info(f"Deleted directory: {path}")
            return True
    except Exception as file_err:
        app.logger.error(f"Failed to delete {path}: {file_err}")
    return False

def build_utterances(segments, duration,
                     gap_s=None, pad_s=None, min_len=None, max_len=None):
    """Merge Whisper segments into MFA utterances, split only at real pauses.

    Production used to hand MFA one utterance per file, so a 20-60 minute recording
    was a single alignment graph and one mismatch could fail or distort the whole
    thing: on the 1369 s reference file that cost 279 s and produced no alignment at
    all. Splitting at gaps of `gap_s` or more bounds the damage to one utterance and
    aligned the same file in 30 s.

    A split happens at a qualifying gap once the utterance is at least `min_len`
    long, or when adding the next segment would push it past `max_len`. Each
    utterance is padded `pad_s` into the adjacent silence without ever overlapping
    its neighbours. Returns dicts with start, end, text and the segment indices.
    """
    gap_s = MFA_UTTERANCE_GAP_SEC if gap_s is None else gap_s
    pad_s = MFA_UTTERANCE_PAD_SEC if pad_s is None else pad_s
    min_len = MFA_UTTERANCE_MIN_SEC if min_len is None else min_len
    max_len = MFA_UTTERANCE_MAX_SEC if max_len is None else max_len

    numbered = [(i, s) for i, s in enumerate(segments) if s.get("text", "").strip()]
    groups = []
    current = []
    for i, seg in numbered:
        if current:
            previous = current[-1][1]
            gap = seg["start"] - previous["end"]
            length = previous["end"] - current[0][1]["start"]
            would_be = seg["end"] - current[0][1]["start"]
            if gap >= gap_s and (length >= min_len or would_be > max_len):
                groups.append(current)
                current = []
        current.append((i, seg))
    if current:
        groups.append(current)

    utterances = [
        {
            "start": g[0][1]["start"],
            "end": g[-1][1]["end"],
            "text": " ".join(s["text"].strip() for _, s in g),
            "segments": [i for i, _ in g],
        }
        for g in groups
    ]
    edges = [(u["start"], u["end"]) for u in utterances]
    for k, u in enumerate(utterances):
        start, end = edges[k]
        previous_end = edges[k - 1][1] if k > 0 else 0.0
        next_start = edges[k + 1][0] if k + 1 < len(utterances) else duration
        # Pad into the gap but never past its midpoint, so neighbours cannot overlap.
        u["start"] = round(max(0.0, start - pad_s, (previous_end + start) / 2.0 if k > 0 else 0.0), 4)
        u["end"] = round(min(duration, end + pad_s,
                             (end + next_start) / 2.0 if k + 1 < len(utterances) else duration), 4)
        if u["end"] - u["start"] < 0.1:
            u["end"] = round(min(duration, u["start"] + 0.1), 4)
    for k in range(1, len(utterances)):
        if utterances[k]["start"] < utterances[k - 1]["end"]:
            utterances[k]["start"] = utterances[k - 1]["end"]
    return utterances


def write_textgrid(utterances, duration, path, tier="speaker"):
    """Write a one-tier TextGrid tiling [0, duration] with the utterances as intervals.

    MFA reads a TextGrid beside the WAV as a multi-utterance corpus; the empty
    intervals between utterances are the silences it is told not to align through.
    """
    intervals = []
    cursor = 0.0
    for u in utterances:
        if u["start"] > cursor + 1e-6:
            intervals.append((cursor, u["start"], ""))
        intervals.append((u["start"], u["end"], u["text"]))
        cursor = u["end"]
    if cursor < duration - 1e-6:
        intervals.append((cursor, duration, ""))

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {duration:.4f}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        f'        name = "{tier}"',
        "        xmin = 0",
        f"        xmax = {duration:.4f}",
        f"        intervals: size = {len(intervals)}",
    ]
    for k, (start, end, text) in enumerate(intervals, 1):
        lines += [
            f"        intervals [{k}]:",
            f"            xmin = {start:.4f}",
            f"            xmax = {end:.4f}",
            f'            text = "{text.replace(chr(34), chr(34) * 2)}"',
        ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def export_wav_for_mfa(audio_path, wav_path):
    """Decode to 16 kHz mono signed 16-bit PCM, the rate MFA's models expect.

    pydub's default export kept the source rate and channel count, which made a
    534 MB WAV of the 2783 s stereo file and let MFA hear a channel mix Whisper
    never saw. The same file is 89 MB through this path.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", audio_path,
         "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav_path],
        check=True, capture_output=True, text=True, timeout=600,
    )


def load_mfa_words(alignment_json_path):
    """Word entries from MFA's JSON output, silences and epsilons dropped."""
    with open(alignment_json_path, "r") as fh:
        data = json.load(fh)
    tiers = data.get("tiers")
    if not isinstance(tiers, dict):
        raise ValueError("MFA output has no tiers")
    words_tier = tiers.get("words")
    if words_tier is None:
        words_tier = next((v for k, v in tiers.items() if k.endswith("words")), None)
    if words_tier is None:
        raise ValueError("MFA output has no words tier")
    return [
        {"start": float(entry[0]), "end": float(entry[1]), "text": entry[2]}
        for entry in words_tier.get("entries", [])
        if entry[2] and entry[2] not in {"<eps>", "sil", "spn"}
    ]


def match_mfa_words(segments, mfa_words):
    """Assign MFA words to Whisper segments by sequence match (harness I2).

    The rule this replaces took every MFA word whose start fell inside a Whisper
    segment's span. That double-assigns a word sitting on a boundary, starts a
    segment on its second word when the first drifted, and degrades to nonsense once
    MFA and Whisper disagree about where in the file they are. Matching the two token
    streams with difflib instead means a word is owned by the segment whose text it
    actually came from, whatever the timings say.

    Returns a list of per-Whisper-token records: {"seg", "mfa"} where "mfa" is the
    index into `mfa_words` or None.
    """
    whisper_tokens = []  # (segment index, token)
    for index, segment in enumerate(segments):
        words = segment.get("words")
        if words:
            for word in words:
                for token in norm_words(word["word"]):
                    whisper_tokens.append((index, token))
        else:
            for token in norm_words(segment.get("text", "")):
                whisper_tokens.append((index, token))

    # An MFA label that normalises to several tokens is matched on its first one.
    mfa_tokens = []
    for word in mfa_words:
        tokens = norm_words(word["text"])
        mfa_tokens.append(tokens[0] if tokens else "")

    matcher = difflib.SequenceMatcher(None, mfa_tokens, [t for _, t in whisper_tokens], autojunk=False)
    matched = [None] * len(whisper_tokens)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            matched[block.b + k] = block.a + k
    return [{"seg": seg, "mfa": mfa} for (seg, _), mfa in zip(whisper_tokens, matched)]


def refine_segment_timings(segments, mfa_words):
    """Refine each segment's edges from its own MFA words, then enforce monotonicity.

    A segment MFA did not cover, or covered so partially that its refined span is
    under half the Whisper span, keeps Whisper's word timestamps rather than its
    VAD-padded segment bounds. Returns (timings, empty_fallbacks, short_fallbacks).
    """
    owned = {}
    for record in match_mfa_words(segments, mfa_words):
        if record["mfa"] is not None:
            owned.setdefault(record["seg"], []).append(record["mfa"])

    refined = []
    empty_fallbacks = 0
    short_fallbacks = 0
    span_ratios = []
    for index, segment in enumerate(segments):
        indices = owned.get(index)
        whisper_start, whisper_end = whisper_span(segment)
        whisper_length = whisper_end - whisper_start
        if indices:
            start = mfa_words[min(indices)]["start"]
            end = mfa_words[max(indices)]["end"]
            # Owning some words is not the same as being aligned. A segment that owns
            # 2 of its 30 words gets a span covering those two, so a 20 second stretch
            # of text would be published against a 0.6 second timing. Treat a refined
            # span under half the Whisper span as a failure to cover the segment.
            if whisper_length > 0:
                # Recorded before the substitution, so the distribution describes what
                # the aligner produced rather than what was published.
                span_ratios.append((end - start) / whisper_length)
                if (end - start) < SPAN_RATIO_FLOOR * whisper_length:
                    short_fallbacks += 1
                    start, end = whisper_start, whisper_end
        else:
            empty_fallbacks += 1
            start, end = whisper_start, whisper_end
        refined.append({"start": float(start), "end": float(end), "text": segment["text"].strip()})

    # One pass to make the sequence monotonic: a later segment can never start
    # before its predecessor ended, and no segment can be empty or inverted. A timing
    # the clamp had to move is not the aligner's answer, so it is counted separately.
    clamped = 0
    previous_end = 0.0
    for entry in refined:
        start = max(entry["start"], previous_end)
        end = max(entry["end"], start + 0.05)
        if start != entry["start"] or end != entry["end"]:
            clamped += 1
        entry["start"], entry["end"] = start, end
        previous_end = end

    stats = {
        "empty_fallbacks": empty_fallbacks,
        "short_fallbacks": short_fallbacks,
        "clamped": clamped,
        "span_ratio_p5": _percentile(span_ratios, 5),
        "span_ratio_median": _percentile(span_ratios, 50),
    }
    return refined, stats


def alignment_agreement(segments, refined):
    """Fraction of segments whose refined edges are both within 250 ms of Whisper's.

    Logged per job as agree250: the harness measured 0.44-0.65 under the old rule
    and 0.52-0.79 under this one, so a sharp drop in production is the signal that
    something upstream has changed.
    """
    if not segments:
        return None
    agree = 0
    for segment, entry in zip(segments, refined):
        start, end = whisper_span(segment)
        if abs(entry["start"] - start) <= ALIGNMENT_AGREE_SEC and abs(entry["end"] - end) <= ALIGNMENT_AGREE_SEC:
            agree += 1
    return round(agree / len(segments), 4)


def whisper_timings(segments):
    """The fallback timings: Whisper's own word timestamps, monotonic, no MFA."""
    refined = []
    previous_end = 0.0
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        start, end = whisper_span(segment)
        start = max(float(start), previous_end)
        end = max(float(end), start + 0.05)
        refined.append({"start": start, "end": end, "text": text})
        previous_end = end
    return refined


def alignment_log_fields(stats):
    """The alignment counters as `key=value` pairs for the per-job log line.

    One place builds them so the alignment line and the completion line cannot
    disagree, and so the sweep's parser sees the same names on either.
    """
    return (
        f"agree250={stats.get('agree250')} "
        f"empty_fallbacks={stats.get('empty_fallbacks')} "
        f"span_ratio_lt_0_5={stats.get('short_fallbacks')} "
        f"span_ratio_p5={stats.get('span_ratio_p5')} "
        f"span_ratio_median={stats.get('span_ratio_median')} "
        f"clamped={stats.get('clamped')}"
    )


def run_forced_alignment(audio_path, whisper_segments, guid, duration_sec=None):
    """Refine Whisper's segment timings with Montreal Forced Aligner.

    MFA is always attempted. It gets a 16 kHz mono WAV and a TextGrid of utterances
    built from the Whisper segments, runs once at its default beams, and its words
    are assigned back to segments by sequence match. Every failure path falls back to
    Whisper's own word timestamps rather than losing the job.

    Returns (timings, mfa_applied, stats). `timings` is the API's list of
    {start, end, text}; `stats` carries agree250, the utterance count, the MFA wall
    time and the number of segments MFA did not cover.
    """
    upload_folder = app.config['UPLOAD_FOLDER']
    aligned_output_dir = os.path.join(upload_folder, f"{guid}_aligned")
    alignment_json_path = os.path.join(aligned_output_dir, f"{guid}.json")
    # Assigned before the try so the finally can always clean up, whichever step raised.
    temp_mfa_dir = os.path.join(upload_folder, f"{guid}_mfa_input")
    # MFA's own working directory for this corpus (features, lattices, the corpus
    # .db and a copy of the acoustic model). --clean wipes it before a run; the
    # finally removes it afterwards so nothing accumulates in the container layer.
    mfa_work_dir = os.path.join(MFA_ROOT_DIR, f"{guid}_mfa_input")

    segments = [s for s in whisper_segments if s.get("text", "").strip()]
    # Every counter the sweep parses off the log line, defaulted so a fallback path
    # still emits the full set.
    stats = {"agree250": None, "utterances": 0, "mfa_wall_s": None,
             "empty_fallbacks": None, "short_fallbacks": None, "clamped": None,
             "span_ratio_p5": None, "span_ratio_median": None}
    if not segments:
        return [], False, stats

    def fallback():
        timings = whisper_timings(segments)
        stats["agree250"] = alignment_agreement(segments, timings)
        return timings, False, stats

    if duration_sec is None or duration_sec <= 0:
        duration_sec = max(s["end"] for s in segments)

    # If alignment already exists, avoid re-running MFA
    if os.path.exists(alignment_json_path):
        app.logger.info(f"MFA alignment already exists for {guid}. Skipping re-run.")
    else:
        try:
            # Verify the actual audio file exists and get its correct path
            if not os.path.exists(audio_path):
                app.logger.error(f"Audio file not found for {guid}: {audio_path}")
                return fallback()

            os.makedirs(aligned_output_dir, exist_ok=True)
            os.makedirs(temp_mfa_dir, exist_ok=True)

            temp_audio_path = os.path.join(temp_mfa_dir, f"{guid}.wav")
            try:
                export_wav_for_mfa(audio_path, temp_audio_path)
                utterances = build_utterances(segments, duration_sec)
                write_textgrid(utterances, duration_sec, os.path.join(temp_mfa_dir, f"{guid}.TextGrid"))
                stats["utterances"] = len(utterances)
                app.logger.info(
                    f"Prepared MFA input for {guid}: {len(utterances)} utterances from "
                    f"{len(segments)} segments, {os.path.getsize(temp_audio_path) / 1e6:.0f} MB WAV"
                )
            except Exception as prep_error:
                app.logger.error(f"Error preparing MFA input files for {guid}: {prep_error}")
                return fallback()

            timeout = max(MFA_TIMEOUT_FLOOR_SEC, MFA_TIMEOUT_BASE_SEC + MFA_TIMEOUT_PER_SEC * duration_sec)
            mfa_command = [
                "mfa", "align",
                temp_mfa_dir,          # Directory holding the WAV and its TextGrid
                MFA_DICTIONARY_PATH,   # Pronunciation dictionary
                MFA_ACOUSTIC_MODEL,    # Acoustic model
                aligned_output_dir,    # Output directory
                "--output_format", "json",
                "--include_original_text",
                "--no_tokenization",
                "--clean",             # Start from an empty working directory every run
                "--overwrite",
            ]
            app.logger.info(f"Running MFA for {guid} at default beams (timeout {timeout:.0f}s)")
            started = time.monotonic()
            try:
                result = subprocess.run(mfa_command, check=True, capture_output=True, text=True, timeout=timeout)
                stats["mfa_wall_s"] = round(time.monotonic() - started, 1)
                # Full MFA output (progress bars included) is only useful when
                # something went wrong, so it stays at DEBUG on success.
                app.logger.debug(f"MFA stdout for {guid}: {result.stdout}")
            except subprocess.TimeoutExpired:
                stats["mfa_wall_s"] = round(time.monotonic() - started, 1)
                app.logger.error(f"MFA for {guid} timed out after {timeout:.0f}s; using Whisper timings.")
                return fallback()
            except subprocess.CalledProcessError as e:
                stats["mfa_wall_s"] = round(time.monotonic() - started, 1)
                app.logger.error(
                    f"MFA for {guid} failed with return code {e.returncode}; using Whisper timings. "
                    f"stderr: {e.stderr}\nstdout: {e.stdout}"
                )
                return fallback()

            if not os.path.exists(alignment_json_path):
                app.logger.error(f"MFA output file not found for {guid}: {alignment_json_path}")
                app.logger.error(f"Command used: {' '.join(mfa_command)}")
                return fallback()

        except Exception as e:
            app.logger.error(f"Unexpected error running MFA for {guid}: {str(e)}")
            return fallback()

        finally:
            for path in (temp_mfa_dir, mfa_work_dir):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)

    try:
        mfa_words = load_mfa_words(alignment_json_path)
    except Exception as e:
        app.logger.error(f"Error processing MFA output for {guid}: {str(e)}")
        return fallback()

    if not mfa_words:
        app.logger.error(f"MFA returned no words for {guid}; using Whisper timings.")
        return fallback()

    refined, refine_stats = refine_segment_timings(segments, mfa_words)
    stats.update(refine_stats)
    stats["agree250"] = alignment_agreement(segments, refined)
    app.logger.info(
        f"Alignment for {guid}: {len(mfa_words)} MFA words over {len(segments)} segments, "
        + alignment_log_fields(stats)
    )
    return refined, True, stats


def process_pending_job(cursor, guid, filename):
    # Transcribe a single pending job end-to-end and update its row accordingly
    """Process one pending transcription job and return its resulting status.

    Encapsulates the per-job control flow so it can be unit tested in isolation:
    missing-file handling, the garbage-detection retry/quarantine path, the
    exception retry path, MFA, and the successful-completion path. Garbage results
    and exceptions both go through the attempt counter so a repeatedly bad job is
    moved out of the way instead of re-blocking the head of the queue.
    """
    try:
        file_path = job_audio_path(guid, filename)

        if not os.path.exists(file_path):
            app.logger.error(f"File {file_path} for {guid} not found. Skipping.")
            cursor.execute(
                "UPDATE transcriptions SET status = 'error', completed_at = CURRENT_TIMESTAMP WHERE guid = ?",
                (guid,)
            )
            return 'error'

        # Mark transcription as 'processing' to prevent duplicate execution
        cursor.execute("UPDATE transcriptions SET status = 'processing' WHERE guid = ?", (guid,))

        app.logger.info(f"Processing transcription for {filename} (GUID: {guid})...")
        started = time.monotonic()

        # Step 1: Transcribe with Whisper
        try:
            result = transcribe_audio(file_path, guid)
        except Exception as e:
            app.logger.error(f"Whisper transcription failed for {guid}: {e}")
            return handle_transient_failure(cursor, guid, f"{type(e).__name__}: {e}")

        transcription = result["transcription"]
        whisper_segment_timings = result["timings"]
        # Full diagnostic segments carry the word timestamps the aligner needs; a
        # stubbed or older transcribe_audio only returns the API timings.
        decode_segments = result.get("segments") or whisper_segment_timings
        # Every count describes the pass transcribe_audio selected: when the primary
        # pass looked bad it ran one rescue pass and kept the better of the two, so
        # the gate below must not judge a file on a score the rescue already fixed.
        anomaly_count = result.get("anomaly_count")
        anomaly_windows = result.get("anomaly_windows")
        flagged_segments = result.get("flagged_segments") or []
        rescue_attempted = bool(result.get("rescue_attempted"))
        rescue_selected = bool(result.get("rescue_selected"))

        # Check for garbage transcription
        if is_garbage_transcription(transcription):
            return handle_garbage_result(cursor, guid, "garbage transcription detected")

        # Word-rate floor: coherent prose at half the expected rate passes the
        # alphanumeric check but is still a collapsed decode. Skip clips under
        # 30 s, where a pause or two swings the rate, and unknown durations.
        duration_sec = result.get("duration_sec", 0) or 0
        # Rate over the speech the decoder was actually given, not over the whole
        # file. A recording that is half silence reads as half rate against total
        # duration, which is why the floor had to sit so low to be safe; over speech
        # the same floor means what it says. Falls back to total duration when the
        # decode did not report duration_after_vad.
        speech_seconds = result.get("speech_seconds") or 0
        denom = speech_seconds or duration_sec
        wps = len(transcription.split()) / denom if denom > 0 else 0
        if duration_sec >= 30 and wps < MIN_WORDS_PER_SEC:
            return handle_garbage_result(
                cursor, guid,
                f"word rate {wps:.2f} words/sec over {denom:.0f}s of "
                f"{'speech' if speech_seconds else 'audio'} below floor {MIN_WORDS_PER_SEC}"
            )

        # Anomaly gate: the whole-file word rate is blind to a partial collapse, so a
        # file that transcribes normally for forty minutes and produces nothing for
        # ten still clears the floor above. The window count catches that, and the
        # segment count catches a decode that fell back to high temperatures or
        # tripped faster-whisper's own compression and log-probability thresholds
        # repeatedly.
        if ((anomaly_windows or 0) >= ANOMALY_WINDOWS_MAX
                or (anomaly_count or 0) >= ANOMALY_SEGMENTS_MAX):
            reason = (
                f"anomaly gate: {anomaly_count} flagged segments "
                f"(max {ANOMALY_SEGMENTS_MAX - 1}), {anomaly_windows} low speech windows "
                f"(max {ANOMALY_WINDOWS_MAX - 1})"
            )
            fingerprint = anomaly_fingerprint(len(transcription.split()),
                                              result.get("mean_logprob"))
            cursor.execute(
                "SELECT last_anomaly_fingerprint FROM transcriptions WHERE guid = ?", (guid,)
            )
            row = cursor.fetchone()
            previous_fingerprint = row[0] if row else None

            if same_anomaly_result(previous_fingerprint, fingerprint):
                # The decode is deterministic, so retrying cannot change the verdict.
                # tcf.20150424 proved the point: three identical re-decodes, 510 s of
                # GPU, and a quarantined row with nothing published against a legacy
                # transcript of 8292 words. A quality signal must not black-hole a
                # recording that produced a plausible transcript; quarantine is for
                # results that are unusable, which the garbage check and the word-rate
                # floor above already decide. Publish, with the anomaly fields and the
                # flags set so it can be found and reviewed.
                app.logger.warning(
                    f"Publishing {guid} despite the {reason}: the re-decode reproduced the "
                    f"previous attempt exactly ({fingerprint}), so retrying cannot clear the "
                    f"gate and quarantining would deliver nothing for a transcript of "
                    f"{len(transcription.split())} words."
                )
            else:
                cursor.execute(
                    "UPDATE transcriptions SET last_anomaly_fingerprint = ? WHERE guid = ?",
                    (fingerprint, guid),
                )
                return handle_garbage_result(cursor, guid, reason)

        # Step 2: Run Forced Alignment (MFA). An alignment problem is not a
        # transcription problem: fall back to Whisper's own timings without
        # spending a retry on it.
        app.logger.info(f"Running forced alignment for {filename} (GUID: {guid})...")
        refined_timings, mfa_applied, alignment_stats = run_forced_alignment(
            file_path, decode_segments, guid, duration_sec
        )
        if not refined_timings or all(not seg.get("text") for seg in refined_timings):
            app.logger.warning(f"Alignment returned no usable timings for {guid}; using Whisper timings.")
            refined_timings, mfa_applied = whisper_segment_timings, False

        # Step 3: Update database with refined timings and per-job metrics.
        # attempt_count is left as-is: on a completed row it records how many
        # failed attempts preceded the success, and nothing re-reads it once the
        # job is terminal.
        processing_seconds = round(time.monotonic() - started, 1)
        cursor.execute(
            "UPDATE transcriptions SET transcription = ?, timings = ?, status = 'completed', "
            "processing_seconds = ?, words_per_second = ?, mfa_applied = ?, "
            "anomaly_count = ?, anomaly_windows = ?, flagged_segments = ?, "
            "rescue_attempted = ?, rescue_selected = ?, speech_seconds = ?, "
            "completed_at = CURRENT_TIMESTAMP WHERE guid = ?",
            (
                transcription,
                json.dumps(refined_timings),
                processing_seconds,
                round(wps, 3) if denom > 0 else None,
                1 if mfa_applied else 0,
                anomaly_count,
                anomaly_windows,
                json.dumps(flagged_segments),
                1 if rescue_attempted else 0,
                1 if rescue_selected else 0,
                round(speech_seconds, 2) if speech_seconds else None,
                guid,
            )
        )
        app.logger.info(
            f"Transcription completed for {filename} (GUID: {guid}): "
            f"{len(transcription.split())} words in {duration_sec:.1f}s "
            f"({speech_seconds:.1f}s speech, {wps:.2f} words/sec over "
            f"{'speech' if speech_seconds else 'audio'}, floor {MIN_WORDS_PER_SEC}), "
            f"{processing_seconds:.1f}s processing, mfa_applied={mfa_applied}, "
            f"anomaly_count={anomaly_count}, anomaly_windows={anomaly_windows}, "
            f"flagged_segments={len(flagged_segments)}, "
            f"rescue_attempted={rescue_attempted}, rescue_selected={rescue_selected}, "
            + alignment_log_fields(alignment_stats)
        )
        return 'completed'

    except Exception as e:
        app.logger.error(f"Error processing transcription for {guid}: {e}")
        # Count it against the job only if nothing else already has: an inner
        # handler that raised part-way through may have moved the row on, and
        # counting again would burn a second attempt on one failure. If this
        # read or write fails too, the row stays in 'processing' and the next
        # worker cycle's recovery requeues it.
        cursor.execute("SELECT status FROM transcriptions WHERE guid = ?", (guid,))
        row = cursor.fetchone()
        current = row[0] if row else None
        if current != 'processing':
            app.logger.warning(f"Job {guid} is already '{current}' after the failure; not counting it again.")
            return current
        return handle_transient_failure(cursor, guid, f"{type(e).__name__}: {e}")


def job_artifact_paths(upload_folder, guid, filename):
    """Every path a job can leave behind, on the upload folder and under MFA's root."""
    ext = os.path.splitext(filename or "")[-1].lower()
    return [
        os.path.join(upload_folder, f"{guid}{ext}"),         # Original audio file
        os.path.join(upload_folder, f"{guid}.txt"),          # Transcript used by MFA
        os.path.join(upload_folder, f"{guid}_aligned"),      # MFA output
        os.path.join(upload_folder, f"{guid}_mfa_input"),    # MFA input temp directory
        os.path.join(MFA_ROOT_DIR, f"{guid}_mfa_input"),     # MFA working directory
    ]

def sweep_orphaned_files(cursor, upload_folder, now=None):
    """Remove old entries in upload_folder that no live job refers to.

    The upload folder is a host bind mount and outlives the container-layer
    database, so files from before a redeploy have no row and would otherwise
    never be cleaned. Only entries named after a GUID are considered, only when
    older than ORPHAN_FILE_AGE_SEC, and only when no 'pending' or 'processing'
    row exists for that GUID. Returns the number of entries removed.
    """
    now = time.time() if now is None else now
    try:
        entries = list(os.scandir(upload_folder))
    except FileNotFoundError:
        return 0

    removed = 0
    for entry in entries:
        match = UUID_PREFIX_RE.match(entry.name)
        if not match:
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if now - mtime < ORPHAN_FILE_AGE_SEC:
            continue
        guid = match.group(0).lower()
        cursor.execute(
            "SELECT 1 FROM transcriptions WHERE lower(guid) = ? AND status IN ('pending', 'processing')",
            (guid,)
        )
        if cursor.fetchone():
            continue
        if remove_path(entry.path):
            removed += 1
    if removed:
        app.logger.info(f"Swept {removed} orphaned upload(s) with no live job")
    return removed

def cleanup_old_jobs(cursor, upload_folder):
    """Delete rows that reached a terminal state more than CLEANUP_ROW_AGE ago, with their files.

    Files and rows are removed for the same GUID set, so a row never disappears
    while its files stay behind. Finishes with the orphaned-file sweep. Returns
    (rows_deleted, orphans_swept).
    """
    # Age is measured from completion, not submission: a job that waited a day
    # in a deep queue and finished minutes ago still has a client polling for it.
    cursor.execute(
        "SELECT guid, filename FROM transcriptions "
        "WHERE status IN ('completed', 'error', 'quarantined') "
        "AND COALESCE(completed_at, created_at) <= datetime('now', ?)",
        (CLEANUP_ROW_AGE,)
    )
    old_records = cursor.fetchall()

    for guid, filename in old_records:
        for path in job_artifact_paths(upload_folder, guid, filename):
            remove_path(path)

    if old_records:
        cursor.executemany(
            "DELETE FROM transcriptions WHERE guid = ?",
            [(guid,) for guid, _ in old_records]
        )
        app.logger.info(f"Cleanup: deleted {len(old_records)} old transcription(s) and their files")

    swept = sweep_orphaned_files(cursor, upload_folder)
    return len(old_records), swept


@app.route('/transcriptions', methods=['GET'])
# Endpoint to list all transcription jobs with metadata
def get_all_transcriptions():
    """Returns all transcriptions with status, GUID, submission, completion timestamps, and estimated processing time."""
    # Thread-local connection; teardown_appcontext closes it when the request ends.
    conn = get_db_connection()
    cursor = conn.cursor()
    # Order by created_at to ensure consistent ordering
    cursor.execute("""
        SELECT guid, filename, status, created_at, completed_at, processing_time_est,
               processing_seconds, words_per_second, attempt_count, mfa_applied,
               anomaly_count, anomaly_windows, flagged_segments,
               rescue_attempted, rescue_selected, speech_seconds
        FROM transcriptions
        ORDER BY created_at DESC
    """)

    records = cursor.fetchall()

    result = [{
        'guid': row[0],
        'filename': row[1],
        'status': row[2],
        'submitted_at': row[3],
        'completed_at': row[4] if row[4] is not None else "",
        'processing_time_est': row[5],  # Processing time estimate in seconds
        # Per-job metrics (null until the job completes)
        'processing_seconds': row[6],
        'words_per_second': row[7],
        'attempt_count': row[8],
        'mfa_applied': None if row[9] is None else bool(row[9]),
        # 0.6.0 decode diagnostics (null until the job completes)
        'anomaly_count': row[10],
        'anomaly_windows': row[11],
        'flagged_segments': parse_flagged_segments(row[12]),
        'rescue_attempted': None if row[13] is None else bool(row[13]),
        'rescue_selected': None if row[14] is None else bool(row[14]),
        'speech_seconds': row[15],
    } for row in records]

    return jsonify(result), 200


def is_canonical_uuid(guid):
    """True when guid is the canonical hyphenated form (braces, urn: prefixes and
    unhyphenated hex are rejected, so one job cannot be registered under several
    spellings of the same UUID)."""
    try:
        parsed = uuid.UUID(guid)
    except (ValueError, TypeError, AttributeError):
        return False
    return str(parsed) == guid.lower()


@app.route('/upload', methods=['POST'])
# Endpoint to upload an audio file and register a transcription job
def upload_audio():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Check if GUID is provided
        guid = request.form.get('guid')
        if not guid:
            return jsonify({'error': 'GUID is required'}), 400

        # Validate GUID format
        if not is_canonical_uuid(guid):
            return jsonify({'error': 'Invalid GUID format. Must be a valid UUID v4'}), 400

        file_extension = os.path.splitext(file.filename)[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return jsonify({
                'error': f"Unsupported file type '{file_extension}'. "
                         f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            }), 400

        saved_filename = f"{guid}{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)

        # Existence check, insert and save happen under one lock so a burst of
        # resubmits for the same GUID yields exactly one 201. The row is claimed
        # before any bytes are written, so a loser never overwrites the winner's
        # file. The worker takes the same lock around its pending-row SELECT, so
        # it cannot pick the row up before the file exists.
        with upload_lock:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Case-insensitive, matching the orphan sweep: one UUID in two casings
            # is one job.
            cursor.execute("SELECT 1 FROM transcriptions WHERE lower(guid) = lower(?)", (guid,))
            if cursor.fetchone():
                return jsonify({'error': 'GUID already exists'}), 409  # Conflict status

            try:
                cursor.execute(
                    "INSERT INTO transcriptions (guid, filename, processing_time_est) VALUES (?, ?, 0)",
                    (guid, file.filename)
                )
            except sqlite3.IntegrityError:
                # Not reachable through this process (the lock covers the check),
                # but a second process on the same database could race us.
                return jsonify({'error': 'GUID already exists'}), 409

            try:
                file.save(file_path)
            except Exception as save_error:
                app.logger.error(f"Failed to save upload {guid} ({file.filename}): {save_error}")
                remove_path(file_path)
                cursor.execute("DELETE FROM transcriptions WHERE guid = ?", (guid,))
                return jsonify({'error': 'Internal Server Error'}), 500

            # Probe the duration (ffprobe, no decode). A file that cannot be read is
            # rejected outright: both the file and the row go away.
            try:
                duration_sec = get_audio_duration(file_path)
            except Exception as decode_error:
                app.logger.warning(f"Rejected upload {guid} ({file.filename}): could not decode audio: {decode_error}")
                remove_path(file_path)
                cursor.execute("DELETE FROM transcriptions WHERE guid = ?", (guid,))
                return jsonify({'error': 'Could not decode audio'}), 400

            processing_time_est_sec = estimate_processing_seconds(duration_sec)
            cursor.execute(
                "UPDATE transcriptions SET processing_time_est = ? WHERE guid = ?",
                (processing_time_est_sec, guid)
            )

            # Queue ahead of this job: everything waiting plus the job in flight.
            cursor.execute(
                "SELECT COALESCE(SUM(processing_time_est), 0) FROM transcriptions "
                "WHERE status IN ('pending', 'processing') AND guid != ?",
                (guid,)
            )
            queue_ahead_sec = cursor.fetchone()[0]

        total_processing_time_sec = queue_ahead_sec + processing_time_est_sec
        estimated_completion_utc = utcnow() + timedelta(seconds=total_processing_time_sec)

        app.logger.info(
            f"File {file.filename} received and saved as {saved_filename} with GUID {guid}. "
            f"Duration: {duration_sec:.1f}s. "
            f"Estimated processing time: {processing_time_est_sec / 60:.2f} min. "
            f"Total queue time: {total_processing_time_sec / 60:.2f} min. "
            f"Check back at {format_utc(estimated_completion_utc)}"
        )

        return jsonify({
            'message': 'File uploaded successfully',
            'guid': guid,
            'estimated_completion_utc': format_utc(estimated_completion_utc)
        }), 201  # Created status

    except HTTPException:
        # Werkzeug's own responses (413 for an oversize body) pass through unchanged.
        raise
    except FileNotFoundError as e:
        app.logger.error(f"File not found error: {e}")
        return jsonify({'error': 'File not found'}), 400
    except ValueError as e:
        app.logger.error(f"Value error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error processing request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500



@app.route('/status/<guid>', methods=['GET'])
# Endpoint to check the status or result of a transcription job
def get_transcription(guid):
    """Get the status or results of a specific transcription job by GUID."""
    conn = get_db_connection()  # Thread-local connection, closed by teardown_appcontext
    cursor = conn.cursor()

    # Retrieve the requested transcription details
    cursor.execute("""
        SELECT status, transcription, timings, created_at, processing_time_est,
               processing_seconds, words_per_second, attempt_count, mfa_applied,
               anomaly_count, anomaly_windows, flagged_segments,
               rescue_attempted, rescue_selected, speech_seconds
        FROM transcriptions WHERE guid = ?
    """, (guid,))
    row = cursor.fetchone()

    if row is None:
        return jsonify({'error': 'GUID not found'}), 404

    (status, transcription, timings, created_at, processing_time_est,
     processing_seconds, words_per_second, attempt_count, mfa_applied,
     anomaly_count, anomaly_windows, flagged_segments,
     rescue_attempted, rescue_selected, speech_seconds) = row

    # If already completed, return the results immediately
    if status == 'completed' or status == 'processed':  # Support both new and old status values during transition
        return jsonify({
            'status': 'completed',
            'transcription': transcription or "",
            'timings': json.loads(timings) if timings else [],  # Return timings as a list
            # Additive per-job metrics
            'processing_seconds': processing_seconds,
            'words_per_second': words_per_second,
            'attempt_count': attempt_count,
            'mfa_applied': None if mfa_applied is None else bool(mfa_applied),
            # 0.6.0 decode diagnostics
            'anomaly_count': anomaly_count,
            'anomaly_windows': anomaly_windows,
            'flagged_segments': parse_flagged_segments(flagged_segments),
            'rescue_attempted': None if rescue_attempted is None else bool(rescue_attempted),
            'rescue_selected': None if rescue_selected is None else bool(rescue_selected),
            'speech_seconds': speech_seconds,
        }), 200

    # Handle error / quarantined status (both terminal failures). 'quarantined' means the
    # job repeatedly produced garbage and was removed from the queue; report it as an error
    # so existing clients treat it as a terminal failure rather than polling forever.
    if status == 'error' or status == 'quarantined':
        return jsonify({
            'status': 'error',
            'message': 'Transcription failed'
        }), 200

    # For pending or processing jobs, calculate estimated completion time
    created_at_dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")

    # Sum of processing times for jobs ahead in the queue
    cursor.execute("""
        SELECT COALESCE(SUM(processing_time_est), 0)
        FROM transcriptions
        WHERE status IN ('pending', 'processing')
        AND created_at < ?
    """, (created_at,))
    ahead_sec = cursor.fetchone()[0]

    # Anchor on the submission time, but never report a completion time that has
    # already passed: a slow queue or a requeued job would otherwise show a stale
    # ETA until it finishes.
    computed = created_at_dt + timedelta(seconds=ahead_sec + processing_time_est)
    now_naive_utc = utcnow().replace(tzinfo=None)
    estimated_completion_utc = max(now_naive_utc, computed)
    formatted_completion_time = format_utc(estimated_completion_utc)

    # Handle 'processing' status
    if status == 'processing':
        return jsonify({
            'status': 'processing',
            'message': 'Transcription is currently in progress',
            'estimated_completion_utc': formatted_completion_time
        }), 200

    # Handle 'pending' status
    if status == 'pending':
        return jsonify({
            'status': 'pending',
            'estimated_completion_utc': formatted_completion_time
        }), 200

    return jsonify({'error': 'Unknown status'}), 500


# Worker liveness, read by /health. The worker thread is the only writer.
worker_state = {
    'last_wake': None,      # datetime (UTC) of the last cycle start
    'last_cleanup': None,   # time.monotonic() of the last cleanup pass
    'busy_since': None,     # datetime (UTC) while a job is being processed, else None
    'idle_logged': False,   # so the idle transition is logged once, not every poll
}
worker_thread = None


@app.route('/health', methods=['GET'])
def health():
    """Liveness for the worker thread plus a queue snapshot.

    503 when the worker thread is not running, or when it is not mid-job and has
    not polled within WORKER_STALE_AFTER. A worker inside a long transcription is
    busy, not stale, so a 46 minute file does not trip the check.
    """
    cursor = get_db_connection().cursor()
    cursor.execute(
        "SELECT status, COUNT(*) FROM transcriptions "
        "WHERE status IN ('pending', 'processing') GROUP BY status"
    )
    counts = dict(cursor.fetchall())

    alive = worker_thread is not None and worker_thread.is_alive()
    last_wake = worker_state['last_wake']
    busy = worker_state['busy_since'] is not None
    stale = last_wake is None or (not busy and utcnow() - last_wake > WORKER_STALE_AFTER)
    healthy = alive and not stale

    body = {
        'status': 'ok' if healthy else 'unhealthy',
        'worker_alive': alive,
        'worker_busy': busy,
        'worker_last_wake_utc': format_utc(last_wake) if last_wake else None,
        'pending': counts.get('pending', 0),
        'processing': counts.get('processing', 0),
        'whisper_loaded': bool(whisper_model_loaded()),
    }
    return jsonify(body), (200 if healthy else 503)


def worker_cycle(cursor, state=worker_state):
    """One worker pass: recover orphans, clean up if due, process one pending job.

    Returns True when a job reached a terminal state (the caller should loop
    straight away) and False when the queue was empty or the job was requeued
    for a retry (the caller should sleep first).
    """
    state['last_wake'] = utcnow()

    # Any 'processing' row at this point was orphaned by a previous run or by a
    # failed 'error' write; requeue it before selecting.
    recover_stuck_jobs(cursor)

    now = time.monotonic()
    if state['last_cleanup'] is None or now - state['last_cleanup'] >= CLEANUP_INTERVAL_SEC:
        try:
            cleanup_old_jobs(cursor, app.config['UPLOAD_FOLDER'])
        except Exception as cleanup_error:
            app.logger.error(f"Cleanup error: {cleanup_error}")
        state['last_cleanup'] = now

    # Fetch the oldest pending transcription. The upload lock keeps a row that
    # /upload has inserted but not yet saved the file for out of view.
    with upload_lock:
        cursor.execute("""
            SELECT guid, filename FROM transcriptions
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT 1
        """)
        row = cursor.fetchone()

    if row is None:
        if not state['idle_logged']:
            app.logger.info("No pending transcriptions. Worker idle.")
            state['idle_logged'] = True
        return False

    state['idle_logged'] = False
    guid, filename = row
    state['busy_since'] = utcnow()
    try:
        status = process_pending_job(cursor, guid, filename)
    finally:
        state['busy_since'] = None
    # A requeued job ('pending' again after a garbage result or an exception)
    # must not be retried on the very next iteration, or an instantaneous
    # failure burns every attempt in seconds. Returning False makes the caller
    # sleep for POLL_INTERVAL_SEC before the retry.
    return status != 'pending'


def transcription_worker():
    # Background thread that polls for pending jobs and processes them
    """Background worker: drains the queue, then polls every POLL_INTERVAL_SEC while idle."""
    app.logger.info(
        f"Transcription worker started. Polling every {POLL_INTERVAL_SEC} seconds while idle."
    )

    while True:
        processed = False
        try:
            # Thread-local connection dedicated to the worker thread
            conn = get_db_connection()
            processed = worker_cycle(conn.cursor())
        except Exception as e:
            app.logger.error(f"Worker error: {e}")

        if not processed:
            app.logger.debug(f"Worker sleeping for {POLL_INTERVAL_SEC} seconds")
            time.sleep(POLL_INTERVAL_SEC)
            app.logger.debug("Worker waking up to check for pending transcriptions")


def start_worker():
    """Start the background worker thread (once) and record it for /health."""
    global worker_thread
    if worker_thread is not None and worker_thread.is_alive():
        return worker_thread
    worker_thread = threading.Thread(target=transcription_worker, name="transcription-worker", daemon=True)
    worker_thread.start()
    return worker_thread


# Cleanup function to close database connections when app is shutting down
@app.teardown_appcontext
# Flask hook to clean up DB connections after each request
def shutdown_session(exception=None):
    """Ensure the request thread's connection is closed when the app context ends."""
    close_db_connection()

# Register a function to clean up connections when Flask is shutting down
def cleanup_connections():
    # Called on app shutdown to close any open DB connections
    """Clean up all database connections when the application is shutting down."""
    app.logger.info("Shutting down application, closing database connections...")
    close_db_connection()
    app.logger.info("Database connections closed.")

if __name__ == "__main__":
    # Load the Whisper model once at startup
    model = load_whisper_model()

    # Set up database connection cleanup on app shutdown
    import atexit
    atexit.register(cleanup_connections)

    # Start the worker thread
    app.logger.info("Starting transcription worker thread...")
    start_worker()
    app.logger.info("Transcription worker thread started successfully.")

    # Run the app. Debug mode is off: the Werkzeug debugger exposes an interactive
    # console on unhandled exceptions, which must never be reachable on a service
    # bound to 0.0.0.0. Set FLASK_DEBUG=1 in the environment for local debugging.
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        use_reloader=False
    )
