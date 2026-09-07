import os
import re
import uuid
import sqlite3
import threading  # For background processing and thread-local DB connections
import time
import json
import shutil
import subprocess
from flask import Flask, request, jsonify  # Web framework and request handling
from werkzeug.exceptions import HTTPException
from transcribe import (  # Custom transcription logic
    transcribe_audio,
    load_whisper_model,
    get_audio_duration,
    estimate_processing_seconds,
    whisper_model_loaded,
)
from pydub import AudioSegment  # Audio file manipulation (WAV export for MFA)
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
# material runs 2.5-2.8 words/sec; the 2026-09 decode collapse produced 0.54 and the
# per-pass penalty in transcribe.py already treats anything under 1.4 as low. The
# alphanumeric-ratio check alone is blind to coherent prose at half length.
MIN_WORDS_PER_SEC = float(os.getenv("MIN_WORDS_PER_SEC", "1.0"))

# Where Montreal Forced Aligner keeps its per-corpus working directory. MFA names
# it after the corpus directory basename, so a job's tree is <root>/<guid>_mfa_input.
MFA_ROOT_DIR = os.getenv("MFA_ROOT_DIR", "/mfa")
MFA_DICTIONARY_PATH = "/mfa/pretrained_models/dictionary/english_mfa.dict"
MFA_ACOUSTIC_MODEL = "english_mfa"

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
            mfa_applied INTEGER DEFAULT NULL
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
    ]
    for column, definition in migrations:
        if not column_exists(cursor, 'transcriptions', column):
            cursor.execute(f'ALTER TABLE transcriptions ADD COLUMN {column} {definition}')
            app.logger.info(f"Migrated transcriptions table: added '{column}' column")

def init_db():
    # Initialize the SQLite database schema and performance settings
    """Initialize the database with necessary tables."""
    # Use a temporary connection specifically for initialization to avoid impacting thread-local storage
    with sqlite3.connect(db_file, timeout=db_connection_timeout) as conn:
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
        "UPDATE transcriptions SET status = 'pending', transcription = NULL, timings = NULL WHERE guid = ?",
        (guid,)
    )
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

def run_forced_alignment(audio_path, whisper_segments, guid):
    # Use Montreal Forced Aligner to refine Whisper's segment timings
    """
    Runs Montreal Forced Aligner (MFA) to refine the timestamps from the Whisper transcript,
    ensuring they align with Whisper's segment structure. MFA is always attempted; if it
    fails for any reason the Whisper segments are returned unchanged.

    Returns (segments, mfa_applied): mfa_applied is True only when MFA output was
    parsed and used, False on every fallback path.
    """
    upload_folder = app.config['UPLOAD_FOLDER']
    transcript_path = os.path.join(upload_folder, f"{guid}.txt")
    aligned_output_dir = os.path.join(upload_folder, f"{guid}_aligned")
    alignment_json_path = os.path.join(aligned_output_dir, f"{guid}.json")
    # Assigned before the try so the finally can always clean up, whichever step raised.
    temp_mfa_dir = os.path.join(upload_folder, f"{guid}_mfa_input")
    # MFA's own working directory for this corpus (features, lattices, the corpus
    # .db and a copy of the acoustic model). --clean wipes it before a run; the
    # finally removes it afterwards so nothing accumulates in the container layer.
    mfa_work_dir = os.path.join(MFA_ROOT_DIR, f"{guid}_mfa_input")

    # If alignment already exists, avoid re-running MFA
    if os.path.exists(alignment_json_path):
        app.logger.info(f"MFA alignment already exists for {guid}. Skipping re-run.")
    else:
        try:
            # Verify the actual audio file exists and get its correct path
            if not os.path.exists(audio_path):
                app.logger.error(f"Audio file not found for {guid}: {audio_path}")
                return whisper_segments, False

            # Save transcript to a file for MFA (remove extra spaces and clean up text)
            transcript_content = " ".join(seg["text"].strip() for seg in whisper_segments if seg["text"].strip())

            # Remove repeated words/phrases that might confuse alignment
            transcript_content = re.sub(r'\b(\w+)\s+\1\s+\1\s+\1+', r'\1', transcript_content)

            with open(transcript_path, "w") as f:
                f.write(transcript_content)

            # Ensure output directory exists
            os.makedirs(aligned_output_dir, exist_ok=True)

            # Create a temporary directory with properly named files for MFA
            os.makedirs(temp_mfa_dir, exist_ok=True)

            # Copy files with consistent naming for MFA
            temp_audio_path = os.path.join(temp_mfa_dir, f"{guid}.wav")
            temp_transcript_path = os.path.join(temp_mfa_dir, f"{guid}.txt")

            # Convert audio to WAV format for MFA compatibility
            try:
                audio = AudioSegment.from_file(audio_path)
                audio.export(temp_audio_path, format="wav")
                shutil.copy2(transcript_path, temp_transcript_path)
                app.logger.info(f"Created MFA input files for {guid}: {temp_audio_path}, {temp_transcript_path}")
            except Exception as prep_error:
                app.logger.error(f"Error preparing MFA input files for {guid}: {prep_error}")
                return whisper_segments, False

            # Run MFA command with progressive beam sizes (fallback strategy)
            beam_configs = [
                {"beam": "40", "retry_beam": "100"},  # First try: moderate increase
                {"beam": "100", "retry_beam": "400"}   # Fallback: large increase
            ]

            mfa_success = False
            for i, config in enumerate(beam_configs):
                mfa_command = [
                    "mfa", "align",
                    temp_mfa_dir,  # Directory containing properly named audio & transcript
                    MFA_DICTIONARY_PATH,  # Pronunciation dictionary
                    MFA_ACOUSTIC_MODEL,  # Acoustic model
                    aligned_output_dir, # Output directory
                    "--output_format", "json",
                    "--beam", config["beam"],
                    "--retry_beam", config["retry_beam"],
                    "--clean",  # Start from an empty working directory every run
                ]

                app.logger.info(
                    f"MFA attempt {i+1} for {guid} with beam={config['beam']}, retry_beam={config['retry_beam']}"
                )
                try:
                    result = subprocess.run(mfa_command, check=True, capture_output=True, text=True, timeout=300)
                    mfa_success = True
                    # Full MFA output (progress bars included) is only useful when
                    # something went wrong, so it stays at DEBUG on success.
                    app.logger.debug(f"MFA stdout for {guid}: {result.stdout}")
                    break
                except subprocess.TimeoutExpired:
                    app.logger.warning(f"MFA attempt {i+1} for {guid} timed out after 5 minutes")
                except subprocess.CalledProcessError as e:
                    app.logger.warning(
                        f"MFA attempt {i+1} for {guid} failed with return code {e.returncode}. "
                        f"stderr: {e.stderr}\nstdout: {e.stdout}"
                    )

            if not mfa_success:
                app.logger.error(f"All MFA attempts failed for {guid}; using Whisper timings.")
                return whisper_segments, False

            if not os.path.exists(alignment_json_path):
                app.logger.error(f"MFA output file not found for {guid}: {alignment_json_path}")
                app.logger.error(f"Command used: {' '.join(mfa_command)}")
                return whisper_segments, False  # Return original Whisper segments if no output

        except Exception as e:
            app.logger.error(f"Unexpected error running MFA for {guid}: {str(e)}")
            return whisper_segments, False  # Fallback to Whisper segments

        finally:
            for path in (temp_mfa_dir, mfa_work_dir):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)

    # Read MFA output file
    try:
        with open(alignment_json_path, "r") as f:
            alignment_data = json.load(f)

        # Ensure MFA output format is valid
        if "tiers" not in alignment_data or "words" not in alignment_data["tiers"]:
            app.logger.error(f"'words' tier missing in MFA output for {guid}: {alignment_data}")
            return whisper_segments, False  # Fallback to Whisper's segment timings

        # Extract word-level alignments
        word_entries = alignment_data["tiers"]["words"]["entries"]
        words = [{"start": entry[0], "end": entry[1], "text": entry[2]} for entry in word_entries]

        # Align MFA words within Whisper's segment-level structure
        refined_segments = []
        for segment in whisper_segments:
            whisper_start, whisper_end, segment_text = segment["start"], segment["end"], segment["text"]

            # Find words within the Whisper segment boundary
            segment_words = [word for word in words if whisper_start <= word["start"] <= whisper_end]

            if segment_words:
                refined_start = segment_words[0]["start"]
                refined_end = segment_words[-1]["end"]
            else:
                refined_start, refined_end = whisper_start, whisper_end  # Fallback to Whisper timings

            refined_segments.append({
                "start": refined_start,
                "end": refined_end,
                "text": segment_text
            })

        return refined_segments, True  # Return segment-level alignment

    except Exception as e:
        app.logger.error(f"Error processing MFA output for {guid}: {str(e)}")
        return whisper_segments, False  # Fallback to Whisper segments if JSON parsing fails


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

        # Check for garbage transcription
        if is_garbage_transcription(transcription):
            return handle_garbage_result(cursor, guid, "garbage transcription detected")

        # Word-rate floor: coherent prose at half the expected rate passes the
        # alphanumeric check but is still a collapsed decode. Skip clips under
        # 30 s, where a pause or two swings the rate, and unknown durations.
        duration_sec = result.get("duration_sec", 0) or 0
        wps = len(transcription.split()) / duration_sec if duration_sec > 0 else 0
        if duration_sec >= 30 and wps < MIN_WORDS_PER_SEC:
            return handle_garbage_result(
                cursor, guid,
                f"word rate {wps:.2f} words/sec below floor {MIN_WORDS_PER_SEC}"
            )

        # Step 2: Run Forced Alignment (MFA). An alignment problem is not a
        # transcription problem: fall back to Whisper's own timings without
        # spending a retry on it.
        app.logger.info(f"Running forced alignment for {filename} (GUID: {guid})...")
        refined_timings, mfa_applied = run_forced_alignment(file_path, whisper_segment_timings, guid)
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
            "completed_at = CURRENT_TIMESTAMP WHERE guid = ?",
            (
                transcription,
                json.dumps(refined_timings),
                processing_seconds,
                round(wps, 3) if duration_sec > 0 else None,
                1 if mfa_applied else 0,
                guid,
            )
        )
        app.logger.info(
            f"Transcription completed for {filename} (GUID: {guid}): "
            f"{len(transcription.split())} words in {duration_sec:.1f}s "
            f"({wps:.2f} words/sec, floor {MIN_WORDS_PER_SEC}), "
            f"{processing_seconds:.1f}s processing, mfa_applied={mfa_applied}"
        )
        return 'completed'

    except Exception as e:
        app.logger.error(f"Error processing transcription for {guid}: {e}")
        # Count it against the job; if this write fails too the row stays in
        # 'processing' and the next worker cycle's recovery requeues it.
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
    """Delete terminal rows older than CLEANUP_ROW_AGE together with their files.

    Files and rows are removed for the same GUID set, so a row never disappears
    while its files stay behind. Finishes with the orphaned-file sweep. Returns
    (rows_deleted, orphans_swept).
    """
    cursor.execute(
        "SELECT guid, filename FROM transcriptions "
        "WHERE status IN ('completed', 'error', 'quarantined') "
        "AND created_at <= datetime('now', ?)",
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
               processing_seconds, words_per_second, attempt_count, mfa_applied
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

        # Existence check, save and insert happen under one lock so a burst of
        # resubmits for the same GUID yields exactly one 201 and no overwritten file.
        with upload_lock:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT guid FROM transcriptions WHERE guid = ?", (guid,))
            if cursor.fetchone():
                return jsonify({'error': 'GUID already exists'}), 409  # Conflict status

            file.save(file_path)

            # Probe the duration (ffprobe, no decode). A file that cannot be read is
            # rejected before any row exists, and removed so it is not orphaned.
            try:
                duration_sec = get_audio_duration(file_path)
            except Exception as decode_error:
                app.logger.warning(f"Rejected upload {guid} ({file.filename}): could not decode audio: {decode_error}")
                remove_path(file_path)
                return jsonify({'error': 'Could not decode audio'}), 400

            processing_time_est_sec = estimate_processing_seconds(duration_sec)

            # Queue ahead of this job: everything waiting plus the job in flight.
            cursor.execute(
                "SELECT COALESCE(SUM(processing_time_est), 0) FROM transcriptions "
                "WHERE status IN ('pending', 'processing')"
            )
            queue_ahead_sec = cursor.fetchone()[0]

            try:
                cursor.execute(
                    "INSERT INTO transcriptions (guid, filename, processing_time_est) VALUES (?, ?, ?)",
                    (guid, file.filename, processing_time_est_sec)
                )
            except sqlite3.IntegrityError:
                # Not reachable through this process (the lock covers the check),
                # but a second process on the same database could race us.
                return jsonify({'error': 'GUID already exists'}), 409

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
               processing_seconds, words_per_second, attempt_count, mfa_applied
        FROM transcriptions WHERE guid = ?
    """, (guid,))
    row = cursor.fetchone()

    if row is None:
        return jsonify({'error': 'GUID not found'}), 404

    (status, transcription, timings, created_at, processing_time_est,
     processing_seconds, words_per_second, attempt_count, mfa_applied) = row

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

    Returns True when a job was processed (the caller should loop straight away)
    and False when the queue was empty (the caller should sleep).
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

    # Fetch the oldest pending transcription
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
        process_pending_job(cursor, guid, filename)
    finally:
        state['busy_since'] = None
    return True


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
