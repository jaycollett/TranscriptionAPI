import os
import uuid
import sqlite3
import threading  # For background processing and thread-local DB connections
import time
import math
import json
import shutil
import subprocess
from flask import Flask, request, jsonify  # Web framework and request handling
from transcribe import transcribe_audio, load_whisper_model  # Custom transcription logic
from pydub import AudioSegment  # Audio file manipulation
from datetime import datetime, timedelta, timezone
import logging  # Logging for debugging and monitoring

# Optional import for speaker diarization (graceful fallback if not available)
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.getLogger(__name__).warning("pyannote.audio not available - speaker diarization will be disabled")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER", "/tmp/audio_files")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database configuration

db_file = os.getenv("DB_FILE", "transcriptions.db")
db_connection_timeout = 30  # Connection timeout in seconds

# Maximum number of consecutive garbage transcription results before a job is
# quarantined (moved to a terminal state) instead of being requeued. This keeps a
# single bad sermon from permanently blocking the head of the FIFO queue.
max_garbage_retries = int(os.getenv("MAX_GARBAGE_RETRIES", "3"))

# Ensure database file exists before initializing
if not os.path.exists(db_file):
    open(db_file, 'w').close()

# Thread-local storage for database connections
# This ensures each thread gets its own dedicated connection
local_storage = threading.local()

def get_db_connection():
    # Establish a thread-local SQLite connection with autocommit and cross-thread access
    """
    Returns a SQLite database connection with proper timeout settings.
    Uses thread-local storage to ensure each thread gets its own connection.
    """
    if not hasattr(local_storage, 'connection'):
        local_storage.connection = sqlite3.connect(
            db_file, 
            timeout=db_connection_timeout,
            isolation_level=None,  # Use autocommit mode
            check_same_thread=False  # Allow connection created in one thread to be used in another
        )
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
            attempt_count INTEGER DEFAULT 0
        )
    ''')

    # Create index for status to optimize queue queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON transcriptions(status)')

    # Create index for created_at to optimize cleanup queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON transcriptions(created_at)')

    # Migration: add attempt_count to pre-existing databases (defaults existing rows to 0)
    if not column_exists(cursor, 'transcriptions', 'attempt_count'):
        cursor.execute('ALTER TABLE transcriptions ADD COLUMN attempt_count INTEGER DEFAULT 0')
        app.logger.info("Migrated transcriptions table: added 'attempt_count' column (existing rows default to 0)")

def init_db():
    # Initialize the SQLite database schema and performance settings
    """Initialize the database with necessary tables."""
    # Use a temporary connection specifically for initialization to avoid impacting thread-local storage
    with sqlite3.connect(db_file, timeout=db_connection_timeout) as conn:
        cursor = conn.cursor()

        # Create/migrate schema (tables, indexes, additive columns)
        ensure_schema(cursor)

        # Enable performance optimizations
        cursor.execute('PRAGMA journal_mode = WAL')  # Use Write-Ahead Logging for better concurrency
        cursor.execute('PRAGMA synchronous = NORMAL')  # Slightly less durable but better performance
        
        # Enable foreign key constraints if we add related tables later
        cursor.execute('PRAGMA foreign_keys = ON')
        
        # Set busy timeout to handle database locks
        cursor.execute(f'PRAGMA busy_timeout = {db_connection_timeout * 1000}')
        
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

def handle_garbage_result(cursor, guid, reason):
    # Count a garbage result and either requeue the job or quarantine it
    """Increment the job's attempt counter, then requeue or quarantine it.

    After max_garbage_retries consecutive garbage results the job is moved to the
    terminal 'quarantined' status so it stops being the oldest 'pending' row and the
    FIFO queue can advance to the next sermon. Returns the resulting status string
    ('pending' or 'quarantined').
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
            "UPDATE transcriptions SET status = 'quarantined', completed_at = CURRENT_TIMESTAMP WHERE guid = ?",
            (guid,)
        )
        app.logger.error(
            f"Quarantining transcription {guid} after {attempt_count} garbage result(s) "
            f"({reason}). Moving to terminal 'quarantined' status so the queue can advance."
        )
        return 'quarantined'

    cursor.execute(
        "UPDATE transcriptions SET status = 'pending', transcription = NULL, timings = NULL WHERE guid = ?",
        (guid,)
    )
    app.logger.warning(
        f"Garbage transcription detected for {guid} ({reason}). Resetting to 'pending' "
        f"(attempt {attempt_count} of {max_garbage_retries})."
    )
    return 'pending'

def recover_stuck_jobs(cursor):
    # Reset orphaned in-flight jobs left in 'processing' (e.g. after a container restart)
    """Reset any rows stuck in 'processing' back to 'pending' and return the count.

    The worker only ever selects 'pending' rows, so a job that was mid-transcription
    when the process died would otherwise be stranded forever. Recovery is NOT counted
    as a garbage retry (attempt_count is left untouched) since the job never produced a
    result.
    """
    cursor.execute("SELECT COUNT(*) FROM transcriptions WHERE status = 'processing'")
    stuck = cursor.fetchone()[0]
    if stuck:
        cursor.execute("UPDATE transcriptions SET status = 'pending' WHERE status = 'processing'")
        app.logger.warning(
            f"Startup recovery: reset {stuck} orphaned job(s) stuck in 'processing' back to 'pending'."
        )
    else:
        app.logger.info("Startup recovery: no orphaned 'processing' jobs found.")
    return stuck

def detect_speakers(audio_path):
    """
    Use pyannote-audio to detect the number of unique speakers in an audio file.
    Returns the number of speakers detected.
    """
    if not PYANNOTE_AVAILABLE:
        app.logger.warning("pyannote-audio not available, assuming single speaker")
        return 1
    
    try:
        # Load the speaker diarization pipeline from local models
        # The model is downloaded to cache_dir during build, so use that location
        local_model_cache = "/app/models/pyannote"
        if os.path.exists(local_model_cache):
            app.logger.info(f"Loading pyannote model from local cache: {local_model_cache}")
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", cache_dir=local_model_cache)
        else:
            app.logger.warning("Local pyannote model cache not found, falling back to downloading from HuggingFace")
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        
        # Run speaker diarization
        diarization = pipeline(audio_path)
        
        # Get unique speakers
        speakers = set()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
        
        num_speakers = len(speakers)
        app.logger.info(f"Speaker diarization detected {num_speakers} unique speakers in {audio_path}")
        return num_speakers
        
    except Exception as e:
        app.logger.warning(f"Speaker diarization failed: {e}, assuming single speaker")
        return 1  # Assume single speaker if detection fails

def run_forced_alignment(audio_path, whisper_segments, guid):
    # Use Montreal Forced Aligner to refine Whisper's segment timings
    """
    Runs Montreal Forced Aligner (MFA) to refine the timestamps from the Whisper transcript,
    ensuring they align with Whisper's segment structure.
    """
    upload_folder = app.config['UPLOAD_FOLDER']
    transcript_path = os.path.join(upload_folder, f"{guid}.txt")
    aligned_output_dir = os.path.join(upload_folder, f"{guid}_aligned")
    alignment_json_path = os.path.join(aligned_output_dir, f"{guid}.json")

    # Check for multi-speaker content before running/checking MFA
    # Use proper speaker diarization to detect multiple speakers
    num_speakers = detect_speakers(audio_path)
    
    if num_speakers > 1:
        app.logger.warning(f"Multi-speaker content detected ({num_speakers} speakers). Skipping MFA alignment to maintain accuracy.")
        app.logger.info("Using Whisper segments only for multi-speaker audio to ensure accurate timing alignment")
        return whisper_segments  # Skip MFA entirely for multi-speaker content

    # If alignment already exists, avoid re-running MFA
    if os.path.exists(alignment_json_path):
        app.logger.info(f"MFA alignment already exists for {guid}. Skipping re-run.")
    else:
        try:
            # Verify the actual audio file exists and get its correct path
            if not os.path.exists(audio_path):
                app.logger.error(f"Audio file not found: {audio_path}")
                return whisper_segments
            
            # Save transcript to a file for MFA (remove extra spaces and clean up text)
            transcript_content = " ".join(seg["text"].strip() for seg in whisper_segments if seg["text"].strip())
            
            # Clean transcript for better MFA compatibility
            import re
            # Remove repeated words/phrases that might confuse alignment
            transcript_content = re.sub(r'\b(\w+)\s+\1\s+\1\s+\1+', r'\1', transcript_content)
            
            with open(transcript_path, "w") as f:
                f.write(transcript_content)
            
            # Ensure transcript was written successfully
            if not os.path.exists(transcript_path):
                app.logger.error(f"Failed to create transcript file: {transcript_path}")
                return whisper_segments
            
            # Add a small delay to ensure file is fully written
            time.sleep(0.1)

            # Ensure output directory exists
            os.makedirs(aligned_output_dir, exist_ok=True)
            
            # Create a temporary directory with properly named files for MFA
            temp_mfa_dir = os.path.join(upload_folder, f"{guid}_mfa_input")
            os.makedirs(temp_mfa_dir, exist_ok=True)
            
            # Copy files with consistent naming for MFA
            temp_audio_path = os.path.join(temp_mfa_dir, f"{guid}.wav")
            temp_transcript_path = os.path.join(temp_mfa_dir, f"{guid}.txt")
            
            # Convert audio to WAV format for MFA compatibility
            try:
                audio = AudioSegment.from_file(audio_path)
                audio.export(temp_audio_path, format="wav")
                
                # Copy transcript
                shutil.copy2(transcript_path, temp_transcript_path)
                
                app.logger.info(f"Created MFA input files: {temp_audio_path}, {temp_transcript_path}")
            except Exception as prep_error:
                app.logger.error(f"Error preparing MFA input files: {prep_error}")
                # Cleanup temp directory
                if os.path.exists(temp_mfa_dir):
                    shutil.rmtree(temp_mfa_dir)
                return whisper_segments

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
                    "/mfa/pretrained_models/dictionary/english_mfa.dict",  # Pronunciation dictionary
                    "english_mfa",  # Acoustic model
                    aligned_output_dir, # Output directory
                    "--output_format", "json",
                    "--beam", config["beam"],
                    "--retry_beam", config["retry_beam"]
                ]
                
                try:
                    app.logger.info(f"MFA attempt {i+1} with beam={config['beam']}, retry_beam={config['retry_beam']}")
                    result = subprocess.run(mfa_command, check=True, capture_output=True, text=True, timeout=300)
                    mfa_success = True
                    break
                except subprocess.TimeoutExpired:
                    app.logger.warning(f"MFA attempt {i+1} timed out after 5 minutes")
                    continue
                except subprocess.CalledProcessError as e:
                    app.logger.warning(f"MFA attempt {i+1} failed: {e.stderr}")
                    if i == len(beam_configs) - 1:  # Last attempt
                        raise  # Re-raise the exception to be handled by outer try-catch
                    continue
            
            if not mfa_success:
                raise subprocess.CalledProcessError(1, mfa_command, "All MFA attempts failed")
            
            # Cleanup temporary directory after MFA completes
            try:
                if os.path.exists(temp_mfa_dir):
                    shutil.rmtree(temp_mfa_dir)
            except Exception as cleanup_error:
                app.logger.warning(f"Could not cleanup temp MFA directory: {cleanup_error}")
            
            # Log MFA command output for debugging
            app.logger.info(f"MFA command stdout: {result.stdout}")
            if result.stderr:
                app.logger.warning(f"MFA command stderr: {result.stderr}")
            
            # Check what files were actually created in the output directory
            if os.path.exists(aligned_output_dir):
                output_files = os.listdir(aligned_output_dir)
                app.logger.info(f"Files created in {aligned_output_dir}: {output_files}")
            else:
                app.logger.error(f"Output directory {aligned_output_dir} was not created")
            
            if not os.path.exists(alignment_json_path):
                app.logger.error(f"MFA output file not found: {alignment_json_path}")
                app.logger.error(f"Command used: {' '.join(mfa_command)}")
                return whisper_segments  # Return original Whisper segments if no output

        except subprocess.CalledProcessError as e:
            app.logger.error(f"MFA alignment failed (subprocess error): {e.stderr}")
            app.logger.error(f"Command used: {' '.join(mfa_command)}")
            app.logger.error(f"Return code: {e.returncode}")
            
            # Check if there are error logs in the corpus directory
            log_dir = os.path.join(upload_folder, f"{guid}_corpus/split1/log")
            if os.path.exists(log_dir):
                app.logger.error(f"MFA log directory found at {log_dir}, checking for error logs...")
                try:
                    log_files = os.listdir(log_dir)
                    for log_file in log_files:
                        if "error" in log_file.lower():
                            with open(os.path.join(log_dir, log_file), 'r') as f:
                                app.logger.error(f"Error log content from {log_file}: {f.read()}")
                except Exception as log_error:
                    app.logger.error(f"Error reading MFA logs: {log_error}")
            
            # Cleanup temp directory on error
            try:
                if os.path.exists(temp_mfa_dir):
                    shutil.rmtree(temp_mfa_dir)
            except Exception as cleanup_error:
                app.logger.warning(f"Could not cleanup temp MFA directory after error: {cleanup_error}")
            
            return whisper_segments  # Fallback to Whisper segments

        except Exception as e:
            app.logger.error(f"Unexpected error running MFA: {str(e)}")
            # Cleanup temp directory on error
            try:
                if os.path.exists(temp_mfa_dir):
                    shutil.rmtree(temp_mfa_dir)
            except Exception as cleanup_error:
                app.logger.warning(f"Could not cleanup temp MFA directory after error: {cleanup_error}")
            
            return whisper_segments  # Fallback to Whisper segments

    # Read MFA output file
    try:
        with open(alignment_json_path, "r") as f:
            alignment_data = json.load(f)

        # Ensure MFA output format is valid
        if "tiers" not in alignment_data or "words" not in alignment_data["tiers"]:
            app.logger.error(f"'words' tier missing in MFA output: {alignment_data}")
            return whisper_segments  # Fallback to Whisper's segment timings

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

        return refined_segments  # Return segment-level alignment

    except Exception as e:
        app.logger.error(f"Error processing MFA output: {str(e)}")
        return whisper_segments  # Fallback to Whisper segments if JSON parsing fails


def process_pending_job(cursor, guid, filename):
    # Transcribe a single pending job end-to-end and update its row accordingly
    """Process one pending transcription job and return its resulting status.

    Encapsulates the per-job control flow so it can be unit tested in isolation:
    missing-file handling, the garbage-detection retry/quarantine path, MFA, and the
    successful-completion path (which clears attempt_count). Garbage results are routed
    through handle_garbage_result so a repeatedly bad job is quarantined instead of
    re-blocking the head of the queue.
    """
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{guid}{os.path.splitext(filename)[-1]}")

        if not os.path.exists(file_path):
            app.logger.error(f"File {file_path} not found. Skipping.")
            cursor.execute(
                "UPDATE transcriptions SET status = 'error', completed_at = CURRENT_TIMESTAMP WHERE guid = ?",
                (guid,)
            )
            return 'error'

        # Mark transcription as 'processing' to prevent duplicate execution
        cursor.execute("UPDATE transcriptions SET status = 'processing' WHERE guid = ?", (guid,))

        app.logger.info(f"Processing transcription for {filename} (GUID: {guid})...")

        try:
            # Step 1: Transcribe with Whisper
            result = transcribe_audio(file_path, guid)
            transcription = result["transcription"]
            whisper_segment_timings = result["timings"]

            # Check for garbage transcription
            if is_garbage_transcription(transcription):
                return handle_garbage_result(cursor, guid, "garbage transcription detected")
        except Exception as e:
            app.logger.error(f"Whisper transcription failed for {guid}: {e}")
            cursor.execute(
                "UPDATE transcriptions SET status = 'error', completed_at = CURRENT_TIMESTAMP WHERE guid = ?",
                (guid,)
            )
            return 'error'

        # Step 2: Run Forced Alignment (MFA) if necessary
        app.logger.info(f"Running forced alignment for {filename} (GUID: {guid})...")
        # Check if a processed audio file exists (from noise reduction during
        # transcription). Handle any audio format, not just MP3. Processed files are
        # always written as MP3 by preprocess_audio_for_transcription.
        file_root, _ = os.path.splitext(file_path)
        processed_file_path = f"{file_root}_processed.mp3"
        audio_file_for_mfa = processed_file_path if os.path.exists(processed_file_path) else file_path
        app.logger.info(f"Using audio file for MFA: {audio_file_for_mfa}")
        refined_timings = run_forced_alignment(audio_file_for_mfa, whisper_segment_timings, guid)

        # Garbage check again — just in case MFA corrupted it
        if is_garbage_transcription(transcription) or not refined_timings or all(not seg.get("text") for seg in refined_timings):
            return handle_garbage_result(cursor, guid, "post-alignment transcription/timing looks corrupted")

        if refined_timings is None:
            app.logger.error(f"Forced alignment failed for {guid}. Falling back to Whisper's timings.")
            refined_timings = whisper_segment_timings

        # Step 3: Update database with refined timings. Clear attempt_count so a job that
        # succeeds after earlier garbage retries is never wrongly quarantined later.
        cursor.execute(
            "UPDATE transcriptions SET transcription = ?, timings = ?, status = 'completed', attempt_count = 0, completed_at = CURRENT_TIMESTAMP WHERE guid = ?",
            (transcription, json.dumps(refined_timings), guid)
        )
        app.logger.info(f"Transcription completed for {filename} (GUID: {guid})")
        return 'completed'

    except Exception as e:
        app.logger.error(f"Error processing transcription for {guid}: {e}")
        # Ensure we mark the job as error if anything goes wrong
        cursor.execute(
            "UPDATE transcriptions SET status = 'error', completed_at = CURRENT_TIMESTAMP WHERE guid = ?",
            (guid,)
        )
        return 'error'


@app.route('/transcriptions', methods=['GET'])
# Endpoint to list all transcription jobs with metadata
def get_all_transcriptions():
    """Returns all transcriptions with status, GUID, submission, completion timestamps, and estimated processing time."""
    conn = get_db_connection()
    # No need to close the connection - it's stored in thread-local storage and will be reused
    
    cursor = conn.cursor()
    # Order by created_at to ensure consistent ordering
    cursor.execute("""
        SELECT guid, filename, status, created_at, completed_at, processing_time_est 
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
        'processing_time_est': row[5]  # Processing time estimate in seconds
    } for row in records]

    return jsonify(result), 200


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
        try:
            uuid.UUID(guid, version=4)  # Ensures it's a valid GUID
        except ValueError:
            return jsonify({'error': 'Invalid GUID format. Must be a valid UUID v4'}), 400

        # Check if GUID already exists in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT guid FROM transcriptions WHERE guid = ?", (guid,))
        if cursor.fetchone():
            return jsonify({'error': 'GUID already exists'}), 409  # Conflict status

        # Save file with the provided GUID
        file_extension = os.path.splitext(file.filename)[-1]
        saved_filename = f"{guid}{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(file_path)

        # **Analyze the audio file to get duration**
        audio = AudioSegment.from_file(file_path)
        duration_sec = len(audio) / 1000  # Convert milliseconds to seconds
        psf = 15.1  # Processing speed factor
        processing_time_est_sec = math.ceil(duration_sec / psf) * 5 + 45

        # **Check pending transcriptions and sum up processing times**
        conn = get_db_connection()  # Reuses the same connection from thread-local storage
        cursor = conn.cursor()
        
        # Use a more efficient single query to get sum
        cursor.execute("SELECT SUM(processing_time_est) FROM transcriptions WHERE status = 'pending'")
        result = cursor.fetchone()
        pending_times_sum = result[0] if result[0] is not None else 0

        total_processing_time_sec = pending_times_sum + processing_time_est_sec
        estimated_completion_utc = datetime.now(timezone.utc) + timedelta(seconds=total_processing_time_sec)

        # **Insert into the database** - done in the same connection
        cursor.execute(
            "INSERT INTO transcriptions (guid, filename, processing_time_est) VALUES (?, ?, ?)",
            (guid, file.filename, processing_time_est_sec)
        )

        app.logger.info(
            f"File {file.filename} received and saved as {saved_filename} with GUID {guid}. "
            f"Estimated processing time: {processing_time_est_sec / 60:.2f} min. "
            f"Total queue time: {total_processing_time_sec / 60:.2f} min. "
            f"Check back at {estimated_completion_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

        return jsonify({
            'message': 'File uploaded successfully',
            'guid': guid,
            'estimated_completion_utc': estimated_completion_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
        }), 201  # Created status

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
    conn = get_db_connection()  # Uses thread-local connection
    cursor = conn.cursor()

    # Retrieve the requested transcription details
    cursor.execute("""
        SELECT status, transcription, timings, created_at, processing_time_est 
        FROM transcriptions WHERE guid = ?
    """, (guid,))
    row = cursor.fetchone()

    if row is None:
        return jsonify({'error': 'GUID not found'}), 404

    status, transcription, timings, created_at, processing_time_est = row

    # If already completed, return the results immediately
    if status == 'completed' or status == 'processed':  # Support both new and old status values during transition
        return jsonify({
            'status': 'completed',
            'transcription': transcription or "",
            'timings': json.loads(timings) if timings else []  # Return timings as a list
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
    
    # More efficient single query to get sum of processing times for jobs ahead in queue
    cursor.execute("""
        SELECT SUM(processing_time_est) 
        FROM transcriptions 
        WHERE status IN ('pending', 'processing') 
        AND created_at < ?
    """, (created_at,))
    
    result = cursor.fetchone()
    total_processing_time_sec = result[0] if result[0] is not None else 0
    
    # Add this job's processing time
    estimated_completion_utc = created_at_dt + timedelta(seconds=total_processing_time_sec + processing_time_est)
    formatted_completion_time = estimated_completion_utc.strftime('%Y-%m-%d %H:%M:%S UTC')

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


def transcription_worker():
    # Background thread that polls for pending jobs and processes them
    """Background worker that processes pending audio files every N seconds."""
    app.logger.info("Transcription worker started. Checking for pending transcriptions every 30 seconds.")

    # Crash recovery: before the processing loop begins, requeue any job left in
    # 'processing' by a previous run that died mid-transcription (e.g. a container
    # restart). The worker only selects 'pending' rows, so these would otherwise be
    # stranded forever. This does NOT count as a garbage retry.
    try:
        recovery_conn = get_db_connection()
        recover_stuck_jobs(recovery_conn.cursor())
    except Exception as e:
        app.logger.error(f"Startup recovery failed: {e}")

    while True:
        app.logger.info("Worker sleeping for 30 seconds...")
        time.sleep(30)  # Sleep interval between polling cycles
        app.logger.info("Worker waking up to check for pending transcriptions...")

        try:
            # Get thread-local connection - this ensures the worker thread has its own dedicated connection
            conn = get_db_connection()
            cursor = conn.cursor()

            # Fetch pending transcriptions
            cursor.execute("""
                SELECT guid, filename FROM transcriptions 
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1  -- Process only one file at a time
            """)
            records = cursor.fetchall()

            if not records:
                app.logger.info("No pending transcriptions found. Worker going back to sleep.")
                continue

            app.logger.info(f"Found {len(records)} pending transcriptions to process.")

            # Process one file at a time - the oldest pending first
            for guid, filename in records:
                process_pending_job(cursor, guid, filename)

            # Cleanup transcriptions older than 24 hours (once per hour)
            if int(time.time()) % 3600 < 30:  # Run cleanup roughly every hour
                try:
                    cursor.execute("""
                        SELECT guid, filename FROM transcriptions 
                        WHERE status IN ('completed', 'error', 'quarantined') AND created_at <= datetime('now', '-1 day')
                    LIMIT 20
                    """)
                    old_records = cursor.fetchall()
                    
                    if old_records:
                        for guid, filename in old_records:
                            file_root, file_ext = os.path.splitext(filename)
                            upload_folder = app.config['UPLOAD_FOLDER']

                            paths_to_delete = [
                                os.path.join(upload_folder, f"{guid}{file_ext}"),     # Original audio file
                                os.path.join(upload_folder, f"{guid}_processed.mp3"), # Processed audio file
                                os.path.join(upload_folder, f"{guid}.txt"),           # Transcript used by MFA
                                os.path.join(upload_folder, f"{guid}_aligned"),       # MFA output
                                os.path.join(upload_folder, f"{guid}_corpus"),        # MFA working corpus
                                os.path.join(upload_folder, f"{guid}_mfa_input")      # MFA input temp directory
                            ]

                            for path in paths_to_delete:
                                try:
                                    if os.path.isfile(path):
                                        os.remove(path)
                                        app.logger.info(f"Deleted file: {path}")
                                    elif os.path.isdir(path):
                                        shutil.rmtree(path)
                                        app.logger.info(f"Deleted directory: {path}")
                                except Exception as file_err:
                                    app.logger.error(f"Failed to delete {path}: {file_err}")
                        
                        try:
                            cursor.execute("DELETE FROM transcriptions WHERE status IN ('completed', 'error', 'quarantined') AND created_at <= datetime('now', '-1 day')")
                            app.logger.info("Old completed transcriptions deleted")
                        except Exception as db_cleanup_err:
                            app.logger.error(f"Failed to delete old DB records: {db_cleanup_err}")
                except Exception as cleanup_error:
                    app.logger.error(f"Cleanup error: {cleanup_error}")

        except Exception as e:
            app.logger.error(f"Worker error: {e}")


# Cleanup function to close database connections when app is shutting down
@app.teardown_appcontext
# Flask hook to clean up DB connections after each request
def shutdown_session(exception=None):
    """Ensure thread connections are closed when the app context ends."""
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
    atexit_registered = False
    try:
        import atexit
        atexit.register(cleanup_connections)
        atexit_registered = True
    except ImportError:
        app.logger.warning("Could not import atexit module for cleanup - connections may leak on shutdown")
    
    # Start the worker thread
    app.logger.info("Starting transcription worker thread...")
    worker_thread = threading.Thread(target=transcription_worker, daemon=True)
    worker_thread.start()
    app.logger.info("Transcription worker thread started successfully.")
    
    # Run the app
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=True, 
        use_reloader=False
    )
