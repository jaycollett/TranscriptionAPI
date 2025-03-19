import os
import uuid
import sqlite3
import threading
import time
import math
import json 
import subprocess
from flask import Flask, request, jsonify
from transcribe import transcribe_audio, load_whisper_model
from pydub import AudioSegment
from datetime import datetime, timedelta
from contextlib import contextmanager
from queue import Queue, Empty, Full
import logging
import shutil
from typing import Optional, Dict, Any, List, Tuple
import mimetypes
from config import config

# Configure logging with concise format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Thread-local storage for connection pools
import threading
thread_local = threading.local()

# Global model instance
model = None

def create_db_connection() -> sqlite3.Connection:
    """Create a new database connection with proper settings."""
    conn = sqlite3.connect(config.DB_FILE, timeout=config.DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    return conn

def get_thread_pool():
    """Get or create a connection pool for the current thread."""
    if not hasattr(thread_local, 'pool'):
        thread_local.pool = Queue(maxsize=config.MAX_DB_CONNECTIONS)
        # Initialize the pool with connections
        for _ in range(config.MAX_DB_CONNECTIONS):
            try:
                conn = create_db_connection()
                thread_local.pool.put(conn, block=False)
            except Full:
                break
    return thread_local.pool

@contextmanager
def get_db_connection():
    """Get a database connection from the current thread's pool with retry logic."""
    pool = get_thread_pool()
    conn = None
    try:
        # Try to get a connection from the pool
        try:
            conn = pool.get(timeout=5)
        except Empty:
            # If no connection is available, create a new one
            conn = create_db_connection()
            
        # Yield the connection
        yield conn
        
    finally:
        # Put the connection back in the pool when done
        if conn:
            try:
                pool.put(conn, timeout=5)
            except Full:
                conn.close()

def init_db():
    """Initialize database schema."""
    try:
        # Create the schema using a direct connection
        conn = sqlite3.connect(config.DB_FILE, timeout=config.DB_TIMEOUT)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    guid TEXT PRIMARY KEY,
                    filename TEXT,
                    status TEXT DEFAULT 'pending',
                    transcription TEXT DEFAULT NULL,
                    timings TEXT DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP DEFAULT NULL,
                    processing_started_at TIMESTAMP DEFAULT NULL,
                    processing_time_est INTEGER DEFAULT 0,
                    error_message TEXT DEFAULT NULL,
                    retry_count INTEGER DEFAULT 0,
                    file_size INTEGER,
                    mime_type TEXT,
                    confidence FLOAT DEFAULT 0.0
                )
            """)
            conn.commit()
            logger.info("Database schema initialized successfully")
        finally:
            conn.close()
        
        # The thread-local connection pools will be initialized on demand
        # when each thread first accesses the database
        logger.info("Thread-local connection pools will be created on demand")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

# Initialize database at startup
init_db()

def check_disk_space(path: str, required_bytes: int) -> bool:
    """Check if there's enough disk space for the file."""
    _, _, free = shutil.disk_usage(path)
    return free > required_bytes * 2  # Require 2x the file size for safety

def validate_audio_file(file_path: str) -> Optional[str]:
    """Validate audio file type and size."""
    if not os.path.exists(file_path):
        return "File not found"
    
    if os.path.getsize(file_path) > config.MAX_FILE_SIZE:
        return f"File too large. Maximum size is {config.MAX_FILE_SIZE/1024/1024}MB"
    
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type not in config.ALLOWED_AUDIO_TYPES:
        return f"Invalid file type. Allowed types: {', '.join(config.ALLOWED_AUDIO_TYPES)}"
    
    try:
        audio = AudioSegment.from_file(file_path)
        if len(audio) == 0:
            return "Empty audio file"
    except Exception as e:
        return f"Invalid audio file: {str(e)}"
    
    return None

def cleanup_resources(file_path: str, guid: str):
    """Clean up temporary files and resources."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        aligned_dir = os.path.join(config.UPLOAD_FOLDER, f"{guid}_aligned")
        if os.path.exists(aligned_dir):
            shutil.rmtree(aligned_dir)
        transcript_path = os.path.join(config.UPLOAD_FOLDER, f"{guid}.txt")
        if os.path.exists(transcript_path):
            os.remove(transcript_path)
    except Exception as e:
        logger.error(f"Error during cleanup for {guid}: {e}")

def update_transcription_status(conn: sqlite3.Connection, guid: str, status: str, 
                              error_message: Optional[str] = None, 
                              transcription: Optional[str] = None,
                              timings: Optional[str] = None) -> None:
    """Update transcription status with proper error handling."""
    try:
        cursor = conn.cursor()
        if status == 'error':
            cursor.execute("""
                UPDATE transcriptions 
                SET status = ?, error_message = ?, retry_count = retry_count + 1
                WHERE guid = ?
            """, (status, error_message, guid))
        elif status == 'processed':
            cursor.execute("""
                UPDATE transcriptions 
                SET status = ?, transcription = ?, timings = ?, 
                    completed_at = CURRENT_TIMESTAMP, error_message = NULL
                WHERE guid = ?
            """, (status, transcription, timings, guid))
        else:
            cursor.execute("""
                UPDATE transcriptions 
                SET status = ?, error_message = NULL
                WHERE guid = ?
            """, (status, guid))
        conn.commit()
    except Exception as e:
        logger.error(f"Error updating status for {guid}: {e}")
        conn.rollback()
        raise

def run_forced_alignment(audio_path, whisper_segments, guid):
    """
    Runs Montreal Forced Aligner (MFA) to refine the timestamps from the Whisper transcript,
    ensuring they align with Whisper's segment structure.
    """
    upload_folder = config.UPLOAD_FOLDER
    transcript_path = os.path.join(upload_folder, f"{guid}.txt")
    aligned_output_dir = os.path.join(upload_folder, f"{guid}_aligned")
    alignment_json_path = os.path.join(aligned_output_dir, f"{guid}.json")

    # If alignment already exists, avoid re-running MFA
    if os.path.exists(alignment_json_path):
        logger.info(f"MFA alignment exists for {guid}, skipping")
    else:
        try:
            # Save transcript to a file for MFA (remove extra spaces)
            with open(transcript_path, "w") as f:
                f.write(" ".join(seg["text"].strip() for seg in whisper_segments if seg["text"].strip()))

            # Ensure output directory exists
            os.makedirs(aligned_output_dir, exist_ok=True)

            # Run MFA command
            mfa_command = [
                "mfa", "align",
                upload_folder,
                "/mfa/pretrained_models/dictionary/english_mfa.dict",
                "english_mfa",
                aligned_output_dir,
                "--output_format", "json"
            ]

            result = subprocess.run(mfa_command, check=True, capture_output=True, text=True)

            if not os.path.exists(alignment_json_path):
                logger.error(f"MFA output file not found: {alignment_json_path}")
                return whisper_segments

        except subprocess.CalledProcessError as e:
            logger.error(f"MFA alignment failed: {e.stderr}")
            return whisper_segments

        except Exception as e:
            logger.error(f"Unexpected error running MFA: {str(e)}")
            return whisper_segments

    # Read MFA output file
    try:
        with open(alignment_json_path, "r") as f:
            alignment_data = json.load(f)

        # Ensure MFA output format is valid
        if "tiers" not in alignment_data or "words" not in alignment_data["tiers"]:
            logger.error(f"Words tier missing in MFA output")
            return whisper_segments

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
                refined_start, refined_end = whisper_start, whisper_end

            refined_segments.append({
                "start": refined_start,
                "end": refined_end,
                "text": segment_text
            })

        return refined_segments

    except Exception as e:
        logger.error(f"Error processing MFA output: {str(e)}")
        return whisper_segments

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check database connection using the thread-local pool
        with get_db_connection() as conn:
            conn.execute("SELECT 1")
        
        # Check if model is loaded
        if model is None:
            return jsonify({"status": "error", "message": "Whisper model not loaded"}), 500
        
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_audio():
    guid = None
    file_path = None
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
            uuid.UUID(guid, version=4)
        except ValueError:
            return jsonify({'error': 'Invalid GUID format. Must be a valid UUID v4'}), 400

        # Check if GUID already exists in the database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT guid FROM transcriptions WHERE guid = ?", (guid,))
            if cursor.fetchone():
                return jsonify({'error': 'GUID already exists'}), 409

        # Save file with the provided GUID
        file_extension = os.path.splitext(file.filename)[-1]
        saved_filename = f"{guid}{file_extension}"
        file_path = os.path.join(config.UPLOAD_FOLDER, saved_filename)

        # Check disk space before saving
        if not check_disk_space(config.UPLOAD_FOLDER, config.MAX_FILE_SIZE):
            return jsonify({'error': 'Insufficient disk space'}), 507

        file.save(file_path)

        # Validate audio file
        error = validate_audio_file(file_path)
        if error:
            cleanup_resources(file_path, guid)
            return jsonify({'error': error}), 400

        # Get file metadata
        file_size = os.path.getsize(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)

        # Analyze the audio file to get duration
        audio = AudioSegment.from_file(file_path)
        duration_sec = len(audio) / 1000

        # Calculate base processing time based on duration
        base_processing_time = duration_sec / config.PROCESSING_SPEED_FACTOR

        # Adjust processing time based on file size (larger files may take longer)
        size_factor = min(1.5, max(1.0, file_size / (100 * 1024 * 1024)))  # 100MB as baseline
        adjusted_processing_time = base_processing_time * size_factor

        # Add overhead for multiple passes and MFA
        # We have 4 passes (high_accuracy, balanced, aggressive, higherbalance)
        # Plus potential retry pass (0.5x)
        # Plus MFA processing (0.5x)
        passes_factor = 5.0  # 4 passes + 0.5 retry + 0.5 MFA
        final_processing_time = adjusted_processing_time * passes_factor

        # Calculate queue position and estimated completion
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get current processing jobs
            cursor.execute("""
                SELECT COUNT(*) as processing_count
                FROM transcriptions 
                WHERE status = 'processing'
            """)
            processing_count = cursor.fetchone()['processing_count']

            # Get pending jobs with their estimated times
            cursor.execute("""
                SELECT processing_time_est
                FROM transcriptions 
                WHERE status = 'pending'
                ORDER BY created_at ASC
            """)
            pending_times = [row['processing_time_est'] for row in cursor.fetchall()]

            # Calculate total queue time
            total_queue_time = sum(pending_times)

            # Add buffer for system overhead and potential delays
            system_overhead = 1.2  # 20% buffer for system overhead
            total_processing_time = final_processing_time * system_overhead

            # Calculate estimated completion time
            estimated_completion_utc = datetime.utcnow() + timedelta(seconds=total_processing_time + total_queue_time)

            # Calculate queue position
            queue_position = len(pending_times) + processing_count

            # Insert into the database with the total processing time (including overhead)
            cursor.execute("""
                INSERT INTO transcriptions 
                (guid, filename, processing_time_est, file_size, mime_type) 
                VALUES (?, ?, ?, ?, ?)
            """, (guid, file.filename, total_processing_time, file_size, mime_type))
            conn.commit()

        # Format times for logging
        processing_time_min = total_processing_time / 60  # Use total_processing_time which includes overhead
        queue_time_min = total_queue_time / 60
        total_time_min = processing_time_min + queue_time_min

        logger.info(
            f"File {file.filename} saved as {saved_filename} with GUID {guid}.\n"
            f"  Processing time: {processing_time_min:.1f} min\n"
            f"  Queue position: {queue_position}\n"
            f"  Queue time: {queue_time_min:.1f} min\n"
            f"  Total time: {total_time_min:.1f} min\n"
            f"  Estimated completion: {estimated_completion_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

        return jsonify({
            'message': 'File uploaded successfully',
            'guid': guid,
            'queue_position': queue_position,
            'processing_time_minutes': round(processing_time_min, 1),
            'queue_time_minutes': round(queue_time_min, 1),
            'total_time_minutes': round(total_time_min, 1),
            'estimated_completion_utc': estimated_completion_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
        }), 201

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        if guid and file_path:
            cleanup_resources(file_path, guid)
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/status/<guid>', methods=['GET'])
def get_status(guid):
    try:
        # Use the thread-local connection pool
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT status, error_message, transcription, timings, 
                       created_at, completed_at, processing_time_est,
                       processing_started_at, confidence
                FROM transcriptions 
                WHERE guid = ?
            """, (guid,))
            result = cursor.fetchone()
            
            if not result:
                return jsonify({'error': 'GUID not found'}), 404
            
            status, error_message, transcription, timings, created_at, completed_at, \
            processing_time_est, processing_started_at, confidence = result
            
            response = {
                'status': status,
                'created_at': created_at,
                'processing_time_est': processing_time_est,
                'confidence': confidence
            }
            
            if error_message:
                response['error_message'] = error_message
            
            if completed_at:
                response['completed_at'] = completed_at
                
            if processing_started_at:
                response['processing_started_at'] = processing_started_at
            
            if transcription:
                response['transcription'] = transcription
                if timings:
                    try:
                        response['timings'] = json.loads(timings)
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding timings JSON for {guid}")
                        response['timings'] = None
            
            return jsonify(response)
            
    except Exception as e:
        logger.error(f"Error fetching status for {guid}: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

def cleanup_old_transcriptions() -> Tuple[int, int]:
    """
    Clean up old transcriptions and their associated files.
    Returns tuple of (files_deleted, records_deleted).
    """
    files_deleted = 0
    records_deleted = 0
    
    try:
        # Use the thread-local connection pool
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get old records in batches
            while True:
                cursor.execute("""
                    SELECT guid, filename 
                    FROM transcriptions 
                    WHERE status = 'processed' 
                    AND created_at <= datetime('now', ?)
                    LIMIT ?
                """, (f'-{config.CLEANUP_AGE_DAYS} days', config.CLEANUP_BATCH_SIZE))
                
                old_records = cursor.fetchall()
                if not old_records:
                    break
                
                for old_guid, old_filename in old_records:
                    try:
                        # Clean up all associated files
                        cleanup_resources(
                            os.path.join(config.UPLOAD_FOLDER, f"{old_guid}{os.path.splitext(old_filename)[-1]}"),
                            old_guid
                        )
                        files_deleted += 1
                        logger.info(f"Deleted old file: {old_filename}")
                    except Exception as e:
                        logger.error(f"Error deleting file {old_filename}: {e}")
                
                # Delete the records
                cursor.execute("""
                    DELETE FROM transcriptions 
                    WHERE guid IN ({})
                """.format(','.join('?' * len(old_records))), [r[0] for r in old_records])
                
                records_deleted += len(old_records)
                conn.commit()
                
                # Log progress
                logger.info(f"Cleanup progress: {records_deleted} records processed")
                
                # Small delay to prevent database lock contention
                time.sleep(0.1)
            
            if records_deleted > 0:
                logger.info(f"Cleanup completed: {files_deleted} files, {records_deleted} records deleted")
            
            return files_deleted, records_deleted
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return files_deleted, records_deleted

def transcription_worker():
    """Worker thread to process pending transcriptions."""
    global model
    logger.info("Transcription worker started")
    
    while True:
        try:
            logger.info("Checking for pending transcriptions...")
            
            # Use thread-local connection pool
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get pending transcription with all relevant fields
                cursor.execute("""
                    SELECT guid, filename, processing_time_est, file_size, mime_type,
                           retry_count, error_message
                    FROM transcriptions 
                    WHERE status = 'pending'
                    ORDER BY created_at ASC
                    LIMIT 1
                """)
                pending = cursor.fetchone()
                
                if pending:
                    guid, filename, processing_time_est, file_size, mime_type, \
                    retry_count, error_message = pending
                    
                    logger.info(f"Found 1 pending transcription")
                    logger.info(f"Processing: {filename} (GUID: {guid})")
                    
                    # Update status to processing
                    cursor.execute("""
                        UPDATE transcriptions 
                        SET status = 'processing', 
                            processing_started_at = CURRENT_TIMESTAMP,
                            error_message = NULL  -- Clear any previous errors
                        WHERE guid = ?
                    """, (guid,))
                    conn.commit()
                    
                    try:
                        # Process the transcription
                        audio_path = os.path.join(config.UPLOAD_FOLDER, filename)
                        result = transcribe_audio(audio_path, guid, model)
                        
                        # Update status to completed with all fields
                        cursor.execute("""
                            UPDATE transcriptions 
                            SET status = 'completed',
                                completed_at = CURRENT_TIMESTAMP,
                                transcription = ?,
                                timings = ?,
                                confidence = ?,
                                error_message = NULL,
                                retry_count = 0
                            WHERE guid = ?
                        """, (
                            result["transcription"],
                            json.dumps(result["timings"]),
                            result.get("confidence", 0.0),
                            guid
                        ))
                        conn.commit()
                        
                        logger.info(f"Transcription completed for {filename}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {guid}: {str(e)}")
                        
                        # Update status to failed with all fields
                        cursor.execute("""
                            UPDATE transcriptions 
                            SET status = 'failed',
                                error_message = ?,
                                completed_at = CURRENT_TIMESTAMP,
                                retry_count = retry_count + 1,
                                transcription = NULL,
                                timings = NULL,
                                confidence = 0.0
                            WHERE guid = ?
                        """, (str(e), guid))
                        conn.commit()
                        
                        # Clean up any temporary files
                        try:
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                            transcript_path = os.path.join(config.UPLOAD_FOLDER, f"{guid}.txt")
                            if os.path.exists(transcript_path):
                                os.remove(transcript_path)
                        except Exception as cleanup_error:
                            logger.error(f"Error during cleanup for {guid}: {str(cleanup_error)}")
                else:
                    logger.info("No pending transcriptions found")
            
            # Wait before checking again
            time.sleep(15)  # Increased from 5 to 15 seconds
            
        except Exception as e:
            logger.error(f"Error in transcription worker: {str(e)}")
            time.sleep(15)  # Also increased error retry delay to 15 seconds

def cleanup_worker():
    """Dedicated worker thread for cleaning up old transcriptions."""
    logger.info("Cleanup worker started")
    
    while True:
        try:
            # Simply call the cleanup function which now handles its own connection
            files_deleted, records_deleted = cleanup_old_transcriptions()
            if records_deleted > 0:
                logger.info(f"Cleanup worker processed {files_deleted} files, {records_deleted} records")
            
            # Sleep until next cleanup cycle
            time.sleep(config.CLEANUP_INTERVAL)
            
        except Exception as e:
            logger.error(f"Cleanup worker error: {e}")
            time.sleep(60)  # Wait on error before retrying

if __name__ == "__main__":
    # Load model at startup
    logger.info("Loading Whisper model...")
    try:
        model = load_whisper_model()  # Load model once at startup
        logger.info("Whisper model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load Whisper model at startup: {str(e)}")
        raise
    
    # Start worker threads
    logger.info("Starting worker threads")
    transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    
    transcription_thread.start()
    cleanup_thread.start()
    
    logger.info("Worker threads started")
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.DEBUG, use_reloader=False)
