import os
import uuid
import sqlite3
import threading
import time
from flask import Flask, request, jsonify
from transcribe import transcribe_audio, load_whisper_model
from pydub import AudioSegment
from datetime import datetime, timedelta

# Configure Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/audio_files'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db_file = 'transcriptions.db'

# Ensure database file exists before initializing
if not os.path.exists(db_file):
    open(db_file, 'w').close()

# Load the Whisper model once at startup
app.logger.info("🔄 Loading Whisper model...")
model = load_whisper_model()
app.logger.info("✅ Whisper model loaded successfully.")

# Initialize SQLite database
def init_db():
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                guid TEXT PRIMARY KEY,
                filename TEXT,
                status TEXT DEFAULT 'pending',
                transcription TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP DEFAULT NULL,
                processing_time_est INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
init_db()

@app.route('/transcriptions', methods=['GET'])
def get_all_transcriptions():
    """Returns all transcriptions with status, GUID, submission, completion timestamps, and estimated processing time."""
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT guid, filename, status, created_at, completed_at, processing_time_est FROM transcriptions")
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
        with sqlite3.connect(db_file) as conn:
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
        duration_min = duration_sec / 60  # Convert seconds to minutes

        # **Calculate estimated processing time**
        processing_time_est = (duration_min * 2) * 1.03  # +3% buffer
        processing_time_est_sec = processing_time_est * 60  # Convert to seconds

        # **Check pending transcriptions and sum up processing times**
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT processing_time_est FROM transcriptions WHERE status = 'pending'")
            pending_times = [row[0] for row in cursor.fetchall()]

        total_processing_time_sec = sum(pending_times) + processing_time_est_sec
        estimated_completion_utc = datetime.utcnow() + timedelta(seconds=total_processing_time_sec)

        # **Insert into the database**
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO transcriptions (guid, filename, processing_time_est) VALUES (?, ?, ?)",
                (guid, file.filename, processing_time_est_sec)
            )
            conn.commit()

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

    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500



@app.route('/status/<guid>', methods=['GET'])
def get_transcription(guid):
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()

        # Retrieve the requested transcription details
        cursor.execute("SELECT status, transcription, created_at, processing_time_est FROM transcriptions WHERE guid = ?", (guid,))
        row = cursor.fetchone()

        if row is None:
            return jsonify({'error': 'GUID not found'}), 404

        status, transcription, created_at, processing_time_est = row

        if status == 'processed':
            return jsonify({
                'status': 'processed',
                'transcription': transcription
            }), 200

        # If the file is still pending, calculate the estimated completion time based on the queue
        cursor.execute(
            "SELECT created_at, processing_time_est FROM transcriptions WHERE status = 'pending' ORDER BY created_at ASC"
        )
        pending_jobs = cursor.fetchall()

        total_processing_time_sec = 0
        file_found = False
        for job_created_at, job_processing_time in pending_jobs:
            job_created_at_dt = datetime.strptime(job_created_at, "%Y-%m-%d %H:%M:%S")

            # Sum processing time until we reach the requested file in the queue
            if job_created_at == created_at:
                file_found = True
                break  # Stop adding times once we find the requested file

            total_processing_time_sec += job_processing_time

        if not file_found:
            return jsonify({'error': 'Unexpected error: GUID found in DB but not in pending queue'}), 500

        # Compute the new estimated UTC completion time
        created_at_dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        estimated_completion_utc = created_at_dt + timedelta(seconds=total_processing_time_sec + processing_time_est)

        return jsonify({
            'status': 'pending',
            'estimated_completion_utc': estimated_completion_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
        }), 200


def transcription_worker():
    """ Background worker that processes pending audio files every 30 seconds. """
    app.logger.info("🟢 Transcription worker started. Checking for pending transcriptions every 30 seconds.")

    while True:
        app.logger.info("⏳ Worker sleeping for 30 seconds...")
        time.sleep(30)
        app.logger.info("🔍 Worker waking up to check for pending transcriptions...")

        try:
            with sqlite3.connect(db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT guid, filename FROM transcriptions WHERE status = 'pending'")
                records = cursor.fetchall()

                if not records:
                    app.logger.info("❌ No pending transcriptions found. Worker going back to sleep.")
                else:
                    app.logger.info(f"📌 Found {len(records)} pending transcriptions to process.")
                
                for guid, filename in records:
                    file_path = os.path.join("/tmp/audio_files", f"{guid}{os.path.splitext(filename)[-1]}")

                    if not os.path.exists(file_path):
                        app.logger.error(f"🚨 File {file_path} not found. Skipping.")
                        continue

                    app.logger.info(f"🎙️ Processing transcription for {filename} (GUID: {guid})...")
                    transcription = transcribe_audio(file_path, guid)

                    cursor.execute("UPDATE transcriptions SET transcription = ?, status = 'processed', completed_at = CURRENT_TIMESTAMP WHERE guid = ?", (transcription, guid))
                    conn.commit()
                    app.logger.info(f"✅ Transcription completed for {filename} (GUID: {guid})")

                # Cleanup transcriptions older than 24 hours
                cursor.execute("SELECT guid, filename FROM transcriptions WHERE status = 'processed' AND created_at <= datetime('now', '-1 day')")
                old_records = cursor.fetchall()
                
                for guid, filename in old_records:
                    file_path = os.path.join("/tmp/audio_files", f"{guid}{os.path.splitext(filename)[-1]}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        app.logger.info(f"🗑️ Deleted old audio file {file_path}")
                
                cursor.execute("DELETE FROM transcriptions WHERE status = 'processed' AND created_at <= datetime('now', '-1 day')")
                conn.commit()
                app.logger.info("🧹 Old processed transcriptions deleted")

        except Exception as e:
            app.logger.error(f"❌ Worker error: {e}")

if __name__ == "__main__":
    app.logger.info("🔥 Starting transcription worker thread...")
    worker_thread = threading.Thread(target=transcription_worker, daemon=True)
    worker_thread.start()
    app.logger.info("✅ Transcription worker thread started successfully.")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
