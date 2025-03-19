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


# Configure Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER", "/tmp/audio_files")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db_file = 'transcriptions.db'

# Ensure database file exists before initializing
if not os.path.exists(db_file):
    open(db_file, 'w').close()

# Initialize SQLite database with timings column included in the CREATE TABLE statement
def init_db():
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                guid TEXT PRIMARY KEY,
                filename TEXT,
                status TEXT DEFAULT 'pending',
                transcription TEXT DEFAULT NULL,
                timings TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP DEFAULT NULL,
                processing_time_est INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
init_db()

def run_forced_alignment(audio_path, whisper_segments, guid):
    """
    Runs Montreal Forced Aligner (MFA) to refine the timestamps from the Whisper transcript,
    ensuring they align with Whisper's segment structure.
    """
    upload_folder = app.config['UPLOAD_FOLDER']
    transcript_path = os.path.join(upload_folder, f"{guid}.txt")
    aligned_output_dir = os.path.join(upload_folder, f"{guid}_aligned")
    alignment_json_path = os.path.join(aligned_output_dir, f"{guid}.json")

    # If alignment already exists, avoid re-running MFA
    if os.path.exists(alignment_json_path):
        app.logger.info(f"✅ MFA alignment already exists for {guid}. Skipping re-run.")
    else:
        try:
            # Save transcript to a file for MFA (remove extra spaces)
            with open(transcript_path, "w") as f:
                f.write(" ".join(seg["text"].strip() for seg in whisper_segments if seg["text"].strip()))  # Remove extra spaces

            # Ensure output directory exists
            os.makedirs(aligned_output_dir, exist_ok=True)

            # Run MFA command
            mfa_command = [
                "mfa", "align",
                upload_folder,  # Directory containing audio & transcript
                "/mfa/pretrained_models/dictionary/english_mfa.dict",  # Pronunciation dictionary
                "english_mfa",  # Acoustic model
                aligned_output_dir,  # Output directory
                "--output_format", "json"
            ]

            result = subprocess.run(mfa_command, check=True, capture_output=True, text=True)

            if not os.path.exists(alignment_json_path):
                app.logger.error(f"❌ MFA output file not found: {alignment_json_path}")
                return whisper_segments  # Return original Whisper segments if no output

        except subprocess.CalledProcessError as e:
            app.logger.error(f"❌ MFA alignment failed (subprocess error): {e.stderr}")
            return whisper_segments  # Fallback to Whisper segments

        except Exception as e:
            app.logger.error(f"❌ Unexpected error running MFA: {str(e)}")
            return whisper_segments  # Fallback to Whisper segments

    # Read MFA output file
    try:
        with open(alignment_json_path, "r") as f:
            alignment_data = json.load(f)

        # Ensure MFA output format is valid
        if "tiers" not in alignment_data or "words" not in alignment_data["tiers"]:
            app.logger.error(f"❌ 'words' tier missing in MFA output: {alignment_data}")
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
        app.logger.error(f"❌ Error processing MFA output: {str(e)}")
        return whisper_segments  # Fallback to Whisper segments if JSON parsing fails


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
        psf = 15.1  # Processing speed factor
        processing_time_est_sec = math.ceil(duration_sec / psf) * 3

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
        cursor.execute("SELECT status, transcription, timings, created_at, processing_time_est FROM transcriptions WHERE guid = ?", (guid,))
        row = cursor.fetchone()

        if row is None:
            return jsonify({'error': 'GUID not found'}), 404

        status, transcription, timings, created_at, processing_time_est = row

        if status == 'processed':
            return jsonify({
                'status': 'processed',
                'transcription': transcription or "",
                'timings': json.loads(timings) if timings else []  # Return timings as a list
            }), 200

        # **Include both 'pending' and 'processing' jobs in the queue calculation**
        cursor.execute(
            "SELECT created_at, processing_time_est FROM transcriptions WHERE status IN ('pending', 'processing') ORDER BY created_at ASC"
        )
        processing_queue = cursor.fetchall()

        total_processing_time_sec = 0
        file_found = False
        for job_created_at, job_processing_time in processing_queue:
            job_created_at_dt = datetime.strptime(job_created_at, "%Y-%m-%d %H:%M:%S")

            # Sum processing time until we reach the requested file in the queue
            if job_created_at == created_at:
                file_found = True
                break  # Stop adding times once we find the requested file

            total_processing_time_sec += job_processing_time

        # Compute the estimated UTC completion time
        created_at_dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        estimated_completion_utc = created_at_dt + timedelta(seconds=total_processing_time_sec + processing_time_est)

        # **Handle 'processing' status separately**
        if status == 'processing':
            return jsonify({
                'status': 'processing',
                'message': 'Transcription is currently in progress',
                'estimated_completion_utc': estimated_completion_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
            }), 200

        # **Handle 'pending' status**
        if status == 'pending':
            return jsonify({
                'status': 'pending',
                'estimated_completion_utc': estimated_completion_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
            }), 200

        return jsonify({'error': 'Unknown status'}), 500


def transcription_worker():
    """Background worker that processes pending audio files every N seconds."""
    app.logger.info("🟢 Transcription worker started. Checking for pending transcriptions every 15 seconds.")

    while True:
        app.logger.info("⏳ Worker sleeping for 30 seconds...")
        time.sleep(30)
        app.logger.info("🔍 Worker waking up to check for pending transcriptions...")

        try:
            with sqlite3.connect(db_file) as conn:
                cursor = conn.cursor()

                # Fetch pending transcriptions
                cursor.execute("SELECT guid, filename FROM transcriptions WHERE status = 'pending'")
                records = cursor.fetchall()

                if not records:
                    app.logger.info("❌ No pending transcriptions found. Worker going back to sleep.")
                    continue

                app.logger.info(f"📌 Found {len(records)} pending transcriptions to process.")

                for guid, filename in records:
                    file_path = os.path.join("/tmp/audio_files", f"{guid}{os.path.splitext(filename)[-1]}")

                    if not os.path.exists(file_path):
                        app.logger.error(f"🚨 File {file_path} not found. Skipping.")
                        continue

                    # **Mark transcription as 'processing' to prevent duplicate execution**
                    cursor.execute("UPDATE transcriptions SET status = 'processing' WHERE guid = ?", (guid,))
                    conn.commit()

                    app.logger.info(f"🎙️ Processing transcription for {filename} (GUID: {guid})...")

                    try:
                        # **Step 1: Transcribe with Whisper**
                        result = transcribe_audio(file_path, guid)
                        transcription = result["transcription"]
                        whisper_segment_timings = result["timings"]
                    except Exception as e:
                        app.logger.error(f"❌ Whisper transcription failed for {guid}: {e}")
                        continue

                    # **Step 2: Run Forced Alignment (MFA) if necessary**
                    app.logger.info(f"🎯 Running forced alignment for {filename} (GUID: {guid})...")
                    refined_timings = run_forced_alignment(file_path, whisper_segment_timings, guid)

                    if refined_timings is None:
                        app.logger.error(f"❌ Forced alignment failed for {guid}. Falling back to Whisper's timings.")
                        refined_timings = whisper_segment_timings

                    # **Step 3: Update database with refined timings**
                    cursor.execute(
                        "UPDATE transcriptions SET transcription = ?, timings = ?, status = 'processed', completed_at = CURRENT_TIMESTAMP WHERE guid = ?",
                        (transcription, json.dumps(refined_timings), guid)
                    )
                    conn.commit()
                    app.logger.info(f"✅ Transcription completed for {filename} (GUID: {guid})")

                # **Cleanup transcriptions older than 24 hours**
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
    # Load the Whisper model once at startup
    model = load_whisper_model()
    app.logger.info("🔥 Starting transcription worker thread...")
    worker_thread = threading.Thread(target=transcription_worker, daemon=True)
    worker_thread.start()
    app.logger.info("✅ Transcription worker thread started successfully.")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
