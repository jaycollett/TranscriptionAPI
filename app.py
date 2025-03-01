import os
import uuid
import sqlite3
import threading
import time
from flask import Flask, request, jsonify
from transcribe import transcribe_audio, load_whisper_model

# Configure Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/audio_files'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db_file = 'transcriptions.db'

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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
init_db()

@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        guid = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[-1]
        saved_filename = f"{guid}{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        
        file.save(file_path)  # Save the uploaded file
        
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO transcriptions (guid, filename) VALUES (?, ?)", (guid, file.filename))
            conn.commit()
        
        app.logger.info(f"File {file.filename} received and saved as {saved_filename} with GUID {guid}")
        return jsonify({'guid': guid}), 200
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/transcription/<guid>', methods=['GET'])
def get_transcription(guid):
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, transcription FROM transcriptions WHERE guid = ?", (guid,))
        row = cursor.fetchone()

        if row is None:
            return jsonify({'error': 'GUID not found'}), 404

        status, transcription = row
        if status == 'pending':
            return jsonify({'status': 'pending'}), 200
        else:
            return jsonify({'status': 'processed', 'transcription': transcription}), 200

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
                    transcription = transcribe_audio(file_path)

                    cursor.execute("UPDATE transcriptions SET transcription = ?, status = 'processed' WHERE guid = ?", (transcription, guid))
                    conn.commit()
                    app.logger.info(f"✅ Transcription completed for {filename} (GUID: {guid})")

                # Cleanup transcriptions older than 24 hours
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
