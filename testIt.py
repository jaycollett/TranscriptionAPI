import requests
import time
import uuid
import os

# API URLs
API_BASE_URL = "http://localhost:5032"
UPLOAD_URL = f"{API_BASE_URL}/upload"
STATUS_URL = f"{API_BASE_URL}/status"

# File Path
AUDIO_FILE_PATH = "/home/jay/Downloads/simple.mp3"

# Generate a unique GUID for this job
guid = str(uuid.uuid4())

def upload_audio():
    """Uploads the audio file to the API for processing."""
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"❌ Error: Audio file not found at {AUDIO_FILE_PATH}")
        return None

    with open(AUDIO_FILE_PATH, 'rb') as file:
        files = {'file': file}
        data = {'guid': guid}
        response = requests.post(UPLOAD_URL, files=files, data=data)

    if response.status_code == 201:
        response_json = response.json()
        print(f"✅ Upload successful! GUID: {response_json['guid']}")
        print(f"🔄 Estimated completion time: {response_json['estimated_completion_utc']}")
        return response_json['guid']
    else:
        try:
            error = response.json()
        except Exception:
            error = response.text
        print(f"❌ Upload failed: {error}")
        return None

def check_transcription():
    """Checks the transcription status until it's completed."""
    while True:
        response = requests.get(f"{STATUS_URL}/{guid}")

        if response.status_code == 200:
            response_json = response.json()
            status = response_json['status']

            if status == 'pending':
                print(f"⏳ Still processing... Estimated completion: {response_json.get('estimated_completion_utc', 'Unknown')}")
            elif status == 'processing':
                print("🚀 Transcription is currently in progress...")
            elif status == 'processed':
                print("✅ Transcription complete!")
                print(f"📝 Transcription:\n{response_json.get('transcription', 'No transcription available')}")
                print("⏰ Timings:")
                timings = response_json.get('timings', [])
                for segment in timings:
                    start = segment.get('start', 'Unknown')
                    end = segment.get('end', 'Unknown')
                    text = segment.get('text', 'No text')
                    print(f"  Start: {start}, End: {end}, Text: {text}")
                return
        else:
            try:
                error = response.json()
            except Exception:
                error = response.text
            print(f"❌ Error checking status: {error}")

        time.sleep(20)  # Wait 20 seconds before checking again

if __name__ == "__main__":
    print("🚀 Submitting transcription job...")
    if upload_audio():
        check_transcription()
