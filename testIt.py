import requests
import time
import uuid
import os

# API URLs
API_BASE_URL = "http://localhost:5030"
UPLOAD_URL = f"{API_BASE_URL}/upload"
TRANSCRIPTION_URL = f"{API_BASE_URL}/transcription"

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
        print(f"❌ Upload failed: {response.json()}")
        return None

def check_transcription():
    """Checks the transcription status until it's completed."""
    while True:
        response = requests.get(f"{TRANSCRIPTION_URL}/{guid}")

        if response.status_code == 200:
            response_json = response.json()
            status = response_json['status']

            if status == 'pending':
                print(f"⏳ Still processing... Estimated completion: {response_json['estimated_completion_utc']}")
            elif status == 'processed':
                print("✅ Transcription complete!")
                print(f"📝 Transcription:\n{response_json['transcription']}")
                return
        else:
            print(f"❌ Error checking status: {response.json()}")

        time.sleep(20)  # Wait 20 seconds before checking again

if __name__ == "__main__":
    print("🚀 Submitting transcription job...")
    if upload_audio():
        check_transcription()
