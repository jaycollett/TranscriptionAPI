import requests
import logging
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = "http://localhost:5030/transcriptions"

def check_transcription_queue():
    """Fetches and displays the current transcription queue with estimated wait times."""
    while True:
        logging.info("Fetching transcription queue...")
        try:
            response = requests.get(API_URL)
            if response.status_code == 200:
                transcriptions = response.json()
                
                # Filter pending jobs
                pending_jobs = [t for t in transcriptions if t['status'] == 'pending']
                total_pending = len(pending_jobs)

                # Calculate total estimated wait time
                total_wait_time_sec = sum(t['processing_time_est'] for t in pending_jobs)
                total_wait_time_human = str(timedelta(seconds=total_wait_time_sec))

                if total_pending == 0:
                    logging.info("No pending transcriptions in queue.")
                else:
                    logging.info(f"Total Pending Jobs: {total_pending}")
                    logging.info(f"Estimated Total Wait Time: {total_wait_time_human} (hh:mm:ss)")

                    for t in pending_jobs:
                        # Convert `submitted_at` to datetime
                        submitted_time = datetime.strptime(t['submitted_at'], "%Y-%m-%d %H:%M:%S")

                        # Calculate estimated completion UTC time
                        estimated_completion_time = submitted_time + timedelta(seconds=t['processing_time_est'])
                        estimated_completion_str = estimated_completion_time.strftime("%Y-%m-%d %H:%M:%S UTC")

                        # Log the details
                        logging.info(f"GUID: {t['guid']}, Filename: {t['filename']}, Submitted: {t['submitted_at']}, Estimated Completion: {estimated_completion_str}")

            else:
                logging.error(f"Failed to fetch transcriptions. Status Code: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logging.error(f"Error fetching transcription queue: {e}")

        logging.info("Waiting 30 seconds before next check...\n")
        time.sleep(30)

if __name__ == "__main__":
    check_transcription_queue()
