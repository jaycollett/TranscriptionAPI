import logging
import time
from datetime import datetime, timedelta, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = "http://localhost:5030/transcriptions"


def estimate_completions(transcriptions, now=None):
    """Return [(job, estimated_completion_datetime)] for every pending job, in queue order.

    The queue is FIFO and single-worker, so a job finishes only after everything
    ahead of it: the job currently processing plus every pending job submitted
    earlier. Each estimate is therefore the running sum of processing_time_est
    over those jobs, anchored on now, which is the same arithmetic /status uses.
    """
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    in_flight = sum(t['processing_time_est'] for t in transcriptions if t['status'] == 'processing')
    pending = sorted(
        (t for t in transcriptions if t['status'] == 'pending'),
        key=lambda t: t['submitted_at'],
    )
    estimates = []
    cumulative = in_flight
    for job in pending:
        cumulative += job['processing_time_est']
        estimates.append((job, now + timedelta(seconds=cumulative)))
    return estimates


def check_transcription_queue():
    """Fetches and displays the current transcription queue with estimated wait times."""
    import requests  # Host-side dependency (requirements-dev.txt), not needed by the service

    while True:
        logging.info("Fetching transcription queue...")
        try:
            response = requests.get(API_URL, timeout=10)
            if response.status_code == 200:
                transcriptions = response.json()
                estimates = estimate_completions(transcriptions)

                if not estimates:
                    logging.info("No pending transcriptions in queue.")
                else:
                    total_wait_sec = int((estimates[-1][1] - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds())
                    logging.info(f"Total Pending Jobs: {len(estimates)}")
                    logging.info(f"Estimated Total Wait Time: {timedelta(seconds=max(total_wait_sec, 0))} (hh:mm:ss)")

                    for job, eta in estimates:
                        logging.info(
                            f"GUID: {job['guid']}, Filename: {job['filename']}, "
                            f"Submitted: {job['submitted_at']}, "
                            f"Estimated Completion: {eta.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        )

            else:
                logging.error(f"Failed to fetch transcriptions. Status Code: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logging.error(f"Error fetching transcription queue: {e}")

        logging.info("Waiting 30 seconds before next check...\n")
        time.sleep(30)


if __name__ == "__main__":
    check_transcription_queue()
