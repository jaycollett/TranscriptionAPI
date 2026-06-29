"""Tests for the queue retry-cap / quarantine and crash-recovery hardening.

Covers the two production failure modes:
  1. A job that keeps producing garbage staying the oldest 'pending' row forever and
     blocking the whole FIFO queue (fixed by the attempt-count quarantine).
  2. In-flight jobs left in 'processing' after a container restart being stranded
     forever (fixed by startup recovery).
"""

import os
import sqlite3
import uuid

import pytest

from conftest import get_row, insert_job, next_pending_guid

GOOD_TEXT = (
    "This is a clean sermon transcription with plenty of real words so that the "
    "garbage detector treats it as valid output and the job completes successfully."
)
GOOD_TIMINGS = [{"start": 0.0, "end": 2.5, "text": "hello world"}]


# --------------------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------------------
def test_migration_adds_attempt_count_to_legacy_db(app_module):
    """ensure_schema() adds attempt_count to a pre-existing table without it,
    defaulting existing rows to 0."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    cur = conn.cursor()
    # Legacy schema: no attempt_count column.
    cur.execute(
        """
        CREATE TABLE transcriptions (
            guid TEXT PRIMARY KEY,
            filename TEXT,
            status TEXT DEFAULT 'pending',
            transcription TEXT DEFAULT NULL,
            timings TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP DEFAULT NULL,
            processing_time_est INTEGER DEFAULT 0
        )
        """
    )
    cur.execute("INSERT INTO transcriptions (guid, filename) VALUES ('legacy', 'old.mp3')")

    assert not app_module.column_exists(cur, "transcriptions", "attempt_count")
    app_module.ensure_schema(cur)
    assert app_module.column_exists(cur, "transcriptions", "attempt_count")

    cur.execute("SELECT attempt_count FROM transcriptions WHERE guid = 'legacy'")
    assert cur.fetchone()[0] == 0

    # Idempotent: a second call must not raise.
    app_module.ensure_schema(cur)
    conn.close()


# --------------------------------------------------------------------------------------
# Quarantine after N garbage retries
# --------------------------------------------------------------------------------------
def test_quarantine_after_max_garbage_retries(app_module, db, monkeypatch):
    """After max_garbage_retries garbage results the job is quarantined, not requeued."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    guid = str(uuid.uuid4())
    insert_job(db, guid, status="processing")
    cur = db.cursor()

    # First two garbage results requeue the job.
    assert app_module.handle_garbage_result(cur, guid, "garbage") == "pending"
    assert get_row(db, guid)["status"] == "pending"
    assert get_row(db, guid)["attempt_count"] == 1

    assert app_module.handle_garbage_result(cur, guid, "garbage") == "pending"
    assert get_row(db, guid)["attempt_count"] == 2

    # Third garbage result trips the cap -> quarantine (terminal).
    assert app_module.handle_garbage_result(cur, guid, "garbage") == "quarantined"
    row = get_row(db, guid)
    assert row["status"] == "quarantined"
    assert row["attempt_count"] == 3
    assert row["completed_at"] is not None


def test_custom_retry_cap_is_respected(app_module, db, monkeypatch):
    """A cap of 1 quarantines on the very first garbage result."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 1)
    guid = str(uuid.uuid4())
    insert_job(db, guid, status="processing")
    assert app_module.handle_garbage_result(db.cursor(), guid, "garbage") == "quarantined"
    assert get_row(db, guid)["status"] == "quarantined"


# --------------------------------------------------------------------------------------
# Queue advances past a quarantined job
# --------------------------------------------------------------------------------------
def test_queue_advances_after_quarantine(app_module, db, monkeypatch):
    """Once the head-of-line job is quarantined, the FIFO select returns the next job."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    bad = str(uuid.uuid4())
    good = str(uuid.uuid4())
    # `bad` is older, so it is the head of the FIFO queue.
    insert_job(db, bad, status="pending", created_at="2026-01-01 00:00:00")
    insert_job(db, good, status="pending", created_at="2026-01-01 00:05:00")

    # The bad job blocks the queue while it is the oldest pending row.
    assert next_pending_guid(db) == bad

    cur = db.cursor()
    for _ in range(3):
        app_module.handle_garbage_result(cur, bad, "garbage")

    assert get_row(db, bad)["status"] == "quarantined"
    # Queue now advances to the next sermon instead of looping on the bad one forever.
    assert next_pending_guid(db) == good


# --------------------------------------------------------------------------------------
# Crash recovery of orphaned 'processing' rows
# --------------------------------------------------------------------------------------
def test_recover_stuck_jobs_resets_processing(app_module, db):
    """Startup recovery requeues 'processing' rows and leaves attempt_count untouched."""
    stuck1 = str(uuid.uuid4())
    stuck2 = str(uuid.uuid4())
    waiting = str(uuid.uuid4())
    done = str(uuid.uuid4())
    insert_job(db, stuck1, status="processing", attempt_count=2)
    insert_job(db, stuck2, status="processing", attempt_count=0)
    insert_job(db, waiting, status="pending", attempt_count=0)
    insert_job(db, done, status="completed", attempt_count=0)

    recovered = app_module.recover_stuck_jobs(db.cursor())

    assert recovered == 2
    assert get_row(db, stuck1)["status"] == "pending"
    assert get_row(db, stuck2)["status"] == "pending"
    # Recovery must NOT count as a garbage retry.
    assert get_row(db, stuck1)["attempt_count"] == 2
    # Unrelated jobs are untouched.
    assert get_row(db, waiting)["status"] == "pending"
    assert get_row(db, done)["status"] == "completed"


def test_recover_stuck_jobs_noop_when_none(app_module, db):
    """Recovery returns 0 and changes nothing when there are no 'processing' rows."""
    guid = str(uuid.uuid4())
    insert_job(db, guid, status="pending")
    assert app_module.recover_stuck_jobs(db.cursor()) == 0
    assert get_row(db, guid)["status"] == "pending"


# --------------------------------------------------------------------------------------
# Full per-job flow via process_pending_job
# --------------------------------------------------------------------------------------
@pytest.fixture
def upload_dir(tmp_path, app_module, monkeypatch):
    """Point UPLOAD_FOLDER at a temp dir for process_pending_job file checks."""
    monkeypatch.setitem(app_module.app.config, "UPLOAD_FOLDER", str(tmp_path))
    return tmp_path


def _make_audio(upload_dir, guid, filename="sermon.mp3"):
    ext = os.path.splitext(filename)[-1]
    (upload_dir / f"{guid}{ext}").write_bytes(b"fake audio")
    return filename


def test_good_transcription_completes_and_clears_counter(app_module, db, upload_dir, monkeypatch):
    """A successful transcription completes and resets attempt_count to 0 (never quarantined)."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {"transcription": GOOD_TEXT, "timings": GOOD_TIMINGS},
    )
    monkeypatch.setattr(app_module, "run_forced_alignment", lambda p, t, g: GOOD_TIMINGS)

    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    # Pre-seed a non-zero attempt_count to prove success clears it.
    insert_job(db, guid, filename=filename, status="pending", attempt_count=2)

    status = app_module.process_pending_job(db.cursor(), guid, filename)

    assert status == "completed"
    row = get_row(db, guid)
    assert row["status"] == "completed"
    assert row["attempt_count"] == 0
    assert row["transcription"] == GOOD_TEXT
    assert row["timings"] is not None
    assert row["completed_at"] is not None


def test_process_garbage_requeues_then_quarantines(app_module, db, upload_dir, monkeypatch):
    """process_pending_job routes garbage through the retry/quarantine path end-to-end."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {"transcription": "", "timings": []},  # empty -> garbage
    )

    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending", attempt_count=0)
    cur = db.cursor()

    assert app_module.process_pending_job(cur, guid, filename) == "pending"
    assert get_row(db, guid)["attempt_count"] == 1
    assert app_module.process_pending_job(cur, guid, filename) == "pending"
    assert get_row(db, guid)["attempt_count"] == 2
    assert app_module.process_pending_job(cur, guid, filename) == "quarantined"
    assert get_row(db, guid)["status"] == "quarantined"


def test_process_missing_file_marks_error(app_module, db, upload_dir, monkeypatch):
    """A job whose audio file is gone is marked 'error', not requeued or quarantined."""
    guid = str(uuid.uuid4())
    insert_job(db, guid, filename="missing.mp3", status="pending")
    status = app_module.process_pending_job(db.cursor(), guid, "missing.mp3")
    assert status == "error"
    assert get_row(db, guid)["status"] == "error"


def test_process_whisper_exception_marks_error(app_module, db, upload_dir, monkeypatch):
    """A transcription exception marks the job 'error' (does not loop forever)."""
    def boom(path, guid):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(app_module, "transcribe_audio", boom)
    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")
    assert app_module.process_pending_job(db.cursor(), guid, filename) == "error"
    assert get_row(db, guid)["status"] == "error"
