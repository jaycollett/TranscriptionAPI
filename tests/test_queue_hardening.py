"""Tests for the queue retry-cap / quarantine and crash-recovery hardening.

Covers the two production failure modes:
  1. A job that keeps producing garbage staying the oldest 'pending' row forever and
     blocking the whole FIFO queue (fixed by the attempt-count quarantine).
  2. In-flight jobs left in 'processing' after a container restart being stranded
     forever (fixed by startup recovery).
"""

import json
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
def _make_audio(upload_dir, guid, filename="sermon.mp3"):
    ext = os.path.splitext(filename)[-1]
    (upload_dir / f"{guid}{ext}").write_bytes(b"fake audio")
    return filename


def test_good_transcription_completes_and_keeps_attempt_history(app_module, db, upload_dir, monkeypatch):
    """A successful transcription completes; attempt_count is kept as the retry history
    (a completed row is terminal, so the counter can never trigger a quarantine)."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {"transcription": GOOD_TEXT, "timings": GOOD_TIMINGS},
    )
    monkeypatch.setattr(app_module, "run_forced_alignment", lambda p, t, g: (GOOD_TIMINGS, True))

    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    # Pre-seed a non-zero attempt_count to prove success clears it.
    insert_job(db, guid, filename=filename, status="pending", attempt_count=2)

    status = app_module.process_pending_job(db.cursor(), guid, filename)

    assert status == "completed"
    row = get_row(db, guid)
    assert row["status"] == "completed"
    assert row["attempt_count"] == 2
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


def test_process_whisper_exception_retries_then_succeeds(app_module, db, upload_dir, monkeypatch):
    """A transient exception (CUDA OOM) requeues the job; a later run can still complete it."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    calls = []

    def flaky(path, guid):
        calls.append(guid)
        if len(calls) <= 2:
            raise RuntimeError("CUDA out of memory")
        return {"transcription": GOOD_TEXT, "timings": GOOD_TIMINGS}

    monkeypatch.setattr(app_module, "transcribe_audio", flaky)
    monkeypatch.setattr(app_module, "run_forced_alignment", lambda p, t, g: (GOOD_TIMINGS, True))
    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")
    cur = db.cursor()

    assert app_module.process_pending_job(cur, guid, filename) == "pending"
    assert get_row(db, guid)["attempt_count"] == 1
    assert app_module.process_pending_job(cur, guid, filename) == "pending"
    assert get_row(db, guid)["attempt_count"] == 2
    assert app_module.process_pending_job(cur, guid, filename) == "completed"
    assert get_row(db, guid)["status"] == "completed"


def test_process_whisper_exception_marks_error_after_cap(app_module, db, upload_dir, monkeypatch):
    """A deterministic crash still terminates: 'error' once the retry cap is reached."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)

    def boom(path, guid):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(app_module, "transcribe_audio", boom)
    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")
    cur = db.cursor()

    assert app_module.process_pending_job(cur, guid, filename) == "pending"
    assert app_module.process_pending_job(cur, guid, filename) == "pending"
    assert app_module.process_pending_job(cur, guid, filename) == "error"
    row = get_row(db, guid)
    assert row["status"] == "error"
    assert row["attempt_count"] == 3
    assert row["completed_at"] is not None


def test_empty_alignment_falls_back_to_whisper_timings(app_module, db, upload_dir, monkeypatch):
    """An alignment that returns nothing is not a garbage attempt: the job completes on
    Whisper's own timings and attempt_count is untouched."""
    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {"transcription": _prose(250), "timings": GOOD_TIMINGS, "duration_sec": 100.0},
    )
    monkeypatch.setattr(app_module, "run_forced_alignment", lambda p, t, g: ([], False))
    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending", attempt_count=0)

    assert app_module.process_pending_job(db.cursor(), guid, filename) == "completed"
    row = get_row(db, guid)
    assert row["attempt_count"] == 0
    assert json.loads(row["timings"]) == GOOD_TIMINGS
    assert row["mfa_applied"] == 0


def test_completed_row_records_metrics(app_module, db, upload_dir, monkeypatch):
    """processing_seconds, words_per_second and mfa_applied are stored on completion."""
    words = int(100 * 2.5)
    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {"transcription": _prose(words), "timings": GOOD_TIMINGS, "duration_sec": 100.0},
    )
    monkeypatch.setattr(app_module, "run_forced_alignment", lambda p, t, g: (GOOD_TIMINGS, True))
    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, filename) == "completed"
    row = get_row(db, guid)
    assert row["words_per_second"] == pytest.approx(2.5)
    assert row["processing_seconds"] is not None and row["processing_seconds"] >= 0
    assert row["mfa_applied"] == 1


def test_uppercase_extension_is_found_on_disk(app_module, db, upload_dir, monkeypatch):
    """The upload saves <guid><ext lowercased>; the worker must look for the same name."""
    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {"transcription": GOOD_TEXT, "timings": GOOD_TIMINGS},
    )
    monkeypatch.setattr(app_module, "run_forced_alignment", lambda p, t, g: (GOOD_TIMINGS, True))
    guid = str(uuid.uuid4())
    (upload_dir / f"{guid}.mp3").write_bytes(b"fake audio")
    insert_job(db, guid, filename="Sermon.MP3", status="pending")
    assert app_module.process_pending_job(db.cursor(), guid, "Sermon.MP3") == "completed"


# --------------------------------------------------------------------------------------
# Words-per-second floor on the transcription result
# --------------------------------------------------------------------------------------
def _prose(word_count):
    """Coherent-looking text of exactly word_count words (passes the alnum-ratio check)."""
    return " ".join(f"word{i}" for i in range(word_count))


def _stub_transcription(app_module, monkeypatch, word_count, duration_sec):
    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {
            "transcription": _prose(word_count),
            "timings": GOOD_TIMINGS,
            "duration_sec": duration_sec,
        },
    )
    monkeypatch.setattr(app_module, "run_forced_alignment", lambda p, t, g: (GOOD_TIMINGS, True))


def test_low_word_rate_on_long_file_is_requeued(app_module, db, upload_dir, monkeypatch):
    """0.5 words/sec on a 755 s file is a collapsed decode: requeue, not complete."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    monkeypatch.setattr(app_module, "MIN_WORDS_PER_SEC", 1.0)
    _stub_transcription(app_module, monkeypatch, word_count=int(755 * 0.5), duration_sec=755.0)

    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, filename) == "pending"
    row = get_row(db, guid)
    assert row["status"] == "pending"
    assert row["attempt_count"] == 1
    assert row["transcription"] is None


def test_normal_word_rate_completes(app_module, db, upload_dir, monkeypatch):
    """2.7 words/sec is normal sermon material and must complete."""
    monkeypatch.setattr(app_module, "MIN_WORDS_PER_SEC", 1.0)
    _stub_transcription(app_module, monkeypatch, word_count=int(755 * 2.7), duration_sec=755.0)

    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, filename) == "completed"
    row = get_row(db, guid)
    assert row["status"] == "completed"
    assert row["attempt_count"] == 0


def test_unknown_duration_is_not_gated_by_word_rate(app_module, db, upload_dir, monkeypatch):
    """duration_sec == 0 means the duration could not be read; the floor must not apply."""
    monkeypatch.setattr(app_module, "MIN_WORDS_PER_SEC", 1.0)
    _stub_transcription(app_module, monkeypatch, word_count=40, duration_sec=0)

    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, filename) == "completed"
    assert get_row(db, guid)["status"] == "completed"


def test_short_clip_is_not_gated_by_word_rate(app_module, db, upload_dir, monkeypatch):
    """Under 30 s a pause or two swings the rate; a sparse short clip still completes."""
    monkeypatch.setattr(app_module, "MIN_WORDS_PER_SEC", 1.0)
    # 12 words in 20 s is 0.6 words/sec, below the floor, but the clip is too short to gate.
    _stub_transcription(app_module, monkeypatch, word_count=12, duration_sec=20.0)

    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, filename) == "completed"
    assert get_row(db, guid)["status"] == "completed"
