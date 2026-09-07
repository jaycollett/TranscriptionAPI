"""Tests for the worker loop, the hourly cleanup, and the orphaned-file sweep.

Covers the production defects behind cleanup that never ran (and orphaned files when
it did), files that survive a redeploy without a row, rows stranded in 'processing'
while the worker is alive, and the 30 s idle gap between back-to-back jobs.
"""

import logging
import os
import time
import uuid

import pytest

from conftest import get_row, insert_job

GOOD_TEXT = " ".join(f"word{i}" for i in range(300))
GOOD_TIMINGS = [{"start": 0.0, "end": 2.5, "text": "hello world"}]


def _touch_job_files(upload_dir, guid, ext=".mp3", mfa_root=None):
    paths = [
        upload_dir / f"{guid}{ext}",
        upload_dir / f"{guid}.txt",
    ]
    for p in paths:
        p.write_bytes(b"x")
    (upload_dir / f"{guid}_aligned").mkdir()
    (upload_dir / f"{guid}_aligned" / f"{guid}.json").write_text("{}")
    (upload_dir / f"{guid}_mfa_input").mkdir()
    if mfa_root is not None:
        (mfa_root / f"{guid}_mfa_input").mkdir()
        (mfa_root / f"{guid}_mfa_input" / "corpus.db").write_bytes(b"x")


def _age(path, days):
    old = time.time() - days * 86400
    os.utime(path, (old, old))


# --------------------------------------------------------------------------------------
# cleanup_old_jobs: rows and files go together
# --------------------------------------------------------------------------------------
def test_cleanup_removes_old_terminal_rows_and_their_files(app_module, db, upload_dir):
    mfa_root = app_module.MFA_ROOT_DIR
    old_done = str(uuid.uuid4())
    old_error = str(uuid.uuid4())
    fresh_done = str(uuid.uuid4())
    old_pending = str(uuid.uuid4())
    insert_job(db, old_done, filename="a.MP3", status="completed", created_at="2026-01-01 00:00:00")
    insert_job(db, old_error, filename="b.wav", status="error", created_at="2026-01-01 00:00:00")
    insert_job(db, fresh_done, filename="c.mp3", status="completed")
    insert_job(db, old_pending, filename="d.mp3", status="pending", created_at="2026-01-01 00:00:00")
    from pathlib import Path
    _touch_job_files(upload_dir, old_done, ext=".mp3", mfa_root=Path(mfa_root))
    _touch_job_files(upload_dir, old_error, ext=".wav")
    _touch_job_files(upload_dir, fresh_done)
    _touch_job_files(upload_dir, old_pending)

    deleted, _swept = app_module.cleanup_old_jobs(db.cursor(), str(upload_dir))

    assert deleted == 2
    assert get_row(db, old_done) is None
    assert get_row(db, old_error) is None
    assert get_row(db, fresh_done) is not None
    assert get_row(db, old_pending) is not None
    for guid in (old_done, old_error):
        assert not list(upload_dir.glob(f"{guid}*")), f"files for {guid} should be gone"
    assert not (Path(mfa_root) / f"{old_done}_mfa_input").exists()
    assert (upload_dir / f"{fresh_done}.mp3").exists()
    assert (upload_dir / f"{old_pending}.mp3").exists()


def test_cleanup_deletes_every_old_row_not_just_twenty(app_module, db, upload_dir):
    """The old code deleted files for 20 rows but rows without limit; both must agree."""
    guids = [str(uuid.uuid4()) for _ in range(25)]
    for g in guids:
        insert_job(db, g, status="completed", created_at="2026-01-01 00:00:00")
        (upload_dir / f"{g}.mp3").write_bytes(b"x")

    deleted, _ = app_module.cleanup_old_jobs(db.cursor(), str(upload_dir))

    assert deleted == 25
    assert all(get_row(db, g) is None for g in guids)
    assert not list(upload_dir.glob("*.mp3"))


def test_cleanup_with_nothing_to_do(app_module, db, upload_dir):
    assert app_module.cleanup_old_jobs(db.cursor(), str(upload_dir)) == (0, 0)


def test_cleanup_ages_rows_by_completion_not_submission(app_module, db, upload_dir):
    """A job that waited two days in the queue and completed five minutes ago is
    still being polled for; only rows that finished a day ago are removed."""
    waited_long = str(uuid.uuid4())
    finished_long_ago = str(uuid.uuid4())
    insert_job(db, waited_long, status="completed", created_at="2026-01-01 00:00:00")
    insert_job(db, finished_long_ago, status="completed", created_at="2026-01-01 00:00:00")
    cur = db.cursor()
    cur.execute("UPDATE transcriptions SET completed_at = datetime('now', '-5 minutes') WHERE guid = ?", (waited_long,))
    cur.execute("UPDATE transcriptions SET completed_at = datetime('now', '-2 days') WHERE guid = ?", (finished_long_ago,))

    deleted, _ = app_module.cleanup_old_jobs(cur, str(upload_dir))

    assert deleted == 1
    assert get_row(db, waited_long) is not None
    assert get_row(db, finished_long_ago) is None


def test_cleanup_falls_back_to_created_at_without_completed_at(app_module, db, upload_dir):
    """Legacy terminal rows with no completed_at are aged by created_at."""
    legacy = str(uuid.uuid4())
    insert_job(db, legacy, status="error", created_at="2026-01-01 00:00:00")
    assert get_row(db, legacy)["completed_at"] is None
    deleted, _ = app_module.cleanup_old_jobs(db.cursor(), str(upload_dir))
    assert deleted == 1


# --------------------------------------------------------------------------------------
# Orphaned-file sweep (files outlive the container-layer database)
# --------------------------------------------------------------------------------------
def test_sweep_removes_old_files_with_no_row(app_module, db, upload_dir):
    stray = str(uuid.uuid4())
    (upload_dir / f"{stray}.mp3").write_bytes(b"x")
    (upload_dir / f"{stray}_aligned").mkdir()
    _age(upload_dir / f"{stray}.mp3", 3)
    _age(upload_dir / f"{stray}_aligned", 3)

    removed = app_module.sweep_orphaned_files(db.cursor(), str(upload_dir))

    assert removed == 2
    assert not list(upload_dir.iterdir())


def test_sweep_keeps_recent_files_and_live_jobs(app_module, db, upload_dir):
    recent = str(uuid.uuid4())
    live_old = str(uuid.uuid4())
    processing_old = str(uuid.uuid4())
    (upload_dir / f"{recent}.mp3").write_bytes(b"x")
    (upload_dir / f"{live_old}.mp3").write_bytes(b"x")
    (upload_dir / f"{processing_old}.mp3").write_bytes(b"x")
    _age(upload_dir / f"{live_old}.mp3", 3)
    _age(upload_dir / f"{processing_old}.mp3", 3)
    insert_job(db, live_old, status="pending")
    insert_job(db, processing_old.upper(), status="processing")

    removed = app_module.sweep_orphaned_files(db.cursor(), str(upload_dir))

    assert removed == 0
    assert (upload_dir / f"{recent}.mp3").exists()
    assert (upload_dir / f"{live_old}.mp3").exists()
    assert (upload_dir / f"{processing_old}.mp3").exists()


def test_sweep_ignores_entries_not_named_after_a_guid(app_module, db, upload_dir):
    (upload_dir / "notes.txt").write_bytes(b"x")
    _age(upload_dir / "notes.txt", 10)
    assert app_module.sweep_orphaned_files(db.cursor(), str(upload_dir)) == 0
    assert (upload_dir / "notes.txt").exists()


def test_sweep_removes_old_file_of_completed_job(app_module, db, upload_dir):
    """A terminal row does not protect its files: only pending/processing do."""
    done = str(uuid.uuid4())
    (upload_dir / f"{done}.mp3").write_bytes(b"x")
    _age(upload_dir / f"{done}.mp3", 3)
    insert_job(db, done, status="completed")
    assert app_module.sweep_orphaned_files(db.cursor(), str(upload_dir)) == 1


def test_sweep_missing_folder_is_a_noop(app_module, db, tmp_path):
    assert app_module.sweep_orphaned_files(db.cursor(), str(tmp_path / "nope")) == 0


# --------------------------------------------------------------------------------------
# worker_cycle
# --------------------------------------------------------------------------------------
@pytest.fixture
def state():
    return {"last_wake": None, "last_cleanup": None, "busy_since": None, "idle_logged": False}


@pytest.fixture
def stub_pipeline(app_module, monkeypatch):
    processed = []

    def fake_transcribe(path, guid):
        processed.append(guid)
        return {"transcription": GOOD_TEXT, "timings": GOOD_TIMINGS, "duration_sec": 100.0}

    monkeypatch.setattr(app_module, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(app_module, "run_forced_alignment", lambda p, t, g: (GOOD_TIMINGS, True))
    return processed


def _queue(db, upload_dir, n):
    guids = []
    for i in range(n):
        g = str(uuid.uuid4())
        insert_job(db, g, filename="s.mp3", status="pending", created_at=f"2026-01-01 00:0{i}:00")
        (upload_dir / f"{g}.mp3").write_bytes(b"x")
        guids.append(g)
    return guids


def test_cycle_requeues_orphaned_processing_row_every_wake(app_module, db, upload_dir, state, stub_pipeline):
    stuck = str(uuid.uuid4())
    insert_job(db, stuck, filename="s.mp3", status="processing", attempt_count=1)
    (upload_dir / f"{stuck}.mp3").write_bytes(b"x")

    assert app_module.worker_cycle(db.cursor(), state) is True

    # Recovered to pending, then picked up and completed in the same cycle; the
    # recovery itself did not count as an attempt.
    row = get_row(db, stuck)
    assert row["status"] == "completed"
    assert row["attempt_count"] == 1
    assert stub_pipeline == [stuck]


def test_cycle_returns_true_per_job_and_false_when_empty(app_module, db, upload_dir, state, stub_pipeline):
    guids = _queue(db, upload_dir, 2)
    cur = db.cursor()

    assert app_module.worker_cycle(cur, state) is True
    assert app_module.worker_cycle(cur, state) is True
    assert app_module.worker_cycle(cur, state) is False

    assert stub_pipeline == guids, "oldest first"
    assert state["last_wake"] is not None
    assert state["busy_since"] is None


def test_cycle_requeue_returns_false_so_the_retry_waits(app_module, db, upload_dir, state, monkeypatch):
    """A failed attempt requeues the job as 'pending'; the cycle must report no
    progress so the loop sleeps before retrying instead of spinning through the
    attempt cap in seconds."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    calls = []

    def flaky(path, guid):
        calls.append(guid)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return {"transcription": GOOD_TEXT, "timings": GOOD_TIMINGS, "duration_sec": 100.0}

    monkeypatch.setattr(app_module, "transcribe_audio", flaky)
    monkeypatch.setattr(app_module, "run_forced_alignment", lambda p, t, g: (GOOD_TIMINGS, True))
    guid = _queue(db, upload_dir, 1)[0]
    cur = db.cursor()

    assert app_module.worker_cycle(cur, state) is False, "requeued: caller must sleep"
    assert get_row(db, guid)["status"] == "pending"
    assert app_module.worker_cycle(cur, state) is True
    assert get_row(db, guid)["status"] == "completed"


def test_worker_sleeps_between_a_failed_attempt_and_its_retry(app_module, db, upload_dir, monkeypatch):
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    monkeypatch.setattr(app_module, "get_db_connection", lambda: db)
    monkeypatch.setattr(app_module, "worker_state",
                        {"last_wake": None, "last_cleanup": None, "busy_since": None, "idle_logged": False})
    attempts = []

    def always_fails(path, guid):
        attempts.append(guid)
        raise RuntimeError("unreadable")

    monkeypatch.setattr(app_module, "transcribe_audio", always_fails)
    guid = _queue(db, upload_dir, 1)[0]
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise _StopLoop()

    monkeypatch.setattr(app_module.time, "sleep", fake_sleep)
    with pytest.raises(_StopLoop):
        app_module.transcription_worker()

    # One attempt, sleep, second attempt, sleep: never two attempts back to back.
    assert attempts == [guid, guid]
    assert sleeps == [app_module.POLL_INTERVAL_SEC] * 2
    assert get_row(db, guid)["attempt_count"] == 2
    assert get_row(db, guid)["status"] == "pending"


def test_cycle_runs_cleanup_on_first_wake_then_hourly(app_module, db, upload_dir, state, monkeypatch, stub_pipeline):
    calls = []
    monkeypatch.setattr(app_module, "cleanup_old_jobs", lambda cur, folder: calls.append(folder) or (0, 0))
    clock = {"t": 1000.0}
    monkeypatch.setattr(app_module.time, "monotonic", lambda: clock["t"])
    cur = db.cursor()

    app_module.worker_cycle(cur, state)
    assert calls == [str(upload_dir)]
    clock["t"] += 1800
    app_module.worker_cycle(cur, state)
    assert len(calls) == 1, "not due yet"
    clock["t"] += 1800
    app_module.worker_cycle(cur, state)
    assert len(calls) == 2, "an hour has passed"


def test_cycle_cleanup_runs_even_when_queue_is_empty(app_module, db, upload_dir, state, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "cleanup_old_jobs", lambda cur, folder: calls.append(1) or (0, 0))
    assert app_module.worker_cycle(db.cursor(), state) is False
    assert calls == [1]


def test_cycle_survives_cleanup_failure(app_module, db, upload_dir, state, monkeypatch, stub_pipeline):
    def broken(cur, folder):
        raise OSError("disk on fire")

    monkeypatch.setattr(app_module, "cleanup_old_jobs", broken)
    _queue(db, upload_dir, 1)
    assert app_module.worker_cycle(db.cursor(), state) is True
    assert len(stub_pipeline) == 1


def test_idle_transition_is_logged_once(app_module, db, upload_dir, state, caplog):
    caplog.set_level(logging.DEBUG, logger="app")
    cur = db.cursor()
    app_module.worker_cycle(cur, state)
    app_module.worker_cycle(cur, state)
    app_module.worker_cycle(cur, state)
    idle_lines = [r for r in caplog.records if "Worker idle" in r.getMessage()]
    assert len(idle_lines) == 1
    assert idle_lines[0].levelno == logging.INFO


def test_idle_message_logs_again_after_work(app_module, db, upload_dir, state, stub_pipeline, caplog):
    caplog.set_level(logging.INFO, logger="app")
    cur = db.cursor()
    app_module.worker_cycle(cur, state)   # idle -> logged
    _queue(db, upload_dir, 1)
    app_module.worker_cycle(cur, state)   # job
    app_module.worker_cycle(cur, state)   # idle again -> logged again
    assert sum("Worker idle" in r.getMessage() for r in caplog.records) == 2


def test_job_scoped_log_lines_carry_the_guid(app_module, db, upload_dir, stub_pipeline, caplog):
    caplog.set_level(logging.INFO, logger="app")
    guid = _queue(db, upload_dir, 1)[0]
    app_module.process_pending_job(db.cursor(), guid, "s.mp3")
    job_lines = [r.getMessage() for r in caplog.records if "Processing transcription" in r.getMessage()
                 or "completed for" in r.getMessage() or "forced alignment" in r.getMessage()]
    assert job_lines, "expected job-scoped log lines"
    assert all(guid in line for line in job_lines)


# --------------------------------------------------------------------------------------
# transcription_worker loop: no sleep between back-to-back jobs
# --------------------------------------------------------------------------------------
class _StopLoop(Exception):
    pass


def test_worker_drains_queue_before_sleeping(app_module, db, upload_dir, monkeypatch, stub_pipeline):
    guids = _queue(db, upload_dir, 3)
    monkeypatch.setattr(app_module, "get_db_connection", lambda: db)
    monkeypatch.setattr(app_module, "worker_state",
                        {"last_wake": None, "last_cleanup": None, "busy_since": None, "idle_logged": False})
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        raise _StopLoop()

    monkeypatch.setattr(app_module.time, "sleep", fake_sleep)

    with pytest.raises(_StopLoop):
        app_module.transcription_worker()

    assert stub_pipeline == guids, "all three jobs ran before the first sleep"
    assert sleeps == [app_module.POLL_INTERVAL_SEC]
    assert all(get_row(db, g)["status"] == "completed" for g in guids)


def test_worker_wake_and_sleep_lines_are_debug(app_module, db, upload_dir, monkeypatch, caplog):
    monkeypatch.setattr(app_module, "get_db_connection", lambda: db)
    monkeypatch.setattr(app_module, "worker_state",
                        {"last_wake": None, "last_cleanup": None, "busy_since": None, "idle_logged": False})
    caplog.set_level(logging.DEBUG, logger="app")

    def fake_sleep(seconds):
        raise _StopLoop()

    monkeypatch.setattr(app_module.time, "sleep", fake_sleep)
    with pytest.raises(_StopLoop):
        app_module.transcription_worker()

    sleep_lines = [r for r in caplog.records if "sleeping" in r.getMessage()]
    assert sleep_lines and all(r.levelno == logging.DEBUG for r in sleep_lines)


def test_worker_error_is_logged_and_loop_continues(app_module, db, upload_dir, monkeypatch, caplog):
    def broken():
        raise RuntimeError("database is locked")

    monkeypatch.setattr(app_module, "get_db_connection", broken)
    caplog.set_level(logging.ERROR, logger="app")

    def fake_sleep(seconds):
        raise _StopLoop()

    monkeypatch.setattr(app_module.time, "sleep", fake_sleep)
    with pytest.raises(_StopLoop):
        app_module.transcription_worker()
    assert any("Worker error" in r.getMessage() for r in caplog.records)


def test_start_worker_records_thread_for_health(app_module, monkeypatch):
    started = []

    class _FakeThread:
        def __init__(self, target, name, daemon):
            self.target = target

        def start(self):
            started.append(self)

        def is_alive(self):
            return True

    monkeypatch.setattr(app_module.threading, "Thread", _FakeThread)
    monkeypatch.setattr(app_module, "worker_thread", None)
    t = app_module.start_worker()
    assert started == [t]
    assert app_module.worker_thread is t
    # Idempotent while alive.
    assert app_module.start_worker() is t
    assert len(started) == 1
