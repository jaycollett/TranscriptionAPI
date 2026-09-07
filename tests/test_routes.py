"""Route tests for /upload, /status, /transcriptions and /health via Flask's test client.

The public contract is pinned here: status values, field names and types on every
response, and the 201/400/404/409 codes. New fields are asserted as additive only.
"""

import io
import json
import threading
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from conftest import get_row, insert_job

FIXED_NOW = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)


def _upload(client, guid, filename="sermon.mp3", data=b"fake audio bytes"):
    return client.post(
        "/upload",
        data={"file": (io.BytesIO(data), filename), "guid": guid},
        content_type="multipart/form-data",
    )


@pytest.fixture
def frozen_now(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "utcnow", lambda: FIXED_NOW)
    return FIXED_NOW


# --------------------------------------------------------------------------------------
# /upload
# --------------------------------------------------------------------------------------
def test_upload_success_returns_contract_fields(client, route_db, upload_dir, app_module, monkeypatch, frozen_now):
    monkeypatch.setattr(app_module, "get_audio_duration", lambda path: 755.0)
    monkeypatch.setattr(app_module, "estimate_processing_seconds", lambda d: 295)
    guid = str(uuid.uuid4())

    resp = _upload(client, guid)

    assert resp.status_code == 201
    body = resp.get_json()
    assert body["message"] == "File uploaded successfully"
    assert body["guid"] == guid
    expected = (FIXED_NOW + timedelta(seconds=295)).strftime("%Y-%m-%d %H:%M:%S UTC")
    assert body["estimated_completion_utc"] == expected
    assert (upload_dir / f"{guid}.mp3").read_bytes() == b"fake audio bytes"
    row = get_row(route_db, guid)
    assert row["status"] == "pending"
    assert row["filename"] == "sermon.mp3"


def test_upload_eta_counts_pending_and_processing(client, route_db, app_module, monkeypatch, frozen_now):
    """The in-flight job is ahead of the new one just as much as the waiting ones are."""
    monkeypatch.setattr(app_module, "estimate_processing_seconds", lambda d: 100)
    insert_job(route_db, str(uuid.uuid4()), status="processing", processing_time_est=600)
    insert_job(route_db, str(uuid.uuid4()), status="pending", processing_time_est=300)
    insert_job(route_db, str(uuid.uuid4()), status="completed", processing_time_est=9999)

    resp = _upload(client, str(uuid.uuid4()))

    assert resp.status_code == 201
    expected = (FIXED_NOW + timedelta(seconds=600 + 300 + 100)).strftime("%Y-%m-%d %H:%M:%S UTC")
    assert resp.get_json()["estimated_completion_utc"] == expected


@pytest.mark.parametrize("payload, expected_error", [
    ({}, "No file part"),
    ({"file": (io.BytesIO(b""), "")}, "No selected file"),
    ({"file": (io.BytesIO(b"x"), "a.mp3")}, "GUID is required"),
])
def test_upload_missing_parts_return_400(client, payload, expected_error):
    resp = client.post("/upload", data=payload, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == expected_error


@pytest.mark.parametrize("bad_guid", [
    "not-a-uuid",
    "{6ba7b810-9dad-11d1-80b4-00c04fd430c8}",
    "urn:uuid:6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "6ba7b8109dad11d180b400c04fd430c8",
])
def test_upload_rejects_non_canonical_guid(client, upload_dir, bad_guid):
    resp = _upload(client, bad_guid)
    assert resp.status_code == 400
    assert "Invalid GUID" in resp.get_json()["error"]
    assert not list(upload_dir.iterdir()), "nothing may be saved for a rejected GUID"


def test_upload_echoes_guid_unchanged(client, app_module, monkeypatch):
    """Canonical uppercase hex is accepted and echoed as sent, not rewritten."""
    guid = str(uuid.uuid4()).upper()
    resp = _upload(client, guid)
    assert resp.status_code == 201
    assert resp.get_json()["guid"] == guid


def test_upload_duplicate_guid_returns_409(client, route_db):
    guid = str(uuid.uuid4())
    assert _upload(client, guid).status_code == 201
    resp = _upload(client, guid, data=b"different bytes")
    assert resp.status_code == 409
    assert resp.get_json()["error"] == "GUID already exists"


def test_upload_integrity_error_returns_409(client, app_module, monkeypatch, upload_dir):
    """A row that appears between the check and the insert (another process on the
    same database) is a conflict, not a 500."""
    import sqlite3

    real_cursor_factory = app_module.get_db_connection

    class _RacingCursor:
        def __init__(self, cursor):
            self._cursor = cursor

        def execute(self, sql, params=()):
            if sql.lstrip().upper().startswith("INSERT"):
                raise sqlite3.IntegrityError("UNIQUE constraint failed: transcriptions.guid")
            return self._cursor.execute(sql, params)

        def fetchone(self):
            return self._cursor.fetchone()

    class _Conn:
        def __init__(self, conn):
            self._conn = conn

        def cursor(self):
            return _RacingCursor(self._conn.cursor())

    monkeypatch.setattr(app_module, "get_db_connection", lambda: _Conn(real_cursor_factory()))
    guid = str(uuid.uuid4())
    resp = _upload(client, guid)
    assert resp.status_code == 409
    assert not (upload_dir / f"{guid}.mp3").exists(), "the loser must not write over the winner's file"


def test_upload_duplicate_guid_in_other_casing_returns_409(client, route_db):
    guid = str(uuid.uuid4())
    assert _upload(client, guid).status_code == 201
    resp = _upload(client, guid.upper())
    assert resp.status_code == 409


def test_upload_save_failure_removes_row_and_returns_500(client, route_db, upload_dir, app_module, monkeypatch):
    import werkzeug.datastructures

    def broken_save(self, dst, buffer_size=16384):
        raise OSError("No space left on device")

    monkeypatch.setattr(werkzeug.datastructures.FileStorage, "save", broken_save)
    guid = str(uuid.uuid4())
    resp = _upload(client, guid)
    assert resp.status_code == 500
    assert get_row(route_db, guid) is None
    assert not (upload_dir / f"{guid}.mp3").exists()


def test_upload_row_is_not_visible_to_the_worker_before_the_file_exists(client, route_db, upload_dir, app_module, monkeypatch):
    """The worker selects pending rows under upload_lock, so a row inserted by an
    upload still in progress is never picked up while its file is missing."""
    seen_while_locked = []
    original_get_duration = app_module.get_audio_duration

    def probe_during_upload(path):
        # Mid-upload: the row exists but the lock is held. A worker cycle on
        # another thread must block rather than pick the row up.
        acquired = app_module.upload_lock.acquire(timeout=0.05)
        seen_while_locked.append(acquired)
        if acquired:
            app_module.upload_lock.release()
        return 60.0

    monkeypatch.setattr(app_module, "get_audio_duration", probe_during_upload)
    guid = str(uuid.uuid4())
    assert _upload(client, guid).status_code == 201
    assert seen_while_locked == [False], "the lock must be held across insert, save and probe"
    del original_get_duration


def test_upload_concurrent_same_guid_one_wins(client, route_db, app_module, monkeypatch, upload_dir):
    """Two simultaneous uploads of one GUID: exactly one 201, one 409, one row, one file."""
    gate = threading.Event()

    def slow_duration(path):
        gate.wait(timeout=5)
        return 60.0

    monkeypatch.setattr(app_module, "get_audio_duration", slow_duration)
    guid = str(uuid.uuid4())
    results = []

    def worker(payload):
        with app_module.app.test_client() as c:
            results.append(_upload(c, guid, data=payload).status_code)

    threads = [threading.Thread(target=worker, args=(b"first",)),
               threading.Thread(target=worker, args=(b"second",))]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(timeout=10)

    assert sorted(results) == [201, 409]
    cur = route_db.cursor()
    cur.execute("SELECT COUNT(*) FROM transcriptions WHERE guid = ?", (guid,))
    assert cur.fetchone()[0] == 1
    assert (upload_dir / f"{guid}.mp3").exists()


def test_upload_undecodable_audio_returns_400_and_removes_file(client, route_db, upload_dir, app_module, monkeypatch):
    def cannot_decode(path):
        raise RuntimeError("Decoding failed. ffmpeg returned error code: 1")

    monkeypatch.setattr(app_module, "get_audio_duration", cannot_decode)
    guid = str(uuid.uuid4())

    resp = _upload(client, guid, data=b"not audio")

    assert resp.status_code == 400
    assert resp.get_json() == {"error": "Could not decode audio"}
    assert not (upload_dir / f"{guid}.mp3").exists()
    assert get_row(route_db, guid) is None


@pytest.mark.parametrize("filename", ["script.sh", "a.mp3.sh", "noext", "archive.zip"])
def test_upload_rejects_disallowed_extension(client, upload_dir, filename):
    resp = _upload(client, str(uuid.uuid4()), filename=filename)
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.get_json()["error"]
    assert not list(upload_dir.iterdir())


@pytest.mark.parametrize("filename", ["Sermon.MP3", "talk.M4A", "audio.flac"])
def test_upload_lowercases_extension_on_disk(client, upload_dir, filename):
    guid = str(uuid.uuid4())
    assert _upload(client, guid, filename=filename).status_code == 201
    ext = filename.rsplit(".", 1)[1].lower()
    assert (upload_dir / f"{guid}.{ext}").exists()


def test_upload_oversize_body_returns_413(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 200)
    resp = _upload(client, str(uuid.uuid4()), data=b"x" * 1000)
    assert resp.status_code == 413


def test_upload_size_cap_is_one_gib(app_module):
    assert app_module.app.config["MAX_CONTENT_LENGTH"] == 1024 ** 3


# --------------------------------------------------------------------------------------
# /status/<guid>
# --------------------------------------------------------------------------------------
def test_status_unknown_guid_404(client):
    resp = client.get(f"/status/{uuid.uuid4()}")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "GUID not found"


def test_status_pending_eta_is_never_in_the_past(client, route_db, frozen_now):
    """A job submitted an hour ago with a 5 minute estimate reports now, not 55 minutes ago."""
    guid = str(uuid.uuid4())
    created = (FIXED_NOW - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    insert_job(route_db, guid, status="pending", created_at=created, processing_time_est=300)

    resp = client.get(f"/status/{guid}")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "pending"
    assert body["estimated_completion_utc"] == FIXED_NOW.strftime("%Y-%m-%d %H:%M:%S UTC")


def test_status_pending_eta_sums_jobs_ahead(client, route_db, frozen_now):
    """created_at + (pending and processing ahead) + own estimate, when that is in the future."""
    ahead_processing = str(uuid.uuid4())
    ahead_pending = str(uuid.uuid4())
    guid = str(uuid.uuid4())
    t0 = FIXED_NOW - timedelta(minutes=1)
    insert_job(route_db, ahead_processing, status="processing",
               created_at=(t0 - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"), processing_time_est=600)
    insert_job(route_db, ahead_pending, status="pending",
               created_at=(t0 - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"), processing_time_est=300)
    insert_job(route_db, guid, status="pending",
               created_at=t0.strftime("%Y-%m-%d %H:%M:%S"), processing_time_est=100)

    body = client.get(f"/status/{guid}").get_json()

    expected = (t0 + timedelta(seconds=600 + 300 + 100)).strftime("%Y-%m-%d %H:%M:%S UTC")
    assert body["estimated_completion_utc"] == expected


def test_status_processing_has_message_and_eta(client, route_db, frozen_now):
    guid = str(uuid.uuid4())
    insert_job(route_db, guid, status="processing", processing_time_est=60)
    body = client.get(f"/status/{guid}").get_json()
    assert body["status"] == "processing"
    assert body["message"] == "Transcription is currently in progress"
    assert body["estimated_completion_utc"].endswith(" UTC")


def test_status_completed_returns_results_and_metrics(client, route_db):
    guid = str(uuid.uuid4())
    timings = [{"start": 0.0, "end": 2.5, "text": "hello world"}]
    cur = route_db.cursor()
    cur.execute(
        "INSERT INTO transcriptions (guid, filename, status, transcription, timings, "
        "processing_seconds, words_per_second, attempt_count, mfa_applied) "
        "VALUES (?, ?, 'completed', ?, ?, ?, ?, ?, ?)",
        (guid, "a.mp3", "hello world", json.dumps(timings), 185.2, 2.71, 1, 1),
    )

    body = client.get(f"/status/{guid}").get_json()

    assert body["status"] == "completed"
    assert body["transcription"] == "hello world"
    assert body["timings"] == timings
    assert isinstance(body["timings"][0]["start"], float)
    assert body["processing_seconds"] == pytest.approx(185.2)
    assert body["words_per_second"] == pytest.approx(2.71)
    assert body["attempt_count"] == 1
    assert body["mfa_applied"] is True


def test_status_legacy_processed_maps_to_completed(client, route_db):
    guid = str(uuid.uuid4())
    insert_job(route_db, guid, status="processed")
    body = client.get(f"/status/{guid}").get_json()
    assert body["status"] == "completed"
    assert body["transcription"] == ""
    assert body["timings"] == []
    assert body["mfa_applied"] is None


@pytest.mark.parametrize("stored_status", ["error", "quarantined"])
def test_status_terminal_failures_report_error(client, route_db, stored_status):
    guid = str(uuid.uuid4())
    insert_job(route_db, guid, status=stored_status)
    body = client.get(f"/status/{guid}").get_json()
    assert body == {"status": "error", "message": "Transcription failed"}


def test_status_values_stay_within_contract(client, route_db):
    statuses = {"pending", "processing", "completed", "error"}
    for stored in ["pending", "processing", "completed", "processed", "error", "quarantined"]:
        guid = str(uuid.uuid4())
        insert_job(route_db, guid, status=stored)
        assert client.get(f"/status/{guid}").get_json()["status"] in statuses


# --------------------------------------------------------------------------------------
# /transcriptions
# --------------------------------------------------------------------------------------
def test_transcriptions_lists_contract_fields_plus_metrics(client, route_db):
    older = str(uuid.uuid4())
    newer = str(uuid.uuid4())
    insert_job(route_db, older, status="completed", created_at="2026-01-01 00:00:00", processing_time_est=120)
    insert_job(route_db, newer, status="pending", created_at="2026-01-02 00:00:00", processing_time_est=60)

    resp = client.get("/transcriptions")

    assert resp.status_code == 200
    rows = resp.get_json()
    assert [r["guid"] for r in rows] == [newer, older], "newest first"
    for row in rows:
        for key in ("guid", "filename", "status", "submitted_at", "completed_at", "processing_time_est"):
            assert key in row
        for key in ("processing_seconds", "words_per_second", "attempt_count", "mfa_applied"):
            assert key in row
    pending = rows[0]
    assert pending["completed_at"] == ""
    assert pending["processing_time_est"] == 60
    assert pending["processing_seconds"] is None
    assert pending["mfa_applied"] is None
    assert pending["attempt_count"] == 0


def test_transcriptions_empty_list(client):
    resp = client.get("/transcriptions")
    assert resp.status_code == 200
    assert resp.get_json() == []


# --------------------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------------------
class _FakeThread:
    def __init__(self, alive):
        self._alive = alive

    def is_alive(self):
        return self._alive


@pytest.fixture
def worker_state(app_module, monkeypatch):
    state = {"last_wake": None, "last_cleanup": None, "busy_since": None, "idle_logged": False}
    monkeypatch.setattr(app_module, "worker_state", state)
    return state


def test_health_503_when_worker_not_started(client, app_module, monkeypatch, worker_state):
    monkeypatch.setattr(app_module, "worker_thread", None)
    resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.get_json()
    assert body["status"] == "unhealthy"
    assert body["worker_alive"] is False
    assert body["worker_last_wake_utc"] is None


def test_health_ok_when_worker_recently_woke(client, route_db, app_module, monkeypatch, worker_state, frozen_now):
    monkeypatch.setattr(app_module, "worker_thread", _FakeThread(alive=True))
    monkeypatch.setattr(app_module, "whisper_model_loaded", lambda: True)
    worker_state["last_wake"] = FIXED_NOW - timedelta(seconds=20)
    insert_job(route_db, str(uuid.uuid4()), status="pending")
    insert_job(route_db, str(uuid.uuid4()), status="pending")
    insert_job(route_db, str(uuid.uuid4()), status="processing")
    insert_job(route_db, str(uuid.uuid4()), status="completed")

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["worker_alive"] is True
    assert body["worker_last_wake_utc"] == (FIXED_NOW - timedelta(seconds=20)).strftime("%Y-%m-%d %H:%M:%S UTC")
    assert body["pending"] == 2
    assert body["processing"] == 1
    assert body["whisper_loaded"] is True


def test_health_503_when_worker_thread_died(client, app_module, monkeypatch, worker_state, frozen_now):
    monkeypatch.setattr(app_module, "worker_thread", _FakeThread(alive=False))
    worker_state["last_wake"] = FIXED_NOW - timedelta(seconds=5)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.get_json()["worker_alive"] is False


def test_health_503_when_idle_worker_has_not_woken_in_five_minutes(client, app_module, monkeypatch, worker_state, frozen_now):
    monkeypatch.setattr(app_module, "worker_thread", _FakeThread(alive=True))
    worker_state["last_wake"] = FIXED_NOW - timedelta(minutes=6)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.get_json()["worker_alive"] is True


def test_health_ok_while_busy_on_a_long_job(client, app_module, monkeypatch, worker_state, frozen_now):
    """A worker inside a 46 minute transcription has not polled in a while but is not dead."""
    monkeypatch.setattr(app_module, "worker_thread", _FakeThread(alive=True))
    worker_state["last_wake"] = FIXED_NOW - timedelta(minutes=40)
    worker_state["busy_since"] = FIXED_NOW - timedelta(minutes=40)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["worker_busy"] is True
