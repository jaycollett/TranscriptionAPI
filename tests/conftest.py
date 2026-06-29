"""Test fixtures for the TranscriptionAPI queue-hardening tests.

app.py imports the heavy transcription stack (faster-whisper, torch, pydub) at module
import time and runs init_db() against DB_FILE. To keep the unit tests fast, deterministic,
and runnable without a GPU, we stub those modules in sys.modules and point DB_FILE /
UPLOAD_FOLDER at throwaway temp locations BEFORE importing app.
"""

import os
import sqlite3
import sys
import tempfile
import types

import pytest

# Repo root (one level up from tests/) must be importable so `import app` works.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- Stub the heavy imports app.py pulls in at import time ---------------------------
_fake_transcribe = types.ModuleType("transcribe")
_fake_transcribe.transcribe_audio = lambda *a, **k: {"transcription": "", "timings": []}
_fake_transcribe.load_whisper_model = lambda *a, **k: None
sys.modules.setdefault("transcribe", _fake_transcribe)

_fake_pydub = types.ModuleType("pydub")
_fake_pydub.AudioSegment = object
sys.modules.setdefault("pydub", _fake_pydub)

# --- Redirect DB + uploads to temp locations before app's module-level init_db() runs --
_TMP_DIR = tempfile.mkdtemp(prefix="transcriptionapi-tests-")
os.environ.setdefault("DB_FILE", os.path.join(_TMP_DIR, "transcriptions.db"))
os.environ.setdefault("UPLOAD_FOLDER", os.path.join(_TMP_DIR, "audio_files"))


@pytest.fixture(scope="session")
def app_module():
    """Import the app module once with stubs/env in place."""
    import app

    return app


@pytest.fixture
def db(app_module):
    """Fresh in-memory SQLite DB with the production schema applied per test."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    app_module.ensure_schema(conn.cursor())
    yield conn
    conn.close()


def insert_job(conn, guid, filename="sermon.mp3", status="pending",
               attempt_count=0, created_at=None):
    """Insert a transcription row, optionally with an explicit created_at."""
    cur = conn.cursor()
    if created_at is not None:
        cur.execute(
            "INSERT INTO transcriptions (guid, filename, status, attempt_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (guid, filename, status, attempt_count, created_at),
        )
    else:
        cur.execute(
            "INSERT INTO transcriptions (guid, filename, status, attempt_count) "
            "VALUES (?, ?, ?, ?)",
            (guid, filename, status, attempt_count),
        )


def get_row(conn, guid):
    """Return a job row as a dict keyed by column name."""
    cur = conn.cursor()
    cur.execute(
        "SELECT guid, filename, status, transcription, timings, attempt_count, completed_at "
        "FROM transcriptions WHERE guid = ?",
        (guid,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    keys = ["guid", "filename", "status", "transcription", "timings",
            "attempt_count", "completed_at"]
    return dict(zip(keys, row))


def next_pending_guid(conn):
    """Mirror the worker's FIFO selection: oldest pending job first."""
    cur = conn.cursor()
    cur.execute(
        "SELECT guid FROM transcriptions WHERE status = 'pending' "
        "ORDER BY created_at ASC LIMIT 1"
    )
    row = cur.fetchone()
    return row[0] if row else None
