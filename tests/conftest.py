"""Test fixtures for the TranscriptionAPI unit tests.

app.py imports the heavy transcription stack (faster-whisper, torch, pydub) at module
import time and runs init_db() against DB_FILE. To keep the unit tests fast, deterministic,
and runnable without a GPU, we stub those modules in sys.modules and point DB_FILE /
UPLOAD_FOLDER at throwaway temp locations BEFORE importing app.
"""

import importlib.util
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

# --- Stub the heavy imports app.py pulls in at module import time ---------------------
def _whisper_span(segment):
    """Real behaviour of transcribe.whisper_span; app.py's alignment depends on it."""
    words = segment.get("words")
    if words:
        return float(words[0]["start"]), float(words[-1]["end"])
    return float(segment["start"]), float(segment["end"])


_fake_transcribe = types.ModuleType("transcribe")
_fake_transcribe.transcribe_audio = lambda *a, **k: {"transcription": "", "timings": []}
_fake_transcribe.load_whisper_model = lambda *a, **k: None
_fake_transcribe.get_audio_duration = lambda path: 60.0
_fake_transcribe.estimate_processing_seconds = lambda duration_sec: 45
_fake_transcribe.whisper_model_loaded = lambda: False
_fake_transcribe.whisper_span = _whisper_span
sys.modules.setdefault("transcribe", _fake_transcribe)


def _make_pydub_stub():
    """A bare pydub package with AudioSegment and a utils.mediainfo submodule."""
    fake_pydub = types.ModuleType("pydub")

    class _AudioSegment:
        @staticmethod
        def from_file(path):
            raise AssertionError("audio must not be decoded in these tests")

    fake_pydub.AudioSegment = _AudioSegment
    fake_utils = types.ModuleType("pydub.utils")
    fake_utils.mediainfo = lambda path: {}
    fake_pydub.utils = fake_utils
    return fake_pydub, fake_utils


_fake_pydub, _fake_pydub_utils = _make_pydub_stub()
sys.modules.setdefault("pydub", _fake_pydub)
sys.modules.setdefault("pydub.utils", _fake_pydub_utils)

# --- Redirect DB + uploads to temp locations before app's module-level init_db() runs --
_TMP_DIR = tempfile.mkdtemp(prefix="transcriptionapi-tests-")
os.environ.setdefault("DB_FILE", os.path.join(_TMP_DIR, "transcriptions.db"))
os.environ.setdefault("UPLOAD_FOLDER", os.path.join(_TMP_DIR, "audio_files"))
os.environ.setdefault("MFA_ROOT_DIR", os.path.join(_TMP_DIR, "mfa"))


@pytest.fixture(scope="session")
def app_module():
    """Import the app module once with stubs/env in place."""
    import app

    return app


@pytest.fixture(scope="module")
def transcribe_module():
    """Load the real transcribe.py with torch/faster-whisper/pydub stubbed out.

    conftest stubs `transcribe` in sys.modules for the app.py tests, so the real module
    is loaded from its file path under a different name. No GPU, no model, no audio.
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = object

    fake_pydub, fake_pydub_utils = _make_pydub_stub()

    keys = ("torch", "faster_whisper", "pydub", "pydub.utils")
    saved = {k: sys.modules.get(k) for k in keys}
    sys.modules["torch"] = fake_torch
    sys.modules["faster_whisper"] = fake_fw
    sys.modules["pydub"] = fake_pydub
    sys.modules["pydub.utils"] = fake_pydub_utils
    try:
        spec = importlib.util.spec_from_file_location(
            "transcribe_under_test", os.path.join(REPO_ROOT, "transcribe.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


@pytest.fixture
def db(app_module):
    """Fresh in-memory SQLite DB with the production schema applied per test."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    app_module.ensure_schema(conn.cursor())
    yield conn
    conn.close()


@pytest.fixture
def upload_dir(tmp_path, app_module, monkeypatch):
    """Point UPLOAD_FOLDER (and MFA's root) at temp dirs for file-touching tests."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    mfa_root = tmp_path / "mfa"
    mfa_root.mkdir()
    monkeypatch.setitem(app_module.app.config, "UPLOAD_FOLDER", str(uploads))
    monkeypatch.setattr(app_module, "MFA_ROOT_DIR", str(mfa_root))
    return uploads


@pytest.fixture
def client(app_module, tmp_path, upload_dir, monkeypatch):
    """Flask test client backed by a fresh on-disk database in tmp_path.

    The routes open thread-local connections against app.db_file, so the module
    global is pointed at a per-test file and the schema is created there. Any
    connection left on this thread is closed afterwards so the next test starts
    clean.
    """
    monkeypatch.setattr(app_module, "db_file", str(tmp_path / "routes.db"))
    app_module.close_db_connection()
    app_module.init_db()
    app_module.app.config["TESTING"] = True
    yield app_module.app.test_client()
    app_module.close_db_connection()


@pytest.fixture
def route_db(app_module, client):
    """Direct connection to the same database the test client's routes use."""
    conn = sqlite3.connect(app_module.db_file, isolation_level=None)
    yield conn
    conn.close()


def insert_job(conn, guid, filename="sermon.mp3", status="pending",
               attempt_count=0, created_at=None, processing_time_est=0):
    """Insert a transcription row, optionally with an explicit created_at."""
    cur = conn.cursor()
    if created_at is not None:
        cur.execute(
            "INSERT INTO transcriptions "
            "(guid, filename, status, attempt_count, created_at, processing_time_est) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (guid, filename, status, attempt_count, created_at, processing_time_est),
        )
    else:
        cur.execute(
            "INSERT INTO transcriptions "
            "(guid, filename, status, attempt_count, processing_time_est) "
            "VALUES (?, ?, ?, ?, ?)",
            (guid, filename, status, attempt_count, processing_time_est),
        )


def get_row(conn, guid):
    """Return a job row as a dict keyed by column name."""
    cur = conn.cursor()
    cur.execute(
        "SELECT guid, filename, status, transcription, timings, attempt_count, completed_at, "
        "processing_seconds, words_per_second, mfa_applied, "
        "anomaly_count, anomaly_windows, flagged_segments, "
        "rescue_attempted, rescue_selected "
        "FROM transcriptions WHERE guid = ?",
        (guid,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    keys = ["guid", "filename", "status", "transcription", "timings",
            "attempt_count", "completed_at", "processing_seconds", "words_per_second",
            "mfa_applied", "anomaly_count", "anomaly_windows", "flagged_segments",
        "rescue_attempted", "rescue_selected"]
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
