"""Tests for the 0.6.0 anomaly gate, its columns and the additive API fields.

The whole-file word rate is blind to a partial collapse: a file that transcribes
normally for forty minutes and produces nothing for ten still clears
MIN_WORDS_PER_SEC. The window count catches that; the segment count catches a decode
that fell back to high temperatures or repeatedly tripped faster-whisper's own
compression and log-probability thresholds. On the six reference files a healthy
decode scored zero on both.
"""

import json
import os
import uuid

import pytest

from conftest import get_row, insert_job

TIMINGS = [
    {"start": 0.0, "end": 2.0, "text": "hello world"},
    {"start": 2.0, "end": 4.0, "text": "second segment"},
]
FLAGGED = [{"index": 3, "start": 30.0, "end": 42.0, "flags": ["loop"]}]


def _prose(word_count):
    return " ".join(f"word{i}" for i in range(word_count))


def _make_audio(upload_dir, guid, filename="sermon.mp3"):
    ext = os.path.splitext(filename)[-1]
    (upload_dir / f"{guid}{ext}").write_bytes(b"fake audio")
    return filename


@pytest.fixture
def stub_decode(app_module, monkeypatch):
    """Install a transcribe_audio returning healthy prose with the given diagnostics."""
    def install(anomaly_count=0, anomaly_windows=0, flagged_segments=None, words=270,
                duration_sec=100.0):
        monkeypatch.setattr(
            app_module, "transcribe_audio",
            lambda path, guid: {
                "transcription": _prose(words),
                "timings": TIMINGS,
                "segments": TIMINGS,
                "duration_sec": duration_sec,
                "anomaly_count": anomaly_count,
                "anomaly_windows": anomaly_windows,
                "flagged_segments": flagged_segments if flagged_segments is not None else [],
                "speech_seconds": duration_sec * 0.8,
                "mean_logprob": -0.0612345678901234,
            },
        )
        monkeypatch.setattr(
            app_module, "run_forced_alignment",
            lambda p, t, g, d=None: (TIMINGS, True, {"agree250": 0.61, "utterances": 4}),
        )
    return install


def _run(app_module, db, upload_dir):
    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")
    status = app_module.process_pending_job(db.cursor(), guid, filename)
    return guid, status


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------
def test_a_clean_decode_completes(app_module, db, upload_dir, stub_decode):
    stub_decode(anomaly_count=0, anomaly_windows=0)
    guid, status = _run(app_module, db, upload_dir)
    assert status == "completed"
    row = get_row(db, guid)
    assert row["anomaly_count"] == 0
    assert row["anomaly_windows"] == 0


def test_anomalies_below_the_thresholds_still_complete(app_module, db, upload_dir, stub_decode,
                                                       monkeypatch):
    """Four flagged segments and one low window is noise, not a collapse."""
    monkeypatch.setattr(app_module, "ANOMALY_SEGMENTS_MAX", 5)
    monkeypatch.setattr(app_module, "ANOMALY_WINDOWS_MAX", 2)
    stub_decode(anomaly_count=4, anomaly_windows=1)
    guid, status = _run(app_module, db, upload_dir)
    assert status == "completed"
    assert get_row(db, guid)["anomaly_count"] == 4


def test_too_many_low_windows_requeues_the_job(app_module, db, upload_dir, stub_decode,
                                               monkeypatch):
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    monkeypatch.setattr(app_module, "ANOMALY_WINDOWS_MAX", 2)
    stub_decode(anomaly_count=0, anomaly_windows=2)

    guid, status = _run(app_module, db, upload_dir)

    assert status == "pending"
    row = get_row(db, guid)
    assert row["attempt_count"] == 1
    assert row["transcription"] is None
    # The diagnostics of the rejected attempt are cleared with its transcript.
    assert row["anomaly_count"] is None


def test_too_many_flagged_segments_requeues_the_job(app_module, db, upload_dir, stub_decode,
                                                    monkeypatch):
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    monkeypatch.setattr(app_module, "ANOMALY_SEGMENTS_MAX", 5)
    stub_decode(anomaly_count=5, anomaly_windows=0)

    guid, status = _run(app_module, db, upload_dir)

    assert status == "pending"
    assert get_row(db, guid)["attempt_count"] == 1


def test_the_gate_reason_names_both_counts(app_module, db, upload_dir, stub_decode, monkeypatch,
                                           caplog):
    """The log line is the only record of why a good-looking transcript was thrown away."""
    import logging

    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    stub_decode(anomaly_count=7, anomaly_windows=3)
    with caplog.at_level(logging.WARNING):
        _run(app_module, db, upload_dir)
    assert "anomaly gate" in caplog.text
    assert "7 flagged segments" in caplog.text
    assert "3 low speech windows" in caplog.text


def test_an_identical_redecode_is_published_not_quarantined(app_module, db, upload_dir,
                                                            stub_decode, monkeypatch):
    """The tcf.20150424 case: three identical re-decodes, 510 s of GPU, nothing published.

    The decode is deterministic, so a job the anomaly gate rejects re-decodes to the
    same transcript and retrying cannot change the verdict. The first attempt still
    requeues, because the result might not be reproducible; the second recognises
    itself and publishes. A recording that produced 8292 plausible words must never be
    black-holed by a quality signal.
    """
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    stub_decode(anomaly_count=9, anomaly_windows=4, words=8292)

    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, filename) == "pending"
    assert app_module.process_pending_job(db.cursor(), guid, filename) == "completed"

    row = get_row(db, guid)
    assert row["status"] == "completed"
    assert row["transcription"] is not None
    assert len(row["transcription"].split()) == 8292
    # The anomaly fields are published with it so the job can be found and reviewed.
    assert row["anomaly_count"] == 9
    assert row["anomaly_windows"] == 4


def test_the_publish_despite_anomalies_is_logged(app_module, db, upload_dir, stub_decode,
                                                 monkeypatch, caplog):
    import logging

    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    stub_decode(anomaly_count=9, anomaly_windows=4, words=8292)
    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")

    app_module.process_pending_job(db.cursor(), guid, filename)
    with caplog.at_level(logging.WARNING):
        app_module.process_pending_job(db.cursor(), guid, filename)

    assert "Publishing" in caplog.text
    assert "reproduced the previous attempt exactly" in caplog.text
    assert "8292 words" in caplog.text


def test_a_changing_anomalous_result_still_quarantines(app_module, db, upload_dir,
                                                       monkeypatch):
    """A decode that differs each time has not proved retrying is futile.

    Quarantine still exists; what it no longer does is consume a deterministic result
    three times and deliver nothing.
    """
    monkeypatch.setattr(app_module, "max_garbage_retries", 2)
    counter = {"n": 0}

    def varying(path, guid):
        counter["n"] += 1
        return {
            "transcription": _prose(200 + counter["n"]),
            "timings": TIMINGS,
            "segments": TIMINGS,
            "duration_sec": 100.0,
            "anomaly_count": 9,
            "anomaly_windows": 4,
            "flagged_segments": [],
            "mean_logprob": -0.5 - counter["n"],
        }

    monkeypatch.setattr(app_module, "transcribe_audio", varying)

    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, filename) == "pending"
    assert app_module.process_pending_job(db.cursor(), guid, filename) == "quarantined"


def test_the_garbage_check_and_word_rate_floor_still_quarantine(app_module, db, upload_dir,
                                                                monkeypatch):
    """Quarantine is reserved for results that are actually unusable."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 2)
    monkeypatch.setattr(app_module, "MIN_WORDS_PER_SEC", 1.0)
    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {
            "transcription": _prose(50), "timings": TIMINGS, "segments": TIMINGS,
            "duration_sec": 755.0, "speech_seconds": 700.0,
            "anomaly_count": 0, "anomaly_windows": 0, "flagged_segments": [],
        },
    )

    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, filename) == "pending"
    assert app_module.process_pending_job(db.cursor(), guid, filename) == "quarantined"


@pytest.mark.parametrize("previous, current, same", [
    ("8292:-0.123456789012", "8292:-0.123456789012", True),
    ("8292:-0.123456789012", "8292:-0.123456700000", True),    # inside the epsilon
    ("8292:-0.123456789012", "8292:-0.223456789012", False),
    ("8292:-0.123456789012", "8291:-0.123456789012", False),   # a different word count
    (None, "8292:-0.1", False),
    ("", "8292:-0.1", False),
    ("8292:none", "8292:none", True),
    ("8292:none", "8292:-0.1", False),
    ("8292:garbage", "8292:-0.1", False),
])
def test_same_anomaly_result(app_module, previous, current, same):
    assert app_module.same_anomaly_result(previous, current) is same


def test_the_fingerprint_survives_the_requeue(app_module, db, upload_dir, stub_decode,
                                              monkeypatch):
    """It is cleared with the transcript and the whole check stops working."""
    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    stub_decode(anomaly_count=9, anomaly_windows=4, words=8292)
    guid = str(uuid.uuid4())
    filename = _make_audio(upload_dir, guid)
    insert_job(db, guid, filename=filename, status="pending")

    app_module.process_pending_job(db.cursor(), guid, filename)

    cur = db.cursor()
    cur.execute("SELECT last_anomaly_fingerprint FROM transcriptions WHERE guid = ?", (guid,))
    assert cur.fetchone()[0] is not None


def test_a_missing_anomaly_count_does_not_gate(app_module, db, upload_dir, monkeypatch):
    """An older or stubbed transcribe_audio returns no diagnostics; that is not a failure."""
    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {"transcription": _prose(270), "timings": TIMINGS,
                            "duration_sec": 100.0},
    )
    monkeypatch.setattr(app_module, "run_forced_alignment",
                        lambda p, t, g, d=None: (TIMINGS, True, {"agree250": None}))
    guid, status = _run(app_module, db, upload_dir)
    assert status == "completed"
    row = get_row(db, guid)
    assert row["anomaly_count"] is None
    assert json.loads(row["flagged_segments"]) == []


# --------------------------------------------------------------------------------------
# Columns and the alignment hand-off
# --------------------------------------------------------------------------------------
def test_flagged_segments_are_stored_as_json(app_module, db, upload_dir, stub_decode):
    stub_decode(anomaly_count=0, anomaly_windows=0, flagged_segments=FLAGGED)
    guid, status = _run(app_module, db, upload_dir)
    assert status == "completed"
    assert json.loads(get_row(db, guid)["flagged_segments"]) == FLAGGED


def test_the_new_columns_migrate_onto_an_older_database(app_module):
    """ensure_schema is the migration path; it must add the columns to a 0.5.x table."""
    import sqlite3

    conn = sqlite3.connect(":memory:", isolation_level=None)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE transcriptions (
            guid TEXT PRIMARY KEY, filename TEXT, status TEXT DEFAULT 'pending',
            transcription TEXT, timings TEXT, created_at TIMESTAMP, completed_at TIMESTAMP,
            processing_time_est INTEGER DEFAULT 0
        )
    """)
    cursor.execute("INSERT INTO transcriptions (guid, filename) VALUES ('g', 'a.mp3')")

    app_module.ensure_schema(cursor)

    for column in ("anomaly_count", "anomaly_windows", "flagged_segments"):
        assert app_module.column_exists(cursor, "transcriptions", column)
    # The pre-existing row survives with nulls in the new columns.
    cursor.execute("SELECT anomaly_count, anomaly_windows, flagged_segments FROM transcriptions")
    assert cursor.fetchone() == (None, None, None)
    conn.close()


def test_the_aligner_receives_the_diagnostic_segments(app_module, db, upload_dir, monkeypatch):
    """MFA word matching needs the word timestamps, which only `segments` carries."""
    detailed = [{"start": 0.0, "end": 2.0, "text": "hello world",
                 "words": [{"start": 0.1, "end": 0.6, "word": "hello", "probability": 0.9},
                           {"start": 0.6, "end": 1.9, "word": " world", "probability": 0.9}]}]
    seen = {}

    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {"transcription": _prose(270), "timings": TIMINGS,
                            "segments": detailed, "duration_sec": 100.0,
                            "anomaly_count": 0, "anomaly_windows": 0, "flagged_segments": []},
    )

    def capture(path, segments, guid, duration=None):
        seen["segments"] = segments
        seen["duration"] = duration
        return TIMINGS, True, {"agree250": 0.6}

    monkeypatch.setattr(app_module, "run_forced_alignment", capture)
    _run(app_module, db, upload_dir)

    assert seen["segments"] is detailed
    assert seen["duration"] == pytest.approx(100.0)


def test_a_requeue_removes_the_stale_alignment_output(app_module, db, upload_dir):
    """Otherwise the retry refines its new transcript against the old attempt's words."""
    guid = str(uuid.uuid4())
    insert_job(db, guid, status="pending")
    aligned = upload_dir / f"{guid}_aligned"
    aligned.mkdir()
    (aligned / f"{guid}.json").write_text("{}")

    app_module.handle_garbage_result(db.cursor(), guid, "anomaly gate")

    assert not aligned.exists()


# --------------------------------------------------------------------------------------
# The additive API fields
# --------------------------------------------------------------------------------------
def _completed_row(route_db, guid, **overrides):
    values = {
        "transcription": "hello world",
        "timings": json.dumps(TIMINGS),
        "anomaly_count": 2,
        "anomaly_windows": 1,
        "flagged_segments": json.dumps(FLAGGED),
    }
    values.update(overrides)
    cur = route_db.cursor()
    cur.execute(
        "INSERT INTO transcriptions (guid, filename, status, transcription, timings, "
        "anomaly_count, anomaly_windows, flagged_segments) "
        "VALUES (?, 'sermon.mp3', 'completed', ?, ?, ?, ?, ?)",
        (guid, values["transcription"], values["timings"], values["anomaly_count"],
         values["anomaly_windows"], values["flagged_segments"]),
    )


def test_status_returns_the_new_fields(client, route_db):
    guid = str(uuid.uuid4())
    _completed_row(route_db, guid)

    body = client.get(f"/status/{guid}").get_json()

    # The 0.5.x contract is untouched.
    assert body["status"] == "completed"
    assert body["transcription"] == "hello world"
    assert body["timings"] == TIMINGS
    # Additive.
    assert body["anomaly_count"] == 2
    assert body["anomaly_windows"] == 1
    assert body["flagged_segments"] == FLAGGED


def test_status_flagged_segments_is_a_list_on_older_rows(client, route_db):
    """A row written before 0.6.0 has a null column; the client should not have to care."""
    guid = str(uuid.uuid4())
    _completed_row(route_db, guid, anomaly_count=None, anomaly_windows=None,
                   flagged_segments=None)

    body = client.get(f"/status/{guid}").get_json()

    assert body["anomaly_count"] is None
    assert body["flagged_segments"] == []


def test_transcriptions_lists_the_new_fields(client, route_db):
    guid = str(uuid.uuid4())
    _completed_row(route_db, guid)

    rows = client.get("/transcriptions").get_json()

    row = next(r for r in rows if r["guid"] == guid)
    assert row["anomaly_count"] == 2
    assert row["anomaly_windows"] == 1
    assert row["flagged_segments"] == FLAGGED


def test_unparseable_flagged_segments_does_not_break_the_response(client, route_db):
    guid = str(uuid.uuid4())
    _completed_row(route_db, guid, flagged_segments="not json")

    body = client.get(f"/status/{guid}").get_json()
    assert body["flagged_segments"] == []


# --------------------------------------------------------------------------------------
# The word-rate floor over speech, and the speech_seconds field
# --------------------------------------------------------------------------------------
def test_the_floor_divides_by_speech_not_total_duration(app_module, db, upload_dir,
                                                        monkeypatch):
    """A recording that is half silence must not read as half rate.

    270 words over 100 s of audio is 2.7 words/sec, but only 60 s of that was speech,
    so the honest rate is 4.5. Dividing by total duration is what forced the floor
    down to 1.0 to be safe on quiet material.
    """
    import uuid

    from conftest import get_row, insert_job

    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {
            "transcription": _prose(270), "timings": TIMINGS, "segments": TIMINGS,
            "duration_sec": 100.0, "speech_seconds": 60.0,
            "anomaly_count": 0, "anomaly_windows": 0, "flagged_segments": [],
        },
    )
    monkeypatch.setattr(app_module, "run_forced_alignment",
                        lambda p, t, g, d=None: (TIMINGS, True, {"agree250": 0.6}))

    guid = str(uuid.uuid4())
    (upload_dir / f"{guid}.mp3").write_bytes(b"fake audio")
    insert_job(db, guid, filename="sermon.mp3", status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, "sermon.mp3") == "completed"
    row = get_row(db, guid)
    assert row["words_per_second"] == pytest.approx(4.5)
    assert row["speech_seconds"] == pytest.approx(60.0)


def test_the_floor_falls_back_to_duration_without_speech_seconds(app_module, db, upload_dir,
                                                                 monkeypatch):
    """An older or stubbed decode reports no duration_after_vad."""
    import uuid

    from conftest import get_row, insert_job

    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {
            "transcription": _prose(270), "timings": TIMINGS, "segments": TIMINGS,
            "duration_sec": 100.0,
            "anomaly_count": 0, "anomaly_windows": 0, "flagged_segments": [],
        },
    )
    monkeypatch.setattr(app_module, "run_forced_alignment",
                        lambda p, t, g, d=None: (TIMINGS, True, {"agree250": 0.6}))

    guid = str(uuid.uuid4())
    (upload_dir / f"{guid}.mp3").write_bytes(b"fake audio")
    insert_job(db, guid, filename="sermon.mp3", status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, "sermon.mp3") == "completed"
    row = get_row(db, guid)
    assert row["words_per_second"] == pytest.approx(2.7)
    assert row["speech_seconds"] is None


def test_a_collapse_over_speech_is_requeued(app_module, db, upload_dir, monkeypatch):
    """60 words over 600 s of actual speech is 0.1 words/sec, well under the floor."""
    import uuid

    from conftest import get_row, insert_job

    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    monkeypatch.setattr(app_module, "MIN_WORDS_PER_SEC", 1.0)
    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {
            "transcription": _prose(60), "timings": TIMINGS, "segments": TIMINGS,
            "duration_sec": 900.0, "speech_seconds": 600.0,
            "anomaly_count": 0, "anomaly_windows": 0, "flagged_segments": [],
        },
    )

    guid = str(uuid.uuid4())
    (upload_dir / f"{guid}.mp3").write_bytes(b"fake audio")
    insert_job(db, guid, filename="sermon.mp3", status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, "sermon.mp3") == "pending"
    assert get_row(db, guid)["attempt_count"] == 1


def test_speech_seconds_reaches_both_endpoints(client, route_db):
    import json
    import uuid

    guid = str(uuid.uuid4())
    route_db.cursor().execute(
        "INSERT INTO transcriptions (guid, filename, status, transcription, timings, "
        "speech_seconds) VALUES (?, 'sermon.mp3', 'completed', 'hello world', ?, 612.5)",
        (guid, json.dumps(TIMINGS)),
    )

    assert client.get(f"/status/{guid}").get_json()["speech_seconds"] == pytest.approx(612.5)
    rows = client.get("/transcriptions").get_json()
    assert next(r for r in rows if r["guid"] == guid)["speech_seconds"] == pytest.approx(612.5)


def test_speech_seconds_is_null_on_older_rows(client, route_db):
    import uuid

    guid = str(uuid.uuid4())
    route_db.cursor().execute(
        "INSERT INTO transcriptions (guid, filename, status, transcription) "
        "VALUES (?, 'sermon.mp3', 'completed', 'hello world')",
        (guid,),
    )
    assert client.get(f"/status/{guid}").get_json()["speech_seconds"] is None


def test_speech_seconds_migrates_onto_an_older_database(app_module):
    import sqlite3

    conn = sqlite3.connect(":memory:", isolation_level=None)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE transcriptions (
            guid TEXT PRIMARY KEY, filename TEXT, status TEXT DEFAULT 'pending',
            transcription TEXT, timings TEXT, created_at TIMESTAMP, completed_at TIMESTAMP,
            processing_time_est INTEGER DEFAULT 0
        )
    """)
    app_module.ensure_schema(cursor)
    assert app_module.column_exists(cursor, "transcriptions", "speech_seconds")
    conn.close()
