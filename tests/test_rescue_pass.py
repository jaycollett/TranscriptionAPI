"""Tests for the anomaly-triggered rescue pass.

The five-pass decode bought redundancy by paying for it on every file. The rescue
buys the same redundancy only where the primary pass shows evidence of trouble: one
extra pass, with previous-text conditioning off and the ladder started above the rung
the primary already failed on, and the better of the two kept on the anomaly score.
"""

import pytest


class _Word:
    def __init__(self, start, end, word, probability=0.9):
        self.start, self.end, self.word, self.probability = start, end, word, probability


class _Seg:
    """A faster-whisper Segment as far as serialize_segment is concerned."""

    def __init__(self, start, end, text, temperature=0.0, compression_ratio=1.4,
                 avg_logprob=-0.06, no_speech_prob=0.01):
        self.id, self.seek = 0, int(start * 100)
        self.start, self.end, self.text = start, end, text
        self.temperature = temperature
        self.compression_ratio = compression_ratio
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        tokens = text.split()
        step = (end - start) / max(len(tokens), 1)
        self.words = [
            _Word(start + i * step, start + (i + 1) * step, (" " if i else "") + t)
            for i, t in enumerate(tokens)
        ]


def _clean(n=20):
    return [_Seg(i * 10.0, i * 10.0 + 10.0,
                 "and so the word of the Lord came to him once again saying")
            for i in range(n)]


def _anomalous(n=20):
    """A clean file with three segments that tripped the temperature ladder."""
    segments = _clean(n)
    for i in (5, 9, 14):
        segments[i].temperature = 0.8
        segments[i].compression_ratio = 3.1
    return segments


@pytest.fixture
def two_pass(transcribe_module, monkeypatch, tmp_path):
    """Drive transcribe_audio with a scripted model; returns the call log and result."""
    def run(pass_segments):
        calls = []

        class _FakeModel:
            def transcribe(self, audio, **kwargs):
                calls.append(kwargs)
                index = min(len(calls) - 1, len(pass_segments) - 1)
                return (pass_segments[index], None)

        monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
        monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 200.0)
        monkeypatch.setattr(transcribe_module, "measure_mean_dbfs", lambda path, duration=0: -20.0)
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        result = transcribe_module.transcribe_audio(str(audio), "guid")
        return calls, result
    return run


# --------------------------------------------------------------------------------------
# The trigger
# --------------------------------------------------------------------------------------
def test_a_clean_primary_pass_does_not_trigger_a_rescue(two_pass):
    calls, result = two_pass([_clean()])
    assert len(calls) == 1, "a healthy file must cost exactly one pass"
    assert result["rescue_attempted"] is False
    assert result["rescue_selected"] is False
    assert result["anomaly_count"] == 0


def test_an_anomalous_primary_pass_triggers_exactly_one_rescue(two_pass):
    calls, result = two_pass([_anomalous(), _clean()])
    assert len(calls) == 2, "never more than one rescue"
    assert result["rescue_attempted"] is True


@pytest.mark.parametrize("count, windows, expected", [
    (0, 0, False),
    (1, 0, True),
    (0, 1, True),
    (3, 2, True),
])
def test_the_trigger_thresholds(transcribe_module, count, windows, expected):
    assert transcribe_module.should_attempt_rescue(count, windows) is expected


def test_the_rescue_can_be_switched_off(transcribe_module, monkeypatch):
    monkeypatch.setattr(transcribe_module, "RESCUE_ENABLED", False)
    assert transcribe_module.should_attempt_rescue(9, 9) is False


def test_the_trigger_is_far_below_the_quarantine_gate(transcribe_module, app_module):
    """A second opinion is cheap and a requeue is not, so the rescue must fire first."""
    assert transcribe_module.RESCUE_ANOMALY_SEGMENTS < app_module.ANOMALY_SEGMENTS_MAX
    assert transcribe_module.RESCUE_ANOMALY_WINDOWS < app_module.ANOMALY_WINDOWS_MAX


# --------------------------------------------------------------------------------------
# The rescue pass's parameters
# --------------------------------------------------------------------------------------
def test_the_rescue_changes_only_the_two_levers(two_pass):
    calls, _result = two_pass([_anomalous(), _clean()])
    primary, rescue = calls

    assert primary["condition_on_previous_text"] is True
    assert rescue["condition_on_previous_text"] is False, \
        "unconditioning is what stops a loop feeding itself across windows"
    assert tuple(rescue["temperature"]) == (0.2, 0.4, 0.6, 0.8, 1.0)
    assert rescue["temperature"][0] > primary["temperature"][0], \
        "repeating the rung the primary failed on cannot help"

    for key in ("beam_size", "best_of", "patience", "vad_filter", "vad_parameters",
                "compression_ratio_threshold", "log_prob_threshold", "no_speech_threshold",
                "word_timestamps", "language", "hallucination_silence_threshold"):
        assert rescue[key] == primary[key], f"{key} must not differ between the passes"


def test_the_rescue_sees_the_same_vad_threshold(two_pass):
    calls, _result = two_pass([_anomalous(), _clean()])
    assert calls[0]["vad_parameters"] == calls[1]["vad_parameters"]


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------
def _record(label, anomaly_count=0, anomaly_windows=0, words=100, mean_logprob=-0.05):
    return {
        "label": label,
        "segments": [],
        "transcript": "",
        "words": words,
        "anomaly_count": anomaly_count,
        "anomaly_windows": anomaly_windows,
        "flagged_segments": [],
        "windows": 3,
        "mean_logprob": mean_logprob,
    }


def test_the_lower_anomaly_score_wins(transcribe_module):
    primary = _record("primary", anomaly_count=3, anomaly_windows=1, words=2000)
    rescue = _record("rescue", anomaly_count=0, anomaly_windows=0, words=1900)
    assert transcribe_module.select_pass([primary, rescue])["label"] == "rescue"
    # And the other way round: fewer anomalies wins even with fewer words.
    primary = _record("primary", anomaly_count=0, anomaly_windows=0, words=1900)
    rescue = _record("rescue", anomaly_count=2, anomaly_windows=0, words=2000)
    assert transcribe_module.select_pass([primary, rescue])["label"] == "primary"


def test_the_score_is_the_sum_of_segments_and_windows(transcribe_module):
    primary = _record("primary", anomaly_count=3, anomaly_windows=0)
    rescue = _record("rescue", anomaly_count=0, anomaly_windows=4)
    assert transcribe_module.select_pass([primary, rescue])["label"] == "primary"


def test_the_first_tie_break_is_word_count(transcribe_module):
    """Every disagreement the 0.5.x confidence rule got wrong preferred fewer words."""
    primary = _record("primary", anomaly_count=1, words=8461, mean_logprob=-0.02)
    rescue = _record("rescue", anomaly_count=1, words=8904, mean_logprob=-0.09)
    assert transcribe_module.select_pass([primary, rescue])["label"] == "rescue"


def test_the_second_tie_break_is_mean_logprob(transcribe_module):
    primary = _record("primary", anomaly_count=1, words=2000, mean_logprob=-0.20)
    rescue = _record("rescue", anomaly_count=1, words=2000, mean_logprob=-0.05)
    assert transcribe_module.select_pass([primary, rescue])["label"] == "rescue"


def test_an_exact_tie_keeps_the_primary(transcribe_module):
    primary = _record("primary")
    rescue = _record("rescue")
    assert transcribe_module.select_pass([primary, rescue])["label"] == "primary"


def test_an_empty_rescue_never_wins(transcribe_module):
    primary = _record("primary", anomaly_count=1, words=2000)
    empty = _record("rescue", anomaly_count=1, words=0, mean_logprob=None)
    assert transcribe_module.select_pass([primary, empty])["label"] == "primary"


def test_the_returned_counts_describe_the_selected_pass(two_pass):
    """The rescue fixes the anomalies; the reported score must be the fixed one."""
    _calls, result = two_pass([_anomalous(), _clean()])
    assert result["rescue_selected"] is True
    assert result["anomaly_count"] == 0
    assert result["flagged_segments"] == []


def test_a_worse_rescue_is_discarded(two_pass):
    """A rescue that makes things worse must not be published."""
    worse = _anomalous()
    for segment in worse:
        segment.temperature = 0.8
    _calls, result = two_pass([_anomalous(), worse])
    assert result["rescue_attempted"] is True
    assert result["rescue_selected"] is False
    assert result["anomaly_count"] == 3


def test_the_invariant_holds_on_the_selected_pass(two_pass):
    _calls, result = two_pass([_anomalous(), _clean()])
    assert " ".join(t["text"] for t in result["timings"]) == result["transcription"]


# --------------------------------------------------------------------------------------
# The gate sees the selected pass, and the fields reach the API
# --------------------------------------------------------------------------------------
def _prose(n):
    return " ".join(f"word{i}" for i in range(n))


TIMINGS = [{"start": 0.0, "end": 2.0, "text": "hello world"}]


def test_the_gate_judges_the_selected_pass_not_the_primary(app_module, db, upload_dir,
                                                           monkeypatch):
    """A file only the rescue could save must complete, not be quarantined."""
    import os
    import uuid

    from conftest import get_row, insert_job

    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {
            "transcription": _prose(270), "timings": TIMINGS, "segments": TIMINGS,
            "duration_sec": 100.0,
            # The primary scored 9 and 4; these are the rescue's numbers.
            "anomaly_count": 0, "anomaly_windows": 0, "flagged_segments": [],
            "rescue_attempted": True, "rescue_selected": True,
        },
    )
    monkeypatch.setattr(app_module, "run_forced_alignment",
                        lambda p, t, g, d=None: (TIMINGS, True, {"agree250": 0.6}))

    guid = str(uuid.uuid4())
    (upload_dir / f"{guid}.mp3").write_bytes(b"fake audio")
    insert_job(db, guid, filename="sermon.mp3", status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, "sermon.mp3") == "completed"
    row = get_row(db, guid)
    assert row["rescue_attempted"] == 1
    assert row["rescue_selected"] == 1
    assert os.path.exists(str(upload_dir / f"{guid}.mp3"))


def test_a_rescue_that_did_not_help_still_gates(app_module, db, upload_dir, monkeypatch):
    import uuid

    from conftest import get_row, insert_job

    monkeypatch.setattr(app_module, "max_garbage_retries", 3)
    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {
            "transcription": _prose(270), "timings": TIMINGS, "segments": TIMINGS,
            "duration_sec": 100.0,
            "anomaly_count": 8, "anomaly_windows": 3, "flagged_segments": [],
            "rescue_attempted": True, "rescue_selected": False,
        },
    )

    guid = str(uuid.uuid4())
    (upload_dir / f"{guid}.mp3").write_bytes(b"fake audio")
    insert_job(db, guid, filename="sermon.mp3", status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, "sermon.mp3") == "pending"
    assert get_row(db, guid)["attempt_count"] == 1


def test_the_rescue_fields_reach_both_endpoints(client, route_db):
    import json
    import uuid

    guid = str(uuid.uuid4())
    route_db.cursor().execute(
        "INSERT INTO transcriptions (guid, filename, status, transcription, timings, "
        "anomaly_count, anomaly_windows, flagged_segments, rescue_attempted, rescue_selected) "
        "VALUES (?, 'sermon.mp3', 'completed', 'hello world', ?, 0, 0, '[]', 1, 1)",
        (guid, json.dumps(TIMINGS)),
    )

    status = client.get(f"/status/{guid}").get_json()
    assert status["rescue_attempted"] is True
    assert status["rescue_selected"] is True

    listing = client.get("/transcriptions").get_json()
    row = next(r for r in listing if r["guid"] == guid)
    assert row["rescue_attempted"] is True
    assert row["rescue_selected"] is True


def test_the_rescue_fields_are_null_on_older_rows(client, route_db):
    import uuid

    guid = str(uuid.uuid4())
    route_db.cursor().execute(
        "INSERT INTO transcriptions (guid, filename, status, transcription) "
        "VALUES (?, 'sermon.mp3', 'completed', 'hello world')",
        (guid,),
    )
    body = client.get(f"/status/{guid}").get_json()
    assert body["rescue_attempted"] is None
    assert body["rescue_selected"] is None


def test_the_rescue_columns_migrate_onto_an_older_database(app_module):
    import sqlite3

    conn = sqlite3.connect(":memory:", isolation_level=None)
    cursor = conn.cursor()
    # The 0.5.x table, before either 0.6.0 migration.
    cursor.execute("""
        CREATE TABLE transcriptions (
            guid TEXT PRIMARY KEY, filename TEXT, status TEXT DEFAULT 'pending',
            transcription TEXT, timings TEXT, created_at TIMESTAMP, completed_at TIMESTAMP,
            processing_time_est INTEGER DEFAULT 0, attempt_count INTEGER DEFAULT 0,
            processing_seconds REAL, words_per_second REAL, mfa_applied INTEGER
        )
    """)
    app_module.ensure_schema(cursor)
    for column in ("rescue_attempted", "rescue_selected"):
        assert app_module.column_exists(cursor, "transcriptions", column)
    conn.close()
