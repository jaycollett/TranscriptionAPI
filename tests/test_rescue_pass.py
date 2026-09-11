"""Tests for the anomaly-triggered rescue pass.

The five-pass decode bought redundancy by paying for it on every file. The rescue
buys the same redundancy only where the primary pass shows evidence of trouble: one
extra pass, with previous-text conditioning off and the ladder started above the rung
the primary already failed on, and the better of the two kept on the anomaly score.
"""

import json

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


# 26 words per 10 s segment, i.e. 2.6 words/sec, the measured healthy rate. The floor
# is 1.5, so a fixture at the old 1.3 would read as an omission.
NORMAL_RATE_BODY = (
    "and so the word of the Lord came to him once again saying "
    "behold I will send my messenger before your face to prepare the way"
)


def _clean(n=20):
    return [_Seg(i * 10.0, i * 10.0 + 10.0, NORMAL_RATE_BODY) for i in range(n)]


def _anomalous(n=20):
    """A file with a dropped passage: the shape that now fires the rescue.

    Segments 8-13 carry almost no words over 60 s of speech. Their per-segment fields
    are entirely normal, which is the point: a confident omission looks fine to every
    signal except the word rate.
    """
    segments = _clean(n)
    for i in (8, 9, 10, 11, 12, 13):
        segments[i] = _Seg(i * 10.0, i * 10.0 + 10.0, "and then")
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


def test_a_rescue_that_is_still_anomalous_does_not_trigger_another(two_pass):  # noqa: E501
    """A `while should_attempt_rescue(...)` regression would hang the worker.

    The assertion is the point: the failure has to be a red test, not a wedged
    thread with no log line.
    """
    calls, result = two_pass([_anomalous(), _anomalous()])
    assert len(calls) == 2, "the rescue must never itself trigger a rescue"
    assert result["rescue_attempted"] is True


@pytest.mark.parametrize("windows, expected", [(0, False), (1, True), (3, True)])
def test_the_trigger_thresholds(transcribe_module, windows, expected):
    assert transcribe_module.should_attempt_rescue(windows) is expected


def test_the_segment_count_no_longer_triggers_anything(transcribe_module):
    """Measured across 102 recordings: it never exceeded 1, identified none of the
    seven confirmed bad transcripts, and every one of the six files it flagged was
    healthy. It is a diagnostic, not a decision."""
    assert transcribe_module.should_attempt_rescue(0) is False


def test_the_rescue_can_be_switched_off(transcribe_module, monkeypatch):
    monkeypatch.setattr(transcribe_module, "RESCUE_ENABLED", False)
    assert transcribe_module.should_attempt_rescue(9) is False


def test_a_single_low_window_is_enough_to_fire_the_rescue(transcribe_module):
    """It is the only signal that can see a confident omission, so it must reach the fix.

    The model drops read-aloud passages without crossing the compression-ratio or
    log-probability thresholds, so no segment looks anomalous and no temperature rung
    above the first is ever tried. A stretch of speech carrying almost no words is the
    whole of the evidence, and the rescue is the remedy.
    """
    assert transcribe_module.should_attempt_rescue(1) is True


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
    """Within the retention floor, the anomaly score decides."""
    primary = _record("primary", anomaly_count=3, anomaly_windows=1, words=2000)
    rescue = _record("rescue", anomaly_count=0, anomaly_windows=0, words=1990)
    assert transcribe_module.select_pass([primary, rescue])["label"] == "rescue"
    # And the other way round: fewer anomalies wins even with fewer words.
    primary = _record("primary", anomaly_count=0, anomaly_windows=0, words=1990)
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
    _calls, result = two_pass([_anomalous(), _clean(21)])
    assert result["rescue_selected"] is True
    assert result["anomaly_count"] == 0
    assert result["flagged_segments"] == []


def _with_dropped_passage(n=20):
    """A clean decode with one 60 s stretch the model silently declined to transcribe."""
    return _anomalous(n)


def test_a_dropped_passage_fires_the_rescue_and_the_recovery_is_published(two_pass):
    """The whole read-aloud path, end to end.

    The primary silently drops a passage, which shows up only as a low-rate window.
    That fires the rescue, the rescue recovers the passage, and the selection rules
    have to prefer it: its low-window count falls so it wins on anomaly total, and it
    gains words rather than losing them so neither the retention floor nor the 40 word
    cap can block it.
    """
    calls, result = two_pass([_with_dropped_passage(), _clean()])

    assert len(calls) == 2, "the low-rate window must fire the rescue"
    assert result["rescue_attempted"] is True
    assert result["rescue_selected"] is True, "the recovery must be the published pass"
    assert result["anomaly_windows"] == 0, "the recovered pass has no low window left"
    assert len(result["transcription"].split()) > 400
    assert " ".join(t["text"] for t in result["timings"]) == result["transcription"]


def test_the_recovery_is_not_blocked_by_the_retention_floor(transcribe_module):
    """A rescue that gains words is always eligible; the floor only stops losses."""
    primary = _record("primary", anomaly_windows=2, words=3900)
    rescue = _record("rescue", anomaly_windows=0, words=4180)
    assert transcribe_module.retains_enough_words(primary, rescue) is True
    assert transcribe_module.select_pass([primary, rescue])["label"] == "rescue"


def test_a_rescue_that_does_not_recover_leaves_the_primary(two_pass):
    """The two question-and-answer stretches: audience audio that is simply not there.

    Nothing recovers those, and the rescue must not be able to make things worse by
    firing on them.
    """
    dropped = _with_dropped_passage()
    calls, result = two_pass([dropped, _with_dropped_passage()])

    assert len(calls) == 2
    assert result["rescue_attempted"] is True
    assert result["rescue_selected"] is False, "an equal rescue never displaces the primary"
    assert result["anomaly_windows"] >= 1, "and the window count is still reported"


def test_a_worse_rescue_is_discarded(two_pass):
    """A rescue that makes things worse must not be published.

    Both passes drop the same passage, so the scores tie and the primary is kept.
    """
    _calls, result = two_pass([_anomalous(), _anomalous()])
    assert result["rescue_attempted"] is True
    assert result["rescue_selected"] is False
    assert result["anomaly_windows"] >= 1


def test_the_invariant_holds_on_the_selected_pass(two_pass):
    _calls, result = two_pass([_anomalous(), _clean()])
    assert " ".join(t["text"] for t in result["timings"]) == result["transcription"]


# --------------------------------------------------------------------------------------
# The word-retention floor
# --------------------------------------------------------------------------------------
def test_a_rescue_that_truncates_is_discarded(transcribe_module):
    """Measured on tcf.20240319b with the trigger forced on.

    The primary scored one anomalous segment out of 323 with 3599 words; the rescue
    scored none with 3525. On the anomaly score alone the rescue wins and the service
    publishes 74 fewer words, which is the same failure the retired confidence rule
    made on the retreat file. The retention floor keeps the primary.
    """
    primary = _record("primary", anomaly_count=1, words=3599, mean_logprob=-0.162)
    rescue = _record("rescue", anomaly_count=0, words=3525, mean_logprob=-0.118)
    assert transcribe_module.select_pass([primary, rescue])["label"] == "primary"


def test_a_rescue_within_the_floor_still_wins_on_anomalies(transcribe_module):
    """The floor is a guard, not a veto: a rescue that fixes anomalies cheaply wins."""
    primary = _record("primary", anomaly_count=4, words=3599)
    rescue = _record("rescue", anomaly_count=0, words=3580)  # 0.5 percent loss
    assert transcribe_module.select_pass([primary, rescue])["label"] == "rescue"


def test_a_rescue_with_more_words_is_always_eligible(transcribe_module):
    primary = _record("primary", anomaly_count=2, words=3500)
    rescue = _record("rescue", anomaly_count=0, words=3700)
    assert transcribe_module.select_pass([primary, rescue])["label"] == "rescue"


@pytest.mark.parametrize("rescue_words, eligible", [
    (10000, True),
    (9960, True),    # exactly at the 40 word cap
    (9959, False),   # inside the 99 percent ratio, past the cap
    (9900, False),
    (5000, False),
])
def test_the_retention_floor_boundary(transcribe_module, rescue_words, eligible):
    """The cap binds on a long file, where 1 percent is far more than 40 words."""
    primary = _record("primary", words=10000)
    rescue = _record("rescue", words=rescue_words)
    assert transcribe_module.retains_enough_words(primary, rescue) is eligible


@pytest.mark.parametrize("primary_words, rescue_words, eligible", [
    (400, 397, True),    # 0.75 percent of a short clip, 3 words
    (400, 395, False),   # 1.25 percent, inside the cap but past the ratio
])
def test_the_ratio_binds_on_a_short_file(transcribe_module, primary_words, rescue_words,
                                         eligible):
    """On a short clip the 40 word cap is no constraint, so the ratio has to be there."""
    primary = _record("primary", words=primary_words)
    rescue = _record("rescue", words=rescue_words)
    assert transcribe_module.retains_enough_words(primary, rescue) is eligible


def test_the_retreat_sized_loss_the_ratio_alone_would_have_allowed(transcribe_module):
    """1 percent of 8904 words is 89, more than the two Psalm 91 runs that started this."""
    primary = _record("primary", anomaly_count=2, words=8904)
    rescue = _record("rescue", anomaly_count=0, words=8815)
    assert transcribe_module.retains_enough_words(primary, rescue) is False
    assert transcribe_module.select_pass([primary, rescue])["label"] == "primary"


def test_an_empty_primary_does_not_divide_by_zero(transcribe_module):
    primary = _record("primary", words=0)
    rescue = _record("rescue", words=0)
    assert transcribe_module.retains_enough_words(primary, rescue) is True


def test_the_discarded_rescue_is_logged(transcribe_module, caplog):
    """A silently discarded second decode is a fault nobody would ever find."""
    import logging

    primary = _record("primary", anomaly_count=1, words=3599)
    rescue = _record("rescue", anomaly_count=0, words=3525)
    with caplog.at_level(logging.WARNING):
        transcribe_module.select_pass([primary, rescue])
    assert "retention floor" in caplog.text
    assert "3525" in caplog.text and "3599" in caplog.text


def test_the_forced_213b_case_keeps_the_primary(transcribe_module):
    """Measured on tcf.20240213b with the trigger forced on: a tie on anomalies.

    Both passes scored zero; the rescue had 2085 words against 2104. The word-count
    tie-break alone already keeps the primary here, and the retention floor is a
    second line for the case where the rescue also scores better.
    """
    primary = _record("primary", anomaly_count=0, words=2104, mean_logprob=-0.136)
    rescue = _record("rescue", anomaly_count=0, words=2085, mean_logprob=-0.109)
    assert transcribe_module.select_pass([primary, rescue])["label"] == "primary"


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


def test_a_rescue_that_did_not_help_still_publishes(app_module, db, upload_dir, monkeypatch):
    """The rescue is the only thing an anomaly drives, and it gets exactly one go.

    When it does not recover the passage the primary is kept and the file is
    published with its counts, because a flagged transcript beats no transcript and
    nothing about these signals is reliable enough to throw work away on.
    """
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
    monkeypatch.setattr(app_module, "run_forced_alignment",
                        lambda p, t, g, d=None: (TIMINGS, True, {"agree250": 0.6}))

    guid = str(uuid.uuid4())
    (upload_dir / f"{guid}.mp3").write_bytes(b"fake audio")
    insert_job(db, guid, filename="sermon.mp3", status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, "sermon.mp3") == "completed"
    row = get_row(db, guid)
    assert row["attempt_count"] == 0
    assert row["anomaly_windows"] == 3
    assert row["rescue_attempted"] == 1
    assert row["rescue_selected"] == 0


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


# ---------------------------------------------------------------------------------
# Keeping the pass that was not published
# ---------------------------------------------------------------------------------
def _pass(label, words, text):
    return {
        "label": label,
        "words": words,
        "transcript": text,
        "anomaly_count": 0,
        "anomaly_windows": 0,
        "mean_logprob": -0.3,
        "uncovered_s": 1.0,
        "uncovered_max_gap_s": 1.0,
    }


def test_retain_both_passes_is_off_by_default(transcribe_module, tmp_path, monkeypatch):
    """Production sets no directory, so nothing is written and nothing is touched."""
    monkeypatch.setattr(transcribe_module, "RESCUE_TRANSCRIPT_DIR", "")
    primary = _pass("primary", 3, "one two three")
    rescue = _pass("rescue", 2, "one two")
    transcribe_module.retain_both_passes("g", "/audio/x.mp3", [primary, rescue], primary)
    assert list(tmp_path.iterdir()) == []


def test_retain_both_passes_keeps_the_discarded_transcript(
    transcribe_module, tmp_path, monkeypatch
):
    """The refused rescue's text is the thing no other record holds, so it is the
    thing this has to survive with."""
    monkeypatch.setattr(transcribe_module, "RESCUE_TRANSCRIPT_DIR", str(tmp_path))
    primary = _pass("primary", 3, "one two three")
    rescue = _pass("rescue", 2, "one two")
    transcribe_module.retain_both_passes("abc-123", "/audio/x.mp3", [primary, rescue], primary)

    written = json.loads((tmp_path / "abc-123.json").read_text())
    assert written["guid"] == "abc-123"
    assert written["file"] == "x.mp3"
    assert written["published"] == "primary"
    assert [p["label"] for p in written["passes"]] == ["primary", "rescue"]
    assert [p["transcription"] for p in written["passes"]] == ["one two three", "one two"]
    assert [p["words"] for p in written["passes"]] == [3, 2]


def test_retain_both_passes_never_fails_the_job(transcribe_module, tmp_path, monkeypatch):
    """A review facility that can lose a transcript is worse than no review facility."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setattr(transcribe_module, "RESCUE_TRANSCRIPT_DIR", str(blocker))
    primary = _pass("primary", 3, "one two three")
    transcribe_module.retain_both_passes("g", "/audio/x.mp3", [primary], primary)
