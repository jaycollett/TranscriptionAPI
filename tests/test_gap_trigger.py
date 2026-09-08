"""Tests for the gap rescue trigger added in 0.6.1.

The largest contiguous uncovered stretch of speech was already measured and already
used, as a quarantine gate at 20 s. Tuned as a gate a false positive withholds a
finished transcript, so the threshold sat high and was never evaluated below 10 s.
Tuned as a trigger for a second decode a false positive costs GPU time on a file that
was fine, so the same signal is worth reading at 8 s: on the 32-file fidelity subset
it reaches all seven confirmed content losses there, against three for the low-rate
window, and the 20 s value reaches two.

One measurement, two prices. The two thresholds are separate constants and the tests
below hold them apart, because conflating them is the one way this change could hurt.
"""

import logging

import pytest

from test_rescue_pass import NORMAL_RATE_BODY, TIMINGS, _clean, _prose, _Seg


def _with_gap(gap_s, before=10, after=8):
    """Segments at a healthy word rate with one contiguous hole of `gap_s` in them.

    Every segment carries the normal 2.6 words/sec, so no window drops below the 1.5
    floor: the hole is the only abnormal thing about the file. That is the shape the
    gap trigger exists for and the shape the window check is weakest on, because a
    hole contributes no words but, on a coverage clock, no time either.
    """
    segments = [_Seg(i * 10.0, i * 10.0 + 10.0, NORMAL_RATE_BODY) for i in range(before)]
    resume = before * 10.0 + gap_s
    segments += [_Seg(resume + i * 10.0, resume + i * 10.0 + 10.0, NORMAL_RATE_BODY)
                 for i in range(after)]
    return segments, resume + after * 10.0


@pytest.fixture
def two_pass_vad(transcribe_module, monkeypatch, tmp_path):
    """Drive transcribe_audio with a scripted model and real voice-activity intervals.

    The intervals are the point. Without them `coverage_report` reports a zero gap by
    design, so the gap trigger cannot fire at all: a job whose speech detector failed
    is never rescued on a measurement nobody took.
    """
    def run(pass_segments, speech_end, duration=None):
        calls = []

        class _FakeModel:
            def transcribe(self, audio, **kwargs):
                calls.append(kwargs)
                index = min(len(calls) - 1, len(pass_segments) - 1)
                return (pass_segments[index], None)

        monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
        monkeypatch.setattr(transcribe_module, "get_audio_duration",
                            lambda path: duration or speech_end)
        monkeypatch.setattr(transcribe_module, "measure_mean_dbfs",
                            lambda path, duration=0: -20.0)
        monkeypatch.setattr(transcribe_module, "prepare_audio",
                            lambda path, vad: ("AUDIO", [(0.0, speech_end)], 1.0))
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        return calls, transcribe_module.transcribe_audio(str(audio), "guid")
    return run


@pytest.fixture
def one_pass(transcribe_module, monkeypatch, tmp_path):
    """The same, with no speech intervals: the detector failed and coverage is unknown."""
    def run(segments):
        calls = []

        class _FakeModel:
            def transcribe(self, audio, **kwargs):
                calls.append(kwargs)
                return (segments, None)

        def _boom(path, vad):
            raise RuntimeError("no speech detector")

        monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
        monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 200.0)
        monkeypatch.setattr(transcribe_module, "measure_mean_dbfs",
                            lambda path, duration=0: -20.0)
        monkeypatch.setattr(transcribe_module, "prepare_audio", _boom)
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        return calls, transcribe_module.transcribe_audio(str(audio), "guid")
    return run


# --------------------------------------------------------------------------------------
# The threshold
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("gap_s, expected", [
    (0.0, ()), (7.9, ()), (8.0, ("gap",)), (8.5, ("gap",)), (46.8, ("gap",)),
])
def test_the_gap_trigger_threshold(transcribe_module, gap_s, expected):
    """8.0, just under the 8.5 the subset measured, so a small coverage shift keeps it."""
    assert transcribe_module.rescue_triggers(0, gap_s) == expected


@pytest.mark.parametrize("windows, gap_s, expected", [
    (0, 0.0, ()),
    (1, 0.0, ("window",)),
    (0, 12.0, ("gap",)),
    (1, 12.0, ("window", "gap")),
    (3, 46.8, ("window", "gap")),
])
def test_either_trigger_fires_and_both_are_named(transcribe_module, windows, gap_s,
                                                 expected):
    assert transcribe_module.rescue_triggers(windows, gap_s) == expected
    assert transcribe_module.should_attempt_rescue(windows, gap_s) is bool(expected)


def test_the_window_trigger_still_works_without_a_gap_argument(transcribe_module):
    """0.6.0 called this with one argument; the gap is an addition, not a replacement."""
    assert transcribe_module.should_attempt_rescue(1) is True
    assert transcribe_module.should_attempt_rescue(0) is False


def test_the_kill_switch_stops_both_triggers(transcribe_module, monkeypatch):
    monkeypatch.setattr(transcribe_module, "RESCUE_ENABLED", False)
    assert transcribe_module.rescue_triggers(9, 600.0) == ()
    assert transcribe_module.should_attempt_rescue(9, 600.0) is False


def test_a_zero_threshold_switches_the_gap_trigger_off_rather_than_on(transcribe_module,
                                                                     monkeypatch):
    """A bare `>=` would fire on every file, including files with no gap at all."""
    monkeypatch.setattr(transcribe_module, "RESCUE_MAX_GAP_S", 0.0)
    assert transcribe_module.rescue_triggers(0, 0.0) == ()
    assert transcribe_module.rescue_triggers(0, 600.0) == ()
    assert transcribe_module.rescue_triggers(1, 600.0) == ("window",)


def test_the_quarantine_threshold_is_not_the_trigger_threshold(transcribe_module):
    """One measurement, two prices. Conflating them is the whole risk of this change.

    ANOMALY_MAX_UNCOVERED_GAP_SEC feeds the anomaly score. Nothing quarantines on that
    score any more, but 20 s is still the number chosen for a signal that could
    withhold a transcript. RESCUE_MAX_GAP_S buys one more decode and nothing else.
    They have to be able to move independently.
    """
    assert transcribe_module.ANOMALY_MAX_UNCOVERED_GAP_SEC == 20.0
    assert transcribe_module.RESCUE_MAX_GAP_S == 8.0
    assert (transcribe_module.RESCUE_MAX_GAP_S
            < transcribe_module.ANOMALY_MAX_UNCOVERED_GAP_SEC)


# --------------------------------------------------------------------------------------
# End to end through the decode
# --------------------------------------------------------------------------------------
def test_a_long_gap_fires_exactly_one_rescue(two_pass_vad):
    segments, end = _with_gap(12.0)
    calls, result = two_pass_vad([segments, _clean()], end)
    assert result["primary_anomaly_windows"] == 0, "the window check must not see this"
    assert result["primary_uncovered_max_gap_s"] == pytest.approx(12.0, abs=0.05)
    assert result["rescue_attempted"] is True
    assert result["rescue_triggers"] == "gap"
    assert len(calls) == 2, "never more than one rescue, whichever trigger fired"


def test_a_short_gap_at_a_healthy_rate_costs_one_pass(two_pass_vad):
    segments, end = _with_gap(6.0)
    calls, result = two_pass_vad([segments], end)
    assert result["rescue_attempted"] is False
    assert result["rescue_triggers"] == ""
    assert len(calls) == 1


def test_the_gap_trigger_reads_the_primary_not_the_selected_pass(two_pass_vad):
    """The rescue can close the gap; the decision was already taken on the primary."""
    clean = [_Seg(i * 10.0, i * 10.0 + 10.0, NORMAL_RATE_BODY) for i in range(19)]
    segments, end = _with_gap(12.0)
    _calls, result = two_pass_vad([segments, clean], end)
    assert result["primary_uncovered_max_gap_s"] == pytest.approx(12.0, abs=0.05)
    assert result["rescue_attempted"] is True


def test_both_triggers_are_reported_when_both_fire(two_pass_vad):
    """A dropped passage that is both thinly covered and partly uncovered."""
    segments, end = _with_gap(30.0)
    for i in (2, 3, 4, 5, 6, 7):
        segments[i] = _Seg(i * 10.0, i * 10.0 + 10.0, "and then")
    calls, result = two_pass_vad([segments, _clean()], end)
    assert result["rescue_triggers"] == "window+gap"
    assert len(calls) == 2


def test_a_failed_detector_disables_the_gap_trigger_rather_than_firing_it(one_pass):
    """No intervals means no coverage measurement, so the gap reads 0.0, not infinity."""
    calls, result = one_pass(_clean())
    assert result["primary_uncovered_max_gap_s"] == 0.0
    assert result["rescue_attempted"] is False
    assert len(calls) == 1


# --------------------------------------------------------------------------------------
# It is a trigger and nothing else
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("gap_s", [8.0, 12.0, 20.0, 46.79, 600.0, 100000.0])
@pytest.mark.parametrize("rescue_attempted, rescue_selected",
                         [(True, False), (True, True)])
def test_no_gap_value_alone_can_produce_a_terminal_empty_row(app_module, db, upload_dir,
                                                             monkeypatch, gap_s,
                                                             rescue_attempted,
                                                             rescue_selected):
    """The 20 s gap value was a quarantine input. This one must never be.

    If any gap value could quarantine a job, requeue it, or complete it with nothing
    in the row, the change would have imported the gate's price into the trigger,
    which is the exact confusion this release exists to avoid. Every value here
    publishes the transcript the decode produced.
    """
    import uuid

    from conftest import get_row, insert_job

    monkeypatch.setattr(
        app_module, "transcribe_audio",
        lambda path, guid: {
            "transcription": _prose(270), "timings": TIMINGS, "segments": TIMINGS,
            "duration_sec": 100.0, "speech_seconds": 100.0,
            "anomaly_count": 0, "anomaly_windows": 0, "flagged_segments": [],
            "rescue_attempted": rescue_attempted, "rescue_selected": rescue_selected,
            "rescue_triggers": "gap", "rescue_unpublished_run": 0,
            "uncovered_max_gap_s": gap_s, "primary_uncovered_max_gap_s": gap_s,
        },
    )
    monkeypatch.setattr(app_module, "run_forced_alignment",
                        lambda p, t, g, d=None: (TIMINGS, True, {"agree250": 0.6}))

    guid = str(uuid.uuid4())
    (upload_dir / f"{guid}.mp3").write_bytes(b"fake audio")
    insert_job(db, guid, filename="sermon.mp3", status="pending")

    assert app_module.process_pending_job(db.cursor(), guid, "sermon.mp3") == "completed"
    row = get_row(db, guid)
    assert row["status"] == "completed"
    assert row["transcription"] == _prose(270), "the row keeps the transcript it had"
    assert row["attempt_count"] == 0, "no gap value may spend a retry"


def test_the_worker_never_reads_the_gap_measurement_at_all(app_module):
    """The trigger lives inside the decode. The job's control flow never sees it.

    A grep in a test, because the failure this guards against is somebody later
    reaching for the gap in the quarantine path, where it was never priced to be.
    """
    import inspect

    source = inspect.getsource(app_module)
    assert "RESCUE_MAX_GAP_S" not in source
    assert "uncovered_max_gap_s" not in source


def test_a_gap_triggered_rescue_that_is_refused_still_publishes(two_pass_vad):
    """Firing more often is not recovering more, and it must not be losing more.

    The rescue runs, comes back short, the retention floor discards it and the primary
    is published unchanged. That is the measured outcome on at least one recording in
    the fidelity subset, where the rescue already runs with both levers at their best
    values and is correctly refused.
    """
    segments, end = _with_gap(12.0)
    truncated = segments[:12]
    _calls, result = two_pass_vad([segments, truncated], end)
    assert result["rescue_attempted"] is True
    assert result["rescue_selected"] is False
    assert result["transcription"], "a refused rescue never empties the transcript"
    assert result["primary_words"] == len(result["transcription"].split())


# --------------------------------------------------------------------------------------
# The directional cross-check
# --------------------------------------------------------------------------------------
def test_the_cross_check_reports_the_longest_run_only_the_rescue_has(transcribe_module):
    published = "alpha bravo charlie delta echo foxtrot"
    candidate = "alpha bravo one two three four five charlie delta echo foxtrot"
    assert transcribe_module.longest_unpublished_run(published, candidate) == 5


def test_the_cross_check_is_directional(transcribe_module):
    """A run the published pass has and the second decode lacks is not evidence.

    The largest one-sided run in the 32-file experiment was 137 words the second
    decode did not have, and it was a repetition loop. Symmetric disagreement would
    have scored that file as the worst in the corpus.
    """
    published = "alpha bravo the sins we commit the sins we commit charlie"
    candidate = "alpha bravo charlie"
    assert transcribe_module.longest_unpublished_run(published, candidate) == 0


def test_the_cross_check_counts_a_replaced_passage_not_only_a_missing_one(
        transcribe_module):
    published = "alpha bravo mumble charlie delta"
    candidate = "alpha bravo for the word of God is living and active charlie delta"
    assert transcribe_module.longest_unpublished_run(published, candidate) == 9


def test_the_cross_check_ignores_case_and_punctuation(transcribe_module):
    published = "Alpha, bravo; charlie."
    candidate = "alpha bravo charlie"
    assert transcribe_module.longest_unpublished_run(published, candidate) == 0


def test_the_cross_check_handles_an_empty_side(transcribe_module):
    assert transcribe_module.longest_unpublished_run("", "alpha bravo charlie") == 3
    assert transcribe_module.longest_unpublished_run("alpha bravo", "") == 0
    assert transcribe_module.longest_unpublished_run("", "") == 0
    assert transcribe_module.longest_unpublished_run(None, None) == 0


def test_the_cross_check_is_none_when_no_second_decode_ran(two_pass_vad):
    """It is read off two decodes that already exist. It never causes one."""
    segments, end = _with_gap(6.0)
    calls, result = two_pass_vad([segments], end)
    assert len(calls) == 1
    assert result["rescue_unpublished_run"] is None


def test_the_cross_check_survives_the_rescue_being_refused(two_pass_vad):
    """The case it exists for, and the reason it is measured against the published pass.

    The rescue fills the hole with a passage the primary never had, and loses more
    elsewhere than the retention floor allows, so it is discarded and the primary is
    published. The transcript is unchanged and the log still says a second decode
    found sixteen words nobody can read in the published file. That is a review
    marker, not a decision.
    """
    passage = ("and Moses wrote all the words of the Lord "
               "and rose up early in the morning")
    segments, end = _with_gap(12.0)
    rescue = segments[:6] + [_Seg(100.0, 112.0, passage)] + segments[10:]
    _calls, result = two_pass_vad([segments, rescue], end)
    assert result["rescue_attempted"] is True
    assert result["rescue_selected"] is False, "the retention floor must refuse this one"
    assert result["rescue_unpublished_run"] >= 12


def test_the_cross_check_is_zero_when_the_rescue_was_published(two_pass_vad):
    """Nothing the published pass lacks, because the published pass is that decode."""
    segments, end = _with_gap(12.0)
    longer = segments + [_Seg(end, end + 10.0, NORMAL_RATE_BODY)]
    _calls, result = two_pass_vad([segments, longer], end)
    assert result["rescue_selected"] is True
    assert result["rescue_unpublished_run"] == 0


def test_the_triggers_and_the_cross_check_reach_the_completion_line(two_pass_vad,
                                                                   caplog):
    """The firing mix has to be readable off production logs, not inferred."""
    segments, end = _with_gap(12.0)
    with caplog.at_level(logging.INFO):
        two_pass_vad([segments, _clean()], end)
    completion = [r.message for r in caplog.records
                  if r.message.startswith("Transcription completed for GUID")]
    assert completion, "the completion line is where the mix is observed"
    assert "rescue_triggers=gap" in completion[-1]
    assert "rescue_unpublished_run=" in completion[-1]
