"""Tests for run_forced_alignment and the pieces it is built from.

0.6.0 replaced the whole-file alignment with the harness's I1 + I2 + I5 + I6 path:
utterances built from Whisper segments instead of one utterance per file, a 16 kHz
mono WAV instead of a source-rate pydub export, one MFA attempt at default beams
instead of a 40/100 then 100/400 ladder, and difflib word assignment instead of a
start-time window. On the six reference files that turned one outright failure and
one second-attempt rescue into six first-attempt successes, cut MFA wall time from
36-279 s to 24-35 s, and raised agree250 from 0.44-0.65 to 0.52-0.79 with no
non-monotonic or overlapping segments.

No MFA binary and no ffmpeg: subprocess.run is stubbed.
"""

import json
import logging
import os
import subprocess
import uuid

import pytest

SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "hello world"},
    {"start": 2.5, "end": 5.0, "text": "second segment"},
]


def _worded(start, end, text):
    """A segment carrying word timestamps, as the decode now hands them over."""
    tokens = text.split()
    step = (end - start) / len(tokens)
    return {
        "start": start,
        "end": end,
        "text": text,
        "words": [
            {"start": round(start + i * step, 4), "end": round(start + (i + 1) * step, 4),
             "word": (" " if i else "") + t, "probability": 0.9}
            for i, t in enumerate(tokens)
        ],
    }


@pytest.fixture
def alignment_env(app_module, upload_dir, monkeypatch):
    guid = str(uuid.uuid4())
    audio = upload_dir / f"{guid}.mp3"
    audio.write_bytes(b"fake audio")
    return guid, str(audio)


def _mfa_output(guid, upload_dir, entries):
    out_dir = upload_dir / f"{guid}_aligned"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{guid}.json").write_text(json.dumps({"tiers": {"words": {"entries": entries}}}))


def _stub_subprocess(monkeypatch, app_module, guid, upload_dir, entries=None, mfa_rc=0,
                     commands=None):
    """Stand in for both the ffmpeg WAV export and the MFA run."""
    def fake_run(cmd, **kwargs):
        if commands is not None:
            commands.append(cmd)
        if cmd[0] == "ffmpeg":
            # The corpus directory exists by now; produce the file MFA would read.
            with open(cmd[-1], "wb") as fh:
                fh.write(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if mfa_rc != 0:
            raise subprocess.CalledProcessError(mfa_rc, cmd, output="", stderr="beam too narrow")
        if entries is not None:
            _mfa_output(guid, upload_dir, entries)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)


# --------------------------------------------------------------------------------------
# Utterance construction (I1)
# --------------------------------------------------------------------------------------
def test_utterances_split_only_at_real_gaps(app_module):
    """Segments separated by less than the gap threshold stay in one utterance."""
    segments = [
        {"start": 0.0, "end": 5.0, "text": "one"},
        {"start": 5.1, "end": 10.0, "text": "two"},     # 0.1 s gap: no split
        {"start": 12.0, "end": 17.0, "text": "three"},  # 2.0 s gap after 10 s: split
    ]
    utterances = app_module.build_utterances(segments, 20.0)
    assert len(utterances) == 2
    assert utterances[0]["text"] == "one two"
    assert utterances[1]["text"] == "three"
    assert utterances[0]["segments"] == [0, 1]
    assert utterances[1]["segments"] == [2]


def test_a_short_utterance_is_not_split_at_a_gap(app_module):
    """Under the 8 s minimum a qualifying gap is not enough to split."""
    segments = [
        {"start": 0.0, "end": 2.0, "text": "one"},
        {"start": 3.0, "end": 5.0, "text": "two"},
    ]
    assert len(app_module.build_utterances(segments, 10.0)) == 1


def test_utterances_are_padded_but_never_overlap(app_module):
    segments = [
        {"start": 1.0, "end": 10.0, "text": "one"},
        {"start": 10.5, "end": 20.0, "text": "two"},
    ]
    utterances = app_module.build_utterances(segments, 25.0, gap_s=0.4, min_len=1.0)
    assert len(utterances) == 2
    assert utterances[0]["start"] < 1.0, "padded into the leading silence"
    assert utterances[1]["end"] > 20.0
    assert utterances[1]["start"] >= utterances[0]["end"]
    assert utterances[0]["start"] >= 0.0
    assert utterances[-1]["end"] <= 25.0


def test_a_long_run_is_split_at_the_max_length(app_module):
    """A 30 s ceiling keeps one alignment graph from becoming the whole file again."""
    segments = [{"start": i * 5.0, "end": i * 5.0 + 4.5, "text": f"segment {i}"}
                for i in range(20)]
    utterances = app_module.build_utterances(segments, 100.0)
    assert len(utterances) > 1
    assert max(u["end"] - u["start"] for u in utterances) <= 31.0


def test_empty_segments_are_left_out_of_the_utterances(app_module):
    segments = [
        {"start": 0.0, "end": 2.0, "text": "one"},
        {"start": 2.0, "end": 3.0, "text": "   "},
        {"start": 3.0, "end": 5.0, "text": "two"},
    ]
    utterances = app_module.build_utterances(segments, 6.0)
    assert utterances[0]["text"] == "one two"


def test_textgrid_tiles_the_whole_file(app_module, tmp_path):
    """MFA reads the empty intervals as the silence it must not align through."""
    utterances = [{"start": 1.0, "end": 5.0, "text": 'he said "peace"', "segments": [0]}]
    path = tmp_path / "a.TextGrid"
    app_module.write_textgrid(utterances, 10.0, str(path))
    content = path.read_text()

    assert 'Object class = "TextGrid"' in content
    assert 'name = "speaker"' in content
    assert "xmax = 10.0000" in content
    # Leading silence, the utterance, trailing silence.
    assert "intervals: size = 3" in content
    # Quotes in the transcript are doubled, per the ooTextFile format.
    assert 'text = "he said ""peace"""' in content


# --------------------------------------------------------------------------------------
# Word assignment (I2)
# --------------------------------------------------------------------------------------
def test_words_are_assigned_by_sequence_not_by_time(app_module):
    """The window rule assigned by start time, so drift moved words to the wrong segment.

    Here MFA's timings are shifted a second late; the sequence match still puts each
    word in the segment whose text it came from.
    """
    segments = [_worded(0.0, 2.0, "the Lord is good"), _worded(2.0, 4.0, "and his mercy endures")]
    mfa_words = [
        {"start": 1.0, "end": 1.2, "text": "the"},
        {"start": 1.2, "end": 1.5, "text": "Lord"},
        {"start": 1.5, "end": 1.7, "text": "is"},
        {"start": 1.7, "end": 2.1, "text": "good"},
        {"start": 3.0, "end": 3.2, "text": "and"},
        {"start": 3.2, "end": 3.4, "text": "his"},
        {"start": 3.4, "end": 3.7, "text": "mercy"},
        {"start": 3.7, "end": 4.2, "text": "endures"},
    ]
    refined, rstats = app_module.refine_segment_timings(segments, mfa_words)

    assert rstats["empty_fallbacks"] == 0
    assert refined[0]["start"] == pytest.approx(1.0)
    assert refined[0]["end"] == pytest.approx(2.1)
    assert refined[1]["start"] == pytest.approx(3.0)
    assert refined[1]["end"] == pytest.approx(4.2)


def test_a_boundary_word_is_owned_by_one_segment_only(app_module):
    """The window rule gave a word on the seam to both neighbours."""
    segments = [_worded(0.0, 1.0, "alpha beta"), _worded(1.0, 2.0, "gamma delta")]
    mfa_words = [
        {"start": 0.0, "end": 0.5, "text": "alpha"},
        {"start": 0.5, "end": 1.0, "text": "beta"},
        {"start": 1.0, "end": 1.5, "text": "gamma"},
        {"start": 1.5, "end": 2.0, "text": "delta"},
    ]
    refined, rstats = app_module.refine_segment_timings(segments, mfa_words)
    assert refined[0]["end"] == pytest.approx(1.0)
    assert refined[1]["start"] == pytest.approx(1.0)


def test_case_and_punctuation_do_not_break_the_match(app_module):
    segments = [_worded(0.0, 2.0, "The Lord, is good!")]
    mfa_words = [
        {"start": 0.1, "end": 0.3, "text": "the"},
        {"start": 0.3, "end": 0.6, "text": "lord"},
        {"start": 0.6, "end": 0.8, "text": "is"},
        {"start": 0.8, "end": 1.4, "text": "good"},
    ]
    refined, rstats = app_module.refine_segment_timings(segments, mfa_words)
    assert rstats["empty_fallbacks"] == 0
    assert refined[0]["start"] == pytest.approx(0.1)
    assert refined[0]["end"] == pytest.approx(1.4)


def test_uncovered_segment_falls_back_to_whisper_word_timestamps(app_module):
    """Not to the segment bounds: those carry the VAD's 300 ms of padding."""
    covered = _worded(0.0, 2.0, "alpha beta")
    uncovered = _worded(2.0, 6.0, "gamma delta")
    # Whisper's own words for the uncovered segment run 2.0-4.0, inside the 2.0-6.0 span.
    uncovered["words"][0]["start"] = 2.4
    uncovered["words"][-1]["end"] = 5.2

    mfa_words = [
        {"start": 0.1, "end": 0.5, "text": "alpha"},
        {"start": 0.5, "end": 1.1, "text": "beta"},
    ]
    refined, rstats = app_module.refine_segment_timings([covered, uncovered], mfa_words)

    assert rstats["empty_fallbacks"] == 1
    assert refined[1]["start"] == pytest.approx(2.4)
    assert refined[1]["end"] == pytest.approx(5.2)


def test_monotonicity_is_enforced(app_module):
    """MFA can hand back a word later than the next segment's first word."""
    segments = [_worded(0.0, 2.0, "alpha beta"), _worded(2.0, 4.0, "gamma delta")]
    mfa_words = [
        {"start": 0.0, "end": 3.5, "text": "alpha"},
        {"start": 0.5, "end": 3.9, "text": "beta"},
        {"start": 1.0, "end": 1.5, "text": "gamma"},   # starts before segment 0 ended
        {"start": 1.5, "end": 2.0, "text": "delta"},
    ]
    refined, rstats = app_module.refine_segment_timings(segments, mfa_words)

    assert refined[1]["start"] >= refined[0]["end"]
    for entry in refined:
        assert entry["end"] > entry["start"]


def test_agreement_is_measured_against_whisper_word_timestamps(app_module):
    segments = [_worded(0.0, 2.0, "alpha beta"), _worded(2.0, 4.0, "gamma delta")]
    exact = app_module.whisper_timings(segments)
    assert app_module.alignment_agreement(segments, exact) == pytest.approx(1.0)

    shifted = [dict(t, start=t["start"] + 5.0, end=t["end"] + 5.0) for t in exact]
    assert app_module.alignment_agreement(segments, shifted) == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# The MFA invocation
# --------------------------------------------------------------------------------------
def test_mfa_runs_once_at_default_beams(app_module, upload_dir, alignment_env, monkeypatch):
    guid, audio = alignment_env
    commands = []
    _stub_subprocess(monkeypatch, app_module, guid, upload_dir,
                     entries=[[0.1, 0.5, "hello"], [0.6, 2.4, "world"],
                              [2.6, 3.0, "second"], [3.1, 4.9, "segment"]],
                     commands=commands)

    _timings, applied, stats = app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)

    assert applied is True
    mfa_commands = [c for c in commands if c[0] == "mfa"]
    assert len(mfa_commands) == 1, "the second beam attempt is gone"
    cmd = mfa_commands[0]
    assert cmd[:2] == ["mfa", "align"]
    assert cmd[2] == os.path.join(str(upload_dir), f"{guid}_mfa_input")
    assert cmd[5] == os.path.join(str(upload_dir), f"{guid}_aligned")
    assert "--beam" not in cmd and "--retry_beam" not in cmd
    for flag in ("--include_original_text", "--no_tokenization", "--clean", "--overwrite"):
        assert flag in cmd
    assert stats["utterances"] == 1
    assert stats["agree250"] is not None


def test_wav_is_exported_at_16k_mono(app_module, upload_dir, alignment_env, monkeypatch):
    """A source-rate stereo export made a 534 MB WAV of the 2783 s file; this makes 89 MB."""
    guid, audio = alignment_env
    commands = []
    _stub_subprocess(monkeypatch, app_module, guid, upload_dir,
                     entries=[[0.1, 0.5, "hello"]], commands=commands)

    app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)

    ffmpeg = next(c for c in commands if c[0] == "ffmpeg")
    assert "-ac" in ffmpeg and ffmpeg[ffmpeg.index("-ac") + 1] == "1"
    assert "-ar" in ffmpeg and ffmpeg[ffmpeg.index("-ar") + 1] == "16000"
    assert "-sample_fmt" in ffmpeg and ffmpeg[ffmpeg.index("-sample_fmt") + 1] == "s16"
    assert ffmpeg[-1].endswith(".wav")


def test_a_textgrid_is_written_not_a_flat_transcript(app_module, upload_dir, alignment_env,
                                                     monkeypatch):
    """One utterance per file was the defect; the corpus is now a TextGrid."""
    guid, audio = alignment_env
    written = {}

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            with open(cmd[-1], "wb") as fh:
                fh.write(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        corpus = cmd[2]
        written["files"] = sorted(os.listdir(corpus))
        _mfa_output(guid, upload_dir, [[0.1, 0.5, "hello"]])
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)

    assert written["files"] == [f"{guid}.TextGrid", f"{guid}.wav"]


def test_timeout_scales_with_duration_and_has_a_floor(app_module, upload_dir, alignment_env,
                                                      monkeypatch):
    timeouts = []

    def run_with(guid, audio, duration):
        def fake_run(cmd, **kwargs):
            if cmd[0] == "ffmpeg":
                with open(cmd[-1], "wb") as fh:
                    fh.write(b"RIFF")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            timeouts.append(kwargs.get("timeout"))
            _mfa_output(guid, upload_dir, [[0.1, 0.5, "hello"]])
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(app_module.subprocess, "run", fake_run)
        app_module.run_forced_alignment(audio, SEGMENTS, guid, duration)

    guid, audio = alignment_env
    run_with(guid, audio, 755.0)
    assert timeouts[-1] == pytest.approx(60 + 0.5 * 755.0)

    # A short clip still gets the floor rather than a 62 s budget.
    guid2 = str(uuid.uuid4())
    audio2 = upload_dir / f"{guid2}.mp3"
    audio2.write_bytes(b"fake")
    run_with(guid2, str(audio2), 4.0)
    assert timeouts[-1] == pytest.approx(120.0)


def test_working_directories_are_removed_on_success(app_module, upload_dir, alignment_env,
                                                    monkeypatch):
    guid, audio = alignment_env
    mfa_work = os.path.join(app_module.MFA_ROOT_DIR, f"{guid}_mfa_input")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            with open(cmd[-1], "wb") as fh:
                fh.write(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        os.makedirs(mfa_work, exist_ok=True)
        _mfa_output(guid, upload_dir, [[0.1, 0.5, "hello"]])
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)

    assert not os.path.exists(os.path.join(str(upload_dir), f"{guid}_mfa_input"))
    assert not os.path.exists(mfa_work)


def test_working_directories_are_removed_on_failure(app_module, upload_dir, alignment_env,
                                                    monkeypatch):
    guid, audio = alignment_env
    _stub_subprocess(monkeypatch, app_module, guid, upload_dir, mfa_rc=1)

    timings, applied, _stats = app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)

    assert applied is False
    assert not os.path.exists(os.path.join(str(upload_dir), f"{guid}_mfa_input"))
    assert not os.path.exists(os.path.join(app_module.MFA_ROOT_DIR, f"{guid}_mfa_input"))
    assert [t["text"] for t in timings] == [s["text"] for s in SEGMENTS]


def test_failure_returns_whisper_word_timings(app_module, upload_dir, alignment_env, monkeypatch):
    """The fallback is the word timestamps, not the VAD-padded segment bounds."""
    guid, audio = alignment_env
    segments = [_worded(0.0, 3.0, "alpha beta"), _worded(3.0, 6.0, "gamma delta")]
    segments[0]["words"][0]["start"] = 0.4
    _stub_subprocess(monkeypatch, app_module, guid, upload_dir, mfa_rc=1)

    timings, applied, stats = app_module.run_forced_alignment(audio, segments, guid, 6.0)

    assert applied is False
    assert timings[0]["start"] == pytest.approx(0.4)
    assert stats["agree250"] == pytest.approx(1.0)


def test_mfa_timeout_falls_back(app_module, upload_dir, alignment_env, monkeypatch):
    guid, audio = alignment_env

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            with open(cmd[-1], "wb") as fh:
                fh.write(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    _timings, applied, _stats = app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)
    assert applied is False


def test_wav_export_failure_falls_back(app_module, upload_dir, alignment_env, monkeypatch):
    guid, audio = alignment_env

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="no such codec")

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    timings, applied, _stats = app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)
    assert applied is False
    assert len(timings) == 2


def test_mfa_stdout_logged_only_on_failure(app_module, upload_dir, alignment_env, monkeypatch,
                                           caplog):
    """MFA's progress bars are megabytes of noise on a healthy run."""
    guid, audio = alignment_env
    _stub_subprocess(monkeypatch, app_module, guid, upload_dir, entries=[[0.1, 0.5, "hello"]])
    with caplog.at_level(logging.INFO):
        app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)
    assert "progress-bar-noise" not in caplog.text

    guid2 = str(uuid.uuid4())
    audio2 = upload_dir / f"{guid2}.mp3"
    audio2.write_bytes(b"fake")

    def failing(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            with open(cmd[-1], "wb") as fh:
                fh.write(b"RIFF")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise subprocess.CalledProcessError(1, cmd, output="progress-bar-noise", stderr="boom")

    monkeypatch.setattr(app_module.subprocess, "run", failing)
    with caplog.at_level(logging.ERROR):
        app_module.run_forced_alignment(str(audio2), SEGMENTS, guid2, 5.0)
    assert "progress-bar-noise" in caplog.text


def test_existing_alignment_is_reused_without_running_mfa(app_module, upload_dir, alignment_env,
                                                          monkeypatch):
    guid, audio = alignment_env
    _mfa_output(guid, upload_dir, [[0.1, 0.5, "hello"], [0.6, 2.4, "world"]])

    def fail(cmd, **kwargs):
        raise AssertionError("MFA must not run when its output already exists")

    monkeypatch.setattr(app_module.subprocess, "run", fail)
    _timings, applied, _stats = app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)
    assert applied is True


def test_malformed_mfa_json_falls_back(app_module, upload_dir, alignment_env):
    guid, audio = alignment_env
    out_dir = upload_dir / f"{guid}_aligned"
    out_dir.mkdir()
    (out_dir / f"{guid}.json").write_text(json.dumps({"tiers": {"phones": {"entries": []}}}))

    timings, applied, _stats = app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)
    assert applied is False
    assert [t["text"] for t in timings] == [s["text"] for s in SEGMENTS]


def test_silence_labels_are_dropped_from_the_words_tier(app_module, upload_dir, alignment_env):
    guid, audio = alignment_env
    _mfa_output(guid, upload_dir, [
        [0.0, 0.1, "sil"], [0.1, 0.5, "hello"], [0.5, 0.6, "<eps>"], [0.6, 2.4, "world"],
        [2.6, 3.0, "second"], [3.1, 4.9, "segment"],
    ])
    timings, applied, _stats = app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)
    assert applied is True
    assert timings[0]["start"] == pytest.approx(0.1)


def test_missing_audio_falls_back(app_module, upload_dir, alignment_env, monkeypatch):
    guid, _audio = alignment_env
    timings, applied, _stats = app_module.run_forced_alignment(
        str(upload_dir / "missing.mp3"), SEGMENTS, guid, 5.0
    )
    assert applied is False
    assert [t["text"] for t in timings] == [s["text"] for s in SEGMENTS]


def test_no_segments_returns_nothing_to_align(app_module, upload_dir, alignment_env):
    guid, audio = alignment_env
    timings, applied, _stats = app_module.run_forced_alignment(audio, [], guid, 5.0)
    assert timings == []
    assert applied is False


def test_no_dead_corpus_probe_remains(app_module):
    import inspect
    assert "_corpus" not in inspect.getsource(app_module.run_forced_alignment)


def test_a_barely_covered_segment_falls_back_to_the_whisper_span(app_module):
    """Owning some aligner words is not the same as being aligned.

    A 20 second segment that MFA matched on 2 of its 30 words would otherwise be
    published with a 0.6 second timing carrying twenty seconds of text, which is
    worse than having no refinement at all: it is confidently wrong, and the
    empty-window fallback never fires because the segment is not empty.
    """
    covered = _worded(0.0, 2.0, "alpha beta")
    sparse = _worded(2.0, 22.0, " ".join(f"word{i}" for i in range(30)))

    mfa_words = [
        {"start": 0.1, "end": 0.5, "text": "alpha"},
        {"start": 0.5, "end": 1.1, "text": "beta"},
        # Only two of the sparse segment's thirty words came back.
        {"start": 2.2, "end": 2.4, "text": "word0"},
        {"start": 2.4, "end": 2.8, "text": "word1"},
    ]
    refined, rstats = app_module.refine_segment_timings([covered, sparse], mfa_words)

    assert rstats["empty_fallbacks"] == 0, "the segment owns words, so the empty fallback cannot fire"
    assert rstats["short_fallbacks"] == 1
    whisper_start, whisper_end = app_module.whisper_span(sparse)
    assert refined[1]["end"] == pytest.approx(whisper_end)
    assert refined[1]["end"] - refined[1]["start"] > 15.0


def test_a_well_covered_segment_keeps_its_refined_span(app_module):
    """The guard must not undo the refinement it exists to protect."""
    segment = _worded(0.0, 10.0, " ".join(f"word{i}" for i in range(10)))
    mfa_words = [
        {"start": 0.2 + i, "end": 0.9 + i, "text": f"word{i}"} for i in range(10)
    ]
    refined, rstats = app_module.refine_segment_timings([segment], mfa_words)

    assert rstats["empty_fallbacks"] == 0 and rstats["short_fallbacks"] == 0
    assert rstats["span_ratio_median"] > 0.9
    assert rstats["clamped"] == 0
    assert refined[0]["start"] == pytest.approx(0.2)
    assert refined[0]["end"] == pytest.approx(9.9)


def test_short_fallbacks_are_reported_in_the_stats(app_module, upload_dir, alignment_env):
    """The sweep needs to see how often this fires."""
    guid, audio = alignment_env
    _mfa_output(guid, upload_dir, [
        [0.1, 0.5, "hello"], [0.6, 2.4, "world"],
        [2.6, 2.7, "second"],  # only one of the second segment's two words
    ])
    _timings, applied, stats = app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)

    assert applied is True
    assert stats["short_fallbacks"] is not None
    assert stats["empty_fallbacks"] == 0


def test_the_sweep_counters_are_emitted_as_key_value_pairs(app_module, upload_dir,
                                                           alignment_env, caplog):
    """The sweep parses these off the log line; /status returns only refined timings.

    Names are load-bearing: the parser ingests any key=value on the line, so the keys
    are what it keys on.
    """
    import logging

    guid, audio = alignment_env
    _mfa_output(guid, upload_dir, [
        [0.1, 0.5, "hello"], [0.6, 2.4, "world"],
        [2.6, 3.0, "second"], [3.1, 4.9, "segment"],
    ])
    with caplog.at_level(logging.INFO):
        app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)

    for key in ("agree250=", "empty_fallbacks=", "span_ratio_lt_0_5=",
                "span_ratio_p5=", "span_ratio_median=", "clamped="):
        assert key in caplog.text, f"the sweep parser needs {key} on the alignment line"


def test_the_counters_survive_an_alignment_failure(app_module, upload_dir, alignment_env,
                                                   monkeypatch, caplog):
    """A fallback path must still emit the full set, or the sweep sees a ragged table."""
    import logging
    import subprocess

    guid, audio = alignment_env
    _stub_subprocess(monkeypatch, app_module, guid, upload_dir, mfa_rc=1)
    with caplog.at_level(logging.INFO):
        _timings, applied, stats = app_module.run_forced_alignment(audio, SEGMENTS, guid, 5.0)

    assert applied is False
    assert subprocess is not None
    for key in ("agree250", "empty_fallbacks", "short_fallbacks", "clamped",
                "span_ratio_p5", "span_ratio_median"):
        assert key in stats, f"{key} missing from the fallback stats"


def test_the_clamp_is_counted(app_module):
    """A timing the monotonic clamp moved is not the aligner's answer."""
    segments = [_worded(0.0, 2.0, "alpha beta"), _worded(2.0, 4.0, "gamma delta")]
    mfa_words = [
        {"start": 0.0, "end": 3.5, "text": "alpha"},
        {"start": 0.5, "end": 3.9, "text": "beta"},
        {"start": 1.0, "end": 1.5, "text": "gamma"},   # starts before segment 0 ended
        {"start": 1.5, "end": 2.0, "text": "delta"},
    ]
    _refined, rstats = app_module.refine_segment_timings(segments, mfa_words)
    assert rstats["clamped"] >= 1


def test_span_ratios_describe_what_the_aligner_produced(app_module):
    """Recorded before the substitution, so the fix cannot hide the problem it fixes."""
    sparse = _worded(0.0, 20.0, " ".join(f"word{i}" for i in range(30)))
    mfa_words = [
        {"start": 0.2, "end": 0.4, "text": "word0"},
        {"start": 0.4, "end": 0.8, "text": "word1"},
    ]
    _refined, rstats = app_module.refine_segment_timings([sparse], mfa_words)

    assert rstats["short_fallbacks"] == 1
    assert rstats["span_ratio_median"] < 0.5, \
        "the ratio must record the aligner's short span, not the substituted one"
