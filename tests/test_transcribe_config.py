"""Tests for the decode configuration in transcribe.py.

These are regression guards for the defects that made the twelve 2024 teachings
transcribe at 0.15-1.24 words/sec against a normal 2.5-2.8:

  1. a bogus "noise detected" gate that was always taken, feeding every file through
     an unprofiled noisereduce pass,
  2. an initial_prompt asserting a single speaker on multi-voice class and Q&A audio,
  3. a scalar `temperature`, which disables faster-whisper's temperature fallback and
     removes the built-in escape from a mid-file decode collapse.

Plus the 0.6.0 contract: exactly one decode pass at beam 5 (harness C1, which
reproduced the five-pass winner in a fifth of the wall time) and a VAD threshold that
depends on the file's level (C4, which loses 290 words on the -28.6 dBFS retreat
recording at the loud threshold).

The real module is loaded by the shared `transcribe_module` fixture in conftest.py
with the heavy GPU imports stubbed. No GPU, no model, no audio.
"""

import pytest

# --------------------------------------------------------------------------------------
# Defect 3: temperature must be a sequence, or faster-whisper never falls back
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("base", [0.0, 0.2, 0.3, 0.5])
def test_temperature_ladder_is_a_usable_fallback_sequence(transcribe_module, base):
    ladder = transcribe_module.temperature_ladder(base)
    assert isinstance(ladder, tuple)
    assert ladder[0] == pytest.approx(base)
    assert ladder[-1] == pytest.approx(1.0)
    assert len(ladder) > 1, "a single-element ladder is the same as a scalar: no fallback"
    assert list(ladder) == sorted(ladder)
    assert len(set(ladder)) == len(ladder)


def test_temperature_ladder_at_ceiling_is_not_empty(transcribe_module):
    assert transcribe_module.temperature_ladder(1.0) == (1.0,)


def test_temperature_ladder_clamps_out_of_range_base(transcribe_module):
    """Whisper temperatures live in [0, 1]; a bad base must not produce a bad ladder."""
    assert transcribe_module.temperature_ladder(1.2) == (1.0,)
    low = transcribe_module.temperature_ladder(-0.2)
    assert low[0] == pytest.approx(0.0)
    assert low[-1] == pytest.approx(1.0)
    assert all(0.0 <= t <= 1.0 for t in low)


def test_default_ladder_is_the_measured_one(transcribe_module):
    """The C1 ladder. Anything coarser was measured (C10) and made no difference."""
    assert transcribe_module.temperature_ladder(
        transcribe_module.WHISPER_TEMPERATURE_BASE
    ) == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# --------------------------------------------------------------------------------------
# What actually reaches model.transcribe()
# --------------------------------------------------------------------------------------
@pytest.fixture
def decode_calls(transcribe_module, monkeypatch, tmp_path):
    """Run transcribe_audio against a fake model and capture every decode kwarg set."""
    calls = []

    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            calls.append({"audio": audio, **kwargs})
            return ([], None)

    monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
    monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 755.164)
    monkeypatch.setattr(transcribe_module, "measure_mean_dbfs", lambda path, duration=0: -23.1)
    monkeypatch.setattr(transcribe_module, "upload_folder", str(tmp_path))

    audio_path = tmp_path / "teaching.mp3"
    audio_path.write_bytes(b"not really audio")

    transcribe_module.transcribe_audio(
        str(audio_path), "f5e593e9-c7b8-4809-a476-2a82d7b613f3"
    )
    return calls, str(audio_path)


def test_original_audio_is_decoded_untouched(decode_calls):
    """No denoise, no renormalisation, no second MP3 encode: Whisper sees the source file."""
    calls, audio_path = decode_calls
    assert calls, "no transcription pass ran"
    for call in calls:
        assert call["audio"] == audio_path
        assert not call["audio"].endswith("_processed.mp3")


def test_no_denoise_helper_remains(transcribe_module):
    """Guard against the noise gate being reintroduced."""
    assert not hasattr(transcribe_module, "preprocess_audio_for_transcription")


def test_no_boundary_regex_helper_remains(transcribe_module):
    """clean_boundary_duplicates edited the text and not the timings; it is gone."""
    assert not hasattr(transcribe_module, "clean_boundary_duplicates")


def test_exactly_one_decode_pass_runs(decode_calls):
    """C1: five passes were four noisy variants of one decode plus the beam search."""
    calls, _ = decode_calls
    assert len(calls) == 1, f"expected a single decode pass, got {len(calls)}"


def test_single_pass_uses_the_measured_beam_settings(decode_calls):
    calls, _ = decode_calls
    call = calls[0]
    assert call["beam_size"] == 5
    assert call["best_of"] == 5
    assert call["patience"] == pytest.approx(1.0)
    assert call["language"] == "en"
    assert call["word_timestamps"] is True
    assert call["condition_on_previous_text"] is True
    assert call["compression_ratio_threshold"] == pytest.approx(2.4)
    assert call["log_prob_threshold"] == pytest.approx(-1.0)
    assert call["no_speech_threshold"] == pytest.approx(0.6)
    assert call["prompt_reset_on_temperature"] == pytest.approx(0.5)


def test_decode_passes_a_temperature_sequence(decode_calls):
    """A scalar here is the bug: faster-whisper wraps it and disables fallback."""
    calls, _ = decode_calls
    temperature = calls[0]["temperature"]
    assert isinstance(temperature, (tuple, list)), f"scalar temperature: {temperature!r}"
    assert tuple(temperature) == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def test_no_prompt_and_no_hotwords(decode_calls):
    """The old prompt asserted one speaker; measured hotwords primed repetition."""
    call = decode_calls[0][0]
    assert call.get("initial_prompt") is None
    assert call.get("hotwords") is None


def test_model_is_built_with_one_worker(transcribe_module):
    """A second CUDA worker only ever doubled the resident model: the loop is serial."""
    assert transcribe_module.WHISPER_NUM_WORKERS == 1


# --------------------------------------------------------------------------------------
# Level-aware VAD (C4 plus the retreat-file failure)
# --------------------------------------------------------------------------------------
def test_vad_is_enabled_with_the_measured_parameters(decode_calls):
    calls, _ = decode_calls
    call = calls[0]
    assert call["vad_filter"] is True
    params = call["vad_parameters"]
    assert params["min_speech_duration_ms"] == 250
    assert params["min_silence_duration_ms"] == 1000
    assert params["speech_pad_ms"] == 300


def test_hallucination_threshold_can_actually_fire(decode_calls):
    """2.0 was dead weight: the VAD only ever leaves 2 x 300 ms of silence in a chunk."""
    threshold = decode_calls[0][0]["hallucination_silence_threshold"]
    assert threshold == pytest.approx(0.5)
    assert threshold < 2 * 300 / 1000.0


@pytest.mark.parametrize(
    "mean_dbfs, expected",
    [
        (-23.1, "loud"),    # the 755 s reference file
        (-18.0, "loud"),
        (-24.5, "loud"),    # quietest file that worked on the loud profile
        (-26.0, "loud"),    # the cutover itself is the loud branch
        (-26.1, "quiet"),
        (-28.4, "quiet"),   # the retreat file, which loses 3.0 percent of its words
    ],
)
def test_profile_follows_the_level(transcribe_module, mean_dbfs, expected):
    profile, why = transcribe_module.choose_vad_profile(mean_dbfs)
    assert profile == expected
    assert f"{mean_dbfs:.1f}" in why


def test_the_two_profiles_are_the_measured_ones(transcribe_module):
    """A softer threshold alone is not enough: the quiet branch needs the whole profile.

    Measured 2026-09-07 on the retreat file: the loud profile gives 1157 segments and
    8639 words, the loud profile at threshold 0.35 gives 4693 segments and 8407 words,
    and this profile gives 435 segments and 8904 words, reproducing C1 exactly.
    """
    loud = transcribe_module.vad_parameters("loud")
    quiet = transcribe_module.vad_parameters("quiet")

    assert loud == {"threshold": 0.5, "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 1000, "speech_pad_ms": 300}
    assert quiet == {"threshold": 0.35, "min_speech_duration_ms": 250,
                     "min_silence_duration_ms": 300, "speech_pad_ms": 400}
    # The long minimum silence is what shreds quiet audio, so it must differ.
    assert quiet["min_silence_duration_ms"] < loud["min_silence_duration_ms"]
    assert quiet["speech_pad_ms"] > loud["speech_pad_ms"]


def test_the_hallucination_filter_is_off_on_the_quiet_profile(transcribe_module):
    """The quiet profile's 800 ms of padded silence was measured without the filter."""
    assert transcribe_module.hallucination_threshold("loud") == pytest.approx(0.5)
    assert transcribe_module.hallucination_threshold("quiet") is None


def test_unmeasurable_level_takes_the_cautious_profile(transcribe_module):
    """Fragmenting quiet speech loses words; cutting extra seams on loud speech does not."""
    profile, why = transcribe_module.choose_vad_profile(None)
    assert profile == "quiet"
    assert "unknown" in why


@pytest.mark.parametrize("mean_dbfs, expected, profile", [(-23.1, 0.5, "loud"),
                                                          (-28.4, 0.35, "quiet")])
def test_measured_level_reaches_the_decode_call(transcribe_module, monkeypatch, tmp_path,
                                                mean_dbfs, expected, profile):
    """End to end: the level measured for the file picks the profile Whisper is given."""
    calls = []

    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return ([], None)

    monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
    monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 3388.0)
    monkeypatch.setattr(transcribe_module, "measure_mean_dbfs", lambda path, duration=0: mean_dbfs)
    audio_path = tmp_path / "session3.mp3"
    audio_path.write_bytes(b"not really audio")

    result = transcribe_module.transcribe_audio(str(audio_path), "guid")

    assert calls[0]["vad_parameters"] == transcribe_module.vad_parameters(profile)
    assert calls[0]["vad_parameters"]["threshold"] == pytest.approx(expected)
    assert calls[0]["hallucination_silence_threshold"] == \
        transcribe_module.hallucination_threshold(profile)
    assert result["vad_threshold"] == pytest.approx(expected)
    assert result["vad_profile"] == profile
    assert result["mean_dbfs"] == pytest.approx(mean_dbfs)


def test_mean_dbfs_is_parsed_from_volumedetect(transcribe_module, monkeypatch, tmp_path):
    """ffmpeg reports mean_volume on stderr; the last match is the audio stream's."""
    stderr = (
        "[Parsed_volumedetect_0 @ 0x1] n_samples: 54263040\n"
        "[Parsed_volumedetect_0 @ 0x1] mean_volume: -28.4 dB\n"
        "[Parsed_volumedetect_0 @ 0x1] max_volume: -3.2 dB\n"
    )

    class _Completed:
        returncode = 0
        stdout = ""

    _Completed.stderr = stderr
    monkeypatch.setattr(transcribe_module.subprocess, "run", lambda *a, **k: _Completed())
    assert transcribe_module.measure_mean_dbfs(str(tmp_path / "a.mp3"), 3388.0) == pytest.approx(-28.4)


def test_mean_dbfs_is_none_when_nothing_can_measure_it(transcribe_module, monkeypatch, tmp_path):
    """A missing filter and an undecodable file must not fail the job, only the branch."""
    def boom(*a, **k):
        raise OSError("no ffmpeg")

    monkeypatch.setattr(transcribe_module.subprocess, "run", boom)
    assert transcribe_module.measure_mean_dbfs(str(tmp_path / "a.mp3"), 60.0) is None
