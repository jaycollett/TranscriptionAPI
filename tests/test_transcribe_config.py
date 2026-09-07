"""Tests for the decode configuration in transcribe.py.

These are regression guards for the three defects that made the twelve 2024 teachings
transcribe at 0.15-1.24 words/sec against a normal 2.5-2.8:

  1. a bogus "noise detected" gate that was always taken, feeding every file through
     an unprofiled noisereduce pass,
  2. an initial_prompt asserting a single speaker on multi-voice class and Q&A audio,
  3. a scalar `temperature`, which disables faster-whisper's temperature fallback and
     removes the built-in escape from a mid-file decode collapse.

Plus one guard for the pass-5 `condition_on_previous_text: False` setting, which the
decode call used to override with a hardcoded True.

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


# --------------------------------------------------------------------------------------
# Defects 1 and 2: what actually reaches model.transcribe()
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


def test_no_single_speaker_prompt(decode_calls):
    """These are multi-voice class and Q&A recordings; asserting one speaker is false."""
    calls, _ = decode_calls
    for call in calls:
        prompt = call.get("initial_prompt")
        assert prompt is None or "single speaker" not in prompt.lower()


def test_every_pass_passes_a_temperature_sequence(decode_calls):
    """A scalar here is the bug: faster-whisper wraps it and disables fallback."""
    calls, _ = decode_calls
    assert len(calls) == 5, "expected the five configured passes"
    for call in calls:
        temperature = call["temperature"]
        assert isinstance(temperature, (tuple, list)), f"scalar temperature: {temperature!r}"
        assert len(temperature) > 1
        assert temperature[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Pass 5 declares condition_on_previous_text=False; the decode call must honor it
# --------------------------------------------------------------------------------------
def test_pass_five_disables_conditioning_on_previous_text(decode_calls):
    """The passes list declares the flag once; the call must not hardcode True."""
    calls, _ = decode_calls
    assert len(calls) == 5, "expected the five configured passes"
    flags = [call["condition_on_previous_text"] for call in calls]
    assert flags.count(False) == 1, f"exactly one pass should disable it, got {flags!r}"
    assert flags.count(True) == 4, f"the other four passes should keep it, got {flags!r}"
    assert flags[-1] is False, "the unconditioned pass is the fifth one"
