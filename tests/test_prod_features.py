"""Tests for the production-live transcription features that were merged alongside the
queue-hardening: speaker diarization and the multi-speaker MFA-skip decision.

These cover the *decision logic* only - no GPU, no pyannote models, no MFA binary - so
they stay fast and deterministic in CI. The heavy diarization/alignment work itself is
exercised in production, not here.
"""

import uuid

import pytest


# --------------------------------------------------------------------------------------
# Speaker diarization graceful fallback
# --------------------------------------------------------------------------------------
def test_detect_speakers_falls_back_to_one_without_pyannote(app_module, monkeypatch):
    """When pyannote is unavailable, detect_speakers assumes a single speaker (1)
    instead of raising, so the pipeline degrades gracefully on hosts without it."""
    monkeypatch.setattr(app_module, "PYANNOTE_AVAILABLE", False)
    assert app_module.detect_speakers("/nonexistent/audio.wav") == 1


# --------------------------------------------------------------------------------------
# Multi-speaker content skips MFA forced alignment
# --------------------------------------------------------------------------------------
def test_multi_speaker_audio_skips_mfa(app_module, monkeypatch):
    """If diarization reports >1 speaker, run_forced_alignment returns the Whisper
    segments untouched and never shells out to MFA (which is inaccurate on multi-speaker
    audio)."""
    monkeypatch.setattr(app_module, "detect_speakers", lambda path: 3)

    # MFA must NOT be invoked for multi-speaker audio.
    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run (MFA) must not run for multi-speaker audio")

    monkeypatch.setattr(app_module.subprocess, "run", fail_if_called)

    whisper_segments = [{"start": 0.0, "end": 2.5, "text": "hello world"}]
    result = app_module.run_forced_alignment("/tmp/audio.wav", whisper_segments, str(uuid.uuid4()))

    assert result == whisper_segments


def test_single_speaker_audio_invokes_mfa(app_module, monkeypatch, tmp_path):
    """With a single speaker, run_forced_alignment proceeds into the MFA path (shells out
    to the aligner) rather than short-circuiting on the diarization check. MFA failing here
    causes a safe fallback to the Whisper segments."""
    import subprocess as _subprocess

    monkeypatch.setattr(app_module, "detect_speakers", lambda path: 1)
    monkeypatch.setitem(app_module.app.config, "UPLOAD_FOLDER", str(tmp_path))

    # Minimal AudioSegment stub so MFA input prep (WAV export) succeeds.
    class _FakeAudio:
        @staticmethod
        def from_file(path):
            return _FakeAudio()

        def export(self, out_path, format=None):
            with open(out_path, "wb"):
                pass

    monkeypatch.setattr(app_module, "AudioSegment", _FakeAudio)

    mfa_calls = []

    def record_mfa(*args, **kwargs):
        mfa_calls.append(args[0] if args else None)
        # Simulate MFA failing so the function falls back to Whisper segments.
        raise _subprocess.CalledProcessError(1, args[0] if args else "mfa", stderr="boom")

    monkeypatch.setattr(app_module.subprocess, "run", record_mfa)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake audio")

    whisper_segments = [{"start": 0.0, "end": 2.5, "text": "hello world"}]
    result = app_module.run_forced_alignment(str(audio_path), whisper_segments, str(uuid.uuid4()))

    # Single-speaker audio must reach the MFA invocation (not skip it)...
    assert mfa_calls, "single-speaker audio should invoke MFA, not skip alignment"
    # ...and on MFA failure it falls back safely to the original Whisper segments.
    assert result == whisper_segments
