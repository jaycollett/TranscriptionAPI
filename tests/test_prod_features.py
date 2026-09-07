"""Tests for the forced-alignment step in app.py.

run_forced_alignment must always attempt MFA (there is no longer a diarization
short-circuit ahead of it) and must fall back to the Whisper segments when the aligner
fails. This covers the control flow only: no GPU, no MFA binary.
"""

import uuid


def test_run_forced_alignment_always_invokes_mfa_and_falls_back_on_failure(
    app_module, monkeypatch, tmp_path
):
    """run_forced_alignment shells out to MFA unconditionally; when MFA fails it returns
    the original Whisper segments rather than raising."""
    import subprocess as _subprocess

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

    assert mfa_calls, "run_forced_alignment should always reach the MFA invocation"
    assert result == whisper_segments
