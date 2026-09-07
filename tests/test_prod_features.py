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

    commands = []

    def record(cmd, **kwargs):
        commands.append(cmd)
        if cmd[0] == "ffmpeg":
            # Stand in for the 16 kHz mono WAV export MFA reads.
            with open(cmd[-1], "wb") as fh:
                fh.write(b"RIFF")
            return _subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        # Simulate MFA failing so the function falls back to Whisper timings.
        raise _subprocess.CalledProcessError(1, cmd, stderr="boom")

    monkeypatch.setattr(app_module.subprocess, "run", record)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake audio")

    whisper_segments = [{"start": 0.0, "end": 2.5, "text": "hello world"}]
    timings, applied, _stats = app_module.run_forced_alignment(
        str(audio_path), whisper_segments, str(uuid.uuid4()), 2.5
    )

    assert any(cmd[0] == "mfa" for cmd in commands), \
        "run_forced_alignment should always reach the MFA invocation"
    assert applied is False
    assert timings == [{"start": 0.0, "end": 2.5, "text": "hello world"}]
