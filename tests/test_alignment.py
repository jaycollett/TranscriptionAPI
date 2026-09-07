"""Tests for run_forced_alignment: the MFA command, working-directory cleanup, the
except/finally structure, log volume, and the JSON parsing that maps MFA words back
onto Whisper segments. No MFA binary; subprocess.run is stubbed.
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


class _FakeAudio:
    @staticmethod
    def from_file(path):
        return _FakeAudio()

    def export(self, out_path, format=None):
        with open(out_path, "wb"):
            pass


@pytest.fixture
def alignment_env(app_module, upload_dir, monkeypatch):
    monkeypatch.setattr(app_module, "AudioSegment", _FakeAudio)
    guid = str(uuid.uuid4())
    audio = upload_dir / f"{guid}.mp3"
    audio.write_bytes(b"fake audio")
    return guid, str(audio)


def _mfa_output(guid, upload_dir, entries):
    out_dir = upload_dir / f"{guid}_aligned"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{guid}.json").write_text(json.dumps({"tiers": {"words": {"entries": entries}}}))


def test_mfa_command_uses_clean_flag_and_input_dir(app_module, upload_dir, alignment_env, monkeypatch):
    guid, audio = alignment_env
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        _mfa_output(guid, upload_dir, [[0.1, 0.5, "hello"], [0.6, 2.4, "world"]])
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)

    segments, applied = app_module.run_forced_alignment(audio, SEGMENTS, guid)

    assert applied is True
    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:2] == ["mfa", "align"]
    assert "--clean" in cmd
    assert cmd[2] == str(upload_dir / f"{guid}_mfa_input")
    assert "--output_format" in cmd and "json" in cmd


def test_mfa_working_directories_are_removed_on_success(app_module, upload_dir, alignment_env, monkeypatch):
    guid, audio = alignment_env
    mfa_root = app_module.MFA_ROOT_DIR

    def fake_run(cmd, **kwargs):
        # MFA writes its corpus under <root>/<corpus basename>
        work = os.path.join(mfa_root, f"{guid}_mfa_input")
        os.makedirs(work, exist_ok=True)
        with open(os.path.join(work, "corpus.db"), "wb"):
            pass
        _mfa_output(guid, upload_dir, [[0.1, 0.5, "hello"]])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    app_module.run_forced_alignment(audio, SEGMENTS, guid)

    assert not os.path.exists(os.path.join(mfa_root, f"{guid}_mfa_input"))
    assert not (upload_dir / f"{guid}_mfa_input").exists()


def test_mfa_working_directories_are_removed_on_failure(app_module, upload_dir, alignment_env, monkeypatch):
    guid, audio = alignment_env
    mfa_root = app_module.MFA_ROOT_DIR

    def fake_run(cmd, **kwargs):
        os.makedirs(os.path.join(mfa_root, f"{guid}_mfa_input"), exist_ok=True)
        raise subprocess.CalledProcessError(1, cmd, output="progress bar", stderr="boom")

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    segments, applied = app_module.run_forced_alignment(audio, SEGMENTS, guid)

    assert (segments, applied) == (SEGMENTS, False)
    assert not os.path.exists(os.path.join(mfa_root, f"{guid}_mfa_input"))
    assert not (upload_dir / f"{guid}_mfa_input").exists()


def test_both_beam_configs_are_tried_before_giving_up(app_module, upload_dir, alignment_env, monkeypatch):
    guid, audio = alignment_env
    beams = []

    def fake_run(cmd, **kwargs):
        beams.append(cmd[cmd.index("--beam") + 1])
        raise subprocess.CalledProcessError(1, cmd, stderr="no")

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    assert app_module.run_forced_alignment(audio, SEGMENTS, guid) == (SEGMENTS, False)
    assert beams == ["40", "100"]


def test_timeout_falls_through_to_next_beam(app_module, upload_dir, alignment_env, monkeypatch):
    guid, audio = alignment_env
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd, 300)
        _mfa_output(guid, upload_dir, [[0.1, 0.5, "hello"]])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    _, applied = app_module.run_forced_alignment(audio, SEGMENTS, guid)
    assert applied is True
    assert len(calls) == 2


def test_makedirs_failure_does_not_raise_unbound_local(app_module, upload_dir, alignment_env, monkeypatch, caplog):
    """If a step before temp_mfa_dir was assigned raises, cleanup must not blow up
    with UnboundLocalError (which the old code swallowed into a misleading log)."""
    guid, audio = alignment_env
    caplog.set_level(logging.ERROR, logger="app")

    def broken_makedirs(path, exist_ok=False):
        raise OSError("read-only file system")

    monkeypatch.setattr(app_module.os, "makedirs", broken_makedirs)
    ran = []
    monkeypatch.setattr(app_module.subprocess, "run", lambda *a, **k: ran.append(1))

    assert app_module.run_forced_alignment(audio, SEGMENTS, guid) == (SEGMENTS, False)
    assert not ran
    messages = [r.getMessage() for r in caplog.records]
    assert not any("UnboundLocalError" in m or "referenced before assignment" in m for m in messages)
    assert any("read-only file system" in m for m in messages)


def test_mfa_stdout_logged_only_on_failure(app_module, upload_dir, alignment_env, monkeypatch, caplog):
    guid, audio = alignment_env
    caplog.set_level(logging.INFO, logger="app")

    def ok_run(cmd, **kwargs):
        _mfa_output(guid, upload_dir, [[0.1, 0.5, "hello"]])
        return subprocess.CompletedProcess(cmd, 0, stdout="PROGRESS BAR NOISE", stderr="")

    monkeypatch.setattr(app_module.subprocess, "run", ok_run)
    app_module.run_forced_alignment(audio, SEGMENTS, guid)
    info_or_higher = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    assert not any("PROGRESS BAR NOISE" in m for m in info_or_higher)

    caplog.clear()
    guid2 = str(uuid.uuid4())
    audio2 = upload_dir / f"{guid2}.mp3"
    audio2.write_bytes(b"x")

    def bad_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(2, cmd, output="FAILED STDOUT", stderr="FAILED STDERR")

    monkeypatch.setattr(app_module.subprocess, "run", bad_run)
    app_module.run_forced_alignment(str(audio2), SEGMENTS, guid2)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("FAILED STDOUT" in m and "FAILED STDERR" in m and guid2 in m for m in warnings)


def test_existing_alignment_is_reused_without_running_mfa(app_module, upload_dir, alignment_env, monkeypatch):
    guid, audio = alignment_env
    _mfa_output(guid, upload_dir, [[0.2, 0.6, "hello"], [0.7, 2.3, "world"], [2.6, 4.9, "second"]])
    ran = []
    monkeypatch.setattr(app_module.subprocess, "run", lambda *a, **k: ran.append(1))

    segments, applied = app_module.run_forced_alignment(audio, SEGMENTS, guid)

    assert not ran
    assert applied is True
    assert segments[0] == {"start": 0.2, "end": 2.3, "text": "hello world"}
    assert segments[1] == {"start": 2.6, "end": 4.9, "text": "second segment"}


def test_segment_without_words_keeps_whisper_timings(app_module, upload_dir, alignment_env):
    guid, audio = alignment_env
    _mfa_output(guid, upload_dir, [[0.2, 0.6, "hello"]])
    segments, applied = app_module.run_forced_alignment(audio, SEGMENTS, guid)
    assert applied is True
    assert segments[1] == SEGMENTS[1]


def test_malformed_mfa_json_falls_back(app_module, upload_dir, alignment_env):
    guid, audio = alignment_env
    out_dir = upload_dir / f"{guid}_aligned"
    out_dir.mkdir()
    (out_dir / f"{guid}.json").write_text(json.dumps({"tiers": {"phones": {}}}))
    assert app_module.run_forced_alignment(audio, SEGMENTS, guid) == (SEGMENTS, False)


def test_missing_audio_falls_back(app_module, upload_dir, alignment_env, monkeypatch):
    guid, _ = alignment_env
    ran = []
    monkeypatch.setattr(app_module.subprocess, "run", lambda *a, **k: ran.append(1))
    result = app_module.run_forced_alignment(str(upload_dir / "missing.mp3"), SEGMENTS, guid)
    assert result == (SEGMENTS, False)
    assert not ran


def test_no_dead_corpus_probe_remains(app_module):
    import inspect

    assert "_corpus" not in inspect.getsource(app_module.run_forced_alignment)
