"""Tests for the non-decode helpers in transcribe.py: the ffprobe duration probe, the
shared processing estimate, and the removal of the transcript file write. The decode
parameters themselves are covered by test_transcribe_config.py and are not touched.
"""

import math

import pytest


# --------------------------------------------------------------------------------------
# get_audio_duration: ffprobe first, full decode only as a fallback
# --------------------------------------------------------------------------------------
def test_duration_comes_from_mediainfo_without_decoding(transcribe_module, monkeypatch):
    monkeypatch.setattr(transcribe_module, "mediainfo", lambda path: {"duration": "755.164000"})

    class _NeverDecode:
        @staticmethod
        def from_file(path):
            raise AssertionError("full decode must not happen when ffprobe has a duration")

    monkeypatch.setattr(transcribe_module, "AudioSegment", _NeverDecode)
    assert transcribe_module.get_audio_duration("/x/a.mp3") == pytest.approx(755.164)


@pytest.mark.parametrize("info", [{}, {"duration": "N/A"}, {"duration": ""}, None])
def test_duration_falls_back_to_decode_when_mediainfo_has_none(transcribe_module, monkeypatch, info):
    monkeypatch.setattr(transcribe_module, "mediainfo", lambda path: info)

    class _Decoded:
        def __len__(self):
            return 12_500

    class _Audio:
        @staticmethod
        def from_file(path):
            return _Decoded()

    monkeypatch.setattr(transcribe_module, "AudioSegment", _Audio)
    assert transcribe_module.get_audio_duration("/x/a.mp3") == pytest.approx(12.5)


def test_duration_falls_back_when_mediainfo_raises(transcribe_module, monkeypatch):
    def broken(path):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(transcribe_module, "mediainfo", broken)

    class _Decoded:
        def __len__(self):
            return 3_000

    monkeypatch.setattr(transcribe_module, "AudioSegment",
                        type("A", (), {"from_file": staticmethod(lambda p: _Decoded())}))
    assert transcribe_module.get_audio_duration("/x/a.mp3") == pytest.approx(3.0)


def test_undecodable_input_raises(transcribe_module, monkeypatch):
    monkeypatch.setattr(transcribe_module, "mediainfo", lambda path: {})

    class _Audio:
        @staticmethod
        def from_file(path):
            raise RuntimeError("Decoding failed")

    monkeypatch.setattr(transcribe_module, "AudioSegment", _Audio)
    with pytest.raises(RuntimeError):
        transcribe_module.get_audio_duration("/x/garbage.mp3")


def test_duration_is_not_cached_by_path(transcribe_module, monkeypatch):
    """The old lru_cache keyed on the path; a re-upload to the same name must re-probe."""
    values = iter(["10.0", "20.0"])
    monkeypatch.setattr(transcribe_module, "mediainfo", lambda path: {"duration": next(values)})
    assert transcribe_module.get_audio_duration("/x/same.mp3") == 10.0
    assert transcribe_module.get_audio_duration("/x/same.mp3") == 20.0


# --------------------------------------------------------------------------------------
# estimate_processing_seconds: one formula for /upload and the log line
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("duration", [0, 1, 30.0, 755.164, 2783.0])
def test_estimate_matches_the_historical_formula(transcribe_module, duration):
    expected = math.ceil(duration / 15.1) * 5 + 45
    assert transcribe_module.estimate_processing_seconds(duration) == expected


def test_estimate_returns_an_int(transcribe_module):
    assert isinstance(transcribe_module.estimate_processing_seconds(755.164), int)


# --------------------------------------------------------------------------------------
# whisper_model_loaded and the transcript write
# --------------------------------------------------------------------------------------
def test_whisper_model_loaded_tracks_the_singleton(transcribe_module, monkeypatch):
    monkeypatch.setattr(transcribe_module, "_whisper_model", None)
    assert transcribe_module.whisper_model_loaded() is False
    monkeypatch.setattr(transcribe_module, "_whisper_model", object())
    assert transcribe_module.whisper_model_loaded() is True


def test_transcribe_audio_does_not_write_a_transcript_file(transcribe_module, monkeypatch, tmp_path):
    class _Seg:
        def __init__(self, start, end, text):
            self.start, self.end, self.text, self.words = start, end, text, []

    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            return ([_Seg(0.0, 1.0, "hello there"), _Seg(1.0, 2.0, "general")], None)

    monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
    monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 2.0)
    monkeypatch.setattr(transcribe_module, "upload_folder", str(tmp_path))
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"x")
    guid = "f5e593e9-c7b8-4809-a476-2a82d7b613f3"

    result = transcribe_module.transcribe_audio(str(audio), guid)

    assert result["transcription"] == "hello there general"
    assert result["timings"] == [
        {"start": 0.0, "end": 1.0, "text": "hello there"},
        {"start": 1.0, "end": 2.0, "text": "general"},
    ]
    assert result["duration_sec"] == 2.0
    assert not (tmp_path / f"{guid}.txt").exists()
