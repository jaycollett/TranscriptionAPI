"""Tests for the 0.6.0 per-segment diagnostics and the segment-level boundary dedupe.

The anomaly score (harness C2) and the loop flag (C7) replace the duration-weighted
word-probability rule, which on four of the six reference files preferred the pass
with fewer words: on the retreat recording it ranked the passes in reverse order of
word count and dropped two runs of Psalm 91 that the winning decode contained.

The boundary dedupe replaces `clean_boundary_duplicates`, a regex over the transcript
string that left the timings untouched, so text and timings disagreed on every
production job, and that deleted legitimate repetition anywhere in the file.
"""

import pytest


def _segment(start, end, text, **fields):
    """A serialised segment with healthy defaults and one word per whitespace token.

    Word timestamps are spread evenly across the span, which is enough for the
    windowing and dedupe rules; nothing here depends on their exact values.
    """
    tokens = text.split()
    step = (end - start) / len(tokens) if tokens else 0.0
    words = [
        {
            "start": round(start + i * step, 4),
            "end": round(start + (i + 1) * step, 4),
            "word": (" " if i else "") + token,
            "probability": 0.9,
        }
        for i, token in enumerate(tokens)
    ]
    segment = {
        "start": start,
        "end": end,
        "text": text,
        "avg_logprob": -0.06,
        "compression_ratio": 1.5,
        "no_speech_prob": 0.01,
        "temperature": 0.0,
        "words": words,
    }
    segment.update(fields)
    return segment


# --------------------------------------------------------------------------------------
# The anomaly score
# --------------------------------------------------------------------------------------
def test_healthy_segments_score_zero(transcribe_module):
    segments = [_segment(i * 10.0, i * 10.0 + 10.0, "the word of the Lord came to him again")
                for i in range(30)]
    count, flagged = transcribe_module.annotate_segments(segments)
    assert count == 0
    assert flagged == []
    assert all(s["flags"] == [] for s in segments)


@pytest.mark.parametrize(
    "field, value, flag",
    [
        ("temperature", 0.6, "temperature"),
        ("temperature", 0.5, "temperature"),
        ("compression_ratio", 2.5, "compression_ratio"),
        ("avg_logprob", -1.4, "avg_logprob"),
        ("no_speech_prob", 0.8, "no_speech"),
    ],
)
def test_each_anomaly_reason_is_counted(transcribe_module, field, value, flag):
    segments = [_segment(0.0, 10.0, "a normal stretch of teaching here", **{field: value})]
    count, flagged = transcribe_module.annotate_segments(segments)
    assert count == 1
    assert flagged[0]["flags"] == [flag]
    assert flagged[0]["index"] == 0


def test_temperature_below_the_threshold_is_not_an_anomaly(transcribe_module):
    segments = [_segment(0.0, 10.0, "a normal stretch of teaching here", temperature=0.4)]
    assert transcribe_module.annotate_segments(segments) == (0, [])


def test_high_no_speech_on_an_empty_segment_is_not_an_anomaly(transcribe_module):
    """no_speech_prob is only evidence of a phantom when there is text to be phantom."""
    segments = [_segment(0.0, 10.0, "", no_speech_prob=0.9)]
    assert transcribe_module.annotate_segments(segments) == (0, [])


def test_one_segment_with_several_reasons_counts_once(transcribe_module):
    segments = [_segment(0.0, 10.0, "words words", temperature=0.8, compression_ratio=3.1,
                         avg_logprob=-1.9)]
    count, flagged = transcribe_module.annotate_segments(segments)
    assert count == 1
    assert set(flagged[0]["flags"]) >= {"temperature", "compression_ratio", "avg_logprob"}


def test_truncation_is_caught_by_the_window_check(transcribe_module):
    """A 30 percent truncation: the file transcribes normally, then stops producing words.

    The whole-file rate would still be 1.9 words/sec here, over the MIN_WORDS_PER_SEC
    floor, which is exactly the case the window check exists for.
    """
    segments = []
    # 300 s of normal speech at about 2.7 words/sec.
    for i in range(30):
        segments.append(_segment(i * 10.0, i * 10.0 + 10.0,
                                 "and so the word of the Lord came to him once again saying"))
    # 130 s of near-silence: speech spans continue, words do not.
    for i in range(13):
        segments.append(_segment(300.0 + i * 10.0, 300.0 + i * 10.0 + 10.0, "yes"))

    windows, low = transcribe_module.low_speech_windows(segments, 430.0)
    assert len(windows) >= 7
    assert low >= 2, f"expected the quiet stretch to trip at least two windows: {windows}"


def test_a_clean_file_trips_no_windows(transcribe_module):
    segments = [_segment(i * 10.0, i * 10.0 + 10.0,
                         "and so the word of the Lord came to him once again saying")
                for i in range(30)]
    _windows, low = transcribe_module.low_speech_windows(segments, 300.0)
    assert low == 0


def test_windows_use_speech_time_not_file_time(transcribe_module):
    """A ten minute break between segments must not read as ten minutes of collapse."""
    segments = [
        _segment(0.0, 60.0, " ".join(["word"] * 180)),
        _segment(660.0, 720.0, " ".join(["word"] * 180)),
    ]
    _windows, low = transcribe_module.low_speech_windows(segments, 720.0)
    assert low == 0


# --------------------------------------------------------------------------------------
# The loop flag (C7)
# --------------------------------------------------------------------------------------
def test_a_twenty_second_loop_is_flagged(transcribe_module):
    """A decoder loop: one phrase repeated for twenty seconds inside a single segment."""
    text = " ".join(["I want you to know that"] * 12)
    segments = [_segment(100.0, 120.0, text)]
    transcribe_module.annotate_segments(segments)
    assert "loop" in segments[0]["flags"]
    assert transcribe_module.repeated_4gram_rate(text) > 0.3


def test_rhetorical_repetition_does_not_requeue_a_job(transcribe_module):
    """Every segment the harness flagged on the reference set was genuine repetition.

    The flag is surfaced for review; it must never be part of the count that can send
    a job back to the queue.
    """
    text = "Eternal life is knowing God. Eternal life is knowing God."
    segments = [_segment(10.0, 16.0, text)]
    count, flagged = transcribe_module.annotate_segments(segments)
    assert segments[0]["flags"] == ["loop"]
    assert count == 0, "a loop flag is not an anomaly"
    assert flagged[0]["flags"] == ["loop"]


def test_short_segments_have_no_ngram_rate(transcribe_module):
    assert transcribe_module.repeated_4gram_rate("only three words") == 0.0
    assert transcribe_module.repeated_4gram_rate("") == 0.0


# --------------------------------------------------------------------------------------
# Boundary de-duplication
# --------------------------------------------------------------------------------------
def test_repeat_inside_one_segment_survives_intact(transcribe_module):
    """The regex this replaces turned this into "He is risen.  indeed."."""
    text = "He is risen. He is risen indeed."
    segments = [_segment(0.0, 4.0, text)]
    result = transcribe_module.deduplicate_segment_boundaries(segments)
    assert len(result) == 1
    assert result[0]["text"] == text


@pytest.mark.parametrize(
    "text",
    [
        "pray without ceasing. Pray without ceasing.",
        "day by day by day",
        "Holy, holy, holy is the Lord God Almighty",
    ],
)
def test_legitimate_repetition_is_never_touched(transcribe_module, text):
    segments = [_segment(0.0, 5.0, text)]
    result = transcribe_module.deduplicate_segment_boundaries(segments)
    assert result[0]["text"] == text


def test_boundary_duplicate_is_removed_and_the_start_moves(transcribe_module):
    first = _segment(0.0, 4.0, "and he said to them the Lord is good")
    second = _segment(4.0, 9.0, "the Lord is good and his mercy endures forever")
    result = transcribe_module.deduplicate_segment_boundaries([first, second])

    assert len(result) == 2
    assert result[0]["text"] == "and he said to them the Lord is good"
    assert result[1]["text"] == "and his mercy endures forever"
    # The start moves to the first surviving word, not the old segment bound.
    assert result[1]["start"] == pytest.approx(result[1]["words"][0]["start"])
    assert result[1]["start"] > 4.0


def test_the_longest_overlap_wins(transcribe_module):
    """Trimming the shortest match first would leave "the Lord" stranded."""
    first = _segment(0.0, 4.0, "I will bless the Lord")
    second = _segment(4.0, 8.0, "the Lord at all times")
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert result[1]["text"] == "at all times"


def test_case_and_punctuation_do_not_hide_a_duplicate(transcribe_module):
    first = _segment(0.0, 3.0, "he answered, the Lord is good.")
    second = _segment(3.0, 7.0, "The Lord is good, and I will praise him")
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert result[1]["text"] == "and I will praise him"


def test_a_single_shared_word_is_not_a_duplicate(transcribe_module):
    """One word is ordinary English, not a seam artefact; the floor is two."""
    first = _segment(0.0, 3.0, "and he went out to the")
    second = _segment(3.0, 6.0, "the mountain to pray alone")
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert result[1]["text"] == "the mountain to pray alone"


def test_a_fully_duplicated_segment_is_dropped(transcribe_module):
    first = _segment(0.0, 3.0, "glory to God in the highest")
    second = _segment(3.0, 5.0, "in the highest")
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert len(result) == 1
    assert result[0]["text"] == "glory to God in the highest"


def test_dedupe_works_without_word_timestamps(transcribe_module):
    """MFA and the tests both hand around segments that carry text only."""
    first = {"start": 0.0, "end": 3.0, "text": "the Lord is good", "words": None}
    second = {"start": 3.0, "end": 6.0, "text": "the Lord is good to me", "words": None}
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert result[1]["text"] == "to me"


def test_dedupe_does_not_mutate_its_input(transcribe_module):
    first = _segment(0.0, 3.0, "the Lord is good")
    second = _segment(3.0, 7.0, "the Lord is good to me")
    transcribe_module.deduplicate_segment_boundaries([first, second])
    assert second["text"] == "the Lord is good to me"


# --------------------------------------------------------------------------------------
# The API invariant
# --------------------------------------------------------------------------------------
def test_transcription_equals_the_joined_timings(transcribe_module, monkeypatch, tmp_path):
    """The one invariant 0.5.x broke on every job.

    `clean_boundary_duplicates` edited the transcript string and not the timings, so
    the two disagreed by 2-27 words per file and MFA received a third variant.
    """
    class _Word:
        def __init__(self, start, end, word):
            self.start, self.end, self.word, self.probability = start, end, word, 0.9

    class _Seg:
        def __init__(self, id_, start, end, text):
            self.id, self.seek = id_, 0
            self.start, self.end, self.text = start, end, text
            self.avg_logprob, self.compression_ratio = -0.05, 1.4
            self.no_speech_prob, self.temperature = 0.01, 0.0
            tokens = text.split()
            step = (end - start) / len(tokens)
            self.words = [
                _Word(start + i * step, start + (i + 1) * step, (" " if i else "") + t)
                for i, t in enumerate(tokens)
            ]

    raw = [
        _Seg(0, 0.0, 4.0, "and he said to them the Lord is good"),
        _Seg(1, 4.0, 9.0, "the Lord is good and his mercy endures forever"),
        _Seg(2, 9.0, 13.0, "He is risen. He is risen indeed."),
    ]

    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            return (raw, None)

    monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
    monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 13.0)
    monkeypatch.setattr(transcribe_module, "measure_mean_dbfs", lambda path, duration=0: -20.0)
    audio_path = tmp_path / "a.mp3"
    audio_path.write_bytes(b"x")

    result = transcribe_module.transcribe_audio(str(audio_path), "guid")

    assert " ".join(t["text"] for t in result["timings"]) == result["transcription"]
    # The seam duplicate went, the in-segment repetition stayed.
    assert "the Lord is good and his mercy" in result["transcription"]
    assert result["transcription"].count("the Lord is good") == 1
    assert "He is risen. He is risen indeed." in result["transcription"]
    # Timings stay ordered and non-overlapping.
    starts = [t["start"] for t in result["timings"]]
    assert starts == sorted(starts)
    for previous, current in zip(result["timings"], result["timings"][1:]):
        assert current["start"] >= previous["start"]


def test_transcribe_audio_returns_the_diagnostic_keys(transcribe_module, monkeypatch, tmp_path):
    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            return ([], None)

    monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
    monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 60.0)
    monkeypatch.setattr(transcribe_module, "measure_mean_dbfs", lambda path, duration=0: -20.0)
    audio_path = tmp_path / "a.mp3"
    audio_path.write_bytes(b"x")

    result = transcribe_module.transcribe_audio(str(audio_path), "guid")

    for key in ("transcription", "timings", "duration_sec", "segments", "anomaly_count",
                "anomaly_windows", "flagged_segments", "speech_seconds", "mean_dbfs",
                "vad_threshold"):
        assert key in result, f"missing {key}"
    assert result["timings"] == []
    assert result["anomaly_count"] == 0
    assert result["anomaly_windows"] == 0
