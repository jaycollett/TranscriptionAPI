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
    # Four words, the minimum the rule will act on.
    result = transcribe_module.deduplicate_segment_boundaries([first, second])

    assert len(result) == 2
    assert result[0]["text"] == "and he said to them the Lord is good"
    assert result[1]["text"] == "and his mercy endures forever"
    # The start moves to the first surviving word, not the old segment bound.
    assert result[1]["start"] == pytest.approx(result[1]["words"][0]["start"])
    assert result[1]["start"] > 4.0


def test_the_longest_overlap_wins(transcribe_module):
    """Trimming a shorter match first would leave the tail of the repeat stranded."""
    first = _segment(0.0, 4.0, "and I will bless the Lord at all times")
    second = _segment(4.0, 8.0, "the Lord at all times and his praise shall be")
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert result[1]["text"] == "and his praise shall be"


def test_case_and_punctuation_do_not_hide_a_duplicate(transcribe_module):
    first = _segment(0.0, 3.0, "he answered, the Lord is good.")
    second = _segment(3.0, 7.0, "The Lord is good, and I will praise him")
    # "the Lord is good" is four words, so the rule acts on it.
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert result[1]["text"] == "and I will praise him"


@pytest.mark.parametrize("tail, head", [
    ("and he went out to the", "the mountain to pray alone"),
    ("he said to them the Lord", "the Lord be with you always"),
    ("it was good in the sight of", "the sight of God and man"),
])
def test_a_short_shared_run_is_not_a_duplicate(transcribe_module, tail, head):
    """One to three shared words across a seam is ordinary English, not an artefact.

    At a two-word minimum this rule deleted genuine repetition; the floor is four.
    """
    first = _segment(0.0, 3.0, tail)
    second = _segment(3.0, 6.0, head)
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert result[1]["text"] == head


def test_a_fully_duplicated_segment_is_dropped(transcribe_module):
    first = _segment(0.0, 3.0, "glory to God in the highest heaven")
    second = _segment(3.0, 5.0, "in the highest heaven")
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert len(result) == 1
    assert result[0]["text"] == "glory to God in the highest heaven"


def test_dedupe_works_without_word_timestamps(transcribe_module):
    """MFA and the tests both hand around segments that carry text only."""
    first = {"start": 0.0, "end": 3.0, "text": "the Lord is good", "words": None}
    second = {"start": 3.0, "end": 6.0, "text": "the Lord is good to me", "words": None}
    # Four shared words.
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


# --------------------------------------------------------------------------------------
# The omission case: the failure the window check exists for
# --------------------------------------------------------------------------------------
def test_an_omission_is_counted_from_uncovered_vad_speech(transcribe_module):
    """40 minutes transcribed, 10 minutes emitted nothing at all.

    This is the shape the window check exists for and the shape it used to miss: the
    decoder produces no segment for the dropped audio, so it contributes no span, and
    a clock built only from the segments shrinks to fit the words that survived. The
    remaining 40 minutes look perfectly healthy at 2.6 words/sec. Only the VAD speech
    faster-whisper reports knows the other 10 minutes existed.
    """
    segments = [
        _segment(i * 10.0, i * 10.0 + 10.0,
                 "and so the word of the Lord came to him once again saying")
        for i in range(240)  # 2400 s of speech, about 2.6 words/sec
    ]
    speech_seconds = 3000.0  # the decoder was given 600 s more than it emitted

    _windows, blind = transcribe_module.low_speech_windows(segments, 3600.0)
    assert blind == 0, "the segment-span clock cannot see an omission; that is the bug"

    _windows, low = transcribe_module.low_speech_windows(segments, 3600.0, speech_seconds)
    assert low == 10, "600 s of uncovered VAD speech is ten silent windows"


def test_a_total_omission_is_counted(transcribe_module):
    """No segments at all: every second of VAD speech is uncovered."""
    _windows, low = transcribe_module.low_speech_windows([], 3600.0, 600.0)
    assert low == 10


def test_full_coverage_charges_nothing(transcribe_module):
    """A healthy file must not collect phantom windows from rounding."""
    segments = [
        _segment(i * 10.0, i * 10.0 + 10.0,
                 "and so the word of the Lord came to him once again saying")
        for i in range(30)
    ]
    _windows, low = transcribe_module.low_speech_windows(segments, 300.0, 300.0)
    assert low == 0


def test_a_small_coverage_shortfall_is_not_charged(transcribe_module):
    """VAD padding means coverage is always a little under the reported speech.

    Only whole 60 s units are charged, so the few seconds of padding around each
    chunk cannot manufacture a window on a healthy file.
    """
    segments = [
        _segment(i * 10.0, i * 10.0 + 10.0,
                 "and so the word of the Lord came to him once again saying")
        for i in range(30)
    ]
    _windows, low = transcribe_module.low_speech_windows(segments, 300.0, 355.0)
    assert low == 0, "55 s of padding is not a missing minute"


def test_the_omission_reaches_the_returned_counts(transcribe_module, monkeypatch, tmp_path):
    """End to end: duration_after_vad has to be read before the pass is scored."""
    class _Info:
        duration_after_vad = 3000.0

    class _Word:
        def __init__(self, start, end, word):
            self.start, self.end, self.word, self.probability = start, end, word, 0.9

    class _Seg:
        def __init__(self, start, end, text):
            self.id, self.seek = 0, 0
            self.start, self.end, self.text = start, end, text
            self.avg_logprob, self.compression_ratio = -0.05, 1.4
            self.no_speech_prob, self.temperature = 0.01, 0.0
            tokens = text.split()
            step = (end - start) / len(tokens)
            self.words = [
                _Word(start + i * step, start + (i + 1) * step, (" " if i else "") + t)
                for i, t in enumerate(tokens)
            ]

    raw = [_Seg(i * 10.0, i * 10.0 + 10.0,
                "and so the word of the Lord came to him once again saying")
           for i in range(240)]

    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            return (raw, _Info())

    monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
    monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 3600.0)
    monkeypatch.setattr(transcribe_module, "measure_mean_dbfs", lambda path, duration=0: -20.0)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")

    result = transcribe_module.transcribe_audio(str(audio), "guid")

    assert result["anomaly_windows"] == 10
    assert result["speech_seconds"] == pytest.approx(3000.0)


# --------------------------------------------------------------------------------------
# Liturgical repetition that straddles a seam
# --------------------------------------------------------------------------------------
def test_repetition_straddling_a_seam_survives(transcribe_module):
    """The reviewer's reproduction, and the reason the minimum is four words.

    At a two-word minimum this published as "The Lord is with you. He is risen.
    indeed. Alleluia.", which is the old regex's defect exactly. Every other
    preservation test keeps the repetition inside one segment, which is not the
    failing shape.
    """
    first = _segment(0.0, 4.0, "The Lord is with you. He is risen.")
    second = _segment(4.0, 8.0, "He is risen indeed. Alleluia.")
    result = transcribe_module.deduplicate_segment_boundaries([first, second])

    assert len(result) == 2
    assert result[0]["text"] == "The Lord is with you. He is risen."
    assert result[1]["text"] == "He is risen indeed. Alleluia."
    assert transcribe_module.transcript_from_segments(result) == \
        "The Lord is with you. He is risen. He is risen indeed. Alleluia."


@pytest.mark.parametrize("tail, head", [
    ("Christ has died. Christ is risen.", "Christ is risen. Christ will come again."),
    ("and the people said, Amen and amen.", "Amen and amen, forever and ever."),
    ("and they cried out, Holy, holy.", "Holy is the Lord of hosts."),
])
def test_call_and_response_across_a_seam_survives(transcribe_module, tail, head):
    first = _segment(0.0, 4.0, tail)
    second = _segment(4.0, 8.0, head)
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert result[1]["text"] == head, "a three word liturgical echo is content, not an artefact"


def test_a_four_word_response_across_a_seam_is_still_trimmed(transcribe_module):
    """The residual risk of the four-word minimum, stated rather than hidden.

    "Lord, hear our prayer" is four words, so a genuine call and response split
    across a seam is still removed. Four is where the rule stops eating ordinary
    English, not where it stops being wrong; a longer minimum would start missing
    real seam duplicates, which are also four to five words. This is exactly why
    every trim is logged: the sweep can count how often this shape occurs against
    how often a true duplicate does, and the minimum can be set on evidence.
    """
    first = _segment(0.0, 4.0, "Let us pray. Lord, hear our prayer.")
    second = _segment(4.0, 8.0, "Lord, hear our prayer, we beseech you.")
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert result[1]["text"] == "we beseech you."


def test_every_trim_is_logged_with_what_it_removed(transcribe_module, caplog):
    """The sweep has to be able to count the true-positive rate from the logs."""
    import logging

    first = _segment(0.0, 4.0, "and he said to them the Lord is good")
    second = _segment(4.0, 9.0, "the Lord is good and his mercy endures forever")
    with caplog.at_level(logging.INFO):
        transcribe_module.deduplicate_segment_boundaries([first, second], guid="abc-123")

    assert "Boundary dedupe" in caplog.text
    assert "abc-123" in caplog.text
    assert "4 words" in caplog.text
    assert "the Lord is good" in caplog.text


def test_a_trim_that_empties_a_segment_says_so(transcribe_module, caplog):
    import logging

    first = _segment(0.0, 3.0, "glory to God in the highest heaven")
    second = _segment(3.0, 5.0, "in the highest heaven")
    with caplog.at_level(logging.INFO):
        result = transcribe_module.deduplicate_segment_boundaries([first, second], guid="g")

    assert len(result) == 1
    assert "emptied and dropped" in caplog.text


# --------------------------------------------------------------------------------------
# The temperature step clamp
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("step", [0.0, -0.2])
def test_a_non_positive_temperature_step_cannot_hang_the_ladder(transcribe_module, step):
    """A zero or negative step makes the ladder loop forever and wedges the worker."""
    ladder = transcribe_module.temperature_ladder(0.0, step=max(0.01, step))
    assert ladder[-1] == pytest.approx(1.0)
    assert len(ladder) < 500


def test_the_configured_step_is_clamped_positive(transcribe_module):
    assert transcribe_module.WHISPER_TEMPERATURE_STEP > 0
