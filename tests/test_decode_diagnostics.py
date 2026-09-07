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
# Boundary de-duplication: the test is temporal, not lexical
# --------------------------------------------------------------------------------------
def _timed(pairs, **fields):
    """A segment built from explicit (start, end, word) triples.

    The word timestamps are the whole point here: they are what separates the same
    audio decoded twice from a speaker saying something twice.
    """
    words = [
        {"start": s, "end": e, "word": (" " if i else "") + w, "probability": 0.9}
        for i, (s, e, w) in enumerate(pairs)
    ]
    segment = {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": "".join(w["word"] for w in words).strip(),
        "avg_logprob": -0.06,
        "compression_ratio": 1.5,
        "no_speech_prob": 0.01,
        "temperature": 0.0,
        "words": words,
    }
    segment.update(fields)
    return segment


def _run(start, words, step=0.4):
    """Evenly spaced (start, end, word) triples beginning at `start`."""
    return [(round(start + i * step, 3), round(start + (i + 1) * step, 3), w)
            for i, w in enumerate(words)]


SEAM_WORDS = ["the", "Lord", "is", "good"]


def _artifact_pair():
    """The same four words decoded into both segments, over the same seconds.

    This is what a seam artifact is: one stretch of audio landing in two chunks.
    """
    tail = _run(8.0, ["and", "he", "said"] + SEAM_WORDS)
    # The next segment re-decodes 9.2-10.8, the span the tail's last four words occupy.
    head = _run(9.2, SEAM_WORDS + ["and", "his", "mercy", "endures"])
    return _timed(tail), _timed(head)


def _repetition_pair():
    """The speaker says the same four words twice, one after the other."""
    tail = _run(8.0, ["and", "he", "said"] + SEAM_WORDS)
    # Begins after the first occurrence finished: sequential, not overlapping.
    head = _run(11.0, SEAM_WORDS + ["and", "his", "mercy", "endures"])
    return _timed(tail), _timed(head)


def test_a_true_seam_artifact_is_trimmed(transcribe_module):
    """Overlapping spans mean the decoder produced the same audio twice."""
    first, second = _artifact_pair()
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert len(result) == 2
    assert result[1]["text"] == "and his mercy endures"
    assert result[1]["start"] == pytest.approx(result[1]["words"][0]["start"])


def test_the_same_words_said_twice_are_kept(transcribe_module):
    """Identical text, identical word count; only the timestamps differ."""
    first, second = _repetition_pair()
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert len(result) == 2
    assert result[1]["text"] == "the Lord is good and his mercy endures"


def test_the_1_kings_acclamation_survives(transcribe_module):
    """1 Kings 18:39. The sweep found four consecutive segments of this deleted.

    "The LORD, he is God; the LORD, he is God" is an acclamation whose entire force
    is the doubling, and the word-count rule emptied and dropped the segments
    carrying it.
    """
    first = _timed(_run(10.0, ["The", "LORD,", "he", "is", "God;"]))
    second = _timed(_run(12.6, ["the", "LORD,", "he", "is", "God."]))
    result = transcribe_module.deduplicate_segment_boundaries([first, second])

    assert len(result) == 2, "neither segment may be dropped"
    assert result[0]["text"] == "The LORD, he is God;"
    assert result[1]["text"] == "the LORD, he is God."


def test_anaphora_survives(transcribe_module):
    """"I want to love" repeated as a rhetorical opener; two emptied segments."""
    first = _timed(_run(4.0, ["I", "want", "to", "love", "him"]))
    second = _timed(_run(6.6, ["I", "want", "to", "love", "you"]))
    result = transcribe_module.deduplicate_segment_boundaries([first, second])

    assert len(result) == 2
    assert result[0]["text"] == "I want to love him"
    assert result[1]["text"] == "I want to love you"


def test_a_trim_never_empties_a_segment(transcribe_module):
    """Even a true overlap must not delete a segment whole.

    Dropping segments whole is how the scripture passage vanished, so when the
    repeated run is the entire segment the trim is declined.
    """
    tail = _timed(_run(8.0, ["glory", "to", "God", "in", "the", "highest"]))
    head = _timed(_run(9.2, ["in", "the", "highest"] + []))
    # Make the run four words so it is a candidate at all.
    tail = _timed(_run(8.0, ["glory", "to", "God"] + SEAM_WORDS))
    head = _timed(_run(9.2, SEAM_WORDS))
    result = transcribe_module.deduplicate_segment_boundaries([tail, head])

    assert len(result) == 2, "the segment must survive rather than be dropped"
    assert result[1]["text"] == "the Lord is good"


def test_declining_to_empty_a_segment_is_logged(transcribe_module, caplog):
    import logging

    tail = _timed(_run(8.0, ["glory", "to", "God"] + SEAM_WORDS))
    head = _timed(_run(9.2, SEAM_WORDS))
    with caplog.at_level(logging.INFO):
        transcribe_module.deduplicate_segment_boundaries([tail, head], guid="g")
    assert "declined to trim" in caplog.text


def test_a_short_overlapping_run_is_below_the_minimum(transcribe_module):
    """Four words stays a secondary condition; three overlapping words are ignored."""
    tail = _timed(_run(8.0, ["and", "he", "said", "the", "Lord"]))
    head = _timed(_run(8.8, ["the", "Lord", "be", "with", "you"]))
    result = transcribe_module.deduplicate_segment_boundaries([tail, head])
    assert result[1]["text"] == "the Lord be with you"


def test_without_word_timestamps_nothing_is_trimmed(transcribe_module):
    """No timestamps means no way to tell an artifact from repetition; publish both."""
    first = {"start": 0.0, "end": 3.0, "text": "the Lord is good", "words": None}
    second = {"start": 3.0, "end": 6.0, "text": "the Lord is good to me", "words": None}
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert result[1]["text"] == "the Lord is good to me"


def test_the_step_can_be_switched_off(transcribe_module, monkeypatch):
    """One variable disables the whole thing if overlap does not separate the cases."""
    monkeypatch.setattr(transcribe_module, "BOUNDARY_DEDUPE_ENABLED", False)
    first, second = _artifact_pair()
    result = transcribe_module.deduplicate_segment_boundaries([first, second])
    assert result[1]["text"] == "the Lord is good and his mercy endures"


def test_the_overlap_is_on_the_log_line(transcribe_module, caplog):
    """The sweep scores the rule from these, so the measurement has to be in them."""
    import logging

    first, second = _artifact_pair()
    with caplog.at_level(logging.INFO):
        transcribe_module.deduplicate_segment_boundaries([first, second], guid="abc-123")

    assert "Boundary dedupe" in caplog.text
    assert "abc-123" in caplog.text
    assert "overlap_sec=" in caplog.text
    assert "the Lord is good" in caplog.text


def test_a_kept_run_is_also_logged_with_its_overlap(transcribe_module, caplog):
    import logging

    first, second = _repetition_pair()
    with caplog.at_level(logging.INFO):
        transcribe_module.deduplicate_segment_boundaries([first, second], guid="abc-123")

    assert "kept a repeated run" in caplog.text
    assert "overlap_sec=" in caplog.text


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


def test_dedupe_does_not_mutate_its_input(transcribe_module):
    first, second = _artifact_pair()
    original = second["text"]
    transcribe_module.deduplicate_segment_boundaries([first, second])
    assert second["text"] == original


def test_the_longest_overlapping_run_wins(transcribe_module):
    """Trimming a shorter match first would leave the tail of the repeat stranded."""
    long_run = ["the", "Lord", "at", "all", "times"]
    tail = _timed(_run(8.0, ["and", "I", "will", "bless"] + long_run))
    head = _timed(_run(9.4, long_run + ["and", "his", "praise"]))
    result = transcribe_module.deduplicate_segment_boundaries([tail, head])
    assert result[1]["text"] == "and his praise"


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

    # Segment 1 re-decodes the seconds segment 0 already covered, which is what a
    # seam artifact is; segment 2 repeats inside itself and must survive whole.
    raw = [
        _Seg(0, 0.0, 4.0, "and he said to them the Lord is good"),
        _Seg(1, 2.5, 9.0, "the Lord is good and his mercy endures forever"),
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
# Coverage: bounded by the speech it is measured against, by construction
# --------------------------------------------------------------------------------------
def _dense(n, start=0.0, span=10.0):
    """`n` consecutive 10 s segments of normal-rate speech."""
    return [_segment(start + i * span, start + (i + 1) * span,
                     "and so the word of the Lord came to him once again saying")
            for i in range(n)]


def test_merge_intervals_sorts_and_absorbs_overlaps(transcribe_module):
    """A merge that only checks its predecessor double counts unsorted input."""
    merged = transcribe_module.merge_intervals([(5.0, 8.0), (0.0, 6.0), (20.0, 22.0)])
    assert merged == [(0.0, 8.0), (20.0, 22.0)]
    assert transcribe_module.merge_intervals([(1.0, 1.0), (2.0, 1.0)]) == []


def test_intersect_intervals(transcribe_module):
    a = [(0.0, 10.0), (20.0, 30.0)]
    b = [(5.0, 25.0)]
    assert transcribe_module.intersect_intervals(a, b) == [(5.0, 10.0), (20.0, 25.0)]
    assert transcribe_module.intersect_intervals(a, []) == []


def test_coverage_cannot_exceed_the_speech_it_is_measured_against(transcribe_module):
    """The defect: a plain sum of segment spans is unbounded above.

    Two overlapping segments and one running well past the end of the speech region.
    Summing spans gives 10 + 10 + 30 = 50 s against 20 s of speech, a ratio of 2.5;
    the sweep saw 1.087 in the wild, and above 1.0 the uncovered figure goes negative
    and the omission check is dead.
    """
    segments = [
        _segment(0.0, 10.0, "one two three four five"),
        _segment(5.0, 15.0, "six seven eight nine ten"),      # overlaps the first
        _segment(15.0, 45.0, "eleven twelve thirteen"),        # runs past the speech
    ]
    speech = [(0.0, 20.0)]

    covered = sum(e - s for s, e in transcribe_module.covered_speech(segments, speech))
    naive = sum(s["end"] - s["start"] for s in segments)

    assert naive == pytest.approx(50.0), "the old measure overcounts to 2.5x"
    assert covered == pytest.approx(20.0)
    assert covered <= sum(e - s for s, e in speech) + 1e-9


@pytest.mark.parametrize("segments, speech", [
    ([_segment(0.0, 10.0, "a b c"), _segment(5.0, 15.0, "d e f")], [(0.0, 12.0)]),
    ([_segment(0.0, 100.0, "a b c")], [(10.0, 20.0)]),
    (_dense(30), [(0.0, 250.0)]),
])
def test_the_coverage_ratio_is_never_above_one(transcribe_module, segments, speech):
    covered = sum(e - s for s, e in transcribe_module.covered_speech(segments, speech))
    total = sum(e - s for s, e in speech)
    assert covered <= total + 1e-9


def test_overlapping_segments_do_not_hide_an_omission(transcribe_module):
    """The regression the sweep found: inflated coverage disables the check.

    2400 s of speech is covered by segments that also overlap each other by 5 s
    apiece. Summed naively they exceed the 3000 s of speech and the omission
    disappears; intersected, the missing 600 s is still missing.
    """
    segments = []
    for i in range(240):
        segments.append(_segment(i * 10.0, i * 10.0 + 15.0,
                                 "and so the word of the Lord came to him once again saying"))
    speech = [(0.0, 3000.0)]

    naive = sum(s["end"] - s["start"] for s in segments)
    assert naive > 3000.0, "the naive sum exceeds the speech, which is the bug"

    _windows, low = transcribe_module.low_speech_windows(segments, 3600.0, speech)
    assert low >= 3, "the omission must still be charged"


# --------------------------------------------------------------------------------------
# The omission case: the failure the window check exists for
# --------------------------------------------------------------------------------------
def test_an_omission_is_counted_from_uncovered_speech(transcribe_module):
    """40 minutes transcribed, 10 minutes emitted nothing at all.

    This is the shape the window check exists for and the shape it used to miss: the
    decoder produces no segment for the dropped audio, so it contributes no span, and
    a clock built only from the segments shrinks to fit the words that survived. The
    remaining 40 minutes look perfectly healthy at 2.6 words/sec.
    """
    segments = _dense(240)  # 2400 s covered
    speech = [(0.0, 3000.0)]

    _windows, blind = transcribe_module.low_speech_windows(segments, 3600.0)
    assert blind == 0, "the segment-span clock cannot see an omission; that is the bug"

    _windows, low = transcribe_module.low_speech_windows(segments, 3600.0, speech)
    # 600 s uncovered, 300 s inside the 10 percent tolerance, 300 s charged.
    assert low == 5


def test_a_total_omission_is_counted(transcribe_module):
    """No segments at all: every second of speech is uncovered."""
    _windows, low = transcribe_module.low_speech_windows([], 3600.0, [(0.0, 600.0)])
    assert low == 9, "600 s uncovered less the 60 s tolerance is nine windows"


def test_full_coverage_charges_nothing(transcribe_module):
    segments = _dense(30)
    _windows, low = transcribe_module.low_speech_windows(segments, 300.0, [(0.0, 300.0)])
    assert low == 0


def test_a_small_coverage_shortfall_is_not_charged(transcribe_module):
    """Sub-second pauses between segments sum to minutes over a sermon."""
    segments = _dense(30)
    _windows, low = transcribe_module.low_speech_windows(segments, 400.0, [(0.0, 355.0)])
    assert low == 0, "55 s of breath pauses is not a missing minute"


@pytest.mark.parametrize("stem, speech_total, covered_frac", [
    ("tcf.20240213b", 734.9, 685.6 / 734.9),
    ("tcf.20240319b", 1259.4, 1192.5 / 1259.4),
    ("tcf.20240213a", 1438.9, 1337.4 / 1438.9),   # the worst measured, 7.1 percent
])
def test_healthy_files_are_not_charged_an_omission(transcribe_module, stem, speech_total,
                                                   covered_frac):
    """The measured coverage of three real reference decodes must score zero.

    On tcf.20240213a the 101.5 s uncovered is 373 sub-second pauses whose largest
    single member is 1.8 s; there is no omission anywhere in the file.
    """
    covered = speech_total * covered_frac
    segments = [_segment(0.0, covered, " ".join(["word"] * int(covered * 2.6)))]
    _windows, low = transcribe_module.low_speech_windows(
        segments, speech_total * 1.4, [(0.0, speech_total)]
    )
    assert low == 0, f"{stem} is healthy and must not be charged an omission"


@pytest.mark.parametrize("speech_total, covered_s, expected", [
    (1000.0, 950.0, 0),    # 5 percent, inside the tolerance
    (1000.0, 900.0, 0),    # exactly at the tolerance
    (1000.0, 830.0, 1),    # 17 percent, 70 s past
    (1000.0, 500.0, 6),    # half the speech missing
])
def test_the_tolerance_boundary(transcribe_module, speech_total, covered_s, expected):
    segments = [_segment(0.0, covered_s, " ".join(["word"] * int(covered_s * 2.6)))]
    _windows, low = transcribe_module.low_speech_windows(
        segments, speech_total * 1.5, [(0.0, speech_total)]
    )
    assert low == expected


def test_the_omission_reaches_the_returned_counts(transcribe_module, monkeypatch, tmp_path):
    """End to end: the detector runs and its intervals reach the score."""
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
            return (raw, None)

    monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
    monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 3600.0)
    monkeypatch.setattr(transcribe_module, "measure_mean_dbfs", lambda path, duration=0: -20.0)
    monkeypatch.setattr(transcribe_module, "prepare_audio",
                        lambda path, vad: ("AUDIO", [(0.0, 3000.0)], 4.2))
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")

    result = transcribe_module.transcribe_audio(str(audio), "guid")

    assert result["anomaly_windows"] == 5
    assert result["speech_seconds"] == pytest.approx(3000.0)


def test_a_detector_failure_disables_the_check_not_the_job(transcribe_module, monkeypatch,
                                                           tmp_path):
    """Losing the intervals must not lose the transcript."""
    def boom(path, vad):
        raise RuntimeError("no audio backend")

    seen = {}

    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            seen["source"] = audio
            return ([], None)

    monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
    monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 60.0)
    monkeypatch.setattr(transcribe_module, "measure_mean_dbfs", lambda path, duration=0: -20.0)
    monkeypatch.setattr(transcribe_module, "prepare_audio", boom)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")

    result = transcribe_module.transcribe_audio(str(audio), "guid")

    assert seen["source"] == str(audio), "the decode falls back to the file path"
    assert result["anomaly_windows"] == 0


def test_the_decoded_audio_is_reused_for_the_decode(transcribe_module, monkeypatch, tmp_path):
    """Decoding once and handing the array to the model keeps the cost to the detector."""
    seen = {}

    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            seen["source"] = audio
            return ([], None)

    monkeypatch.setattr(transcribe_module, "load_whisper_model", lambda: _FakeModel())
    monkeypatch.setattr(transcribe_module, "get_audio_duration", lambda path: 60.0)
    monkeypatch.setattr(transcribe_module, "measure_mean_dbfs", lambda path, duration=0: -20.0)
    monkeypatch.setattr(transcribe_module, "prepare_audio",
                        lambda path, vad: ("DECODED", [(0.0, 50.0)], 1.0))
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")

    transcribe_module.transcribe_audio(str(audio), "guid")
    assert seen["source"] == "DECODED"


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


def test_a_four_word_response_across_a_seam_now_survives(transcribe_module):
    """The residual risk of the word-count rule, closed by the temporal test.

    "Lord, hear our prayer" is four words, so the word-count rule trimmed it and a
    genuine call and response split across a seam lost its second half. The two
    occurrences are sequential in time, so the overlap test keeps both.
    """
    first = _timed(_run(20.0, ["Let", "us", "pray.", "Lord,", "hear", "our", "prayer."]))
    second = _timed(_run(23.2, ["Lord,", "hear", "our", "prayer,", "we", "beseech", "you."]))
    result = transcribe_module.deduplicate_segment_boundaries([first, second])

    assert len(result) == 2
    assert result[1]["text"] == "Lord, hear our prayer, we beseech you."


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
