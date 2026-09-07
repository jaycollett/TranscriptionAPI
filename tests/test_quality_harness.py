"""Unit tests for the GPU-free parts of tools/quality_harness: text normalisation,
decode metrics, the production and C2 selection rules, the C7 flagger, utterance
building and TextGrid writing, the window and I2 refinement rules, and the
verbatim production helper copies.
"""

import os
import sys

import pytest

HARNESS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "quality_harness")
if HARNESS_DIR not in sys.path:
    sys.path.insert(0, HARNESS_DIR)

import align  # noqa: E402
import configs  # noqa: E402
import metrics  # noqa: E402
import prod_replica  # noqa: E402
import selection  # noqa: E402
from decode import build_clips  # noqa: E402
from textnorm import norm_words  # noqa: E402


def seg(start, end, text, lp=-0.2, cr=1.3, nsp=0.01, temp=0.0, seek=None, probs=0.9):
    words = []
    toks = text.split()
    step = (end - start) / max(len(toks), 1)
    for i, tok in enumerate(toks):
        words.append({"start": start + i * step, "end": start + (i + 1) * step, "word": " " + tok, "probability": probs})
    return {
        "id": 0,
        "seek": seek if seek is not None else int(start * 100),
        "start": start,
        "end": end,
        "text": " " + text,
        "avg_logprob": lp,
        "compression_ratio": cr,
        "no_speech_prob": nsp,
        "temperature": temp,
        "words": words,
    }


# ------------------------------------------------------------------ textnorm
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello, World!", ["hello", "world"]),
        ("don’t Stop.", ["don't", "stop"]),
        ("John 3:16", ["john", "3", "16"]),
        ("  ", []),
        ("'quoted'", ["quoted"]),
    ],
)
def test_norm_words(text, expected):
    assert norm_words(text) == expected


# ------------------------------------------------------------------ metrics
def test_decode_metrics_basic_counts():
    segs = [seg(0, 10, "the quick brown fox jumps"), seg(10, 20, "over the lazy dog again", temp=0.6, cr=2.6)]
    m = metrics.decode_metrics(segs, "the quick brown fox jumps over the lazy dog again", 20.0, [(0, 20)], [(0, 20)])
    assert m["words"] == 10
    assert m["wps"] == 0.5
    assert m["seg_count"] == 2
    assert m["temp_ge_0_5"] == 1
    assert m["fallback_windows"] == 1
    assert m["cr_max"] == 2.6
    assert m["ngram_reject_flags"] == 1
    assert m["seam_count"] == 1
    assert m["removed_s"] == 0.0


def test_repeated_ngram_rate_detects_loops():
    loop = " ".join(["and he said to them"] * 6)
    clean = "one two three four five six seven eight nine ten eleven twelve"
    assert metrics.repeated_ngram_rate([seg(0, 30, loop)], 4) > 0.8
    assert metrics.repeated_ngram_rate([seg(0, 30, clean)], 4) == 0.0
    assert metrics.segment_repeated_4gram_rate(loop) > 0.3
    assert metrics.segment_repeated_4gram_rate("holy holy holy is the lord") == 0.0


def test_window_wps_uses_speech_clock():
    # 120 s file, speech only in 0-60; 60 words there. The second window is not speech.
    segs = [seg(0, 60, " ".join(f"w{i}" for i in range(60)))]
    windows, low = metrics.window_wps(segs, [(0, 60)], 120.0)
    assert windows == [1.0]
    assert low == 1
    windows, low = metrics.window_wps(segs, [], 120.0)
    assert windows == [1.0, 0.0]
    assert low == 2


def test_phantom_segments_match_stoplist():
    segs = [seg(0, 1, "Thank you."), seg(1, 5, "Thank you for coming tonight")]
    hits = metrics.phantom_segments(segs)
    assert len(hits) == 1 and hits[0]["text"] == "Thank you."


def test_agreement_localises_differences():
    a = [seg(0, 5, "we walk by faith not by sight")]
    b = [seg(0, 5, "we walk by faith not by site")]
    agr = metrics.agreement(a, b)
    assert agr["replace"] == 1 and agr["insert"] == 0 and agr["delete"] == 0
    assert agr["opcodes"][0]["a"] == "sight" and agr["opcodes"][0]["b"] == "site"
    assert agr["opcodes"][0]["t"] == pytest.approx(5 * 6 / 7, abs=0.01)
    assert metrics.agreement(a, a)["ratio"] == 1.0


def test_api_invariants():
    timings = [{"start": 0, "end": 2, "text": "a b"}, {"start": 1.5, "end": 3, "text": "c"}]
    inv = metrics.api_invariants("a b c", timings)
    assert inv["join_equals_transcript"] and inv["timings_overlaps"] == 1 and inv["timings_nonmono"] == 0


# ------------------------------------------------------------------ selection
def test_prod_rule_matches_transcribe_formula():
    good = {"segments": [seg(0, 100, " ".join(["w"] * 200), probs=0.9)], "transcript": " ".join(["w"] * 200)}
    short = {"segments": [seg(0, 100, " ".join(["w"] * 50), probs=0.99)], "transcript": " ".join(["w"] * 50)}
    best, per = selection.prod_rule([good, short], 100.0)
    # short has the higher raw confidence but 0.5 wps, scaled by 0.5/1.4
    assert per[1]["confidence"] == pytest.approx(0.99)
    assert per[1]["adjusted"] == pytest.approx(0.99 * 0.5 / 1.4, abs=1e-5)
    assert best == 0


def test_prod_rule_zero_duration_picks_first():
    a = {"segments": [seg(0, 10, "a b c", probs=0.5)], "transcript": "a b c"}
    b = {"segments": [seg(0, 10, "a b c", probs=0.9)], "transcript": "a b c"}
    best, per = selection.prod_rule([a, b], 0.0)
    assert best == 0 and all(p["adjusted"] == 0 for p in per)


def test_segscore_rejects_loop_and_truncation_where_prod_does_not():
    words = " ".join(f"w{i}" for i in range(150))
    clean = {"segments": [seg(0, 50, words), seg(50, 100, words)], "transcript": words + " " + words}
    # Loop: same word count, one segment tripped the ladder and compresses well.
    looped = {
        "segments": [seg(0, 50, words), seg(50, 100, " ".join(["and he said to them"] * 30), temp=0.8, cr=2.9)],
        "transcript": words + " " + " ".join(["and he said to them"] * 30),
    }
    # Truncation: the second half dropped entirely, still 1.5 wps so the 1.4 floor
    # never fires; the surviving words carry the highest probabilities.
    truncated = {"segments": [seg(0, 50, words, probs=0.95)], "transcript": words}
    prod_best, _ = selection.prod_rule([clean, looped, truncated], 100.0)
    seg_best, per = selection.segscore_rule([clean, looped, truncated], 100.0, [(0, 100)])
    assert prod_best == 2  # highest word probability wins under production
    assert seg_best == 0
    assert per[1]["anomalies"] == 1
    assert per[2]["low_windows"] == 1


def test_clip_choice_prefers_fewest_anomalies_then_logprob():
    cands = [("a", [seg(0, 5, "x y z", lp=-0.9, temp=0.6)]), ("b", [seg(0, 5, "x y z", lp=-0.5)]), ("c", [seg(0, 5, "x y", lp=-0.1)])]
    assert selection.clip_choice(cands) == 2
    assert selection.clip_choice(cands[:2]) == 1
    assert selection.clip_choice([("a", []), ("b", [seg(0, 5, "x")])]) == 1


def test_build_clips_groups_runs_under_30s():
    clips = build_clips([(0, 10), (12, 25), (26, 40), (41, 45), (50, 100)])
    assert clips == [(0, 25), (26, 45), (50, 80), (80, 100)]


# ------------------------------------------------------------------ prod replica
def test_prod_helpers_match_transcribe(transcribe_module):
    for base in (0.0, 0.2, 0.3):
        assert prod_replica._temperature_ladder(base) == transcribe_module.temperature_ladder(base)
    for text in ("He is risen. He is risen indeed.", "day by day by day", "no repeats here at all"):
        assert prod_replica._clean_boundary_duplicates(text) == transcribe_module.clean_boundary_duplicates(text)


def test_prod_pass_kwargs_are_the_production_call():
    kw = prod_replica.prod_pass_kwargs(prod_replica.PROD_PASSES[4])
    assert kw["beam_size"] == 15 and kw["patience"] == 3.5 and kw["condition_on_previous_text"] is False
    assert kw["temperature"] == (0.2, 0.4, 0.6, 0.8, 1.0)
    assert kw["vad_parameters"] == {"threshold": 0.35, "min_speech_duration_ms": 250, "min_silence_duration_ms": 300}
    assert kw["word_timestamps"] == "all" and kw["suppress_tokens"] == [-1]


def test_configs_are_deltas_from_base():
    c1 = configs.get_config("C1")
    assert c1["transcribe"] == configs.BASE["transcribe"]
    assert configs.get_config("C3")["transcribe"]["vad_filter"] is False
    assert configs.get_config("C10")["transcribe"]["temperature"] == [0.0, 0.4, 0.8, 1.0]
    assert configs.get_config("C8")["pipeline"] == "batched"
    assert len(configs.GLOSSARY.split()) < 30
    with pytest.raises(KeyError):
        configs.get_config("nope")


# ------------------------------------------------------------------ alignment helpers
def test_prod_transcript_text_collapses_four_repeats():
    segs = [{"text": " go go go go go now"}, {"text": " ok ok ok"}]
    # The production regex collapses exactly four repeats per match, so a fifth survives.
    assert align.prod_transcript_text(segs) == "go go now ok ok ok"


def test_build_utterances_splits_only_at_gaps():
    segs = [
        {"start": 0.0, "end": 4.0, "text": "a"},
        {"start": 4.1, "end": 9.0, "text": "b"},  # gap 0.1: never split
        {"start": 9.6, "end": 14.0, "text": "c"},  # gap 0.6, utterance is 9 s: split
        {"start": 14.2, "end": 40.0, "text": "d"},  # gap 0.2: no split even though > 30 s
        {"start": 41.0, "end": 45.0, "text": "e"},  # gap 1.0: split
    ]
    utts = align.build_utterances(segs, 50.0)
    assert [u["text"] for u in utts] == ["a b", "c d", "e"]
    assert utts[0]["start"] == 0.0
    assert utts[0]["end"] == pytest.approx(9.15)
    assert utts[1]["start"] == pytest.approx(9.45)
    assert utts[1]["end"] == pytest.approx(40.15)
    assert utts[2]["start"] == pytest.approx(40.85)
    assert utts[2]["end"] == pytest.approx(45.15)
    for k in range(1, len(utts)):
        assert utts[k]["start"] >= utts[k - 1]["end"]


def test_build_utterances_pads_at_most_half_the_gap():
    segs = [{"start": 0.0, "end": 10.0, "text": "a"}, {"start": 10.2, "end": 20.0, "text": "b"}]
    # Force a split at the 0.2 s gap by lowering gap_s.
    utts = align.build_utterances(segs, 20.0, gap_s=0.1)
    assert utts[0]["end"] == pytest.approx(10.1) and utts[1]["start"] == pytest.approx(10.1)


def test_write_textgrid_tiles_the_file(tmp_path):
    utts = [{"start": 1.0, "end": 3.0, "text": 'say "hi"'}, {"start": 4.0, "end": 6.0, "text": "bye"}]
    path = tmp_path / "x.TextGrid"
    align.write_textgrid(utts, 7.0, str(path))
    text = path.read_text()
    assert "intervals: size = 5" in text
    assert 'text = "say ""hi"""' in text
    assert 'name = "speaker"' in text
    assert "xmax = 7.0000" in text


def test_refine_window_reproduces_production_rule():
    segs = [seg(0.0, 2.0, "one two"), seg(2.0, 4.0, "three four")]
    words = [
        {"start": 0.1, "end": 0.5, "text": "one"},
        {"start": 0.6, "end": 1.0, "text": "two"},
        {"start": 1.9, "end": 2.4, "text": "three"},  # MFA onset leads Whisper: captured by segment 0
        {"start": 2.3, "end": 3.0, "text": "four"},
    ]
    refined, empties = align.refine_window(segs, words)
    assert refined[0]["end"] == 2.4 and refined[1]["start"] == 2.3
    assert align.count_nonmono(refined) == 1
    assert empties == 0


def test_refine_i2_matches_by_sequence_and_enforces_monotonic():
    segs = [seg(0.0, 2.0, "one two"), seg(2.0, 4.0, "three four"), seg(4.0, 5.0, "five")]
    words = [
        {"start": 0.1, "end": 0.5, "text": "one"},
        {"start": 0.6, "end": 1.0, "text": "two"},
        {"start": 1.9, "end": 2.4, "text": "three"},
        {"start": 2.5, "end": 3.0, "text": "four"},
    ]
    refined, empties, pre = align.refine_i2(segs, words)
    assert refined[0] == {"start": 0.1, "end": 1.0, "text": " one two", "owned": 2}
    assert refined[1]["start"] == 1.9 and refined[1]["end"] == 3.0
    assert refined[2]["fallback"] and empties == 1
    assert pre == 0 and align.count_nonmono(refined) == 0
    m = align.alignment_metrics(segs, refined, words, align.match_words(segs, words)[0], empties, "one two three four five")
    assert m["words_transcript"] == 5 and m["words_mfa"] == 4 and m["matched_words"] == 4
    assert m["drift_words"] == 0.0
    assert m["agree250"] == pytest.approx(1 / 3, abs=1e-4)  # only "five" (fallback) keeps Whisper edges within 250 ms


def test_i2_ignores_mfa_word_missing_from_whisper():
    segs = [seg(0.0, 2.0, "one two"), seg(2.0, 4.0, "three four")]
    words = [
        {"start": 0.1, "end": 0.5, "text": "one"},
        {"start": 0.6, "end": 1.0, "text": "amen"},
        {"start": 1.2, "end": 1.5, "text": "two"},
        {"start": 2.1, "end": 2.4, "text": "three"},
        {"start": 2.5, "end": 3.0, "text": "four"},
    ]
    records, ratio = align.match_words(segs, words)
    assert [r["mfa"] for r in records] == [0, 2, 3, 4]
    assert ratio < 1.0


def test_glossary_stats_counts_terms_and_leakage():
    terms = ["Holy Spirit", "Bonhoeffer", "Acts"]
    g = metrics.glossary_stats("The Holy Spirit came in Acts. Bonhoeffer wrote. holy spirit bonhoeffer acts", terms)
    assert g["per_term"] == {"Holy Spirit": 2, "Bonhoeffer": 2, "Acts": 2}
    assert g["total_hits"] == 6
    assert g["leakage"] == 2  # "holy spirit bonhoeffer" and "bonhoeffer acts" in listed order
    assert len(configs.GLOSSARY_TERMS) == len(configs.GLOSSARY.split(","))
