"""Unit tests for the GPU-free parts of the fidelity experiment.

The decoding needs a GPU and an audio archive. Everything that decides what the run
*means* is arithmetic over dictionaries and strings, and that is what these pin: the
infix word error rate that scores the scripture benchmark, the citation parser that finds
the passages, the two profile selectors the experiment compares, the agreement statistics
that stand in for the missing detector, and the configuration overlay's contract that
config A changes nothing.
"""

import json
import os
import sys

import pytest

SWEEP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "quality_sweep"
)
if SWEEP_DIR not in sys.path:
    sys.path.insert(0, SWEEP_DIR)

import agreement  # noqa: E402
import scripture  # noqa: E402
import select_fidelity_set  # noqa: E402


# --- infix word error rate -------------------------------------------------------------


def test_infix_wer_is_zero_when_the_reference_appears_verbatim_inside_the_hypothesis():
    reference = "moth and rust destroy".split()
    hypothesis = "so he read do not lay up moth and rust destroy and then he paused".split()
    wer, _ = scripture.infix_wer(reference, hypothesis)
    assert wer == 0.0


def test_infix_wer_is_one_when_the_passage_is_absent():
    reference = "moth and rust destroy".split()
    hypothesis = "he talked about something else entirely for a while".split()
    wer, _ = scripture.infix_wer(reference, hypothesis)
    assert wer == 1.0


def test_infix_wer_is_never_above_one_even_when_nothing_aligns():
    # Deleting the whole reference costs len(reference), so no alignment can be worse.
    reference = "a b c d e f g h".split()
    hypothesis = "z y x w v u t s r q p".split()
    wer, _ = scripture.infix_wer(reference, hypothesis)
    assert wer <= 1.0


def test_infix_wer_charges_one_error_per_substituted_word():
    reference = "moth and rust destroy".split()
    hypothesis = "before moth and rust consume after".split()
    wer, _ = scripture.infix_wer(reference, hypothesis)
    assert wer == pytest.approx(0.25)


def test_infix_wer_is_not_diluted_by_the_length_of_the_surrounding_sermon():
    """The whole point of the free suffix: a 4 word reading inside 4000 words of
    preaching scores on the reading, not on the sermon."""
    reference = "moth and rust destroy".split()
    short = "moth and rust destroy".split()
    long_hypothesis = ["filler"] * 2000 + short + ["filler"] * 2000
    assert scripture.infix_wer(reference, short)[0] == \
        scripture.infix_wer(reference, long_hypothesis)[0]


def test_infix_wer_handles_an_empty_hypothesis_and_an_empty_reference():
    assert scripture.infix_wer([], ["a"]) == (None, None)
    assert scripture.infix_wer(["a"], [])[0] == 1.0


def test_containment_counts_shared_ngrams_and_declines_short_references():
    reference = "a b c d e f".split()
    assert scripture.containment(reference, reference) == 1.0
    assert scripture.containment(reference, "x y z".split()) == 0.0
    assert scripture.containment("a b".split(), "a b".split()) is None


# --- citation parsing ------------------------------------------------------------------


def test_citations_read_the_forms_whisper_actually_writes():
    text = ("Matthew 6.19, 1 Peter 1.22, Second Timothy 3.16, Psalm 91.1, "
            "1 Kings 18.39, Revelations 21.4, Exodus 24:3")
    found = [(scripture.BOOKS[b - 1], c, v) for b, c, v, _ in scripture.citations(text)]
    assert found == [
        ("Matthew", 6, 19), ("1 Peter", 1, 22), ("2 Timothy", 3, 16), ("Psalms", 91, 1),
        ("1 Kings", 18, 39), ("Revelation", 21, 4), ("Exodus", 24, 3),
    ]


def test_citations_returns_the_offset_just_past_the_reference():
    text = "turn with me to Matthew 6.19 and follow along"
    (_book, _chapter, _verse, offset), = scripture.citations(text)
    assert text[offset:].startswith(" and follow along")


def test_citations_ignores_a_bare_book_name_with_no_verse():
    assert scripture.citations("the gospel of Matthew is where we are") == []


def test_citations_ignores_an_unknown_book():
    assert scripture.citations("Hezekiah 4.2 is not a book") == []


# --- the two profile selectors ---------------------------------------------------------


def test_level_selector_is_the_shipped_one_sided_rule():
    assert select_fidelity_set.level_profile(-20.0) == "loud"
    assert select_fidelity_set.level_profile(-26.0) == "loud"
    assert select_fidelity_set.level_profile(-26.1) == "quiet"
    assert select_fidelity_set.level_profile(None) == "quiet"


def test_fidelity_selector_sends_low_sample_rate_or_low_bit_rate_to_the_gentle_profile():
    assert select_fidelity_set.fidelity_profile(22050, 128000) == "quiet"
    assert select_fidelity_set.fidelity_profile(48000, 51000) == "quiet"
    assert select_fidelity_set.fidelity_profile(48000, 128000) == "loud"
    assert select_fidelity_set.fidelity_profile(44100, 64000) == "loud"


def test_fidelity_selector_falls_back_to_the_gentle_profile_when_the_encoding_is_unknown():
    """Same reasoning as the level rule's unknown branch: fragmenting quiet or
    low-fidelity speech loses words, cutting extra seams on good audio does not."""
    assert select_fidelity_set.fidelity_profile(None, None) == "quiet"


def test_the_two_selectors_genuinely_disagree_on_a_loud_low_bitrate_file():
    """tcf.20150424's shape: -17.8 dBFS, 48 kHz, 51 kbps. The level rule calls it loud
    and the fidelity rule calls it low, which is the whole point of running D."""
    assert select_fidelity_set.level_profile(-17.8) == "loud"
    assert select_fidelity_set.fidelity_profile(48000, 51452) == "quiet"


def test_read_levels_parses_the_tab_separated_measurement_file(tmp_path):
    path = tmp_path / "levels.csv"
    path.write_text("a.mp3\t-19.1\t-0.0\tmp3,48000,1,79767\n"
                    "b.mp3\t-28.4\t0.0\tmp3,22050,2,37889\n")
    levels = select_fidelity_set.read_levels(str(path))
    assert levels["a.mp3"] == {"mean_dbfs": -19.1, "codec": "mp3", "sample_rate": 48000,
                               "channels": 1, "bit_rate": 79767}
    assert levels["b.mp3"]["sample_rate"] == 22050


# --- agreement statistics ---------------------------------------------------------------


def test_compare_reports_perfect_agreement_and_no_run_for_identical_decodes():
    words = "the lord is my shepherd i shall not want".split()
    result = agreement.compare(words, words)
    assert result["agreement"] == 1.0
    assert result["max_run"] == 0


def test_compare_measures_a_contiguous_omission_as_a_run_on_the_longer_side():
    full = "before " + " ".join(f"w{i}" for i in range(40)) + " after"
    dropped = "before after"
    result = agreement.compare(full.split(), dropped.split())
    assert result["max_run"] == 40
    assert result["max_run_side"] == "a"


def test_compare_scores_a_substitution_as_disagreement_but_not_as_a_long_run():
    a = "moth and rust destroy the treasure".split()
    b = "moth and rust consume the treasure".split()
    result = agreement.compare(a, b)
    assert result["max_run"] == 1
    assert 0 < 1 - result["agreement"] < 0.2


def test_separation_reports_the_false_positive_cost_of_every_threshold():
    per_file = {
        "bad1.mp3": {"run": 90},
        "bad2.mp3": {"run": 10},
        "ok1.mp3": {"run": 0},
        "ok2.mp3": {"run": 20},
        "ok3.mp3": {"run": 1},
    }
    sep = agreement.separation(per_file, "run", {"bad1.mp3", "bad2.mp3"}, True)
    # Catching the worst bad file costs nothing; catching both costs the healthy file
    # that scores between them.
    assert sep["curve"][0]["caught"] == 1
    assert sep["curve"][0]["false_positives"] == 0
    assert sep["curve"][1]["caught"] == 2
    assert sep["curve"][1]["false_positives"] == 1


def test_separation_returns_none_when_one_side_of_the_split_is_empty():
    per_file = {"a.mp3": {"run": 1}, "b.mp3": {"run": 2}}
    assert agreement.separation(per_file, "run", set(), True) is None


def test_unhealthy_uses_only_evidence_independent_of_the_statistic_under_test():
    clean = {"speech_seconds": 1000, "words": 2600, "uncovered_max_gap_s": 3.0}
    assert agreement.unhealthy(clean, 2600) == []
    assert agreement.unhealthy({**clean, "words": 1000}, 2600) != []
    assert agreement.unhealthy({**clean, "uncovered_max_gap_s": 25.0}, 2600) != []
    assert agreement.unhealthy(clean, 3000) != []


# --- the configuration overlay ------------------------------------------------------------


class _FakeTranscribe:
    """Just enough of the image's module for the overlay to bind against."""

    LOUD_PROFILE = "loud"
    QUIET_PROFILE = "quiet"
    WHISPER_TEMPERATURE_BASE = 0.0

    @staticmethod
    def choose_vad_profile(mean_dbfs):
        return ("loud", "level") if (mean_dbfs or 0) >= -26.0 else ("quiet", "level")

    @staticmethod
    def rescue_decode_kwargs(primary):
        out = dict(primary)
        out["condition_on_previous_text"] = False
        out["temperature"] = (0.2, 0.4, 0.6, 0.8, 1.0)
        return out


@pytest.fixture()
def fake_transcribe():
    module = _FakeTranscribe()
    # Bind the staticmethods onto the instance so the overlay can replace them.
    module.choose_vad_profile = _FakeTranscribe.choose_vad_profile
    module.rescue_decode_kwargs = _FakeTranscribe.rescue_decode_kwargs
    return module


def _fidelity_pass():
    # fidelity_pass inserts /app on the path for the image's transcribe module, which is
    # absent outside the container; the import itself is deferred to main() so this is safe.
    import fidelity_pass
    return fidelity_pass


def test_config_a_is_the_shipped_code_with_nothing_overlaid(fake_transcribe):
    module = _fidelity_pass()
    before = (fake_transcribe.WHISPER_TEMPERATURE_BASE,
              fake_transcribe.choose_vad_profile,
              fake_transcribe.rescue_decode_kwargs)
    module.apply_config(fake_transcribe, "A")
    after = (fake_transcribe.WHISPER_TEMPERATURE_BASE,
             fake_transcribe.choose_vad_profile,
             fake_transcribe.rescue_decode_kwargs)
    assert before == after


def test_config_b_moves_only_the_primary_ladder_base(fake_transcribe):
    module = _fidelity_pass()
    module.apply_config(fake_transcribe, "B")
    assert fake_transcribe.WHISPER_TEMPERATURE_BASE == 0.2
    # The rescue and the selector are untouched, so any difference is the ladder's.
    assert fake_transcribe.rescue_decode_kwargs({})["condition_on_previous_text"] is False


def test_config_c_restores_conditioning_but_keeps_the_rescue_ladder(fake_transcribe):
    module = _fidelity_pass()
    module.apply_config(fake_transcribe, "C")
    kwargs = fake_transcribe.rescue_decode_kwargs({"beam_size": 5})
    assert kwargs["condition_on_previous_text"] is True
    assert kwargs["temperature"] == (0.2, 0.4, 0.6, 0.8, 1.0)
    assert kwargs["beam_size"] == 5
    # The primary is untouched, so C's first pass is A's first pass.
    assert fake_transcribe.WHISPER_TEMPERATURE_BASE == 0.0


def test_config_d_keys_the_profile_on_encoding_and_ignores_the_level(fake_transcribe):
    module = _fidelity_pass()
    module.apply_config(fake_transcribe, "D")
    module.CURRENT.clear()
    module.CURRENT.update({"sample_rate": 48000, "bit_rate": 51452})
    profile, why = fake_transcribe.choose_vad_profile(-17.8)  # loud by level
    assert profile == "quiet"
    assert "51452" in why
    module.CURRENT.update({"sample_rate": 48000, "bit_rate": 128000})
    assert fake_transcribe.choose_vad_profile(-30.0)[0] == "loud"  # quiet by level


def test_config_d_takes_the_gentle_profile_when_the_encoding_is_unknown(fake_transcribe):
    module = _fidelity_pass()
    module.apply_config(fake_transcribe, "D")
    module.CURRENT.clear()
    assert fake_transcribe.choose_vad_profile(-10.0)[0] == "quiet"


def test_an_unknown_config_label_is_refused(fake_transcribe):
    module = _fidelity_pass()
    with pytest.raises(SystemExit):
        module.apply_config(fake_transcribe, "Z")


# --- the committed passage set -----------------------------------------------------------


def test_the_scripture_passages_file_is_well_formed_and_covers_the_known_four():
    path = os.path.join(SWEEP_DIR, "scripture_passages.json")
    with open(path) as handle:
        passages = json.load(handle)
    known = {(p["file"], p["book_name"], p["chapter"]) for p in passages
             if p.get("source") == "known"}
    assert known == {
        ("tcf.20250607.mp3", "Exodus", 24),
        ("tcf.20210217.mp3", "Matthew", 6),
        ("tcf.20210210.mp3", "Matthew", 12),
        ("ucf20211106b.mp3", "1 Peter", 1),
    }
    for entry in passages:
        assert entry["first_verse"] <= entry["last_verse"]
        assert entry["ref_words"] >= 20
        assert 1 <= entry["book"] <= 66


def test_every_passage_reference_is_in_the_committed_cache():
    """The cache is committed so the benchmark can be re-scored with no network."""
    bible = scripture.Bible(os.path.join(SWEEP_DIR, "scripture_cache.json"), offline=True)
    with open(os.path.join(SWEEP_DIR, "scripture_passages.json")) as handle:
        passages = json.load(handle)
    for entry in passages:
        text = bible.passage("web", entry["book"], entry["chapter"],
                             entry["first_verse"], entry["last_verse"])
        assert text, f"no cached reference for {entry['label']}"


# --- the analyzer -------------------------------------------------------------------------


def _pass(words_by_file, **extra):
    out = {}
    for name, text in words_by_file.items():
        out[name] = {"transcription": text, "words": len(text.split()), "segments": 10,
                     "duration_sec": 600.0, "speech_seconds": 500.0,
                     "uncovered_max_gap_s": 2.0, "wall_s": 10.0, "vad_profile": "loud",
                     "anomaly_windows": 0, "rescue_attempted": False,
                     "rescue_selected": False, **extra.get(name, {})}
    return out


def _file_list(names, **overrides):
    return {"files": [{"file": n, "tags": ["control"], "bad_reasons": [],
                       "selector_split": False, "legacy_words": 100,
                       **overrides.get(n, {})} for n in names]}


def test_analyzer_scores_every_arm_against_the_control():
    import fidelity_analyze
    base = " ".join(f"w{i}" for i in range(100))
    configs = {
        "A": _pass({"a.mp3": base, "b.mp3": base}),
        "B": _pass({"a.mp3": base, "b.mp3": base + " extra words here"}),
    }
    doc = fidelity_analyze.build(configs, _file_list(["a.mp3", "b.mp3"]), {"A": "", "B": ""})
    assert doc["arms"] == ["B"]
    assert doc["per_file"]["a.mp3"]["B"]["word_delta"] == 0
    assert doc["per_file"]["a.mp3"]["B"]["agreement"] == 1.0
    assert doc["per_file"]["b.mp3"]["B"]["word_delta"] == 3
    assert doc["summary"]["B"]["n"] == 2


def test_analyzer_treats_a_replicate_as_an_arm_so_the_noise_floor_is_visible():
    import fidelity_analyze
    base = " ".join(f"w{i}" for i in range(100))
    configs = {
        "A": _pass({"a.mp3": base}),
        "A2": _pass({"a.mp3": base + " drifted"}),
        "B": _pass({"a.mp3": base}),
    }
    doc = fidelity_analyze.build(configs, _file_list(["a.mp3"]), {"A": "", "A2": "", "B": ""})
    assert doc["arms"] == ["A2", "B"]
    assert "A2" in doc["detector"] and "B" in doc["detector"]


def test_analyzer_renders_without_an_arm_that_was_not_run():
    import fidelity_analyze
    base = " ".join(f"w{i}" for i in range(50))
    configs = {"A": _pass({"a.mp3": base, "b.mp3": base}),
               "D": _pass({"a.mp3": base})}
    doc = fidelity_analyze.build(configs, _file_list(["a.mp3", "b.mp3"]), {"A": "", "D": ""})
    assert "D" in doc["per_file"]["a.mp3"]
    assert "D" not in doc["per_file"]["b.mp3"]
    assert "b.mp3" in fidelity_analyze.render(doc)


def test_analyzer_skips_a_control_entry_that_errored():
    import fidelity_analyze
    configs = {"A": {"a.mp3": {"error": "boom"}}, "B": _pass({"a.mp3": "one two"})}
    doc = fidelity_analyze.build(configs, _file_list(["a.mp3"]), {"A": "", "B": ""})
    assert doc["per_file"] == {}


def test_a_replicate_gets_a_default_note_saying_what_it_is(tmp_path):
    import fidelity_analyze
    path = tmp_path / "a2.json"
    path.write_text(json.dumps({"files": {}}))
    note, _ = fidelity_analyze.load_config(str(path), "A2")
    assert "noise floor" in note
