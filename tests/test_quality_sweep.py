"""Unit tests for the GPU-free parts of tools/quality_sweep.

The sweep itself needs a GPU, an audio archive and hours of wall time. Everything that
decides what the sweep means is arithmetic over dictionaries, and that is what these
tests pin: the stratum boundaries, the selection rules, the analyzer's distributions,
the regression rule, the words-per-second split, the level table and the runner's
timeout and multipart helpers.
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

import analyze  # noqa: E402
import runner  # noqa: E402
import select_files  # noqa: E402
import strata  # noqa: E402


# --- strata ---------------------------------------------------------------------------


def test_duration_buckets_are_half_open_and_cover_the_archive():
    assert strata.bucket_for(899.9, strata.DURATION_BUCKETS) == "short"
    assert strata.bucket_for(900.0, strata.DURATION_BUCKETS) == "medium"
    assert strata.bucket_for(1799.9, strata.DURATION_BUCKETS) == "medium"
    assert strata.bucket_for(2700.0, strata.DURATION_BUCKETS) == "very_long"
    assert strata.bucket_for(None, strata.DURATION_BUCKETS) == "unknown"


def test_level_buckets_straddle_the_vad_cutover():
    assert strata.bucket_for(-26.0, strata.LEVEL_BUCKETS) == "cutover_low"
    assert strata.bucket_for(-26.1, strata.LEVEL_BUCKETS) == "quiet"
    assert strata.bucket_for(-28.1, strata.LEVEL_BUCKETS) == "very_quiet"
    assert strata.bucket_for(-12.2, strata.LEVEL_BUCKETS) == "loud"


def test_vad_branch_follows_the_release_rule():
    assert strata.vad_branch(-24.5) == 0.5
    assert strata.vad_branch(-26.0) == 0.5
    assert strata.vad_branch(-26.01) == 0.35
    assert strata.vad_branch(-28.6) == 0.35
    assert strata.vad_branch(None) is None


def test_era_boundaries_match_the_release_tags():
    assert strata.era_for("2025-03-06 20:23:29") == "E1_0.1.5"
    assert strata.era_for("2025-03-20 00:00:00") == "E2_0.2.x"
    assert strata.era_for("2026-06-28 09:00:00") == "E3_0.3.x"
    assert strata.era_for("2026-09-07 06:31:32") == "E4_0.5.x"
    assert strata.era_for(None) == "unknown"


def test_multi_voice_detection():
    assert strata.is_multi_voice("women_retreat_2025_session3.mp3")
    assert strata.is_multi_voice("tcf.20240416_Formation_Class_Holy_Spirit.mp3")
    assert strata.is_multi_voice("cf.20220402.mens_breakfast_In_Christ.mp3")
    assert not strata.is_multi_voice("tcf.20240213b.mp3")


def test_readers_parse_the_measured_input_formats(tmp_path):
    durations = tmp_path / "durations.csv"
    durations.write_text("755,10485760,tcf.20240213b.mp3\n3388,20971520,women_retreat.mp3\n")
    levels = tmp_path / "levels.csv"
    levels.write_text(
        "tcf.20240213b.mp3\t-23.1\t-0.0\tmp3,48000,1,139000\n"
        "women_retreat.mp3\t-28.4\tNA\tmp3,48000,1,93000\n"
    )
    assert strata.read_durations(str(durations))["tcf.20240213b.mp3"] == 755.0
    parsed = strata.read_levels(str(levels))
    assert parsed["tcf.20240213b.mp3"]["mean_dbfs"] == -23.1
    assert parsed["tcf.20240213b.mp3"]["sample_rate"] == 48000
    assert parsed["women_retreat.mp3"]["max_dbfs"] is None


# --- selection ------------------------------------------------------------------------


def make_row(name, duration, dbfs, bitrate=90000, rate=48000, channels=1, era_ts="2025-03-06",
             wps=2.7):
    return strata.describe(
        name,
        {"words": int(duration * wps), "wps": wps, "transcription_finished_at": era_ts,
         "sermon_guid": "g-" + name},
        {"mean_dbfs": dbfs, "max_dbfs": -0.1, "codec": "mp3", "sample_rate": rate,
         "channels": channels, "bit_rate": bitrate},
        duration,
    )


def test_spread_takes_the_endpoints_and_even_steps():
    rows = {f"f{i}.mp3": make_row(f"f{i}.mp3", 600, -25.9 + i * 0.1) for i in range(10)}
    picked = select_files.spread(list(rows), rows, lambda r: r["mean_dbfs"], 4)
    assert len(picked) == 4
    levels = sorted(rows[n]["mean_dbfs"] for n in picked)
    assert levels[0] == pytest.approx(-25.9)
    assert levels[-1] == pytest.approx(-25.0)


def test_spread_returns_everything_when_the_quota_exceeds_the_pool():
    rows = {f"f{i}.mp3": make_row(f"f{i}.mp3", 600, -25.0 + i) for i in range(3)}
    assert len(select_files.spread(list(rows), rows, lambda r: r["mean_dbfs"], 9)) == 3


def test_mandatory_rules_pick_the_populations_they_name():
    rows = [
        make_row("quiet.mp3", 600, -29.0),
        make_row("rerun.mp3", 600, -20.0, era_ts="2026-09-07 06:31:32"),
        make_row("defective.mp3", 600, -20.0, era_ts="2026-07-01 00:00:00"),
        make_row("verylong.mp3", 3000, -20.0),
        make_row("stereo.mp3", 600, -20.0, channels=2),
        make_row("slow.mp3", 600, -20.0, wps=1.1),
        make_row("normal.mp3", 600, -20.0),
    ]
    sets = select_files.mandatory_sets(rows)
    assert sets["below_vad_cutover"] == ["quiet.mp3"]
    assert sets["rerun_2026_09_07"] == ["rerun.mp3"]
    assert sets["defective_era_E3"] == ["defective.mp3"]
    assert sets["very_long"] == ["verylong.mp3"]
    assert sets["non_mono_or_non_mp3"] == ["stereo.mp3"]
    assert sets["lowest_legacy_wps"][0] == "slow.mp3"


def test_stratified_fill_reaches_the_target_and_never_repeats():
    rows = [make_row(f"f{i:03d}.mp3", 600 + i * 30, -24.0 - (i % 5)) for i in range(60)]
    chosen = ["f000.mp3", "f001.mp3"]
    out = select_files.stratified_fill(rows, chosen, 20)
    assert len(out) == 20
    assert len(set(out)) == 20
    assert out[:2] == chosen


def test_stratified_fill_is_deterministic():
    rows = [make_row(f"f{i:03d}.mp3", 600 + i * 30, -24.0 - (i % 5)) for i in range(60)]
    first = select_files.stratified_fill(rows, [], 15)
    second = select_files.stratified_fill(rows, [], 15)
    assert first == second


def test_committed_file_list_is_one_hundred_files_with_reasons():
    path = os.path.join(SWEEP_DIR, "file_list.json")
    with open(path) as handle:
        doc = json.load(handle)
    assert len(doc["files"]) == 100
    assert len({row["file"] for row in doc["files"]}) == 100
    assert all(row["reasons"] for row in doc["files"])
    assert doc["strata_counts"]["total"] == 100


# --- analyzer arithmetic ---------------------------------------------------------------


def test_percentile_interpolates_and_tolerates_empty():
    assert analyze.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert analyze.percentile([1, 2, 3, 4], 0) == 1
    assert analyze.percentile([1, 2, 3, 4], 100) == 4
    assert analyze.percentile([], 50) is None
    assert analyze.percentile([None, 2, None], 50) == 2


def test_distribution_reports_the_expected_keys():
    dist = analyze.distribution([1.0, 2.0, 3.0, 4.0, 5.0])
    assert dist["n"] == 5
    assert dist["mean"] == pytest.approx(3.0)
    assert dist["median"] == pytest.approx(3.0)
    assert dist["min"] == 1.0 and dist["max"] == 5.0
    assert analyze.distribution([])["n"] == 0


def test_choose_wps_split_finds_a_real_gap():
    low = [0.2, 0.3, 0.45, 1.1]
    high = [2.6 + i * 0.01 for i in range(40)]
    split = analyze.choose_wps_split(low + high)
    assert split["method"] == "largest_gap_in_lower_quartile"
    assert 1.1 < split["threshold"] < 2.6
    assert split["below"] == 4
    assert split["above"] == 40


def test_choose_wps_split_says_so_when_the_tail_is_smooth():
    values = [2.4 + i * 0.01 for i in range(60)]
    split = analyze.choose_wps_split(values)
    assert split["method"] == "no_natural_gap_p5_fallback"
    assert split["threshold"] == pytest.approx(analyze.percentile(values, 5), rel=1e-6)


def test_choose_wps_split_handles_a_tiny_sample():
    assert analyze.choose_wps_split([2.7, 2.8])["method"] == "insufficient_data"


# --- analyzer over fixtures -------------------------------------------------------------


def fixture_state():
    def record(name, words, wall, duration, payload=None, outcome="completed"):
        payload = dict(payload or {})
        return {
            "file": name,
            "duration_s": duration,
            "outcome": outcome,
            "wall_s": wall,
            "payload": payload,
            "derived": {
                "words": words,
                "chars": words * 5,
                "wps": round(words / duration, 4),
                "timings": {"count": 10, "non_monotonic": 0, "overlaps": 0, "text_matches": True},
            },
        }

    return {
        "files": {
            # holds: 1000 -> 1010 words
            "hold.mp3": record("hold.mp3", 1010, 60.0, 400,
                               {"mfa_applied": True, "attempt_count": 1, "anomaly_count": 0,
                                "rescue_attempted": False, "rescue_selected": False,
                                "flagged_segments": []}),
            # regression: 1000 -> 900 words, ten percent down
            "drop.mp3": record("drop.mp3", 900, 70.0, 400,
                               {"mfa_applied": False, "attempt_count": 2, "anomaly_count": 3,
                                "rescue_attempted": True, "rescue_selected": True,
                                "flagged_segments": [{"start": 1, "end": 2, "reason": "rep4"}]}),
            # quiet file on the 0.35 branch, recovers strongly
            "quiet.mp3": record("quiet.mp3", 1400, 90.0, 400,
                                {"mfa_applied": True, "attempt_count": 1, "anomaly_count": 1,
                                 "rescue_attempted": True, "rescue_selected": False,
                                 "flagged_segments": []}),
            "failed.mp3": {"file": "failed.mp3", "duration_s": 400, "outcome": "error",
                           "wall_s": 20.0},
        }
    }


def fixture_baseline():
    return {
        "hold.mp3": {"words": 1000, "wps": 2.5, "transcription_finished_at": "2025-03-06"},
        "drop.mp3": {"words": 1000, "wps": 2.5, "transcription_finished_at": "2026-07-01"},
        "quiet.mp3": {"words": 1000, "wps": 2.5, "transcription_finished_at": "2025-03-06"},
        "failed.mp3": {"words": 1000, "wps": 2.5, "transcription_finished_at": "2025-03-06"},
    }


def fixture_file_list():
    return {
        "files": [
            {"file": "hold.mp3", "duration_s": 400, "mean_dbfs": -20.0,
             "level_bucket": "normal", "duration_bucket": "short", "legacy_era": "E1_0.1.5",
             "reasons": ["stratified_fill"], "multi_voice": False},
            {"file": "drop.mp3", "duration_s": 400, "mean_dbfs": -21.0,
             "level_bucket": "normal", "duration_bucket": "short", "legacy_era": "E3_0.3.x",
             "reasons": ["defective_era_E3"], "multi_voice": False},
            {"file": "quiet.mp3", "duration_s": 400, "mean_dbfs": -29.0,
             "level_bucket": "very_quiet", "duration_bucket": "short", "legacy_era": "E1_0.1.5",
             "reasons": ["below_vad_cutover"], "multi_voice": True},
            {"file": "failed.mp3", "duration_s": 400, "mean_dbfs": -20.0,
             "level_bucket": "normal", "duration_bucket": "short", "legacy_era": "E1_0.1.5",
             "reasons": ["stratified_fill"], "multi_voice": False},
        ],
        "supplementary": [],
    }


def test_build_rows_joins_result_baseline_and_strata():
    rows = {r["file"]: r for r in
            analyze.build_rows(fixture_state(), fixture_baseline(), fixture_file_list())}
    assert rows["hold.mp3"]["word_delta"] == 10
    assert rows["hold.mp3"]["word_delta_pct"] == pytest.approx(0.01)
    assert rows["drop.mp3"]["word_delta_pct"] == pytest.approx(-0.10)
    assert rows["quiet.mp3"]["vad_branch_expected"] == 0.35
    assert rows["hold.mp3"]["mfa_applied_present"] is True
    assert rows["failed.mp3"]["mfa_applied_present"] is False


def test_regressions_use_the_five_percent_rule():
    rows = analyze.build_rows(fixture_state(), fixture_baseline(), fixture_file_list())
    names = [r["file"] for r in analyze.regressions(rows)]
    assert names == ["drop.mp3"]
    assert [r["file"] for r in analyze.regressions(rows, tolerance=0.2)] == []


def test_rates_count_only_rows_that_carry_the_field():
    rows = analyze.build_rows(fixture_state(), fixture_baseline(), fixture_file_list())
    attempted = analyze.rate(rows, "rescue_attempted")
    assert attempted == {"present": 3, "true": 2, "false": 1, "rate": pytest.approx(0.6667, abs=1e-4)}
    selected = analyze.rate(rows, "rescue_selected")
    assert selected["true"] == 1
    mfa = analyze.rate(rows, "mfa_applied")
    assert mfa["present"] == 3 and mfa["true"] == 2


def test_counter_of_histograms_scalar_fields():
    rows = analyze.build_rows(fixture_state(), fixture_baseline(), fixture_file_list())
    assert analyze.counter_of(rows, "attempt_count") == {1: 2, 2: 1}
    assert analyze.counter_of(rows, "anomaly_count") == {0: 1, 1: 1, 3: 1}


def test_flag_summary_counts_files_and_segments():
    rows = analyze.build_rows(fixture_state(), fixture_baseline(), fixture_file_list())
    flags = analyze.flag_summary(rows)
    assert flags["files_with_field"] == 3
    assert flags["files_flagged"] == 1
    assert flags["segments_total"] == 1


def test_summarise_population_totals_and_outcomes():
    rows = analyze.build_rows(fixture_state(), fixture_baseline(), fixture_file_list())
    pop = analyze.summarise_population(rows)
    assert pop["files"] == 4
    assert pop["completed"] == 3
    assert pop["comparable"] == 3
    assert pop["outcomes"] == {"completed": 3, "error": 1}
    assert pop["word_delta"]["median"] == pytest.approx(10.0)
    assert pop["regressions"]["count"] == 1
    assert pop["timings"]["text_matches_timings"] == 3
    assert pop["timings"]["overlapping_files"] == 0


def test_level_table_marks_the_cutover_branch():
    rows = analyze.build_rows(fixture_state(), fixture_baseline(), fixture_file_list())
    table = {(e["dbfs_lo"], e["dbfs_hi"]): e for e in analyze.level_table(rows)}
    quiet = table[(-29, -28)]
    assert quiet["vad_branch_expected"] == 0.35
    assert quiet["at_or_above_cutover"] is False
    assert quiet["files"] == 1
    loud = table[(-21, -20)]
    assert loud["vad_branch_expected"] == 0.5
    assert loud["regressions"] == 1


def test_analyse_splits_populations_by_era_and_reason():
    analysis = analyze.analyse(fixture_state(), fixture_baseline(), fixture_file_list())
    assert analysis["files_attempted"] == 4
    assert "era_E1_0.1.5" in analysis["populations"]
    assert "era_E3_0.3.x" in analysis["populations"]
    assert analysis["populations"]["era_E3_0.3.x"]["regressions"]["count"] == 1
    assert analysis["populations"]["multi_voice"]["files"] == 1
    assert analysis["populations"]["overall"]["completed"] == 3


def test_render_markdown_produces_every_section():
    analysis = analyze.analyse(fixture_state(), fixture_baseline(), fixture_file_list())
    text = analyze.render_markdown(analysis)
    for heading in ("## Populations", "## Service behaviour",
                    "## Level against the VAD cutover", "## Regressions to read by hand",
                    "## Per file"):
        assert heading in text
    assert "drop.mp3" in text
    assert "not benchmark quality" in text


# --- runner helpers ---------------------------------------------------------------------


def test_timeout_scales_with_duration_and_has_a_floor():
    assert runner.timeout_for(0) == runner.TIMEOUT_FLOOR_S
    assert runner.timeout_for(None) == runner.TIMEOUT_FLOOR_S
    assert runner.timeout_for(200) == runner.TIMEOUT_FLOOR_S
    assert runner.timeout_for(3388) == pytest.approx(180.0 + 3.0 * 3388)


def test_multipart_body_carries_the_guid_and_the_file():
    body, ctype = runner.encode_multipart({"guid": "abc"}, "file", "x.mp3", b"BYTES")
    assert ctype.startswith("multipart/form-data; boundary=")
    assert b'name="guid"' in body
    assert b"abc" in body
    assert b'filename="x.mp3"' in body
    assert b"BYTES" in body
    assert body.rstrip().endswith(b"--")


def test_timing_stats_detect_overlap_and_regression_in_order():
    good = [{"start": 0.0, "end": 1.0, "text": "a"}, {"start": 1.0, "end": 2.0, "text": "b"}]
    assert runner.timing_stats(good) == {"count": 2, "non_monotonic": 0, "overlaps": 0}
    bad = [{"start": 0.0, "end": 2.0, "text": "a"}, {"start": 1.0, "end": 3.0, "text": "b"},
           {"start": 0.5, "end": 4.0, "text": "c"}]
    stats = runner.timing_stats(bad)
    assert stats["overlaps"] == 2
    assert stats["non_monotonic"] == 1
    assert runner.timing_stats([])["count"] == 0


def test_summarise_checks_the_transcript_against_the_timings():
    payload = {
        "transcription": "hello there world",
        "timings": [{"start": 0, "end": 1, "text": "hello there"},
                    {"start": 1, "end": 2, "text": "world"}],
    }
    summary = runner.summarise(payload, 60.0)
    assert summary["words"] == 3
    assert summary["wps"] == pytest.approx(0.05)
    assert summary["timings"]["text_matches"] is True

    payload["timings"][1]["text"] = "planet"
    assert runner.summarise(payload, 60.0)["timings"]["text_matches"] is False


def test_scratch_sweep_removes_only_the_jobs_guid(tmp_path):
    out = tmp_path / "out"
    scratch = tmp_path / "scratch"
    out.mkdir()
    scratch.mkdir()
    guid = "11111111-2222-3333-4444-555555555555"
    (scratch / f"{guid}.mp3").write_bytes(b"x" * 100)
    (scratch / f"{guid}.txt").write_text("hello")
    nested = scratch / f"{guid}_mfa_input"
    nested.mkdir()
    (nested / "audio.wav").write_bytes(b"y" * 500)
    keeper = scratch / "someone-elses.mp3"
    keeper.write_bytes(b"z" * 10)

    r = runner.Runner("http://127.0.0.1:1", str(tmp_path), str(out), scratch_dirs=[str(scratch)])
    freed = r.sweep_scratch(guid)

    assert freed >= 600
    assert keeper.exists()
    assert not (scratch / f"{guid}.mp3").exists()
    assert not nested.exists()


def test_scratch_sweep_is_a_no_op_without_scratch_dirs(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    r = runner.Runner("http://127.0.0.1:1", str(tmp_path), str(out))
    assert r.sweep_scratch("any-guid") == 0
