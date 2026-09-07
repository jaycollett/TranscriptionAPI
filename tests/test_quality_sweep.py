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
import determinism  # noqa: E402
import gaps  # noqa: E402
import legacy_eras  # noqa: E402
import runner  # noqa: E402
import select_files  # noqa: E402
import service_log  # noqa: E402
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
    good = [{"start": 0.0, "end": 1.0, "text": "a"}, {"start": 1.0, "end": 3.0, "text": "b"}]
    stats = runner.timing_stats(good)
    assert stats["count"] == 2 and stats["non_monotonic"] == 0 and stats["overlaps"] == 0
    assert stats["seg_mean_s"] == pytest.approx(1.5)
    assert stats["seg_max_s"] == pytest.approx(2.0)
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


# --- rescue, level table and the cutover view --------------------------------------------


def test_rescue_detail_lists_selected_files_with_their_delta():
    rows = analyze.build_rows(fixture_state(), fixture_baseline(), fixture_file_list())
    detail = analyze.rescue_detail(rows)
    assert detail["attempted"] == ["drop.mp3", "quiet.mp3"]
    assert [r["file"] for r in detail["selected"]] == ["drop.mp3"]
    assert detail["selected"][0]["word_delta"] == -100
    assert detail["selected"][0]["word_delta_pct"] == pytest.approx(-0.10)


def test_rescue_detail_is_empty_when_the_field_is_absent():
    state = {"files": {"a.mp3": {"file": "a.mp3", "outcome": "completed", "duration_s": 10,
                                 "payload": {}, "derived": {"words": 10, "wps": 1.0,
                                                            "timings": {}}}}}
    baseline = {"a.mp3": {"words": 10, "wps": 1.0, "transcription_finished_at": "2025-03-06"}}
    file_list = {"files": [{"file": "a.mp3", "duration_s": 10, "mean_dbfs": -20.0,
                            "level_bucket": "normal", "duration_bucket": "short",
                            "legacy_era": "E1_0.1.5", "reasons": [], "multi_voice": False}],
                 "supplementary": []}
    rows = analyze.build_rows(state, baseline, file_list)
    detail = analyze.rescue_detail(rows)
    assert detail["attempted"] == [] and detail["selected"] == []


def test_level_table_carries_segment_geometry():
    state = fixture_state()
    state["files"]["quiet.mp3"]["derived"]["timings"] = {
        "count": 900, "non_monotonic": 0, "overlaps": 0, "text_matches": True,
        "seg_mean_s": 1.2, "seg_max_s": 4.0,
    }
    rows = analyze.build_rows(state, fixture_baseline(), fixture_file_list())
    table = {(e["dbfs_lo"], e["dbfs_hi"]): e for e in analyze.level_table(rows)}
    quiet = table[(-29, -28)]
    assert quiet["segments_median"] == 900
    assert quiet["seg_mean_s_median"] == pytest.approx(1.2)
    assert quiet["words_total"] == 1400


def test_cutover_view_splits_on_the_edge_and_the_straddling_band():
    rows = analyze.build_rows(fixture_state(), fixture_baseline(), fixture_file_list())
    # Move two files into the band straddling the edge, one each side.
    rows[0]["mean_dbfs"] = -25.0
    rows[1]["mean_dbfs"] = -27.0
    view = analyze.cutover_view(rows)
    assert view["edge_dbfs"] == -26.0
    below, above = view["arms"]
    assert below["files"] == 2  # quiet.mp3 at -29 and the file moved to -27
    assert above["files"] == 2
    band_below, band_above = view["straddling_band"]
    assert band_below["files"] == 1  # -27 is inside -28 to -26; -29 is not
    assert band_above["files"] == 1  # -25 is inside -26 to -24; -20 is not


def test_cutover_view_ignores_files_with_no_level():
    rows = analyze.build_rows(fixture_state(), fixture_baseline(), fixture_file_list())
    for row in rows:
        row["mean_dbfs"] = None
    view = analyze.cutover_view(rows)
    assert view["arms"][0]["files"] == 0 and view["arms"][1]["files"] == 0


# --- determinism -------------------------------------------------------------------------


def det_record(name, words, anomaly, rescue_attempted, rescue_selected, duration=1000,
               segments=100):
    return {
        "file": name,
        "duration_s": duration,
        "outcome": "completed",
        "payload": {
            "anomaly_count": anomaly,
            "attempt_count": 1,
            "processing_seconds": 50.0,
            "rescue_attempted": rescue_attempted,
            "rescue_selected": rescue_selected,
            "mfa_applied": True,
        },
        "derived": {"words": words, "wps": words / duration,
                    "timings": {"count": segments, "seg_mean_s": 6.0}},
    }


def test_pair_reports_spreads_and_no_flip_when_the_runs_agree():
    row = determinism.pair(
        det_record("a.mp3", 3599, 0, False, False),
        det_record("a.mp3", 3598, 0, False, False),
    )
    assert row["word_spread"] == -1
    assert row["word_spread_pct"] == pytest.approx(-1 / 3599, abs=1e-5)
    assert row["anomaly_count_spread"] == 0
    assert row["rescue_decision_changed"] is False


def test_pair_flags_a_changed_rescue_decision():
    row = determinism.pair(
        det_record("a.mp3", 3599, 1, True, True),
        det_record("a.mp3", 3560, 0, False, False),
    )
    assert row["rescue_attempted_flipped"] is True
    assert row["rescue_selected_flipped"] is True
    assert row["rescue_decision_changed"] is True
    assert row["anomaly_count_spread"] == -1


def test_pair_leaves_flip_unknown_when_a_field_is_absent():
    a = det_record("a.mp3", 10, 0, False, False)
    b = det_record("a.mp3", 10, 0, False, False)
    del a["payload"]["rescue_attempted"]
    row = determinism.pair(a, b)
    assert row["rescue_attempted_flipped"] is None
    assert row["rescue_decision_changed"] is False


def test_compare_summarises_over_the_shared_files():
    state_a = {"files": {
        "a.mp3": det_record("a.mp3", 1000, 0, False, False, duration=500),
        "b.mp3": det_record("b.mp3", 3599, 2, True, True, duration=1369),
        "only_a.mp3": det_record("only_a.mp3", 10, 0, False, False),
    }}
    state_b = {"files": {
        "a.mp3": det_record("a.mp3", 1000, 0, False, False, duration=500),
        "b.mp3": det_record("b.mp3", 3585, 0, False, False, duration=1369),
    }}
    result = determinism.compare(state_a, state_b)
    assert result["pairs"] == 2
    assert result["max_abs_word_spread"] == 14
    assert result["max_abs_anomaly_spread"] == 2
    assert result["rescue_decision_changed"] == 1
    text = determinism.render_markdown(result)
    assert "b.mp3" in text and "only_a.mp3" not in text


def test_text_digest_reads_the_transcript_when_present(tmp_path):
    directory = tmp_path / "transcripts"
    directory.mkdir()
    (directory / "a.txt").write_text("Hello, there!")
    (directory / "b.txt").write_text("hello there")
    first = determinism.text_digest(str(directory), "a.mp3")
    second = determinism.text_digest(str(directory), "b.mp3")
    assert first == second  # normalisation makes punctuation and case irrelevant
    assert determinism.text_digest(str(directory), "missing.mp3") is None
    assert determinism.text_digest(None, "a.mp3") is None


# --- legacy era comparison ----------------------------------------------------------------


def era_rows():
    rows = []
    for i in range(6):
        rows.append(make_row(f"e1_{i}.mp3", 1000, -20.0, era_ts="2025-03-06", wps=2.8))
    for i in range(6):
        rows.append(make_row(f"e2_{i}.mp3", 1000, -20.0, era_ts="2025-06-01", wps=2.6))
    for i in range(2):
        rows.append(make_row(f"e4_{i}.mp3", 1000, -20.0, era_ts="2026-09-07", wps=3.0))
    return rows


def test_arms_drop_every_era_that_is_not_being_compared():
    single, multi = legacy_eras.arms(era_rows())
    assert len(single) == 6 and len(multi) == 6
    assert all(r["legacy_era"] == "E1_0.1.5" for r in single)


def test_raw_comparison_reports_the_median_difference_and_the_covariates():
    raw = legacy_eras.raw_comparison(era_rows())
    assert raw["single_pass"]["n"] == 6 and raw["multi_pass"]["n"] == 6
    assert raw["median_difference"] == pytest.approx(0.2, abs=1e-3)
    assert raw["covariates"]["single_pass_duration_median"] == 1000


def test_stratified_comparison_pools_only_cells_with_both_arms():
    rows = era_rows()
    # A cell with one multi-pass file must not contribute.
    rows.append(make_row("lonely.mp3", 3000, -29.0, era_ts="2025-06-01", wps=1.0))
    result = legacy_eras.stratified_comparison(rows)
    assert result["usable_cells"] == 1
    assert result["pooled_difference"] == pytest.approx(0.2, abs=1e-3)
    assert result["cells_favouring_single_pass"] == 1
    lonely = [c for c in result["cells"] if c["duration_bucket"] == "very_long"][0]
    assert lonely["usable"] is False and lonely["difference"] is None


def test_stratified_comparison_handles_no_usable_cells():
    rows = [make_row("a.mp3", 1000, -20.0, era_ts="2025-03-06")]
    result = legacy_eras.stratified_comparison(rows)
    assert result["usable_cells"] == 0
    assert result["pooled_difference"] is None


# --- service log harvesting ---------------------------------------------------------------

GUID_A = "7e02caa2-a1a9-4ba0-9cde-c072a68d9a26"
GUID_B = "7be50407-9e38-44ee-ad6e-c07f8dba392b"

LOG_LINES = [
    f"2026-09-07 15:40:50 - INFO - app - File a.mp3 received and saved as {GUID_A}.mp3 "
    f"with GUID {GUID_A}. Duration: 5095.7s. Estimated processing time: 4.88 min",
    "2026-09-07 15:41:00 - INFO - faster_whisper - Processing audio with duration 84:55.700",
    "2026-09-07 15:41:01 - INFO - faster_whisper - VAD filter removed 01:35.500 of audio",
    f"2026-09-07 15:41:30 - INFO - app - Seam trim for {GUID_A}: removed 2-word overlap "
    f"'he is risen' from segment 41 following 'he is risen indeed'",
    f"2026-09-07 15:41:31 - INFO - app - Seam trim for {GUID_A}: removed 3-word overlap "
    f"'pray without ceasing' from segment 88 following 'pray without ceasing'",
    f"2026-09-07 15:44:28 - INFO - app - Alignment for {GUID_A}: 11648 MFA words over 758 "
    f"segments, agree250=0.7718, clamped=4, 103 segments on Whisper word timings",
    f"2026-09-07 15:44:28 - INFO - app - Transcription completed for a.mp3 (GUID: {GUID_A}): "
    "13457 words in 5095.7s (2.64 words/sec, floor 1.0), 196.1s processing, "
    "mfa_applied=True, anomaly_count=0, anomaly_windows=0, flagged_segments=1, "
    "rescue_attempted=False, rescue_selected=False, speech_seconds=5000.2",
    f"2026-09-07 15:44:34 - INFO - app - File b.mp3 received and saved as {GUID_B}.mp3 "
    f"with GUID {GUID_B}. Duration: 3875.1s.",
    "2026-09-07 15:44:40 - INFO - faster_whisper - VAD filter removed 00:10.000 of audio",
    f"2026-09-07 15:48:00 - INFO - app - Transcription completed for b.mp3 (GUID: {GUID_B}): "
    "9000 words in 3875.1s (2.32 words/sec, floor 1.0), 180.0s processing, "
    "mfa_applied=True, anomaly_count=2, rescue_attempted=True, rescue_selected=True",
]


def test_parse_builds_one_record_per_guid():
    jobs = service_log.parse(LOG_LINES)
    assert list(jobs) == [GUID_A, GUID_B]
    assert jobs[GUID_A]["file"] == "a.mp3"
    assert jobs[GUID_B]["file"] == "b.mp3"


def test_parse_reads_the_completion_key_value_pairs_generically():
    fields = service_log.parse(LOG_LINES)[GUID_A]["fields"]
    assert fields["anomaly_count"] == 0
    assert fields["mfa_applied"] is True
    assert fields["rescue_selected"] is False
    assert fields["speech_seconds"] == 5000.2
    # A key the module never lists is still captured.
    assert fields["flagged_segments"] == 1


def test_parse_captures_the_alignment_counters():
    alignment = service_log.parse(LOG_LINES)[GUID_A]["alignment"]
    assert alignment["mfa_words"] == 11648
    assert alignment["mfa_segments"] == 758
    assert alignment["whisper_fallback_segments"] == 103
    assert alignment["agree250"] == 0.7718
    assert alignment["clamped"] == 4


def test_parse_collects_every_dedupe_trim_with_its_overlap_and_text():
    job = service_log.parse(LOG_LINES)[GUID_A]
    assert job["trim_count"] == 2
    assert job["trimmed_words"] == 5
    first = job["trims"][0]
    assert first["overlap_words"] == 2
    assert "he is risen" in first["quoted"]
    assert "he is risen indeed" in first["quoted"]


def test_parse_attributes_vad_lines_to_the_current_job():
    jobs = service_log.parse(LOG_LINES)
    assert jobs[GUID_A]["vad_removed_first_s"] == pytest.approx(95.5)
    assert jobs[GUID_B]["vad_removed_first_s"] == pytest.approx(10.0)


def test_speech_seconds_prefers_the_logged_value_then_falls_back():
    jobs = service_log.parse(LOG_LINES)
    # a.mp3 logs speech_seconds directly.
    assert jobs[GUID_A]["speech_seconds"] == pytest.approx(5000.2)
    # b.mp3 does not, so it is duration minus what VAD removed.
    assert jobs[GUID_B]["speech_seconds"] == pytest.approx(3875.1 - 10.0)


def test_speech_seconds_is_none_without_either_source():
    record = {"fields": {}, "duration_logged_s": None, "decode_duration_s": None,
              "vad_removed_first_s": None}
    assert service_log.speech_seconds(record) is None


def test_by_file_keys_records_on_the_filename():
    indexed = service_log.by_file(service_log.parse(LOG_LINES))
    assert set(indexed) == {"a.mp3", "b.mp3"}
    assert indexed["a.mp3"]["guid"] == GUID_A


def test_parse_ignores_a_log_with_nothing_it_recognises():
    assert service_log.parse(["hello", "2026-09-07 - INFO - werkzeug - GET /health 200"]) == {}


# --- coverage and dedupe in the analyzer ---------------------------------------------------


def service_jobs_fixture():
    return {
        "hold.mp3": {
            "guid": "g1", "file": "hold.mp3", "fields": {"speech_seconds": 380.0},
            "alignment": {"clamped": 2, "whisper_fallback_segments": 5},
            "trim_count": 1, "trimmed_words": 3, "speech_seconds": 380.0,
        },
        "drop.mp3": {
            "guid": "g2", "file": "drop.mp3", "fields": {"speech_seconds": 390.0},
            "alignment": {}, "trim_count": 0, "trimmed_words": 0, "speech_seconds": 390.0,
        },
        "quiet.mp3": {
            "guid": "g3", "file": "quiet.mp3", "fields": {"speech_seconds": 395.0},
            "alignment": {}, "trim_count": 0, "trimmed_words": 0, "speech_seconds": 395.0,
        },
    }


def state_with_spans():
    state = fixture_state()
    for name, span, count in (("hold.mp3", 370.0, 40), ("drop.mp3", 180.0, 20),
                              ("quiet.mp3", 350.0, 45)):
        state["files"][name]["derived"]["timings"] = {
            "count": count, "non_monotonic": 0, "overlaps": 0, "text_matches": True,
            "seg_mean_s": span / count, "seg_max_s": 20.0, "span_total_s": span,
        }
    return state


def test_attach_service_log_computes_coverage_and_the_speech_rate():
    rows = {r["file"]: r for r in analyze.build_rows(
        state_with_spans(), fixture_baseline(), fixture_file_list(), service_jobs_fixture()
    )}
    hold = rows["hold.mp3"]
    assert hold["speech_seconds"] == 380.0
    assert hold["coverage"] == pytest.approx(370.0 / 380.0, abs=1e-4)
    assert hold["new_wps_speech"] == pytest.approx(1010 / 380.0, abs=1e-4)
    assert hold["legacy_wps_speech"] == pytest.approx(1000 / 380.0, abs=1e-4)
    assert hold["clamped"] == 2
    assert hold["whisper_fallback_segments"] == 5


def test_attach_service_log_leaves_everything_none_without_a_record():
    rows = {r["file"]: r for r in analyze.build_rows(
        state_with_spans(), fixture_baseline(), fixture_file_list(), None
    )}
    assert rows["hold.mp3"]["coverage"] is None
    assert rows["hold.mp3"]["trimmed_words"] is None


def test_coverage_summary_counts_the_files_below_each_band():
    rows = analyze.build_rows(
        state_with_spans(), fixture_baseline(), fixture_file_list(), service_jobs_fixture()
    )
    cov = analyze.coverage_summary(rows)
    assert cov["measured"] == 3
    # drop.mp3 covers 180 of 390 speech seconds, well under half; quiet.mp3 covers
    # 350 of 395, under the 0.95 band but nowhere near the 0.5 one.
    assert cov["counts"]["under_0.5"] == 1
    assert cov["counts"]["under_0.95"] == 2
    assert cov["counts"]["under_0.7"] == 1
    assert cov["worst"][0]["file"] == "drop.mp3"


def test_dedupe_summary_separates_the_trim_from_the_decode():
    rows = analyze.build_rows(
        state_with_spans(), fixture_baseline(), fixture_file_list(), service_jobs_fixture()
    )
    ded = analyze.dedupe_summary(rows)
    assert ded["measured"] == 3
    assert ded["files_trimmed"] == 1
    assert ded["words_removed_total"] == 3
    entry = ded["files"][0]
    assert entry["file"] == "hold.mp3"
    assert entry["words"] == 1010
    assert entry["words_before_dedupe"] == 1013


def test_analyse_and_render_survive_a_run_with_a_service_log():
    analysis = analyze.analyse(
        state_with_spans(), fixture_baseline(), fixture_file_list(), service_jobs_fixture()
    )
    assert analysis["populations"]["overall"]["coverage"]["measured"] == 3
    text = analyze.render_markdown(analysis)
    assert "## Segment coverage of the VAD speech seconds" in text
    assert "## Seam de-duplication" in text
    assert "words before" in text


def test_analyse_and_render_survive_a_run_without_a_service_log():
    analysis = analyze.analyse(fixture_state(), fixture_baseline(), fixture_file_list())
    text = analyze.render_markdown(analysis)
    assert "Not measured: no service log was supplied." in text


# --- interval arithmetic and uncovered speech ----------------------------------------------


def test_merge_coalesces_and_sorts():
    assert gaps.merge([(5, 7), (0, 3), (2, 4)]) == [(0.0, 4.0), (5.0, 7.0)]
    assert gaps.merge([(1, 2), (2, 3)]) == [(1.0, 3.0)]
    assert gaps.merge([(3, 1), (None, 5)]) == []


def test_complement_covers_the_holes_and_the_edges():
    assert gaps.complement([(2, 4)], 0, 10) == [(0, 2), (4, 10)]
    assert gaps.complement([], 0, 5) == [(0, 5)]
    assert gaps.complement([(0, 5)], 0, 5) == []
    # Intervals outside the window are ignored.
    assert gaps.complement([(-5, -1), (20, 30)], 0, 10) == [(0, 10)]


def test_subtract_removes_overlaps_and_keeps_the_rest():
    assert gaps.subtract([(0, 10)], [(2, 4)]) == [(0, 2), (4, 10)]
    assert gaps.subtract([(0, 10)], [(0, 10)]) == []
    assert gaps.subtract([(0, 10)], []) == [(0.0, 10.0)]
    assert gaps.subtract([(0, 5), (10, 15)], [(3, 12)]) == [(0, 3), (12, 15)]


def test_total_sums_interval_lengths():
    assert gaps.total([(0, 2), (5, 6.5)]) == 3.5
    assert gaps.total([]) == 0


def test_analyse_file_finds_a_contiguous_omission():
    # 100 s file, silent 40 to 50, decoder emitted everything except 60 to 90.
    silences = [(40, 50)]
    spans = [(0, 40), (50, 60), (90, 100)]
    result = gaps.analyse_file(silences, spans, 100)
    assert result["speech_s"] == 90.0
    assert result["uncovered_s"] == 30.0
    assert result["uncovered_fraction"] == pytest.approx(30 / 90, abs=1e-4)
    assert result["largest_gap_s"] == 30.0
    assert result["gap_count"] == 1
    assert result["gaps_over"]["over_15.0s"] == 1
    assert result["top_gaps"][0] == {"start": 60.0, "end": 90.0, "length_s": 30.0}


def test_analyse_file_treats_scattered_breathing_differently_from_an_omission():
    # Same uncovered total, spread over many small gaps instead of one long one.
    spans = []
    cursor = 0.0
    while cursor < 100:
        spans.append((cursor, cursor + 7.0))
        cursor += 10.0
    scattered = gaps.analyse_file([], spans, 100)
    contiguous = gaps.analyse_file([], [(0, 70)], 100)
    assert scattered["uncovered_s"] == pytest.approx(contiguous["uncovered_s"], abs=1.0)
    assert scattered["largest_gap_s"] == pytest.approx(3.0)
    assert contiguous["largest_gap_s"] == pytest.approx(30.0)
    assert scattered["gaps_over"]["over_15.0s"] == 0
    assert contiguous["gaps_over"]["over_15.0s"] == 1


def test_analyse_file_reports_full_coverage_as_zero_uncovered():
    result = gaps.analyse_file([], [(0, 100)], 100)
    assert result["uncovered_s"] == 0.0
    assert result["largest_gap_s"] == 0.0
    assert result["uncovered_fraction"] == 0.0


def test_analyse_file_needs_a_duration():
    assert gaps.analyse_file([], [(0, 10)], 0) is None
    assert gaps.analyse_file([], [(0, 10)], None) is None


def test_recommend_scales_the_p95_and_respects_the_floor():
    # p95 of twenty values interpolates between the 19th and 20th, so one high outlier
    # moves it only a little: the recommendation follows the body, not the tail.
    rec = gaps.recommend([0.01] * 19 + [0.08])
    assert rec["p95"] == pytest.approx(0.0135, abs=1e-4)
    assert rec["value"] == pytest.approx(max(rec["p95"] * 1.5, 0.02), abs=1e-4)
    # A corpus whose body really is high pulls the recommendation up with it.
    high = gaps.recommend([0.06] * 19 + [0.09])
    assert high["value"] > 0.09
    # A corpus with almost no uncovered speech still gets the floor, not zero.
    assert gaps.recommend([0.0] * 20)["value"] == 0.02
    assert gaps.recommend([]) is None


def test_summarise_counts_files_over_the_shipped_tolerance():
    per_file = {
        "clean.mp3": gaps.analyse_file([], [(0, 99)], 100),
        "omitted.mp3": gaps.analyse_file([], [(0, 50)], 100),
    }
    summary = gaps.summarise(per_file, tolerance=0.10)
    assert summary["files"] == 2
    assert summary["over_tolerance"]["count"] == 1
    assert summary["over_tolerance"]["files"][0]["file"] == "omitted.mp3"
    assert summary["worst_largest_gap"][0]["file"] == "omitted.mp3"
    text = gaps.render_markdown(summary, per_file)
    assert "omitted.mp3" in text and "Recommended tolerance" in text


def test_build_skips_files_that_did_not_complete(tmp_path):
    silence = tmp_path / "silence.jsonl"
    silence.write_text(
        json.dumps({"file": "a.mp3", "threshold_db": -34.0, "silences": [[10, 20]]}) + "\n"
        + json.dumps({"file": "b.mp3", "threshold_db": -34.0, "silences": []}) + "\n"
    )
    timings = tmp_path / "timings"
    timings.mkdir()
    (timings / "a.json").write_text(json.dumps(
        {"timings": [{"start": 0, "end": 10}, {"start": 20, "end": 100}]}
    ))
    (timings / "b.json").write_text(json.dumps({"timings": [{"start": 0, "end": 100}]}))
    state = {"files": {
        "a.mp3": {"outcome": "completed", "duration_s": 100},
        "b.mp3": {"outcome": "error", "duration_s": 100},
    }}
    built = gaps.build(str(silence), str(timings), state)
    assert set(built) == {"a.mp3"}
    assert built["a.mp3"]["uncovered_s"] == 0.0
    assert built["a.mp3"]["threshold_db"] == -34.0


# --- rc2 log lines -------------------------------------------------------------------------

RC2_LINES = [
    f"INFO - transcribe - Audio level for {GUID_A}: -28.4 dBFS; VAD profile 'quiet' "
    f"(-28.4 dBFS below the -26.0 dBFS cutover): {{'threshold': 0.35}}",
    f"INFO - transcribe - primary pass for {GUID_A} in 78.8s: 8904 words, 435 segments "
    f"covering 3100.0s of 3231.0s VAD speech, anomaly_count=3, anomaly_windows=1 of 54, "
    f"flagged_segments=2",
    f"INFO - transcribe - rescue pass for {GUID_A} in 74.1s: 8880 words, 430 segments "
    f"covering 3190.0s of 3231.0s VAD speech, anomaly_count=0, anomaly_windows=0 of 54, "
    f"flagged_segments=1",
    f"INFO - transcribe - Boundary dedupe for {GUID_A}: dropped 5 words repeated across "
    f"the seam at 812.30s: 'lord in your mercy hear'",
    f"INFO - app - Alignment for {GUID_A}: 8918 MFA words over 435 segments, agree250=0.786 "
    f"empty_fallbacks=2 span_ratio_lt_0_5=7 span_ratio_p5=0.41 span_ratio_median=0.98 clamped=3",
]


def test_rc2_level_line_gives_the_profile_actually_taken():
    job = service_log.parse(RC2_LINES)[GUID_A]
    assert job["audio_level_dbfs"] == -28.4
    assert job["vad_profile"] == "quiet"


def test_rc2_pass_lines_give_coverage_for_each_pass():
    job = service_log.parse(RC2_LINES)[GUID_A]
    assert len(job["passes"]) == 2
    primary = job["primary_pass"]
    assert primary["covered_s"] == 3100.0
    assert primary["vad_speech_s"] == 3231.0
    assert primary["uncovered_s"] == 131.0
    assert primary["uncovered_fraction"] == pytest.approx(131 / 3231, abs=1e-5)
    assert primary["windows_total"] == 54
    assert primary["anomaly_windows"] == 1
    assert job["rescue_pass"]["words"] == 8880
    assert job["service_uncovered_fraction"] == primary["uncovered_fraction"]


def test_rc2_trim_line_is_parsed_exactly_not_loosely():
    job = service_log.parse(RC2_LINES)[GUID_A]
    assert job["trim_count"] == 1
    assert job["trimmed_words"] == 5
    trim = job["trims"][0]
    assert trim["matched"] == "exact"
    assert trim["overlap_words"] == 5
    assert trim["at_s"] == 812.30
    assert trim["removed"] == "lord in your mercy hear"
    assert trim["emptied"] is False
    assert job["loose_trim_matches"] == 0


def test_rc2_alignment_counters_include_the_span_ratios_and_the_clamp():
    alignment = service_log.parse(RC2_LINES)[GUID_A]["alignment"]
    assert alignment["span_ratio_lt_0_5"] == 7
    assert alignment["span_ratio_p5"] == 0.41
    assert alignment["span_ratio_median"] == 0.98
    assert alignment["clamped"] == 3
    assert alignment["empty_fallbacks"] == 2


def test_a_reworded_trim_line_is_still_captured_as_a_loose_match():
    lines = [f"INFO - transcribe - Seam de-dup for {GUID_A}: removed 4 words 'a b c d'"]
    job = service_log.parse(lines)[GUID_A]
    assert job["trim_count"] == 1
    assert job["trims"][0]["matched"] == "loose"
    assert job["loose_trim_matches"] == 1


def test_a_trim_logged_once_per_pass_is_counted_once():
    # When the rescue pass runs, the dedupe runs again and logs the same trim a second
    # time. Only the distinct phrases correspond to text missing from the transcript.
    line = (f"INFO - transcribe - Boundary dedupe for {GUID_A}: dropped 4 words repeated "
            f"across the seam at 2297.68s: 'I want to love,' (segment emptied and dropped)")
    job = service_log.parse([line, line, line])[GUID_A]
    assert job["trims_logged"] == 3
    assert job["trim_count"] == 1
    assert job["trimmed_words"] == 4


def test_two_trims_at_different_times_stay_separate():
    a = (f"INFO - transcribe - Boundary dedupe for {GUID_A}: dropped 4 words repeated "
         f"across the seam at 100.00s: 'the lord he is'")
    b = (f"INFO - transcribe - Boundary dedupe for {GUID_A}: dropped 5 words repeated "
         f"across the seam at 200.00s: 'the lord he is god'")
    job = service_log.parse([a, b])[GUID_A]
    assert job["trim_count"] == 2
    assert job["trimmed_words"] == 9
