"""Join the sweep results to the legacy baseline and write the report.

Everything below the `main` boundary is pure arithmetic over dictionaries, so the unit
tests in `tests/test_quality_sweep.py` exercise the real functions against small
fixtures rather than a mock of them.

Three things shape the output:

1. **Populations.** The legacy transcripts were not all produced by the same code. Each
   row carries the era of the release that wrote it (`strata.ERAS`), and the report
   breaks every table down by era as well as overall. The 0.5.4 re-run era (E4) is not
   legacy output at all and is reported separately so it is never averaged in with it.
2. **The low-rate split.** The brief expected two populations, healthy and collapsed.
   The split is computed from the data (`choose_wps_split`) instead of assumed, and the
   report states the threshold it found and how many files fall each side.
3. **The level-versus-VAD table.** One row per 1 dB of mean level, with the branch the
   0.6.0 rule takes at -26 dBFS marked, so the cutover can be moved on measured rows.

    python3 tools/quality_sweep/analyze.py \
        --results /home/jay/sweep/run/state.json \
        --baseline /home/jay/sweep/legacy_baseline.json \
        --file-list tools/quality_sweep/file_list.json \
        --out-json /home/jay/sweep/analysis.json --out-md /home/jay/sweep/report.md
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import service_log  # noqa: E402
from strata import VAD_CUTOVER_DBFS, vad_branch  # noqa: E402

# A file is a regression when 0.6.0 returns more than this fraction fewer words than the
# baseline. These are the rows to read by hand; nothing is decided automatically.
REGRESSION_TOLERANCE = 0.05

# Level table resolution, dBFS. One row per degree from -34 to -12 covers the archive
# (min -30.4, max -12.2) with room either side.
LEVEL_TABLE_LO = -34
LEVEL_TABLE_HI = -12

# Fields the 0.6.0 service is expected to add. Absent fields are reported as absent
# rather than as zero, so a missing field is never read as a clean result.
ADDITIVE_FIELDS = (
    "processing_seconds",
    "words_per_second",
    "attempt_count",
    "mfa_applied",
    "anomaly_count",
    "anomaly_windows",
    "flagged_segments",
    "rescue_attempted",
    "rescue_selected",
)


def percentile(values, pct):
    """Linear-interpolated percentile of a numeric list; None when empty."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    k = (len(data) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def distribution(values, digits=3):
    """n, mean and the usual percentiles of a numeric list."""
    data = [v for v in values if v is not None]
    if not data:
        return {"n": 0}
    out = {
        "n": len(data),
        "mean": sum(data) / len(data),
        "min": min(data),
        "p5": percentile(data, 5),
        "p25": percentile(data, 25),
        "median": percentile(data, 50),
        "p75": percentile(data, 75),
        "p95": percentile(data, 95),
        "max": max(data),
    }
    return {k: (round(v, digits) if isinstance(v, float) else v) for k, v in out.items()}


def choose_wps_split(values, floor_pct=1.0, ceiling_pct=25.0):
    """Find a data-driven cut between a low-rate tail and the healthy body.

    The rule is the largest absolute gap between consecutive sorted values inside the
    lower quarter of the distribution, above the bottom percentile so a single outlier
    cannot define the split. When the tail is smooth (no gap wider than a tenth of the
    interquartile range) there is no natural split and the function says so, falling back
    to the 5th percentile so the report still has a line to draw.
    """
    data = sorted(v for v in values if v is not None)
    if len(data) < 4:
        return {"method": "insufficient_data", "threshold": None, "below": 0, "above": len(data)}
    lo = percentile(data, floor_pct)
    hi = percentile(data, ceiling_pct)
    iqr = (percentile(data, 75) or 0) - (percentile(data, 25) or 0)
    best = None
    for i in range(len(data) - 1):
        if data[i] < lo or data[i] > hi:
            continue
        gap = data[i + 1] - data[i]
        if best is None or gap > best[0]:
            best = (gap, data[i], data[i + 1])
    if best is None or best[0] < iqr / 10.0:
        threshold = percentile(data, 5)
        method = "no_natural_gap_p5_fallback"
    else:
        threshold = (best[1] + best[2]) / 2.0
        method = "largest_gap_in_lower_quartile"
    below = sum(1 for v in data if v < threshold)
    return {
        "method": method,
        "threshold": round(threshold, 4) if threshold is not None else None,
        "widest_gap": round(best[0], 4) if best else None,
        "iqr": round(iqr, 4),
        "below": below,
        "above": len(data) - below,
    }


def build_rows(state, baseline, file_list, service_jobs=None):
    """One flat row per attempted file, joining result, baseline, strata and the log."""
    strata = {entry["file"]: entry for entry in file_list.get("files", [])}
    for entry in file_list.get("supplementary", []):
        strata[entry["file"]] = entry

    rows = []
    for filename, record in sorted(state.get("files", {}).items()):
        entry = strata.get(filename, {})
        base = baseline.get(filename) or {}
        payload = record.get("payload") or {}
        derived = record.get("derived") or {}
        legacy_words = base.get("words")
        new_words = derived.get("words")
        delta = None
        delta_pct = None
        if legacy_words and new_words is not None:
            delta = new_words - legacy_words
            delta_pct = delta / legacy_words
        row = {
            "file": filename,
            "outcome": record.get("outcome"),
            "duration_s": record.get("duration_s") or entry.get("duration_s"),
            "mean_dbfs": entry.get("mean_dbfs"),
            "level_bucket": entry.get("level_bucket"),
            "duration_bucket": entry.get("duration_bucket"),
            "bitrate_bucket": entry.get("bitrate_bucket"),
            "sample_rate": entry.get("sample_rate"),
            "channels": entry.get("channels"),
            "multi_voice": entry.get("multi_voice"),
            "reasons": entry.get("reasons") or [],
            "legacy_era": entry.get("legacy_era") or "unknown",
            "legacy_words": legacy_words,
            "legacy_wps": base.get("wps"),
            "new_words": new_words,
            "new_wps": derived.get("wps"),
            "word_delta": delta,
            "word_delta_pct": round(delta_pct, 5) if delta_pct is not None else None,
            "wall_s": record.get("wall_s"),
            "vad_branch_expected": vad_branch(entry.get("mean_dbfs")),
            "timings": derived.get("timings") or {},
            "segments": (derived.get("timings") or {}).get("count"),
            "seg_mean_s": (derived.get("timings") or {}).get("seg_mean_s"),
            "span_total_s": (derived.get("timings") or {}).get("span_total_s"),
        }
        for field in ADDITIVE_FIELDS:
            row[field] = payload.get(field, None)
            row[field + "_present"] = field in payload
        attach_service_log(row, (service_jobs or {}).get(filename))
        rows.append(row)
    return rows


def attach_service_log(row, job):
    """Fold one job's harvested log record into its row.

    Three things come only from the log. `speech_seconds` is what the VAD actually kept,
    which turns into the second words-per-second figure and into the coverage ratio.
    `coverage` is the share of that speech the emitted segments claim to cover: audio the
    decoder emitted nothing for lands here as a shortfall, which is the omission the
    anomaly window check was blind to. `trimmed_words` is what the seam de-duplication
    removed, so a word delta can be split into what the decode produced and what the
    dedupe took away afterwards.
    """
    row["speech_seconds"] = None
    row["coverage"] = None
    row["new_wps_speech"] = None
    row["legacy_wps_speech"] = None
    row["trim_count"] = None
    row["trimmed_words"] = None
    row["trims"] = []
    row["words_restored"] = None
    row["word_delta_restored"] = None
    row["word_delta_pct_restored"] = None
    row["alignment_log"] = {}
    row["service_log_fields"] = {}
    row["vad_profile_taken"] = None
    row["audio_level_service_dbfs"] = None
    row["service_uncovered_fraction"] = None
    row["rescue_pass_logged"] = None
    if not job:
        return row

    row["service_log_fields"] = job.get("fields") or {}
    row["alignment_log"] = job.get("alignment") or {}
    row["trim_count"] = job.get("trim_count")
    row["trimmed_words"] = job.get("trimmed_words")
    row["trims"] = job.get("trims") or []
    row["vad_profile_taken"] = job.get("vad_profile")
    row["audio_level_service_dbfs"] = job.get("audio_level_dbfs")
    row["service_uncovered_fraction"] = job.get("service_uncovered_fraction")
    row["rescue_pass_logged"] = bool(job.get("rescue_pass"))
    primary = job.get("primary_pass") or {}
    row["windows_total"] = primary.get("windows_total")
    row["primary_words"] = primary.get("words")
    speech = job.get("speech_seconds")
    if speech:
        row["speech_seconds"] = speech
        if row["span_total_s"] is not None:
            row["coverage"] = round(row["span_total_s"] / speech, 4)
        if row["new_words"] is not None:
            row["new_wps_speech"] = round(row["new_words"] / speech, 4)
        if row["legacy_words"]:
            row["legacy_wps_speech"] = round(row["legacy_words"] / speech, 4)
    # The corrected build declines every trim whose occurrences merely abut, so its
    # expected word count is this run's plus whatever the seam rule removed. Both bases
    # are carried so a delta can be read as measured or as the corrected build would
    # produce it.
    if row["new_words"] is not None and row.get("trimmed_words") is not None:
        row["words_restored"] = row["new_words"] + row["trimmed_words"]
        if row["legacy_words"]:
            row["word_delta_restored"] = row["words_restored"] - row["legacy_words"]
            row["word_delta_pct_restored"] = round(
                row["word_delta_restored"] / row["legacy_words"], 5
            )
    for key in ("clamped", "clamped_timings", "span_ratio_lt_0_5", "span_ratio_p5",
                "span_ratio_median", "empty_fallbacks", "whisper_fallback_segments",
                "mfa_words", "mfa_segments", "agree250", "utterances"):
        if key in row["alignment_log"]:
            row[key] = row["alignment_log"][key]
    return row


def completed(rows):
    return [r for r in rows if r["outcome"] == "completed"]


def comparable(rows):
    """Completed rows that also have a legacy word count to compare against."""
    return [r for r in completed(rows) if r["legacy_words"] and r["new_words"] is not None]


def regressions(rows, tolerance=REGRESSION_TOLERANCE, key="word_delta_pct"):
    """Completed files that returned more than `tolerance` fewer words than the baseline.

    `key` selects the basis: `word_delta_pct` is what this run produced, and
    `word_delta_pct_restored` is what the corrected build would produce, since it declines
    the seam trims that removed those words.
    """
    out = [
        r for r in comparable(rows)
        if r.get(key) is not None and r[key] < -tolerance
    ]
    out.sort(key=lambda r: r[key])
    return out


# Coverage bands. A file whose segments cover less than this share of the VAD's speech
# seconds has audio the decoder emitted nothing for, which is the omission the anomaly
# window check could not see because it built its speech clock from those same segments.
COVERAGE_BANDS = (0.5, 0.7, 0.8, 0.9, 0.95)


def coverage_summary(rows):
    """Segment-span coverage of the VAD's speech seconds, over the files that have both."""
    values = [r["coverage"] for r in completed(rows) if r.get("coverage") is not None]
    below = {
        f"under_{band}": sum(1 for v in values if v < band) for band in COVERAGE_BANDS
    }
    worst = sorted(
        (r for r in completed(rows) if r.get("coverage") is not None),
        key=lambda r: r["coverage"],
    )[:10]
    return {
        "measured": len(values),
        "distribution": distribution(values, digits=4),
        "counts": below,
        "worst": [
            {
                "file": r["file"],
                "coverage": r["coverage"],
                "span_total_s": r["span_total_s"],
                "speech_seconds": r["speech_seconds"],
                "duration_s": r["duration_s"],
                "anomaly_count": r.get("anomaly_count"),
                "anomaly_windows": r.get("anomaly_windows"),
            }
            for r in worst
        ],
    }


def dedupe_summary(rows):
    """What the seam de-duplication removed, per file and in total.

    The trim runs after the decode, so a file's word delta against the baseline is the
    decode's contribution plus this. Separating them is the only way to tell a decode
    that produced fewer words from a decode whose words were deleted afterwards.
    """
    measured = [r for r in completed(rows) if r.get("trim_count") is not None]
    trimmed = [r for r in measured if (r.get("trim_count") or 0) > 0]
    return {
        "measured": len(measured),
        "files_trimmed": len(trimmed),
        "trims_total": sum(r["trim_count"] or 0 for r in measured),
        "words_removed_total": sum(r["trimmed_words"] or 0 for r in measured),
        "words_removed_per_file": distribution(
            [r["trimmed_words"] for r in measured], digits=1
        ),
        "files": [
            {
                "file": r["file"],
                "trim_count": r["trim_count"],
                "trimmed_words": r["trimmed_words"],
                "words": r["new_words"],
                "words_before_dedupe": (
                    (r["new_words"] or 0) + (r["trimmed_words"] or 0)
                    if r["new_words"] is not None else None
                ),
                "word_delta_pct": r["word_delta_pct"],
            }
            for r in sorted(trimmed, key=lambda r: -(r["trimmed_words"] or 0))
        ],
        # Every trimmed phrase, so each one can be judged by eye. A four-word minimum
        # still removes a liturgical response that straddles a seam, so whether four is
        # the right floor is a question about these phrases, not about the count.
        "phrases": [
            {
                "file": r["file"],
                "at_s": t.get("at_s"),
                "overlap_words": t.get("overlap_words"),
                "removed": t.get("removed"),
                "emptied_segment": t.get("emptied"),
            }
            for r in trimmed
            for t in (r.get("trims") or [])
        ],
    }


def rescue_detail(rows):
    """Every file the rescue pass was selected on, with the word delta it produced.

    The rescue re-decodes once with previous-text conditioning off and is kept only if it
    retains at least 99 percent of the primary's words, so a selected rescue should never
    cost more than one percent. The delta reported here is against the legacy baseline,
    which is what the reader can act on; `words` is the absolute count for tracing.
    """
    attempted = [r for r in rows if r.get("rescue_attempted_present") and r.get("rescue_attempted")]
    selected = [r for r in rows if r.get("rescue_selected_present") and r.get("rescue_selected")]
    return {
        "attempted": [r["file"] for r in attempted],
        "selected": [
            {
                "file": r["file"],
                "duration_s": r["duration_s"],
                "mean_dbfs": r["mean_dbfs"],
                "anomaly_count": r.get("anomaly_count"),
                "legacy_words": r["legacy_words"],
                "words": r["new_words"],
                "word_delta": r["word_delta"],
                "word_delta_pct": r["word_delta_pct"],
            }
            for r in sorted(selected, key=lambda r: (r["word_delta_pct"] is None,
                                                     r["word_delta_pct"] or 0))
        ],
    }


def rate(rows, field):
    """`{present, true, false, rate}` for a boolean additive field."""
    present = [r for r in rows if r.get(field + "_present")]
    truthy = [r for r in present if r.get(field)]
    return {
        "present": len(present),
        "true": len(truthy),
        "false": len(present) - len(truthy),
        "rate": round(len(truthy) / len(present), 4) if present else None,
    }


def counter_of(rows, field):
    """Value histogram for a scalar additive field, absent values excluded."""
    values = [r.get(field) for r in rows if r.get(field + "_present")]
    return dict(sorted(collections.Counter(
        v if isinstance(v, (int, str, bool)) or v is None else len(v) for v in values
    ).items(), key=lambda kv: (kv[0] is None, kv[0])))


def flag_summary(rows):
    """How many files carry flagged segments and how many segments in total."""
    present = [r for r in rows if r.get("flagged_segments_present")]
    counts = [len(r["flagged_segments"] or []) for r in present]
    return {
        "files_with_field": len(present),
        "files_flagged": sum(1 for c in counts if c),
        "segments_total": sum(counts),
        "per_file": distribution(counts) if counts else {"n": 0},
    }


def outcome_counts(rows):
    return dict(sorted(collections.Counter(r["outcome"] for r in rows).items()))


def timing_health(rows):
    """The transcript-versus-timings consistency checks, over completed rows."""
    done = completed(rows)
    with_flag = [r for r in done if r["timings"].get("text_matches") is not None]
    return {
        "checked": len(with_flag),
        "text_matches_timings": sum(1 for r in with_flag if r["timings"]["text_matches"]),
        "non_monotonic_files": sum(1 for r in done if (r["timings"].get("non_monotonic") or 0) > 0),
        "overlapping_files": sum(1 for r in done if (r["timings"].get("overlaps") or 0) > 0),
        "segments": distribution([r["timings"].get("count") for r in done]),
    }


def summarise_population(rows):
    """Every headline number for one population of rows."""
    comp = comparable(rows)
    return {
        "files": len(rows),
        "outcomes": outcome_counts(rows),
        "completed": len(completed(rows)),
        "comparable": len(comp),
        "word_delta": distribution([r["word_delta"] for r in comp], digits=1),
        "word_delta_pct": distribution([r["word_delta_pct"] for r in comp], digits=5),
        "legacy_wps": distribution([r["legacy_wps"] for r in comp]),
        "new_wps": distribution([r["new_wps"] for r in comp]),
        "regressions": {
            "tolerance": REGRESSION_TOLERANCE,
            "count": len(regressions(rows)),
            "files": [r["file"] for r in regressions(rows)],
        },
        "word_delta_restored": distribution(
            [r.get("word_delta_restored") for r in comp], digits=1
        ),
        "word_delta_pct_restored": distribution(
            [r.get("word_delta_pct_restored") for r in comp], digits=5
        ),
        "regressions_restored": {
            "tolerance": REGRESSION_TOLERANCE,
            "count": len(regressions(rows, key="word_delta_pct_restored")),
            "files": [
                r["file"] for r in regressions(rows, key="word_delta_pct_restored")
            ],
        },
        "rescue_attempted": rate(rows, "rescue_attempted"),
        "rescue_selected": rate(rows, "rescue_selected"),
        "rescue_detail": rescue_detail(rows),
        "segments": distribution([r.get("segments") for r in completed(rows)], digits=1),
        "seg_mean_s": distribution([r.get("seg_mean_s") for r in completed(rows)]),
        "coverage": coverage_summary(rows),
        "new_wps_speech": distribution([r.get("new_wps_speech") for r in completed(rows)]),
        "legacy_wps_speech": distribution([r.get("legacy_wps_speech") for r in comp]),
        "dedupe": dedupe_summary(rows),
        "mfa_applied": rate(rows, "mfa_applied"),
        "attempt_count": counter_of(rows, "attempt_count"),
        "anomaly_count": counter_of(rows, "anomaly_count"),
        "anomaly_windows": counter_of(rows, "anomaly_windows"),
        "flagged_segments": flag_summary(rows),
        "processing_seconds": distribution([
            r["processing_seconds"] for r in rows if r.get("processing_seconds_present")
        ], digits=1),
        "wall_s": distribution([r["wall_s"] for r in rows], digits=1),
        "timings": timing_health(rows),
        "alignment": alignment_health(rows),
        "service_uncovered_fraction": distribution(
            [r.get("service_uncovered_fraction") for r in completed(rows)], digits=5
        ),
    }


def alignment_health(rows):
    """The aligner's own counters, which 0.6.0 emits on the alignment line."""
    done = completed(rows)

    def values(key):
        return [r[key] for r in done if r.get(key) is not None]

    return {
        "agree250": distribution(values("agree250"), digits=4),
        "span_ratio_p5": distribution(values("span_ratio_p5"), digits=4),
        "span_ratio_median": distribution(values("span_ratio_median"), digits=4),
        "span_ratio_lt_0_5_total": sum(values("span_ratio_lt_0_5")),
        "span_ratio_lt_0_5_files": sum(1 for v in values("span_ratio_lt_0_5") if v),
        "clamped_total": sum(values("clamped")),
        "clamped_files": sum(1 for v in values("clamped") if v),
        "empty_fallbacks_total": sum(values("empty_fallbacks")),
        "empty_fallbacks_files": sum(1 for v in values("empty_fallbacks") if v),
        "measured": len(values("agree250")),
    }


def level_table(rows, lo=LEVEL_TABLE_LO, hi=LEVEL_TABLE_HI):
    """One row per 1 dB of mean level, for recalibrating the VAD cutover."""
    buckets = collections.defaultdict(list)
    for row in rows:
        level = row.get("mean_dbfs")
        if level is None:
            buckets["unknown"].append(row)
            continue
        edge = max(lo, min(hi - 1, int(level // 1)))
        buckets[edge].append(row)

    table = []
    for edge in sorted(k for k in buckets if k != "unknown"):
        members = buckets[edge]
        comp = comparable(members)
        table.append({
            "dbfs_lo": edge,
            "dbfs_hi": edge + 1,
            "vad_branch_expected": vad_branch(edge + 0.5),
            "at_or_above_cutover": (edge + 0.5) >= VAD_CUTOVER_DBFS,
            "files": len(members),
            "completed": len(completed(members)),
            "comparable": len(comp),
            "legacy_wps_median": percentile([r["legacy_wps"] for r in comp], 50),
            "new_wps_median": percentile([r["new_wps"] for r in comp], 50),
            "word_delta_pct_median": percentile([r["word_delta_pct"] for r in comp], 50),
            "words_total": sum(r["new_words"] or 0 for r in completed(members)),
            "segments_median": percentile(
                [r.get("segments") for r in completed(members)], 50
            ),
            "coverage_median": percentile(
                [r.get("coverage") for r in completed(members)], 50
            ),
            "vad_profile_taken": sorted({
                r["vad_profile_taken"] for r in members if r.get("vad_profile_taken")
            }) or None,
            "seg_mean_s_median": percentile(
                [r.get("seg_mean_s") for r in completed(members)], 50
            ),
            "regressions": len(regressions(members)),
            "anomaly_count_mean": (
                round(sum(r["anomaly_count"] or 0 for r in members
                          if r.get("anomaly_count_present"))
                      / max(1, sum(1 for r in members if r.get("anomaly_count_present"))), 3)
                if any(r.get("anomaly_count_present") for r in members) else None
            ),
            "files_flagged": flag_summary(members)["files_flagged"],
        })
    if buckets.get("unknown"):
        table.append({"dbfs_lo": None, "dbfs_hi": None, "files": len(buckets["unknown"])})
    return table


def cutover_view(rows, edge=VAD_CUTOVER_DBFS, band=2.0):
    """Aggregates either side of the VAD cutover, and in the band straddling it.

    The question the sweep exists to answer is whether -26 dBFS is in the right place.
    The two arms say whether the profiles behave differently at all; the two halves of
    the straddling band say whether the edge is drawn where the behaviour changes. A
    profile that shreds quiet audio shows up as a collapse in `seg_mean_s_median` and a
    jump in `segments_median`, which is exactly what the 0.5 threshold did on the
    retreat file before the level-selected profile existed.
    """
    def summarise(members, label):
        comp = comparable(members)
        return {
            "label": label,
            "files": len(members),
            "completed": len(completed(members)),
            "word_delta_pct_median": percentile([r["word_delta_pct"] for r in comp], 50),
            "new_wps_median": percentile([r["new_wps"] for r in comp], 50),
            "segments_median": percentile([r.get("segments") for r in completed(members)], 50),
            "seg_mean_s_median": percentile([r.get("seg_mean_s") for r in completed(members)], 50),
            "anomaly_count_median": percentile(
                [r["anomaly_count"] for r in members if r.get("anomaly_count_present")], 50
            ),
            "rescue_attempted_rate": rate(members, "rescue_attempted")["rate"],
            "regressions": len(regressions(members)),
        }

    known = [r for r in rows if r.get("mean_dbfs") is not None]
    below = [r for r in known if r["mean_dbfs"] < edge]
    at_or_above = [r for r in known if r["mean_dbfs"] >= edge]
    band_below = [r for r in below if r["mean_dbfs"] >= edge - band]
    band_above = [r for r in at_or_above if r["mean_dbfs"] < edge + band]
    return {
        "edge_dbfs": edge,
        "arms": [
            summarise(below, f"below {edge} dBFS (quiet profile)"),
            summarise(at_or_above, f"at or above {edge} dBFS (loud profile)"),
        ],
        "straddling_band": [
            summarise(band_below, f"{edge - band} to {edge} dBFS (quiet profile)"),
            summarise(band_above, f"{edge} to {edge + band} dBFS (loud profile)"),
        ],
    }


def analyse(state, baseline, file_list, service_jobs=None):
    rows = build_rows(state, baseline, file_list, service_jobs)
    comp = comparable(rows)
    split = choose_wps_split([r["legacy_wps"] for r in comp])

    populations = collections.OrderedDict()
    populations["overall"] = summarise_population(rows)

    by_era = collections.defaultdict(list)
    for row in rows:
        by_era[row["legacy_era"]].append(row)
    for era in sorted(by_era):
        populations["era_" + era] = summarise_population(by_era[era])

    if split.get("threshold") is not None:
        low = [r for r in rows if (r["legacy_wps"] or 0) and r["legacy_wps"] < split["threshold"]]
        high = [r for r in rows if r["legacy_wps"] and r["legacy_wps"] >= split["threshold"]]
        populations["legacy_low_rate"] = summarise_population(low)
        populations["legacy_healthy"] = summarise_population(high)

    known_collapse = [
        r for r in rows
        if "rerun_2026_09_07" in r["reasons"] or "known_collapse_no_baseline" in r["reasons"]
    ]
    if known_collapse:
        populations["known_collapse_set"] = summarise_population(known_collapse)

    multi = [r for r in rows if r.get("multi_voice")]
    if multi:
        populations["multi_voice"] = summarise_population(multi)

    return {
        "files_attempted": len(rows),
        "legacy_wps_split": split,
        "populations": populations,
        "level_table": level_table(rows),
        "cutover": cutover_view(rows),
        "duration_buckets": {
            bucket: summarise_population([r for r in rows if r["duration_bucket"] == bucket])
            for bucket in sorted({r["duration_bucket"] for r in rows if r["duration_bucket"]})
        },
        "rows": rows,
    }


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers, body):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(analysis, title="TranscriptionAPI 0.6.0 validation sweep"):
    out = [f"# {title}", ""]
    split = analysis["legacy_wps_split"]
    out.append(
        f"{analysis['files_attempted']} files attempted. Legacy words-per-second split: "
        f"method `{split['method']}`, threshold {_fmt(split['threshold'])}, "
        f"{split['below']} below and {split['above']} at or above."
    )
    out.append("")
    out.append(
        "Wall-clock timings in this run share GPU 0 with production and are not "
        "benchmark quality."
    )
    out.append("")

    out.append("## Populations")
    out.append("")
    headers = [
        "population", "files", "completed", "comparable", "word delta median",
        "delta pct median", "delta pct restored", "legacy wps median", "new wps median",
        "new wps over speech", "regressions", "regr restored",
    ]
    body = []
    for name, pop in analysis["populations"].items():
        body.append([
            name,
            str(pop["files"]),
            str(pop["completed"]),
            str(pop["comparable"]),
            _fmt(pop["word_delta"].get("median"), 1),
            _fmt(pop["word_delta_pct"].get("median"), 4),
            _fmt(pop["word_delta_pct_restored"].get("median"), 4),
            _fmt(pop["legacy_wps"].get("median")),
            _fmt(pop["new_wps"].get("median")),
            _fmt(pop["new_wps_speech"].get("median")),
            str(pop["regressions"]["count"]),
            str(pop["regressions_restored"]["count"]),
        ])
    out.append(_table(headers, body))
    out.append("")

    out.append("## Service behaviour")
    out.append("")
    headers = [
        "population", "outcomes", "rescue attempted", "rescue selected", "mfa applied",
        "attempts", "anomaly counts", "files flagged",
    ]
    body = []
    for name, pop in analysis["populations"].items():
        body.append([
            name,
            ", ".join(f"{k}={v}" for k, v in pop["outcomes"].items()) or "-",
            _fmt(pop["rescue_attempted"]["rate"]),
            _fmt(pop["rescue_selected"]["rate"]),
            _fmt(pop["mfa_applied"]["rate"]),
            ", ".join(f"{k}:{v}" for k, v in pop["attempt_count"].items()) or "-",
            ", ".join(f"{k}:{v}" for k, v in pop["anomaly_count"].items()) or "-",
            str(pop["flagged_segments"]["files_flagged"]),
        ])
    out.append(_table(headers, body))
    out.append("")

    out.append("## Level against the VAD cutover")
    out.append("")
    out.append(
        f"The 0.6.0 rule takes VAD threshold 0.5 at or above {VAD_CUTOVER_DBFS} dBFS and "
        "0.35 below it. Rows below the line are the 0.35 branch as shipped."
    )
    out.append("")
    headers = [
        "dBFS", "profile", "files", "done", "words", "legacy wps", "new wps",
        "delta pct", "segments", "seg mean s", "coverage", "anomaly mean", "regr", "flagged",
    ]
    body = []
    for entry in analysis["level_table"]:
        if entry.get("dbfs_lo") is None:
            body.append(["unknown", "-", str(entry["files"])] + ["-"] * 11)
            continue
        body.append([
            f"{entry['dbfs_lo']} to {entry['dbfs_hi']}",
            "quiet" if not entry["at_or_above_cutover"] else "loud",
            str(entry["files"]),
            str(entry["completed"]),
            str(entry["words_total"]),
            _fmt(entry["legacy_wps_median"]),
            _fmt(entry["new_wps_median"]),
            _fmt(entry["word_delta_pct_median"], 4),
            _fmt(entry["segments_median"], 0),
            _fmt(entry["seg_mean_s_median"], 2),
            _fmt(entry["coverage_median"], 3),
            _fmt(entry["anomaly_count_mean"]),
            str(entry["regressions"]),
            str(entry["files_flagged"]),
        ])
    out.append(_table(headers, body))
    out.append("")

    out.append("### Either side of the cutover")
    out.append("")
    headers = [
        "arm", "files", "done", "delta pct", "new wps", "segments", "seg mean s",
        "anomaly", "rescue rate", "regr",
    ]
    body = []
    for group in ("arms", "straddling_band"):
        for arm in analysis["cutover"][group]:
            body.append([
                arm["label"],
                str(arm["files"]),
                str(arm["completed"]),
                _fmt(arm["word_delta_pct_median"], 4),
                _fmt(arm["new_wps_median"]),
                _fmt(arm["segments_median"], 0),
                _fmt(arm["seg_mean_s_median"], 2),
                _fmt(arm["anomaly_count_median"], 1),
                _fmt(arm["rescue_attempted_rate"]),
                str(arm["regressions"]),
            ])
    out.append(_table(headers, body))
    out.append("")

    overall = analysis["populations"]["overall"]
    cov = overall["coverage"]
    out.append("## Segment coverage of the VAD speech seconds")
    out.append("")
    if cov["measured"]:
        counts = ", ".join(f"{k.replace('under_', 'under ')}: {v}" for k, v in cov["counts"].items())
        out.append(
            f"Measured on {cov['measured']} files. Coverage median "
            f"{_fmt(cov['distribution'].get('median'), 4)}, p5 "
            f"{_fmt(cov['distribution'].get('p5'), 4)}, min "
            f"{_fmt(cov['distribution'].get('min'), 4)}. Files below each band: {counts}."
        )
        out.append("")
        out.append(
            "Coverage is the total span of the emitted segments over the seconds VAD kept. "
            "A shortfall is audio the decoder emitted nothing for, which is exactly what "
            "the segment-derived speech clock could not see."
        )
        out.append("")
        headers = ["file", "coverage", "span total s", "speech s", "dur s",
                   "anomaly", "anomaly windows"]
        body = [
            [r["file"], _fmt(r["coverage"], 4), _fmt(r["span_total_s"], 1),
             _fmt(r["speech_seconds"], 1), _fmt(r["duration_s"], 0),
             _fmt(r["anomaly_count"], 0), _fmt(r["anomaly_windows"], 0)]
            for r in cov["worst"]
        ]
        out.append(_table(headers, body))
    else:
        out.append("Not measured: no service log was supplied.")
    out.append("")

    ded = overall["dedupe"]
    out.append("## Seam de-duplication")
    out.append("")
    if ded["measured"]:
        out.append(
            f"{ded['files_trimmed']} of {ded['measured']} files had a trim. "
            f"{ded['trims_total']} trims removed {ded['words_removed_total']} words in total."
        )
        out.append("")
        if ded["files"]:
            headers = ["file", "trims", "words removed", "words after",
                       "words before", "delta pct vs legacy"]
            body = [
                [r["file"], str(r["trim_count"]), str(r["trimmed_words"]),
                 _fmt(r["words"], 0), _fmt(r["words_before_dedupe"], 0),
                 _fmt(r["word_delta_pct"], 4)]
                for r in ded["files"]
            ]
            out.append(_table(headers, body))
            out.append("")
        if ded["phrases"]:
            out.append(
                "Every trimmed phrase, for judging by eye. A four-word minimum still "
                "removes a liturgical response that straddles a seam, so whether four is "
                "the right floor is a question about these phrases and not about the count."
            )
            out.append("")
            headers = ["file", "at s", "words", "phrase removed", "emptied segment"]
            body = [
                [p["file"], _fmt(p["at_s"], 2), _fmt(p["overlap_words"], 0),
                 (p["removed"] or "").replace("|", "/"),
                 "yes" if p["emptied_segment"] else "no"]
                for p in ded["phrases"]
            ]
            out.append(_table(headers, body))
    else:
        out.append("Not measured: no service log was supplied.")
    out.append("")

    align = overall["alignment"]
    out.append("## Alignment counters")
    out.append("")
    if align["measured"]:
        out.append(
            f"Measured on {align['measured']} files. agree250 median "
            f"{_fmt(align['agree250'].get('median'), 4)}, p5 "
            f"{_fmt(align['agree250'].get('p5'), 4)}. Span ratio median of medians "
            f"{_fmt(align['span_ratio_median'].get('median'), 4)}, p5 of p5s "
            f"{_fmt(align['span_ratio_p5'].get('p5'), 4)}. "
            f"{align['span_ratio_lt_0_5_total']} segments fell below the 0.5 span ratio "
            f"across {align['span_ratio_lt_0_5_files']} files; "
            f"{align['clamped_total']} timings came from the monotonic clamp across "
            f"{align['clamped_files']} files; {align['empty_fallbacks_total']} empty "
            f"fallbacks across {align['empty_fallbacks_files']} files."
        )
    else:
        out.append("Not measured: no service log was supplied.")
    out.append("")

    detail = analysis["populations"]["overall"]["rescue_detail"]
    out.append("## Rescue pass")
    out.append("")
    out.append(
        f"Attempted on {len(detail['attempted'])} files, selected on "
        f"{len(detail['selected'])}."
    )
    out.append("")
    if detail["selected"]:
        headers = ["file", "dur s", "dBFS", "anomaly", "legacy words", "words",
                   "delta", "delta pct"]
        body = [
            [
                row["file"],
                _fmt(row["duration_s"], 0),
                _fmt(row["mean_dbfs"], 1),
                _fmt(row["anomaly_count"], 0),
                _fmt(row["legacy_words"], 0),
                _fmt(row["words"], 0),
                _fmt(row["word_delta"], 0),
                _fmt(row["word_delta_pct"], 4),
            ]
            for row in detail["selected"]
        ]
        out.append(_table(headers, body))
        out.append("")

    regressed = analysis["populations"]["overall"]["regressions"]["files"]
    out.append("## Regressions to read by hand")
    out.append("")
    if regressed:
        out.append(
            f"{len(regressed)} files returned more than "
            f"{int(REGRESSION_TOLERANCE * 100)} percent fewer words than the baseline."
        )
    else:
        out.append("No file returned more than 5 percent fewer words than the baseline.")
    out.append("")

    out.append("## Per file")
    out.append("")
    headers = [
        "file", "outcome", "dur s", "dBFS", "era", "legacy words", "new words",
        "delta pct", "legacy wps", "new wps", "wps speech", "coverage", "trims",
        "segs", "seg mean s", "wall s", "anomaly", "resc a/s", "flags", "mfa", "attempts",
    ]
    body = []
    for row in sorted(analysis["rows"], key=lambda r: (r["word_delta_pct"] is None,
                                                       r["word_delta_pct"] or 0)):
        flags = row.get("flagged_segments")
        body.append([
            row["file"],
            str(row["outcome"]),
            _fmt(row["duration_s"], 0),
            _fmt(row["mean_dbfs"], 1),
            row["legacy_era"],
            _fmt(row["legacy_words"], 0),
            _fmt(row["new_words"], 0),
            _fmt(row["word_delta_pct"], 4),
            _fmt(row["legacy_wps"]),
            _fmt(row["new_wps"]),
            _fmt(row.get("new_wps_speech")),
            _fmt(row.get("coverage"), 3),
            _fmt(row.get("trimmed_words"), 0),
            _fmt(row.get("segments"), 0),
            _fmt(row.get("seg_mean_s"), 2),
            _fmt(row["wall_s"], 1),
            _fmt(row["anomaly_count"], 0),
            f"{row.get('rescue_attempted')}/{row.get('rescue_selected')}",
            str(len(flags)) if isinstance(flags, list) else "-",
            _fmt(row["mfa_applied"], 0),
            _fmt(row["attempt_count"], 0),
        ])
    out.append(_table(headers, body))
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="the runner's state.json")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--title", default="TranscriptionAPI 0.6.0 validation sweep")
    parser.add_argument(
        "--service-log",
        help="the sweep container's captured log, for speech seconds, the "
             "de-duplication trims and the alignment counters",
    )
    args = parser.parse_args(argv)

    with open(args.results) as handle:
        state = json.load(handle)
    with open(args.baseline) as handle:
        baseline = json.load(handle)["baseline"]
    with open(args.file_list) as handle:
        file_list = json.load(handle)

    service_jobs = None
    if args.service_log and os.path.exists(args.service_log):
        with open(args.service_log, errors="replace") as handle:
            service_jobs = service_log.by_file(service_log.parse(handle))
        print(f"harvested {len(service_jobs)} jobs from {args.service_log}")

    analysis = analyse(state, baseline, file_list, service_jobs)
    with open(args.out_json, "w") as handle:
        json.dump(analysis, handle, indent=1, sort_keys=False)
        handle.write("\n")
    with open(args.out_md, "w") as handle:
        handle.write(render_markdown(analysis, args.title))
        handle.write("\n")
    print(f"wrote {args.out_json} and {args.out_md}")
    print(f"{analysis['files_attempted']} files, "
          f"{analysis['populations']['overall']['completed']} completed, "
          f"{analysis['populations']['overall']['regressions']['count']} regressions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
