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


def build_rows(state, baseline, file_list):
    """One flat row per attempted file, joining result, baseline and strata."""
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
        }
        for field in ADDITIVE_FIELDS:
            row[field] = payload.get(field, None)
            row[field + "_present"] = field in payload
        rows.append(row)
    return rows


def completed(rows):
    return [r for r in rows if r["outcome"] == "completed"]


def comparable(rows):
    """Completed rows that also have a legacy word count to compare against."""
    return [r for r in completed(rows) if r["legacy_words"] and r["new_words"] is not None]


def regressions(rows, tolerance=REGRESSION_TOLERANCE):
    """Completed files that returned more than `tolerance` fewer words than the baseline."""
    out = [
        r for r in comparable(rows)
        if r["word_delta_pct"] is not None and r["word_delta_pct"] < -tolerance
    ]
    out.sort(key=lambda r: r["word_delta_pct"])
    return out


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
        "rescue_attempted": rate(rows, "rescue_attempted"),
        "rescue_selected": rate(rows, "rescue_selected"),
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


def analyse(state, baseline, file_list):
    rows = build_rows(state, baseline, file_list)
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
        "delta pct median", "legacy wps median", "new wps median", "regressions",
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
            _fmt(pop["legacy_wps"].get("median")),
            _fmt(pop["new_wps"].get("median")),
            str(pop["regressions"]["count"]),
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
        "dBFS", "branch", "files", "completed", "legacy wps", "new wps",
        "delta pct", "regressions", "anomaly mean", "flagged",
    ]
    body = []
    for entry in analysis["level_table"]:
        if entry.get("dbfs_lo") is None:
            body.append(["unknown", "-", str(entry["files"]), "-", "-", "-", "-", "-", "-", "-"])
            continue
        body.append([
            f"{entry['dbfs_lo']} to {entry['dbfs_hi']}",
            _fmt(entry["vad_branch_expected"], 2),
            str(entry["files"]),
            str(entry["completed"]),
            _fmt(entry["legacy_wps_median"]),
            _fmt(entry["new_wps_median"]),
            _fmt(entry["word_delta_pct_median"], 4),
            str(entry["regressions"]),
            _fmt(entry["anomaly_count_mean"]),
            str(entry["files_flagged"]),
        ])
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
        "delta pct", "legacy wps", "new wps", "wall s", "anomaly", "flags", "mfa", "attempts",
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
            _fmt(row["wall_s"], 1),
            _fmt(row["anomaly_count"], 0),
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
    args = parser.parse_args(argv)

    with open(args.results) as handle:
        state = json.load(handle)
    with open(args.baseline) as handle:
        baseline = json.load(handle)["baseline"]
    with open(args.file_list) as handle:
        file_list = json.load(handle)

    analysis = analyse(state, baseline, file_list)
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
