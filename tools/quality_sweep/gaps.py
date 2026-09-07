"""Find the speech the decoder emitted nothing for, and how it is distributed in time.

0.6.0 tolerates uncovered speech up to 10 percent of detected speech before it calls a
decode incomplete. That number was calibrated on six files against a healthy worst case of
7.1 percent, so the first thing a hundred files can settle is whether 10 percent is the
right line, and the second is whether the total is even the right statistic.

The reason to doubt the total is structural. Breathing, pauses and the ragged edges of
every segment leave uncovered speech scattered in fractions of a second all through a
healthy file. A real omission is one contiguous stretch the decoder skipped. Two files
can share an uncovered total of 8 percent where one is a thousand breaths and the other
is a missing ninety seconds, and only the second is a defect. So this module reports both
the total and the largest single contiguous uncovered stretch, and the report says whether
the largest-gap statistic separates the two populations more cleanly than the total does.

Detected speech here is measured independently of the service, by `silence.sh` running
ffmpeg's silencedetect at a threshold set from each file's own mean level. Using the
service's own VAD would mean trusting the same clock the decode used, and the whole point
is to check it from outside.

    python3 tools/quality_sweep/gaps.py --silence /home/jay/sweep/silence.jsonl \
        --timings /home/jay/sweep/run/timings --results /home/jay/sweep/run/state.json \
        --out-json /home/jay/sweep/gaps.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze import distribution, percentile  # noqa: E402

# Gap sizes worth counting separately. A second of uncovered speech is a breath; half a
# minute is a paragraph nobody transcribed.
GAP_BANDS = (2.0, 5.0, 15.0, 30.0, 60.0)


def merge(intervals):
    """Sort and coalesce overlapping or touching intervals."""
    clean = sorted(
        (float(a), float(b)) for a, b in intervals if a is not None and b is not None and b > a
    )
    out = []
    for start, end in clean:
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def complement(intervals, lo, hi):
    """Everything in [lo, hi] that `intervals` does not cover."""
    out = []
    cursor = lo
    for start, end in merge(intervals):
        if end <= lo or start >= hi:
            continue
        start = max(start, lo)
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, min(end, hi))
    if cursor < hi:
        out.append((cursor, hi))
    return out


def subtract(intervals, removals):
    """`intervals` minus `removals`, both as interval lists."""
    out = []
    removals = merge(removals)
    for start, end in merge(intervals):
        cursor = start
        for r_start, r_end in removals:
            if r_end <= cursor:
                continue
            if r_start >= end:
                break
            if r_start > cursor:
                out.append((cursor, min(r_start, end)))
            cursor = max(cursor, r_end)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return [(a, b) for a, b in out if b > a]


def total(intervals):
    return round(sum(b - a for a, b in intervals), 3)


def analyse_file(silences, spans, duration):
    """Uncovered-speech figures for one file.

    `silences` are the detected silent stretches, `spans` the segments the decoder
    emitted, `duration` the file length. Speech is the complement of silence; uncovered
    speech is that minus the spans.
    """
    if not duration:
        return None
    speech = complement(silences, 0.0, float(duration))
    speech_s = total(speech)
    uncovered = subtract(speech, spans)
    lengths = sorted((b - a for a, b in uncovered), reverse=True)
    uncovered_s = round(sum(lengths), 3)
    return {
        "duration_s": round(float(duration), 3),
        "speech_s": speech_s,
        "covered_s": round(speech_s - uncovered_s, 3),
        "uncovered_s": uncovered_s,
        "uncovered_fraction": round(uncovered_s / speech_s, 5) if speech_s else None,
        "largest_gap_s": round(lengths[0], 3) if lengths else 0.0,
        "largest_gap_fraction": (
            round(lengths[0] / speech_s, 5) if lengths and speech_s else None
        ),
        "gap_count": len(lengths),
        "gaps_over": {f"over_{band}s": sum(1 for v in lengths if v > band) for band in GAP_BANDS},
        "top_gaps": [
            {"start": round(a, 2), "end": round(b, 2), "length_s": round(b - a, 2)}
            for a, b in sorted(uncovered, key=lambda p: p[1] - p[0], reverse=True)[:5]
        ],
    }


def load_spans(timings_dir, filename):
    """Segment spans for one file, from the timings the runner saved."""
    stem = os.path.splitext(filename)[0]
    path = os.path.join(timings_dir, stem + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        doc = json.load(handle)
    spans = []
    for entry in doc.get("timings") or []:
        start, end = entry.get("start"), entry.get("end")
        if start is not None and end is not None:
            spans.append((float(start), float(end)))
    return spans


def build(silence_path, timings_dir, state):
    """`{filename: figures}` for every file that has both a silence scan and timings."""
    out = {}
    with open(silence_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            filename = doc["file"]
            record = (state.get("files") or {}).get(filename)
            if not record or record.get("outcome") != "completed":
                continue
            spans = load_spans(timings_dir, filename)
            if spans is None:
                continue
            silences = [
                (a, b) for a, b in doc.get("silences") or [] if a is not None and b is not None
            ]
            figures = analyse_file(silences, spans, record.get("duration_s"))
            if figures:
                figures["threshold_db"] = doc.get("threshold_db")
                out[filename] = figures
    return out


def summarise(per_file, tolerance=0.10):
    """Corpus-level distributions, and what tolerance the data would support."""
    fractions = [v["uncovered_fraction"] for v in per_file.values() if v["uncovered_fraction"] is not None]
    largest = [v["largest_gap_s"] for v in per_file.values()]
    largest_fraction = [
        v["largest_gap_fraction"] for v in per_file.values() if v["largest_gap_fraction"] is not None
    ]
    over_tolerance = sorted(
        (
            {"file": name, **{k: v[k] for k in ("uncovered_fraction", "largest_gap_s", "speech_s")}}
            for name, v in per_file.items()
            if v["uncovered_fraction"] is not None and v["uncovered_fraction"] > tolerance
        ),
        key=lambda r: -r["uncovered_fraction"],
    )
    worst_gap = sorted(
        (
            {"file": name, **{k: v[k] for k in ("largest_gap_s", "uncovered_fraction", "speech_s")},
             "top_gaps": v["top_gaps"]}
            for name, v in per_file.items()
        ),
        key=lambda r: -(r["largest_gap_s"] or 0),
    )[:15]
    return {
        "files": len(per_file),
        "tolerance_under_test": tolerance,
        "uncovered_fraction": distribution(fractions, digits=5),
        "largest_gap_s": distribution(largest, digits=2),
        "largest_gap_fraction": distribution(largest_fraction, digits=5),
        "over_tolerance": {"count": len(over_tolerance), "files": over_tolerance},
        "recommended_tolerance": recommend(fractions),
        "worst_largest_gap": worst_gap,
        "gap_band_totals": {
            f"over_{band}s": sum(1 for v in per_file.values() if v["gaps_over"][f"over_{band}s"])
            for band in GAP_BANDS
        },
    }


def recommend(fractions, headroom=1.5, floor=0.02):
    """A tolerance the measured corpus supports: headroom over the healthy body.

    The rule is the 95th percentile of the observed uncovered fractions times `headroom`,
    never below `floor`. It is a starting point stated in the open, not a derivation: it
    assumes most of the corpus is healthy, which is exactly what the largest-gap table is
    there to check.
    """
    if not fractions:
        return None
    p95 = percentile(fractions, 95)
    return {
        "p95": round(p95, 5),
        "headroom": headroom,
        "value": round(max(p95 * headroom, floor), 4),
        "rule": "95th percentile of uncovered fraction times headroom, floored",
    }


def render_markdown(summary, per_file):
    out = ["# Uncovered speech", ""]
    dist = summary["uncovered_fraction"]
    gap = summary["largest_gap_s"]
    out.append(
        f"{summary['files']} files. Uncovered speech as a fraction of detected speech: "
        f"median {dist.get('median')}, p75 {dist.get('p75')}, p95 {dist.get('p95')}, "
        f"max {dist.get('max')}. Largest contiguous uncovered stretch: median "
        f"{gap.get('median')} s, p95 {gap.get('p95')} s, max {gap.get('max')} s."
    )
    out.append("")
    rec = summary["recommended_tolerance"]
    out.append(
        f"{summary['over_tolerance']['count']} files exceed the shipped tolerance of "
        f"{summary['tolerance_under_test']}. Recommended tolerance from this corpus: "
        f"{rec['value']} ({rec['rule']}, p95 {rec['p95']})."
    )
    out.append("")
    out.append("Files with the largest single contiguous uncovered stretch:")
    out.append("")
    out.append("| file | largest gap s | uncovered fraction | speech s | top gaps |")
    out.append("|---|---|---|---|---|")
    for row in summary["worst_largest_gap"]:
        top = "; ".join(f"{g['start']}-{g['end']} ({g['length_s']}s)" for g in row["top_gaps"][:3])
        out.append(
            f"| {row['file']} | {row['largest_gap_s']} | {row['uncovered_fraction']} | "
            f"{row['speech_s']} | {top} |"
        )
    out.append("")
    out.append("Per file:")
    out.append("")
    out.append("| file | speech s | uncovered s | fraction | largest gap s | gaps >5s | >15s | >30s |")
    out.append("|---|---|---|---|---|---|---|---|")
    for name in sorted(per_file, key=lambda n: -(per_file[n]["uncovered_fraction"] or 0)):
        v = per_file[name]
        out.append(
            f"| {name} | {v['speech_s']} | {v['uncovered_s']} | {v['uncovered_fraction']} | "
            f"{v['largest_gap_s']} | {v['gaps_over']['over_5.0s']} | "
            f"{v['gaps_over']['over_15.0s']} | {v['gaps_over']['over_30.0s']} |"
        )
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silence", required=True)
    parser.add_argument("--timings", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    args = parser.parse_args(argv)

    with open(args.results) as handle:
        state = json.load(handle)
    per_file = build(args.silence, args.timings, state)
    summary = summarise(per_file, args.tolerance)

    if args.out_json:
        with open(args.out_json, "w") as handle:
            json.dump({"summary": summary, "per_file": per_file}, handle, indent=1)
            handle.write("\n")
    if args.out_md:
        with open(args.out_md, "w") as handle:
            handle.write(render_markdown(summary, per_file))
            handle.write("\n")
    print(f"{summary['files']} files, {summary['over_tolerance']['count']} over tolerance, "
          f"recommended {summary['recommended_tolerance']['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
