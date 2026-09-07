"""Decide, for every seam de-duplication trim, whether the two occurrences overlap in time.

The question this answers is whether a better rule than "four or more words" exists.

A genuine decoder seam artifact is the same audio decoded twice: the tail of segment N and
the head of segment N+1 describe the same seconds, so their time ranges overlap. Genuine
repetition is the speaker saying something twice: the two occurrences are sequential and
disjoint. If the trims separate on that, overlap is the right rule and word count is not.

**What is measured, and how precisely.** The service logs each trim with the pre-trim start
of segment N+1, which is exactly the start of the first matched word on the later side. The
end of the matched run on the earlier side is the end of segment N, because the rule only
ever matches the *last* k words of N. So:

    overlap = end_of_segment_N - start_of_segment_N_plus_1

is positive when the two occurrences share time and negative when they abut with a gap.

The one imprecision is that `end_of_segment_N` is read from the timings the API returns,
which are post-alignment, while the logged start is pre-alignment. Alignment moves segment
edges by a median of 0.07 to 0.23 s in the harness measurements, so treat any result inside
roughly +/- 0.3 s as unresolved rather than as a verdict. A four or five word phrase runs
1.5 to 2.5 s, so a true double-decode should clear that noise floor comfortably; that it
clears it is itself part of the finding.

A definitive version needs one more field in the service's own log line: the word-timestamp
span of the matched run on each side, both of which `_segment_tokens` already has in hand at
the moment it logs. That is a one-line change and it removes the caveat entirely.

    python3 tools/quality_sweep/trims.py --service-log /home/jay/sweep/run/service.log \
        --timings /home/jay/sweep/run/timings --out-json /home/jay/sweep/trims.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import service_log  # noqa: E402
from analyze import distribution  # noqa: E402

# Alignment moves a segment edge by this much, so a measured overlap smaller than this is
# not evidence either way.
NOISE_FLOOR_S = 0.3


def load_segments(timings_dir, filename):
    """Sorted (start, end) pairs for a file, from the timings the runner saved."""
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
    return sorted(spans)


def preceding_segment(segments, at_s):
    """The segment immediately before `at_s`: the one the matched run ends in."""
    best = None
    for start, end in segments:
        if start < at_s:
            if best is None or start > best[0]:
                best = (start, end)
        else:
            break
    return best


def classify(overlap_s, noise_floor=NOISE_FLOOR_S):
    """`overlapping`, `abutting` or `unresolved` for one measured overlap."""
    if overlap_s is None:
        return "unknown"
    if overlap_s > noise_floor:
        return "overlapping"
    if overlap_s < -noise_floor:
        return "abutting"
    return "unresolved"


def measure(trim, segments, noise_floor=NOISE_FLOOR_S):
    """One row: the two spans as far as they are known, and whether they overlap."""
    at_s = trim.get("at_s")
    row = {
        "at_s": at_s,
        "overlap_words": trim.get("overlap_words"),
        "removed": trim.get("removed"),
        "emptied_segment": trim.get("emptied"),
        "segment_n_end_s": None,
        "segment_n_start_s": None,
        "overlap_s": None,
        "gap_s": None,
        "verdict": "unknown",
    }
    if at_s is None or not segments:
        return row
    previous = preceding_segment(segments, at_s)
    if previous is None:
        return row
    row["segment_n_start_s"] = round(previous[0], 3)
    row["segment_n_end_s"] = round(previous[1], 3)
    overlap = previous[1] - at_s
    row["overlap_s"] = round(overlap, 3)
    row["gap_s"] = round(-overlap, 3) if overlap < 0 else 0.0
    row["verdict"] = classify(overlap, noise_floor)
    return row


def build(jobs, timings_dir, noise_floor=NOISE_FLOOR_S):
    """Every trim across every job, with its overlap measurement."""
    rows = []
    for record in jobs.values():
        filename = record.get("file")
        if not filename or not record.get("trims"):
            continue
        segments = load_segments(timings_dir, filename) or []
        for trim in record["trims"]:
            row = measure(trim, segments, noise_floor)
            row["file"] = filename
            row["guid"] = record.get("guid")
            rows.append(row)
    rows.sort(key=lambda r: (r["file"], r["at_s"] or 0))
    return rows


def summarise(rows, noise_floor=NOISE_FLOOR_S):
    """Whether the overlap test separates the trims, and by how much."""
    verdicts = {}
    for row in rows:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
    measured = [r["overlap_s"] for r in rows if r["overlap_s"] is not None]
    overlapping = [v for v in measured if v > noise_floor]
    abutting = [v for v in measured if v < -noise_floor]
    unresolved = [v for v in measured if abs(v) <= noise_floor]
    separates = bool(overlapping) and bool(abutting) and not unresolved
    return {
        "trims": len(rows),
        "files": len({r["file"] for r in rows}),
        "words_removed": sum(r["overlap_words"] or 0 for r in rows),
        "segments_emptied": sum(1 for r in rows if r["emptied_segment"]),
        "verdicts": verdicts,
        "noise_floor_s": noise_floor,
        "overlap_s": distribution(measured, digits=3),
        "overlapping": distribution(overlapping, digits=3),
        "abutting": distribution(abutting, digits=3),
        "unresolved_count": len(unresolved),
        "separates_cleanly": separates,
    }


def render_markdown(summary, rows):
    out = ["# Seam de-duplication trims: overlap or abut", ""]
    out.append(
        f"{summary['trims']} trims across {summary['files']} files removed "
        f"{summary['words_removed']} words; {summary['segments_emptied']} trims emptied "
        f"their segment entirely."
    )
    out.append("")
    verdicts = ", ".join(f"{k} {v}" for k, v in sorted(summary["verdicts"].items()))
    out.append(
        f"Verdicts at a +/- {summary['noise_floor_s']} s noise floor: {verdicts}. "
        + (
            "The two populations separate with nothing in between, so overlap in time is a "
            "usable rule."
            if summary["separates_cleanly"]
            else "The populations do not separate cleanly, so overlap in time is not on its "
            "own a reliable rule."
        )
    )
    out.append("")
    out.append(
        "Overlap is the end of segment N minus the pre-trim start of segment N+1. Positive "
        "means the two occurrences share time, which is what decoding the same audio twice "
        "looks like. Negative means they abut, which is what a speaker saying it twice looks "
        "like. Segment N's end is post-alignment and the start is pre-alignment, so results "
        "inside the noise floor are unresolved rather than decided."
    )
    out.append("")
    out.append("| file | at s | words | phrase | seg N end | overlap s | verdict | emptied |")
    out.append("|---|---|---|---|---|---|---|---|")
    for row in rows:
        out.append(
            f"| {row.get('file')} | {row.get('at_s')} | {row.get('overlap_words')} | "
            f"{(row.get('removed') or '').replace('|', '/')} | "
            f"{row.get('segment_n_end_s')} | {row.get('overlap_s')} | "
            f"{row.get('verdict')} | {'yes' if row.get('emptied_segment') else 'no'} |"
        )
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-log", required=True)
    parser.add_argument("--timings", required=True)
    parser.add_argument("--noise-floor", type=float, default=NOISE_FLOOR_S)
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    args = parser.parse_args(argv)

    with open(args.service_log, errors="replace") as handle:
        jobs = service_log.parse(handle)
    rows = build(jobs, args.timings, args.noise_floor)
    summary = summarise(rows, args.noise_floor)

    if args.out_json:
        with open(args.out_json, "w") as handle:
            json.dump({"summary": summary, "trims": rows}, handle, indent=1)
            handle.write("\n")
    if args.out_md:
        with open(args.out_md, "w") as handle:
            handle.write(render_markdown(summary, rows))
            handle.write("\n")
    print(render_markdown(summary, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
