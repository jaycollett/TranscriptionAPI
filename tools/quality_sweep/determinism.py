"""Compare two sweep runs of the same files to measure run-to-run variation.

The 0.6.0 decode is not bit-reproducible on longer files. That matters more than a word
or two would normally, because the rescue pass fires on a threshold over the primary
decode's anomaly count: if the anomaly count sits on the threshold, one run can take the
rescue path and the next can skip it, and the two runs then differ by a whole decode
rather than by rounding.

This module pairs a file's record in run A with its record in run B and reports the word
count, anomaly count and words-per-second spread, whether the rescue decision changed,
and whether the transcript was identical. `rescue_decision_changed` is the number that
answers the question; the word spread only says how large the noise is.

    python3 tools/quality_sweep/determinism.py \
        --run-a /home/jay/sweep/run/state.json \
        --run-b /home/jay/sweep/repeat/state.json \
        --out-json /home/jay/sweep/determinism.json --out-md /home/jay/sweep/determinism.md
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from norm import norm_words  # noqa: E402

# Fields compared between the two runs. Booleans are compared for a flip; numbers for a
# spread.
NUMERIC_FIELDS = ("anomaly_count", "attempt_count", "processing_seconds")
BOOLEAN_FIELDS = ("rescue_attempted", "rescue_selected", "mfa_applied")


def text_digest(transcripts_dir, filename):
    """SHA-256 of the normalised transcript, or None when the text was not kept."""
    if not transcripts_dir:
        return None
    stem = os.path.splitext(filename)[0]
    path = os.path.join(transcripts_dir, stem + ".txt")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return hashlib.sha256(" ".join(norm_words(handle.read())).encode("utf-8")).hexdigest()


def pair(record_a, record_b, digest_a=None, digest_b=None):
    """One row comparing the same file across two runs."""
    derived_a = record_a.get("derived") or {}
    derived_b = record_b.get("derived") or {}
    payload_a = record_a.get("payload") or {}
    payload_b = record_b.get("payload") or {}

    words_a = derived_a.get("words")
    words_b = derived_b.get("words")
    row = {
        "file": record_a.get("file"),
        "duration_s": record_a.get("duration_s"),
        "outcome_a": record_a.get("outcome"),
        "outcome_b": record_b.get("outcome"),
        "words_a": words_a,
        "words_b": words_b,
        "word_spread": (words_b - words_a) if words_a is not None and words_b is not None else None,
        "wps_a": derived_a.get("wps"),
        "wps_b": derived_b.get("wps"),
        "segments_a": (derived_a.get("timings") or {}).get("count"),
        "segments_b": (derived_b.get("timings") or {}).get("count"),
        "identical_text": (
            None if digest_a is None or digest_b is None else digest_a == digest_b
        ),
    }
    if words_a:
        row["word_spread_pct"] = round((words_b - words_a) / words_a, 5) if words_b is not None else None
    else:
        row["word_spread_pct"] = None

    for field in NUMERIC_FIELDS:
        value_a = payload_a.get(field)
        value_b = payload_b.get(field)
        row[field + "_a"] = value_a
        row[field + "_b"] = value_b
        row[field + "_spread"] = (
            value_b - value_a
            if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float))
            else None
        )
    for field in BOOLEAN_FIELDS:
        value_a = payload_a.get(field)
        value_b = payload_b.get(field)
        row[field + "_a"] = value_a
        row[field + "_b"] = value_b
        row[field + "_flipped"] = (
            None if value_a is None or value_b is None else bool(value_a) != bool(value_b)
        )
    row["rescue_decision_changed"] = bool(
        row.get("rescue_attempted_flipped") or row.get("rescue_selected_flipped")
    )
    return row


def compare(state_a, state_b, transcripts_a=None, transcripts_b=None):
    files_a = state_a.get("files", {})
    files_b = state_b.get("files", {})
    rows = []
    for filename in sorted(set(files_a) & set(files_b)):
        rows.append(pair(
            files_a[filename],
            files_b[filename],
            text_digest(transcripts_a, filename),
            text_digest(transcripts_b, filename),
        ))

    spreads = [abs(r["word_spread"]) for r in rows if r["word_spread"] is not None]
    pct = [abs(r["word_spread_pct"]) for r in rows if r["word_spread_pct"] is not None]
    anomaly = [
        abs(r["anomaly_count_spread"]) for r in rows if r.get("anomaly_count_spread") is not None
    ]
    return {
        "pairs": len(rows),
        "identical_text": sum(1 for r in rows if r["identical_text"]),
        "text_compared": sum(1 for r in rows if r["identical_text"] is not None),
        "max_abs_word_spread": max(spreads) if spreads else None,
        "max_abs_word_spread_pct": max(pct) if pct else None,
        "max_abs_anomaly_spread": max(anomaly) if anomaly else None,
        "rescue_decision_changed": sum(1 for r in rows if r["rescue_decision_changed"]),
        "rows": rows,
    }


def render_markdown(result):
    out = ["# Run-to-run variation", ""]
    out.append(
        f"{result['pairs']} files submitted twice. "
        f"Largest word spread {result['max_abs_word_spread']} "
        f"({result['max_abs_word_spread_pct']} of the file). "
        f"Largest anomaly-count spread {result['max_abs_anomaly_spread']}. "
        f"Rescue decision changed on {result['rescue_decision_changed']} of "
        f"{result['pairs']} files. "
        f"{result['identical_text']} of {result['text_compared']} transcripts identical."
    )
    out.append("")
    out.append(
        "| file | dur s | words A | words B | spread | anomaly A | anomaly B | "
        "rescue att A/B | rescue sel A/B | changed | identical text |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in sorted(result["rows"], key=lambda r: -(r["duration_s"] or 0)):
        out.append(
            f"| {row['file']} | {row['duration_s']} | {row['words_a']} | {row['words_b']} | "
            f"{row['word_spread']} | {row['anomaly_count_a']} | {row['anomaly_count_b']} | "
            f"{row['rescue_attempted_a']}/{row['rescue_attempted_b']} | "
            f"{row['rescue_selected_a']}/{row['rescue_selected_b']} | "
            f"{'yes' if row['rescue_decision_changed'] else 'no'} | {row['identical_text']} |"
        )
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--transcripts-a")
    parser.add_argument("--transcripts-b")
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    args = parser.parse_args(argv)

    with open(args.run_a) as handle:
        state_a = json.load(handle)
    with open(args.run_b) as handle:
        state_b = json.load(handle)

    transcripts_a = args.transcripts_a or os.path.join(
        os.path.dirname(os.path.abspath(args.run_a)), "transcripts"
    )
    transcripts_b = args.transcripts_b or os.path.join(
        os.path.dirname(os.path.abspath(args.run_b)), "transcripts"
    )
    result = compare(state_a, state_b, transcripts_a, transcripts_b)
    text = render_markdown(result)
    if args.out_json:
        with open(args.out_json, "w") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")
    if args.out_md:
        with open(args.out_md, "w") as handle:
            handle.write(text)
            handle.write("\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
