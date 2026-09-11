"""Does disagreement between two decodes predict a bad transcript?

The open question from the rc2 sweep is that `tcf.20150424` publishes about 15 percent
short of its legacy transcript while every signal the service has reads clean: zero
flagged segments, zero low-rate windows, no uncovered gap over 20 s. The signal set is
incomplete, not merely mistuned.

This module tests a signal the service does not currently have and could get cheaply:
the agreement between two independent decodes of the same audio. Nothing new is decoded
here. The corpus already holds two full decodes of the same hundred recordings from two
different builds of the same release line, so the question can be answered from data on
disk before any GPU time is spent.

Two statistics per file, both over `norm_words` tokens so case and punctuation never
count:

- `agreement`: `SequenceMatcher.ratio()`, i.e. 2M/(len(a)+len(b)) over matched tokens.
  A whole-file number, so a small contiguous omission in a long recording barely moves
  it.
- `max_run`: the longest single stretch of consecutive tokens that one decode has and
  the other does not, in words. This is the local statistic, and it is the shape the
  read-aloud defect actually has: one contiguous block of scripture present in one
  decode and absent from the other.

The ground truth is the same `unhealthy` predicate `candidate_anomaly.py` uses, which
is deliberately independent of both statistics here: word rate over speech, the largest
uncovered gap, and the delta against the legacy transcript.
"""

import argparse
import difflib
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm import norm_words  # noqa: E402

WORSE_TOLERANCE = 0.05


def unhealthy(entry, legacy_words):
    """Independent evidence that a file's transcript is bad. Mirrors candidate_anomaly."""
    reasons = []
    speech = entry.get("speech_seconds") or 0
    words = entry.get("words") or 0
    if speech and words / speech < 1.5:
        reasons.append(f"words/sec over speech {words / speech:.2f}")
    if (entry.get("uncovered_max_gap_s") or 0) >= 20.0:
        reasons.append(f"largest uncovered gap {entry['uncovered_max_gap_s']}s")
    if legacy_words:
        delta = (words - legacy_words) / legacy_words
        if delta < -WORSE_TOLERANCE:
            reasons.append(f"{delta * 100:.1f}% against legacy")
    return reasons


def compare(a_words, b_words):
    """Agreement ratio and the longest one-sided run, from one diff of the two decodes."""
    matcher = difflib.SequenceMatcher(a=a_words, b=b_words, autojunk=False)
    blocks = matcher.get_matching_blocks()
    matched = sum(block.size for block in blocks)
    total = len(a_words) + len(b_words)
    agreement = (2.0 * matched / total) if total else None

    # Walk the gaps between matching blocks. Each gap is a stretch a_i..a_j present only
    # in A and b_i..b_j present only in B; the longer side is the run length, because a
    # substitution of one word for another is not an omission of either.
    max_run = 0
    max_run_side = None
    max_run_at = None
    pos_a = pos_b = 0
    for block in blocks:
        run_a = block.a - pos_a
        run_b = block.b - pos_b
        if max(run_a, run_b) > max_run:
            max_run = max(run_a, run_b)
            max_run_side = "a" if run_a >= run_b else "b"
            max_run_at = pos_a if run_a >= run_b else pos_b
        pos_a = block.a + block.size
        pos_b = block.b + block.size
    return {
        "agreement": round(agreement, 5) if agreement is not None else None,
        "max_run": max_run,
        "max_run_side": max_run_side,
        # Where in the file the run sits, as a fraction, so an opening-minutes defect is
        # visible as one.
        "max_run_frac": (round(max_run_at / max(1, len(a_words) if max_run_side == "a"
                                                else len(b_words)), 4)
                         if max_run_at is not None else None),
    }


def load_rows(analysis_path):
    """legacy word counts per file, from the rc2 analysis."""
    with open(analysis_path) as handle:
        doc = json.load(handle)
    return {row["file"]: row for row in doc.get("rows", [])}


def separation(per_file, key, bad_names, higher_is_worse):
    """How well `key` separates the known-bad files from the rest.

    Reported as the threshold that catches each number of bad files and the false
    positives it costs, because "does it separate" is a question about the whole
    trade-off curve and not about one cutoff.
    """
    scored = [(name, rec[key]) for name, rec in per_file.items() if rec.get(key) is not None]
    bad = [(n, v) for n, v in scored if n in bad_names]
    good = [(n, v) for n, v in scored if n not in bad_names]
    if not bad or not good:
        return None

    # Rank the bad files among all files, worst first.
    ordered = sorted(scored, key=lambda kv: -kv[1] if higher_is_worse else kv[1])
    rank = {name: i + 1 for i, (name, _) in enumerate(ordered)}

    curve = []
    bad_sorted = sorted(bad, key=lambda kv: -kv[1] if higher_is_worse else kv[1])
    for caught, (name, value) in enumerate(bad_sorted, 1):
        if higher_is_worse:
            fps = sum(1 for _, v in good if v >= value)
        else:
            fps = sum(1 for _, v in good if v <= value)
        curve.append({"caught": caught, "of": len(bad), "threshold": value,
                      "last_file": name, "false_positives": fps,
                      "flagged_total": caught + fps})

    return {
        "bad": sorted(({"file": n, "value": v, "rank": rank[n]} for n, v in bad),
                      key=lambda d: d["rank"]),
        "good_median": round(statistics.median(v for _, v in good), 5),
        "good_p95": round(sorted(v for _, v in good)[int(0.95 * (len(good) - 1))], 5),
        "good_max": round(max(v for _, v in good), 5),
        "good_min": round(min(v for _, v in good), 5),
        "n_good": len(good),
        "n_bad": len(bad),
        "curve": curve,
    }


def build(rc2_dir, candidate_path, analysis_path):
    with open(candidate_path) as handle:
        candidate = json.load(handle)["files"]
    rows = load_rows(analysis_path)

    per_file = {}
    bad_names = set()
    for name, entry in sorted(candidate.items()):
        if entry.get("error") or not entry.get("transcription"):
            continue
        stem = os.path.splitext(name)[0]
        rc2_path = os.path.join(rc2_dir, stem + ".txt")
        if not os.path.exists(rc2_path):
            continue
        with open(rc2_path) as handle:
            a_words = norm_words(handle.read())
        b_words = norm_words(entry["transcription"])
        if not a_words or not b_words:
            continue
        row = rows.get(name) or {}
        legacy = row.get("legacy_words")
        reasons = unhealthy(entry, legacy)
        if reasons:
            bad_names.add(name)
        rec = compare(a_words, b_words)
        rec.update({
            "rc2_words": len(a_words),
            "candidate_words": len(b_words),
            "word_delta": len(b_words) - len(a_words),
            "legacy_words": legacy,
            "duration_sec": entry.get("duration_sec"),
            "unhealthy_reasons": reasons,
            # The run as a share of the shorter decode, so a 90-word omission in a
            # 1200-word recording is not read as equivalent to one in 9000 words.
            "max_run_rate": round(rec["max_run"] / min(len(a_words), len(b_words)), 5),
            "disagreement": round(1.0 - rec["agreement"], 5),
        })
        per_file[name] = rec

    return {
        "n": len(per_file),
        "bad_files": sorted(bad_names),
        "separation": {
            "disagreement": separation(per_file, "disagreement", bad_names, True),
            "max_run": separation(per_file, "max_run", bad_names, True),
            "max_run_rate": separation(per_file, "max_run_rate", bad_names, True),
        },
        "per_file": per_file,
    }


def fmt(value, digits=4):
    return "-" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def render(doc):
    out = ["# Does a second decode's disagreement detect a bad transcript?", ""]
    out.append(f"{doc['n']} recordings, each decoded twice by two builds of the same "
               f"release line: the rc2 sweep transcript and the shipped 0.6.0 "
               f"decode-only pass. No new decoding.")
    out.append("")
    out.append(f"Known bad, by evidence independent of both statistics "
               f"({len(doc['bad_files'])} files): " + ", ".join(doc["bad_files"]))
    out.append("")

    for key, title, note in [
        ("disagreement", "Whole-file disagreement (1 - agreement)",
         "A whole-file ratio. One contiguous omission in a long recording moves it very "
         "little, which is the reason to measure the run as well."),
        ("max_run", "Longest one-sided run, in words",
         "The longest stretch of consecutive words one decode has and the other does not."),
        ("max_run_rate", "Longest one-sided run, as a share of the transcript",
         "The same run normalised by the shorter decode's length."),
    ]:
        sep = doc["separation"].get(key)
        out.append(f"## {title}")
        out.append("")
        out.append(note)
        out.append("")
        if not sep:
            out.append("Not computable.")
            out.append("")
            continue
        out.append(f"Healthy ({sep['n_good']} files): median {fmt(sep['good_median'], 5)}, "
                   f"p95 {fmt(sep['good_p95'], 5)}, max {fmt(sep['good_max'], 5)}.")
        out.append("")
        out.append("| bad file | value | rank of " + str(sep["n_good"] + sep["n_bad"]) + " |")
        out.append("|---|---|---|")
        for entry in sep["bad"]:
            out.append(f"| {entry['file']} | {fmt(entry['value'], 5)} | {entry['rank']} |")
        out.append("")
        out.append("| threshold catches | of | at value | healthy files also flagged | total flagged |")
        out.append("|---|---|---|---|---|")
        for point in sep["curve"]:
            out.append(f"| {point['caught']} | {point['of']} | {fmt(point['threshold'], 5)} "
                       f"| {point['false_positives']} | {point['flagged_total']} |")
        out.append("")

    out.append("## Every file, ranked by longest one-sided run")
    out.append("")
    out.append("| file | run | run rate | disagreement | rc2 words | 0.6.0 words | legacy | bad |")
    out.append("|---|---|---|---|---|---|---|---|")
    ranked = sorted(doc["per_file"].items(), key=lambda kv: -kv[1]["max_run"])
    for name, rec in ranked:
        out.append(f"| {name} | {rec['max_run']} | {fmt(rec['max_run_rate'], 4)} "
                   f"| {fmt(rec['disagreement'], 4)} | {rec['rc2_words']} "
                   f"| {rec['candidate_words']} | {rec['legacy_words'] or '-'} "
                   f"| {'yes' if rec['unhealthy_reasons'] else ''} |")
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rc2-transcripts", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args(argv)

    doc = build(args.rc2_transcripts, args.candidate, args.analysis)
    with open(args.out_json, "w") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
    with open(args.out_md, "w") as handle:
        handle.write(render(doc))
    print(f"{doc['n']} files, {len(doc['bad_files'])} known bad -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
