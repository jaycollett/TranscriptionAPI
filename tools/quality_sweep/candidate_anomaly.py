"""Read the anomaly cap off the candidate's own decode of the sweep corpus.

Answers four things from `candidate_pass.py`'s output:

1. The distribution of `anomaly_count` and `anomaly_windows`, so a cap can be set from the
   corpus tail with headroom instead of from a round number.
2. Whether the count scales with segment count. If it does, an absolute cap means different
   things on a 435-segment recording and a 3645-segment one, and a rate is the better
   shape. The test is the rank correlation plus how well each of the two separates the
   files a human would call unhealthy.
3. Whether reproducibility changed: the repeat submissions, split by whether the published
   pass came from the rescue, which samples unseeded.
4. Which files came out materially worse than their rc2 counterparts.

    python3 candidate_anomaly.py --candidate candidate_pass.json \
        --rc2 /home/jay/sweep/analysis.json --transcripts /home/jay/sweep/run/transcripts \
        --out-json candidate_anomaly.json --out-md candidate_anomaly.md
"""

import argparse
import collections
import difflib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze import distribution, percentile  # noqa: E402
from norm import norm_words  # noqa: E402

# A file is materially worse than its rc2 counterpart when it loses more than this share of
# rc2's words. Same tolerance as the sweep's regression rule, for consistency.
WORSE_TOLERANCE = 0.05


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0] * len(values)
    for position, index in enumerate(order):
        out[index] = position
    return out


def spearman(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    if len(pairs) < 10:
        return None
    a, b = zip(*pairs)
    ra, rb = rank(list(a)), rank(list(b))
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 4) if den else None


def histogram(values):
    return dict(sorted(collections.Counter(values).items()))


def at_or_above(values, k):
    return sum(1 for v in values if v >= k)


def unhealthy(name, entry, rc2_row):
    """Whether a file's own output looks bad, judged without reference to the anomaly count.

    Deliberately independent of the signal under test: a cap that fires on these is doing
    its job, and one that fires elsewhere is not.
    """
    reasons = []
    speech = entry.get("speech_seconds") or 0
    words = entry.get("words") or 0
    if speech and words / speech < 1.5:
        reasons.append(f"words/sec over speech {words / speech:.2f}")
    if (entry.get("uncovered_max_gap_s") or 0) >= 20.0:
        reasons.append(f"largest uncovered gap {entry['uncovered_max_gap_s']}s")
    if rc2_row and rc2_row.get("legacy_words"):
        delta = (words - rc2_row["legacy_words"]) / rc2_row["legacy_words"]
        if delta < -WORSE_TOLERANCE:
            reasons.append(f"{delta * 100:.1f}% against legacy")
    return reasons


def compare_to_rc2(entry, rc2_words, transcript_path):
    """Word delta and similarity against the rc2 transcript for the same file."""
    out = {"rc2_words": rc2_words, "word_delta": None, "word_delta_pct": None,
           "similarity": None}
    words = entry.get("words")
    if rc2_words and words is not None:
        out["word_delta"] = words - rc2_words
        out["word_delta_pct"] = round((words - rc2_words) / rc2_words, 5)
    if transcript_path and os.path.exists(transcript_path) and entry.get("transcription"):
        with open(transcript_path) as handle:
            old = norm_words(handle.read())
        new = norm_words(entry["transcription"])
        if old and new:
            out["similarity"] = round(
                difflib.SequenceMatcher(a=old, b=new, autojunk=False).quick_ratio(), 4
            )
    return out


def analyse(candidate, rc2_rows, transcripts_dir):
    files = {n: e for n, e in candidate["files"].items() if not e.get("error")}
    errors = {n: e for n, e in candidate["files"].items() if e.get("error")}

    counts = [e["anomaly_count"] for e in files.values() if e.get("anomaly_count") is not None]
    windows = [e["anomaly_windows"] for e in files.values() if e.get("anomaly_windows") is not None]
    segments = [e["segments"] for e in files.values()]

    rates = []
    for entry in files.values():
        if entry.get("segments"):
            rates.append(100.0 * (entry.get("anomaly_count") or 0) / entry["segments"])

    per_file = {}
    for name, entry in files.items():
        row = rc2_rows.get(name) or {}
        stem = os.path.splitext(name)[0]
        per_file[name] = {
            "anomaly_count": entry.get("anomaly_count"),
            "anomaly_windows": entry.get("anomaly_windows"),
            "segments": entry.get("segments"),
            "rate_per_100_segments": (
                round(100.0 * (entry.get("anomaly_count") or 0) / entry["segments"], 3)
                if entry.get("segments") else None
            ),
            "words": entry.get("words"),
            "speech_seconds": entry.get("speech_seconds"),
            "uncovered_max_gap_s": entry.get("uncovered_max_gap_s"),
            "rescue_attempted": entry.get("rescue_attempted"),
            "rescue_selected": entry.get("rescue_selected"),
            "flag_counts": entry.get("flag_counts"),
            "unhealthy_reasons": unhealthy(name, entry, row),
            "vs_rc2": compare_to_rc2(
                entry, row.get("new_words"),
                os.path.join(transcripts_dir, stem + ".txt") if transcripts_dir else None,
            ),
        }

    bad = [n for n, v in per_file.items() if v["unhealthy_reasons"]]
    healthy = [n for n in per_file if n not in bad]

    def split(key):
        return ([per_file[n][key] for n in healthy if per_file[n][key] is not None],
                [per_file[n][key] for n in bad if per_file[n][key] is not None])

    healthy_counts, bad_counts = split("anomaly_count")
    healthy_rates, bad_rates = split("rate_per_100_segments")

    worse = sorted(
        ({"file": n, **v["vs_rc2"], "anomaly_count": v["anomaly_count"]}
         for n, v in per_file.items()
         if v["vs_rc2"]["word_delta_pct"] is not None
         and v["vs_rc2"]["word_delta_pct"] < -WORSE_TOLERANCE),
        key=lambda r: r["word_delta_pct"],
    )

    repeats = {}
    for name, entry in (candidate.get("repeats") or {}).items():
        first = files.get(name)
        if not first or entry.get("error"):
            continue
        repeats[name] = {
            "words_a": first.get("words"), "words_b": entry.get("words"),
            "word_spread": (entry.get("words") or 0) - (first.get("words") or 0),
            "anomaly_a": first.get("anomaly_count"), "anomaly_b": entry.get("anomaly_count"),
            "anomaly_spread": (entry.get("anomaly_count") or 0) - (first.get("anomaly_count") or 0),
            "windows_a": first.get("anomaly_windows"), "windows_b": entry.get("anomaly_windows"),
            "rescue_a": first.get("rescue_selected"), "rescue_b": entry.get("rescue_selected"),
            "identical_text": (first.get("transcription") or "") == (entry.get("transcription") or ""),
        }

    return {
        "files_scored": len(files),
        "errors": {n: e["error"] for n, e in errors.items()},
        "anomaly_count": {
            "histogram": histogram(counts),
            "distribution": distribution(counts, digits=3),
            "at_or_above": {str(k): at_or_above(counts, k) for k in (1, 2, 3, 4, 5, 6, 8, 10)},
            "p99": percentile(counts, 99),
        },
        "anomaly_windows": {
            "histogram": histogram(windows),
            "distribution": distribution(windows, digits=3),
            "at_or_above": {str(k): at_or_above(windows, k) for k in (1, 2, 3)},
        },
        "segments": distribution(segments, digits=1),
        "rate_per_100_segments": distribution(rates, digits=4),
        "scaling": {
            "spearman_anomaly_vs_segments": spearman(segments, counts),
            "spearman_rate_vs_segments": spearman(segments, rates),
        },
        "separation": {
            "unhealthy_files": bad,
            "healthy_n": len(healthy_counts), "unhealthy_n": len(bad_counts),
            "absolute": {
                "healthy": distribution(healthy_counts, digits=3),
                "unhealthy": distribution(bad_counts, digits=3),
            },
            "rate": {
                "healthy": distribution(healthy_rates, digits=4),
                "unhealthy": distribution(bad_rates, digits=4),
            },
        },
        "worse_than_rc2": worse,
        "repeats": repeats,
        "per_file": per_file,
    }


def render_markdown(result):
    out = ["# The anomaly signal on the candidate", ""]
    ac = result["anomaly_count"]
    out.append(f"{result['files_scored']} files decoded with the candidate's own scoring.")
    out.append("")
    out.append("## anomaly_count")
    out.append("")
    out.append("| anomalous segments | files |")
    out.append("|---|---|")
    for k, v in ac["histogram"].items():
        out.append(f"| {k} | {v} |")
    d = ac["distribution"]
    out.append("")
    out.append(f"median {d.get('median')}, p75 {d.get('p75')}, p95 {d.get('p95')}, "
               f"p99 {ac['p99']}, max {d.get('max')}. At or above: "
               + ", ".join(f"{k}: {v}" for k, v in ac["at_or_above"].items()))
    out.append("")
    aw = result["anomaly_windows"]
    out.append("## anomaly_windows at the 1.5 floor")
    out.append("")
    out.append(", ".join(f"{k} windows: {v} files" for k, v in aw["histogram"].items()))
    out.append("")
    out.append("## Does the count scale with segment count")
    out.append("")
    s = result["scaling"]
    out.append(f"Spearman of anomaly_count against segment count: "
               f"**{s['spearman_anomaly_vs_segments']}**. Of the rate per hundred segments "
               f"against segment count: {s['spearman_rate_vs_segments']}.")
    out.append("")
    sep = result["separation"]
    out.append(f"Splitting on independent evidence of a bad transcript "
               f"({sep['unhealthy_n']} files) against the rest ({sep['healthy_n']}):")
    out.append("")
    out.append("| statistic | healthy median | healthy p95 | unhealthy median | unhealthy max |")
    out.append("|---|---|---|---|---|")
    for label, key in (("absolute count", "absolute"), ("per 100 segments", "rate")):
        h, u = sep[key]["healthy"], sep[key]["unhealthy"]
        out.append(f"| {label} | {h.get('median')} | {h.get('p95')} | "
                   f"{u.get('median')} | {u.get('max')} |")
    out.append("")
    out.append("## Reproducibility")
    out.append("")
    out.append("| file | words A | words B | spread | anomaly A | anomaly B | rescue A/B | identical |")
    out.append("|---|---|---|---|---|---|---|---|")
    for name, r in sorted(result["repeats"].items()):
        out.append(f"| {name} | {r['words_a']} | {r['words_b']} | {r['word_spread']} | "
                   f"{r['anomaly_a']} | {r['anomaly_b']} | {r['rescue_a']}/{r['rescue_b']} | "
                   f"{r['identical_text']} |")
    out.append("")
    out.append("## Materially worse than rc2")
    out.append("")
    if result["worse_than_rc2"]:
        out.append("| file | rc2 words | delta | delta pct | similarity | anomaly |")
        out.append("|---|---|---|---|---|---|")
        for r in result["worse_than_rc2"]:
            out.append(f"| {r['file']} | {r['rc2_words']} | {r['word_delta']} | "
                       f"{r['word_delta_pct']} | {r['similarity']} | {r['anomaly_count']} |")
    else:
        out.append("No file lost more than 5 percent of its rc2 word count.")
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--rc2", required=True)
    parser.add_argument("--transcripts")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args(argv)

    with open(args.candidate) as handle:
        candidate = json.load(handle)
    with open(args.rc2) as handle:
        rc2_rows = {r["file"]: r for r in json.load(handle)["rows"]}

    result = analyse(candidate, rc2_rows, args.transcripts)
    with open(args.out_json, "w") as handle:
        json.dump(result, handle, indent=1)
        handle.write("\n")
    with open(args.out_md, "w") as handle:
        handle.write(render_markdown(result))
        handle.write("\n")
    print(render_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
