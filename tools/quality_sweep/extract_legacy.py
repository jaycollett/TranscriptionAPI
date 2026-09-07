"""Build the legacy baseline from a read-only copy of the orchestrator database.

The live `sermon_orchestrator.db` is never opened by this script. Copy it first and pass
the copy; the SQLite URI adds `mode=ro` so even the copy cannot be written.

Rows are keyed by the basename of `orig_audio_url`, which is exactly the filename in the
reference archive. Only aggregates are written out: word and character counts, words per
second over the ffprobe duration, and a SHA-256 of the normalised text so a later run can
tell whether a transcript changed without holding the text. Transcript text stays on the
GPU host.

    cp /path/to/sermon_orchestrator.db /home/jay/sweep/legacy.db
    python3 tools/quality_sweep/extract_legacy.py \
        --db /home/jay/sweep/legacy.db \
        --audio-dir /home/jay/SourceCode/SermonPreprocessorAPI/data/audiofiles \
        --durations /home/jay/sweep/durations.csv \
        --out /home/jay/sweep/legacy_baseline.json
"""

import argparse
import collections
import hashlib
import json
import os
import sqlite3
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from norm import norm_words  # noqa: E402
from strata import era_for, read_durations  # noqa: E402

TRANSCRIPTION_STEP = "GenerateEnglishTranscription"


def basename_of(url):
    """Filename an `orig_audio_url` points at, query string and escaping removed."""
    path = urllib.parse.urlsplit(url or "").path
    return urllib.parse.unquote(os.path.basename(path))


def timing_count(raw):
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return len(parsed) if isinstance(parsed, list) else None


def build(db_path, audio_dir, durations_path):
    durations = read_durations(durations_path)
    files = set(os.listdir(audio_dir))
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    finished = {
        guid: stamp
        for guid, stamp in con.execute(
            "select sermon_guid, max(finished_at) from sermon_steps "
            "where step_name = ? and status = 'completed' group by sermon_guid",
            (TRANSCRIPTION_STEP,),
        )
    }

    baseline = {}
    skipped = []
    for guid, title, url, fetched, text, timings in con.execute(
        "select sermon_guid, title, orig_audio_url, fetched_date, "
        "english_transcription, english_transcription_timings from sermon_metadata"
    ):
        name = basename_of(url)
        if name not in files:
            skipped.append({"guid": guid, "reason": "no audio file", "url": url})
            continue
        if not text or not text.strip():
            skipped.append({"guid": guid, "reason": "empty transcript", "file": name})
            continue
        words = norm_words(text)
        duration = durations.get(name)
        stamp = finished.get(guid)
        baseline[name] = {
            "sermon_guid": guid,
            "title": title,
            "fetched_date": fetched,
            "transcription_finished_at": stamp,
            "legacy_era": era_for(stamp),
            "duration_s": duration,
            "words": len(words),
            "chars": len(text),
            "wps": round(len(words) / duration, 4) if duration else None,
            "sha256_norm": hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest(),
            "timing_count": timing_count(timings),
        }
    return baseline, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="a COPY of sermon_orchestrator.db")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--durations", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    baseline, skipped = build(args.db, args.audio_dir, args.durations)
    with open(args.out, "w") as handle:
        json.dump({"baseline": baseline, "skipped": skipped}, handle, indent=1, sort_keys=True)
        handle.write("\n")

    eras = collections.Counter(row["legacy_era"] for row in baseline.values())
    print(f"wrote {args.out}: {len(baseline)} rows, {len(skipped)} skipped")
    for era, count in sorted(eras.items()):
        print(f"  {era}: {count}")
    for entry in skipped:
        print(f"  skipped {entry.get('file') or entry.get('url')}: {entry['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
