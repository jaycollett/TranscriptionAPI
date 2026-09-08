"""Decode-only pass over the sweep's hundred files, using the candidate's own scoring.

The question is what `anomaly_count` looks like on the candidate, so that the segment cap
can be set from the corpus rather than from a round number. Reimplementing the score would
answer a different question, so this imports the image's own `transcribe` module and calls
`transcribe_audio`, which is the exact decode-and-score path the service runs, minus the
aligner, which lives in `app.py` and is not in the loop here.

Resumable: each file's record is written as it completes and an existing record is skipped,
because the pass takes hours and an interrupted run should not start over.

    python3 candidate_pass.py --file-list file_list.json --audio-dir /audio \
        --out /work/candidate_pass.json
"""

import argparse
import json
import os
import sys
import time
import uuid

sys.path.insert(0, "/app")


def load_records(path):
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    return {"files": {}, "repeats": {}}


def save_records(path, records):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(records, handle, indent=1, sort_keys=True)
    os.replace(tmp, path)


def flag_counts(flagged):
    counts = {}
    for entry in flagged or []:
        for flag in entry.get("flags") or []:
            counts[flag] = counts.get(flag, 0) + 1
    return counts


def summarise(result, wall):
    """The fields worth keeping. The transcript is kept for the rc2 comparison."""
    segments = result.get("segments") or []
    return {
        "words": len((result.get("transcription") or "").split()),
        "segments": len(segments),
        "anomaly_count": result.get("anomaly_count"),
        "anomaly_windows": result.get("anomaly_windows"),
        "flagged_segments": len(result.get("flagged_segments") or []),
        "flag_counts": flag_counts(result.get("flagged_segments")),
        "rescue_attempted": result.get("rescue_attempted"),
        "rescue_selected": result.get("rescue_selected"),
        # 0.6.1 names which signal asked for the second decode, so the firing mix is
        # observable rather than inferred, and reports the longest run of words the
        # unpublished pass had that the published one lacks. That run is the directional
        # check: a rescue that is merely different scores near zero, one that recovered a
        # passage scores the length of it.
        "rescue_triggers": result.get("rescue_triggers"),
        "rescue_unpublished_run": result.get("rescue_unpublished_run"),
        "primary_words": result.get("primary_words"),
        "primary_uncovered_max_gap_s": result.get("primary_uncovered_max_gap_s"),
        "vad_profile": result.get("vad_profile"),
        "mean_logprob": result.get("mean_logprob"),
        "speech_seconds": result.get("speech_seconds"),
        "duration_sec": result.get("duration_sec"),
        "mean_dbfs": result.get("mean_dbfs"),
        "vad_threshold": result.get("vad_threshold"),
        "uncovered_s": result.get("uncovered_s"),
        "uncovered_max_gap_s": result.get("uncovered_max_gap_s"),
        "primary_anomaly_count": result.get("primary_anomaly_count"),
        "primary_anomaly_windows": result.get("primary_anomaly_windows"),
        "wall_s": round(wall, 2),
        "transcription": result.get("transcription") or "",
    }


def run_one(transcribe, audio_dir, name):
    started = time.time()
    result = transcribe.transcribe_audio(os.path.join(audio_dir, name), str(uuid.uuid4()))
    return summarise(result, time.time() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeat", action="append", default=[],
                        help="submit these a second time, for the determinism check")
    args = parser.parse_args(argv)

    import transcribe

    with open(args.file_list) as handle:
        doc = json.load(handle)
    names = [r["file"] for r in doc["files"]] + [r["file"] for r in doc.get("supplementary", [])]

    records = load_records(args.out)
    for index, name in enumerate(names, 1):
        if name in records["files"]:
            print(f"[{index}/{len(names)}] skip {name}", flush=True)
            continue
        try:
            entry = run_one(transcribe, args.audio_dir, name)
        except Exception as exc:  # a failure on one file must not end the pass
            entry = {"error": f"{type(exc).__name__}: {exc}"}
        records["files"][name] = entry
        save_records(args.out, records)
        print(f"[{index}/{len(names)}] {name} words={entry.get('words')} "
              f"segs={entry.get('segments')} anomaly={entry.get('anomaly_count')} "
              f"windows={entry.get('anomaly_windows')} "
              f"rescue={entry.get('rescue_attempted')}/{entry.get('rescue_selected')} "
              f"wall={entry.get('wall_s')}s", flush=True)

    for name in args.repeat:
        if name in records["repeats"]:
            continue
        try:
            entry = run_one(transcribe, args.audio_dir, name)
        except Exception as exc:
            entry = {"error": f"{type(exc).__name__}: {exc}"}
        records["repeats"][name] = entry
        save_records(args.out, records)
        print(f"[repeat] {name} words={entry.get('words')} "
              f"anomaly={entry.get('anomaly_count')} "
              f"rescue={entry.get('rescue_attempted')}/{entry.get('rescue_selected')}",
              flush=True)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
