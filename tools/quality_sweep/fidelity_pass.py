"""Decode the fidelity experiment's file set under four configurations.

The three open items 0.6.0 left behind each name a specific change, and each is a
single-lever change to the shipped decode:

- **A, control**: 0.6.0 exactly as shipped. Nothing is overlaid.
- **B**: the primary temperature ladder starts at 0.2 instead of 0.0. faster-whisper
  beam-searches only at temperature 0.0 and samples with `beam_size=1` at every rung
  above it, so this is not a nudge: it replaces beam search with sampling on the first
  attempt. That is the deferred sampling-versus-beam question, and the rc2 passage probe
  showed it recovering 4 of 4 lost read-aloud passages.
- **C**: the rescue pass keeps `condition_on_previous_text=True` while still starting its
  ladder at 0.2. The rescue changes two things at once, the probe suggested they may not
  compose, and the retention guard refuses the Exodus rescue because it loses 200-340
  words elsewhere. If disabling conditioning is what costs those words, C keeps the
  recovery and pays less for it. C's primary pass is A's primary pass bit for bit, so C
  is only worth running where A fires a rescue.
- **D**: the VAD profile is selected on sample rate and bit rate instead of level. The
  rc2 fragmentation analysis ranked encoding quality above level and found level
  non-monotonic, so no one-sided level threshold can express the shape. D is only worth
  running where the two selectors disagree.

Nothing here edits `transcribe.py`. Each configuration is an overlay applied to the
image's own module at run time: the module globals the decode reads at call time are set,
and the two selection functions are replaced. That keeps the control genuinely equal to
the shipped code (config A applies no overlay at all) and keeps every difference between
configurations attributable to the named lever.

    python3 fidelity_pass.py --file-list fidelity_file_list.json --audio-dir /audio \
        --levels /work/levels.csv --config A --out /work/fidelity/A.json
"""

import argparse
import csv
import json
import os
import sys
import time
import uuid

sys.path.insert(0, "/app")

# Matches select_fidelity_set.py; both read the same rc2 fragmentation analysis.
FIDELITY_SAMPLE_RATE_HZ = 22050
FIDELITY_BIT_RATE = 64000

# Set by the driver before each file so the fidelity selector, which the decode calls
# with the level and nothing else, can see the encoding of the file being decoded.
CURRENT = {}


def read_levels(path):
    out = {}
    with open(path) as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 4:
                continue
            parts = row[3].split(",")
            out[row[0]] = {
                "sample_rate": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None,
                "bit_rate": int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None,
            }
    return out


def apply_config(transcribe, label):
    """Overlay one configuration onto the image's transcribe module. Returns a note."""
    if label == "A":
        return "0.6.0 as shipped; no overlay"

    if label == "B":
        transcribe.WHISPER_TEMPERATURE_BASE = 0.2
        return "primary temperature ladder from 0.2 (sampling, not beam search)"

    if label == "C":
        shipped = transcribe.rescue_decode_kwargs

        def rescue_keeping_context(primary_kwargs):
            kwargs = shipped(primary_kwargs)
            kwargs["condition_on_previous_text"] = True
            return kwargs

        transcribe.rescue_decode_kwargs = rescue_keeping_context
        return "rescue ladder from 0.2 with previous-text conditioning left ON"

    if label == "D":
        def fidelity_profile(mean_dbfs):
            sample_rate = CURRENT.get("sample_rate")
            bit_rate = CURRENT.get("bit_rate")
            if sample_rate is None and bit_rate is None:
                return transcribe.QUIET_PROFILE, "encoding unknown, using the quiet profile"
            low = ((sample_rate is not None and sample_rate <= FIDELITY_SAMPLE_RATE_HZ)
                   or (bit_rate is not None and bit_rate < FIDELITY_BIT_RATE))
            why = f"{sample_rate} Hz, {bit_rate} bps"
            return ((transcribe.QUIET_PROFILE, f"low fidelity ({why})") if low
                    else (transcribe.LOUD_PROFILE, f"full fidelity ({why})"))

        transcribe.choose_vad_profile = fidelity_profile
        return (f"VAD profile keyed on sample rate <= {FIDELITY_SAMPLE_RATE_HZ} Hz or "
                f"bit rate < {FIDELITY_BIT_RATE} bps, not on level")

    raise SystemExit(f"unknown config {label}")


def summarise(result, wall):
    segments = result.get("segments") or []
    flags = {}
    for entry in result.get("flagged_segments") or []:
        for flag in entry.get("flags") or []:
            flags[flag] = flags.get(flag, 0) + 1
    return {
        "words": len((result.get("transcription") or "").split()),
        "segments": len(segments),
        "anomaly_count": result.get("anomaly_count"),
        "anomaly_windows": result.get("anomaly_windows"),
        "primary_anomaly_windows": result.get("primary_anomaly_windows"),
        "flagged_segments": len(result.get("flagged_segments") or []),
        "flag_counts": flags,
        "rescue_attempted": result.get("rescue_attempted"),
        "rescue_selected": result.get("rescue_selected"),
        "mean_logprob": result.get("mean_logprob"),
        "speech_seconds": result.get("speech_seconds"),
        "duration_sec": result.get("duration_sec"),
        "mean_dbfs": result.get("mean_dbfs"),
        "vad_profile": result.get("vad_profile"),
        "vad_threshold": result.get("vad_threshold"),
        "uncovered_s": result.get("uncovered_s"),
        "uncovered_max_gap_s": result.get("uncovered_max_gap_s"),
        "wall_s": round(wall, 2),
        "transcription": result.get("transcription") or "",
    }


def load(path):
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    return {"config": None, "note": None, "files": {}}


def save(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--levels", required=True)
    parser.add_argument("--config", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--all-files", action="store_true",
                        help="ignore the per-config subset and decode every file")
    args = parser.parse_args(argv)

    import transcribe

    note = apply_config(transcribe, args.config)
    print(f"config {args.config}: {note}", flush=True)

    with open(args.file_list) as handle:
        doc = json.load(handle)
    names = (([r["file"] for r in doc["files"]]) if args.all_files
             else doc["config_subsets"][args.config])
    levels = read_levels(args.levels)

    records = load(args.out)
    records["config"] = args.config
    records["note"] = note
    for index, name in enumerate(names, 1):
        if name in records["files"]:
            print(f"[{index}/{len(names)}] skip {name}", flush=True)
            continue
        CURRENT.clear()
        CURRENT.update(levels.get(name) or {})
        started = time.time()
        try:
            result = transcribe.transcribe_audio(
                os.path.join(args.audio_dir, name), str(uuid.uuid4()))
            entry = summarise(result, time.time() - started)
        except Exception as exc:  # one bad file must not end the pass
            entry = {"error": f"{type(exc).__name__}: {exc}"}
        records["files"][name] = entry
        save(args.out, records)
        print(f"[{index}/{len(names)}] {name} words={entry.get('words')} "
              f"segs={entry.get('segments')} profile={entry.get('vad_profile')} "
              f"windows={entry.get('anomaly_windows')} "
              f"rescue={entry.get('rescue_attempted')}/{entry.get('rescue_selected')} "
              f"wall={entry.get('wall_s')}s", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
