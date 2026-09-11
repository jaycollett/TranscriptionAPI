"""Choose the ~25 recordings for the fidelity experiment, and label what each is for.

The experiment compares four decode configurations, and a file earns its place by being
able to discriminate between them:

- **scripture**: a read-aloud passage with a public-domain reference, which is the only
  ground truth this corpus has.
- **bad**: one of the files with independent evidence of a bad transcript under 0.6.0
  (word rate over speech under 1.5, an uncovered gap of 20 s or more, or more than 5
  percent short of its legacy transcript).
- **fragmented**: the top of the segments-per-minute distribution, which is what the
  profile selector is supposed to fix.
- **selector_split**: a file where the shipped level-keyed selector and the proposed
  fidelity-keyed selector choose *different* profiles. Only these can move config D;
  everywhere else D is A by construction, so running D there would burn GPU time to
  reproduce a number we already have.
- **control**: healthy files spread over duration, level, sample rate and era, so a
  regression anywhere shows up.

The output is a file list plus, for each config, the subset it actually needs to run.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The shipped selector.
LEVEL_CUTOVER_DBFS = -26.0
# The proposed one. Both thresholds come from the rc2 fragmentation analysis: 22.05 kHz
# files run a median 13.2 segments per minute against 9.1 at 48 kHz, and of the twenty
# most fragmented files 9 are 22.05 kHz and 8 are under 64 kbps against only 3 below
# -26 dBFS. Level is not merely weaker, it is non-monotonic (both the quietest and the
# loudest band fragment, the middle does not), so no one-sided level threshold can
# express the shape at any value and the level term is dropped rather than re-tuned.
FIDELITY_SAMPLE_RATE_HZ = 22050
FIDELITY_BIT_RATE = 64000


def level_profile(mean_dbfs):
    if mean_dbfs is None:
        return "quiet"
    return "loud" if mean_dbfs >= LEVEL_CUTOVER_DBFS else "quiet"


def fidelity_profile(sample_rate, bit_rate):
    """Low fidelity takes the gentle profile; everything else takes the loud one."""
    if sample_rate is None and bit_rate is None:
        return "quiet"
    if (sample_rate is not None and sample_rate <= FIDELITY_SAMPLE_RATE_HZ) or \
       (bit_rate is not None and bit_rate < FIDELITY_BIT_RATE):
        return "quiet"
    return "loud"


def read_levels(path):
    out = {}
    with open(path) as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 4:
                continue
            name, mean, _peak, meta = row[0], row[1], row[2], row[3]
            parts = meta.split(",")
            try:
                sample_rate = int(parts[1])
            except (IndexError, ValueError):
                sample_rate = None
            try:
                bit_rate = int(parts[3])
            except (IndexError, ValueError):
                bit_rate = None
            try:
                mean_dbfs = float(mean)
            except ValueError:
                mean_dbfs = None
            out[name] = {"mean_dbfs": mean_dbfs, "codec": parts[0] if parts else None,
                         "sample_rate": sample_rate,
                         "channels": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None,
                         "bit_rate": bit_rate}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--passages", required=True)
    parser.add_argument("--target", type=int, default=25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    levels = read_levels(args.levels)
    candidate = json.load(open(args.candidate))["files"]
    rows = {r["file"]: r for r in json.load(open(args.analysis))["rows"]}
    passages = json.load(open(args.passages))
    passage_files = {p["file"] for p in passages}

    info = {}
    for name, entry in candidate.items():
        if entry.get("error"):
            continue
        meta = levels.get(name, {})
        duration = entry.get("duration_sec") or 0
        segments = entry.get("segments") or 0
        speech = entry.get("speech_seconds") or 0
        words = entry.get("words") or 0
        legacy = (rows.get(name) or {}).get("legacy_words")
        reasons = []
        if speech and words / speech < 1.5:
            reasons.append("word rate")
        if (entry.get("uncovered_max_gap_s") or 0) >= 20.0:
            reasons.append("uncovered gap")
        if legacy and (words - legacy) / legacy < -0.05:
            reasons.append("short vs legacy")
        info[name] = {
            "duration_sec": duration,
            "segments_per_min": round(segments / (duration / 60.0), 2) if duration else None,
            "mean_dbfs": meta.get("mean_dbfs"),
            "sample_rate": meta.get("sample_rate"),
            "bit_rate": meta.get("bit_rate"),
            "words": words,
            "legacy_words": legacy,
            "bad_reasons": reasons,
            "level_profile": level_profile(meta.get("mean_dbfs")),
            "fidelity_profile": fidelity_profile(meta.get("sample_rate"), meta.get("bit_rate")),
            "rescue_attempted": entry.get("rescue_attempted"),
        }
        info[name]["selector_split"] = (
            info[name]["level_profile"] != info[name]["fidelity_profile"])

    # tcf.20150424 errored in the rc2 run so it has no rc2 transcript, but it decodes
    # under 0.6.0 and it is the one file no signal sees. It must be in.
    chosen = {}

    def add(name, tag):
        if name in info:
            chosen.setdefault(name, []).append(tag)

    for name in sorted(passage_files):
        add(name, "scripture")
    for name, rec in sorted(info.items()):
        if rec["bad_reasons"]:
            add(name, "bad")
    ranked = sorted((r["segments_per_min"] or 0, n) for n, r in info.items())
    for _, name in ranked[::-1][:6]:
        add(name, "fragmented")
    splits = sorted(n for n, r in info.items() if r["selector_split"])
    # Take splits in both directions so D is tested where it makes a file gentler and
    # where it makes one harsher.
    to_quiet = [n for n in splits if info[n]["fidelity_profile"] == "quiet"]
    to_loud = [n for n in splits if info[n]["fidelity_profile"] == "loud"]
    for name in to_quiet[:5] + to_loud[:3]:
        add(name, "selector_split")

    # Controls: healthy files, spread over the duration and level range, filling to the
    # target. Sorted by a spread key so the fill is deterministic and not clustered.
    healthy = [n for n, r in sorted(info.items())
               if not r["bad_reasons"] and n not in chosen]
    healthy.sort(key=lambda n: (info[n]["duration_sec"], n))
    if healthy:
        need = max(0, args.target - len(chosen))
        step = max(1, len(healthy) // max(1, need))
        for name in healthy[::step][:need]:
            add(name, "control")

    out = {
        "thresholds": {
            "level_cutover_dbfs": LEVEL_CUTOVER_DBFS,
            "fidelity_sample_rate_hz": FIDELITY_SAMPLE_RATE_HZ,
            "fidelity_bit_rate": FIDELITY_BIT_RATE,
        },
        "files": [],
    }
    for name in sorted(chosen):
        rec = dict(info[name])
        rec["file"] = name
        rec["tags"] = sorted(set(chosen[name]))
        out["files"].append(rec)
    out["config_subsets"] = {
        # A and B decode everything: A is the control and B is the change under test,
        # and the A-vs-B agreement check needs both on every file.
        "A": [r["file"] for r in out["files"]],
        "B": [r["file"] for r in out["files"]],
        # C differs from A only in the rescue pass, and the rescue's trigger is computed
        # from the primary, which C shares with A bit for bit. On a file where A runs no
        # rescue, C is A.
        "C": [r["file"] for r in out["files"] if info[r["file"]]["rescue_attempted"]],
        # D differs from A only where the two selectors disagree.
        "D": [r["file"] for r in out["files"] if info[r["file"]]["selector_split"]],
    }
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)

    print(f"{len(out['files'])} files")
    for name, tags in sorted(chosen.items()):
        rec = info[name]
        print(f"  {name:34s} {','.join(sorted(set(tags))):28s} "
              f"{rec['duration_sec']:7.0f}s {str(rec['mean_dbfs']):>6s}dB "
              f"{str(rec['sample_rate']):>5s}Hz {str(rec['bit_rate']):>7s}bps "
              f"seg/min={str(rec['segments_per_min']):>6s} "
              f"{rec['level_profile']}->{rec['fidelity_profile']}"
              f"{' SPLIT' if rec['selector_split'] else ''}")
    print()
    for label, subset in out["config_subsets"].items():
        print(f"  config {label}: {len(subset)} decodes")
    print(f"  total decodes: {sum(len(s) for s in out['config_subsets'].values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
