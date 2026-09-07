"""Harvest per-job facts from the sweep service's container log.

Some of what the sweep needs to measure is not in the `/status` payload: the seconds VAD
removed, the phrases the seam de-duplication trimmed, and whatever counters the alignment
step logs. All of it is in the container's own log, so `run_sweep.sh` captures that log
for the whole run and this module turns it into one record per GUID.

The parsing is deliberately loose in one specific way. The service's completion and
alignment lines are already written as `key=value` pairs, so this module scans every such
pair generically instead of listing the keys it knows, coercing each to a bool, an int or
a float where it looks like one. A corrected image that starts reporting `speech_seconds`,
a VAD profile, a clamp count or a span ratio is therefore picked up with no change here.

The service processes one job at a time and the runner submits the next file only after
the previous one is terminal, so a line that carries no GUID of its own belongs to the
most recently mentioned GUID. That is how the faster-whisper VAD lines, which know
nothing about jobs, are attributed.

    python3 tools/quality_sweep/service_log.py --log /home/jay/sweep/run/service.log \
        --out-json /home/jay/sweep/service_log.json
"""

import argparse
import collections
import json
import re

GUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

RE_GUID = re.compile(GUID)
RE_RECEIVED = re.compile(
    r"File (?P<file>\S+) received and saved as .*? with GUID (?P<guid>" + GUID + r")"
)
RE_COMPLETED = re.compile(
    r"Transcription completed for (?P<file>\S+) \(GUID: (?P<guid>" + GUID + r")\):"
    r"\s*(?P<words>\d+) words in (?P<duration>[\d.]+)s"
)
RE_ALIGNMENT = re.compile(r"Alignment for (?P<guid>" + GUID + r"):(?P<rest>.*)")
RE_VAD_REMOVED = re.compile(r"VAD filter removed (?P<hh>\d+):(?P<mm>\d+)\.(?P<ms>\d+) of audio")
RE_PROCESSING_DURATION = re.compile(r"Processing audio with duration (?P<hh>\d+):(?P<mm>\d+)\.(?P<ms>\d+)")
RE_KV = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>-?[\w.]+)")
RE_MFA_WORDS = re.compile(r"(?P<mfa_words>\d+) MFA words over (?P<mfa_segments>\d+) segments")
RE_WHISPER_FALLBACK = re.compile(r"(?P<whisper_fallback_segments>\d+) segments on Whisper word timings")

# Any line about the seam de-duplication. The corrected image logs each trim at INFO with
# the GUID; the shape of that line is not pinned here on purpose, so a wording change does
# not silently drop the data. Whatever matches is kept whole, and the structured fields are
# filled in when the richer pattern also matches.
RE_TRIM_LINE = re.compile(r"\b(trim|trimmed|dedup|de-dup|duplicate overlap|seam)\b", re.IGNORECASE)
RE_TRIM_DETAIL = re.compile(
    r"(?P<overlap>\d+)[ -]word", re.IGNORECASE
)
RE_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def _seconds(match, prefix=""):
    """faster-whisper prints MM:SS.mmm; return it as seconds."""
    return (
        int(match.group(prefix + "hh")) * 60
        + int(match.group(prefix + "mm"))
        + float("0." + match.group(prefix + "ms"))
    )


def _coerce(key, value):
    """Turn a logged value into a bool, an int, a float or the string it already was."""
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    if value in ("None", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse(lines):
    """`{guid: record}` for every job the log mentions."""
    jobs = collections.OrderedDict()
    current = None

    def job(guid):
        if guid not in jobs:
            jobs[guid] = {
                "guid": guid,
                "file": None,
                "vad_removed_s": [],
                "decode_durations_s": [],
                "trims": [],
                "fields": {},
                "alignment": {},
            }
        return jobs[guid]

    for raw in lines:
        line = raw.rstrip("\n")

        received = RE_RECEIVED.search(line)
        if received:
            current = received.group("guid")
            job(current)["file"] = received.group("file")
            continue

        completed = RE_COMPLETED.search(line)
        if completed:
            current = completed.group("guid")
            record = job(current)
            record["file"] = completed.group("file")
            record["words_logged"] = int(completed.group("words"))
            record["duration_logged_s"] = float(completed.group("duration"))
            for key, value in RE_KV.findall(line):
                record["fields"][key] = _coerce(key, value)
            continue

        alignment = RE_ALIGNMENT.search(line)
        if alignment:
            current = alignment.group("guid")
            record = job(current)
            rest = alignment.group("rest")
            for pattern in (RE_MFA_WORDS, RE_WHISPER_FALLBACK):
                found = pattern.search(rest)
                if found:
                    record["alignment"].update(
                        {k: int(v) for k, v in found.groupdict().items()}
                    )
            for key, value in RE_KV.findall(rest):
                record["alignment"][key] = _coerce(key, value)
            continue

        found_guid = RE_GUID.search(line)
        if found_guid:
            current = found_guid.group(0)

        if RE_TRIM_LINE.search(line) and current:
            record = job(current)
            detail = RE_TRIM_DETAIL.search(line)
            quoted = [a or b for a, b in RE_QUOTED.findall(line)]
            record["trims"].append({
                "overlap_words": int(detail.group("overlap")) if detail else None,
                "quoted": quoted,
                "line": line.strip(),
            })
            continue

        removed = RE_VAD_REMOVED.search(line)
        if removed and current:
            job(current)["vad_removed_s"].append(round(_seconds(removed), 3))
            continue

        processing = RE_PROCESSING_DURATION.search(line)
        if processing and current:
            job(current)["decode_durations_s"].append(round(_seconds(processing), 3))

    for record in jobs.values():
        record["trim_count"] = len(record["trims"])
        record["trimmed_words"] = sum(
            t["overlap_words"] for t in record["trims"] if t["overlap_words"]
        )
        # The first decode is the primary; a second entry means the rescue pass ran.
        record["vad_removed_first_s"] = (
            record["vad_removed_s"][0] if record["vad_removed_s"] else None
        )
        record["decode_duration_s"] = (
            record["decode_durations_s"][0] if record["decode_durations_s"] else None
        )
        record["speech_seconds"] = speech_seconds(record)
    return jobs


def speech_seconds(record):
    """Speech seconds for a job: the service's own number if it logs one, else duration minus VAD."""
    logged = record["fields"].get("speech_seconds")
    if isinstance(logged, (int, float)):
        return float(logged)
    duration = record.get("duration_logged_s") or record.get("decode_duration_s")
    removed = record.get("vad_removed_first_s")
    if duration is None or removed is None:
        return None
    return round(duration - removed, 3)


def by_file(jobs):
    """`{filename: record}`, keeping the last job for a file if it was submitted twice."""
    out = {}
    for record in jobs.values():
        if record.get("file"):
            out[record["file"]] = record
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args(argv)

    with open(args.log, errors="replace") as handle:
        jobs = parse(handle)
    with open(args.out_json, "w") as handle:
        json.dump({"jobs": list(jobs.values())}, handle, indent=1)
        handle.write("\n")

    trims = sum(r["trim_count"] for r in jobs.values())
    with_speech = sum(1 for r in jobs.values() if r["speech_seconds"] is not None)
    print(f"{len(jobs)} jobs, {trims} de-duplication trims, "
          f"{with_speech} with speech seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
