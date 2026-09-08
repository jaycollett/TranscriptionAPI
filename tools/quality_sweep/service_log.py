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
# `+` is in the value class for rescue_triggers=window+gap, which 0.6.1 logs on the
# completion line so the firing mix of the two triggers can be read off a sweep rather
# than guessed at. Without it the value silently truncates to the first trigger.
RE_KV = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>-?[\w.+]+)")
RE_MFA_WORDS = re.compile(r"(?P<mfa_words>\d+) MFA words over (?P<mfa_segments>\d+) segments")
RE_WHISPER_FALLBACK = re.compile(r"(?P<whisper_fallback_segments>\d+) segments on Whisper word timings")

# The level line names the profile the decode actually used, rather than the one the -26
# rule predicts, so the level table can report what happened instead of what should have.
RE_AUDIO_LEVEL = re.compile(
    r"Audio level for (?P<guid>" + GUID + r"): (?P<level>-?[\d.]+) dBFS; "
    r"VAD profile '(?P<profile>\w+)'"
)

# The per-pass line carries the service's own coverage of its VAD speech, which is the
# number the 10 percent tolerance is measured against.
RE_PASS = re.compile(
    r"(?P<pass_name>primary|rescue) pass for (?P<guid>" + GUID + r") in (?P<seconds>[\d.]+)s: "
    r"(?P<words>\d+) words, (?P<segments>\d+) segments covering (?P<covered>[\d.]+)s of "
    r"(?P<speech>[\d.]+)s VAD speech"
)
RE_WINDOWS_OF = re.compile(r"anomaly_windows=(?P<windows>\d+) of (?P<windows_total>\d+)")

# The post-transcription quality gate. A failure resets the job to pending and burns a
# whole re-decode, so the reason and the attempt number are worth counting: a gate that
# fails on a deterministic decode can never be satisfied by retrying and ends in
# quarantine after three full decodes.
RE_GATE_FAIL = re.compile(
    r"Transcription (?P<guid>" + GUID + r") failed \((?P<reason>.+)\)\. "
    r"Resetting to 'pending' \(attempt (?P<attempt>\d+) of (?P<max_attempts>\d+)\)"
)
RE_QUARANTINE = re.compile(
    r"(?P<guid>" + GUID + r").{0,120}?quarantin", re.IGNORECASE
)
RE_RESCUE_DISCARD = re.compile(
    r"Discarding the rescue pass: (?P<rescue_words>\d+) words against the primary "
    r"pass's (?P<primary_words>\d+) loses (?P<lost>\d+)"
)

# The seam de-duplication logs every trim as
#   Boundary dedupe for <guid>: dropped 4 words repeated across the seam at 123.45s: 'phrase'
# Verified against the rc2 image's own source before the run. The loose pattern below is
# kept as a fallback so a wording change is noticed rather than silently dropping trims.
RE_TRIM = re.compile(
    r"Boundary dedupe(?: for (?P<guid>" + GUID + r"))?: dropped (?P<overlap>\d+) words "
    r"repeated across the seam at (?P<at>[\d.]+)s: (?P<removed>.*)"
)
RE_TRIM_LINE = re.compile(
    r"(boundary dedupe|\btrim\b|\btrimmed\b|dedup|de-dup|duplicate overlap|\bseam\b)",
    re.IGNORECASE,
)
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
                "passes": [],
                "gate_failures": [],
                "rescue_discards": [],
                "quarantined": False,
                "fields": {},
                "alignment": {},
                "audio_level_dbfs": None,
                "vad_profile": None,
            }
        return jobs[guid]

    for raw in lines:
        line = raw.rstrip("\n")

        gate = RE_GATE_FAIL.search(line)
        if gate:
            current = gate.group("guid")
            job(current)["gate_failures"].append({
                "reason": gate.group("reason").strip(),
                "attempt": int(gate.group("attempt")),
                "max_attempts": int(gate.group("max_attempts")),
            })
            continue

        discard = RE_RESCUE_DISCARD.search(line)
        if discard and current:
            job(current)["rescue_discards"].append({
                "rescue_words": int(discard.group("rescue_words")),
                "primary_words": int(discard.group("primary_words")),
                "lost": int(discard.group("lost")),
            })
            continue

        quarantine = RE_QUARANTINE.search(line)
        if quarantine:
            current = quarantine.group("guid")
            job(current)["quarantined"] = True
            continue

        level = RE_AUDIO_LEVEL.search(line)
        if level:
            current = level.group("guid")
            record = job(current)
            record["audio_level_dbfs"] = float(level.group("level"))
            record["vad_profile"] = level.group("profile")
            continue

        run = RE_PASS.search(line)
        if run:
            current = run.group("guid")
            record = job(current)
            entry = {
                "pass": run.group("pass_name"),
                "seconds": float(run.group("seconds")),
                "words": int(run.group("words")),
                "segments": int(run.group("segments")),
                "covered_s": float(run.group("covered")),
                "vad_speech_s": float(run.group("speech")),
            }
            entry["uncovered_s"] = round(entry["vad_speech_s"] - entry["covered_s"], 3)
            entry["uncovered_fraction"] = (
                round(entry["uncovered_s"] / entry["vad_speech_s"], 5)
                if entry["vad_speech_s"] else None
            )
            windows = RE_WINDOWS_OF.search(line)
            if windows:
                entry["anomaly_windows"] = int(windows.group("windows"))
                entry["windows_total"] = int(windows.group("windows_total"))
            for key, value in RE_KV.findall(line):
                entry.setdefault(key, _coerce(key, value))
            record["passes"].append(entry)
            continue

        trim = RE_TRIM.search(line)
        if trim:
            current = trim.group("guid") or current
            if current:
                quoted = [a or b for a, b in RE_QUOTED.findall(trim.group("removed"))]
                job(current)["trims"].append({
                    "overlap_words": int(trim.group("overlap")),
                    "at_s": float(trim.group("at")),
                    "removed": quoted[0] if quoted else trim.group("removed").strip(),
                    "quoted": quoted,
                    "emptied": "segment emptied" in line,
                    "line": line.strip(),
                    "matched": "exact",
                })
            continue

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
                "at_s": None,
                "removed": quoted[0] if quoted else None,
                "quoted": quoted,
                "emptied": "segment emptied" in line,
                "line": line.strip(),
                "matched": "loose",
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
        # A file whose rescue pass ran logs the same trim once per decode, and the
        # selection log line can repeat it again, so the raw line count multiplies by the
        # number of passes. Only the distinct (timestamp, phrase) pairs correspond to
        # text actually missing from the transcript.
        seen = set()
        distinct = []
        for trim in record["trims"]:
            key = (trim.get("at_s"), trim.get("overlap_words"), trim.get("removed"))
            if key in seen:
                continue
            seen.add(key)
            distinct.append(trim)
        record["trims_logged"] = len(record["trims"])
        record["trims"] = distinct
        record["trim_count"] = len(distinct)
        record["trimmed_words"] = sum(
            t["overlap_words"] for t in distinct if t["overlap_words"]
        )
        # The first decode is the primary; a second entry means the rescue pass ran.
        record["vad_removed_first_s"] = (
            record["vad_removed_s"][0] if record["vad_removed_s"] else None
        )
        record["decode_duration_s"] = (
            record["decode_durations_s"][0] if record["decode_durations_s"] else None
        )
        record["speech_seconds"] = speech_seconds(record)
        primary = next((p for p in record["passes"] if p["pass"] == "primary"), None)
        record["primary_pass"] = primary
        record["rescue_pass"] = next(
            (p for p in record["passes"] if p["pass"] == "rescue"), None
        )
        record["service_uncovered_fraction"] = (
            primary["uncovered_fraction"] if primary else None
        )
        record["loose_trim_matches"] = sum(
            1 for t in record["trims"] if t.get("matched") == "loose"
        )
        record["gate_failure_count"] = len(record["gate_failures"])
        record["gate_failure_reasons"] = sorted({g["reason"] for g in record["gate_failures"]})
        record["decode_attempts"] = len([p for p in record["passes"] if p["pass"] == "primary"])
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
