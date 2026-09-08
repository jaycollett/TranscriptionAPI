"""Stratum definitions and the input readers shared by selection and analysis.

Three inputs describe the 655-file reference archive:

- `durations.csv`   `seconds,bytes,filename` from ffprobe.
- `levels.csv`      `filename<TAB>mean_dbfs<TAB>max_dbfs<TAB>codec,sample_rate,channels,bit_rate`
                    from `ffmpeg -af volumedetect` plus ffprobe, measured over the whole file.
- `legacy_baseline.json`  per-file aggregates of the transcript the legacy production
                    code stored in the orchestrator database. Aggregates only: the
                    transcript text never leaves the GPU host.

Everything else in this package works from these dictionaries, so the pure functions
below are what the unit tests exercise.
"""

import csv
import json

# Duration buckets, seconds. The archive is 221 s to 5095 s with a median of 1322 s.
DURATION_BUCKETS = (
    ("short", 0.0, 900.0),
    ("medium", 900.0, 1800.0),
    ("long", 1800.0, 2700.0),
    ("very_long", 2700.0, float("inf")),
)

# Mean-level buckets, dBFS. The -26 edge is the 0.6.0 VAD cutover: at or above it the
# release uses VAD threshold 0.5, below it 0.35. The buckets straddling the cutover are
# deliberately narrow so the sweep can move the edge on measured data.
LEVEL_BUCKETS = (
    ("very_quiet", float("-inf"), -28.0),
    ("quiet", -28.0, -26.0),
    ("cutover_low", -26.0, -24.0),
    ("cutover_high", -24.0, -22.0),
    ("normal", -22.0, -19.0),
    ("loud", -19.0, float("inf")),
)

# Bit-rate buckets, bits per second.
BITRATE_BUCKETS = (
    ("very_low", 0.0, 50_000.0),
    ("low", 50_000.0, 80_000.0),
    ("medium", 80_000.0, 120_000.0),
    ("high", 120_000.0, float("inf")),
)

# The 0.6.0 level-aware VAD rule, so the analyzer can report which branch each file took
# without importing the service.
VAD_CUTOVER_DBFS = -26.0
VAD_THRESHOLD_LOUD = 0.5
VAD_THRESHOLD_QUIET = 0.35

# Legacy code eras, keyed by the orchestrator's GenerateEnglishTranscription completion
# time. Boundaries are the release tag dates, so each era names the code that produced
# the stored transcript.
ERAS = (
    ("E1_0.1.5", "", "2025-03-20"),
    ("E2_0.2.x", "2025-03-20", "2026-06-28"),
    ("E3_0.3.x", "2026-06-28", "2026-09-07"),
    ("E4_0.5.x", "2026-09-07", "9999"),
)

ERA_NOTES = {
    "E1_0.1.5": "single model.transcribe(beam_size=7, best_of=7), no prompt, no denoise",
    "E2_0.2.x": "five passes, scalar temperature (fallback disabled), single-speaker prompt",
    "E3_0.3.x": "five passes, scalar temperature, prompt, always-on denoise",
    "E4_0.5.x": "0.4.0-0.5.4, all four decode defects fixed; not legacy output",
}

# Recordings with more than one voice: classes, retreats, men's breakfasts and Q&A.
MULTI_VOICE_MARKERS = (
    "women_retreat",
    "mens_breakfast",
    "mens_fast",
    "mens_",
    "formation_class",
    "class",
    "q_and_a",
    "qanda",
)


def bucket_for(value, buckets, default="unknown"):
    """Name of the half-open bucket [lo, hi) containing `value`."""
    if value is None:
        return default
    for name, lo, hi in buckets:
        if lo <= value < hi:
            return name
    return default


def era_for(finished_at):
    """Legacy code era for an orchestrator completion timestamp."""
    if not finished_at:
        return "unknown"
    for name, lo, hi in ERAS:
        if lo <= finished_at < hi:
            return name
    return "unknown"


def vad_branch(mean_dbfs):
    """The VAD threshold 0.6.0 picks for a file at this mean level."""
    if mean_dbfs is None:
        return None
    return VAD_THRESHOLD_LOUD if mean_dbfs >= VAD_CUTOVER_DBFS else VAD_THRESHOLD_QUIET


def is_multi_voice(filename):
    """True for recordings whose name marks them as class, retreat or Q&A material."""
    lowered = filename.lower()
    return any(marker in lowered for marker in MULTI_VOICE_MARKERS)


def read_durations(path):
    """`{filename: seconds}` from the ffprobe duration CSV."""
    out = {}
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 3 and row[0]:
                out[row[2]] = float(row[0])
    return out


def read_levels(path):
    """`{filename: {mean_dbfs, max_dbfs, codec, sample_rate, channels, bit_rate}}`."""
    out = {}
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            name, mean, peak, props = parts[0], parts[1], parts[2], parts[3].split(",")
            entry = {
                "mean_dbfs": None if mean == "NA" else float(mean),
                "max_dbfs": None if peak == "NA" else float(peak),
                "codec": props[0] if props else None,
                "sample_rate": int(props[1]) if len(props) > 1 and props[1].isdigit() else None,
                "channels": int(props[2]) if len(props) > 2 and props[2].isdigit() else None,
                "bit_rate": int(props[3]) if len(props) > 3 and props[3].isdigit() else None,
            }
            out[name] = entry
    return out


def read_baseline(path):
    """The `baseline` mapping of legacy_baseline.json."""
    with open(path) as handle:
        return json.load(handle)["baseline"]


def describe(filename, baseline_row, level_row, duration):
    """Every stratum label for one file, as one flat dict."""
    mean_dbfs = (level_row or {}).get("mean_dbfs")
    return {
        "file": filename,
        "duration_s": duration,
        "duration_bucket": bucket_for(duration, DURATION_BUCKETS),
        "mean_dbfs": mean_dbfs,
        "max_dbfs": (level_row or {}).get("max_dbfs"),
        "level_bucket": bucket_for(mean_dbfs, LEVEL_BUCKETS),
        "vad_threshold": vad_branch(mean_dbfs),
        "sample_rate": (level_row or {}).get("sample_rate"),
        "channels": (level_row or {}).get("channels"),
        "bit_rate": (level_row or {}).get("bit_rate"),
        "bitrate_bucket": bucket_for((level_row or {}).get("bit_rate"), BITRATE_BUCKETS),
        "multi_voice": is_multi_voice(filename),
        "legacy_era": era_for((baseline_row or {}).get("transcription_finished_at")),
        "legacy_words": (baseline_row or {}).get("words"),
        "legacy_wps": (baseline_row or {}).get("wps"),
        "sermon_guid": (baseline_row or {}).get("sermon_guid"),
    }
