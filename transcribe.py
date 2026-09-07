"""Single-pass Faster-Whisper decode with per-segment diagnostics.

0.6.0 replaced the five-pass decode and its duration-weighted confidence rule with
one deterministic beam-5 pass (harness config C1) plus a level-aware VAD (C4) and a
per-segment anomaly score (C2/C7). The measurements behind every constant here are in
docs/QUALITY_PROPOSAL.md and tools/quality_harness/results/2026-09-07/summary.md.
"""

import json
import logging
import math
import os
import re
import subprocess
import threading
import time
import warnings
from collections import Counter

import torch  # type: ignore
from faster_whisper import WhisperModel  # type: ignore
from pydub import AudioSegment  # type: ignore
from pydub.utils import mediainfo  # type: ignore

from textnorm import norm_words

# Set audio file location from environment variable or default to /tmp/audio_files
upload_folder = os.getenv("UPLOAD_FOLDER", "/tmp/audio_files")

# Configure logging to include timestamp, log level, logger name, and message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('transcribe')

# Suppress specific warning from Faster-Whisper about FP16 usage
warnings.filterwarnings("ignore", category=UserWarning, module="faster_whisper")

# Determine device type (GPU if available, else CPU) and set compute precision
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"

logger.info(f"Running Faster-Whisper on {device.upper()} with compute type {compute_type}")

# Sentinel variable to ensure the model is loaded only once and reused
_whisper_model = None
_model_lock = threading.Lock()


# ---------------------------------------------------------------------------------
# Tunables. Every one of these is an environment variable so the decode can be
# retuned on a running container without a rebuild; the defaults are the measured
# 0.6.0 configuration and changing one invalidates the harness baseline it was
# measured against.
# ---------------------------------------------------------------------------------
def _env_float(name, default):
    raw = os.getenv(name)
    if raw in (None, ""):
        return float(default)
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not a number; using the default {default}")
        return float(default)


def _env_int(name, default):
    raw = os.getenv(name)
    if raw in (None, ""):
        return int(default)
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not an integer; using the default {default}")
        return int(default)


# Decode (harness C1). beam_size 5 with best_of 5 and patience 1.0 reproduced the
# five-pass winner at a fifth of the wall time on all six reference files.
WHISPER_BEAM_SIZE = _env_int("WHISPER_BEAM_SIZE", 5)
WHISPER_BEST_OF = _env_int("WHISPER_BEST_OF", 5)
WHISPER_PATIENCE = _env_float("WHISPER_PATIENCE", 1.0)
WHISPER_TEMPERATURE_BASE = _env_float("WHISPER_TEMPERATURE_BASE", 0.0)
# Clamped positive: temperature_ladder walks upward by this, so zero or a negative
# value never terminates and would hang the worker thread on the first job with no
# log line to say why.
WHISPER_TEMPERATURE_STEP = max(0.01, _env_float("WHISPER_TEMPERATURE_STEP", 0.2))
WHISPER_COMPRESSION_RATIO_THRESHOLD = _env_float("WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4)
WHISPER_LOG_PROB_THRESHOLD = _env_float("WHISPER_LOG_PROB_THRESHOLD", -1.0)
WHISPER_NO_SPEECH_THRESHOLD = _env_float("WHISPER_NO_SPEECH_THRESHOLD", 0.6)
WHISPER_PROMPT_RESET_ON_TEMPERATURE = _env_float("WHISPER_PROMPT_RESET_ON_TEMPERATURE", 0.5)
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
# One CUDA worker: the five-pass loop was sequential anyway, so the second worker
# only ever doubled the resident model.
WHISPER_NUM_WORKERS = _env_int("WHISPER_NUM_WORKERS", 1)

# VAD. The level chooses between two whole profiles, not just a threshold. Both were
# measured on the reference set and neither works on the other's material:
#
#   loud  (harness C4): seams drop 3-4x and removed audio stays under 6 percent on the
#         five files at -17 to -24.5 dBFS, and it is the best-aligning decode of any
#         config on the 755 s file.
#   quiet (harness BASE, the C1 profile): on the -28.4 dBFS retreat recording the loud
#         profile produces 1157 segments and loses 3.0 percent of the words, and
#         relaxing only its threshold to 0.35 is not enough. Measured 2026-09-07:
#         threshold 0.35 with the loud profile's 1000 ms silence and 300 ms padding
#         gives 4693 segments and 8407 words; this profile gives 435 segments and 8904
#         words, reproducing C1 exactly. The long minimum silence is what shreds quiet
#         audio, not the threshold and not the hallucination filter.
#
# PROVISIONAL. The cutover sits between the quietest file that worked on the loud
# profile (-24.4 dBFS) and the one that failed (-28.4 dBFS), which is a boundary
# calibrated on exactly two files. The 100-file corpus sweep found 34 files within
# 2 dB of it and the level histogram densest right at the line, so a large share of
# the archive is decided by a threshold that two recordings placed. Left at -26 so
# the sweep can measure both sides; expect it to move, or to be replaced by
# something that does not put a hard step through the middle of the distribution.
VAD_LEVEL_CUTOVER_DBFS = _env_float("VAD_LEVEL_CUTOVER_DBFS", -26.0)
VAD_MIN_SPEECH_MS = _env_int("VAD_MIN_SPEECH_DURATION_MS", 250)

VAD_THRESHOLD = _env_float("VAD_THRESHOLD", 0.5)
VAD_MIN_SILENCE_MS = _env_int("VAD_MIN_SILENCE_DURATION_MS", 1000)
VAD_SPEECH_PAD_MS = _env_int("VAD_SPEECH_PAD_MS", 300)

VAD_THRESHOLD_QUIET = _env_float("VAD_THRESHOLD_QUIET", 0.35)
VAD_MIN_SILENCE_MS_QUIET = _env_int("VAD_MIN_SILENCE_DURATION_MS_QUIET", 300)
VAD_SPEECH_PAD_MS_QUIET = _env_int("VAD_SPEECH_PAD_MS_QUIET", 400)

# Drop text the model produced over silence longer than this. It has to be under the
# 2 x speech_pad_ms of silence the VAD leaves around each chunk or it can never fire,
# which is why the 2.0 the service used to carry was dead weight. Off on the quiet
# profile, whose 400 ms padding puts 800 ms of deliberate silence in every gap, and
# which was measured without it.
WHISPER_HALLUCINATION_SILENCE_THRESHOLD = _env_float("WHISPER_HALLUCINATION_SILENCE_THRESHOLD", 0.5)

# Anomaly score (harness C2) and the loop flagger (C7).
ANOMALY_TEMPERATURE = _env_float("ANOMALY_TEMPERATURE", 0.5)
ANOMALY_NO_SPEECH_PROB = _env_float("ANOMALY_NO_SPEECH_PROB", 0.5)
ANOMALY_WINDOW_SEC = _env_float("ANOMALY_WINDOW_SEC", 60.0)
ANOMALY_WINDOW_MIN_WPS = _env_float("ANOMALY_WINDOW_MIN_WPS", 1.2)
# A window shorter than this is not scored: a 12 s tail is not evidence of collapse.
ANOMALY_WINDOW_MIN_TAIL_SEC = 20.0
# Share of VAD speech a healthy decode is allowed to leave uncovered before any of it
# counts as an omission.
#
# Segment spans never tile VAD speech exactly: the decoder leaves a fraction of a
# second between segments at every breath, and those pauses sum. Measured on the six
# reference files, a healthy decode leaves 0 to 7.1 percent of VAD speech uncovered,
# and on the worst of them (tcf.20240213a) that 101.5 s is 373 sub-second gaps whose
# largest single member is 1.8 s. Charging raw uncovered time therefore reports a
# missing minute on files that are missing nothing, which is what the first
# implementation of this check did: it fired on 2 of 6 healthy files, triggered the
# rescue on both, and on tcf.20240213a the rescue was then selected and published 9
# fewer words at lower agreement than the primary.
#
# 10 percent leaves margin over the measured 7.1 percent worst case while still
# catching the omissions this exists for: a 20 percent omission clears it by 300 s,
# five windows. Calibrated on six files; the sweep should re-measure the healthy
# distribution and set it from that.
ANOMALY_UNCOVERED_TOLERANCE = _env_float("ANOMALY_UNCOVERED_TOLERANCE", 0.10)
LOOP_4GRAM_RATE = _env_float("LOOP_4GRAM_RATE", 0.3)

# Anomaly-triggered rescue pass. The five-pass decode bought redundancy by paying for
# it on every file, including the 99 percent that never needed it; this buys the same
# redundancy only where the primary pass shows evidence of trouble. Worst case is two
# passes, still well under five. The trigger is deliberately far below the quarantine
# gate (ANOMALY_SEGMENTS_MAX / ANOMALY_WINDOWS_MAX in app.py): a second opinion is
# cheap and a requeue is not, so the rescue fires long before the job is at risk.
RESCUE_ENABLED = os.getenv("RESCUE_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
# 2, not 1. At 1 the trigger sits at the minimum representable value, so a single
# flipped segment decides the code path, and the decode is measurably not
# bit-reproducible on long files: two runs of the identical configuration on
# tcf.20240319b gave 3599 and 3598 words with 1 and 0 anomalous segments.
RESCUE_ANOMALY_SEGMENTS = _env_int("RESCUE_ANOMALY_SEGMENTS", 2)
RESCUE_ANOMALY_WINDOWS = _env_int("RESCUE_ANOMALY_WINDOWS", 1)
# The rescue ladder starts above 0.0: the primary pass already decoded this audio at
# that rung and produced the anomalies, so repeating it is the one outcome guaranteed
# not to help.
RESCUE_TEMPERATURE_BASE = _env_float("RESCUE_TEMPERATURE_BASE", 0.2)
# A rescue that clears the anomaly by deleting content has not fixed the transcript,
# it has truncated it. Measured 2026-09-07 on tcf.20240319b with the trigger forced
# on: the primary scored one anomalous segment out of 323 with 3599 words, the rescue
# scored none with 3525, and the anomaly score alone would have published the pass
# with 74 fewer words. That is the same "prefer the shorter transcript" failure the
# retired duration-weighted rule made on the retreat file, arriving through a
# different door, so the rescue has to keep at least this share of the primary's
# words to be eligible at all.
RESCUE_MIN_WORD_RETENTION = _env_float("RESCUE_MIN_WORD_RETENTION", 0.99)
# The ratio alone scales the wrong way. It was calibrated on a 74 word loss at 3599
# words, but the same 1 percent on the 8904 word retreat file permits an 89 word
# loss, which is larger than the two Psalm 91 runs (61 and 58 words) whose loss
# motivated this whole rewrite. Both conditions have to hold.
RESCUE_MAX_WORD_LOSS = _env_int("RESCUE_MAX_WORD_LOSS", 40)

# Boundary de-duplication: the longest and shortest overlap between the tail of one
# segment and the head of the next that is treated as a duplicate.
#
# 4, not 2. A two or three word coincidence across a seam is ordinary English, and at
# 2 this rule reproduced the defect it was written to remove: the segments
# ["The Lord is with you. He is risen.", "He is risen indeed. Alleluia."] published as
# "The Lord is with you. He is risen. indeed. Alleluia.", which is the old regex's
# failure exactly. Every trim is logged so the true-positive rate can be counted from
# production rather than assumed.
BOUNDARY_DEDUPE_MIN_WORDS = _env_int("BOUNDARY_DEDUPE_MIN_WORDS", 4)
BOUNDARY_DEDUPE_MAX_WORDS = _env_int("BOUNDARY_DEDUPE_MAX_WORDS", 5)

# Real-time factor for the client-facing estimate. Measured on the six reference
# files: Whisper about 0.027 and MFA about 0.022 of real time on the 12 GB card, so
# 0.06 leaves headroom for a queue wake and the level measurement.
PROCESSING_REALTIME_FACTOR = _env_float("PROCESSING_REALTIME_FACTOR", 0.06)
PROCESSING_FIXED_OVERHEAD_SEC = _env_int("PROCESSING_FIXED_OVERHEAD_SEC", 60)

_MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


def whisper_model_loaded():
    """True once load_whisper_model() has completed; used by the /health endpoint."""
    return _whisper_model is not None


def estimate_processing_seconds(duration_sec):
    """Estimated wall-clock seconds to transcribe duration_sec of audio.

    One formula shared by /upload (the ETA the client is given) and the
    transcription log line, so the two can never drift apart.

    The 0.5.x PROCESSING_SPEED_FACTOR constant is retired: it was calibrated to the
    five-pass decode, which ran at 0.13-0.14 of real time. The single pass plus
    per-utterance MFA runs at about 0.05, so the old formula over-promised the queue
    by a factor of three.
    """
    return math.ceil(float(duration_sec) * PROCESSING_REALTIME_FACTOR) + PROCESSING_FIXED_OVERHEAD_SEC


def load_whisper_model():
    """Load the Faster-Whisper model once and reuse it."""
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            start_time = time.time()
            logger.info("Loading Faster-Whisper model...")
            model_name = os.environ.get("MODEL", "large-v3-turbo")
            kwargs = {
                "device": device,
                "compute_type": compute_type,
                "cpu_threads": os.cpu_count(),
                "num_workers": WHISPER_NUM_WORKERS,
            }
            # Try to load from local cache first
            local_model_path = "/app/models/whisper"
            if os.path.exists(local_model_path):
                logger.info(f"Loading Whisper model from local cache: {local_model_path}")
                kwargs["download_root"] = local_model_path
            else:
                logger.info("Local Whisper model cache not found, downloading from internet")
            _whisper_model = WhisperModel(model_name, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Faster-Whisper model '{model_name}' loaded successfully in {elapsed:.2f} seconds")
    return _whisper_model


def normalize_timestamp(ts):
    """Convert timestamp to a consistent format (float)"""
    if isinstance(ts, (tuple, list)):
        return float(ts[0])
    return float(ts)


# NOTE (2026-09-07): preprocess_audio_for_transcription() was removed.
# It normalised to -20 dBFS, then measured "noise" as mean(abs(samples))/32768 on the
# NORMALISED signal. That is a crest-factor measurement of the speech itself, not a
# noise floor, so every file in the archive scored 0.046-0.060 against the 0.015
# threshold and the "only if noise is detected" branch was never skipped. It then ran
# noisereduce with no noise profile over material that is ~78% speech, so the profile
# it estimated was largely speech: measured effect was speech-active frames falling
# from 76% to 65% and dynamic range widening by 8 dB, plus a second lossy MP3 encode.
# faster-whisper's own feature extractor normalises the input, so feeding it the
# original file is both simpler and measurably better: the same model/config on the
# untouched 755 s file produces 2.77 words/sec against 0.15-1.24 through this stage.
# Deleted rather than repaired: a correct noise-floor estimate would need non-speech
# frame detection, and doing nothing is already the better answer.

# NOTE (2026-09-07): clean_boundary_duplicates() was removed in 0.6.0.
# It ran a repeated-phrase regex over the transcript string and left the timings
# untouched, so `" ".join(t["text"] for t in timings) == transcription` was false on
# every production job, and a third variant of the word sequence reached MFA. The
# regex also deleted legitimate repetition anywhere in the file, not just at a seam:
# "He is risen. He is risen indeed." became "He is risen.  indeed.", "pray without
# ceasing. Pray without ceasing." became "pray without ceasing. .", and "day by day
# by day" became "day by  day". deduplicate_segment_boundaries() below replaces it
# with a check that only ever looks across a segment seam and that edits the
# segments themselves, so text and timings stay one sequence.


def temperature_ladder(base, step=None):
    """Return the temperature fallback ladder faster-whisper expects.

    faster-whisper only performs temperature fallback when `temperature` is a
    sequence; a scalar is wrapped into a single-element list, which disables the
    fallback entirely. Fallback is the built-in escape from a decode that trips the
    compression-ratio or log-probability thresholds, i.e. exactly the mid-file
    collapse this service was producing. At or above
    WHISPER_PROMPT_RESET_ON_TEMPERATURE faster-whisper also resets the previous-text
    prompt, which breaks any repetition loop already under way.

    The ladder runs from `base` up to 1.0 inclusive; `base` is clamped to [0.0, 1.0].
    """
    if step is None:
        step = WHISPER_TEMPERATURE_STEP
    base = min(max(float(base), 0.0), 1.0)
    steps = []
    t = base
    while t < 1.0 + 1e-9:
        steps.append(round(t, 2))
        t += step
    if not steps or steps[-1] < 1.0:
        steps.append(1.0)
    return tuple(steps)


def get_audio_duration(file_path):
    """Return the duration of the audio file in seconds without modifying it.

    Reads the container header through ffprobe (pydub's mediainfo), which is
    instant and allocates nothing; decoding a 46 minute stereo MP3 to PCM just to
    measure it costs roughly 490 MB. The full decode is kept only as a fallback
    for files whose header carries no duration, and it raises on undecodable
    input so callers can reject the file.
    """
    try:
        info = mediainfo(file_path) or {}
        duration = info.get("duration")
        if duration not in (None, "", "N/A"):
            return float(duration)
        logger.warning(f"ffprobe reported no duration for {file_path}; decoding to measure it")
    except Exception as e:
        logger.warning(f"ffprobe duration lookup failed for {file_path}: {e}; decoding to measure it")
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # Convert milliseconds to seconds


# ---------------------------------------------------------------------------------
# Level-aware VAD
# ---------------------------------------------------------------------------------
def measure_mean_dbfs(file_path, duration_sec=0):
    """Mean RMS level of the file in dBFS, or None if it cannot be measured.

    ffmpeg's volumedetect filter reports exactly the quantity the -26 dBFS cutover
    was derived from, so it is the primary measurement; pydub's dBFS (the same RMS
    over a full decode) is the fallback for a build without the filter. Both decode
    the whole file, which costs a few seconds on a sermon and is why the result is
    measured once per job and passed around rather than recomputed.
    """
    timeout = max(120.0, float(duration_sec or 0) * 0.25)
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-i", file_path,
             "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=timeout,
        )
        # volumedetect writes to stderr; take the last match in case of several streams.
        matches = _MEAN_VOLUME_RE.findall(proc.stderr or "")
        if matches:
            return float(matches[-1])
        logger.warning(f"volumedetect reported no mean_volume for {file_path} (rc={proc.returncode})")
    except Exception as e:
        logger.warning(f"volumedetect failed for {file_path}: {e}")

    try:
        level = AudioSegment.from_file(file_path).dBFS
        if level is not None and level != float("-inf") and not math.isnan(level):
            return float(level)
    except Exception as e:
        logger.warning(f"pydub level measurement failed for {file_path}: {e}")
    return None


LOUD_PROFILE = "loud"
QUIET_PROFILE = "quiet"


def vad_parameters(profile):
    """The VAD settings for the named profile."""
    if profile == QUIET_PROFILE:
        return {
            "threshold": VAD_THRESHOLD_QUIET,
            "min_speech_duration_ms": VAD_MIN_SPEECH_MS,
            "min_silence_duration_ms": VAD_MIN_SILENCE_MS_QUIET,
            "speech_pad_ms": VAD_SPEECH_PAD_MS_QUIET,
        }
    return {
        "threshold": VAD_THRESHOLD,
        "min_speech_duration_ms": VAD_MIN_SPEECH_MS,
        "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
        "speech_pad_ms": VAD_SPEECH_PAD_MS,
    }


def hallucination_threshold(profile):
    """The hallucination filter for the named profile; None disables it."""
    return None if profile == QUIET_PROFILE else WHISPER_HALLUCINATION_SILENCE_THRESHOLD


def choose_vad_profile(mean_dbfs):
    """Pick the VAD profile for a file at `mean_dbfs`; returns (profile, why).

    Quiet material is the failure case and it needs the whole profile, not a softer
    threshold: on the -28.4 dBFS retreat recording the loud profile loses 3.0 percent
    of the words, and the loud profile with only its threshold relaxed to 0.35 loses
    5.6 percent. An unmeasurable level therefore takes the quiet profile, which is
    what 0.5.x used on every file: fragmenting quiet speech loses words, while cutting
    a few extra seams on loud speech does not.
    """
    if mean_dbfs is None:
        return QUIET_PROFILE, "level unknown, using the quiet profile"
    if mean_dbfs >= VAD_LEVEL_CUTOVER_DBFS:
        return LOUD_PROFILE, f"{mean_dbfs:.1f} dBFS at or above the {VAD_LEVEL_CUTOVER_DBFS:.1f} dBFS cutover"
    return QUIET_PROFILE, f"{mean_dbfs:.1f} dBFS below the {VAD_LEVEL_CUTOVER_DBFS:.1f} dBFS cutover"


# ---------------------------------------------------------------------------------
# Per-segment diagnostics, anomaly score and loop flag
# ---------------------------------------------------------------------------------
def serialize_segment(seg):
    """Flatten a faster-whisper Segment into plain JSON-able data.

    Everything downstream (the anomaly score, the boundary de-duplication, MFA word
    matching and the timing fallback) works on these dicts, so nothing after the
    decode holds a reference to the library's objects.
    """
    words = None
    raw_words = getattr(seg, "words", None)
    if raw_words:
        words = []
        for w in raw_words:
            words.append({
                "start": normalize_timestamp(w.start),
                "end": normalize_timestamp(w.end),
                "word": w.word,
                "probability": float(w.probability) if w.probability is not None else None,
            })

    def _opt_float(name):
        value = getattr(seg, name, None)
        return float(value) if value is not None else None

    return {
        "id": getattr(seg, "id", None),
        "seek": getattr(seg, "seek", None),
        "start": normalize_timestamp(seg.start),
        "end": normalize_timestamp(seg.end),
        "text": (seg.text or "").strip(),
        "avg_logprob": _opt_float("avg_logprob"),
        "compression_ratio": _opt_float("compression_ratio"),
        "no_speech_prob": _opt_float("no_speech_prob"),
        "temperature": _opt_float("temperature"),
        "words": words,
    }


def repeated_4gram_rate(text):
    """Share of a segment's 4-grams that occur more than once (harness C7).

    On the six reference files every segment over 0.3 was genuine rhetorical
    repetition, so this marks a segment for review; it never rejects a job.
    """
    words = norm_words(text)
    if len(words) < 4:
        return 0.0
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    counts = Counter(grams)
    return sum(c for c in counts.values() if c > 1) / len(grams)


def segment_flags(segment):
    """Diagnostic flags for one segment: the C2 anomaly reasons plus the C7 loop mark.

    'loop' is deliberately not an anomaly reason. It reads rhetorical repetition as
    often as a decoder loop, so it is surfaced for review and left out of the count
    that can requeue a job.
    """
    flags = []
    temperature = segment.get("temperature")
    if temperature is not None and temperature >= ANOMALY_TEMPERATURE:
        flags.append("temperature")
    compression_ratio = segment.get("compression_ratio")
    if compression_ratio is not None and compression_ratio > WHISPER_COMPRESSION_RATIO_THRESHOLD:
        flags.append("compression_ratio")
    avg_logprob = segment.get("avg_logprob")
    if avg_logprob is not None and avg_logprob < WHISPER_LOG_PROB_THRESHOLD:
        flags.append("avg_logprob")
    no_speech_prob = segment.get("no_speech_prob")
    if (no_speech_prob is not None and no_speech_prob > ANOMALY_NO_SPEECH_PROB
            and segment.get("text", "").strip()):
        flags.append("no_speech")
    if repeated_4gram_rate(segment.get("text", "")) > LOOP_4GRAM_RATE:
        flags.append("loop")
    return flags


ANOMALY_FLAGS = frozenset({"temperature", "compression_ratio", "avg_logprob", "no_speech"})


def annotate_segments(segments):
    """Attach `flags` to every segment; returns (anomaly_count, flagged_segments)."""
    anomaly_count = 0
    flagged = []
    for index, segment in enumerate(segments):
        flags = segment_flags(segment)
        segment["flags"] = flags
        if not flags:
            continue
        flagged.append({
            "index": index,
            "start": round(segment["start"], 3),
            "end": round(segment["end"], 3),
            "flags": flags,
        })
        if ANOMALY_FLAGS.intersection(flags):
            anomaly_count += 1
    return anomaly_count, flagged


def speech_spans(segments):
    """Merge the segments' own spans into non-overlapping speech runs.

    The window check needs a clock that skips silence, or a file with a ten minute
    break before the Q&A reads as a collapse. faster-whisper does not hand back the
    VAD chunks it used, and re-running the VAD would mean decoding the audio a second
    time, so the segments' own spans stand in for them.

    On their own they are not enough, and this is the trap: audio the decoder emitted
    nothing for contributes no span, so an omission shrinks the clock instead of
    showing up as a low-rate window, and the check is blind to the one failure it
    exists for. `low_speech_windows` therefore compares this coverage against the VAD
    speech faster-whisper reports and charges the difference.
    """
    spans = []
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if end <= start:
            continue
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans


class SpeechClock:
    """Maps file time to cumulative speech time given speech spans in seconds."""

    def __init__(self, spans):
        self.spans = sorted((float(s), float(e)) for s, e in spans)
        self.total = sum(e - s for s, e in self.spans)
        self.before = []
        acc = 0.0
        for s, e in self.spans:
            self.before.append(acc)
            acc += e - s

    def speech_time(self, t):
        if not self.spans:
            return t
        for (s, e), before in zip(self.spans, self.before):
            if t < s:
                return before
            if t <= e:
                return before + (t - s)
        return self.total


def flat_words(segments):
    """Every word of a segment list in order, as (start, end, normalised token)."""
    out = []
    for segment in segments:
        for word in segment.get("words") or []:
            for token in norm_words(word["word"]):
                out.append((word["start"], word["end"], token))
    return out


def low_speech_windows(segments, duration_sec, speech_seconds=None):
    """Count 60 s windows of speech carrying under 1.2 words/sec (harness C2).

    Returns (windows, low_count). The whole-file word rate hides a partial collapse:
    at a normal 2.6 words/sec about 62 percent of a file has to vanish before the 1.0
    words/sec floor trips, so a 20-40 percent omission clears the floor, the rescue
    trigger and the quarantine gate. This is what is supposed to catch it.

    Two things are counted. Windows the decoder did produce words for are scored on
    their word rate. Then VAD speech the decoder produced no segment for is charged:
    `speech_seconds` is faster-whisper's `duration_after_vad`, the audio it was
    actually given, so speech no surviving segment covers is speech that went
    missing. Without the second half an omission simply shrinks the clock and is
    invisible.

    Only the share beyond ANOMALY_UNCOVERED_TOLERANCE is charged. Segment spans never
    tile VAD speech exactly, and the sub-second pauses between segments sum to minutes
    over a sermon; charging those reports an omission on a file that is missing
    nothing. See the constant for the measurements.
    """
    spans = speech_spans(segments)
    clock = SpeechClock(spans)
    covered = clock.total

    uncovered_windows = 0
    if speech_seconds:
        uncovered = max(0.0, float(speech_seconds) - covered)
        excess = uncovered - ANOMALY_UNCOVERED_TOLERANCE * float(speech_seconds)
        uncovered_windows = int(max(0.0, excess) // ANOMALY_WINDOW_SEC)

    if covered <= 0:
        return [], uncovered_windows

    counts = Counter()
    for start, end, _ in flat_words(segments):
        midpoint = (start + end) / 2.0
        counts[int(clock.speech_time(midpoint) // ANOMALY_WINDOW_SEC)] += 1
    full_windows = int(covered // ANOMALY_WINDOW_SEC)
    tail = covered - full_windows * ANOMALY_WINDOW_SEC
    windows = [counts[i] / ANOMALY_WINDOW_SEC for i in range(full_windows)]
    if tail >= ANOMALY_WINDOW_MIN_TAIL_SEC:
        windows.append(counts[full_windows] / tail)
    low = sum(1 for w in windows if w < ANOMALY_WINDOW_MIN_WPS)
    return windows, low + uncovered_windows


# ---------------------------------------------------------------------------------
# Boundary de-duplication
# ---------------------------------------------------------------------------------
def _segment_tokens(segment):
    """(word_index, normalised token) for a segment, skipping pure punctuation.

    A word that normalises to several tokens ("well-known") is represented by its
    whole normalised form, so trimming can only ever remove whole Whisper words.
    """
    tokens = []
    words = segment.get("words")
    if words:
        for index, word in enumerate(words):
            normalised = " ".join(norm_words(word["word"]))
            if normalised:
                tokens.append((index, normalised))
    else:
        for index, raw in enumerate(segment.get("text", "").split()):
            normalised = " ".join(norm_words(raw))
            if normalised:
                tokens.append((index, normalised))
    return tokens


def _text_from_words(words):
    """Rebuild segment text from Whisper word tokens, which carry their own spacing."""
    return "".join(w["word"] for w in words).strip()


def deduplicate_segment_boundaries(segments, guid=None):
    """Drop a phrase repeated across a segment seam, editing the segments themselves.

    When the last k words of segment N (k from BOUNDARY_DEDUPE_MAX_WORDS down to
    BOUNDARY_DEDUPE_MIN_WORDS) equal the first k words of segment N+1, ignoring case
    and punctuation, those k words are removed from the start of N+1 and its start
    moves to the first surviving word's timestamp. Nothing inside a segment is ever
    touched, so "He is risen. He is risen indeed." inside one segment survives whole.

    The minimum of 4 is load-bearing rather than conservative. At 2 this rule deleted
    genuine liturgical repetition that happened to straddle a seam, which is the exact
    defect the old regex was removed for. Every trim is logged so the true-positive
    rate can be counted from production instead of assumed.

    Returns a new list; segments emptied by the trim are dropped.
    """
    result = []
    for segment in segments:
        segment = dict(segment)
        if segment.get("words") is not None:
            segment["words"] = [dict(w) for w in segment["words"]]
        if not segment.get("text", "").strip():
            continue
        if result:
            previous = result[-1]
            trim = _boundary_overlap(previous, segment)
            if trim:
                where = f"{segment.get('start', 0.0):.2f}s"
                trimmed, removed = _trim_leading_words(segment, trim)
                logger.info(
                    f"Boundary dedupe{'' if guid is None else f' for {guid}'}: dropped {trim} words "
                    f"repeated across the seam at {where}: {removed!r}"
                    + ("" if trimmed is not None else " (segment emptied and dropped)")
                )
                if trimmed is None:
                    continue
                segment = trimmed
        result.append(segment)
    return result


def _boundary_overlap(previous, segment):
    """Number of leading words of `segment` duplicated at the end of `previous`."""
    previous_tokens = [t for _, t in _segment_tokens(previous)]
    segment_tokens = [t for _, t in _segment_tokens(segment)]
    upper = min(BOUNDARY_DEDUPE_MAX_WORDS, len(previous_tokens), len(segment_tokens))
    # Longest overlap first: "the Lord is good" must not be trimmed as "is good".
    for k in range(upper, BOUNDARY_DEDUPE_MIN_WORDS - 1, -1):
        if previous_tokens[-k:] == segment_tokens[:k]:
            return k
    return 0


def _trim_leading_words(segment, count):
    """Remove `count` leading words from a segment.

    Returns (segment, removed_text); the segment is None when nothing survives the
    trim. `removed_text` is what was dropped, for the log line.
    """
    tokens = _segment_tokens(segment)
    words = segment.get("words")
    if count >= len(tokens):
        whole = _text_from_words(words) if words else segment.get("text", "").strip()
        return None, whole
    first_kept = tokens[count][0]
    if words:
        removed = _text_from_words(words[:first_kept])
        survivors = words[first_kept:]
        text = _text_from_words(survivors)
        if not text:
            return None, removed
        segment["words"] = survivors
        segment["text"] = text
        segment["start"] = normalize_timestamp(survivors[0]["start"])
    else:
        parts = segment.get("text", "").split()
        removed = " ".join(parts[:first_kept])
        text = " ".join(parts[first_kept:]).strip()
        if not text:
            return None, removed
        segment["text"] = text
    return segment, removed


def whisper_span(segment):
    """(start, end) from a segment's own word timestamps, falling back to its bounds.

    This is the reference the refined MFA timings are compared against and the
    fallback used for any segment MFA did not cover; the segment bounds are padded
    by the VAD, so the words are the tighter and more honest answer.
    """
    words = segment.get("words")
    if words:
        return float(words[0]["start"]), float(words[-1]["end"])
    return float(segment["start"]), float(segment["end"])


def timings_from_segments(segments):
    """The API's timings list: {start, end, text} floats, in order."""
    timings = []
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        start, end = whisper_span(segment)
        timings.append({"start": start, "end": end, "text": text})
    return timings


def transcript_from_segments(segments):
    """The transcription string, joined so it always equals the timings' texts."""
    return " ".join(s["text"].strip() for s in segments if s.get("text", "").strip())


# ---------------------------------------------------------------------------------
# Passes: scoring, the rescue trigger and the selection rule
# ---------------------------------------------------------------------------------
def summarize_pass(segments, duration_sec, label, speech_seconds=None, guid=None):
    """Post-process one pass's segments and score it, returning a comparable record.

    Empty segments are dropped, seam duplicates are trimmed, flags are attached and
    the anomaly counts are computed, so two passes are always compared on the output
    that would actually be published rather than on the raw decode.

    `speech_seconds` is this pass's `duration_after_vad`. Without it the window check
    cannot see an omission, so it is threaded through rather than read after the fact.
    """
    kept = [s for s in segments if s.get("text", "").strip()]
    deduped = deduplicate_segment_boundaries(kept, guid=guid)
    anomaly_count, flagged = annotate_segments(deduped)
    windows, low_windows = low_speech_windows(deduped, duration_sec, speech_seconds)
    transcript = transcript_from_segments(deduped)
    logprobs = [s["avg_logprob"] for s in deduped if s.get("avg_logprob") is not None]
    return {
        "label": label,
        "segments": deduped,
        "transcript": transcript,
        "words": len(transcript.split()),
        "anomaly_count": anomaly_count,
        "anomaly_windows": low_windows,
        "flagged_segments": flagged,
        "windows": len(windows),
        "mean_logprob": (sum(logprobs) / len(logprobs)) if logprobs else None,
    }


def should_attempt_rescue(anomaly_count, anomaly_windows):
    """True when the primary pass looks bad enough to be worth a second opinion."""
    if not RESCUE_ENABLED:
        return False
    return anomaly_count >= RESCUE_ANOMALY_SEGMENTS or anomaly_windows >= RESCUE_ANOMALY_WINDOWS


def rescue_decode_kwargs(primary_kwargs):
    """The rescue pass: the primary configuration with the two levers that matter changed.

    `condition_on_previous_text=False` is the one setting that stops a repetition loop
    feeding itself from window to window, which is the failure the anomaly flags
    actually describe; the higher ladder base skips the rung the primary pass already
    failed on. Model, beam, VAD and thresholds are deliberately identical, so a
    difference in the result is attributable to those two changes and not to noise.
    """
    kwargs = dict(primary_kwargs)
    kwargs["condition_on_previous_text"] = False
    kwargs["temperature"] = temperature_ladder(RESCUE_TEMPERATURE_BASE)
    return kwargs


def pass_sort_key(record):
    """Ordering for select_pass: fewer anomalies, then more words, then better logprob.

    Word count is the second key because every disagreement the 0.5.x confidence rule
    got wrong was a case where it preferred the shorter transcript: on the retreat
    recording it ranked the passes in reverse order of word count and dropped two runs
    of Psalm 91. Mean log-probability only breaks a remaining tie; the duration-weighted
    word probability that used to decide is not used at all.
    """
    mean_logprob = record["mean_logprob"]
    return (
        record["anomaly_count"] + record["anomaly_windows"],
        -record["words"],
        -(mean_logprob if mean_logprob is not None else float("-inf")),
    )


def retains_enough_words(primary, candidate):
    """True when `candidate` keeps enough of `primary`'s words to be worth considering.

    The anomaly score counts segments; this counts content. A rescue that clears a
    flagged segment by dropping part of the transcript has removed the evidence rather
    than the defect, and nothing downstream would ever notice.

    Both a ratio and an absolute cap, because either alone scales wrongly. The ratio
    was calibrated on a 74 word loss at 3599 words; the same 1 percent on the 8904
    word retreat file would permit 89 words, more than the two Psalm 91 runs (61 and
    58 words) whose loss is the reason this rewrite exists. The cap alone would be
    absurdly strict on a short clip.
    """
    if primary["words"] <= 0:
        return True
    lost = primary["words"] - candidate["words"]
    if lost <= 0:
        return True
    return (
        candidate["words"] >= primary["words"] * RESCUE_MIN_WORD_RETENTION
        and lost <= RESCUE_MAX_WORD_LOSS
    )


def select_pass(passes):
    """Pick the best of the decoded passes.

    A candidate that fails the word-retention floor is not eligible however good its
    anomaly score looks. Among the eligible passes the order is fewest anomalies,
    then most words, then best mean log-probability; `min` keeps the first of equal
    candidates and the primary is always first, so the rescue only displaces it on a
    strict win.
    """
    if not passes:
        raise ValueError("select_pass needs at least one pass")
    primary = passes[0]
    eligible = [primary]
    for candidate in passes[1:]:
        if retains_enough_words(primary, candidate):
            eligible.append(candidate)
        else:
            logger.warning(
                f"Discarding the {candidate['label']} pass: {candidate['words']} words against the "
                f"{primary['label']} pass's {primary['words']} loses "
                f"{primary['words'] - candidate['words']}, past the "
                f"{RESCUE_MIN_WORD_RETENTION:.0%} retention floor or the "
                f"{RESCUE_MAX_WORD_LOSS} word cap, despite scoring "
                f"{candidate['anomaly_count'] + candidate['anomaly_windows']} "
                f"against {primary['anomaly_count'] + primary['anomaly_windows']}"
            )
    return min(eligible, key=pass_sort_key)


# ---------------------------------------------------------------------------------
# The decode
# ---------------------------------------------------------------------------------
def transcribe_audio(file_path, guid):
    """Decode `file_path` in one pass and return the transcript with diagnostics.

    Returns a dict with the API-facing keys (`transcription`, `timings`,
    `duration_sec`) plus the diagnostics 0.6.0 added: `segments` (each with its
    faster-whisper fields, word timestamps and `flags`), `anomaly_count`,
    `anomaly_windows`, `flagged_segments`, `speech_seconds`, `mean_dbfs` and
    `vad_threshold`. `" ".join(t["text"] for t in timings) == transcription` always
    holds.
    """
    start_time = time.time()
    logger.info(f"Starting transcription for: {file_path} (GUID: {guid})")

    try:
        duration_sec = get_audio_duration(file_path)
        estimated_processing_time = estimate_processing_seconds(duration_sec)
        logger.info(
            f"Audio duration: {duration_sec:.2f} seconds. "
            f"Estimated processing time: ~{estimated_processing_time / 60:.2f} min"
        )
    except Exception as e:
        logger.warning(f"Could not calculate estimated processing time: {e}")
        duration_sec = 0

    # No preprocessing: faster-whisper normalises internally and the previous
    # denoise stage measurably degraded every file (see note above).
    logger.info(f"Using audio file for transcription: {file_path}")

    mean_dbfs = measure_mean_dbfs(file_path, duration_sec)
    profile, why = choose_vad_profile(mean_dbfs)
    vad = vad_parameters(profile)
    logger.info(f"Audio level for {guid}: {'unknown' if mean_dbfs is None else f'{mean_dbfs:.1f} dBFS'}; "
                f"VAD profile '{profile}' ({why}): {vad}, "
                f"hallucination_silence_threshold={hallucination_threshold(profile)}")

    model = load_whisper_model()  # Load the Whisper model (cached)

    ladder = temperature_ladder(WHISPER_TEMPERATURE_BASE)
    decode_kwargs = {
        "language": WHISPER_LANGUAGE,
        "beam_size": WHISPER_BEAM_SIZE,
        "best_of": WHISPER_BEST_OF,
        "patience": WHISPER_PATIENCE,
        "temperature": ladder,
        "compression_ratio_threshold": WHISPER_COMPRESSION_RATIO_THRESHOLD,
        "log_prob_threshold": WHISPER_LOG_PROB_THRESHOLD,
        "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
        "condition_on_previous_text": True,
        "prompt_reset_on_temperature": WHISPER_PROMPT_RESET_ON_TEMPERATURE,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": vad,
        "hallucination_silence_threshold": hallucination_threshold(profile),
        # No initial_prompt and no hotwords. The old prompt asserted "a single
        # speaker", which is false for these multi-voice class and Q&A recordings,
        # and a prompt is prepended as previous-text context, so it primes the
        # repetition loops it was meant to prevent. Measured hotwords (harness C9)
        # raised the repeated 4-gram rate from 0.031 to 0.140 for no spelling gain.
    }

    def run_pass(kwargs, label):
        """Decode once and score the result; returns (record, speech_seconds)."""
        decode_start = time.time()
        try:
            raw_segments, info = model.transcribe(file_path, **kwargs)
            segments = [serialize_segment(s) for s in raw_segments]
        except Exception as e:
            logger.error(f"Error during the {label} decode for {guid}: {e}")
            raise
        # duration_after_vad is the speech the decoder actually saw. It is read before
        # scoring, not after: it is the honest denominator for a word rate and the
        # only reference the window check has for audio the decoder skipped entirely.
        speech = float(getattr(info, "duration_after_vad", 0.0) or 0.0) if info is not None else 0.0
        record = summarize_pass(segments, duration_sec, label, speech_seconds=speech, guid=guid)
        record["decode_seconds"] = round(time.time() - decode_start, 2)
        record["speech_seconds"] = speech
        covered = sum(e - s for s, e in speech_spans(record["segments"]))
        logger.info(
            f"{label} pass for {guid} in {record['decode_seconds']:.2f}s: {record['words']} words, "
            f"{len(record['segments'])} segments covering {covered:.1f}s of {speech:.1f}s VAD speech, "
            f"anomaly_count={record['anomaly_count']}, "
            f"anomaly_windows={record['anomaly_windows']} of {record['windows']}, "
            f"flagged_segments={len(record['flagged_segments'])}"
        )
        return record, speech

    logger.info(
        f"Decoding {guid}: beam_size={WHISPER_BEAM_SIZE}, best_of={WHISPER_BEST_OF}, "
        f"patience={WHISPER_PATIENCE}, temperature ladder {ladder}"
    )
    primary, speech_seconds = run_pass(decode_kwargs, "primary")
    passes = [primary]

    rescue_attempted = should_attempt_rescue(primary["anomaly_count"], primary["anomaly_windows"])
    if rescue_attempted:
        logger.warning(
            f"Primary pass for {guid} scored anomaly_count={primary['anomaly_count']} "
            f"(trigger {RESCUE_ANOMALY_SEGMENTS}) and anomaly_windows={primary['anomaly_windows']} "
            f"(trigger {RESCUE_ANOMALY_WINDOWS}); running one rescue pass with "
            f"condition_on_previous_text=False from temperature {RESCUE_TEMPERATURE_BASE}"
        )
        rescue, rescue_speech = run_pass(rescue_decode_kwargs(decode_kwargs), "rescue")
        passes.append(rescue)

    selected = select_pass(passes)
    rescue_selected = selected["label"] == "rescue"
    if rescue_attempted:
        if rescue_selected:
            speech_seconds = rescue_speech
        logger.info(
            f"Selected the {selected['label']} pass for {guid}: "
            + " vs ".join(
                f"{p['label']} score {p['anomaly_count'] + p['anomaly_windows']} "
                f"({p['words']} words, mean logprob {p['mean_logprob']})"
                for p in passes
            )
        )

    segments = selected["segments"]
    transcription = selected["transcript"]
    timings = timings_from_segments(segments)

    wps = selected["words"] / duration_sec if duration_sec > 0 else 0.0
    total_time = time.time() - start_time
    logger.info(
        f"Transcription completed for GUID: {guid} in {total_time:.2f} seconds: "
        f"{selected['words']} words in {duration_sec:.1f}s audio ({wps:.2f} words/sec), "
        f"{speech_seconds:.1f}s speech, rescue_attempted={rescue_attempted}, "
        f"rescue_selected={rescue_selected}, anomaly_count={selected['anomaly_count']}, "
        f"anomaly_windows={selected['anomaly_windows']}"
    )

    return {
        "transcription": transcription,
        "timings": timings,
        "duration_sec": duration_sec,
        # Diagnostics. `segments` carries the word timestamps the alignment needs and
        # is not exposed by the API. Every count describes the SELECTED pass.
        "segments": segments,
        "anomaly_count": selected["anomaly_count"],
        "anomaly_windows": selected["anomaly_windows"],
        "flagged_segments": selected["flagged_segments"],
        "rescue_attempted": rescue_attempted,
        "rescue_selected": rescue_selected,
        "speech_seconds": speech_seconds,
        "mean_dbfs": mean_dbfs,
        "vad_profile": profile,
        "vad_threshold": vad["threshold"],
    }


if __name__ == "__main__":
    # For testing purposes
    test_file = os.path.join(upload_folder, "test_audio.mp3")
    test_guid = "00000000-0000-0000-0000-000000000000"
    result = transcribe_audio(test_file, test_guid)  # Run test transcription
    print(json.dumps({k: v for k, v in result.items() if k != "segments"}, indent=2))
