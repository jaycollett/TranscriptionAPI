"""Single-pass Faster-Whisper decode with per-segment diagnostics.

0.6.0 replaced the five-pass decode and its duration-weighted confidence rule with
one deterministic beam-5 pass (harness config C1) plus a level-aware VAD (C4) and a
per-segment anomaly score (C2/C7). The measurements behind every constant here are in
docs/QUALITY_PROPOSAL.md and tools/quality_harness/results/2026-09-07/summary.md.

0.6.1 adds one trigger, RESCUE_MAX_GAP_S: a primary pass with a long contiguous
uncovered stretch of speech now buys a second decode. The signal was already measured
and already used, priced for a quarantine gate at 20 s; priced for a second decode it
belongs at 8. See docs/QUALITY_PROPOSAL.md section 4.4 and
tools/quality_sweep/results/2026-09-08-fidelity/.
"""

import difflib
import hashlib
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
# 1.5, raised from 1.2 on the read-aloud diagnosis. Measured per 60 s window over the
# four omitted passages, the control scores 0.42, 0.38, 0.82 and 1.20 words/sec: the
# 1.2 floor caught three and missed the fourth by exactly nothing. Against a corpus
# median of 2.65 over speech and a measured healthy minimum of 2.37 across the six
# reference files, 1.5 keeps 0.87 of headroom, so this buys the fourth passage without
# spending the margin.
#
# This is the only signal that can see a confident omission. The model drops
# read-aloud passages without crossing the compression-ratio or log-probability
# thresholds, so no temperature rung above the first is ever attempted and five rungs
# of fallback go unused; nothing in the per-segment fields registers anything wrong.
# A stretch of speech carrying almost no words is the whole of the evidence.
ANOMALY_WINDOW_MIN_WPS = _env_float("ANOMALY_WINDOW_MIN_WPS", 1.5)
# A window shorter than this is not scored: a 12 s tail is not evidence of collapse.
ANOMALY_WINDOW_MIN_TAIL_SEC = 20.0
# Share of speech a healthy decode is allowed to leave uncovered before any of it
# counts as an omission.
#
# Segment spans never tile speech exactly: the decoder leaves a fraction of a second
# between segments at every breath, and those pauses sum. Charging raw uncovered time
# reports a missing minute on files that are missing nothing, which is what the first
# implementation did: it fired on 2 of 6 healthy files, triggered the rescue on both,
# and on tcf.20240213a the rescue was then selected and published 9 fewer words at
# lower agreement than the primary.
#
# This is now a loose backstop only, at 0.25. The 100-file sweep measured both this
# total and the largest contiguous uncovered stretch, and the total turned out to be
# contaminated: the two speech detectors disagree by 0.81 to 1.27 times, so the files
# with the highest uncovered fraction are largely those where the detectors disagree
# about what counts as speech, not files missing words. Any threshold on a total
# inherits that sensitivity, which is why the value kept having to move (0.10, then
# 0.15) as the measurement improved rather than converging.
ANOMALY_UNCOVERED_TOLERANCE = _env_float("ANOMALY_UNCOVERED_TOLERANCE", 0.25)
# The gap check. An omission is contiguous: the decoder stops emitting for a stretch
# and then resumes. A breath is not. Measuring the largest single uncovered stretch is
# local, so no global offset between the two speech detectors moves it, unlike the
# total above. Across 101 files: median 2.13 s, p95 17.1 s, max 46.8 s. The six
# reference files top out at 3.92 s while their uncovered totals run to 9.0 percent.
#
# 20 s is a POLICY VALUE, not a measured boundary. The number of files tripping the
# check falls smoothly with the threshold (11 at 10 s, 6 at 15, 4 at 20, 2 at 25, 1 at
# 30) with no knee anywhere in it, so this is a choice about how much review volume is
# wanted and nothing in the data argues for one value over its neighbours. Move it on
# review capacity, not on a search for the "right" number; there isn't one.
ANOMALY_MAX_UNCOVERED_GAP_SEC = _env_float("ANOMALY_MAX_UNCOVERED_GAP_SEC", 20.0)
LOOP_4GRAM_RATE = _env_float("LOOP_4GRAM_RATE", 0.3)

# Anomaly-triggered rescue pass. The five-pass decode bought redundancy by paying for
# it on every file, including the 99 percent that never needed it; this buys the same
# redundancy only where the primary pass shows evidence of trouble. Worst case is two
# passes, still well under five. Nothing downstream can quarantine on an anomaly any
# more, so this is the only thing either signal now drives: the rescue is the remedy,
# not the punishment.
RESCUE_ENABLED = os.getenv("RESCUE_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
# The low-rate window is the whole trigger. The per-segment anomaly count was measured
# across 102 recordings and 204 decodes and retired: it never exceeded 1 on any file,
# identified zero of seven independently confirmed bad transcripts, and all six files
# that scored a flag were healthy and within 0.2 percent of their previous word
# counts, so it was mildly anti-correlated with quality. Every one of those six
# carried a temperature flag, which means the only thing that creates an anomalous
# segment on this corpus is that the ladder engaged: a fact about the decode path, not
# about the output. All twelve rescues in that run came from the window trigger.
RESCUE_ANOMALY_WINDOWS = _env_int("RESCUE_ANOMALY_WINDOWS", 1)
# The gap trigger, added in 0.6.1. This reads exactly the same measurement as
# ANOMALY_MAX_UNCOVERED_GAP_SEC above, at a quarter of its threshold, and the reason
# the two numbers differ is the reason this trigger was missed for a whole release.
#
# THE SAME SIGNAL WANTS A DIFFERENT THRESHOLD DEPENDING ON WHAT FIRING COSTS. Tuned
# as a quarantine gate, a false positive withholds a finished transcript from the
# people waiting for it, so the signal was priced at 20 s and never evaluated below
# 10. Tuned as a trigger for a second decode, a false positive costs GPU time on a
# file that was fine, so the same signal belongs far lower. The measurement never
# changed; only what firing costs did.
#
# Measured over the 32-file fidelity subset, 2026-09-08. Against the seven files
# where a second decode and the legacy archive agree that the shipped transcript lost
# a contiguous run of 25 words or more, 8.5 s reaches all seven and selects eight
# files of 32; 10 s reaches five; 20 s, the quarantine value, reaches two. The
# default sits at 8.0 rather than at the measured 8.5 so a small shift in a file's
# coverage does not drop one of the seven back out. Every cheaper signal was measured
# on the same subset and separates nothing: uncovered speech fraction runs 0.053 to
# 0.134 on the bad files and 0.011 to 0.128 on the healthy ones, with the three
# highest values in the subset belonging to healthy recordings; coverage ratio is the
# same statistic inverted; word rate over speech and flagged segments catch none at
# zero false positives.
#
# It is a trigger and nothing else. No value of this constant can quarantine a job,
# requeue it, or publish an empty row: the rescue is one extra decode, and the
# retention floor and word cap still decide whether its output is kept. Set it to 0
# to switch the gap trigger off without touching the window trigger.
RESCUE_MAX_GAP_S = _env_float("RESCUE_MAX_GAP_S", 8.0)
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
# Where to keep the pass that was not published, when a rescue runs.
#
# Only the selected pass survives a job: the other one is decoded, compared, scored and
# then dropped on the floor. That cost a whole session. The 0.6.1 corpus run recorded
# that nine rescues were refused and by how many words, but not what either transcript
# said, so the question of whether those refusals threw away content could not be
# answered from disk and needed the audio decoded a second time.
#
# Set this to a directory and every job that runs a rescue drops one JSON file there
# holding both transcripts and both sets of counts. Unset, which is the default and is
# what production runs, nothing is written and nothing changes. It is a review facility
# for sweeps, so a failure to write it must never fail the job.
RESCUE_TRANSCRIPT_DIR = os.getenv("RESCUE_TRANSCRIPT_DIR", "").strip()

# Seeding the sampler.
#
# The rescue decodes from temperature 0.2, and above 0.0 faster-whisper samples rather
# than beam-searches, so the rescue is a draw from a distribution and nothing seeded it.
# The cost was measured on 2026-09-10: selection differed on two of eleven recordings
# between two runs of the same build, and on tcf.20240713 a 67 word swing between two
# draws of the same rescue flipped the shipped decision from refuse to publish. While
# the input to a quality rule is a random draw nobody recorded, no quality rule in this
# service can be validated.
#
# "guid", the default, derives the seed from the job GUID. "off" restores the previous
# behaviour, so seeding can be withdrawn on a running container if it ever proves
# harmful, without a rebuild.
#
# Why per-identifier and not a constant: a constant seed would fix every recording to
# the same corner of the sampler, and the standing objection to seeding, that it could
# be systematically worse on some material, would be fair. A per-GUID seed keeps the
# draws as varied across the archive as they are now, and makes each individual sermon
# reproducible. Each recording is decoded once either way; unseeded only means nobody
# can say which draw it got.
_SEED_MODES = ("guid", "off")
RESCUE_SEED_MODE = os.getenv("RESCUE_SEED_MODE", "guid").strip().lower() or "guid"
if RESCUE_SEED_MODE not in _SEED_MODES:
    logger.warning(
        f"RESCUE_SEED_MODE={RESCUE_SEED_MODE!r} is not one of {_SEED_MODES}; "
        f"using the default 'guid'"
    )
    RESCUE_SEED_MODE = "guid"

# Boundary de-duplication.
#
# The test is temporal, not lexical. A genuine seam artifact is the same audio decoded
# twice, so the repeated words at the end of segment N and at the start of segment N+1
# describe overlapping stretches of time. A speaker saying something twice produces two
# sequential, disjoint stretches. Word timestamps separate those two cases; word counts
# do not, which is what the corpus sweep established: across the first eight files all
# ten trims under the word-count rule were repetition rather than decoder artifacts,
# including four consecutive segments emptied on 1 Kings 18:39, "The LORD, he is God;
# the LORD, he is God", an acclamation whose entire force is the doubling.
#
# The whole step can be turned off with one variable. If overlap turns out not to
# separate the cases either, the right answer is to accept an occasional doubled
# phrase at a seam, and that should be a config change rather than a release.
# OFF by default since the 100-file sweep. Across 64 trims on 32 files it found zero
# plausible decoder artifacts, about 38 clear false positives including four
# consecutive dropped segments on the 1 Kings 18:39 acclamation, and 22 ambiguous
# sentence restarts. Thirty-one trims measured as sequential; of the ten that measured
# as overlapping, two overlap by 3.7 to 11.7 s against a phrase lasting 1.4 s, which is
# physically impossible for a double decode and marks them as measurement artifacts.
# The expected benefit is indistinguishable from zero and the demonstrated harm is
# deleted scripture. The implementation and its tests are kept intact so it can be
# switched back on if a real artifact is ever observed; it is disabled on evidence,
# not removed.
BOUNDARY_DEDUPE_ENABLED = os.getenv("BOUNDARY_DEDUPE_ENABLED", "0").strip().lower() \
    not in ("0", "false", "no", "")
# Kept as a secondary condition: a short coincidence is not worth acting on even when
# the timestamps do overlap.
BOUNDARY_DEDUPE_MIN_WORDS = _env_int("BOUNDARY_DEDUPE_MIN_WORDS", 4)
# How much the two spans may fall short of overlapping and still be trimmed. Zero by
# default, so a strict overlap is required. Continuous speech abuts: a speaker
# repeating a phrase across a segment boundary ends one occurrence and begins the next
# within a few tens of milliseconds, so treating abutment as overlap would put every
# anaphora back in scope. Raise it only on evidence that real artifacts are being
# missed, and expect to trade scripture for them.
BOUNDARY_DEDUPE_OVERLAP_SLACK_SEC = _env_float("BOUNDARY_DEDUPE_OVERLAP_SLACK_SEC", 0.0)
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


SAMPLE_RATE = 16000


def prepare_audio(file_path, vad_params):
    """Decode once and run the voice-activity detector ourselves.

    Returns (audio, speech_intervals, seconds_spent). `audio` is handed to
    `model.transcribe` in place of the path, so decoding it here costs nothing extra:
    faster-whisper would have done exactly this internally. The detector pass is the
    real added cost, and it buys the only thing that makes the omission check
    meaningful. faster-whisper reports `duration_after_vad` but not the intervals, and
    a total cannot be intersected with anything, so coverage measured against it is
    unbounded above and the check dies as soon as the ratio passes 1.0.

    Measured in the 0.5.4 image: 1.58 s on the 755 s file and 6.38 s on the 3388 s
    file, about 0.2 percent of real time, decode included.

    The same VadOptions the decode will use, so the intervals are the ones it saw.
    On any failure the caller falls back to the file path with no intervals, which
    disables the omission check rather than the job.
    """
    started = time.time()
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(file_path, sampling_rate=SAMPLE_RATE)
    chunks = get_speech_timestamps(audio, VadOptions(**vad_params))
    intervals = [(c["start"] / SAMPLE_RATE, c["end"] / SAMPLE_RATE) for c in chunks]
    return audio, intervals, time.time() - started


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


def merge_intervals(intervals):
    """Sort and merge intervals into a non-overlapping, ascending list.

    Sorting first matters: Whisper's timestamps come back through
    `restore_speech_timestamps` and adjacent segments can overlap or arrive slightly
    out of order, and a merge that only looks at its immediate predecessor double
    counts the overlap instead of absorbing it.
    """
    ordered = sorted((float(s), float(e)) for s, e in intervals if e > s)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def speech_spans(segments):
    """The segments' own spans, merged into non-overlapping ascending runs."""
    return merge_intervals((s["start"], s["end"]) for s in segments)


def intersect_intervals(a, b):
    """Intersection of two interval lists, each merged and ascending."""
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            out.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract_intervals(whole, taken):
    """`whole` minus `taken`, both merged and ascending."""
    out = []
    for start, end in whole:
        cursor = start
        for taken_start, taken_end in taken:
            if taken_end <= cursor or taken_start >= end:
                continue
            if taken_start > cursor:
                out.append((cursor, min(taken_start, end)))
            cursor = max(cursor, taken_end)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return out


def uncovered_stretches(segments, speech_intervals):
    """The speech regions no surviving segment covers, as intervals.

    The largest of these is what the omission gate reads. An omission is contiguous,
    a breath is not, and unlike a total this is local: an offset between the detector
    that produced `speech_intervals` and the one inside the decode shifts every
    boundary a little but does not create a long stretch out of nothing.
    """
    speech = merge_intervals(speech_intervals)
    return subtract_intervals(speech, intersect_intervals(speech_spans(segments), speech))


def coverage_report(segments, speech_intervals):
    """Every coverage figure the gate and the log line need, computed once.

    `covered_spans` falls back to the segments' own spans when there are no
    intervals, so the word-rate clock still works with the omission check disabled.
    """
    if not speech_intervals:
        own = speech_spans(segments)
        return {
            "speech_s": 0.0,
            "covered_s": sum(e - s for s, e in own),
            "uncovered_s": 0.0,
            "uncovered_max_gap_s": 0.0,
            "uncovered_gaps_over_threshold": 0,
            "covered_spans": own,
            "speech_spans": own,
        }
    speech = merge_intervals(speech_intervals)
    covered_spans = intersect_intervals(speech_spans(segments), speech)
    gaps = subtract_intervals(speech, covered_spans)
    return {
        "speech_s": sum(e - s for s, e in speech),
        "covered_s": sum(e - s for s, e in covered_spans),
        "uncovered_s": sum(e - s for s, e in gaps),
        "uncovered_max_gap_s": max((e - s for s, e in gaps), default=0.0),
        "uncovered_gaps_over_threshold": sum(
            1 for s, e in gaps if (e - s) >= ANOMALY_MAX_UNCOVERED_GAP_SEC
        ),
        "covered_spans": covered_spans,
        "speech_spans": speech,
    }


def covered_speech(segments, speech_intervals):
    """The speech the decoder actually produced segments for.

    The merged segment spans intersected with the voice-activity intervals, so the
    result is bounded above by the speech itself. A plain sum of segment spans is not:
    Whisper's timestamps are padded by the VAD and can run past the speech region or
    overlap each other, and comparing that sum against the VAD total produced coverage
    ratios up to 1.087 in the corpus sweep. Above 1.0 the uncovered figure goes
    negative, the tolerance can never fire and the omission check is silently dead,
    which is the defect this whole check exists to avoid, one step downstream.
    """
    return intersect_intervals(speech_spans(segments), merge_intervals(speech_intervals))


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


def low_speech_windows(segments, duration_sec, speech_intervals=None):
    """Count 60 s windows of speech carrying under 1.2 words/sec (harness C2).

    Returns (windows, low_count). The whole-file word rate hides a partial collapse:
    at a normal 2.6 words/sec about 62 percent of a file has to vanish before the 1.0
    words/sec floor trips, so a 20-40 percent omission clears the floor, the rescue
    trigger and the quarantine gate. This is what is supposed to catch it.

    Two independent checks, neither a substitute for the other.

    The window check scores every 60 s window of speech across the whole file on its
    word rate. It is what catches a stretch the decoder covered thinly: sparse
    segments over a passage it should have transcribed fully. A coverage measure is
    blind to that at any threshold, because output is present, just wrong.

    The gap check charges one long contiguous stretch of speech that no segment
    covers at all. The window check is weak on that shape in the opposite direction:
    a hole contributes no words but also, on a coverage clock, no time.

    The sweep's two word-count regressions are the first shape (73 and 91 words gone
    from one passage, largest uncovered gap 5.4 and 6.7 s) and a truncated decode is
    the second. Keeping both is the point.

    The intervals themselves are required, not a total. Coverage has to be the merged
    segment spans intersected with them, or it is not bounded by the speech it is
    compared against and the check dies quietly the moment the ratio passes 1.0.

    Only the share beyond ANOMALY_UNCOVERED_TOLERANCE is charged. Segment spans never
    tile speech exactly, and the sub-second pauses between segments sum to minutes
    over a sermon; charging those reports an omission on a file that is missing
    nothing. See the constant for the measurements.
    """
    report = coverage_report(segments, speech_intervals)
    speech_total = report["speech_s"]
    uncovered_total = report["uncovered_s"]
    # The word-rate clock runs over the speech, not over the parts of it the decoder
    # happened to cover. That distinction is the whole sensitivity of this check.
    # Clocking on coverage skips the gaps between sparse segments, so a stretch where
    # the decoder emitted a little instead of nothing gets folded into its healthy
    # neighbours and the rate never drops: on the sweep's two word-count regressions,
    # 73 and 91 words dropped from a single passage, the largest uncovered gap was
    # only 5.4 and 6.7 s because output was thin rather than absent. Clocking on
    # speech, that passage keeps its full duration and its few words, so the rate
    # falls where it actually fell.
    clock = SpeechClock(report["speech_spans"])
    covered = clock.total

    gap_events = 0
    backstop_windows = 0
    if speech_total > 0:
        # One count per qualifying gap, not one per minute of it. On a speech clock a
        # long hole already scores through the window check below, because its full
        # duration is on the clock with no words in it. What this adds is the shape
        # that check dilutes: a hole shorter than a window, which moves one window's
        # rate without sinking it.
        gap_events = report["uncovered_gaps_over_threshold"]
        # Loose backstop for an omission smeared across many medium gaps, which no
        # single stretch would catch. Deliberately generous: see the constant.
        excess = uncovered_total - ANOMALY_UNCOVERED_TOLERANCE * speech_total
        backstop_windows = int(max(0.0, excess) // ANOMALY_WINDOW_SEC)

    if covered <= 0:
        return [], max(gap_events, backstop_windows)

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
    # The strongest single piece of evidence, not the sum of all three. They are three
    # views of the same missing content, and adding them double counts: a 46.8 s hole,
    # the largest the sweep saw on an otherwise healthy file, both empties one window
    # and trips the gap check, and summing those would quarantine it on one event when
    # it should be a review case. Taking the maximum keeps each check able to raise
    # the score on its own while one event stays one event.
    return windows, max(low, gap_events, backstop_windows)


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

    The run has to overlap in time to be trimmed. Word counts cannot tell a decoder
    artifact from a speaker saying something twice, and the corpus sweep showed the
    word-count rule deleting scripture: ten trims across eight files, none of them a
    decoder artifact. See BOUNDARY_DEDUPE_ENABLED for the constants and the reasoning.

    A trim never empties a segment. Dropping a segment whole is how four consecutive
    segments of 1 Kings 18:39 disappeared, so if the duplicate is the entire segment
    the trim is declined and logged.

    Returns a new list.
    """
    result = []
    for segment in segments:
        segment = dict(segment)
        if segment.get("words") is not None:
            segment["words"] = [dict(w) for w in segment["words"]]
        if not segment.get("text", "").strip():
            continue
        if result and BOUNDARY_DEDUPE_ENABLED:
            segment = _dedupe_against(result[-1], segment, guid)
        result.append(segment)
    return result


def _dedupe_against(previous, segment, guid):
    """Trim `segment`'s head if it repeats `previous`'s tail as an artifact."""
    trim, overlap, reason = _boundary_overlap(previous, segment)
    tag = "" if guid is None else f" for {guid}"
    where = f"{segment.get('start', 0.0):.2f}s"
    overlap_field = "none" if overlap is None else f"{overlap:.3f}"

    if not trim:
        if overlap is not None:
            logger.info(
                f"Boundary dedupe{tag}: kept a repeated run at {where}, {reason} "
                f"(overlap_sec={overlap_field})"
            )
        return segment

    trimmed, removed = _trim_leading_words(segment, trim)
    if trimmed is None:
        # Never drop a segment whole. The duplicate is the entire segment, which is
        # what an acclamation looks like, and deleting it is the failure mode that
        # took out four consecutive segments of scripture.
        logger.info(
            f"Boundary dedupe{tag}: declined to trim at {where}, the run is the whole "
            f"segment and trimming would empty it (words={trim} overlap_sec={overlap_field}): "
            f"{removed!r}"
        )
        return segment

    logger.info(
        f"Boundary dedupe{tag}: dropped {trim} words repeated across the seam at "
        f"{where} (overlap_sec={overlap_field}): {removed!r}"
    )
    return trimmed


def _word_span(segment, indices):
    """(start, end) of the words at `indices`, or None without word timestamps."""
    words = segment.get("words")
    if not words or not indices:
        return None
    picked = [words[i] for i in indices if 0 <= i < len(words)]
    if not picked:
        return None
    return float(picked[0]["start"]), float(picked[-1]["end"])


def _boundary_overlap(previous, segment):
    """Find a repeated run across the seam and decide whether it is an artifact.

    Returns (k, overlap_sec, reason). `k` is 0 when nothing should be trimmed, and
    `reason` says why for the log line. `overlap_sec` is how much the two occurrences
    overlap in time; positive means the same audio was decoded twice, negative means
    the speaker said it twice.
    """
    previous_entries = _segment_tokens(previous)
    segment_entries = _segment_tokens(segment)
    previous_tokens = [t for _, t in previous_entries]
    segment_tokens = [t for _, t in segment_entries]

    upper = min(BOUNDARY_DEDUPE_MAX_WORDS, len(previous_tokens), len(segment_tokens))
    # Longest run first: "the Lord is good" must not be trimmed as "is good".
    for k in range(upper, BOUNDARY_DEDUPE_MIN_WORDS - 1, -1):
        if previous_tokens[-k:] != segment_tokens[:k]:
            continue
        tail = _word_span(previous, [i for i, _ in previous_entries[-k:]])
        head = _word_span(segment, [i for i, _ in segment_entries[:k]])
        if tail is None or head is None:
            # No word timestamps means no way to tell an artifact from repetition,
            # and the safe answer is to publish both.
            return 0, None, "no word timestamps to test the overlap"
        overlap = min(tail[1], head[1]) - max(tail[0], head[0])
        if overlap > -BOUNDARY_DEDUPE_OVERLAP_SLACK_SEC:
            return k, overlap, "spans overlap"
        return 0, overlap, "occurrences are sequential, so the speaker said it twice"
    return 0, None, ""


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
def summarize_pass(segments, duration_sec, label, speech_intervals=None, guid=None):
    """Post-process one pass's segments and score it, returning a comparable record.

    Empty segments are dropped, seam duplicates are trimmed, flags are attached and
    the anomaly counts are computed, so two passes are always compared on the output
    that would actually be published rather than on the raw decode.

    `speech_intervals` are the voice-activity regions for this audio. Without them the
    window check cannot see an omission, so they are threaded through rather than
    reconstructed after the fact.
    """
    kept = [s for s in segments if s.get("text", "").strip()]
    deduped = deduplicate_segment_boundaries(kept, guid=guid)
    anomaly_count, flagged = annotate_segments(deduped)
    windows, low_windows = low_speech_windows(deduped, duration_sec, speech_intervals)
    coverage = coverage_report(deduped, speech_intervals)
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
        "uncovered_s": round(coverage["uncovered_s"], 2),
        "uncovered_max_gap_s": round(coverage["uncovered_max_gap_s"], 2),
        "covered_s": round(coverage["covered_s"], 2),
    }


def rescue_triggers(anomaly_windows, max_gap_s=0.0):
    """Which signals ask for a second decode, in a stable order, for the log line.

    Two signals, deliberately independent, either one sufficient.

    `window` is the low-rate window: a stretch the decoder covered thinly, sparse
    segments over a passage it should have transcribed fully. Output is present and
    wrong, so no coverage measure sees it at any threshold.

    `gap` is the largest contiguous uncovered stretch of speech: a hole the decoder
    emitted nothing into. The window check is weak on that shape from the other side,
    because a hole contributes no words but, on a coverage clock, no time either. On
    the fidelity subset the gap trigger reached all seven confirmed content losses and
    the window trigger reached three, and the three the window trigger caught were all
    files the gap trigger also caught, so this widens the net rather than replacing it.

    Returning the names rather than a bool is what makes the firing mix observable:
    the completion line reports which of the two fired, so the split can be read off
    production logs instead of guessed at.
    """
    if not RESCUE_ENABLED:
        return ()
    fired = []
    if anomaly_windows >= RESCUE_ANOMALY_WINDOWS:
        fired.append("window")
    # The `> 0` guard makes RESCUE_MAX_GAP_S=0 switch the gap trigger off. Without it
    # a zero threshold would fire on every file, including the ones with no measured
    # gap at all, which is the opposite of what setting it to zero reads as.
    if RESCUE_MAX_GAP_S > 0 and max_gap_s >= RESCUE_MAX_GAP_S:
        fired.append("gap")
    return tuple(fired)


def should_attempt_rescue(anomaly_windows, max_gap_s=0.0):
    """True when the primary pass looks bad enough to be worth a second opinion.

    Either trigger is enough, and neither can do anything worse than spend one more
    decode: it runs with previous-text conditioning off and its ladder starting at
    0.2, the two changes measured to recover the lost read-aloud passages, and the
    retention floor still decides whether its output is published.
    """
    return bool(rescue_triggers(anomaly_windows, max_gap_s))


def longest_unpublished_run(published_text, candidate_text):
    """Longest contiguous run of words `candidate_text` has that `published_text` lacks.

    The review marker the fidelity experiment identified, and it is directional on
    purpose. Symmetric disagreement between two decodes is nearly useless as a signal,
    because it fires just as loudly when the second decode is the wrong one: the
    largest one-sided run in the whole 32-file experiment was 137 words, and it was a
    repetition loop the second decode was right to lack. Only a run the second decode
    has and the published pass does not is evidence that something was lost.

    Free here, and only here. It needs two decodes, which is why it cannot be a
    trigger, but when the rescue has already run both transcripts are in memory and
    this is one difflib pass over their word lists, a fraction of a second against a
    decode measured in minutes. Nothing extra is decoded to compute it.

    Zero when the rescue was published, by construction: the published pass is then
    the second decode and there is nothing it has that itself lacks. A non-zero value
    means the rescue found a run that the pass actually published does not contain,
    which is a file worth a human reading, not a file to reject.
    """
    published = norm_words(published_text or "")
    candidate = norm_words(candidate_text or "")
    if not candidate:
        return 0
    if not published:
        return len(candidate)
    longest = 0
    matcher = difflib.SequenceMatcher(a=published, b=candidate, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        # `replace` counts as well as `insert`. A run the rescue has can line up
        # against different words in the published pass rather than against nothing,
        # and difflib reports that as a replacement; ignoring it would hide exactly
        # the case where a passage was decoded as something else instead of dropped.
        if tag in ("insert", "replace"):
            longest = max(longest, j2 - j1)
    return longest


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


def retain_both_passes(guid, file_path, passes, selected):
    """Write both decoded passes to `RESCUE_TRANSCRIPT_DIR`, when one is configured.

    The published transcript is recoverable from the database; the pass that lost is
    recoverable from nowhere, and it is exactly the thing anyone reviewing the retention
    guard needs. One file per job, named for the GUID, holding the text and the counts
    for every pass that ran and which one was published.

    No-op unless the directory is set, and every failure is swallowed: this exists to
    make a past run reviewable, and no review facility is worth losing a transcript over.
    """
    if not RESCUE_TRANSCRIPT_DIR:
        return
    try:
        os.makedirs(RESCUE_TRANSCRIPT_DIR, exist_ok=True)
        record = {
            "guid": guid,
            "file": os.path.basename(file_path or ""),
            "published": selected["label"],
            "passes": [
                {
                    "label": p["label"],
                    "words": p["words"],
                    "anomaly_count": p["anomaly_count"],
                    "anomaly_windows": p["anomaly_windows"],
                    "mean_logprob": p["mean_logprob"],
                    "uncovered_s": p["uncovered_s"],
                    "uncovered_max_gap_s": p["uncovered_max_gap_s"],
                    "seed": p.get("seed"),
                    "transcription": p["transcript"],
                }
                for p in passes
            ],
        }
        path = os.path.join(RESCUE_TRANSCRIPT_DIR, f"{guid}.json")
        with open(path, "w") as handle:
            json.dump(record, handle, indent=1, sort_keys=True)
        logger.info(
            f"Kept both passes for {guid} at {path}: "
            + ", ".join(f"{p['label']} {p['words']} words" for p in passes)
            + f", published the {selected['label']}"
        )
    except Exception as exc:  # noqa: BLE001 - a review file must never fail a job
        logger.warning(f"Could not keep both passes for {guid}: {exc}")


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
# Seeding
# ---------------------------------------------------------------------------------
# CTranslate2 reserves the largest unsigned 32 bit value as its "no seed was set"
# sentinel and draws from the system generator when it sees it, so a derived seed has
# to stay clear of that one value or it would silently mean the opposite of seeding.
SEED_MODULUS = 2**32 - 1


def seed_for_guid(guid):
    """The decoder seed this job runs under, or None when seeding is off.

    blake2b over the GUID bytes rather than Python's `hash()`: `hash()` is salted per
    process, so the same GUID would seed differently in every worker and in every
    restart, which is precisely the reproducibility this exists to provide. The
    derivation is pinned by a test against an expected value, because changing it
    silently would change every transcript the service produces from then on.
    """
    if RESCUE_SEED_MODE != "guid":
        return None
    digest = hashlib.blake2b(str(guid or "").encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SEED_MODULUS


def apply_decode_seed(guid):
    """Seed the decoder for `guid` and return the seed, or None when nothing was set.

    Called immediately before every decode, primary and rescue alike. The primary
    beam-searches from temperature 0.0 and is already deterministic, so seeding it
    changes nothing; doing it uniformly is simpler than reasoning about which pass
    samples, and it keeps the whole job reproducible if the primary's base temperature
    is ever raised.

    `ctranslate2.set_random_seed` is process-global, which is safe here because the
    worker is single threaded and decodes one job at a time: app.py starts exactly one
    `transcription-worker` thread, `worker_cycle` claims a single pending job and runs
    it to completion before looking for the next, and the model is loaded with
    `num_workers=1`. No two decodes are ever in flight in one process.
    `tests/test_seeded_sampler.py` asserts that, so a future change to the assumption
    fails a test rather than quietly randomising the archive again.

    ctranslate2 is imported here rather than at module scope so the unit tests, which
    stub the GPU stack, do not have to carry it. A failure to seed is logged and the
    job continues: an unseeded transcript is the behaviour of every release up to 0.6.1
    and is worth less than a refused job.
    """
    seed = seed_for_guid(guid)
    if seed is None:
        return None
    try:
        import ctranslate2  # noqa: PLC0415 - kept out of the module import for the tests

        ctranslate2.set_random_seed(seed)
    except Exception as exc:  # noqa: BLE001 - an unseeded decode beats a failed job
        logger.warning(f"Could not seed the decoder for {guid}: {exc}")
        return None
    return seed


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

    # Decode once and run the detector ourselves. `audio` then goes to the model in
    # place of the path, so the decode is not repeated; the detector pass is the added
    # cost and is logged. Without the intervals the omission check has nothing to
    # measure coverage against, so a failure here disables it loudly rather than
    # leaving it silently dead.
    speech_intervals = None
    try:
        decoded, speech_intervals, vad_seconds = prepare_audio(file_path, vad)
        source = decoded
        logger.info(
            f"Voice activity for {guid}: {len(speech_intervals)} speech regions totalling "
            f"{sum(e - st for st, e in speech_intervals):.1f}s of {duration_sec:.1f}s "
            f"in {vad_seconds:.2f}s"
        )
    except Exception as e:
        logger.warning(
            f"Could not decode or segment {file_path} for {guid}: {e}. Falling back to the "
            f"file path; the omission check is disabled for this job."
        )
        source = file_path

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
        # Immediately before the decode, so nothing between here and the sampler can
        # consume draws from the generator and shift the result.
        seed = apply_decode_seed(guid)
        try:
            raw_segments, info = model.transcribe(source, **kwargs)
            segments = [serialize_segment(s) for s in raw_segments]
        except Exception as e:
            logger.error(f"Error during the {label} decode for {guid}: {e}")
            raise
        # duration_after_vad is the speech the decoder actually saw. It is read before
        # scoring, not after: it is the honest denominator for a word rate and the
        # only reference the window check has for audio the decoder skipped entirely.
        record = summarize_pass(segments, duration_sec, label,
                                speech_intervals=speech_intervals, guid=guid)
        record["decode_seconds"] = round(time.time() - decode_start, 2)
        record["seed"] = seed
        # Prefer our own intervals: they are the basis coverage is measured on, so the
        # ratio below is bounded by construction. duration_after_vad is the fallback
        # when the detector could not run.
        if speech_intervals is not None:
            speech = sum(e - st for st, e in merge_intervals(speech_intervals))
        else:
            speech = float(getattr(info, "duration_after_vad", 0.0) or 0.0) if info is not None else 0.0
        record["speech_seconds"] = speech
        covered = record["covered_s"]
        ratio = covered / speech if speech > 0 else 0.0
        logger.info(
            f"{label} pass for {guid} in {record['decode_seconds']:.2f}s "
            f"(seed={'off' if seed is None else seed}): {record['words']} words, "
            f"{len(record['segments'])} segments covering {covered:.1f}s of {speech:.1f}s speech "
            f"(coverage_ratio={ratio:.3f} uncovered_total_s={record['uncovered_s']} "
            f"uncovered_max_gap_s={record['uncovered_max_gap_s']}), "
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

    # The triggers read the PRIMARY pass. Both are measured on the pass that would
    # otherwise be published without a second opinion.
    triggers = rescue_triggers(primary["anomaly_windows"], primary["uncovered_max_gap_s"])
    rescue_attempted = bool(triggers)
    unpublished_run = None
    if rescue_attempted:
        logger.warning(
            f"Primary pass for {guid} fired the {'+'.join(triggers)} trigger: "
            f"anomaly_windows={primary['anomaly_windows']} (fires at {RESCUE_ANOMALY_WINDOWS}), "
            f"uncovered_max_gap_s={primary['uncovered_max_gap_s']} "
            f"(fires at {RESCUE_MAX_GAP_S}); running one rescue pass with "
            f"condition_on_previous_text=False from temperature {RESCUE_TEMPERATURE_BASE}"
        )
        rescue, rescue_speech = run_pass(rescue_decode_kwargs(decode_kwargs), "rescue")
        passes.append(rescue)

    selected = select_pass(passes)
    rescue_selected = selected["label"] == "rescue"
    if rescue_attempted:
        retain_both_passes(guid, file_path, passes, selected)
        # The directional cross-check, against the pass that is actually published.
        unpublished_run = longest_unpublished_run(selected["transcript"], rescue["transcript"])
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
        f"rescue_triggers={'+'.join(triggers) if triggers else 'none'}, "
        f"rescue_selected={rescue_selected}, "
        f"rescue_unpublished_run={unpublished_run}, "
        f"seed={'off' if selected.get('seed') is None else selected['seed']}, "
        f"seed_mode={RESCUE_SEED_MODE}, "
        f"anomaly_count={selected['anomaly_count']}, "
        f"anomaly_windows={selected['anomaly_windows']}, "
        f"primary_uncovered_max_gap_s={primary['uncovered_max_gap_s']}"
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
        # Which signals fired, joined, or "" when the rescue did not run. A log and
        # review field only: nothing downstream branches on it.
        "rescue_triggers": "+".join(triggers),
        "rescue_selected": rescue_selected,
        # The seed the published pass decoded under, or None when seeding is off. It is
        # what makes a published transcript reproducible, so it is carried out of the
        # decode as well as logged.
        "seed": selected.get("seed"),
        # The directional cross-check. None when no rescue ran, since it takes two
        # decodes and no second decode is run to obtain it.
        "rescue_unpublished_run": unpublished_run,
        "mean_logprob": selected["mean_logprob"],
        # The primary's own figures, for review: they say what the file looked like
        # before the rescue ran, which is not recoverable from the published pass.
        "primary_words": primary["words"],
        "primary_anomaly_windows": primary["anomaly_windows"],
        "primary_uncovered_max_gap_s": primary["uncovered_max_gap_s"],
        "uncovered_s": selected["uncovered_s"],
        "uncovered_max_gap_s": selected["uncovered_max_gap_s"],
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
