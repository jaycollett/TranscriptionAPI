"""Named decode configurations as JSON deltas from BASE (deep-decode.md section 2).

A config is a dict with:
  pipeline    "sequential" | "batched" | "prod" | "shared_clips"
  transcribe  kwargs passed to WhisperModel.transcribe / BatchedInferencePipeline.transcribe
  batch_size  batched pipeline only
  selection   {"rule": ...} documenting how the final segments are chosen
  note        one line on what the delta is meant to test

PROD is not expressed as a delta: its five passes are replicated verbatim in
prod_replica.py so the control reproduces production exactly.
"""

import copy

BASE = {
    "name": "BASE",
    "model": {"name": "large-v3-turbo", "compute_type": "float16", "num_workers": 1},
    "pipeline": "sequential",
    "transcribe": {
        "language": "en",
        "word_timestamps": True,
        "beam_size": 5,
        "best_of": 5,
        "patience": 1.0,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
        "prompt_reset_on_temperature": 0.5,
        "hallucination_silence_threshold": None,
        "vad_filter": True,
        "vad_parameters": {
            "threshold": 0.35,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 400,
        },
        "initial_prompt": None,
        "hotwords": None,
    },
    "selection": {"rule": "single"},
}

# Glossary for C9: Bible book names and theological terms that occur in the
# production transcripts of the reference set (refined after stage 1 by grepping
# the C1 transcripts for capitalised terms). Kept under 30 tokens on purpose: the
# hotwords are prepended to every window outside the 223-token previous-text cap.
GLOSSARY = (
    "Jesus, Holy Spirit, Bonhoeffer, Sermon on the Mount, Great Commission, Acts, Moses, "
    "Joshua, Peter, Paul, Colossians, Philemon, Habakkuk, Nehemiah, Ephesians, Galatians, "
    "Corinthians, Thessalonians, Pentecost, sanctification, propitiation"
)
GLOSSARY_TERMS = [t.strip() for t in GLOSSARY.split(",")]

C4_VAD = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 1000,
    "speech_pad_ms": 300,
}

# The 0.6.0 release candidate. Identical to C4 except that the threshold is chosen
# per file from its measured level, and the decode is followed by the production
# post-processing (segment-level boundary dedupe, anomaly score, loop flags). Both
# come from transcribe.py itself rather than a copy, so what this measures is what
# ships; `level_aware_vad` and `production_postprocess` are the two flags decode.py
# reads to do that.
RC060_VAD = dict(C4_VAD)


def _delta(name, note, transcribe=None, **top):
    cfg = copy.deepcopy(BASE)
    cfg["name"] = name
    cfg["note"] = note
    if transcribe:
        cfg["transcribe"].update(copy.deepcopy(transcribe))
    for key, value in top.items():
        cfg[key] = copy.deepcopy(value)
    return cfg


CONFIGS = {
    "PROD": {
        "name": "PROD",
        "pipeline": "prod",
        "note": "five-pass production replica with the duration-weighted word-probability rule",
        "selection": {"rule": "duration_weighted_wordprob_wps1.4"},
    },
    "C1": _delta("C1", "one beam-5 temperature-0 pass with the full ladder (BASE)"),
    # Identical to C1; run on the short file to measure determinism.
    "C1_REPEAT": _delta("C1_REPEAT", "second run of C1 for the determinism check"),
    # The greedy half of C6: same ladder, beam 1.
    "GREEDY": _delta("GREEDY", "greedy pass for the C6 agreement gate", {"beam_size": 1, "best_of": 1}),
    "C3": _delta(
        "C3",
        "no VAD, hallucination gating on",
        {"vad_filter": False, "hallucination_silence_threshold": 2.0, "no_speech_threshold": 0.6},
    ),
    "C4": _delta(
        "C4",
        "conservative VAD (cuts only at pauses over 1 s) with a threshold that can fire",
        {"vad_parameters": C4_VAD, "hallucination_silence_threshold": 0.5},
    ),
    "C5": _delta(
        "C5",
        "shared clip boundaries from C4 VAD, per-clip arbitration across three passes",
        {"vad_filter": False},
        pipeline="shared_clips",
        selection={"rule": "per_clip_anomaly_then_logprob"},
    ),
    "C8": _delta(
        "C8",
        "BatchedInferencePipeline, batch 8, temperature 0 only, its own VAD",
        {
            "temperature": [0.0],
            "vad_parameters": {"threshold": 0.5, "min_silence_duration_ms": 500, "speech_pad_ms": 300},
        },
        pipeline="batched",
        batch_size=8,
    ),
    "C9": _delta("C9", "domain glossary as hotwords on every window", {"hotwords": GLOSSARY}),
    "C10": _delta("C10", "coarser fallback ladder", {"temperature": [0.0, 0.4, 0.8, 1.0]}),
    "RC060": _delta(
        "RC060",
        "the 0.6.0 candidate: C1 decode, C4 VAD at a level-chosen threshold, production post-processing",
        {"vad_parameters": dict(RC060_VAD), "hallucination_silence_threshold": 0.5},
        level_aware_vad=True,
        production_postprocess=True,
    ),
}

# Keys the batched pipeline does not accept or ignores; dropped before the call.
BATCHED_DROP = {"prompt_reset_on_temperature"}


def get_config(name):
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; known: {', '.join(sorted(CONFIGS))}")
    return copy.deepcopy(CONFIGS[name])
