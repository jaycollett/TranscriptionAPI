"""The production decode, replicated verbatim from transcribe.py (main, 0.5.2).

`temperature_ladder` and `clean_boundary_duplicates` are imported from the deployed
transcribe.py when it is importable (/app in the image, the repo root on a dev
machine) so the control cannot drift from production; the copies below are the
fallback and are asserted equal in the unit tests.
"""

import os
import re
import sys

PROD_PASSES = [
    {"temperature": 0.2, "patience": 3.0, "beam_size": 5},
    {"temperature": 0.0, "patience": 2.8, "beam_size": 7},
    {"temperature": 0.0, "patience": 2.5, "beam_size": 1},  # Greedy
    {"temperature": 0.3, "patience": 3.2, "beam_size": 10},
    {"temperature": 0.2, "patience": 3.5, "beam_size": 15, "condition_on_previous_text": False},
]

PROD_VAD = {"threshold": 0.35, "min_speech_duration_ms": 250, "min_silence_duration_ms": 300}


def _temperature_ladder(base, step=0.2):
    base = min(max(float(base), 0.0), 1.0)
    steps = []
    t = base
    while t < 1.0 + 1e-9:
        steps.append(round(t, 2))
        t += step
    if not steps or steps[-1] < 1.0:
        steps.append(1.0)
    return tuple(steps)


def _clean_boundary_duplicates(text):
    pattern = r"\b(\w+\s+\w+(?:\s+\w+){0,3})[.,;!?\s]*\1\b"
    while True:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            break
        start, end = match.span()
        phrase = match.group(1)
        phrase_len = len(phrase)
        duplicate_pos = text[start:end].lower().find(phrase.lower(), phrase_len)
        if duplicate_pos > 0:
            duplicate_pos += start
            text = text[:duplicate_pos] + text[duplicate_pos + phrase_len :]
    return text


def _load_production_helpers():
    """Import the two helpers from the real transcribe.py if it can be found."""
    candidates = ["/app", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]
    for root in candidates:
        path = os.path.join(root, "transcribe.py")
        if not os.path.exists(path):
            continue
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("transcribe_prod", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.temperature_ladder, module.clean_boundary_duplicates, path
        except Exception as exc:  # torch or pydub missing on a dev box
            sys.stderr.write(f"prod_replica: could not import {path}: {exc}; using local copies\n")
            break
    return _temperature_ladder, _clean_boundary_duplicates, None


temperature_ladder, clean_boundary_duplicates, PROD_HELPERS_SOURCE = _load_production_helpers()


def prod_pass_kwargs(params):
    """The exact keyword arguments transcribe.py passes to model.transcribe for one pass."""
    return {
        "language": "en",
        "vad_filter": True,
        "vad_parameters": dict(PROD_VAD),
        "beam_size": params["beam_size"],
        "temperature": temperature_ladder(params["temperature"]),
        "word_timestamps": "all",
        "suppress_tokens": [-1],
        "condition_on_previous_text": params.get("condition_on_previous_text", True),
        "patience": params["patience"],
    }


def prod_transcript(segments):
    """transcribe.py's final transcript: joined non-empty segment texts, then the dedupe regex."""
    joined = " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())
    return clean_boundary_duplicates(joined)


def prod_timings(segments):
    return [
        {"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()}
        for seg in segments
        if seg["text"].strip()
    ]
