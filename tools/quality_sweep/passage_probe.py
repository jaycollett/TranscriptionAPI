"""Find out why 0.6.0 drops read-aloud passages, by decoding whole files five ways.

The sweep found six confirmed omissions across five files, and they share a shape: a
passage that is absent from the whole-file decode reappears when the same seconds are cut
out and decoded on their own. The one thing isolation removes is accumulated previous-text
context, so the leading hypothesis is that a decoder conditioned on preceding conversational
preaching suppresses a switch of register into formal reading.

This runs faster-whisper directly, not the service, so a single decode parameter can be
varied with everything else held fixed. The control reproduces the rc2 production kwargs
exactly, read from the image's own `transcribe.py`.

Scoring is containment, not similarity: for each known passage, the reference text is what
the isolated clip decodes to, and the score is the fraction of the reference's 5-grams that
appear anywhere in the whole-file transcript. A passage that is present scores near 1; one
that is dropped scores near 0. Containment is used rather than a diff because the passage
may land at a different offset in each config's output.

    python3 passage_probe.py --audio-dir /audio --out /work/passage_probe.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from norm import norm_words  # noqa: E402

# The rc2 production decode, verbatim from the image's transcribe.py.
LADDER = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
LADDER_FROM_0_2 = (0.2, 0.4, 0.6, 0.8, 1.0)
LOUD_VAD = {"threshold": 0.5, "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 1000, "speech_pad_ms": 300}
QUIET_VAD = {"threshold": 0.35, "min_speech_duration_ms": 250,
             "min_silence_duration_ms": 300, "speech_pad_ms": 400}

BASE_KWARGS = {
    "language": "en",
    "beam_size": 5,
    "best_of": 5,
    "patience": 1.0,
    "temperature": LADDER,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": True,
    "prompt_reset_on_temperature": 0.5,
    "word_timestamps": True,
    "vad_filter": True,
    "vad_parameters": LOUD_VAD,
    "hallucination_silence_threshold": 0.5,
}

# One change each, so a difference has one cause.
CONFIGS = {
    "control": {},
    "no_context": {"condition_on_previous_text": False},
    "no_vad": {"vad_filter": False, "vad_parameters": None},
    "quiet_profile": {"vad_parameters": QUIET_VAD},
    "ladder_from_0_2": {"temperature": LADDER_FROM_0_2},
}

# The six confirmed omissions, as (file, start, end, label). Bounds are the uncovered
# stretch the sweep measured, widened where the omission was located by transcript diff
# rather than by a coverage hole.
PASSAGES = [
    ("tcf.20250607.mp3", 76.35, 101.10, "Exodus 24 read aloud"),
    ("tcf.20210604.mp3", 290.18, 314.80, "C.S. Lewis, The Problem of Pain"),
    ("ucf20211106b.mp3", 1225.64, 1250.82, "1 Peter 1:22-25 read aloud"),
    ("tcf.20241105.mp3", 1299.39, 1326.69, "Q&A exchange, 89 words"),
    ("tcf.20241105.mp3", 1657.65, 1674.07, "Q&A exchange, 48 words"),
    ("tcf.20210210.mp3", 5.0, 35.0, "Matthew 12:15-16 read aloud"),
    ("tcf.20210217.mp3", 20.0, 50.0, "Matthew 6:19-21 read aloud"),
]

CLIP_PAD_S = 3.0
NGRAM = 5
WINDOW_S = 60.0
LOW_WPS = 1.2


def ngrams(words, n=NGRAM):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def containment(reference_words, candidate_words, n=NGRAM):
    """Fraction of the reference's n-grams that appear in the candidate."""
    ref = ngrams(reference_words, n)
    if not ref:
        return None
    cand = ngrams(candidate_words, n)
    return round(len(ref & cand) / len(ref), 4)


def decode(model, path, overrides, clip=None):
    """Decode a file (or a clip of it) and return words, text and segments."""
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    if kwargs.get("vad_parameters") is None:
        kwargs.pop("vad_parameters", None)
    started = time.time()
    if clip:
        import numpy as np
        from faster_whisper.audio import decode_audio
        audio = decode_audio(path, sampling_rate=16000)
        lo = max(0, int(clip[0] * 16000))
        hi = min(len(audio), int(clip[1] * 16000))
        audio = np.asarray(audio[lo:hi])
        segments, info = model.transcribe(audio, **kwargs)
    else:
        segments, info = model.transcribe(path, **kwargs)
    out = []
    for seg in segments:
        out.append({"start": seg.start, "end": seg.end, "text": seg.text})
    text = " ".join(s["text"].strip() for s in out).strip()
    return {
        "text": text,
        "words": norm_words(text),
        "segments": out,
        "decode_seconds": round(time.time() - started, 2),
        "speech_seconds": round(float(getattr(info, "duration_after_vad", 0.0) or 0.0), 2),
    }


def window_word_rates(segments, start, end, window_s=WINDOW_S):
    """Word rate per fixed window over a stretch, for the low-rate window check.

    The question is whether a per-window check would have fired on the omitted stretch
    when a whole-file rate does not. Words are attributed to the window their segment
    starts in, which is what a windowed gate over segment output would see.
    """
    rates = []
    cursor = start
    while cursor < end:
        stop = min(cursor + window_s, end)
        words = 0
        for seg in segments:
            if cursor <= seg["start"] < stop:
                words += len(norm_words(seg["text"]))
        span = stop - cursor
        rates.append({
            "start": round(cursor, 1),
            "end": round(stop, 1),
            "words": words,
            "wps": round(words / span, 3) if span else None,
            "below_floor": (words / span) < LOW_WPS if span else None,
        })
        cursor = stop
    return rates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="large-v3-turbo",
                        help="model name; resolved against --download-root like the service")
    parser.add_argument("--download-root", default="/app/models/whisper")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    args = parser.parse_args(argv)

    from faster_whisper import WhisperModel
    # Load exactly as the service does: the name, resolved against the image's cache.
    kwargs = {"device": "cuda", "compute_type": "float16", "num_workers": 1}
    if os.path.exists(args.download_root):
        kwargs["download_root"] = args.download_root
    model = WhisperModel(args.model, **kwargs)

    wanted = [c for c in args.configs.split(",") if c in CONFIGS]
    files = sorted({p[0] for p in PASSAGES})

    # Reference text for each passage: the isolated clip under the control config.
    references = {}
    for name, start, end, label in PASSAGES:
        path = os.path.join(args.audio_dir, name)
        clip = (max(0.0, start - CLIP_PAD_S), end + CLIP_PAD_S)
        got = decode(model, path, CONFIGS["control"], clip=clip)
        references[(name, start)] = got
        print(f"reference {name} @{start}: {len(got['words'])} words ({label})", flush=True)

    results = {"configs": wanted, "files": {}, "references": {
        f"{n}@{s}": {"words": len(r["words"]), "text": r["text"]}
        for (n, s), r in references.items()
    }}

    for name in files:
        path = os.path.join(args.audio_dir, name)
        results["files"][name] = {}
        for config in wanted:
            got = decode(model, path, CONFIGS[config])
            entry = {
                "words": len(got["words"]),
                "segments": len(got["segments"]),
                "decode_seconds": got["decode_seconds"],
                "speech_seconds": got["speech_seconds"],
                "passages": {},
            }
            for pname, start, end, label in PASSAGES:
                if pname != name:
                    continue
                ref = references[(pname, start)]
                entry["passages"][f"{start}"] = {
                    "label": label,
                    "reference_words": len(ref["words"]),
                    "containment": containment(ref["words"], got["words"]),
                    "windows": window_word_rates(got["segments"], start - 30, end + 30),
                }
            results["files"][name][config] = entry
            best = ", ".join(
                f"{k}={v['containment']}" for k, v in entry["passages"].items()
            )
            print(f"{name:24s} {config:16s} {entry['words']:6d} words "
                  f"{entry['decode_seconds']:7.1f}s  {best}", flush=True)

    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)
        handle.write("\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
