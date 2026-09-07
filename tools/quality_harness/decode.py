"""Run one named config on one decoded audio array and serialise the result.

Only this module touches faster-whisper. Everything it returns is plain dicts so the
metrics, selection rules and report can be replayed from the JSON on disk.
"""

import copy
import logging
import os
import subprocess
import threading
import time

from configs import C4_VAD, get_config
from prod_replica import PROD_PASSES, PROD_VAD, prod_pass_kwargs, prod_timings, prod_transcript
from selection import clip_choice, prod_rule, segscore_rule

log = logging.getLogger("harness.decode")

SAMPLE_RATE = 16000
MODEL_DIR = "/app/models/whisper"


def load_model():
    from faster_whisper import WhisperModel

    kwargs = {"device": "cuda", "compute_type": "float16", "cpu_threads": os.cpu_count(), "num_workers": 1}
    if os.path.isdir(MODEL_DIR):
        kwargs["download_root"] = MODEL_DIR
    t0 = time.time()
    model = WhisperModel("large-v3-turbo", **kwargs)
    log.info("model loaded in %.1f s (%s)", time.time() - t0, kwargs)
    return model


def decode_audio_file(path):
    from faster_whisper.audio import decode_audio

    audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
    return audio, float(audio.shape[0]) / SAMPLE_RATE


def vad_chunks_seconds(audio, vad_params, max_speech_duration_s=None):
    """Speech chunks in seconds for the given VAD parameters (the library's own VAD)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    params = dict(vad_params or {})
    if max_speech_duration_s is not None:
        params["max_speech_duration_s"] = max_speech_duration_s
    chunks = get_speech_timestamps(audio, VadOptions(**params))
    return [(c["start"] / SAMPLE_RATE, c["end"] / SAMPLE_RATE) for c in chunks]


class GpuSampler:
    """Samples nvidia-smi memory.used on GPU 0 once a second while a run is in flight."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def read_mb():
        try:
            out = subprocess.run(
                ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def _loop(self):
        while not self._stop.is_set():
            value = self.read_mb()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.baseline = self.read_mb()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self):
        peak = max(self.samples) if self.samples else None
        return {
            "gpu_baseline_mb": self.baseline,
            "gpu_peak_total_mb": peak,
            "gpu_peak_mb": (peak - self.baseline) if (peak is not None and self.baseline is not None) else None,
        }


def serialize_segment(seg):
    words = None
    if seg.words:
        words = [
            {"start": float(w.start), "end": float(w.end), "word": w.word, "probability": float(w.probability)}
            for w in seg.words
        ]
    return {
        "id": seg.id,
        "seek": seg.seek,
        "start": float(seg.start),
        "end": float(seg.end),
        "text": seg.text,
        "avg_logprob": float(seg.avg_logprob) if seg.avg_logprob is not None else None,
        "compression_ratio": float(seg.compression_ratio) if seg.compression_ratio is not None else None,
        "no_speech_prob": float(seg.no_speech_prob) if seg.no_speech_prob is not None else None,
        "temperature": float(seg.temperature) if seg.temperature is not None else None,
        "words": words,
    }


def _jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def run_sequential(model, audio, kwargs):
    t0 = time.time()
    segments, info = model.transcribe(audio, **kwargs)
    segments = [serialize_segment(s) for s in segments]
    wall = time.time() - t0
    return {
        "params": {k: _jsonable(v) for k, v in kwargs.items()},
        "segments": segments,
        "transcript": " ".join(s["text"].strip() for s in segments if s["text"].strip()),
        "wall_s": round(wall, 2),
        "duration_after_vad": float(info.duration_after_vad),
    }


def run_batched(model, audio, kwargs, batch_size):
    from faster_whisper import BatchedInferencePipeline

    pipeline = BatchedInferencePipeline(model)
    t0 = time.time()
    segments, info = pipeline.transcribe(audio, batch_size=batch_size, **kwargs)
    segments = [serialize_segment(s) for s in segments]
    wall = time.time() - t0
    return {
        "params": {k: _jsonable(v) for k, v in kwargs.items()} | {"batch_size": batch_size},
        "segments": segments,
        "transcript": " ".join(s["text"].strip() for s in segments if s["text"].strip()),
        "wall_s": round(wall, 2),
        "duration_after_vad": float(info.duration_after_vad),
    }


def build_clips(chunks, max_len=30.0):
    """Group VAD speech runs into clips of at most `max_len` seconds.

    A single run longer than max_len is cut at max_len boundaries (the only case
    where a clip edge can fall inside speech).
    """
    runs = []
    for s, e in chunks:
        while e - s > max_len:
            runs.append((s, s + max_len))
            s += max_len
        runs.append((s, e))
    clips = []
    for s, e in runs:
        if clips and e - clips[-1][0] <= max_len:
            clips[-1] = (clips[-1][0], e)
        else:
            clips.append((s, e))
    return clips


def _segments_in_clip(segments, clip):
    s, e = clip
    return [seg for seg in segments if s <= (seg["start"] + seg["end"]) / 2.0 < e + 1e-6]


def run_shared_clips(model, audio, cfg):
    chunks = vad_chunks_seconds(audio, C4_VAD)
    clips = build_clips(chunks)
    flat = [t for clip in clips for t in clip]
    base = dict(cfg["transcribe"])
    base["vad_filter"] = False
    base.pop("vad_parameters", None)
    pass_a = run_sequential(model, audio, base | {"clip_timestamps": flat})
    pass_b = run_sequential(model, audio, base | {"clip_timestamps": flat, "condition_on_previous_text": False})
    batched_kwargs = {k: v for k, v in base.items() if k not in {"vad_filter", "clip_timestamps"}}
    batched_kwargs["temperature"] = [0.0]
    pass_c = run_batched(
        model, audio, batched_kwargs | {"clip_timestamps": [{"start": s, "end": e} for s, e in clips]}, 8
    )
    passes = [("cond", pass_a), ("nocond", pass_b), ("batched", pass_c)]
    chosen = []
    decisions = []
    for clip in clips:
        cands = [(name, _segments_in_clip(p["segments"], clip)) for name, p in passes]
        idx = clip_choice(cands)
        chosen.extend(cands[idx][1])
        texts = [" ".join(s["text"].strip() for s in c[1]) for c in cands]
        decisions.append(
            {
                "clip": [round(clip[0], 2), round(clip[1], 2)],
                "chosen": cands[idx][0],
                "disagree": len(set(texts)) > 1,
            }
        )
    return {
        "passes": [{"name": n} | p for n, p in passes],
        "segments": chosen,
        "clips": len(clips),
        "clip_decisions": decisions,
        "clips_disagreeing": sum(1 for d in decisions if d["disagree"]),
        "wall_s": round(sum(p["wall_s"] for _, p in passes), 2),
    }


def run_config(model, audio, duration, name, ref_chunks):
    """Run config `name`; return a result dict ready to be written as segments.json."""
    cfg = get_config(name)
    result = {"config": name, "pipeline": cfg["pipeline"], "note": cfg.get("note")}
    with GpuSampler() as gpu:
        t0 = time.time()
        if cfg["pipeline"] == "prod":
            passes = []
            for i, params in enumerate(PROD_PASSES):
                kwargs = prod_pass_kwargs(params)
                log.info("PROD pass %d/%d %s", i + 1, len(PROD_PASSES), params)
                p = run_sequential(model, audio, kwargs)
                p["pass"] = i + 1
                p["ladder_base"] = params["temperature"]
                passes.append(p)
                log.info("PROD pass %d: %d words in %.1f s", i + 1, len(p["transcript"].split()), p["wall_s"])
            best, per_pass = prod_rule(passes, duration)
            seg_best, seg_per_pass = segscore_rule(passes, duration, ref_chunks)
            winner = passes[best]["segments"]
            result.update(
                {
                    "passes": passes,
                    "segments": winner,
                    "transcript": prod_transcript(winner),
                    "transcript_raw": passes[best]["transcript"],
                    "timings": prod_timings(winner),
                    "ladder_base": PROD_PASSES[best]["temperature"],
                    "selection": {
                        "rule": cfg["selection"]["rule"],
                        "chosen_pass": best + 1,
                        "per_pass": per_pass,
                        "segscore_chosen_pass": seg_best + 1,
                        "segscore_per_pass": seg_per_pass,
                        "segscore_agrees": seg_best == best,
                    },
                    "vad_chunks": vad_chunks_seconds(audio, PROD_VAD),
                }
            )
        elif cfg["pipeline"] == "sequential":
            kwargs = dict(cfg["transcribe"])
            if not kwargs.get("vad_filter"):
                kwargs.pop("vad_parameters", None)
            p = run_sequential(model, audio, kwargs)
            result.update(
                {
                    "passes": [p],
                    "segments": p["segments"],
                    "transcript": p["transcript"],
                    "timings": prod_timings(p["segments"]),
                    "ladder_base": kwargs["temperature"][0],
                    "selection": {"rule": "single"},
                    "vad_chunks": vad_chunks_seconds(audio, kwargs["vad_parameters"]) if kwargs.get("vad_filter") else [],
                    "duration_after_vad": p["duration_after_vad"],
                }
            )
        elif cfg["pipeline"] == "batched":
            kwargs = dict(cfg["transcribe"])
            p = run_batched(model, audio, kwargs, cfg["batch_size"])
            result.update(
                {
                    "passes": [p],
                    "segments": p["segments"],
                    "transcript": p["transcript"],
                    "timings": prod_timings(p["segments"]),
                    "ladder_base": kwargs["temperature"][0],
                    "selection": {"rule": "single"},
                    "vad_chunks": vad_chunks_seconds(audio, kwargs["vad_parameters"], max_speech_duration_s=30),
                    "duration_after_vad": p["duration_after_vad"],
                }
            )
        elif cfg["pipeline"] == "shared_clips":
            r = run_shared_clips(model, audio, cfg)
            result.update(
                {
                    "passes": r["passes"],
                    "segments": r["segments"],
                    "transcript": " ".join(s["text"].strip() for s in r["segments"] if s["text"].strip()),
                    "timings": prod_timings(r["segments"]),
                    "ladder_base": 0.0,
                    "selection": {
                        "rule": cfg["selection"]["rule"],
                        "clips": r["clips"],
                        "clips_disagreeing": r["clips_disagreeing"],
                        "clip_decisions": r["clip_decisions"],
                    },
                    "vad_chunks": vad_chunks_seconds(audio, C4_VAD),
                }
            )
        else:
            raise ValueError(f"unknown pipeline {cfg['pipeline']}")
        result["wall_s"] = round(time.time() - t0, 2)
    result.update(gpu.summary())
    result["rtf"] = round(result["wall_s"] / duration, 4) if duration else None
    result["config_json"] = copy.deepcopy(cfg)
    return result


def as_float_list(chunks):
    return [[round(float(s), 3), round(float(e), 3)] for s, e in chunks]


__all__ = [
    "GpuSampler",
    "as_float_list",
    "build_clips",
    "decode_audio_file",
    "load_model",
    "run_config",
    "vad_chunks_seconds",
]
