"""Decode metrics from deep-decode.md section 3, computed on serialised segments.

A serialised segment is a dict with start, end, text, avg_logprob, compression_ratio,
no_speech_prob, temperature, seek and words (list of {start, end, word, probability}).
Everything here is pure Python so it runs on the Mac in the unit tests and on the GPU
host inside the container.
"""

import difflib
import statistics
from collections import Counter

from textnorm import norm_words

# Known Whisper hallucination strings (normalised). A segment whose whole text
# normalises to one of these is counted as a phantom candidate.
PHANTOM_STOPLIST = {
    "thank you",
    "thank you very much",
    "thanks for watching",
    "thank you for watching",
    "amen",
    "bye",
    "goodbye",
    "you",
    "the end",
    "music",
    "so",
    "okay",
    "oh",
    "subscribe",
    "please subscribe",
    "like and subscribe",
    "thank you for listening",
    "see you next time",
}

WINDOW_S = 60.0
LOW_WPS = 1.2


def percentile(values, pct):
    if not values:
        return None
    data = sorted(values)
    k = (len(data) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def flat_words(segments):
    """All words of a segment list in order, each as (start, end, normalised word)."""
    out = []
    for seg in segments:
        for w in seg.get("words") or []:
            for token in norm_words(w["word"]):
                out.append((w["start"], w["end"], token))
    return out


def words_from_segments(segments):
    """Normalised word list from segment text (falls back to text when words are absent)."""
    out = []
    for seg in segments:
        out.extend(norm_words(seg["text"]))
    return out


class SpeechClock:
    """Maps file time to cumulative speech time given VAD speech chunks in seconds."""

    def __init__(self, chunks):
        self.chunks = sorted((float(s), float(e)) for s, e in chunks)
        self.total = sum(e - s for s, e in self.chunks)
        self.before = []
        acc = 0.0
        for s, e in self.chunks:
            self.before.append(acc)
            acc += e - s

    def speech_time(self, t):
        if not self.chunks:
            return t
        for (s, e), before in zip(self.chunks, self.before):
            if t < s:
                return before
            if t <= e:
                return before + (t - s)
        return self.total


def window_wps(segments, speech_chunks, duration, window_s=WINDOW_S):
    """Words per second over `window_s` windows of speech time.

    With `speech_chunks` the clock is cumulative VAD speech, so windows never span
    long silences; without them the clock is file time. Returns per-window wps and
    the number of windows under LOW_WPS. The trailing partial window is kept when it
    covers at least 20 s.
    """
    clock = SpeechClock(speech_chunks) if speech_chunks else None
    total = clock.total if clock else duration
    if total <= 0:
        return [], 0
    counts = Counter()
    for start, end, _ in flat_words(segments):
        mid = (start + end) / 2.0
        t = clock.speech_time(mid) if clock else mid
        counts[int(t // window_s)] += 1
    n_windows = int(total // window_s)
    tail = total - n_windows * window_s
    windows = []
    for i in range(n_windows):
        windows.append(counts[i] / window_s)
    if tail >= 20.0:
        windows.append(counts[n_windows] / tail)
    low = sum(1 for w in windows if w < LOW_WPS)
    return windows, low


def repeated_ngram_rate(segments, n, window_s=WINDOW_S):
    """Share of n-grams that occur more than once within any 60 s window (file time)."""
    words = flat_words(segments)
    if len(words) < n:
        return 0.0
    buckets = {}
    for i in range(len(words) - n + 1):
        gram = tuple(w for _, _, w in words[i : i + n])
        bucket = int(words[i][0] // window_s)
        buckets.setdefault(bucket, []).append(gram)
    total = 0
    repeated = 0
    for grams in buckets.values():
        counts = Counter(grams)
        total += len(grams)
        repeated += sum(c for c in counts.values() if c > 1)
    return repeated / total if total else 0.0


def segment_repeated_4gram_rate(text):
    words = norm_words(text)
    if len(words) < 4:
        return 0.0
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(grams)


def ngram_reject_flags(segments, cr_threshold=2.4, rate_threshold=0.3):
    """C7: post-hoc segment flagger. Returns the flagged segments with the reason."""
    flagged = []
    for seg in segments:
        reasons = []
        cr = seg.get("compression_ratio")
        if cr is not None and cr > cr_threshold:
            reasons.append(f"compression_ratio {cr:.2f}")
        rate = segment_repeated_4gram_rate(seg["text"])
        if rate > rate_threshold:
            reasons.append(f"repeated_4gram_rate {rate:.2f}")
        if reasons:
            flagged.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "reasons": reasons,
                    "text": seg["text"][:200],
                }
            )
    return flagged


def phantom_segments(segments):
    hits = []
    for seg in segments:
        key = " ".join(norm_words(seg["text"]))
        if key in PHANTOM_STOPLIST:
            hits.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    "no_speech_prob": seg.get("no_speech_prob"),
                }
            )
    return hits


def api_invariants(transcript, timings):
    joined = " ".join(t["text"].strip() for t in timings if t["text"].strip())
    nonmono = 0
    overlaps = 0
    prev_end = None
    for t in timings:
        if t["end"] < t["start"]:
            nonmono += 1
        if prev_end is not None and t["start"] < prev_end:
            overlaps += 1
        prev_end = t["end"]
    return {
        "join_equals_transcript": joined == transcript,
        "timings_nonmono": nonmono,
        "timings_overlaps": overlaps,
    }


def decode_metrics(segments, transcript, duration, speech_chunks, ref_speech_chunks, ladder_base=0.0):
    """All per-config metrics. `speech_chunks` are this config's own VAD chunks (or []),
    `ref_speech_chunks` the shared reference VAD used for wps_speech and window wps."""
    words = len(transcript.split())
    seg_lens = [s["end"] - s["start"] for s in segments]
    word_durs = [w["end"] - w["start"] for s in segments for w in (s.get("words") or [])]
    logprobs = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
    crs = [s["compression_ratio"] for s in segments if s.get("compression_ratio") is not None]
    temps = [s.get("temperature") for s in segments]
    no_speech = [s.get("no_speech_prob") for s in segments if s.get("no_speech_prob") is not None]
    fallback_seeks = {
        s.get("seek", i)
        for i, s in enumerate(segments)
        if s.get("temperature") is not None and s["temperature"] > ladder_base + 1e-9
    }
    ref_clock = SpeechClock(ref_speech_chunks) if ref_speech_chunks else None
    speech_s = ref_clock.total if ref_clock else duration
    windows, low = window_wps(segments, ref_speech_chunks, duration)
    own_speech = sum(e - s for s, e in speech_chunks) if speech_chunks else duration
    return {
        "words": words,
        "wps": round(words / duration, 4) if duration else None,
        "wps_speech": round(words / speech_s, 4) if speech_s else None,
        "speech_s_ref": round(speech_s, 2),
        "seg_count": len(segments),
        "seg_mean_len_s": round(statistics.fmean(seg_lens), 2) if seg_lens else None,
        "seg_max_len_s": round(max(seg_lens), 2) if seg_lens else None,
        "word_dur_p50": round(percentile(word_durs, 50), 3) if word_durs else None,
        "word_dur_p95": round(percentile(word_durs, 95), 3) if word_durs else None,
        "logprob_mean": round(statistics.fmean(logprobs), 4) if logprobs else None,
        "logprob_min": round(min(logprobs), 4) if logprobs else None,
        "cr_max": round(max(crs), 3) if crs else None,
        "temp_ge_0_5": sum(1 for t in temps if t is not None and t >= 0.5),
        "no_speech_ratio": round(sum(1 for p in no_speech if p > 0.5) / len(no_speech), 4)
        if no_speech
        else None,
        "fallback_windows": len(fallback_seeks),
        "repeated_3gram_rate": round(repeated_ngram_rate(segments, 3), 4),
        "repeated_4gram_rate": round(repeated_ngram_rate(segments, 4), 4),
        "window_wps_min": round(min(windows), 3) if windows else None,
        "low_windows": low,
        "windows": len(windows),
        "phantom_segments": len(phantom_segments(segments)),
        "phantom_detail": phantom_segments(segments),
        "seam_count": len(speech_chunks) if speech_chunks else 0,
        "removed_s": round(duration - own_speech, 2) if speech_chunks else 0.0,
        "ngram_reject_flags": len(ngram_reject_flags(segments)),
        "ngram_reject_detail": ngram_reject_flags(segments),
    }


def agreement(segments_a, segments_b, max_opcodes=300):
    """Word-level agreement between two segment lists, localised in A's time."""
    wa = flat_words(segments_a)
    wb = flat_words(segments_b)
    ta = [w for _, _, w in wa]
    tb = [w for _, _, w in wb]
    if not ta and not tb:
        return {"ratio": 1.0, "insert": 0, "delete": 0, "replace": 0, "opcodes": []}
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    counts = Counter()
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        counts[tag] += 1
        if len(ops) < max_opcodes:
            if i1 < len(wa):
                at = wa[min(i1, len(wa) - 1)][0]
            elif wa:
                at = wa[-1][1]
            else:
                at = 0.0
            ops.append(
                {
                    "op": tag,
                    "t": round(at, 2),
                    "a": " ".join(ta[i1:i2]),
                    "b": " ".join(tb[j1:j2]),
                }
            )
    return {
        "ratio": round(sm.ratio(), 5),
        "words_a": len(ta),
        "words_b": len(tb),
        "insert": counts["insert"],
        "delete": counts["delete"],
        "replace": counts["replace"],
        "opcodes": ops,
    }
