"""Best-pass selection rules.

`prod_rule` is the production rule from transcribe.py (duration-weighted word
probability with the 1.4 wps scaler); `segscore_rule` is C2 from deep-decode.md.
Both operate on serialised segments so they can be replayed offline.
"""

from metrics import window_wps


def weighted_confidence(segments):
    """Verbatim port of transcribe.calculate_weighted_confidence on serialised segments."""
    total_duration = 0.0
    weighted_sum = 0.0
    for segment in segments:
        words = segment.get("words")
        if not words:
            continue
        for word in words:
            prob = word.get("probability")
            if prob is None or prob <= 0:
                continue
            word_start = word.get("start")
            word_end = word.get("end")
            if word_start is not None and word_end is not None:
                duration = word_end - word_start
            else:
                seg_duration = segment.get("end", 0) - segment.get("start", 0)
                duration = seg_duration / len(words) if len(words) > 0 else 0
            weighted_sum += duration * prob
            total_duration += duration
    return weighted_sum / total_duration if total_duration > 0 else 0.0


def prod_rule(passes, duration_sec):
    """Return (best_index, per_pass) exactly as transcribe.transcribe_audio does."""
    per_pass = []
    for p in passes:
        conf = weighted_confidence(p["segments"])
        wc = len(p["transcript"].split())
        wps = wc / duration_sec if duration_sec > 0 else 0
        adjusted = conf * (wps / 1.4) if wps < 1.4 else conf
        per_pass.append({"confidence": round(conf, 6), "words": wc, "wps": round(wps, 4), "adjusted": round(adjusted, 6)})
    best = max(range(len(per_pass)), key=lambda i: per_pass[i]["adjusted"]) if per_pass else 0
    # numpy.argmax returns the first maximum; max() with key does the same.
    return best, per_pass


def anomaly_count(segments):
    n = 0
    for s in segments:
        t = s.get("temperature")
        if (
            (t is not None and t >= 0.5)
            or (s.get("compression_ratio") or 0) > 2.4
            or (s.get("avg_logprob") if s.get("avg_logprob") is not None else 0) < -1.0
            or ((s.get("no_speech_prob") or 0) > 0.5 and s["text"].strip())
        ):
            n += 1
    return n


def segscore_rule(passes, duration_sec, speech_chunks):
    """C2: anomalies + low 60 s windows, tie-break on mean avg_logprob (higher wins)."""
    per_pass = []
    for p in passes:
        segs = p["segments"]
        anomalies = anomaly_count(segs)
        _, low = window_wps(segs, speech_chunks, duration_sec)
        lps = [s["avg_logprob"] for s in segs if s.get("avg_logprob") is not None]
        mean_lp = sum(lps) / len(lps) if lps else float("-inf")
        per_pass.append({"anomalies": anomalies, "low_windows": low, "score": anomalies + low, "mean_logprob": round(mean_lp, 4)})
    best = min(range(len(per_pass)), key=lambda i: (per_pass[i]["score"], -per_pass[i]["mean_logprob"])) if per_pass else 0
    return best, per_pass


def clip_choice(candidates):
    """C5 per-clip rule: fewest anomalies, then highest mean avg_logprob.

    `candidates` is a list of (pass_name, segments_in_clip). Returns the index chosen.
    """
    def key(item):
        segs = item[1]
        lps = [s["avg_logprob"] for s in segs if s.get("avg_logprob") is not None]
        mean_lp = sum(lps) / len(lps) if lps else float("-inf")
        # An empty clip from one pass while another has text is treated as a miss.
        return (anomaly_count(segs), 0 if segs else 1, -mean_lp)

    return min(range(len(candidates)), key=lambda i: key(candidates[i]))
