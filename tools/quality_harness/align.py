"""Forced-alignment experiments: A0 (production MFA path) versus I1 (per-utterance
TextGrid, I5 16 kHz WAV, I6 duration-scaled timeout), with the production window
refinement rule and the I2 difflib rule, and the deep-alignment.md section 3 metrics.
"""

import bisect
import csv
import difflib
import glob
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import time

from metrics import percentile
from textnorm import norm_words

log = logging.getLogger("harness.align")

MFA_DICTIONARY_PATH = "/mfa/pretrained_models/dictionary/english_mfa.dict"
MFA_ACOUSTIC_MODEL = "english_mfa"
MFA_MODELS_DIR = "/mfa/pretrained_models"


def ensure_mfa_models(mfa_root):
    """Make the image's pretrained models visible under MFA_ROOT_DIR.

    Production runs with MFA_ROOT_DIR=/mfa, where `mfa model download` put the
    models; the harness uses a bind-mounted root so a symlink keeps the production
    command line (model by name) valid.
    """
    if not mfa_root or not os.path.isdir(MFA_MODELS_DIR):
        return
    target = os.path.join(mfa_root, "pretrained_models")
    if os.path.islink(target):
        return
    if os.path.isdir(target):
        if os.path.exists(os.path.join(target, "acoustic", "english_mfa.zip")):
            return
        # `mfa model list` creates an empty skeleton of model-type directories.
        shutil.rmtree(target)
    os.makedirs(mfa_root, exist_ok=True)
    os.symlink(MFA_MODELS_DIR, target)


# ---------------------------------------------------------------- input preparation
def prod_transcript_text(segments):
    """app.run_forced_alignment: joined raw segment texts, 4+ identical words collapsed."""
    text = " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())
    return re.sub(r"\b(\w+)\s+\1\s+\1\s+\1+", r"\1", text)


def wav_via_pydub(audio_path, wav_path):
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    audio.export(wav_path, format="wav")


def wav_via_ffmpeg(audio_path, wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", audio_path, "-vn", "-ac", "1", "-ar", "16000",
         "-sample_fmt", "s16", wav_path],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )


def build_utterances(segments, duration, gap_s=0.4, pad_s=0.15, min_len=8.0, max_len=30.0):
    """Merge Whisper segments into utterances split only at gaps >= gap_s.

    A split happens at a qualifying gap once the utterance is at least min_len long,
    or when adding the next segment would push it past max_len. Each utterance is
    padded pad_s into the adjacent gap without overlapping its neighbours and never
    shorter than 0.1 s. Returns dicts with start, end, text and segment indices.
    """
    segs = [(i, s) for i, s in enumerate(segments) if s["text"].strip()]
    groups = []
    current = []
    for i, seg in segs:
        if current:
            prev = current[-1][1]
            gap = seg["start"] - prev["end"]
            length = prev["end"] - current[0][1]["start"]
            would_be = seg["end"] - current[0][1]["start"]
            if gap >= gap_s and (length >= min_len or would_be > max_len):
                groups.append(current)
                current = []
        current.append((i, seg))
    if current:
        groups.append(current)

    utts = []
    for g in groups:
        utts.append(
            {
                "start": g[0][1]["start"],
                "end": g[-1][1]["end"],
                "text": " ".join(s["text"].strip() for _, s in g),
                "segments": [i for i, _ in g],
            }
        )
    edges = [(u["start"], u["end"]) for u in utts]
    for k, u in enumerate(utts):
        s, e = edges[k]
        prev_end = edges[k - 1][1] if k > 0 else 0.0
        next_start = edges[k + 1][0] if k + 1 < len(utts) else duration
        # Pad into the gap but never past its midpoint, so neighbours cannot overlap.
        u["start"] = round(max(0.0, s - pad_s, (prev_end + s) / 2.0 if k > 0 else 0.0), 4)
        u["end"] = round(min(duration, e + pad_s, (e + next_start) / 2.0 if k + 1 < len(utts) else duration), 4)
        if u["end"] - u["start"] < 0.1:
            u["end"] = round(min(duration, u["start"] + 0.1), 4)
    for k in range(1, len(utts)):
        if utts[k]["start"] < utts[k - 1]["end"]:
            utts[k]["start"] = utts[k - 1]["end"]
    return utts


def _tg_escape(text):
    return text.replace('"', '""')


def write_textgrid(utterances, duration, path, tier="speaker"):
    """One interval tier tiling [0, duration]; empty intervals between utterances."""
    intervals = []
    cursor = 0.0
    for u in utterances:
        if u["start"] > cursor + 1e-6:
            intervals.append((cursor, u["start"], ""))
        intervals.append((u["start"], u["end"], u["text"]))
        cursor = u["end"]
    if cursor < duration - 1e-6:
        intervals.append((cursor, duration, ""))
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {duration:.4f}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        f'        name = "{tier}"',
        "        xmin = 0",
        f"        xmax = {duration:.4f}",
        f"        intervals: size = {len(intervals)}",
    ]
    for k, (s, e, text) in enumerate(intervals, 1):
        lines += [
            f"        intervals [{k}]:",
            f"            xmin = {s:.4f}",
            f"            xmax = {e:.4f}",
            f'            text = "{_tg_escape(text)}"',
        ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- running MFA
def run_mfa(corpus_dir, out_dir, args, timeout):
    cmd = ["mfa", "align", corpus_dir, MFA_DICTIONARY_PATH, MFA_ACOUSTIC_MODEL, out_dir, "--output_format", "json"] + list(args)
    log.info("mfa: %s (timeout %.0f s)", " ".join(cmd), timeout)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "wall_s": round(time.time() - t0, 2),
            "timed_out": False,
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": None,
            "wall_s": round(time.time() - t0, 2),
            "timed_out": True,
            "stdout_tail": (exc.stdout or b"")[-3000:] if isinstance(exc.stdout, (bytes, str)) else "",
            "stderr_tail": (exc.stderr or b"")[-3000:] if isinstance(exc.stderr, (bytes, str)) else "",
        }


def _collect_work_dir_diagnostics(mfa_root, corpus_name, out_dir):
    """Gather alignment_analysis.csv and the OOV list from MFA's working and output dirs."""
    work = os.path.join(mfa_root, corpus_name) if mfa_root else None
    diag = {"oov_count": None, "oov_sample": []}
    roots = [d for d in (work, out_dir) if d and os.path.isdir(d)]
    if not roots:
        return diag, None
    oov_files = []
    for root in roots:
        oov_files += glob.glob(os.path.join(root, "**", "oovs_found*.txt"), recursive=True)
        oov_files += glob.glob(os.path.join(root, "**", "utterance_oovs.txt"), recursive=True)
        for csv_path in glob.glob(os.path.join(root, "**", "alignment_analysis.csv"), recursive=True):
            target = os.path.join(out_dir, "alignment_analysis.csv")
            if os.path.abspath(csv_path) != os.path.abspath(target) and not os.path.exists(target):
                shutil.copy2(csv_path, target)
    diag["diagnostic_files"] = sorted(os.path.relpath(p, os.path.dirname(p.rstrip("/"))) for p in oov_files)
    words = set()
    for path in oov_files:
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if os.path.basename(path).startswith("utterance_oovs"):
                        words.update(parts[1:])
                    else:
                        words.add(parts[0])
        except OSError:
            pass
    if oov_files:
        diag["oov_count"] = len(words)
        diag["oov_sample"] = sorted(words)[:40]
    return diag, work


def _analysis_stats(out_dir):
    path = os.path.join(out_dir, "alignment_analysis.csv")
    if not os.path.exists(path):
        return None
    dev_gt10 = snr_le1 = n = 0
    lls = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n += 1
            try:
                if float(row.get("duration_deviation") or 0) > 10:
                    dev_gt10 += 1
            except ValueError:
                pass
            try:
                if float(row.get("snr") or 99) <= 1.0:
                    snr_le1 += 1
            except ValueError:
                pass
            try:
                lls.append(float(row.get("speech_log_likelihood")))
            except (TypeError, ValueError):
                pass
    return {
        "utterances": n,
        "duration_deviation_gt10": dev_gt10,
        "snr_le1": snr_le1,
        "speech_log_likelihood_mean": round(statistics.fmean(lls), 3) if lls else None,
    }


def _find_json(out_dir, stem):
    path = os.path.join(out_dir, f"{stem}.json")
    if os.path.exists(path):
        return path
    hits = glob.glob(os.path.join(out_dir, "**", "*.json"), recursive=True)
    return hits[0] if hits else None


def load_mfa_words(json_path):
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    tiers = data.get("tiers", {})
    words_tier = tiers.get("words") or next((v for k, v in tiers.items() if k.endswith("words")), None)
    utt_tier = tiers.get("utterances") or next((v for k, v in tiers.items() if k.endswith("utterances")), None)
    words = [
        {"start": float(s), "end": float(e), "text": label}
        for s, e, label in (words_tier or {}).get("entries", [])
        if label and label not in {"<eps>", "sil"}
    ]
    utts = [
        {"start": float(s), "end": float(e), "text": label}
        for s, e, label in (utt_tier or {}).get("entries", [])
        if label
    ]
    return words, utts


def utterances_with_words(utterances, words):
    """Count utterances whose span contains at least one MFA word start."""
    starts = sorted(w["start"] for w in words)
    n = 0
    for u in utterances:
        i = bisect.bisect_left(starts, u["start"] - 1e-6)
        if i < len(starts) and starts[i] <= u["end"] + 1e-6:
            n += 1
    return n


def align_path(path_name, audio_path, segments, duration, workdir, mfa_root, stem):
    """Run one alignment path ("a0" or "i1"). Returns the process record and MFA words."""
    corpus_name = re.sub(r"[^A-Za-z0-9_]", "_", f"{stem}_{path_name}_mfa_input")
    corpus_dir = os.path.join(workdir, corpus_name)
    out_dir = os.path.join(workdir, f"{corpus_name}_aligned")
    shutil.rmtree(corpus_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(corpus_dir)
    os.makedirs(out_dir)
    name = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    wav = os.path.join(corpus_dir, f"{name}.wav")
    record = {"path": path_name, "attempts": [], "prep_wall_s": None}
    t0 = time.time()
    try:
        if path_name == "a0":
            wav_via_pydub(audio_path, wav)
            with open(os.path.join(corpus_dir, f"{name}.txt"), "w", encoding="utf-8") as fh:
                fh.write(prod_transcript_text(segments))
            record["words_submitted"] = len(prod_transcript_text(segments).split())
            record["utterances_submitted"] = 1
            plan = [
                (["--beam", "40", "--retry_beam", "100", "--clean"], 300),
                (["--beam", "100", "--retry_beam", "400", "--clean"], 300),
            ]
        else:
            wav_via_ffmpeg(audio_path, wav)
            utts = build_utterances(segments, duration)
            write_textgrid(utts, duration, os.path.join(corpus_dir, f"{name}.TextGrid"))
            record["utterances_submitted"] = len(utts)
            record["words_submitted"] = sum(len(u["text"].split()) for u in utts)
            record["utterance_len_p50"] = round(percentile([u["end"] - u["start"] for u in utts], 50), 2)
            record["utterance_len_max"] = round(max(u["end"] - u["start"] for u in utts), 2)
            plan = [(["--include_original_text", "--no_tokenization", "--clean", "--overwrite"], 60 + 0.5 * duration)]
    finally:
        record["prep_wall_s"] = round(time.time() - t0, 2)
    record["wav_bytes"] = os.path.getsize(wav) if os.path.exists(wav) else None

    success = False
    for k, (args, timeout) in enumerate(plan, 1):
        res = run_mfa(corpus_dir, out_dir, args, timeout)
        res["attempt"] = k
        record["attempts"].append(res)
        diag, work = _collect_work_dir_diagnostics(mfa_root, corpus_name, out_dir)
        res.update(diag)
        if res["returncode"] == 0 and _find_json(out_dir, name):
            success = True
            record["attempt_used"] = k
            break
        log.warning("mfa %s attempt %d failed rc=%s timed_out=%s", path_name, k, res["returncode"], res["timed_out"])
    record["mfa_success"] = success
    record["mfa_wall_s"] = round(sum(a["wall_s"] for a in record["attempts"]), 2)
    record["mfa_rtf"] = round(record["mfa_wall_s"] / duration, 4) if duration else None
    record["timeouts"] = sum(1 for a in record["attempts"] if a["timed_out"])
    record["analysis"] = _analysis_stats(out_dir) if success else None
    words, utts_out = [], []
    if success:
        json_path = _find_json(out_dir, name)
        words, utts_out = load_mfa_words(json_path)
        record["mfa_json"] = json_path
    record["words_mfa"] = len(words)
    # The utterances tier echoes every input interval, so success is measured as
    # utterances that received at least one word in the words tier.
    record["utterances_aligned"] = utterances_with_words(utts_out, words) if utts_out else None
    # MFA's working tree can be hundreds of MB; remove it once diagnostics are captured.
    if mfa_root:
        shutil.rmtree(os.path.join(mfa_root, corpus_name), ignore_errors=True)
    if os.path.exists(wav):
        os.remove(wav)
    return record, words, utts_out


# ---------------------------------------------------------------- refinement rules
def refine_window(segments, mfa_words):
    """app.run_forced_alignment's rule: words whose start lies inside the Whisper span."""
    refined = []
    empties = 0
    for seg in segments:
        ws, we = seg["start"], seg["end"]
        owned = [w for w in mfa_words if ws <= w["start"] <= we]
        if owned:
            refined.append({"start": owned[0]["start"], "end": owned[-1]["end"], "text": seg["text"], "owned": len(owned)})
        else:
            empties += 1
            refined.append({"start": ws, "end": we, "text": seg["text"], "owned": 0, "fallback": True})
    return refined, empties


def match_words(segments, mfa_words):
    """difflib match between MFA tokens and Whisper word tokens (I2 core).

    Returns per-Whisper-token records (segment index, whisper start, matched MFA
    index or None) and the match ratio.
    """
    whisper_tokens = []  # (seg_idx, start, end, token)
    for si, seg in enumerate(segments):
        if seg.get("words"):
            for w in seg["words"]:
                for tok in norm_words(w["word"]):
                    whisper_tokens.append((si, w["start"], w["end"], tok))
        else:
            for tok in norm_words(seg["text"]):
                whisper_tokens.append((si, seg["start"], seg["end"], tok))
    mfa_tokens = []  # (mfa_idx, token)
    for mi, w in enumerate(mfa_words):
        toks = norm_words(w["text"])
        # An MFA label that normalises to several tokens is matched on its first one.
        mfa_tokens.append((mi, toks[0] if toks else ""))
    sm = difflib.SequenceMatcher(None, [t[1] for t in mfa_tokens], [t[3] for t in whisper_tokens], autojunk=False)
    matched = [None] * len(whisper_tokens)
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            matched[block.b + k] = mfa_tokens[block.a + k][0]
    records = [
        {"seg": si, "w_start": ws, "w_end": we, "token": tok, "mfa": mi}
        for (si, ws, we, tok), mi in zip(whisper_tokens, matched)
    ]
    return records, round(sm.ratio(), 4)


def refine_i2(segments, mfa_words, records=None):
    """I2: owned words come from the sequence match; edges are then made monotonic."""
    if records is None:
        records, _ = match_words(segments, mfa_words)
    owned = {}
    for r in records:
        if r["mfa"] is not None:
            owned.setdefault(r["seg"], []).append(r["mfa"])
    refined = []
    empties = 0
    for si, seg in enumerate(segments):
        idx = owned.get(si)
        if idx:
            refined.append(
                {"start": mfa_words[min(idx)]["start"], "end": mfa_words[max(idx)]["end"], "text": seg["text"], "owned": len(idx)}
            )
        else:
            empties += 1
            refined.append({"start": seg["start"], "end": seg["end"], "text": seg["text"], "owned": 0, "fallback": True})
    pre_nonmono = count_nonmono(refined)
    prev_end = 0.0
    for r in refined:
        r["start"] = max(r["start"], prev_end)
        r["end"] = max(r["end"], r["start"] + 0.05)
        prev_end = r["end"]
    return refined, empties, pre_nonmono


def count_nonmono(refined):
    n = 0
    for k, r in enumerate(refined):
        if r["end"] < r["start"]:
            n += 1
        if k > 0 and r["start"] < refined[k - 1]["end"]:
            n += 1
    return n


def alignment_metrics(segments, refined, mfa_words, records, empties, transcript_text):
    """deep-alignment.md section 3 timing-quality and text-consistency metrics."""
    d_start = []
    agree = 0
    ratios = []
    for seg, r in zip(segments, refined):
        ds = abs(r["start"] - seg["start"])
        de = abs(r["end"] - seg["end"])
        d_start.append(ds)
        if ds <= 0.25 and de <= 0.25:
            agree += 1
        n_words = len(norm_words(seg["text"]))
        if n_words:
            ratios.append(r.get("owned", 0) / n_words)
    matched = [r for r in records if r["mfa"] is not None]
    drift = sum(1 for r in matched if abs(mfa_words[r["mfa"]]["start"] - r["w_start"]) > 1.0)
    gaps = []
    for k in range(1, len(mfa_words)):
        g = mfa_words[k]["start"] - mfa_words[k - 1]["end"]
        if g >= 0.2:
            gaps.append(mfa_words[k]["start"])
    snapped = 0
    for r in refined:
        if any(abs(r["start"] - g) <= 0.02 for g in gaps):
            snapped += 1
    n = len(segments)
    return {
        "segments": n,
        "agree250": round(agree / n, 4) if n else None,
        "delta_start_p50": round(percentile(d_start, 50), 3) if d_start else None,
        "delta_start_p95": round(percentile(d_start, 95), 3) if d_start else None,
        "nonmono": count_nonmono(refined),
        "owned_ratio_lt_0_8": round(sum(1 for x in ratios if x < 0.8) / len(ratios), 4) if ratios else None,
        "owned_ratio_gt_1_2": round(sum(1 for x in ratios if x > 1.2) / len(ratios), 4) if ratios else None,
        "empty_fallbacks": empties,
        "drift_words": round(drift / len(matched), 4) if matched else None,
        "matched_words": len(matched),
        "pause_snapped": round(snapped / n, 4) if n else None,
        "words_transcript": len(transcript_text.split()),
        "words_timings": sum(len(s["text"].split()) for s in segments),
        "words_mfa": len(mfa_words),
    }
