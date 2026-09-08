"""Analyse the four-configuration fidelity experiment against the shipped 0.6.0.

The experiment decoded a 32-file subset of the sweep corpus four ways and re-submitted
the shipped configuration a second time. Nothing here decodes anything; it reads the
records the run left behind.

    A  0.6.0 exactly as shipped                                       32 files
    B  primary temperature ladder from 0.2, so faster-whisper samples
       on the first attempt instead of beam-searching                 32 files
    C  rescue ladder from 0.2 with previous-text conditioning left ON  8 files
    D  VAD profile keyed on sample rate and bit rate, not on level    20 files
    replicate  a second submission of A, for run-to-run variation     32 files

Three things are measured.

**Passage fidelity.** Where a passage is read aloud from scripture the correct words are
knowable, so those stretches are scored by word error rate against a public-domain
translation. The World English Bible is the reference of record because it is the only
one cached for every passage; the American Standard, King James, American King James,
Young's Literal and Webster texts are scored too, for the four passages that have them,
purely to show the ordering between configurations does not depend on the reference.

The preacher's translation is unknown and almost certainly differs from all of them, so
the absolute rate is inflated and **is not accuracy and must never be reported as
accuracy**. It is a comparator: the same reference scores every configuration, so the
difference between two configurations on the same passage is meaningful even though
neither number is.

**Contiguous content.** Every one-sided run of twelve or more words between two decodes,
checked against the legacy archive. A run the legacy transcript also has is content one
decode lost; a run nobody but one decode has is something it invented. This is the
evidence that matters, not the word count: a decode can gain a hundred words of
repetition loop while losing a scripture reading.

**Detection.** Which signal available on a published transcript separates the recordings
independently known to be bad from the healthy ones, and how many healthy recordings a
threshold that catches the bad ones would also flag.

    python3 tools/quality_sweep/fidelity.py --data /home/jay/sweep/fidelity \
        --lists tools/quality_sweep --legacy /home/jay/sweep/legacy_text.json \
        --out-json fidelity.json --out-md fidelity.md
"""

import argparse
import difflib
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from norm import norm_words  # noqa: E402

# The reference of record, and the translations used only for the sensitivity check.
REFERENCE = "web"
ALTERNATES = ("web", "asv", "kjv", "akjv", "ylt", "wb")

# A one-sided difference this long is a passage rather than a wording choice. Below it
# the two decodes are arguing about fillers; at and above it one of them lost something.
RUN_MIN_WORDS = 12

# Long enough that a legacy transcript containing it is confirming the same passage and
# not a stock phrase.
LEGACY_PROBE_NGRAM = 6


# --------------------------------------------------------------------------- passages

def reference_words(cache, passage, translation=REFERENCE):
    key = "%s/%d/%d" % (translation, passage["book"], passage["chapter"])
    chapter = cache.get(key) or {}
    return norm_words(" ".join(
        chapter.get(str(v), "")
        for v in range(passage["first_verse"], passage["last_verse"] + 1)
    ))


def ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(0, max(0, len(words) - n + 1))}


def containment(ref, hyp, n=5):
    """Share of the reference's n-grams that appear anywhere in the hypothesis."""
    want = ngrams(ref, n)
    return round(len(want & ngrams(hyp, n)) / len(want), 4) if want else None


def best_window(ref, hyp, span_factor=3):
    """Locate the stretch of `hyp` most likely to hold `ref`, by 3-gram overlap.

    Scoring the whole transcript would be quadratic in its length for no gain: the
    passage is somewhere, and only its neighbourhood can align to it.
    """
    r = len(ref)
    want = ngrams(ref, 3)
    if not r or not hyp or not want:
        return 0, len(hyp)
    step = max(1, r // 8)
    best, best_at = -1, 0
    for start in range(0, max(1, len(hyp) - r + 1), step):
        score = len(want & ngrams(hyp[start:start + r], 3))
        if score > best:
            best, best_at = score, start
    pad = r * (span_factor - 1) // 2
    return max(0, best_at - pad), min(len(hyp), best_at + r + pad)


def substring_wer(ref, window):
    """Edit distance of `ref` against the best substring of `window`, over len(ref).

    Free start and free end on the hypothesis side, so the score measures how well the
    passage was transcribed and not where in the recording it sits.
    """
    if not ref:
        return None
    prev = [0] * (len(window) + 1)
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(window)
        token = ref[i - 1]
        for j in range(1, len(window) + 1):
            cost = 0 if token == window[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return round(min(prev) / len(ref), 4)


def score_passage(ref, transcript):
    lo, hi = best_window(ref, transcript)
    return {"wer": substring_wer(ref, transcript[lo:hi]),
            "containment5": containment(ref, transcript, 5)}


# ------------------------------------------------------------------- contiguous content

def diff_stats(left, right):
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    only_left = only_right = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            only_left = max(only_left, i2 - i1)
        if tag in ("insert", "replace"):
            only_right = max(only_right, j2 - j1)
    denom = max(1, min(len(left), len(right)))
    return {
        "disagreement": round(1 - (2 * matched) / max(1, len(left) + len(right)), 5),
        "run_only_left": only_left,
        "run_only_right": only_right,
        "run_max": max(only_left, only_right),
        "run_rate": round(max(only_left, only_right) / denom, 5),
    }


def one_sided_runs(left, right, minimum=RUN_MIN_WORDS):
    """Every contiguous run of `minimum`+ words one decode has and the other does not."""
    runs = []
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if (i2 - i1) >= minimum and (i2 - i1) > (j2 - j1):
            runs.append(("left", i2 - i1, i1 / max(1, len(left)), left[i1:i2]))
        if (j2 - j1) >= minimum and (j2 - j1) > (i2 - i1):
            runs.append(("right", j2 - j1, j1 / max(1, len(right)), right[j1:j2]))
    return runs


def legacy_has(legacy_words, run, n=LEGACY_PROBE_NGRAM):
    """Whether the legacy transcript contains this run, judged on any n-gram of it."""
    if not legacy_words:
        return None
    hay = " ".join(legacy_words)
    return any(" ".join(run[i:i + n]) in hay for i in range(0, max(1, len(run) - n + 1)))


# ---------------------------------------------------------------------------- detection

def detector_power(rows, key, bad, higher_is_worse=True):
    """How many bad files a threshold catches within each false-positive budget."""
    values = [(r["file"], r[key]) for r in rows if r.get(key) is not None]
    if not values:
        return None
    best = {}
    for threshold in sorted({v for _, v in values}, reverse=higher_is_worse):
        if higher_is_worse:
            flagged = {f for f, v in values if v >= threshold}
        else:
            flagged = {f for f, v in values if v <= threshold}
        caught, false_positives = len(flagged & bad), len(flagged - bad)
        for budget in (0, 1, 2, 3, 5):
            if false_positives <= budget and caught > best.get(budget, (0, None))[0]:
                best[budget] = (caught, threshold)
    bad_values = sorted(v for f, v in values if f in bad)
    healthy = sorted(v for f, v in values if f not in bad)
    return {
        "signal": key,
        "direction": "high" if higher_is_worse else "low",
        "bad_median": round(statistics.median(bad_values), 5),
        "bad_range": [bad_values[0], bad_values[-1]],
        "healthy_median": round(statistics.median(healthy), 5),
        "healthy_range": [healthy[0], healthy[-1]],
        "at_budget": {str(k): {"caught": v[0], "threshold": v[1]} for k, v in best.items()},
        "n_bad": len(bad_values),
        "n_healthy": len(healthy),
    }


# --------------------------------------------------------------------------------- shape

def shape(entry):
    """Segment shape: how often the decoder cut, and how long its segments run."""
    covered = (entry.get("speech_seconds") or 0) - (entry.get("uncovered_s") or 0)
    minutes = (entry.get("duration_sec") or 0) / 60.0
    return {
        "segments_per_min": round(entry["segments"] / minutes, 2) if minutes else None,
        "segment_mean_s": round(covered / entry["segments"], 2) if entry["segments"] else None,
    }


# --------------------------------------------------------------------------------- main

def build(data_dir, lists_dir, legacy_path):
    def load(path):
        with open(path) as handle:
            return json.load(handle)

    configs = {k: load(os.path.join(data_dir, k + ".json"))["files"]
               for k in ("A", "B", "C", "D")}
    replicate = load(os.path.join(data_dir, "replicate.json"))
    file_list = load(os.path.join(lists_dir, "fidelity_file_list.json"))
    passages = load(os.path.join(lists_dir, "scripture_passages.json"))
    cache = load(os.path.join(lists_dir, "scripture_cache.json"))
    legacy_raw = load(legacy_path) if legacy_path else {}
    legacy = {name: norm_words(text) for name, text in legacy_raw.items()}

    meta = {f["file"]: f for f in file_list["files"]}
    bad = {f["file"] for f in file_list["files"] if "bad" in (f.get("tags") or [])}
    words = {cfg: {name: norm_words(e.get("transcription"))
                   for name, e in entries.items()}
             for cfg, entries in configs.items()}

    out = {"bad_files": sorted(bad), "thresholds": file_list["thresholds"],
           "reference_translation": REFERENCE}

    # Passage fidelity, and the same scoring against every other cached translation.
    out["passages"] = []
    out["translation_sensitivity"] = []
    for passage in passages:
        ref = reference_words(cache, passage)
        row = {"file": passage["file"], "label": passage["label"],
               "ref_words": len(ref), "source": passage["source"]}
        for cfg in ("A", "B", "C", "D"):
            if passage["file"] in words[cfg]:
                row[cfg] = score_passage(ref, words[cfg][passage["file"]])
        out["passages"].append(row)
        for translation in ALTERNATES:
            alt = reference_words(cache, passage, translation)
            if not alt:
                continue
            sens = {"file": passage["file"], "label": passage["label"],
                    "translation": translation, "ref_words": len(alt)}
            for cfg in ("A", "B", "C", "D"):
                if passage["file"] in words[cfg]:
                    sens[cfg] = score_passage(alt, words[cfg][passage["file"]])["wer"]
            out["translation_sensitivity"].append(sens)

    # Every configuration against A, whole file.
    for cfg in ("B", "C", "D"):
        rows = []
        for name in sorted(set(words["A"]) & set(words[cfg])):
            left, right = configs["A"][name], configs[cfg][name]
            row = diff_stats(words["A"][name], words[cfg][name])
            row.update({
                "file": name, "bad": name in bad,
                "words_a": left["words"], "words_x": right["words"],
                "delta_pct": round((right["words"] - left["words"]) / max(1, left["words"]), 5),
                "legacy_words": (meta.get(name) or {}).get("legacy_words"),
                "segments_a": left["segments"], "segments_x": right["segments"],
                "uncovered_a": left["uncovered_s"], "uncovered_x": right["uncovered_s"],
                "max_gap_a": left["uncovered_max_gap_s"], "max_gap_x": right["uncovered_max_gap_s"],
                "windows_a": left["anomaly_windows"], "windows_x": right["anomaly_windows"],
                "flagged_a": left["flagged_segments"], "flagged_x": right["flagged_segments"],
                "logprob_a": round(left["mean_logprob"], 5),
                "logprob_x": round(right["mean_logprob"], 5),
                "rescue_a": "%s/%s" % (left["rescue_attempted"], left["rescue_selected"]),
                "rescue_x": "%s/%s" % (right["rescue_attempted"], right["rescue_selected"]),
                "wall_a": left["wall_s"], "wall_x": right["wall_s"],
                "profile_a": left["vad_profile"], "profile_x": right["vad_profile"],
                "sample_rate": (meta.get(name) or {}).get("sample_rate"),
                "bit_rate": (meta.get(name) or {}).get("bit_rate"),
                "mean_dbfs": left["mean_dbfs"],
            })
            row.update({"a_" + k: v for k, v in shape(left).items()})
            row.update({"x_" + k: v for k, v in shape(right).items()})
            rows.append(row)
        out["a_vs_" + cfg.lower()] = rows

    # Contiguous content, A against B, checked against the legacy archive.
    runs = []
    for name in sorted(set(words["A"]) & set(words["B"])):
        for side, size, position, run in one_sided_runs(words["A"][name], words["B"][name]):
            runs.append({
                "file": name,
                "side": "A-only" if side == "left" else "B-only",
                "words": size,
                "position": round(position, 3),
                "in_legacy": legacy_has(legacy.get(name), run),
                "text": " ".join(run),
            })
    runs.sort(key=lambda r: -r["words"])
    out["runs"] = runs

    # Detection, on the shipped configuration's own published statistics.
    b_runs = {r["file"]: r for r in out["a_vs_b"]}
    rows = []
    for name, entry in sorted(configs["A"].items()):
        speech = entry.get("speech_seconds") or 0.0
        uncovered = entry.get("uncovered_s") or 0.0
        row = {
            "file": name, "bad": name in bad,
            "uncovered_fraction": round(uncovered / speech, 5) if speech else None,
            "coverage": round(1 - uncovered / speech, 5) if speech else None,
            "max_gap_s": entry.get("uncovered_max_gap_s"),
            "anomaly_windows": entry.get("anomaly_windows"),
            "flagged_segments": entry.get("flagged_segments"),
            "wps_speech": round(entry["words"] / speech, 3) if speech else None,
            "words": entry["words"],
            "legacy_words": (meta.get(name) or {}).get("legacy_words"),
        }
        row.update(shape(entry))
        # The cross-check statistics. `crosscheck_run` is directional on purpose: only a
        # run the second decode has and the published one lacks is evidence of a loss.
        # The symmetric statistic also fires when the second decode is the wrong one.
        row["crosscheck_run"] = b_runs[name]["run_only_right"]
        row["crosscheck_run_reverse"] = b_runs[name]["run_only_left"]
        row["crosscheck_disagreement"] = b_runs[name]["disagreement"]
        rows.append(row)
    out["detector_rows"] = rows

    # A second label set, derived from the run inventory rather than from the earlier
    # sweep: the files where a second decode and the legacy archive agree that the
    # shipped transcript lost a passage. Independent of every coverage statistic.
    confirmed = sorted({r["file"] for r in runs
                        if r["side"] == "B-only" and r["words"] >= 25 and r["in_legacy"]})
    out["confirmed_omissions"] = confirmed

    out["detectors"] = {}
    for label, targets in (("sweep_bad", bad), ("confirmed_omissions", set(confirmed))):
        out["detectors"][label] = [
            d for d in (
                detector_power(rows, "uncovered_fraction", targets),
                detector_power(rows, "coverage", targets, higher_is_worse=False),
                detector_power(rows, "max_gap_s", targets),
                detector_power(rows, "anomaly_windows", targets),
                detector_power(rows, "flagged_segments", targets),
                detector_power(rows, "wps_speech", targets, higher_is_worse=False),
                detector_power(rows, "crosscheck_disagreement", targets),
                detector_power(rows, "crosscheck_run", targets),
                detector_power(rows, "crosscheck_run_reverse", targets),
            ) if d
        ]

    # What a cheap gap filter would select for a second decode, and whether the files
    # with a confirmed loss are inside that selection.
    out["gap_filter"] = []
    for threshold in (5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0):
        selected = {r["file"] for r in rows if (r["max_gap_s"] or 0) >= threshold}
        out["gap_filter"].append({
            "threshold_s": threshold,
            "selected": len(selected),
            "of": len(rows),
            "confirmed_reached": len(selected & set(confirmed)),
            "confirmed_total": len(confirmed),
            "sweep_bad_reached": len(selected & bad),
        })

    varied = [r for r in replicate if r["dis"] > 0]
    out["replicate"] = replicate
    out["replicate_summary"] = {
        "files": len(replicate),
        "identical": sum(1 for r in replicate if r["dis"] == 0),
        "varied": len(varied),
        "varied_where_a_pass_sampled": sum(
            1 for r in varied if "True" in r["rescue_a"] or "True" in r["rescue_b"]),
        "max_disagreement": max((r["dis"] for r in replicate), default=0),
        "max_abs_delta_pct": max((abs(r["dpct"]) for r in replicate), default=0),
    }
    return out


def render(out, handle):
    def w(line=""):
        handle.write(line + "\n")

    w("# The four-configuration fidelity experiment")
    w()
    w("Decode-only records from a 32-file subset of the sweep corpus, read after the fact.")
    w("No alignment ran in any of these configurations, so nothing here measures the")
    w("per-utterance MFA stage that 0.6.0 introduced.")
    w()
    w("**Word error rates on this page are not accuracy.** The preacher's translation is")
    w("unknown and differs from every public-domain text, which inflates every rate. The")
    w("same reference scores every configuration, so a difference between two")
    w("configurations on one passage is meaningful; the level is not.")
    w()

    w("## Passage fidelity, word error rate against the %s" % out["reference_translation"].upper())
    w()
    w("| file | passage | ref words | A | B | C | D |")
    w("|---|---|---|---|---|---|---|")
    for p in out["passages"]:
        if not p["ref_words"]:
            continue
        cells = ["%.3f" % p[c]["wer"] if p.get(c) else "-" for c in ("A", "B", "C", "D")]
        w("| %s | %s | %d | %s |" % (p["file"], p["label"], p["ref_words"], " | ".join(cells)))
    w()

    w("### The same passages against every cached translation")
    w()
    w("| file | passage | translation | A | B | C | D |")
    w("|---|---|---|---|---|---|---|")
    for r in out["translation_sensitivity"]:
        cells = ["%.3f" % r[c] if r.get(c) is not None else "-" for c in ("A", "B", "C", "D")]
        w("| %s | %s | %s | %s |" % (r["file"], r["label"], r["translation"], " | ".join(cells)))
    w()

    w("## Contiguous content, A against B")
    w()
    w("Every one-sided run of %d or more words, and whether the legacy archive has it." % RUN_MIN_WORDS)
    w()
    w("| side | words | file | at | in legacy | opening |")
    w("|---|---|---|---|---|---|")
    for r in out["runs"]:
        w("| %s | %d | %s | %.0f%% | %s | %s |" % (
            r["side"], r["words"], r["file"], 100 * r["position"], r["in_legacy"],
            " ".join(r["text"].split()[:12])))
    w()

    w("## Detection")
    w()
    for label, detectors in out["detectors"].items():
        w("### Against the %s label set (%d of %d files)" % (
            label.replace("_", " "), detectors[0]["n_bad"],
            detectors[0]["n_bad"] + detectors[0]["n_healthy"]))
        w()
        w("| signal | bad median | bad range | healthy median | healthy range | 0 FP | 1 FP | 2 FP | 3 FP |")
        w("|---|---|---|---|---|---|---|---|---|")
        for d in detectors:
            cells = []
            for budget in ("0", "1", "2", "3"):
                hit = d["at_budget"].get(budget)
                cells.append("%d @%g" % (hit["caught"], hit["threshold"]) if hit else "0")
            w("| %s | %g | %g to %g | %g | %g to %g | %s |" % (
                d["signal"], d["bad_median"], d["bad_range"][0], d["bad_range"][1],
                d["healthy_median"], d["healthy_range"][0], d["healthy_range"][1],
                " | ".join(cells)))
        w()

    w("### A gap filter as the first stage of a two-stage check")
    w()
    w("| largest gap at least | files selected | of | confirmed losses reached | of |")
    w("|---|---|---|---|---|")
    for g in out["gap_filter"]:
        w("| %.0f s | %d | %d | %d | %d |" % (
            g["threshold_s"], g["selected"], g["of"],
            g["confirmed_reached"], g["confirmed_total"]))
    w()

    w("## Run-to-run variation of the shipped configuration")
    w()
    s = out["replicate_summary"]
    w("%d files submitted twice: %d byte-identical, %d varied, %d of the varying ones on a "
      "path where a pass sampled. Largest disagreement %.5f, largest word delta %.2f percent."
      % (s["files"], s["identical"], s["varied"], s["varied_where_a_pass_sampled"],
         s["max_disagreement"], s["max_abs_delta_pct"]))
    w()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="directory holding A/B/C/D and replicate")
    parser.add_argument("--lists", required=True, help="directory holding the file and passage lists")
    parser.add_argument("--legacy", help="legacy_text.json, for confirming a lost run")
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    args = parser.parse_args(argv)

    out = build(args.data, args.lists, args.legacy)
    if args.out_json:
        with open(args.out_json, "w") as handle:
            json.dump(out, handle, indent=1, sort_keys=True)
            handle.write("\n")
    if args.out_md:
        with open(args.out_md, "w") as handle:
            render(out, handle)
    if not args.out_json and not args.out_md:
        render(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
