"""What the discarded rescue pass on tcf.20250607 actually costs, and what a different
tie-break rule would do to the rest of the corpus.

0.6.1 discarded the rescue on `tcf.20250607` even though that rescue held a 156 word run
containing the Exodus 24 reading, because the rescue was 340 words shorter overall. The
open question is whether those 340 words are content the primary heard and the rescue
missed, or filler the two decodes merely word differently. Word count cannot answer it;
contiguous runs checked against the legacy archive can, and that is the method the
fidelity experiment already used.

Nothing here decodes. It reads the records the runs left behind:

  * `fidelity/A.json`   0.6.0 as shipped. Its published transcript for this file is the
                        primary pass, since the rescue was discarded. The primary is a
                        beam search from temperature 0.0 and is deterministic, so this is
                        byte-for-byte the primary the 0.6.1 run published.
  * `fidelity/B.json`   the rescue configuration decoded whole-file, and on this recording
                        its own rescue was selected. This is a sample from the same
                        unseeded sampler the 0.6.1 rescue draws from, not the same draw.
  * `legacy_text.json`  the archive transcript, the independent third party.
  * `run/timings/`      per-utterance timings from the sweep run, to date the runs.

Part two parses the 0.6.1 service log and replays `select_pass` over all twenty rescues
under alternative rules, so a proposed tie-break can be scored on the corpus rather than
on the one recording that motivated it.

    python3 rescue_deficit.py --fidelity /home/jay/sweep/fidelity \
        --legacy /home/jay/sweep/legacy_text.json \
        --timings /home/jay/sweep/run/timings \
        --candidate /home/jay/sweep/rc061/candidate_pass.json \
        --primary-only /home/jay/sweep/rc061/primary_only.json \
        --log /home/jay/sweep/rc061.log \
        --file tcf.20250607.mp3 --out-json rescue_deficit.json --out-md rescue_deficit.md
"""

import argparse
import collections
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from norm import norm_words  # noqa: E402

# A one-sided difference this long is a passage rather than a wording choice; the same
# threshold the fidelity analysis used, so the two are directly comparable.
RUN_MIN_WORDS = 12
# Below the headline threshold, to show the shape of what is left over.
RUN_FLOOR = 4
# Any n-gram of a run appearing in the legacy transcript is enough to call it real: the
# legacy decoder was a different model on the same audio, so an exact match of six words
# is not something two independent decoders invent.
LEGACY_PROBE_NGRAM = 6
# A run that says the same four words twice inside itself is a loop, not a passage.
LOOP_NGRAM = 4

# The 0.6.1 constants, so the replay is the shipped rule and not a paraphrase of it.
RESCUE_MIN_WORD_RETENTION = 0.99
RESCUE_MAX_WORD_LOSS = 40


# ------------------------------------------------------------------ contiguous content

def one_sided_runs(left, right, minimum=RUN_MIN_WORDS):
    """Every contiguous run of `minimum`+ words one decode has and the other does not.

    Yields (side, size, index in that side's stream, words, index in the left stream).
    """
    runs = []
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if (i2 - i1) >= minimum and (i2 - i1) > (j2 - j1):
            runs.append(("left", i2 - i1, i1, left[i1:i2], i1))
        if (j2 - j1) >= minimum and (j2 - j1) > (i2 - i1):
            # `anchor` is where the run belongs in the left transcript, which is the one
            # the timings were produced for. A run only the right side has still has a
            # place on the clock: the point the left side skipped over.
            runs.append(("right", j2 - j1, j1, right[j1:j2], i1))
    return runs


def deficit_budget(left, right):
    """Where the word difference between two decodes lives, by run length.

    A decode can be four hundred words shorter because it lost a reading, or because it
    dropped a filler word in four hundred places. Those are not the same defect and the
    totals do not distinguish them, so bucket every one-sided stretch by its length.
    """
    buckets = collections.OrderedDict(
        (name, {"left_words": 0, "right_words": 0, "left_runs": 0, "right_runs": 0})
        for name in ("1", "2-3", "4-11", "12+"))

    def bucket_for(size):
        if size <= 1:
            return "1"
        if size <= 3:
            return "2-3"
        if size <= 11:
            return "4-11"
        return "12+"

    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        # Only the excess counts. A five-for-three replacement is two words of deficit,
        # not five words of loss and three of gain.
        excess = (i2 - i1) - (j2 - j1)
        if excess > 0:
            row = buckets[bucket_for(i2 - i1)]
            row["left_words"] += excess
            row["left_runs"] += 1
        elif excess < 0:
            row = buckets[bucket_for(j2 - j1)]
            row["right_words"] += -excess
            row["right_runs"] += 1
    return buckets


def single_word_deficit(left, right, top=25):
    """The vocabulary of the one-word-at-a-time difference.

    Four hundred words spread one at a time over a forty-seven minute recording is either
    four hundred lost words or a different transcription convention for fillers. The words
    themselves settle it, and nothing else does.
    """
    counter = collections.Counter()
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if (i2 - i1) - (j2 - j1) == 1 and (i2 - i1) <= 3:
            # The one word this stretch has too many, whichever it is.
            extra = [w for w in left[i1:i2] if w not in right[j1:j2]] or list(left[i1:i2])
            counter[extra[0]] += 1
    return counter.most_common(top)


def legacy_has(legacy_words, run, n=LEGACY_PROBE_NGRAM):
    """Whether the legacy transcript contains this run, judged on any n-gram of it."""
    if not legacy_words:
        return None
    hay = " ".join(legacy_words)
    return any(" ".join(run[i:i + n]) in hay for i in range(0, max(1, len(run) - n + 1)))


def legacy_coverage(legacy_words, run, n=LEGACY_PROBE_NGRAM):
    """How much of the run the legacy transcript corroborates, not merely whether any of
    it does. A single matching n-gram inside a twenty word run means something different
    from every n-gram matching."""
    if not legacy_words or len(run) < n:
        return None
    hay = " ".join(legacy_words)
    grams = [" ".join(run[i:i + n]) for i in range(len(run) - n + 1)]
    hits = sum(1 for g in grams if g in hay)
    return round(hits / len(grams), 4)


def repeats_itself(run, n=LOOP_NGRAM):
    """The most times any n-gram of the run appears inside the run."""
    if len(run) < n:
        return 1
    counts = collections.Counter(
        tuple(run[i:i + n]) for i in range(len(run) - n + 1))
    return max(counts.values())


def classify(run, legacy):
    """real content, repetition, or ambiguous, on the legacy archive as the arbiter."""
    repeat = repeats_itself(run)
    covered = legacy_coverage(legacy, run)
    if repeat >= 3:
        return "repetition", repeat, covered
    if legacy_has(legacy, run):
        return "real content", repeat, covered
    return "ambiguous", repeat, covered


# ------------------------------------------------------------------------- timestamping

def load_timings(path):
    """Word index to seconds, from the sweep run's per-utterance timings."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as handle:
        doc = json.load(handle)
    spans = []
    index = 0
    for utterance in doc.get("timings") or []:
        count = len(norm_words(utterance.get("text") or ""))
        if not count:
            continue
        spans.append((index, index + count, utterance.get("start"), utterance.get("end")))
        index += count
    return {"spans": spans, "words": index}


def stamp(timings, index):
    """The second at which the word at `index` was spoken, or None."""
    if not timings:
        return None
    for lo, hi, start, end in timings["spans"]:
        if lo <= index < hi:
            if start is None or end is None:
                return None
            share = (index - lo) / max(1, hi - lo)
            return round(start + share * (end - start), 2)
    return None


def clock(seconds):
    if seconds is None:
        return "-"
    return "%d:%02d" % (int(seconds // 60), int(seconds % 60))


# ---------------------------------------------------------------------- the log replay

STARTED = re.compile(r"Starting transcription for: (?P<path>\S+) \(GUID: (?P<guid>[0-9a-f-]+)\)")
PASS = re.compile(
    r"(?P<label>primary|rescue) pass for (?P<guid>[0-9a-f-]+) in [0-9.]+s: "
    r"(?P<words>\d+) words, (?P<segments>\d+) segments .*?"
    r"uncovered_max_gap_s=(?P<gap>[0-9.]+)\), "
    r"anomaly_count=(?P<anomaly>\d+), anomaly_windows=(?P<windows>\d+)")
SELECTED = re.compile(
    r"Selected the (?P<label>primary|rescue) pass for (?P<guid>[0-9a-f-]+): "
    r"primary score (?P<pscore>\d+) \((?P<pwords>\d+) words, mean logprob (?P<plp>-?[0-9.]+)\) "
    r"vs rescue score (?P<rscore>\d+) \((?P<rwords>\d+) words, mean logprob (?P<rlp>-?[0-9.]+)\)")


def parse_log(path):
    """Both passes and the shipped decision, for every file that ran a rescue."""
    names, passes, decisions = {}, collections.defaultdict(dict), {}
    with open(path, errors="replace") as handle:
        for line in handle:
            match = STARTED.search(line)
            if match:
                names[match.group("guid")] = os.path.basename(match.group("path"))
                continue
            match = PASS.search(line)
            if match:
                passes[match.group("guid")][match.group("label")] = {
                    "words": int(match.group("words")),
                    "segments": int(match.group("segments")),
                    "max_gap_s": float(match.group("gap")),
                    "anomaly_count": int(match.group("anomaly")),
                    "anomaly_windows": int(match.group("windows")),
                }
                continue
            match = SELECTED.search(line)
            if match:
                decisions[match.group("guid")] = {
                    "shipped": match.group("label"),
                    "primary_logprob": float(match.group("plp")),
                    "rescue_logprob": float(match.group("rlp")),
                }
    rows = []
    for guid, decision in decisions.items():
        both = passes.get(guid) or {}
        if "primary" not in both or "rescue" not in both:
            continue
        primary = dict(both["primary"], label="primary",
                       mean_logprob=decision["primary_logprob"])
        rescue = dict(both["rescue"], label="rescue",
                      mean_logprob=decision["rescue_logprob"])
        rows.append({"file": names.get(guid, guid), "guid": guid,
                     "shipped": decision["shipped"],
                     "primary": primary, "rescue": rescue})
    rows.sort(key=lambda r: r["file"])
    return rows


def retains_enough_words(primary, candidate):
    """The shipped retention guard, verbatim in behaviour."""
    if primary["words"] <= 0:
        return True
    lost = primary["words"] - candidate["words"]
    if lost <= 0:
        return True
    return (candidate["words"] >= primary["words"] * RESCUE_MIN_WORD_RETENTION
            and lost <= RESCUE_MAX_WORD_LOSS)


def score(entry):
    return entry["anomaly_count"] + entry["anomaly_windows"]


# The shipped ordering, and the three variants the tie-break proposal asked for, as
# named functions so another tool can import the shipped one and apply exactly it.
def pass_sort_key_words(entry):
    """0.6.1: fewest anomalies, then most words, then best mean log-probability."""
    return (score(entry), -entry["words"], -entry["mean_logprob"])


def pass_sort_key_gap_tiebreak(entry):
    """A tie on the anomaly score falls to the smaller uncovered gap."""
    return (score(entry), entry["max_gap_s"], -entry["words"], -entry["mean_logprob"])


def pass_sort_key_gap_first(entry):
    """The uncovered gap ahead of the anomaly score entirely."""
    return (entry["max_gap_s"], score(entry), -entry["words"], -entry["mean_logprob"])


RULES = {}


def _better(primary, rescue, key):
    """The rescue only displaces the primary on a strict win, as `min` does in
    `select_pass` where the primary is always first."""
    return "rescue" if key(rescue) < key(primary) else "primary"


def rule(name):
    def register(fn):
        RULES[name] = fn
        return fn
    return register


@rule("shipped")
def rule_shipped(primary, rescue):
    """0.6.1: retention gate, then fewest anomalies, most words, best logprob; ties to
    the primary because it is first into `min`."""
    if not retains_enough_words(primary, rescue):
        return "primary"
    return _better(primary, rescue, pass_sort_key_words)


@rule("gap_tiebreak")
def rule_gap_tiebreak(primary, rescue):
    """The proposal: identical, except a tie on anomaly score falls to the smaller
    uncovered gap instead of to the primary. The retention gate still runs first."""
    if not retains_enough_words(primary, rescue):
        return "primary"
    return _better(primary, rescue, pass_sort_key_gap_tiebreak)


@rule("gap_tiebreak_no_retention")
def rule_gap_tiebreak_open(primary, rescue):
    """The proposal with the retention gate removed, which is what it would take for the
    tie-break to reach tcf.20250607 at all."""
    return _better(primary, rescue, pass_sort_key_gap_tiebreak)


@rule("gap_first")
def rule_gap_first(primary, rescue):
    """Gap ahead of the anomaly score entirely, retention still gating."""
    if not retains_enough_words(primary, rescue):
        return "primary"
    return _better(primary, rescue, pass_sort_key_gap_first)


def replay(rows):
    out = []
    for row in rows:
        verdicts = {name: fn(row["primary"], row["rescue"]) for name, fn in RULES.items()}
        lost = row["primary"]["words"] - row["rescue"]["words"]
        out.append({
            "file": row["file"],
            "shipped": row["shipped"],
            "primary_words": row["primary"]["words"],
            "rescue_words": row["rescue"]["words"],
            "words_lost": lost,
            "retention_ok": retains_enough_words(row["primary"], row["rescue"]),
            "primary_score": score(row["primary"]),
            "rescue_score": score(row["rescue"]),
            "primary_gap_s": row["primary"]["max_gap_s"],
            "rescue_gap_s": row["rescue"]["max_gap_s"],
            "verdicts": verdicts,
        })
    out.sort(key=lambda r: r["file"])
    return out


# -------------------------------------------------------------------------------- build

def _files_of(path):
    """The `files` map of a sweep result document, keyed by recording name."""
    with open(path) as handle:
        doc = json.load(handle)
    return doc["files"] if isinstance(doc, dict) and "files" in doc else doc


def load_pair(args):
    """The primary and rescue transcript maps, from whichever source was given.

    Two sources exist and the caller picks one. `--fidelity` is a *directory* holding the
    fidelity experiment's `A.json` and `B.json`, which is what the first run of this tool
    used; passing a single JSON file there fails with `NotADirectoryError` and that
    mistake cost a session. `--primary-json` and `--rescue-json` name the two documents
    outright, which is what a decode-only re-run produces and what the error above was
    reaching for. Giving neither, or a directory that is not one, says so plainly.
    """
    if args.primary_json or args.rescue_json:
        if not (args.primary_json and args.rescue_json):
            raise SystemExit(
                "--primary-json and --rescue-json go together; give both or neither")
        return _files_of(args.primary_json), _files_of(args.rescue_json)
    if not args.fidelity:
        raise SystemExit(
            "give either --fidelity DIR (holding A.json and B.json) or both of "
            "--primary-json FILE and --rescue-json FILE")
    if not os.path.isdir(args.fidelity):
        raise SystemExit(
            "--fidelity wants the directory holding A.json and B.json, not a file; "
            "%r is not a directory. To name two transcript documents directly, use "
            "--primary-json and --rescue-json." % args.fidelity)
    return (_files_of(os.path.join(args.fidelity, "A.json")),
            _files_of(os.path.join(args.fidelity, "B.json")))


def build(args):
    def load(path):
        with open(path) as handle:
            return json.load(handle)

    shipped, rescue_cfg = load_pair(args)
    legacy_raw = load(args.legacy)

    name = args.file
    primary_words = norm_words(shipped[name]["transcription"])
    rescue_words = norm_words(rescue_cfg[name]["transcription"])
    legacy = norm_words(legacy_raw.get(name) or "")

    stem = os.path.splitext(name)[0]
    timings = load_timings(os.path.join(args.timings, stem + ".json") if args.timings else None)

    out = {
        "file": name,
        "counts": {
            "primary_words": len(primary_words),
            "rescue_words": len(rescue_words),
            "legacy_words": len(legacy),
            "rescue_minus_primary": len(rescue_words) - len(primary_words),
            "legacy_minus_primary": len(legacy) - len(primary_words),
            "legacy_minus_rescue": len(legacy) - len(rescue_words),
        },
        "timings_words": (timings or {}).get("words"),
        "run_min_words": RUN_MIN_WORDS,
        "run_floor": RUN_FLOOR,
        "legacy_probe_ngram": LEGACY_PROBE_NGRAM,
    }

    runs = []
    for side, size, index, run, anchor in one_sided_runs(primary_words, rescue_words, RUN_FLOOR):
        kind, repeat, covered = classify(run, legacy)
        entry = {
            "side": "primary-only" if side == "left" else "rescue-only",
            "words": size,
            "word_index": index,
            "position": round(index / max(1, len(primary_words if side == "left" else rescue_words)), 4),
            "class": kind,
            "max_self_repeat": repeat,
            "legacy_ngram_coverage": covered,
            "in_legacy": legacy_has(legacy, run),
            "text": " ".join(run),
        }
        entry["start_s"] = stamp(timings, anchor)
        entry["clock"] = clock(entry["start_s"])
        runs.append(entry)
    runs.sort(key=lambda r: (-r["words"], r["word_index"]))
    out["runs"] = runs
    out["runs_at_threshold"] = [r for r in runs if r["words"] >= RUN_MIN_WORDS]
    out["budget"] = deficit_budget(primary_words, rescue_words)
    out["single_word_deficit"] = single_word_deficit(primary_words, rescue_words)

    if args.log:
        out["replay"] = replay(parse_log(args.log))

    if args.candidate:
        candidate = load(args.candidate)["files"]
        out["unpublished_runs"] = {
            k: v.get("rescue_unpublished_run")
            for k, v in sorted(candidate.items())
            if v.get("rescue_attempted")
        }
    if args.primary_only:
        primary_only = load(args.primary_only)["files"]
        candidate = load(args.candidate)["files"] if args.candidate else {}
        recovered = {}
        for k, entry in sorted(primary_only.items()):
            published = candidate.get(k)
            if not published:
                continue
            left = norm_words(entry["transcription"])
            right = norm_words(published["transcription"])
            found = one_sided_runs(left, right, RUN_MIN_WORDS)
            lg = norm_words(legacy_raw.get(k) or "")
            recovered[k] = {
                "primary_only_words": len(left),
                "published_words": len(right),
                "primary_only_gap_s": entry.get("uncovered_max_gap_s"),
                "published_gap_s": published.get("uncovered_max_gap_s"),
                "rescue_only_runs": [
                    {"words": size, "class": classify(run, lg)[0],
                     "text": " ".join(run)[:220]}
                    for side, size, index, run, anchor in found if side == "right"],
                "primary_only_runs": [
                    {"words": size, "class": classify(run, lg)[0],
                     "text": " ".join(run)[:220]}
                    for side, size, index, run, anchor in found if side == "left"],
            }
        out["selected_rescues"] = recovered
    return out


# ------------------------------------------------------------------------------- report

def render(out):
    lines = []
    w = lines.append
    counts = out["counts"]
    w("# The discarded rescue on %s" % out["file"])
    w("")
    w("Word counts are the question, not the answer: primary %d, rescue %d, legacy archive %d."
      % (counts["primary_words"], counts["rescue_words"], counts["legacy_words"]))
    w("")

    w("## Where the word difference lives")
    w("")
    w("Every one-sided stretch between the two decodes, bucketed by its length. Only the")
    w("excess counts, so a five-for-three replacement is two words and not five.")
    w("")
    w("| run length | primary-only runs | primary-only words | rescue-only runs | rescue-only words |")
    w("|---|---|---|---|---|")
    for label, row in out["budget"].items():
        w("| %s | %d | %d | %d | %d |" % (label, row["left_runs"], row["left_words"],
                                          row["right_runs"], row["right_words"]))
    w("")

    w("### The one-word-at-a-time part of it")
    w("")
    w("The word the primary has and the rescue does not, wherever the difference is a")
    w("single word. Counts over the whole recording.")
    w("")
    w("| word | times | word | times | word | times |")
    w("|---|---|---|---|---|---|")
    pairs = out["single_word_deficit"]
    for i in range(0, len(pairs), 3):
        row = pairs[i:i + 3]
        while len(row) < 3:
            row.append(("", ""))
        w("| " + " | ".join("%s | %s" % (a, b) for a, b in row) + " |")
    w("")

    for side, heading in (("primary-only", "What the primary has and the rescue lacks"),
                          ("rescue-only", "What the rescue has and the primary lacks")):
        w("## %s" % heading)
        w("")
        rows = [r for r in out["runs_at_threshold"] if r["side"] == side]
        if not rows:
            w("No contiguous run of %d or more words. The longest is %d."
              % (out["run_min_words"],
                 max([r["words"] for r in out["runs"] if r["side"] == side] or [0])))
            w("")
            continue
        w("| words | at | class | legacy n-gram coverage | text |")
        w("|---|---|---|---|---|")
        for r in rows:
            w("| %d | %s | %s | %s | %s |"
              % (r["words"], r.get("clock", "-"), r["class"],
                 "-" if r["legacy_ngram_coverage"] is None else "%.2f" % r["legacy_ngram_coverage"],
                 r["text"][:160].replace("|", "/")))
        w("")

    w("## The longest of what is left, below the threshold")
    w("")
    w("| side | words | at | class | legacy | text |")
    w("|---|---|---|---|---|---|")
    for r in [x for x in out["runs"] if x["words"] < out["run_min_words"]][:20]:
        w("| %s | %d | %s | %s | %s | %s |"
          % (r["side"], r["words"], r.get("clock", "-"), r["class"],
             "yes" if r["in_legacy"] else "no", r["text"][:100].replace("|", "/")))
    w("")

    if "replay" in out:
        w("## Every rescue in the corpus, under four selection rules")
        w("")
        w("| file | primary | rescue | lost | retention | scores | gaps | shipped | gap tie-break | gap tie-break, no retention | gap first |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in out["replay"]:
            v = r["verdicts"]
            w("| %s | %d | %d | %d | %s | %d v %d | %.1f v %.1f | %s | %s | %s | %s |"
              % (r["file"], r["primary_words"], r["rescue_words"], r["words_lost"],
                 "ok" if r["retention_ok"] else "fails",
                 r["primary_score"], r["rescue_score"],
                 r["primary_gap_s"], r["rescue_gap_s"], r["shipped"],
                 v["gap_tiebreak"], v["gap_tiebreak_no_retention"], v["gap_first"]))
        w("")
        for name in ("gap_tiebreak", "gap_tiebreak_no_retention", "gap_first"):
            changed = [r["file"] for r in out["replay"] if r["verdicts"][name] != r["shipped"]]
            w("- `%s` changes %d of %d decisions%s"
              % (name, len(changed), len(out["replay"]),
                 (": " + ", ".join(changed)) if changed else ""))
        w("")

    if "selected_rescues" in out:
        w("## The eleven selected rescues, primary-only against published")
        w("")
        w("| file | primary-only | published | gap before | gap after | rescue-only runs | primary-only runs |")
        w("|---|---|---|---|---|---|---|")
        for name, row in out["selected_rescues"].items():
            w("| %s | %d | %d | %s | %s | %s | %s |"
              % (name, row["primary_only_words"], row["published_words"],
                 row["primary_only_gap_s"], row["published_gap_s"],
                 ", ".join("%d %s" % (x["words"], x["class"]) for x in row["rescue_only_runs"]) or "none",
                 ", ".join("%d %s" % (x["words"], x["class"]) for x in row["primary_only_runs"]) or "none"))
        w("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity",
                        help="directory holding the fidelity A.json and B.json")
    parser.add_argument("--primary-json",
                        help="a sweep result document to read the primary transcript from")
    parser.add_argument("--rescue-json",
                        help="a sweep result document to read the rescue transcript from")
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--timings")
    parser.add_argument("--candidate")
    parser.add_argument("--primary-only")
    parser.add_argument("--log")
    parser.add_argument("--file", action="append", dest="files",
                        help="a recording to analyse; repeat for several")
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    parser.add_argument("--out-dir",
                        help="write <stem>.json and <stem>.md per file here; required "
                             "when more than one --file is given")
    args = parser.parse_args(argv)

    names = args.files or ["tcf.20250607.mp3"]
    if len(names) > 1 and not args.out_dir:
        raise SystemExit("several --file need --out-dir")
    if len(names) == 1 and not (args.out_dir or (args.out_json and args.out_md)):
        raise SystemExit("give --out-dir, or both --out-json and --out-md")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    for name in names:
        args.file = name
        out = build(args)
        if args.out_dir:
            stem = os.path.splitext(name)[0]
            out_json = os.path.join(args.out_dir, stem + ".json")
            out_md = os.path.join(args.out_dir, stem + ".md")
        else:
            out_json, out_md = args.out_json, args.out_md
        with open(out_json, "w") as handle:
            json.dump(out, handle, indent=1, sort_keys=True)
        with open(out_md, "w") as handle:
            handle.write(render(out))
        print("wrote %s and %s" % (out_json, out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
