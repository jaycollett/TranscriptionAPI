"""Score the proposed replacement for the rescue retention guard across the corpus.

0.6.1 refuses a rescue that loses more than 1 percent or more than 40 words of the
primary, whichever bites first. The proposal is to drop both and refuse a rescue only
when it loses a **contiguous run of N or more words that the legacy archive
corroborates**, on the argument that a word count cannot tell a dropped scripture
reading from four hundred dropped instances of "um".

A rule is only worth having if it is right on the whole corpus, so this replays all
twenty rescues of the 0.6.1 run under the shipped rule and under the proposal at several
values of N, and reports every file whose decision changes.

The evidence per file is two transcripts, from whichever source has them:

  * `--fidelity DIR`      the 32-file fidelity experiment. `A.json` is 0.6.0 as shipped,
                          a beam search from temperature 0.0 and therefore the same
                          primary 0.6.1 published; `B.json` is a whole-file decode with
                          the ladder from 0.2, which is the rescue configuration but a
                          different unseeded draw.
  * `--passes DIR`        one JSON per job from `RESCUE_TRANSCRIPT_DIR`, holding both
                          passes of a single decode. This is the matched pair and is
                          preferred wherever it exists.
  * `--primary-only` and `--candidate`
                          the two corpus runs. For a file whose rescue was SELECTED,
                          `candidate_pass.json` is the rescue and `primary_only.json`
                          is the primary, which is the only pair that exists for those.

`P` is the longest contiguous run the primary holds, the rescue lacks, and the legacy
archive corroborates. `R` is the same the other way round. The proposal refuses the
rescue exactly when `P >= N`; everything after that is the shipped ordering, so a rescue
still has to beat the primary on anomalies, then words, then log-probability.

    python3 tools/quality_sweep/retention_threshold.py \
        --log /home/jay/sweep/rc061.log \
        --legacy /home/jay/sweep/legacy_text.json \
        --fidelity /home/jay/sweep/fidelity \
        --passes /home/jay/sweep/deficit/redecode/passes \
        --primary-only /home/jay/sweep/rc061/primary_only.json \
        --candidate /home/jay/sweep/rc061/candidate_pass.json \
        --out-json retention_threshold.json --out-md retention_threshold.md
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from norm import norm_words  # noqa: E402
from rescue_deficit import (  # noqa: E402
    classify,
    one_sided_runs,
    parse_log,
    pass_sort_key_words,
    retains_enough_words,
    score,
)

# The values of N the analysis was asked to try.
THRESHOLDS = (8, 12, 20, 30)
# Below this a one-sided run is a wording choice, not a passage; the floor the rest of
# this work uses, so the two are comparable. Runs shorter than this are not searched.
SEARCH_FLOOR = 4


def longest_corroborated(left, right, legacy):
    """(P, R): the longest legacy-corroborated one-sided run on each side.

    A run the legacy archive does not have is not evidence of anything, because the two
    decodes disagreeing proves only that they disagree. A run that repeats itself is a
    loop and is not content however long it is; `classify` rules both out.
    """
    best = {"left": (0, ""), "right": (0, "")}
    for side, size, _index, run, _anchor in one_sided_runs(left, right, SEARCH_FLOOR):
        kind, _repeat, _covered = classify(run, legacy)
        if kind != "real content":
            continue
        if size > best[side][0]:
            best[side] = (size, " ".join(run))
    return best["left"], best["right"]


def load_pairs(args, wanted):
    """`{file: (primary_words, rescue_words, source)}` for every rescue we can pair up.

    Preference order is the matched pair first, then the corpus pair, then the fidelity
    pair, because a matched pair is the decode that actually happened and the fidelity
    rescue is a different draw.
    """
    pairs = {}

    def keep(name, primary, rescue, source):
        if name in wanted and name not in pairs:
            pairs[name] = (norm_words(primary), norm_words(rescue), source)

    if args.passes:
        for path in sorted(glob.glob(os.path.join(args.passes, "*.json"))):
            with open(path) as handle:
                doc = json.load(handle)
            by_label = {p["label"]: p["transcription"] for p in doc["passes"]}
            if "primary" in by_label and "rescue" in by_label:
                keep(doc["file"], by_label["primary"], by_label["rescue"], "matched pair")

    if args.primary_only and args.candidate:
        with open(args.primary_only) as handle:
            primary_only = json.load(handle)["files"]
        with open(args.candidate) as handle:
            candidate = json.load(handle)["files"]
        for name, entry in primary_only.items():
            published = candidate.get(name)
            if not published:
                continue
            # Only meaningful where the rescue was published: otherwise both documents
            # hold the same primary and the diff is empty by construction.
            if name in wanted and wanted[name] == "rescue":
                keep(name, entry["transcription"], published["transcription"], "corpus pair")

    if args.fidelity:
        with open(os.path.join(args.fidelity, "A.json")) as handle:
            shipped = json.load(handle)["files"]
        with open(os.path.join(args.fidelity, "B.json")) as handle:
            rescue_cfg = json.load(handle)["files"]
        for name in sorted(set(shipped) & set(rescue_cfg)):
            keep(name, shipped[name]["transcription"], rescue_cfg[name]["transcription"],
                 "fidelity, different draw")

    return pairs


def decide_shipped(primary, rescue):
    """0.6.1 exactly: the retention guard as a hard pre-filter, then the ordering."""
    if not retains_enough_words(primary, rescue):
        return "primary"
    return "rescue" if pass_sort_key_words(rescue) < pass_sort_key_words(primary) else "primary"


def decide_proposed(primary, rescue, deficit_run, threshold):
    """The proposal: the same ordering, gated on lost contiguous content instead.

    `deficit_run` is P. `None` means the file has no transcript pair, so the rule cannot
    be evaluated and the caller is told rather than guessed at.
    """
    if deficit_run is None:
        return None
    if deficit_run >= threshold:
        return "primary"
    return "rescue" if pass_sort_key_words(rescue) < pass_sort_key_words(primary) else "primary"


def build(args):
    rows = parse_log(args.log)
    wanted = {r["file"]: r["shipped"] for r in rows}
    with open(args.legacy) as handle:
        legacy_raw = json.load(handle)
    pairs = load_pairs(args, wanted)

    out = []
    for row in rows:
        name = row["file"]
        pair = pairs.get(name)
        entry = {
            "file": name,
            "shipped": row["shipped"],
            "primary_words": row["primary"]["words"],
            "rescue_words": row["rescue"]["words"],
            "words_lost": row["primary"]["words"] - row["rescue"]["words"],
            "retention_ok": retains_enough_words(row["primary"], row["rescue"]),
            "primary_score": score(row["primary"]),
            "rescue_score": score(row["rescue"]),
            "primary_gap_s": row["primary"]["max_gap_s"],
            "rescue_gap_s": row["rescue"]["max_gap_s"],
            "source": None,
            "primary_only_run": None,
            "primary_only_text": None,
            "rescue_only_run": None,
            "rescue_only_text": None,
            "verdicts": {},
        }
        if pair:
            left, right, source = pair
            legacy = norm_words(legacy_raw.get(name) or "")
            (p_words, p_text), (r_words, r_text) = longest_corroborated(left, right, legacy)
            entry.update({
                "source": source,
                "primary_only_run": p_words,
                "primary_only_text": p_text[:240],
                "rescue_only_run": r_words,
                "rescue_only_text": r_text[:240],
                "pair_primary_words": len(left),
                "pair_rescue_words": len(right),
            })
        for n in THRESHOLDS:
            entry["verdicts"]["N=%d" % n] = decide_proposed(
                row["primary"], row["rescue"], entry["primary_only_run"], n)
        entry["verdicts"]["shipped_replay"] = decide_shipped(row["primary"], row["rescue"])
        out.append(entry)
    return {"thresholds": list(THRESHOLDS), "search_floor": SEARCH_FLOOR, "files": out}


def render(doc):
    lines = []
    w = lines.append
    rows = doc["files"]
    w("# The retention guard against a contiguous-content rule")
    w("")
    w("`P` is the longest run the primary holds, the rescue lacks, and the legacy archive")
    w("corroborates. `R` is the same the other way round. The proposal refuses the rescue")
    w("when `P >= N` and otherwise applies the shipped ordering unchanged.")
    w("")
    w("| file | primary | rescue | lost | retention | P | R | evidence | shipped | "
      + " | ".join("N=%d" % n for n in doc["thresholds"]) + " |")
    w("|---|---|---|---|---|---|---|---|---|" + "---|" * len(doc["thresholds"]))
    for r in rows:
        w("| %s | %d | %d | %d | %s | %s | %s | %s | %s | %s |"
          % (r["file"], r["primary_words"], r["rescue_words"], r["words_lost"],
             "ok" if r["retention_ok"] else "fails",
             "-" if r["primary_only_run"] is None else r["primary_only_run"],
             "-" if r["rescue_only_run"] is None else r["rescue_only_run"],
             r["source"] or "none", r["shipped"],
             " | ".join(r["verdicts"]["N=%d" % n] or "?" for n in doc["thresholds"])))
    w("")
    for n in doc["thresholds"]:
        key = "N=%d" % n
        flips = [r for r in rows if r["verdicts"][key] and r["verdicts"][key] != r["shipped"]]
        unknown = [r["file"] for r in rows if r["verdicts"][key] is None]
        w("- **N=%d** changes %d of %d decisions%s%s"
          % (n, len(flips), len(rows),
             (": " + ", ".join("%s %s to %s" % (f["file"], f["shipped"], f["verdicts"][key])
                               for f in flips)) if flips else "",
             (" (not evaluable: " + ", ".join(unknown) + ")") if unknown else ""))
    w("")
    w("## What each flip costs or buys")
    w("")
    w("| file | direction | P | the primary-only run | R | the rescue-only run |")
    w("|---|---|---|---|---|---|")
    for r in rows:
        if r["primary_only_run"] is None:
            continue
        flipped = sorted(n for n in doc["thresholds"]
                         if r["verdicts"]["N=%d" % n] != r["shipped"])
        if not flipped:
            continue
        w("| %s | %s to %s at N=%s | %d | %s | %d | %s |"
          % (r["file"], r["shipped"],
             r["verdicts"]["N=%d" % flipped[0]],
             ",".join(str(n) for n in flipped),
             r["primary_only_run"], (r["primary_only_text"] or "-")[:140].replace("|", "/"),
             r["rescue_only_run"], (r["rescue_only_text"] or "-")[:140].replace("|", "/")))
    w("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--fidelity")
    parser.add_argument("--passes")
    parser.add_argument("--primary-only")
    parser.add_argument("--candidate")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args(argv)

    doc = build(args)
    with open(args.out_json, "w") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
    with open(args.out_md, "w") as handle:
        handle.write(render(doc))
    print("wrote %s and %s" % (args.out_json, args.out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
