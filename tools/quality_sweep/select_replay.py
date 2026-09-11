"""Replay an alternative selection ordering over the passes 0.6.2 retained on disk.

`RESCUE_TRANSCRIPT_DIR` keeps both decoded passes whenever a rescue runs, so the question
"what would a different `select_pass` have published" is answerable offline, by replaying
the rule over transcripts that already exist, instead of by decoding the corpus again.

Two orderings are compared, both keeping the retention floor as an eligibility gate and
both keeping the anomaly score first:

- `current`     anomaly score, then most words, then best mean log-probability.
- `content_first`  anomaly score, then most **corroborated contiguous content**, then most
  words, then best mean log-probability.

Corroborated contiguous content is the length of the longest run of consecutive words a
pass has that the other pass does not, and that also appears in the legacy archive. The
corroboration is what makes it evidence rather than length: a decoder can produce a long
run of invented words, but it cannot invent a run that independently matches what a
different decoder wrote for the same seconds years earlier. On the scripture readings this
is precisely the signal that separates a rescue holding Exodus 24 from one holding filler.

A flip is then judged on content and not on count, by 5-gram recall against the legacy
transcript: publishing the other pass is an improvement only if recall goes up.

    python3 select_replay.py --retained /home/jay/sweep/service/rescue_transcripts \
        --legacy-db /home/jay/sweep/legacy.db --out-json replay.json --out-md replay.md
"""

import argparse
import json
import os
import sqlite3
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from norm import norm_words  # noqa: E402

# The retention guard, mirrored from the service so the replay filters candidates exactly
# as the service does. Both a ratio and an absolute cap; either alone scales wrongly.
RESCUE_MIN_WORD_RETENTION = 0.99
RESCUE_MAX_WORD_LOSS = 40

# A run shorter than this is not contiguous content, it is a coincidence of phrasing.
MIN_RUN_WORDS = 8
NGRAM = 5


def retains_enough_words(primary_words, candidate_words):
    if primary_words <= 0:
        return True
    lost = primary_words - candidate_words
    if lost <= 0:
        return True
    return lost <= RESCUE_MAX_WORD_LOSS and candidate_words >= primary_words * RESCUE_MIN_WORD_RETENTION


def current_key(record):
    logprob = record.get("mean_logprob")
    return (
        record["anomaly_count"] + record["anomaly_windows"],
        -record["words"],
        -(logprob if logprob is not None else float("-inf")),
    )


def content_first_key(record):
    logprob = record.get("mean_logprob")
    return (
        record["anomaly_count"] + record["anomaly_windows"],
        -record.get("corroborated_run", 0),
        -record["words"],
        -(logprob if logprob is not None else float("-inf")),
    )


def longest_unique_run(words, other_words, corroborating=None, min_run=MIN_RUN_WORDS):
    """Longest run of consecutive `words` absent from `other_words`.

    When `corroborating` is given, the run must also appear there, which is what turns a
    long stretch of output into evidence that content was recovered.
    """
    other = {tuple(other_words[i:i + NGRAM]) for i in range(len(other_words) - NGRAM + 1)}
    corroborator = None
    if corroborating is not None:
        corroborator = {
            tuple(corroborating[i:i + NGRAM])
            for i in range(len(corroborating) - NGRAM + 1)
        }

    best, run = 0, 0
    best_span = None
    span_start = 0
    for i in range(len(words) - NGRAM + 1):
        gram = tuple(words[i:i + NGRAM])
        novel = gram not in other
        backed = corroborator is None or gram in corroborator
        if novel and backed:
            if run == 0:
                span_start = i
            run += 1
            if run > best:
                best, best_span = run, (span_start, i + NGRAM)
        else:
            run = 0
    length = best + NGRAM - 1 if best else 0
    if length < min_run:
        return 0, None
    return length, best_span


def legacy_recall(legacy_words, candidate_words):
    if not legacy_words:
        return None
    legacy = {tuple(legacy_words[i:i + NGRAM]) for i in range(len(legacy_words) - NGRAM + 1)}
    cand = {tuple(candidate_words[i:i + NGRAM]) for i in range(len(candidate_words) - NGRAM + 1)}
    return round(len(legacy & cand) / len(legacy), 5) if legacy else None


def load_legacy(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = {}
    for url, text in con.execute(
        "select orig_audio_url, english_transcription from sermon_metadata"
    ):
        if not text:
            continue
        name = urllib.parse.unquote(os.path.basename(urllib.parse.urlsplit(url or "").path))
        out[name] = norm_words(text)
    return out


def replay_one(record, legacy):
    passes = [dict(p) for p in record["passes"]]
    if len(passes) < 2:
        return None
    for entry in passes:
        entry["norm"] = norm_words(entry.get("transcription") or "")
    primary = passes[0]
    legacy_words = legacy.get(record.get("file") or "")

    for entry in passes:
        others = [p for p in passes if p is not entry]
        other_words = others[0]["norm"] if others else []
        run, span = longest_unique_run(entry["norm"], other_words, legacy_words)
        entry["corroborated_run"] = run
        entry["corroborated_span"] = span
        entry["uncorroborated_run"], _ = longest_unique_run(entry["norm"], other_words, None)
        entry["legacy_recall"] = legacy_recall(legacy_words, entry["norm"])

    eligible = [primary] + [
        p for p in passes[1:] if retains_enough_words(primary["words"], p["words"])
    ]
    chosen_now = min(eligible, key=current_key)
    chosen_new = min(eligible, key=content_first_key)

    return {
        "guid": record.get("guid"),
        "file": record.get("file"),
        "published_label": record.get("published"),
        "current": chosen_now["label"],
        "content_first": chosen_new["label"],
        "flips": chosen_now["label"] != chosen_new["label"],
        "eligible": [p["label"] for p in eligible],
        "recall_current": chosen_now["legacy_recall"],
        "recall_content_first": chosen_new["legacy_recall"],
        "recall_gain": (
            round(chosen_new["legacy_recall"] - chosen_now["legacy_recall"], 5)
            if chosen_now["legacy_recall"] is not None and chosen_new["legacy_recall"] is not None
            else None
        ),
        "passes": [
            {
                "label": p["label"],
                "words": p["words"],
                "anomaly_score": p["anomaly_count"] + p["anomaly_windows"],
                "corroborated_run": p["corroborated_run"],
                "uncorroborated_run": p["uncorroborated_run"],
                "legacy_recall": p["legacy_recall"],
                "seed": p.get("seed"),
                "eligible": p["label"] in [e["label"] for e in eligible],
                "excerpt": (
                    " ".join(p["norm"][p["corroborated_span"][0]:p["corroborated_span"][1]][:28])
                    if p["corroborated_span"] else ""
                ),
            }
            for p in passes
        ],
    }


def summarise(rows):
    flips = [r for r in rows if r["flips"]]
    better = [r for r in flips if (r["recall_gain"] or 0) > 0.0005]
    worse = [r for r in flips if (r["recall_gain"] or 0) < -0.0005]
    same = [r for r in flips if r not in better and r not in worse]
    return {
        "rescues": len(rows),
        "flips": len(flips),
        "flips_better": len(better),
        "flips_worse": len(worse),
        "flips_indifferent": len(same),
        "better_files": [r["file"] for r in better],
        "worse_files": [r["file"] for r in worse],
        "total_recall_gain": round(sum(r["recall_gain"] or 0 for r in flips), 5),
    }


def render_markdown(summary, rows):
    out = ["# Replaying `content_first` over the retained passes", ""]
    out.append(
        f"{summary['rescues']} rescues retained. `content_first` would publish a different "
        f"pass on **{summary['flips']}** of them: {summary['flips_better']} better by legacy "
        f"recall, {summary['flips_worse']} worse, {summary['flips_indifferent']} indifferent. "
        f"Net recall change {summary['total_recall_gain']}."
    )
    out.append("")
    out.append("| file | current | content_first | flips | recall now | recall after | gain |")
    out.append("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -(r["recall_gain"] or 0)):
        out.append(
            f"| {r['file']} | {r['current']} | {r['content_first']} | "
            f"{'yes' if r['flips'] else 'no'} | {r['recall_current']} | "
            f"{r['recall_content_first']} | {r['recall_gain']} |"
        )
    out.append("")
    out.append("## The corroborated runs that drive each flip")
    out.append("")
    for r in rows:
        if not r["flips"]:
            continue
        out.append(f"**{r['file']}** publishes the {r['content_first']} instead of the "
                   f"{r['current']}:")
        for p in r["passes"]:
            out.append(f"- {p['label']}: {p['words']} words, anomaly {p['anomaly_score']}, "
                       f"corroborated run {p['corroborated_run']}, recall {p['legacy_recall']}"
                       + (f" — \"{p['excerpt']}...\"" if p["excerpt"] else ""))
        out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained", required=True)
    parser.add_argument("--legacy-db", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args(argv)

    legacy = load_legacy(args.legacy_db)
    rows = []
    for name in sorted(os.listdir(args.retained)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(args.retained, name)) as handle:
            record = json.load(handle)
        row = replay_one(record, legacy)
        if row:
            rows.append(row)

    summary = summarise(rows)
    with open(args.out_json, "w") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=1)
        handle.write("\n")
    with open(args.out_md, "w") as handle:
        handle.write(render_markdown(summary, rows))
        handle.write("\n")
    print(render_markdown(summary, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
