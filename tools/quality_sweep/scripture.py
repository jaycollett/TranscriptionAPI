"""A scripture benchmark: the only ground truth this corpus contains.

Every quality comparison up to now has been relative, because no human-corrected
transcript of any of these recordings exists. But when a preacher reads Matthew 6:19-21
aloud, the words he says are knowable independently of any transcript the service ever
produced. That is a reference, and it lands exactly on the material 0.6.0 is weakest at:
the read-aloud passage.

## The caveat, which is not a footnote

**The absolute numbers here are not accuracy.** The preacher's translation is unknown and
almost certainly is not the World English Bible this scores against, so a correct
transcript of an ESV reading is charged for every place the ESV and the WEB differ. A
word error rate of 0.30 on a passage does not mean 30 percent of the words were
mis-transcribed.

What the number is good for is comparison. The same reference scores every configuration,
so the difference between two configurations on the same passage is meaningful even when
neither absolute value is. Read the deltas; do not quote the levels.

Two statistics per passage per configuration:

- `wer`: the minimum word error rate of the reference against any substring of the
  hypothesis transcript. Free start and free end on the hypothesis side, so the passage
  is located rather than assumed to be at a known offset, and the score is not
  contaminated by the rest of the sermon.
- `containment`: the fraction of the reference's 5-grams appearing anywhere in the
  hypothesis. This is the statistic `passage_probe.py` used, kept so the two runs can be
  read together, and it answers the coarser question of whether the passage is present at
  all.

## Discovery

The four known passages were found by hand. The rest are found by scanning transcripts
for a scripture citation and testing whether a long quotation follows it: fetch the cited
chapter, take the cited verse and the ones after it, and measure the WER of that reference
against the window of transcript following the citation. A citation the preacher merely
mentions scores badly; one he then reads scores well. The verse range is grown greedily
from the cited verse for as long as the WER keeps improving, so the benchmark entry covers
what was actually read rather than what was cited.

Reference text comes from getbible.net, which serves the World English Bible and the
American Standard Version, both public domain. Chapters are cached in
`scripture_cache.json` so a rerun needs no network.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm import norm_words  # noqa: E402

NGRAM = 5
API = "https://api.getbible.net/v2/{translation}/{book}/{chapter}.json"

# getbible numbers books 1-66 in canonical order. The spoken forms Whisper produces are
# what the citation regex has to match, so ordinals appear as words and as digits.
BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges",
    "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes",
    "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea",
    "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
    "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
    "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians",
    "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus",
    "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
    "Jude", "Revelation",
]
BOOK_NUMBER = {name.lower(): i + 1 for i, name in enumerate(BOOKS)}
# Spoken and written forms that are not the canonical name. "Psalm 91" is far commoner
# from a pulpit than "Psalms 91", and Whisper writes what it hears.
ALIASES = {"psalm": "Psalms", "song of songs": "Song of Solomon",
           "revelations": "Revelation", "canticles": "Song of Solomon"}
BOOK_NUMBER.update({alias: BOOK_NUMBER[target.lower()] for alias, target in ALIASES.items()})
# Books whose spoken name has no ordinal ambiguity, plus the ordinal-carrying ones handled
# separately in the citation pattern.
PLAIN_BOOKS = [b for b in BOOKS if not b[0].isdigit()] + list(ALIASES)
ORDINAL_BOOKS = sorted({b.split(" ", 1)[1] for b in BOOKS if b[0].isdigit()})

_ORDINAL = r"(?:1|2|3|first|second|third|1st|2nd|3rd|i|ii|iii)"
CITATION = re.compile(
    r"\b(?:(?P<ord>" + _ORDINAL + r")\s+(?P<obook>" + "|".join(ORDINAL_BOOKS) + r")"
    r"|(?P<book>" + "|".join(sorted(PLAIN_BOOKS, key=len, reverse=True)) + r"))"
    r"[\s,]+(?P<chapter>\d{1,3})\s*[:.]\s*(?P<verse>\d{1,3})",
    re.IGNORECASE,
)
ORDINAL_VALUE = {"1": 1, "first": 1, "1st": 1, "i": 1,
                 "2": 2, "second": 2, "2nd": 2, "ii": 2,
                 "3": 3, "third": 3, "3rd": 3, "iii": 3}


# ---------------------------------------------------------------------------------
# Reference text
# ---------------------------------------------------------------------------------
class Bible:
    """Chapter text from getbible.net, cached on disk so a rerun needs no network."""

    def __init__(self, cache_path, offline=False):
        self.cache_path = cache_path
        self.offline = offline
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as handle:
                self.cache = json.load(handle)
        self.dirty = False

    def chapter(self, translation, book_number, chapter):
        key = f"{translation}/{book_number}/{chapter}"
        if key in self.cache:
            return self.cache[key]
        if self.offline:
            return None
        url = API.format(translation=translation, book=book_number, chapter=chapter)
        # getbible refuses urllib's default agent with a 403, so say who we are.
        request = urllib.request.Request(url, headers={
            "User-Agent": "TranscriptionAPI-quality-sweep/1.0",
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                doc = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            print(f"  fetch failed {key}: {exc}", file=sys.stderr)
            self.cache[key] = None
            self.dirty = True
            return None
        verses = {str(v["verse"]): v["text"] for v in doc.get("verses", [])}
        self.cache[key] = verses
        self.dirty = True
        time.sleep(0.2)
        return verses

    def passage(self, translation, book_number, chapter, first, last):
        verses = self.chapter(translation, book_number, chapter)
        if not verses:
            return None
        text = " ".join(verses[str(v)] for v in range(first, last + 1) if str(v) in verses)
        return text or None

    def save(self):
        if self.dirty:
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w") as handle:
                json.dump(self.cache, handle, indent=0, sort_keys=True)
            os.replace(tmp, self.cache_path)
            self.dirty = False


# ---------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------
def infix_wer(reference, hypothesis):
    """Minimum word error rate of `reference` against any substring of `hypothesis`.

    Ordinary edit distance with the hypothesis's prefix and suffix free: row 0 is all
    zeros, so an alignment may begin anywhere, and the answer is the minimum of the last
    row, so it may end anywhere. This is what makes the score a measurement of the
    passage rather than of the sermon it sits in: a 60-word reading inside an 8000-word
    transcript is located by the alignment, not assumed to be at a known offset.

    Returns (wer, hypothesis_end_index). A reference that is absent scores 1.0, because
    deleting every reference word is then the cheapest alignment.
    """
    if not reference:
        return None, None
    if not hypothesis:
        return 1.0, None
    previous = [0] * (len(hypothesis) + 1)
    for i in range(1, len(reference) + 1):
        current = [i] + [0] * len(hypothesis)
        ref_word = reference[i - 1]
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if ref_word == hypothesis[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    best = min(previous)
    # Cap at 1.0: deleting the whole reference always costs len(reference), so a score
    # above that means the alignment found nothing worth keeping.
    return round(min(best / len(reference), 1.0), 4), previous.index(best)


def containment(reference, hypothesis, n=NGRAM):
    """Fraction of the reference's n-grams that appear anywhere in the hypothesis."""
    if len(reference) < n:
        return None
    ref_grams = {tuple(reference[i:i + n]) for i in range(len(reference) - n + 1)}
    hyp_grams = {tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1)}
    return round(len(ref_grams & hyp_grams) / len(ref_grams), 4)


def score_passage(reference_words, hypothesis_words):
    wer, end = infix_wer(reference_words, hypothesis_words)
    return {
        "wer": wer,
        "containment": containment(reference_words, hypothesis_words),
        "ref_words": len(reference_words),
        "hyp_end": end,
    }


# ---------------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------------
def citations(text):
    """Every scripture citation in `text`, as (book_number, chapter, verse, char_offset)."""
    found = []
    for match in CITATION.finditer(text):
        if match.group("obook"):
            ordinal = ORDINAL_VALUE.get((match.group("ord") or "").lower())
            if not ordinal:
                continue
            name = f"{ordinal} {match.group('obook')}"
        else:
            name = match.group("book")
        number = BOOK_NUMBER.get(name.lower())
        if number is None and name.lower() == "psalm":
            number = BOOK_NUMBER["psalms"]
        if number is None:
            continue
        found.append((number, int(match.group("chapter")), int(match.group("verse")),
                      match.end()))
    return found


def grow_passage(bible, translation, book, chapter, verse, window_words,
                 max_verses=12, wer_bar=0.45):
    """Extend the reference from `verse` for as long as the alignment keeps improving.

    A citation is a pointer, not a length: the preacher says "Matthew 6:19" and then reads
    three verses, or one, or none. Growing the reference greedily and keeping the longest
    range that still aligns well is what turns a citation into a benchmark entry covering
    what was actually read.
    """
    best = None
    for last in range(verse, verse + max_verses):
        text = bible.passage(translation, book, chapter, verse, last)
        if not text:
            break
        reference = norm_words(text)
        if len(reference) < 12:
            continue
        wer, _ = infix_wer(reference, window_words)
        if wer is None or wer > wer_bar:
            # Once the alignment falls apart, adding more verses will not repair it.
            if best is not None:
                break
            continue
        if best is None or len(reference) > best["ref_words"]:
            best = {"first_verse": verse, "last_verse": last, "wer": wer,
                    "ref_words": len(reference), "text": text}
    return best


def discover(transcripts, bible, translation, window=450, wer_bar=0.45, min_words=25):
    """Find read-aloud passages: a citation followed by text that matches the passage."""
    hits = []
    for name, text in sorted(transcripts.items()):
        seen = set()
        words_all = norm_words(text)
        for book, chapter, verse, offset in citations(text):
            key = (book, chapter, verse)
            if key in seen:
                continue
            seen.add(key)
            # The words following the citation, which is where a reading would be.
            after = norm_words(text[offset:offset + window * 8])[:window]
            if len(after) < 30:
                continue
            best = grow_passage(bible, translation, book, chapter, verse, after,
                                wer_bar=wer_bar)
            if not best or best["ref_words"] < min_words:
                continue
            # Score against the whole transcript too: a passage read a few sentences
            # after the citation still counts, and the window score can understate it.
            whole = infix_wer(norm_words(best["text"]), words_all)[0]
            hits.append({
                "file": name, "book": book, "book_name": BOOKS[book - 1],
                "chapter": chapter, "first_verse": best["first_verse"],
                "last_verse": best["last_verse"], "ref_words": best["ref_words"],
                "window_wer": best["wer"], "whole_wer": whole,
                "label": f"{BOOKS[book - 1]} {chapter}:{best['first_verse']}"
                         f"-{best['last_verse']}",
            })
            print(f"  {name}: {hits[-1]['label']} "
                  f"{best['ref_words']}w wer={best['wer']:.3f}", flush=True)
        bible.save()
    return hits


# ---------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------
def load_transcripts(spec):
    """Read transcripts from a directory of .txt files or a {name: text} JSON."""
    if os.path.isdir(spec):
        out = {}
        for entry in sorted(os.listdir(spec)):
            if entry.endswith(".txt"):
                with open(os.path.join(spec, entry)) as handle:
                    out[entry[:-4] + ".mp3"] = handle.read()
        return out
    with open(spec) as handle:
        doc = json.load(handle)
    if "files" in doc:  # a candidate_pass.json
        return {k: v.get("transcription", "") for k, v in doc["files"].items()
                if v.get("transcription")}
    return doc


def cmd_discover(args):
    bible = Bible(args.cache, offline=args.offline)
    transcripts = {}
    for spec in args.transcripts:
        for name, text in load_transcripts(spec).items():
            # Prefer the first source given: legacy first means a passage 0.6.0 dropped
            # is still discoverable.
            transcripts.setdefault(name, text)
    if args.only:
        keep = set(json.load(open(args.only)))
        transcripts = {k: v for k, v in transcripts.items() if k in keep}
    print(f"scanning {len(transcripts)} transcripts")
    hits = discover(transcripts, bible, args.translation, wer_bar=args.wer_bar)
    bible.save()
    with open(args.out, "w") as handle:
        json.dump(hits, handle, indent=1, sort_keys=True)
    print(f"{len(hits)} passages -> {args.out}")
    return 0


def cmd_score(args):
    bible = Bible(args.cache, offline=args.offline)
    with open(args.passages) as handle:
        passages = json.load(handle)
    configs = {}
    for spec in args.config:
        label, path = spec.split("=", 1)
        configs[label] = load_transcripts(path)

    rows = []
    for entry in passages:
        text = bible.passage(args.translation, entry["book"], entry["chapter"],
                             entry["first_verse"], entry["last_verse"])
        if not text:
            print(f"  no reference for {entry['label']}", file=sys.stderr)
            continue
        reference = norm_words(text)
        row = {"file": entry["file"], "label": entry["label"],
               "ref_words": len(reference), "configs": {}}
        for label, transcripts in configs.items():
            hypothesis = transcripts.get(entry["file"])
            if hypothesis is None:
                row["configs"][label] = {"missing": True}
                continue
            row["configs"][label] = score_passage(reference, norm_words(hypothesis))
        rows.append(row)
    bible.save()

    doc = {"translation": args.translation, "configs": sorted(configs), "rows": rows}
    with open(args.out, "w") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
    with open(args.out_md, "w") as handle:
        handle.write(render(doc))
    print(f"{len(rows)} passages x {len(configs)} configs -> {args.out_md}")
    return 0


def render(doc):
    labels = doc["configs"]
    out = ["# Scripture benchmark", ""]
    out.append(f"Reference translation: **{doc['translation']}**. "
               "The preacher's translation is unknown and is probably not this one, so "
               "**the absolute word error rates below are not accuracy** and must not be "
               "quoted as such: a correct transcript of an ESV reading is charged for "
               "every place the ESV and this reference differ. The same reference scores "
               "every configuration, so the *differences between configurations* on the "
               "same passage are meaningful.")
    out.append("")
    out.append("## Word error rate over the passage stretch (lower is better)")
    out.append("")
    out.append("| file | passage | ref words | " + " | ".join(labels) + " |")
    out.append("|---|---|---|" + "---|" * len(labels))
    for row in doc["rows"]:
        cells = []
        for label in labels:
            entry = row["configs"].get(label) or {}
            cells.append("-" if entry.get("missing") else f"{entry['wer']:.3f}")
        out.append(f"| {row['file']} | {row['label']} | {row['ref_words']} | "
                   + " | ".join(cells) + " |")
    out.append("")
    out.append("## Containment: is the passage present at all (higher is better)")
    out.append("")
    out.append("| file | passage | " + " | ".join(labels) + " |")
    out.append("|---|---|" + "---|" * len(labels))
    for row in doc["rows"]:
        cells = []
        for label in labels:
            entry = row["configs"].get(label) or {}
            value = entry.get("containment")
            cells.append("-" if entry.get("missing") or value is None else f"{value:.3f}")
        out.append(f"| {row['file']} | {row['label']} | " + " | ".join(cells) + " |")
    out.append("")

    out.append("## Summary")
    out.append("")
    out.append("| config | mean WER | median WER | passages present (containment >= 0.8) |")
    out.append("|---|---|---|---|")
    for label in labels:
        values = [r["configs"][label]["wer"] for r in doc["rows"]
                  if not r["configs"].get(label, {}).get("missing")]
        cont = [r["configs"][label].get("containment") for r in doc["rows"]
                if not r["configs"].get(label, {}).get("missing")]
        cont = [c for c in cont if c is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        median = sorted(values)[len(values) // 2]
        out.append(f"| {label} | {mean:.3f} | {median:.3f} | "
                   f"{sum(1 for c in cont if c >= 0.8)} of {len(cont)} |")
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scripture_cache.json"))
    parser.add_argument("--translation", default="web")
    parser.add_argument("--offline", action="store_true",
                        help="use only the cache; never reach the network")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("discover")
    d.add_argument("--transcripts", action="append", required=True)
    d.add_argument("--only", help="JSON list of filenames to restrict the scan to")
    d.add_argument("--wer-bar", type=float, default=0.45)
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_discover)

    s = sub.add_parser("score")
    s.add_argument("--passages", required=True)
    s.add_argument("--config", action="append", required=True,
                   help="label=path, where path is a transcripts dir or a JSON")
    s.add_argument("--out", required=True)
    s.add_argument("--out-md", required=True)
    s.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
