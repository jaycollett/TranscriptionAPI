"""Re-score the two passages whose isolated-clip reference was too short to be usable.

`passage_probe.py` builds each reference by decoding the passage's own seconds in
isolation. That works when the isolated decode recovers the passage, which it did for four
of the six, but on `tcf.20210210` it returned 23 words and on `tcf.20250607` only 9, so
those two references described almost nothing and their containment scores were flat
across every config.

Both have a better reference available. The two regression files lost a contiguous run
that is present in the legacy transcript, so the legacy run itself is the reference: it is
independent of this release entirely. `tcf.20250607` needs the clip re-encoded by ffmpeg
rather than sliced out of the decoded waveform, which is how the service was fed it when
it returned the full 102 words.
"""

import argparse
import difflib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from norm import norm_words  # noqa: E402
from passage_probe import CONFIGS, containment, decode, window_word_rates  # noqa: E402

# The two files whose reference comes from the legacy transcript, and the passage each
# lost, identified by the first words of the run the diff found.
LEGACY_REFERENCED = {
    "tcf.20210210.mp3": "jesus aware of this withdrew",
    "tcf.20210217.mp3": "do not lay up for yourself",
}
FFMPEG_CLIP = {"tcf.20250607.mp3": (76.35, 101.10)}


def legacy_text(db_path, filename):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    for url, text in con.execute(
        "select orig_audio_url, english_transcription from sermon_metadata"
    ):
        name = urllib.parse.unquote(os.path.basename(urllib.parse.urlsplit(url or "").path))
        if name == filename:
            return text or ""
    return ""


def missing_run(legacy_words, new_words, opener):
    """The longest run the new transcript lost, preferring the one starting at `opener`."""
    matcher = difflib.SequenceMatcher(a=legacy_words, b=new_words, autojunk=False)
    runs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace") and (i2 - i1) > (j2 - j1):
            runs.append((i2 - i1, i1, i2))
    if not runs:
        return None
    opener_words = opener.split()
    for _, i1, i2 in sorted(runs, reverse=True):
        if legacy_words[i1:i1 + len(opener_words)] == opener_words:
            return legacy_words[i1:i2]
    return legacy_words[sorted(runs, reverse=True)[0][1]:sorted(runs, reverse=True)[0][2]]


def ffmpeg_clip_reference(model, audio_dir, name, start, end, pad=3.0):
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp3")
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", str(max(0.0, start - pad)), "-t", str(end - start + 2 * pad),
            "-i", os.path.join(audio_dir, name), "-c:a", "libmp3lame", "-q:a", "4", clip,
        ], check=True)
        return decode(model, clip, CONFIGS["control"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--legacy-db", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--download-root", default="/app/models/whisper")
    args = parser.parse_args(argv)

    from faster_whisper import WhisperModel
    kwargs = {"device": "cuda", "compute_type": "float16", "num_workers": 1}
    if os.path.exists(args.download_root):
        kwargs["download_root"] = args.download_root
    model = WhisperModel("large-v3-turbo", **kwargs)

    results = {"files": {}, "references": {}}
    targets = list(LEGACY_REFERENCED) + list(FFMPEG_CLIP)

    decoded = {}
    for name in targets:
        decoded[name] = {}
        for config in CONFIGS:
            decoded[name][config] = decode(model, os.path.join(args.audio_dir, name),
                                           CONFIGS[config])
            print(f"decoded {name} {config}: {len(decoded[name][config]['words'])} words",
                  flush=True)

    references = {}
    for name, opener in LEGACY_REFERENCED.items():
        legacy = norm_words(legacy_text(args.legacy_db, name))
        run = missing_run(legacy, decoded[name]["control"]["words"], opener)
        references[name] = run or []
        print(f"reference {name}: {len(references[name])} legacy words", flush=True)
    for name, (start, end) in FFMPEG_CLIP.items():
        got = ffmpeg_clip_reference(model, args.audio_dir, name, start, end)
        references[name] = got["words"]
        print(f"reference {name}: {len(references[name])} clip words", flush=True)

    for name in targets:
        results["references"][name] = {
            "words": len(references[name]),
            "text": " ".join(references[name]),
        }
        results["files"][name] = {}
        for config in CONFIGS:
            got = decoded[name][config]
            score = containment(references[name], got["words"])
            entry = {"words": len(got["words"]), "containment": score,
                     "decode_seconds": got["decode_seconds"]}
            if name in FFMPEG_CLIP:
                start, end = FFMPEG_CLIP[name]
                entry["windows"] = window_word_rates(got["segments"], start - 30, end + 30)
            results["files"][name][config] = entry
            print(f"{name:22s} {config:16s} words {entry['words']:6d}  "
                  f"containment {score}", flush=True)

    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)
        handle.write("\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
