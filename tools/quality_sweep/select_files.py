"""Choose the 100-file validation sample and write tools/quality_sweep/file_list.json.

The sample is not random. It is a mandatory core of everything the 0.6.0 changes are
supposed to act on, filled out with a stratified draw so the remainder spans the
archive's duration, level and bit-rate ranges. `README.md` explains why each rule is
there; `RULES` below is the machine-readable version and the report quotes it.

Run it from the repo root with the three inputs measured on the GPU host:

    python3 tools/quality_sweep/select_files.py \
        --durations durations.csv --levels levels.csv \
        --baseline legacy_baseline.json --out tools/quality_sweep/file_list.json

The output is deterministic: every tie breaks on the filename and the one shuffle is
seeded, so re-running it on the same inputs reproduces the same 100 files.
"""

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strata import (  # noqa: E402
    describe,
    read_baseline,
    read_durations,
    read_levels,
)

TARGET = 100
SEED = 20260907

# The six files the 2026-09-07 harness run measured. Every acceptance threshold in
# docs/QUALITY_PROPOSAL.md section 4 is stated against these, so the sweep has to carry
# them or the release gate has no continuity.
HARNESS_REFERENCE = (
    "tcf.20240213b.mp3",
    "tcf.20240116.mp3",
    "tcf.20240213a.mp3",
    "tcf.20240319b.mp3",
    "tcf.20240416_Formation_Class_Holy_Spirit.mp3",
    "women_retreat_2025_session3.mp3",
)

# The 2024 teachings that collapsed under the pre-0.4.0 code. Their stored transcripts
# were overwritten by the 0.5.4 re-run on 2026-09-07, so their baseline is fixed-code
# output, not legacy output. Two of them have no transcript at all: the orchestrator's
# re-run lost the submission five times and gave up.
KNOWN_COLLAPSE_NO_BASELINE = (
    "tcf.20240326b.mp3",
    "tcf.20240514.mp3",
)

RULES = (
    ("rerun_2026_09_07", "every file the 0.5.4 re-run produced (legacy era E4): the known collapses"),
    ("defective_era_E3", "every file transcribed by 0.3.x, the era with all four decode defects live"),
    ("below_vad_cutover", "every file whose mean level is under -26 dBFS: the 0.35 VAD branch"),
    ("harness_reference", "the six files the 2026-09-07 harness measured"),
    ("lowest_legacy_wps", "the ten lowest legacy words-per-second rates in the archive"),
    ("lowest_bitrate", "the ten lowest bit rates"),
    ("very_long", "every recording over 2700 s"),
    ("non_mono_or_non_mp3", "the stereo files and the one AAC file"),
    ("multi_voice", "class, retreat, men's breakfast and Q&A recordings"),
    ("rare_sample_rate", "two files from every sample rate the archive holds fewer than ten of"),
    ("cutover_zone", "twelve files between -26 and -24 dBFS, spread evenly across the edge"),
    ("stratified_fill", "a seeded draw over duration x level cells to reach 100"),
)

# The -26 to -24 dBFS band holds 33 files, more than the sample can spend on one stratum.
# Twelve are taken at even spacing across the band so the edge is resolved without the
# band crowding out the rest of the archive.
CUTOVER_ZONE_QUOTA = 12

# Sample rates the archive holds fewer than ten of (32 kHz, five files) would otherwise
# never be drawn; two of each are taken so no encoder path is unmeasured.
RARE_RATE_MAX = 10
RARE_RATE_QUOTA = 2


def spread(names, rows_by_name, key, quota):
    """`quota` names spaced evenly along `key`, always including both endpoints."""
    ordered = sorted(names, key=lambda n: (key(rows_by_name[n]), n))
    if quota >= len(ordered):
        return ordered
    if quota == 1:
        return ordered[:1]
    step = (len(ordered) - 1) / (quota - 1)
    picked = []
    for i in range(quota):
        candidate = ordered[int(round(i * step))]
        if candidate not in picked:
            picked.append(candidate)
    for name in ordered:
        if len(picked) >= quota:
            break
        if name not in picked:
            picked.append(name)
    return sorted(picked)


def mandatory_sets(rows):
    """`{rule: [filename, ...]}` for every deliberate inclusion, in rule order."""
    by_name = {row["file"]: row for row in rows}
    names = sorted(by_name)

    def take(predicate, limit=None, key=None):
        picked = [n for n in names if predicate(by_name[n])]
        if key is not None:
            picked.sort(key=lambda n: (key(by_name[n]), n))
        if limit is not None:
            picked = picked[:limit]
        return picked

    sets = collections.OrderedDict()
    sets["rerun_2026_09_07"] = take(lambda r: r["legacy_era"] == "E4_0.5.x")
    sets["defective_era_E3"] = take(lambda r: r["legacy_era"] == "E3_0.3.x")
    sets["below_vad_cutover"] = take(
        lambda r: r["mean_dbfs"] is not None and r["mean_dbfs"] < -26.0
    )
    sets["harness_reference"] = [n for n in HARNESS_REFERENCE if n in by_name]
    sets["lowest_legacy_wps"] = take(
        lambda r: r["legacy_wps"] is not None, limit=10, key=lambda r: r["legacy_wps"]
    )
    sets["lowest_bitrate"] = take(
        lambda r: r["bit_rate"] is not None, limit=10, key=lambda r: r["bit_rate"]
    )
    sets["very_long"] = take(lambda r: r["duration_bucket"] == "very_long")
    sets["non_mono_or_non_mp3"] = take(
        lambda r: (r["channels"] or 1) != 1 or not r["file"].lower().endswith(".mp3")
    )
    sets["multi_voice"] = take(lambda r: r["multi_voice"])
    rate_counts = collections.Counter(
        by_name[n]["sample_rate"] for n in names if by_name[n]["sample_rate"]
    )
    rare_rates = {rate for rate, count in rate_counts.items() if count < RARE_RATE_MAX}
    rare = []
    for rate in sorted(rare_rates):
        members = [n for n in names if by_name[n]["sample_rate"] == rate]
        rare.extend(members[:RARE_RATE_QUOTA])
    sets["rare_sample_rate"] = sorted(rare)
    sets["cutover_zone"] = spread(
        take(lambda r: r["level_bucket"] == "cutover_low"),
        by_name,
        lambda r: r["mean_dbfs"],
        CUTOVER_ZONE_QUOTA,
    )
    return sets


def stratified_fill(rows, chosen, target, seed=SEED):
    """Fill `chosen` to `target` by round-robin over duration x level cells.

    Cells are visited in order of how far they are below their share of the archive, so
    the fill pulls the sample toward the archive's shape rather than toward whichever
    cell happens to be biggest.
    """
    by_name = {row["file"]: row for row in rows}
    remaining = sorted(set(by_name) - set(chosen))
    rng = random.Random(seed)
    rng.shuffle(remaining)

    def cell(name):
        row = by_name[name]
        return (row["duration_bucket"], row["level_bucket"])

    pool = collections.defaultdict(list)
    for name in remaining:
        pool[cell(name)].append(name)

    archive = collections.Counter(cell(n) for n in by_name)
    picked = collections.Counter(cell(n) for n in chosen)
    total = len(by_name)
    out = list(chosen)

    while len(out) < target:
        candidates = [c for c in pool if pool[c]]
        if not candidates:
            break
        # Deficit against the cell's archive share, largest first; ties on the cell name.
        candidates.sort(key=lambda c: (-(archive[c] / total * target - picked[c]), c))
        cell_key = candidates[0]
        name = pool[cell_key].pop(0)
        out.append(name)
        picked[cell_key] += 1
    return out


def build(durations_path, levels_path, baseline_path):
    durations = read_durations(durations_path)
    levels = read_levels(levels_path)
    baseline = read_baseline(baseline_path)

    rows = [
        describe(name, baseline[name], levels.get(name), durations.get(name))
        for name in sorted(baseline)
    ]
    by_name = {row["file"]: row for row in rows}

    sets = mandatory_sets(rows)
    chosen = []
    reasons = collections.defaultdict(list)
    for rule, members in sets.items():
        for name in members:
            reasons[name].append(rule)
            if name not in chosen:
                chosen.append(name)

    if len(chosen) > TARGET:
        raise SystemExit(
            f"mandatory rules already select {len(chosen)} files, over the {TARGET} target"
        )

    chosen = stratified_fill(rows, chosen, TARGET)
    for name in chosen:
        if not reasons[name]:
            reasons[name].append("stratified_fill")

    files = []
    for name in chosen:
        row = dict(by_name[name])
        row["reasons"] = reasons[name]
        files.append(row)
    files.sort(key=lambda r: (-(r["duration_s"] or 0), r["file"]))

    supplementary = []
    for name in KNOWN_COLLAPSE_NO_BASELINE:
        if name in baseline:
            continue
        row = describe(name, None, levels.get(name), durations.get(name))
        row["reasons"] = ["known_collapse_no_baseline"]
        supplementary.append(row)

    return {
        "generated_from": {
            "durations": os.path.basename(durations_path),
            "levels": os.path.basename(levels_path),
            "baseline": os.path.basename(baseline_path),
            "archive_files": len(baseline),
            "seed": SEED,
        },
        "rules": [{"rule": r, "why": w} for r, w in RULES],
        "strata_counts": strata_counts(files),
        "files": files,
        "supplementary": supplementary,
    }


def strata_counts(files):
    """Counts per stratum for the selected files, for the README and the report."""
    def count(key):
        return dict(sorted(collections.Counter(row[key] for row in files).items()))

    return {
        "total": len(files),
        "duration_bucket": count("duration_bucket"),
        "level_bucket": count("level_bucket"),
        "bitrate_bucket": count("bitrate_bucket"),
        "sample_rate": {str(k): v for k, v in count("sample_rate").items()},
        "channels": {str(k): v for k, v in count("channels").items()},
        "legacy_era": count("legacy_era"),
        "vad_threshold": {str(k): v for k, v in count("vad_threshold").items()},
        "multi_voice": {str(k): v for k, v in count("multi_voice").items()},
        "reasons": dict(
            sorted(collections.Counter(r for row in files for r in row["reasons"]).items())
        ),
    }


def archive_counts(durations_path, levels_path, baseline_path):
    """The same counts over the whole archive, so the sample can be compared to it."""
    durations = read_durations(durations_path)
    levels = read_levels(levels_path)
    baseline = read_baseline(baseline_path)
    rows = [
        describe(name, baseline[name], levels.get(name), durations.get(name))
        for name in sorted(baseline)
    ]
    for row in rows:
        row["reasons"] = []
    return strata_counts(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", required=True)
    parser.add_argument("--levels", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--print-archive", action="store_true")
    args = parser.parse_args(argv)

    doc = build(args.durations, args.levels, args.baseline)
    with open(args.out, "w") as handle:
        json.dump(doc, handle, indent=1, sort_keys=False)
        handle.write("\n")
    print(f"wrote {args.out}: {len(doc['files'])} files, "
          f"{len(doc['supplementary'])} supplementary")
    print(json.dumps(doc["strata_counts"], indent=1))
    if args.print_archive:
        print("archive:")
        print(json.dumps(archive_counts(args.durations, args.levels, args.baseline), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
