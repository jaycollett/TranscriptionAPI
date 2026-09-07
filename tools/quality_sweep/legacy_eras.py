"""Compare the legacy words-per-second distributions of the single-pass and multi-pass eras.

The archive's stored transcripts come from four releases (see `strata.ERAS`). E1 is a
single `model.transcribe(beam_size=7, best_of=7)` under 0.1.5; E2 and E3 are the
five-pass scheme with the duration-weighted best-pass selector. 0.6.0 returns to a single
pass, so the natural question is whether single-pass output historically looked any worse
on this corpus.

**This is observational, not controlled.** Era is a proxy for date: E1 is the March 2025
bulk import of everything published up to then, E2 and E3 are what has been published
since. Speaker, room, recording chain and encoder all drift with date, and nothing here
separates a decoder effect from a corpus effect. Two files were never decoded twice, so
there is no paired measurement anywhere in this data. Treat a difference as a prompt to
look, never as evidence that one decoder is better.

The stratified comparison reduces, but does not remove, the confound: it compares eras
only inside cells of the same duration and level bucket, and pools the within-cell
differences weighted by the smaller arm of each cell.

    python3 tools/quality_sweep/legacy_eras.py \
        --baseline legacy_baseline.json --levels levels.csv --durations durations.csv
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze import distribution, percentile  # noqa: E402
from strata import describe, read_baseline, read_durations, read_levels  # noqa: E402

SINGLE_PASS_ERAS = ("E1_0.1.5",)
MULTI_PASS_ERAS = ("E2_0.2.x", "E3_0.3.x")

# A cell needs this many files in each arm before its difference means anything.
MIN_PER_ARM = 3


def _round(value, digits=3):
    return None if value is None else round(value, digits)


def arms(rows):
    """Split rows into the single-pass and multi-pass arms, dropping every other era."""
    single = [r for r in rows if r["legacy_era"] in SINGLE_PASS_ERAS and r["legacy_wps"]]
    multi = [r for r in rows if r["legacy_era"] in MULTI_PASS_ERAS and r["legacy_wps"]]
    return single, multi


def raw_comparison(rows):
    """The unadjusted difference in words per second between the two arms."""
    single, multi = arms(rows)
    single_wps = [r["legacy_wps"] for r in single]
    multi_wps = [r["legacy_wps"] for r in multi]
    single_median = percentile(single_wps, 50)
    multi_median = percentile(multi_wps, 50)
    return {
        "single_pass": distribution(single_wps),
        "multi_pass": distribution(multi_wps),
        "median_difference": (
            round(single_median - multi_median, 4)
            if single_median is not None and multi_median is not None
            else None
        ),
        "covariates": {
            "single_pass_duration_median": percentile([r["duration_s"] for r in single], 50),
            "multi_pass_duration_median": percentile([r["duration_s"] for r in multi], 50),
            "single_pass_dbfs_median": percentile([r["mean_dbfs"] for r in single], 50),
            "multi_pass_dbfs_median": percentile([r["mean_dbfs"] for r in multi], 50),
        },
    }


def stratified_comparison(rows, min_per_arm=MIN_PER_ARM):
    """Within-cell differences over duration x level cells, pooled by the smaller arm.

    Weighting by the smaller arm keeps a cell with 60 single-pass files and 3 multi-pass
    ones from dominating the pooled number on the strength of an arm that is nearly empty.
    """
    cells = collections.defaultdict(lambda: {"single": [], "multi": []})
    for row in rows:
        if not row["legacy_wps"]:
            continue
        key = (row["duration_bucket"], row["level_bucket"])
        if row["legacy_era"] in SINGLE_PASS_ERAS:
            cells[key]["single"].append(row["legacy_wps"])
        elif row["legacy_era"] in MULTI_PASS_ERAS:
            cells[key]["multi"].append(row["legacy_wps"])

    table, weighted_sum, weight_total, usable = [], 0.0, 0, 0
    higher_single = 0
    for key in sorted(cells):
        single = cells[key]["single"]
        multi = cells[key]["multi"]
        entry = {
            "duration_bucket": key[0],
            "level_bucket": key[1],
            "single_n": len(single),
            "multi_n": len(multi),
            "single_median": _round(percentile(single, 50)),
            "multi_median": _round(percentile(multi, 50)),
            "usable": len(single) >= min_per_arm and len(multi) >= min_per_arm,
        }
        if entry["usable"]:
            entry["difference"] = round(entry["single_median"] - entry["multi_median"], 4)
            weight = min(len(single), len(multi))
            weighted_sum += entry["difference"] * weight
            weight_total += weight
            usable += 1
            if entry["difference"] > 0:
                higher_single += 1
        else:
            entry["difference"] = None
        table.append(entry)

    return {
        "cells": table,
        "usable_cells": usable,
        "cells_favouring_single_pass": higher_single,
        "pooled_difference": round(weighted_sum / weight_total, 4) if weight_total else None,
        "pooled_weight": weight_total,
        "min_per_arm": min_per_arm,
    }


def compare(baseline_path, levels_path, durations_path):
    baseline = read_baseline(baseline_path)
    levels = read_levels(levels_path)
    durations = read_durations(durations_path)
    rows = [
        describe(name, baseline[name], levels.get(name), durations.get(name))
        for name in sorted(baseline)
    ]
    return {
        "note": (
            "Observational. Era is a proxy for date, so speaker, room and encoder drift "
            "with it and no file was ever decoded by both. Not evidence about decoders."
        ),
        "raw": raw_comparison(rows),
        "stratified": stratified_comparison(rows),
    }


def render_markdown(result):
    raw = result["raw"]
    out = ["# Legacy single-pass against multi-pass words per second", "", result["note"], ""]
    out.append("## Raw")
    out.append("")
    out.append("| arm | n | mean | p5 | p25 | median | p75 | p95 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for label, key in (("single pass (E1, 0.1.5)", "single_pass"),
                       ("multi pass (E2 and E3, 0.2.x and 0.3.x)", "multi_pass")):
        dist = raw[key]
        out.append(
            f"| {label} | {dist['n']} | {dist['mean']} | {dist['p5']} | {dist['p25']} | "
            f"{dist['median']} | {dist['p75']} | {dist['p95']} |"
        )
    out.append("")
    cov = raw["covariates"]
    out.append(
        f"Median difference (single minus multi): {raw['median_difference']} words per second. "
        f"The arms are not alike to begin with: median duration "
        f"{cov['single_pass_duration_median']} s against "
        f"{cov['multi_pass_duration_median']} s, median level "
        f"{cov['single_pass_dbfs_median']} dBFS against "
        f"{cov['multi_pass_dbfs_median']} dBFS."
    )
    out.append("")
    out.append("## Stratified by duration and level")
    out.append("")
    strat = result["stratified"]
    out.append("| duration | level | single n | multi n | single median | multi median | difference |")
    out.append("|---|---|---|---|---|---|---|")
    for cell in strat["cells"]:
        out.append(
            f"| {cell['duration_bucket']} | {cell['level_bucket']} | {cell['single_n']} | "
            f"{cell['multi_n']} | {cell['single_median']} | {cell['multi_median']} | "
            f"{cell['difference'] if cell['difference'] is not None else 'too few'} |"
        )
    out.append("")
    out.append(
        f"Pooled over {strat['usable_cells']} cells with at least "
        f"{strat['min_per_arm']} files in each arm, weighted by the smaller arm: "
        f"{strat['pooled_difference']} words per second. The single-pass arm is higher in "
        f"{strat['cells_favouring_single_pass']} of those {strat['usable_cells']} cells."
    )
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--levels", required=True)
    parser.add_argument("--durations", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    args = parser.parse_args(argv)

    result = compare(args.baseline, args.levels, args.durations)
    text = render_markdown(result)
    if args.out_json:
        with open(args.out_json, "w") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")
    if args.out_md:
        with open(args.out_md, "w") as handle:
            handle.write(text)
            handle.write("\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
