"""Read the four fidelity configurations against each other, one open item at a time.

`fidelity_pass.py` produces one JSON per configuration. This joins them and answers the
three questions 0.6.0 left open, plus the cheap-detector question:

1. **Should the primary pass sample rather than beam-search?** B against A on the
   scripture benchmark (the only ground truth available) and on the whole set as a
   regression check.
2. **Is a second decode's disagreement the missing detector?** A against B, which unlike
   the two builds already on disk really are two different decode paths, scored against
   the independent evidence of a bad transcript.
3. **Should the VAD profile be keyed on encoding rather than level?** D against A on the
   files where the two selectors disagree.

Plus C against A, which separates the rescue's two levers: C keeps the higher ladder base
and puts previous-text conditioning back on.
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agreement import compare, separation  # noqa: E402
from norm import norm_words  # noqa: E402


def load_config(path):
    with open(path) as handle:
        doc = json.load(handle)
    return doc.get("note"), doc["files"]


def pct(new, old):
    return None if not old else round((new - old) / old * 100, 2)


def seg_per_min(entry):
    duration = entry.get("duration_sec") or 0
    return round((entry.get("segments") or 0) / (duration / 60.0), 2) if duration else None


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build(configs, file_list, notes):
    control = configs["A"]
    bad_names = {r["file"] for r in file_list["files"] if r["bad_reasons"]}
    tags = {r["file"]: r["tags"] for r in file_list["files"]}
    split = {r["file"] for r in file_list["files"] if r["selector_split"]}

    per_file = {}
    for name, base in sorted(control.items()):
        if base.get("error"):
            continue
        row = {"tags": tags.get(name, []), "bad": name in bad_names,
               "selector_split": name in split,
               "duration_sec": base.get("duration_sec"),
               "A": {"words": base.get("words"), "segments": base.get("segments"),
                     "seg_per_min": seg_per_min(base),
                     "vad_profile": base.get("vad_profile"),
                     "anomaly_windows": base.get("anomaly_windows"),
                     "rescue_attempted": base.get("rescue_attempted"),
                     "rescue_selected": base.get("rescue_selected"),
                     "uncovered_max_gap_s": base.get("uncovered_max_gap_s"),
                     "wall_s": base.get("wall_s")}}
        base_words = norm_words(base.get("transcription") or "")
        for label in ("B", "C", "D"):
            entry = (configs.get(label) or {}).get(name)
            if not entry or entry.get("error"):
                continue
            cell = {"words": entry.get("words"), "segments": entry.get("segments"),
                    "seg_per_min": seg_per_min(entry),
                    "vad_profile": entry.get("vad_profile"),
                    "anomaly_windows": entry.get("anomaly_windows"),
                    "rescue_attempted": entry.get("rescue_attempted"),
                    "rescue_selected": entry.get("rescue_selected"),
                    "uncovered_max_gap_s": entry.get("uncovered_max_gap_s"),
                    "wall_s": entry.get("wall_s"),
                    "word_delta": (entry.get("words") or 0) - (base.get("words") or 0),
                    "word_delta_pct": pct(entry.get("words") or 0, base.get("words") or 0)}
            cell.update(compare(base_words, norm_words(entry.get("transcription") or "")))
            cell["disagreement"] = (round(1.0 - cell["agreement"], 5)
                                   if cell["agreement"] is not None else None)
            cell["max_run_rate"] = (round(cell["max_run"] / max(1, min(
                len(base_words), entry.get("words") or 1)), 5))
            row[label] = cell
        per_file[name] = row

    # The detector question, on the A/B pair.
    ab = {name: row["B"] for name, row in per_file.items() if "B" in row}
    detector = {
        "disagreement": separation(ab, "disagreement", bad_names, True),
        "max_run": separation(ab, "max_run", bad_names, True),
        "max_run_rate": separation(ab, "max_run_rate", bad_names, True),
    }

    summary = {}
    for label in ("B", "C", "D"):
        rows = [row[label] for row in per_file.values() if label in row]
        if not rows:
            continue
        deltas = [r["word_delta_pct"] for r in rows if r["word_delta_pct"] is not None]
        summary[label] = {
            "n": len(rows),
            "median_word_delta_pct": round(statistics.median(deltas), 3) if deltas else None,
            "mean_word_delta_pct": round(statistics.mean(deltas), 3) if deltas else None,
            "files_worse_1pct": sum(1 for d in deltas if d <= -1.0),
            "files_worse_5pct": sum(1 for d in deltas if d <= -5.0),
            "files_better_1pct": sum(1 for d in deltas if d >= 1.0),
            "median_agreement": round(statistics.median(
                r["agreement"] for r in rows if r["agreement"] is not None), 5),
            "rescue_selected": sum(1 for r in rows if r.get("rescue_selected")),
            "median_seg_per_min": round(statistics.median(
                r["seg_per_min"] for r in rows if r["seg_per_min"] is not None), 2),
            "total_wall_s": round(sum(r["wall_s"] or 0 for r in rows), 1),
        }
    summary["A"] = {
        "n": len(per_file),
        "median_seg_per_min": round(statistics.median(
            row["A"]["seg_per_min"] for row in per_file.values()
            if row["A"]["seg_per_min"] is not None), 2),
        "rescue_selected": sum(1 for row in per_file.values()
                               if row["A"].get("rescue_selected")),
        "total_wall_s": round(sum(row["A"]["wall_s"] or 0 for row in per_file.values()), 1),
    }

    return {"notes": notes, "bad_files": sorted(bad_names), "summary": summary,
            "detector": detector, "per_file": per_file}


def render(doc):
    out = ["# The fidelity experiment: four configurations, one corpus", ""]
    for label in sorted(doc["notes"]):
        out.append(f"- **{label}**: {doc['notes'][label]}")
    out.append("")
    out.append(f"Independent evidence of a bad transcript under 0.6.0 "
               f"({len(doc['bad_files'])} files): " + ", ".join(doc["bad_files"]))
    out.append("")

    out.append("## Whole-set summary, every configuration against A")
    out.append("")
    out.append("| config | files | median word delta | mean word delta | worse by 1% | "
               "worse by 5% | better by 1% | median agreement with A | median seg/min | "
               "rescues kept | wall |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for label in ("A", "B", "C", "D"):
        s = doc["summary"].get(label)
        if not s:
            continue
        out.append(
            f"| {label} | {s['n']} | {fmt(s.get('median_word_delta_pct'), 2)}% "
            f"| {fmt(s.get('mean_word_delta_pct'), 2)}% | {s.get('files_worse_1pct', '-')} "
            f"| {s.get('files_worse_5pct', '-')} | {s.get('files_better_1pct', '-')} "
            f"| {fmt(s.get('median_agreement'), 4)} | {fmt(s.get('median_seg_per_min'), 2)} "
            f"| {s.get('rescue_selected')} | {fmt(s.get('total_wall_s'), 0)}s |")
    out.append("")

    out.append("## Open item 2: does A-vs-B disagreement detect a bad transcript")
    out.append("")
    out.append("A and B are genuinely different decode paths, one beam-searching and one "
               "sampling, so unlike the two builds already on disk their disagreement "
               "measures the decode rather than a downstream gate.")
    out.append("")
    for key, title in [("max_run", "Longest one-sided run, in words"),
                       ("max_run_rate", "Longest one-sided run, as a share of the transcript"),
                       ("disagreement", "Whole-file disagreement")]:
        sep = doc["detector"].get(key)
        out.append(f"### {title}")
        out.append("")
        if not sep:
            out.append("Not computable.")
            out.append("")
            continue
        out.append(f"Healthy ({sep['n_good']} files): median {fmt(sep['good_median'], 5)}, "
                   f"p95 {fmt(sep['good_p95'], 5)}, max {fmt(sep['good_max'], 5)}.")
        out.append("")
        out.append("| bad file | value | rank |")
        out.append("|---|---|---|")
        for entry in sep["bad"]:
            out.append(f"| {entry['file']} | {fmt(entry['value'], 5)} | {entry['rank']} |")
        out.append("")
        out.append("| catches | of | threshold | healthy also flagged | total flagged |")
        out.append("|---|---|---|---|---|")
        for point in sep["curve"]:
            out.append(f"| {point['caught']} | {point['of']} | {fmt(point['threshold'], 5)} "
                       f"| {point['false_positives']} | {point['flagged_total']} |")
        out.append("")

    out.append("## Open item 3: the profile selector, on the files where the two disagree")
    out.append("")
    out.append("| file | dBFS | A profile | D profile | A seg/min | D seg/min | "
               "A words | D words | delta | agreement |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, row in sorted(doc["per_file"].items()):
        if "D" not in row:
            continue
        a, d = row["A"], row["D"]
        out.append(f"| {name} | {fmt(a.get('mean_dbfs'), 1)} | {a['vad_profile']} "
                   f"| {d['vad_profile']} | {fmt(a['seg_per_min'], 2)} "
                   f"| {fmt(d['seg_per_min'], 2)} | {a['words']} | {d['words']} "
                   f"| {fmt(d['word_delta_pct'], 2)}% | {fmt(d['agreement'], 4)} |")
    out.append("")

    out.append("## The rescue's two levers: C against A where A fires a rescue")
    out.append("")
    out.append("| file | A rescue kept | C rescue kept | A words | C words | delta | agreement |")
    out.append("|---|---|---|---|---|---|---|")
    for name, row in sorted(doc["per_file"].items()):
        if "C" not in row:
            continue
        a, c = row["A"], row["C"]
        out.append(f"| {name} | {a.get('rescue_selected')} | {c.get('rescue_selected')} "
                   f"| {a['words']} | {c['words']} | {fmt(c['word_delta_pct'], 2)}% "
                   f"| {fmt(c['agreement'], 4)} |")
    out.append("")

    out.append("## Every file")
    out.append("")
    out.append("| file | tags | A words | B delta | C delta | D delta | A seg/min | "
               "B seg/min | A gap | B gap | bad |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, row in sorted(doc["per_file"].items()):
        a = row["A"]
        cells = []
        for label in ("B", "C", "D"):
            cell = row.get(label)
            cells.append("-" if not cell else f"{fmt(cell['word_delta_pct'], 2)}%")
        b = row.get("B") or {}
        out.append(f"| {name} | {','.join(row['tags'])} | {a['words']} | " + " | ".join(cells)
                   + f" | {fmt(a['seg_per_min'], 2)} | {fmt(b.get('seg_per_min'), 2)} "
                   f"| {fmt(a.get('uncovered_max_gap_s'), 1)} "
                   f"| {fmt(b.get('uncovered_max_gap_s'), 1)} "
                   f"| {'yes' if row['bad'] else ''} |")
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, help="label=path")
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args(argv)

    configs, notes = {}, {}
    for spec in args.config:
        label, path = spec.split("=", 1)
        notes[label], configs[label] = load_config(path)
    with open(args.file_list) as handle:
        file_list = json.load(handle)

    doc = build(configs, file_list, notes)
    with open(args.out_json, "w") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
    with open(args.out_md, "w") as handle:
        handle.write(render(doc))
    print(f"{len(doc['per_file'])} files -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
