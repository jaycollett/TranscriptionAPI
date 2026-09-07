"""Aggregate whatever is on disk under the results directory into per-file Markdown
tables and one machine-readable results.json. Runs without the GPU stack, so the
tables can be regenerated on the Mac from copied results.
"""

import json
import os

from configs import GLOSSARY_TERMS
from metrics import agreement, glossary_stats
from selection import anomaly_count

DECODE_COLUMNS = [
    ("words", "words"),
    ("wps", "wps"),
    ("wps_speech", "wps_sp"),
    ("seg_count", "segs"),
    ("seg_mean_len_s", "seg_mean"),
    ("seg_max_len_s", "seg_max"),
    ("word_dur_p95", "wdur_p95"),
    ("logprob_mean", "lp_mean"),
    ("logprob_min", "lp_min"),
    ("cr_max", "cr_max"),
    ("temp_ge_0_5", "t>=.5"),
    ("fallback_windows", "fallbk"),
    ("no_speech_ratio", "nosp"),
    ("repeated_3gram_rate", "rep3"),
    ("repeated_4gram_rate", "rep4"),
    ("window_wps_min", "win_min"),
    ("low_windows", "low_win"),
    ("phantom_segments", "phantom"),
    ("ngram_reject_flags", "c7_flags"),
    ("seam_count", "seams"),
    ("removed_s", "removed_s"),
    ("wall_s", "wall_s"),
    ("rtf", "rtf"),
    ("gpu_peak_total_mb", "gpu_mb"),
    ("agree_c1", "agree_C1"),
]

ALIGN_COLUMNS = [
    ("mfa_success", "ok"),
    ("attempt_used", "attempt"),
    ("mfa_wall_s", "mfa_s"),
    ("mfa_rtf", "mfa_rtf"),
    ("utterances_submitted", "utt_sub"),
    ("utterances_aligned", "utt_ok"),
    ("agree250", "agree250"),
    ("delta_start_p50", "ds_p50"),
    ("delta_start_p95", "ds_p95"),
    ("nonmono", "nonmono"),
    ("pre_nonmono", "pre_nonmono"),
    ("empty_fallbacks", "empty"),
    ("owned_ratio_lt_0_8", "own<.8"),
    ("owned_ratio_gt_1_2", "own>1.2"),
    ("drift_words", "drift"),
    ("pause_snapped", "snap"),
    ("words_transcript", "w_tr"),
    ("words_timings", "w_tim"),
    ("words_mfa", "w_mfa"),
    ("dev_gt10", "dev>10"),
    ("oov_count", "oov"),
]


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if abs(value) < 1000 else f"{value:.0f}"
    return str(value)


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(out)


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_file_results(file_dir):
    """Everything stored for one audio file: file.json, per-config metrics and segments."""
    file_meta = _load(os.path.join(file_dir, "file.json")) if os.path.exists(os.path.join(file_dir, "file.json")) else {}
    configs = {}
    for name in sorted(os.listdir(file_dir)):
        cdir = os.path.join(file_dir, name)
        mpath = os.path.join(cdir, "metrics.json")
        spath = os.path.join(cdir, "segments.json")
        if os.path.isdir(cdir) and os.path.exists(mpath) and os.path.exists(spath):
            configs[name] = {"metrics": _load(mpath), "segments": _load(spath)}
    aligns = {}
    adir = os.path.join(file_dir, "align")
    if os.path.isdir(adir):
        for name in sorted(os.listdir(adir)):
            mpath = os.path.join(adir, name, "metrics.json")
            if os.path.exists(mpath):
                aligns[name] = _load(mpath)
    return file_meta, configs, aligns


def cross_config(configs):
    """Pairwise agreement of every config with C1, the determinism check and the C6 gate."""
    out = {"pairs": {}, "determinism": None, "greedy_gate": None}
    if "C1" in configs:
        for name, data in configs.items():
            if name == "C1":
                continue
            agr = agreement(configs["C1"]["segments"]["segments"], data["segments"]["segments"])
            out["pairs"][f"C1~{name}"] = agr
            data["metrics"]["agree_c1"] = agr["ratio"]
        if "C1_REPEAT" in configs:
            agr = out["pairs"]["C1~C1_REPEAT"]
            out["determinism"] = {
                "ratio": agr["ratio"],
                "identical_text": agr["insert"] + agr["delete"] + agr["replace"] == 0,
                "identical_segments": configs["C1"]["segments"]["segments"] == configs["C1_REPEAT"]["segments"]["segments"],
            }
        if "GREEDY" in configs:
            agr = out["pairs"]["C1~GREEDY"]
            c1 = configs["C1"]
            gr = configs["GREEDY"]
            clean = (
                anomaly_count(c1["segments"]["segments"]) == 0
                and anomaly_count(gr["segments"]["segments"]) == 0
                and c1["metrics"]["low_windows"] == 0
                and gr["metrics"]["low_windows"] == 0
            )
            out["greedy_gate"] = {
                "ratio": agr["ratio"],
                "anomaly_free": clean,
                "fast_path": agr["ratio"] >= 0.985 and clean,
                "disagreements": agr["insert"] + agr["delete"] + agr["replace"],
                "greedy_wall_s": gr["metrics"].get("wall_s"),
            }
    if "PROD" in configs and "C1" in configs:
        out["pairs"]["PROD~C1"] = out["pairs"].get("C1~PROD")
    return out


def file_markdown(stem, file_meta, configs, aligns, cross):
    lines = [f"## {stem} ({file_meta.get('duration', 0):.0f} s, ref speech {file_meta.get('ref_speech_s', 0):.0f} s)", ""]
    rows = []
    for name, data in configs.items():
        m = data["metrics"]
        rows.append([name] + [m.get(key) for key, _ in DECODE_COLUMNS])
    lines.append(_table(["config"] + [label for _, label in DECODE_COLUMNS], rows))
    lines.append("")
    if "PROD" in configs:
        sel = configs["PROD"]["segments"].get("selection", {})
        lines.append(
            f"PROD selection: production rule chose pass {sel.get('chosen_pass')}; "
            f"C2 SEGSCORE would choose pass {sel.get('segscore_chosen_pass')} "
            f"({'same' if sel.get('segscore_agrees') else 'different'})."
        )
        prow = []
        for k, (p, s) in enumerate(zip(sel.get("per_pass", []), sel.get("segscore_per_pass", [])), 1):
            pm = configs["PROD"]["segments"].get("pass_metrics", [{}] * 5)[k - 1] if configs["PROD"]["segments"].get("pass_metrics") else {}
            prow.append(
                [k, p["words"], p["confidence"], p["adjusted"], s["anomalies"], s["low_windows"], s["mean_logprob"],
                 pm.get("wall_s"), pm.get("temp_ge_0_5"), pm.get("repeated_4gram_rate")]
            )
        lines.append("")
        lines.append(_table(["pass", "words", "conf", "adjusted", "c2_anom", "c2_low", "lp_mean", "wall_s", "t>=.5", "rep4"], prow))
        lines.append("")
    if cross.get("determinism"):
        d = cross["determinism"]
        lines.append(
            f"Determinism (C1 vs C1_REPEAT): ratio {d['ratio']}, identical text {d['identical_text']}, "
            f"identical segments {d['identical_segments']}."
        )
    if cross.get("greedy_gate"):
        g = cross["greedy_gate"]
        lines.append(
            f"C6 GREEDY_GATE: C1~GREEDY ratio {g['ratio']} ({g['disagreements']} disagreement runs), "
            f"anomaly-free {g['anomaly_free']}, fast path {'taken' if g['fast_path'] else 'not taken'}."
        )
    if "C5" in configs:
        sel = configs["C5"]["segments"].get("selection", {})
        lines.append(
            f"C5: {sel.get('clips')} clips, {sel.get('clips_disagreeing')} with pass disagreement; "
            f"chosen per pass {sel.get('chosen_counts')}."
        )
    if "C9" in configs and "C1" in configs:
        g9 = glossary_stats(configs["C9"]["segments"]["transcript"], GLOSSARY_TERMS)
        g1 = glossary_stats(configs["C1"]["segments"]["transcript"], GLOSSARY_TERMS)
        lines.append(
            f"C9 GLOSSARY: term hits C9 {g9['total_hits']} vs C1 {g1['total_hits']} "
            f"(C9 {g9['per_term']}; C1 {g1['per_term']}); prompt leakage C9 {g9['leakage']}."
        )
    lines.append("")
    if aligns:
        lines.append("Alignment (rows: decode config, MFA path, refinement rule):")
        lines.append("")
        rows = []
        for name, m in aligns.items():
            for rule in ("window", "i2"):
                r = m.get("rules", {}).get(rule)
                if not r:
                    continue
                merged = {**m.get("process", {}), **r}
                merged["dev_gt10"] = (m.get("process", {}).get("analysis") or {}).get("duration_deviation_gt10")
                merged["oov_count"] = (m.get("process", {}).get("attempts") or [{}])[-1].get("oov_count")
                rows.append([name, rule] + [merged.get(key) for key, _ in ALIGN_COLUMNS])
        lines.append(_table(["run", "rule"] + [label for _, label in ALIGN_COLUMNS], rows))
        lines.append("")
    return "\n".join(lines)


def build_report(results_dir):
    all_results = {}
    md = ["# Quality harness results", ""]
    for stem in sorted(os.listdir(results_dir)):
        fdir = os.path.join(results_dir, stem)
        if not os.path.isdir(fdir) or not os.path.exists(os.path.join(fdir, "file.json")):
            continue
        file_meta, configs, aligns = load_file_results(fdir)
        cross = cross_config(configs)
        section = file_markdown(stem, file_meta, configs, aligns, cross)
        with open(os.path.join(fdir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write(section + "\n")
        md.append(section)
        all_results[stem] = {
            "file": file_meta,
            "configs": {n: d["metrics"] for n, d in configs.items()},
            "selection": {n: d["segments"].get("selection") for n, d in configs.items()},
            "cross": cross,
            "align": aligns,
        }
    with open(os.path.join(results_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=1)
    return all_results
