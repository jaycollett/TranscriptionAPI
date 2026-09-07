"""Transcription quality experiment harness.

Runs named decode configs (configs.py, deep-decode.md C1-C10 plus the PROD control)
and alignment paths (align.py, deep-alignment.md A0 versus I1/I2/I5/I6) on a list
of audio files, writing per (config, file) the transcript, the segment list with
per-segment diagnostics and a metrics JSON, then a Markdown summary and results.json.

Meant to run inside the deployed image on the GPU host:

  python /harness/harness.py run    --audio /audio/a.mp3,/audio/b.mp3 --configs PROD,C1,C3
  python /harness/harness.py align  --audio /audio/a.mp3 --configs C1,PROD --paths a0,i1
  python /harness/harness.py report

All state lives under --out (default /harness/results); `report` only reads it, so
the tables can be regenerated on any machine from a copy of that directory.

The RC060 config imports transcribe.py directly for its level rule and its post-decode
stage, so a release-candidate run needs `transcribe.py` and `textnorm.py` copied into
this directory alongside the harness. Every other config is self-contained.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from align import (  # noqa: E402
    align_path,
    alignment_metrics,
    ensure_mfa_models,
    match_words,
    refine_i2,
    refine_window,
)
from metrics import api_invariants, decode_metrics  # noqa: E402
from prod_replica import PROD_HELPERS_SOURCE, PROD_VAD  # noqa: E402
from report import build_report  # noqa: E402

log = logging.getLogger("harness")


def stem_of(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def file_dir(out, path):
    return os.path.join(out, stem_of(path))


# ---------------------------------------------------------------- run
def cmd_run(args):
    from decode import as_float_list, decode_audio_file, load_model, run_config, vad_chunks_seconds

    model = load_model()
    log.info("production helpers imported from: %s", PROD_HELPERS_SOURCE or "local copies")
    for audio_path in args.audio:
        fdir = file_dir(args.out, audio_path)
        os.makedirs(fdir, exist_ok=True)
        t0 = time.time()
        audio, duration = decode_audio_file(audio_path)
        log.info("decoded %s: %.1f s in %.1f s", audio_path, duration, time.time() - t0)
        meta_path = os.path.join(fdir, "file.json")
        if os.path.exists(meta_path):
            meta = read_json(meta_path)
        else:
            ref = vad_chunks_seconds(audio, PROD_VAD)
            meta = {
                "stem": stem_of(audio_path),
                "path": audio_path,
                "duration": round(duration, 3),
                "ref_vad": PROD_VAD,
                "ref_speech_chunks": as_float_list(ref),
                "ref_speech_s": round(sum(e - s for s, e in ref), 2),
            }
            write_json(meta_path, meta)
        ref_chunks = [tuple(c) for c in meta["ref_speech_chunks"]]
        for name in args.configs:
            cdir = os.path.join(fdir, name)
            if os.path.exists(os.path.join(cdir, "metrics.json")) and not args.force:
                log.info("%s/%s exists, skipping (use --force to rerun)", meta["stem"], name)
                continue
            log.info("=== %s on %s", name, meta["stem"])
            try:
                result = run_config(model, audio, duration, name, ref_chunks, audio_path=audio_path)
            except Exception:
                log.error("config %s failed on %s:\n%s", name, meta["stem"], traceback.format_exc())
                write_json(os.path.join(cdir, "error.json"), {"error": traceback.format_exc()})
                continue
            own_chunks = [tuple(c) for c in result.get("vad_chunks", [])]
            metrics = decode_metrics(
                result["segments"], result["transcript"], duration, own_chunks, ref_chunks, result.get("ladder_base", 0.0)
            )
            metrics.update(
                {
                    "config": name,
                    "wall_s": result["wall_s"],
                    "rtf": result["rtf"],
                    "gpu_peak_mb": result.get("gpu_peak_mb"),
                    "gpu_peak_total_mb": result.get("gpu_peak_total_mb"),
                    "invariants": api_invariants(result["transcript"], result["timings"]),
                }
            )
            if result.get("level"):
                metrics["mean_dbfs"] = result["level"]["mean_dbfs"]
                metrics["vad_threshold"] = result["level"]["threshold"]
                metrics["vad_profile"] = result["level"].get("profile")
            if result.get("production"):
                production = result["production"]
                metrics["anomaly_count"] = production["anomaly_count"]
                metrics["anomaly_windows"] = production["anomaly_windows"]
                metrics["flagged_segments"] = len(production["flagged_segments"])
                metrics["words_raw"] = production["words_before_dedupe"]
                metrics["dedupe_removed_words"] = production["words_before_dedupe"] - metrics["words"]
                metrics["rescue_attempted"] = production.get("rescue_attempted")
                metrics["rescue_selected"] = production.get("rescue_selected")
                metrics["selected_pass"] = production.get("selected_pass")
                metrics["uncovered_s"] = production.get("uncovered_s")
                metrics["uncovered_max_gap_s"] = production.get("uncovered_max_gap_s")
                if production.get("pass_scores"):
                    metrics["pass_scores"] = production["pass_scores"]
                    metrics["pass_wall_s"] = production["pass_wall_s"]
            if result["pipeline"] == "prod":
                metrics["words_raw"] = len(result["transcript_raw"].split())
                metrics["dedupe_removed_words"] = metrics["words_raw"] - metrics["words"]
                metrics["chosen_pass"] = result["selection"]["chosen_pass"]
                metrics["segscore_pass"] = result["selection"]["segscore_chosen_pass"]
                pass_metrics = []
                for p in result["passes"]:
                    pm = decode_metrics(p["segments"], p["transcript"], duration, own_chunks, ref_chunks, p["ladder_base"])
                    pm = {k: v for k, v in pm.items() if not k.endswith("_detail")}
                    pm["wall_s"] = p["wall_s"]
                    pass_metrics.append(pm)
                result["pass_metrics"] = pass_metrics
                metrics["pass_wall_s"] = [p["wall_s"] for p in result["passes"]]
            result["vad_chunks"] = as_float_list(own_chunks)
            os.makedirs(cdir, exist_ok=True)
            with open(os.path.join(cdir, "transcript.txt"), "w", encoding="utf-8") as fh:
                fh.write(result["transcript"] + "\n")
            write_json(os.path.join(cdir, "segments.json"), result)
            write_json(os.path.join(cdir, "metrics.json"), metrics)
            log.info(
                "%s on %s: %d words, wps %.3f, segs %d, wall %.1f s, gpu peak +%s MB",
                name, meta["stem"], metrics["words"], metrics["wps"] or 0, metrics["seg_count"], metrics["wall_s"],
                metrics.get("gpu_peak_mb"),
            )
        del audio
    build_report(args.out)


# ---------------------------------------------------------------- align
def cmd_align(args):
    mfa_root = os.environ.get("MFA_ROOT_DIR")
    ensure_mfa_models(mfa_root)
    workdir = args.work
    os.makedirs(workdir, exist_ok=True)
    for audio_path in args.audio:
        fdir = file_dir(args.out, audio_path)
        meta = read_json(os.path.join(fdir, "file.json"))
        duration = meta["duration"]
        for name in args.configs:
            spath = os.path.join(fdir, name, "segments.json")
            if not os.path.exists(spath):
                log.warning("no decode output for %s/%s; run it first", meta["stem"], name)
                continue
            result = read_json(spath)
            segments = [s for s in result["segments"] if s["text"].strip()]
            for path_name in args.paths:
                adir = os.path.join(fdir, "align", f"{name}__{path_name}")
                if os.path.exists(os.path.join(adir, "metrics.json")) and not args.force:
                    log.info("%s exists, skipping", adir)
                    continue
                log.info("=== align %s %s on %s", name, path_name, meta["stem"])
                try:
                    record, mfa_words, _ = align_path(
                        path_name, audio_path, segments, duration, workdir, mfa_root, f"{meta['stem']}_{name}"
                    )
                except Exception:
                    log.error("align %s/%s failed:\n%s", name, path_name, traceback.format_exc())
                    write_json(os.path.join(adir, "error.json"), {"error": traceback.format_exc()})
                    continue
                os.makedirs(adir, exist_ok=True)
                if record.get("mfa_json") and os.path.exists(record["mfa_json"]):
                    shutil.copy2(record["mfa_json"], os.path.join(adir, "mfa.json"))
                    src_dir = os.path.dirname(record["mfa_json"])
                    csv_path = os.path.join(src_dir, "alignment_analysis.csv")
                    if os.path.exists(csv_path):
                        shutil.copy2(csv_path, os.path.join(adir, "alignment_analysis.csv"))
                out = {"process": record, "rules": {}}
                if record["mfa_success"]:
                    records, match_ratio = match_words(segments, mfa_words)
                    win, win_empty = refine_window(segments, mfa_words)
                    i2, i2_empty, pre_nonmono = refine_i2(segments, mfa_words, records)
                    out["rules"]["window"] = alignment_metrics(segments, win, mfa_words, records, win_empty, result["transcript"])
                    out["rules"]["i2"] = alignment_metrics(segments, i2, mfa_words, records, i2_empty, result["transcript"])
                    out["rules"]["i2"]["pre_nonmono"] = pre_nonmono
                    out["match_ratio"] = match_ratio
                    write_json(os.path.join(adir, "refined_window.json"), win)
                    write_json(os.path.join(adir, "refined_i2.json"), i2)
                write_json(os.path.join(adir, "metrics.json"), out)
                log.info(
                    "align %s %s on %s: success %s attempt %s wall %.1f s; window agree250 %s; i2 agree250 %s",
                    name, path_name, meta["stem"], record["mfa_success"], record.get("attempt_used"), record["mfa_wall_s"],
                    out["rules"].get("window", {}).get("agree250"), out["rules"].get("i2", {}).get("agree250"),
                )
        for leftover in os.listdir(workdir):
            shutil.rmtree(os.path.join(workdir, leftover), ignore_errors=True)
    build_report(args.out)


def cmd_report(args):
    results = build_report(args.out)
    print(json.dumps({k: list(v["configs"]) for k, v in results.items()}, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="/harness/results")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--audio", required=True, type=lambda s: s.split(","))
    run.add_argument("--configs", required=True, type=lambda s: s.split(","))
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=cmd_run)
    al = sub.add_parser("align")
    al.add_argument("--audio", required=True, type=lambda s: s.split(","))
    al.add_argument("--configs", required=True, type=lambda s: s.split(","))
    al.add_argument("--paths", default="a0,i1", type=lambda s: s.split(","))
    al.add_argument("--work", default="/harness/work")
    al.add_argument("--force", action="store_true")
    al.set_defaults(func=cmd_align)
    rep = sub.add_parser("report")
    rep.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
