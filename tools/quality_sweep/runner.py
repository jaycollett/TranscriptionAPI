"""Submit the sweep file list to an isolated transcription service and record every result.

The runner talks to one service, one file at a time, over HTTP. It never touches the
production container, the production database or the production upload folder: the base
URL, the output directory and the state file are all arguments, and the sweep service is
expected to be `transcription-api-sweep` on port 5031 with its own `DB_FILE`,
`UPLOAD_FOLDER` and `MFA_ROOT_DIR` under /home/jay/sweep.

For each file it POSTs `/upload` with a fresh UUID v4, polls `/status/<guid>` until the
job reaches a terminal state, and writes one JSON record per file. Every key the service
returns is kept, so the additive 0.6.0 fields (`anomaly_count`, `anomaly_windows`,
`flagged_segments`, `rescue_attempted`, `rescue_selected`, `mfa_applied`,
`attempt_count`, `processing_seconds`, `words_per_second`) land in the record without the
runner needing to know about them; `KNOWN_ADDITIVE` only decides what gets a column in
the progress line.

The run takes hours, so it is resumable: the state file records the outcome of every
finished file and a rerun skips those unless `--force` names them. A quarantined,
errored or timed-out job is recorded with its reason and the runner moves to the next
file rather than stopping.

Stdlib only: it runs under the plain `python3` of the GPU host with no virtualenv.

    python3 tools/quality_sweep/runner.py \
        --file-list tools/quality_sweep/file_list.json \
        --audio-dir /home/jay/SourceCode/SermonPreprocessorAPI/data/audiofiles \
        --base-url http://127.0.0.1:5031 \
        --out-dir /home/jay/sweep/run
"""

import argparse
import json
import mimetypes
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from norm import norm_words  # noqa: E402

# Fields worth showing on the progress line. Anything else the service returns is still
# stored in the record.
KNOWN_ADDITIVE = (
    "processing_seconds",
    "words_per_second",
    "attempt_count",
    "mfa_applied",
    "anomaly_count",
    "rescue_attempted",
    "rescue_selected",
)

TERMINAL = ("completed", "error")

# Per-file poll timeout. The service retries a failed job up to three times, so the
# budget has to cover three decodes of the file plus alignment; measured end-to-end
# real-time factors are 0.14-0.19 on 0.5.x and about 0.07 expected on 0.6.0.
TIMEOUT_BASE_S = 180.0
TIMEOUT_PER_AUDIO_S = 3.0
TIMEOUT_FLOOR_S = 900.0

# Free space below this on the output filesystem stops the run rather than filling the
# disk the production container also writes to.
MIN_FREE_BYTES = 5 * 1024 ** 3


def timeout_for(duration_s):
    """Poll budget in seconds for a file of this duration."""
    if not duration_s:
        return TIMEOUT_FLOOR_S
    return max(TIMEOUT_FLOOR_S, TIMEOUT_BASE_S + TIMEOUT_PER_AUDIO_S * float(duration_s))


def encode_multipart(fields, file_field, filename, payload):
    """A multipart/form-data body: `fields` as text parts, one file part."""
    boundary = "----sweep" + uuid.uuid4().hex
    sep = ("--" + boundary).encode()
    chunks = []
    for name, value in sorted(fields.items()):
        chunks.append(sep)
        chunks.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        chunks.append(b"")
        chunks.append(str(value).encode())
    ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    chunks.append(sep)
    chunks.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"'.encode()
    )
    chunks.append(f"Content-Type: {ctype}".encode())
    chunks.append(b"")
    chunks.append(payload)
    chunks.append(("--" + boundary + "--").encode())
    chunks.append(b"")
    body = b"\r\n".join(chunks)
    return body, f"multipart/form-data; boundary={boundary}"


def request_json(url, data=None, content_type=None, timeout=120):
    """`(status_code, parsed_json_or_text)` for one HTTP call, errors included."""
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            code = response.getcode()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        code = exc.code
    except (urllib.error.URLError, OSError) as exc:
        return None, {"error": f"transport: {exc}"}
    try:
        return code, json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return code, {"error": "non-json response", "body": raw[:500].decode("utf-8", "replace")}


def timing_stats(timings):
    """Structural checks on the timings list that do not need the audio."""
    if not isinstance(timings, list) or not timings:
        return {"count": 0, "non_monotonic": None, "overlaps": None, "text_matches": None}
    starts, ends = [], []
    for entry in timings:
        if not isinstance(entry, dict):
            continue
        starts.append(entry.get("start"))
        ends.append(entry.get("end"))
    non_monotonic = sum(
        1
        for i in range(1, len(starts))
        if starts[i] is not None and starts[i - 1] is not None and starts[i] < starts[i - 1]
    )
    overlaps = sum(
        1
        for i in range(1, len(starts))
        if starts[i] is not None and ends[i - 1] is not None and starts[i] < ends[i - 1]
    )
    return {"count": len(timings), "non_monotonic": non_monotonic, "overlaps": overlaps}


def summarise(payload, duration_s):
    """Derived quantities the analyzer needs, computed once here."""
    text = payload.get("transcription") or ""
    words = norm_words(text)
    timings = payload.get("timings") or []
    stats = timing_stats(timings)
    joined = " ".join(
        (entry.get("text") or "") for entry in timings if isinstance(entry, dict)
    ).strip()
    stats["text_matches"] = (joined == text.strip()) if timings else None
    return {
        "words": len(words),
        "chars": len(text),
        "wps": round(len(words) / duration_s, 4) if duration_s else None,
        "timings": stats,
    }


class Runner:
    def __init__(self, base_url, audio_dir, out_dir, poll_interval=10.0):
        self.base_url = base_url.rstrip("/")
        self.audio_dir = audio_dir
        self.out_dir = out_dir
        self.poll_interval = poll_interval
        self.state_path = os.path.join(out_dir, "state.json")
        os.makedirs(os.path.join(out_dir, "records"), exist_ok=True)
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as handle:
                return json.load(handle)
        return {"started_at": time.time(), "files": {}}

    def _save_state(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(self.state, handle, indent=1, sort_keys=True)
        os.replace(tmp, self.state_path)

    def free_bytes(self):
        return shutil.disk_usage(self.out_dir).free

    def submit(self, filename, guid):
        path = os.path.join(self.audio_dir, filename)
        with open(path, "rb") as handle:
            payload = handle.read()
        body, ctype = encode_multipart({"guid": guid}, "file", filename, payload)
        return request_json(self.base_url + "/upload", body, ctype, timeout=600)

    def poll(self, guid, budget_s):
        """Poll one job to a terminal state. Returns `(payload, waited_s, reason)`."""
        deadline = time.monotonic() + budget_s
        last = None
        while time.monotonic() < deadline:
            code, payload = request_json(f"{self.base_url}/status/{guid}", timeout=120)
            if code == 200 and isinstance(payload, dict):
                last = payload
                if payload.get("status") in TERMINAL:
                    return payload, budget_s - (deadline - time.monotonic()), None
            elif code == 404:
                return payload, budget_s - (deadline - time.monotonic()), "guid not found"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self.poll_interval, remaining))
        return last, budget_s, "timeout"

    def run_one(self, entry):
        filename = entry["file"]
        duration_s = entry.get("duration_s")
        guid = str(uuid.uuid4())
        record = {
            "file": filename,
            "guid": guid,
            "duration_s": duration_s,
            "submitted_at": time.time(),
            "strata": {k: entry.get(k) for k in entry if k not in ("file",)},
        }

        code, response = self.submit(filename, guid)
        record["upload_status"] = code
        if code != 201:
            # 409 means the GUID collided, which a fresh UUID makes vanishingly unlikely;
            # anything else is a real submit failure. Both are recorded, not retried here.
            record["outcome"] = "submit_failed"
            record["error"] = response
            record["finished_at"] = time.time()
            return record

        budget = timeout_for(duration_s)
        record["timeout_s"] = budget
        payload, waited, reason = self.poll(guid, budget)
        record["wall_s"] = round(time.time() - record["submitted_at"], 2)
        record["polled_s"] = round(waited, 2)
        record["finished_at"] = time.time()

        if reason:
            record["outcome"] = reason.replace(" ", "_")
            record["last_payload"] = payload
            return record

        record["outcome"] = payload.get("status")
        record["payload"] = {k: v for k, v in payload.items() if k != "timings"}
        record["derived"] = summarise(payload, duration_s)
        if payload.get("status") == "completed":
            self._write_side_files(filename, guid, payload)
        return record

    def _write_side_files(self, filename, guid, payload):
        stem = os.path.splitext(filename)[0]
        text_dir = os.path.join(self.out_dir, "transcripts")
        timing_dir = os.path.join(self.out_dir, "timings")
        os.makedirs(text_dir, exist_ok=True)
        os.makedirs(timing_dir, exist_ok=True)
        with open(os.path.join(text_dir, stem + ".txt"), "w") as handle:
            handle.write(payload.get("transcription") or "")
        with open(os.path.join(timing_dir, stem + ".json"), "w") as handle:
            json.dump({"guid": guid, "timings": payload.get("timings") or []}, handle)

    def run(self, entries, force=()):
        force = set(force)
        done = 0
        for index, entry in enumerate(entries, 1):
            filename = entry["file"]
            prior = self.state["files"].get(filename)
            if prior and prior.get("outcome") and filename not in force:
                print(f"[{index}/{len(entries)}] skip {filename} ({prior['outcome']})")
                continue
            if self.free_bytes() < MIN_FREE_BYTES:
                print(f"stopping: only {self.free_bytes() / 1024 ** 3:.1f} GB free")
                break
            print(f"[{index}/{len(entries)}] {filename} ({entry.get('duration_s')} s) ...", flush=True)
            record = self.run_one(entry)
            self.state["files"][filename] = record
            self._save_state()
            with open(os.path.join(self.out_dir, "records", filename + ".json"), "w") as handle:
                json.dump(record, handle, indent=1, sort_keys=True)
            extras = " ".join(
                f"{k}={record.get('payload', {}).get(k)}"
                for k in KNOWN_ADDITIVE
                if isinstance(record.get("payload"), dict) and k in record["payload"]
            )
            derived = record.get("derived") or {}
            print(
                f"    -> {record['outcome']} words={derived.get('words')} "
                f"wps={derived.get('wps')} wall={record.get('wall_s')}s {extras}",
                flush=True,
            )
            done += 1
        return done


def load_entries(file_list_path, include_supplementary):
    with open(file_list_path) as handle:
        doc = json.load(handle)
    entries = list(doc["files"])
    if include_supplementary:
        entries.extend(doc.get("supplementary") or [])
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N entries")
    parser.add_argument("--only", action="append", default=[], help="run only these filenames")
    parser.add_argument("--force", action="append", default=[], help="rerun these filenames")
    parser.add_argument("--no-supplementary", action="store_true")
    args = parser.parse_args(argv)

    entries = load_entries(args.file_list, not args.no_supplementary)
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["file"] in wanted]
    if args.limit:
        entries = entries[: args.limit]

    runner = Runner(args.base_url, args.audio_dir, args.out_dir, args.poll_interval)
    print(f"{len(entries)} entries, {runner.free_bytes() / 1024 ** 3:.1f} GB free at start")
    processed = runner.run(entries, force=args.force)
    print(f"processed {processed} files, {runner.free_bytes() / 1024 ** 3:.1f} GB free at end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
