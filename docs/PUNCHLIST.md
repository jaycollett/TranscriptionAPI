# Punchlist

Date: 2026-09-07. Merged from two reviews of the service as it stood on `main`
before the 0.4.0 release: a service review (queue, routes, worker, deployment,
CI) and a pipeline review (decode configuration, alignment, scoring). Production
evidence is the devmachine overlay container (image
`transcription-api:0.3.0-decode-fix`, logs 2026-09-07 06:11 to 06:40 UTC, jobs
f5e593e9 at 755 s and 95f4fe20 at 2783 s). Code references are given by file and
function because the line numbers in the original reviews were taken from `main`
and have since moved.

Constraints that every item respects:

- **API contract.** `/status` returns `status` in {pending, processing,
  completed, error}, `transcription` (string), `timings` (list of
  {start, end, text} with float times), `estimated_completion_utc`, `message`.
  `/upload` takes a multipart `file` plus `guid` and returns `message`, `guid`,
  `estimated_completion_utc` with 201/400/409/500. `/transcriptions` returns
  `guid`, `filename`, `status`, `submitted_at`, `completed_at`,
  `processing_time_est`. Additive optional fields and new endpoints are allowed;
  renames, removals, status-value changes and type changes are not.
- **Stateless by design.** The SQLite queue lives in the container layer and is
  lost on redeploy on purpose (clients resubmit on 404). No volume for the
  database, ever. See `docs/SESSION_KNOWLEDGE.md`.
- **GPU validation gate.** Anything that touches the decode parameters in
  `transcribe.py` (passes, temperature ladder, VAD, beam sizes), the Dockerfile
  apt/CUDA lines or `ENV CUDA_*` settings, or the base image tag must be
  validated on devmachine with the recipe in `docs/SESSION_KNOWLEDGE.md`
  (reference file `tcf.20240213b.mp3`, 755 s, expect 2.5-2.8 words/sec) before
  it replaces production. Those items are not part of 0.5.0.

Verification legend: **Mac** means a unit test with the existing conftest stubs
(no GPU, no MFA binary); **GPU** means it needs the devmachine container.

Status values: `Open`, `Resolved in 0.4.0 (...)`, `Deferred to a GPU-validated
release`, `Done in 0.5.0 (<commit subject>)`.

## Service

Baseline before 0.5.0: `pytest tests -q --cov=app` passed 26/26 at roughly 42%
coverage of `app.py`; the three routes, the worker loop and cleanup, and the MFA
JSON parsing were uncovered.

### P0 - correctness or data-loss bug today

**1. Speaker diarization is dead in production: every job 401s against Hugging Face and falls back to one speaker**
`app.py:detect_speakers`, Dockerfile pyannote snapshot, runDocker.sh token
handling (all removed in 0.4.0). Log at 06:30:04: `HEAD
.../pyannote/segmentation-3.0/.../pytorch_model.bin 401`, then `Speaker
diarization failed ... assuming single speaker`. The Dockerfile only snapshotted
the `speaker-diarization-3.1` pipeline repo; its config referenced
`pyannote/segmentation-3.0` and `pyannote/wespeaker-voxceleb-resnet34-LM`,
which were never cached, so every job reached the network with no token. Net
effect: the multi-speaker MFA-skip never triggered and each job made two
outbound HTTPS calls on the hot path.
Recommended change (at the time): snapshot the two sub-model repos, set
`HF_HUB_OFFLINE=1`, load the pipeline once at startup. Risk: low. Verify: GPU.
Status: Resolved in 0.4.0 (speaker diarization removed; the build no longer needs a Hugging Face token)

**2. Hourly cleanup effectively never runs, and when it does it orphans files**
`app.py:transcription_worker` (cleanup block). Three compounding defects: (a)
`if not records: continue` skips the cleanup block whenever the queue is empty,
so cleanup can only run immediately after a job finishes; (b) the gate
`int(time.time()) % 3600 < 30` is evaluated at that instant, which lands in the
window with probability 30/3600 = 0.8% per completed job; (c) file deletion is
`LIMIT 20` but the `DELETE FROM transcriptions` is unbounded, so rows 21+ lose
their DB record while their audio, transcript, `_aligned/` and `_mfa_input/`
stay on disk forever.
Recommended change: extract `cleanup_old_jobs(cursor, upload_folder)` and call
it on every wake when more than an hour has passed since the last run
(monotonic clock), before the "no pending" `continue`; delete files and rows for
the same GUID set in one pass; drop the `{guid}_processed.mp3` and
`{guid}_corpus` entries from the per-job path list, since neither is created
any more. Age rows by `COALESCE(completed_at, created_at)`, not `created_at`:
a job that waited a day in a deep queue and completed minutes ago still has a
client polling for it. Risk: low. Verify: Mac, unit test with `created_at`
two days old, assert files and row both gone; a second test with 25 rows; a
third with a row created two days ago but completed five minutes ago, kept.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop; aging by completion time from review)

**3. MFA leaves a full working corpus under /mfa for every job (container-layer disk leak)**
`app.py:run_forced_alignment` (mfa command). `mfa align` is run without
`--clean` or `--temp_directory`, so MFA writes its corpus (features, lattices,
`.db`, copied acoustic model) to `$MFA_ROOT_DIR/<guid>_mfa_input/`. docker-diff
of the live container shows three such trees (163 entries) after 20 minutes of
uptime; nothing in cleanup touches `/mfa`. The `{guid}_corpus` path the code
expects in the error-log probe is never created.
Recommended change: add `--clean` to the mfa command and remove
`$MFA_ROOT_DIR/<guid>_mfa_input` (env `MFA_ROOT_DIR`, default `/mfa`) in a
`finally` after alignment; remove the dead `_corpus` log-dir probe. Risk: low.
Verify: Mac for the command flags and the rmtree call; GPU for `ls /mfa` after
a job showing only `pretrained_models`.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop; GPU check of /mfa still pending)

**4. The CI image build cannot succeed: no build arg is ever passed, and the Dockerfile asserts on it**
`.github/workflows/BuildAndPublish.yml` build step, Dockerfile pyannote layer
(removed in 0.4.0). `build-push-action` had no `build-args:` or `secrets:`, and
the Dockerfile did `assert hf_token`, so both the release build and the Friday
rebuild failed at that layer. Production confirmed it: the running image was a
locally built `transcription-api:0.3.0-decode-fix`, not a ghcr.io tag. The
overlay's `docker history` also showed the token in `ARG` and `ENV` layers.
Status: Resolved in 0.4.0 (speaker diarization removed; the build no longer needs a Hugging Face token)
Note (0.5.3): the token-in-history concern is closed by 0.4.0. The CI build
still failed on 0.5.1 and 0.5.2, but at a different layer: `mfa model download`
hit the unauthenticated GitHub API limit (60/hour) from the shared runner pool.
0.5.3 passes the workflow's GITHUB_TOKEN as a BuildKit secret mount, which is
not a layer, so `docker history` stays clean. See `docs/SESSION_KNOWLEDGE.md`.

**5. Audio files survive redeploys but their DB rows do not, so they are never cleaned**
`runDocker.sh` (`-v ./tmp:/tmp`), `app.py:transcription_worker` cleanup. The
stateless design drops the queue on redeploy, but `UPLOAD_FOLDER=/tmp/audio_files`
is a host bind mount that persists. Cleanup is driven purely by DB rows, so
every file from before a redeploy is orphaned permanently on the host.
Recommended change (keeps the stateless design): add a filesystem sweep to the
cleanup pass that removes any entry in `UPLOAD_FOLDER` whose mtime is older
than 2 days and whose GUID prefix has no non-terminal row. Risk: low. Verify:
Mac, unit test with a stray old file and an empty DB.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

### P1 - reliability, runtime, or operability

**6. A row can be stranded in 'processing' while the worker is alive**
`app.py:process_pending_job` (outer except), `app.py:transcription_worker`. The
outer `except` marks the row 'error' with another DB write; if the original
exception was the database (locked, disk full), that write raises too, escapes
to the worker's `except`, and the row stays 'processing' until a restart. With
a single synchronous worker, any 'processing' row at the top of the loop is by
definition orphaned.
Recommended change: call `recover_stuck_jobs(cursor)` at the top of every wake
(a cheap `SELECT COUNT`), not just at startup. Risk: none for correctness.
Verify: Mac, insert a 'processing' row, run one loop iteration, assert
'pending'.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**7. Transient exceptions (CUDA OOM, MFA prep I/O) mark the sermon permanently 'error' with no retry**
`app.py:process_pending_job` (both except blocks). `attempt_count` exists but is
only used for garbage results; a single OOM on a large file terminates the job.
Recommended change: route exceptions through the same counter: increment
`attempt_count`, requeue as 'pending' while below `max_garbage_retries`, else
'error'. Status vocabulary unchanged. A requeue must not be retried on the very
next worker iteration (an instantaneous failure would burn every attempt in
seconds): `worker_cycle` reports a requeue as no progress so the loop sleeps
`POLL_INTERVAL_SEC` first. The outer handler in `process_pending_job` counts
only if the row is still 'processing', so a failure that already moved the
row on is not counted twice. Risk: a deterministic crash costs up to three
runs before failing; acceptable. Verify: Mac, stub `transcribe_audio` to raise
twice then succeed; assert a sleep between a failing attempt and its retry.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop; retry spacing and single counting from review)

**8. Duplicate-GUID race returns 500 and overwrites the winner's file**
`app.py:upload_audio`. SELECT then `file.save` then INSERT with no lock. Two
concurrent uploads of one GUID (the client does burst resubmits: five
`POST /upload 409` within one second at 06:28:18) both pass the SELECT, both
save to the same path, and the loser's INSERT raises `sqlite3.IntegrityError`,
which becomes a 500.
Recommended change: wrap the sequence in a module-level `threading.Lock`,
insert the row before saving the file (so a loser never writes over the
winner's bytes; a failed save deletes the row and returns 500), compare GUIDs
case-insensitively, and catch `sqlite3.IntegrityError` explicitly as 409. The
worker takes the same lock around its pending-row SELECT so it cannot pick up
a row whose file is not yet saved. Contract unchanged. Verify: Mac, test
client, two threads, one 201 and one 409; the 409 path writes no file.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop; insert-before-save ordering from review)

**9. Upload decodes untrusted input before the row exists; failures leak the file and return 500**
`app.py:upload_audio`. The duration probe on a corrupt or non-audio upload
raises `CouldntDecodeError` (a plain `Exception`) after the file is on disk and
before any row is written, so the client gets 500 and the file is never cleaned
(no row).
Recommended change: catch decode failure, `os.remove(file_path)`, return 400
`{'error': 'Could not decode audio'}` (400 is already in the contract). Verify:
Mac, upload `b"not audio"` with a stubbed probe that raises.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**10. Duration probe decodes the entire file to PCM**
`app.py:upload_audio`, `transcribe.py:get_audio_duration`.
`AudioSegment.from_file` renders the whole file; a 46 minute stereo MP3 (the
2783 s job in the logs) is roughly 490 MB of PCM in the request thread just to
read `len(audio)`. The `lru_cache` on `get_audio_duration` is keyed by path and
buys nothing.
Recommended change: `pydub.utils.mediainfo(file_path)['duration']` (ffprobe, no
decode) in both places, falling back to an `AudioSegment` decode only when
mediainfo reports no duration; drop the cache. Verify: Mac unit; GPU RSS during
upload.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**11. ETA at upload ignores the in-flight job; /status counts it**
`app.py:upload_audio` vs `app.py:get_transcription`. `/upload` sums only
`status = 'pending'`; `/status` sums `('pending','processing')`. Log at
06:32:34: upload of 8beac1f2 reported "Total queue time 34.42 min" while
95f4fe20 (16.17 min estimate, processing since 06:31:07) was excluded, so the
first ETA is about 15 min optimistic and jumps on the first poll.
Recommended change: use the same `IN ('pending','processing')` predicate in
`/upload`. A later refinement could add a `started_at` column and subtract
elapsed time for the in-flight job. Verify: Mac, route test with one processing
and one pending row.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**12. /status ETA is anchored on created_at and can sit in the past indefinitely**
`app.py:get_transcription`. `estimated_completion_utc = created_at + sum(ahead) + own`.
Once the queue runs slower than the estimate, or a job is requeued by a garbage
retry (`created_at` unchanged), the client is told a completion time that has
already passed until the job finishes.
Recommended change: return `max(now_utc, created_at + sum(ahead) + own)`; format
unchanged. Also collapse the `psf = 15.1` formula duplicated in
`app.py:upload_audio` and `transcribe.py:transcribe_audio` (log-only there) into
one `estimate_processing_seconds(duration)` used by both files. Verify: Mac,
freeze time, row created an hour ago, assert ETA >= now.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**13. Worker sleeps 30 s between back-to-back jobs and never adapts**
`app.py:transcription_worker`. Logs: job f5e593e9 completed 06:30:37, next
pending job started 06:31:07. With a five-deep queue that is 2.5 idle GPU
minutes per batch.
Recommended change: after a job returns, loop immediately; sleep only when the
SELECT is empty. Verify: Mac, unit test with two pending rows and a stubbed
`time.sleep` that records calls.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**14. pyannote pipeline is reconstructed from disk (and the network) on every job; no warmup**
`app.py:detect_speakers` (removed in 0.4.0). Whisper is cached in
`load_whisper_model()`; the diarization pipeline was not, and it also performed
the failing HTTP calls in item 1 on every job.
Status: Resolved in 0.4.0 (speaker diarization removed; the build no longer needs a Hugging Face token)

**15. No /health endpoint, no worker liveness signal**
`app.py` main block. The worker is a daemon thread; if it exits, Flask keeps
answering and jobs stay 'pending' with nothing observable. No Docker
`HEALTHCHECK`, so `runDocker.sh` cannot wait for readiness.
Recommended change (new endpoint, additive): `GET /health` returning
`{status, worker_alive, worker_last_wake_utc, pending, processing, whisper_loaded}`
with 503 when the worker thread is dead or has not woken in 5 minutes; add a
`HEALTHCHECK` to the Dockerfile. The image installs `wget` but not `curl`, so
the check uses `wget -q -O /dev/null http://localhost:5000/health || exit 1`.
Verify: Mac route test; GPU `docker inspect --format '{{.State.Health.Status}}'`.
Status: Done in 0.5.0 (Add a container health check on the /health endpoint)

**16. Tests do not run in CI; 58% of app.py is unexercised**
`.github/workflows/BuildAndPublish.yml` has a single build job; there is no
test or lint job. Uncovered: all three routes, worker loop, cleanup, MFA JSON
parsing (including the word-window matching), `is_garbage_transcription`, and
the ETA math.
Recommended change: add a `test` job (pytest plus `ruff check`) on push and
pull_request, and make the publish job `needs` it. Backfill route tests with
Flask's test client (the conftest stubs already make `import app` cheap).
Verify: Mac, the suite runs in under a second today.
Status: Done in 0.5.0 (Run the test suite and lint in CI ahead of the image build)

**17. runDocker.sh takes the service down for the whole build and has no rollback**
`runDocker.sh`. It stops and removes the container before `docker build`; no
`set -e`, so a failed build falls through to `docker run transcription-api:latest`
and starts whatever image last held that tag while printing "Container
started!".
Recommended change: `set -euo pipefail`; build into a versioned tag
(`git describe --always --dirty`) before stopping anything, retag the previous
image `transcription-api:rollback`, then stop/rm/run and poll `/health` for up
to 3 minutes. Verify: GPU.
Status: Done in 0.5.0 (Build before stopping, keep a rollback tag and wait for health in runDocker.sh)

**18. Log noise: three INFO lines every 30 s, plus MFA progress bars**
`app.py:transcription_worker`, `app.py:run_forced_alignment`. 70 of the 300
sampled log lines are sleep/wake/no-pending chatter (8,640 lines per day idle);
MFA stdout is dumped in full including rich progress bars.
Recommended change: log the idle transition once, use DEBUG for wake/sleep, log
MFA stdout only on failure. Include the GUID in every job-scoped line. Verify:
Mac with `caplog`.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**19. No per-job metrics surfaced**
`transcribe.py` logs words/sec and pass timing but nothing is stored. Additive
fields on `/transcriptions` rows (`processing_seconds`, `words_per_second`,
`attempt_count`, `mfa_applied`) and on the completed `/status` body would make
calibration of `psf` and garbage-gate tuning possible without grepping logs.
Contract: additive optional fields only; columns added via the existing
`ensure_schema` migration pattern. Verify: Mac route tests.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**20. Dead second garbage check and unreachable `None` branch in process_pending_job**
`app.py:process_pending_job` (post-alignment check), `transcribe.py:transcribe_audio`
(transcript write). The second `is_garbage_transcription(transcription)` tests
the same input already tested before alignment, so it is always False;
`run_forced_alignment` always returns a list (never None); the
`not refined_timings` / all-empty-text branch counts an alignment problem as a
garbage attempt and reruns five Whisper passes. Also `{guid}.txt` is written by
`transcribe.py` and overwritten by `run_forced_alignment`; a write failure in
`transcribe.py` returns an empty transcription that is then counted as garbage.
Recommended change: delete the second check; if timings come back empty fall
back to the Whisper timings without touching `attempt_count`; drop the
transcript write from `transcribe.py`. Verify: Mac, existing tests plus one for
the empty-timings fallback.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

### P2 - polish, hygiene

**21. GUID validation accepts non-v4 and non-canonical forms and stores them raw**
`app.py:upload_audio`. `uuid.UUID(guid, version=4)` overwrites the version bits
rather than checking them, so the nil UUID, `{UPPER-CASE}` and `urn:uuid:...`
forms are all accepted and stored verbatim; the same UUID in two casings becomes
two jobs.
Recommended change: reject unless `str(uuid.UUID(guid)) == guid.lower()` (400,
in contract); do not rewrite the echoed guid. Verify: Mac.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**22. Upload: no size cap, no extension allowlist, extension taken from the client filename**
`app.py:upload_audio`. `os.path.splitext` cannot inject a path separator
(verified), but the extension is unbounded and unvalidated (`a.mp3\n.sh` yields
`.sh`, 300-char extensions are accepted). LAN-only, so hygiene not security.
Recommended change: `MAX_CONTENT_LENGTH` of 1 GiB (413 is new but only
reachable for oversize bodies), allowlist
`{.mp3,.wav,.m4a,.flac,.ogg,.aac,.mp4}`, lowercase it, 400 otherwise. Verify:
Mac.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**23. `temp_mfa_dir` referenced before assignment in the except blocks (masked)**
`app.py:run_forced_alignment`. If anything raises between the transcript write
and the `temp_mfa_dir` assignment (`os.makedirs`), the cleanup in the except
block raises `UnboundLocalError`, which the inner `except Exception` swallows as
"Could not cleanup temp MFA directory". Not a crash, but a misleading log.
Recommended change: assign `temp_mfa_dir` before the `try` and use `finally`
for cleanup. Verify: Mac, monkeypatch `os.makedirs` to raise.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**24. SQLite pragmas and connection comments are misleading (no bug)**
`app.py:init_db`, `app.py:get_db_connection`, route comments.
`journal_mode=WAL` persists; `synchronous`, `foreign_keys` and `busy_timeout`
are per-connection and only hit the init connection (`timeout=30` in `connect`
already covers busy waits). `teardown_appcontext` closes each request thread's
connection, so the "reuses the same connection" comments are wrong; the worker
has no app context and is unaffected. `check_same_thread=False` is unnecessary
with thread-local connections.
Recommended change: apply the per-connection pragmas in `get_db_connection`,
drop the flag, fix the comments. Verify: Mac.
Status: Done in 0.5.0 (Harden the queue, the upload path and the worker loop)

**25. Stale `transcriptions` SQLite file tracked in git and copied into the image**
`transcriptions` is a 20 KB SQLite database from the initial commit; `COPY . .`
ships it plus `runDocker.sh`, `checkQueue.py`, `testIt.py` and `.github/`.
Recommended change: `git rm transcriptions`; add `*.db*`, `transcriptions`,
`runDocker.sh`, `.github/`, `checkQueue.py`, `testIt.py` to `.dockerignore` and
`*.db*` to `.gitignore`. The `*.md` exclusion is harmless.
Status: Done in 0.5.0 (Stop tracking the stale SQLite database and keep host tooling out of the image)

**26. `requests` used by the helper scripts is not declared**
`checkQueue.py`, `testIt.py`; it only works in the container because the conda
base ships `requests 2.28.2`.
Recommended change: `requirements-dev.txt` with `requests`, `pytest`,
`pytest-cov`, `ruff`; use it in CI.
Status: Done in 0.5.0 (Declare the dev tooling and pin the disk-space action to a commit)

**27. checkQueue.py ETA math ignores jobs ahead**
`checkQueue.py:check_transcription_queue` computes
`submitted_at + processing_time_est` per job, wrong for anything but the head
of the queue.
Recommended change: accumulate the sum in submission order or call
`/status/<guid>`. Verify: Mac.
Status: Done in 0.5.0 (Account for jobs ahead in the checkQueue.py ETA)

**28. CI hygiene: unpinned action**
`.github/workflows/BuildAndPublish.yml` uses `jlumbroso/free-disk-space@main`
(mutable ref) while every other action is pinned. The `latest` semantics
(newest version tag by `-v:refname`) are consistent between release and weekly
runs; just document that a hotfix of an older line is not retagged `latest`.
Recommended change: pin to a commit SHA.
Status: Done in 0.5.0 (Declare the dev tooling and pin the disk-space action to a commit)

**29. Dockerfile items for a separate GPU-validated release**
Beyond the header's deferred list (non-root USER, `cuda-toolkit-12-2` to runtime
libs): `CUDA_LAUNCH_BLOCKING=1` serialises every kernel launch, a debugging
setting that costs throughput on all five Whisper passes; the base pin is
`v3.4.1` while production runs `v3.3.9` (image label), so the next from-scratch
build also changes the aligner. Keep both out of any code-only release;
validate on devmachine against the same two reference files (755 s and
2783 s), comparing words/sec and wall time. The Werkzeug dev server is
acceptable for a LAN-only single client; if ever replaced, use single-process
`waitress` so the worker thread is not duplicated.
2026-09-07 update: `cuda-libraries-12-2` and the `CUDA_LAUNCH_BLOCKING`
removal shipped in 0.5.2 (base pin `v3.4.2` since 0.5.1). The residual Trivy
findings (libnghttp2-14, the `/opt/conda` bootstrap packages, pip's vendored
msgpack and setuptools in `/env`) are cleared in 0.5.4; see the session
knowledge entry "0.5.4: residual CVE pass". The non-root USER is the only
item still open.
Status: Mostly shipped (0.5.2, 0.5.4); non-root USER still deferred

**30. torch 2.12.1 carries a low-severity advisory (torch.jit.script)**
`requirements.txt` (`torch==2.12.1`, `torchvision==0.27.1`). GitHub Dependabot
alert #2, raised on `main` once the pins landed: torch <= 2.12.1, low
severity, memory corruption through `torch.jit.script`, fixed in 2.13.0. The
service never calls `torch.jit.script` (torch is imported only for the CUDA
availability check; decoding runs through ctranslate2), so there is no
reachable path today.
Recommended change: bump to torch 2.13.0 when it is validated. torch and
torchvision must move together, since torchvision pins its exact torch
version, and the CUDA build of both has to match the host driver. The bump
changes the validated stack, so it needs the reference-file recipe on
devmachine (2.5-2.8 words/sec on `tcf.20240213b.mp3`, CUDA visible to both
torch and ctranslate2) before it replaces production. Risk: low for the
advisory itself, moderate for the dependency swap. Verify: GPU.
Status: Deferred to a GPU-validated release

## Pipeline

Scope: transcription outcomes only. None of these are implemented in 0.5.0;
every one that changes what reaches `model.transcribe()` or the MFA command
needs the GPU validation recipe before it ships. Evidence: `transcribe.py` and
`app.py` on `main`, the container logs for jobs f5e593e9 (755 s) and 95f4fe20
(2783 s), and the faster-whisper 1.2.1 and MFA 3.x sources.

### P0 - likely affecting output correctness today

**1. Three of the five passes never beam-search; their beam_size and patience are dead parameters**
`transcribe.py:transcribe_audio` (passes list) and faster-whisper
`generate_with_fallback`. In 1.2.1 the decode kwargs are
`{"beam_size": options.beam_size, "patience": options.patience}` only when the
attempt temperature is 0; for temperature > 0 they are
`{"beam_size": 1, "num_hypotheses": best_of, "sampling_topk": 0, "sampling_temperature": t}`.
Passes 1, 4 and 5 have ladder bases 0.2, 0.3 and 0.2, so their first (and
normally only) attempt is a greedy sampling decode with best_of 5. "beam 15,
patience 3.5" is really "sample at T 0.2", which is why pass 5 (27.9 s) ran
faster than pass 2 with beam 7 (37.5 s), and passes 1, 4 and 5 are
non-deterministic. Only pass 2 beam-searches; pass 3 is greedy (patience is
meaningless with beam 1).
Recommended change: ladder base 0.0 for every pass so beam search runs, and
patience 1.0-1.5 (ctranslate2 patience multiplies finished hypotheses before
stopping; 3.5 is pure cost). Effect: deterministic beam decodes, lower WER on
hard words. Risk: low; beam 5 at T 0 is the reference Whisper configuration.
Verify: GPU, run pass 1 twice on the reference file; outputs should be identical
and the log should show T 0 attempts.
Status: Open

**2. Speaker diarization has never run: every job 401s on the gated sub-model and silently assumes one speaker**
`app.py:detect_speakers` (removed in 0.4.0). Logs at 06:30:04 and 06:39:33:
HEAD on `pyannote/segmentation-3.0/pytorch_model.bin` returned 401, then
"assuming single speaker". Root cause: the Dockerfile snapshotted only the
pipeline repo; its config referenced two gated sub-models that were never
cached, and the runtime token was cleared. So the multi-speaker MFA gate was
dead and whole-file MFA ran on Q&A recordings, which is exactly what 0.4.0 now
does by design.
Status: Resolved in 0.4.0 (speaker diarization removed; the build no longer needs a Hugging Face token)

**3. MFA words tier includes silence intervals, so refined segment boundaries can snap to pauses**
`app.py:run_forced_alignment` (word_entries loop). MFA's JSON export appends
every interval as `[begin, end, label]` with no label filter, so the words tier
contains pause entries with label `""`. The code keeps them, so
`segment_words[0]` is frequently a pause that starts inside the Whisper window,
and `refined_start`/`refined_end` become pause edges rather than word edges.
Recommended change: filter entries whose label is empty or in
`{"<eps>", "sil", "spn", "<unk>"}`. Effect: segment starts land on the first
phoneme instead of up to a second early. Risk: none. Verify: GPU,
`jq '.tiers.words.entries[:12]' /tmp/audio_files/<guid>_aligned/<guid>.json` on
the reference job; count empty labels before and after. A Mac unit test can
cover the filter itself.
Status: Open

**4. Word-to-segment assignment by start-time window is wrong for shifted, overlapping or drifting segments**
`app.py:run_forced_alignment` (`whisper_start <= word["start"] <= whisper_end`).
MFA onsets land 50-150 ms before Whisper's, so the first word of segment i+1
falls inside segment i: segment i's end extends over it and segment i+1 starts
on its second word. Whisper segments after `restore_speech_timestamps` can
overlap by a few hundred ms, so boundary words are double-assigned, and when
MFA drifts a 12-word segment captures 3 or 20 MFA words and its edges come from
the wrong words.
Recommended change: MFA returns words in transcript order and its transcript is
the concatenation of the segment texts, so walk cumulative word counts (segment
i gets MFA words `[k_i, k_i + n_i)` after MFA's normalisation: lowercase, strip
punctuation, split on whitespace); if totals disagree, assign each MFA word to
the segment containing its midpoint; then enforce
`start_i = max(start_i, end_{i-1})`. Effect: monotonic, non-overlapping segments
whose edges are the true first and last word. Risk: low with the fallback.
Verify: GPU, on the reference job assert no `timings[i+1].start < timings[i].end`
and that every segment's refined span differs from Whisper's by under 0.5 s.
The mapping itself is Mac-testable with a fixture JSON.
Status: Open

**5. Boundary de-duplication edits the transcript string but not the timings, and it deletes deliberate repetition**
`transcribe.py:clean_boundary_duplicates` and the `final_timings` loop.
`final_transcript` is cleaned but `final_timings` comes from the raw segments,
so `/status.transcription` and the concatenated `timings[].text` disagree, and
MFA (which rebuilds its transcript from `whisper_segments`) aligns a third
variant. The regex `\b(\w+\s+\w+(?:\s+\w+){0,3})[.,;!?\s]*\1\b` is
case-insensitive and position-free: "He is risen. He is risen indeed" becomes
"He is risen. indeed", call-and-response loses content, and stranded
punctuation and double spaces remain.
Recommended change: apply de-duplication only where the last k words (k >= 3)
of segment i equal the first k words of segment i+1, modify that segment's
text, and derive `transcription` from the cleaned segments so the two fields
always agree. Effect: consistent API output; rhetorical repeats preserved.
Risk: fewer cross-segment stutter removals; acceptable now that temperature
fallback handles loops. Verify: Mac for the function;
GPU `" ".join(t["text"] for t in timings) == transcription` on the reference
job and grep the output for rhetorical repeats that used to vanish.
Status: Open

### P1 - measurable quality or runtime win

**6. Five full passes cost 5x runtime for outputs that agree to within 1 percent**
`transcribe.py:transcribe_audio`. 755 s file: 34.4, 37.5, 22.2, 27.1, 27.9 s
(150 s total), word counts 2096-2124, confidences 0.9736-0.9766 except the
pass 5 outlier. 2783 s file: 506 s total, word counts 7817-7847 (0.4 percent
spread), confidences 0.9784-0.9801. The selector is ranking noise.
Recommended change: one primary pass (beam 5, ladder from 0.0,
`condition_on_previous_text=True`), accepted if it passes an anomaly gate: no
segment with `temperature >= 0.5`, `compression_ratio > 2.4` or
`avg_logprob < -1.0`; overall wps 1.8-3.8; no 60 s window of VAD speech below
1.2 wps. Only on failure run a second, differently-conditioned pass (item 7)
and pick by fewest anomalous segments, tie-break on mean `avg_logprob`. Effect:
150 s becomes about 35 s and 506 s about 95 s on clean files, with no loss on
the logged evidence. Risk: a regional error in a clean-looking pass is no longer
outvoted; the window check mitigates. Verify: GPU, reference file wps 2.5-2.8
and WER under 2 percent against the current 5-pass output
(`pip install jiwer`).
Status: Open

**7. Use BatchedInferencePipeline as the second-opinion pass and as the speed path**
`transcribe.py:transcribe_audio`.
`BatchedInferencePipeline(model).transcribe(path, batch_size=8, language="en", beam_size=5, word_timestamps=True, vad_filter=True, vad_parameters=...)`
decodes VAD-derived chunks (up to `chunk_length` s,
`min_silence_duration_ms=160` internally) in parallel and hardcodes
`condition_on_previous_text=False` and `hallucination_silence_threshold=None`
(verified in 1.2.1). Independent chunks cannot carry a repetition loop across
windows, the sequential pass's main failure mode. Expect 3-5x over sequential
on the 3060 with turbo fp16 (about 1.6 GB of weights; batch 8 fits comfortably
now that no diarization model is resident). Effect: a genuinely different
decode for arbitration, or a 10 s primary pass if speed outranks cross-window
context. Risk: weaker proper-noun consistency without context; possible OOM at
batch 16, so start at 8. Verify: GPU, time and wps on the reference file at
batch 8 and 16; `nvidia-smi` peak.
Status: Open

**8. Best-pass metric: duration-weighted word probability rewards hallucinations and cannot see partial collapse**
`transcribe.py:calculate_weighted_confidence` and the 1.4 wps penalty. Whisper
gives hallucinated words in silence long durations, so duration weighting
up-weights exactly the words to distrust. The 1.4 wps floor fires only on
near-total collapse: a pass missing its last 30 percent still shows 1.95 wps
and its surviving words keep high probabilities, so it can win.
Recommended change: score by anomalous-segment count from the fields
faster-whisper already returns per `Segment` (`compression_ratio`,
`avg_logprob`, `no_speech_prob`, `temperature`) plus the 60 s window check,
then mean `avg_logprob`; cross-pass WER (jiwer) is the cheapest ground truth
when two passes exist. Effect: selection tracks real defects instead of 0.002
noise. Risk: none. Verify: Mac, inject a synthetic collapse (truncate one
pass's segments at 70 percent) and confirm it is not selected; GPU for the
reference file.
Status: Open

**9. Enable hallucination_silence_threshold and set the decode thresholds explicitly**
`transcribe.py:transcribe_audio` model.transcribe call. 1.2.1 defaults in force
today: `compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`,
`no_speech_threshold=0.6`, `hallucination_silence_threshold=None`. Sermons have
long pauses (prayer, music, transitions) where Whisper invents "Thank you" or
restates the last phrase. `hallucination_silence_threshold` (sequential path
only, needs `word_timestamps=True`, already on) skips silences longer than the
threshold when the following words look anomalous.
Recommended change: `hallucination_silence_threshold=2.0`, and write the three
thresholds explicitly at their defaults so they are visible and env-tunable.
Leave `repetition_penalty=1.0` and `no_repeat_ngram_size=0`: liturgical
repetition is real content and an n-gram ban forces substitutions; the
compression-ratio fallback is the right loop breaker. Effect: fewer phantom
phrases after pauses. Risk: low; a real word after a very long pause could be
skipped, observable in the window check. Verify: GPU, count segments with
`no_speech_prob > 0.5` and non-empty text on the reference file before and
after.
Status: Open

**10. VAD settings create many seams and admit non-speech**
`transcribe.py:transcribe_audio` vad_parameters. 1.2.1 `VadOptions` defaults:
threshold 0.5, min_silence_duration_ms 2000, speech_pad_ms 400,
min_speech_duration_ms 0. The service uses threshold 0.35 (more permissive:
hum, congregation noise and music count as speech) and min_silence 300 ms; with
400 ms padding each side, gaps under 800 ms are re-joined anyway, so the
effective behaviour is "cut at every pause over about 1.1 s". Each cut is a
seam in the concatenated audio the model sees and a discontinuity
`restore_speech_timestamps` must map, stretching word timestamps across it.
Logs: only 13.9 s of 755 s (1.8 percent) and 97 s of 2783 s were removed, so
the filter buys little and costs seams.
Recommended change: threshold 0.5, min_silence_duration_ms 1000,
speech_pad_ms 300, keep min_speech_duration_ms 250. Effect: fewer seams, less
non-speech decoded. Risk: quiet speakers below 0.5 on a distant mic; check the
"VAD filter removed" line stays under 5 percent. Verify: GPU, the
removed-duration log line and word-timestamp continuity at former seam points.
Status: Open

**11. CUDA_LAUNCH_BLOCKING=1 is baked into the image and serialises every kernel launch**
`Dockerfile` ENV. This is a debugging switch; it forces synchronous launches for
ctranslate2 and torch in the process and costs decode throughput.
Recommended change: remove it (keep `expandable_segments`). Effect: 10-30
percent faster passes once on GPU. Risk: none for production, but it is an
`ENV CUDA_*` line and therefore gated. Verify: GPU, pass 1 time on the
reference file with and without.
Status: Deferred to a GPU-validated release

**12. Diarization will run on CPU and reload from disk on every job once item 2 lands**
`app.py:detect_speakers` (removed in 0.4.0). No `pipeline.to(torch.device("cuda"))`,
and `Pipeline.from_pretrained` ran per job. With diarization gone there is
nothing to move to the GPU or cache.
Status: Resolved in 0.4.0 (speaker diarization removed; the build no longer needs a Hugging Face token)

**13. A raw speaker count is the wrong multi-speaker gate, and skipping MFA is the wrong response**
`app.py:run_forced_alignment` (`num_speakers > 1` branch, removed in 0.4.0).
One "amen" from the congregation made the file two-speaker and disabled timing
refinement for the whole sermon. 0.4.0 removed the gate and always runs MFA,
which is the outcome this item asked for. If multi-speaker handling is ever
wanted again (speech-time share per speaker, `--uses_speaker_adaptation false`,
an optional `speaker` field on `timings` entries), treat it as a new feature
with its own GPU validation, not as a restore of the old code.
Status: Resolved in 0.4.0 (speaker diarization removed; the build no longer needs a Hugging Face token)

**14. MFA aligns the whole file as one utterance with forced beams instead of per-segment utterances**
`app.py:run_forced_alignment` (single-line `{guid}.txt`, beam 40/100 then
100/400). A 46 min single utterance is why the beams had to be widened; errors
propagate across the file, and MFA's "only 1 speakers" warning shows it cannot
parallelise.
Recommended change: write a TextGrid with one interval per Whisper segment as
the transcript, so each segment is an utterance with known bounds; use MFA
defaults (beam 10, retry_beam 40) and a duration-scaled timeout
(60 + 0.25 x duration s) instead of a flat 300 s that, when it fires, is
followed by a wider and slower attempt. Export the WAV with
`ffmpeg -ac 1 -ar 16000 -sample_fmt s16` rather than pydub at source rate (5x
smaller, faster MFA load). Effect: drift bounded to a segment, fewer timeouts,
faster MFA. Risk: low; interval bounds come from Whisper's segment times.
Verify: GPU, `alignment_analysis.csv` per-utterance log-likelihoods, MFA wall
time vs the 24 s baseline on the reference file.
Status: Open

**15. Word-level timestamps are computed five times and then thrown away**
`transcribe.py:run_transcription_pass` (`word_timestamps` on, used only for
confidence weighting; per-segment `confidence` dicts are built twice and
discarded) and `app.py:run_forced_alignment` (MFA word alignments reduced to
segment edges).
Recommended change: add an additive, optional `words` list to each `timings`
entry, `[{"start", "end", "text", "probability"}]`, from MFA when alignment
succeeded and Whisper otherwise (flagged: new optional key; `start`/`end`/`text`
and the `/status` shape unchanged). Effect: downstream gets word timing for
free. Risk: `timings` JSON grows about 5x in the DB; still small. Verify: GPU,
every `words[].start` lies within its segment and words are monotonic on the
reference job.
Status: Open

### P2 - polish

**16. Duration estimate formula is calibrated to the 5-pass era and over-estimates already**
`transcribe.py:transcribe_audio` and `app.py:upload_audio`
(`ceil(d/15.1)*5 + 45`). 755 s file: estimate 295 s vs about 185 s actual
(150 s Whisper + 24 s MFA + polling); 2783 s: 970 s vs roughly 600 s. After
items 6-7 it will be 3-4x high.
Recommended change: `d x (passes x rtf_whisper + rtf_mfa) + 45 s` with measured
RTFs (sequential beam 5 about 0.045, greedy 0.03, MFA about 0.035), and use the
per-job `processing_seconds` column from service item 19 for a rolling median.
`estimated_completion_utc` format unchanged. Verify: GPU, estimate vs
`completed_at - created_at` over a week.
Status: Open

**17. Stale MFA output is reused after a post-alignment garbage requeue**
`app.py:run_forced_alignment` ("alignment already exists" branch). A retry
re-transcribes but reuses `<guid>.json` built from the previous transcript.
Recommended change: delete `<guid>_aligned` on requeue or key the cache by a
transcript hash. Verify: Mac, force a requeue and confirm the aligned directory
is gone; GPU to confirm MFA reruns.
Status: Open

**18. Transcript cleaning regex before MFA changes the word count**
`app.py:run_forced_alignment` (`\b(\w+)\s+\1\s+\1\s+\1+`). Case-sensitive, only
4+ repeats, and it desynchronises MFA's word sequence from the segments (breaks
item 4's index mapping).
Recommended change: drop it; loops are handled upstream now. Verify: Mac.
Status: Open

**19. Model construction hygiene**
`transcribe.py:load_whisper_model` and the transcribe call. `num_workers=2` adds
a ctranslate2 replica for a strictly sequential worker (`LIMIT 1`); set 1.
`word_timestamps="all"` is a truthy string where 1.2.1 declares `bool`; use
`True`. `suppress_tokens=[-1]` is the default. `get_audio_duration` decodes the
whole MP3 with pydub; `ffprobe` is instant (the duration part is service item
10 and ships in 0.5.0; the model arguments are gated).
Status: Open

**20. Language pinning is correct; keep it**
`language="en"` skips per-pass detection and prevents mid-file language flips on
Hebrew/Greek terms. Keep `task="transcribe"` and `multilingual=False`
(defaults). For consistently misspelled congregation names, A/B `hotwords`
(inserted into every window prompt, so watch for the priming effect the old
`initial_prompt` had) rather than an initial prompt.
Status: Open

**21. OOV proper names align as spn in MFA**
`app.py:run_forced_alignment`. Biblical names outside `english_mfa` get `spn`
phones and coarse boundaries.
Recommended change: download the matching G2P model
(`mfa model download g2p`, verify the exact name for the mfa dictionary in 3.3)
and pass `--g2p_model_path`. Verify: GPU, OOV count in MFA's stdout drops.
Status: Open

**22. Garbage gate reads only the first 1000 characters**
`app.py:is_garbage_transcription`. Alnum ratio over the preview cannot see a
collapse after minute two. The wps floor covers total collapse; the per-window
check in item 6 should also feed this gate so partial collapse is requeued
rather than published.
Status: Open
