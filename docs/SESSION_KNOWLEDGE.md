# Session Knowledge

Accumulated, dated, non-obvious learnings about this service: what works, what
does not, and why. Read it before changing the decode configuration, the
dependency set, or the deployment path. Add an entry when you learn something
that is not recoverable from the code or git history, and correct an entry when
new evidence proves it wrong.

## 2026-09-07 - Decode collapse: three defects, one release

Production transcription fell to 0.15-1.24 words/sec on a set of 2024
teachings (normal material runs 2.5-2.8). Three defects in `transcribe.py`
combined to produce it, plus one parameter that was never reaching the model.

1. **Always-on denoise gate.** `preprocess_audio_for_transcription()` normalised
   to -20 dBFS and then measured "noise" as `mean(abs(samples)) / 32768` on the
   normalised signal. That is a crest-factor measurement of the speech itself,
   not a noise floor, so every file scored 0.046-0.060 against the 0.015
   threshold and the "only if noise is detected" branch was always taken. It then
   ran `noisereduce` with no noise profile over material that is about 78%
   speech, so the estimated profile was mostly speech: speech-active frames fell
   from 76% to 65%, dynamic range widened by 8 dB, and the result went through a
   second lossy MP3 encode. Deleted rather than repaired; faster-whisper
   normalises its own input and the untouched file decodes better.
2. **Single-speaker prompt.** `initial_prompt` asserted "a single speaker in
   clear English" on multi-voice class and Q&A recordings. The prompt is
   prepended as previous-text context, so it can prime the repetition loops it
   was meant to prevent. Removed; the configuration measured at 2.77 words/sec
   used no prompt.
3. **Scalar temperature disabling fallback.** `temperature` was passed as a
   scalar. faster-whisper wraps a scalar into a one-element list, which disables
   temperature fallback entirely, and fallback is the built-in escape from a
   decode that trips the compression-ratio or log-probability thresholds. Each
   pass now passes `temperature_ladder(base)`, a tuple from the pass's base up
   to 1.0 (above 0.5 faster-whisper also resets the previous-text prompt, which
   breaks a loop already under way).
4. **Pass 5 parameter never honored.** The passes list declared
   `condition_on_previous_text: False` on pass 5, but the `model.transcribe()`
   call hardcoded `True`. The call now reads it from the pass parameters.

Regression tests for all four live in `tests/test_transcribe_config.py`. They
load the real module with torch, faster-whisper and pydub stubbed and assert on
what actually reaches `model.transcribe()`.

**Word-rate floor in the quality gate.** `MIN_WORDS_PER_SEC` (env, default 1.0)
is checked in `process_pending_job` right after the alphanumeric-ratio check,
which is blind to coherent prose at half the normal rate. Words per second is
computed from the `duration_sec` that `transcribe_audio` now returns. The floor
is skipped for clips under 30 s and for unknown durations (`duration_sec` of 0),
and a failing job routes through the same retry/quarantine path as any other
garbage result. The success log line reports the job's words/sec so the floor
can be tuned from production logs.

**Beam search is inert on the sampled passes.** In faster-whisper 1.2.1 any
temperature rung above 0 decodes by sampling with `beam_size=1` and ignores
`beam_size` and `patience`. Passes 1, 4 and 5 have bases of 0.2, 0.3 and 0.2,
so they never beam-search and their listed beam sizes (5, 10, 15) do nothing;
only passes 2 and 3 (base 0.0) beam-search on their first rung. This is
pre-existing behavior kept as-is for this release; the measured numbers were
produced with it.

**Diarization was failing during validation.** The 2.78-2.82 words/sec numbers
were measured with pyannote diarization failing (401 on the gated
`pyannote/segmentation-3.0` model, which the build-time download does not
fetch) and falling back to one speaker, so MFA always ran. The feature has
since been removed (entry below), which is a no-op at runtime.

**Measured before/after** on the same 755 s file: 0.54 to 2.745 words/sec;
410 tokens of mostly punctuation to 2,073 words; runtime about 16 min to
3m32s. Post-fix production logs on a 2783 s file show 2.78-2.82 words/sec on
every pass with adjusted confidence around 0.98.

**Production package set that produced those numbers:** Python 3.13.11,
torch 2.12.1+cu130, ctranslate2 4.8.0, faster-whisper 1.2.1, pyannote-audio
4.0.5, MFA 3.3.10.dev0 from a June 2026 `:latest` base image. Host driver CUDA
13.0 on an RTX 3060 12 GB. `requirements.txt` now pins the direct dependencies
to exactly this set. numpy, scipy, MFA, kalpy, torchaudio and torchcodec come
from the base's conda environment and are deliberately not pinned. The conda
torchaudio in that image is built against the base's CPU torch; nothing in this
service imports it directly, and no torchaudio release matching torch 2.12.1
exists on PyPI (2.11.0 is the newest).

**The pinned v3.4.1 Dockerfile had never been built.** The base tag was pinned
from `:latest` to `:v3.4.1` in the 2026-07-29 CVE pass on a workstation that
cannot build or run the image (linux/amd64 + NVIDIA GPU + Hugging Face token
required). Every production image so far came from a `:latest` base. The first
full build from the pinned Dockerfile is a real upgrade and needs the
validation recipe below before it replaces anything. `Dockerfile.release` is
the overlay path that rebuilds only the application source on top of a retained
production image, so a code change can ship without touching the stack.

**Release process caveat.** The GitHub Actions workflow builds from the
from-scratch `Dockerfile` on every published release and pushes `:latest`. It
used to fail because it passed no `HUGGINGFACE_TOKEN_BUILD`; with diarization
removed (entry below) the build needs no secret and will succeed, and it will
then publish whatever the from-scratch build produces (pinned v3.4.1 base,
pinned requirements, freshly downloaded models), which is a different stack
from the one validated here. A GitHub release must therefore only be
published after that exact build has passed the validation recipe below on
the GPU host. Until then, ship code-only changes through `Dockerfile.release`.

**GLIBCXX note.** A bare `docker exec <container> python -c "import torch"`
fails with `GLIBCXX_3.4.29 not found` because the system libstdc++ is too old,
while the live process works because `/env/lib/libstdc++` is mapped into it.
For exec-based checks pass the library path explicitly:

```
docker exec -e LD_LIBRARY_PATH=/env/lib:/usr/local/cuda/lib64 <container> \
    python -c "import torch, ctranslate2; print(torch.cuda.is_available(), ctranslate2.get_cuda_device_count())"
```

### Validation recipe

Run this on the GPU host after any change to the decode configuration, the
dependency pins, or the base image, before the image replaces production.

- File: `/home/jay/SourceCode/SermonPreprocessorAPI/data/audiofiles/tcf.20240213b.mp3`
  (755 s).
- Expect 2.5-2.8 words/sec in the `Pass N stats` log lines. The 2026-09 collapse
  produced 0.54 on this file.
- Read the first and last lines of the transcript. Word count alone is satisfied
  by a hallucination loop, so check that the ending is real material and not a
  repeated phrase.
- Check the per-pass weighted and adjusted confidence in the logs; a healthy run
  sits around 0.97-0.98 with no pass penalised for word rate.
- Inside the container (with the LD_LIBRARY_PATH above) confirm
  `torch.cuda.is_available()` is `True` and `ctranslate2.get_cuda_device_count()`
  is at least 1. A CPU fallback runs, slowly, and looks like a decode problem.

## 2026-09-07 - 0.5.0: service hardening, no decode changes

`docs/PUNCHLIST.md` is the merged service and pipeline review; 0.5.0 closes
the service items that a Mac unit test can verify and leaves every pipeline
item open. Nothing in the decode call, the Dockerfile CUDA lines or the base
image changed, so the 0.4.0 validation numbers still apply to this release.
The contract of `/upload`, `/status` and `/transcriptions` is unchanged;
`GET /health` and four per-job metric fields (`processing_seconds`,
`words_per_second`, `attempt_count`, `mfa_applied`) are additive.

What changed, in one place:

- Cleanup is `cleanup_old_jobs()`, runs on the first wake and then hourly on
  a monotonic clock whether or not the queue is empty, and deletes files and
  rows for the same GUID set. Rows are aged by `COALESCE(completed_at,
  created_at)`, so a job that waited a day in the queue and finished minutes
  ago is not deleted from under a polling client. `sweep_orphaned_files()` removes GUID-named
  entries in `UPLOAD_FOLDER` older than two days with no pending or
  processing row; that folder is a host bind mount and outlives the
  container-layer database, so it is the only way pre-redeploy files go away.
- `recover_stuck_jobs()` runs at the top of every worker cycle. With a single
  synchronous worker any 'processing' row seen there is orphaned by
  definition.
- Exceptions from transcription go through the attempt counter like garbage
  results: requeued below `MAX_GARBAGE_RETRIES`, 'error' at it. A completed
  row now keeps its `attempt_count` as the retry history instead of being
  reset to 0; nothing re-reads it once the job is terminal. The outer
  handler in `process_pending_job` only counts a failure if the row is still
  'processing', so an inner handler that raised after writing cannot cause a
  double count.
- The worker loops straight into the next job after a terminal outcome and
  sleeps `POLL_INTERVAL_SEC` when the queue is empty or when the job it just
  ran was requeued; without that spacing an instantaneous failure (OOM,
  unreadable file) burned all three attempts in under a second. The idle
  transition is logged once; wake/sleep lines are DEBUG.
- `/health` returns 503 when the worker thread is dead or an idle worker has
  not polled in five minutes. A worker mid-job is reported `worker_busy` and
  is exempt from the staleness rule, otherwise a 46 minute file would mark
  the container unhealthy. Both Dockerfiles carry a `HEALTHCHECK` using
  `wget` (the image has no curl) with a 180 s start period for the model load.
- `/upload` holds a lock across check, insert and save, in that order: the
  row is claimed before any bytes are written so a duplicate never overwrites
  the winner's file, a failed save deletes the row and returns 500, and an
  undecodable file removes both file and row and returns 400. The duplicate
  check is case-insensitive (`lower(guid)`), matching the orphan sweep. The
  worker takes the same lock around its pending-row SELECT so it cannot pick
  up a row whose file has not been saved yet. GUIDs must be in canonical
  hyphenated form (uppercase hex is accepted and echoed as sent), extensions
  are allowlisted and lowercased on disk, bodies are capped at 1 GiB.
  Werkzeug's 413 is an `HTTPException`; the handler's broad
  `except Exception` has to re-raise it or it becomes a 500.
- `Dockerfile.release` copies only `app.py`, `transcribe.py` and
  `requirements.txt`; `checkQueue.py` is host tooling and is excluded by
  `.dockerignore`, so listing it in the COPY fails the overlay build.
- Duration comes from ffprobe via `pydub.utils.mediainfo`, with a full decode
  only when the header has no duration. `estimate_processing_seconds()` is the
  one copy of the `ceil(d / 15.1) * 5 + 45` formula. `/upload` and `/status`
  both count pending and processing jobs, and `/status` never reports an ETA
  in the past.
- `run_forced_alignment` returns `(segments, mfa_applied)`, passes `--clean`,
  and removes both its input directory and MFA's working tree under
  `MFA_ROOT_DIR` (default `/mfa`) in a `finally`. Empty alignment output falls
  back to Whisper timings without spending a retry. `transcribe.py` no longer
  writes the MFA transcript file.
- CI: `Test.yml` runs ruff and pytest on push and pull request and is called
  by the publish workflow, which `needs` it. `ruff.toml` pins the rule
  selection because ruff 0.16 ships a different default rule set from 0.15
  and `requirements-dev.txt` only sets a version floor; without the file the
  lint step went from clean to 37 findings on a tool upgrade alone.
- `runDocker.sh` builds into a `git describe` tag before stopping anything,
  retags the serving image `transcription-api:rollback`, and polls `/health`
  for up to three minutes after starting the new container.

CI note: the 0.4.0 release run pushed the image but failed the Trivy gate on
base-image and cuda-toolkit findings; 64 of the 80 were in the Nsight Systems
Go binary that `cuda-toolkit-12-2` installs and the service never runs. That
is exactly what the deferred `cuda-toolkit-12-2` to `cuda-runtime-12-2` change
in the Dockerfile header (service item 29) addresses, and it stays gated on
GPU validation.

Test suite: 127 tests, under half a second, `app.py` at 92% line coverage (was 42%).
The routes are exercised with Flask's test client against a per-test SQLite
file; the worker loop is tested through `worker_cycle()` and a `time.sleep`
stub that raises to break the loop. One test-client limitation: a filename
containing a newline is dropped by Werkzeug's multipart encoder before the
server sees it, so the extension allowlist test uses `a.mp3.sh` instead.

### Still needs GPU verification (devmachine)

None of these can be observed on the Mac; check them on the first 0.5.0
deploy before calling the release done.

- Service item 3: after a job, `ls /mfa` inside the container should show
  only `pretrained_models`. If MFA 3.4 names its working directory
  differently from `<corpus basename>`, the `finally` removes nothing and the
  `--clean` flag is the only thing keeping the tree bounded.
- Service item 15: `docker inspect --format '{{.State.Health.Status}}'
  transcription-api` should read `healthy` once the model has loaded, and
  `/health` should stay 200 through a long job (the `worker_busy` exemption).
- Service item 17: run `runDocker.sh` once end to end; confirm the
  `transcription-api:rollback` tag exists afterwards and that the health poll
  reports the JSON body rather than timing out.
- `get_audio_duration` now shells out to `ffprobe`; confirm the upload log
  line shows a sensible `Duration:` for an MP3 and that a corrupt upload gets
  a 400 rather than a 500.

## 2026-09-07 - Speaker diarization removed

pyannote speaker diarization and the multi-speaker MFA-skip branch in
`run_forced_alignment` are gone. MFA is now always attempted, with the existing
fallback to Whisper timings when it fails.

- 98% of the recordings this service handles are single speaker, so the branch
  had almost nothing to decide.
- It never worked in production. The Dockerfile snapshotted only the
  `pyannote/speaker-diarization-3.1` pipeline repo, not the gated sub-models it
  loads (`pyannote/segmentation-3.0` and the embedding model), so every job hit
  a 401 at load time, logged "assuming single speaker", and took the MFA path
  anyway. Removal is therefore a no-op at runtime; the validated 2.78-2.82
  words/sec numbers were produced in exactly this state.
- Repairing it would have cost a GPU pass per job and would have been the first
  code path to decode audio through the base image's conda torchcodec, which is
  built against the base's CPU torch while pip replaced torch with 2.12.1+cu130.
  That mismatch is untested and not worth exposing for a 2% case.
- Dropping it removes the only build-time secret (`HUGGINGFACE_TOKEN_BUILD`),
  so the CI weekly image refresh can run unattended, and `pyannote.audio` leaves
  the pinned requirements along with its dependency tree.

If multi-speaker handling is ever wanted again, treat it as a new feature with
its own validation on the GPU host, not as a restore of this code.

## 2026-09-07 - Service is stateless by design

The SQLite queue lives in the container layer and is lost on redeploy. That is
intentional: clients treat a 404 from `/status/<guid>` as "resubmit", so a
fresh container starts with an empty queue and the callers refill it. Do not
add a volume for the database and do not add WAL checkpointing to preserve it;
either one turns a redeploy into a migration problem for state that nobody
needs to survive.

## 2026-09-07 - 0.4.0 validation record

Reference file tcf.20240213b.mp3 (755.2 s), both from-scratch builds on the
mmcauliffe/montreal-forced-aligner:v3.4.1 base with the pinned requirements.

| Build | Words | Words/sec | Pass confidence | Whisper time | MFA | Notes |
|---|---|---|---|---|---|---|
| 0.4.0-rc1 (with diarization) | 2081 | 2.756 | 0.9735 to 0.9753 | 145 s | first attempt | crashed at startup until LD_LIBRARY_PATH put /env/lib first; diarization 401 as before |
| 0.4.0-rc2 (release) | 2079 | 2.753 | 0.9736 to 0.9797 | about 145 s | first attempt | no pyannote or torchcodec lines in the log |

Both transcripts open on the home-groups talk and close on "So, questions or
comments?" with timings spanning 0 to 754.7 s. Environment inside the image:
Python 3.13.14, torch 2.12.1+cu130 with CUDA available on the RTX 3060,
ctranslate2 4.8.0 seeing one device, MFA 3.4.2.dev0 (what the v3.4.1 image tag
actually ships), Ubuntu 20.04.5. Whole job about 3.5 minutes end to end.

## 2026-09-07 - 0.5.0 validation record

0.5.0-rc1, from-scratch build on the same v3.4.1 base and pins as 0.4.0, on the
reference file tcf.20240213b.mp3: 2081 words, 2.756 words/sec, 127 timings, MFA
aligned on the first attempt, 145 s for the five Whisper passes and 180 s end to
end. The completed status body and the /transcriptions row both carried the
additive fields (processing_seconds 180.0, words_per_second 2.756, attempt_count
0, mfa_applied true). GPU-only checks: /health returned 200 six seconds after
start and Docker reported healthy; /health showed worker_busy true with status ok
while a job ran; zero <guid>_mfa_input directories remained under /mfa after the
job; corrupt and wrong-extension uploads returned 400; a duplicate GUID returned
409 in both lower and upper case; no worker sleep/wake lines in the log. The
upstream client resubmitted its in-flight jobs within minutes of the container
swap, as the stateless design intends.

## 2026-09-07 - 0.5.1: base image v3.4.2

Dependabot PR #8 applied by hand (the Dockerfile had moved too far for a clean
merge). The first candidate died at import: pydub needs the audioop module that
Python 3.13 removed, and the v3.4.1 base had carried the audioop-lts backport as a
side effect while v3.4.2 does not. audioop-lts==0.2.2 is now declared in
requirements.txt. Lesson: anything the service imports must be pinned in
requirements.txt, never inherited from the base's conda environment.

Validation of 0.5.1-rc2 on the reference file: 2086 words, 2.762 words/sec, 187
timings, MFA aligned on the first attempt, 139 s for the five Whisper passes,
170 s end to end, zero MFA working directories left, Docker healthy. Image
environment: Python 3.13.15, torch 2.12.1+cu130 with CUDA available, ctranslate2
4.8.0 seeing one device, numpy 2.5.2, MFA 3.4.3.dev0 (what the v3.4.2 tag ships),
Ubuntu 20.04.5.

## 2026-09-07 - 0.5.2: cuda-libraries instead of the toolkit, no CUDA_LAUNCH_BLOCKING

cuda-toolkit-12-2 replaced by cuda-libraries-12-2 (cuda-runtime-12-2 was rejected:
apt-cache shows it depends on cuda-drivers, which would install the NVIDIA driver
inside the container). ctranslate2 links no CUDA library at load time (ldd shows
only its own libctranslate2 and libgomp); it dlopens libcudart, libcublas and
libcudnn at first use, which cuda-libraries-12-2 plus libcudnn9-cuda-12 provide.
torch brings its own CUDA 13 libraries from pip. CUDA_LAUNCH_BLOCKING=1 removed.

Validation of 0.5.2-rc1 on the reference file: 2079 words, 2.753 words/sec, 135
timings, MFA aligned on the first attempt, torch CUDA available and ctranslate2
seeing one device. Five Whisper passes took 106 s against 139 to 145 s on 0.4.0,
0.5.0 and 0.5.1, so the synchronous-launch setting had been costing roughly a
quarter of the decode time. End to end 138 s against 170 to 180 s.

## 2026-09-07 - 0.5.3: CI model download rate limit

The 0.5.1 and 0.5.2 release runs (34099157643, 34099726604) failed in the
Dockerfile's `mfa model download` step with MFA's ModelsConnectionError:
"Current hourly rate limit (60 per hour) has been exceeded for the GitHub
API". `mfa model download` resolves models through the GitHub releases API of
MontrealCorpusTools/mfa-models; unauthenticated that API allows 60 requests
per hour per source address, and the shared GitHub Actions runner pool
exhausts it, while the same build on the GPU host succeeds because it is the
only caller from its address.

Fix: the workflow passes `${{ secrets.GITHUB_TOKEN }}` to build-push-action as
a BuildKit secret (`secrets: github_token=...`), and the Dockerfile mounts it
with `RUN --mount=type=secret,id=github_token` and appends `--github_token` to
both `mfa model download` calls only when `/run/secrets/github_token` is
present. A local build supplies no secret and downloads unauthenticated as
before. GITHUB_TOKEN is rate limited at 1,000 requests per hour per repository
and needs no scope for public releases. Secret mounts are not layers, so the
token is not in the image or its history (the concern that closed service
item 4 in 0.4.0). No `# syntax=` directive is needed: `RUN --mount` is in the
stable Dockerfile reference and BuildKit has been the default builder since
Docker 23.0; `runDocker.sh` exports `DOCKER_BUILDKIT=1` anyway so a stray
`DOCKER_BUILDKIT=0` in the calling shell cannot select the legacy builder.

Release-run history since 0.4.0: 0.5.0 was the only run whose image build
completed (0.5.1 and 0.5.2 died at the MFA download). The 0.4.0 and 0.5.0 runs
pushed their images and failed only at the Trivy gate, on the base-image and
cuda-toolkit findings that 0.5.2's cuda-libraries change addresses.

Validation of 0.5.3-rc1 on the reference file: 2084 words, 2.760 words/sec, 194
timings, MFA aligned on the first attempt, 103 s for the five Whisper passes,
134 s end to end, zero MFA working directories left. The local build ran the MFA
download unauthenticated through the new conditional, as intended.

## 2026-09-07 - 0.5.4: residual CVE pass

The 0.5.2 release run (34099726604) pushed its image and failed only at the
Trivy gate (CRITICAL/HIGH, fixable, `vuln-type os,library`): 1 Ubuntu finding
and 15 Python findings. A local scan of `transcription-api:0.5.3` with the
current Trivy database counts 16 (one more vendored setuptools entry, below).
None of them were in the environment the service runs from. Where each lived,
found by inspecting the 0.5.3 image on the GPU host, and what fixed it:

- `libnghttp2-14 1.40.0-1build1` (CVE-2023-44487): Ubuntu 20.04 package in the
  MFA base image. `libnghttp2-14` is now on the existing apt-get install line,
  which pulls the focal-security build (1.40.0-1ubuntu0.3) in the same layer.
  No blanket `apt-get upgrade`.
- `certifi 2022.12.7`, `cryptography 39.0.1`, `pyOpenSSL 23.0.0`,
  `setuptools 65.6.3`, `urllib3 1.26.14` (13 findings between them): live
  site-packages of `/opt/conda`, the mambaforge 22.11 environment the MFA
  image was bootstrapped from (Python 3.10.9, conda 22.11.1, pip 23.0). There
  is no `/opt/conda/pkgs` cache; the `(PKG-INFO)` label in the report just
  means certifi and setuptools are egg-info installs there. The service never
  touches this environment: ENTRYPOINT is tini, CMD is `python app.py`, PATH
  puts `/env/bin` ahead of `/opt/conda/bin`, and `/env/bin/mfa` has an
  `/env/bin/python` shebang. Fix: one RUN with `/opt/conda/bin/pip install`
  of those five packages plus `requests` (2.28.2 pinned `urllib3<1.27`, and
  pyOpenSSL 23 pinned `cryptography<40`), followed by `pip check`. Result:
  certifi 2026.7.22, cryptography 50.0.1, pyOpenSSL 26.4.0, setuptools 84.0.0,
  urllib3 2.7.0, requests 2.34.2, cffi 2.1.1; `pip check` clean; conda
  22.11.1 still answers `--version` and `info`.
- `msgpack 1.1.2` (GHSA-6v7p-g79w-8964) and a second `setuptools 70.3.0`
  (CVE-2025-47273), both reported without a file path: these are the
  libraries pip 26.2.1 vendors inside `/env`, declared in
  `pip/_vendor/bom.cdx.json`, a CycloneDX SBOM that Trivy reads. `/env`'s own
  msgpack is 1.2.1 and its setuptools is 81, both clean. pip 26.2.1 is the
  newest release on PyPI (2026-08-04) and its `vendor.txt` still lists
  `msgpack==1.1.2` and `setuptools==70.3.0`, so no pip upgrade clears them.
  Fix: `python -m pip uninstall -y pip` in a RUN after the Whisper bake, so
  the requirements and model layers stay cached. Nothing runs pip in the built
  image (`validate-rc.sh` execs only `python` and `mfa`). Deleting just
  `bom.cdx.json` was rejected: it hides the finding and leaves the code.

Trivy on `transcription-api:0.5.4-rc1` on the GPU host: 0 findings on every
target. The scan also warns "Third-party SBOM may lead to inaccurate
vulnerability detection"; that is the Qt SPDX files under `/env/lib/qt6/sbom`
and av's `auditwheel.cdx.json`, all clean, not a finding. Image 15.9 GB, the
same as 0.5.3. In the rc image torch reports CUDA available and ctranslate2
one device; `mfa version` is 3.4.3.dev0+gd2dc283bd.d20260820, unchanged.
`Dockerfile.release` needs nothing: it overlays code on a retained image.

Adding a package to the apt line invalidates every layer after it, so the
first build after this change re-downloads the MFA models, the requirements
and the Whisper model. On the GPU host that was 3.5 minutes with warm caches.

Validation of 0.5.4-rc1 on the reference file: 2077 words, 2.750 words/sec, 146
timings, MFA aligned on the first attempt, zero MFA working directories left.
The five Whisper passes took 194 s because the quality experiment harness was
sharing GPU 0 at the time; that is contention, not a regression (0.5.2 and 0.5.3
measured 103 to 106 s on an idle GPU).
