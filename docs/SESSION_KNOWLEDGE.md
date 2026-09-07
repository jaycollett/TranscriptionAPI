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

## 2026-09-07 - Quality analysis and harness results

Two code analyses of the decode path and the post-Whisper pipeline, and a
harness run of their experiments on six reference files inside the 0.5.2 image,
are written up in `docs/QUALITY_PROPOSAL.md`; the measured tables and every
number behind them are under `tools/quality_harness/results/2026-09-07/`. The
short version: one beam-5 pass reproduces the five-pass winner 4-8x faster, the
duration-weighted confidence prefers the pass with fewer words (it dropped 119
words of Psalm 91 on the retreat file), per-utterance MFA aligns the 1369 s file
that whole-file MFA cannot, and no human-corrected transcript exists so none of
this is a word error rate. The proposal defines release 0.6.0 and its
acceptance thresholds against that results directory.


## 2026-09-07 - 0.6.0: single-pass decode, anomaly gate, per-utterance alignment

The first release that changes what the service transcribes. Everything here
implements a measured winner from `docs/QUALITY_PROPOSAL.md`; the numbers cited
are from `tools/quality_harness/results/2026-09-07/`, and nothing was adopted
that the harness had not run on all six reference files.

**One pass replaces five.** `beam_size=5`, `best_of=5`, `patience=1.0`, ladder
(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), `condition_on_previous_text=True`,
`prompt_reset_on_temperature=0.5`, `word_timestamps=True`, no prompt, no
hotwords, `num_workers=1`. The five-pass loop was never doing what it looked
like it was doing: passes 1, 4 and 5 sampled at temperature 0.2-0.3, where
faster-whisper ignores `beam_size` and `patience` entirely, so the selector was
ranking four noisy variants of a single beam decode. The single pass agrees
0.987-1.000 with the production winner and produces more words on every file, at
20 s instead of 103 s on the 755 s file and 70 s instead of 360 s on the
2783 s file. Every parameter is an environment variable, so the decode can be
retuned on a running container; the defaults are the measured configuration and
changing one invalidates the harness baseline.

**The file's level selects a whole VAD profile, not a threshold.** The
conservative C4 settings (threshold 0.5, `min_silence_duration_ms` 1000,
`speech_pad_ms` 300) cut seams 3-4x and were the best-aligning decode of any
config on the 755 s file. On the one quiet file in the 655-file archive they are
a disaster: at -28.4 dBFS they fragment speech into 1157 segments and lose 3.0
percent of the words, and mean log-probability falls from -0.056 to -0.162.

Relaxing only the threshold to 0.35 does not fix it and makes it worse: measured
on that file, threshold 0.35 with C4's 1000 ms silence and 300 ms padding gives
4693 segments and 8407 words, a 5.6 percent loss. The BASE/C1 profile entire
(threshold 0.35, 300 ms silence, 400 ms padding, no hallucination filter) gives
435 segments and 8904 words and reproduces C1 exactly. **The long minimum silence
is what shreds quiet audio, not the threshold and not the hallucination filter.**
That is the non-obvious part and it is why the level picks a profile rather than
a number:

  loud  (at or above -26 dBFS): 0.5 / 250 / 1000 / 300, hallucination filter 0.5.
  quiet (below -26 dBFS): 0.35 / 250 / 300 / 400, no hallucination filter.

An unmeasurable level takes the quiet profile, because fragmenting quiet speech
loses words while cutting extra seams on loud speech does not.

**The -26 dBFS cutover is provisional and should not be read as settled.** It sits
between the quietest file that worked on the loud profile (-24.4 dBFS) and the one
that failed (-28.4 dBFS), which is a boundary calibrated on two recordings. The
100-file corpus sweep then measured the level distribution and found 34 files within
2 dB of the line, with the histogram densest right at it: a third of the archive is
decided by a threshold two files placed, and a file drifting 1 dB between recordings
would flip profiles and change its output. It is left at -26 for the sweep so both
sides get measured. Expect it to move, or to be replaced by something that does not
put a hard step through the densest part of the distribution. The level comes from `ffmpeg -af
volumedetect`, costs a full decode of the audio (a few seconds on a sermon), and
the level, the profile and the parameters are logged on every job.
`hallucination_silence_threshold` is 0.5 on the loud profile, not the 2.0 the
punchlist proposed: the VAD pads every gap by 300 ms each side, so a 2.0 s
threshold can never fire. It is off on the quiet profile, whose 400 ms padding
puts 800 ms of silence in every gap and which was measured without it.

**The anomaly score replaces the duration-weighted word probability.** The old
metric weighted each word's probability by its duration, which up-weights
exactly the words to distrust, since Whisper gives hallucinated words in silence
long durations. It disagreed with the anomaly rule on four of the six files and
in every disagreement it preferred the pass with fewer words; on the retreat
recording it ranked the passes in reverse order of word count and its winner had
dropped two runs of Psalm 91 that the losing pass contained. `anomaly_count`
counts segments at temperature 0.5 or above, compression ratio over 2.4, average
log-probability under -1.0, or `no_speech_prob` over 0.5 with non-empty text.
`anomaly_windows` counts 60 s windows of speech carrying under 1.2 words/sec,
which is the only signal that sees a partial collapse: a file that transcribes
normally for forty minutes and produces nothing for ten still clears the
whole-file `MIN_WORDS_PER_SEC` floor.

The window clock skips silence, or a ten-minute break before the Q&A would read
as a collapse. faster-whisper does not hand back the VAD chunks it used and
re-running the VAD would mean decoding the audio a second time, so the segments'
own spans stand in for them: under `vad_filter` a segment only exists where the
VAD found speech. This is a deliberate approximation and it is the one place the
production metric differs in construction from the harness's, which had the real
chunks.

**Segment spans alone cannot see an omission, and the obvious fix reports one on
every healthy file.** Audio the decoder emitted nothing for contributes no span,
so an omission shrinks the clock rather than showing up as a low-rate window: the
surviving 40 minutes of a 50 minute file look perfectly healthy. The whole-file
floor is not a backstop either, because at 2.6 words/sec about 62 percent of a
file has to vanish before the 1.0 floor trips. So `duration_after_vad` is now read
before the pass is scored and speech no segment covers is charged to the window
count.

The trap is charging it raw. Segment spans never tile speech exactly, and the
sub-second pauses between segments sum to minutes over a sermon. On
`tcf.20240213a` the uncovered time is 373 sub-second gaps whose largest single
member is 1.8 s: there is no omission anywhere in it. Charging raw uncovered time
fired on 2 of 6 healthy files, triggered the rescue on both, and on
`tcf.20240213a` the rescue was then selected and published 9 fewer words at lower
agreement than the primary. The gate made the output worse on a file that was
fine. `ANOMALY_UNCOVERED_TOLERANCE` is the share of speech a decode may leave
uncovered before any of it counts, and it is scale-free on purpose: a 60 s
absolute bucket is meaningless without knowing whether the file is 12 minutes or
56.

**The measure has to be bounded by what it is compared against, and the obvious
implementation is not.** The first version summed segment spans, merged only
against the immediately preceding span, and compared that against
`duration_after_vad`. Those are different quantities: Whisper's timestamps are
padded by the VAD and can overlap each other or run past the speech region, so the
sum is unbounded above. The corpus sweep found coverage ratios up to 1.087 on 5 of
its first 10 files, and on 3 of the 6 reference files the naive ratio is 1.019 to
1.036. Above 1.0 the uncovered figure is zero or negative, the tolerance can never
fire, and the omission check is silently dead again, one step downstream of where
it was dead before.

Coverage is now the merged segment spans **intersected** with the voice-activity
intervals, which is bounded by construction; the same six files come out at 0.919
to 0.989. Two details matter. The merge has to sort first, because a merge that
only checks its immediate predecessor double counts unsorted input, and Whisper
timestamps do arrive slightly out of order after `restore_speech_timestamps`. And
the intervals themselves are required: a total cannot be intersected with
anything, which is why `duration_after_vad` is not enough.

**So the service now runs the voice-activity detector itself.** faster-whisper
reports the total but not the regions. The audio is decoded once with
`decode_audio` and handed to `model.transcribe` in place of the path, which is
what faster-whisper would have done internally, so the decode is not repeated and
the marginal cost is one detector pass, logged per job with the region count and
seconds. A failure there falls back to the file path and disables the omission
check loudly rather than losing the job.

Measured in the 0.5.4 image on the GPU host, decode plus detector is 1.58 s on the
755 s file and 6.38 s on the 3388 s retreat file, about 0.2 percent of real time.
That figure includes the audio decode, which the model then reuses instead of doing
itself, so the true marginal cost of getting the intervals is lower still. It is a
rounding error against a 21 to 80 s decode, and the check is worthless without it.

Correcting the arithmetic also moved the tolerance. The healthy worst case was
measured at 7.1 percent using the inflated coverage; intersected, the healthy
range across the six files is 1.1 to 9.0 percent, worst on `tcf.20240319b`. A 10
percent tolerance would have shipped with 12 s of headroom on a 1259 s file, which
is not a tolerance, so the default is 15 percent. All six measured coverages are
pinned as tests. Six files is still not a distribution and the sweep should set
this from the corpus.

**A rescue pass, not five of them.** (The segment-count half of the trigger and the
quarantine gate described in this section were retired later the same day; see the
entry on retiring the per-segment anomaly count. What follows is the state as first
built.) The redundancy the five-pass design was
reaching for is kept, but paid for only where it is needed. If the primary pass
scores any anomaly at all (`RESCUE_ANOMALY_SEGMENTS` 1, `RESCUE_ANOMALY_WINDOWS`
1) exactly one rescue pass runs, so the worst case is two passes. The rescue
changes two things and nothing else: `condition_on_previous_text=False`, which is
the one lever that stops a repetition loop feeding itself from window to window,
and a ladder starting at 0.2, because the primary already decoded this audio at
0.0 and produced the anomalies, so repeating that rung is the one outcome
guaranteed not to help. Model, beam, VAD threshold and every faster-whisper
threshold are identical, so a difference in the result is attributable to those
two changes rather than to noise. Selection is on the anomaly score: lower
`anomaly_count + anomaly_windows`, then more words, then higher mean
`avg_logprob`, with an exact tie keeping the primary. Word count is the first
tie-break precisely because every disagreement the retired confidence rule got
wrong was one where it preferred the shorter transcript.

That tie-break is not sufficient on its own, and the validation run proved it. A
rescue is only eligible at all if it keeps at least `RESCUE_MIN_WORD_RETENTION`
(0.99) of the primary's words. Measured on `tcf.20240319b` with the trigger
forced on: the primary scored one anomalous segment out of 323 with 3599 words
and the rescue scored none with 3525, so on the anomaly score alone the service
would have published 74 fewer words to remove a single flagged segment. Any
metric that counts defects prefers the transcript with less content to be
defective about; that is the retired rule's failure wearing a new hat, and every
such rule in this service needs a content floor beside it. A discarded rescue is
logged with both word counts and both scores.

The order is primary, maybe rescue, select, then gate. The quarantine thresholds
(`ANOMALY_SEGMENTS_MAX` 5) see the selected result, so a
file only the rescue could save is not requeued on the primary's score. The
rescue trigger sits far below the gate on purpose: a second opinion is cheap and
a requeue is not.

**The loop flag is a marker, never a rejection.** Segments whose repeated 4-gram
rate exceeds 0.3 are flagged `loop`. All six such segments across the reference
set were genuine rhetorical repetition ("Eternal life is knowing God. Eternal
life is knowing God.", "willing to open your home, willing to open your home");
no decoder loop occurred on any file. So the flag is surfaced in
`flagged_segments` for review and is deliberately excluded from the count that
can requeue a job. If a first-week job is flagged for a real loop, that is the
evidence needed to promote it.

**The seam de-duplication test is temporal, not lexical.** Word count cannot tell
a decoder artifact from a speaker saying something twice, and the corpus sweep
proved it: ten trims across its first eight files, none of them an artifact,
including four consecutive segments emptied on 1 Kings 18:39, "The LORD, he is
God; the LORD, he is God", an acclamation whose entire force is the doubling, and
two more on "I want to love" as anaphora. Raising the minimum from 2 to 4 words
did not help, because length is not the distinguishing feature.

A seam artifact is one stretch of audio decoded twice, so the repeated words at
the end of segment N and the start of segment N+1 describe overlapping time
ranges. A speaker saying it twice produces two sequential, disjoint ranges. The
word timestamps separate the cases cleanly and nothing else does. Only an overlap
is trimmed; the overlap in seconds goes on the log line so the rule can be scored
from production.

`BOUNDARY_DEDUPE_OVERLAP_SLACK_SEC` defaults to 0.0, not to a small positive
value, and that is deliberate: continuous speech abuts. A speaker repeating a
phrase across a segment boundary ends one occurrence and begins the next within
tens of milliseconds, so treating abutment as overlap puts every anaphora straight
back in scope. Raise it only on evidence that real artifacts are being missed.

A trim never empties a segment. If the repeated run is the whole segment the trim
is declined and logged, because dropping segments whole is exactly how the
scripture passage vanished. `BOUNDARY_DEDUPE_ENABLED` turns the whole step off in
one variable so that decision stays a config change.

The measured effect on the six reference files: the word-count rule made one trim,
on the class file, and the temporal rule keeps it. Its two occurrences are 0.86 s
apart, so that one apparent true positive was repetition too, and across the
entire reference set there is no evidence a seam artifact has ever occurred.

**Text and timings are one sequence again.** `clean_boundary_duplicates` is
deleted. It ran a repeated-phrase regex over the transcript string and left the
timings untouched, so `" ".join(t["text"] for t in timings) == transcription` was
false on every production job (2-27 words per file) and a second regex in
`run_forced_alignment` made MFA's word sequence a third variant. It was also
position-free, so it deleted legitimate repetition anywhere in the file: "He is
risen. He is risen indeed." became "He is risen.  indeed.", "pray without
ceasing. Pray without ceasing." became "pray without ceasing. .", and "day by day
by day" became "day by  day". The replacement only ever looks across a segment
seam, trims a 2-5 word phrase repeated there, moves the start to the first
surviving word's timestamp, and edits the segments themselves. The invariant is
now a test.

**Alignment: the utterance is the unit of failure.** Production wrote one `.txt`
beside the WAV, so a 20-60 minute file was a single alignment graph and one
mismatch could take the whole thing down. It did: the 1369 s file failed both
beam attempts in 279 s and produced no alignment at all, and the 1463 s file
needed the 100/400 retry at 195 s. A TextGrid of 8-30 s utterances, split only at
gaps of 0.4 s or more and padded 0.15 s into the gap, aligned all six files on
the first attempt at MFA's default beams in 24-35 s, including the file
whole-file MFA could not do. Failure is now bounded by the utterance: on the
1369 s file, 43 of 326 segments keep Whisper timings where the old path lost
everything. The WAV goes through `ffmpeg -ac 1 -ar 16000 -sample_fmt s16` instead
of a source-rate pydub export, which takes the 2783 s file from 534 MB to 89 MB
and means MFA hears the same channel mix Whisper decoded.

Words are assigned to segments by `difflib.SequenceMatcher` over normalised
tokens rather than by "MFA word start falls inside the Whisper span". The window
rule double-assigned boundary words, started segments on their second word, and
degraded to nonsense under drift (p95 start delta 17.5 s on the class file under
the old path against 0.65 s under this one). Sequence matching lifted agree250
from 0.44-0.65 to 0.52-0.79 and produced zero non-monotonic or overlapping
segments on any run. A segment MFA did not cover falls back to Whisper's own word
timestamps, not the segment bounds, which carry the VAD's padding. agree250 is
logged per job so a drop is visible without rerunning the harness.

**No ground truth exists.** Every transcript in the archive was produced by this
service and none has been human-corrected, so nothing here is a word error rate.
Agreement between configurations is a consistency measure. Where two configs
disagreed the differing runs were read by hand; that is how the Psalm 91 loss was
found, and it is the only reason word count is trusted as a tie-break.

**Environment variables added.** Decode: `WHISPER_BEAM_SIZE`, `WHISPER_BEST_OF`,
`WHISPER_PATIENCE`, `WHISPER_TEMPERATURE_BASE`, `WHISPER_TEMPERATURE_STEP`,
`WHISPER_COMPRESSION_RATIO_THRESHOLD`, `WHISPER_LOG_PROB_THRESHOLD`,
`WHISPER_NO_SPEECH_THRESHOLD`, `WHISPER_PROMPT_RESET_ON_TEMPERATURE`,
`WHISPER_HALLUCINATION_SILENCE_THRESHOLD`, `WHISPER_NUM_WORKERS`,
`WHISPER_LANGUAGE`. VAD: `VAD_THRESHOLD`, `VAD_THRESHOLD_QUIET`,
`VAD_LEVEL_CUTOVER_DBFS`, `VAD_MIN_SPEECH_DURATION_MS`,
`VAD_MIN_SILENCE_DURATION_MS`, `VAD_SPEECH_PAD_MS`,
`VAD_MIN_SILENCE_DURATION_MS_QUIET`, `VAD_SPEECH_PAD_MS_QUIET`. Rescue:
`RESCUE_ENABLED`, `RESCUE_ANOMALY_WINDOWS`,
`RESCUE_TEMPERATURE_BASE`, `RESCUE_MIN_WORD_RETENTION`, `RESCUE_MAX_WORD_LOSS`.
Anomaly windows: `ANOMALY_UNCOVERED_TOLERANCE`. De-duplication:
`BOUNDARY_DEDUPE_ENABLED`, `BOUNDARY_DEDUPE_MIN_WORDS`,
`BOUNDARY_DEDUPE_OVERLAP_SLACK_SEC`.
Alignment:
`MFA_UTTERANCE_GAP_SEC`, `MFA_UTTERANCE_PAD_SEC`, `MFA_UTTERANCE_MIN_SEC`,
`MFA_UTTERANCE_MAX_SEC`, `MFA_TIMEOUT_FLOOR_SEC`, `MFA_TIMEOUT_BASE_SEC`,
`MFA_TIMEOUT_PER_SEC`. Estimate: `PROCESSING_REALTIME_FACTOR`,
`PROCESSING_FIXED_OVERHEAD_SEC`.

**API.** The `/upload`, `/status` and `/transcriptions` contract is unchanged.
Five additive fields on completed rows: `anomaly_count`, `anomaly_windows`,
`flagged_segments`, `rescue_attempted`, `rescue_selected`, all null on rows
written before 0.6.0 (`flagged_segments` reads back as `[]` so a client need not
tell a pre-0.6.0 row from a clean one). The estimate formula is
`ceil(duration * 0.06) + 60`; `PROCESSING_SPEED_FACTOR` is retired.

**A requeue now clears its evidence.** `_count_failed_attempt` deletes
`<guid>_aligned` and nulls the diagnostic columns, because `run_forced_alignment`
skips MFA when its output already exists and a retry would otherwise refine its
new transcript against the previous attempt's word list.

**The harness measures what ships.** `RC060` and `RC060_RESCUE` import
`transcribe.py` for the level rule, the post-decode stage, the rescue trigger and
the selection rule, rather than reimplementing them, so a validation run cannot
pass against a second implementation that has drifted. That means a
release-candidate run needs `transcribe.py` and `textnorm.py` copied into
`/home/jay/harness` alongside the harness; every other config is self-contained.
`tools/quality_harness/prod_replica.py` deliberately keeps its own copy of
`clean_boundary_duplicates`, because the PROD control has to stay bug-for-bug
identical to 0.5.x for the 2026-09-07 baseline to remain reproducible.

**What to watch in the first week.** `words_per_second` should sit at 2.4-3.0
over speech. No job should be requeued by `anomaly_windows` on normal material.
`flagged_segments` should read as rhetorical repetition. `rescue_attempted`
should be rare; if it fires on most jobs the trigger is too tight, and if it
fires and `rescue_selected` is always false the rescue is not earning its
runtime. A flagged segment that is a real loop, or a normal recording requeued by
the window check, is a threshold to revisit before 0.6.1.

## 2026-09-07 - 0.6.0 harness validation on devmachine

The candidate was run through `tools/quality_harness` on the GPU host inside
`transcription-api:0.5.4` on all six reference files, as `RC060` (decode plus the
production post-stage) and `RC060_RESCUE` (the same plus the rescue), with the
alignment stage on `RC060` through the `i1` path and the `i2` rule. Three things
came out of it that were not visible from the Mac.

**The quiet VAD branch had to be a whole profile.** Covered in the 0.6.0 entry
above; the first run lost 3.0 percent of the retreat file's words and the fix was
measured, not guessed.

**The rescue can lose more than it fixes, and now cannot.** With the trigger
forced on, `tcf.20240319b` produced a primary pass with one anomalous segment out
of 323 and 3599 words, and a rescue with none and 3525. The anomaly score alone
selects the rescue and publishes 74 fewer words. A second forced run gave 2
against 0 and 3601 against 3518. `RESCUE_MIN_WORD_RETENTION` (0.99) now makes a
rescue ineligible if it drops more than 1 percent of the primary's words, and the
discard is logged with both word counts and both anomaly scores. The lesson
generalises beyond the rescue: any metric that counts defects prefers the
transcript with less content to be defective about, so every such rule in this
service needs a content floor beside it.

**On the loud files the alignment change is exactly neutral, and the decode is
what moved.** `agree250` for RC060 came out below the 2026-09-07 C1 row on three
files, which looks like an alignment regression and is not one. Aligning the
baseline's own C4 decodes through the same path gives agree250 identical to
RC060's on four files and within 0.0023 on the fifth, so RC060's alignment
reproduces the baseline's C4 alignment exactly and every difference from the C1
row is the C4 VAD, not the I1/I2 work. The C4 VAD helps alignment on two files
(755 s +0.089, class +0.067) and hurts on three (1463 s -0.057, 2783 s -0.006,
1369 s -0.001), net positive but not uniform. Comparing a candidate's alignment
against a baseline built on a different decode measures the decode; the
like-for-like row is the one to read.

Two smaller observations. The decode is not bit-reproducible on the longer files:
two runs of the identical configuration on `tcf.20240319b` gave 3599 and 3598
words (323 and 322 segments), while the 755 s file reproduced exactly, matching
the 2026-09-07 C1/C1_REPEAT determinism check. That is enough to move a file
across the rescue trigger, which is why the trigger had to be measured with the
threshold forced rather than by waiting for it to fire. And RC060's pre-dedupe
word count equals C4's exactly on all five loud files, which is the cleanest
evidence that the decode is unchanged from the measured configuration and only
the boundary de-duplication differs.


## 2026-09-07 - 0.6.0: the 100-file corpus sweep

The six reference files chose the design; the sweep is what tested it. Headline
results across the corpus: **median word delta +0.18 percent** against the legacy
transcripts over 99 files, with two regressions, both above 2.5 words per second
over speech and neither related to a trim. The legacy low-rate tail, the material
the whole rewrite was for, improved by 1.46 percent. Alignment applied on **100
percent** of files, every one on the first attempt, against a legacy path that
failed outright on one of six reference files. The rescue fired on 5 percent of
files and was kept on 4 percent. Five of five repeat submissions came back
byte-identical, including the file that varied between runs during reference
testing.

Four things the sweep found that six files could not.

**A quality gate must never be able to deliver nothing.** `tcf.20150424` failed
the anomaly gate three times, re-decoding *identically* each time, the same word
count and the same mean log-probability to fifteen digits. It burned 510 seconds
of GPU and ended quarantined with nothing published, against a legacy transcript
of 8292 words. The decode is deterministic, so a retry cannot clear a gate the
first attempt failed; the loop was pure cost with a guaranteed outcome, and the
outcome was worse than publishing. `process_pending_job` now fingerprints the
rejected result (word count plus mean log-probability) and, when a re-decode
reproduces it, publishes with the anomaly fields and flags set instead of
requeueing. Quarantine is reserved for results that are actually unusable, which
means the garbage check and the word-rate floor. The general lesson is worth more
than the fix: a signal that says "this looks wrong" must degrade to publishing
with a marker, never to silence, because a wrong transcript can be corrected and a
missing one cannot.

**Two checks, and neither substitutes for the other.** The coverage checks catch
holes: stretches of speech with nothing decoded over them. The per-window word-rate
check catches thin output: a passage the decoder covered but sparsely. Both of the
sweep's word-count regressions are the second shape, 73 and 91 words dropped from a
single passage, and their largest uncovered gaps are only 5.4 and 6.7 s because the
decoder emitted sparse segments across the omitted stretch rather than nothing at
all. No coverage threshold reaches that at any value, because output is present,
just wrong.

That made one detail load-bearing that had been quietly wrong: **the word-rate clock
has to run over the speech, not over the parts of it the decoder covered.** Clocking
on coverage skips the gaps between sparse segments, so a thin passage is compressed
into a few seconds of clock time and folded into its healthy neighbours, and the
rate never drops. Clocking on speech, it keeps its real duration and its few words.
Measured on the six reference files the minimum window rate is 2.37 words/sec
against a 1.2 floor, so the change costs no headroom.

The score is the **maximum** of the three signals, not their sum. They are three
views of the same missing content: a 46.8 s hole, the largest the sweep saw on an
otherwise healthy file, both empties one window and trips the gap check, and summing
those would quarantine it on a single event when it should be a review case. Taking
the maximum lets each check raise the score alone while one event stays one event.

**The uncovered-speech total is contaminated; the largest contiguous stretch is
not.** The sweep measured both over 101 files. The two speech detectors, ours and
the one inside the decode, disagree by 0.81 to 1.27 times, so the files with the
highest uncovered fraction are largely the ones where the detectors disagree about
what counts as speech, not files missing words. Any threshold on a total inherits
that sensitivity, which is why the tolerance kept having to move (0.10, then 0.15)
as the measurement improved rather than converging on a value. The largest
contiguous gap has no such problem: it is local, so a global offset between the
detectors shifts every boundary slightly without creating a long stretch out of
nothing. Its distribution separates cleanly, median 2.13 s, p95 17.1 s, max 46.8 s,
with a distinct tail. The gate now reads that, defaulting to 20 s, and the total
survives only as a loose backstop at 0.25 for an omission smeared across many
medium gaps. **20 s is a policy value, not a discovery**: the count of files
tripping the check falls smoothly with the threshold (11 at 10 s, 6 at 15, 4 at 20,
2 at 25, 1 at 30) with no knee anywhere in it. It is a choice about review volume,
and nothing in the data argues for one value over its neighbours, so move it on
capacity rather than on a search for the right number. Both numbers are logged per job. The six reference files agree with
the sweep: their uncovered totals reach 9.0 percent while no single stretch exceeds
3.92 s.

**The boundary de-duplication is off by default.** Across 64 trims on 32 files the
sweep found zero plausible decoder artifacts, roughly 38 clear false positives
including the four consecutive dropped segments on the 1 Kings 18:39 acclamation,
and 22 ambiguous sentence restarts. Thirty-one trims measured as sequential. Of
the ten that measured as overlapping, two overlap by 3.7 to 11.7 seconds against a
phrase lasting 1.4 seconds, which is physically impossible for a double decode and
marks them as measurement artifacts rather than evidence. So the expected benefit
is indistinguishable from zero and the demonstrated harm is deleted scripture.
`BOUNDARY_DEDUPE_ENABLED` defaults to off. The temporal implementation and its
tests are kept intact: this is disabled on evidence, not removed, and one variable
turns it back on if a real artifact is ever observed. The reference set says the
same thing in miniature: the single trim the word-count rule made there measures
as 0.86 s sequential, so even that apparent true positive was repetition.

**The profile selector is keyed on the wrong signal (0.6.1 investigation, do not
fix now).** Level does not merely predict fragmentation weakly, it predicts it
**non-monotonically**. Recordings quieter than -26 dBFS fragment at 12.1 segments
per minute, the -26 to -19 middle at about 9, and everything louder than -19 at
14.2. Both extremes fragment and the middle does not. A one-sided threshold cannot
express that shape at any value, so the question "where should the cutover sit" has
no answer: the selector is the wrong shape, not badly tuned. Consistent with that,
the eight most fragmented recordings all took the loud profile, the worst at 0.88
seconds per segment on one of the loudest files in the set, and the quiet profile's
worst case is better than the loud profile's.

What does track fragmentation is fidelity: 22.05 kHz recordings run 13.2 segments
per minute against 9.1 at 48 kHz, and bit rate moves with it. So the selector
should key on sample rate and bit rate first, both of which are free from the
container header and need no audio analysis at all. Any level term that survives
has to be two-sided, selecting the quiet profile at both ends rather than below a
line. Left alone for 0.6.0 so the sweep's numbers stay interpretable.


## 2026-09-07 - 0.6.0: the read-aloud omission, and why the fallback never fired

What 0.6.0 lost, and the sweep found, was read-aloud passages: scripture read from
the text, a quoted book, formal reading rather than preaching. Six cases across five
files, including both word-count regressions, all recovered when the same audio is
decoded in isolation. One coherent defect, not five scattered ones.

**It is not a misconfigured parameter.** Four different single-parameter changes each
recover a different subset of the passages and no two recover the same pair; between
them they cover all four. That is a deterministic decode landing in a bad path, which
is consistent with five of five repeat submissions coming back byte-identical. The
strongest single lever is starting the temperature ladder at 0.2, which recovers four
of four, and what that actually does is stop beam-searching: faster-whisper
beam-searches only at temperature 0.0 and samples at every rung above it. The
passages are lost by the beam search specifically.

**The real defect is that the fallback never fires.** The model drops these passages
*confidently*. Neither the compression-ratio nor the log-probability threshold is
crossed, so no rung above the first is ever attempted and five rungs of the ladder go
unused. Every per-segment guard the decode has looks at how sure the model was, and
the model was sure. A silent, confident omission is invisible to all of them.

That leaves exactly one signal that can see it: a stretch of speech carrying almost
no words. Hence three changes.

`ANOMALY_WINDOW_MIN_WPS` goes from 1.2 to 1.5. Measured per 60 s window over the
omitted stretches the control scores 0.42, 0.38, 0.82 and 1.20 words/sec, so the old
floor caught three of four and missed the fourth by exactly nothing. Against a corpus
median of 2.65 over speech and a healthy minimum of 2.37 across the six reference
files, 1.5 keeps 0.87 of headroom. Re-checked after the change: no reference file
trips it and no false rescue fires.

**The signal is wired to the remedy, not only to the punishment.** A single low-rate
window fires the rescue pass, which decodes with previous-text conditioning off and
its ladder starting at 0.2. Those are precisely the two most effective variants
combined, which is a coincidence worth naming: the rescue was designed for repetition
loops and happens to be the right treatment for confident omission as well, because
both are failures of the primary decode's path rather than of its parameters.

**The retry stop fingerprints the primary pass, not the published one.** The primary
decodes at temperature 0.0, where faster-whisper beam-searches, so it reproduces
itself. The rescue decodes from 0.2, where faster-whisper samples, and no seed is set
anywhere, so a rescue-published transcript differs on every attempt. Fingerprinting
the selected pass therefore never matched on exactly the files the rescue exists for,
and they burned three decode pairs before quarantining with nothing: the black hole
returning through the door built to close it. And because the gate reads the
per-segment count while selection ranks on count plus windows, a rescue that traded
segment flags for windows could win selection and then trip the cap, quarantining a
file the primary would have cleared. So the quarantine decision now needs BOTH the
published pass and the primary over the cap, and on the final attempt it publishes
with the flags set regardless. The anomaly gate can requeue; it can no longer end in
silence.

**A low-rate window never requeues, however many there are.** The decode is
deterministic, so a requeue re-runs the identical decode and burns three attempts
into the same quarantine. That is exactly what happened to tcf.20150424: 510 s of GPU
to publish nothing against a legacy transcript of 8292 words. Only the per-segment
anomaly count can requeue now. The window count fires the rescue, the better pass is
selected on the existing rules and published, and the count is stored on the row with
a log line so the job can be found and reviewed.

The selection works in the right direction here without any special case. When the
rescue recovers a passage its low-window count falls, so it wins on anomaly total;
and because it *gains* words rather than losing them, neither the retention floor nor
the 40 word cap can block it. Those guards exist to stop a rescue that clears a flag
by deleting content, which is the opposite case.

**Deferred to 0.6.1, recorded not done: whether the primary pass should sample rather
than beam-search.** Starting the ladder at 0.2 recovers all four passages, which is
the best result of any single change. It is not in 0.6.0 because it changes the core
decode for every file on the evidence of seven passages rather than a corpus, and
because beam search is the reference Whisper configuration and the thing the 0.5.x
five-pass design was measured against. It deserves its own release with its own
sweep, not a late amendment to this one.

Two of the six cases are not expected to recover and should not. The two
question-and-answer stretches in tcf.20241105 are an audience member off microphone:
the audio is not there to transcribe, and the surrounding conversation holds those
windows at 2.1 to 2.5 words/sec so they do not trip the floor either. One correction
to the earlier record: the C.S. Lewis quotation in tcf.20210604 was never lost. It
scores 0.938 under the current configuration and its coverage gap was a timing
artifact. Confirmed omissions are three, plus the two partial question-and-answer
cases.


## 2026-09-07 - 0.6.0: retiring the per-segment anomaly count

Measured across 102 recordings and 204 decodes including repeats, `anomaly_count` was
retired as a decision input. It never exceeded 1 on any file on either build. It
identified **zero of seven** independently confirmed bad transcripts, while all six
files that scored a flag were healthy and within 0.2 percent of their previous word
counts, which makes it mildly *anti*-correlated with quality. Every one of those six
carried a temperature flag, so the only thing that creates an anomalous segment on
this corpus is that the ladder engaged: a fact about the decode path, not about the
output. The window check caught six of the seven bad files, the two signals were
perfectly disjoint (every file with a flagged segment had zero low-rate windows and
the reverse), and all twelve rescues in the run came from the window trigger.

So `ANOMALY_SEGMENTS_MAX` and `RESCUE_ANOMALY_SEGMENTS` are deleted rather than left
configured and inert, along with the retry fingerprint machinery whose only consumer
was the gate they fed. `anomaly_count`, `anomaly_windows` and `flagged_segments` stay
as published diagnostics and in the log line: they cost nothing and are useful for
review, they simply do not decide anything. **No anomaly signal can quarantine a job
any more.** Quarantine belongs to the garbage check and the word-rate floor, which
reject output that is unusable rather than merely suspect.

**Why a threshold on that signal was never going to be stable.** A file varied by one
word between two runs with no rescue on either, and that single word flipped its
anomaly count. It carries a temperature flag, so the ladder engaged inside the primary
pass, and that is the whole mechanism: the primary is reproducible only while the
ladder stays on its first rung. Once it engages, the higher rungs sample unseeded and
the primary itself varies. A signal that only fires when sampling occurred is
therefore unstable exactly where it is non-zero, which is exactly where a threshold on
it would be read. The signal is not noisy despite firing rarely; it is noisy *because*
of what makes it fire.

**Not done, deliberately: converting the cap to a rate.** One flag in a 34-segment
recording is 2.94 per hundred while the same single flag in a 3646-segment recording
is 0.027, so normalising would make short files look catastrophic. The
scale-dependence is real in principle and invisible in this corpus, and the cure is
worse than the disease.

**Known gap, undetected, for 0.6.1: `tcf.20150424`.** It publishes about 15 percent
short of its legacy transcript with zero flagged segments, zero low-rate windows and
no uncovered gap over 20 seconds. It is the one bad file that neither signal sees.
Publishing it is still better than the quarantine it used to receive, but the loss is
real and nothing currently flags it for review. Any 0.6.1 work on detection should
start here, because it is the counterexample that says the present signal set is
incomplete rather than merely imperfect.

**Harness anomaly numbers are a separate series from the service's.** Reference runs
scored this corpus's oscillating file at 0, 1, 2 and once 5 anomalous segments, while
the corpus run scored it 1 twice and could not reproduce a 5 across 102 files. The 5
came from the harness path. The two are not configuration drift: the harness decode
arguments and the service's were compared field by field and are identical, and a unit
test pins them equal. But they are two copies of the same settings rather than one, so
they are equal by test rather than by construction, and the harness does not exercise
`transcribe_audio` itself. Treat harness anomaly counts as their own measurement
series until that gap is closed, and note that the reference runs shared GPU 0 with
the sweep container, which is the likeliest source of the extra spread.
