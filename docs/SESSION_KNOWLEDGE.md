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

Correcting the arithmetic also moved the tolerance. The healthy worst case was
measured at 7.1 percent using the inflated coverage; intersected, the healthy
range across the six files is 1.1 to 9.0 percent, worst on `tcf.20240319b`. A 10
percent tolerance would have shipped with 12 s of headroom on a 1259 s file, which
is not a tolerance, so the default is 15 percent. All six measured coverages are
pinned as tests. Six files is still not a distribution and the sweep should set
this from the corpus.

**A rescue pass, not five of them.** The redundancy the five-pass design was
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
(`ANOMALY_WINDOWS_MAX` 2, `ANOMALY_SEGMENTS_MAX` 5) see the selected result, so a
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
`RESCUE_ENABLED`, `RESCUE_ANOMALY_SEGMENTS`, `RESCUE_ANOMALY_WINDOWS`,
`RESCUE_TEMPERATURE_BASE`, `RESCUE_MIN_WORD_RETENTION`, `RESCUE_MAX_WORD_LOSS`.
Anomaly windows: `ANOMALY_UNCOVERED_TOLERANCE`. De-duplication:
`BOUNDARY_DEDUPE_ENABLED`, `BOUNDARY_DEDUPE_MIN_WORDS`,
`BOUNDARY_DEDUPE_OVERLAP_SLACK_SEC`.
Gate: `ANOMALY_WINDOWS_MAX`, `ANOMALY_SEGMENTS_MAX`. Alignment:
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
