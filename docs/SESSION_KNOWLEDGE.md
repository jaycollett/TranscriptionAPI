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
