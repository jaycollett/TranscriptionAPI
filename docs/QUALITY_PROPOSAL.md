# Transcription quality: findings and the 0.6.0 proposal

Date: 2026-09-07. Evidence: the two code analyses of `main` at 0.5.1 (decode
and best-pass selection, flaws F1-F10 and experiments C1-C10; post-Whisper
alignment, flaws F1-F13 and experiments I1-I7), the faster-whisper 1.2.1 and
MFA 3.4 sources, and the harness run recorded under
`tools/quality_harness/results/2026-09-07/` (`summary.md` for the tables,
`results.json` for every number below). Item numbers in brackets refer to the
Pipeline section of `docs/PUNCHLIST.md`.

## 1. Purpose and method

The question was whether the five-pass decode, the pass selection, the VAD
settings, the boundary de-duplication and the whole-file MFA alignment do
what they were written to do, and what a release that fixes them should
contain. Each analysis read the service code against the library sources it
calls, listed the defects, and defined experiments as deltas from one base
configuration. The harness (`tools/quality_harness/`) ran them inside the
production image (`transcription-api:0.5.2`) on the GPU host, with a
verbatim replica of the production decode and selection rule (`PROD`) as the
control.

**Reference set.** Six files from the production archive, chosen to cover
length, format and recording conditions: the 755 s validation file
(`tcf.20240213b`), a 1463 s and a 2783 s sermon (the latter stereo), the
1369 s file on which production MFA had failed (`tcf.20240319b`), a 1454 s
multi-voice class recording at 22.05 kHz and 49 kbps, and a 3388 s retreat
session whose mean level is -28.6 dB against -18 to -24.5 dB for everything
else (the only quiet file in the 655-file archive).

**No ground truth exists.** Every transcript in the archive was produced by
this service and none has been human-corrected, so no word error rate was
computed. The harness measures relative and structural quantities instead:

- Word count and words per second over file and over VAD speech time.
- Cross-config agreement: `difflib.SequenceMatcher` ratio on normalised
  words, with the insert, delete and replace runs localised by time.
- Anomaly counts from the per-segment fields faster-whisper returns:
  segments at temperature 0.5 or above, compression ratio over 2.4, average
  log-probability under -1.0, fallback windows, 60 s speech windows under
  1.2 words per second, and a stoplist of phantom phrases.
- Repeated 3- and 4-gram rates, the C7 flagger, VAD seam count and removed
  seconds, determinism between two runs, wall time and GPU peak.
- For alignment: `agree250`, the fraction of segments whose refined start
  and end are both within 250 ms of Whisper's; median and p95 start delta;
  non-monotonic and overlapping segment counts; empty-window fallbacks; the
  share of segments owning under 0.8 or over 1.2 of their words; MFA attempt
  used, wall time and utterances aligned.

Agreement with C1 is a consistency measure, not accuracy. Where two configs
disagree the localised runs were read by hand; the Psalm 91 case below is one
of those.

## 2. Findings, ranked by measured effect

### 2.1 C1: one beam-5 pass replaces five passes [1, 6]

Passes 1, 4 and 5 sample at temperature 0.2-0.3 with `beam_size=1`, so only
pass 2 beam-searches and the selector ranks four noisy variants of one decode
(F3). A single deterministic beam-5 pass with the full temperature ladder
reproduces the production winner at a fraction of the cost.

| file | dur s | PROD words | C1 words | agreement | PROD wall s | C1 wall s |
|---|---|---|---|---|---|---|
| tcf.20240213b | 755 | 2076 | 2104 | 0.994 | 103.2 | 20.4 |
| tcf.20240319b | 1369 | 3574 | 3578 | 0.992 | 175.2 | 40.6 |
| tcf.20240416 class | 1454 | 3881 | 4010 | 0.987 | 276.8 | 35.9 |
| tcf.20240213a | 1463 | 4021 | 4051 | 0.994 | 206.5 | 44.8 |
| tcf.20240116 | 2783 | 7803 | 7821 | 1.000 | 360.0 | 70.4 |
| women_retreat s3 | 3388 | 8461 | 8904 | 0.971 | 469.7 | 78.8 |

C1 run twice on the 755 s file gave identical text and identical segments
(ratio 1.0). Where PROD and C1 differ most (the class file and the retreat
file) the difference is words PROD's chosen pass dropped, see 2.2. The 103 s
PROD replica time matches the 106 s measured for 0.5.2 in production.

### 2.2 C2: anomaly score instead of duration-weighted word probability [8]

The production metric weights each word's teacher-forced probability by its
duration (F1). Across the six files it disagreed with the anomaly rule on
four, and in every disagreement it preferred the pass with fewer words.

| file | PROD pick (words, conf) | C2 pick (words, mean logprob) | C1 words |
|---|---|---|---|
| tcf.20240116 | pass 2 (7820, 0.980) | pass 2, same | 7821 |
| tcf.20240213b | pass 5 (2092, 0.981) | pass 5, same | 2104 |
| tcf.20240213a | pass 2 (4023, 0.980) | pass 5 (4007, -0.082) | 4051 |
| tcf.20240319b | pass 3 (3576, 0.979) | pass 2 (3656, -0.069) | 3578 |
| tcf.20240416 class | pass 5 (3908, 0.979) | pass 4 (4007, -0.064) | 4010 |
| women_retreat s3 | pass 3 (8467, 0.976) | pass 2 (8919, -0.053) | 8904 |

On the retreat file the confidence ranks the passes in reverse order of word
count: pass 2 with 8919 words scores lowest at 0.942 and pass 3 with 8467
wins. The C1-to-PROD diff shows what pass 3 lost: 290 deleted words,
including runs at 2085.8 s (61 words) and 2108.9 s (58 words) that are the
speaker reading Psalm 91:9-11 and 14-15 ("because you have made the Lord your
dwelling place ... I will rescue him and honor him"). C1 has both. No pass on
any file tripped an anomaly, so C2 decided on mean log-probability, which
tracked word count every time.

### 2.3 I1: per-utterance MFA instead of one utterance per file [14]

Production writes one `.txt` beside the WAV, so a 20-60 minute file is one
alignment graph and a single mismatch can fail or distort the whole file (F1).
I1 writes a TextGrid with utterances of 8-30 s built from Whisper segments,
split only at gaps of 0.4 s or more and padded 0.15 s into the gap, and runs
MFA at its default beams. Rows compare the production path (a0, beam 40/100
then 100/400, 300 s timeout each) with I1, both from the C1 decode and both
scored with the I2 rule.

| file | a0 attempt | a0 MFA s | a0 WAV MB | I1 MFA s | I1 WAV MB | utts | agree250 a0 / I1 |
|---|---|---|---|---|---|---|---|
| tcf.20240213b | 1 | 36.4 | 72 | 24.7 | 24 | 43/43 | 0.490 / 0.516 |
| tcf.20240319b | failed | 278.8 | 131 | 30.0 | 44 | 79/79 | none / 0.564 |
| tcf.20240416 class | 1 | 37.1 | 64 | 28.3 | 47 | 23/23 | 0.679 / 0.711 |
| tcf.20240213a | 2 | 194.8 | 140 | 26.9 | 47 | 71/71 | 0.577 / 0.636 |
| tcf.20240116 | 1 | 54.1 | 534 | 32.9 | 89 | 46/46 | 0.796 / 0.777 |
| women_retreat s3 | 1 | 57.6 | 325 | 35.4 | 108 | 52/52 | 0.770 / 0.786 |

The 1369 s file reproduces the production failure: attempt 1 returned code 1
after 78.7 s and attempt 2 after 200.1 s, 278.8 s for no alignment (the PROD
decode fails the same way in 288.7 s); I1 aligned it on the first attempt in
30 s. The 1463 s file needed the 100/400 attempt under a0 and 26.9 s under I1.
On the class file the PROD decode under a0 shows the predicted drift: p95
start delta 17.5 s and 10.8 percent of matched words over 1 s from Whisper's
position, against 0.65 s and 0.4 percent under I1. Failure is now bounded by
the utterance: the 1369 s file keeps Whisper timings for 43 segments (394 of
3578 words) where a0 lost the whole file.

### 2.4 I2: sequence-matched word assignment [4]

The window rule assigns MFA words by start time inside Whisper's segment
bounds (F4), which double-assigns boundary words and starts segments on their
second word. I2 matches MFA tokens to Whisper tokens with
`difflib.SequenceMatcher`, takes segment edges from owned words, and enforces
monotonic output. All rows are C1 decode, I1 path.

| file | agree250 window / I2 | start delta p50 | non-monotonic | own < 0.8 |
|---|---|---|---|---|
| tcf.20240213b | 0.458 / 0.516 | 0.27 / 0.23 | 0 / 0 | 0.072 / 0 |
| tcf.20240319b | 0.509 / 0.564 | 0.23 / 0.16 | 0 / 0 | 0.212 / 0.132 |
| tcf.20240416 class | 0.454 / 0.711 | 0.185 / 0.09 | 0 / 0 | 0.005 / 0 |
| tcf.20240213a | 0.519 / 0.636 | 0.23 / 0.15 | 0 / 0 | 0.065 / 0.026 |
| tcf.20240116 | 0.570 / 0.777 | 0.12 / 0.07 | 1 / 0 | 0.024 / 0 |
| women_retreat s3 | 0.653 / 0.786 | 0.13 / 0.08 | 0 / 0 | 0.002 / 0 |

I2 never produced a non-monotonic or overlapping segment on any run; the
window rule produced one on the 2783 s file and two on the 1463 s file under
a0.

### 2.5 C4: conservative VAD, with a quiet-file failure [10]

Production VAD (threshold 0.35, min silence 300 ms, pad 400 ms) removes 1-5
percent of audio and cuts a seam at every pause over 0.8 s (F5). C4 uses
threshold 0.5, min silence 1000 ms, pad 300 ms and
`hallucination_silence_threshold` 0.5 (a value that can actually fire under
VAD; 2.0 cannot, F6).

| file | C1 seams | C4 seams | C1 removed s | C4 removed s | C4 words vs C1 | agreement | C4 wall s |
|---|---|---|---|---|---|---|---|
| tcf.20240213b | 107 | 27 | 13.9 | 20.2 | 2107 / 2104 | 0.994 | 20.5 |
| tcf.20240319b | 165 | 43 | 85.8 | 109.9 | 3601 / 3578 | 0.978 | 43.0 |
| tcf.20240416 class | 300 | 84 | 51.6 | 72.4 | 4008 / 4010 | 0.997 | 34.5 |
| tcf.20240213a | 172 | 36 | 15.7 | 24.5 | 4023 / 4051 | 0.992 | 40.3 |
| tcf.20240116 | 408 | 121 | 96.9 | 123.5 | 7822 / 7821 | 0.998 | 68.3 |
| women_retreat s3 | 564 | 155 | 156.8 | 192.6 | 8614 / 8904 | 0.979 | 102.6 |

Seams drop 3-4x and removed audio stays under 6 percent. On the 755 s file
C4 also aligns best of any decode (agree250 0.605 under I1 and I2 against
0.516 for C1). The retreat file is the failure: at -28.6 dB the 0.5 threshold
fragments speech into 2171 segments of 1.4 s mean (C1: 435 of 7.7 s), mean
log-probability falls from -0.056 to -0.141, 290 words are lost and wall time
rises 30 percent. The threshold has to depend on the file's level (section 3).

### 2.6 C7: repeated n-gram flagger

Segments whose repeated 4-gram rate exceeds 0.3 were flagged on the C1
decode: 2, 0, 1, 0, 2 and 1 per file, six in total. Every one was read and
every one is genuine rhetorical repetition ("Eternal life is knowing God.
Eternal life is knowing God.", "willing to open your home, willing to open
your home"). No decoder loop occurred on any file. The flagger is therefore
useful as a marker for downstream review, not as a rejection rule.

### 2.7 F7: boundary de-duplication desynchronises text from timings [5, 18]

`clean_boundary_duplicates` edits the transcript string but not the timings.
On every PROD run `" ".join(timings[].text) == transcription` was false and on
every other config it was true; the regex removed 16, 2, 27, 2, 17 and 6 words
per file. Tested directly: "He is risen. He is risen indeed." becomes
"He is risen.  indeed."; "pray without ceasing. Pray without ceasing." becomes
"pray without ceasing. ."; "day by day by day" becomes "day by  day". The
second regex in `run_forced_alignment` (collapse of four identical words) then
makes MFA's word sequence a third variant.

### 2.8 Negative results

| experiment | result | verdict |
|---|---|---|
| C9 hotwords glossary | 755 s: 2251 words (+147 on C1), agreement 0.913, repeated 4-gram rate 0.140 vs 0.031, 5 fallback windows. Class file: agreement 0.834, max compression 11.4, 12 fallbacks, one low window. Glossary hits 28 vs 25, 2 vs 2, 77 vs 79. No prompt leakage. | Rejected: hotwords prime repetition on every window and gain nothing on spelling. |
| C8 batched pipeline (batch 8) | 11.2-48.5 s wall (2-3x faster than C1) but 1-4 percent fewer words on every file (2072 vs 2104, 8569 vs 8904), agreement 0.977-0.994, timing overlaps on three files (7 on the retreat file), segments up to 57 s, agree250 0.400 vs 0.516 after I1 and I2. | Deferred: speed path or second opinion only; not a primary decode. |
| C5 per-clip arbitration | 2050, 3426 and 3724 words on the three files run (3-7 percent under C1), agreement 0.961-0.984, 73-122 s wall; the rule chose the batched candidate in 58 of 58, 55 of 56 and 30 of 30 clips because its per-chunk log-probability is structurally higher. | Rejected as designed: the rule is biased toward the batched pass and the result loses words. |
| C3 no VAD, hallucination threshold 2.0 | Agreement 0.976-0.993, wall equal to C1. On the 1369 s file +124 words and no fallback windows (C1 had one at min log-probability -1.25); on the class file 6 phantom segments (C1: 1) and 442 segments of 2.9 s; agree250 0.445 vs 0.516 on the 755 s file. | Deferred: the 1369 s gain deserves a follow-up, the timing and phantom costs rule it out now. |
| C10 coarser ladder | Identical text to C1 on all three files (ratio 1.0); one extra temperature 0.5 segment on the 1369 s file. | Rejected: no effect. |
| C6 greedy agreement gate | Fast path taken on the 755 s file (ratio 0.9957) and refused on the 1369 s (0.9923, an anomaly present) and class (0.977) files. | Deferred: only worth having with a slow path (C5) that is not adopted. |

## 3. Proposed release 0.6.0

Concrete changes against `transcribe.py` and `app.py` as they stand on
`main`.

**Decode (`transcribe.py`, `transcribe_audio`).** Replace the five-entry
`passes` list and the loop over it with one call: `beam_size=5`, `best_of=5`,
`patience=1.0`, `temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`,
`compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`,
`no_speech_threshold=0.6`, `condition_on_previous_text=True`,
`word_timestamps=True`, `language="en"`, no prompt, no hotwords. Construct the
model with `num_workers=1` [19]. Keep the per-segment `Segment` fields
(`temperature`, `compression_ratio`, `avg_logprob`, `no_speech_prob`) on the
returned segments; the harness's C1 is exactly this configuration. A second
pass runs only if the first fails the gate below, with
`condition_on_previous_text=False` and the ladder starting at 0.4, and the
pass with the lower anomaly score is kept.

**Selection and gate (`transcribe.py` and `app.py`, `process_pending_job`).**
Delete `calculate_weighted_confidence` and the 1.4 words-per-second scaler.
The anomaly score counts segments with temperature at or above 0.5,
compression ratio over 2.4, average log-probability under -1.0, or
`no_speech_prob` over 0.5 with non-empty text, plus 60 s windows of VAD
speech under 1.2 words per second; ties break on mean log-probability.
`transcribe_audio` returns `anomaly_score`, `low_windows` and
`speech_seconds`. `process_pending_job` keeps the `MIN_WORDS_PER_SEC` floor,
computed over `speech_seconds` when available, and also requeues when
`low_windows` is non-zero, so a partial collapse is caught where the
whole-file rate is not [22]. Both numbers go to the success log line.

**VAD (`transcribe.py`).** `vad_parameters` becomes
`{"threshold": T, "min_speech_duration_ms": 250, "min_silence_duration_ms": 1000, "speech_pad_ms": 300}`
with `hallucination_silence_threshold=0.5`. `T` is level-aware: measure the
file's mean level once with `ffmpeg -af volumedetect` (or the equivalent RMS
in dBFS from the decoded samples), use `T=0.5` when the mean is at or above
-26 dBFS and `T=0.35` below it. The -26 dBFS cut sits between the loudest
file that worked at 0.5 (-24.5 dBFS) and the one that failed (-28.6 dBFS);
it is a starting point, and the 0.35 branch has not been measured with the
1000 ms and 300 ms values. It must be tuned on `women_retreat_2025_session3`
before release, with the acceptance test that its word count and agreement
against C1 stay within the section 4 thresholds.

**Text and timings (`transcribe.py`).** Remove `clean_boundary_duplicates`
and the `re.sub` in `run_forced_alignment`. De-duplicate per segment: where
the last k words (k of 2 or more) of segment i equal, case-insensitive and
punctuation-stripped, the first k words of segment i+1, trim them from
segment i+1; collapse four or more identical consecutive words inside one
segment. Then `transcription = " ".join(seg.text)` and the MFA text is built
from the same segments, so the three word sequences are one. Each timing
entry carries Whisper's `words` list internally for I2's fallback (not
exposed in the API in this release).

**Alignment (`app.py`, `run_forced_alignment`).** Export the WAV with
`ffmpeg -y -nostdin -i <in> -vn -ac 1 -ar 16000 -sample_fmt s16` instead of
pydub at source rate (I5; 534 MB becomes 89 MB on the 2783 s file, and MFA
hears the same channel mix Whisper decoded). Write `<guid>.TextGrid` with one
`speaker` tier of utterances built as in the harness's `build_utterances`
(gap 0.4 s, pad 0.15 s, 8-30 s). Run
`mfa align <dir> <dict> english_mfa <out> --output_format json --include_original_text --no_tokenization --clean --overwrite`
at default beams, one attempt, timeout `60 + 0.5 * duration_sec` (I6; the
100/400 attempt is dropped). Replace the window loop with the harness's
`refine_i2`: normalise both sides, `SequenceMatcher` with `autojunk=False`,
segment edges from owned words, Whisper word timestamps for words MFA did not
return, then `start_i = max(start_i, end_{i-1})`. Utterances MFA drops keep
Whisper timings and are counted in the log. Delete `<guid>_aligned` in
`_count_failed_attempt` so a requeue cannot reuse stale words [17].

**Additive API field.** `flagged_segments`: a list of `{start, end, reason}`
for segments whose repeated 4-gram rate exceeds 0.3, stored as a column and
returned by `/status` and `/transcriptions`. Empty on most jobs. Nothing is
removed on its account.

**Unchanged.** The `/upload`, `/status` and `/transcriptions` contract and
status vocabulary; `large-v3-turbo` float16; the base image and every pin in
`requirements.txt`; the queue, retry and cleanup behaviour of 0.5.0; MFA
dictionary and acoustic model.

**Expected effects, from the measured rows.** Whisper 20 s instead of
103-106 s on the 755 s file and 70 s instead of 360 s on the 2783 s file. MFA
24-35 s on every reference file instead of 36-279 s, first attempt every time
(a0 failed one file of six and needed the second attempt on another). End to
end about 50 s on the 755 s file against 138 s in the 0.5.2 validation. Word
counts at C1's level: 28 more words on the 755 s file, 129 on the class file
and 443 on the retreat file than production publishes today. Timing agreement
within 250 ms rises from 0.44-0.65 to 0.52-0.79 with zero overlaps.

## 4. Validation plan for 0.6.0

Beyond the standard release-candidate recipe in `docs/SESSION_KNOWLEDGE.md`:

1. Add the 0.6.0 configuration to `tools/quality_harness/configs.py` as `RC`
   (decode, VAD rule and per-segment dedupe) and an `rc` alignment path, and
   run `run` and `align` on all six reference files with the same image the
   candidate was built from. Compare against `results/2026-09-07/`.
2. Acceptance thresholds, all six files unless stated:
   - Agreement with the 2026-09-07 C1 transcript at or above 0.995 on
     `tcf.20240213b`, and at or above the C4 row's agreement on every other
     file.
   - Word count within 1 percent of C1's per file.
   - `agree250` under the I2 rule at or above the C1 I1 row per file.
   - Zero MFA failures, zero timeouts, every file on the first attempt.
   - Wall time under 60 s end to end for the 755 s file.
   - `" ".join(timings[].text) == transcription`, zero non-monotonic and zero
     overlapping timings on every file.
   - The retreat file's word count within 1 percent of C1 with the level-aware
     threshold taking the 0.35 branch, and the other five files taking 0.5.
3. Run the candidate twice on the 755 s file; text and segments must be
   identical.
4. First week in production: for every completed job, log and review
   `words_per_second` and `flagged_segments`. Expect 2.4-3.0 words per second
   over speech, no job requeued by `low_windows` on normal material, and
   flagged segments that read as rhetorical repetition. A job with a flagged
   segment that is a real loop, or a normal recording requeued by the window
   check, is a threshold to revisit before 0.6.1.

## 5. Deferred and rejected

- C8 batched pipeline: deferred; 2-3x faster than C1 but loses 1-4 percent of
  words and produces overlapping timings.
- C5 per-clip arbitration and C6 greedy gate: rejected in the measured form;
  the clip rule prefers the batched candidate structurally and the gate has no
  slow path worth taking.
- C3 no VAD: deferred; better on the hardest file, worse timing and more
  phantoms on the class file. Revisit with a phantom filter.
- C9 hotwords: rejected; primes repetition, no spelling gain [20].
- C10 coarser ladder: rejected; no effect.
- I7 G2P for OOV names and digit expansion [21]: deferred, not run; needs the
  G2P model in the image first.
- I8 `--single_speaker`: deferred, not run; disables speaker adaptation.
- `words` per timing entry in the API [15]: deferred to a later additive
  release; the data is kept internally for I2.
- Duration estimate recalibration [16]: deferred to 0.6.1 with the measured
  real-time factors (Whisper about 0.03, MFA about 0.02).
- Whole-transcript alphanumeric gate [22]: deferred; the window check covers
  the case that mattered.
- `hallucination_silence_threshold=2.0` under VAD [9]: rejected; it cannot
  fire while VAD pads gaps to 0.8 s.
- Punchlist item 3 (silence intervals in the MFA words tier): withdrawn; MFA
  3.4 strips them by default and the harness saw `words_mfa` equal to the
  submitted count within tokenisation on the clean files.

## 6. Rerunning the harness

The harness runs inside the production image on the GPU host with the audio
directory and a working directory bind-mounted. Copy `tools/quality_harness/`
to `/home/jay/harness` on devmachine, then:

```
docker run --rm --name transcription-harness --gpus device=0 \
  -v /home/jay/harness:/harness \
  -v /home/jay/SourceCode/SermonPreprocessorAPI/data/audiofiles:/audio:ro \
  -e MFA_ROOT_DIR=/harness/mfa_tmp \
  transcription-api:0.5.2 \
  python /harness/harness.py run --audio /audio/tcf.20240213b.mp3 --configs PROD,C1,C4
```

Stages: `run --audio <a,b> --configs <names>` decodes each config on each
file into `/harness/results/<stem>/<config>/` (transcript, segments with
diagnostics, metrics); `align --audio <a> --configs <names> --paths a0,i1`
runs the production alignment path (`a0`) and the per-utterance path (`i1`)
on those decodes and scores both refinement rules (`window`, `i2`); `report`
regenerates `summary.md` and `results.json` and needs no GPU. `--force`
re-runs a cached pair. The 2026-09-07 run: the 755 s file with
`PROD,C1,C1_REPEAT,GREEDY,C3,C4,C8,C10,C5` then `C9`; the class and 1369 s
files with `PROD,C1,GREEDY,C3,C4,C8,C10,C9,C5`; the 1463 s, 2783 s and
retreat files with `PROD,C1,C4,C8`; alignment `C1,PROD` with `a0,i1` on the
first three files, `C3,C4,C8` with `i1` on the 755 s file, `C1` with `a0,i1`
on the last three. Use `--gpus device=0` (the 12 GB card); the 8 GB card is
held by `ollama-vision.service`.
