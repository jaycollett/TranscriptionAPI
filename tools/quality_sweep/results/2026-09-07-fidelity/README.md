# The fidelity experiment, 2026-09-07

0.6.0 shipped with three questions open, and this run answers them with measurement
rather than argument. Nothing here changed production: the service on port 5030 was never
stopped, restarted, exec'd into or sent traffic, and the only production-owned path in
any command was the audio archive, mounted read-only.

## The three open items

1. **`tcf.20250607` still loses Exodus 24.** The rescue recovers the reading and loses
   200-340 words elsewhere, so the retention guard refuses it. The deferred question is
   whether the primary pass should sample rather than beam-search.
2. **`tcf.20150424` publishes about 15 percent short of legacy and nothing sees it.**
   Zero flagged segments, zero low-rate windows, no gap over 20 s. The signal set is
   incomplete, not mistuned.
3. **The profile selector is keyed on level, which does not predict fragmentation.**
   Sample rate and bit rate do; level is non-monotonic.

## The measurement that had been missing

Every comparison before this one was relative, because no human-corrected transcript of
any of these recordings exists. This corpus does contain one source of real ground truth:
**passages read aloud from scripture.** `scripture.py` builds a benchmark of 17 such
passages across 13 recordings, four found by hand in the rc2 follow-up and 13 found by
scanning transcripts for a citation and testing whether the passage it cites is then
read, and scores each configuration by the word error rate of a public-domain reference
against the best-matching stretch of the transcript.

**The absolute rates are not accuracy and must never be quoted as such.** The preacher's
translation is unknown and is not the World English Bible this scores against. The
measured size of that mismatch is in `scripture.md`: a transcript that demonstrably
contains a reading still scores around 0.42. The reference is the same for every
configuration, so the differences between configurations on the same passage carry
information even though the levels do not.

The World English Bible was chosen over the ASV, KJV, AKJV, Webster and Young's by
scoring all six against the **legacy** transcripts, which are none of the four
configurations under test, so the choice cannot favour one arm over another. WEB scored a
mean 0.416 against 0.441 for the next best.

## The four configurations

| config | change | subset | why that subset |
|---|---|---|---|
| A | 0.6.0 as shipped, no overlay | all 32 | the control |
| A2 | 0.6.0 as shipped, decoded a second time | all 32 | the noise floor |
| B | primary temperature ladder from 0.2 | all 32 | changes the decode on every file |
| C | rescue ladder from 0.2 with previous-text conditioning left **on** | 8 | C's primary is A's primary bit for bit, so C is A wherever no rescue fires |
| D | VAD profile keyed on sample rate and bit rate, not level | 20 | D is A wherever the two selectors agree |

`fidelity_pass.py` applies each configuration as an overlay on the image's own
`transcribe` module rather than editing it, so config A is the shipped code exactly and
every difference is attributable to one named lever. The overlay was verified before the
run: A reproduced the shipped pass's word count on the smoke file exactly, B moved the
ladder, and D selected the quiet profile on a file the level rule calls loud.

## The 32 recordings

Chosen by `select_fidelity_set.py` so that every file can discriminate between at least
two configurations: the 4 known read-aloud passages, all 7 files with independent
evidence of a bad transcript, the 13 files carrying a discovered scripture passage, the
6 most fragmented recordings, files on both sides of the selector disagreement, and a
healthy control group spread over duration and level.

## Files here

| file | what it is |
|---|---|
| `free_analysis.md` | the no-GPU test of whether two decodes already on disk detect a bad transcript, and why the answer is negative |
| `agreement_heldout.md`, `.json` | its per-file numbers |
| `noise_floor.md`, `replicate.json` | 0.6.0 against itself over the same 32 files |
| `scripture.md`, `.json` | the scripture benchmark, every configuration |
| `fidelity.md`, `.json` | the four configurations joined, one open item at a time |
| `fidelity_log.txt` | the run log |

## Rerunning

On the GPU host, with `tools/quality_sweep/` copied to `/home/jay/sweep/tools`:

```
./run_fidelity.sh          # the four configuration passes, detached, one log
./reports.sh               # every report; no GPU
```

`fidelity_pass.py` is resumable: a finished file is skipped, so an interrupted run
continues where it stopped. The scripture reference cache is committed, so
`scripture.py --offline score` needs no network.
