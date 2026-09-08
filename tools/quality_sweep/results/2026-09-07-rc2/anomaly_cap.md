# The anomaly segment cap, read from the 100-file sweep

## Is the recorded number comparable

In rc2, `anomaly_count` is exactly the segment-flag count and nothing else:
`annotate_segments` counts segments carrying any of `temperature` (>= 0.5),
`compression_ratio` (> 2.4), `avg_logprob` (< -1.0) or `no_speech` (> 0.5 with non-empty
text). `loop` is deliberately excluded. `anomaly_windows` is a separate field and was never
summed into it.

So structurally this is the same quantity `ANOMALY_SEGMENTS_MAX` reads, and neither of the
two changes since (the window clock running over real speech, and scoring taking the
maximum of the signals rather than their sum) touches the per-segment predicate.

**But the numbers are not on the same scale in practice, and there is direct evidence.**
`tcf.20240319b` is in this sweep. It scored `anomaly_count` of **1** on both of its repeat
submissions. The same file now scores 5 and 2. A file cannot go from 1 to 5 on an unchanged
predicate unless its inputs changed: either the decode moved, or more segments now survive
into `annotate_segments`, which the seam-dedupe change would do since the count runs on the
de-duplicated list. Whatever the reason, **the distribution below describes rc2 and cannot
be transplanted onto the candidate's scale.** It should be re-measured on the candidate
before the cap is set from it.

## The rc2 distribution, over 101 completed files

| anomalous segments | files |
|---|---|
| 0 | **99** |
| 1 | **2** |
| 2 or more | **0** |

Mean 0.020. p50 0, p75 0, p90 0, p95 0, p99 1, max 1.
At or above 3: **0**. At 4: **0**. At 5: **0**. At 6 or more: **0**.

`anomaly_windows` is equally sparse: 99 files at 0, 2 files at 1, none higher.

On this build there is no cluster at 4 to 6 and nothing approaches 5. The whole corpus sits
at 0 or 1, so on rc2's scale a cap of 5 was not tight, it was unreachable.

## Are the highest scorers poor transcripts

No. Both files that scored 1 are healthy by every independent measure:

| file | anomaly | word delta vs legacy | words/sec over speech | flags present |
|---|---|---|---|---|
| tcf.20240319b | 1 | **+0.81%** | 2.87 | temperature 1, compression_ratio 1, loop 2 |
| tcf.20251212 | 1 | **-0.07%** | 2.84 | temperature 1, compression_ratio 1 |

Neither is a transcript I would call bad, and neither appears on the regression list, the
uncovered-speech list or the quarantine list. The one anomalous segment in each is a single
segment that fell back to temperature 0.5 and came back with a high compression ratio, in a
file that is otherwise indistinguishable from the corpus median. A cap firing on these
would be firing on good output.

Across all 101 files there are only **2 temperature flags and 2 compression-ratio flags in
total**, against 73 `loop` marks which are correctly excluded from the count.

## Does fallback explain the run-to-run variation

**On this build, no, and I have a direct counterexample.** Exactly two files engaged the
temperature ladder far enough to flag a segment at 0.5 or above: `tcf.20240319b` and
`tcf.20251212`. `tcf.20240319b` is one of the five files I submitted twice, and it produced
a **byte-identical transcript** on both runs, with `anomaly_count` 1 both times.

So the one file in my determinism set where fallback demonstrably fired did not vary. That
is a single file and it does not disprove the mechanism, but it does mean "fallback fired"
was not sufficient for variation on rc2. The likeliest reconciliation is that ctranslate2's
sampling is seeded per call, making the rungs above zero reproducible within a build; if
the candidate now oscillates on the same file, that reproducibility is what changed, and it
is worth confirming directly rather than inferring.

The other four repeat files showed no temperature flag at all, so they were beam-search
only and their identical output says nothing either way.

## What I would advise

The rc2 distribution cannot set the candidate's cap, because the same file moved from 1 to
5 between builds. What it does establish is that on a corpus of a hundred varied
recordings, healthy output produced at most one anomalous segment, and the two files that
produced one were good transcripts. If the candidate's counts have shifted upward by
roughly the factor that file shows, a cap of 5 will sit inside the healthy body rather than
above it. Re-run the segment-flag count over the same hundred files on the candidate, which
is a decode-only pass and costs about two hours, and set the cap at the p99 of that
distribution with headroom rather than at a round number.
