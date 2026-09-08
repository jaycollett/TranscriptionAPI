# The free analysis: can a second decode's disagreement detect a bad transcript?

Run before any GPU time was spent, over data already on disk. The result is negative, and
the reason it is negative is worth more than the statistic: **the two decodes I already
held are not two configurations.**

## What was compared

101 recordings, each with two full transcripts:

- the rc2 sweep run (`run/transcripts/`), the release candidate as it stood at 15:47;
- the shipped 0.6.0 decode-only pass (`rc4/candidate_pass.json`).

Statistics per file, over `norm_words` tokens: whole-file agreement
(`2M/(len(a)+len(b))` from one `SequenceMatcher` diff) and the longest single stretch of
consecutive words one decode has and the other does not. The second is the shape the
read-aloud defect actually has, and the first is nearly blind to it: 91 dropped words in
a 1600-word transcript move the whole-file ratio by under 0.03.

The ground truth is the independent `unhealthy` predicate from `candidate_anomaly.py`:
word rate over speech under 1.5, an uncovered gap of 20 s or more, or more than 5 percent
short of the legacy transcript. Neither statistic above is an input to it.

## The result

| statistic | healthy median | healthy p95 | healthy max | bad files in the top 5 |
|---|---|---|---|---|
| longest one-sided run | 0 words | 6 | 74 | 2 of 6 |
| whole-file disagreement | 0.00000 | 0.00485 | 0.03690 | 2 of 6 |

Six of the seven independently-bad files are comparable (`tcf.20150424` has no rc2
transcript at all; see below). Of those six, **four disagree by exactly zero**: the two
decodes produced byte-identical text. Ranked by longest run, the six sit at positions 1,
3, 61, 75, 95 and 99 of 101.

A threshold catching the worst bad file costs zero false positives; catching two costs
two; catching a third requires a threshold of zero, which flags 95 of the 95 healthy
files. There is no usable operating point beyond the first two.

## Why the answer is negative, and what it actually shows

The five files with any material disagreement are exactly the five where **0.6.0's own
rescue pass fired and was selected**:

| file | run | side | rc2 words | 0.6.0 words | legacy | rescue kept |
|---|---|---|---|---|---|---|
| tcf.20210217 | 91 | 0.6.0 | 1595 | 1687 | 1683 | yes |
| tcf.20240326b | 74 | 0.6.0 | 2753 | 2829 | - | yes |
| tcf.20260626 | 70 | 0.6.0 | 3750 | 3770 | 3697 | yes |
| tcf.202606623b | 70 | 0.6.0 | 1307 | 1349 | 1375 | yes |
| tcf.20260717 | 66 | 0.6.0 | 3708 | 3786 | 3779 | yes |

Every run is on the 0.6.0 side, and in every case the added words move the transcript
*towards* the legacy word count. So this measurement is not detecting decode variance at
all. It is detecting the rescue that 0.6.0 added between rc2 and release, and confirming
that the rescue recovers real content rather than inventing it. That is a useful thing to
have confirmed, and it is not the thing the analysis set out to test.

The two builds share every decode parameter. They differ in gates, thresholds and
post-processing. Where those do not fire, the decode is the same decode, so agreement
between them is close to a tautology: 95 of 101 files agree to the word. A detector built
on this pair would fire only where the rescue already fired, which is where the service
already knows something happened.

## The one file that matters most cannot be tested from held data

`tcf.20150424` is the file the whole open item is about: it publishes about 15 percent
short of its legacy transcript with zero flagged segments, zero low-rate windows and no
uncovered gap over 20 s. It errored in the rc2 sweep run and has no rc2 transcript, so it
has no second decode on disk at all. It is the one file in the corpus for which this
analysis has nothing to say.

## What it costs to answer properly

A genuinely different decode path. Configuration B in the fidelity experiment is exactly
that: the primary temperature ladder from 0.2, which makes faster-whisper sample instead
of beam-search. A against B is two different decoders on the same audio, so their
disagreement measures the decode. That pairing is carried into `fidelity_analyze.py` and
scored against the same independent evidence, with one addition this analysis showed to be
necessary: a **noise floor**. Two runs of A itself are not always identical, so a
disagreement smaller than the one A produces against itself is not evidence of anything.
