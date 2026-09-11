# The noise floor: 0.6.0 against itself

Before any configuration can be compared to 0.6.0, 0.6.0 has to be compared to itself.
Two passes decoded the same 32 recordings with the same image, on the same host, with
nothing overlaid: the decode-only pass run at 21:42 and config A of the fidelity run at
00:20. Same code, same parameters, same audio. Every difference between them is the
decode's own run-to-run variation.

## The result

**9 of 32 files are not reproducible.** 23 produced byte-identical transcripts; the other
9 did not.

| statistic | worst case across the 32 |
|---|---|
| absolute word delta | **6.48 percent** (tcf.20210210, 1172 against 1248) |
| longest one-sided run | **137 words** (tcf.20201003.mens_) |
| whole-file disagreement | **0.032** |
| rescue decision flipped | **3 files** |
| `anomaly_windows` changed | **3 files** |

| file | duration | A | A2 | delta | disagreement | longest run | rescue A | rescue A2 |
|---|---|---|---|---|---|---|---|---|
| tcf.20210210 | 470 s | 1248 | 1172 | +6.48% | 0.03220 | 73 | fired, kept | fired, refused |
| tcf.20201003.mens_ | 1639 s | 4521 | 4384 | +3.12% | 0.01529 | 137 | none | none |
| tcf.20260626 | 1505 s | 3799 | 3752 | +1.25% | 0.00843 | 49 | fired, kept | fired, kept |
| women_retreat_2025_session4 | 1626 s | 3158 | 3195 | -1.16% | 0.01824 | 9 | none | fired, kept |
| tcf.20260331b | 764 s | 1962 | 1958 | +0.20% | 0.00102 | 4 | none | none |
| tcf.20260717 | 1419 s | 3767 | 3762 | +0.13% | 0.00172 | 3 | fired, kept | fired, kept |
| women_retreat_2026_session1 | 3184 s | 7881 | 7886 | -0.06% | 0.02656 | 88 | fired, kept | fired, refused |
| tcf.20210217 | 617 s | 1683 | 1683 | 0.00% | 0.00059 | 1 | fired, kept | fired, kept |
| tcf.20220729 | 2215 s | 5667 | 5667 | 0.00% | 0.00388 | 9 | none | none |

Duration does not explain it: 7 of 20 files under 1800 s vary and 2 of 12 over it do.

## Why this matters more than it looks

**It breaks the read-aloud story into two halves.** `tcf.20210210` loses Matthew 12:15-21
in one run and keeps it in the other. Scored against the World English Bible the same
passage reads 0.824 in the pass that lost it and 0.407 in the pass that kept it, against
0.407 for the legacy transcript. So "0.6.0 drops this passage" is not a property of the
release on this file; it is a coin flip, and the coin lands on the omission often enough
that two separate investigations have caught it doing so.

**It undermines the rescue trigger specifically.** `anomaly_windows` is the only signal
that fires the rescue, and it moved from 1 to 0 on two files and from 2 to 0 on a third
between two runs of the same code. The rescue therefore fires or does not fire on a
threshold whose input is not stable, and on three of the nine varying files the published
transcript is decided by which side of that flip the run landed on.

**It sets the bar for every other comparison.** Two runs of the *same* configuration
produce word deltas up to 6.5 percent and one-sided runs up to 137 words on healthy
material. So a single file where configuration B differs from A by 3 percent, or shows a
100-word run, is not evidence about B. Only differences that are consistent across many
files, or larger than 6.5 percent on one file, carry information. Every recommendation in
this experiment is stated against that floor.

**It contradicts what the release believed about itself.** `docs/QUALITY_PROPOSAL.md` 4.1
records "five of five repeat submissions byte-identical", and `anomaly_cap.md` reasons
from a byte-identical repeat that "ctranslate2's sampling is seeded per call, making the
rungs above zero reproducible within a build". Both are true of the five files that were
checked and false of the corpus: those five were drawn to span duration, not to span the
files that vary. Determinism was measured on a sample that happened to contain none of
the unstable population.

## What it does not show

This is not a defect introduced by 0.6.0's changes, and nothing here says the varying
transcripts are worse. Float16 GEMM reductions on a GPU are not order-deterministic, and
beam search amplifies a tie into a different token sequence. The finding is about what
can and cannot be concluded from a single decode of a single file, not about the quality
of any one of them.
