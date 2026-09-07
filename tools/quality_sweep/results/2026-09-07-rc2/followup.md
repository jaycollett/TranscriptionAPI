# Follow-up analysis on the 0.6.0-rc2 sweep

Three questions answered from the sweep data, plus one direct experiment: the gap regions
were cut out of the source audio and re-submitted to the same service in isolation, which
is what turns "no segments here" into "the decoder cannot transcribe this".

## 1. The replacement gate, verified retroactively

### How many files trip

| largest contiguous uncovered gap | files of 101 |
|---|---|
| >= 10 s | 11 |
| >= 15 s | 6 |
| >= 20 s | 4 |
| >= 25 s | 2 |
| >= 30 s | 1 |

The curve is steep between 10 s and 20 s and flat after: 11, 6, 4, 2, 1. There is no
natural knee, so the choice is a policy about review volume rather than a discovery. The
total backstop at 0.25 catches 2 files.

### What each tripping file actually contains

Every gap of 15 s or more was cut out with 3 s of padding and re-transcribed on its own.
That is the decisive test: if the decoder produces speech from the isolated clip, the
full-file decode omitted it.

| file | gap | level in gap | isolated re-decode | verdict |
|---|---|---|---|---|
| tcf.20250607 | 76.3-101.1 (24.8 s) | -18.7 dB | **102 words**: Exodus 24 read aloud | **missing speech** |
| tcf.20210604 | 290.2-314.8 (24.6 s) | -18.6 dB | **67 words**: the C.S. Lewis quotation from The Problem of Pain | **missing speech** |
| ucf20211106b | 1225.6-1250.8 (25.2 s) | -20.9 dB | **57 words**: 1 Peter 1:22-25 read aloud | **missing speech** |
| tcf.20241105 | 1299.4-1326.7 (27.3 s) | -23.2 dB | **89 words** of Q&A discussion | **missing speech** |
| tcf.20241105 | 1657.7-1674.1 (16.4 s) | -21.4 dB | **48 words** of Q&A discussion | **missing speech** |
| tcf.20241105 | 2585.8-2632.6 (46.8 s) | -25.7 dB | 20 words over 8.2 s of 46.7 s of VAD speech; quarantined on the word-rate floor | **not transcribable** |
| tcf.20211203 | 1059.0-1077.5 (18.5 s) | -35.4 dB | 19 words, all of which are the surrounding context | **legitimate pause** |
| tcf.20220729 | 499.1-516.2 (17.1 s) | -30.9 dB | 13 words over 7.1 s of 23.1 s; quarantined as garbage | **mostly pause** |

The level in the gap separates the two classes cleanly on this sample: every gap at
-18 to -23 dB is missing speech, and every gap at -30 dB or quieter is a pause. The
46.8 s case sits between them at -25.7 dB and is neither.

### The priority case: tcf.20241105, 46.8 s

**0.6.0 did not drop 90 seconds of a sermon.** Re-decoding that 46.8 s in isolation, with
no surrounding context to blame, produces the same near-silence: 20 words covering 8.2 s
of the 46.7 s Silero calls speech, a word rate of 0.43/s that fails the floor. The region
sits between "One or two more thoughts, questions." and "Yeah, I don't, so I don't know
that I have a ton of advice here", which is an audience member asking a long question off
microphone. The level profile is flat at -25 to -27 dB for the whole 46.8 s, about 7 dB
under the speaker, then jumps to -18.7 dB exactly where the answer begins. This is distant
off-mic speech the decoder cannot resolve, and it reproduces in isolation, so it is a
property of the audio and not of the release.

The file's own word delta is -3.25 percent against legacy, and the differences are spread
over 70 runs across the whole file rather than concentrated in this gap, which is what a
recording with a poorly-miked audience looks like.

### The finding that matters more than the gate

**Neither regression would trip the proposed gate, and both are real contiguous
omissions.** `tcf.20210210` drops 73 consecutive words yet its largest gap is 5.37 s;
`tcf.20210217` drops 91 yet its largest gap is 6.65 s. In both cases the decoder emitted
*some* segments over the omitted stretch, just far too few words, so no large hole appears
in the coverage. Their uncovered totals are 0.119 and 0.105, above the old 0.10 but far
below the 0.25 backstop.

A coverage gate, at any threshold and on either statistic, is blind to sparse-but-present
output. The instrument that does catch it is a word-rate-per-window check, which is
exactly the low-speech-window check that fired four windows on `tcf.20150424`. The
recommendation is therefore not to replace the window check with the gap check but to keep
both: the gap check for holes, the window check for thin stretches.

## 2. The two regressions: contiguous omissions, both scripture

| file | legacy | new | delta | similarity | shape |
|---|---|---|---|---|---|
| tcf.20210210 | 1248 | 1173 | -6.01% | 0.9682 | one 73-word run at 1.1% through the file, plus 3 words at the start |
| tcf.20210217 | 1683 | 1595 | -5.23% | 0.9713 | one 91-word run at 1.1% through the file |

Both are single contiguous omissions, not distributed wording noise, and both are a
scripture reading in the first two percent of the recording:

- tcf.20210210 loses Matthew 12:15-16, "jesus aware of this withdrew from there and many
  followed him and he healed them all and ordered them not to make him known".
- tcf.20210217 loses Matthew 6:19-21, "do not lay up for yourself treasures on earth where
  moth and rust destroy and where thieves break in and steal".

This is the same class as three of the four confirmed gap omissions above (Exodus 24,
1 Peter 1, the Lewis quotation). **Across the sweep, what 0.6.0 loses is read-aloud
passages**: a change of register from preaching to reading, often in the opening minutes
before any context is established. That is a single coherent defect, not five unrelated
ones, and it is worth attacking directly rather than through gate thresholds.

## 3. What actually predicts fragmentation

Spearman rank correlation against segments per minute, over the 101 completed files:

| signal | rho |
|---|---|
| gap count per minute | +0.862 |
| uncovered fraction | +0.704 |
| agree250 | -0.503 |
| **bit rate** | **-0.181** |
| **sample rate** | **-0.175** |
| speech / total duration | +0.132 |
| detector disagreement | +0.118 |
| era | -0.117 |
| **mean dBFS** | **+0.090** |
| multi voice | -0.075 |
| duration | +0.037 |
| words per second over speech | -0.013 |

The top two are definitional (more segments makes more inter-segment gaps) and agree250 is
a consequence, not a cause: fragmentation degrades alignment. Among the independent
signals, encoding quality leads and level is last but one.

Group medians make the size of it clearer than the rank correlation does:

| sample rate | n | median segments/min |
|---|---|---|
| 22.05 kHz | 24 | **13.2** |
| 32 kHz | 2 | 19.5 |
| 44.1 kHz | 23 | 11.2 |
| 48 kHz | 52 | **9.1** |

Low fidelity (22.05 kHz or under 64 kbps): 34 files, median 13.2. Everything else: 67
files, median 9.3. Of the twenty most fragmented files, 9 are 22.05 kHz and 8 are under
64 kbps, against only 3 below -26 dBFS.

**Level is not merely weak, it is non-monotonic:**

| mean level | n | median segments/min |
|---|---|---|
| below -26 dB | 20 | 12.1 |
| -26 to -22 | 26 | 9.1 |
| -22 to -19 | 40 | 9.9 |
| above -19 | 15 | **14.2** |

Both ends fragment and the middle does not, and the loudest band is the worst of all. A
one-sided threshold at -26 dB cannot express that shape whatever value it takes, which is
a stronger objection to the current selector than the position of the cut.

**Ranked answer for the next release.** Key the profile on sample rate and bit rate first
(a 22.05 kHz or sub-64 kbps file is the one that fragments), and if a level term is kept
at all it must be two-sided, treating both the very quiet and the very loud as needing the
gentler profile. Level alone, one-sided, is the wrong signal.
