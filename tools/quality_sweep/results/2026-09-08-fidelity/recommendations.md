# The fidelity experiment: what to change in 0.6.1, and what to leave alone

Read `fidelity.md` for the tables. This is the argument.

## The measurement, and the caveat that governs it

Where a passage is read aloud from scripture the correct words are knowable, so those
stretches are scored by word error rate against a public-domain translation. The World
English Bible is the reference of record; the American Standard, King James, American
King James, Young's Literal and Webster texts score the four passages that have them, to
show the ordering does not depend on the reference.

**The rates are not accuracy and must never be quoted as accuracy.** The preacher's
translation is unknown and differs from all six, so every rate is inflated by the
translation difference. What survives that inflation is the comparison: the same
reference scores every configuration, so a difference between two configurations on one
passage is real even though neither level is. Where the two configurations are the same
transcript the rate is identical to three decimals, which is the sanity check that the
scorer is measuring the transcript and not the reference.

The second, stronger instrument needs no reference at all. Every one-sided run of twelve
or more words between two decodes was extracted and checked against the legacy archive.
A run the legacy transcript also contains is content one of the two decodes lost, judged
by a baseline that predates this release line entirely. All twenty such runs are in the
legacy transcript, so every one is a real difference in content rather than a wording
choice.

Word count is not evidence and is not used as such. The single largest one-sided run in
the whole experiment is 137 words that A has and B does not, and it is a repetition loop:
"the sins we commit" fifteen times over, at 97 percent through `tcf.20201003`. On word
count A wins that file by 3 percent. On content it lost.

## 1. Should the primary pass sample rather than beam-search? No.

**What sampling gains.** Fourteen contiguous runs totalling 578 words are present in B
and in the legacy archive but absent from A. Ten of them, 492 words, are scripture read
or quoted aloud: Exodus 24:3-6, 1 Peter 1:22-24, 1 Peter 2:9-10, Psalm 25:17-21, Isaiah 64:4,
Colossians 1:11-12, Hebrews 4:16, Ephesians 4:11, Philippians 2:13 and an Ephesians 5
paraphrase. Two of those were the open cases: Exodus 24 in `tcf.20250607` moves from
0.834 to 0.442 against the World English Bible, and 1 Peter 1 in `ucf20211106b` from
0.676 to 0.437, with the same ordering under all six translations. **Eight of the ten were
not previously known to be missing at all.** No pre-identified passage scores worse under
B, and the long-gap count falls: five files with a gap of 15 s or more become two, three
of 20 s or more become one. Flagged segments fall from 34 to 23, low-rate windows from 3
to 1, and wall clock is unchanged at 1.6 percent faster.

**What sampling costs.** Six runs totalling 273 words go the other way. One is the 137
word loop, which B is right to lack. The other five, 136 words, are real content A keeps:
including Psalm 1:1 at the very start of `tcf.20230603_mens_fast`, a read-aloud scripture
passage, which is the exact defect the change is meant to cure, relocated to a different
file.

**Why that is not enough to change the primary decode.**

- *It is one draw.* Sampling is unseeded. The evidence for sampling is a single
  submission of each file, and the same configuration family contradicts itself inside
  this dataset: A's own rescue pass, which decodes from temperature 0.2 with conditioning
  off, ran on `tcf.20250607` and was not selected, on the very file where B's sampled
  primary recovered Exodus 24. Selection prefers the pass with more words once the anomaly
  totals tie, so that rescue either did not recover the passage or recovered it while
  scoring worse on anomalies. Either way, two nearby sampled draws, opposite outcomes on
  the same passage. A 578 to 136 split is suggestive, but with one draw per file it cannot be
  separated from luck, and the same argument that credits B for Exodus 24 has to credit
  chance for Psalm 1:1.
- *It makes the whole corpus irreproducible.* The shipped configuration returns a
  byte-identical transcript on 23 of 32 repeat submissions. The nine that vary are the
  ones where some pass sampled, and they vary by up to 0.032 disagreement and 6.5 percent
  of words. Under B every file is on that path. Nothing in 0.6.0 depends on determinism
  any more, since the retry fingerprint was removed, but a support answer of "resubmit and
  you get a different transcript" is a real cost for a service whose output is published.
- *It moves the aligner's input and nobody measured the aligner.* Total segments fall
  from 18 325 to 15 449 across the 32 files, and individual files move violently in both
  directions: `tcf.20230603_mens_fast` from 1389 segments to 356, `tcf.20150424` from 1156
  to 1811. 0.6.0's per-utterance MFA runs once per segment. Every configuration in this
  experiment is decode-only, so the stage most exposed to this change was never exercised.
- *It does not fix the known-short file.* `tcf.20150424` publishes 7060 words against a
  legacy 8292 under A and 7112 under B. Still about 14 percent short.
- Beam search at temperature 0 is the reference Whisper configuration and the baseline
  every earlier measurement in this project was taken against.

**Recommendation: leave `WHISPER_TEMPERATURE_BASE` at 0.0.** The recovery is real and
should be captured, but through the second pass that already exists rather than by
replacing the first one for every file. See item 2, which turns the same decode into both
the detector and the remedy on the minority of files that need it.

If this is revisited, the experiment that would settle it is three submissions of B on the
same 32 files with the aligner enabled, reporting the run inventory per draw and
`agree250`. That is a decode plus alignment sweep, not an analysis, so it is not run here.

## 2. What detects a transcript that is short but unflagged?

Two label sets are used. The **sweep bad** set is the seven recordings the earlier work
identified independently. The **confirmed omissions** set is the seven files where the
second decode and the legacy archive agree that the shipped transcript lost a run of 25
words or more. They overlap in four files and disagree usefully: the sweep set contains
three files (`tcf.20210210`, `tcf.20210217`, `tcf.20260626`) whose passages the shipped
rescue already recovers, and the confirmed set contains three (`tcf.20211203`,
`tcf.20221203`, `women_retreat_2025_session1`) that nobody had noticed were losing
scripture.

**Uncovered speech fraction does not work, and the stalled note was wrong.** Bad files
run 0.053 to 0.134, healthy files 0.011 to 0.128. The three highest values in the whole
subset are healthy recordings. At every false-positive budget from zero to five it catches
one of seven. Coverage ratio is the same statistic inverted and fails identically. Word
rate over speech catches none at zero false positives, and flagged segments none at any
budget. These are all measuring how ragged the segment edges are, which is a property of
the speaker and the room.

**Largest contiguous uncovered gap works, at a threshold nobody tried.** Against the
confirmed set it reaches all seven at 8.5 s, flagging eight files of 32 and one healthy
file. The earlier work only ever evaluated this signal at 10 s and above, and only as a
quarantine gate, where a false positive costs a published transcript. As a trigger for a
second decode a false positive costs GPU time, so the threshold belongs far lower. At 20 s
it reaches two of seven; at 8 s, all seven.

**Cross-configuration disagreement is a real detector, but only in its directional form.**
Symmetric disagreement between two decodes is nearly useless: 1 of 7 at zero false
positives against the sweep set, because it also fires when the second decode is the one
that is wrong. `tcf.20201003` has the highest reverse run in the corpus, 137 words, and
they are a hallucination loop. The statistic that works is **the longest run the second
decode has that the published pass lacks**: 4 of 7 at one false positive against the
sweep set, and against the confirmed set it recovers all seven, including `tcf.20150424`,
which no free signal sees at all. That last figure is partly circular, since the confirmed
set was defined by this statistic, and it is reported as a description of the label set
rather than as a score. What is not circular is that the runs it finds are corroborated
by the legacy archive, and that the gap filter, an entirely independent statistic, selects
the same files.

**Recommendation: change the rescue trigger, and read the second pass as the detector.**

- Add `RESCUE_MAX_GAP_S`, default 8.0, so a primary pass whose largest contiguous
  uncovered stretch reaches it fires the rescue, alongside the existing
  `anomaly_windows >= 1`.
- Publish the longest run the rescue has that the primary lacks as a diagnostic and a log
  line, so a reviewer can see what the second decode found.
- Do not quarantine and do not requeue on either. The existing selection already prefers
  the pass with more words, so the same decode that detects the loss repairs it.

Measured effect on this subset: the rescue would fire on 8 files of 32 rather than 3, and
would reach every confirmed loss instead of two of seven. Cost: about 25 percent more
decode on this subset. On the 101-file sweep 11 files had a gap of 10 s or more, so budget
15 to 25 percent, not 100 percent. Risk: those files become irreproducible between
submissions, which is already true of every file the rescue touches today. To
re-validate: rerun the 101-file sweep with the new trigger and confirm that no file loses
words to a selected rescue, that the retention floor still blocks the cases it was built
for, and that the extra rescues do not move `agree250`.

## 3. Should the profile selector key on sample rate and bit rate? No.

On the 20 files whose profile changes, the fidelity-keyed selector buys almost nothing and
costs words.

| statistic | A, keyed on level | D, keyed on sample and bit rate |
|---|---|---|
| segments per minute, median | 15.16 | 14.66 |
| segments per minute, mean | 20.72 | 18.01 |
| mean segment length, median | 3.48 s | 3.50 s |
| files less fragmented / more / unchanged | - | 9 / 6 / 5 |
| word delta against A, median | - | -0.06 percent |
| word delta against A, mean | - | -0.43 percent |
| files losing more than 1 percent of words | - | 7 |
| words against the legacy archive, subset total | -981 | -1357 |
| uncovered speech, subset total | 2440 s | 2717 s |

Mean segment length does not move at all. The mean segments per minute improves because
four badly fragmented files improve a lot, and six other files get worse: `ucf20211106b`
goes from 10.3 to 31.7 segments per minute, a threefold fragmentation, and `tcf.20150424`
from 23.1 to 36.6. The one clear fidelity win is on that same `ucf20211106b`, where the
quiet profile recovers 1 Peter 1:22-24 from 0.676 to 0.437, and it arrives attached to the
threefold fragmentation.

Alignment quality could not be measured. These are decode-only runs with no MFA output,
and alignment is precisely what fragmentation damages, so the one cost that matters most
is unobserved.

**Recommendation: leave the level rule alone.** The earlier finding that level is
non-monotonic stands, and it remains a reason to distrust the current rule. It is not a
reason to adopt this one, which trades a marginal median improvement for measurable word
loss, more uncovered speech and an unmeasured alignment risk. If the selector is revisited
it should be as a two-sided level rule, which is what the non-monotonicity actually
implies, measured with the aligner running.

## 4. Do the rescue's two levers compose? No, and conditioning-off is the one doing the work.

C runs the rescue with the ladder at 0.2 but leaves `condition_on_previous_text` on, which
isolates the second lever.

- `tcf.20210217`. A publishes 1683 words, exactly the legacy count, with Matthew 6 intact
  at 0.435. C publishes 1594 and loses a 91-word run: Matthew 6 goes to 0.770. Same
  ordering under all six translations (0.435/0.770 WEB, 0.514/0.788 ASV, 0.498/0.787 KJV,
  0.455/0.772 AKJV, 0.598/0.817 YLT, 0.481/0.783 Webster). Starting the ladder higher does
  not substitute for turning conditioning off; removing conditioning-off costs the passage
  the rescue exists to recover.
- `tcf.20260626`. C loses a 70-word run that A keeps.
- `tcf.20250607`. A and C are identical, and both lose Exodus 24 at 0.834 with 5-gram
  containment of 0.013.

**Leave `RESCUE_TEMPERATURE_BASE` and the rescue's conditioning setting exactly as they
are.** The combination in 0.6.0 is the right one and the two levers are not
interchangeable.

**On recovering Exodus 24 economically: it cannot be done by tuning the rescue.** The
rescue already runs on that file, with both levers set the way the passage probe said was
best, and its output was not selected. Only a differently sampled
draw recovered it, and only once. What the trigger change in item 2 buys for that file is
not a guaranteed recovery but a second attempt and a review marker, instead of publishing
196 words over the legacy count while silently missing six verses of Exodus.
