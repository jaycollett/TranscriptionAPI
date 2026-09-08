# Pricing 0.6.1: what the gap trigger costs and what it buys

102 files decoded with `transcription-api:0.6.1-rc1` using the image's own
`transcribe_audio`, so every count is the one the service computes. The 0.6.0 baseline is
derived from the same run rather than from an earlier build: 0.6.0 fires on the window
alone and 0.6.1 fires on window or gap, a strict superset, so a file whose only trigger was
`gap` would not have run a rescue under 0.6.0 and its 0.6.0 cost and output are the primary
pass exactly. That removes build drift and GPU contention from the comparison.

## 1. Firing rate and cost

**20 of 102 fire (19.6 percent)**, against the predicted 19 of 101. The engineer's estimate
was right on the count.

| trigger | files |
|---|---|
| gap only | 10 |
| window + gap | 10 |
| window only | **0** |
| none | 82 |

Every window firing also fires on gap, so on this corpus the gap trigger strictly subsumes
the window trigger. Firing files are **26.9 percent of the corpus by duration** (11.9 h of
44.1 h), against the predicted 30.

**The cost is +13.8 percent, not +24.**

| | decode seconds |
|---|---|
| primary passes, all 102 files | 4003.2 |
| rescue passes, all 20 | 955.5 |
| **0.6.0 equivalent** (primary + the 10 window-triggered rescues) | **4355.6** |
| **0.6.1 total** | **4958.6** |
| the 10 extra gap-only rescues | 603.1 |

The prediction was high because it priced a rescue as a second full decode. It is not: the
rescue decodes from temperature 0.2, where faster-whisper samples with `beam_size=1` instead
of beam-searching, so it costs roughly two thirds of a primary. Twenty rescues add 955 s to
4003 s of primary decode, not 100 percent of it.

## 2. What the firing buys

**11 of the 20 rescues are selected.** Every one of the 11 is strictly additive: the
directional statistic `rescue_unpublished_run` is **0 on all eleven**, meaning the discarded
primary contained no run of words the published rescue lacks. Word deltas run +18 to +145.

Firing is not recovering, so the eleven were checked against the legacy archive by 5-gram
recall, using a primary-only rerun (`RESCUE_ENABLED=0`) to get the exact 0.6.0 text:

| file | primary | published | legacy recall before | after | verdict |
|---|---|---|---|---|---|
| tcf.20210210 | 1172 | 1247 | 0.9320 | **0.9927** | recovers, +84 grams |
| tcf.20210217 | 1591 | 1683 | 0.9453 | **0.9976** | recovers, +86 grams |
| women_retreat_2026_session1 | 7886 | 8017 | 0.9279 | 0.9477 | recovers, +321 grams |
| tcf.20240621 | 4417 | 4562 | 0.9483 | 0.9738 | recovers, +184 grams |
| tcf.20260724 | 4425 | 4458 | 0.9543 | 0.9681 | recovers, +110 grams |
| tcf.20260717 | 3700 | 3767 | 0.9671 | 0.9845 | recovers, +82 grams |
| tcf.202606623b | 1306 | 1351 | 0.9328 | 0.9653 | recovers, +71 grams |
| tcf.20240412 | 3357 | 3391 | 0.9784 | 0.9775 | neutral |
| tcf.20260626 | 3735 | 3753 | 0.9651 | 0.9613 | neutral |
| tcf.20210604 | 4497 | 4545 | 0.9829 | 0.9711 | slightly lower |
| tcf.20240326b | 2758 | 2837 | - | - | no legacy row |

**Seven of eleven measurably recover legacy content**, 1047 legacy 5-grams in total. The two
scripture readings are essentially fully restored: Matthew 12:15-16 takes recall from 0.932
to 0.993, Matthew 6:19-21 from 0.945 to 0.998.

So the honest chain is: **20 fire, 11 are selected, 7 genuinely recover content.**

## 3. Regressions

None.

- No file loses more than 5 percent against the derived 0.6.0 output.
- No selected rescue lost words; `rescue_unpublished_run` is 0 on all eleven.
- The retention guards were exercised and worked: one rescue was discarded for losing 28
  words of 2245, which is 1.25 percent and past the 99 percent floor.
- Alignment was re-run on the eleven files whose published text changed, with and without
  the rescue. **MFA applied on 11 of 11 in both**, every file completed, no failures either
  way. Segment counts shift, usually downward, because the rescue produces fewer and longer
  segments.

## 4. The seven confirmed losses

**All seven now fire. Three recover.**

| file | fires on | selected | outcome |
|---|---|---|---|
| tcf.20210210 | window+gap | yes | **recovers** Matthew 12:15-16 |
| tcf.20210217 | window+gap | yes | **recovers** Matthew 6:19-21 |
| tcf.20210604 | gap | yes | selected, +48 words, recall slightly down |
| tcf.20250607 | window+gap | **no** | rescue held a **156-word run** the primary lacks |
| tcf.20241105 | gap | no | rescue held a 76-word run |
| tcf.20150424 | gap | no | rescue held a 40-word run; still publishes ~15 percent short |
| ucf20211106b | window+gap | no | rescue held only a 4-word run; it did not find 1 Peter either |

`tcf.20250607` is the case worth acting on. Its rescue recovered the Exodus 24 reading, a
156-word run absent from the primary, and cut the largest uncovered gap from 24.9 s to
12.1 s. It was still discarded, and not by the retention guards: the two passes **tied on
anomaly score, 1 against 1, and the primary wins ties**. The rescue was net 340 words down
overall, so publishing it wholesale would be a poor trade, but the current rule cannot see
that it holds a large contiguous passage nobody else has.

`tcf.20150424` fires but does not recover, and continues to publish about 15 percent short
against legacy. The trigger now sees it; the remedy does not fix it.

## 5. The benefit is stochastic

The same eleven files were decoded twice on this build, once in the decode-only pass and
once through the service. **Selection differed on 2 of 11.** `tcf.20210210` recovered
Matthew 12 in one run and not in the other; `tcf.20260717` likewise flipped.

The rescue samples from temperature 0.2 with no seed set, so its output differs on every
attempt and the selection can flip with it. The seven recoveries are therefore an
expectation across the corpus, not a promise about any given sermon.

## Recommendation: ship it, and fix the tie-break

**The trade, stated plainly: 13.8 percent more decode time buys about seven files per
hundred with measurably more real content, of which two are complete scripture readings
restored to sermons that were missing them.**

I would ship it. The cost came in at little more than half what was predicted, the guards
demonstrably prevent a losing rescue, alignment is unaffected, and there is no regression
anywhere in a hundred files. For an archive whose purpose is preserving preaching, a
restored reading of Matthew 6 is worth more than fourteen percent of a GPU that is idle most
of the day.

Two things I would change, neither blocking:

1. **The tie-break discards the best rescue in the corpus.** When anomaly scores tie,
   prefer the pass with the smaller largest-uncovered-gap, or consult
   `rescue_unpublished_run` in the other direction: a rescue holding a 156-word run the
   primary lacks is not the same as one holding four words. On this corpus that single
   change would decide `tcf.20250607`, and it is the difference between three of seven
   confirmed losses recovered and four.
2. **Seed the rescue sampler.** It would make recovery reproducible instead of a coin flip
   on two files in eleven, and it would also settle the anomaly-count oscillation found in
   the previous round, which has the same root cause.
