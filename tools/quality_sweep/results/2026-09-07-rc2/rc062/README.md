# 0.6.2 on the hundred files: the seeded baseline, and what selection ordering would change

102 files through `transcription-api:0.6.2` on the isolated sweep service, with
`RESCUE_TRANSCRIPT_DIR` enabled so both passes of every rescue are on disk. Job identifiers
are UUIDv5 of the filename, so the decoder seed is a function of the recording and this run
is reproducible by anyone who submits the same files under the same identifiers. Every seed
is in `seeds.json`.

**102 of 102 completed. No errors, no quarantines, MFA applied on 102 of 102.**

## 1. The baseline (replaces the 0.6.1 figures)

| | |
|---|---|
| fired | **20 of 102 (19.6 percent)** |
| trigger: gap only | 10 |
| trigger: window + gap | 10 |
| trigger: window only | 0 |
| rescue selected | **9 of 20** |
| genuine recoveries against the legacy archive | **6 of 9** |

The six that recover, by 5-gram recall against the legacy transcript:

| file | primary | published | recall before | after | gain |
|---|---|---|---|---|---|
| tcf.20210217 | 1591 | 1685 | 0.94532 | **0.99210** | +0.0468 |
| tcf.20240621 | 4417 | 4560 | 0.94827 | 0.97922 | +0.0310 |
| tcf.202606623b | 1306 | 1348 | 0.93283 | 0.96679 | +0.0340 |
| tcf.20260717 | 3700 | 3766 | 0.96708 | 0.98621 | +0.0191 |
| women_retreat_2026_session1 | 7886 | 7977 | 0.92786 | 0.94216 | +0.0143 |
| tcf.20260626 | 3735 | 3802 | 0.96513 | 0.97317 | +0.0080 |

Two selected rescues slightly reduce recall (tcf.20210604 −0.006, tcf.20240713 −0.013) and
one has no legacy row. As before: **20 fire, 9 are selected, 6 genuinely recover.**

Of the seven confirmed content losses, only **Matthew 6:19-21 on tcf.20210217** recovers in
this run. Matthew 12 on tcf.20210210 does not: under its seeded draw the rescue came back
35 words short and failed the retention ratio, so it was never eligible.

## 2. Regression against the 0.6.1 corpus

One file loses more than five percent of its 0.6.1 word count: **tcf.20210210**, 1247 to
1173. That is the same file as above. Its 0.6.1 result came from an unseeded draw that
happened to recover Matthew 12; the seeded draw did not. Nothing was broken, the lottery
was simply re-run with a fixed ticket.

No selected rescue lost content against its own primary. Alignment is unchanged in
aggregate: MFA applied on 102 of 102, every file completed, and `agree250` over 204
alignments has mean 0.607, median 0.640, p5 0.366.

## 3. Did seeding cost anything

No. Across the 100 files with a legacy row, comparing 0.6.2 against 0.6.1 by legacy recall:

| | |
|---|---|
| mean change | **−0.00071** |
| median change | **0.00000** |
| better | 6 files |
| worse | 5 files |
| unchanged | **89 files** |
| aggregate recall | 0.95543 → 0.95472 |

Word counts move +3.9 on average. **The seeded draws are indistinguishable from the
unseeded ones in aggregate**, which is what a per-identifier seed should do: it fixes which
draw each recording gets without biasing the draw. Eighty-nine of a hundred recordings are
bit-identical in content terms; the eleven that move are the rescue files, and they move in
both directions in almost equal number.

## 4. Replaying `content_first`: it changes nothing

Over all 20 retained rescues, `content_first` publishes **the same pass as the current rule
on 20 of 20. Zero flips.**

The statistic is working, not silent: corroborated contiguous runs of 8 to 94 words were
found on both sides of most pairs. The ordering never decides anything because:

- On 9 of 20 the rescue is **ineligible on the retention guard**, so no ordering can reach
  it, `content_first` included.
- On the remaining 11, wherever the two passes tie on anomaly score, the pass holding the
  larger corroborated run is also the pass with more words, so both rules pick the same
  one. Where they differ on anomaly score, that key decides first in both rules.

It does behave correctly on `tcf.20241105`, keeping the primary as the earlier analysis
said it should, though it does so because the rescue is ineligible rather than because the
ordering preferred the primary.

**The six scripture readings would not publish under `content_first`.** For every one of
them the rescue that holds the reading is excluded by the retention guard before ordering
is consulted. The ordering is not the lever.

## 5. What the lever actually is, and why I am not recommending pulling it

The retention guard rejects a candidate for losing words. Publishing the nine
retention-blocked rescues anyway would, on balance, produce *better* transcripts:

| file | words lost | recall published | recall if rescue published | change |
|---|---|---|---|---|
| tcf.20250607 (Exodus 24) | 309 | 0.79564 | **0.90086** | **+0.1052** |
| tcf.20241105 | 390 | 0.76092 | 0.83060 | +0.0697 |
| ucf20211106b (1 Peter) | 73 | 0.86863 | 0.92131 | +0.0527 |
| tcf.20221203 | 217 | 0.91027 | 0.96158 | +0.0513 |
| women_retreat_2025_session1 | 28 | 0.95450 | 0.98626 | +0.0318 |
| tcf.20260428 | 137 | 0.90902 | 0.91495 | +0.0059 |
| tcf.20150424 | 85 | 0.82393 | 0.80976 | −0.0142 |
| tcf.20210210 | 35 | 0.93204 | 0.90049 | −0.0316 |
| tcf.20211203 | 271 | 0.88968 | 0.85645 | −0.0332 |

**Six better, three worse, net +0.238 recall over nine files.** The words those rescues
"lose" are largely words the legacy transcript does not have either; what they gain is
content it does. Across all 19 rescues with a legacy row, **13 have higher legacy recall
than their primary** — the rescue is more often the better transcript than the rules credit.

But I cannot recommend a content-aware exemption, because **nothing observable at decision
time separates the six that help from the three that hurt.** Spearman against the recall
change, over the 19 rescues:

| candidate signal | rho |
|---|---|
| anomaly score improvement | 0.24 |
| reduction in largest uncovered gap | 0.16 |
| words lost | 0.05 |
| rescue's own largest gap | 0.04 |
| reduction in total uncovered speech | −0.25 |
| corroborated run length | does not separate (64 hurts, 14 helps) |

Every correlation is weak, and the corroborated-run statistic that motivated
`content_first` does not predict benefit either: the largest corroborated run in the blocked
set (64 words, tcf.20211203) belongs to a rescue that would *reduce* recall by 0.033.

Legacy recall is the only thing that separates them, and it is unavailable for a new sermon,
which is the whole reason this archive is useful for calibration and useless as a runtime
signal.

## Recommendation

**Do not ship `content_first`.** On this corpus it is a no-op: twenty rescues, zero flips.
The earlier measurement of five recordings gained was taken on unseeded draws that no longer
exist, which is exactly the class of result 0.6.2 was built to retire.

The real finding is that the retention guard, not the ordering, decides these cases, and it
is currently deciding six of nine of them the wrong way. That is worth pursuing, but not
blindly: a naive relaxation would also publish three transcripts that are measurably worse.
What would make it tractable is a runtime signal for "this pass holds content the other
lacks" that actually correlates with being right, and none of the five I tested does.

Two cheaper things worth having in the meantime, both enabled by the retention now being on
by default: keep `RESCUE_TRANSCRIPT_DIR` on in production, so the discarded pass survives on
the jobs where it mattered, and revisit this with a human reading a handful of the blocked
pairs. Nine files is a small enough set to adjudicate by eye, and eye adjudication is the
only method here that has ever separated the cases correctly.
