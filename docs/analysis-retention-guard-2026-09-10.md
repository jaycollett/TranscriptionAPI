# Should the rescue retention guard count content instead of words

Answer: no. **Leave the guard alone.** The proposal gains nothing on the corpus and
costs between one and three published rescues, and the reason it gains nothing is that
the retention guard is not what refuses these rescues.

Evidence is in `tools/quality_sweep/results/2026-09-10-retention-guard/`, produced by
`tools/quality_sweep/rescue_deficit.py` and `tools/quality_sweep/retention_threshold.py`
against the 0.6.1 corpus run on devmachine.

## What the run actually did

102 files, 20 rescues fired, 11 selected, **9 refused**. The earlier "six refused" list
was a subset of twelve `tcf.*` files. `/home/jay/sweep/rc061/selected_list.json` holds
exactly the eleven selected files and lists no refusals at all, which is where the
mismatch came from; `rc061.log` carries nine `Discarding the rescue pass:` lines and is
authoritative.

## Step 2: is the word deficit filler

Per refusal, `P` is the longest contiguous run the primary holds, the rescue lacks, and
the legacy archive corroborates. `R` is the same the other way round.

| file | lost | P | R | what the rescue holds that the primary does not |
|---|---|---|---|---|
| tcf.20150424 | 79 | 4 | 40 | Colossians 1:11-12, coverage 1.00 |
| tcf.20211203 | 296 | 4 | 59 | 1 Peter 2:9-10, coverage 1.00 |
| tcf.20221203 | 120 | 5 | 25 | Ephesians 4:11, coverage 1.00, plus 24 and 19 word runs |
| tcf.20240713 | 48 | 0 | 27 | Job 42:5-6, coverage 1.00 |
| tcf.20241105 | 344 | 69 | 8 | nothing at twelve words or more |
| tcf.20250607 | 340 | 4 | 156 | Exodus 24 read aloud, coverage 0.95 |
| tcf.20260428 | 149 | 35 | 34 | four runs, only the 13 word one well corroborated |
| ucf20211106b | 66 | 0 | 64 | 1 Peter 1:22, coverage 1.00 |
| women_retreat_2025_session1 | 28 | 24 | 26 | Hebrews 4:16, coverage 1.00 |

Six of the nine are one-sided: the primary holds no corroborated run of even twelve
words, its longest one-sided stretch is four or five, and the rescue holds a scripture
reading. Four of those readings, Colossians 1:11-12, 1 Peter 2:9-10, Ephesians 4:11 and
Hebrews 4:16, are on the list of eight passages `QUALITY_PROPOSAL.md` section 4.4 says
are missing from current transcripts. They are not missing because the model cannot hear
them. They were transcribed, and then discarded.

Two are close trades. `women_retreat_2025_session1` loses a 24 word aside and gains
26 words of Hebrews 4:16. `tcf.20260428` is 35 against 34 and the primary's run has the
better corroboration.

**One is a genuine save.** On `tcf.20241105` the primary holds 69, 24 and 19 word runs
the rescue lost and the rescue holds nothing at all above eight words. The guard was
right on that file. It is also the file with the second largest word deficit, so on this
recording the word count and the content agreed.

So the deficit is filler on six of nine, a trade on two, and real loss on one. The
original finding on `tcf.20250607` holds and generalises, but not universally.

## Step 3: the replacement guard is worse at every threshold

Refuse a rescue only when `P >= N`, dropping the 99 percent floor and the 40 word cap.

| N | decisions changed | rescues gained | rescues lost |
|---|---|---|---|
| 8 | 3 of 20 | **0** | 3 |
| 12 | 2 of 20 | **0** | 2 |
| 20 | 2 of 20 | **0** | 2 |
| 30 | 1 of 20 | **0** | 1 |

Zero gained at every threshold. Not one of the nine refusals publishes, including
`tcf.20250607`, the recording the whole question came from.

### Why nothing is gained

Because the retention guard is not the thing refusing them. Remove it and the ordering
refuses them again, for the same reason and on the same evidence.

`select_pass` orders eligible passes by `(anomaly score, -words, -mean logprob)` with the
primary first, so a tie on the anomaly score falls to whichever pass has **more words**.
Of the nine refusals, seven are tied on the anomaly score and lose on that `-words` term;
the other two, `tcf.20150424` and `tcf.20211203`, lose on the anomaly score itself before
words are consulted. The retention guard changed the outcome on none of the nine.

The plan document said the guard was "the last place a raw word count still makes a
decision". It is not. There are two word-count gates in series, and replacing only the
first leaves the second doing the same job. This does not reopen the tie-break question
as it was framed before, and that correction stands: the guard is a hard pre-filter, so
changing the tie-break alone flips nothing either. Both statements are true at once,
which is exactly the problem. Neither gate can be changed alone to any effect.

### What it costs

| file | P, and what the primary keeps | R, and what the rescue loses |
|---|---|---|
| tcf.20260626 | 49, Philippians 2:1-2 | 70, Philippians 2:5-6, the kenosis hymn |
| tcf.202606623b | 20, 2 Corinthians 4:5 | 70, Galatians 2:7 |
| tcf.20260724 | 9, a fragment of Philippians 4:5 | 42, Philippians 4:6-7 |

In all three the rescue's run is longer and reads as the fuller passage, so every flip is
a regression on content and not only on count. A rule that fixes nothing and breaks three
is not a fix.

### Two variants, for the record

`net_content` refuses only when `P >= N` **and** `P > R`, so a rescue that trades one
corroborated passage for a longer one is not punished. That removes all three
regressions and changes nothing else: **0 of 20 decisions at every N**. It is a strictly
safer rule than the proposal and than the shipped guard, and it is also a no-op, so it
buys nothing but a cleaner explanation.

`content_first` is `net_content` plus ranking corroborated contiguous content ahead of
raw words inside the ordering. That one does move: **5 rescues gained, 0 lost**, at every
N from 8 to 30, publishing `tcf.20221203`, `tcf.20240713`, `tcf.20250607`, `ucf20211106b`
and `women_retreat_2025_session1`, and correctly keeping the primary on `tcf.20241105`
and `tcf.20260428`. It is not being recommended here. It changes the selection ordering,
not the guard, it has never been run on the corpus, and nothing in this project has
earned the right to ship a selection rule that has not been measured across a hundred
files first. It is recorded so the next session knows where the lever actually is.

## Step 4: decision

**0.6.2 should not change the retention guard.** Specifically:

- Do not drop the 99 percent floor or the 40 word cap. They changed no outcome on this
  corpus, so removing them is not a risk, but replacing them with a contiguous-content
  threshold is: the proposal costs published scripture and recovers none.
- Do not adopt any N. There is no threshold at which the proposed rule gains anything.
- `RESCUE_TRANSCRIPT_DIR` is merged and is the one change this analysis produced that is
  worth having: a job that runs a rescue now keeps both transcripts, so this question is
  answerable from disk next time instead of needing the audio decoded again. It is off by
  default and production behaviour is unchanged.

Conservative refusal keeps the primary, which is the safe failure, and on `tcf.20241105`
it was the correct one.

What would need re-validating if the `content_first` direction is ever taken up: firing
rate, selection rate, genuine recoveries and any file losing more than five percent, on
the full hundred-file corpus, exactly as 0.6.1 was validated. That work has not been
done and this analysis is not a substitute for it.

## Caveats that travel with these numbers

- **The rescue is a different draw.** The primary is a deterministic beam search from
  temperature 0.0 and reproduced the corpus run to within two words on all three
  re-decodes. The rescue samples. On the same three files the fresh draw came out 67, 106
  and 42 words away from the corpus draw, and on `tcf.20240713` that was enough to flip
  the shipped decision from refuse to publish. `P` and `R` for six of the twenty files
  come from the fidelity experiment's `B.json`, which is a rescue-configuration decode but
  a different draw again. These figures describe the rescue in distribution, not the
  particular transcript that was thrown away.
- **The legacy archive corroborates, it does not prove.** It is machine output from four
  code eras. `classify` calls a run "real content" on a single matching six-gram, which is
  permissive: the 69 word run on `tcf.20241105` has n-gram coverage 0.27, against 1.00 for
  every scripture reading in the table above. Where the two disagree, the readings with
  full coverage are the ones to trust.
- The twenty-file replay uses the log's word counts and anomaly scores, which are the
  shipped run's own numbers, so the rule comparison is exact even where the transcript
  pairing is approximate.

## The seed question

**Seed it, derived from the job, not from a constant.** Set the sampler seed from a hash
of the GUID so every recording gets its own draw and every recording gets the same draw
every time. `ctranslate2.set_random_seed` is available in the 4.8.0 the image ships.

The argument against seeding assumes an unseeded draw buys diversity. It does not buy
anything for any individual sermon, because each sermon is decoded once and never again:
it gets exactly one draw either way, and unseeded only means nobody can say which one. A
constant seed would deserve the objection, since it fixes every recording to the same
corner of the sampler and could be systematically poor on some; a per-GUID seed does not,
because the draws stay as varied across the archive as they are now while each one becomes
reproducible.

The cost is real and should be paid rather than argued away. Seeding invalidates the seven
measured recoveries of 0.6.1, which were obtained unseeded, so it needs its own corpus run
to re-establish them. Set against that, this analysis found a 67 word swing between two
draws of the same rescue on `tcf.20240713` that flipped the published transcript, and the
anomaly-count oscillation has the same root. As things stand no quality rule in this
service can be validated, because the input to the rule is a coin flip. That is the more
expensive problem.
