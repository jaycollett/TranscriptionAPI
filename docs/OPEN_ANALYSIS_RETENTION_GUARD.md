# Open analysis: should the rescue retention guard count content instead of words

Status: in progress, 2026-09-10. Step 0 (count reconciliation) is done and is recorded
below. Steps 1 to 4 follow. Written so this can be finished by a session with no prior
context; everything needed to resume is below, and each section says whether it is
settled or still open.

Progress log, newest last:

- 2026-09-10 Counts reconciled. 20 rescues fired, 11 selected, **9 refused**, not six.
  The "six refused" table below was a subset and is corrected in place.
- 2026-09-10 Fidelity coverage checked. Six of the nine refused rescues already have
  both transcripts on disk. Only three need a re-decode.

## The question in one sentence

The rescue pass is refused when it has fewer words than the primary pass, but on the one
recording examined closely those missing words were almost entirely "um" and "uh", and the
rescue held a scripture reading the primary had dropped entirely, so the guard rejected the
better transcript. Does that hold across the other five refused rescues, and if so should the
guard count contiguous content rather than raw words?

## Why this matters

The service transcribes sermons. A dropped scripture reading is the worst failure it has. The
0.6.1 rescue pass exists to recover exactly that, and on at least one recording the guard
meant to protect the transcript is what prevented the recovery.

It also matters because it is the same mistake in a new place. Across the whole 2026-09-07
effort, word count was the wrong instrument every single time it was used. The largest
difference between any two configurations all day was 137 words and it was a hallucinated
loop repeating one phrase fifteen times, so the pass with more words had worse content. The
retention guard is the last place a raw word count still makes a decision.

## What is already established

On `tcf.20250607.mp3`, the refused rescue versus the published primary:

| measure | value |
|---|---|
| primary words | 7818 |
| rescue words | 7610 |
| legacy archive words | 7619 |
| runs of 12+ words the primary has and the rescue lacks | 0, longest is 7 |
| runs of 12+ words the rescue has and the primary lacks | 1, of 156 words |
| that run | Exodus 24 read aloud, 0.95 n-gram coverage against the archive |

The 340 word deficit breaks down as 207 single-word runs, 70 runs of two or three, and 11 runs
of four to eleven. Of the single words, "um" accounts for 83 and "uh" for 40. The legacy
archive word count sits beside the rescue, not the primary, so the primary is the outlier and
its surplus is the model faithfully transcribing hesitation.

Conclusion for that one recording: the rescue is the better transcript and a word-count rule
refused it. The tie-break was never the blocker; the retention guard was. Changing the
tie-break alone flips nothing.

That is settled in the code, not inferred. `select_pass` in `transcribe.py` builds an
eligibility list first and only sorts what survives it, so `retains_enough_words` is a hard
pre-filter and a candidate that fails it never reaches the tie-break at all. On this recording
the rescue is 340 words down against a 40 word cap, so it fails eligibility by a factor of
eight and is discarded before any tie is considered. One earlier report attributed the
discard to the tie-break and recommended changing it; that recommendation is a dead end and
should not be revisited. The guard is the thing to change, or nothing is.

Full detail is on devmachine at `/home/jay/sweep/rescue_deficit.md` and `.json`.

## The nine refused rescues

Settled 2026-09-10. `/home/jay/sweep/rc061.log` is authoritative and the two lists are
now reconciled: **102 files ran, 20 fired a rescue, 11 selected it, 9 refused it.** The
earlier "six refused, six selected" table was a subset of twelve `tcf.*` files, not the
corpus. `/home/jay/sweep/rc061/selected_list.json` holds exactly the eleven selected
files, which is where the eleven came from; it does not list the refusals at all, so it
cannot be used to enumerate them. The full twenty-row table in
`docs/analysis-rescue-deficit-tcf20250607.md` agrees with the log line for line, and
there are nine `Discarding the rescue pass:` lines in the log, one per refusal.

`lost` is primary words minus rescue words; `gaps` is largest uncovered gap, primary
versus rescue. `on disk` says whether both transcripts already exist in the 32-file
fidelity set, which decides whether a re-decode is needed.

| file | primary | rescue | lost | anomaly | gaps | on disk | analysed |
|---|---|---|---|---|---|---|---|
| tcf.20150424 | 7060 | 6981 | 79 | 0 v 1 | 12.0 v 27.6 | yes | no |
| tcf.20211203 | 9996 | 9700 | 296 | 0 v 1 | 16.5 v 5.7 | yes | no |
| tcf.20221203 | 9695 | 9575 | 120 | 0 v 0 | 8.8 v 8.5 | yes | no |
| tcf.20240713 | 6531 | 6483 | 48 | 0 v 0 | 13.4 v 11.0 | no | no |
| tcf.20241105 | 7786 | 7442 | 344 | 0 v 0 | 11.9 v 8.6 | no | no |
| tcf.20250607 | 7815 | 7475 | 340 | 1 v 1 | 24.9 v 12.1 | yes | **yes** |
| tcf.20260428 | 13470 | 13321 | 149 | 0 v 0 | 10.5 v 2.7 | no | no |
| ucf20211106b | 5050 | 4984 | 66 | 1 v 1 | 20.6 v 19.2 | yes | no |
| women_retreat_2025_session1 | 2245 | 2217 | 28 | 0 v 0 | 8.5 v 1.5 | yes | no |

The last three were missing from the earlier list. `women_retreat_2025_session1` is worth
noting on its own: it loses 28 words, which is inside the 40 word cap, so the 99 percent
floor alone refused it. It is the only refusal the cap would have let through.

The eleven rescues that were selected and published, for the regression check:
`tcf.20210210`, `tcf.20210217`, `tcf.20210604`, `tcf.20240326b`, `tcf.20240412`,
`tcf.20240621`, `tcf.20260626`, `tcf.202606623b`, `tcf.20260717`, `tcf.20260724`,
`women_retreat_2026_session1`.

## Why it is blocked

Two blockers, both found on 2026-09-10:

1. **The analysis tool takes a directory, not a file.** `rescue_deficit.py` expects
   `--fidelity` to be a directory containing `A.json`. Passing a JSON path fails with
   `NotADirectoryError`. The fidelity directory is `/home/jay/sweep/fidelity/`.
2. **The corpus run stored metrics but not transcripts.** `/home/jay/sweep/rc061/` holds only
   `candidate_pass.json`, `primary_only.json` and `selected_list.json`, which carry word
   counts, anomaly counts, gaps and rescue outcomes but no text. The deficit analysis needs
   both the primary and the rescue transcript for each file in order to diff them, and only
   `tcf.20250607` has them, which is why only it was analysed.

The 32-file fidelity set at `/home/jay/sweep/fidelity/` does have transcripts. Overlap
checked 2026-09-10: it covers **six of the nine refused rescues** (`tcf.20150424`,
`tcf.20211203`, `tcf.20221203`, `tcf.20250607`, `ucf20211106b`,
`women_retreat_2025_session1`) and five of the eleven selected ones (`tcf.20210210`,
`tcf.20210217`, `tcf.20260626`, `tcf.20260717`, `women_retreat_2026_session1`). Only
three refusals need a decode: `tcf.20240713`, `tcf.20241105`, `tcf.20260428`.

One caveat that has to travel with every fidelity-sourced result. `A.json` is 0.6.0 as
shipped, a beam search from temperature 0.0, deterministic, so it is the same primary the
0.6.1 run published and its word counts agree with the log to within normalisation.
`B.json` is a whole-file decode with the ladder starting at 0.2, which is what the rescue
does, but it is **a different unseeded draw**, not the refused draw. On `tcf.20250607` the
refused rescue was 7475 words and `B.json` is 7610. So a fidelity-sourced result answers
"does a rescue-configuration decode of this recording drop corroborated content the
primary holds", which is the question the guard needs answered in distribution, but it is
not literally the transcript that was thrown away. Results from a re-decode are marked as
such and are the matched pair.

## How to finish it

### Step 1, get the transcripts

For each of the five unanalysed files, obtain both the primary transcript and the rescue
transcript. Cheapest route is to re-decode those five recordings with transcripts retained.
On an idle card this is well under an hour. Decode only, no alignment needed, since the
question is purely about text.

While doing this, fix the underlying gap: the sweep tooling should retain both transcripts
whenever a rescue runs, so this question is answerable from disk next time. That is a small
change to the runner and worth making permanent.

### Step 2, run the deficit analysis

`rescue_deficit.py` already does exactly what is needed. Per file it produces the run-length
histogram, the single-word breakdown, the longest run on each side, and classification of each
significant run against the legacy archive. Run it for all five, then read the results
together with the existing one.

The classification that matters is: **does the primary hold any contiguous run of twelve or
more words, corroborated by the legacy archive, that the rescue lacks?** If not, the deficit is
filler and the rescue is safe to publish.

### Step 3, evaluate the replacement guard

Proposed rule: refuse a rescue only when it loses a contiguous run of N or more words that the
legacy archive corroborates. Drop the 99 percent word floor and the 40 word cap entirely.

Try N at 8, 12, 20 and 30. For each threshold, report which of the twelve rescues publish, and
for every flip against today's behaviour say whether it improves or regresses the transcript,
judged on content. A rule that fixes one recording and breaks three is not a fix.

Note that the legacy archive is itself machine output from older code, so it corroborates
rather than proves. Where it disagrees, prefer the reading that a human would recognise as a
sentence.

### Step 4, decide and validate

If a threshold is clearly right, implement it as 0.6.2 and validate on the full hundred-file
corpus the same way 0.6.1 was: firing rate, selection rate, genuine recoveries, and any file
losing more than five percent. If no threshold is clearly right, say so and leave the guard
alone; conservative refusal keeps the primary, which is the safe failure.

## Also open, same area

**Should the rescue sampler be seeded?** It samples unseeded, so selection differed on two of
eleven recordings between two runs of the same build, meaning recovery is an expectation
across the archive rather than a promise for any one sermon. Seeding makes it reproducible and
would also settle an anomaly-count oscillation with the same root cause. The argument against
is that it converts a random draw into a fixed one that may be systematically worse on some
recordings, and that the seven measured recoveries in 0.6.1 were obtained unseeded, so seeding
invalidates that measurement and needs its own corpus run.

**Eight scripture passages are missing from current transcripts.** The 2026-09-08 fidelity
experiment found that making the primary pass sample rather than beam-search recovers readings
from 1 Peter 2:9-10, Psalm 25:17-21, Isaiah 64:4, Colossians 1:11-12, Hebrews 4:16,
Ephesians 4:11, Philippians 2:13 and an Ephesians 5 paraphrase, none of which anyone knew were
missing. It was not adopted because it is one unseeded draw per recording, it moves segment
counts by up to fourfold under a release that aligns per segment, and the experiment was
decode-only so alignment was never exercised. Deferred, not rejected. See
`docs/QUALITY_PROPOSAL.md` section 4.4.

## Environment

- GPU host is devmachine, user `jay` at `192.168.0.228`. Password is `DEVMACHINE_SSH_PASSWORD`
  in `~/.config/homelab/credentials.env`; it contains shell metacharacters, so always route it
  through a temp file with `sshpass -f` and never interpolate it into a command line.
- **Production runs `transcription-api:0.6.1` on port 5030 and must not be touched, restarted
  or sent traffic.** Use a separate container, port, database, upload folder and aligner root
  under `/home/jay/sweep`. A seeded aligner root already exists at
  `/home/jay/sweep/service/mfa_root`; a private root must be seeded with the image's pretrained
  models or alignment silently fails while still reporting success.
- Audio archive: `/home/jay/SourceCode/SermonPreprocessorAPI/data/audiofiles/`, 655 files.
- Legacy transcripts: `/home/jay/sweep/legacy_text.json`, extracted read-only from the
  orchestrator database. They are machine output from four different code eras; see the
  2026-09-07 session knowledge entry.
- Analysis tooling: `tools/quality_sweep/` in this repo, and the working copies under
  `/home/jay/sweep/tools/`.
- Disk on devmachine runs tight. Prune release candidate images and the build cache after any
  build.

## Guard rails learned the hard way

Four separate quality rules were written on 2026-09-07 and every one would have destroyed good
content: boundary de-duplication deleted a scripture acclamation, the anomaly gate discarded a
healthy 8292 word transcript, the uncovered-speech check counted breath pauses as missing
audio, and the per-segment anomaly count flagged six healthy recordings while missing all seven
bad ones. Each was a punishment attached to a signal nobody had measured on real recordings
first.

Two rules follow. Never add a gate that can withhold or shorten a transcript without first
measuring the signal's distribution across the corpus. And when a rule and a transcript
disagree, suspect the rule.
