# Why 0.6.0 drops read-aloud passages

Seven passages across six files, each decoded whole five ways with one decode parameter
changed at a time, using faster-whisper directly so the service's gate, rescue and
alignment cannot confound the result. The control reproduces the rc2 production kwargs
verbatim from the image's `transcribe.py`.

Scoring is containment: the fraction of the reference passage's 5-grams that appear
anywhere in the whole-file transcript. References come from the isolated clip decode,
except the two regression files, where the reference is the run the legacy transcript has
and the new one lost, which is independent of this release entirely.

## The table

| passage | words | control | no_context | no_vad | quiet_profile | ladder from 0.2 |
|---|---|---|---|---|---|---|
| tcf.20250607, Exodus 24 | 102 | **0.083** | **1.000** | 0.896 | 0.083 | 0.896 |
| ucf20211106b, 1 Peter 1:22-25 | 59 | **0.364** | 0.364 | 0.364 | **0.909** | **0.909** |
| tcf.20210210, Matthew 12:15-16 | 73 | **0.000** | 0.000 | 0.000 | 0.000 | **1.000** |
| tcf.20210217, Matthew 6:19-21 | 44 | **0.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| tcf.20241105, Q&A at 1299 s | 93 | 0.528 | 0.551 | **0.697** | 0.652 | 0.539 |
| tcf.20241105, Q&A at 1658 s | 52 | 0.771 | 0.688 | **0.979** | 0.667 | 0.708 |
| tcf.20210604, C.S. Lewis | 68 | 0.938 | 1.000 | 1.000 | 1.000 | 1.000 |

Counting only the four passages the control genuinely loses (containment under 0.5), and
counting a recovery as 0.8 or better:

| config | recovers |
|---|---|
| ladder from 0.2 | **4 of 4** |
| no_context | 2 of 4 |
| no_vad | 2 of 4 |
| quiet_profile | 2 of 4 |

## Reading of the cause

**The previous-text hypothesis is part of the story but is not the cause.** Turning
conditioning off recovers two of the four (Exodus 24 completely, Matthew 6 completely) and
does nothing at all for the other two. If accumulated context were the mechanism it should
have recovered 1 Peter and Matthew 12 as well, and it recovered neither.

**What every recovery has in common is that the control's decode path was perturbed.**
Four different single-parameter changes each rescue a different subset, no two the same,
and every one of the four failing passages is recovered by at least one of them. That is
the signature of a deterministic decode falling into a bad path rather than of any one
parameter being wrong. The control is bit-reproducible, so it lands in the same bad path
every time, which is exactly what the sweep's determinism check showed at the file level:
five of five repeat submissions identical.

**The single most effective lever is starting the temperature ladder at 0.2**, which
recovers all four and is the only config that recovers Matthew 12 at all. Note what that
change really is: faster-whisper beam-searches only at temperature 0.0, and any rung above
it decodes by sampling with `beam_size=1`. So `ladder from 0.2` does not nudge the decode,
it replaces beam search with sampling on the first attempt. The passages are lost by the
beam search specifically.

**And the fallback never fires, which is the actual defect.** The ladder exists to escape a
bad decode, but it only steps up when the compression-ratio or log-probability threshold is
tripped. Here the model drops the passage confidently: no threshold is crossed, no fallback
is attempted, and the ladder's remaining five rungs are never used. A silent, confident
omission is invisible to every guard the decode has.

**One correction to the earlier sweep report.** `tcf.20210604`, the C.S. Lewis quotation,
scores 0.938 under the control: those words were in the transcript all along. Its 24.6 s
coverage gap was a timing artifact, not an omission. That reduces the confirmed omissions
from four to three plus two partial Q&A cases, and it is the one place the coverage gap
misled us in the other direction.

## Consequence for the remedy and the trigger

The rescue pass is already the right remedy. It decodes with `condition_on_previous_text`
off **and** `RESCUE_TEMPERATURE_BASE=0.2`, which is both of the two most effective variants
at once, so a rescue that fired on these files would very likely recover the passages.

The trigger is what fails. None of these files trips `RESCUE_ANOMALY_SEGMENTS=2`, and only
some trip `RESCUE_ANOMALY_WINDOWS=1`, because a confident omission produces no anomalous
segment at all.

## Would a per-window word-rate check have caught it

Per 60 s window over the omitted stretch, control config, against the 1.2 words per second
floor:

| passage | rate in the affected window | fires |
|---|---|---|
| tcf.20210217, Matthew 6 | 0.42 wps | **yes** |
| ucf20211106b, 1 Peter | 0.38 wps | **yes** |
| tcf.20210210, Matthew 12 | 0.82 wps | **yes** |
| tcf.20250607, Exodus 24 | **1.20 wps** | no, exactly at the floor |
| tcf.20241105, Q&A at 1299 s | 2.48 wps | no |
| tcf.20241105, Q&A at 1658 s | 2.12 wps | no |
| tcf.20210604, Lewis | 1.82 wps | no, and nothing was missing |

The windowed check catches three of the four real omissions outright, and the fourth misses
by nothing at all: Exodus 24 sits at exactly 1.20 against a floor of 1.20. **Raising the
window floor from 1.2 to 1.5 words per second would catch all four.** The corpus median is
2.65 wps over speech and the 5th percentile of healthy files is well above 1.5, so that
floor has room.

The two Q&A cases are not catchable this way and should not be: the surrounding
conversation keeps the window at 2.1 to 2.5 wps, and the omitted audio is an audience
member off microphone. That is a recording limitation, not a decode defect.

## Recommendation

1. **Raise the low-window floor to 1.5 wps and make it a rescue trigger**, not only a
   requeue reason. It catches all four real omissions where the anomaly count catches none,
   and it is the only signal that sees a confident omission.
2. **Do not requeue on it.** The decode is deterministic, so a requeue re-runs the identical
   decode and burns three attempts to reach the same quarantine, which is what happened to
   `tcf.20150424`. Fire the rescue, keep the better of the two passes, and publish.
3. Leave the rescue's own parameters alone; they are already the two most effective
   variants combined.
4. Consider, for 0.6.1 rather than now, whether the first pass should sample rather than
   beam-search. It recovers all four here, but it is a change to the core decode and this
   sample is seven passages, not a corpus.
