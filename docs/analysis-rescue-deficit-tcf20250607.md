# The discarded rescue on tcf.20250607.mp3

Word counts are the question, not the answer: primary 7818, rescue 7610, legacy archive 7619.

## Where the word difference lives

Every one-sided stretch between the two decodes, bucketed by its length. Only the
excess counts, so a five-for-three replacement is two words and not five.

| run length | primary-only runs | primary-only words | rescue-only runs | rescue-only words |
|---|---|---|---|---|
| 1 | 207 | 207 | 14 | 14 |
| 2-3 | 70 | 147 | 6 | 6 |
| 4-11 | 11 | 44 | 3 | 16 |
| 12+ | 0 | 0 | 1 | 154 |

### The one-word-at-a-time part of it

The word the primary has and the rescue does not, wherever the difference is a
single word. Counts over the whole recording.

| word | times | word | times | word | times |
|---|---|---|---|---|---|
| um | 83 | uh | 40 | i | 12 |
| and | 11 | yeah | 8 | the | 8 |
| right | 4 | it's | 4 | that | 3 |
| it | 3 | like | 3 | is | 3 |
| in | 2 | there's | 2 | somebody | 2 |
| of | 2 | sure | 2 | a | 2 |
| but | 2 | if | 1 | gotcha | 1 |
| i'm | 1 | most | 1 | really | 1 |
| that's | 1 |  |  |  |  |

## What the primary has and the rescue lacks

No contiguous run of 12 or more words. The longest is 7.

## What the rescue has and the primary lacks

| words | at | class | legacy n-gram coverage | text |
|---|---|---|---|---|
| 156 | 1:41 | real content | 0.95 | moses came and told the people all the words that the lord of the lord and all the rules and all the people answered with one voice and said all the words that  |

## The longest of what is left, below the threshold

| side | words | at | class | legacy | text |
|---|---|---|---|---|---|
| rescue-only | 8 | 18:20 | real content | yes | shannon gets on to me all the time |
| rescue-only | 7 | 27:51 | real content | yes | do you have anything to say about |
| primary-only | 7 | 39:18 | ambiguous | no | all you know we should we should |
| primary-only | 6 | 37:46 | ambiguous | no | uh i think they've got it |
| primary-only | 5 | 33:26 | ambiguous | no | is it it's a uh |
| rescue-only | 5 | 35:08 | ambiguous | no | is it going to be |
| primary-only | 4 | 11:37 | real content | yes | i don't want to |
| primary-only | 4 | 17:53 | ambiguous | no | um over the reading |
| primary-only | 4 | 21:43 | ambiguous | no | in his he turned |
| primary-only | 4 | 24:24 | ambiguous | no | uh it you know |
| primary-only | 4 | 25:15 | real content | yes | gets you focused yep |
| primary-only | 4 | 28:19 | ambiguous | no | um i am reading |
| primary-only | 4 | 29:57 | ambiguous | no | or evangelical yeah evangelical |
| primary-only | 4 | 34:22 | real content | yes | a succinct statement of |

## Every rescue in the corpus, under four selection rules

| file | primary | rescue | lost | retention | scores | gaps | shipped | gap tie-break | gap tie-break, no retention | gap first |
|---|---|---|---|---|---|---|---|---|---|---|
| tcf.20150424.mp3 | 7060 | 6981 | 79 | fails | 0 v 1 | 12.0 v 27.6 | primary | primary | primary | primary |
| tcf.20210210.mp3 | 1172 | 1247 | -75 | ok | 1 v 0 | 26.0 v 1.3 | rescue | rescue | rescue | rescue |
| tcf.20210217.mp3 | 1591 | 1683 | -92 | ok | 1 v 1 | 23.0 v 22.9 | rescue | rescue | rescue | rescue |
| tcf.20210604.mp3 | 4497 | 4545 | -48 | ok | 0 v 0 | 10.4 v 1.6 | rescue | rescue | rescue | rescue |
| tcf.20211203.mp3 | 9996 | 9700 | 296 | fails | 0 v 1 | 16.5 v 5.7 | primary | primary | primary | primary |
| tcf.20221203.mp3 | 9695 | 9575 | 120 | fails | 0 v 0 | 8.8 v 8.5 | primary | primary | rescue | primary |
| tcf.20240326b.mp3 | 2758 | 2837 | -79 | ok | 1 v 0 | 26.2 v 2.0 | rescue | rescue | rescue | rescue |
| tcf.20240412.mp3 | 3357 | 3391 | -34 | ok | 0 v 0 | 8.2 v 1.4 | rescue | rescue | rescue | rescue |
| tcf.20240621.mp3 | 4417 | 4562 | -145 | ok | 5 v 0 | 27.3 v 1.2 | rescue | rescue | rescue | rescue |
| tcf.20240713.mp3 | 6531 | 6483 | 48 | fails | 0 v 0 | 13.4 v 11.0 | primary | primary | rescue | primary |
| tcf.20241105.mp3 | 7786 | 7442 | 344 | fails | 0 v 0 | 11.9 v 8.6 | primary | primary | rescue | primary |
| tcf.20250607.mp3 | 7815 | 7475 | 340 | fails | 1 v 1 | 24.9 v 12.1 | primary | primary | rescue | primary |
| tcf.20260428.mp3 | 13470 | 13321 | 149 | fails | 0 v 0 | 10.5 v 2.7 | primary | primary | rescue | primary |
| tcf.20260626.mp3 | 3735 | 3753 | -18 | ok | 1 v 1 | 25.7 v 21.3 | rescue | rescue | rescue | rescue |
| tcf.202606623b.mp3 | 1306 | 1351 | -45 | ok | 1 v 0 | 23.4 v 9.8 | rescue | rescue | rescue | rescue |
| tcf.20260717.mp3 | 3700 | 3767 | -67 | ok | 1 v 0 | 22.5 v 1.1 | rescue | rescue | rescue | rescue |
| tcf.20260724.mp3 | 4425 | 4458 | -33 | ok | 0 v 0 | 17.3 v 16.9 | rescue | rescue | rescue | rescue |
| ucf20211106b.mp3 | 5050 | 4984 | 66 | fails | 1 v 1 | 20.6 v 19.2 | primary | primary | rescue | primary |
| women_retreat_2025_session1.mp3 | 2245 | 2217 | 28 | fails | 0 v 0 | 8.5 v 1.5 | primary | primary | rescue | primary |
| women_retreat_2026_session1.mp3 | 7886 | 8017 | -131 | ok | 2 v 0 | 21.1 v 2.9 | rescue | rescue | rescue | rescue |

- `gap_tiebreak` changes 0 of 20 decisions
- `gap_tiebreak_no_retention` changes 7 of 20 decisions: tcf.20221203.mp3, tcf.20240713.mp3, tcf.20241105.mp3, tcf.20250607.mp3, tcf.20260428.mp3, ucf20211106b.mp3, women_retreat_2025_session1.mp3
- `gap_first` changes 0 of 20 decisions

## The eleven selected rescues, primary-only against published

| file | primary-only | published | gap before | gap after | rescue-only runs | primary-only runs |
|---|---|---|---|---|---|---|
| tcf.20210210.mp3 | 1173 | 1248 | 25.98 | 1.32 | 73 real content | none |
| tcf.20210217.mp3 | 1595 | 1687 | 22.97 | 22.87 | 91 real content | none |
| tcf.20210604.mp3 | 4505 | 4558 | 10.38 | 1.56 | 56 real content | none |
| tcf.20240326b.mp3 | 2757 | 2843 | 26.22 | 2.02 | 74 ambiguous | none |
| tcf.20240412.mp3 | 3371 | 3406 | 8.23 | 1.41 | 14 real content, 24 real content | none |
| tcf.20240621.mp3 | 4422 | 4567 | 27.26 | 1.16 | 58 real content, 86 real content | none |
| tcf.20260626.mp3 | 3754 | 3771 | 25.68 | 21.32 | 70 real content | 49 real content |
| tcf.202606623b.mp3 | 1307 | 1352 | 23.36 | 9.84 | 70 real content | 20 real content |
| tcf.20260717.mp3 | 3727 | 3791 | 22.49 | 1.14 | 66 real content | none |
| tcf.20260724.mp3 | 4432 | 4466 | 17.28 | 16.95 | 42 real content | none |
| women_retreat_2026_session1.mp3 | 7891 | 8022 | 21.11 | 2.88 | 59 real content, 88 real content, 16 real content | none |

