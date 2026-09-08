# Seam de-duplication trims: overlap or abut

64 trims across 32 files removed 278 words; 30 trims emptied their segment entirely.

Verdicts at a +/- 0.3 s noise floor: abutting 31, overlapping 10, unknown 5, unresolved 18. The populations do not separate cleanly, so overlap in time is not on its own a reliable rule.

Against the corrected temporal rule: it would still trim 10, would decline 31, and 23 cannot be resolved inside the noise floor. Declining the non-overlapping ones restores 235 words.

Overlap is the end of segment N minus the pre-trim start of segment N+1. Positive means the two occurrences share time, which is what decoding the same audio twice looks like. Negative means they abut, which is what a speaker saying it twice looks like. Segment N's end is post-alignment and the start is pre-alignment, so results inside the noise floor are unresolved rather than decided.

Of the measured overlaps, 3 exceed twice the phrase's own duration, which a genuine double-decode cannot do, so those are measurement artifacts rather than evidence. 16 are within the physically possible range.

| file | at s | words | phrase | seg N end | overlap s | phrase s | plausible | verdict | emptied |
|---|---|---|---|---|---|---|---|---|---|
| cf.20220402.mens_breakfast_In_Christ.mp3 | 120.96 | 5 | in the way he's thinking | 121.88 | 0.92 | 1.786 | True | overlapping | no |
| tcf.20150424.mp3 | 463.41 | 5 | Does that remind you of | None | None | None | None | unknown | no |
| tcf.20150424.mp3 | 649.92 | 5 | I want to talk about | None | None | None | None | unknown | no |
| tcf.20150424.mp3 | 649.96 | 5 | I want to talk about | None | None | None | None | unknown | no |
| tcf.20150424.mp3 | 2297.68 | 4 | I want to love, | None | None | None | None | unknown | yes |
| tcf.20150424.mp3 | 2298.18 | 4 | I want to love. | None | None | None | None | unknown | yes |
| tcf.20200515b.mp3 | 1112.5 | 4 | Are you a slave | 1112.12 | -0.38 | 1.429 | None | abutting | no |
| tcf.20200612.mp3 | 452.85 | 4 | This is the time | 452.75 | -0.1 | 1.429 | None | unresolved | no |
| tcf.20201003.mens_.mp3 | 1486.63 | 4 | There's a ticking clock | 1485.36 | -1.27 | 1.429 | None | abutting | no |
| tcf.20210604.mp3 | 393.68 | 5 | This part of my life | 393.46 | -0.22 | 1.786 | None | unresolved | no |
| tcf.20210604.mp3 | 768.06 | 4 | Repentance moves beyond conviction. | 766.64 | -1.42 | 1.429 | None | abutting | yes |
| tcf.20210604.mp3 | 878.51 | 4 | What shall we do? | 879.95 | 1.44 | 1.429 | True | overlapping | yes |
| tcf.20210604.mp3 | 1303.36 | 5 | Exactly what has to change. | 1302.82 | -0.54 | 1.786 | None | abutting | yes |
| tcf.20210611.mp3 | 464.11 | 4 | where the difference lies | 463.62 | -0.49 | 1.429 | None | abutting | no |
| tcf.20210611.mp3 | 649.07 | 4 | Repentance insists on reality. | 647.71 | -1.36 | 1.429 | None | abutting | yes |
| tcf.20211204.mp3 | 1012.06 | 4 | Facts are stubborn things | 1010.93 | -1.13 | 1.429 | None | abutting | no |
| tcf.20211204.mp3 | 2131.93 | 4 | Better than Chuck Norris, | 2130.88 | -1.05 | 1.429 | None | abutting | no |
| tcf.20220121.mp3 | 625.32 | 4 | We don't like limits | 625.21 | -0.11 | 1.429 | None | unresolved | no |
| tcf.20220121.mp3 | 773.79 | 4 | the heart of Sabbath | 773.94 | 0.15 | 1.429 | True | unresolved | no |
| tcf.20220121.mp3 | 1270.58 | 5 | need a little bit more. | 1270.23 | -0.35 | 1.786 | None | abutting | no |
| tcf.20220205.mp3 | 1128.04 | 4 | I for an eye, | 1126.72 | -1.32 | 1.429 | None | abutting | yes |
| tcf.20230603_mens_fast.mp3 | 459.19 | 4 | The blessed man is | 459.24 | 0.05 | 1.429 | True | unresolved | no |
| tcf.20230603_mens_fast.mp3 | 587.31 | 4 | You know every word | 587.23 | -0.08 | 1.429 | None | unresolved | no |
| tcf.20230603_mens_fast.mp3 | 678.61 | 4 | The king of Israel | 678.64 | 0.03 | 1.429 | True | unresolved | no |
| tcf.20231006.mp3 | 2738.44 | 4 | I know that guy. | 2750.17 | 11.73 | 1.429 | False | overlapping | no |
| tcf.20231229.mp3 | 1091.42 | 5 | For the people of God, | 1090.76 | -0.66 | 1.786 | None | abutting | no |
| tcf.20240116b.mp3 | 147.86 | 4 | TCF has a vessel. | 144.15 | -3.71 | 1.429 | None | abutting | no |
| tcf.20240130b.mp3 | 98.74 | 5 | if you were to ask | 98.23 | -0.51 | 1.786 | None | abutting | no |
| tcf.20240319a.mp3 | 272.32 | 5 | Don't tell anybody about this. | 272.32 | 0.0 | 1.786 | True | unresolved | yes |
| tcf.20240326b.mp3 | 18.74 | 4 | It might become that. | 17.03 | -1.71 | 1.429 | None | abutting | yes |
| tcf.20240412.mp3 | 568.47 | 4 | I was a jerk. | 567.82 | -0.65 | 1.429 | None | abutting | yes |
| tcf.20240412.mp3 | 710.51 | 4 | Encourage the faint-hearted. | 708.85 | -1.66 | 1.429 | None | abutting | yes |
| tcf.20240412.mp3 | 853.0 | 5 | I can do these things. | 852.32 | -0.68 | 1.786 | None | abutting | yes |
| tcf.20240412.mp3 | 1261.68 | 5 | He gave them a job. | 1261.08 | -0.6 | 1.786 | None | abutting | yes |
| tcf.20240416_Formation_Class_Holy_Spirit.mp3 | 952.0 | 5 | The spirit has perfect memory | 951.29 | -0.71 | 1.786 | None | abutting | no |
| tcf.20240713.mp3 | 2015.09 | 4 | out in the wild | 2015.03 | -0.06 | 1.429 | None | unresolved | yes |
| tcf.20241129.mp3 | 802.82 | 5 | Saying thank you to God. | 803.81 | 0.99 | 1.786 | True | overlapping | yes |
| tcf.20241129.mp3 | 1547.62 | 4 | The Lord is God. | 1546.44 | -1.18 | 1.429 | None | abutting | yes |
| tcf.20241129.mp3 | 1768.84 | 4 | Yahweh, he is God. | 1772.87 | 4.03 | 1.429 | False | overlapping | yes |
| tcf.20241129.mp3 | 1772.84 | 5 | The Lord, he is God. | 1772.87 | 0.03 | 1.786 | True | unresolved | yes |
| tcf.20241129.mp3 | 2963.63 | 4 | They're really into this | 2962.71 | -0.92 | 1.429 | None | abutting | no |
| tcf.20241129.mp3 | 3021.81 | 4 | this is not a | 3021.05 | -0.76 | 1.429 | None | abutting | no |
| tcf.20250426.mp3 | 391.45 | 4 | in a sinful world. | 391.51 | 0.06 | 1.429 | True | unresolved | yes |
| tcf.20250426.mp3 | 1490.54 | 4 | you know the time | 1491.39 | 0.85 | 1.429 | True | overlapping | yes |
| tcf.20250725.mp3 | 196.74 | 5 | It is a total gift | 196.63 | -0.11 | 1.786 | None | unresolved | no |
| tcf.20250725.mp3 | 1525.46 | 5 | to see what it is | 1525.61 | 0.15 | 1.786 | True | unresolved | yes |
| tcf.20251212.mp3 | 675.7 | 4 | He has to say | 675.51 | -0.19 | 1.429 | None | unresolved | no |
| tcf.20260331b.mp3 | 42.56 | 4 | That is not true. | 42.57 | 0.01 | 1.429 | True | unresolved | no |
| tcf.20260626.mp3 | 885.07 | 4 | you can't be all | 885.17 | 0.1 | 1.429 | True | unresolved | no |
| tcf.20260717.mp3 | 268.26 | 5 | None of that counts anymore | 267.85 | -0.41 | 1.786 | None | abutting | no |
| tcf.20260717.mp3 | 478.24 | 5 | They didn't count for anything. | 477.34 | -0.9 | 1.786 | None | abutting | yes |
| tcf.20260717.mp3 | 1086.71 | 4 | Write down your resume. | 1085.48 | -1.23 | 1.429 | None | abutting | yes |
| tcf.20260717.mp3 | 1171.23 | 5 | Write down your anti-resume. | 1174.97 | 3.74 | 1.786 | False | overlapping | yes |
| tcf.20260724.mp3 | 326.65 | 5 | Rejoice in the Lord always. | 326.04 | -0.61 | 1.786 | None | abutting | yes |
| tcf.20260724.mp3 | 714.79 | 4 | All is really well. | 713.67 | -1.12 | 1.429 | None | abutting | yes |
| tcf.20260724.mp3 | 1439.32 | 4 | It's God who provides. | 1440.45 | 1.13 | 1.429 | True | overlapping | yes |
| tcf.20260904.mp3 | 360.94 | 4 | God builds a house. | 360.39 | -0.55 | 1.429 | None | abutting | yes |
| tcf.20260905.mp3 | 884.08 | 4 | When we're more vulnerable | 883.05 | -1.03 | 1.429 | None | abutting | no |
| tcf.20260905.mp3 | 959.26 | 4 | No one will know | 958.32 | -0.94 | 1.429 | None | abutting | no |
| ucf.20211106c.mp3 | 193.63 | 5 | It's good to have that | 193.56 | -0.07 | 1.786 | None | unresolved | no |
| ucf.20211106c.mp3 | 483.82 | 4 | Exodus 20, verse 20. | 485.41 | 1.59 | 1.429 | True | overlapping | yes |
| ucf.20211106c.mp3 | 2232.22 | 4 | What can we do | 2231.06 | -1.16 | 1.429 | None | abutting | no |
| ucf.20211106c.mp3 | 2643.77 | 4 | I know a guy. | 2644.68 | 0.91 | 1.429 | True | overlapping | yes |
| women_retreat_2026_session3.mp3 | 835.47 | 4 | We can't do that | 835.35 | -0.12 | 1.429 | None | unresolved | no |

