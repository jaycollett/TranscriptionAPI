# The four-configuration fidelity experiment

Decode-only records from a 32-file subset of the sweep corpus, read after the fact.
No alignment ran in any of these configurations, so nothing here measures the
per-utterance MFA stage that 0.6.0 introduced.

**Word error rates on this page are not accuracy.** The preacher's translation is
unknown and differs from every public-domain text, which inflates every rate. The
same reference scores every configuration, so a difference between two
configurations on one passage is meaningful; the level is not.

## Passage fidelity, word error rate against the WEB

| file | passage | ref words | A | B | C | D |
|---|---|---|---|---|---|---|
| tcf.20210210.mp3 | Matthew 12:15-21 | 108 | 0.407 | 0.407 | 0.407 | - |
| tcf.20210217.mp3 | Matthew 6:18-25 | 209 | 0.435 | 0.435 | 0.770 | - |
| tcf.20210611.mp3 | Romans 5:1-6 | 102 | 0.363 | 0.363 | - | 0.363 |
| tcf.20211203.mp3 | Hebrews 4:12-12 | 43 | 0.256 | 0.256 | - | - |
| tcf.20221203.mp3 | Numbers 3:8-8 | 26 | 0.385 | 0.385 | - | 0.385 |
| tcf.20240319a.mp3 | Galatians 2:20-20 | 46 | 0.348 | 0.348 | - | - |
| tcf.20250607.mp3 | Exodus 24:3-8 | 163 | 0.834 | 0.442 | 0.834 | - |
| tcf.20260331a.mp3 | Acts 2:38-38 | 33 | 0.303 | 0.303 | - | - |
| tcf.20260331a.mp3 | 1 Corinthians 8:6-6 | 33 | 0.273 | 0.273 | - | - |
| tcf.20260331a.mp3 | 1 Timothy 3:16-16 | 35 | 0.429 | 0.429 | - | - |
| tcf.20260331b.mp3 | Philippians 2:14-15 | 38 | 0.237 | 0.237 | - | - |
| tcf.20260626.mp3 | Colossians 3:16-16 | 32 | 0.406 | 0.406 | 0.406 | 0.406 |
| tcf.20260710.mp3 | Daniel 12:3-3 | 25 | 0.160 | 0.160 | - | 0.160 |
| ucf20211106b.mp3 | John 3:36-36 | 26 | 0.308 | 0.308 | 0.308 | 0.308 |
| ucf20211106b.mp3 | Ephesians 2:12-13 | 51 | 0.392 | 0.392 | 0.392 | 0.392 |
| ucf20211106b.mp3 | 1 Peter 1:22-24 | 71 | 0.676 | 0.437 | 0.676 | 0.437 |
| women_retreat_2025_session1.mp3 | Hebrews 4:16-16 | 27 | 0.296 | 0.259 | - | 0.296 |

### The same passages against every cached translation

| file | passage | translation | A | B | C | D |
|---|---|---|---|---|---|---|
| tcf.20210210.mp3 | Matthew 12:15-21 | web | 0.407 | 0.407 | 0.407 | - |
| tcf.20210210.mp3 | Matthew 12:15-21 | asv | 0.368 | 0.368 | 0.368 | - |
| tcf.20210210.mp3 | Matthew 12:15-21 | kjv | 0.387 | 0.387 | 0.387 | - |
| tcf.20210210.mp3 | Matthew 12:15-21 | akjv | 0.378 | 0.378 | 0.378 | - |
| tcf.20210210.mp3 | Matthew 12:15-21 | ylt | 0.552 | 0.552 | 0.552 | - |
| tcf.20210210.mp3 | Matthew 12:15-21 | wb | 0.383 | 0.383 | 0.383 | - |
| tcf.20210217.mp3 | Matthew 6:18-25 | web | 0.435 | 0.435 | 0.770 | - |
| tcf.20210217.mp3 | Matthew 6:18-25 | asv | 0.514 | 0.514 | 0.788 | - |
| tcf.20210217.mp3 | Matthew 6:18-25 | kjv | 0.498 | 0.498 | 0.787 | - |
| tcf.20210217.mp3 | Matthew 6:18-25 | akjv | 0.455 | 0.455 | 0.772 | - |
| tcf.20210217.mp3 | Matthew 6:18-25 | ylt | 0.598 | 0.598 | 0.817 | - |
| tcf.20210217.mp3 | Matthew 6:18-25 | wb | 0.481 | 0.481 | 0.783 | - |
| tcf.20210611.mp3 | Romans 5:1-6 | web | 0.363 | 0.363 | - | 0.363 |
| tcf.20210611.mp3 | Romans 5:1-6 | asv | 0.452 | 0.452 | - | 0.452 |
| tcf.20210611.mp3 | Romans 5:1-6 | kjv | 0.556 | 0.556 | - | 0.556 |
| tcf.20210611.mp3 | Romans 5:1-6 | akjv | 0.545 | 0.545 | - | 0.545 |
| tcf.20210611.mp3 | Romans 5:1-6 | ylt | 0.500 | 0.500 | - | 0.500 |
| tcf.20210611.mp3 | Romans 5:1-6 | wb | 0.510 | 0.510 | - | 0.510 |
| tcf.20211203.mp3 | Hebrews 4:12-12 | web | 0.256 | 0.256 | - | - |
| tcf.20221203.mp3 | Numbers 3:8-8 | web | 0.385 | 0.385 | - | 0.385 |
| tcf.20240319a.mp3 | Galatians 2:20-20 | web | 0.348 | 0.348 | - | - |
| tcf.20250607.mp3 | Exodus 24:3-8 | web | 0.834 | 0.442 | 0.834 | - |
| tcf.20250607.mp3 | Exodus 24:3-8 | asv | 0.819 | 0.404 | 0.819 | - |
| tcf.20250607.mp3 | Exodus 24:3-8 | kjv | 0.808 | 0.362 | 0.808 | - |
| tcf.20250607.mp3 | Exodus 24:3-8 | akjv | 0.808 | 0.333 | 0.808 | - |
| tcf.20250607.mp3 | Exodus 24:3-8 | ylt | 0.878 | 0.552 | 0.878 | - |
| tcf.20250607.mp3 | Exodus 24:3-8 | wb | 0.807 | 0.341 | 0.807 | - |
| tcf.20260331a.mp3 | Acts 2:38-38 | web | 0.303 | 0.303 | - | - |
| tcf.20260331a.mp3 | 1 Corinthians 8:6-6 | web | 0.273 | 0.273 | - | - |
| tcf.20260331a.mp3 | 1 Timothy 3:16-16 | web | 0.429 | 0.429 | - | - |
| tcf.20260331b.mp3 | Philippians 2:14-15 | web | 0.237 | 0.237 | - | - |
| tcf.20260626.mp3 | Colossians 3:16-16 | web | 0.406 | 0.406 | 0.406 | 0.406 |
| tcf.20260710.mp3 | Daniel 12:3-3 | web | 0.160 | 0.160 | - | 0.160 |
| ucf20211106b.mp3 | John 3:36-36 | web | 0.308 | 0.308 | 0.308 | 0.308 |
| ucf20211106b.mp3 | Ephesians 2:12-13 | web | 0.392 | 0.392 | 0.392 | 0.392 |
| ucf20211106b.mp3 | 1 Peter 1:22-24 | web | 0.676 | 0.437 | 0.676 | 0.437 |
| ucf20211106b.mp3 | 1 Peter 1:22-24 | asv | 0.721 | 0.485 | 0.721 | 0.485 |
| ucf20211106b.mp3 | 1 Peter 1:22-24 | kjv | 0.753 | 0.597 | 0.753 | 0.597 |
| ucf20211106b.mp3 | 1 Peter 1:22-24 | akjv | 0.727 | 0.584 | 0.727 | 0.584 |
| ucf20211106b.mp3 | 1 Peter 1:22-24 | ylt | 0.776 | 0.645 | 0.776 | 0.645 |
| ucf20211106b.mp3 | 1 Peter 1:22-24 | wb | 0.750 | 0.605 | 0.750 | 0.605 |
| women_retreat_2025_session1.mp3 | Hebrews 4:16-16 | web | 0.296 | 0.259 | - | 0.296 |

## Contiguous content, A against B

Every one-sided run of 12 or more words, and whether the legacy archive has it.

| side | words | file | at | in legacy | opening |
|---|---|---|---|---|---|
| B-only | 156 | tcf.20250607.mp3 | 3% | True | moses came and told the people all the words that the lord |
| A-only | 137 | tcf.20201003.mens_.mp3 | 97% | True | i am convinced the sins we commit but the sins we commit |
| B-only | 64 | ucf20211106b.mp3 | 54% | True | all right having purified your souls by your obedience to the truth |
| B-only | 59 | tcf.20211203.mp3 | 64% | True | but you're a chosen race a royal priesthood a holy nation a |
| A-only | 49 | tcf.20210514.mp3 | 48% | True | honor was universally regarded as the ultimate asset for human beings and |
| B-only | 44 | women_retreat_2026_session1.mp3 | 42% | True | distresses consider my affliction and my trouble and forgive all my sins |
| B-only | 42 | women_retreat_2026_session1.mp3 | 63% | True | from of old no one has heard or perceived by the ear |
| B-only | 40 | tcf.20150424.mp3 | 31% | True | may you be strengthened with all power according to his glorious might |
| A-only | 27 | tcf.20230603_mens_fast.mp3 | 1% | True | blessed is the man who walks not in the counsel of the |
| B-only | 26 | women_retreat_2025_session1.mp3 | 59% | True | in time of need let us then with confidence not groveling with |
| B-only | 26 | women_retreat_2026_session1.mp3 | 15% | True | semicolon orientation of our lives toward the father and his purposes trusting |
| B-only | 25 | tcf.20221203.mp3 | 92% | True | in ephesians chapter 4 he gave some as apostles some as prophets |
| B-only | 24 | tcf.20221203.mp3 | 57% | True | to two different directors i would hope they would talk to the |
| A-only | 24 | women_retreat_2025_session1.mp3 | 90% | True | have a cheerful or optimistic view of things often without a valid |
| A-only | 23 | women_retreat_2026_session1.mp3 | 82% | True | jesus and the holy spirit number two also there was waiting for |
| B-only | 19 | tcf.20221203.mp3 | 66% | True | always been kind of an annoying thing to me in the scriptures |
| B-only | 18 | tcf.20210514.mp3 | 15% | True | submit themselves to christ who submitted themselves to his father's authority wives |
| B-only | 18 | tcf.20260710.mp3 | 34% | True | for it is god who works in you both to will and |
| B-only | 17 | tcf.20150424.mp3 | 60% | True | people i don't want to do it for the pastor i don't |
| A-only | 13 | women_retreat_2025_session1.mp3 | 58% | True | often on his side but the path to victory in those cases |

## Detection

### Against the sweep bad label set (7 of 32 files)

| signal | bad median | bad range | healthy median | healthy range | 0 FP | 1 FP | 2 FP | 3 FP |
|---|---|---|---|---|---|---|---|---|
| uncovered_fraction | 0.07826 | 0.05296 to 0.13371 | 0.07604 | 0.01107 to 0.12828 | 1 @0.13371 | 1 @0.13371 | 1 @0.13371 | 1 @0.13371 |
| coverage | 0.92174 | 0.86629 to 0.94704 | 0.92396 | 0.87172 to 0.98893 | 1 @0.86629 | 1 @0.86629 | 1 @0.86629 | 1 @0.86629 |
| max_gap_s | 16.16 | 1.32 to 24.9 | 2.22 | 0.71 to 16.46 | 3 @20.6 | 5 @12 | 5 @12 | 5 @12 |
| anomaly_windows | 0 | 0 to 1 | 0 | 0 to 0 | 3 @1 | 3 @1 | 3 @1 | 3 @1 |
| flagged_segments | 1 | 0 to 2 | 1 | 0 to 5 | 0 | 0 | 0 | 0 |
| wps_speech | 2.689 | 2.418 to 3.036 | 2.787 | 2.007 to 3.148 | 0 | 0 | 2 @2.655 | 2 @2.655 |
| crosscheck_disagreement | 0.01086 | 0.00059 to 0.04796 | 0.00713 | 0 to 0.03347 | 1 @0.04796 | 1 @0.04796 | 2 @0.02102 | 2 @0.02102 |
| crosscheck_run | 40 | 1 to 156 | 3 | 0 to 59 | 2 @64 | 4 @40 | 4 @40 | 4 @40 |
| crosscheck_run_reverse | 4 | 1 to 23 | 4 | 0 to 137 | 0 | 0 | 0 | 0 |

### Against the confirmed omissions label set (7 of 32 files)

| signal | bad median | bad range | healthy median | healthy range | 0 FP | 1 FP | 2 FP | 3 FP |
|---|---|---|---|---|---|---|---|---|
| uncovered_fraction | 0.08783 | 0.05163 to 0.13371 | 0.07604 | 0.01107 to 0.12828 | 1 @0.13371 | 1 @0.13371 | 1 @0.13371 | 1 @0.13371 |
| coverage | 0.91217 | 0.86629 to 0.94837 | 0.92396 | 0.87172 to 0.98893 | 1 @0.86629 | 1 @0.86629 | 1 @0.86629 | 1 @0.86629 |
| max_gap_s | 16.16 | 8.54 to 24.9 | 2.14 | 0.71 to 22.87 | 1 @24.9 | 7 @8.54 | 7 @8.54 | 7 @8.54 |
| anomaly_windows | 0 | 0 to 1 | 0 | 0 to 1 | 0 | 2 @1 | 2 @1 | 2 @1 |
| flagged_segments | 1 | 1 to 3 | 0 | 0 to 5 | 0 | 0 | 0 | 0 |
| wps_speech | 2.708 | 2.079 to 3.036 | 2.787 | 2.007 to 3.148 | 0 | 2 @2.418 | 2 @2.418 | 2 @2.418 |
| crosscheck_disagreement | 0.02102 | 0.01086 to 0.04796 | 0.00432 | 0 to 0.01998 | 4 @0.02102 | 5 @0.01997 | 5 @0.01997 | 6 @0.01744 |
| crosscheck_run | 44 | 25 to 156 | 2 | 0 to 18 | 7 @25 | 7 @25 | 7 @25 | 7 @25 |
| crosscheck_run_reverse | 5 | 4 to 24 | 3 | 0 to 137 | 0 | 0 | 0 | 2 @23 |

### A gap filter as the first stage of a two-stage check

| largest gap at least | files selected | of | confirmed losses reached | of |
|---|---|---|---|---|
| 5 s | 10 | 32 | 7 | 7 |
| 8 s | 8 | 32 | 7 | 7 |
| 10 s | 6 | 32 | 5 | 7 |
| 12 s | 6 | 32 | 5 | 7 |
| 15 s | 5 | 32 | 4 | 7 |
| 20 s | 3 | 32 | 2 | 7 |
| 25 s | 0 | 32 | 0 | 7 |

## Run-to-run variation of the shipped configuration

32 files submitted twice: 23 byte-identical, 9 varied, 6 of the varying ones on a path where a pass sampled. Largest disagreement 0.03220, largest word delta 6.48 percent.

