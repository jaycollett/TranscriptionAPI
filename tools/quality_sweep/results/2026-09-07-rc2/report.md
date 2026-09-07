# TranscriptionAPI 0.6.0 validation sweep

102 files attempted. Legacy words-per-second split: method `largest_gap_in_lower_quartile`, threshold 2.336, 10 below and 89 at or above.

Wall-clock timings in this run share GPU 0 with production and are not benchmark quality.

## Populations

| population | files | completed | comparable | word delta median | delta pct median | delta pct restored | legacy wps median | new wps median | new wps over speech | regressions | regr restored |
|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 102 | 101 | 99 | 4.0 | 0.0018 | 0.0024 | 2.649 | 2.648 | 2.806 | 2 | 2 |
| era_E1_0.1.5 | 51 | 50 | 50 | 0.0 | 0.0000 | 0.0003 | 2.685 | 2.696 | 2.804 | 2 | 2 |
| era_E2_0.2.x | 18 | 18 | 18 | 22.0 | 0.0050 | 0.0050 | 2.628 | 2.643 | 2.776 | 0 | 0 |
| era_E3_0.3.x | 21 | 21 | 21 | 8.0 | 0.0022 | 0.0022 | 2.609 | 2.606 | 2.775 | 0 | 0 |
| era_E4_0.5.x | 10 | 10 | 10 | 19.0 | 0.0030 | 0.0037 | 2.661 | 2.718 | 2.867 | 0 | 0 |
| era_unknown | 2 | 2 | 0 | - | - | - | - | - | 2.810 | 0 | 0 |
| legacy_low_rate | 10 | 10 | 10 | 55.0 | 0.0146 | 0.0146 | 2.182 | 2.237 | 2.708 | 0 | 0 |
| legacy_healthy | 90 | 89 | 89 | 4.0 | 0.0016 | 0.0022 | 2.663 | 2.674 | 2.807 | 2 | 2 |
| known_collapse_set | 12 | 12 | 10 | 19.0 | 0.0030 | 0.0037 | 2.661 | 2.718 | 2.863 | 0 | 0 |
| multi_voice | 12 | 12 | 12 | 29.0 | 0.0075 | 0.0085 | 2.665 | 2.686 | 2.858 | 0 | 0 |

## Service behaviour

| population | outcomes | rescue attempted | rescue selected | mfa applied | attempts | anomaly counts | files flagged |
|---|---|---|---|---|---|---|---|
| overall | completed=101, error=1 | 0.050 | 0.040 | 1.000 | 0:101 | 0:99, 1:2 | 43 |
| era_E1_0.1.5 | completed=50, error=1 | 0.040 | 0.040 | 1.000 | 0:50 | 0:50 | 19 |
| era_E2_0.2.x | completed=18 | 0.111 | 0.111 | 1.000 | 0:18 | 0:17, 1:1 | 7 |
| era_E3_0.3.x | completed=21 | 0.000 | 0.000 | 1.000 | 0:21 | 0:21 | 7 |
| era_E4_0.5.x | completed=10 | 0.100 | 0.000 | 1.000 | 0:10 | 0:9, 1:1 | 9 |
| era_unknown | completed=2 | 0.000 | 0.000 | 1.000 | 0:2 | 0:2 | 1 |
| legacy_low_rate | completed=10 | 0.200 | 0.200 | 1.000 | 0:10 | 0:10 | 6 |
| legacy_healthy | completed=89, error=1 | 0.034 | 0.022 | 1.000 | 0:89 | 0:87, 1:2 | 36 |
| known_collapse_set | completed=12 | 0.083 | 0.000 | 1.000 | 0:12 | 0:11, 1:1 | 10 |
| multi_voice | completed=12 | 0.083 | 0.083 | 1.000 | 0:12 | 0:12 | 5 |

## Level against the VAD cutover

The 0.6.0 rule takes VAD threshold 0.5 at or above -26.0 dBFS and 0.35 below it. Rows below the line are the 0.35 branch as shipped.

| dBFS | profile | files | done | words | legacy wps | new wps | delta pct | segments | seg mean s | coverage | anomaly mean | regr | flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -31 to -30 | quiet | 1 | 1 | 3397 | 2.666 | 2.677 | 0.0041 | 584 | 1.80 | 0.853 | 0.000 | 0 | 0 |
| -29 to -28 | quiet | 1 | 1 | 8916 | 2.500 | 2.632 | 0.0529 | 435 | 7.58 | 1.021 | 0.000 | 0 | 1 |
| -28 to -27 | quiet | 7 | 7 | 27395 | 2.604 | 2.582 | 0.0004 | 235 | 4.29 | 0.911 | 0.000 | 0 | 2 |
| -27 to -26 | quiet | 11 | 11 | 32843 | 2.738 | 2.742 | 0.0022 | 138 | 4.19 | 0.966 | 0.091 | 1 | 3 |
| -26 to -25 | loud | 4 | 4 | 12169 | 2.778 | 2.767 | -0.0035 | 160 | 5.02 | 0.936 | 0.000 | 0 | 2 |
| -25 to -24 | loud | 12 | 12 | 59308 | 2.660 | 2.676 | 0.0020 | 307 | 6.39 | 1.011 | 0.000 | 0 | 5 |
| -24 to -23 | loud | 7 | 7 | 30167 | 2.649 | 2.613 | 0.0010 | 296 | 3.52 | 0.917 | 0.000 | 0 | 4 |
| -23 to -22 | loud | 3 | 3 | 18254 | 2.635 | 2.735 | 0.0047 | 195 | 6.43 | 1.012 | 0.000 | 0 | 3 |
| -22 to -21 | loud | 13 | 13 | 48396 | 2.640 | 2.679 | 0.0022 | 289 | 6.17 | 0.989 | 0.000 | 0 | 7 |
| -21 to -20 | loud | 15 | 15 | 45989 | 2.634 | 2.632 | -0.0011 | 150 | 4.01 | 0.910 | 0.000 | 1 | 4 |
| -20 to -19 | loud | 12 | 12 | 50120 | 2.639 | 2.644 | 0.0035 | 217 | 6.62 | 1.012 | 0.000 | 0 | 7 |
| -19 to -18 | loud | 10 | 10 | 45739 | 2.626 | 2.681 | 0.0055 | 374 | 3.30 | 0.886 | 0.000 | 0 | 2 |
| -18 to -17 | loud | 6 | 5 | 25267 | 2.664 | 2.640 | -0.0086 | 334 | 3.38 | 0.936 | 0.200 | 0 | 3 |

### Either side of the cutover

| arm | files | done | delta pct | new wps | segments | seg mean s | anomaly | rescue rate | regr |
|---|---|---|---|---|---|---|---|---|---|
| below -26.0 dBFS (quiet profile) | 20 | 20 | 0.0031 | 2.655 | 243 | 4.29 | 0.0 | 0.150 | 1 |
| at or above -26.0 dBFS (loud profile) | 82 | 81 | 0.0012 | 2.648 | 265 | 4.92 | 0.0 | 0.025 | 1 |
| -28.0 to -26.0 dBFS (quiet profile) | 18 | 18 | 0.0019 | 2.678 | 186 | 4.29 | 0.0 | 0.167 | 1 |
| -26.0 to -24.0 dBFS (loud profile) | 16 | 16 | 0.0008 | 2.713 | 284 | 6.39 | 0.0 | 0.000 | 0 |

## Segment coverage of the VAD speech seconds

Measured on 101 files. Coverage median 0.9359, p5 0.8413, min 0.7925. Files below each band: under 0.5: 0, under 0.7: 0, under 0.8: 1, under 0.9: 35, under 0.95: 54.

Coverage is the total span of the emitted segments over the seconds VAD kept. A shortfall is audio the decoder emitted nothing for, which is exactly what the segment-derived speech clock could not see.

| file | coverage | span total s | speech s | dur s | anomaly | anomaly windows |
|---|---|---|---|---|---|---|
| tcf.20260626.mp3 | 0.7925 | 1133.8 | 1430.6 | 1505 | 0 | 0 |
| tcf.20260724.mp3 | 0.8159 | 1324.9 | 1623.8 | 1731 | 0 | 0 |
| tcf.20260904.mp3 | 0.8190 | 870.8 | 1063.3 | 1130 | 0 | 0 |
| tcf.20210217.mp3 | 0.8346 | 508.8 | 609.6 | 616 | 0 | 0 |
| tcf.202606623b.mp3 | 0.8401 | 435.8 | 518.7 | 535 | 0 | 0 |
| tcf.20210604.mp3 | 0.8413 | 1313.1 | 1560.7 | 1660 | 0 | 0 |
| tcf.20210514.mp3 | 0.8438 | 1741.6 | 2064.0 | 2141 | 0 | 0 |
| tcf.20260717.mp3 | 0.8447 | 1128.6 | 1336.2 | 1419 | 0 | 0 |
| tcf.20250426.mp3 | 0.8460 | 1446.6 | 1710.1 | 1841 | 0 | 0 |
| tcf.20220205.mp3 | 0.8522 | 1109.4 | 1301.8 | 1371 | 0 | 0 |

## Seam de-duplication

31 of 101 files had a trim. 59 trims removed 255 words in total.

| file | trims | words removed | words after | words before | delta pct vs legacy |
|---|---|---|---|---|---|
| tcf.20241129.mp3 | 6 | 26 | 6886 | 6912 | 0.0012 |
| tcf.20260717.mp3 | 4 | 19 | 3708 | 3727 | -0.0188 |
| tcf.20210604.mp3 | 4 | 18 | 4487 | 4505 | -0.0160 |
| tcf.20240412.mp3 | 4 | 18 | 3353 | 3371 | -0.0074 |
| ucf.20211106c.mp3 | 4 | 17 | 5877 | 5894 | -0.0073 |
| tcf.20220121.mp3 | 3 | 13 | 3981 | 3994 | 0.0010 |
| tcf.20260724.mp3 | 3 | 13 | 4419 | 4432 | -0.0292 |
| tcf.20230603_mens_fast.mp3 | 3 | 12 | 5755 | 5767 | 0.0097 |
| tcf.20250725.mp3 | 2 | 10 | 4441 | 4451 | 0.0004 |
| tcf.20210611.mp3 | 2 | 8 | 4973 | 4981 | 0.0002 |
| tcf.20211204.mp3 | 2 | 8 | 6819 | 6827 | -0.0045 |
| tcf.20250426.mp3 | 2 | 8 | 4994 | 5002 | 0.0111 |
| tcf.20260905.mp3 | 2 | 8 | 3656 | 3664 | -0.0019 |
| cf.20220402.mens_breakfast_In_Christ.mp3 | 1 | 5 | 4018 | 4023 | 0.0037 |
| tcf.20231229.mp3 | 1 | 5 | 4068 | 4073 | -0.0022 |
| tcf.20240130b.mp3 | 1 | 5 | 3670 | 3675 | 0.0030 |
| tcf.20240319a.mp3 | 1 | 5 | 6191 | 6196 | -0.0091 |
| tcf.20240416_Formation_Class_Holy_Spirit.mp3 | 1 | 5 | 4010 | 4015 | 0.0378 |
| tcf.20200515b.mp3 | 1 | 4 | 3434 | 3438 | 0.0000 |
| tcf.20200612.mp3 | 1 | 4 | 1467 | 1471 | -0.0020 |
| tcf.20201003.mens_.mp3 | 1 | 4 | 4407 | 4411 | -0.0034 |
| tcf.20220205.mp3 | 1 | 4 | 3491 | 3495 | -0.0188 |
| tcf.20231006.mp3 | 1 | 4 | 6717 | 6721 | 0.0315 |
| tcf.20240116b.mp3 | 1 | 4 | 1977 | 1981 | 0.0010 |
| tcf.20240326b.mp3 | 1 | 4 | 2753 | 2757 | - |
| tcf.20240713.mp3 | 1 | 4 | 6545 | 6549 | -0.0086 |
| tcf.20251212.mp3 | 1 | 4 | 4113 | 4117 | -0.0007 |
| tcf.20260331b.mp3 | 1 | 4 | 1955 | 1959 | 0.0077 |
| tcf.20260626.mp3 | 1 | 4 | 3750 | 3754 | 0.0143 |
| tcf.20260904.mp3 | 1 | 4 | 2845 | 2849 | 0.0312 |
| women_retreat_2026_session3.mp3 | 1 | 4 | 2460 | 2464 | -0.0037 |

Every trimmed phrase, for judging by eye. A four-word minimum still removes a liturgical response that straddles a seam, so whether four is the right floor is a question about these phrases and not about the count.

| file | at s | words | phrase removed | emptied segment |
|---|---|---|---|---|
| cf.20220402.mens_breakfast_In_Christ.mp3 | 120.96 | 5 | in the way he's thinking | no |
| tcf.20200515b.mp3 | 1112.50 | 4 | Are you a slave | no |
| tcf.20200612.mp3 | 452.85 | 4 | This is the time | no |
| tcf.20201003.mens_.mp3 | 1486.63 | 4 | There's a ticking clock | no |
| tcf.20210604.mp3 | 393.68 | 5 | This part of my life | no |
| tcf.20210604.mp3 | 768.06 | 4 | Repentance moves beyond conviction. | yes |
| tcf.20210604.mp3 | 878.51 | 4 | What shall we do? | yes |
| tcf.20210604.mp3 | 1303.36 | 5 | Exactly what has to change. | yes |
| tcf.20210611.mp3 | 464.11 | 4 | where the difference lies | no |
| tcf.20210611.mp3 | 649.07 | 4 | Repentance insists on reality. | yes |
| tcf.20211204.mp3 | 1012.06 | 4 | Facts are stubborn things | no |
| tcf.20211204.mp3 | 2131.93 | 4 | Better than Chuck Norris, | no |
| tcf.20220121.mp3 | 625.32 | 4 | We don't like limits | no |
| tcf.20220121.mp3 | 773.79 | 4 | the heart of Sabbath | no |
| tcf.20220121.mp3 | 1270.58 | 5 | need a little bit more. | no |
| tcf.20220205.mp3 | 1128.04 | 4 | I for an eye, | yes |
| tcf.20230603_mens_fast.mp3 | 459.19 | 4 | The blessed man is | no |
| tcf.20230603_mens_fast.mp3 | 587.31 | 4 | You know every word | no |
| tcf.20230603_mens_fast.mp3 | 678.61 | 4 | The king of Israel | no |
| tcf.20231006.mp3 | 2738.44 | 4 | I know that guy. | no |
| tcf.20231229.mp3 | 1091.42 | 5 | For the people of God, | no |
| tcf.20240116b.mp3 | 147.86 | 4 | TCF has a vessel. | no |
| tcf.20240130b.mp3 | 98.74 | 5 | if you were to ask | no |
| tcf.20240319a.mp3 | 272.32 | 5 | Don't tell anybody about this. | yes |
| tcf.20240326b.mp3 | 18.74 | 4 | It might become that. | yes |
| tcf.20240412.mp3 | 568.47 | 4 | I was a jerk. | yes |
| tcf.20240412.mp3 | 710.51 | 4 | Encourage the faint-hearted. | yes |
| tcf.20240412.mp3 | 853.00 | 5 | I can do these things. | yes |
| tcf.20240412.mp3 | 1261.68 | 5 | He gave them a job. | yes |
| tcf.20240416_Formation_Class_Holy_Spirit.mp3 | 952.00 | 5 | The spirit has perfect memory | no |
| tcf.20240713.mp3 | 2015.09 | 4 | out in the wild | yes |
| tcf.20241129.mp3 | 802.82 | 5 | Saying thank you to God. | yes |
| tcf.20241129.mp3 | 1547.62 | 4 | The Lord is God. | yes |
| tcf.20241129.mp3 | 1768.84 | 4 | Yahweh, he is God. | yes |
| tcf.20241129.mp3 | 1772.84 | 5 | The Lord, he is God. | yes |
| tcf.20241129.mp3 | 2963.63 | 4 | They're really into this | no |
| tcf.20241129.mp3 | 3021.81 | 4 | this is not a | no |
| tcf.20250426.mp3 | 391.45 | 4 | in a sinful world. | yes |
| tcf.20250426.mp3 | 1490.54 | 4 | you know the time | yes |
| tcf.20250725.mp3 | 196.74 | 5 | It is a total gift | no |
| tcf.20250725.mp3 | 1525.46 | 5 | to see what it is | yes |
| tcf.20251212.mp3 | 675.70 | 4 | He has to say | no |
| tcf.20260331b.mp3 | 42.56 | 4 | That is not true. | no |
| tcf.20260626.mp3 | 885.07 | 4 | you can't be all | no |
| tcf.20260717.mp3 | 268.26 | 5 | None of that counts anymore | no |
| tcf.20260717.mp3 | 478.24 | 5 | They didn't count for anything. | yes |
| tcf.20260717.mp3 | 1086.71 | 4 | Write down your resume. | yes |
| tcf.20260717.mp3 | 1171.23 | 5 | Write down your anti-resume. | yes |
| tcf.20260724.mp3 | 326.65 | 5 | Rejoice in the Lord always. | yes |
| tcf.20260724.mp3 | 714.79 | 4 | All is really well. | yes |
| tcf.20260724.mp3 | 1439.32 | 4 | It's God who provides. | yes |
| tcf.20260904.mp3 | 360.94 | 4 | God builds a house. | yes |
| tcf.20260905.mp3 | 884.08 | 4 | When we're more vulnerable | no |
| tcf.20260905.mp3 | 959.26 | 4 | No one will know | no |
| ucf.20211106c.mp3 | 193.63 | 5 | It's good to have that | no |
| ucf.20211106c.mp3 | 483.82 | 4 | Exodus 20, verse 20. | yes |
| ucf.20211106c.mp3 | 2232.22 | 4 | What can we do | no |
| ucf.20211106c.mp3 | 2643.77 | 4 | I know a guy. | yes |
| women_retreat_2026_session3.mp3 | 835.47 | 4 | We can't do that | no |

## Alignment counters

Measured on 101 files. agree250 median 0.6287, p5 0.3738. Span ratio median of medians 0.9905, p5 of p5s 0.5998. 868 segments fell below the 0.5 span ratio across 48 files; 836 timings came from the monotonic clamp across 44 files; 719 empty fallbacks across 16 files.

## Rescue pass

Attempted on 5 files, selected on 4.

| file | dur s | dBFS | anomaly | legacy words | words | delta | delta pct |
|---|---|---|---|---|---|---|---|
| women_retreat_2025_session4.mp3 | 1625 | -28.0 | 0 | 3216 | 3191 | -25 | -0.0078 |
| ucf.20211106c.mp3 | 2869 | -26.9 | 0 | 5920 | 5877 | -43 | -0.0073 |
| tcf.20240621.mp3 | 1735 | -20.3 | 0 | 4570 | 4565 | -5 | -0.0011 |
| tcf.20260331b.mp3 | 764 | -27.1 | 0 | 1940 | 1955 | 15 | 0.0077 |

## Regressions to read by hand

2 files returned more than 5 percent fewer words than the baseline.

## Per file

| file | outcome | dur s | dBFS | era | legacy words | new words | delta pct | legacy wps | new wps | wps speech | coverage | trims | segs | seg mean s | wall s | anomaly | resc a/s | flags | mfa | attempts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcf.20150424.mp3 | error | 2998 | -17.8 | E1_0.1.5 | 8292 | 0 | -1.0000 | 2.766 | 0.000 | 0.000 | - | 23 | 0 | - | 510.3 | - | None/None | - | - | - |
| tcf.20210210.mp3 | completed | 469 | -20.3 | E1_0.1.5 | 1248 | 1173 | -0.0601 | 2.661 | 2.501 | 2.527 | 0.859 | 0 | 81 | 4.92 | 60.1 | 0 | False/False | 0 | True | 0 |
| tcf.20210217.mp3 | completed | 616 | -26.4 | E1_0.1.5 | 1683 | 1595 | -0.0523 | 2.732 | 2.589 | 2.616 | 0.835 | 0 | 138 | 3.69 | 60.1 | 0 | False/False | 0 | True | 0 |
| tcf.202606623b.mp3 | completed | 535 | -19.9 | E3_0.3.x | 1375 | 1307 | -0.0495 | 2.570 | 2.443 | 2.520 | 0.840 | 0 | 110 | 3.96 | 60.1 | 0 | False/False | 0 | True | 0 |
| tcf.20210319.mp3 | completed | 2396 | -23.1 | E1_0.1.5 | 5155 | 4914 | -0.0467 | 2.151 | 2.051 | 2.791 | 1.274 | 0 | 255 | 8.80 | 150.2 | 0 | False/False | 4 | True | 0 |
| tcf.20241105.mp3 | completed | 2981 | -20.0 | E1_0.1.5 | 8056 | 7794 | -0.0325 | 2.702 | 2.615 | 2.866 | 0.949 | 0 | 749 | 3.45 | 160.3 | 0 | False/False | 1 | True | 0 |
| tcf.20260724.mp3 | completed | 1731 | -20.7 | E3_0.3.x | 4552 | 4419 | -0.0292 | 2.630 | 2.553 | 2.721 | 0.816 | 13 | 951 | 1.39 | 120.1 | 0 | False/False | 0 | True | 0 |
| tcf.20220205.mp3 | completed | 1371 | -18.8 | E1_0.1.5 | 3558 | 3491 | -0.0188 | 2.595 | 2.546 | 2.682 | 0.852 | 4 | 835 | 1.33 | 100.1 | 0 | False/False | 0 | True | 0 |
| tcf.20260717.mp3 | completed | 1419 | -23.5 | E3_0.3.x | 3779 | 3708 | -0.0188 | 2.663 | 2.613 | 2.775 | 0.845 | 19 | 341 | 3.31 | 90.1 | 0 | False/False | 0 | True | 0 |
| tcf.20210604.mp3 | completed | 1660 | -20.5 | E1_0.1.5 | 4560 | 4487 | -0.0160 | 2.747 | 2.703 | 2.875 | 0.841 | 18 | 582 | 2.26 | 100.1 | 0 | False/False | 0 | True | 0 |
| tcf.20220729.mp3 | completed | 2215 | -27.9 | E1_0.1.5 | 5769 | 5690 | -0.0137 | 2.604 | 2.569 | 2.790 | 0.892 | 0 | 644 | 2.83 | 130.2 | 0 | False/False | 1 | True | 0 |
| tcf.20240301a.mp3 | completed | 238 | -21.5 | E1_0.1.5 | 660 | 653 | -0.0106 | 2.773 | 2.744 | 2.831 | 0.996 | 0 | 34 | 6.76 | 60.1 | 0 | False/False | 1 | True | 0 |
| women_retreat_2026_session1.mp3 | completed | 3183 | -23.9 | E3_0.3.x | 7975 | 7891 | -0.0105 | 2.506 | 2.479 | 2.681 | 0.874 | 0 | 918 | 2.80 | 150.2 | 0 | False/False | 0 | True | 0 |
| tcf.20240319a.mp3 | completed | 2345 | -17.7 | E4_0.5.x | 6248 | 6191 | -0.0091 | 2.664 | 2.640 | 2.860 | 0.941 | 5 | 602 | 3.38 | 170.2 | 0 | False/False | 1 | True | 0 |
| tcf.20240713.mp3 | completed | 2404 | -17.5 | E1_0.1.5 | 6602 | 6545 | -0.0086 | 2.746 | 2.723 | 2.831 | 0.874 | 4 | 1324 | 1.52 | 200.1 | 0 | False/False | 0 | True | 0 |
| women_retreat_2025_session4.mp3 | completed | 1625 | -28.0 | E2_0.2.x | 3216 | 3191 | -0.0078 | 1.979 | 1.964 | 2.028 | 0.911 | 0 | 235 | 6.10 | 140.1 | 0 | True/True | 0 | True | 0 |
| tcf.20240412.mp3 | completed | 1309 | -18.9 | E1_0.1.5 | 3378 | 3353 | -0.0074 | 2.581 | 2.562 | 2.748 | 0.875 | 18 | 289 | 3.69 | 90.1 | 0 | False/False | 0 | True | 0 |
| ucf.20211106c.mp3 | completed | 2869 | -26.9 | E1_0.1.5 | 5920 | 5877 | -0.0073 | 2.063 | 2.048 | 2.428 | 0.874 | 17 | 816 | 2.59 | 350.2 | 0 | True/True | 0 | True | 0 |
| tcf20260825.mp3 | completed | 1477 | -21.7 | E3_0.3.x | 4174 | 4147 | -0.0065 | 2.826 | 2.808 | 2.946 | 0.902 | 0 | 396 | 3.21 | 100.1 | 0 | False/False | 0 | True | 0 |
| tcf.20211204.mp3 | completed | 2603 | -20.4 | E1_0.1.5 | 6850 | 6819 | -0.0045 | 2.632 | 2.620 | 2.748 | 0.890 | 8 | 571 | 3.87 | 160.1 | 0 | False/False | 1 | True | 0 |
| tcf.20240405.mp3 | completed | 1456 | -25.8 | E1_0.1.5 | 4126 | 4108 | -0.0044 | 2.834 | 2.821 | 2.949 | 1.012 | 0 | 220 | 6.41 | 90.1 | 0 | False/False | 2 | True | 0 |
| tcf.20200818.mp3 | completed | 412 | -25.3 | E1_0.1.5 | 1009 | 1005 | -0.0040 | 2.449 | 2.439 | 2.481 | 0.887 | 0 | 99 | 3.63 | 60.1 | 0 | False/False | 0 | True | 0 |
| women_retreat_2026_session3.mp3 | completed | 863 | -18.4 | E3_0.3.x | 2469 | 2460 | -0.0037 | 2.861 | 2.850 | 3.073 | 0.920 | 4 | 193 | 3.81 | 80.1 | 0 | False/False | 0 | True | 0 |
| tcf.20201003.mens_.mp3 | completed | 1639 | -18.7 | E1_0.1.5 | 4422 | 4407 | -0.0034 | 2.698 | 2.689 | 2.851 | 0.863 | 4 | 459 | 2.91 | 110.1 | 0 | False/False | 0 | True | 0 |
| tcf.20260728a.mp3 | completed | 1435 | -20.0 | E3_0.3.x | 3809 | 3797 | -0.0032 | 2.654 | 2.646 | 2.828 | 1.036 | 0 | 203 | 6.85 | 100.1 | 0 | False/False | 1 | True | 0 |
| tcf.20201021.mp3 | completed | 603 | -25.2 | E1_0.1.5 | 1641 | 1636 | -0.0031 | 2.721 | 2.713 | 2.752 | 0.986 | 0 | 86 | 6.81 | 70.1 | 0 | False/False | 0 | True | 0 |
| tcf.20200922.mp3 | completed | 378 | -20.2 | E1_0.1.5 | 1164 | 1161 | -0.0026 | 3.079 | 3.071 | 3.150 | 0.864 | 0 | 130 | 2.45 | 70.1 | 0 | False/False | 0 | True | 0 |
| tcf.20260710.mp3 | completed | 1562 | -20.5 | E3_0.3.x | 4075 | 4065 | -0.0024 | 2.609 | 2.602 | 2.734 | 1.006 | 0 | 225 | 6.65 | 100.1 | 0 | False/False | 1 | True | 0 |
| tcf.20210131.mp3 | completed | 625 | -20.4 | E1_0.1.5 | 1767 | 1763 | -0.0023 | 2.827 | 2.821 | 2.869 | 0.910 | 0 | 141 | 3.97 | 80.1 | 0 | False/False | 0 | True | 0 |
| tcf.20231229.mp3 | completed | 1455 | -24.3 | E1_0.1.5 | 4077 | 4068 | -0.0022 | 2.802 | 2.796 | 2.957 | 1.029 | 5 | 219 | 6.46 | 100.1 | 0 | False/False | 2 | True | 0 |
| tcf.20200612.mp3 | completed | 562 | -21.4 | E1_0.1.5 | 1470 | 1467 | -0.0020 | 2.616 | 2.610 | 2.657 | 0.895 | 4 | 152 | 3.25 | 60.1 | 0 | False/False | 0 | True | 0 |
| tcf.20260905.mp3 | completed | 1415 | -19.7 | E3_0.3.x | 3663 | 3656 | -0.0019 | 2.589 | 2.584 | 2.736 | 0.858 | 8 | 495 | 2.31 | 100.1 | 0 | False/False | 2 | True | 0 |
| tcf.20240621.mp3 | completed | 1735 | -20.3 | E1_0.1.5 | 4570 | 4565 | -0.0011 | 2.634 | 2.631 | 2.779 | 0.919 | 0 | 377 | 4.01 | 140.1 | 0 | True/True | 1 | True | 0 |
| tcf.20251212.mp3 | completed | 1492 | -26.6 | E2_0.2.x | 4116 | 4113 | -0.0007 | 2.759 | 2.757 | 2.844 | 0.887 | 4 | 313 | 4.10 | 100.2 | 1 | False/False | 1 | True | 0 |
| tcf.20250704.mp3 | completed | 1088 | -24.5 | E2_0.2.x | 2842 | 2840 | -0.0007 | 2.612 | 2.610 | 2.776 | 1.025 | 0 | 152 | 6.90 | 80.1 | 0 | False/False | 0 | True | 0 |
| tcf.20260627.mp3 | completed | 2279 | -24.5 | E3_0.3.x | 6012 | 6009 | -0.0005 | 2.638 | 2.637 | 2.811 | 1.033 | 0 | 327 | 6.76 | 120.1 | 0 | False/False | 3 | True | 0 |
| tcf.20200422.mp3 | completed | 486 | -27.9 | E1_0.1.5 | 1355 | 1355 | 0.0000 | 2.788 | 2.788 | 2.789 | 0.901 | 0 | 102 | 4.29 | 60.1 | 0 | False/False | 0 | True | 0 |
| tcf.20200515b.mp3 | completed | 1263 | -24.2 | E1_0.1.5 | 3434 | 3434 | 0.0000 | 2.719 | 2.719 | 2.818 | 0.878 | 4 | 398 | 2.69 | 90.1 | 0 | False/False | 1 | True | 0 |
| tcf.20200729.mp3 | completed | 543 | -20.2 | E1_0.1.5 | 1421 | 1421 | 0.0000 | 2.617 | 2.617 | 2.678 | 0.990 | 0 | 77 | 6.83 | 70.1 | 0 | False/False | 0 | True | 0 |
| tcf.20200814.mp3 | completed | 480 | -20.8 | E1_0.1.5 | 1264 | 1264 | 0.0000 | 2.633 | 2.633 | 2.672 | 0.965 | 0 | 69 | 6.62 | 70.1 | 0 | False/False | 0 | True | 0 |
| tcf.20201126.mp3 | completed | 257 | -26.3 | E1_0.1.5 | 734 | 734 | 0.0000 | 2.856 | 2.856 | 2.931 | 0.921 | 0 | 55 | 4.19 | 50.1 | 0 | False/False | 0 | True | 0 |
| tcf.20201214-15.mp3 | completed | 586 | -21.5 | E1_0.1.5 | 1601 | 1601 | 0.0000 | 2.732 | 2.732 | 2.791 | 0.992 | 0 | 88 | 6.47 | 70.1 | 0 | False/False | 0 | True | 0 |
| tcf.20210611.mp3 | completed | 1728 | -20.2 | E1_0.1.5 | 4972 | 4973 | 0.0002 | 2.877 | 2.878 | 2.990 | 0.876 | 8 | 486 | 3.00 | 100.1 | 0 | False/False | 0 | True | 0 |
| tcf.20250725.mp3 | completed | 1629 | -27.1 | E2_0.2.x | 4439 | 4441 | 0.0004 | 2.725 | 2.726 | 2.751 | 0.870 | 10 | 906 | 1.55 | 100.1 | 0 | False/False | 0 | True | 0 |
| tcf.20260102.mp3 | completed | 1867 | -25.5 | E2_0.2.x | 5416 | 5420 | 0.0007 | 2.901 | 2.903 | 2.979 | 0.887 | 0 | 542 | 2.98 | 110.1 | 0 | False/False | 2 | True | 0 |
| ucf.20251109.mp3 | completed | 1267 | -24.5 | E2_0.2.x | 3190 | 3193 | 0.0009 | 2.518 | 2.520 | 2.724 | 0.898 | 0 | 287 | 3.67 | 90.1 | 0 | False/False | 0 | True | 0 |
| tcf.20220121.mp3 | completed | 1454 | -21.3 | E1_0.1.5 | 3977 | 3981 | 0.0010 | 2.735 | 2.738 | 2.986 | 1.050 | 13 | 210 | 6.67 | 90.1 | 0 | False/False | 6 | True | 0 |
| tcf.20240116b.mp3 | completed | 798 | -23.6 | E4_0.5.x | 1975 | 1977 | 0.0010 | 2.475 | 2.477 | 2.807 | 1.022 | 4 | 109 | 6.61 | 70.1 | 0 | False/False | 1 | True | 0 |
| tcf.20241129.mp3 | completed | 3093 | -21.4 | E1_0.1.5 | 6878 | 6886 | 0.0012 | 2.224 | 2.226 | 2.821 | 0.902 | 26 | 921 | 2.39 | 140.2 | 0 | False/False | 1 | True | 0 |
| tcf.20200726.mp3 | completed | 450 | -26.9 | E1_0.1.5 | 1232 | 1234 | 0.0016 | 2.738 | 2.742 | 2.770 | 0.979 | 0 | 65 | 6.71 | 70.1 | 0 | False/False | 1 | True | 0 |
| tcf.20260313.mp3 | completed | 1636 | -24.6 | E2_0.2.x | 4002 | 4009 | 0.0018 | 2.446 | 2.450 | 2.734 | 1.075 | 0 | 214 | 7.37 | 100.1 | 0 | False/False | 2 | True | 0 |
| sermon_from_thomas_wingfold_by_george_macdonald.m4a | completed | 628 | -26.3 | E1_0.1.5 | 1860 | 1864 | 0.0022 | 2.962 | 2.968 | 2.996 | 0.978 | 0 | 96 | 6.34 | 60.1 | 0 | False/False | 0 | True | 0 |
| tcf.202606623.mp3 | completed | 2498 | -21.5 | E3_0.3.x | 6496 | 6510 | 0.0022 | 2.600 | 2.606 | 2.818 | 1.042 | 0 | 346 | 6.96 | 160.2 | 0 | False/False | 0 | True | 0 |
| tcf.20260801.mp3 | completed | 1266 | -24.7 | E3_0.3.x | 3665 | 3673 | 0.0022 | 2.895 | 2.901 | 3.035 | 1.007 | 0 | 203 | 6.00 | 90.1 | 0 | False/False | 0 | True | 0 |
| tcf.20260807.mp3 | completed | 1299 | -19.6 | E3_0.3.x | 3349 | 3357 | 0.0024 | 2.578 | 2.584 | 2.740 | 1.032 | 0 | 183 | 6.91 | 90.1 | 0 | False/False | 0 | True | 0 |
| tcf.20240116.mp3 | completed | 2783 | -24.4 | E4_0.5.x | 7815 | 7837 | 0.0028 | 2.808 | 2.816 | 2.947 | 1.016 | 0 | 423 | 6.38 | 190.3 | 0 | False/False | 1 | True | 0 |
| tcf.20240130a.mp3 | completed | 2084 | -23.9 | E4_0.5.x | 5520 | 5536 | 0.0029 | 2.649 | 2.656 | 2.802 | 1.019 | 0 | 296 | 6.80 | 110.2 | 0 | False/False | 2 | True | 0 |
| tcf.20240213a.mp3 | completed | 1463 | -23.9 | E4_0.5.x | 4022 | 4034 | 0.0030 | 2.749 | 2.757 | 2.803 | 0.912 | 0 | 373 | 3.52 | 100.1 | 0 | False/False | 0 | True | 0 |
| tcf.20240130b.mp3 | completed | 1322 | -22.1 | E4_0.5.x | 3659 | 3670 | 0.0030 | 2.768 | 2.776 | 2.895 | 1.012 | 5 | 195 | 6.58 | 90.1 | 0 | False/False | 1 | True | 0 |
| tcf.20260821.mp3 | completed | 1502 | -19.7 | E3_0.3.x | 4058 | 4071 | 0.0032 | 2.702 | 2.710 | 2.807 | 1.007 | 0 | 218 | 6.70 | 90.1 | 0 | False/False | 1 | True | 0 |
| tcf.20200430.mp3 | completed | 568 | -21.9 | E1_0.1.5 | 1607 | 1613 | 0.0037 | 2.829 | 2.840 | 2.853 | 0.913 | 0 | 106 | 4.87 | 70.1 | 0 | False/False | 0 | True | 0 |
| cf.20220402.mens_breakfast_In_Christ.mp3 | completed | 1498 | -19.1 | E1_0.1.5 | 4003 | 4018 | 0.0037 | 2.672 | 2.682 | 2.865 | 1.029 | 5 | 212 | 6.81 | 100.1 | 0 | False/False | 3 | True | 0 |
| ucf.20251108a.mp3 | completed | 1269 | -30.4 | E2_0.2.x | 3383 | 3397 | 0.0041 | 2.666 | 2.677 | 2.755 | 0.853 | 0 | 584 | 1.80 | 90.1 | 0 | False/False | 0 | True | 0 |
| tcf.20210514.mp3 | completed | 2141 | -18.4 | E1_0.1.5 | 6203 | 6230 | 0.0043 | 2.897 | 2.910 | 3.018 | 0.844 | 0 | 1285 | 1.35 | 130.1 | 0 | False/False | 0 | True | 0 |
| tcf.20260428.mp3 | completed | 5095 | -22.8 | E2_0.2.x | 13427 | 13490 | 0.0047 | 2.635 | 2.648 | 2.806 | 1.014 | 0 | 758 | 6.43 | 230.3 | 0 | False/False | 1 | True | 0 |
| tcf.20200809.mp3 | completed | 255 | -27.3 | E1_0.1.5 | 754 | 758 | 0.0053 | 2.957 | 2.973 | 2.990 | 0.916 | 0 | 47 | 4.94 | 60.1 | 0 | False/False | 0 | True | 0 |
| women_retreat_2025_session2.mp3 | completed | 1842 | -24.9 | E2_0.2.x | 5461 | 5490 | 0.0053 | 2.965 | 2.981 | 3.066 | 1.000 | 0 | 280 | 6.39 | 100.1 | 0 | False/False | 0 | True | 0 |
| tcf.20200807.mp3 | completed | 1352 | -19.9 | E1_0.1.5 | 3548 | 3571 | 0.0065 | 2.624 | 2.641 | 2.746 | 0.876 | 0 | 690 | 1.65 | 100.1 | 0 | False/False | 0 | True | 0 |
| tcf.20240524.mp3 | completed | 1645 | -18.5 | E1_0.1.5 | 4369 | 4398 | 0.0066 | 2.656 | 2.674 | 2.778 | 1.008 | 0 | 238 | 6.70 | 100.1 | 0 | False/False | 4 | True | 0 |
| tcf.20260331a.mp3 | completed | 3794 | -25.0 | E2_0.2.x | 9942 | 10011 | 0.0069 | 2.620 | 2.639 | 2.847 | 1.052 | 0 | 547 | 6.76 | 160.2 | 0 | False/False | 0 | True | 0 |
| tcf.20260331b.mp3 | completed | 764 | -27.1 | E2_0.2.x | 1940 | 1955 | 0.0077 | 2.539 | 2.559 | 2.777 | 1.048 | 4 | 111 | 6.65 | 90.1 | 0 | True/True | 0 | True | 0 |
| tcf.20260731-1.mp3 | completed | 669 | -20.9 | E3_0.3.x | 1747 | 1761 | 0.0080 | 2.611 | 2.632 | 2.801 | 1.030 | 0 | 94 | 6.89 | 70.1 | 0 | False/False | 3 | True | 0 |
| tcf.20240319b.mp3 | completed | 1369 | -17.3 | E4_0.5.x | 3582 | 3611 | 0.0081 | 2.616 | 2.638 | 2.867 | 0.936 | 0 | 323 | 3.65 | 100.1 | 1 | False/False | 3 | True | 0 |
| tcf.20230603_mens_fast.mp3 | completed | 1978 | -18.1 | E1_0.1.5 | 5700 | 5755 | 0.0097 | 2.882 | 2.909 | 3.019 | 0.870 | 12 | 1389 | 1.19 | 130.1 | 0 | False/False | 0 | True | 0 |
| tcf.20210114.mp3 | completed | 511 | -21.6 | E1_0.1.5 | 1530 | 1546 | 0.0105 | 2.994 | 3.025 | 3.062 | 0.989 | 0 | 81 | 6.17 | 70.1 | 0 | False/False | 2 | True | 0 |
| tcf.20240213b.mp3 | completed | 755 | -23.4 | E4_0.5.x | 2085 | 2107 | 0.0106 | 2.762 | 2.791 | 2.867 | 0.917 | 0 | 195 | 3.46 | 80.1 | 0 | False/False | 1 | True | 0 |
| tcf.20250426.mp3 | completed | 1841 | -24.4 | E2_0.2.x | 4939 | 4994 | 0.0111 | 2.683 | 2.713 | 2.920 | 0.846 | 8 | 966 | 1.50 | 120.1 | 0 | False/False | 0 | True | 0 |
| women_retreat_2025_session1.mp3 | completed | 1242 | -26.9 | E2_0.2.x | 2226 | 2255 | 0.0130 | 1.792 | 1.816 | 2.089 | 0.966 | 0 | 251 | 4.15 | 80.1 | 0 | False/False | 3 | True | 0 |
| tcf.20211203.mp3 | completed | 3875 | -27.4 | E1_0.1.5 | 9870 | 10005 | 0.0137 | 2.547 | 2.582 | 2.774 | 0.931 | 0 | 785 | 4.28 | 190.2 | 0 | False/False | 1 | True | 0 |
| tcf.20260626.mp3 | completed | 1505 | -24.1 | E3_0.3.x | 3697 | 3750 | 0.0143 | 2.457 | 2.492 | 2.621 | 0.792 | 4 | 683 | 1.66 | 100.1 | 0 | False/False | 0 | True | 0 |
| tcf.20240326a.mp3 | completed | 2006 | -21.5 | E4_0.5.x | 5296 | 5375 | 0.0149 | 2.640 | 2.679 | 2.880 | 1.037 | 0 | 289 | 6.70 | 150.2 | 0 | True/False | 3 | True | 0 |
| ucf20211106b.mp3 | completed | 2248 | -21.3 | E1_0.1.5 | 4974 | 5055 | 0.0163 | 2.213 | 2.249 | 2.420 | 0.937 | 0 | 386 | 5.07 | 120.1 | 0 | False/False | 1 | True | 0 |
| tcf.20221203.mp3 | completed | 3704 | -18.7 | E1_0.1.5 | 9514 | 9703 | 0.0199 | 2.569 | 2.620 | 2.710 | 0.897 | 0 | 3645 | 0.88 | 210.2 | 0 | False/False | 0 | True | 0 |
| tcf.20260830.mp3 | completed | 1754 | -19.4 | E3_0.3.x | 4561 | 4668 | 0.0235 | 2.600 | 2.661 | 2.841 | 1.040 | 0 | 242 | 7.06 | 100.1 | 0 | False/False | 0 | True | 0 |
| women_retreat_2026_session4.mp3 | completed | 996 | -20.2 | E3_0.3.x | 2761 | 2830 | 0.0250 | 2.772 | 2.841 | 2.965 | 1.009 | 0 | 150 | 6.42 | 80.1 | 0 | False/False | 0 | True | 0 |
| tcf.20250607.mp3 | completed | 2846 | -19.9 | E2_0.2.x | 7619 | 7818 | 0.0261 | 2.677 | 2.747 | 2.848 | 0.922 | 0 | 754 | 3.36 | 220.2 | 0 | False/False | 1 | True | 0 |
| tcf.20260904.mp3 | completed | 1130 | -21.1 | E3_0.3.x | 2759 | 2845 | 0.0312 | 2.442 | 2.518 | 2.676 | 0.819 | 4 | 607 | 1.44 | 90.1 | 0 | False/False | 0 | True | 0 |
| tcf.20231006.mp3 | completed | 2875 | -21.6 | E1_0.1.5 | 6512 | 6717 | 0.0315 | 2.265 | 2.336 | 2.909 | 1.178 | 4 | 360 | 7.56 | 130.2 | 0 | False/False | 2 | True | 0 |
| tcf.20260728b.mp3 | completed | 817 | -19.1 | E3_0.3.x | 1987 | 2053 | 0.0332 | 2.432 | 2.513 | 2.618 | 1.017 | 0 | 115 | 6.93 | 80.1 | 0 | False/False | 0 | True | 0 |
| ucf.20251107.mp3 | completed | 2654 | -26.2 | E2_0.2.x | 6610 | 6845 | 0.0355 | 2.491 | 2.579 | 2.770 | 1.046 | 0 | 389 | 6.65 | 210.3 | 0 | False/False | 0 | True | 0 |
| tcf.20240416_Formation_Class_Holy_Spirit.mp3 | completed | 1454 | -19.6 | E4_0.5.x | 3864 | 4010 | 0.0378 | 2.658 | 2.758 | 2.902 | 1.023 | 5 | 216 | 6.54 | 90.1 | 0 | False/False | 1 | True | 0 |
| tcf.20250912.mp3 | completed | 1924 | -26.1 | E2_0.2.x | 5358 | 5585 | 0.0424 | 2.785 | 2.903 | 2.949 | 0.996 | 0 | 299 | 6.31 | 110.1 | 0 | False/False | 0 | True | 0 |
| tcf.20201002.mp3 | completed | 505 | -27.0 | E1_0.1.5 | 1395 | 1455 | 0.0430 | 2.762 | 2.881 | 2.922 | 0.984 | 0 | 77 | 6.36 | 60.1 | 0 | False/False | 0 | True | 0 |
| tcf.20200417.mp3 | completed | 1540 | -21.0 | E1_0.1.5 | 4079 | 4261 | 0.0446 | 2.649 | 2.767 | 2.876 | 1.021 | 0 | 215 | 7.04 | 90.1 | 0 | False/False | 0 | True | 0 |
| women_retreat_2026_session2.mp3 | completed | 1974 | -18.4 | E3_0.3.x | 4741 | 4977 | 0.0498 | 2.402 | 2.521 | 2.668 | 1.026 | 0 | 250 | 7.66 | 110.1 | 0 | False/False | 1 | True | 0 |
| women_retreat_2025_session3.mp3 | completed | 3387 | -28.4 | E2_0.2.x | 8468 | 8916 | 0.0529 | 2.500 | 2.632 | 2.760 | 1.021 | 0 | 435 | 7.58 | 150.2 | 0 | False/False | 1 | True | 0 |
| tcf.20200925.mp3 | completed | 489 | -26.6 | E1_0.1.5 | 1195 | 1286 | 0.0761 | 2.444 | 2.630 | 2.681 | 0.895 | 0 | 123 | 3.49 | 70.1 | 0 | False/False | 0 | True | 0 |
| tcf.20200622.mp3 | completed | 396 | -20.4 | E1_0.1.5 | 899 | 1027 | 0.1424 | 2.270 | 2.593 | 2.635 | 0.899 | 0 | 89 | 3.94 | 60.1 | 0 | False/False | 0 | True | 0 |
| tcf.20200926.mp3 | completed | 400 | -22.4 | E1_0.1.5 | 885 | 1094 | 0.2362 | 2.212 | 2.735 | 2.781 | 0.909 | 0 | 89 | 4.02 | 60.1 | 0 | False/False | 1 | True | 0 |
| tcf.20200626b.mp3 | completed | 342 | -18.7 | E1_0.1.5 | 719 | 965 | 0.3421 | 2.102 | 2.822 | 2.885 | 0.982 | 0 | 53 | 6.20 | 60.1 | 0 | False/False | 0 | True | 0 |
| tcf.20240326b.mp3 | completed | 1194 | -17.8 | unknown | - | 2753 | - | - | 2.306 | 2.764 | 0.882 | 4 | 265 | 3.32 | 90.1 | 0 | False/False | 0 | True | 0 |
| tcf.20240514.mp3 | completed | 2226 | -17.6 | unknown | - | 6167 | - | - | 2.770 | 2.857 | 1.004 | 0 | 334 | 6.49 | 110.2 | 0 | False/False | 2 | True | 0 |

