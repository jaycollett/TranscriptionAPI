# The retention guard against a contiguous-content rule

`P` is the longest run the primary holds, the rescue lacks, and the legacy archive
corroborates. `R` is the same the other way round. The proposal refuses the rescue
when `P >= N` and otherwise applies the shipped ordering unchanged.

| file | primary | rescue | lost | retention | P | R | evidence | shipped | N=8 | N=12 | N=20 | N=30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcf.20150424.mp3 | 7060 | 6981 | 79 | fails | 4 | 40 | fidelity, different draw | primary | primary | primary | primary | primary |
| tcf.20210210.mp3 | 1172 | 1247 | -75 | ok | 0 | 73 | corpus pair | rescue | rescue | rescue | rescue | rescue |
| tcf.20210217.mp3 | 1591 | 1683 | -92 | ok | 0 | 91 | corpus pair | rescue | rescue | rescue | rescue | rescue |
| tcf.20210604.mp3 | 4497 | 4545 | -48 | ok | 0 | 56 | corpus pair | rescue | rescue | rescue | rescue | rescue |
| tcf.20211203.mp3 | 9996 | 9700 | 296 | fails | 4 | 59 | fidelity, different draw | primary | primary | primary | primary | primary |
| tcf.20221203.mp3 | 9695 | 9575 | 120 | fails | 5 | 25 | fidelity, different draw | primary | primary | primary | primary | primary |
| tcf.20240326b.mp3 | 2758 | 2837 | -79 | ok | 0 | 0 | corpus pair | rescue | rescue | rescue | rescue | rescue |
| tcf.20240412.mp3 | 3357 | 3391 | -34 | ok | 0 | 24 | corpus pair | rescue | rescue | rescue | rescue | rescue |
| tcf.20240621.mp3 | 4417 | 4562 | -145 | ok | 0 | 86 | corpus pair | rescue | rescue | rescue | rescue | rescue |
| tcf.20240713.mp3 | 6531 | 6483 | 48 | fails | 0 | 27 | matched pair | primary | primary | primary | primary | primary |
| tcf.20241105.mp3 | 7786 | 7442 | 344 | fails | 69 | 8 | matched pair | primary | primary | primary | primary | primary |
| tcf.20250607.mp3 | 7815 | 7475 | 340 | fails | 4 | 156 | fidelity, different draw | primary | primary | primary | primary | primary |
| tcf.20260428.mp3 | 13470 | 13321 | 149 | fails | 35 | 34 | matched pair | primary | primary | primary | primary | primary |
| tcf.20260626.mp3 | 3735 | 3753 | -18 | ok | 49 | 70 | corpus pair | rescue | primary | primary | primary | primary |
| tcf.202606623b.mp3 | 1306 | 1351 | -45 | ok | 20 | 70 | corpus pair | rescue | primary | primary | primary | rescue |
| tcf.20260717.mp3 | 3700 | 3767 | -67 | ok | 0 | 66 | corpus pair | rescue | rescue | rescue | rescue | rescue |
| tcf.20260724.mp3 | 4425 | 4458 | -33 | ok | 9 | 42 | corpus pair | rescue | primary | rescue | rescue | rescue |
| ucf20211106b.mp3 | 5050 | 4984 | 66 | fails | 0 | 64 | fidelity, different draw | primary | primary | primary | primary | primary |
| women_retreat_2025_session1.mp3 | 2245 | 2217 | 28 | fails | 24 | 26 | fidelity, different draw | primary | primary | primary | primary | primary |
| women_retreat_2026_session1.mp3 | 7886 | 8017 | -131 | ok | 0 | 88 | corpus pair | rescue | rescue | rescue | rescue | rescue |

## Every rule at every threshold

`proposed` is the rule as written. `net_content` adds `P > R`, so a rescue that
trades one corroborated passage for a longer one is not refused. `content_first`
also ranks corroborated contiguous content ahead of raw words in the ordering.

- **proposed, N=8** changes 3 of 20: 0 rescues gained, 3 lost (tcf.20260626.mp3, tcf.202606623b.mp3, tcf.20260724.mp3)
- **proposed, N=12** changes 2 of 20: 0 rescues gained, 2 lost (tcf.20260626.mp3, tcf.202606623b.mp3)
- **proposed, N=20** changes 2 of 20: 0 rescues gained, 2 lost (tcf.20260626.mp3, tcf.202606623b.mp3)
- **proposed, N=30** changes 1 of 20: 0 rescues gained, 1 lost (tcf.20260626.mp3)

- **net_content, N=8** changes 0 of 20: 0 rescues gained, 0 lost
- **net_content, N=12** changes 0 of 20: 0 rescues gained, 0 lost
- **net_content, N=20** changes 0 of 20: 0 rescues gained, 0 lost
- **net_content, N=30** changes 0 of 20: 0 rescues gained, 0 lost

- **content_first, N=8** changes 5 of 20: 5 rescues gained (tcf.20221203.mp3, tcf.20240713.mp3, tcf.20250607.mp3, ucf20211106b.mp3, women_retreat_2025_session1.mp3), 0 lost
- **content_first, N=12** changes 5 of 20: 5 rescues gained (tcf.20221203.mp3, tcf.20240713.mp3, tcf.20250607.mp3, ucf20211106b.mp3, women_retreat_2025_session1.mp3), 0 lost
- **content_first, N=20** changes 5 of 20: 5 rescues gained (tcf.20221203.mp3, tcf.20240713.mp3, tcf.20250607.mp3, ucf20211106b.mp3, women_retreat_2025_session1.mp3), 0 lost
- **content_first, N=30** changes 5 of 20: 5 rescues gained (tcf.20221203.mp3, tcf.20240713.mp3, tcf.20250607.mp3, ucf20211106b.mp3, women_retreat_2025_session1.mp3), 0 lost

## What each flip costs or buys

| file | direction | P | the primary-only run | R | the rescue-only run |
|---|---|---|---|---|---|
| tcf.20260626.mp3 | rescue to primary at N=8,12,20,30 | 49 | so if there is any encouragement in christ any comfort from love any participation in the spirit any affection and sympathy complete my joy  | 70 | have this mind among yourselves which is yours in christ jesus who though he was in the form of god did not count equality with god a thing  |
| tcf.202606623b.mp3 | rescue to primary at N=8,12,20 | 20 | for what we proclaim is not ourselves but jesus christ as lord with ourselves as your servants for jesus sake | 70 | when the apostles saw that i had been entrusted with the gospel to the uncircumcised just as peter had been entrusted with the gospel to the |
| tcf.20260724.mp3 | rescue to primary at N=8 | 9 | think it makes more sense let your gentleness be | 42 | do not be anxious about anything but in everything by prayer and supplication with thanksgiving let your requests be made known to god and t |

