# Legacy single-pass against multi-pass words per second

Observational. Era is a proxy for date, so speaker, room and encoder drift with it and no file was ever decoded by both. Not evidence about decoders.

## Raw

| arm | n | mean | p5 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|---|---|
| single pass (E1, 0.1.5) | 531 | 2.759 | 2.507 | 2.674 | 2.766 | 2.861 | 2.972 |
| multi pass (E2 and E3, 0.2.x and 0.3.x) | 112 | 2.622 | 2.412 | 2.514 | 2.62 | 2.728 | 2.875 |

Median difference (single minus multi): 0.1461 words per second. The arms are not alike to begin with: median duration 804.0 s against 1627.0 s, median level -20.7 dBFS against -21.3 dBFS.

## Stratified by duration and level

| duration | level | single n | multi n | single median | multi median | difference |
|---|---|---|---|---|---|---|
| long | cutover_high | 8 | 1 | 2.725 | 2.774 | too few |
| long | cutover_low | 2 | 7 | 2.767 | 2.692 | too few |
| long | loud | 29 | 7 | 2.791 | 2.582 | 0.209 |
| long | normal | 53 | 14 | 2.804 | 2.677 | 0.127 |
| long | quiet | 1 | 2 | 2.604 | 2.638 | too few |
| medium | cutover_high | 8 | 13 | 2.735 | 2.626 | 0.109 |
| medium | cutover_low | 4 | 11 | 2.76 | 2.505 | 0.255 |
| medium | loud | 79 | 4 | 2.747 | 2.618 | 0.129 |
| medium | normal | 71 | 34 | 2.767 | 2.616 | 0.151 |
| medium | quiet | 0 | 4 | None | 2.352 | too few |
| medium | very_quiet | 0 | 1 | None | 2.666 | too few |
| short | cutover_high | 59 | 1 | 2.746 | 2.432 | too few |
| short | cutover_low | 7 | 0 | 2.777 | None | too few |
| short | loud | 16 | 3 | 2.769 | 2.861 | -0.092 |
| short | normal | 179 | 4 | 2.773 | 2.555 | 0.218 |
| short | quiet | 8 | 1 | 2.775 | 2.539 | too few |
| very_long | cutover_high | 0 | 2 | None | 2.57 | too few |
| very_long | cutover_low | 0 | 1 | None | 2.62 | too few |
| very_long | loud | 2 | 0 | 2.667 | None | too few |
| very_long | normal | 3 | 1 | 2.265 | 2.677 | too few |
| very_long | quiet | 2 | 0 | 2.305 | None | too few |
| very_long | very_quiet | 0 | 1 | None | 2.5 | too few |

Pooled over 8 cells with at least 3 files in each arm, weighted by the smaller arm: 0.1459 words per second. The single-pass arm is higher in 7 of those 8 cells.

