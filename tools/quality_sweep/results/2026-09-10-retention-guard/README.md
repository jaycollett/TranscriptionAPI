# Retention guard analysis, 2026-09-10

Per-recording deficit reports for the refused rescues of the 0.6.1 corpus run, produced by
`tools/quality_sweep/rescue_deficit.py`.

`fidelity-sourced/` is built from the 32-file fidelity experiment: the primary is `A.json`,
0.6.0 as shipped, a deterministic beam search from temperature 0.0 and therefore the same
primary the 0.6.1 run published; the rescue is `B.json`, a whole-file decode with the ladder
starting at 0.2, which is the rescue configuration but **a different unseeded draw** from the
one that was refused. These reports answer whether a rescue-configuration decode drops
corroborated content the primary holds. They are not the exact transcript that was discarded.

`redecoded/` is built from a decode-only re-run that keeps both passes of one decode, so its
primary and rescue are the matched pair.

Status is tracked in `docs/OPEN_ANALYSIS_RETENTION_GUARD.md`.
