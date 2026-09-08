#!/usr/bin/env bash
# Measure where each reference file is not silent, independently of the service.
#
# The sweep needs to know how much detected speech the decoder emitted nothing for, and
# in particular the largest single contiguous stretch it missed, because a real omission
# is contiguous while breathing is scattered. The service's own coverage number cannot
# answer the second question and is derived from the same VAD the decode used, so this
# measures the audio directly with ffmpeg's silencedetect and hands the analyzer an
# independent set of non-silent intervals to intersect with the emitted segment spans.
#
# The threshold is set per file relative to that file's own mean level, because the
# archive spans -30 to -12 dBFS and a fixed threshold would call a quiet recording silent
# throughout. CPU only and niced, so it cannot compete with the sweep on the GPU.
#
#   ./silence.sh file_list.json levels.csv /home/jay/sweep/silence.jsonl
set -u
LIST="${1:?usage: silence.sh <file_list.json> <levels.csv> <out.jsonl>}"
LEVELS="${2:?}"
OUT="${3:?}"
AUDIO_DIR=/home/jay/SourceCode/SermonPreprocessorAPI/data/audiofiles
# How far below a file's mean level counts as silence, and the shortest gap worth having.
BELOW_MEAN_DB=14
MIN_SILENCE_S=0.4

python3 - "$LIST" "$LEVELS" > /tmp/silence_targets.txt <<'PY'
import json, sys
names = []
with open(sys.argv[1]) as handle:
    doc = json.load(handle)
for row in doc["files"] + doc.get("supplementary", []):
    names.append(row["file"])
levels = {}
with open(sys.argv[2]) as handle:
    for line in handle:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[1] != "NA":
            levels[parts[0]] = float(parts[1])
for name in names:
    print(f"{name}\t{levels.get(name, -20.0)}")
PY

: > "$OUT"
measure() {
    name="$1"; mean="$2"
    thresh=$(python3 -c "print(round($mean - $BELOW_MEAN_DB, 1))")
    nice -n 19 ffmpeg -nostdin -hide_banner -vn -i "${AUDIO_DIR}/${name}" \
        -af "silencedetect=noise=${thresh}dB:d=${MIN_SILENCE_S}" -f null - 2>&1 \
    | python3 -c "
import json, re, sys
name = sys.argv[1]
thresh = float(sys.argv[2])
starts, ends = [], []
for line in sys.stdin:
    m = re.search(r'silence_start: (-?[\d.]+)', line)
    if m:
        starts.append(float(m.group(1)))
    m = re.search(r'silence_end: (-?[\d.]+)', line)
    if m:
        ends.append(float(m.group(1)))
pairs = []
for i, s in enumerate(starts):
    e = ends[i] if i < len(ends) else None
    pairs.append([round(max(s, 0.0), 3), None if e is None else round(e, 3)])
print(json.dumps({'file': name, 'threshold_db': thresh, 'silences': pairs}))
" "$name" "$thresh" >> "$OUT"
}
export -f measure
export AUDIO_DIR BELOW_MEAN_DB MIN_SILENCE_S OUT

while IFS=$'\t' read -r name mean; do
    printf '%s\t%s\n' "$name" "$mean"
done < /tmp/silence_targets.txt | xargs -P 4 -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r n m <<< "{}"; measure "$n" "$m"'

echo "DONE $(wc -l < "$OUT")" > "${OUT}.done"
