#!/usr/bin/env bash
# Measure mean and max level in dBFS plus stream properties for every reference file.
# CPU only, niced, so it cannot compete with production transcription on the GPU.
set -u
AUDIO_DIR=/home/jay/SourceCode/SermonPreprocessorAPI/data/audiofiles
OUT=/home/jay/sweep/levels.csv
TMP=/home/jay/sweep/levels.partial
: > "$TMP"
measure() {
  f="$1"
  base=$(basename "$f")
  vd=$(nice -n 19 ffmpeg -nostdin -hide_banner -vn -i "$f" -af volumedetect -f null - 2>&1)
  mean=$(printf '%s\n' "$vd" | sed -n 's/.*mean_volume: \(-\?[0-9.]*\) dB.*/\1/p' | head -1)
  max=$(printf '%s\n' "$vd" | sed -n 's/.*max_volume: \(-\?[0-9.]*\) dB.*/\1/p' | head -1)
  pr=$(nice -n 19 ffprobe -v error -select_streams a:0 \
        -show_entries stream=sample_rate,channels,bit_rate,codec_name \
        -of default=nw=1:nk=1 "$f" | paste -sd, -)
  printf '%s\t%s\t%s\t%s\n' "$base" "${mean:-NA}" "${max:-NA}" "$pr" >> "$TMP"
}
export -f measure
export TMP
find "$AUDIO_DIR" -maxdepth 1 -type f | sort | xargs -P 6 -I{} bash -c 'measure "$@"' _ {}
sort "$TMP" > "$OUT"
echo "DONE $(wc -l < "$OUT")" > /home/jay/sweep/levels.done
