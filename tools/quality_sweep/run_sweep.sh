#!/usr/bin/env bash
# Start the isolated sweep service and run the file list against it.
#
# Every guard here exists because the sweep shares a host with production. The script
# refuses to run if the container name or the port collide with production, checks that
# GPU 0 has headroom before loading a second Whisper model, and mounts only the sweep's
# own directory. It never stops, restarts or talks to the transcription-api container.
#
#   ./run_sweep.sh 0.6.0                 # start the service and run the sweep
#   ./run_sweep.sh 0.6.0 --keep-running  # leave the container up afterwards
#
# The runner is resumable, so re-running after an interruption picks up where it stopped.
set -euo pipefail

IMAGE_TAG="${1:?usage: run_sweep.sh <image tag> [--keep-running]}"
KEEP="${2:-}"

IMAGE="transcription-api:${IMAGE_TAG}"
CONTAINER="transcription-api-sweep"
PORT=5031
SWEEP_ROOT=/home/jay/sweep
SERVICE_DIR="${SWEEP_ROOT}/service"
OUT_DIR="${SWEEP_ROOT}/run"
REPEAT_DIR="${SWEEP_ROOT}/repeat"
AUDIO_DIR=/home/jay/SourceCode/SermonPreprocessorAPI/data/audiofiles

# Determinism probe. The decode is not bit-reproducible on longer files, and the rescue
# trigger is a threshold on anomaly counts, so a run-to-run flip could change whether a
# file takes the rescue path. These five span 238 s to 3388 s and include the 1369 s file
# the engineer measured and the quiet file on the 0.35 VAD branch. Each is submitted once
# more here; the pair is the main run's result against this one.
REPEAT_FILES=(
    tcf.20240301a.mp3
    tcf.20240213b.mp3
    tcf.20240319b.mp3
    tcf.20240116.mp3
    women_retreat_2025_session3.mp3
)
MIN_GPU_FREE_MIB=4096
MIN_DISK_FREE_GB=8
HEALTH_TIMEOUT_SEC=300

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Isolation guards. Production is transcription-api on 5030; nothing here may touch it.
if [ "${CONTAINER}" = "transcription-api" ] || [ "${PORT}" = "5030" ]; then
    echo "refusing to run: the sweep would collide with production" >&2
    exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "image ${IMAGE} not found on this host" >&2
    exit 1
fi

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)"
if [ "${free_mib}" -lt "${MIN_GPU_FREE_MIB}" ]; then
    echo "GPU 0 has only ${free_mib} MiB free, under the ${MIN_GPU_FREE_MIB} MiB floor." >&2
    echo "Production is mid-job. Wait and re-run rather than starting a second model." >&2
    exit 1
fi
free_gb="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
if [ "${free_gb}" -lt "${MIN_DISK_FREE_GB}" ]; then
    echo "only ${free_gb} GB free on /, under the ${MIN_DISK_FREE_GB} GB floor" >&2
    exit 1
fi
echo "GPU 0 free: ${free_mib} MiB. Disk free: ${free_gb} GB."

mkdir -p "${SERVICE_DIR}/uploads" "${OUT_DIR}" "${REPEAT_DIR}"

# A private MFA_ROOT_DIR has to be seeded with the image's pretrained models. MFA
# resolves `english_mfa` under $MFA_ROOT_DIR/pretrained_models, so an empty root makes
# every alignment fail with "Could not find a model named english_mfa" and the service
# falls back to Whisper timings without erroring. Copy the models out of the image once.
if [ ! -f "${SERVICE_DIR}/mfa_root/pretrained_models/acoustic/english_mfa.zip" ]; then
    echo "seeding ${SERVICE_DIR}/mfa_root from ${IMAGE}..."
    mkdir -p "${SERVICE_DIR}/mfa_root"
    seeder="$(docker create "${IMAGE}")"
    docker cp "${seeder}:/mfa/pretrained_models" "${SERVICE_DIR}/mfa_root/pretrained_models"
    docker cp "${seeder}:/mfa/global_config.yaml" "${SERVICE_DIR}/mfa_root/global_config.yaml" 2>/dev/null || true
    docker rm -f "${seeder}" >/dev/null
fi

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run -dit \
    --name "${CONTAINER}" \
    --gpus="device=0" \
    -p "${PORT}:5000" \
    -e MODEL="large-v3-turbo" \
    -e DB_FILE="/data/sweep.db" \
    -e UPLOAD_FOLDER="/data/uploads" \
    -e MFA_ROOT_DIR="/data/mfa_root" \
    -v "${SERVICE_DIR}:/data" \
    "${IMAGE}" >/dev/null

cleanup() {
    if [ "${KEEP}" != "--keep-running" ]; then
        docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
        echo "removed ${CONTAINER}"
    fi
}
trap cleanup EXIT

deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SEC ))
while :; do
    if body="$(curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null)"; then
        echo "sweep service healthy: ${body}"
        break
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
        echo "sweep service did not come up in ${HEALTH_TIMEOUT_SEC}s" >&2
        docker logs --tail 40 "${CONTAINER}" >&2 || true
        exit 1
    fi
    sleep 5
done

python3 "${HERE}/runner.py" \
    --file-list "${HERE}/file_list.json" \
    --audio-dir "${AUDIO_DIR}" \
    --base-url "http://127.0.0.1:${PORT}" \
    --out-dir "${OUT_DIR}" \
    --poll-interval 10 \
    --scratch-dir "${SERVICE_DIR}/uploads" \
    --scratch-dir "${SERVICE_DIR}/mfa_root"

repeat_args=()
for name in "${REPEAT_FILES[@]}"; do
    repeat_args+=(--only "${name}")
done
python3 "${HERE}/runner.py" \
    --file-list "${HERE}/file_list.json" \
    --audio-dir "${AUDIO_DIR}" \
    --base-url "http://127.0.0.1:${PORT}" \
    --out-dir "${REPEAT_DIR}" \
    --poll-interval 10 \
    --no-supplementary \
    --scratch-dir "${SERVICE_DIR}/uploads" \
    --scratch-dir "${SERVICE_DIR}/mfa_root" \
    "${repeat_args[@]}"

python3 "${HERE}/analyze.py" \
    --results "${OUT_DIR}/state.json" \
    --baseline "${SWEEP_ROOT}/legacy_baseline.json" \
    --file-list "${HERE}/file_list.json" \
    --out-json "${SWEEP_ROOT}/analysis.json" \
    --out-md "${SWEEP_ROOT}/report.md"

# The container runs as root, so the uploaded copies and the _aligned directories it
# leaves behind are root-owned inside root-owned directories and the runner (running as
# the login user) cannot unlink them. Clear them with a throwaway container on the same
# volume. Only the sweep's own directory is mounted, and pretrained_models is kept.
docker run --rm -v "${SERVICE_DIR}:/data" --entrypoint sh "${IMAGE}" -c '
    rm -rf /data/uploads/* /data/mfa_root/*_mfa_input /data/mfa_root/extracted_models            /data/mfa_root/joblib_cache
' || true
du -sh "${SERVICE_DIR}"
df -h / | tail -1
