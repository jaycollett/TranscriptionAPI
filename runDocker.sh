#!/bin/bash
# Build, deploy and verify the transcription service on the GPU host.
#
# The new image is built and tagged before the running container is touched, so
# a failed build leaves production alone. The image that was serving is retagged
# transcription-api:rollback first, so it can be put back with the same run
# command and that tag if the new one does not come up healthy.
set -euo pipefail

IMAGE="transcription-api"
CONTAINER="transcription-api"
VERSION="$(git describe --tags --always --dirty 2>/dev/null || date +untagged-%Y%m%d%H%M%S)"
HEALTH_URL="http://localhost:5030/health"
HEALTH_TIMEOUT_SEC=180

# The Dockerfile uses a BuildKit secret mount (RUN --mount=type=secret) for the
# MFA model download. BuildKit has been the default builder since Docker 23.0,
# but a DOCKER_BUILDKIT=0 in the calling shell would select the legacy builder,
# which rejects the --mount flag. No secret is passed here: the local build
# downloads the models unauthenticated, which is fine from a single host.
export DOCKER_BUILDKIT=1

echo "Building ${IMAGE}:${VERSION}..."
docker build -t "${IMAGE}:${VERSION}" .

# Keep the image that is serving now reachable as :rollback.
if docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    previous="$(docker inspect --format '{{.Image}}' "${CONTAINER}")"
    docker tag "${previous}" "${IMAGE}:rollback"
    echo "Previous image ${previous} retagged as ${IMAGE}:rollback"
fi
docker tag "${IMAGE}:${VERSION}" "${IMAGE}:latest"

echo "Stopping existing container..."
docker stop "${CONTAINER}" 2>/dev/null || true
docker rm "${CONTAINER}" 2>/dev/null || true

echo "Starting ${IMAGE}:${VERSION}..."
docker run -dit \
  --name "${CONTAINER}" \
  --gpus="device=0" \
  --restart unless-stopped \
  -v ./tmp:/tmp \
  -p 5030:5000 \
  -e MODEL="large-v3-turbo" \
  -e UPLOAD_FOLDER="/tmp/audio_files" \
  "${IMAGE}:${VERSION}"

echo "Waiting up to ${HEALTH_TIMEOUT_SEC}s for ${HEALTH_URL} (model load takes a while)..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SEC ))
while :; do
    if body="$(curl -sf --max-time 5 "${HEALTH_URL}" 2>/dev/null)"; then
        echo "Healthy: ${body}"
        echo "API is available at http://localhost:5030"
        echo "Running ${IMAGE}:${VERSION}; previous image is ${IMAGE}:rollback"
        echo "Use the testIt.py script to test transcription"
        exit 0
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
        echo "Service did not become healthy within ${HEALTH_TIMEOUT_SEC}s. Last container log lines:" >&2
        docker logs --tail 40 "${CONTAINER}" >&2 || true
        echo "To roll back: docker stop ${CONTAINER}; docker rm ${CONTAINER}; then rerun the docker run" >&2
        echo "command above with ${IMAGE}:rollback in place of ${IMAGE}:${VERSION}." >&2
        exit 1
    fi
    sleep 5
done
