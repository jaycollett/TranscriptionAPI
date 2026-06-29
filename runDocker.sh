#!/bin/bash

# Hugging Face token, required at build time to download the pyannote model.
# Provide it via the environment (do NOT hardcode secrets here):
#   export HUGGINGFACE_TOKEN=hf_xxx   (or put it in ~/.hf_token)
if [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ -f "$HOME/.hf_token" ]; then
    HUGGINGFACE_TOKEN="$(cat "$HOME/.hf_token")"
fi
export HUGGINGFACE_TOKEN

# Stop any existing containers
echo "Stopping existing containers..."
docker stop transcription-api 2>/dev/null || true
docker rm transcription-api 2>/dev/null || true

# Build the Docker image with Hugging Face token
echo "Building Docker image..."
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN environment variable is required for building"
    echo "Please set it with: export HUGGINGFACE_TOKEN=your_token_here"
    exit 1
fi
docker build --build-arg HUGGINGFACE_TOKEN_BUILD="$HUGGINGFACE_TOKEN" -t transcription-api .

# Run the Docker container with improved settings
echo "Starting container..."
docker run -dit \
  --name transcription-api \
  --gpus="device=0" \
  --restart unless-stopped \
  -v ./tmp:/tmp \
  -p 5030:5000 \
  -e MODEL="large-v3-turbo" \
  -e UPLOAD_FOLDER="/tmp/audio_files" \
  transcription-api:latest

echo "Container started!"
echo "API is available at http://localhost:5030"
echo "Use the testIt.py script to test transcription"
