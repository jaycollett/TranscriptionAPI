#!/bin/bash

# Stop any existing containers
echo "Stopping existing containers..."
docker stop transcription-api 2>/dev/null || true
docker rm transcription-api 2>/dev/null || true

# Build the Docker image
echo "Building Docker image..."
docker build -t transcription-api .

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
