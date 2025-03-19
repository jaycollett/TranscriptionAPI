docker stop transcription-api
docker rm transcription-api
docker build -t transcription-api .
docker run -d \
  --name transcription-api \
  -p 5030:5000 \
  --gpus="device=0" \
  transcription-api
