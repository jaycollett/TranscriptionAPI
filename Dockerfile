# Use NVIDIA CUDA runtime with Ubuntu 22.04 as the base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies (including Python and FFmpeg)
RUN apt-get update && \
    apt-get install -y python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install application dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py /app/
COPY transcribe.py /app/
COPY init_db.sql /app/

# Expose the Flask port
EXPOSE 5000


ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
ENV CUDA_LAUNCH_BLOCKING=0

# Start Flask with the worker inside the same container
CMD ["python3", "app.py"]
