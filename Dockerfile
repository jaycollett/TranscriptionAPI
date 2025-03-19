# Use the MFA base image
FROM mmcauliffe/montreal-forced-aligner:latest

# Switch to root to install packages
USER root

# Set environment variables for CUDA and PyTorch
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_LAUNCH_BLOCKING=1 \
    MFA_MODEL_PATH="/mfa/pretrained_models" \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64" \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_PATH="/usr/local/cuda" \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    CUDA_MODULE_LOADING=LAZY \
    CUDA_CACHE_PATH=/tmp/cuda-cache \
    CUDA_CACHE_MAXSIZE=4294967296

# Install basic system dependencies first
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Add NVIDIA package repository for cuDNN
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb
# Install system dependencies and NVIDIA CUDA libraries
RUN apt-get update && apt-get install -y \
    wget gnupg cmake git sox ffmpeg unzip \
    cuda-toolkit-12-2 libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Install remaining system dependencies and NVIDIA CUDA libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    sox \
    ffmpeg \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /usr/local/cuda/doc \
    && rm -rf /usr/local/cuda/compute-sanitizer \
    && rm -rf /usr/local/cuda/extras \
    && rm -rf /usr/local/cuda/nsight-* \
    && rm -rf /usr/local/cuda/tools \
    && rm -rf /usr/local/cuda/samples

# Create MFA model directory if missing
RUN mkdir -p ${MFA_MODEL_PATH}

# Download MFA pre-trained models directly to the correct location
RUN mfa model download acoustic english_mfa && \
    mfa model download dictionary english_mfa && \
    rm -rf /root/.cache/mfa

# Ensure the models exist
RUN ls -lah ${MFA_MODEL_PATH}

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip/*

# Copy application files
COPY . .

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Set default environment variables
ENV API_HOST=0.0.0.0 \
    API_PORT=5000 \
    UPLOAD_FOLDER=/data/audio_files \
    DB_FILE=/data/transcriptions.db \
    MAX_FILE_SIZE=524288000 \
    MAX_DB_CONNECTIONS=5 \
    MAX_RETRIES=3 \
    RETRY_DELAY=5 \
    DB_TIMEOUT=30 \
    CLEANUP_AGE_DAYS=1 \
    CLEANUP_BATCH_SIZE=100 \
    CLEANUP_INTERVAL=3600 \
    PROCESSING_SPEED_FACTOR=15.1 \
    WHISPER_MODEL=large-v3-turbo \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE_TYPE=float32 \
    DEBUG=false

# Create data directory
RUN mkdir -p /data/audio_files

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
