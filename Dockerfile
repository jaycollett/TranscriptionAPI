FROM mmcauliffe/montreal-forced-aligner:latest

# Switch to root to install packages
USER root

# Set environment variables for CUDA and PyTorch
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_LAUNCH_BLOCKING=1 \
    MFA_MODEL_PATH="/mfa/pretrained_models"

# Add NVIDIA package repository for cuDNN
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb

# Install system dependencies and NVIDIA CUDA libraries
RUN apt-get update && apt-get install -y \
    wget gnupg cmake git sox ffmpeg unzip \
    libsndfile1 libsndfile1-dev \
    cuda-toolkit-12-2 libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Ensure NVIDIA paths are available
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64"

# Create MFA model directory if missing
RUN mkdir -p ${MFA_MODEL_PATH}

# Download MFA pre-trained models directly to the correct location
RUN mfa model download acoustic english_mfa && \
    mfa model download dictionary english_mfa

# Ensure the models exist
RUN ls -lah ${MFA_MODEL_PATH}

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Accept Hugging Face token as build argument (only used during build)
ARG HUGGINGFACE_TOKEN_BUILD
ENV HUGGINGFACE_TOKEN_BUILD=${HUGGINGFACE_TOKEN_BUILD}

# Download pyannote models to avoid runtime Hugging Face token requirement
RUN python -c "import os; from pyannote.audio import Pipeline; from huggingface_hub import snapshot_download; os.makedirs('/app/models/pyannote', exist_ok=True); hf_token = os.environ.get('HUGGINGFACE_TOKEN_BUILD'); assert hf_token, 'HUGGINGFACE_TOKEN build argument is required'; snapshot_download(repo_id='pyannote/speaker-diarization-3.1', cache_dir='/app/models/pyannote', token=hf_token); print('Pyannote models downloaded successfully')"

# Clear the build token environment variable for security
ENV HUGGINGFACE_TOKEN_BUILD=""

# Download Whisper models to cache them
RUN python -c "from faster_whisper import WhisperModel; import os; os.makedirs('/app/models/whisper', exist_ok=True); model = WhisperModel('large-v3-turbo', download_root='/app/models/whisper'); print('Whisper models downloaded successfully')"

# Copy application files
COPY . .


# Expose port (internal port 5000, mapped to external 5030)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
