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

# Copy application files
COPY . .

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
