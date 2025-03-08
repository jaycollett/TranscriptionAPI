FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set environment variables to ensure proper CUDA execution and reduce VRAM fragmentation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_LAUNCH_BLOCKING=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA package repository for cuDNN
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb

# Install latest available cuDNN 9 for CUDA 12
RUN apt-get update && apt-get install -y \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements.txt *before* copying the rest of the app (improves caching)
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files *after* installing dependencies (improves caching)
COPY . .

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
