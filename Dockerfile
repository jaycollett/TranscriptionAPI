# GPU transcription/alignment service.
#
# The base image is Montreal Forced Aligner, which carries its own conda Python
# plus the Kaldi/OpenFST stack, and CUDA is a runtime requirement rather than a
# build-time one. Neither Alpine nor a multi-stage venv split applies here: the
# application runs inside the base image's interpreter.
#
# 2026-07-29 CVE pass. What changed:
#   * base tag pinned `:latest` -> `:v3.4.1`. An unpinned :latest means the
#     image that was scanned is not necessarily the image that was deployed,
#     and it makes every rebuild a silent, untested base upgrade.
#   * the cuda-keyring .deb is now fetched, installed and deleted inside a
#     single layer, so the .deb never persists in the image.
#   * the -dev packages (libsndfile1-dev, libcudnn9-dev-cuda-12) are gone.
#     They ship C headers and static archives; nothing in this image compiles
#     against either one, so they were pure scan surface.
#
# Deliberately NOT changed, because this image cannot be built or started on
# the hardening workstation (it needs linux/amd64 + an NVIDIA GPU, and the
# build used to require a Hugging Face token). Each of these is worth doing on a
# machine that can actually verify it:
#   * cuda-toolkit-12-2 -> cuda-runtime-12-2 + cuda-libraries-12-2. The full
#     toolkit ships nvcc, the profilers and the samples; ctranslate2 and
#     faster-whisper use prebuilt kernels and should not need any of it. This
#     is the single largest remaining CVE contributor in the image.
#   * a non-root USER. The MFA base runs as root and writes to /mfa and the
#     model cache at runtime; switching users needs a real GPU run to confirm.
#
# 2026-09-07: speaker diarization (pyannote) was removed, and with it the
# HUGGINGFACE_TOKEN_BUILD build arg and the gated-model download. The build
# now needs no secret at all, so CI can rebuild the image unattended.
FROM mmcauliffe/montreal-forced-aligner:v3.4.1

# Switch to root to install packages
USER root

# Set environment variables for CUDA and PyTorch
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_LAUNCH_BLOCKING=1 \
    MFA_MODEL_PATH="/mfa/pretrained_models"

# Add the NVIDIA package repository, then install the system dependencies and
# the CUDA/cuDNN libraries in one layer, so neither the keyring .deb nor the
# apt lists survive into an image layer.
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg cmake git sox ffmpeg unzip \
    libsndfile1 \
    cuda-toolkit-12-2 libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Ensure NVIDIA paths are available
ENV PATH="/usr/local/cuda/bin:${PATH}"
# The conda environment's libstdc++ must come first. The v3.4.1 base ships numpy
# 2.4 built against GLIBCXX_3.4.29, which Ubuntu 20.04's system libstdc++ does not
# provide; without /env/lib on the path `import numpy` (and therefore torch) fails
# at startup. Verified 2026-09-07 on the GPU host: with this order torch still
# reports CUDA available and ctranslate2 still sees the device.
ENV LD_LIBRARY_PATH="/env/lib:/usr/local/cuda/lib64"

# Create the MFA model directory and download the pre-trained models into it.
RUN mkdir -p ${MFA_MODEL_PATH} && \
    mfa model download acoustic english_mfa && \
    mfa model download dictionary english_mfa && \
    ls -lah ${MFA_MODEL_PATH}

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Whisper models to cache them
RUN python -c "from faster_whisper import WhisperModel; import os; os.makedirs('/app/models/whisper', exist_ok=True); model = WhisperModel('large-v3-turbo', download_root='/app/models/whisper'); print('Whisper models downloaded successfully')"

# Copy application files
COPY . .


# Expose port (internal port 5000, mapped to external 5030)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
