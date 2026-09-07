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
# 2026-09-07 (0.5.2): two of the deferred items above are now done.
#   * cuda-toolkit-12-2 -> cuda-libraries-12-2. The toolkit meta-package pulled in
#     cuda-compiler, cuda-tools (Nsight Systems, whose bundled Go binary carried
#     64 of the 80 Trivy findings on the 0.4.0 image), cuda-libraries-dev and
#     cuda-documentation. Nothing here compiles CUDA: torch ships its own CUDA 13
#     libraries via pip, and ctranslate2 dlopens libcudart/libcublas 12 and
#     libcudnn 9 at runtime, all of which cuda-libraries-12-2 plus
#     libcudnn9-cuda-12 provide under /usr/local/cuda/lib64 and /usr/lib.
#     cuda-runtime-12-2 was NOT used: apt-cache shows it depends on cuda-drivers,
#     which would install the NVIDIA driver inside the container.
#   * CUDA_LAUNCH_BLOCKING=1 removed. It serialises every kernel launch and is a
#     debugging setting; it cost throughput on all five Whisper passes.
# Still deferred: a non-root USER (the MFA base runs as root and writes to /mfa).
FROM mmcauliffe/montreal-forced-aligner:v3.4.2

# Switch to root to install packages
USER root

# Set environment variables for CUDA and PyTorch
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
    cuda-libraries-12-2 libcudnn9-cuda-12 \
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

# Liveness. /health answers 503 when the worker thread is dead or an idle
# worker has stopped polling. The image has wget (used above) but no curl. The
# start period covers loading the Whisper model, which happens before the
# worker thread and the HTTP server exist.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:5000/health || exit 1

# Run the Flask app
CMD ["python", "app.py"]
