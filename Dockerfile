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
    libsndfile1 libnghttp2-14 \
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

# 2026-09-07 (0.5.4): residual CVE pass. /opt/conda is the mambaforge bootstrap
# environment the MFA base image was built from (Python 3.10, conda 22.11). The
# service runs from /env and never imports from /opt/conda, but its 2023-era
# packages are what the release gate reports:
#   certifi 2022.12.7    CVE-2023-37920
#   cryptography 39.0.1  CVE-2023-50782, CVE-2024-26130, CVE-2026-26007,
#                        GHSA-537c-gmf6-5ccf
#   pyOpenSSL 23.0.0     CVE-2026-27459
#   setuptools 65.6.3    CVE-2024-6345, CVE-2025-47273
#   urllib3 1.26.14      CVE-2023-43804, CVE-2025-66418, CVE-2025-66471,
#                        CVE-2026-21441, CVE-2026-44431
# requests moves with urllib3 (2.28 pins urllib3<1.27) so that `pip check`
# stays clean. Only the named packages are touched, with that environment's
# own pip; nothing in /env changes here. libnghttp2-14 (CVE-2023-44487) is
# handled by the apt line above, which pulls the focal-security build.
RUN /opt/conda/bin/pip install --no-cache-dir \
        "certifi>=2023.7.22" "cryptography>=48.0.1" "pyOpenSSL>=26.0.0" \
        "setuptools>=78.1.1" "urllib3>=2.7.0" "requests>=2.32.0" && \
    /opt/conda/bin/pip check

# Create the MFA model directory and download the pre-trained models into it.
#
# `mfa model download` resolves the model through the GitHub releases API of
# MontrealCorpusTools/mfa-models. Unauthenticated, that API allows 60 requests
# per hour per source address, and the shared GitHub Actions runner pool burns
# through that, so the release build fails with ModelsConnectionError while a
# build on the GPU host succeeds. The workflow passes its GITHUB_TOKEN as the
# BuildKit secret `github_token` (1,000 requests per hour per repository); a
# local build supplies no secret, the file is absent, and the download runs
# unauthenticated as before. Secret mounts are not image layers, so the token
# is never persisted in the image or its history.
RUN --mount=type=secret,id=github_token,required=false \
    set -e; TOKEN_ARG=""; \
    if [ -s /run/secrets/github_token ]; then TOKEN_ARG="--github_token $(cat /run/secrets/github_token)"; fi; \
    mkdir -p ${MFA_MODEL_PATH} && \
    mfa model download acoustic english_mfa $TOKEN_ARG && \
    mfa model download dictionary english_mfa $TOKEN_ARG && \
    ls -lah ${MFA_MODEL_PATH}

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Whisper models to cache them
RUN python -c "from faster_whisper import WhisperModel; import os; os.makedirs('/app/models/whisper', exist_ok=True); model = WhisperModel('large-v3-turbo', download_root='/app/models/whisper'); print('Whisper models downloaded successfully')"

# 2026-09-07 (0.5.4): pip 26.2.1 in /env ships a CycloneDX inventory of the
# libraries it vendors (pip/_vendor/bom.cdx.json) and Trivy reads it, so the
# gate reports the vendored msgpack 1.1.2 (GHSA-6v7p-g79w-8964) and the
# pkg_resources copy of setuptools 70.3.0 (CVE-2025-47273). The newest pip on
# PyPI vendors the same versions, so no upgrade clears them, and /env's own
# msgpack 1.2.1 and setuptools 81 are already clean. Nothing runs pip after the
# requirements install above, so pip is removed from the runtime image, which
# takes the vendored code with it rather than only the manifest. This sits
# after the Whisper bake so the model and requirements layers stay cached.
RUN python -m pip uninstall -y pip

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
