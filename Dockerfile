FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/nvidia/npp/lib"

# Only audio bits — Python is in the base image. CUDA runtime libs come from
# nvidia-* pip packages installed alongside torch; libcuda.so.1 (driver) is
# injected at runtime via AddDevice=nvidia.com/gpu=all in the Quadlet unit.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. UID 5000 matches the host owner of mounted volumes when the
# Quadlet unit uses `UserNS=keep-id`. App lives in /app, venv in /opt/venv,
# caches in /home/whisper/.cache.
RUN groupadd --gid 5000 whisper && \
    useradd --uid 5000 --gid 5000 --create-home --shell /bin/bash whisper && \
    python3 -m venv "$VIRTUAL_ENV" && \
    mkdir -p /app /home/whisper/.cache && \
    chown -R whisper:whisper "$VIRTUAL_ENV" /app /home/whisper/.cache

USER whisper
WORKDIR /app

RUN pip install --upgrade pip

# torch from the cu128 index. Wheels declare nvidia-* CUDA runtime packages as
# pip dependencies — they get installed into site-packages/nvidia/* and torch
# adds them to its dlopen path at import time.
# WhisperX → pyannote.audio 4.x → torchcodec 0.7 pins torch ABI to 2.8.*.
RUN pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# Everything else (whisperx, faster-whisper, pyannote.audio, ctranslate2,
# fastapi, uvicorn, ...) is locked in requirements.txt.
COPY --chown=whisper:whisper requirements.txt .
RUN pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r requirements.txt

COPY --chown=whisper:whisper server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
