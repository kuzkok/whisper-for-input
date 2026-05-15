FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y \
        python3 \
        python3-venv \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. UID 5000 matches the host owner of mounted volumes when the
# Quadlet unit uses `UserNS=keep-id`. App lives in /app, venv in /opt/venv,
# caches in /home/whisper/.cache.
RUN groupadd --gid 5000 whisper && \
    useradd --uid 5000 --gid 5000 --create-home --shell /bin/bash whisper && \
    python3 -m venv "$VIRTUAL_ENV" && \
    mkdir -p /app && \
    chown -R whisper:whisper "$VIRTUAL_ENV" /app

USER whisper
WORKDIR /app

RUN pip install --upgrade pip

# Pinned torch from the cu128 index — these wheels dynamically link against
# system CUDA libs (already in the base image) instead of bundling ~3 GB of
# nvidia-* packages like the PyPI variant does.
# WhisperX → pyannote.audio 4.x → torchcodec 0.7 pins torch ABI to 2.8.*.
RUN pip install torch==2.8.0 torchaudio==2.8.0 \
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
