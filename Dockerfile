FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install torch with CUDA 12.4 first (heavy layer, cache it separately)
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu124

# Install Whisper and server dependencies
RUN pip3 install \
        openai-whisper \
        fastapi \
        "uvicorn[standard]" \
        python-multipart

WORKDIR /app
COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
