#!/bin/bash
# Pre-download WhisperX + pyannote models into ./models/ for offline container use.
# Reads HF token from ./secrets/hf_token.
#
# Required HF licenses (accept these on huggingface.co before running):
#   https://huggingface.co/pyannote/speaker-diarization-3.1
#   https://huggingface.co/pyannote/speaker-diarization-community-1
#   https://huggingface.co/pyannote/segmentation-3.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Models and torch-cache live outside the project tree so it can be moved
# without breaking the service. Matches the Volume= paths in the Quadlet unit
# and docker-compose.yml.
DATA_DIR="$HOME/.local/share/whisper-for-input"
MODELS_DIR="$DATA_DIR/models"
TORCH_CACHE_DIR="$DATA_DIR/torch-cache"
TOKEN_FILE="$SCRIPT_DIR/secrets/hf_token"

if [[ ! -f "$TOKEN_FILE" ]]; then
    cat >&2 <<EOF
ERROR: $TOKEN_FILE not found.

Steps:
  1. Get a token at https://huggingface.co/settings/tokens (read access is enough).
  2. Accept the user agreements:
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
  3. mkdir -p "$SCRIPT_DIR/secrets"
     echo 'hf_xxxxxxxxxxxxxxxxxxxx' > "$TOKEN_FILE"
     chmod 600 "$TOKEN_FILE"
  4. Re-run this script.
EOF
    exit 1
fi

HF_TOKEN=$(tr -d '[:space:]' < "$TOKEN_FILE")
if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: $TOKEN_FILE is empty." >&2
    exit 1
fi

export HF_HOME="$MODELS_DIR"
mkdir -p "$MODELS_DIR"

# Pre-create torch cache dir so the container can bind-mount it on first start.
# Contents (WhisperX VAD, torchaudio alignment) are populated by the container
# on first request, not by this script.
mkdir -p "$TORCH_CACHE_DIR"

if command -v hf >/dev/null; then
    HF_CLI=(hf download)
elif command -v huggingface-cli >/dev/null; then
    # Older huggingface_hub versions before the CLI was renamed to `hf`.
    HF_CLI=(huggingface-cli download)
else
    echo "Neither 'hf' nor 'huggingface-cli' found." >&2
    echo "Install with: pip install --user --upgrade 'huggingface_hub[cli]'" >&2
    exit 1
fi

download() {
    local repo="$1"
    # HF кэширует репо в $HF_HOME/hub/models--<org>--<repo>/. Если каталог
    # есть и непустой — пропускаем, чтобы не дёргать сеть HEAD-запросами.
    local cache_dir="$HF_HOME/hub/models--${repo//\//--}"
    echo
    if [[ -d "$cache_dir" && -n "$(ls -A "$cache_dir" 2>/dev/null)" ]]; then
        echo "==> $repo (уже в кэше, пропускаем)"
        return
    fi
    echo "==> $repo"
    "${HF_CLI[@]}" "$repo" --token "$HF_TOKEN"
}

# ASR backend used by WhisperX (faster-whisper / CTranslate2 weights).
# This is the repo that faster-whisper's `_MODELS` maps the `large-v3-turbo`
# alias to. The .container unit passes the full repo ID as WHISPER_MODEL so
# the runtime hits this exact cache entry.
download "dropbox-dash/faster-whisper-large-v3-turbo"

# Diarization pipeline + its component models.
download "pyannote/speaker-diarization-3.1"
# pyannote.audio 4.x делает 3.1-пайплайн зависимым от community-1: при загрузке
# 3.1 он лезет в community-1 за xvec_transform.npz (PLDA), и под HF_HUB_OFFLINE=1
# падает на cache miss. Поэтому community-1 тоже предзагружаем.
download "pyannote/speaker-diarization-community-1"
download "pyannote/segmentation-3.0"
download "pyannote/wespeaker-voxceleb-resnet34-LM"

# Word-level alignment model for Russian.
# English alignment uses a torchaudio-bundled model, no HF download required.
download "jonatasgrosman/wav2vec2-large-xlsr-53-russian"

echo
echo "Done. Models live in $MODELS_DIR."
du -sh "$MODELS_DIR" 2>/dev/null || true
echo "Container mounts this dir at /home/whisper/.cache/huggingface (read-only)."
