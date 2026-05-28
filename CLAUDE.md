# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A monorepo for Whisper-based audio tooling. Top-level layout:

```
whisper-for-input/        ← ASR backend (server + container); name matches the repo
voice-input/              ← push-to-talk client daemon (consumer of /transcribe)
cli/                      ← shell consumers: transcribe.sh, diarize.sh
formatting-transcript/    ← Claude skill that turns raw transcripts into readable articles
docs/, tasks/             ← Claude-tooling artifacts and task specs
install.sh, CLAUDE.md     ← repo-root setup + this file
```

The backend plus its consumers (voice-input, cli, formatting-transcript) live side by side.

Push-to-talk voice input for KDE Wayland is the primary use case. Two cooperating processes on the same host:

- **`whisper-for-input/server.py`** — long-running WhisperX HTTP service inside a rootless Podman container with NVIDIA GPU passthrough. Exposes `POST /transcribe` (ASR only, hot path for voice-input), `POST /diarize` (ASR + word-align + speaker diarization), `GET /health`.
- **`voice-input/voice-input.py`** — user-space daemon that reads `/dev/input/event*` via `evdev`, records with `arecord` while the hotkey (default `KEY_SCROLLLOCK`) is held, POSTs the WAV to the server, pastes the result via `wl-copy` + `xdotool key shift+Insert`. The matching systemd unit lives at `voice-input/voice-input.service`.

The split exists because Whisper model load is multi-second; the resident server avoids paying it per utterance.

**Other consumers:**
- **`cli/transcribe.sh`, `cli/diarize.sh`** — thin curl wrappers around the server's `/transcribe` and `/diarize` endpoints for ad-hoc file transcription from the shell.
- **`formatting-transcript/`** — a Claude skill (`SKILL.md` + `format_transcript.py`) that post-processes a raw transcript into a punctuated, paragraphed article. Imported from the former `audio-tools` repo via merge; its fixtures live in `formatting-transcript/tests/`.

## Architecture notes that span files

**Models live outside the repo on the host** at `~/.local/share/whisper-for-input/models/` (HF cache, read-only mount) and `.../torch-cache/` (writable). The container runs with `HF_HUB_OFFLINE=1`, so anything not pre-fetched by `download-models.sh` will fail at runtime, not silently download. Three places (all under `whisper-for-input/`) reference these paths and **must stay in sync**: `whisper-for-input.container` (`Volume=`), `docker-compose.yml` (`volumes:`), `download-models.sh` (`DATA_DIR`).

**Two model fetch paths, intentionally:**
- `download-models.sh` (host, needs `./secrets/hf_token`) pulls the faster-whisper CTranslate2 weights, pyannote diarization (3.1 + community-1) + segmentation + wespeaker, and the Russian wav2vec2 alignment model into the HF cache. The community-1 repo is needed even though `server.py` selects 3.1 — pyannote.audio 4.x's 3.1 pipeline lazily loads `xvec_transform.npz` from community-1 as its PLDA component, and under `HF_HUB_OFFLINE=1` that fails on cache miss.
- WhisperX's VAD (~17MB) and torchaudio's English alignment weights (~360MB) are **not** in the HF cache — they download to `~/.cache/torch` on first request. That's why `torch-cache` is a separate writable volume.

**UID mapping:** Container runs as `whisper` (UID 5000). Quadlet unit uses `UserNS=keep-id:uid=5000,gid=5000` so host-owned model files appear to the container user without chown. `docker-compose.yml` mirrors this with `userns_mode: "keep-id:uid=5000,gid=5000"`.

**CUDA libs come from pip, not the base image.** `Dockerfile` installs `torch==2.8.0+cu128` from the PyTorch cu128 index; CUDA runtime ships as `nvidia-*` wheels into `site-packages/nvidia/*`. `LD_LIBRARY_PATH` is set to the NPP lib dir explicitly because torch's auto-dlopen path doesn't cover NPP. `libcuda.so.1` (driver) is injected at runtime by `AddDevice=nvidia.com/gpu=all` (Quadlet) or the `deploy.resources` block (Compose). The `torch==2.8.0` pin is forced by `whisperx → pyannote.audio 4.x → torchcodec 0.7` ABI requirements — don't bump torch in isolation.

**Alignment model cache is per-language and lazy.** `server.py:_align_models` keys by language code; the Russian wav2vec2 is pre-downloaded, English is fetched on first English `/diarize`. Languages without pre-cached weights will fail in offline mode.

**`PRELOAD_DIARIZE=1`** loads the pyannote pipeline at startup (adds ~5s + GPU memory). Off by default — diarization is the cold path. The transcribe model always loads at startup via the `lifespan` context manager.

**`INITIAL_PROMPT` is per-endpoint, not per-model.** `whisperx.load_model()` bakes `initial_prompt` into `asr_options` once, so it would otherwise apply to both endpoints. `_set_initial_prompt()` mutates `_model.options` (a faster_whisper `TranscriptionOptions` dataclass) via `dataclasses.replace` before each call: `/transcribe` uses `INITIAL_PROMPT` (voice-input dictation context — improves Russian punctuation and keeps English tech terms in Latin script), `/diarize` uses `None` (arbitrary meeting/video content — the Russian prompt would bias Whisper to translate English speech to Russian even with `language="en"`).

## Common commands

Server-stack commands (build, models, compose) run from inside `whisper-for-input/`; `install.sh` and the voice-input client run from the repo root.

```bash
# Initial setup (once per host) — from repo root
(cd whisper-for-input && ./download-models.sh)        # needs whisper-for-input/secrets/hf_token
./install.sh                                          # builds image, installs systemd units
sudo usermod -a -G input $USER                        # then re-login

# Run via systemd user units (production path)
systemctl --user enable --now whisper-for-input       # server (Quadlet)
systemctl --user enable --now voice-input             # client
journalctl --user -fu whisper-for-input               # live server logs
journalctl --user -fu voice-input                     # live client logs

# Run via docker-compose (alternative) — from whisper-for-input/
cd whisper-for-input
docker compose up -d
docker compose logs -f

# Rebuild image after server.py or Dockerfile changes — from whisper-for-input/
cd whisper-for-input
podman build -t whisper-for-input:latest .
systemctl --user restart whisper-for-input

# Run voice-input directly for debugging (bypasses systemd) — from repo root
python3 voice-input/voice-input.py --key KEY_SCROLLLOCK --lang ru --url http://localhost:8000
python3 voice-input/voice-input.py --list-keys        # list available evdev key names

# Manually hit the server
curl -F 'file=@sample.wav' -F 'language=ru' http://localhost:8000/transcribe
curl -F 'file=@sample.wav' -F 'language=' -F 'format=text' http://localhost:8000/diarize
curl http://localhost:8000/health
```

## Локальная дев-среда и тесты

Чтобы итерироваться над `server.py` без пересборки контейнерного образа, есть локальный venv с тем же стеком, что внутри контейнера (Python 3.12 + cu128 wheels). Это **venv только бэкенда** — он живёт в `whisper-for-input/.venv`, и всё ниже выполняется **из `whisper-for-input/`** (`cd whisper-for-input` сначала): там `server.py`, `requirements*.txt`, `pytest.ini` и `tests/`. Остальным частям монорепо этот стек не нужен: `cli/` — чистый shell, `formatting-transcript/` — stdlib-only Python (зовёт `claude -p`), `voice-input/` — лёгкие `evdev`+`requests` под системным `python3`.

```bash
cd whisper-for-input

# Один раз: поднять venv с зависимостями (~5 мин, ~5 GB)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Если только что склонировал репо — подтянуть LFS-фикстуры:
git lfs pull

# Дев-цикл
.venv/bin/pytest -m smoke           # ~доли секунды, ловит API-дрифт whisperx
.venv/bin/pytest -m unit            # ~секунды, логика обработчиков с моками
.venv/bin/pytest                    # smoke + unit (gpu выключены addopts'ом)
# GPU integration — нужны те же env vars, что у systemd unit/контейнера:
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/npp/lib" \
    HF_HOME="$HOME/.local/share/whisper-for-input/models" \
    HF_HUB_OFFLINE=1 \
    WHISPER_MODEL=dropbox-dash/faster-whisper-large-v3-turbo \
    .venv/bin/pytest -m gpu          # реальные модели + GPU, ~минута на холодную
.venv/bin/uvicorn server:app --reload  # ручная проверка с reload

# Только когда `pytest -m gpu` зелёный — пересобирать образ
podman build -t whisper-for-input:latest .
systemctl --user restart whisper-for-input
```

Audio-фикстуры (`tests/fixtures/{ru_short,jfk}.wav`) хранятся через git-lfs. Sidecar JSON рядом с каждым WAV содержит транскрипт, source URL и лицензию. Менять фикстуры — обычным `git add` после ручной замены файла.

## Editing gotchas

- **Bumping the image tag** in `docker-compose.yml` (`whisper-for-input:20260515.4`) is a manual versioning convention, not auto-generated. The Quadlet unit uses `:latest`; they don't have to match.
- **`voice-input/voice-input.py` is copied to `~/.local/bin/voice-input` by `install.sh`** — editing the repo file alone won't affect the running service. Re-run `install.sh` or copy manually, then `systemctl --user restart voice-input`.
- **`type_text()` uses `wl-copy` + `xdotool key shift+Insert`**, not `ydotool type`. This is intentional for Cyrillic — `ydotool type` mangles non-ASCII on most layouts. Don't "simplify" it back.
- **Container is rootless Podman with `SecurityLabelDisable=true` / `label:disable`**. SELinux relabel (`:z`) on the model volume is needed; don't drop it.
- **`HF_HUB_OFFLINE=1` is a guardrail, not a constraint to work around.** If a model isn't loading, the fix is to add it to `download-models.sh`, not to enable network in the container.
