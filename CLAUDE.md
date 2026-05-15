# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Push-to-talk voice input for KDE Wayland. Two cooperating processes on the same host:

- **`server.py`** — long-running WhisperX HTTP service inside a rootless Podman container with NVIDIA GPU passthrough. Exposes `POST /transcribe` (ASR only, hot path for voice-input), `POST /diarize` (ASR + word-align + speaker diarization), `GET /health`.
- **`voice-input/voice-input.py`** — user-space daemon that reads `/dev/input/event*` via `evdev`, records with `arecord` while the hotkey (default `KEY_SCROLLLOCK`) is held, POSTs the WAV to the server, pastes the result via `wl-copy` + `xdotool key shift+Insert`. The matching systemd unit lives at `voice-input/voice-input.service`.

The split exists because Whisper model load is multi-second; the resident server avoids paying it per utterance.

## Architecture notes that span files

**Models live outside the repo on the host** at `~/.local/share/whisper-for-input/models/` (HF cache, read-only mount) and `.../torch-cache/` (writable). The container runs with `HF_HUB_OFFLINE=1`, so anything not pre-fetched by `download-models.sh` will fail at runtime, not silently download. Three places reference these paths and **must stay in sync**: `whisper-for-input.container` (`Volume=`), `docker-compose.yml` (`volumes:`), `download-models.sh` (`DATA_DIR`).

**Two model fetch paths, intentionally:**
- `download-models.sh` (host, needs `./secrets/hf_token`) pulls the faster-whisper CTranslate2 weights, pyannote diarization + segmentation + wespeaker, and the Russian wav2vec2 alignment model into the HF cache.
- WhisperX's VAD (~17MB) and torchaudio's English alignment weights (~360MB) are **not** in the HF cache — they download to `~/.cache/torch` on first request. That's why `torch-cache` is a separate writable volume.

**UID mapping:** Container runs as `whisper` (UID 5000). Quadlet unit uses `UserNS=keep-id:uid=5000,gid=5000` so host-owned model files appear to the container user without chown. `docker-compose.yml` mirrors this with `userns_mode: "keep-id:uid=5000,gid=5000"`.

**CUDA libs come from pip, not the base image.** `Dockerfile` installs `torch==2.8.0+cu128` from the PyTorch cu128 index; CUDA runtime ships as `nvidia-*` wheels into `site-packages/nvidia/*`. `LD_LIBRARY_PATH` is set to the NPP lib dir explicitly because torch's auto-dlopen path doesn't cover NPP. `libcuda.so.1` (driver) is injected at runtime by `AddDevice=nvidia.com/gpu=all` (Quadlet) or the `deploy.resources` block (Compose). The `torch==2.8.0` pin is forced by `whisperx → pyannote.audio 4.x → torchcodec 0.7` ABI requirements — don't bump torch in isolation.

**Alignment model cache is per-language and lazy.** `server.py:_align_models` keys by language code; the Russian wav2vec2 is pre-downloaded, English is fetched on first English `/diarize`. Languages without pre-cached weights will fail in offline mode.

**`PRELOAD_DIARIZE=1`** loads the pyannote pipeline at startup (adds ~5s + GPU memory). Off by default — diarization is the cold path. The transcribe model always loads at startup via the `lifespan` context manager.

## Common commands

```bash
# Initial setup (once per host)
./download-models.sh                                  # needs ./secrets/hf_token
./install.sh                                          # builds image, installs systemd units
sudo usermod -a -G input $USER                        # then re-login

# Run via systemd user units (production path)
systemctl --user enable --now whisper-for-input       # server (Quadlet)
systemctl --user enable --now voice-input             # client
journalctl --user -fu whisper-for-input               # live server logs
journalctl --user -fu voice-input                     # live client logs

# Run via docker-compose (alternative)
docker compose up -d
docker compose logs -f

# Rebuild image after server.py or Dockerfile changes
podman build -t whisper-for-input:latest .
systemctl --user restart whisper-for-input

# Run voice-input directly for debugging (bypasses systemd)
python3 voice-input/voice-input.py --key KEY_SCROLLLOCK --lang ru --url http://localhost:8000
python3 voice-input/voice-input.py --list-keys        # list available evdev key names

# Manually hit the server
curl -F 'file=@sample.wav' -F 'language=ru' http://localhost:8000/transcribe
curl -F 'file=@sample.wav' -F 'language=' -F 'format=text' http://localhost:8000/diarize
curl http://localhost:8000/health
```

There is no test suite, linter, or build/format command in this repo — don't invent one.

## Editing gotchas

- **Bumping the image tag** in `docker-compose.yml` (`whisper-for-input:20260515.4`) is a manual versioning convention, not auto-generated. The Quadlet unit uses `:latest`; they don't have to match.
- **`voice-input/voice-input.py` is copied to `~/.local/bin/voice-input` by `install.sh`** — editing the repo file alone won't affect the running service. Re-run `install.sh` or copy manually, then `systemctl --user restart voice-input`.
- **`type_text()` uses `wl-copy` + `xdotool key shift+Insert`**, not `ydotool type`. This is intentional for Cyrillic — `ydotool type` mangles non-ASCII on most layouts. Don't "simplify" it back.
- **Container is rootless Podman with `SecurityLabelDisable=true` / `label:disable`**. SELinux relabel (`:z`) on the model volume is needed; don't drop it.
- **`HF_HUB_OFFLINE=1` is a guardrail, not a constraint to work around.** If a model isn't loading, the fix is to add it to `download-models.sh`, not to enable network in the container.
