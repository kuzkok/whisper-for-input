# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A monorepo for Whisper-based audio tooling. Top-level layout:

```
whisper-for-input/        ← ASR backend (server + container); name matches the repo
voice-input/              ← push-to-talk client daemon (consumer of /transcribe)
cli/                      ← shell consumers: transcribe.sh, diarize.sh
formatting-transcript/    ← Claude skill that turns raw transcripts into readable articles
transcript-cleanup/       ← Claude skill that cleans diarized transcripts (names, glossary, punctuation)
docs/, tasks/             ← Claude-tooling artifacts and task specs
install.sh, CLAUDE.md     ← repo-root setup + this file
```

The backend plus its consumers (voice-input, cli, formatting-transcript) live side by side.

Push-to-talk voice input for KDE Wayland is the primary use case. Two cooperating processes on the same host:

- **`whisper-for-input/server.py`** — long-running WhisperX HTTP service inside a rootless Podman container with NVIDIA GPU passthrough. Exposes `POST /transcribe` (ASR only, hot path for voice-input), `POST /diarize` (ASR + word-align + speaker diarization), `GET /health`.
- **`voice-input/voice-input.py`** — user-space daemon that reads `/dev/input/event*` via `evdev`, records with `arecord` while the hotkey (default `KEY_SCROLLLOCK`) is held, POSTs the WAV to the server, pastes the result via `wl-copy` + `xdotool key shift+Insert`. The matching systemd unit lives at `voice-input/voice-input.service`.

The split exists because Whisper model load is multi-second; the resident server avoids paying it per utterance.

**Other consumers:**
- **`cli/transcribe.sh`, `cli/diarize.sh`** — thin curl wrappers around the server's `/transcribe` and `/diarize` endpoints for ad-hoc file transcription from the shell. `diarize.sh --words` saves the raw JSON with word-level speakers (`<input>.words.json`) instead of readable text — input for `whisper-for-input/tools/segmentation_metrics.py`.
- **`formatting-transcript/`** — a Claude skill (`SKILL.md` + `format_transcript.py`) that turns a raw monologue transcript into a readable article: full punctuation, paragraphs, fillers and false starts removed, obvious ASR errors fixed without a glossary, the speaker's wording never paraphrased. The script cuts the text into ~300-word chunks itself (snapping to a sentence end in the source) and formats them in parallel via haiku, neighbours passed as read-only context; haiku's thinking stays on, because without it whole phrases get dropped. Each chunk's answer is saved to `<output>.parts/` as soon as it arrives, so a run killed by the subscription limit (429) resumes with the same command and calls the model only for the missing chunks. Tests with a fake `claude -p` live in `formatting-transcript/tests/` (stdlib `unittest`: `python3 -m unittest discover -s tests` from the skill folder). `install.sh` symlinks the skill folder into `~/.claude/skills/formatting-transcript`.
- **`transcript-cleanup/`** — a Claude skill (`SKILL.md` + `cleanup_transcript.py`) that cleans diarized meeting transcripts (output of `cli/diarize.sh`): maps `SPEAKER_XX` labels to real names via a project context file (two-phase: sonnet builds the speaker map from a meeting skeleton, substitution is deterministic, haiku cleans text in chunks of ~5000 characters, each holding as many whole replicas as fit — replica length varies by orders of magnitude between meetings and podcasts, and model latency follows characters, not replica count; the `[HH:MM:SS] [Speaker]` skeleton never passes through the model). Unlike formatting-transcript, it relies on the context-file glossary and keeps fillers, while formatting-transcript removes fillers and false starts and works without external context — the two skills' edit policies are deliberately incompatible. Tests with a fake `claude -p` live in `transcript-cleanup/tests/` (stdlib `unittest`, no repo venv needed). `install.sh` symlinks the skill folder into `~/.claude/skills/transcript-cleanup`.

## Architecture notes that span files

**Models live outside the repo on the host** at `~/.local/share/whisper-for-input/models/` (HF cache, read-only mount) and `.../torch-cache/` (writable). The container runs with `HF_HUB_OFFLINE=1`, so anything not pre-fetched by `download-models.sh` will fail at runtime, not silently download. Three places (all under `whisper-for-input/`) reference these paths and **must stay in sync**: `whisper-for-input.container` (`Volume=`), `docker-compose.yml` (`volumes:`), `download-models.sh` (`DATA_DIR`).

**Two model fetch paths, intentionally:**
- `download-models.sh` (host, needs `./secrets/hf_token`) pulls the faster-whisper CTranslate2 weights, pyannote diarization (3.1 + community-1) + segmentation + wespeaker, and the Russian wav2vec2 alignment model into the HF cache. The community-1 repo is needed even though `server.py` selects 3.1 — pyannote.audio 4.x's 3.1 pipeline lazily loads `xvec_transform.npz` from community-1 as its PLDA component, and under `HF_HUB_OFFLINE=1` that fails on cache miss.
- WhisperX's VAD (~17MB) and torchaudio's English alignment weights (~360MB) are **not** in the HF cache — they download to `~/.cache/torch` on first request. That's why `torch-cache` is a separate writable volume.

**UID mapping:** Container runs as `whisper` (UID 5000). Quadlet unit uses `UserNS=keep-id:uid=5000,gid=5000` so host-owned model files appear to the container user without chown. `docker-compose.yml` mirrors this with `userns_mode: "keep-id:uid=5000,gid=5000"`.

**CUDA libs come from pip, not the base image.** `Dockerfile` installs `torch==2.8.0+cu128` from the PyTorch cu128 index; CUDA runtime ships as `nvidia-*` wheels into `site-packages/nvidia/*`. `LD_LIBRARY_PATH` is set to the NPP lib dir explicitly because torch's auto-dlopen path doesn't cover NPP. `libcuda.so.1` (driver) is injected at runtime by `AddDevice=nvidia.com/gpu=all` (Quadlet) or the `deploy.resources` block (Compose). The `torch==2.8.0` pin is forced by `whisperx → pyannote.audio 4.x → torchcodec 0.7` ABI requirements — don't bump torch in isolation.

**nltk punkt_tab ships inside the image, unlike the models.** `whisperx.align()` splits text into sentences with nltk and, on a cache miss, calls `nltk.download()`, a blocking HTTP fetch with no timeout that hangs `/diarize` forever in the offline container. `HF_HUB_OFFLINE` does not cover it; nltk has its own downloader. The `Dockerfile` pre-fetches `punkt_tab` into `NLTK_DATA=/home/whisper/nltk_data` (~11MB of static tables, not a model: no token, no lifecycle), and `server.py:_disable_nltk_download()` replaces `nltk.download` at startup so a miss fails loudly instead of hanging. `GET /health` reports `punkt_available`.

**`/transcribe` and `/diarize` are sync handlers on purpose.** Their bodies are blocking (GPU, ffmpeg); as `async def` a single long request froze the whole event loop, `/health` included. FastAPI runs `def` handlers in a threadpool, and `_gpu_lock` keeps the shared models single-request-at-a-time, the serialization the event loop used to provide for free.

**The host has hybrid graphics: the discrete GPU belongs almost entirely to the service.** The laptop has an integrated GPU and the discrete 6GB RTX 3060; the desktop (KDE Wayland, the browser and the rest) renders on the integrated one, and `nvidia-smi` shows only `kwin_wayland` with a few MiB on the discrete card. So the VRAM headroom is not about the desktop: it is eaten by the service's own models (ASR weights, align, pyannote, torch arenas) and only occasionally by an app explicitly started on the discrete GPU via PRIME offload. When chasing a CUDA OOM, look at the service first — `/health` `gpu_memory` and `nvidia-smi --query-compute-apps` — not at the desktop.

**`WHISPER_BATCH_SIZE=8` is a measured ceiling, not a guess.** On the 6GB RTX 3060 each unit of batch costs ~230MB of VRAM in the ASR phase (peak: 2590MB at 4, 3518MB at 8, 4446MB at 12, 5374MB at 16, OOM at 20). Speed barely moves: 10 minutes of audio transcribe in 8.5s at batch 4 vs 7.5s at batch 12, and a 6-second voice-input clip takes ~0.4s regardless. In `/diarize` the ASR phase is a few seconds out of minutes, and the real peak (~5.2GB, both at 10 and 34 minutes of input) comes from align + pyannote, not from the batch. So a bigger batch buys nothing measurable and eats the headroom: 16 works on an idle GPU and dies with `CUDA failed with error out of memory` as soon as anything else lands on the card — the service's own resident models or an app explicitly run on the discrete GPU (see the hybrid-graphics note above).

**`WHISPER_COMPUTE_TYPE=float16`, not `int8_float16` — quantization was losing whole chunks of speech.** On a 46-minute meeting recording the int8 weights replaced a 27-second stretch of live dialogue with a hallucinated YouTube-subtitle phrase; float16 on the same audio transcribes it in full. Measured on that recording: hallucination markers 2 → 0, glossary errors 5 → 4 occurrences, runtime unchanged (877s vs 869s — the ASR phase is a fraction of `/diarize`). float16 weights cost 928 MiB more on the card (2257 MiB resident vs 1329), which on a 6GB card is exactly the headroom the `/diarize` peak needs — hence the offload below. Russian fine-tunes were measured as an alternative and rejected: they trade one script for the other (`dvislobokov` transliterates Latin terms — 16 Latin tokens vs 105; `coriollon` latinizes Russian abbreviations — the project's own acronym appears 2 times instead of 9), and the undistilled large-v3 only fits at `batch_size=2` and runs its ASR phase 5× slower (883s vs 177s) for one glossary occurrence less.

**`/diarize` takes the ASR weights off the card for align + pyannote (`_asr_weights_offloaded` in `server.py`).** The peak is not in the ASR phase: it comes from align+pyannote, where the CTranslate2 weights merely sit resident (measured under float16: 5615 MiB peak of a 5804 MiB card, leaving 190 MiB — an OOM as soon as anything else touches the card). `ctranslate2.models.Whisper.unload_model()` frees 2164 MiB for the duration, bringing the whole-run peak to 4711 MiB, i.e. back to the old int8 level (4655) while keeping float16 quality; the transcript is byte-identical and the runtime unchanged (869s both ways). `to_cpu=False` on purpose — `to_cpu=True` keeps a copy in RAM to reload faster, and the reload is off the hot path anyway. Two details that are not optional: the reload runs **after** `torch.cuda.empty_cache()`, because the weights go back into the memory the align/pyannote arenas just occupied and CTranslate2 allocates outside torch (without it `load_model()` dies with OOM — measured: torch held 4478 MiB, the weights need ~2250, so a request that finished the whole pipeline still answered 500); and the reload sits in `finally`, or a request killed by OOM would leave the server with no ASR model for the next `/transcribe`. `/transcribe` never offloads — it is the voice-input hot path and align/pyannote don't run there. Note the side effect: with the weights gone the torch allocator grows its arenas more freely (reserved 4478 vs 3230), so the win on the peak is ~900 MiB rather than the full 2164.

**Align and pyannote live in RAM between requests and go to the GPU only for their phase (`_diarize_models_parked` in `server.py`).** This is the mirror half of the ASR offload. Left on the card after the first `/diarize`, wav2vec2 + pyannote (~1250 MiB of live tensors, which `empty_cache` can't touch) ate into the ASR phase of the next request, and under float16 every `/diarize` after the first died with `CUDA failed with error out of memory` at the first batch's encode (measured: `device_free` 1988 MiB with `torch_allocated` 1258). The models still compute on the GPU: `get_align_model()` / `get_diarize_pipeline()` move them onto `DEVICE` when handed out, and the context manager parks every cached one back to CPU in `finally`. It is nested inside `_asr_weights_offloaded`, so the models leave the card before the ASR weights come back. pyannote's `Pipeline.to()` only accepts a `torch.device` and moves all its sub-models (segmentation, embedding); the whisperx wrapper keeps it in `.model`. Measured on a 39-minute meeting recording, two `/diarize` back to back: both 200, ~147s each, byte-identical output, whole-card peak 4529 MiB, `torch_allocated` back to 14 MiB after each request. The cost is copying ~1.2GB of weights host↔device per request, not measurable against the pipeline.

**`/diarize` releases VRAM after every request (`torch.cuda.empty_cache()` in the handler's `finally`).** Without it the PyTorch caching allocator kept ~2.7GB of arenas grown during a long request's align+pyannote phases (measured: `torch_reserved` 3962 MiB with `torch_allocated` at 1250 — the live weights; the next request doesn't reuse those arenas, and they can't grow — only ~440 MiB free on the card), so a second long `/diarize` died with CUDA OOM at the first batch's encode — inside CTranslate2, which has its own allocator and no access to torch arenas. `empty_cache` doesn't touch live tensors (the align/pyannote weights stay resident); the next heavy request pays re-growing the arenas, which is not measurable in time (24.7 min of audio: 1m45s with the fix vs 1m47s without). The call sits in `finally` on purpose: a request that died with OOM must also return its arenas, or the process stays wedged until a restart. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was tried too — arenas end up more compact (2900 instead of 3962), but the OOM on a repeated long request remained, so the config is not enabled. `/health` reports `gpu_memory`: `torch_allocated`/`torch_reserved` plus `device_used`/`device_free` (MiB; null until the CUDA context is up — `/health` must not create it itself). `device_used - torch_reserved` ≈ CTranslate2 with CUDA contexts (~1400 MiB).

**Segments are re-cut by word-level speakers, but only where the evidence is strong (`speaker_split.py`).** `whisperx.assign_word_speakers` labels a segment by the dominant overlap with diarization and labels every word separately; the server used to keep the segment label and drop the word one, so a VAD segment covering two people handed all of it to one of them. Cutting on every change of `word['speaker']` was measured and rejected: on two real recordings (24.7 and 38.9 minutes) word labels are systematically off by a word or two at the seams, so the naive cut fixed ~5% of replicas and tore a sentence in half in 30-60 places where there had been 1-2. What ships instead cuts only on an alien run of `MIN_RUN_WORDS`+ words, snapping the cut to a sentence boundary within `SNAP_WINDOW_WORDS`, falling back to a pause of `MIN_PAUSE_S` and, failing that, cutting anyway once the run reaches `FORCE_RUN_WORDS`. A cut that would leave a piece shorter than `MIN_PART_WORDS` is dropped entirely — that piece is not a replica but a shaved-off tail («…и | подключить.»). Where there is no cut but most of the words are alien, the segment label itself is corrected (`_relabelled`): whisperx derives it from overlap duration, so it disagrees with the word majority exactly when someone's whole sentence landed inside the segment. Measured effect: replicas with a long alien run 5→2 and 7→2, words under the wrong label −24% and −37%, replica count +1, sentence-mid starts 1→1 and 2→5. The module is stdlib-only and imports no whisperx on purpose — the assembly logic is unit-tested without a GPU (`tests/test_speaker_split.py`) and reused offline by the metrics tool.

**`/diarize` can return word-level data (`include_words=true`), off by default.** Words are what makes the segment/diarization mismatch measurable from outside, but on a long recording they inflate the response by an order of magnitude and no consumer needs them. `tools/segmentation_metrics.py` computes the segmentation metrics from such a dump without a GPU and prints numbers only — dumps come from private recordings. Note what the metric cannot see: an error of diarization itself (a stretch handed wholesale to the wrong person) leaves no segment/word mismatch and stays invisible.

**Alignment model cache is per-language and lazy.** `server.py:_align_models` keys by language code; the Russian wav2vec2 is pre-downloaded, English is fetched on first English `/diarize`. Languages without pre-cached weights will fail in offline mode.

**`PRELOAD_DIARIZE=1`** loads the pyannote pipeline at startup (adds ~5s + RAM; it is parked on CPU until the first `/diarize`, like between requests). Off by default — diarization is the cold path. The transcribe model always loads at startup via the `lifespan` context manager.

**ASR hints (`initial_prompt`, `hotwords`) are per-endpoint and per-request, not per-model.** `whisperx.load_model()` bakes `initial_prompt` into `asr_options` once and `hotwords` lives in the same faster_whisper `TranscriptionOptions` — neither can be passed to `transcribe()` directly. `_set_asr_hints()` mutates `_model.options` via `dataclasses.replace` before each call and always sets both fields: a forgotten field would leak from one request into the next through the shared model. `/transcribe` uses `INITIAL_PROMPT` (voice-input dictation context — improves Russian punctuation and keeps English tech terms in Latin script). `/diarize` defaults to no prompt (arbitrary meeting/video content — the Russian prompt would bias Whisper to translate English speech to Russian even with `language="en"`), but accepts optional `initial_prompt` and `hotwords` form fields for audio with known context (project terms, names, acronyms): in the whisperx batched pipeline the prompt is rebuilt for every VAD batch, so it applies to the whole recording, not just its start, and `hotwords` sit closest to the decoding window (strongest effect on rare terms). Both are bounded by the Whisper prompt window (~200+ tokens) — a short line of key terms, not a glossary; the server doesn't validate length, faster-whisper silently truncates. `cli/diarize.sh` passes them as `--prompt <str>` / `--prompt-file <path>` / `--hotwords <str>` / `--hotwords-file <path>` (one word per line).

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
# Bump the tag in whisper-for-input.container + docker-compose.yml first, then:
cd whisper-for-input
podman build -t whisper-for-input:20260923.2 -t whisper-for-input:latest .
systemctl --user daemon-reload && systemctl --user restart whisper-for-input

# Run voice-input directly for debugging (bypasses systemd) — from repo root
python3 voice-input/voice-input.py --key KEY_SCROLLLOCK --lang ru --url http://localhost:8000
python3 voice-input/voice-input.py --list-keys        # list available evdev key names

# Manually hit the server
curl -F 'file=@sample.wav' -F 'language=ru' http://localhost:8000/transcribe
curl -F 'file=@sample.wav' -F 'language=' -F 'format=text' http://localhost:8000/diarize
curl http://localhost:8000/health
```

## Локальная дев-среда и тесты

Чтобы итерироваться над `server.py` без пересборки контейнерного образа, есть локальный venv с тем же стеком, что внутри контейнера (Python 3.12 + cu128 wheels). Это **venv только бэкенда** — он живёт в корне репо (`.venv/`), а дев-цикл ниже выполняется **из `whisper-for-input/`** (`cd whisper-for-input`): там `server.py`, `requirements*.txt`, `pytest.ini` и `tests/`, поэтому venv в командах адресуется как `../.venv`. Остальным частям монорепо этот стек не нужен: `cli/` — чистый shell, `formatting-transcript/` — stdlib-only Python (зовёт `claude -p`), `voice-input/` — лёгкие `evdev`+`requests` под системным `python3`.

```bash
# Один раз, из корня репо: поднять venv с зависимостями (~5 мин, ~5 GB)
python3.12 -m venv .venv
.venv/bin/pip install -r whisper-for-input/requirements.txt \
    -r whisper-for-input/requirements-dev.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Если только что склонировал репо — подтянуть LFS-фикстуры:
git lfs pull

# nltk punkt_tab для pytest -m gpu (его требует whisperx.align): в контейнерном
# образе он предзалит Dockerfile'ом, а в хостовом venv его нет — без него gpu-тесты
# падают LookupError'ом. Кладём внутрь venv: nltk ищет sys.prefix/nltk_data
# автоматически, env vars не нужны. -d обязателен — без него downloader кладёт
# в ~/nltk_data и засоряет home:
.venv/bin/python -m nltk.downloader -d .venv/nltk_data punkt_tab

# Дев-цикл — из whisper-for-input/
cd whisper-for-input
../.venv/bin/pytest -m smoke           # ~доли секунды, ловит API-дрифт whisperx
../.venv/bin/pytest -m unit            # ~секунды, логика обработчиков с моками
../.venv/bin/pytest                    # smoke + unit (gpu выключены addopts'ом)
# GPU integration — нужны те же env vars, что у systemd unit/контейнера,
# и остановленный сервис whisper-for-input: резидентные веса в контейнере
# не оставляют места на 6-ГБ карте — тесты падают с CUDA OOM ещё на загрузке
# модели (systemctl --user stop whisper-for-input, после прогона start):
LD_LIBRARY_PATH="$PWD/../.venv/lib/python3.12/site-packages/nvidia/npp/lib" \
    HF_HOME="$HOME/.local/share/whisper-for-input/models" \
    HF_HUB_OFFLINE=1 \
    WHISPER_MODEL=dropbox-dash/faster-whisper-large-v3-turbo \
    ../.venv/bin/pytest -m gpu          # реальные модели + GPU, ~минута на холодную
../.venv/bin/uvicorn server:app --reload  # ручная проверка с reload

# Только когда `pytest -m gpu` зелёный — пересобирать образ
podman build -t whisper-for-input:20260923.2 -t whisper-for-input:latest .
systemctl --user restart whisper-for-input
```

Audio-фикстуры (`tests/fixtures/{ru_short.ogx,jfk.wav}`) хранятся через git-lfs. Sidecar JSON рядом с каждым аудиофайлом содержит транскрипт, source URL и лицензию. Менять фикстуры — обычным `git add` после ручной замены файла.

## Editing gotchas

- **Bumping the image tag** is a manual versioning convention, not auto-generated, and it lives in three places that must agree: `whisper-for-input.container` (`Image=`), `docker-compose.yml` (`image:`), and whatever you pass to `podman build -t`. The Quadlet unit is pinned on purpose so a restart never picks up an unrelated `:latest` rebuild; `install.sh` reads the tag back out of the unit. Build both tags (`-t whisper-for-input:<tag> -t whisper-for-input:latest`).
- **`voice-input/voice-input.py` is copied to `~/.local/bin/voice-input` by `install.sh`** — editing the repo file alone won't affect the running service. Re-run `install.sh` or copy manually, then `systemctl --user restart voice-input`.
- **`type_text()` uses `wl-copy` + `xdotool key shift+Insert`**, not `ydotool type`. This is intentional for Cyrillic — `ydotool type` mangles non-ASCII on most layouts. Don't "simplify" it back.
- **Container is rootless Podman with `SecurityLabelDisable=true` / `label:disable`**. SELinux relabel (`:z`) on the model volume is needed; don't drop it.
- **`HF_HUB_OFFLINE=1` is a guardrail, not a constraint to work around.** If a model isn't loading, the fix is to add it to `download-models.sh`, not to enable network in the container.

## Confidentiality

Audio that users bring for testing (meetings, calls, voice notes) is private, and its transcripts can name real people, companies, and projects. Never copy that material into repository artifacts — task documents (`tasks/`), specs, commit messages, test code or fixtures, logs checked into the repo. Refer to such files generically ("a 24.7-minute meeting recording provided by the user"); don't record filenames or paths that reveal what the recording is. The same applies to identifiers surfaced while working with user content — people's names, company and project names, codenames: keep them out of files, including examples and test data. Public fixtures under `whisper-for-input/tests/fixtures/` with their license sidecars are the exception — they are meant to be in the repo.
