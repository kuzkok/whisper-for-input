"""
Resident WhisperX HTTP server.

Endpoints:
  POST /transcribe  — fast transcription, no diarization. Used by voice-input.
  POST /diarize     — full WhisperX pipeline: transcribe + word-align + speaker diarize.
  GET  /health
"""
import dataclasses
import os
import tempfile
import threading
from contextlib import asynccontextmanager

import nltk
import whisperx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
# Явный импорт сверху — fail-fast на API-дрифте whisperx (атрибут переехал
# из whisperx в whisperx.diarize в 3.8.x). Smoke-тест ловит это на старте,
# а не лениво на первом /diarize.
from whisperx.diarize import DiarizationPipeline

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))
INITIAL_PROMPT = os.environ.get(
    "WHISPER_INITIAL_PROMPT",
    "Разработчик диктует команды и заметки. Используй правильную пунктуацию. Английские термины пиши латиницей: git, deploy, branch, merge, commit, docker, kubernetes, API, SQL.",
)
PRELOAD_DIARIZE = os.environ.get("WHISPERX_PRELOAD_DIARIZE", "0") == "1"

# All HF models are pre-downloaded on the host via download-models.sh.
# The container runs with HF_HUB_OFFLINE=1, so no auth token is needed at runtime.

_model = None
_align_models = {}        # cached per language code: {lang: (model, metadata)}
_diarize_pipeline = None

# Обработчики держат GPU-модели в глобальных переменных и рассчитаны на один
# запрос за раз. Раньше сериализацию бесплатно давал event loop (эндпоинты были
# async def с блокирующим телом); теперь они sync и уходят в threadpool, поэтому
# взаимное исключение нужно явно. Заодно 6 ГБ VRAM не хватит на два пайплайна.
_gpu_lock = threading.Lock()

PUNKT_RESOURCE = "tokenizers/punkt_tab"


def _punkt_available():
    try:
        nltk.data.find(PUNKT_RESOURCE)
    except LookupError:
        return False
    return True


def _disable_nltk_download():
    """Запрещает nltk уходить в сеть, по аналогии с HF_HUB_OFFLINE=1.

    whisperx.align() при промахе по punkt_tab делает nltk.download(), а это
    блокирующий HTTP без таймаута. В офлайн-контейнере он не падает, а висит
    вечно, унося с собой весь запрос. Данные кладутся в образ на этапе сборки
    (см. Dockerfile), так что промах означает поломанный образ: лучше упасть
    с внятной ошибкой, чем зависнуть.
    """
    def _refuse(*args, **kwargs):
        raise RuntimeError(
            f"nltk-данные {PUNKT_RESOURCE} недоступны локально, а загрузка из сети "
            "отключена намеренно. Проверь NLTK_DATA и слой punkt_tab в Dockerfile."
        )

    nltk.download = _refuse


def get_align_model(lang_code):
    if lang_code not in _align_models:
        print(f"Loading alignment model for language '{lang_code}'...", flush=True)
        align_model, metadata = whisperx.load_align_model(language_code=lang_code, device=DEVICE)
        _align_models[lang_code] = (align_model, metadata)
    return _align_models[lang_code]


def get_diarize_pipeline():
    global _diarize_pipeline
    if _diarize_pipeline is None:
        print("Loading pyannote diarization pipeline...", flush=True)
        # model_name указан явно: дефолт пайплайна — pyannote/speaker-diarization-community-1,
        # которую download-models.sh НЕ предзагружает. Под HF_HUB_OFFLINE=1 это уронит
        # контейнер на cache miss. Менять эту строку → обязательно обновлять download-models.sh.
        _diarize_pipeline = DiarizationPipeline(
            model_name="pyannote/speaker-diarization-3.1",
            token=None,
            device=DEVICE,
        )
        print("Diarization pipeline ready.", flush=True)
    return _diarize_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print(
        f"Loading WhisperX model '{MODEL_NAME}' on {DEVICE} ({COMPUTE_TYPE})...",
        flush=True,
    )
    asr_options = {"initial_prompt": INITIAL_PROMPT} if INITIAL_PROMPT else {}
    _model = whisperx.load_model(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        asr_options=asr_options,
    )
    print("Transcribe model ready.", flush=True)
    _disable_nltk_download()
    if not _punkt_available():
        print(
            f"WARNING: nltk-ресурс {PUNKT_RESOURCE} не найден, /diarize упадёт на "
            "этапе выравнивания. Пересобери образ (слой punkt_tab в Dockerfile).",
            flush=True,
        )
    if PRELOAD_DIARIZE:
        get_diarize_pipeline()
    yield


app = FastAPI(lifespan=lifespan)


def _save_upload(file: UploadFile, audio_bytes: bytes) -> str:
    suffix = os.path.splitext(file.filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        return tmp.name


def _normalize_language(language: str):
    lang = language.strip() or None
    if lang == "auto":
        lang = None
    return lang


def _set_initial_prompt(prompt):
    """Per-request мутация _model.options.initial_prompt.

    whisperx 3.8.5 не позволяет передать initial_prompt в model.transcribe(...) —
    он зашит в asr_options на этапе load_model. Чтобы /transcribe и /diarize
    использовали разные промпты (диктовка vs произвольная встреча), мутируем
    options перед каждым вызовом. faster_whisper.TranscriptionOptions — это
    dataclass; whisperx сам пользуется dataclasses.replace по тому же паттерну
    (см. .venv/.../whisperx/asr.py:262).
    """
    _model.options = dataclasses.replace(_model.options, initial_prompt=prompt)


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
):
    """Fast path: transcription only. Compatible with the voice-input client contract.

    Использует INITIAL_PROMPT — этот эндпоинт обслуживает voice-input
    (push-to-talk диктовку разработчика), и подсказка про команды/латиницу
    повышает качество русской транскрипции технических терминов.

    Обработчик объявлен sync намеренно: FastAPI уводит такие в threadpool, и
    event loop остаётся свободным. Тело блокирующее (GPU, ffmpeg), а с async def
    одна долгая обработка вешала весь сервер, включая /health.
    """
    lang = _normalize_language(language)
    audio_bytes = file.file.read()
    tmp_path = _save_upload(file, audio_bytes)
    try:
        with _gpu_lock:
            audio = whisperx.load_audio(tmp_path)
            _set_initial_prompt(INITIAL_PROMPT or None)
            result = _model.transcribe(audio, batch_size=BATCH_SIZE, language=lang)
        text = " ".join(seg["text"].strip() for seg in result["segments"]).strip()
    finally:
        os.unlink(tmp_path)
    return JSONResponse({"text": text})


@app.post("/diarize")
def diarize(
    file: UploadFile = File(...),
    language: str = Form(""),
    min_speakers: int = Form(0),
    max_speakers: int = Form(0),
    format: str = Form("json"),
):
    """Full WhisperX pipeline: ASR + word alignment + speaker diarization.

    format=json  → {"language": "...", "segments": [{start, end, speaker, text}, ...]}
    format=text  → {"language": "...", "text": "[SPEAKER_00] ...\\n\\n[SPEAKER_01] ..."}

    INITIAL_PROMPT здесь НЕ используется: эндпоинт обслуживает произвольные
    встречи/видео, и русская подсказка про разработчика загнала бы английскую
    речь в перевод (Whisper интерпретирует prompt как контекст языка).

    Как и /transcribe, объявлен sync: пайплайн блокирующий и минутами держит
    поток, а event loop должен оставаться отзывчивым.
    """
    lang = _normalize_language(language)
    audio_bytes = file.file.read()
    tmp_path = _save_upload(file, audio_bytes)
    try:
        with _gpu_lock:
            audio = whisperx.load_audio(tmp_path)
            _set_initial_prompt(None)
            result = _model.transcribe(audio, batch_size=BATCH_SIZE, language=lang)
            lang_code = result["language"]

            align_model, metadata = get_align_model(lang_code)
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )

            pipe = get_diarize_pipeline()
            diarize_kwargs = {}
            if min_speakers > 0:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers > 0:
                diarize_kwargs["max_speakers"] = max_speakers
            diarize_segments = pipe(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
    finally:
        os.unlink(tmp_path)

    segments = [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": seg.get("speaker", "UNKNOWN"),
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
    ]

    if format == "text":
        lines = []
        current_speaker = None
        current_text = []
        for s in segments:
            if s["speaker"] != current_speaker:
                if current_text:
                    lines.append(f"[{current_speaker}] {' '.join(current_text)}")
                current_speaker = s["speaker"]
                current_text = [s["text"]]
            else:
                current_text.append(s["text"])
        if current_text:
            lines.append(f"[{current_speaker}] {' '.join(current_text)}")
        return JSONResponse({"language": lang_code, "text": "\n\n".join(lines)})

    return JSONResponse({"language": lang_code, "segments": segments})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "diarize_loaded": _diarize_pipeline is not None,
        "align_languages_loaded": sorted(_align_models.keys()),
        "punkt_available": _punkt_available(),
        "busy": _gpu_lock.locked(),
    }
