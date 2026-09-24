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
from contextlib import asynccontextmanager, contextmanager

import nltk
import torch
import whisperx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
# Явный импорт сверху — fail-fast на API-дрифте whisperx (атрибут переехал
# из whisperx в whisperx.diarize в 3.8.x). Smoke-тест ловит это на старте,
# а не лениво на первом /diarize.
from whisperx.diarize import DiarizationPipeline

from speaker_split import split_segments

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
    """Модель выравнивания для языка, уже на DEVICE.

    Между запросами модели лежат на CPU (см. _diarize_models_parked), поэтому
    закэшированную модель поднимаем на карту при каждой выдаче; для свежей
    загрузки .to() — no-op.
    """
    if lang_code not in _align_models:
        print(f"Loading alignment model for language '{lang_code}'...", flush=True)
        align_model, metadata = whisperx.load_align_model(language_code=lang_code, device=DEVICE)
        _align_models[lang_code] = (align_model, metadata)
    align_model, metadata = _align_models[lang_code]
    return align_model.to(torch.device(DEVICE)), metadata


def _load_diarize_pipeline():
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


def get_diarize_pipeline():
    """Пайплайн pyannote, уже на DEVICE (между запросами он на CPU, как и align).

    .model — pyannote Pipeline внутри обёртки whisperx; его .to() переносит все
    подмодели (segmentation, embedding) и принимает только torch.device.
    """
    pipeline = _load_diarize_pipeline()
    pipeline.model.to(torch.device(DEVICE))
    return pipeline


def _park_diarize_models():
    """Увозит закэшированные align и pyannote с карты в RAM."""
    if DEVICE != "cuda":
        return
    cpu = torch.device("cpu")
    for align_model, _ in _align_models.values():
        align_model.to(cpu)
    if _diarize_pipeline is not None:
        _diarize_pipeline.model.to(cpu)


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
        # Предзагрузка экономит чтение с диска, а не место на карте: до первого
        # /diarize пайплайн лежит в RAM, как и между запросами.
        _load_diarize_pipeline()
        _park_diarize_models()
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


def _format_ts(seconds: float) -> str:
    """Секунды с начала записи → HH:MM:SS, округление вниз до целой секунды."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _word_payload(word):
    """Одно слово из выхода assign_word_speakers для include_words.

    Отсутствующие поля отдаём как None, а не подставляем значения:
    - start/end/score нет у слов, которые wav2vec2 не выровнял (цифры, символы);
    - speaker нет, когда слово не пересеклось ни с одним интервалом диаризации
      (fill_nearest=False по умолчанию, whisperx/diarize.py:189).
    Для метрик сегментации «метки нет» и «метка такая-то» — разные случаи,
    поэтому UNKNOWN уровня сегмента сюда не переносится.
    """
    return {
        "word": word.get("word", ""),
        "start": float(word["start"]) if word.get("start") is not None else None,
        "end": float(word["end"]) if word.get("end") is not None else None,
        "score": float(word["score"]) if word.get("score") is not None else None,
        "speaker": word.get("speaker"),
    }


def _set_asr_hints(initial_prompt, hotwords):
    """Per-request мутация ASR-подсказок в _model.options.

    whisperx 3.8.5 не позволяет передать initial_prompt/hotwords в
    model.transcribe(...) — они зашиты в asr_options на этапе load_model.
    Чтобы эндпоинты (и отдельные запросы /diarize) использовали разные
    подсказки, мутируем options перед каждым вызовом. faster_whisper.
    TranscriptionOptions — это dataclass; whisperx сам пользуется
    dataclasses.replace по тому же паттерну (см. .venv/.../whisperx/asr.py:262).

    Оба поля нужно передавать всегда, а не только «которые меняются»:
    options живут в глобальной модели, забытое значение утечёт в следующий
    запрос (например, hotwords из /diarize — в диктовку voice-input).
    """
    _model.options = dataclasses.replace(
        _model.options, initial_prompt=initial_prompt, hotwords=hotwords
    )


def _asr_ct2_model():
    """ctranslate2.models.Whisper внутри пайплайна whisperx, если он там есть.

    Путь _model.model.model — это whisperx.asr.FasterWhisperPipeline →
    faster_whisper.WhisperModel → CTranslate2. Достаём через getattr, а не
    напрямую: смена версии whisperx не должна ронять /diarize, максимум —
    возвращать выгрузку к прежнему поведению (веса лежат на карте всегда).
    """
    return getattr(getattr(_model, "model", None), "model", None)


@contextmanager
def _asr_weights_offloaded():
    """Снимает веса ASR с карты на время align + диаризации.

    Пик VRAM /diarize приходится не на ASR-фазу, а на align+pyannote, где веса
    ASR только лежат резидентно (измерено на записи 46 минут: пик 5615 MiB из
    5804 под float16, из них ~2250 MiB — сами веса; под int8_float16 пик 4655).
    CTranslate2 держит их в своей памяти, torch.cuda.empty_cache их не трогает,
    поэтому единственный способ освободить карту на тяжёлой фазе — unload_model.

    to_cpu=False, а не True: to_cpu оставляет копию весов в RAM ради быстрого
    возврата, а возврат здесь и так вне горячего пути — /diarize идёт минутами.

    Возврат весов в finally: запрос, упавший на OOM, не имеет права оставить
    сервер без модели — следующим придёт /transcribe от voice-input.
    """
    ct2 = _asr_ct2_model()
    offloadable = DEVICE == "cuda" and hasattr(ct2, "unload_model")
    if offloadable:
        ct2.unload_model(to_cpu=False)
    try:
        yield
    finally:
        if offloadable and not ct2.model_is_loaded:
            # Веса возвращаются ровно в ту память, где только что стояли арены
            # align/pyannote, а CTranslate2 выделяет её мимо torch и его арен
            # не видит. Без empty_cache load_model падает с CUDA out of memory
            # (measured: torch держал 4478 MiB из 5804, весам нужно ~2250),
            # и дошедший до конца пайплайна запрос всё равно отвечает 500.
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            ct2.load_model()


@contextmanager
def _diarize_models_parked():
    """Align + pyannote работают на GPU только внутри блока, после — в RAM.

    Зеркало _asr_weights_offloaded. Оставшись на карте после первого /diarize,
    wav2vec2 и pyannote (~1,2 ГБ живых тензоров, empty_cache их не трогает)
    отнимали память у ASR-фазы следующего запроса, и под float16 он падал с
    CUDA OOM на encode первого батча (measured: device_free 1988 MiB при
    torch_allocated 1258). Считаются они по-прежнему на GPU: геттеры поднимают
    их на карту перед фазой, цена — копирование весов туда-обратно на запрос.

    Блок вложен внутрь _asr_weights_offloaded: модели съезжают с карты раньше,
    чем туда возвращаются веса ASR. В finally — по той же причине, что и там:
    упавший запрос не должен оставить их занимать карту.
    """
    try:
        yield
    finally:
        _park_diarize_models()


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
            _set_asr_hints(INITIAL_PROMPT or None, None)
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
    initial_prompt: str = Form(""),
    hotwords: str = Form(""),
    include_words: bool = Form(False),
):
    """Full WhisperX pipeline: ASR + word alignment + speaker diarization.

    format=json  → {"language": "...", "segments": [{start, end, speaker, text}, ...]}
    format=text  → {"language": "...",
                     "text": "[00:01:23] [SPEAKER_00] ...\n\n[00:05:41] [SPEAKER_01] ..."}

    include_words=true добавляет каждому сегменту поле words со словами и их
    собственными метками спикера. Метка сегмента — доминирующая по пересечению
    с диаризацией (whisperx/diarize.py:229), поэтому речь второго говорящего
    внутри одного VAD-сегмента в ней не видна, а на уровне слов — видна. Флаг
    нужен, чтобы это рассогласование можно было измерить снаружи, не переделывая
    сборку реплик. По умолчанию выключен: на записи в десятки минут слова
    раздувают ответ на порядок, а потребителям (/diarize в CLI, transcript-cleanup)
    они не нужны. Действует только на format=json — в строках format=text места
    под слова нет.

    Глобальный INITIAL_PROMPT здесь НЕ используется: эндпоинт обслуживает
    произвольные встречи/видео, и русская подсказка про разработчика загнала бы
    английскую речь в перевод (Whisper интерпретирует prompt как контекст языка).

    Вместо этого — опциональные per-request подсказки для аудио с известным
    контекстом (термины проекта, имена, аббревиатуры); пустое значение →
    текущее поведение без промпта. Действуют только на этап ASR: выравнивание
    и pyannote их не видят.

    initial_prompt — контекстная фраза («Ретро-встреча команды X»). В batched
    пайплайне whisperx подмешивается в контекст КАЖДОГО батча VAD-сегментов,
    т.е. действует на всё аудио, а не только на начало. Задаёт и язык-контекст:
    промпт не на языке аудио может увести распознавание в перевод.
    hotwords — список ключевых слов через пробел; ставятся вплотную к
    декодируемому окну, на редкие термины действует сильнее, чем initial_prompt.

    Обе подсказки ограничены prompt-окном Whisper (~200+ токенов): это короткая
    строка с ключевыми терминами, а не глоссарий целиком. Сервер не валидирует
    длину; faster-whisper молча обрежет hotwords до половины окна.

    Как и /transcribe, объявлен sync: пайплайн блокирующий и минутами держит
    поток, а event loop должен оставаться отзывчивым.
    """
    lang = _normalize_language(language)
    audio_bytes = file.file.read()
    tmp_path = _save_upload(file, audio_bytes)
    try:
        with _gpu_lock:
            audio = whisperx.load_audio(tmp_path)
            _set_asr_hints(initial_prompt or None, hotwords or None)
            result = _model.transcribe(audio, batch_size=BATCH_SIZE, language=lang)
            lang_code = result["language"]

            # Веса ASR на align+pyannote уже не нужны, а это самая тяжёлая по
            # VRAM фаза — снимаем их с карты до конца пайплайна (см.
            # _asr_weights_offloaded).
            with _asr_weights_offloaded(), _diarize_models_parked():
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
        # После /diarize возвращаем драйверу арены, выросшие под align/pyannote.
        # Без этого caching allocator держит ~2,7 ГБ зарезервированными (арены
        # не фрагментированы, но и не переиспользуются под новый запрос), и
        # повторный длинный /diarize падает с CUDA OOM уже на encode первого
        # батча — measured: reserved 3962 MiB при allocated 1250, free 441.
        # empty_cache не трогает живые тензоры (веса моделей остаются в VRAM),
        # платим повторным выделением арен следующим тяжёлым запросом.
        # Вызов в finally, а не в теле: упавший OOM-запрос тоже обязан
        # вернуть арены, иначе процесс до рестарта остаётся заблокирован.
        if DEVICE == "cuda" and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        os.unlink(tmp_path)

    # Сегмент, накрывший речь двух человек, whisperx отдаёт целиком доминирующему,
    # а word-level разметку выбрасывает. Режем его обратно по меткам слов — иначе
    # вопрос одного участника оказывается внутри реплики другого (см. speaker_split).
    spoken = split_segments(result["segments"])

    segments = []
    for seg in spoken:
        item = {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": seg.get("speaker", "UNKNOWN"),
            "text": seg["text"].strip(),
        }
        if include_words:
            item["words"] = [_word_payload(w) for w in seg.get("words", [])]
        segments.append(item)

    if format == "text":
        lines = []
        current_speaker = None
        current_start = 0.0
        current_text = []
        for s in segments:
            if s["speaker"] != current_speaker:
                if current_text:
                    lines.append(f"[{_format_ts(current_start)}] [{current_speaker}] {' '.join(current_text)}")
                current_speaker = s["speaker"]
                current_start = s["start"]
                current_text = [s["text"]]
            else:
                current_text.append(s["text"])
        if current_text:
            lines.append(f"[{_format_ts(current_start)}] [{current_speaker}] {' '.join(current_text)}")
        return JSONResponse({"language": lang_code, "text": "\n\n".join(lines)})

    return JSONResponse({"language": lang_code, "segments": segments})


def _gpu_memory():
    """Снимок VRAM для /health. None, если CUDA-контекст ещё не поднят.

    Гвард is_initialized(), а не is_available(): mem_get_info() сам создаёт
    CUDA-контекст при первом обращении. В проде контекст уже есть (lifespan
    грузит модель на cuda при старте), а вот /health у процесса без модели —
    unit-тесты, cpu-режим — не должен аллоцировать VRAM на карте.

    Числа в MiB, как в nvidia-smi:
    - torch_allocated — живые тензоры torch (align, pyannote);
    - torch_reserved — всё, что держит caching allocator (allocated + арены);
    - device_* — карта целиком из mem_get_info: total - free включает веса
      CTranslate2, CUDA-контексты и чужие процессы (kwin и т.п.), то есть
      сопоставимо со строкой used в nvidia-smi.
    Разница device_used - torch_reserved — это CTranslate2 с контекстами;
    разница torch_reserved - torch_allocated — зарезервированные, но
    не занятые тензорами арены (кандидат на empty_cache после /diarize).
    """
    if not torch.cuda.is_initialized():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_allocated_mib": round(torch.cuda.memory_allocated() / 2**20),
        "torch_reserved_mib": round(torch.cuda.memory_reserved() / 2**20),
        "device_total_mib": round(total / 2**20),
        "device_used_mib": round((total - free) / 2**20),
        "device_free_mib": round(free / 2**20),
    }


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
        "gpu_memory": _gpu_memory(),
    }
