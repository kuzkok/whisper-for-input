"""Unit тесты: логика обработчиков FastAPI с замоканным whisperx.

Здесь не грузим ни одной реальной модели. Все вызовы в whisperx подменяются
через monkeypatch на простые стабы, которые возвращают заранее заданную
структуру. Цель — проверить, что server.py правильно склеивает / форматирует
выходы, корректно обрабатывает параметры формы и ветви типа format=text.
"""
from __future__ import annotations

import dataclasses
import io
import wave
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


# ── Хелперы ────────────────────────────────────────────────────────────────


def _silence_wav_bytes(seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    """WAV-файл с тишиной — нужен только чтобы multipart upload прошёл."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return buf.getvalue()


@dataclasses.dataclass
class _FakeOptions:
    """Минимальный аналог faster_whisper.TranscriptionOptions.

    Хватает полей, которые мутирует _set_asr_hints: dataclasses.replace в server.py
    работает с любым dataclass с такими полями, тяжёлый настоящий класс не нужен.
    """

    initial_prompt: str | None = None
    hotwords: str | None = None



class _FakeCt2Model:
    """Стаб ctranslate2.models.Whisper: из всего API — только выгрузка весов.

    Настоящий объект лежит в _model.model.model (whisperx-пайплайн → faster_whisper
    → CTranslate2) и держит веса ASR на карте всё время жизни процесса.
    """

    def __init__(self):
        self.model_is_loaded = True
        self.calls: list[tuple] = []

    def unload_model(self, to_cpu=False):
        self.calls.append(("unload", to_cpu))
        self.model_is_loaded = False

    def load_model(self):
        self.calls.append(("load",))
        self.model_is_loaded = True


class _FakeWhisperModel:
    """Стаб faster_whisper.WhisperModel — нужен только как держатель ct2-модели."""

    def __init__(self):
        self.model = _FakeCt2Model()


class _FakeModel:
    """Стаб _model — записывает аргументы вызова и возвращает заранее заданное."""

    def __init__(self, segments: list[dict[str, Any]], language: str = "ru"):
        self._segments = segments
        self._language = language
        self.options = _FakeOptions()
        self.calls: list[dict[str, Any]] = []
        self.model = _FakeWhisperModel()

    def transcribe(self, audio, batch_size, language):  # noqa: ARG002
        # Снапшот подсказок на момент вызова — эндпоинт обязан выставить их
        # ДО transcribe() (мутация options идёт перед ним), и тесты проверяют
        # значения именно в этот момент.
        self.calls.append(
            {
                "batch_size": batch_size,
                "language": language,
                "initial_prompt": self.options.initial_prompt,
                "hotwords": self.options.hotwords,
            }
        )
        return {"segments": self._segments, "language": self._language}


class _FakeTorchModule:
    """Стаб torch-модели (wav2vec2 выравнивания, pyannote Pipeline): только .to().

    Помнит текущее устройство и историю переездов — по ним тесты проверяют, что
    align/pyannote работают на GPU, а между запросами лежат на CPU.
    """

    def __init__(self, device):
        self.device = str(device)
        self.moves: list[str] = []

    def to(self, device):
        self.device = str(device)
        self.moves.append(self.device)
        return self


class _FakeDiarizePipeline:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        # whisperx.diarize.DiarizationPipeline держит pyannote Pipeline в .model.
        self.model = _FakeTorchModule(kwargs.get("device", "cpu"))
        self.devices_at_call: list[str] = []

    def __call__(self, audio, **kwargs):  # noqa: ARG002
        self.calls.append(kwargs)
        self.devices_at_call.append(self.model.device)
        # Возвращаем что-то непустое, реальный assign_word_speakers замокан.
        return "FAKE_DIARIZATION_DF"


# ── Общие fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def patched_server(monkeypatch):
    """Импортирует server, подменяет всё whisperx-окружение, возвращает модуль и хэндлы."""
    import server

    # Дефолтный набор сегментов после транскрипции.
    default_segments = [
        {"start": 0.0, "end": 1.0, "text": "  привет  "},
        {"start": 1.0, "end": 2.0, "text": "мир"},
    ]
    fake_model = _FakeModel(default_segments, language="ru")
    # _set_asr_hints не мокаем: он мутирует fake_model.options по-настоящему
    # (dataclasses.replace на _FakeOptions), а снапшот в _FakeModel.transcribe
    # фиксирует значения на момент вызова.

    # Сегменты после assign_word_speakers — теперь со speaker label.
    # Слова несут собственные метки (поле word["speaker"] из assign_word_speakers).
    # Первый сегмент намеренно смешанный: доминирующий спикер SPEAKER_00, но
    # последнее слово принадлежит SPEAKER_01 — ровно тот случай, который
    # segment-level сборка теряет.
    spoken_segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_00",
            "text": "привет",
            "words": [
                {"word": "при", "start": 0.0, "end": 0.5, "score": 0.9, "speaker": "SPEAKER_00"},
                {"word": "вет", "start": 0.6, "end": 1.0, "score": 0.8, "speaker": "SPEAKER_01"},
            ],
        },
        {
            "start": 1.0,
            "end": 1.5,
            "speaker": "SPEAKER_00",
            "text": "мир",
            "words": [
                {"word": "мир", "start": 1.0, "end": 1.5, "score": 0.7, "speaker": "SPEAKER_00"},
            ],
        },
        {
            "start": 1.5,
            "end": 2.5,
            "speaker": "SPEAKER_01",
            "text": "ответ",
            "words": [
                {"word": "ответ", "start": 1.5, "end": 2.5, "score": 0.6, "speaker": "SPEAKER_01"},
            ],
        },
        {
            "start": 2.5,
            "end": 3.0,
            "speaker": "SPEAKER_00",
            "text": "снова",
            "words": [
                {"word": "снова", "start": 2.5, "end": 3.0, "score": 0.5, "speaker": "SPEAKER_00"},
            ],
        },
    ]

    fake_diarize = _FakeDiarizePipeline()
    diarize_calls = []

    def fake_diarize_factory(*args, **kwargs):
        diarize_calls.append({"args": args, "kwargs": kwargs})
        return fake_diarize

    # Сбрасываем глобальные кэши перед каждым тестом.
    monkeypatch.setattr(server, "_model", fake_model)
    monkeypatch.setattr(server, "_align_models", {})
    monkeypatch.setattr(server, "_diarize_pipeline", None)

    monkeypatch.setattr(server.whisperx, "load_audio", lambda path: np.zeros(16000, dtype=np.float32))
    align_models = []

    def fake_load_align_model(language_code, device):  # noqa: ARG001
        align_models.append(_FakeTorchModule(device))
        return align_models[-1], "FAKE_METADATA"

    align_devices = []

    def fake_align(segments, model, metadata, audio, device, return_char_alignments):  # noqa: ARG001
        align_devices.append(model.device)
        return {"segments": segments}

    monkeypatch.setattr(server.whisperx, "load_align_model", fake_load_align_model)
    monkeypatch.setattr(server.whisperx, "align", fake_align)
    monkeypatch.setattr(
        server.whisperx,
        "assign_word_speakers",
        lambda diarize_segments, transcript: {"segments": spoken_segments},
    )
    # server.py делает `from whisperx.diarize import DiarizationPipeline` сверху,
    # поэтому подменять надо локальное имя в модуле server, а не whisperx.diarize.
    monkeypatch.setattr(server, "DiarizationPipeline", fake_diarize_factory)

    return type(
        "Patched",
        (),
        {
            "module": server,
            "client": TestClient(server.app),
            "fake_model": fake_model,
            "fake_diarize": fake_diarize,
            "diarize_calls": diarize_calls,
            "spoken_segments": spoken_segments,
            "align_models": align_models,
            "align_devices": align_devices,
        },
    )


# ── /health: метрики GPU-памяти ─────────────────────────────────────────────


def test_health_gpu_memory_schema_when_cuda_initialized(monkeypatch, patched_server):
    """gpu_memory с поднятым CUDA-контекстом: MiB-числа из torch.cuda-метрик.
    device_used = total - free (аналог used в nvidia-smi)."""
    monkeypatch.setattr("torch.cuda.is_initialized", lambda: True)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (2 * 2**30, 8 * 2**30))
    monkeypatch.setattr("torch.cuda.memory_allocated", lambda: int(0.5 * 2**30))
    monkeypatch.setattr("torch.cuda.memory_reserved", lambda: int(1.5 * 2**30))
    resp = patched_server.client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["gpu_memory"] == {
        "torch_allocated_mib": 512,
        "torch_reserved_mib": 1536,
        "device_total_mib": 8192,
        "device_used_mib": 6144,
        "device_free_mib": 2048,
    }


def test_health_gpu_memory_null_when_cuda_not_initialized(monkeypatch, patched_server):
    """Без CUDA-контекста gpu_memory = None: /health не должен поднимать
    контекст сам (иначе каждый health-check занимал бы VRAM на карте)."""
    monkeypatch.setattr("torch.cuda.is_initialized", lambda: False)
    resp = patched_server.client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["gpu_memory"] is None


# ── /diarize: возврат арен после тяжёлого запроса ───────────────────────────


@pytest.fixture
def empty_cache_calls(monkeypatch):
    """Мок torch.cuda.empty_cache, считающий вызовы; CUDA «инициализирована»."""
    calls = []
    monkeypatch.setattr("torch.cuda.is_initialized", lambda: True)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: calls.append(1))
    return calls


def test_diarize_calls_empty_cache_afterwards(empty_cache_calls, patched_server):
    """/diarize — тяжёлый запрос: арены align/pyannote возвращаются драйверу,
    иначе повторный длинный /diarize падает с CUDA OOM (measured: 2,7 ГБ арен).

    Вызовов два: первый — внутри offload-контекста, перед возвратом весов ASR
    на карту, второй — в finally обработчика, он покрывает путь, где выгрузка
    не состоялась (CPU-устройство, отсутствующий API CTranslate2) или запрос
    упал ещё в ASR-фазе.
    """
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    assert len(empty_cache_calls) == 2


def test_diarize_calls_empty_cache_even_when_pipeline_fails(empty_cache_calls, patched_server):
    """OOM внутри пайплайна не должен оставлять арены до рестарта: empty_cache
    стоит в finally и выполняется независимо от исхода запроса."""
    def _boom(*args, **kwargs):
        raise RuntimeError("CUDA failed with error out of memory")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(patched_server.module.whisperx, "align", _boom)
    # Обычный TestClient пробрасывает исключение сервера в тест вместо
    # ответа 500 — здесь проверяем именно finally-путь, поэтому глушим.
    client = TestClient(patched_server.module.app, raise_server_exceptions=False)
    resp = client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    monkeypatch.undo()
    assert resp.status_code == 500
    # Два вызова, как и на успешном пути: упавший align тоже проходит через
    # finally offload-контекста (веса надо вернуть) и через finally обработчика.
    assert len(empty_cache_calls) == 2


def test_transcribe_does_not_call_empty_cache(empty_cache_calls, patched_server):
    """/transcribe — hot path voice-input: его арены малы (короткие клипы),
    empty_cache здесь только добавил бы задержку на регрейд арен."""
    resp = patched_server.client.post(
        "/transcribe",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    assert empty_cache_calls == []


# ── /diarize: выгрузка весов ASR на время align + диаризации ───────────────


def test_diarize_offloads_asr_weights_during_align_and_diarization(patched_server):
    """Пик VRAM /diarize приходится на align+pyannote, где веса ASR только лежат.

    Обе фазы обязаны идти с выгруженными весами, а к ответу веса должны
    вернуться: следующий /transcribe не имеет права ждать их загрузки.
    """
    module = patched_server.module
    ct2 = patched_server.fake_model.model.model
    seen = []

    def _align(segments, model, metadata, audio, device, return_char_alignments):  # noqa: ARG001
        seen.append(("align", ct2.model_is_loaded))
        return {"segments": segments}

    def _pipe(audio, **kwargs):  # noqa: ARG001
        seen.append(("diarize", ct2.model_is_loaded))
        return "FAKE_DIARIZATION_DF"

    _pipe.model = _FakeTorchModule("cuda")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module.whisperx, "align", _align)
    monkeypatch.setattr(module, "_diarize_pipeline", None)
    monkeypatch.setattr(module, "DiarizationPipeline", lambda *a, **k: _pipe)

    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    monkeypatch.undo()

    assert resp.status_code == 200
    assert seen == [("align", False), ("diarize", False)]
    assert ct2.model_is_loaded is True
    assert [c[0] for c in ct2.calls] == ["unload", "load"]


def test_diarize_restores_asr_weights_when_pipeline_fails(patched_server):
    """Упавший запрос не имеет права оставить сервер без весов ASR:
    следующий /transcribe (hot path voice-input) должен работать сразу."""
    ct2 = patched_server.fake_model.model.model

    def _boom(*args, **kwargs):
        raise RuntimeError("CUDA failed with error out of memory")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(patched_server.module.whisperx, "align", _boom)
    client = TestClient(patched_server.module.app, raise_server_exceptions=False)
    resp = client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    monkeypatch.undo()

    assert resp.status_code == 500
    assert ct2.model_is_loaded is True
    assert [c[0] for c in ct2.calls] == ["unload", "load"]


def test_diarize_frees_torch_arenas_before_reloading_asr_weights(monkeypatch, patched_server):
    """Веса возвращаются в ту же память, где только что были арены align/pyannote.

    Без предварительного empty_cache load_model падает с CUDA OOM (measured:
    torch держал 4478 MiB из 5804, весам нужно ~2250) — и запрос, дошедший до
    конца пайплайна, всё равно отвечает 500.
    """
    events = []
    monkeypatch.setattr("torch.cuda.is_initialized", lambda: True)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: events.append("empty_cache"))

    ct2 = patched_server.fake_model.model.model
    orig_load = ct2.load_model

    def _load():
        events.append("load_model")
        orig_load()

    monkeypatch.setattr(ct2, "load_model", _load)

    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    assert events.index("empty_cache") < events.index("load_model")


# ── /diarize: align + pyannote живут на CPU между запросами ────────────────


def _post_diarize(client):
    return client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )


def test_diarize_parks_align_and_pyannote_on_cpu_after_request(patched_server):
    """Выгрузка весов ASR — зеркальная половина. Вторая: align (wav2vec2) и
    pyannote, оставшись на карте после первого /diarize, отнимали ~1,2 ГБ у
    ASR-фазы следующего запроса, и под float16 он падал с CUDA OOM на encode
    первого батча. Работают они на GPU, а между запросами лежат на CPU."""
    for _ in range(2):
        assert _post_diarize(patched_server.client).status_code == 200

    align_model = patched_server.align_models[0]
    pipeline = patched_server.fake_diarize
    assert patched_server.align_devices == ["cuda", "cuda"]
    assert pipeline.devices_at_call == ["cuda", "cuda"]
    assert align_model.device == "cpu"
    assert pipeline.model.device == "cpu"


def test_diarize_parks_align_and_pyannote_when_pipeline_fails(patched_server):
    """Упавший запрос (например, OOM внутри pyannote) тоже обязан увезти модели
    с карты — иначе следующий /diarize упадёт уже на ASR-фазе."""
    monkeypatch = pytest.MonkeyPatch()

    def _boom(self, audio, **kwargs):  # noqa: ARG001
        raise RuntimeError("CUDA failed with error out of memory")

    monkeypatch.setattr(_FakeDiarizePipeline, "__call__", _boom)
    client = TestClient(patched_server.module.app, raise_server_exceptions=False)
    resp = _post_diarize(client)
    monkeypatch.undo()

    assert resp.status_code == 500
    assert patched_server.align_models[0].device == "cpu"
    assert patched_server.fake_diarize.model.device == "cpu"


def test_diarize_parks_models_before_reloading_asr_weights(monkeypatch, patched_server):
    """Веса ASR возвращаются на карту после того, как align/pyannote с неё
    съехали: иначе load_model делит память с их ~1,2 ГБ."""
    ct2 = patched_server.fake_model.model.model
    orig_load = ct2.load_model
    devices_at_reload = []

    def _load():
        devices_at_reload.append(
            (patched_server.align_models[0].device, patched_server.fake_diarize.model.device)
        )
        orig_load()

    monkeypatch.setattr(ct2, "load_model", _load)
    assert _post_diarize(patched_server.client).status_code == 200
    assert devices_at_reload == [("cpu", "cpu")]


def test_diarize_models_stay_put_on_cpu_device(monkeypatch, patched_server):
    """WHISPER_DEVICE=cpu: переезжать некуда, модели всё время на CPU."""
    monkeypatch.setattr(patched_server.module, "DEVICE", "cpu")
    assert _post_diarize(patched_server.client).status_code == 200
    assert set(patched_server.align_models[0].moves) <= {"cpu"}
    assert set(patched_server.fake_diarize.model.moves) <= {"cpu"}
    assert patched_server.align_devices == ["cpu"]


def test_transcribe_does_not_offload_asr_weights(patched_server):
    """/transcribe — hot path: выгрузка-загрузка на каждую диктовку стоила бы
    секунд на ровном месте, а align/pyannote здесь не работают."""
    ct2 = patched_server.fake_model.model.model
    resp = patched_server.client.post(
        "/transcribe",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    assert ct2.calls == []
    assert ct2.model_is_loaded is True


# ── /transcribe ─────────────────────────────────────────────────────────────


def test_transcribe_joins_and_trims_segments(patched_server):
    resp = patched_server.client.post(
        "/transcribe",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "привет мир"}


def test_transcribe_empty_language_passes_none_to_model(patched_server):
    resp = patched_server.client.post(
        "/transcribe",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": ""},
    )
    assert resp.status_code == 200
    assert patched_server.fake_model.calls[-1]["language"] is None


def test_transcribe_auto_passes_none_to_model(patched_server):
    resp = patched_server.client.post(
        "/transcribe",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "auto"},
    )
    assert resp.status_code == 200
    assert patched_server.fake_model.calls[-1]["language"] is None


def test_transcribe_explicit_language_passes_through(patched_server):
    resp = patched_server.client.post(
        "/transcribe",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "en"},
    )
    assert resp.status_code == 200
    assert patched_server.fake_model.calls[-1]["language"] == "en"


# ── /diarize format=json ────────────────────────────────────────────────────


def test_diarize_json_structure(patched_server):
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "json"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "ru"
    assert isinstance(body["segments"], list) and body["segments"]
    for seg in body["segments"]:
        assert set(seg.keys()) == {"start", "end", "speaker", "text"}
        assert isinstance(seg["start"], float)
        assert isinstance(seg["end"], float)
        assert seg["speaker"].startswith("SPEAKER_")


def test_diarize_unknown_speaker_fallback(monkeypatch, patched_server):
    """Если у сегмента нет ключа speaker — возвращаем 'UNKNOWN' (server.py:153)."""
    monkeypatch.setattr(
        patched_server.module.whisperx,
        "assign_word_speakers",
        lambda d, t: {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "no-speaker-attr"},
            ]
        },
    )
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "json"},
    )
    assert resp.status_code == 200
    assert resp.json()["segments"][0]["speaker"] == "UNKNOWN"


# ── /diarize format=text — самая нетривиальная логика ──────────────────────


def test_diarize_text_groups_consecutive_same_speaker(patched_server):
    """SPEAKER_00 (привет, мир) → SPEAKER_01 (ответ) → SPEAKER_00 (снова).
    Должно получиться три блока, разделённых пустой строкой."""
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "text"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "ru"
    expected = (
        "[00:00:00] [SPEAKER_00] привет мир\n\n"
        "[00:00:01] [SPEAKER_01] ответ\n\n"
        "[00:00:02] [SPEAKER_00] снова"
    )
    assert body["text"] == expected


def test_diarize_text_single_speaker(monkeypatch, patched_server):
    monkeypatch.setattr(
        patched_server.module.whisperx,
        "assign_word_speakers",
        lambda d, t: {
            "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "один"},
                {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "блок"},
            ]
        },
    )
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "text"},
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "[00:00:00] [SPEAKER_00] один блок"


def test_diarize_text_timestamp_hours(monkeypatch, patched_server):
    """3725.15s → 01:02:05: метка HH:MM:SS с округлением вниз до целой секунды."""
    monkeypatch.setattr(
        patched_server.module.whisperx,
        "assign_word_speakers",
        lambda d, t: {
            "segments": [
                {"start": 3725.15, "end": 3730.0, "speaker": "SPEAKER_01", "text": "долго"},
            ]
        },
    )
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "text"},
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "[01:02:05] [SPEAKER_01] долго"


# ── /diarize: проброс min/max_speakers ─────────────────────────────────────


def test_diarize_min_max_speakers_passed_when_positive(patched_server):
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "min_speakers": "2", "max_speakers": "4"},
    )
    assert resp.status_code == 200
    assert patched_server.fake_diarize.calls[-1] == {"min_speakers": 2, "max_speakers": 4}


def test_diarize_min_max_speakers_omitted_when_zero(patched_server):
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    assert patched_server.fake_diarize.calls[-1] == {}


# ── /diarize: автодетект языка ─────────────────────────────────────────────


def test_diarize_explicit_language_skips_autodetect(patched_server):
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    assert patched_server.fake_model.calls[-1]["language"] == "ru"


def test_diarize_empty_language_triggers_autodetect(patched_server):
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": ""},
    )
    assert resp.status_code == 200
    assert patched_server.fake_model.calls[-1]["language"] is None


# ── /diarize: ленивая загрузка пайплайна ────────────────────────────────────


def test_diarize_pipeline_constructed_with_explicit_model_name(patched_server):
    """Дефолт пайплайна — community-1, которую мы НЕ предзагружаем.
    server.py обязан явно указать pyannote/speaker-diarization-3.1."""
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    assert len(patched_server.diarize_calls) == 1
    kwargs = patched_server.diarize_calls[0]["kwargs"]
    assert kwargs.get("model_name") == "pyannote/speaker-diarization-3.1"
    assert "token" in kwargs


# ── Per-request ASR-подсказки: initial_prompt / hotwords ────────────────────


def test_transcribe_sets_initial_prompt_and_resets_hotwords(patched_server):
    """/transcribe — voice-input диктовка: INITIAL_PROMPT и hotwords=None.
    hotwords от предыдущего /diarize-запроса не должны утекать в диктовку
    (options общие и мутируются на месте)."""
    resp = patched_server.client.post(
        "/transcribe",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    call = patched_server.fake_model.calls[-1]
    assert call["initial_prompt"] == patched_server.module.INITIAL_PROMPT
    assert call["hotwords"] is None


def test_diarize_without_hints_passes_none_for_both(patched_server):
    """/diarize без новых полей — обе подсказки None (продовый русский
    INITIAL_PROMPT уведёт английскую речь в перевод)."""
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    call = patched_server.fake_model.calls[-1]
    assert call["initial_prompt"] is None
    assert call["hotwords"] is None


def test_diarize_initial_prompt_reaches_options(patched_server):
    """initial_prompt из формы — в options к моменту transcribe()."""
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "initial_prompt": "Встреча про Kubernetes и Prometheus."},
    )
    assert resp.status_code == 200
    call = patched_server.fake_model.calls[-1]
    assert call["initial_prompt"] == "Встреча про Kubernetes и Prometheus."
    assert call["hotwords"] is None


def test_diarize_hotwords_reaches_options(patched_server):
    """hotwords из формы — в options к моменту transcribe()."""
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "hotwords": "Kubernetes Prometheus Grafana"},
    )
    assert resp.status_code == 200
    call = patched_server.fake_model.calls[-1]
    assert call["initial_prompt"] is None
    assert call["hotwords"] == "Kubernetes Prometheus Grafana"


def test_diarize_prompt_and_hotwords_together(patched_server):
    """Оба параметра одновременно — оба в options."""
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={
            "language": "ru",
            "initial_prompt": "Ретро команды: ведущий Михаил.",
            "hotwords": "ретро срез закупка",
        },
    )
    assert resp.status_code == 200
    call = patched_server.fake_model.calls[-1]
    assert call["initial_prompt"] == "Ретро команды: ведущий Михаил."
    assert call["hotwords"] == "ретро срез закупка"


def test_diarize_hints_do_not_leak_between_requests(patched_server):
    """Подсказки per-request: запрос БЕЗ полей сбрасывает то, что поставил
    предыдущий запрос (options мутируются на месте, без сброса утекут)."""
    for data in (
        {"language": "ru", "initial_prompt": "утечёт?", "hotwords": "утечёт"},
        {"language": "ru"},  # без полей — обе подсказки должны уйти в None
    ):
        resp = patched_server.client.post(
            "/diarize",
            files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
            data=data,
        )
        assert resp.status_code == 200
    call = patched_server.fake_model.calls[-1]
    assert call["initial_prompt"] is None
    assert call["hotwords"] is None


def test_transcribe_with_empty_initial_prompt_passes_none(monkeypatch, patched_server):
    """Если INITIAL_PROMPT="" — /transcribe тоже передаёт None, а не пустую строку
    (`INITIAL_PROMPT or None` нормализует falsy в None)."""
    monkeypatch.setattr(patched_server.module, "INITIAL_PROMPT", "")
    resp = patched_server.client.post(
        "/transcribe",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    call = patched_server.fake_model.calls[-1]
    assert call["initial_prompt"] is None
    assert call["hotwords"] is None


def test_diarize_pipeline_cached_across_requests(patched_server):
    for _ in range(3):
        resp = patched_server.client.post(
            "/diarize",
            files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
            data={"language": "ru"},
        )
        assert resp.status_code == 200
    # DiarizationPipeline должен быть вызван ровно один раз — get_diarize_pipeline
    # кэширует в _diarize_pipeline.
    assert len(patched_server.diarize_calls) == 1


# ── /diarize include_words: word-level выход ───────────────────────────────


def test_diarize_json_has_no_words_without_flag(patched_server):
    """Контракт по умолчанию не меняется: words отдаются только по запросу,
    иначе ответ на длинной записи распухает на порядок."""
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "json"},
    )
    assert resp.status_code == 200
    for seg in resp.json()["segments"]:
        assert "words" not in seg


def test_diarize_include_words_returns_word_level_speakers(patched_server):
    """include_words=true — у каждого сегмента полный список слов со своими
    метками. Первый сегмент смешанный: SPEAKER_00 + SPEAKER_01 внутри одного
    сегмента, чья метка SPEAKER_00."""
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "json", "include_words": "true"},
    )
    assert resp.status_code == 200
    segments = resp.json()["segments"]
    first = segments[0]
    assert set(first.keys()) == {"start", "end", "speaker", "text", "words"}
    assert [w["word"] for w in first["words"]] == ["при", "вет"]
    assert [w["speaker"] for w in first["words"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert first["words"][0] == {
        "word": "при",
        "start": 0.0,
        "end": 0.5,
        "score": 0.9,
        "speaker": "SPEAKER_00",
    }


def test_diarize_include_words_keeps_missing_speaker_as_none(monkeypatch, patched_server):
    """Слово без пересечения с диаризацией (fill_nearest=False) приходит без
    ключа speaker. Отдаём None, а не UNKNOWN: для метрик «метки нет» и «метка
    такая-то» — разные случаи, а UNKNOWN — валидное значение уровня сегмента."""
    monkeypatch.setattr(
        patched_server.module.whisperx,
        "assign_word_speakers",
        lambda d, t: {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "SPEAKER_00",
                    "text": "тишина",
                    "words": [{"word": "тишина", "start": 0.0, "end": 1.0, "score": 0.4}],
                }
            ]
        },
    )
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "json", "include_words": "true"},
    )
    assert resp.status_code == 200
    assert resp.json()["segments"][0]["words"][0]["speaker"] is None


def test_diarize_include_words_tolerates_unaligned_word(monkeypatch, patched_server):
    """У слов, которые wav2vec2 не выровнял (цифры, символы), нет start/end/score.
    Такое слово не должно ронять ответ — отдаём None в таймингах."""
    monkeypatch.setattr(
        patched_server.module.whisperx,
        "assign_word_speakers",
        lambda d, t: {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "SPEAKER_00",
                    "text": "15",
                    "words": [{"word": "15"}],
                }
            ]
        },
    )
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "json", "include_words": "true"},
    )
    assert resp.status_code == 200
    assert resp.json()["segments"][0]["words"][0] == {
        "word": "15",
        "start": None,
        "end": None,
        "score": None,
        "speaker": None,
    }


def test_diarize_include_words_segment_without_words_key(monkeypatch, patched_server):
    """Сегмент без ключа words (выравнивание ничего не вернуло) → пустой список,
    а не KeyError."""
    monkeypatch.setattr(
        patched_server.module.whisperx,
        "assign_word_speakers",
        lambda d, t: {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "..."}]},
    )
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "json", "include_words": "true"},
    )
    assert resp.status_code == 200
    assert resp.json()["segments"][0]["words"] == []


def test_diarize_include_words_ignored_for_format_text(patched_server):
    """format=text отдаёт строки, места под слова там нет: флаг молча не влияет
    на форму ответа."""
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "text", "include_words": "true"},
    )
    assert resp.status_code == 200
    assert set(resp.json().keys()) == {"language", "text"}


# ── /diarize: разрез сегментов по word-level спикерам ──────────────────────


def _two_speaker_segment():
    """Один VAD-сегмент, накрывший вопрос одного участника и ответ другого.

    Реплика каждого длиннее порога дребезга: метки первых слов реплики
    whisperx путает регулярно, и разрез рассчитан на уверенные куски."""
    words = [
        {"word": "А", "start": 0.0, "end": 0.2, "score": 0.9, "speaker": "SPEAKER_00"},
        {"word": "доступ", "start": 0.2, "end": 0.6, "score": 0.9, "speaker": "SPEAKER_00"},
        {"word": "в", "start": 0.6, "end": 0.7, "score": 0.9, "speaker": "SPEAKER_00"},
        {"word": "репозиторий", "start": 0.7, "end": 1.2, "score": 0.9, "speaker": "SPEAKER_00"},
        {"word": "все", "start": 1.2, "end": 1.4, "score": 0.9, "speaker": "SPEAKER_00"},
        {"word": "получили?", "start": 1.4, "end": 1.9, "score": 0.9, "speaker": "SPEAKER_00"},
        {"word": "Пока", "start": 2.0, "end": 2.3, "score": 0.9, "speaker": "SPEAKER_01"},
        {"word": "нет,", "start": 2.3, "end": 2.6, "score": 0.9, "speaker": "SPEAKER_01"},
        {"word": "заявка", "start": 2.6, "end": 3.0, "score": 0.9, "speaker": "SPEAKER_01"},
        {"word": "ещё", "start": 3.0, "end": 3.3, "score": 0.9, "speaker": "SPEAKER_01"},
        {"word": "в", "start": 3.3, "end": 3.4, "score": 0.9, "speaker": "SPEAKER_01"},
        {"word": "работе.", "start": 3.4, "end": 3.9, "score": 0.9, "speaker": "SPEAKER_01"},
    ]
    return {
        "start": 0.0,
        "end": 3.9,
        "speaker": "SPEAKER_01",  # доминирующий по времени — ему ушёл бы весь сегмент
        "text": " ".join(w["word"] for w in words),
        "words": words,
    }


def test_diarize_splits_segment_with_two_speakers(monkeypatch, patched_server):
    """Ключевой случай задачи: вопрос и ответ в одном VAD-сегменте должны стать
    двумя сегментами с разными метками, а не одним под меткой доминирующего."""
    monkeypatch.setattr(
        patched_server.module.whisperx,
        "assign_word_speakers",
        lambda d, t: {"segments": [_two_speaker_segment()]},
    )
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "json"},
    )
    assert resp.status_code == 200
    segments = resp.json()["segments"]
    assert [s["speaker"] for s in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert segments[0]["text"] == "А доступ в репозиторий все получили?"
    assert segments[1]["text"] == "Пока нет, заявка ещё в работе."


def test_diarize_text_shows_two_replicas_after_split(monkeypatch, patched_server):
    """То же в format=text: две реплики вместо одной слипшейся."""
    monkeypatch.setattr(
        patched_server.module.whisperx,
        "assign_word_speakers",
        lambda d, t: {"segments": [_two_speaker_segment()]},
    )
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru", "format": "text"},
    )
    assert resp.status_code == 200
    lines = resp.json()["text"].split("\n\n")
    assert lines == [
        "[00:00:00] [SPEAKER_00] А доступ в репозиторий все получили?",
        "[00:00:02] [SPEAKER_01] Пока нет, заявка ещё в работе.",
    ]
