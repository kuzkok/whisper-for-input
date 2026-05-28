"""Unit тесты: логика обработчиков FastAPI с замоканным whisperx.

Здесь не грузим ни одной реальной модели. Все вызовы в whisperx подменяются
через monkeypatch на простые стабы, которые возвращают заранее заданную
структуру. Цель — проверить, что server.py правильно склеивает / форматирует
выходы, корректно обрабатывает параметры формы и ветви типа format=text.
"""
from __future__ import annotations

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


class _FakeModel:
    """Стаб _model — записывает аргументы вызова и возвращает заранее заданное."""

    def __init__(self, segments: list[dict[str, Any]], language: str = "ru"):
        self._segments = segments
        self._language = language
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio, batch_size, language):  # noqa: ARG002
        self.calls.append({"batch_size": batch_size, "language": language})
        return {"segments": self._segments, "language": self._language}


class _FakeDiarizePipeline:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []

    def __call__(self, audio, **kwargs):  # noqa: ARG002
        self.calls.append(kwargs)
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
    # Spy на _set_initial_prompt, чтобы юниты могли проверить что эндпоинт
    # выставил правильный prompt перед вызовом transcribe.
    prompt_calls: list = []

    def fake_set_prompt(prompt):
        prompt_calls.append(prompt)

    monkeypatch.setattr(server, "_set_initial_prompt", fake_set_prompt)

    # Сегменты после assign_word_speakers — теперь со speaker label.
    spoken_segments = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "привет"},
        {"start": 1.0, "end": 1.5, "speaker": "SPEAKER_00", "text": "мир"},
        {"start": 1.5, "end": 2.5, "speaker": "SPEAKER_01", "text": "ответ"},
        {"start": 2.5, "end": 3.0, "speaker": "SPEAKER_00", "text": "снова"},
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
    monkeypatch.setattr(
        server.whisperx,
        "load_align_model",
        lambda language_code, device: ("FAKE_ALIGN_MODEL", "FAKE_METADATA"),
    )
    monkeypatch.setattr(
        server.whisperx,
        "align",
        lambda segments, model, metadata, audio, device, return_char_alignments: {
            "segments": segments,
        },
    )
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
            "prompt_calls": prompt_calls,
        },
    )


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
    expected = "[SPEAKER_00] привет мир\n\n[SPEAKER_01] ответ\n\n[SPEAKER_00] снова"
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
    assert resp.json()["text"] == "[SPEAKER_00] один блок"


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


# ── Per-request initial_prompt ──────────────────────────────────────────────


def test_transcribe_sets_initial_prompt_to_configured_value(patched_server):
    """/transcribe — voice-input диктовка, должен ставить INITIAL_PROMPT."""
    resp = patched_server.client.post(
        "/transcribe",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    assert patched_server.prompt_calls == [patched_server.module.INITIAL_PROMPT]


def test_diarize_sets_initial_prompt_to_none(patched_server):
    """/diarize — произвольная встреча, prompt должен быть None
    (русский продовый prompt уведёт английскую речь в перевод)."""
    resp = patched_server.client.post(
        "/diarize",
        files={"file": ("audio.wav", _silence_wav_bytes(), "audio/wav")},
        data={"language": "ru"},
    )
    assert resp.status_code == 200
    assert patched_server.prompt_calls == [None]


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
    assert patched_server.prompt_calls == [None]


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
