"""Integration тесты: реальный WhisperX, реальные модели, реальный GPU.

Эти тесты по умолчанию выключены через addopts = -m "not gpu" в pytest.ini.
Запуск явно: `pytest -m gpu`.

На первом прогоне:
- Модель large-v3-turbo грузится с диска (~3-5с с предзагруженного кэша).
- Английский torchaudio alignment скачивается (~360 MB), кэшируется в ~/.cache/torch.

Модель грузится один раз на сессию через session-scoped TestClient внутри
`with`-блока, который запускает FastAPI lifespan.
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="session")
def integration_client():
    """TestClient, поднятый с реальным lifespan — модель ASR грузится один раз."""
    import server

    with TestClient(server.app) as client:
        yield client


def _open_audio(path):
    return open(path, "rb")


# ── /transcribe ─────────────────────────────────────────────────────────────


def test_transcribe_real_ru(integration_client, ru_short_fixture):
    """/transcribe на русском должен вернуть текст с кириллицей."""
    with _open_audio(ru_short_fixture.audio_path) as f:
        resp = integration_client.post(
            "/transcribe",
            files={"file": (ru_short_fixture.audio_path.name, f, "audio/wav")},
            data={"language": ru_short_fixture.language},
        )
    assert resp.status_code == 200, resp.text
    text = resp.json()["text"]
    assert text, "пустой транскрипт"
    assert re.search(r"[а-яА-ЯёЁ]", text), f"ожидали кириллицу, получили: {text!r}"


def test_transcribe_real_en_returns_non_empty(integration_client, jfk_fixture):
    """/transcribe на английском возвращает непустой текст.

    На алфавит не ассертим: /transcribe умышленно использует русский INITIAL_PROMPT
    для voice-input use case, и это смещает модель в сторону русского контекста
    даже при language="en" — английская речь будет переведена в русский. Это
    приемлемое поведение для основного use case (диктовка разработчика на русском
    с вкраплениями English терминов). Английская транскрипция как таковая
    тестируется через /diarize, который prompt не использует.
    """
    with _open_audio(jfk_fixture.audio_path) as f:
        resp = integration_client.post(
            "/transcribe",
            files={"file": (jfk_fixture.audio_path.name, f, "audio/wav")},
            data={"language": jfk_fixture.language},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["text"], "пустой транскрипт"


# ── /diarize ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture_name", ["ru_short_fixture", "jfk_fixture"])
def test_diarize_json_real(integration_client, fixture_name, request):
    fixture = request.getfixturevalue(fixture_name)
    with _open_audio(fixture.audio_path) as f:
        resp = integration_client.post(
            "/diarize",
            files={"file": (fixture.audio_path.name, f, "audio/wav")},
            data={"language": fixture.language, "format": "json"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["language"] == fixture.language
    segments = body["segments"]
    assert segments, "diarize вернул пустой список сегментов"

    speaker_re = re.compile(r"^SPEAKER_\d+$")
    has_attributed = False
    for seg in segments:
        assert {"start", "end", "speaker", "text"} <= set(seg.keys())
        assert seg["start"] < seg["end"], f"start должен быть меньше end: {seg}"
        if speaker_re.match(seg["speaker"]):
            has_attributed = True
    assert has_attributed, (
        f"ни один сегмент не получил SPEAKER_NN метку (все UNKNOWN?): {segments}"
    )


@pytest.mark.parametrize("fixture_name", ["ru_short_fixture", "jfk_fixture"])
def test_diarize_text_real(integration_client, fixture_name, request):
    fixture = request.getfixturevalue(fixture_name)
    with _open_audio(fixture.audio_path) as f:
        resp = integration_client.post(
            "/diarize",
            files={"file": (fixture.audio_path.name, f, "audio/wav")},
            data={"language": fixture.language, "format": "text"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["language"] == fixture.language
    assert "[SPEAKER_" in body["text"], f"в text-формате нет маркера спикера: {body['text']!r}"


# ── /health после прогрева ──────────────────────────────────────────────────


def test_health_after_diarize_loads(integration_client, ru_short_fixture, jfk_fixture):
    """После реальных вызовов diarize сервер должен рапортовать загруженный pipeline
    и оба языковых alignment-модели."""
    # Прогреваем оба языка, если предыдущие тесты в сессии этого не сделали.
    for fixture in (ru_short_fixture, jfk_fixture):
        with _open_audio(fixture.audio_path) as f:
            resp = integration_client.post(
                "/diarize",
                files={"file": (fixture.audio_path.name, f, "audio/wav")},
                data={"language": fixture.language, "format": "json"},
            )
        assert resp.status_code == 200, resp.text

    health = integration_client.get("/health").json()
    assert health["diarize_loaded"] is True
    assert "ru" in health["align_languages_loaded"]
    assert "en" in health["align_languages_loaded"]


# ── PRELOAD_DIARIZE=1: отдельная сессия ────────────────────────────────────


@pytest.mark.slow
def test_preload_diarize_loads_pipeline_on_startup(monkeypatch):
    """С WHISPERX_PRELOAD_DIARIZE=1 пайплайн должен быть загружен ДО первого /diarize."""
    monkeypatch.setenv("WHISPERX_PRELOAD_DIARIZE", "1")

    # Перезагружаем модуль server, чтобы он подхватил env var на импорте.
    import importlib

    import server

    importlib.reload(server)

    with TestClient(server.app) as client:
        body = client.get("/health").json()
        assert body["diarize_loaded"] is True, (
            "WHISPERX_PRELOAD_DIARIZE=1 не загрузил пайплайн на старте"
        )

    # Откатываем модуль обратно для остальных тестов.
    monkeypatch.delenv("WHISPERX_PRELOAD_DIARIZE", raising=False)
    importlib.reload(server)
