"""Smoke тесты: проверяют import surface, без загрузки моделей.

Идея: ловить API-дрифт whisperx за доли секунды. Этот файл должен падать
немедленно, если server.py импортирует что-то, чего нет в установленной
версии whisperx, или если сигнатура изменилась.
"""
import inspect

import pytest

pytestmark = pytest.mark.smoke


def test_import_server():
    """server.py импортируется без ошибок."""
    import server  # noqa: F401


def test_whisperx_top_level_attrs_used_by_server():
    """Все имена whisperx, которые server.py ожидает на верхнем уровне, существуют."""
    import whisperx

    for name in ("load_model", "load_align_model", "align", "assign_word_speakers", "load_audio"):
        assert hasattr(whisperx, name), f"whisperx.{name} отсутствует"


def test_whisperx_diarize_pipeline_exists_with_expected_signature():
    """DiarizationPipeline переехал в whisperx.diarize, и принимает model_name/token/device."""
    from whisperx.diarize import DiarizationPipeline

    sig = inspect.signature(DiarizationPipeline.__init__)
    params = sig.parameters
    for name in ("model_name", "token", "device"):
        assert name in params, f"DiarizationPipeline.__init__ не принимает {name}: {sig}"


def test_server_imports_diarization_pipeline_at_module_level():
    """server.py делает явный `from whisperx.diarize import DiarizationPipeline`
    наверху файла, чтобы дрифт ловился импортом, а не на первом /diarize."""
    import server

    assert hasattr(server, "DiarizationPipeline"), (
        "server.py должен делать `from whisperx.diarize import DiarizationPipeline` "
        "сверху, чтобы импорт падал немедленно при API-дрифте."
    )


def test_normalize_language():
    from server import _normalize_language

    assert _normalize_language("") is None
    assert _normalize_language("auto") is None
    assert _normalize_language("ru") == "ru"
    assert _normalize_language("  ru  ") == "ru"
    assert _normalize_language("en") == "en"


def test_health_schema():
    """GET /health возвращает ожидаемые ключи. Модели не загружаем —
    обращаемся к app напрямую без lifespan."""
    from fastapi.testclient import TestClient

    import server

    # TestClient без `with` НЕ запускает lifespan, модели не грузятся.
    client = TestClient(server.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    expected_keys = {
        "status",
        "model",
        "device",
        "compute_type",
        "diarize_loaded",
        "align_languages_loaded",
    }
    assert expected_keys <= set(body.keys()), f"missing keys: {expected_keys - set(body.keys())}"
    assert body["status"] == "ok"
    assert body["diarize_loaded"] is False
    assert body["align_languages_loaded"] == []
