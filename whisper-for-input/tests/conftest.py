"""Общие fixtures для всех тестов."""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Чтобы tests/ могли делать `import server` — server.py живёт в корне репо.
sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class AudioFixture:
    name: str
    audio_path: Path
    transcript: str
    language: str
    sample_rate: int
    channels: int
    duration_s: float


_AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".ogx", ".flac", ".m4a")


def _load_audio_fixture(name: str) -> AudioFixture:
    candidates = [FIXTURES_DIR / f"{name}{ext}" for ext in _AUDIO_EXTS]
    audio_path = next((p for p in candidates if p.exists()), None)
    meta = FIXTURES_DIR / f"{name}.json"

    if audio_path is None:
        pytest.skip(
            f"Аудио-фикстура {name} не найдена (искал {', '.join(p.name for p in candidates)}). "
            "Положи файл вручную (см. CLAUDE.md)."
        )
    if not meta.exists():
        pytest.skip(f"Sidecar {meta} не найдена.")

    # Защита от LFS pointer: pointer — текстовый файл, начинается с "version https://git-lfs"
    head = audio_path.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs"):
        pytest.fail(
            f"{audio_path} — это git-lfs pointer, а не реальный аудио-файл. "
            "Запусти `git lfs pull` (или скачай файл руками)."
        )

    data = json.loads(meta.read_text())
    return AudioFixture(
        name=name,
        audio_path=audio_path,
        transcript=data["transcript"],
        language=data["language"],
        sample_rate=data["sample_rate"],
        channels=data["channels"],
        duration_s=data["duration_s"],
    )


@pytest.fixture(scope="session")
def jfk_fixture() -> AudioFixture:
    return _load_audio_fixture("jfk")


@pytest.fixture(scope="session")
def ru_short_fixture() -> AudioFixture:
    return _load_audio_fixture("ru_short")
