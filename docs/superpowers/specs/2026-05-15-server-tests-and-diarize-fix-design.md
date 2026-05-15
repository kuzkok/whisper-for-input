# Тесты для `server.py` + починка API диаризации

**Дата:** 2026-05-15
**Статус:** реализовано 2026-05-15. Все 31 тест зелёные (smoke 6 + unit 17 + integration 8). См. секцию «Реализация» в конце документа.

## Проблема

`server.py` был сгенерирован против устаревшего surface'а WhisperX и ни разу не запускался end-to-end. Первый же реальный вызов `/diarize` падает с `AttributeError: module 'whisperx' has no attribute 'DiarizationPipeline'`. Инспекция установленной библиотеки (`whisperx==3.8.5`, зафиксировано в `requirements.txt`) подтверждает три конкретных расхождения с реальным API, которые поймал бы любой реальный вызов:

1. `whisperx.DiarizationPipeline` больше не существует на верхнем уровне — он переехал в `whisperx.diarize.DiarizationPipeline`.
2. Параметр конструктора `use_auth_token` переименован в `token`.
3. Дефолтная модель пайплайна теперь `pyannote/speaker-diarization-community-1`. `download-models.sh` качает только `pyannote/speaker-diarization-3.1`, а контейнер запускается с `HF_HUB_OFFLINE=1` — без явного `model_name=` пайплайн пытается скачать community-1 и падает на cache miss.
4. **(Найдено integration-тестами.)** Даже когда явно указан `model_name="pyannote/speaker-diarization-3.1"`, pyannote.audio 4.x всё равно лезет в `pyannote/speaker-diarization-community-1` за `xvec_transform.npz` — это PLDA-компонент, который community-1 «расшарил» с 3.1-пайплайном. Поэтому `download-models.sh` обязан качать **обе** модели, а не только 3.1.
5. **(Найдено integration-тестами.)** `INITIAL_PROMPT` (русский, про разработчика) применялся ко **всем** запросам, потому что зашивался в `asr_options` на этапе `whisperx.load_model()`. Для `/transcribe` (voice-input диктовка разработчика) prompt полезен, для `/diarize` (произвольная встреча/видео) — он смещает Whisper в русский контекст и заставляет переводить английскую речь даже при `language="en"`. Решение — мутировать `_model.options` per-request через `dataclasses.replace()` (TranscriptionOptions — dataclass faster_whisper, whisperx внутри сам пользуется тем же паттерном в `whisperx/asr.py:262`).

Остальные функции whisperx, которые использует `server.py` (`load_model`, `load_align_model`, `align`, `assign_word_speakers`, `load_audio`), доступны на верхнем уровне, но их публичные сигнатуры обёрнуты декораторами с `*args, **kwargs` — реальные контракты можно проверить только реальным вызовом. Нужен тестовый стенд, который дёргает настоящую библиотеку, а не stub'ы на уровне импорта.

Пользователь хочет итерироваться локально без пересборки 8.93 GB образа после каждой правки, а пересобирать только когда локальный цикл стал зелёным.

## Цели

- Ловить баги типа API-дрифта, который только что вылез, за доли секунды и до любой пересборки образа.
- Покрыть логику FastAPI-обработчиков (формирование ответа, нормализация параметров, группировка спикеров в `format=text`) без необходимости в GPU.
- Проверить полный пайплайн против реального WhisperX, реальных моделей и реального GPU на машине разработчика.
- Поднять локальный дев-цикл (venv на хосте, без круга через контейнер), который достаточно близко повторяет рантайм контейнера, чтобы быть предсказательным.

## Не цели

- Никакой CI-интеграции в этой итерации (gpu-тесты локальны по умолчанию; smoke и unit *могли бы* крутиться в CI, но никакого пайплайна мы не добавляем).
- Никаких тестов для `voice-input/voice-input.py` — отдельный процесс, отдельные заботы, не имеет отношения к багу диаризации.
- Никаких изменений в `Dockerfile`, `requirements.txt`, `whisper-for-input.container`, `docker-compose.yml`, `install.sh`.
- Никакого рефакторинга `server.py` сверх минимальных правок, нужных чтобы тесты прошли.

## Архитектура

### Раскладка тестов

```
tests/
  __init__.py
  conftest.py             # общие fixtures (TestClient, загрузчики аудио-фикстур)
  fixtures/
    ru_short.wav          # ~3-5с mono 16k из Common Voice ru, ~50-150 KB (LFS)
    ru_short.json         # {transcript, language, source_url, license, duration_s}
    jfk.wav               # ~11с mono 16k из whisper.cpp samples, ~340 KB (LFS)
    jfk.json              # {transcript, language, source_url, license, duration_s}
  test_smoke.py           # @pytest.mark.smoke — без загрузки моделей, миллисекунды
  test_unit.py            # @pytest.mark.unit  — TestClient + monkeypatch'ed whisperx
  test_integration.py     # @pytest.mark.gpu   — реальные модели, реальный GPU
pytest.ini                # маркеры + addopts = -m "not gpu"
requirements-dev.txt      # pytest, httpx, soundfile (для sanity check фикстур)
.gitattributes            # tests/fixtures/*.wav через git-lfs
.gitignore                # добавить .venv/
```

### Три уровня тестов

#### Smoke (`test_smoke.py`, маркер `smoke`)

Без загрузки моделей. Валидирует import surface, от которого зависит сервер. Время прогона — доли секунды. Этот уровень — канарейка для API-дрифта; именно он громко упал бы в день, когда `whisperx.DiarizationPipeline` переехал.

- `import server` отрабатывает без ошибок.
- `whisperx.diarize.DiarizationPipeline` существует и его `__init__` принимает kwargs `model_name`, `token`, `device` (проверка через `inspect.signature`).
- Все имена whisperx верхнего уровня, которые упоминает `server.py`, резолвятся: `load_model`, `load_align_model`, `align`, `assign_word_speakers`, `load_audio`.
- `server._normalize_language` корректно обрабатывает `""`, `"auto"`, `"ru"` (пустое/auto → `None`, `"ru"` → `"ru"`).
- `GET /health` возвращает ожидаемую схему (ключи: `status`, `model`, `device`, `compute_type`, `diarize_loaded`, `align_languages_loaded`). Для этого теста сервер поднимается без реальной модели — `_model` может быть `None`; проверяем только форму ответа.

#### Unit (`test_unit.py`, маркер `unit`)

Использует `fastapi.testclient.TestClient` с полностью monkeypatch'нутым whisperx. Время прогона — доли секунды. Валидирует логику обработчиков в изоляции:

- `/transcribe` склеивает segments в одну строку и тримит её.
- `/transcribe` с пустым полем `language` вызывает модель с `language=None`.
- `/diarize` с `format=json` возвращает `{language, segments}` с ожидаемой структурой каждого сегмента (`start`, `end`, `speaker`, `text`).
- `/diarize` с `format=text` корректно группирует подряд идущие сегменты одного спикера в блоки `[SPEAKER_XX] joined text`, разделённые пустой строкой. (Это самая нетривиальная логика в `server.py:159-172` и приоритетная цель для unit-теста.)
- `/diarize` с `min_speakers=2` и `max_speakers=4` пробрасывает оба значения в diarize-пайплайн; с обоими `0` не передаёт ни одного.
- `/diarize` с `language="ru"` пропускает автодетект (передаёт `language="ru"` в `transcribe`).

Unit-тесты monkeypatch'ат атрибуты модуля `server` (например, `server._model`, глобальный `_align_models`, `whisperx.align`, `whisperx.assign_word_speakers`, `whisperx.diarize.DiarizationPipeline`), чтобы обработчики гоняли свой реальный код против поддельных выходов.

#### Integration (`test_integration.py`, маркер `gpu`)

Реальный WhisperX, реальные модели из `~/.local/share/whisper-for-input/models`, реальный GPU. По умолчанию пропускается (`pytest.ini` ставит `addopts = -m "not gpu"`); явно включается через `pytest -m gpu`.

Модели грузятся один раз на сессию через session-scoped fixture, который оборачивает `TestClient(app)` в `with`-блок (TestClient запускает FastAPI lifespan на enter/exit, заполняя `_model`).

Тесты:

- `/transcribe` параметризованный по `{(ru_short.wav, "ru"), (jfk.wav, "en")}` — возвращает непустой текст в ожидаемом алфавите (кириллица для ru, ASCII для en). Без точного сравнения строк — конкретные транскрипты слишком хрупки между версиями whisper.
- `/diarize?format=json` параметризованный аналогично — `language` в ответе совпадает со входным, `segments` непустой, `start < end` для каждого сегмента, и хотя бы один сегмент имеет `speaker`, матчащийся `^SPEAKER_\d+$`. (Часть коротких сегментов может легитимно остаться `"UNKNOWN"` из `seg.get("speaker", "UNKNOWN")` в `server.py:153`, когда `assign_word_speakers` не приписал им спикера; на этом не падаем — падаем только если *вообще ни один* сегмент не получил атрибуцию.)
- `/diarize?format=text` — ответ содержит хотя бы один маркер `[SPEAKER_`.
- `/health` после обоих вызовов diarize — `diarize_loaded == True`, `align_languages_loaded` содержит и `"ru"`, и `"en"`.
- Отдельный тест (своя сессия, маркеры `gpu` + `slow`) поднимает приложение с `WHISPERX_PRELOAD_DIARIZE=1` и проверяет, что `/health` рапортует `diarize_loaded == True` сразу, до любого вызова `/diarize`.

### Аудио-фикстуры

Обе фикстуры коммитятся в репо как бинарные блобы через git-lfs. Файлы кладёт пользователь вручную; никакого скрипта-загрузчика мы не пишем — для двух статичных файлов это лишняя обвязка.

- **`ru_short.wav`** — один короткий клип (~3–6с) из Common Voice ru (CC-0), сконверченный в mono 16k WAV. Источник можно записать в sidecar JSON.
- **`jfk.wav`** — `https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav` (~340 KB, public domain, mono 16k).

Каждому WAV сопутствует sidecar JSON `{transcript, language, source_url, license, duration_s, sample_rate, channels}`. JSON-ы пишутся вручную и коммитятся обычным образом (не через LFS — текст). Тесты грузят WAV'ы и JSON'ы с диска напрямую.

`.gitattributes` добавляет правило `tests/fixtures/*.wav filter=lfs diff=lfs merge=lfs -text`. В `conftest.py` загрузчик фикстур делает sanity-проверку: открывает WAV через `soundfile.info` и при ошибке (например, попался LFS pointer вместо реального файла из-за непронесённого `git lfs pull`) поднимает понятный `pytest.skip` или fail с подсказкой запустить `git lfs pull`.

### Локальная дев-среда

Один venv в `.venv/` в корне репо, gitignore'нутый. Зеркалит стек контейнера: Python 3.12 + cu128 wheels:

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

`requirements-dev.txt`:

```
pytest>=8
httpx>=0.27
soundfile>=0.12
```

(`httpx` нужен `fastapi.testclient.TestClient`. `soundfile` используется в `download_fixtures.py` и в sanity-проверке, что фикстуры действительно mono 16k.)

`pytest.ini`:

```ini
[pytest]
markers =
    smoke: проверки уровня импорта, без загрузки моделей
    unit: логика обработчиков с замоканным whisperx
    gpu: полный пайплайн против реальных моделей на GPU
addopts = -m "not gpu"
```

Дефолтный вызов `pytest`, таким образом, гоняет smoke + unit (быстро, без GPU). `pytest -m gpu` запускает тяжёлый слой.

### Правки в `server.py`

1. Добавить явный импорт в верхней части файла: `from whisperx.diarize import DiarizationPipeline`. Это заставляет API-дрифт падать на этапе импорта модуля (ловится smoke-тестом), а не лениво на первом `/diarize`.
2. Переписать `get_diarize_pipeline` под новый конструктор: `model_name="pyannote/speaker-diarization-3.1"`, `token=None` (вместо `use_auth_token=`), плюс комментарий про offline-связку с `download-models.sh`.
3. Добавить helper `_set_initial_prompt(prompt)`, который через `dataclasses.replace()` мутирует `_model.options.initial_prompt` per-request.
4. В `/transcribe` вызвать `_set_initial_prompt(INITIAL_PROMPT or None)` перед `_model.transcribe(...)` — voice-input use case, prompt уместен.
5. В `/diarize` вызвать `_set_initial_prompt(None)` перед `_model.transcribe(...)` — произвольный контент, prompt вреден.

`download-models.sh` дополнительно качает `pyannote/speaker-diarization-community-1` (зависимость pyannote.audio 4.x для 3.1-пайплайна) и пропускает уже скачанные репо по наличию каталога `models--<org>--<repo>`.

### Документация

Добавить секцию «Локальная дев-среда / запуск тестов» в `CLAUDE.md` (предпочтительнее, чем отдельный README — `CLAUDE.md` пользователя уже канонический проектный документ). Включает:

- команду установки venv;
- четыре шаблона запуска (smoke / unit / gpu / `uvicorn --reload`);
- «пересобирать образ только после того, как `pytest -m gpu` стало зелёным»;
- упоминание про `git lfs pull` для подтягивания фикстур после клонирования.

## Workflow

TDD-подобная последовательность:

1. Создать `.venv/`, поставить dev-зависимости.
2. Написать smoke + unit + integration тесты под *желаемый* контракт.
3. Запустить `pytest -m smoke` → падает на импорте `whisperx.diarize.DiarizationPipeline` (доказывает, что тест ловит баг).
4. Применить три правки в `server.py`.
5. `pytest -m smoke` → зелено.
6. `pytest -m unit` → зелено.
7. `pytest -m gpu` → зелено (на первом запуске скачаются ~360 MB английских torchaudio alignment-весов в `~/.cache/torch`; последующие запуски — из кэша).
8. Когда всё зелёное, пересобрать образ: `podman build -t whisper-for-input:latest .` затем `systemctl --user restart whisper-for-input`.
9. Smoke-проверка работающего контейнера: `curl -F 'file=@tests/fixtures/ru_short.wav' -F 'language=ru' http://localhost:8000/diarize`.

## Риски и открытые вопросы

- **Английский alignment скачивает 360 MB на первом `pytest -m gpu`.** Это разовая стоимость на машину, кэшируется в `~/.cache/torch`. Задокументировано в docstring тестового файла.
- **LFS pointer вместо реального WAV.** Если кто-то склонирует репо без `git lfs pull`, фикстуры будут текстовыми pointer-файлами, и WhisperX упадёт неинформативно. Митигация: sanity-проверка в `conftest.py`, упомянутая выше.
- **Integration-тесты могут флексить по содержанию транскрипта.** Намеренно ассертим только «непустой + правильный алфавит», без точного текста. Если даже это будет флексить — ужесточим только после реального падения.
- **Тест на `PRELOAD_DIARIZE=1` перезагружает приложение.** Он гоняется в собственной session-scoped fixture с отдельным контекстом TestClient, поэтому не интерферирует с основными integration-тестами. Помечен `slow` дополнительно к `gpu`, чтобы можно было пропускать в быстрой итерации.

## Вне скоупа (явно)

- Тесты для `voice-input/voice-input.py` (evdev hotkey loop, `arecord`, интеграция с `wl-copy`).
- Container-level integration-тесты (запуск настоящего rootless Podman контейнера с volume mounts).
- CI-конфиг.
- Бенчмарки производительности.
- Что-либо в `Dockerfile`, `requirements.txt` или systemd unit'ах.

## Реализация (2026-05-15)

Финальный счёт: **31 тест зелёный** — smoke 6, unit 17, integration 8.

### Файлы

- `tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`, `tests/test_unit.py`, `tests/test_integration.py` — новые.
- `tests/fixtures/jfk.wav` + `jfk.json` — английская фикстура из `whisper.cpp` samples (public domain, 11с mono 16k).
- `tests/fixtures/ru_short.ogx` + `ru_short.json` — русская фикстура с Kaggle (`maratdv` dataset, 14.59с mono 8 kHz, OGG Vorbis). Реальный транскрипт (telesales, B2B Trade) получен через `/diarize` без prompt'а и закоммичен в JSON.
- `pytest.ini`, `requirements-dev.txt` — новые.
- `.gitattributes` — LFS-правила для `tests/fixtures/*.{wav,mp3,ogg,ogx,flac,m4a}`.
- `.gitignore` — добавлены `.venv/`, `.pytest_cache/`, `.*.kate-swp`.
- `server.py` — изменения (детально ниже).
- `download-models.sh` — добавлены `pyannote/speaker-diarization-community-1` и skip-логика по наличию `models--<org>--<repo>` в HF-кэше.
- `CLAUDE.md` — секция «Локальная дев-среда и тесты», заметка про per-endpoint `INITIAL_PROMPT`, упоминание community-1 в архитектурных нотах.

### Что в server.py поменялось

1. `from whisperx.diarize import DiarizationPipeline` сверху (fail-fast на API-дрифте, ловится smoke-тестом).
2. `import dataclasses` сверху.
3. `get_diarize_pipeline()` переписан: `whisperx.diarize.DiarizationPipeline(model_name="pyannote/speaker-diarization-3.1", token=None, device=DEVICE)`.
4. Helper `_set_initial_prompt(prompt)` через `dataclasses.replace(_model.options, initial_prompt=prompt)` (TranscriptionOptions — dataclass faster_whisper, не namedtuple — это всплыло в integration).
5. `/transcribe` вызывает `_set_initial_prompt(INITIAL_PROMPT or None)` перед `_model.transcribe(...)` — voice-input use case.
6. `/diarize` вызывает `_set_initial_prompt(None)` перед `_model.transcribe(...)` — нейтральный контекст для произвольных встреч/видео.

### Что нашли integration-тесты сверх трёх изначальных API drift'ов

4. pyannote.audio 4.x для `pyannote/speaker-diarization-3.1` лениво лезет в `pyannote/speaker-diarization-community-1` за `xvec_transform.npz` (PLDA-компонент). Под `HF_HUB_OFFLINE=1` это уронит контейнер. → `download-models.sh` теперь качает обе модели.
5. `INITIAL_PROMPT` (русский, про разработчика) применялся ко всем запросам через `asr_options` в `load_model()`. Для `/diarize` это смещало Whisper в перевод английской речи в русский. → Per-request мутация `_model.options.initial_prompt`.
6. `TranscriptionOptions` не namedtuple (как сначала предположил спек-черновик), а dataclass faster_whisper. → `dataclasses.replace`, не `_replace`.

### Окружение для локального прогона integration

К стандартному `pytest -m gpu` нужны env vars (зеркалят runtime контейнера):

```bash
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/npp/lib" \
HF_HOME="$HOME/.local/share/whisper-for-input/models" \
HF_HUB_OFFLINE=1 \
WHISPER_MODEL=dropbox-dash/faster-whisper-large-v3-turbo \
.venv/bin/pytest -m gpu
```

`LD_LIBRARY_PATH` — для torchcodec/NPP libs из nvidia-* колёс. `HF_HOME` — указать на пред-скачанный кэш моделей. `HF_HUB_OFFLINE=1` — гарантия что не полезет в сеть. `WHISPER_MODEL` — берём модель из кэша по полному репо-id, иначе faster-whisper попытается скачать дефолтный `large-v3-turbo`.

Также на хосте требуется системный `ffmpeg` (контейнер ставит сам через `apt`, локально — `dnf install ffmpeg` из RPM Fusion).

### Отклонения от изначального плана

- **Фикстура ru_short — `.ogx`, не `.wav`.** Изначально предполагался WAV (~50–150 KB) с Common Voice ru, по факту коммитим OGG Vorbis 8 kHz mono из Kaggle (~47 KB). conftest и `.gitattributes` теперь принимают любой из `.wav/.mp3/.ogg/.ogx/.flac/.m4a` — формат не зашит в код.
- **Long ru-фикстура (14.59с вместо 3–6с).** Не критично — diarize для односпикерного ru-клипа работает; теста на множественных спикерах нет (это требовало бы multi-speaker записи, что переусложнило бы фикстуру).
- **`download_fixtures.py` не написан.** Фикстуры пользователь кладёт вручную; для двух статичных файлов отдельный скрипт-обвязка избыточен.
- **`test_transcribe_real[jfk-ascii]` ослаблен до «непустой текст».** После того как `/transcribe` использует `INITIAL_PROMPT` (а не "пустой в integration"), английский транскрипт намеренно переводится в русский — это и есть voice-input use case. Английский путь сам по себе тестируется через `/diarize` (там prompt = None и язык сохраняется).
- **Лицензия фикстуры `ru_short.ogx` зафиксирована как `"sample"`** (Kaggle dataset `maratdv`). Если репо публикуется — проверить лицензию датасета.
