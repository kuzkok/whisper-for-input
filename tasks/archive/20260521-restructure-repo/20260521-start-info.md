# Реструктуризация репозитория: backend + потребители рядом

Дата: 2026-05-21
Статус: ✅ ВЫПОЛНЕНО 2026-05-29 (слито в master, fast-forward)

## Результат выполнения (2026-05-29)

Сделано на отдельной ветке `restructure-repo` атомарными коммитами, затем
fast-forward в `master`. Итоговые коммиты:

| Коммит | Шаг | Что |
|---|---|---|
| `5033d40` | 1 | Импорт audio-tools через merge неродственной истории |
| `fc8349c` | 2 | `bubble_sort_small.txt` → `formatting-transcript/tests/`, путь в `.gitignore` |
| `e8df1c7` | 3 | Server-стек → `whisper-for-input/` (LFS-указатели целы) |
| `25d9846` | 4 | `cli/transcribe.sh`, `cli/diarize.sh` |
| `ba7069a` | 5 | `install.sh` под новые пути |
| `7cb679e` | 6 | `CLAUDE.md` под новую раскладку |

**Отклонения от плана (согласованы по ходу):**
- Работали на отдельной ветке, а не прямо в master.
- Шаг 2: переносился **только** `bubble_sort_small.txt`. План ошибочно ждал
  в merge ещё `bubble_sort_small.article.md` (gitignored) и `Обзор…txt`
  (untracked) — `git merge` их не приносит, остались только в `../audio-tools`
  на диске. Команды `git mv` для них были бы фейлом.
- Шаг 3: `secrets/` переносился обычным `mv` (gitignored, не tracked) —
  `git mv` на нём упал бы.
- Временный remote `audio-tools-local` удалён после импорта.

**Верификация:** `git status` чистый; `pytest` (smoke+unit) — 23 passed,
8 deselected; пути `install.sh`/`Dockerfile`/`download-models.sh` резолвятся.

**Не сделано (намеренно, на пользователе):** пересоздание `.venv` внутри
`whisper-for-input/` (сейчас работает из корневого), удаление `../audio-tools`,
push в origin.

## Зачем

Сейчас в репо смешаны бэкенд (`server.py` + контейнер) и один потребитель
(`voice-input/`) в подпапке, плюс два untracked shell-скрипта в корне
(`transcribe.sh`, `diarize.sh`). Параллельно живёт отдельный локальный
git-репо `../audio-tools/` с Claude-skill `formatting-transcript/`, который
тоже логически потребитель транскриптов.

Цель: один монорепо со всеми компонентами в top-level папках. Бэкенд
переезжает в свою папку, его имя совпадает с именем репо.

## Итоговая раскладка

```
whisper-for-input/                    ← git root (репо, github remote)
├── whisper-for-input/                ← сервер
│   ├── server.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── whisper-for-input.container
│   ├── download-models.sh
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── pytest.ini
│   ├── .gitattributes      (LFS-правила для tests/fixtures)
│   ├── secrets/hf_token
│   └── tests/              (server-тесты)
├── voice-input/                      (как сейчас)
│   ├── voice-input.py
│   └── voice-input.service
├── cli/                              (shell-потребители)
│   ├── transcribe.sh
│   └── diarize.sh
├── formatting-transcript/            (из audio-tools через subtree merge)
│   ├── format_transcript.py
│   ├── chunk-prompt.md
│   ├── SKILL.md
│   └── tests/
├── docs/                             (как есть, только superpowers/specs/)
├── tasks/                            (этот файл и будущие)
├── install.sh                        (обновлённые пути)
├── CLAUDE.md                         (обновлённые пути)
├── .gitignore
└── .git/
```

## План коммитов

Идём по master атомарными коммитами; любой шаг откатывается через
`git reset --hard HEAD~N`. На spin-loop fix в `voice-input.py`
этот рефакторинг не завязан — фикс делаем отдельной задачей после.

### ✅ 1. `git subtree merge` для audio-tools

```bash
git remote add audio-tools-local /path/to/audio-tools
git fetch audio-tools-local
git merge --allow-unrelated-histories audio-tools-local/master \
    -m "Импорт audio-tools (formatting-transcript) в монорепо"
```

Принесёт в корень: `formatting-transcript/`, `tests/bubble_sort_small.*`,
`tests/Обзор...txt`, `.gitignore` (с правилом `tests/*.article.md`).

Конфликт на `.gitignore` ожидаем — разрешаем слиянием правил (root-овский
+ `tests/*.article.md` пока что; путь правим в шаге 2).

### ✅ 2. Сводим фикстуры formatting-transcript в свою папку

```bash
mkdir -p formatting-transcript/tests
git mv tests/bubble_sort_small.txt formatting-transcript/tests/
git mv tests/bubble_sort_small.article.md formatting-transcript/tests/
git mv tests/Обзор*.txt formatting-transcript/tests/
# .gitignore: tests/*.article.md → formatting-transcript/tests/*.article.md
git commit -m "Свести фикстуры formatting-transcript в formatting-transcript/tests/"
```

### ✅ 3. Вынести server-стек в `whisper-for-input/`

```bash
mkdir whisper-for-input
git mv server.py whisper-for-input/
git mv Dockerfile whisper-for-input/
git mv docker-compose.yml whisper-for-input/
git mv whisper-for-input.container whisper-for-input/
git mv download-models.sh whisper-for-input/
git mv requirements.txt whisper-for-input/
git mv requirements-dev.txt whisper-for-input/
git mv pytest.ini whisper-for-input/
git mv .gitattributes whisper-for-input/.gitattributes
git mv secrets whisper-for-input/secrets
git mv tests whisper-for-input/tests
git commit -m "Вынести server-стек в подпапку whisper-for-input/"
```

Что почему стоит вместе:

- `download-models.sh` использует `$SCRIPT_DIR/secrets/hf_token` —
  переносится с `secrets/` вместе, относительный путь сохраняется.
- `Dockerfile` делает `COPY requirements.txt` / `COPY server.py` —
  build context должен быть `whisper-for-input/`, тогда пути корректны.
- `pytest.ini` имеет `testpaths = tests` — рядом с `tests/` всё работает.
- `.gitattributes` (LFS-правила вида `tests/fixtures/*.wav`) уходит в
  `whisper-for-input/.gitattributes`, чтобы паттерны продолжали матчить
  относительно своего нового положения.
- `whisper-for-input.container` (Quadlet) — внутри только volume-пути
  на хосте (`~/.local/share/whisper-for-input/...`), они не зависят
  от расположения файла.

### ✅ 4. Завести `cli/`

```bash
mkdir cli
mv transcribe.sh diarize.sh cli/   # пока untracked — без git mv
git add cli/
git commit -m "Завести cli/transcribe.sh и cli/diarize.sh"
```

### ✅ 5. Переписать install.sh под новые пути

Правки:

- `cp "$SCRIPT_DIR/whisper-for-input.container" ...`
  → `cp "$SCRIPT_DIR/whisper-for-input/whisper-for-input.container" ...`
- `podman build -t whisper-for-input:latest "$SCRIPT_DIR"`
  → `podman build -t whisper-for-input:latest "$SCRIPT_DIR/whisper-for-input"`
- В подсказке summary: `./download-models.sh` →
  `./whisper-for-input/download-models.sh` (или `cd whisper-for-input && ...`).
- voice-input пути не трогаем — они уже в `voice-input/`.

```bash
git add install.sh
git commit -m "install.sh: пути под whisper-for-input/ как server-папку"
```

### ✅ 6. Обновить CLAUDE.md

Места, где есть пути:

- Описание `server.py` / `voice-input/voice-input.py` — server.py теперь
  в `whisper-for-input/server.py`.
- Секция "Models live outside the repo" — пути в `whisper-for-input.container`,
  `docker-compose.yml`, `download-models.sh` остаются как есть на хосте,
  но сами файлы переехали в `whisper-for-input/`.
- "Common commands": `./download-models.sh`, `podman build`, `docker compose up`
  — добавить `cd whisper-for-input` или поправить пути.
- "Локальная дев-среда и тесты": `python3.12 -m venv .venv` теперь делается
  внутри `whisper-for-input/`; пути к фикстурам `tests/fixtures/*.wav`
  становятся `whisper-for-input/tests/fixtures/*.wav` если запускать pytest
  из корня.
- Editing gotchas: `voice-input/voice-input.py is copied to ~/.local/bin/voice-input`
  — остаётся как есть.
- Добавить упоминание `cli/` и `formatting-transcript/` и кто их потребитель.

```bash
git add CLAUDE.md
git commit -m "CLAUDE.md: пути и команды под новую раскладку"
```

## Что **не** трогаем

- `.venv/` (gitignored) — физически останется в корне старого пути.
  Внутри venv абсолютные пути в shebang'ах — после миграции рекомендуется
  пересоздать в `whisper-for-input/.venv`:
  ```
  cd whisper-for-input
  rm -rf ../.venv
  python3.12 -m venv .venv
  .venv/bin/pip install -r requirements.txt -r requirements-dev.txt \
      --extra-index-url https://download.pytorch.org/whl/cu128
  ```
- Root `.gitignore` — паттерны (`secrets/`, `__pycache__/`, `.venv/`,
  `.pytest_cache/`) матчат на любой глубине, оставляем.
- `~/.local/share/whisper-for-input/` (модели на хосте) — внешний путь,
  не часть репо.
- `docs/superpowers/specs/` — оставляем в корне, это Claude-tooling
  артефакт, не server-specific.
- Старый локальный репо `../audio-tools/` — после subtree merge становится
  избыточным, но удалять его в рамках этой задачи не будем, пусть пользователь
  решит сам.
- github remote и имя репо — пока остаются `whisper-for-input`; пользователь
  планирует переименовать позже отдельной операцией (вероятно в `audio-tools`).

## Проверки после миграции

1. `git status` чистый.
2. `cd whisper-for-input && .venv/bin/pytest -m smoke` (после пересоздания venv).
3. `./install.sh` проходит без ошибок (можно прогнать "вхолостую" — в этой
   ветке podman build займёт время; если образ уже собран, install.sh его
   пропустит).
4. Просмотр `cd whisper-for-input && podman build -t whisper-for-input:latest .`
   (опционально — только если правили Dockerfile).
5. `systemctl --user restart whisper-for-input voice-input` и проверка логов.

## Отложено

- **Фикс spin-loop в voice-input при отвале устройства** — вынесен в отдельную
  задачу `tasks/20260529-voice-input-spin-loop-fix/` (2026-05-29). Ниже —
  исходная диагностика, перенесённая в ту задачу.
  Сам баг диагностирован: при отключении evdev-устройства (Microsoft 2.4GHz
  receiver и т.п.) `selector.select()` возвращает мёртвый fd, `dev.read()`
  бросает OSError, который глушится `except OSError: pass` и fd не
  дерегистрируется → tight loop, 100% CPU. Фикс: добавить
  `DeviceWatcher` с `discover()` (периодический rescan для hotplug) и
  `drop()` на OSError, тесты с fake-устройствами. Делаем после restructure
  в новой раскладке (`voice-input/voice-input.py` + `voice-input/tests/`).
