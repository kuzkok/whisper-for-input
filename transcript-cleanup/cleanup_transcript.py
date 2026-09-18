#!/usr/bin/env python3
"""Чистка диаризованного транскрипта: карта спикеров + глоссарные правки.

Usage: cleanup_transcript.py <source> <context-file> [-o out] [--yes]

Вход — транскрипт вида «[HH:MM:SS] [SPEAKER_XX] текст» (выход cli/diarize.sh)
плюс проектный контекст-файл. Выход — тот же каркас с именами вместо
SPEAKER_XX (где спикер опознан по содержанию) и вычищенным текстом реплик.

Две фазы, обе через headless `claude -p --tools "" --json-schema`:
  1. sonnet строит карту «SPEAKER_XX → имя» по скелету встречи (первые ~30
     слов каждой реплики); карта показывается, подтверждается по stdin
     (или --yes); подстановка имён — детерминированная замена строк здесь.
  2. haiku чистит текст чанками по ~40 реплик; каркас (таймкоды, метки)
     через модель не проходит — скрипт сшивает выход сам.

LLM вызывается только в inference-режиме: JSON на входе, schema-валидный
JSON на выходе, без инструментов. Скелетизация, чанкование, подстановка,
ретраи и обе верификации — детерминированный код.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CHUNK_REPLICAS = 40
CONTEXT_REPLICAS = 3
SKELETON_WORDS = 30
SIGNAL_HEAD_REPLICAS = 2  # первые реплики каждого спикера в сокращённом скелете
MAX_RETRIES = 2

PHASE1_MODEL = "sonnet"
PHASE2_MODEL = "haiku"
PHASE1_TIMEOUT_S = 300
# Фаза 2: 420 с — роутер может отдавать под слотом haiku медленную модель
# (прогон на sonnet через роутер таймаутил чанк на 180 с и уходил в ретрай).
PHASE2_TIMEOUT_S = 420
# Пороги --max-budget-usd: через корпоративный роутер это фиктивный счётчик
# (costBasis unknown, без кэширования промптов) — числа подобраны под него;
# на подписке или официальном API вызов дешевле на порядок, менять не требуется.
# Фаза 2: $1.50 с запасом — счётчик роутера тарифицирует чанк по полному
# входу (~23k токенов с обвязкой CLI); у sonnet-слота 180 с/`$0.30` не хватало.
PHASE1_BUDGET_USD = "0.50"
PHASE2_BUDGET_USD = "1.50"

# Каркас: [HH:MM:SS] [метка] текст. Метка — SPEAKER_XX или UNKNOWN.
LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] \[([^\]]+)\] (.*)$")
QUOTE_LIMIT = 60

SCHEMA_MAP = {
    "type": "object",
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "name": {"type": ["string", "null"]},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "time": {"type": "string"},
                                "type": {"type": "string"},
                            },
                            "required": ["time", "type"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["speaker", "name", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["speakers"],
    "additionalProperties": False,
}

SCHEMA_CLEAN = {
    "type": "object",
    "properties": {
        "cleaned": {"type": "array", "items": {"type": "string"}, "minItems": 1}
    },
    "required": ["cleaned"],
    "additionalProperties": False,
}


@dataclass
class Replica:
    time: str
    label: str  # SPEAKER_XX / UNKNOWN на входе; имя после подстановки
    text: str

    def line(self) -> str:
        return f"[{self.time}] [{self.label}] {self.text}"


# ── парсинг / скелет / чанки ─────────────────────────────────────────────────


def extract_address_forms(context_file: str) -> list[str]:
    """Обращения из таблиц контекст-файла вида «| Обращение в речи | Имя |».

    Парсятся только таблицы, у которых заголовок первой колонки содержит
    «Обращение»; прочие таблицы и списки остаются текстом для LLM. Значения
    левой колонки разворачиваются через запятую («Аня, Анна» → два
    обращения). Правая колонка не нужна: предфильтр лишь сужает вход фазы 1,
    имя подставляет LLM по полному контекст-файлу.
    """
    forms: list[str] = []
    in_address_table = False
    for line in context_file.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_address_table = False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            in_address_table = False
            continue
        if "обращение" in cells[0].lower():
            in_address_table = True
            continue  # заголовочная строка
        if set(cells[0]) <= set("-: "):
            continue  # разделитель |---|---|
        if not in_address_table:
            continue
        for variant in cells[0].split(","):
            v = variant.strip()
            if v:
                forms.append(v)
    return forms


def _stem(form: str) -> str:
    """Основа для матчинга падежных форм: «Вика» → «Вик» (ловит «Вики»,
    «Вике», «Викторию»), «Дима» → «Дим», «Олег» → без усечения.
    Матч без правой границы — предфильтру лучше ошибиться в сторону
    лишней реплики, чем пропустить сигнальную."""
    if len(form) > 3 and form[-1].lower() in "аяоёеиыуэюьй":
        return form[:-1]
    return form


def has_address_hit(text: str, forms: list[str]) -> bool:
    for f in forms:
        if re.search(r"(?<!\w)" + re.escape(_stem(f)), text, re.IGNORECASE):
            return True
    return False


def select_skeleton_indices(replicas: list[Replica], forms: list[str]) -> list[int]:
    """Индексы реплик сокращённого скелета: содержащие обращение, следующая
    после каждой такой (кандидат-адресат) и первые SIGNAL_HEAD_REPLICAS реплик
    каждого спикера (самопрезентация, контекст). Дедуп + хронологический
    порядок. Сокращённый скелет обязан покрывать все метки спикеров —
    инвариант для проверки карты."""
    chosen: set[int] = set()
    head_seen: dict[str, int] = {}
    for i, r in enumerate(replicas):
        if has_address_hit(r.text, forms):
            chosen.add(i)
            if i + 1 < len(replicas):
                chosen.add(i + 1)
        if head_seen.get(r.label, 0) < SIGNAL_HEAD_REPLICAS:
            chosen.add(i)
            head_seen[r.label] = head_seen.get(r.label, 0) + 1
    return sorted(chosen)


def parse_transcript(src_text: str) -> list[Replica]:
    """Строки «[t] [метка] текст» → реплики. Многострочный хвост приклеивается
    к предыдущей реплике (в выходе server.py реплики однострочные, но
    ручная правка файла может внести перенос)."""
    replicas: list[Replica] = []
    for raw in src_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if m:
            replicas.append(Replica(m.group(1), m.group(2), m.group(3)))
        elif replicas:
            replicas[-1].text += " " + line
        else:
            raise ValueError(f"строка вне реплики: {raw[:60]!r}")
    if not replicas:
        raise ValueError("в файле не найдено ни одной реплики вида [HH:MM:SS] [метка] текст")
    return replicas


def skeletonize(replicas: list[Replica], nums: list[int] | None = None) -> list[dict]:
    """Скелет для карты спикеров: метка + первые SKELETON_WORDS слов реплики.
    nums — сквозные номера реплик полного транскрипта (для сокращённого
    скелета), по умолчанию нумерация по порядку."""
    out = []
    for j, r in enumerate(replicas):
        n = nums[j] if nums is not None else j + 1
        words = r.text.split()
        head = " ".join(words[:SKELETON_WORDS])
        if len(words) > SKELETON_WORDS:
            head += "…"
        out.append({"n": n, "time": r.time, "speaker": r.label, "head": head})
    return out


def chunk_replicas(replicas: list[Replica], size: int = CHUNK_REPLICAS) -> list[list[Replica]]:
    """Реплика — атом: между репликами не режем."""
    return [replicas[i : i + size] for i in range(0, len(replicas), size)]


def context_lines(replicas: list[Replica], upto: int, k: int = CONTEXT_REPLICAS) -> str:
    """k полных реплик до позиции upto — read-only контекст края чанка."""
    return "\n".join(r.line() for r in replicas[max(0, upto - k) : upto])


# ── подстановка имён ─────────────────────────────────────────────────────────


def substitute_names(replicas: list[Replica], mapping: dict[str, str]) -> list[Replica]:
    """Детерминированная замена меток: один SPEAKER_XX — одно имя на весь выход.
    LLM метки не переписывает, поэтому согласованность структурна. Пустое имя
    в карте («не опознан») оставляет исходную метку."""
    out = []
    for r in replicas:
        label = mapping.get(r.label) or r.label
        out.append(Replica(r.time, label, r.text))
    return out


# ── вызовы claude -p ─────────────────────────────────────────────────────────


def call_claude(
    system_prompt: str,
    user_msg: str,
    model: str,
    schema: dict,
    timeout_s: int,
    budget_usd: str,
) -> dict:
    try:
        proc = subprocess.run(
            [
                "claude", "-p",
                "--model", model,
                "--tools", "",
                "--no-session-persistence",
                "--max-budget-usd", budget_usd,
                "--system-prompt", system_prompt,
                "--output-format", "json",
                "--json-schema", json.dumps(schema),
                user_msg,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        tail = ((e.stderr or "") + (e.stdout or ""))[-800:]
        raise RuntimeError(f"claude -p rc={e.returncode}; tail: {tail!r}") from e
    envelope = json.loads(proc.stdout)
    if envelope.get("is_error"):
        raise RuntimeError(f"claude -p reported is_error: {envelope.get('result', '')!r}")
    structured = envelope.get("structured_output")
    if not isinstance(structured, dict):
        raise RuntimeError(f"missing/invalid structured_output in envelope: {envelope!r}")
    return structured


# ── фаза 1: карта спикеров ───────────────────────────────────────────────────


def map_speakers(
    map_prompt: str, skeleton: list[dict], context_file: str
) -> tuple[dict[str, str], dict[str, list[dict]]]:
    """Возвращает (метка → имя | не опознана, метка → опоры) по одному вызову
    sonnet. Выход модели компактный (time + type опор, без цитат) — генерация
    короткая; цитаты скрипт подтягивает из скелета по таймкоду. Ретраи:
    пустая/кривая карта — тоже ошибка вызова."""
    payload = {"skeleton": skeleton, "context_file": context_file}
    heads_by_time = {row["time"]: row["head"] for row in skeleton}
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            out = call_claude(
                map_prompt,
                json.dumps(payload, ensure_ascii=False),
                PHASE1_MODEL,
                SCHEMA_MAP,
                PHASE1_TIMEOUT_S,
                PHASE1_BUDGET_USD,
            )
            mapping: dict[str, str] = {}
            evidence: dict[str, list[dict]] = {}
            for row in out["speakers"]:
                mapping[row["speaker"]] = row["name"] or ""
                enriched = []
                for ev in row.get("evidence", []):
                    quote = heads_by_time.get(ev["time"], "")
                    enriched.append({"time": ev["time"], "quote": quote,
                                     "type": ev["type"]})
                evidence[row["speaker"]] = enriched
            known = {r["speaker"] for r in skeleton}
            unknown_meta = known - set(mapping)
            if unknown_meta:
                raise RuntimeError(f"карта не покрывает метки: {sorted(unknown_meta)}")
            if not any(mapping.values()):
                raise RuntimeError("карта пуста: ни одного опознанного спикера")
            return mapping, evidence
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            RuntimeError,
            KeyError,
        ) as e:
            last_err = e
            print(f"  карта спикеров: попытка {attempt + 1} — ошибка: {e}", file=sys.stderr)
    raise RuntimeError(f"карта спикеров: исчерпаны {MAX_RETRIES + 1} попыток; последняя: {last_err}")


def print_map(mapping: dict[str, str], evidence: dict[str, list[dict]]) -> None:
    print("\nКарта спикеров:", file=sys.stderr)
    for sp in sorted(mapping):
        name = mapping[sp]
        if name:
            print(f"  {sp} → {name}", file=sys.stderr)
            for ev in evidence.get(sp, []):
                q = ev["quote"]
                if len(q) > QUOTE_LIMIT:
                    q = q[:QUOTE_LIMIT] + "…"
                print(f"      [{ev['time']}] «{q}» ({ev['type']})", file=sys.stderr)
        else:
            print(f"  {sp} → (не опознан)", file=sys.stderr)


def confirm_map(
    mapping: dict[str, str], evidence: dict[str, list[dict]]
) -> dict[str, str]:
    """stdin: принять / править руками / выйти. Правка — строки
    «SPEAKER_XX Имя Фамилия» или «SPEAKER_XX -» (сброс), пустая строка — конец."""
    while True:
        print_map(mapping, evidence)
        print(
            "\n[a] принять / [e] править / [q] выйти > ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        choice = input().strip().lower()
        if choice in ("a", ""):
            return mapping
        if choice.startswith("q"):
            print("выход без изменений", file=sys.stderr)
            sys.exit(1)
        if choice.startswith("e"):
            print(
                "правки построчно: «SPEAKER_XX Имя Фамилия» или «SPEAKER_XX -» "
                "(сброс); пустая строка — закончить:",
                file=sys.stderr,
            )
            known = set(mapping)
            while True:
                try:
                    row = input().strip()
                except EOFError:
                    break
                if not row:
                    break
                parts = row.split(None, 1)
                if len(parts) != 2 or parts[0] not in known:
                    print(f"  ? не разобрать: {row!r}", file=sys.stderr)
                    continue
                sp, val = parts
                mapping[sp] = "" if val.strip() == "-" else val.strip()
                evidence.pop(sp, None)
            continue
        print("? ответ: a (принять), e (править), q (выйти)", file=sys.stderr)


# ── фаза 2: чистка текста ────────────────────────────────────────────────────


def clean_chunk(
    chunk_prompt: str,
    chunk: list[Replica],
    context: str,
    context_file: str,
    label: str,
) -> list[str]:
    """Очищенные тексты реплик чанка. Каркас не проходит через модель:
    на входе только n + text, метку невозможно испортить."""
    payload = {
        "replicas": [{"n": i, "text": r.text} for i, r in enumerate(chunk, start=1)],
        "context_lines": context,
        "context_file": context_file,
    }
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        if attempt:
            payload["retry_notice"] = (
                "Предыдущая попытка не прошла проверку: длина массива != числу реплик "
                "или есть пустые строки. Обработай каждую реплику и верни по строке на каждую."
            )
        try:
            out = call_claude(
                chunk_prompt,
                json.dumps(payload, ensure_ascii=False),
                PHASE2_MODEL,
                SCHEMA_CLEAN,
                PHASE2_TIMEOUT_S,
                PHASE2_BUDGET_USD,
            )
            cleaned = out["cleaned"]
            if len(cleaned) != len(chunk):
                raise RuntimeError(f"длина массива {len(cleaned)} != {len(chunk)} реплик")
            if any(not s.strip() for s in cleaned):
                raise RuntimeError("пустые строки в cleaned")
            return cleaned
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            RuntimeError,
            KeyError,
        ) as e:
            last_err = e
            print(f"  {label}: попытка {attempt + 1} — ошибка: {e}", file=sys.stderr)
    raise RuntimeError(f"{label}: исчерпаны {MAX_RETRIES + 1} попыток; последняя: {last_err}")


# ── верификации ──────────────────────────────────────────────────────────────


def verify_final(
    src: list[Replica], out: list[Replica], expected: list[Replica]
) -> list[str]:
    """Структурная верификация: каркас не сдвинулся. Список проблем; пустой
    список — ок. Сравниваются число реплик, таймкоды по порядку и метки
    (метки — против детерминированной подстановки фазы 1)."""
    problems = []
    if len(out) != len(src):
        problems.append(f"число реплик {len(out)} != {len(src)}")
    for i, (s, o) in enumerate(zip(src, out), start=1):
        if s.time != o.time:
            problems.append(f"реплика {i}: таймкод {o.time} != {s.time}")
    for i, (o, e) in enumerate(zip(out, expected), start=1):
        if o.label != e.label:
            problems.append(f"реплика {i}: метка {o.label!r} != {e.label!r}")
    return problems


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    args = sys.argv[1:]
    yes = False
    out_path_arg = None
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--yes":
            yes = True
        elif a == "-o" or a == "--output":
            i += 1
            if i >= len(args):
                print("-o требует аргумент", file=sys.stderr)
                return 2
            out_path_arg = args[i]
        elif a in ("-h", "--help"):
            print(__doc__.strip())
            return 0
        else:
            positional.append(a)
        i += 1
    if len(positional) != 2:
        print(
            "usage: cleanup_transcript.py <source> <context-file> [-o out] [--yes]",
            file=sys.stderr,
        )
        return 2

    src_path = Path(positional[0])
    ctx_path = Path(positional[1])
    if not src_path.is_file():
        print(f"файл не найден: {src_path}", file=sys.stderr)
        return 2
    if not ctx_path.is_file():
        print(f"контекст-файл не найден: {ctx_path}", file=sys.stderr)
        return 2
    out_path = Path(out_path_arg) if out_path_arg else src_path.with_suffix(".clean.txt")
    if out_path.exists():
        print(f"выход уже существует, не перезаписываю: {out_path}", file=sys.stderr)
        return 2
    if not sys.stdin.isatty() and not yes:
        print(
            "stdin не tty: интерактивное подтверждение карты недоступно. "
            "Используйте --yes или запускайте в терминале.",
            file=sys.stderr,
        )
        return 2

    here = Path(__file__).resolve().parent
    map_prompt = (here / "map-prompt.md").read_text(encoding="utf-8")
    chunk_prompt = (here / "chunk-prompt.md").read_text(encoding="utf-8")
    context_file = ctx_path.read_text(encoding="utf-8")

    src = parse_transcript(src_path.read_text(encoding="utf-8"))
    print(f"{len(src)} реплик, метки: {sorted({r.label for r in src})}", file=sys.stderr)

    # Фаза 1 — карта спикеров. Полный вход не нужен: детерминированный
    # предфильтр сужает скелет до сигнальных реплик (обращения, ответы,
    # первые реплики спикеров) — LLM решает по малому входу, полный
    # контекст-файл прикладывается как есть.
    forms = extract_address_forms(context_file)
    idx = select_skeleton_indices(src, forms)
    skeleton = skeletonize([src[i] for i in idx], nums=[i + 1 for i in idx])
    print(
        f"скелет: {len(idx)} из {len(src)} реплик (сигнальные + первые "
        f"{SIGNAL_HEAD_REPLICAS} каждого спикера)",
        file=sys.stderr,
    )
    mapping, evidence = map_speakers(map_prompt, skeleton, context_file)
    if yes:
        print_map(mapping, evidence)
        print("--yes: карта принята без подтверждения", file=sys.stderr)
    else:
        mapping = confirm_map(mapping, evidence)

    named = substitute_names(src, mapping)

    # Фаза 2 — чанковая чистка
    chunks = chunk_replicas(named)
    print(f"{len(chunks)} чанков по <= {CHUNK_REPLICAS} реплик", file=sys.stderr)
    cleaned_replicas: list[Replica] = []
    consumed = 0
    delta_chars = 0
    for idx, chunk in enumerate(chunks, start=1):
        label = f"чанк {idx}/{len(chunks)}"
        ctx = context_lines(named, consumed)
        before = sum(len(r.text) for r in chunk)
        cleaned = clean_chunk(chunk_prompt, chunk, ctx, context_file, label)
        after = sum(len(s) for s in cleaned)
        delta_chars += after - before
        print(
            f"{label}: {len(chunk)} реплик, {before}→{after} симв "
            f"({(after - before) / before * 100:+.1f}%)",
            file=sys.stderr,
        )
        for r, text in zip(chunk, cleaned):
            cleaned_replicas.append(Replica(r.time, r.label, text.strip()))
        consumed += len(chunk)

    # Верификация и запись
    problems = verify_final(src, cleaned_replicas, named)
    if problems:
        for p in problems:
            print(f"ВЕРИФИКАЦИЯ: {p}", file=sys.stderr)
        print("выход не записан: каркас разошёлся с входом", file=sys.stderr)
        return 1

    out_path.write_text("\n\n".join(r.line() for r in cleaned_replicas) + "\n", encoding="utf-8")
    total_before = sum(len(r.text) for r in src)
    total_after = sum(len(r.text) for r in cleaned_replicas)
    print(
        f"\nOK: {len(cleaned_replicas)} реплик, таймкоды и метки сходятся; "
        f"текст {total_before}→{total_after} симв "
        f"({(total_after - total_before) / total_before * 100:+.1f}% — мягкий ориентир)",
        file=sys.stderr,
    )
    print(str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
