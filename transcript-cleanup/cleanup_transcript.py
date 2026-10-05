#!/usr/bin/env python3
"""Чистка диаризованного транскрипта: карта спикеров + глоссарные правки.

Usage: cleanup_transcript.py <source> <context-file> [-o out] [--yes] [--workers N]

Вход — транскрипт вида «[HH:MM:SS] [SPEAKER_XX] текст» (выход cli/diarize.sh)
плюс проектный контекст-файл. Выход — тот же каркас с именами вместо
SPEAKER_XX (где спикер опознан по содержанию) и вычищенным текстом реплик.

Две фазы, обе через headless `claude -p --tools "" --json-schema`:
  1. sonnet строит карту «SPEAKER_XX → имя» по скелету встречи (первые ~30
     слов каждой реплики); карта показывается, подтверждается по stdin
     (или --yes); подстановка имён — детерминированная замена строк здесь.
  2. haiku чистит текст чанками по ~CHUNK_CHARS символов (число реплик в
     чанке плавает, реплика — атом); чанки независимы и идут
     пулом потоков (--workers, по умолчанию 3); каркас (таймкоды, метки)
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
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# Чанк фазы 2 меряется символами текста, а не репликами: время ответа haiku
# растёт с объёмом текста, а длина реплики гуляет на порядки — на встрече
# реплика в среднем короткая, в подкасте монолог гостя доходит до 6 тыс.
# символов. Чанк по 15 реплик на подкасте выходил от 1 до 20 тыс. символов, и
# самый толстый определял время всей фазы. 5000 — примерно 15 реплик подкаста
# или 40+ реплик встречи; такие чанки haiku стабильно отдаёт в пределах таймаута.
CHUNK_CHARS = 5000
CONTEXT_REPLICAS = 3
SKELETON_WORDS = 30
SIGNAL_HEAD_REPLICAS = 2  # первые реплики каждого спикера в сокращённом скелете
MAX_RETRIES = 2

PHASE1_MODEL = "sonnet"
PHASE2_MODEL = "haiku"
PHASE1_TIMEOUT_S = 300
# Фаза 2: 420 с — под слотом haiku может оказаться медленная модель (на
# корпоративном роутере чанк таймаутил на 180 с и уходил в ретрай); нативный
# claude-haiku-4-5 укладывается примерно в 4 минуты на чанк из 40 реплик.
# На подкасте с длинными монологами чанк из 40 реплик (~24 тыс. символов) не
# уложился и в 420 с. Чанки теперь меряются символами (CHUNK_CHARS), но одна
# реплика длиннее CHUNK_CHARS всё равно идёт целиком, поэтому таймаут с запасом.
PHASE2_TIMEOUT_S = 1200
# Пороги --max-budget-usd. На подписке счётчик считает list price (costBasis
# list, кэш общего префикса работает — system prompt и контекст-файл чанки
# читают из кэша), реальных списаний нет: пороги остаются страховкой от
# зацикливания. На корпоративном роутере счётчик был фиктивным и щедрее по
# токенам (costBasis unknown, без кэша) — под него числа и подбирались, запас
# сохраняется в обоих режимах.
PHASE1_BUDGET_USD = "0.50"
PHASE2_BUDGET_USD = "1.50"
# Фаза 2 идёт пулом потоков: чанки независимы (контекст края берётся из
# исходника, а не из выхода предыдущего чанка). 3 — компромисс между временем
# прогона и расходом usage-окна подписки; 1 возвращает последовательный прогон.
PHASE2_WORKERS = 3

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
        "cleaned": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        # Догадки: замены, у которых нет пары в контекст-файле. Модель обязана
        # отчитаться о каждой (пустой массив, если их нет) — иначе они уходят в
        # текст незаметно, а именно на них ловятся её выдумки.
        "guesses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "basis": {"type": "string"},
                },
                "required": ["n", "from", "to", "basis"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["cleaned", "guesses"],
    "additionalProperties": False,
}


# Журнал вызовов: envelope каждого вызова несёт total_cost_usd и usage с
# метриками кэша (на подписке — list price, деньги не списываются). Пишется в
# call_claude, а не в возвращаемое значение, чтобы не ломать сигнатуру для
# вызывающих и тестов; фаза 2 пишет из нескольких потоков, отсюда lock.
#
# Кэш префикса живёт на стороне API (ключ — байты префикса + модель, область —
# workspace), переживает отдельные запуски `claude` и держится час: CLI пишет
# записи под 1h TTL. Общий префикс чанков — обвязка CLI и наш system prompt;
# контекст-файл в него не входит, он лежит в payload после `replicas`, которые
# у каждого чанка свои. Растущий cache_read на чанках 2+ — признак, что прогрев
# первым чанком работает; cache_creation на каждом чанке означает, что префикс
# переписывается и прогрев ничего не даёт.
_cost_lock = threading.Lock()
_cost_usd = 0.0
_calls: list[dict] = []

# Журнал догадок: замены без пары в контекст-файле, о которых модель отчиталась
# в `guesses`. Пишется здесь же, а не в возвращаемое значение clean_chunk, по той
# же причине, что и журнал вызовов. Номер реплики внутри чанка превращается в
# таймкод сразу при записи: дальше по коду чанков уже нет, а таймкод — то, по
# чему догадку ищут глазами в выходе.
_guess_lock = threading.Lock()
_guesses: list[dict] = []


def add_cost(usd: float) -> None:
    global _cost_usd
    with _cost_lock:
        _cost_usd += usd


def record_call(model: str, usd: float, usage: dict) -> None:
    def n(key: str) -> int:
        v = usage.get(key)
        return v if isinstance(v, int) else 0

    with _cost_lock:
        _calls.append({
            "model": model,
            "usd": usd,
            "cache_read": n("cache_read_input_tokens"),
            "cache_write": n("cache_creation_input_tokens"),
            "fresh": n("input_tokens"),
        })


def spent_usd() -> float:
    with _cost_lock:
        return _cost_usd


def call_log() -> list[dict]:
    with _cost_lock:
        return list(_calls)


def print_call_log() -> None:
    """Токены и кэш по вызовам — в порядке завершения (фаза 2 идёт пулом)."""
    calls = call_log()
    if not calls:
        return
    print("\nтокены по вызовам (порядок завершения):", file=sys.stderr)
    for i, c in enumerate(calls, start=1):
        print(
            f"  {i:2d} {c['model']:7} из кэша {c['cache_read']:7} | "
            f"в кэш {c['cache_write']:7} | полный тариф {c['fresh']:6} | ${c['usd']:.2f}",
            file=sys.stderr,
        )
    reported = any(c["cache_write"] or c["fresh"] for c in calls)
    if reported and all(c["cache_read"] == 0 for c in calls[1:]):
        print(
            "  кэш префикса не читался ни разу — прогрев первым чанком бесполезен, "
            "префикс переписывается на каждом вызове",
            file=sys.stderr,
        )


def record_guesses(raw: object, chunk: list["Replica"], label: str) -> None:
    """Догадки чанка → журнал. `n` — номер реплики внутри чанка (1-based), он же
    переводится в таймкод. Запись вне диапазона не валит прогон: это отчётность,
    а не выход, и терять из-за неё уже очищенный чанк нечем оправдать."""
    if not isinstance(raw, list):
        return
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        n = item.get("n")
        if not isinstance(n, int) or not 1 <= n <= len(chunk):
            print(f"  {label}: догадка с номером реплики {n!r} вне чанка, пропущена",
                  file=sys.stderr)
            continue
        rows.append({
            "time": chunk[n - 1].time,
            "from": str(item.get("from", "")),
            "to": str(item.get("to", "")),
            "basis": str(item.get("basis", "")),
        })
    if rows:
        with _guess_lock:
            _guesses.extend(rows)


def guess_log() -> list[dict]:
    with _guess_lock:
        return list(_guesses)


def print_guess_log() -> None:
    """Догадки группами «было → стало», с числом вхождений и таймкодами.

    Это главный ручной контроль выхода: замены по парам контекст-файла видны
    грепом, а догадки — нет. Каждая строка либо превращается в пару глоссария,
    либо означает, что модель выдумала.
    """
    rows = guess_log()
    if not rows:
        return
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["from"], r["to"]), []).append(r)
    print(f"\nдогадки без пары в контекст-файле ({len(rows)}):", file=sys.stderr)
    for (frm, to), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        times = ", ".join(i["time"] for i in items[:6])
        if len(items) > 6:
            times += f", … (+{len(items) - 6})"
        print(f"  {frm!r} → {to!r}  ×{len(items)}  [{times}]", file=sys.stderr)
        print(f"      {items[0]['basis']}", file=sys.stderr)


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


def chunk_replicas(replicas: list[Replica], target: int = CHUNK_CHARS) -> list[list[Replica]]:
    """Чанки примерно поровну по символам текста, реплика — атом.

    Число чанков — ceil(всего / target), затем граница k-го чанка ставится на
    стык реплик, ближайший к k/n от общего объёма. Жадное «набивать до порога»
    оставляло бы в конце огрызок и не выравнивало чанки между собой. Реплика
    длиннее target уходит отдельным чанком целиком; пустых чанков не бывает.
    """
    if not replicas:
        return []
    sizes = [max(len(r.text), 1) for r in replicas]
    total = sum(sizes)
    n = min(len(replicas), -(-total // target))
    cum: list[int] = []  # cum[i] — символов в replicas[:i+1]
    acc = 0
    for size in sizes:
        acc += size
        cum.append(acc)
    cuts = [0]
    for k in range(1, n):
        goal = total * k / n
        lo = cuts[-1] + 1  # хотя бы одна реплика в текущем чанке
        hi = len(replicas) - (n - k)  # хотя бы по одной на оставшиеся
        best = min(range(lo, hi + 1), key=lambda i: abs(cum[i - 1] - goal))
        cuts.append(best)
    cuts.append(len(replicas))
    return [replicas[a:b] for a, b in zip(cuts, cuts[1:])]


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
    cost = envelope.get("total_cost_usd")
    cost = float(cost) if isinstance(cost, (int, float)) else 0.0
    usage = envelope.get("usage")
    # до проверок ниже: неудачный вызов тоже стоил денег и тоже трогал кэш
    add_cost(cost)
    record_call(model, cost, usage if isinstance(usage, dict) else {})
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
            # После проверок: догадки неудачной попытки в журнал не попадают.
            record_guesses(out.get("guesses"), chunk, label)
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


def clean_all_chunks(
    chunk_prompt: str,
    named: list[Replica],
    chunks: list[list[Replica]],
    context_file: str,
    workers: int = PHASE2_WORKERS,
) -> list[list[str]]:
    """Очищенные тексты всех чанков — список по чанку, в исходном порядке.

    Чанки независимы: контекст края берётся из `named` (исходник с
    подставленными метками), а не из выхода предыдущего чанка, поэтому фаза 2
    распараллеливается пулом потоков — каждый вызов блокируется на subprocess,
    GIL не мешает. Первый чанк идёт отдельно: он создаёт кэш общего префикса
    (system prompt + контекст-файл), остальные его читают, а холодный веер
    платил бы за запись кэша столько раз, сколько воркеров. Порядок результатов
    восстанавливается по входу, а не по времени завершения. Чанк, исчерпавший
    ретраи, валит всю фазу (выход всё-или-ничего); ещё не начатые чанки при
    этом отменяются, уже запущенные доживают до своего ответа.
    """
    total = len(chunks)
    if not total:
        return []
    offsets: list[int] = []
    consumed = 0
    for chunk in chunks:
        offsets.append(consumed)
        consumed += len(chunk)

    def run(idx: int) -> list[str]:
        chunk = chunks[idx]
        label = f"чанк {idx + 1}/{total}"
        before = sum(len(r.text) for r in chunk)
        cleaned = clean_chunk(
            chunk_prompt, chunk, context_lines(named, offsets[idx]), context_file, label
        )
        after = sum(len(s) for s in cleaned)
        print(
            f"{label}: {len(chunk)} реплик, {before}→{after} симв "
            f"({(after - before) / before * 100:+.1f}%)",
            file=sys.stderr,
        )
        return cleaned

    results = [run(0)]
    rest = list(range(1, total))
    if not rest:
        return results
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results.extend(pool.map(run, rest))  # map отдаёт в порядке входа
    else:
        results.extend(run(i) for i in rest)
    return results


# ── верификации ──────────────────────────────────────────────────────────────

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё\-.]*")


def _words(text: str) -> Counter:
    """Мультимножество слов реплики. Регистр и «ё» нормализуются: их правка
    разрешена политикой и потерей слова не является."""
    out: Counter = Counter()
    for w in WORD_RE.findall(text):
        w = w.lower().replace("ё", "е").strip(".-")
        if w:
            out[w] += 1
    return out


def word_losses(
    src: list[Replica], out: list[Replica], guesses: list[dict]
) -> list[dict]:
    """Реплики, из которых слова пропали без замены и без записи в `guesses`.

    Зачем отдельно от verify_final: тот сверяет каркас, а выброшенная посреди
    реплики фраза каркас не двигает и в дельте символов тонет (измерено: −20
    слов живой речи при итоговой дельте −2.7%, где остальное — объявленное
    сворачивание залипания). Проверка детерминированная, по мультимножеству
    слов, и сознательно грубая: замена слова на слово даёт баланс и сюда не
    попадает, свёрнутый повтор — тоже, если модель объявила его в `guesses`.

    Предупреждение, а не гейт: выход уже записан, ретрай на этом не нужен —
    список нужен глазам, чтобы решить, где смотреть исходник.
    """
    # Объявленная замена вычитается по количеству, объявленное сворачивание —
    # по словам целиком: модель цитирует залипание обрывком («Dark Haus. Dark…
    # 17+ повторов»), а в реплике их бывает вдвое больше, и счёт по вхождениям
    # оставлял бы весь хвост повтора в отчёте как потерю.
    declared: dict[str, Counter] = {}
    collapsed: dict[str, set[str]] = {}
    for g in guesses:
        words = _words(g["from"])
        if not g["to"].strip() or "залипан" in g["basis"].lower():
            collapsed.setdefault(g["time"], set()).update(words)
        else:
            declared.setdefault(g["time"], Counter()).update(words)

    reports = []
    for s, o in zip(src, out):
        lost = _words(s.text) - _words(o.text)
        added = _words(o.text) - _words(s.text)
        lost -= declared.get(s.time, Counter())
        for w in collapsed.get(s.time, ()):
            lost.pop(w, None)
        net = sum(lost.values()) - sum(added.values())
        if net > 0:
            reports.append({"time": s.time, "net": net, "words": dict(lost)})
    return reports


def print_word_losses(reports: list[dict]) -> None:
    """Нетто-баланс и список исчезнувшего — раздельно: часть исчезнувших слов
    заменена (их компенсируют появившиеся), и «−1» рядом с четырьмя словами
    читается как ошибка счёта, хотя это разные величины."""
    if not reports:
        return
    total = sum(r["net"] for r in reports)
    print(
        f"\nпотеря слов без замены и без записи в догадках "
        f"(реплик: {len(reports)}, нетто ~{total} слов):",
        file=sys.stderr,
    )
    for r in reports:
        items = sorted(r["words"].items(), key=lambda kv: (-kv[1], kv[0]))
        shown = ", ".join(f"{w!r}×{n}" if n > 1 else repr(w) for w, n in items[:8])
        if len(items) > 8:
            shown += f", … (+{len(items) - 8})"
        print(f"  [{r['time']}] нетто −{r['net']}; исчезли: {shown}", file=sys.stderr)
    print(
        "  сверить с исходником: сворачивание залипания объявляется в догадках, "
        "эти реплики — нет",
        file=sys.stderr,
    )


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
    workers = PHASE2_WORKERS
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
        elif a == "--workers":
            i += 1
            if i >= len(args):
                print("--workers требует аргумент", file=sys.stderr)
                return 2
            try:
                workers = int(args[i])
            except ValueError:
                print(f"--workers: ожидалось число, получено {args[i]!r}", file=sys.stderr)
                return 2
            if workers < 1:
                print("--workers: минимум 1", file=sys.stderr)
                return 2
        elif a in ("-h", "--help"):
            print(__doc__.strip())
            return 0
        else:
            positional.append(a)
        i += 1
    if len(positional) != 2:
        print(
            "usage: cleanup_transcript.py <source> <context-file> "
            "[-o out] [--yes] [--workers N]",
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
    phase1_cost = spent_usd()
    if yes:
        print_map(mapping, evidence)
        print("--yes: карта принята без подтверждения", file=sys.stderr)
    else:
        mapping = confirm_map(mapping, evidence)

    named = substitute_names(src, mapping)

    # Фаза 2 — чанковая чистка
    chunks = chunk_replicas(named)
    print(
        f"{len(chunks)} чанков по ~{CHUNK_CHARS} симв, воркеров: {workers}",
        file=sys.stderr,
    )
    cleaned_by_chunk = clean_all_chunks(
        chunk_prompt, named, chunks, context_file, workers
    )
    cleaned_replicas: list[Replica] = []
    for chunk, cleaned in zip(chunks, cleaned_by_chunk):
        for r, text in zip(chunk, cleaned):
            cleaned_replicas.append(Replica(r.time, r.label, text.strip()))
    print_call_log()
    print_guess_log()
    print(
        f"счётчик CLI: фаза 1 ${phase1_cost:.2f}, фаза 2 ${spent_usd() - phase1_cost:.2f}, "
        f"итого ${spent_usd():.2f} (list price; на подписке не списывается)",
        file=sys.stderr,
    )

    # Верификация и запись
    problems = verify_final(src, cleaned_replicas, named)
    if problems:
        for p in problems:
            print(f"ВЕРИФИКАЦИЯ: {p}", file=sys.stderr)
        print("выход не записан: каркас разошёлся с входом", file=sys.stderr)
        return 1

    out_path.write_text("\n\n".join(r.line() for r in cleaned_replicas) + "\n", encoding="utf-8")
    print_word_losses(word_losses(src, cleaned_replicas, guess_log()))
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
