#!/usr/bin/env python3
"""Headless transcript formatter — Haiku via `claude -p`, deterministic loop in Python.

Usage: format_transcript.py <source-path> [output-path]
           [--model SLOT] [--settings FILE] [--workers N] [--budget USD]
           [--timeout SEC]

Default output: <source-stem>.article.md next to the source.
--model — слот модели для `claude --model` (по умолчанию haiku); --settings —
файл настроек, который `claude` подмешивает поверх основных (например, env с
ANTHROPIC_BASE_URL роутера, где слоты мапятся на другие модели); --workers —
размер пула; --budget — порог `--max-budget-usd` на вызов (через роутер
счётчик CLI считает чужие модели по своему тарифу и без кэша префикса, так что
порог под haiku обрывает вызов); --timeout — таймаут вызова в секундах (модели
за роутером отвечают медленнее haiku).

Политика (полный текст — chunk-prompt.md): читаемые предложения без пересказа
и искажения сказанного. Можно — любая пунктуация, абзацы, удаление заминок,
паразитов без смысла, повторов и фальстартов, очевидные ASR-правки; нельзя —
пересказ, синонимы, перестановка мыслей, перевод, заголовки, добавления.

Модель вызывается headless с --tools "" — чистый inference, JSON на входе,
schema-валидный JSON на выходе. Разрез на чанки, склейка швов, ретраи, пул
и отчёт потерь слов — детерминированный код здесь.

Чанки независимы: скрипт режет исходник сам (граница — ближайший к целевому
размеру конец предложения в исходнике, если он есть рядом), модель форматирует
ровно свои слова, а соседний текст получает только как контекст. Поэтому чанки
идут пулом: первый — отдельно (прогревает кэш префикса), остальные параллельно.

Ответ каждого чанка сразу сохраняется в `<output>.parts/`; перезапуск с тем
же выходом берёт оттуда готовые чанки и зовёт модель только для остальных
(прогон, упавший на лимите подписки, продолжается после сброса). Статья
пишется только целиком, после чего папка удаляется.

Коды возврата: 0 — выход записан; 2 — ошибка аргументов или выход уже
существует; 1 — чанк исчерпал ретраи, выход не записан. Дельта слов и отчёт
потерь — предупреждения в stderr, rc не меняют. stdout — только путь к выходу.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ~300 слов (20 строк по 100 символов) — по замеру на трёх чанках публичной
# лекции со старым последовательным контрактом: при 40 строках haiku
# выбрасывала целые предложения (4 куска от 5 слов, 61 слово; соседний
# прогон — 5 кусков, 48 слов), при 30 — один чанк потерял 316 слов; при 20 —
# 2 куска по 5–7 слов, оба на фальстартах. Время вызова от размера почти не
# зависит (его съедает thinking) — мелкие чанки окупает пул.
CHUNK_WORDS = 300
# Граница чанка сдвигается к концу предложения в исходнике (`.` `?` `!` `…`),
# если такой есть в пределах окна от целевой точки: Whisper обычно ставит
# пунктуацию, и шов тогда не рвёт предложение. Нет — режем по целевой точке,
# шов посреди предложения склеивает модель по контексту.
SNAP_WINDOW_WORDS = 60
CONTEXT_WORDS = 50
# Пул: вызов блокируется на subprocess, GIL не мешает. 3 — как в
# transcript-cleanup: компромисс между временем прогона и расходом
# usage-окна подписки; 1 возвращает последовательный прогон.
WORKERS = 3
MAX_RETRIES = 2
MODEL = "haiku"
# Файл настроек `claude --settings`; None — вызов с настройками по умолчанию.
SETTINGS: str | None = None
# Thinking — основная доля времени и токенов вызова (~15–25k выходных против
# ~1.5k ответа), но без него haiku выбрасывает законченные фразы. Замер на
# шести чанках публичной лекции, по два прогона: с thinking 559–622 с,
# $0.93–1.00 по счётчику CLI, выпавшие куски ≥5 слов — только фальстарты;
# без thinking (MAX_THINKING_TOKENS=0) 56–58 с, $0.11–0.22, но в обоих
# прогонах пропали целые фразы и элементы перечисления, нетто-потеря −76/−96
# слов против −35/−39. Промежуточного режима нет: `--effort low` на haiku
# выход не уменьшил, MAX_THINKING_TOKENS=2000 как потолок не работает.
THINKING = True
# Пороги подняты под нативный claude-haiku-4-5 на подписке: он думает на чанке
# ~10k thinking-токенов, вызов идёт ~2 минуты и стоит ~$0.09 по счётчику CLI —
# старые 180 с и $0.10 обрывали первый же чанк. На подписке счётчик считает
# list price, реальных списаний нет: порог — страховка от зацикливания.
PER_CALL_TIMEOUT_S = 420
PER_CALL_BUDGET_USD = "0.50"

SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "starts_paragraph": {"type": "boolean"},
    },
    "required": ["text", "starts_paragraph"],
    "additionalProperties": False,
}

# Журнал вызовов — по образцу transcript-cleanup: envelope каждого вызова несёт
# total_cost_usd и usage с метриками кэша (на подписке — list price, деньги не
# списываются). Кэш префикса живёт на стороне API и переживает отдельные
# запуски `claude`: растущий cache_read на чанках 2+ — признак, что общий
# префикс (обвязка CLI + наш system prompt) читается из кэша.
_calls: list[dict] = []


def record_call(usd: float, usage: dict) -> None:
    def n(key: str) -> int:
        v = usage.get(key)
        return v if isinstance(v, int) else 0

    _calls.append({
        "usd": usd,
        "cache_read": n("cache_read_input_tokens"),
        "cache_write": n("cache_creation_input_tokens"),
        "fresh": n("input_tokens"),
        "output": n("output_tokens"),
    })


def print_call_log() -> None:
    if not _calls:
        return
    print("\nтокены по вызовам:", file=sys.stderr)
    for i, c in enumerate(_calls, start=1):
        print(
            f"  {i:2d} из кэша {c['cache_read']:7} | в кэш {c['cache_write']:7} | "
            f"полный тариф {c['fresh']:6} | выход {c['output']:6} | ${c['usd']:.3f}",
            file=sys.stderr,
        )
    total = sum(c["usd"] for c in _calls)
    print(
        f"  вызовов {len(_calls)}, счётчик CLI ${total:.3f} "
        "(list price; на подписке не списывается)",
        file=sys.stderr,
    )


class UsageLimitError(RuntimeError):
    """429 от API — лимит подписки. Ретраи его не снимают, только жгут
    попытки остальных чанков: прогон падает сразу."""


def _error_from(proc_err: subprocess.CalledProcessError) -> RuntimeError:
    """Причина падения `claude -p`: при rc != 0 в stdout обычно всё равно
    лежит envelope — из него берутся статус API и текст ошибки, а не сырой
    хвост на сотни символов служебного JSON."""
    try:
        envelope = json.loads(proc_err.stdout or "")
    except json.JSONDecodeError:
        envelope = None
    if isinstance(envelope, dict):
        cost, usage = envelope.get("total_cost_usd"), envelope.get("usage")
        # упавший вызов тоже мог успеть надумать токенов — в журнал его
        record_call(
            float(cost) if isinstance(cost, (int, float)) else 0.0,
            usage if isinstance(usage, dict) else {},
        )
        status = envelope.get("api_error_status")
        msg = (
            f"claude -p rc={proc_err.returncode}, {envelope.get('subtype')}, "
            f"api {status}: {envelope.get('result', '')!r}"
        )
        return UsageLimitError(msg) if status == 429 else RuntimeError(msg)
    tail = ((proc_err.stderr or "") + (proc_err.stdout or ""))[-800:]
    return RuntimeError(f"claude -p rc={proc_err.returncode}; tail: {tail!r}")


def call_model(system_prompt: str, user_msg: str) -> dict:
    settings = ["--settings", SETTINGS] if SETTINGS else []
    try:
        proc = subprocess.run(
            [
                "claude", "-p",
                "--model", MODEL,
                *settings,
                "--tools", "",
                "--no-session-persistence",
                "--max-budget-usd", PER_CALL_BUDGET_USD,
                "--system-prompt", system_prompt,
                "--output-format", "json",
                "--json-schema", json.dumps(SCHEMA),
                user_msg,
            ],
            capture_output=True,
            text=True,
            env=None if THINKING else {**os.environ, "MAX_THINKING_TOKENS": "0"},
            timeout=PER_CALL_TIMEOUT_S,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise _error_from(e) from e
    envelope = json.loads(proc.stdout)
    cost = envelope.get("total_cost_usd")
    usage = envelope.get("usage")
    # до проверок ниже: неудачный вызов тоже стоил денег и тоже трогал кэш
    record_call(
        float(cost) if isinstance(cost, (int, float)) else 0.0,
        usage if isinstance(usage, dict) else {},
    )
    if envelope.get("is_error"):
        raise RuntimeError(
            f"claude -p reported is_error, {envelope.get('subtype')}: {envelope.get('result', '')!r}"
        )
    structured = envelope.get("structured_output")
    if (
        not isinstance(structured, dict)
        or not isinstance(structured.get("text"), str)
        or not isinstance(structured.get("starts_paragraph"), bool)
    ):
        raise RuntimeError(f"missing/invalid structured_output in envelope: {envelope!r}")
    return structured


# Ответ, который модель завернула второй раз — ```json {"text": …}``` внутри
# поля text. Схема его пропускает (это строка), а в статью он попадает как есть.
WRAPPED_RE = re.compile(r'```|\{\s*"text"\s*:')


def format_chunk(system_prompt: str, chunk: dict, label: str) -> dict:
    """Отформатированный текст чанка и флаг нового абзаца. Пустой `text` —
    тихий пропуск чанка, JSON внутри `text` — двойная обёртка ответа; оба
    ретраятся как ошибка вызова."""
    payload = {
        "context_before": chunk["before"],
        "chunk": chunk["text"],
        "context_after": chunk["after"],
    }
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        if last_err is not None:
            payload["retry_notice"] = (
                f"Предыдущая попытка не прошла проверку: {last_err}. В text — только "
                "отформатированный chunk целиком, простым текстом, без JSON и markdown."
            )
        try:
            out = call_model(system_prompt, json.dumps(payload, ensure_ascii=False))
            if not out["text"].strip():
                raise RuntimeError("пустой text")
            if WRAPPED_RE.search(out["text"]):
                raise RuntimeError("в text JSON или markdown-блок")
            return out
        except UsageLimitError:
            raise
        except subprocess.TimeoutExpired as e:
            # str(e) несёт всю командную строку с промптом — килобайты в stderr
            last_err = RuntimeError(f"таймаут {e.timeout:.0f} с")
            print(f"  {label}: попытка {attempt + 1} — ошибка: {last_err}", file=sys.stderr)
        except (json.JSONDecodeError, RuntimeError) as e:
            last_err = e
            print(f"  {label}: попытка {attempt + 1} — ошибка: {e}", file=sys.stderr)
    raise RuntimeError(f"{label}: исчерпаны {MAX_RETRIES + 1} попытки; последняя: {last_err}")


# ── отчёт потерь слов ────────────────────────────────────────────────────────

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё\-.]*")

# Заминки и паразиты, которые политика разрешает удалять. Их исчезновение из
# отчёта вычитается целиком, по числу вхождений во входе: отличить паразита от
# смыслового «вот» скрипт не может, и шум в отчёте здесь дороже редкой
# пропущенной потери. Многословные — по вхождениям фразы подряд, а не по
# отдельным словам: «как» и «бы» поодиночке — не паразиты.
FILLERS = [
    ("э",), ("ээ",), ("эм",), ("а-а",), ("аа",), ("а",), ("м",), ("мм",), ("м-м",),
    ("ну",), ("вот",), ("значит",), ("типа",), ("короче",),
    ("как", "бы"), ("так", "сказать"), ("в", "общем"), ("это", "самое"),
    ("um",), ("uh",), ("er",), ("like",), ("you", "know"), ("i", "mean"),
]


def _tokens(text: str) -> list[str]:
    """Слова текста по порядку. Регистр и «ё» нормализуются: их правка
    разрешена политикой и потерей слова не является."""
    out = []
    for w in WORD_RE.findall(text):
        w = w.lower().replace("ё", "е").strip(".-")
        if w:
            out.append(w)
    return out


def _excused(tokens: list[str]) -> Counter:
    """Слова входа, чьё исчезновение политика разрешает: заминки и паразиты
    из FILLERS, повтор слова подряд («я я»), обрывок, с которого начинается
    следующее слово («оче очень»)."""
    out: Counter = Counter()
    for i, w in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt is not None and (w == nxt or (len(w) < len(nxt) and nxt.startswith(w))):
            out[w] += 1
    for phrase in FILLERS:
        n = len(phrase)
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i : i + n]) == phrase:
                for w in phrase:
                    out[w] += 1
    return out


def word_losses(label: str, src_text: str, out_text: str) -> dict | None:
    """Нетто-потеря слов одного чанка: мультимножество слов исходного чанка
    против его выхода. Замена слова на слово даёт баланс и
    не печатается; разрешённые политикой удаления вычитаются. Предупреждение, а
    не гейт: выбрасывание законченной фразы видно только здесь — дельта слов
    законно растёт от удалённых паразитов и в ней такая потеря тонет."""
    src_tokens = _tokens(src_text)
    src_words, out_words = Counter(src_tokens), Counter(_tokens(out_text))
    lost = src_words - out_words
    added = out_words - src_words
    lost -= _excused(src_tokens)
    net = sum(lost.values()) - sum(added.values())
    if net <= 0:
        return None
    return {"label": label, "net": net, "words": dict(lost)}


def print_word_losses(reports: list[dict]) -> None:
    if not reports:
        return
    total = sum(r["net"] for r in reports)
    print(
        f"\nпотеря слов сверх заминок, паразитов и повторов "
        f"(вызовов: {len(reports)}, нетто ~{total} слов):",
        file=sys.stderr,
    )
    for r in reports:
        items = sorted(r["words"].items(), key=lambda kv: (-kv[1], kv[0]))
        shown = ", ".join(f"{w!r}×{n}" if n > 1 else repr(w) for w, n in items[:12])
        if len(items) > 12:
            shown += f", … (+{len(items) - 12})"
        print(f"  {r['label']}: нетто −{r['net']}; исчезли: {shown}", file=sys.stderr)
    print(
        "  сверить с исходником: пропавшая законченная фраза — пересказ, "
        "одиночные слова чаще паразиты со смыслом, которые скрипт не узнаёт",
        file=sys.stderr,
    )


SENTENCE_END_RE = re.compile(r"[.?!…][»\"')]*$")


def split_chunks(words: list[str], size: int = CHUNK_WORDS) -> list[dict]:
    """Чанки исходника: слова чанка (`text`) и read-only контекст соседей
    (`before`, `after`). Граница — ближайший к целевой точке конец
    предложения в окне SNAP_WINDOW_WORDS; хвост короче половины чанка
    прирастает к последнему, а не идёт отдельным вызовом."""
    cuts = [0]
    while len(words) - cuts[-1] > size + size // 2:
        target = cuts[-1] + size
        best = None
        for d in range(SNAP_WINDOW_WORDS + 1):
            for i in (target + d, target - d):
                if cuts[-1] < i <= len(words) and SENTENCE_END_RE.search(words[i - 1]):
                    best = i
                    break
            if best is not None:
                break
        cuts.append(best if best is not None else target)
    cuts.append(len(words))
    chunks = []
    for a, b in zip(cuts, cuts[1:]):
        chunks.append({
            "text": " ".join(words[a:b]),
            "before": " ".join(words[max(0, a - CONTEXT_WORDS) : a]),
            "after": " ".join(words[b : b + CONTEXT_WORDS]),
        })
    return chunks


def _part_key(system_prompt: str, chunk: dict) -> str:
    """Отпечаток всего, от чего зависит ответ на чанк: сохранённый ответ
    с другим промптом, моделью, настройками или разрезом не переиспользуется."""
    h = hashlib.sha256()
    for s in (MODEL, SETTINGS or "", system_prompt, chunk["before"], chunk["text"], chunk["after"]):
        h.update(s.encode("utf-8") + b"\0")
    return h.hexdigest()


def _load_part(path: Path, key: str) -> dict | None:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(saved, dict) or saved.get("key") != key:
        return None
    return {"text": saved["text"], "starts_paragraph": saved["starts_paragraph"]}


def _save_part(path: Path, key: str, out: dict) -> None:
    # через временный файл: прерванная запись не оставит битый JSON
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": key, **out}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def format_all_chunks(
    system_prompt: str, chunks: list[dict], parts_dir: Path, workers: int = WORKERS,
) -> list[dict]:
    """Выходы всех чанков в исходном порядке. Первый чанк идёт отдельно: он
    создаёт кэш общего префикса (обвязка CLI + system prompt), остальные его
    читают — холодный веер платил бы за запись кэша столько раз, сколько
    воркеров. Чанк, исчерпавший ретраи, валит прогон; ещё не начатые чанки
    при этом отменяются, уже запущенные доживают до своего ответа.

    Готовый ответ чанка сразу ложится в parts_dir, а сохранённый прошлым
    запуском с тем же отпечатком берётся оттуда без вызова модели: полный
    прогон длинного текста может не уместиться в одно окно лимита подписки."""
    total = len(chunks)
    parts_dir.mkdir(parents=True, exist_ok=True)
    reused = 0

    def run(idx: int) -> dict:
        nonlocal reused
        path = parts_dir / f"{idx + 1:04d}.json"
        key = _part_key(system_prompt, chunks[idx])
        saved = _load_part(path, key)
        if saved is not None:
            reused += 1
            return saved
        out = format_chunk(system_prompt, chunks[idx], f"чанк {idx + 1}/{total}")
        _save_part(path, key, out)
        return out

    try:
        results = [run(0)]
        rest = list(range(1, total))
        if workers > 1 and rest:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results.extend(pool.map(run, rest))  # map отдаёт в порядке входа
        else:
            results.extend(run(i) for i in rest)
    finally:
        if reused:
            print(f"из прошлых запусков взято чанков: {reused} из {total}", file=sys.stderr)
    return results


def _continued(part: str, src_text: str) -> str:
    """Начало чанка, продолжающего предложение соседа: модель часто ставит
    заглавную вопреки промпту. Регистр опускается, только если в исходнике
    первое слово было со строчной и модель его не меняла — имя собственное,
    которое модель исправила или которое уже было с заглавной, не трогается.
    Цена: во входе целиком строчными имя в начале такого шва тоже опустится —
    редкое совпадение, дешевле, чем заглавная посреди каждого разорванного
    предложения."""
    src_words, out_words = _tokens(src_text), _tokens(part)
    src_first = src_text.split()[0] if src_text.split() else ""
    if (
        src_first[:1].islower()
        and src_words[:1] == out_words[:1]
        and part[:1].isupper()
    ):
        return part[0].lower() + part[1:]
    return part


def join_chunks(chunks: list[dict], outs: list[dict]) -> str:
    """Склейка швов. Новый абзац — только если предыдущий чанк кончился
    законченным предложением и модель сказала, что её чанк начинает абзац.
    Иначе через пробел в тот же абзац; шов посреди предложения (предыдущий
    кусок без знака конца) ещё и теряет заглавную в начале следующего."""
    text = outs[0]["text"].strip()
    for chunk, o in zip(chunks[1:], outs[1:]):
        part = o["text"].strip()
        if not SENTENCE_END_RE.search(text):
            text += " " + _continued(part, chunk["text"])
        elif o["starts_paragraph"]:
            text += "\n" + part
        else:
            text += " " + part
    return text


def main() -> int:
    global MODEL, SETTINGS, PER_CALL_BUDGET_USD, PER_CALL_TIMEOUT_S
    parser = argparse.ArgumentParser(description="Сырой транскрипт монолога → Markdown-статья.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--model", default=MODEL, help=f"слот claude --model (по умолчанию {MODEL})")
    parser.add_argument("--settings", default=SETTINGS, help="файл настроек для claude --settings")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"размер пула (по умолчанию {WORKERS})")
    parser.add_argument(
        "--budget", type=float, default=float(PER_CALL_BUDGET_USD),
        help=f"порог claude --max-budget-usd на вызов (по умолчанию {PER_CALL_BUDGET_USD})",
    )
    parser.add_argument(
        "--timeout", type=int, default=PER_CALL_TIMEOUT_S,
        help=f"таймаут вызова, с (по умолчанию {PER_CALL_TIMEOUT_S})",
    )
    args = parser.parse_args()  # ошибка аргументов — exit 2, как и остальные
    if args.workers < 1:
        parser.error("--workers должен быть >= 1")
    if args.budget <= 0:
        parser.error("--budget должен быть > 0")
    if args.timeout <= 0:
        parser.error("--timeout должен быть > 0")
    if args.settings and not Path(args.settings).is_file():
        print(f"settings not found: {args.settings}", file=sys.stderr)
        return 2
    MODEL, SETTINGS, PER_CALL_BUDGET_USD = args.model, args.settings, str(args.budget)
    PER_CALL_TIMEOUT_S = args.timeout

    src = args.source
    if not src.is_file():
        print(f"source not found: {src}", file=sys.stderr)
        return 2
    out = args.output or src.with_suffix(".article.md")
    if out.exists():
        print(f"output already exists, refusing to overwrite: {out}", file=sys.stderr)
        return 2

    words = src.read_text(encoding="utf-8").split()
    if not words:
        print(f"source is empty: {src}", file=sys.stderr)
        return 2
    chunks = split_chunks(words)
    parts_dir = out.with_name(out.name + ".parts")
    system_prompt = (Path(__file__).resolve().parent / "chunk-prompt.md").read_text()

    started = time.monotonic()
    try:
        outs = format_all_chunks(system_prompt, chunks, parts_dir, args.workers)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        print_call_log()
        print(
            f"выход не записан; готовые чанки сохранены в {parts_dir} — "
            "перезапуск той же командой продолжит с места остановки",
            file=sys.stderr,
        )
        return 1

    losses = []
    for i, (chunk, o) in enumerate(zip(chunks, outs), start=1):
        report = word_losses(f"чанк {i}/{len(chunks)}", chunk["text"], o["text"])
        if report:
            losses.append(report)

    # Всё-или-ничего: файл появляется только после последнего чанка — упавший
    # прогон не оставляет полустатью, которую примут за готовую.
    text = join_chunks(chunks, outs) + "\n"
    out.write_text(text, encoding="utf-8")
    shutil.rmtree(parts_dir, ignore_errors=True)

    print_call_log()
    print_word_losses(losses)
    out_words = len(text.split())
    print(
        f"\nOK: чанков {len(chunks)}, время {time.monotonic() - started:.0f} с, "
        f"слов {len(words)}→{out_words} "
        f"({(out_words - len(words)) / len(words) * 100:+.1f}% — мягкий ориентир: "
        "удалённые паразиты и повторы законно дают минус)",
        file=sys.stderr,
    )
    print(str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
