#!/usr/bin/env python3
"""PDF для вычитки: вычищенный транскрипт → документ с оглавлением и разделами.

Usage: reader_pdf.py <transcript.clean.txt> [--chapters <file>] [-o out.pdf] [--title T]

Вход — выход cleanup_transcript.py: реплики «[HH:MM:SS] [Спикер] текст» через
пустую строку, абзацы внутри реплики через одиночный перенос. Выход — PDF
(по умолчанию `<source-stem>.pdf` рядом с исходником) и HTML, из которого он
напечатан (`<out>.html`). Текст реплик шаг не меняет ни на слово.

Разделы берутся из файла глав, если он передан: текст со страницы выпуска как
есть, главой считается строка с ведущим таймкодом и названием. Глава
привязывается к реплике по таймкоду, а если её таймкод лежит внутри длинной
реплики, — к абзацу: скрипт оценивает его интерполяцией по времени, а
окончательный абзац выбирает haiku по названию главы (один вызов на все такие
главы). Без файла глав разделы и их названия размечает sonnet по всему
транскрипту одним вызовом. В обоих случаях модель возвращает только номера (и
названия разделов), текст реплик через её ответ не проходит.

Печать — `google-chrome --headless --print-to-pdf`. Без Chrome в PATH скрипт
пишет только HTML и сообщает в stderr, что PDF не собран (rc 0, в stdout путь к
HTML). Ответы модели сохраняются в `<out>.parts/`: если упала печать,
перезапуск той же командой модель не зовёт. Папка удаляется после успеха.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cleanup_transcript as ct  # noqa: E402

# Глава ближе к началу следующей реплики, чем на SNAP_S секунд, ставится перед
# ней: главы часто размечают по вопросу ведущего, а диаризация начинает его
# реплику на пару секунд позже. Глава не дальше SNAP_S от начала своей реплики
# тоже ставится перед репликой — уточнять там нечего.
SNAP_S = 5
# Сколько абзацев по обе стороны от оценки видит модель. Интерполяция по
# времени грубая (темп речи неровный, паузы), но промах больше чем на 3 абзаца
# на измеренном выпуске не встречался.
ANCHOR_WINDOW = 3
# Средний темп, если его не из чего посчитать (одна реплика): ~15 символов
# в секунду — обычная русская речь.
FALLBACK_CHARS_PER_S = 15.0

ANCHOR_MODEL = "haiku"
ANCHOR_TIMEOUT_S = 420
ANCHOR_BUDGET_USD = "0.50"
# Разметка без глав — один вызов на весь транскрипт, ему нужна структура всего
# разговора, поэтому sonnet: вызов один, разница в цене не важна.
SECTIONS_MODEL = "sonnet"
SECTIONS_TIMEOUT_S = 900
SECTIONS_BUDGET_USD = "2.00"

BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
PRINT_TIMEOUT_S = 180

# Ведущий таймкод H:MM:SS, HH:MM:SS или MM:SS, дальше название.
CHAPTER_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?:(\d{2})(?!\d)(.*)$")
# Оформление перед названием: эмодзи, тире, двоеточия, маркеры списков.
TITLE_LEAD_RE = re.compile(r"^[^\w«\"'(\[]+")

SCHEMA_ANCHORS = {
    "type": "object",
    "properties": {
        "anchors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "paragraph": {"type": "integer"},
                },
                "required": ["n", "paragraph"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["anchors"],
    "additionalProperties": False,
}

SCHEMA_SECTIONS = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "title": {"type": "string"},
                },
                "required": ["start", "title"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sections"],
    "additionalProperties": False,
}


@dataclass
class Chapter:
    seconds: int
    title: str


@dataclass
class Spot:
    """Место главы: реплика и абзац (0-based). candidates — абзацы, из которых
    выбирает модель, если место внутри реплики оценено интерполяцией."""
    replica: int
    paragraph: int
    candidates: list[int] | None = None


@dataclass
class Section:
    replica: int
    paragraph: int
    title: str
    time: str | None  # таймкод в заголовке; None — внутри реплики, времени нет


def hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def seconds_of(time: str) -> int:
    h, m, s = (int(x) for x in time.split(":"))
    return h * 3600 + m * 60 + s


# ── разбор входа ─────────────────────────────────────────────────────────────


def parse_chapters(text: str) -> list[Chapter]:
    """Главы из текста, скопированного со страницы выпуска. Строка без ведущего
    таймкода или без названия после него (длительности соседних выпусков,
    которые копируются заодно) главой не считается. ValueError — глав нет или
    таймкоды не возрастают."""
    chapters: list[Chapter] = []
    for line in text.splitlines():
        m = CHAPTER_RE.match(line)
        if not m:
            continue
        a, b, c, rest = m.groups()
        seconds = int(a) * 3600 + int(b) * 60 + int(c) if b is not None else int(a) * 60 + int(c)
        title = TITLE_LEAD_RE.sub("", rest).strip()
        if not title:
            continue
        if chapters and seconds <= chapters[-1].seconds:
            raise ValueError(
                f"таймкоды глав не возрастают: {hms(seconds)} «{title}» после "
                f"{hms(chapters[-1].seconds)} «{chapters[-1].title}»"
            )
        chapters.append(Chapter(seconds, title))
    if not chapters:
        raise ValueError("в файле глав нет ни одной строки «таймкод название»")
    return chapters


def parse_transcript(src_text: str) -> list[ct.Replica]:
    """Реплики с абзацами: строка продолжения — новый абзац той же реплики
    (через \\n в Replica.text). cleanup_transcript.parse_transcript склеивает
    их пробелом, и фазам 1–2 так и нужно; здесь абзацы — структура документа."""
    replicas: list[ct.Replica] = []
    for raw in src_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = ct.LINE_RE.match(line)
        if m:
            replicas.append(ct.Replica(m.group(1), m.group(2), m.group(3)))
        elif replicas:
            replicas[-1].text += "\n" + line
        else:
            raise ValueError(f"строка вне реплики: {raw[:60]!r}")
    if not replicas:
        raise ValueError("в файле не найдено ни одной реплики вида [HH:MM:SS] [метка] текст")
    return replicas


def paragraphs(r: ct.Replica) -> list[str]:
    return r.text.split("\n")


# ── привязка глав ────────────────────────────────────────────────────────────


def _chars_per_s(replicas: list[ct.Replica]) -> float:
    """Средний темп речи по транскрипту: символы всех реплик, кроме последней,
    на время от первой до последней реплики."""
    span = seconds_of(replicas[-1].time) - seconds_of(replicas[0].time)
    chars = sum(len(r.text) for r in replicas[:-1])
    return chars / span if span > 0 and chars else FALLBACK_CHARS_PER_S


def locate(replicas: list[ct.Replica], t: int) -> Spot:
    """Место главы с таймкодом t (секунды).

    Реплика — последняя, начавшаяся не позже t; если до начала следующей меньше
    SNAP_S, то следующая. Внутри реплики абзац оценивается долей времени между
    её началом и началом следующей, переведённой в долю символов; у последней
    реплики конец оценивается по среднему темпу. Оценка грубая, поэтому вместе
    с ней отдаются абзацы-кандидаты для модели."""
    starts = [seconds_of(r.time) for r in replicas]
    i = max((k for k, s in enumerate(starts) if s <= t), default=0)
    if i + 1 < len(starts) and starts[i + 1] - t <= SNAP_S:
        return Spot(i + 1, 0)
    paras = paragraphs(replicas[i])
    if len(paras) == 1 or t - starts[i] <= SNAP_S:
        return Spot(i, 0)
    if i + 1 < len(starts):
        end = starts[i + 1]
    else:
        end = starts[i] + len(replicas[i].text) / _chars_per_s(replicas)
    frac = min(max((t - starts[i]) / max(end - starts[i], 1), 0.0), 1.0)
    offset = frac * sum(len(p) for p in paras)
    est, acc = len(paras) - 1, 0
    for k, p in enumerate(paras):
        acc += len(p)
        if offset < acc:
            est = k
            break
    lo = max(0, est - ANCHOR_WINDOW)
    hi = min(len(paras), est + ANCHOR_WINDOW + 1)
    return Spot(i, est, list(range(lo, hi)))


def pick_paragraphs(
    prompt: str,
    replicas: list[ct.Replica],
    items: list[tuple[int, str, Spot]],
    titles: list[str],
    parts_dir: Path | None = None,
) -> dict[int, int]:
    """Абзац (0-based) для каждой главы из items — одним вызовом модели.

    items — (номер главы, название, место с кандидатами); titles — названия всех
    глав по порядку, соседние идут в payload как ориентир, где тема кончается.
    Модель видит абзацы-кандидаты только для чтения и возвращает номер абзаца
    (1-based, как в payload). Номер не из кандидатов или пропущенная глава —
    ретрай."""
    chapters = []
    for n, (idx, title, spot) in enumerate(items, start=1):
        paras = paragraphs(replicas[spot.replica])
        chapters.append({
            "n": n,
            "title": title,
            "prev_title": titles[idx - 1] if 0 < idx <= len(titles) else None,
            "next_title": titles[idx + 1] if idx + 1 < len(titles) else None,
            "speaker": replicas[spot.replica].label,
            "estimate": spot.paragraph + 1,
            "paragraphs": [{"i": k + 1, "text": paras[k]} for k in spot.candidates or []],
        })
    payload: dict = {"chapters": chapters}
    key = ct.part_key(ANCHOR_MODEL, prompt, payload)
    saved = ct.load_part(parts_dir, "anchors.json", key)
    if saved is not None:
        print("привязка глав взята из прошлого запуска", file=sys.stderr)
        return {int(k): v for k, v in saved.items()}

    last_err: Exception | None = None
    for attempt in range(ct.MAX_RETRIES + 1):
        if attempt:
            payload["retry_notice"] = (
                f"Предыдущая попытка не прошла проверку: {last_err}. Верни по записи "
                "на каждую главу, номер абзаца — только из её paragraphs."
            )
        try:
            out = ct.call_claude(
                prompt,
                json.dumps(payload, ensure_ascii=False),
                ANCHOR_MODEL,
                SCHEMA_ANCHORS,
                ANCHOR_TIMEOUT_S,
                ANCHOR_BUDGET_USD,
            )
            by_n = {row["n"]: row["paragraph"] for row in out["anchors"]}
            if sorted(by_n) != list(range(1, len(items) + 1)):
                raise RuntimeError(f"номера глав {sorted(by_n)} != 1..{len(items)}")
            result = {}
            for n, (idx, title, spot) in enumerate(items, start=1):
                p = by_n[n] - 1
                if p not in (spot.candidates or []):
                    raise RuntimeError(f"глава {n}: абзац {by_n[n]} не из предложенных")
                result[idx] = p
            ct.save_part(parts_dir, "anchors.json", key, result)
            return result
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            RuntimeError,
            KeyError,
            TypeError,
        ) as e:
            last_err = e
            print(f"  привязка глав: попытка {attempt + 1} — ошибка: {e}", file=sys.stderr)
    raise RuntimeError(f"привязка глав: исчерпаны {ct.MAX_RETRIES + 1} попыток; последняя: {last_err}")


def anchor_chapters(
    prompt: str,
    replicas: list[ct.Replica],
    chapters: list[Chapter],
    parts_dir: Path | None = None,
) -> list[Section]:
    """Разделы по главам. Главы внутри длинных реплик уточняются моделью; если
    она не справилась, остаётся оценка интерполяцией — заголовок стоит рядом,
    а не теряется. Место главы не может оказаться раньше предыдущей: такое
    подтягивается к предыдущей с предупреждением."""
    spots = [locate(replicas, c.seconds) for c in chapters]
    items = [(k, c.title, s) for k, (c, s) in enumerate(zip(chapters, spots)) if s.candidates]
    picked: dict[int, int] = {}
    if items:
        print(f"глав внутри длинных реплик: {len(items)}, абзац выбирает {ANCHOR_MODEL}",
              file=sys.stderr)
        try:
            picked = pick_paragraphs(prompt, replicas, items, [c.title for c in chapters],
                                     parts_dir)
        except RuntimeError as e:
            print(f"  {e}; заголовки этих глав стоят по оценке времени", file=sys.stderr)

    sections: list[Section] = []
    for k, (c, s) in enumerate(zip(chapters, spots)):
        pos = (s.replica, picked.get(k, s.paragraph))
        if sections and pos < (sections[-1].replica, sections[-1].paragraph):
            print(f"  глава «{c.title}» встала раньше предыдущей, ставлю рядом с ней",
                  file=sys.stderr)
            pos = (sections[-1].replica, sections[-1].paragraph)
        sections.append(Section(pos[0], pos[1], c.title, hms(c.seconds)))
    return sections


# ── разметка разделов без глав ───────────────────────────────────────────────


def units(replicas: list[ct.Replica]) -> list[tuple[int, int]]:
    """Единицы разметки: каждый абзац каждой реплики, (реплика, абзац)."""
    return [(i, k) for i, r in enumerate(replicas) for k in range(len(paragraphs(r)))]


def model_sections(
    prompt: str, replicas: list[ct.Replica], parts_dir: Path | None = None
) -> list[Section]:
    """Разделы от модели: она получает пронумерованные единицы с таймкодом и
    спикером и возвращает номера начала разделов и их названия. Первый раздел
    обязан начинаться с единицы 1, номера — строго по возрастанию и в
    диапазоне; иначе ретрай."""
    us = units(replicas)
    payload: dict = {
        "duration_minutes": round(
            (seconds_of(replicas[-1].time) - seconds_of(replicas[0].time)) / 60),
        "units": [
            {"i": n, "time": replicas[i].time, "speaker": replicas[i].label,
             "text": paragraphs(replicas[i])[k]}
            for n, (i, k) in enumerate(us, start=1)
        ],
    }
    key = ct.part_key(SECTIONS_MODEL, prompt, payload)
    saved = ct.load_part(parts_dir, "sections.json", key)
    if saved is None:
        saved = _call_sections(prompt, payload, len(us))
        ct.save_part(parts_dir, "sections.json", key, saved)
    else:
        print("разделы взяты из прошлого запуска", file=sys.stderr)
    out = []
    for row in saved:
        i, k = us[row["start"] - 1]
        out.append(Section(i, k, row["title"], replicas[i].time if k == 0 else None))
    return out


def _call_sections(prompt: str, payload: dict, total: int) -> list[dict]:
    last_err: Exception | None = None
    for attempt in range(ct.MAX_RETRIES + 1):
        if attempt:
            payload["retry_notice"] = (
                f"Предыдущая попытка не прошла проверку: {last_err}. Первый раздел — "
                f"с единицы 1, номера строго по возрастанию, не больше {total}, "
                "у каждого раздела непустое название."
            )
        try:
            out = ct.call_claude(
                prompt,
                json.dumps(payload, ensure_ascii=False),
                SECTIONS_MODEL,
                SCHEMA_SECTIONS,
                SECTIONS_TIMEOUT_S,
                SECTIONS_BUDGET_USD,
            )
            rows = [{"start": r["start"], "title": r["title"].strip()} for r in out["sections"]]
            if not rows:
                raise RuntimeError("ни одного раздела")
            starts = [r["start"] for r in rows]
            if starts[0] != 1:
                raise RuntimeError(f"первый раздел с единицы {starts[0]}, а не с 1")
            if any(b <= a for a, b in zip(starts, starts[1:])):
                raise RuntimeError(f"номера не по возрастанию: {starts}")
            if starts[-1] > total:
                raise RuntimeError(f"номер {starts[-1]} вне 1..{total}")
            if any(not r["title"] for r in rows):
                raise RuntimeError("пустое название раздела")
            return rows
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            RuntimeError,
            KeyError,
            TypeError,
            AttributeError,
        ) as e:
            last_err = e
            print(f"  разделы: попытка {attempt + 1} — ошибка: {e}", file=sys.stderr)
    raise RuntimeError(f"разделы: исчерпаны {ct.MAX_RETRIES + 1} попыток; последняя: {last_err}")


# ── HTML и PDF ───────────────────────────────────────────────────────────────

# Типографика под вычитку с экрана и аннотации: строка ~75 знаков, широкое
# правое поле под заметки, номер страницы внизу. Шрифты — системные, кириллица
# есть во всех из списка.
CSS = """
@page {
  size: A4;
  margin: 20mm 45mm 22mm 22mm;
  @bottom-center { content: counter(page); font: 9pt "Noto Sans", sans-serif; color: #777; }
}
body { font-family: "PT Serif", "Noto Serif", "Liberation Serif", serif;
       font-size: 11pt; line-height: 1.5; color: #111; margin: 0; }
h1, h2, nav, .meta, .note { font-family: "PT Sans", "Noto Sans", "Liberation Sans", sans-serif; }
h1 { font-size: 20pt; line-height: 1.25; margin: 0 0 4pt; }
.note { font-size: 9pt; color: #666; margin: 0 0 18pt; }
nav { break-after: page; }
nav h2 { font-size: 13pt; margin: 0 0 8pt; }
nav ol { list-style: none; padding: 0; margin: 0; font-size: 10pt; }
nav li { display: flex; gap: 8pt; margin: 0 0 4pt; }
nav a { color: inherit; text-decoration: none; flex: 1; }
h2 { font-size: 14pt; line-height: 1.3; margin: 22pt 0 8pt; break-after: avoid; }
.time { color: #888; font-variant-numeric: tabular-nums; font-weight: normal; }
h2 .time { font-size: 10pt; margin-right: 6pt; }
.meta { font-size: 9pt; color: #555; margin: 12pt 0 3pt; break-after: avoid; }
.speaker { font-weight: bold; color: #333; }
p { margin: 0 0 6pt; orphans: 2; widows: 2; }
"""


def render_html(title: str, replicas: list[ct.Replica], sections: list[Section],
                note: str) -> str:
    """HTML документа. Абзацы реплик — <p> без класса и только они: по этому
    признаку тест сверяет текст с входом. Если заголовок раздела встал внутри
    реплики, после него повторяется строка «таймкод, спикер, продолжение»,
    чтобы было видно, кто говорит."""
    e = html.escape
    at: dict[tuple[int, int], list[int]] = {}
    for n, s in enumerate(sections, start=1):
        at.setdefault((s.replica, s.paragraph), []).append(n)

    def heading(n: int) -> str:
        s = sections[n - 1]
        time = f'<span class="time">{e(s.time)}</span>' if s.time else ""
        return f'<h2 id="s{n}">{time}{e(s.title)}</h2>'

    toc = "\n".join(
        f'<li><a href="#s{n}">{e(s.title)}</a><span class="time">{e(s.time or "")}</span></li>'
        for n, s in enumerate(sections, start=1)
    )
    body: list[str] = []
    for i, r in enumerate(replicas):
        for k, p in enumerate(paragraphs(r)):
            for n in at.get((i, k), []):
                body.append(heading(n))
            if k == 0:
                body.append(f'<p class="meta"><span class="time">{e(r.time)}</span> '
                            f'<span class="speaker">{e(r.label)}</span></p>')
            elif (i, k) in at:
                body.append(f'<p class="meta"><span class="time">{e(r.time)}</span> '
                            f'<span class="speaker">{e(r.label)}</span>, продолжение</p>')
            body.append(f"<p>{e(p)}</p>")
    nav = f'<nav><h2>Содержание</h2>\n<ol>\n{toc}\n</ol></nav>' if sections else ""
    return (
        '<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{e(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
        f"<h1>{e(title)}</h1>\n<p class=\"note\">{e(note)}</p>\n{nav}\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )


def find_browser() -> str | None:
    for name in BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    return None


def print_pdf(browser: str, html_path: Path, pdf_path: Path) -> None:
    """Печать HTML в PDF headless-браузером. Профиль — временный: с профилем
    по умолчанию запуск передаётся уже открытому окну Chrome и ничего не
    печатает. RuntimeError — PDF не появился."""
    with tempfile.TemporaryDirectory(prefix="reader-pdf-chrome-") as profile:
        try:
            proc = subprocess.run(
                [
                    browser,
                    "--headless",
                    "--disable-gpu",
                    f"--user-data-dir={profile}",
                    "--no-first-run",
                    "--no-pdf-header-footer",
                    "--generate-pdf-document-outline",
                    f"--print-to-pdf={pdf_path}",
                    html_path.resolve().as_uri(),
                ],
                capture_output=True,
                text=True,
                timeout=PRINT_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"{browser}: печать не уложилась в {PRINT_TIMEOUT_S} с") from e
    if proc.returncode != 0 or not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        pdf_path.unlink(missing_ok=True)
        raise RuntimeError(f"{browser}: rc={proc.returncode}; tail: {tail!r}")


def default_title(src_path: Path) -> str:
    """Имя файла без суффиксов шага: «Как-учатся_-взрослые.clean.txt» →
    «Как учатся взрослые»."""
    stem = src_path.name
    for suffix in (".txt", ".para", ".clean"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return re.sub(r"\s+", " ", re.sub(r"[_]+", " ", stem.replace("-", " "))).strip()


# ── main ─────────────────────────────────────────────────────────────────────

USAGE = "usage: reader_pdf.py <transcript.clean.txt> [--chapters <file>] [-o out.pdf] [--title T]"


def main() -> int:
    args = sys.argv[1:]
    opts: dict[str, str] = {}
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-h", "--help"):
            print(__doc__.strip())
            return 0
        if a in ("--chapters", "-o", "--output", "--title"):
            i += 1
            if i >= len(args):
                print(f"{a} требует аргумент", file=sys.stderr)
                return 2
            opts["-o" if a == "--output" else a] = args[i]
        else:
            positional.append(a)
        i += 1
    if len(positional) != 1:
        print(USAGE, file=sys.stderr)
        return 2

    src_path = Path(positional[0])
    if not src_path.is_file():
        print(f"файл не найден: {src_path}", file=sys.stderr)
        return 2
    out_path = Path(opts["-o"]) if "-o" in opts else src_path.with_suffix(".pdf")
    if out_path.exists():
        print(f"выход уже существует, не перезаписываю: {out_path}", file=sys.stderr)
        return 2
    html_path = out_path.with_suffix(".html")
    parts_dir = out_path.with_name(out_path.name + ".parts")

    try:
        replicas = parse_transcript(src_path.read_text(encoding="utf-8"))
        chapters = None
        if "--chapters" in opts:
            ch_path = Path(opts["--chapters"])
            if not ch_path.is_file():
                print(f"файл глав не найден: {ch_path}", file=sys.stderr)
                return 2
            chapters = parse_chapters(ch_path.read_text(encoding="utf-8"))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    paras = sum(len(paragraphs(r)) for r in replicas)
    print(f"{len(replicas)} реплик, {paras} абзацев", file=sys.stderr)

    here = Path(__file__).resolve().parent
    if chapters is not None:
        print(f"глав: {len(chapters)}", file=sys.stderr)
        prompt = (here / "anchor-prompt.md").read_text(encoding="utf-8")
        sections = anchor_chapters(prompt, replicas, chapters, parts_dir)
        note = "Разделы — главы от авторов выпуска."
    else:
        print(f"файла глав нет, разделы размечает {SECTIONS_MODEL}", file=sys.stderr)
        prompt = (here / "sections-prompt.md").read_text(encoding="utf-8")
        try:
            sections = model_sections(prompt, replicas, parts_dir)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            ct.print_call_log()
            return 1
        note = "Разделы и их названия предложены моделью; текст реплик не менялся."
    ct.print_call_log()
    if ct.call_log():
        print(f"счётчик CLI: ${ct.spent_usd():.2f} (list price; на подписке не списывается)",
              file=sys.stderr)

    title = opts.get("--title") or default_title(src_path)
    html_path.write_text(render_html(title, replicas, sections, note), encoding="utf-8")
    print(f"разделов: {len(sections)}; HTML: {html_path}", file=sys.stderr)

    browser = find_browser()
    if browser is None:
        print(f"PDF не собран: в PATH нет ни одного из {', '.join(BROWSERS)}; "
              f"готов только HTML", file=sys.stderr)
        shutil.rmtree(parts_dir, ignore_errors=True)
        print(str(html_path))
        return 0
    try:
        print_pdf(browser, html_path, out_path)
    except RuntimeError as e:
        print(f"PDF не собран: {e}", file=sys.stderr)
        print(f"HTML и ответы модели сохранены ({parts_dir}) — перезапуск той же "
              "командой модель не зовёт", file=sys.stderr)
        return 1
    shutil.rmtree(parts_dir, ignore_errors=True)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
