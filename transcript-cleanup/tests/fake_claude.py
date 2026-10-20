#!/usr/bin/env python3
"""Фейковый `claude -p` для интеграционных тестов: без сети, детерминированный.

Различает фазы по --model: sonnet → карта спикеров (SPEAKER_00 = Анна Соколова,
SPEAKER_01 = Игорь Левин, остальные null — опоры address_reply по фикстуре
meeting.txt), haiku → механические замены глоссария + точка в конец; замены из GUESSES
дополнительно попадают в поле guesses (догадки без пары в контекст-файле);
haiku с предложениями во входе → фаза 3: абзац с каждого третьего предложения
(4, 7, 10, …). Вызовы reader_pdf.py различаются по payload: «chapters» →
привязка глав (из кандидатов выбирается абзац, где больше всего слов из
названия главы; при равенстве — ближайший к оценке), «units» → разметка
разделов (раздел с каждой четвёртой единицы, «Раздел N»).
Формат envelope совпадает с реальным claude -p --output-format json.

Управляется переменными окружения:
  FAKE_CLAUDE_LOG — файл, куда дописывается по строке на вызов: «map»,
                    «clean», «para», «anchor» или «sections»;
  FAKE_CLAUDE_FAIL_MARK — слово: чанк фазы 2, где оно встречается, получает
                    ошибку вызова (rc 1, is_error);
  FAKE_CLAUDE_BAD_SECTIONS=1 — первая разметка разделов (по журналу
                    FAKE_CLAUDE_LOG) возвращает номер вне диапазона.
"""
import json
import os
import re
import sys

REPLACEMENTS = [
    ("кувер", "Kubernetes"),
    ("апишку", "API"),
    ("апишке", "API"),
    ("Геоджасонки", "GeoJSON-пакеты"),
    ("геоджасонки", "GeoJSON-пакеты"),
    ("Выточление", "Уточнение"),
    ("выточление", "уточнение"),
    ("паладин снимок", "Sentinel-2 снимок"),
    ("пайтера", "Python"),
    ("недобильному", "неделимому"),
]


# Замены, которых в фикстуре project-context.md нет: имитируют догадку по
# контексту, о которой модель обязана отчитаться в guesses.
GUESSES = [
    ("сентинел-данные", "Sentinel-2-данные", "название спутника, пары в контекст-файле нет"),
]


def _stems(text: str) -> set[str]:
    return {w[:5].lower() for w in re.findall(r"\w+", text) if len(w) > 3}


def pick(chapter: dict) -> int:
    title = _stems(chapter["title"])
    return max(
        chapter["paragraphs"],
        key=lambda p: (len(title & _stems(p["text"])), -abs(p["i"] - chapter["estimate"])),
    )["i"]


def main() -> int:
    args = sys.argv[1:]
    model = args[args.index("--model") + 1]
    payload = json.loads(args[-1])
    if "chapters" in payload:
        kind = "anchor"
    elif "units" in payload:
        kind = "sections"
    elif model == "sonnet":
        kind = "map"
    elif payload["replicas"] and "sentences" in payload["replicas"][0]:
        kind = "para"
    else:
        kind = "clean"
    log = os.environ.get("FAKE_CLAUDE_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(kind + "\n")
    mark = os.environ.get("FAKE_CLAUDE_FAIL_MARK")
    if kind == "clean" and mark and any(mark in r["text"] for r in payload["replicas"]):
        print(json.dumps({"is_error": True, "result": "simulated failure"}))
        return 1

    if kind == "anchor":
        out = {"anchors": [
            {"n": ch["n"], "paragraph": pick(ch)} for ch in payload["chapters"]
        ]}
    elif kind == "sections":
        n = len(payload["units"])
        out = {"sections": [
            {"start": i, "title": f"Раздел {k}"}
            for k, i in enumerate(range(1, n + 1, 4), start=1)
        ]}
        first = sum(1 for line in open(log, encoding="utf-8") if line.strip() == "sections") == 1 \
            if log else False
        if os.environ.get("FAKE_CLAUDE_BAD_SECTIONS") == "1" and first:
            out["sections"].append({"start": n + 10, "title": "Мимо"})
    elif kind == "map":
        speakers = sorted({row["speaker"] for row in payload["skeleton"]})
        known = {
            "SPEAKER_00": (
                "Анна Соколова",
                [{"time": "00:01:30", "type": "address_reply"}],
            ),
            "SPEAKER_01": (
                "Игорь Левин",
                [{"time": "00:05:18", "type": "address_reply"}],
            ),
        }
        out = {
            "speakers": [
                {"speaker": sp,
                 "name": known[sp][0] if sp in known else None,
                 "evidence": known[sp][1] if sp in known else []}
                for sp in speakers
            ]
        }
    elif kind == "para":
        out = {"paragraphs": [
            {"n": r["n"], "starts": list(range(4, len(r["sentences"]) + 1, 3))}
            for r in payload["replicas"]
        ]}
    else:
        cleaned = []
        guesses = []
        for r in payload["replicas"]:
            t = r["text"]
            for a, b in REPLACEMENTS:
                t = t.replace(a, b)
            for a, b, basis in GUESSES:
                if a in t:
                    t = t.replace(a, b)
                    guesses.append({"n": r["n"], "from": a, "to": b, "basis": basis})
            if t and t[-1] not in ".!?":
                t += "."
            cleaned.append(t)
        out = {"cleaned": cleaned, "guesses": guesses}

    print(json.dumps({"is_error": False, "structured_output": out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
