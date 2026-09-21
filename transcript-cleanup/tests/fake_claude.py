#!/usr/bin/env python3
"""Фейковый `claude -p` для интеграционных тестов: без сети, детерминированный.

Различает фазы по --model: sonnet → карта спикеров (SPEAKER_00 = Анна Соколова,
SPEAKER_01 = Игорь Левин, остальные null — опоры address_reply по фикстуре
meeting.txt), haiku → механические замены глоссария + точка в конец; замены из GUESSES
дополнительно попадают в поле guesses (догадки без пары в контекст-файле).
Формат envelope совпадает с реальным claude -p --output-format json.
"""
import json
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


def main() -> int:
    args = sys.argv[1:]
    model = args[args.index("--model") + 1]
    payload = json.loads(args[-1])

    if model == "sonnet":
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
