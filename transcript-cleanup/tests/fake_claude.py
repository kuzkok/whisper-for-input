#!/usr/bin/env python3
"""Фейковый `claude -p` для интеграционных тестов: без сети, детерминированный.

Различает фазы по --model: sonnet → карта спикеров (SPEAKER_00 = Анна Соколова,
SPEAKER_01 = Игорь Левин, остальные null — опоры address_reply по фикстуре
meeting.txt), haiku → механические замены глоссария + точка в конец.
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
        for r in payload["replicas"]:
            t = r["text"]
            for a, b in REPLACEMENTS:
                t = t.replace(a, b)
            if t and t[-1] not in ".!?":
                t += "."
            cleaned.append(t)
        out = {"cleaned": cleaned}

    print(json.dumps({"is_error": False, "structured_output": out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
