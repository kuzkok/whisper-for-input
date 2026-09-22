#!/usr/bin/env python3
"""Метрики качества сегментации по дампу /diarize с include_words=true.

Зачем: метка спикера у сегмента — доминирующая по пересечению с диаризацией
(whisperx/diarize.py:229), а у каждого слова своя. Расхождение между ними — речь,
которая приписана не тому говорящему. Считается без ручного эталона: word-level
разметка служит опорой для segment-level.

Что здесь меряется и почему именно так. Первая версия считала «сегменты, внутри
которых больше одной метки слова», и давала 12-15% сегментов и 30-38% реплик.
Проверка глазами показала, что это в основном дребезг: метки слов на стыках
смещены на слово-два («мы просто не | викторизуем»). Настоящие слипания — чужой
ран от ALIEN_RUN_WORDS слов — дают около 5% реплик. Поэтому ключевые числа:

- replicas_with_alien_run: реплики с длинным чужим раном (то, что чиним);
- alien_words: слов не под своей меткой (объём неверно приписанной речи);
- mid_sentence_starts: реплики, начинающиеся со строчной буквы, — разрезы
  посреди фразы, то есть цена, которую платим за разрез;
- stubs: короткие огрызки.

Чего метрика НЕ видит: ошибок самой диаризации. Если pyannote отдал кусок целиком
не тому человеку, рассогласования нет, а ошибка есть.

Использование:
    python3 tools/segmentation_metrics.py dump.words.json [ещё.words.json ...]
    python3 tools/segmentation_metrics.py --json dump.words.json

Колонка «после разреза» применяет к дампу ту же функцию speaker_split.split_segments,
что стоит в /diarize. На дампе, снятом уже после её внедрения, обе колонки совпадут.

Вход — файл, сохранённый `cli/diarize.sh --words`. Скрипт печатает только числа:
дампы приватны (реальные встречи), тексты реплик в вывод не попадают.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speaker_split import split_segments  # noqa: E402

# Чужой ран такой длины — уже не дребезг метки на стыке, а речь другого человека.
ALIEN_RUN_WORDS = 5
# Кусок короче этого считаем огрызком.
STUB_WORDS = 3


def _load(path):
    data = json.loads(Path(path).read_text())
    segments = data["segments"]
    if segments and "words" not in segments[0]:
        sys.exit(f"{path}: в дампе нет word-level данных — нужен include_words=true")
    return segments


def _group_into_replicas(segments):
    """Склейка реплик как в server.py format=text: подряд идущие сегменты
    с одинаковой меткой."""
    replicas = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        if replicas and replicas[-1]["speaker"] == speaker:
            replicas[-1]["words"] += seg.get("words", [])
        else:
            replicas.append({"speaker": speaker, "words": list(seg.get("words", []))})
    return replicas


def _has_alien_run(replica):
    run = 0
    for word in replica["words"]:
        if word["speaker"] is not None and word["speaker"] != replica["speaker"]:
            run += 1
            if run >= ALIEN_RUN_WORDS:
                return True
        else:
            run = 0
    return False


def compute(segments):
    replicas = _group_into_replicas(segments)
    words = [w for r in replicas for w in r["words"]]
    labelled = [w for w in words if w["speaker"] is not None]
    lengths = [len(r["words"]) for r in replicas]

    alien_words = sum(
        1
        for r in replicas
        for w in r["words"]
        if w["speaker"] is not None and w["speaker"] != r["speaker"]
    )
    with_alien_run = sum(1 for r in replicas if _has_alien_run(r))
    mid_sentence = sum(
        1 for r in replicas if r["words"] and r["words"][0]["word"].strip()[:1].islower()
    )
    audio_s = max((seg["end"] for seg in segments), default=0.0)

    def pct(part, whole):
        return round(100.0 * part / whole, 1) if whole else 0.0

    return {
        "audio_minutes": round(audio_s / 60, 1),
        "segments": len(segments),
        "words": len(words),
        "speakers": sorted({w["speaker"] for w in labelled}),
        "words_without_speaker": len(words) - len(labelled),
        "words_without_speaker_pct": pct(len(words) - len(labelled), len(words)),
        "segments_unknown_speaker": sum(1 for s in segments if s.get("speaker") == "UNKNOWN"),
        "replicas": len(replicas),
        "replicas_with_alien_run": with_alien_run,
        "replicas_with_alien_run_pct": pct(with_alien_run, len(replicas)),
        "alien_words": alien_words,
        "alien_words_pct": pct(alien_words, len(labelled)),
        "mid_sentence_starts": mid_sentence,
        "stubs": sum(1 for n in lengths if n <= STUB_WORDS),
        "replica_words_median": statistics.median(lengths) if lengths else 0,
        "replica_words_mean": round(statistics.mean(lengths), 1) if lengths else 0,
        "speaker_changes_per_min": round(len(replicas) / (audio_s / 60), 1) if audio_s else 0,
    }


_ROWS = [
    ("реплик", "replicas"),
    (f"реплик с чужим раном >={ALIEN_RUN_WORDS}", "replicas_with_alien_run"),
    ("слов не под своей меткой", "alien_words"),
    ("начало не с заглавной", "mid_sentence_starts"),
    (f"огрызков (<={STUB_WORDS} слов)", "stubs"),
    ("медиана слов в реплике", "replica_words_median"),
    ("сегментов", "segments"),
]


def _print_report(name, before, after):
    print(f"\n=== {name} ===")
    print(f"аудио {before['audio_minutes']} мин, слов {before['words']}, "
          f"спикеров {len(before['speakers'])}, "
          f"слов без метки {before['words_without_speaker']} ({before['words_without_speaker_pct']}%)")
    print(f"{'':<32}{'как в дампе':>12}{'после разреза':>15}")
    for label, key in _ROWS:
        print(f"{label:<32}{before[key]:>12}{after[key]:>15}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dumps", nargs="+", help="JSON-дампы /diarize с include_words=true")
    parser.add_argument("--json", action="store_true", help="машиночитаемый вывод вместо отчёта")
    args = parser.parse_args()

    results = {}
    for path in args.dumps:
        segments = _load(path)
        # В имени файла может быть название встречи — показываем только basename.
        results[Path(path).name] = {
            "before": compute(segments),
            "after": compute(split_segments(segments)),
        }

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for name, pair in results.items():
            _print_report(name, pair["before"], pair["after"])


if __name__ == "__main__":
    main()
