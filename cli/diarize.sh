#!/usr/bin/env bash
# Прогоняет аудиофайл через POST /diarize и сохраняет диаризованный
# транскрипт рядом с исходником (<input>.txt).
#
# Использование:
#   ./diarize.sh call.m4a
#   ./diarize.sh -l ru --min-speakers 2 --max-speakers 4 call.wav
#   ./diarize.sh -u http://localhost:8000 -o out.txt call.mp3

set -euo pipefail

URL="http://localhost:8000"
LANG="ru"
MIN_SPEAKERS=2
MAX_SPEAKERS=10
OUTPUT=""
INPUT=""

usage() {
    sed -n '2,9p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--lang)         LANG="$2"; shift 2 ;;
        -u|--url)          URL="${2%/}"; shift 2 ;;
        --min-speakers)    MIN_SPEAKERS="$2"; shift 2 ;;
        --max-speakers)    MAX_SPEAKERS="$2"; shift 2 ;;
        -o|--output)       OUTPUT="$2"; shift 2 ;;
        -h|--help)         usage 0 ;;
        -*)                echo "unknown option: $1" >&2; usage 1 ;;
        *)
            if [[ -n "$INPUT" ]]; then
                echo "несколько входных файлов не поддерживаются" >&2
                exit 1
            fi
            INPUT="$1"; shift ;;
    esac
done

if [[ -z "$INPUT" ]]; then
    echo "не указан входной файл" >&2
    usage 1
fi
if [[ ! -f "$INPUT" ]]; then
    echo "файл не найден: $INPUT" >&2
    exit 1
fi
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="${INPUT%.*}.txt"
fi

echo "→ POST $URL/diarize  file=$INPUT lang=${LANG:-auto}" >&2

curl --fail-with-body -sS -X POST "$URL/diarize" \
    -F "file=@${INPUT}" \
    -F "language=${LANG}" \
    -F "min_speakers=${MIN_SPEAKERS}" \
    -F "max_speakers=${MAX_SPEAKERS}" \
    -F "format=text" \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["text"])' \
  > "$OUTPUT"

echo "✔ $OUTPUT" >&2
