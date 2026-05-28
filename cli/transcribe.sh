#!/usr/bin/env bash
# Прогоняет аудио/видео файл через POST /transcribe и сохраняет
# транскрипт рядом с исходником (<input>.txt).
#
# Использование:
#   ./transcribe.sh lecture.mp4
#   ./transcribe.sh -l en talk.m4a
#   ./transcribe.sh -u http://localhost:8000 -o out.txt clip.mkv

set -euo pipefail

URL="http://localhost:8000"
LANG="ru"
OUTPUT=""
INPUT=""

usage() {
    sed -n '2,7p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--lang)    LANG="$2"; shift 2 ;;
        -u|--url)     URL="${2%/}"; shift 2 ;;
        -o|--output)  OUTPUT="$2"; shift 2 ;;
        -h|--help)    usage 0 ;;
        -*)           echo "unknown option: $1" >&2; usage 1 ;;
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

echo "→ POST $URL/transcribe  file=$INPUT lang=${LANG:-auto}" >&2

curl --fail-with-body -sS -X POST "$URL/transcribe" \
    -F "file=@${INPUT}" \
    -F "language=${LANG}" \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["text"])' \
  > "$OUTPUT"

echo "✔ $OUTPUT" >&2
