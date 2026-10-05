#!/usr/bin/env python3
"""Фейковый `claude -p` для интеграционных тестов: без сети, детерминированный.

Возвращает chunk как есть (абзацы не начинает) — выход скрипта обязан совпасть
со входом пословно. Формат envelope совпадает с реальным
claude -p --output-format json.

Управляется переменными окружения:
  FAKE_CLAUDE_LOG — файл, куда дописывается первое слово каждого chunk'а
                    (один вызов — одна строка);
  FAKE_CLAUDE_FAIL_MARK — слово: chunk, содержащий его, получает ответ
                    лимита подписки (api 429, rc 1).
"""
import json
import os
import sys


def main() -> int:
    payload = json.loads(sys.argv[-1])
    chunk = payload["chunk"]

    log = os.environ.get("FAKE_CLAUDE_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(chunk.split()[0] + "\n")

    mark = os.environ.get("FAKE_CLAUDE_FAIL_MARK")
    if mark and mark in chunk.split():
        print(json.dumps({
            "is_error": True, "subtype": "error", "api_error_status": 429,
            "result": "usage limit reached",
        }))
        return 1

    print(json.dumps({
        "is_error": False,
        "total_cost_usd": 0.01,
        "usage": {},
        "structured_output": {"text": chunk, "starts_paragraph": False},
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
