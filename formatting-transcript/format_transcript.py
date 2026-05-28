#!/usr/bin/env python3
"""Headless transcript formatter — Haiku via `claude -p`, deterministic loop in Python.

Usage: format_transcript.py <source-path> [output-path]

Default output: <source-stem>.article.md next to the source.

Iron law of the skill — preserve every source word in order, only add `.`, `,`,
capitalization, and paragraph breaks. The model is invoked headlessly with
--tools "" so it cannot call any tools — pure inference, JSON in / JSON out,
schema-enforced via --json-schema. The chunk loop, echo / empty-write detectors,
and retry logic live here in Python as plain code.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CHUNK_LINES = 40
MAX_RETRIES = 2
MODEL = "haiku"
PER_CALL_TIMEOUT_S = 180
PER_CALL_BUDGET_USD = "0.10"

SCHEMA = {
    "type": "object",
    "properties": {
        "write": {"type": "string"},
        "buffer": {"type": "string"},
    },
    "required": ["write", "buffer"],
    "additionalProperties": False,
}


def call_haiku(system_prompt: str, user_msg: str) -> dict:
    proc = subprocess.run(
        [
            "claude", "-p",
            "--model", MODEL,
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
        timeout=PER_CALL_TIMEOUT_S,
        check=True,
    )
    envelope = json.loads(proc.stdout)
    if envelope.get("is_error"):
        raise RuntimeError(f"claude -p reported is_error: {envelope.get('result', '')!r}")
    structured = envelope.get("structured_output")
    if not isinstance(structured, dict) or "write" not in structured or "buffer" not in structured:
        raise RuntimeError(f"missing/invalid structured_output in envelope: {envelope!r}")
    return structured


def process_chunk(
    system_prompt: str,
    chunk: str,
    trailing: str,
    is_final: bool,
    retry: bool = False,
) -> tuple[str, str]:
    payload = {
        "chunk": chunk,
        "trailing_buffer": trailing,
        "is_final": is_final,
    }
    if retry:
        payload["retry_notice"] = (
            "Previous attempt failed the silent-skip detector — either your buffer "
            "was byte-equal to the input trailing_buffer (echo) or your write was "
            "empty. Re-process the chunk text fully."
        )
    out = call_haiku(system_prompt, json.dumps(payload, ensure_ascii=False))
    return out["write"], out["buffer"]


def chunk_with_retry(
    system_prompt: str,
    chunk: str,
    trailing: str,
    is_final: bool,
    label: str,
) -> tuple[str, str]:
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            write, new_buf = process_chunk(
                system_prompt, chunk, trailing, is_final, retry=attempt > 0
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            RuntimeError,
        ) as e:
            last_err = e
            print(f"  {label}: attempt {attempt + 1} — error: {e}", file=sys.stderr)
            continue
        if trailing.strip() and new_buf.strip() == trailing.strip():
            print(f"  {label}: attempt {attempt + 1} — buffer echo, retrying", file=sys.stderr)
            continue
        if not write.strip():
            print(f"  {label}: attempt {attempt + 1} — empty write, retrying", file=sys.stderr)
            continue
        return write, new_buf
    raise RuntimeError(
        f"{label}: exhausted {MAX_RETRIES + 1} attempts"
        + (f"; last error: {last_err}" if last_err else "")
    )


def normalize(src: Path) -> list[str]:
    proc = subprocess.run(
        ["fold", "-s", "-w", "100", str(src)],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.splitlines()


def word_count(path: Path) -> int:
    proc = subprocess.run(
        ["wc", "-w", str(path)], capture_output=True, text=True, check=True
    )
    return int(proc.stdout.split()[0])


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: format_transcript.py <source> [output]", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    if not src.is_file():
        print(f"source not found: {src}", file=sys.stderr)
        return 2
    out = Path(sys.argv[2]) if len(sys.argv) == 3 else src.with_suffix(".article.md")
    if out.exists():
        print(f"output already exists, refusing to overwrite: {out}", file=sys.stderr)
        return 2

    lines = normalize(src)
    n_chunks = (len(lines) + CHUNK_LINES - 1) // CHUNK_LINES
    print(f"{len(lines)} lines → {n_chunks} chunks", file=sys.stderr)

    system_prompt = (Path(__file__).resolve().parent / "chunk-prompt.md").read_text()

    out.write_text("", encoding="utf-8")
    buf = ""
    for i in range(n_chunks):
        offset = i * CHUNK_LINES
        chunk = "\n".join(lines[offset : offset + CHUNK_LINES])
        is_final = offset + CHUNK_LINES >= len(lines)
        label = f"chunk {i + 1}/{n_chunks}"
        print(f"{label} (lines {offset + 1}-{min(offset + CHUNK_LINES, len(lines))})...", file=sys.stderr, end=" ", flush=True)
        write, buf = chunk_with_retry(system_prompt, chunk, buf, is_final, label)
        with out.open("a", encoding="utf-8") as f:
            f.write(write + "\n")
        print(f"+{len(write.split())} words", file=sys.stderr)

    src_words = word_count(src)
    out_words = word_count(out)
    delta_pct = abs(out_words - src_words) / max(src_words, 1) * 100
    status = "OK" if delta_pct < 2.0 else "WARNING"
    print(
        f"\n[{status}] source={src_words} output={out_words} delta={delta_pct:.2f}%",
        file=sys.stderr,
    )
    return 0 if delta_pct < 2.0 else 1


if __name__ == "__main__":
    sys.exit(main())
