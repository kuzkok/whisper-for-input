---
name: formatting-transcript
description: Use when converting a raw transcript (YouTube auto-captions, Whisper output, lecture audio transcription, any source) into a readable article — only adds periods, commas, capitalization, and paragraph breaks while preserving every original word. Triggers on requests like "приведи транскрипт в вид статьи", "расставь знаки препинания", "разбей на абзацы", "format transcript". Works for Russian (incl. mixed-in English tech terms) and English.
---

# Formatting Transcript

Convert a raw transcript file (`.txt`, `.vtt`/`.srt` body, plain auto-captions, Whisper output, any audio-to-text source) into a clean Markdown article. **Iron Law: every word of the source must remain in the output, in the same order.** Only `.`, `,`, capitalization, and paragraph breaks are added. Nothing is paraphrased, summarized, reordered, translated, or omitted.

## How it runs

This skill is implemented as a Python script — the chunk loop is fully deterministic in Python (no LLM in the orchestration), and Haiku is invoked headlessly per chunk via `claude -p --tools "" --json-schema ...` so the model can only return schema-validated JSON, no tool calls, no marker blocks.

**To run it, resolve the script path by trying the two conventional skill install locations (project-local override → user-global), then invoke via Python:**

```bash
SCRIPT="$(realpath .claude/skills/formatting-transcript/format_transcript.py 2>/dev/null \
  || realpath "$HOME/.claude/skills/formatting-transcript/format_transcript.py" 2>/dev/null)"
python "$SCRIPT" "<source-path>" ["<output-path>"]
```

No `find` and no glob — two literal `realpath` lookups so Claude Code's static permission analyzer does not prompt per run. If both fail, abort and tell the user the skill is not installed.

Defaults:
- Output path: `<source-stem>.article.md` next to the source if not provided.
- The script refuses to overwrite an existing output file — pass an explicit second argument or delete the existing file first.
- Progress and per-chunk word counts stream to stderr; the final ±2% verification is the success gate (exit code 0 = within tolerance, 1 = drift detected).

## What the script does

1. **Normalize**: `fold -s -w 100 "<source>"` — soft-wraps long lines at column 100 at whitespace boundaries, so byte-exact word content is preserved while becoming sliceable. Output goes to memory, not a temp file.
2. **Chunk**: 40 normalized lines per chunk → N chunks.
3. **Per chunk**: call Haiku with the chunk text, the previous chunk's trailing buffer, and `is_final`. Haiku returns `{"write": "...", "buffer": "..."}` validated against a JSON schema. Append `write` to the output file; carry `buffer` into the next chunk.
4. **Per-chunk integrity** (Python code, not prompt instruction):
   - **Echo detector**: if `buffer == trailing_buffer` and `trailing_buffer` was non-empty → silent skip → retry (cap 2).
   - **Empty write**: if the model returned no `write` content → retry (cap 2).
   - On retry, an extra `retry_notice` field is added to the input.
5. **Final verify**: `wc -w` source vs output. ≥ ±2% drift → exit 1, prints WARNING.

The subagent's full rules live in `chunk-prompt.md` alongside the script (used as `--system-prompt`).

## When to use

- User asks to "format a transcript", "make it readable", "add punctuation", "split into paragraphs", "превратить в статью", "отформатируй транскрипт", "расставь знаки препинания"
- Source is monolithic text without punctuation, auto-captions stripped of timestamps, Whisper output, etc.
- Goal: readable article that preserves the speaker's exact phrasing

## When NOT to use

- User wants a summary, retelling, or shortening — different task
- User wants translation — different task
- User wants section headings, TOC, or structural rewriting — that is editing, not this skill
- Source is already properly punctuated AND paragraphed — no work needed

## What is preserved vs added

| Added | Preserved verbatim |
|-------|--------------------|
| `.`, `,` | every source word in original order |
| Capitalization at sentence starts and proper nouns | filler words («э», «ну», "um", "you know") |
| Paragraph breaks (single `\n`) | existing source punctuation (`?`, `!`, quotes, dashes) |
| | `[музыка]` / `[music]` markers |
| | tech terms (`React`, `Docker`, `npm install`) verbatim |

## What is forbidden

- Adding any punctuation other than `.` and `,`: no `?`, `!`, `:`, `;`, `—`, `«»`, `""`, `''`, `()`, `[]`
- Changing existing punctuation from the source
- Removing or replacing filler words
- Reordering words
- Translating or transliterating tech terms
- Inserting headings unless the user explicitly asks

These are enforced by the system prompt in `chunk-prompt.md` and by the final ±2% word-count check.

## Failure modes and what to do

| Symptom | Meaning | Action |
|---------|---------|--------|
| Script exits with `WARNING source=X output=Y delta=N%` | Word count drifted by ≥2% | Inspect the output to find what was dropped or invented. Most likely a chunk where Haiku violated the Iron Law — re-run that chunk manually with `--debug api`. |
| `exhausted N attempts; last error: ...` | A single chunk failed echo / empty-write detector or the API call N+1 times in a row | Re-run the whole script; if it repeats on the same chunk, that chunk's text may have unusual structure. Reduce CHUNK_LINES in the script as a workaround. |
| `claude -p` permission prompts | `--tools ""` should prevent any tool call from being attempted, so this shouldn't happen. If it does, check that the `--tools ""` flag is being passed. |
| Output file already exists | The script refuses to overwrite | Delete the existing file or pass an explicit output path argument. |

## Cost guidance

Per call: ~9k cache_creation + ~9k cache_read tokens (mostly Claude Code's harness scaffolding, not our system prompt) + ~1k chunk-specific = ~$0.015 first chunk, ~$0.004 subsequent (cache hit). For a 20-chunk transcript, expect ~$0.10–0.15 total. `--max-budget-usd 0.10` is set as a per-call safety cap.

If you need to drive cost much lower, switch the script to `--bare` mode, which strips the Claude Code system prompt but requires `ANTHROPIC_API_KEY` instead of OAuth/keychain. Not enabled by default to keep the script working out of the box.
