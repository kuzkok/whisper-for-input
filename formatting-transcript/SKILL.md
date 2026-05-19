---
name: formatting-transcript
description: Use when converting a raw transcript (YouTube auto-captions, Whisper output, lecture audio transcription, any source) into a readable article — only adds periods, commas, capitalization, and paragraph breaks while preserving every original word. Triggers on requests like "приведи транскрипт в вид статьи", "расставь знаки препинания", "разбей на абзацы", "format transcript". Works for Russian (incl. mixed-in English tech terms) and English.
---

# Formatting Transcript

## Overview

Convert a raw transcript file (`.txt`, `.vtt`/`.srt` body, plain auto-captions, Whisper output, any audio-to-text source) into a clean Markdown article.

**The Iron Law: every word of the source must remain in the output, in the same order.** The only things added are basic punctuation marks (`.`, `,`), capitalization, and paragraph breaks. Nothing is paraphrased, summarized, reordered, or omitted.

Per-chunk processing happens in a Haiku subagent so the main session's context stays bounded and the run is several times faster than processing inline on Opus.

**Permissions**: each chunk's subagent appends to the output via `tee -a`. To avoid a permission prompt per chunk, ensure `Bash(tee:*)` is in `.claude/settings.local.json` under `permissions.allow` before starting. If absent, the first chunk will prompt — the user can click "always allow" once and continue.

## When to use

- User asks to "format a transcript", "make it readable", "add punctuation", "split into paragraphs", "превратить в статью", "отформатируй транскрипт"
- Source is monolithic text without punctuation, or auto-captions stripped of timestamps, or Whisper output
- Goal: readable article that preserves the speaker's exact phrasing

## When NOT to use

- User wants a summary, retelling, or shortening — different task
- User wants translation — different task
- User wants section headings, TOC, or structural rewriting — that is editing, not this skill
- Source is already properly punctuated AND paragraphed — no work needed

## What is allowed

- Add `.` and `,`
- Capitalize sentence starts and proper nouns
- Insert blank lines between paragraphs

## What is forbidden

- Adding any other punctuation: `?`, `!`, `:`, `;`, `—`, `«»`, `""`, `''`, `()`, `[]`
- Changing existing punctuation from the source (preserve as-is, including existing `?`, `!`, quotes, dashes)
- Removing filler words («э», «ну», «значит», «как бы», "um", "you know", "like") — they stay
- Replacing colloquial words with literary equivalents
- Reordering words within a sentence
- Adding or removing words "for clarity"
- Translating or transliterating tech terms (`React`, `Docker`, `npm install` stay verbatim in any language)
- Inserting headings (`#`, `##`, `###`) unless the user explicitly asks
- Processing chunks inline in the main agent — always dispatch to a Haiku subagent (see step 4)

## Workflow

### 1. Resolve paths

- **Source**: file path provided by the user
- **Default output**: `<source-stem>.article.md` next to the source (e.g. `lecture.txt` → `lecture.article.md`)
- **Working copy**: `<source-stem>.normalized.txt` next to the source — a temporary, line-normalized version used for chunk slicing
- If output would overwrite an existing file, ask the user before proceeding

### 2. Normalize line widths and count

Transcripts often arrive as **one huge single line** (YouTube `ytdlp_download_transcript`, Whisper raw output, etc.). Slicing such a file by line offset would load the entire transcript at once. So **always normalize line widths before chunked processing**:

```bash
fold -s -w 100 "<source-path>" > "<source-stem>.normalized.txt"
wc -l "<source-stem>.normalized.txt"
```

- `fold -s -w 100` soft-wraps at column 100, breaking only at whitespace (`-s`) so words and UTF-8 characters are never split.
- 40 such lines ≈ 3–4 KB ≈ a safe, predictable chunk for a Haiku subagent.
- The wrap is purely about chunk size; the **original wording, including word order and `[музыка]` / `[music]` markers, is preserved byte-for-byte**. Verify with `wc -w` before and after — counts must be identical.
- Use `wc -l` on the normalized file to get total lines `L`. Number of chunks `N = ceil(L / 40)`.

### 3. Plan the chunk schedule

| Chunk | offset | limit | is_final |
|-------|--------|-------|----------|
| 1     | 1      | 40    | false    |
| 2     | 41     | 40    | false    |
| 3     | 81     | 40    | false    |
| …     | …      | 40    | false    |
| N     | 1+40*(N-1) | 40 | **true** |

### 4. For each chunk: dispatch a Haiku subagent

**Sequentially** (NOT in parallel — each subagent needs the trailing buffer from the previous one), dispatch a subagent using the `Agent` tool with `subagent_type: "general-purpose"` and `model: "haiku"`.

The subagent itself:
1. Reads its own chunk slice via `Read(<normalized_path>, offset, limit)` — main agent does NOT read the chunk
2. Prepends the trailing buffer from the previous chunk (empty for chunk 1)
3. Adds `.`, `,`, paragraph breaks per the rules
4. Appends the finished paragraphs to the output file via Bash `tee -a ... <<'EOF'` heredoc (use `tee -a`, NOT `cat >>` — Claude Code's permission system treats shell redirection `>>` as a "write operation" requiring per-file approval; `tee -a` writes via its file argument and is permitted by a single `Bash(tee:*)` rule)
5. Returns ONLY the trailing buffer string (or empty for the final chunk)

This bounds the main agent's context to dispatch params + the short buffer string between chunks. The chunk content and processed paragraphs never enter the main agent's context.

#### Subagent prompt template

Substitute the bracketed values per chunk:

```
You are formatting one chunk of a raw transcript. Follow the rules below exactly.

RULES (Iron Law: preserve every word of the source in the same order):
- Add ONLY `.` and `,`. No other punctuation marks: no `?`, `!`, `:`, `;`, `—`, `«»`, `""`, `''`, `()`, `[]`.
- Insert blank lines between paragraphs (~3–7 sentences per paragraph; more is OK if the speaker stays on one topic).
- Capitalize sentence starts and proper nouns.
- Preserve EVERY source word in original order. Filler words, colloquialisms, repeated words, typos — all stay.
- Preserve `[музыка]` / `[music]` markers verbatim.
- Preserve any existing punctuation in the source as-is. Do not change `?` to `.`, do not strip existing quotes or dashes.
- Tech terms stay verbatim (`React`, `Docker`, `Kubernetes`, `npm install`).
- No headings (`#`, `##`, `###`).

INPUT:
- normalized_path: <ABSOLUTE_PATH_TO_NORMALIZED_FILE>
- offset: <OFFSET>
- limit: 40
- trailing_buffer: <BUFFER_STRING_OR_EMPTY>
- is_final: <true|false>
- output_path: <ABSOLUTE_PATH_TO_OUTPUT_FILE>

STEPS:
1. Read the chunk: `Read(normalized_path, offset, limit)`. Strip the `cat -n` line-number prefix that Read adds (everything before the tab on each line). Join the lines with single spaces.
2. Prepend the trailing_buffer (with a space separator if both are non-empty) to form the chunk text.
3. Add `.` and `,` per the rules. Group into paragraphs.
4. If is_final is false: cut at the last complete sentence. Everything after that is the NEW trailing_buffer. If the text ends on a conjunction (`и`, `или`, `но`, `а`, `что`, `чтобы`, `that`, `because`, `so`), a preposition (`в`, `на`, `с`, `к`, `in`, `on`, `with`, `for`), an article (`the`, `a`, `an`), or any grammatically incomplete fragment, save more to the buffer. When in doubt, save more.
5. If is_final is true: terminate the last sentence with a period if it lacks one. The whole text is finished; the new trailing_buffer is empty.
6. Append the finished paragraphs to output_path with a Bash heredoc (use `tee -a`, NOT `cat >>` — `>>` triggers Claude Code's write-redirection check and prompts per chunk):
   ```
   tee -a "<output_path>" <<'OUT_EOF'
   
   <finished paragraphs separated by blank lines>
   OUT_EOF
   ```
   The leading blank line preserves paragraph separation between chunks. `tee -a` will echo the content to stdout — that's harmless, it just appears in the tool result.
7. End your response with EXACTLY these three lines (and nothing else after):
   ```
   ===BUFFER===
   <trailing buffer string, or blank line if final>
   ===END===
   ```
```

### 5. Collect the buffer between chunks

After each subagent returns, extract the text between `===BUFFER===` and `===END===` markers (regex: `===BUFFER===\n(.*?)\n===END===` with DOTALL). Pass that string as `trailing_buffer` to the next chunk's subagent. Discard the rest of the subagent's response.

If a subagent response lacks the markers, treat it as a failure — re-dispatch the same chunk with an emphasized reminder about the output format.

### 6. Verify preservation and clean up

```bash
wc -w <source>
wc -w <output>
```

Output word count must be within roughly ±2% of source (allowance for stripped timestamps or VTT cue headers). If the difference is larger, a subagent removed or invented words — inspect the output and report to the user before claiming completion.

After verification passes, delete the temporary normalized file:

```bash
rm <source-stem>.normalized.txt
```

If verification fails, **keep** the normalized file — it helps debug which chunk went wrong.

## Paragraph break heuristics

Subagents should insert a blank line when:

- Speaker shifts topic or subtopic
- After a clear conclusion or summary marker («итак», «поэтому», «короче», "so", "in summary", "to wrap up")
- A paragraph exceeds ~7 sentences and the speaker pauses on a sub-point

Do **not** insert a heading instead of a blank line. Paragraphs only.

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Subagent added `«»`, `—`, `:`, `?`, `!`, etc. | Re-dispatch the chunk with a stronger reminder: ONLY `.` and `,` are allowed. |
| Subagent deleted filler («э-э», «ну», "um") | Re-dispatch the chunk; filler is part of the speaker's text. |
| Main agent ran chunks in parallel | Sequential only — each chunk needs the previous chunk's buffer. |
| Main agent read the chunk itself via Read | Re-do: dispatch the subagent with offset/limit/output_path so the chunk text lives in subagent context, not main. |
| Skipped `fold` because "file looks fine" | Run it anyway. It is idempotent — already-wrapped files pass through. |
| Lost the last words of a chunk | Subagent must save trailing buffer at non-final chunks. |
| Subagent response missed the `===BUFFER===` markers | Re-dispatch with output-format reminder emphasized. |
| Word count off by >2% | A subagent dropped or added words. Inspect the chunk that flipped the count. |

## Red flags — STOP and revert

- "It reads better with em-dash here" → No. Only `.` and `,`.
- "I'll merge these two near-identical sentences" → No. The speaker said both.
- "A heading here would help structure" → No. Paragraphs only, unless user explicitly asked.
- "Word count is off by 10%" → Content was changed. Find what was removed or invented.

## Quick reference

| Task | How |
|------|-----|
| Normalize line widths | `fold -s -w 100 <source> > <source-stem>.normalized.txt` |
| Total lines | `wc -l <source-stem>.normalized.txt` |
| Number of chunks | `N = ceil(total_lines / 40)` |
| Default output path | `<source-stem>.article.md` next to source |
| Dispatch chunk i | `Agent(subagent_type="general-purpose", model="haiku", prompt=<filled subagent prompt with offset=1+40*(i-1), limit=40, trailing_buffer=<prev_buffer>, is_final=(i==N), output_path=...>)` |
| Extract buffer from response | regex `===BUFFER===\n(.*?)\n===END===` (DOTALL) |
| Verify preservation | `wc -w` source vs output, expect ±2% |
| Clean up | `rm <source-stem>.normalized.txt` after verification passes |
