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
- Insert paragraph breaks (single newline between paragraphs — **no blank line**)

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
- Inspecting source / normalized / output files between chunk dispatches with `Read`, `Grep`, or Bash (`sed`, `cat`, `head`, `tail`, `awk`, `wc`, `diff`, etc.) — see "Main agent discipline" below

## Main agent discipline — no detours between chunks

The chunk loop must be deterministic, like a `for` loop in a script. Between dispatches the main agent does **exactly two operations**, in this order:

1. Extract `trailing_buffer` from the previous subagent's response via the marker regex (`===BUFFER===\n(.*?)\n===END===`, DOTALL).
2. Dispatch the next chunk's subagent with the new offset, `is_final`, and that buffer.

Everything else is forbidden until **all N chunks have returned**:

- No `Read` on the source, normalized, or output file.
- No Bash commands of any kind — no `sed`, `cat`, `head`, `tail`, `awk`, `grep`, `wc`, `diff`, `ls`, no peeking at chunk boundaries, no spot-checking the output.
- No "let me just verify the buffer looks right" — trust the marker regex; if it matches, dispatch.
- No status narration that requires inspecting content («N слов записано», «чанк завершился на полном предложении») — the main agent has no way to know these without forbidden inspection. A short progress line like "chunk i/N dispatched" is fine.
- No clarifying questions to the user mid-loop unless a subagent returns a hard failure (missing markers, error).

All verification — `wc -w`, content sanity checks, anything that reads the files — happens **only in step 6**, after the final chunk's subagent has returned. If you feel the urge to look at a file mid-loop, that is the drift this rule exists to prevent. Resist it; the loop is supposed to be boring.

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

Create a `TodoWrite` list with exactly N items, one per chunk (`Chunk 1/N`, `Chunk 2/N`, …, `Chunk N/N`). Mark each one completed the moment the buffer has been extracted and the next dispatch is queued. The todo list is the loop counter — it locks the main agent into the for-loop shape and makes any detour visually obvious.

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
You are formatting one chunk of a raw transcript. Follow the rules and tool whitelist below exactly. Do not improvise.

RULES (Iron Law: preserve every word of the source in the same order):
- Add ONLY `.` and `,`. No other punctuation marks: no `?`, `!`, `:`, `;`, `—`, `«»`, `""`, `''`, `()`, `[]`.
- Separate paragraphs with a SINGLE newline (`\n`), NOT a blank line (`\n\n`). One paragraph per line. ~3–7 sentences per paragraph; more is OK if the speaker stays on one topic.
- Capitalize sentence starts and proper nouns.
- Preserve EVERY source word in original order. Filler words, colloquialisms, repeated words, typos — all stay.
- Preserve `[музыка]` / `[music]` markers verbatim.
- Preserve any existing punctuation in the source as-is. Do not change `?` to `.`, do not strip existing quotes or dashes.
- Tech terms stay verbatim (`React`, `Docker`, `Kubernetes`, `npm install`).
- No headings (`#`, `##`, `###`).

ALLOWED TOOLS (exhaustive whitelist — you may use ONLY these two tool calls, in this order, and nothing else):
1. ONE `Read(normalized_path, offset, limit)` call to load your chunk.
2. ONE `Bash` call whose command **literally starts with the word `tee `** — specifically `tee -a "<output_path>" <<'OUT_EOF' ... OUT_EOF`. The very first token of the shell command is `tee`. No prefix command, no pipe, no command substitution, no parentheses, no `cat`, no `echo`, no `printf`.

Total tool calls per run: exactly 2. No exceptions. After that, end your response with the buffer marker block (text only, not a tool call).

Why "starts with `tee`": Claude Code's permission analyzer matches `Bash(tee:*)` only when the command's first word is `tee`. Any prefix (e.g. `cat <<EOF | tee -a ...`) makes the analyzer see `cat` instead of `tee`, fails the static check, and forces a per-chunk permission prompt that breaks the unattended run.

FORBIDDEN ACTIONS (any one of these is a hard failure — abort and return an error instead of doing them):
- Do NOT write or execute scripts of any kind. No `python`, no `python3`, no `node`, no `perl`, no `ruby`, no inline `-c '...'`.
- Do NOT use text-processing utilities: no `sed`, `awk`, `grep`, `cut`, `tr`, `sort`, `uniq`, `head`, `tail`, `cat`, `wc`, `diff`, `fold`, `xargs`.
- Do NOT use shell pipelines (`|`), command substitution (`$(...)` / backticks), subshells (`(...)`), redirection operators other than the heredoc on `tee` (no `>`, `>>`, `<`, `2>&1`), and no command chaining (`&&`, `||`, `;`). Your Bash command is a single `tee -a ... <<'OUT_EOF' ... OUT_EOF` invocation, nothing more.
- Do NOT prefix the heredoc with `cat`: `cat <<'EOF' | tee -a ...` is **wrong** even though it would write the file — the leading `cat` breaks Claude Code's static permission check. Feed the heredoc DIRECTLY to `tee`: `tee -a "<path>" <<'OUT_EOF' ... OUT_EOF`.
- Do NOT create scratch / helper / temporary files. No `cat > /tmp/...`, no `Write`, no `Edit`, no `mkdir`, no `touch`. The only file you write to is `output_path`, via the single `tee -a` heredoc.
- Do NOT make multiple `Read` calls. One chunk = one Read. Do not re-read to "double-check".
- Do NOT make multiple `Bash` calls. One chunk = one `tee -a` call. Do not split paragraphs across calls.
- Do NOT shell out to "find the cutoff point" or "count words" or "verify the buffer". Steps 2–5 below are mental operations on the chunk text inside your own response; they do not use tools.

CORRECT vs WRONG Bash command shape:

✅ CORRECT (first word is `tee`, heredoc is the direct stdin):
```
tee -a "/abs/path/output.md" <<'OUT_EOF'
Первый абзац.
Второй абзац.
OUT_EOF
```

❌ WRONG (first word is `cat`, pipeline triggers permission prompt):
```
cat <<'EOF' | tee -a "/abs/path/output.md"
Первый абзац.
EOF
```

❌ WRONG (uses `>>` redirection, also triggers permission prompt):
```
cat <<'EOF' >> "/abs/path/output.md"
Первый абзац.
EOF
```

❌ WRONG (`echo` prefix):
```
echo "Первый абзац." | tee -a "/abs/path/output.md"
```

INPUT:
- normalized_path: <ABSOLUTE_PATH_TO_NORMALIZED_FILE>
- offset: <OFFSET>
- limit: 40
- trailing_buffer: <BUFFER_STRING_OR_EMPTY>
- is_final: <true|false>
- output_path: <ABSOLUTE_PATH_TO_OUTPUT_FILE>

STEPS (steps 2–5 are done in your head — NO tool calls between Read and tee):
1. [TOOL: Read] `Read(normalized_path, offset, limit)`. Strip the `cat -n` line-number prefix that Read adds (everything before the tab on each line). Join the lines with single spaces. This is your only Read.
2. [MENTAL] Prepend the trailing_buffer (with a space separator if both are non-empty) to form the chunk text.
3. [MENTAL] Add `.` and `,` per the rules. Group into paragraphs.
4. [MENTAL] If is_final is false: pick the cut point yourself by reading the text — find the last sentence that ends on a clear terminal word and a complete thought. Everything after that point is the NEW trailing_buffer. If the text ends on a conjunction (`и`, `или`, `но`, `а`, `что`, `чтобы`, `that`, `because`, `so`), a preposition (`в`, `на`, `с`, `к`, `in`, `on`, `with`, `for`), an article (`the`, `a`, `an`), or any grammatically incomplete fragment, save more to the buffer. When in doubt, save more. **This is a linguistic judgment you make directly — do not write a script, regex, or `rfind` lookup to find the cut.**
5. [MENTAL] If is_final is true: terminate the last sentence with a period if it lacks one. The whole text is finished; the new trailing_buffer is empty.
6. [TOOL: Bash] Append the finished paragraphs to output_path with ONE Bash heredoc (use `tee -a`, NOT `cat >>` — `>>` triggers Claude Code's write-redirection check and prompts per chunk):
   ```
   tee -a "<output_path>" <<'OUT_EOF'
   <finished paragraphs, ONE paragraph per line, no blank lines between them>
   OUT_EOF
   ```
   No leading blank line. Each paragraph is one line; paragraphs are separated by a single `\n`. The heredoc's trailing newline cleanly separates this chunk's last paragraph from the next chunk's first paragraph (still single-newline). `tee -a` echoes content to stdout — harmless, just appears in the tool result. This is your only Bash call.
7. End your response with EXACTLY these three lines (and nothing else after):
   ```
   ===BUFFER===
   <trailing buffer string, or blank line if final>
   ===END===
   ```

SUBAGENT RED FLAGS — if any of these thoughts arise, STOP and just do the mental work in your response:
- "Let me write a quick Python script to find the cutoff phrase" → No. Pick the cut by reading the text.
- "I'll `cat > /tmp/format.py` to handle this cleanly" → No. No scratch files. Ever.
- "Let me run `sed` / `awk` to clean this up" → No. Punctuation is added in your head, written via `tee -a`.
- "I'll do a `Read` again to double-check the chunk" → No. One Read per chunk.
- "Let me `wc -w` to verify the buffer is right size" → No. Trust your judgment; main agent verifies at the end.
- "I'll split this into two `tee -a` calls so each paragraph is separate" → No. One heredoc with all paragraphs.
- "`cat <<'EOF' | tee -a "..."` is the same thing, more idiomatic" → No. The first word of your Bash command must be `tee`. The pipeline form fails the static permission check and prompts the user per chunk, breaking the unattended run.
- "I'll use `>>` redirection, it's simpler than `tee -a`" → No. `>>` is treated as a write operation by Claude Code's permission system and prompts the user per chunk. Use `tee -a` and only `tee -a`.

If you find yourself reaching for any forbidden tool: that is the violation this prompt exists to prevent. The Iron Law of this skill is that processing happens in your head; tools only ferry text in (Read) and out (tee -a).
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

Subagents should start a new paragraph (single `\n`, **not** a blank line) when:

- Speaker shifts topic or subtopic
- After a clear conclusion or summary marker («итак», «поэтому», «короче», "so", "in summary", "to wrap up")
- A paragraph exceeds ~7 sentences and the speaker pauses on a sub-point

Do **not** insert a heading or a blank line between paragraphs. One paragraph per line, separated only by `\n`.

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Subagent added `«»`, `—`, `:`, `?`, `!`, etc. | Re-dispatch the chunk with a stronger reminder: ONLY `.` and `,` are allowed. |
| Subagent deleted filler («э-э», «ну», "um") | Re-dispatch the chunk; filler is part of the speaker's text. |
| Subagent wrote a helper script (`cat > /tmp/*.py`, `python -c`, `sed`, `awk`, etc.) to "find the cutoff" or "process the text" | Hard violation of the subagent tool whitelist. Re-dispatch with the ALLOWED TOOLS / FORBIDDEN ACTIONS blocks emphasized. Steps 2–5 are mental — no shell, no Python. |
| Subagent made >1 Read or >1 Bash call per chunk | Re-dispatch. Exactly one Read (load chunk) and one Bash (`tee -a` heredoc). Anything else is drift. |
| Subagent used `Write` / `Edit` / created temp files | Re-dispatch. The only file write is the single `tee -a` to `output_path`. |
| Subagent used `cat <<'EOF' \| tee -a ...` (heredoc piped through `cat` into `tee`) — triggers per-chunk permission prompt | Permission analyzer matches `Bash(tee:*)` only when the command's first word is `tee`. Re-dispatch with the CORRECT vs WRONG block emphasized: heredoc must be the DIRECT stdin of `tee -a`, no `cat` prefix, no pipe. |
| Subagent used `>>` redirection (`cat <<'EOF' >> "<path>"`) | Same root cause — redirection isn't allowed by the `Bash(tee:*)` rule and prompts per chunk. Re-dispatch using direct `tee -a "<path>" <<'OUT_EOF' ... OUT_EOF`. |
| Main agent ran chunks in parallel | Sequential only — each chunk needs the previous chunk's buffer. |
| Main agent read the chunk itself via Read | Re-do: dispatch the subagent with offset/limit/output_path so the chunk text lives in subagent context, not main. |
| Main agent ran `sed`/`cat`/`head`/`wc`/`Read` between chunk dispatches to "verify" something | Forbidden by "Main agent discipline". Trust the marker regex; verification happens only in step 6. |
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
