---
name: formatting-transcript
description: Use when converting a raw transcript (YouTube auto-captions, Whisper output, lecture audio transcription, any source) into a readable article — only adds periods, commas, capitalization, and paragraph breaks while preserving every original word. Triggers on requests like "приведи транскрипт в вид статьи", "расставь знаки препинания", "разбей на абзацы", "format transcript". Works for Russian (incl. mixed-in English tech terms) and English.
---

# Formatting Transcript

## Overview

Convert a raw transcript file (`.txt`, `.vtt`/`.srt` body, plain auto-captions, Whisper output, any audio-to-text source) into a clean Markdown article.

**The Iron Law: every word of the source must remain in the output, in the same order.** The only things added are basic punctuation marks (`.`, `,`), capitalization, and paragraph breaks. Nothing is paraphrased, summarized, reordered, or omitted.

Per-chunk processing happens in a Haiku subagent so the main session's context stays bounded and the run is several times faster than processing inline on Opus.

**Permissions**: each chunk's subagent appends to the output via `tee -a`. To avoid a permission prompt per chunk, `Bash(tee:*)` must be in **the current project's** `.claude/settings.local.json` under `permissions.allow`. User-level (`~/.claude/settings.json`) is intentionally NOT used — granting `tee` unconditionally for every project is too broad. The check is local-only by design.

**Before any chunk dispatch, the main agent MUST verify the rule is present locally**:

```bash
grep -q '"Bash(tee:\*)"' .claude/settings.local.json 2>/dev/null && echo present || echo missing
```

If `missing`, ask the user once: "Add `Bash(tee:*)` to this project's `.claude/settings.local.json`? Otherwise every chunk will prompt." On confirmation, edit the file directly (create it if absent, with `{"permissions": {"allow": ["Bash(tee:*)"]}}`) — do not rely on Claude Code's interactive "always allow" flow, which sometimes asks twice (once for project scope, once for user scope) and the second prompt can land mid-loop. A direct edit before dispatch is unambiguous.

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
- Inspecting source / normalized / output files between chunk dispatches with `Read`, `Grep`, or Bash (`sed`, `cat`, `head`, `tail`, `awk`, `diff`, etc.). The ONE exception is `wc -w "<output_path>"` for the per-chunk growth check — see "Main agent discipline" below

## Main agent discipline — no detours between chunks

The chunk loop must be deterministic, like a `for` loop in a script — but with two cheap structural signals per chunk that catch silent failures (echo, no growth) before they accumulate.

**Hard preconditions before step 4 (first chunk dispatch):**

1. A `TodoWrite` call with exactly N items (`Chunk 1/N` … `Chunk N/N`) has been made.
2. The output file has been `touch`-ed so it exists and is empty.

No `Agent` dispatch happens before both are done. The todo list is the loop counter — without it visible in the harness UI, drift becomes invisible to both you and the user. The empty output file lets the per-chunk `wc -w` growth check (below) return 0 on the first dispatch instead of erroring on a missing file. If you find yourself about to dispatch chunk 1 and either is missing, STOP, do them now, then resume.

Between dispatches the main agent does **exactly these five steps**, in this order, on the just-returned subagent response:

1. **Extract buffer** — regex `===BUFFER===\n(.*?)\n===END===` (DOTALL). Record whether the markers were present.
2. **Echo detector** — compare the extracted buffer to `trailing_buffer_in` (the buffer passed IN to that subagent). If they are byte-equal AND `trailing_buffer_in` was non-empty → `echoed = true`.
3. **Growth check** — run `wc -w <output_path>` (the ONE Bash command permitted mid-loop). Compute `delta = new_words - prev_words` (track `prev_words`, starts at 0).
4. **Decide** per the rules in step 5 of the workflow (re-dispatch / abort / advance). Mark the todo `completed` and update `prev_words` ONLY on `advance`.
5. **Dispatch** the next chunk's subagent with the new offset, `is_final`, and the extracted buffer as the new `trailing_buffer_in`. Set the next chunk's todo `in_progress`.

Everything else is forbidden until **all N chunks have returned**:

- No `Read` on the source, normalized, or output file.
- No Bash commands of any kind EXCEPT the single `wc -w <output_path>` for the growth check above — no `sed`, `cat`, `head`, `tail`, `awk`, `grep`, `diff`, `ls`, no other `wc` invocations (no `wc -l`, no `wc` on source/normalized), no peeking at chunk boundaries, no spot-checking the output content.
- `wc -w` returns ONLY a numeric word count, never content. Reading any portion of the output text mid-loop is forbidden — even `head -c 100`, even one line. The two structural signals (echo, growth) are all the per-chunk verification the main agent gets; do not try to compensate with eyeballing.
- No status narration that requires inspecting content («чанк завершился на полном предложении») — the main agent has no way to know that without forbidden inspection. A short progress line like "chunk i/N dispatched, delta=+612 words" is fine.
- No clarifying questions to the user mid-loop unless a chunk has hit the retry cap (step 5).

All other verification — content sanity checks, final source-vs-output word count — happens **only in step 6**, after the final chunk's subagent has returned. If you feel the urge to look at file content mid-loop, that is the drift this rule exists to prevent. Resist it; the loop is supposed to be boring, with two cheap structural signals per chunk and nothing more.

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

### 3a. Set up loop infrastructure (TodoWrite + touch) — STOP gate before any dispatch

**Both of these are hard preconditions for step 4. Do not call `Agent` until both are done.**

1. **TodoWrite the loop counter.** Call `TodoWrite` exactly once with N items, one per chunk: `Chunk 1/N`, `Chunk 2/N`, …, `Chunk N/N`. All start as `pending`. Mark the first as `in_progress` either in this initial call or together with the first dispatch.
2. **Touch the output file.** `touch "<output_path>"` so the file exists empty. The per-chunk growth check (step 5) runs `wc -w <output_path>` after every chunk; without a pre-existing file the first run errors instead of returning 0. Initialize `prev_words = 0` in your head.

Why these are gates, not optional bookkeeping: without the todo list visible in the harness UI, drift (skipped chunks, off-by-one in offsets, parallel dispatches) becomes invisible to both you and the user. Without the empty output file, the growth check fails open on chunk 1. Both must be true before any `Agent` call.

Common failure mode: dispatching chunks 1–2 first and creating the todo list afterwards. That defeats the purpose — by the time the list appears, the early chunks are already past the gate's protection. **If you catch yourself about to call `Agent` for chunk 1 and either gate is missing, STOP, do both now, then proceed.**

Throughout the loop: mark each chunk `completed` ONLY after the step 5 per-chunk verification passes; mark the next one `in_progress` in the same or adjacent `TodoWrite` call.

### 4. For each chunk: dispatch a Haiku subagent

**Sequentially** (NOT in parallel — each subagent needs the trailing buffer from the previous one), dispatch a subagent using the `Agent` tool with `subagent_type: "general-purpose"` and `model: "haiku"`.

**Why the dispatch prompt is thin**: the subagent's full rules live in `chunk-prompt.md` next to this SKILL.md (~10 KB). Inlining those rules in every `Agent(prompt=...)` call duplicates them N times in the main agent's context — at 20 chunks that is ~40k tokens of pure boilerplate. Instead, the dispatch tells the subagent to Read `chunk-prompt.md` once, and the main agent only carries the per-chunk variables.

**Before the first dispatch** (after the step 3a TodoWrite gate), resolve the absolute path to `chunk-prompt.md` once and reuse it. It lives in the same directory as this `SKILL.md` — i.e. inside the `formatting-transcript/` skill folder, wherever Claude Code loaded the skill from. Do NOT `find .` from the current working directory: the skill almost never lives in the user's project tree. Instead try the two conventional install locations in priority order — project-local override → user-global:

```bash
realpath ".claude/skills/formatting-transcript/chunk-prompt.md" 2>/dev/null \
  || realpath "$HOME/.claude/skills/formatting-transcript/chunk-prompt.md" 2>/dev/null
```

The command is intentionally `find`-free and glob-free — any unquoted `*` (even one safely tucked inside single quotes inside a `find -path`) trips Claude Code's static permission analyzer and prompts the user once per skill startup. Two literal `realpath` calls do not.

Cache the first non-empty result. If both lookups fail, abort and tell the user — do not inline the prompt or fall back to a tree-wide `find`. Plugin-bundled installs (`~/.claude/plugins/<plugin>/skills/formatting-transcript/`) are not auto-discovered; the user should symlink such a path into `~/.claude/skills/formatting-transcript/` or pass the absolute path to chunk-prompt.md manually.

The subagent itself:
1. Reads `chunk-prompt.md` once (its only Read of the rules file)
2. Reads its own chunk slice via `Read(<normalized_path>, offset, limit)` — main agent does NOT read the chunk
3. Prepends the trailing buffer from the previous chunk (empty for chunk 1)
4. Adds `.`, `,`, paragraph breaks per the rules
5. Appends the finished paragraphs to the output file via Bash `tee -a ... <<'EOF'` heredoc (use `tee -a`, NOT `cat >>` — Claude Code's permission system treats shell redirection `>>` as a "write operation" requiring per-file approval; `tee -a` writes via its file argument and is permitted by a single `Bash(tee:*)` rule)
6. Returns ONLY the trailing buffer string (or empty for the final chunk) between `===BUFFER===` / `===END===` markers

This bounds the main agent's context to a small dispatch wrapper (~200 tokens) + the short buffer string between chunks. The chunk content, processed paragraphs, and rules themselves never enter the main agent's context.

#### Subagent dispatch prompt template

Thin wrapper — the actual rules live in `chunk-prompt.md`. Substitute the bracketed values per chunk:

```
You are a Haiku subagent for the formatting-transcript skill. First, Read the rules at:
  <ABSOLUTE_PATH_TO_CHUNK_PROMPT_MD>

Then execute on this chunk with these inputs:
- normalized_path: <ABSOLUTE_PATH_TO_NORMALIZED_FILE>
- offset: <OFFSET>
- limit: 40
- trailing_buffer: <BUFFER_STRING_OR_EMPTY>
- is_final: <true|false>
- output_path: <ABSOLUTE_PATH_TO_OUTPUT_FILE>

Exactly three tool calls total: (1) Read of chunk-prompt.md, (2) Read of the chunk slice, (3) one Bash whose first word is `tee` to append the WRITE-part to output_path. End your response with the ===BUFFER=== / ===END=== marker block per the rules file.
```

That is the entire dispatch prompt. Do not paste the rules from `chunk-prompt.md` into it — defeats the whole point.

### 5. Per-chunk verification: extract buffer, echo detector, growth check

After each subagent returns, run these three checks in order:

1. **Extract buffer** — regex `===BUFFER===\n(.*?)\n===END===` (DOTALL). Record `markers_present` (bool).
2. **Echo detector** — `echoed = (extracted_buffer == trailing_buffer_in) AND (trailing_buffer_in != "")`. The echoed-input failure mode means the subagent silent-skipped the chunk — it returned its input unchanged instead of processing forward.
3. **Growth check** — run `wc -w "<output_path>"`, compute `delta = new_words - prev_words`.

Then decide:

| `delta` | `markers_present` AND NOT `echoed` | Action |
|---------|------------------------------------|--------|
| `> 0`   | true (success)                     | **Advance**: mark todo `completed`, set `prev_words = new_words`, dispatch next chunk. |
| `> 0`   | false (markers missing OR echoed)  | **Abort and report**: file grew with potentially-correct content, but buffer state is corrupted. Re-dispatching would double-write. Tell the user which chunk hit the ambiguous state, the offset, and the file's current word count so they can inspect and decide. Do NOT re-dispatch automatically. |
| `== 0`  | (any)                              | **Re-dispatch with retry preamble**: subagent did not write to the file, so retrying is safe (no double-write risk). Track retry count per chunk. |

**Retry cap: 2 retries per chunk.** On the 3rd consecutive failure for the same chunk (delta still 0), abort and tell the user — do not keep looping.

**Retry preamble** — prepend this to the standard dispatch template on every retry:

```
ATTENTION: previous attempt at this chunk failed the silent-skip detector — either no `===BUFFER===` / `===END===` block in your response, OR your output buffer was byte-identical to the input trailing_buffer (echo), OR the output file did not grow at all. All three mean you did not actually process the chunk. Re-read the chunk slice, do the mental work, and write the WRITE-part via a single `tee -a` Bash. The BUFFER-part you return MUST differ from your input trailing_buffer; the file MUST grow.
```

Discard everything in the subagent's response outside the marker block.

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
| Subagent wrote the full combined text to file AND returned its tail as buffer (boundary duplication) | Re-dispatch with the "TWO DISJOINT parts" framing emphasized: WRITE-part and BUFFER-part are disjoint; the trailing_buffer is held back from this write, not echoed alongside it. |
| Subagent's returned BUFFER is byte-identical to the input `trailing_buffer` (echo / silent skip) | Caught by main-agent echo-detector in step 5. If `delta == 0` (file didn't grow either) → re-dispatch with the retry preamble, cap 2 retries. If `delta > 0` (file grew with echoed buffer) → abort and report ambiguous state to the user; re-dispatching would double-write. |
| `wc -w "<output_path>"` did not increase after chunk dispatch (`delta == 0`) | Caught by main-agent growth check in step 5. Subagent silent-skipped — file unchanged, safe to re-dispatch. Cap 2 retries; on 3rd failure abort and report. |
| Main agent forgot to `touch "<output_path>"` before chunk 1 | Growth check errors on missing file. Catch this at the step 3a gate — `touch` is a precondition equal to `TodoWrite`. |
| Subagent wrote a helper script (`cat > /tmp/*.py`, `python -c`, `sed`, `awk`, etc.) to "find the cutoff" or "process the text" | Hard violation of the subagent tool whitelist. Re-dispatch with the ALLOWED TOOLS / FORBIDDEN ACTIONS blocks emphasized. Steps 2–5 are mental — no shell, no Python. |
| Subagent made >2 Reads or >1 Bash call per chunk | Re-dispatch. Exactly two Reads (chunk-prompt.md + chunk slice) and one Bash (`tee -a` heredoc). Anything else is drift. |
| Main agent inlined the full subagent rules in the dispatch prompt | Defeats the whole reason `chunk-prompt.md` exists — each dispatch then carries ~2k tokens of rules into main-agent context, ballooning past 10 chunks. Re-dispatch with the thin wrapper that just points the subagent to `chunk-prompt.md`. |
| Subagent used `Write` / `Edit` / created temp files | Re-dispatch. The only file write is the single `tee -a` to `output_path`. |
| Subagent used `cat <<'EOF' \| tee -a ...` (heredoc piped through `cat` into `tee`) — triggers per-chunk permission prompt | Permission analyzer matches `Bash(tee:*)` only when the command's first word is `tee`. Re-dispatch with the CORRECT vs WRONG block emphasized: heredoc must be the DIRECT stdin of `tee -a`, no `cat` prefix, no pipe. |
| Subagent used `>>` redirection (`cat <<'EOF' >> "<path>"`) | Same root cause — redirection isn't allowed by the `Bash(tee:*)` rule and prompts per chunk. Re-dispatch using direct `tee -a "<path>" <<'OUT_EOF' ... OUT_EOF`. |
| Main agent dispatched chunk 1 (or 1–2) before calling `TodoWrite` | Violates the step 3a gate. The todo list is the loop counter — it must exist BEFORE any `Agent` call. Catch yourself, call `TodoWrite` with all N items now, mark already-processed chunks as `completed`, then continue. |
| Main agent ran chunks in parallel | Sequential only — each chunk needs the previous chunk's buffer. |
| Main agent skipped the `Bash(tee:*)` pre-check and ran into a permission prompt mid-loop | Always run the local-grep check from "Permissions" before step 4. If absent, edit `.claude/settings.local.json` directly — do not rely on the interactive "always allow" flow. |
| Main agent read the chunk itself via Read | Re-do: dispatch the subagent with offset/limit/output_path so the chunk text lives in subagent context, not main. |
| Main agent ran `sed`/`cat`/`head`/`Read` between chunk dispatches, or ran `wc -l`/`wc` on source/normalized files, or `wc -w` on anything other than `<output_path>` | Forbidden by "Main agent discipline". The only Bash permitted mid-loop is exactly `wc -w "<output_path>"` for the per-chunk growth check; all other inspection is for step 6. |
| Skipped `fold` because "file looks fine" | Run it anyway. It is idempotent — already-wrapped files pass through. |
| Lost the last words of a chunk | Subagent must save trailing buffer at non-final chunks. |
| Subagent response missed the `===BUFFER===` markers | Re-dispatch with output-format reminder emphasized. |
| Word count off by >2% | A subagent dropped or added words. Inspect the chunk that flipped the count. |

## Red flags — STOP and revert

- "It reads better with em-dash here" → No. Only `.` and `,`.
- "I'll merge these two near-identical sentences" → No. The speaker said both.
- "A heading here would help structure" → No. Paragraphs only, unless user explicitly asked.
- "Word count is off by 10%" → Content was changed. Find what was removed or invented.
- "The same phrase appears at the end of chunk N and start of chunk N+1" → Boundary duplication. The subagent wrote the trailing_buffer to the file instead of holding it back. WRITE-part and BUFFER-part must be disjoint.

## Quick reference

| Task | How |
|------|-----|
| Normalize line widths | `fold -s -w 100 <source> > <source-stem>.normalized.txt` |
| Total lines | `wc -l <source-stem>.normalized.txt` |
| Number of chunks | `N = ceil(total_lines / 40)` |
| Default output path | `<source-stem>.article.md` next to source |
| Resolve chunk-prompt.md path | `realpath ".claude/skills/formatting-transcript/chunk-prompt.md" 2>/dev/null \|\| realpath "$HOME/.claude/skills/formatting-transcript/chunk-prompt.md" 2>/dev/null` — two literal paths, no `find`, no glob (avoids the permission analyzer's `*` warning). Run once, cache the absolute path. |
| Dispatch chunk i | `Agent(subagent_type="general-purpose", model="haiku", prompt=<thin wrapper: pointer to chunk-prompt.md absolute path + variables offset=1+40*(i-1), limit=40, trailing_buffer=<prev_buffer>, is_final=(i==N), output_path=...>)` |
| Pre-touch output file | `touch "<output_path>"` once in step 3a so per-chunk `wc -w` returns 0 instead of erroring |
| Extract buffer from response | regex `===BUFFER===\n(.*?)\n===END===` (DOTALL) |
| Per-chunk echo check | `extracted_buffer == trailing_buffer_in AND trailing_buffer_in != ""` → silent skip; if `delta == 0` re-dispatch (cap 2), else abort |
| Per-chunk growth check | `wc -w "<output_path>"` → `delta = new - prev_words`; if `delta == 0` re-dispatch (cap 2); update `prev_words` only on success |
| Verify preservation | `wc -w` source vs output, expect ±2% |
| Clean up | `rm <source-stem>.normalized.txt` after verification passes |
