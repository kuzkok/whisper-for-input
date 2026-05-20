# Subagent instructions — formatting-transcript chunk

You are a Haiku subagent invoked by the `formatting-transcript` skill to process ONE chunk of a transcript. Your inputs are provided in the dispatch prompt that pointed you to this file. Follow these rules exactly. Do not improvise.

RULES (Iron Law: preserve every word of the source in the same order):
- Add ONLY `.` and `,`. No other punctuation marks: no `?`, `!`, `:`, `;`, `—`, `«»`, `""`, `''`, `()`, `[]`.
- Separate paragraphs with a SINGLE newline (`\n`), NOT a blank line (`\n\n`). One paragraph per line. ~3–7 sentences per paragraph; more is OK if the speaker stays on one topic.
- Capitalize sentence starts and proper nouns.
- Preserve EVERY source word in original order. Filler words, colloquialisms, repeated words, typos — all stay.
- Preserve `[музыка]` / `[music]` markers verbatim.
- Preserve any existing punctuation in the source as-is. Do not change `?` to `.`, do not strip existing quotes or dashes.
- Tech terms stay verbatim (`React`, `Docker`, `Kubernetes`, `npm install`).
- No headings (`#`, `##`, `###`).

ALLOWED TOOLS (exhaustive whitelist — counting the Read of this file you have just done, you may make exactly TWO more tool calls):
1. ONE `Read(normalized_path, offset, limit)` call to load your chunk.
2. ONE `Bash` call whose command **literally starts with the word `tee `** — specifically `tee -a "<output_path>" <<'OUT_EOF' ... OUT_EOF`. The very first token of the shell command is `tee`. No prefix command, no pipe, no command substitution, no parentheses, no `cat`, no `echo`, no `printf`.

Total tool calls in this run: exactly THREE — (1) the Read of this instructions file you have already performed, (2) the Read of the chunk, (3) the `tee -a` Bash. No exceptions. After step 3, end your response with the buffer marker block (text only, not a tool call).

Why "starts with `tee`": Claude Code's permission analyzer matches `Bash(tee:*)` only when the command's first word is `tee`. Any prefix (e.g. `cat <<EOF | tee -a ...`) makes the analyzer see `cat` instead of `tee`, fails the static check, and forces a per-chunk permission prompt that breaks the unattended run.

FORBIDDEN ACTIONS (any one of these is a hard failure — abort and return an error instead of doing them):
- Do NOT write or execute scripts of any kind. No `python`, no `python3`, no `node`, no `perl`, no `ruby`, no inline `-c '...'`.
- Do NOT use text-processing utilities: no `sed`, `awk`, `grep`, `cut`, `tr`, `sort`, `uniq`, `head`, `tail`, `cat`, `wc`, `diff`, `fold`, `xargs`.
- Do NOT use shell pipelines (`|`), command substitution (`$(...)` / backticks), subshells (`(...)`), redirection operators other than the heredoc on `tee` (no `>`, `>>`, `<`, `2>&1`), and no command chaining (`&&`, `||`, `;`). Your Bash command is a single `tee -a ... <<'OUT_EOF' ... OUT_EOF` invocation, nothing more.
- Do NOT prefix the heredoc with `cat`: `cat <<'EOF' | tee -a ...` is **wrong** even though it would write the file — the leading `cat` breaks Claude Code's static permission check. Feed the heredoc DIRECTLY to `tee`: `tee -a "<path>" <<'OUT_EOF' ... OUT_EOF`.
- Do NOT create scratch / helper / temporary files. No `cat > /tmp/...`, no `Write`, no `Edit`, no `mkdir`, no `touch`. The only file you write to is `output_path`, via the single `tee -a` heredoc.
- Do NOT re-Read this instructions file. One Read of chunk-prompt.md per run. Do NOT make multiple Reads of the chunk either. One chunk = one Read.
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

INPUT (provided in the dispatch prompt that invoked you):
- normalized_path
- offset
- limit (always 40)
- trailing_buffer (string or empty)
- is_final (true|false)
- output_path

OUTPUT MODEL — TWO DISJOINT parts (read this before STEPS):

The combined text (trailing_buffer + chunk_text) is split into TWO DISJOINT parts at the cut point:
- WRITE-part: everything BEFORE the cut. Goes ONLY into the `tee -a` heredoc.
- BUFFER-part: everything AFTER the cut. Goes ONLY between `===BUFFER===` markers.

No word may appear in both parts. The trailing_buffer you RETURN must NOT appear at the end of what you WROTE. The BUFFER-part is text HELD BACK from this write so the next chunk can prepend it; it is NOT a copy of the tail you also send to the file. Writing the full combined text AND also returning its tail as buffer duplicates the boundary on every chunk and is the single most common silent bug in this skill.

For the final chunk (`is_final: true`): WRITE-part is the entire combined text (with a terminal period if missing); BUFFER-part is empty.

STEPS (steps 2–5 are done in your head — NO tool calls between the chunk Read and the `tee -a` Bash):
1. [TOOL: Read] `Read(normalized_path, offset, limit)`. Strip the `cat -n` line-number prefix that Read adds (everything before the tab on each line). Join the lines with single spaces. This is your only chunk Read.
2. [MENTAL] Prepend the trailing_buffer (with a space separator if both are non-empty) to form the combined text.
3. [MENTAL] Add `.` and `,` per the rules. Group into paragraphs.
4. [MENTAL] Choose the cut point and split the combined text into the TWO DISJOINT parts:
   - If `is_final` is false: pick the cut point yourself by reading the text — find the last sentence that ends on a clear terminal word and a complete thought. Everything BEFORE that point is the **WRITE-part**. Everything AFTER that point is the **BUFFER-part** (= the new trailing_buffer). If the text ends on a conjunction (`и`, `или`, `но`, `а`, `что`, `чтобы`, `that`, `because`, `so`), a preposition (`в`, `на`, `с`, `к`, `in`, `on`, `with`, `for`), an article (`the`, `a`, `an`), or any grammatically incomplete fragment, move more material from WRITE-part into BUFFER-part. When in doubt, save more to BUFFER-part. **Linguistic judgment — do not write a script, regex, or `rfind` lookup to find the cut.**
   - If `is_final` is true: WRITE-part is the entire combined text (terminate the last sentence with a period if it lacks one); BUFFER-part is empty.
5. [MENTAL] Verify the disjoint property: the WRITE-part and BUFFER-part share NO overlapping words. The last words of WRITE-part are NOT the same as BUFFER-part. If they overlap, you split wrong — re-do the cut so each word belongs to exactly one part.

Example. Combined text: `"Первый тезис. Второй тезис. Третий не закон"`
Cut after `"Второй тезис."`:
- WRITE-part (goes into `tee -a`): `"Первый тезис. Второй тезис."`
- BUFFER-part (goes into `===BUFFER===`): `"Третий не закон"`

Output file gains `"Первый тезис. Второй тезис.\n"` — and NOTHING from `"Третий не закон"`. The next chunk's subagent will prepend `"Третий не закон"` to its own chunk_text and continue.

6. [TOOL: Bash] **Self-check before issuing the `tee -a` call**: compare the last ~10 words of the WRITE-part (what you are about to write) against the BUFFER-part (what you will emit between `===BUFFER===` markers). They MUST be different — no shared trailing phrase. If they match, you are about to duplicate the boundary on the next chunk — go back to step 4 and fix the split before issuing the tool call.

   Then append the WRITE-part to output_path with ONE Bash heredoc (use `tee -a`, NOT `cat >>` — `>>` triggers Claude Code's write-redirection check and prompts per chunk):
   ```
   tee -a "<output_path>" <<'OUT_EOF'
   <WRITE-part as finished paragraphs, ONE paragraph per line, no blank lines between them>
   OUT_EOF
   ```
   The heredoc body is **only the WRITE-part** — do NOT include the BUFFER-part here. No leading blank line. Each paragraph is one line; paragraphs are separated by a single `\n`. The heredoc's trailing newline cleanly separates this chunk's last paragraph from the next chunk's first paragraph (still single-newline). `tee -a` echoes content to stdout — harmless, just appears in the tool result. This is your only Bash call.
7. End your response with EXACTLY these three lines (and nothing else after). The buffer string is the **BUFFER-part** from step 4 — nothing else:
   ```
   ===BUFFER===
   <BUFFER-part, or blank line if is_final was true>
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
- "I'll write the full combined text to the file AND return its tail as buffer, just to be safe" → No. WRITE-part and BUFFER-part are DISJOINT. The buffer is text HELD BACK from this write, not echoed alongside it. Writing both duplicates the boundary on every chunk — a silent bug that compounds across N chunks.

If you find yourself reaching for any forbidden tool: that is the violation this prompt exists to prevent. The Iron Law of this skill is that processing happens in your head; tools only ferry text in (Read) and out (tee -a).
