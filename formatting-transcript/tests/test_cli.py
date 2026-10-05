"""Интеграционные тесты format_transcript через subprocess с фейковым claude -p.

Фейк (fake_claude.py) копируется в tmp-каталог и ставится первым в PATH —
скрипт находит его как настоящий бинарник. Вход — синтетический текст из
1000 слов с точкой каждые 10 слов: при CHUNK_WORDS = 300 это 3 чанка.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "format_transcript.py"

WORDS = [f"w{i}" + ("." if i % 10 == 9 else "") for i in range(1000)]
FAIL_WORD = "w650"  # попадает в третий чанк


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)

        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "claude"
        shutil.copy2(HERE / "fake_claude.py", fake)
        fake.chmod(0o755)
        self.log = tmp / "calls.log"
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_CLAUDE_LOG": str(self.log),
        }
        self.env.pop("FAKE_CLAUDE_FAIL_MARK", None)

        self.src = tmp / "src.txt"
        self.src.write_text(" ".join(WORDS), encoding="utf-8")
        self.out = tmp / "out.md"
        self.parts = tmp / "out.md.parts"

    def run_script(self, *args: str, fail_mark: str | None = None) -> subprocess.CompletedProcess:
        env = dict(self.env)
        if fail_mark:
            env["FAKE_CLAUDE_FAIL_MARK"] = fail_mark
        return subprocess.run(
            ["python3", str(SCRIPT), str(self.src), str(self.out), *args],
            capture_output=True, text=True, env=env,
            stdin=subprocess.DEVNULL, timeout=60,
        )

    def calls(self) -> int:
        n = len(self.log.read_text().splitlines()) if self.log.exists() else 0
        self.log.unlink(missing_ok=True)
        return n

    def test_full_run(self):
        proc = self.run_script()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), str(self.out))
        self.assertEqual(self.out.read_text().split(), WORDS)
        self.assertEqual(self.calls(), 3)
        self.assertFalse(self.parts.exists())

    def test_parallel_run_matches_sequential(self):
        seq = self.run_script("--workers", "1")
        self.assertEqual(seq.returncode, 0, seq.stderr)
        seq_text = self.out.read_text()
        self.out.unlink()
        par = self.run_script("--workers", "3")
        self.assertEqual(par.returncode, 0, par.stderr)
        self.assertEqual(self.out.read_text(), seq_text)

    def test_refuses_to_overwrite(self):
        self.out.write_text("old")
        proc = self.run_script()
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(self.out.read_text(), "old")
        self.assertEqual(self.calls(), 0)

    def test_usage_limit_keeps_parts_and_resume_reuses_them(self):
        failed = self.run_script("--workers", "1", fail_mark=FAIL_WORD)
        self.assertEqual(failed.returncode, 1)
        self.assertIn("429", failed.stderr)
        self.assertFalse(self.out.exists())
        self.assertEqual(sorted(p.name for p in self.parts.iterdir()), ["0001.json", "0002.json"])
        self.calls()

        resumed = self.run_script("--workers", "1")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(self.calls(), 1)  # только упавший чанк
        self.assertIn("взято чанков: 2 из 3", resumed.stderr)
        self.assertEqual(self.out.read_text().split(), WORDS)
        self.assertFalse(self.parts.exists())

    def test_parts_not_reused_with_other_model(self):
        failed = self.run_script("--workers", "1", fail_mark=FAIL_WORD)
        self.assertEqual(failed.returncode, 1)
        self.calls()

        rerun = self.run_script("--workers", "1", "--model", "sonnet")
        self.assertEqual(rerun.returncode, 0, rerun.stderr)
        self.assertEqual(self.calls(), 3)
        self.assertNotIn("взято чанков", rerun.stderr)

    def test_corrupt_part_is_recomputed(self):
        failed = self.run_script("--workers", "1", fail_mark=FAIL_WORD)
        self.assertEqual(failed.returncode, 1)
        (self.parts / "0001.json").write_text("{not json")
        self.calls()

        resumed = self.run_script("--workers", "1")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(self.calls(), 2)
        self.assertEqual(self.out.read_text().split(), WORDS)


if __name__ == "__main__":
    unittest.main()
