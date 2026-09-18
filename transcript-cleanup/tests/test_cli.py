"""Интеграционные тесты cleanup_transcript через subprocess с фейковым claude -p.

Фейк (fake_claude.py) копируется в tmp-каталог и ставится первым в PATH —
скрипт находит его как настоящий бинарник. stdin=DEVNULL имитирует не-tty.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "cleanup_transcript.py"
FIXTURE_TRANSCRIPT = HERE / "fixtures" / "meeting.txt"
FIXTURE_CONTEXT = HERE / "fixtures" / "project-context.md"

LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] \[([^\]]+)\] (.*)$")


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
        self.env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

        self.src = tmp / "meeting.txt"
        shutil.copy2(FIXTURE_TRANSCRIPT, self.src)
        self.ctx = tmp / "context.md"
        shutil.copy2(FIXTURE_CONTEXT, self.ctx)
        self.out = tmp / "meeting.clean.txt"

    def run_script(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["python3", str(SCRIPT), *args],
            capture_output=True, text=True, env=self.env,
            stdin=subprocess.DEVNULL, timeout=60,
        )

    def test_non_tty_without_yes_refuses(self):
        proc = self.run_script(str(self.src), str(self.ctx), "-o", str(self.out))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--yes", proc.stderr)

    def test_full_run_with_yes(self):
        proc = self.run_script(str(self.src), str(self.ctx),
                               "-o", str(self.out), "--yes")
        self.assertEqual(proc.returncode, 0, proc.stderr)

        src_reps = parse(FIXTURE_TRANSCRIPT.read_text())
        out_reps = parse(self.out.read_text())

        # Каркас: число реплик, таймкоды по порядку (критерии приёмки 3, 5)
        self.assertEqual(len(out_reps), len(src_reps))
        self.assertEqual([r[0] for r in out_reps], [r[0] for r in src_reps])

        # Подстановка: SPEAKER_00 → Анна, SPEAKER_01 → Игорь, UNKNOWN остался
        labels = [r[1] for r in out_reps]
        self.assertEqual(set(labels),
                         {"Анна Соколова", "Игорь Левин", "UNKNOWN"})
        for src_r, out_r in zip(src_reps, out_reps):
            if src_r[1] == "SPEAKER_00":
                self.assertEqual(out_r[1], "Анна Соколова")
            elif src_r[1] == "SPEAKER_01":
                self.assertEqual(out_r[1], "Игорь Левин")
            else:
                self.assertEqual(out_r[1], "UNKNOWN")

        # Согласованность имён: одна метка — одно имя по всему выходу (критерий 2)
        m = {}
        for src_r, out_r in zip(src_reps, out_reps):
            self.assertEqual(m.setdefault(src_r[1], out_r[1]), out_r[1])

        # Глоссарные замены (критерий 4) и сохранность спорной пары
        text = "\n".join(r[2] for r in out_reps)
        self.assertIn("Kubernetes", text)
        self.assertNotIn("кувер", text)
        self.assertIn("GeoJSON", text)
        self.assertNotIn("геоджасонки", text)
        self.assertIn("Горелый", text)  # спорная пара — не тронута
        self.assertIn("Sentinel-2", text)

    def test_refuses_overwrite(self):
        self.out.write_text("старый выход\n")
        proc = self.run_script(str(self.src), str(self.ctx),
                               "-o", str(self.out), "--yes")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("не перезаписываю", proc.stderr)
        self.assertEqual(self.out.read_text(), "старый выход\n")

    def test_usage_error(self):
        proc = self.run_script()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage", proc.stderr)


def parse(text: str):
    return [LINE_RE.match(l).groups()
            for l in text.splitlines() if l.startswith("[")]


if __name__ == "__main__":
    unittest.main()
