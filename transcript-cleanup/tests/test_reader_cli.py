"""Интеграционные тесты reader_pdf через subprocess с поддельными claude и Chrome.

Подделки копируются в tmp/bin, PATH сужается до этого каталога: настоящий
Chrome скрипт не найдёт, а поддельный — найдёт, если тест его положил. python3
ставится туда же ссылкой, иначе шебанг подделок не сработает.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "reader_pdf.py"
FIXTURE_TRANSCRIPT = HERE / "fixtures" / "podcast.clean.txt"
FIXTURE_CHAPTERS = HERE / "fixtures" / "podcast.chapters.txt"

sys.path.insert(0, str(HERE))
from test_reader_units import _TextCollector  # noqa: E402

sys.path.insert(0, str(HERE.parent))
import reader_pdf as rp  # noqa: E402


class ReaderCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        self.bin = tmp / "bin"
        self.bin.mkdir()
        (self.bin / "python3").symlink_to(sys.executable)
        fake = self.bin / "claude"
        shutil.copy2(HERE / "fake_claude.py", fake)
        fake.chmod(0o755)
        self.log = tmp / "calls.log"
        self.env = {**os.environ, "PATH": str(self.bin), "FAKE_CLAUDE_LOG": str(self.log)}

        self.src = tmp / "podcast.clean.txt"
        shutil.copy2(FIXTURE_TRANSCRIPT, self.src)
        self.chapters = tmp / "chapters.txt"
        shutil.copy2(FIXTURE_CHAPTERS, self.chapters)
        self.pdf = tmp / "podcast.clean.pdf"
        self.html = tmp / "podcast.clean.html"
        self.parts = tmp / "podcast.clean.pdf.parts"

    def add_fake_chrome(self):
        chrome = self.bin / "google-chrome"
        shutil.copy2(HERE / "fake_chrome.py", chrome)
        chrome.chmod(0o755)

    def run_script(self, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True, text=True, env={**self.env, **(env_extra or {})},
            stdin=subprocess.DEVNULL, timeout=60,
        )

    def calls(self) -> list[str]:
        return self.log.read_text().split() if self.log.exists() else []

    def parsed_html(self) -> _TextCollector:
        p = _TextCollector()
        p.feed(self.html.read_text(encoding="utf-8"))
        return p

    def test_chapters_to_pdf(self):
        self.add_fake_chrome()
        proc = self.run_script(str(self.src), "--chapters", str(self.chapters))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), str(self.pdf))
        self.assertTrue(self.pdf.read_text().startswith("%PDF"))
        self.assertFalse(self.parts.exists())
        self.assertEqual(self.calls(), ["anchor"])  # одна глава внутри реплики

        parsed = self.parsed_html()
        self.assertEqual(parsed.toc, ["Интро", "Подготовка почвы", "Компост и мульча",
                                      "Полив", "Финал"])
        # Текст реплик тот же, абзац в абзац (критерий 2)
        reps = rp.parse_transcript(FIXTURE_TRANSCRIPT.read_text())
        self.assertEqual(parsed.paras, [p for r in reps for p in rp.paragraphs(r)])

        # «Компост и мульча» — перед третьим абзацем длинной реплики, а не перед
        # вторым, куда указывает интерполяция (критерий 1)
        html = self.html.read_text(encoding="utf-8")
        long = rp.paragraphs(reps[3])
        h = html.index("Компост и мульча</h2>")
        self.assertLess(html.index(long[1]), h)
        self.assertLess(h, html.index(long[2]))
        # «Полив» за 2 с до реплики — перед репликой, а не внутри предыдущей
        self.assertLess(html.index("Полив</h2>"), html.index("А как насчёт полива?"))
        self.assertLess(html.index(long[4]), html.index("Полив</h2>"))

    def test_without_chapters_model_sections(self):
        self.add_fake_chrome()
        proc = self.run_script(str(self.src))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.calls(), ["sections"])
        parsed = self.parsed_html()
        self.assertEqual(parsed.toc, ["Раздел 1", "Раздел 2", "Раздел 3"])
        self.assertIn("предложены моделью", self.html.read_text(encoding="utf-8"))

    def test_bad_sections_retried(self):
        self.add_fake_chrome()
        proc = self.run_script(str(self.src), env_extra={"FAKE_CLAUDE_BAD_SECTIONS": "1"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.calls(), ["sections", "sections"])
        self.assertEqual(self.parsed_html().toc, ["Раздел 1", "Раздел 2", "Раздел 3"])

    def test_resume_after_print_failure_does_not_call_model(self):
        self.add_fake_chrome()
        failed = self.run_script(str(self.src), "--chapters", str(self.chapters),
                                 env_extra={"FAKE_CHROME_FAIL": "1"})
        self.assertEqual(failed.returncode, 1)
        self.assertIn("PDF не собран", failed.stderr)
        self.assertFalse(self.pdf.exists())
        self.assertTrue(self.parts.is_dir())
        self.assertEqual(self.calls(), ["anchor"])

        again = self.run_script(str(self.src), "--chapters", str(self.chapters))
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(self.calls(), ["anchor"])  # второй прогон модель не звал
        self.assertTrue(self.pdf.exists())
        self.assertFalse(self.parts.exists())

    def test_no_browser_writes_html_and_warns(self):
        proc = self.run_script(str(self.src), "--chapters", str(self.chapters))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PDF не собран", proc.stderr)
        self.assertEqual(proc.stdout.strip(), str(self.html))
        self.assertTrue(self.html.exists())
        self.assertFalse(self.pdf.exists())

    def test_bad_chapters_rc2(self):
        self.chapters.write_text("00:05:00 Б\n00:01:00 А\n", encoding="utf-8")
        proc = self.run_script(str(self.src), "--chapters", str(self.chapters))
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_existing_pdf_not_overwritten(self):
        self.pdf.write_text("old")
        proc = self.run_script(str(self.src), "--chapters", str(self.chapters))
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(self.pdf.read_text(), "old")

    def test_title_option(self):
        proc = self.run_script(str(self.src), "--chapters", str(self.chapters),
                               "--title", "Грядка, выпуск 1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("<title>Грядка, выпуск 1</title>", self.html.read_text(encoding="utf-8"))


@unittest.skipUnless(shutil.which("google-chrome"), "google-chrome не установлен")
class RealChromeTests(unittest.TestCase):
    def test_prints_real_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            html = Path(tmp) / "doc.html"
            pdf = Path(tmp) / "doc.pdf"
            reps = rp.parse_transcript(FIXTURE_TRANSCRIPT.read_text())
            html.write_text(rp.render_html("Грядка", reps, [rp.Section(0, 0, "Интро", "00:00:00")],
                                           "по главам"), encoding="utf-8")
            rp.print_pdf(shutil.which("google-chrome"), html, pdf)
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
