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

    def run_script(self, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["python3", str(SCRIPT), *args],
            capture_output=True, text=True, env={**self.env, **(env_extra or {})},
            stdin=subprocess.DEVNULL, timeout=60,
        )

    def test_non_tty_without_yes_refuses(self):
        proc = self.run_script(str(self.src), str(self.ctx), "-o", str(self.out))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--yes", proc.stderr)

    def test_parallel_run_matches_sequential(self):
        """Пул не меняет результат: фейк детерминированный, выход обязан совпасть."""
        seq_out = Path(self.tmp.name) / "seq.txt"
        par_out = Path(self.tmp.name) / "par.txt"
        seq = self.run_script(str(self.src), str(self.ctx),
                              "-o", str(seq_out), "--yes", "--workers", "1")
        self.assertEqual(seq.returncode, 0, seq.stderr)
        par = self.run_script(str(self.src), str(self.ctx),
                              "-o", str(par_out), "--yes", "--workers", "3")
        self.assertEqual(par.returncode, 0, par.stderr)
        self.assertEqual(seq_out.read_text(), par_out.read_text())
        self.assertIn("воркеров: 3", par.stderr)

    def test_bad_workers_value_refused(self):
        proc = self.run_script(str(self.src), str(self.ctx), "-o", str(self.out),
                               "--yes", "--workers", "0")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--workers", proc.stderr)

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

    def test_long_replica_split_into_paragraphs(self):
        """Фаза 3: длинная реплика выходит абзацами через \\n, текст при этом
        тот же; короткие реплики остаются однострочными."""
        long_text = " ".join(
            f"Предложение номер {i} достаточно длинное, чтобы набрать объём текста." for i in range(1, 21)
        )
        self.assertGreaterEqual(len(long_text), 1000)
        with self.src.open("a", encoding="utf-8") as f:
            f.write(f"\n[00:09:00] [SPEAKER_00] {long_text}\n")
        proc = self.run_script(str(self.src), str(self.ctx),
                               "-o", str(self.out), "--yes")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("фаза 3: 1 реплик", proc.stderr)

        blocks = self.out.read_text().rstrip("\n").split("\n\n")
        last = blocks[-1].splitlines()
        self.assertTrue(last[0].startswith("[00:09:00] [Анна Соколова] Предложение номер 1 "))
        self.assertEqual(len(last), 7)  # 20 предложений, абзац с 4, 7, …, 19
        self.assertTrue(last[1].startswith("Предложение номер 4 "))
        body = " ".join([last[0].split("] ", 2)[2], *last[1:]])
        self.assertEqual(body, long_text)
        self.assertTrue(all(len(b.splitlines()) == 1 for b in blocks[:-1]))

    def add_long_replicas(self):
        """Два длинных хвоста к фикстуре: вход набирает 2 чанка фазы 2,
        маркер МАРКЕР попадает только во второй."""
        filler = "Достаточно длинное предложение для объёма текста. " * 60
        with self.src.open("a", encoding="utf-8") as f:
            f.write(f"\n[00:09:00] [SPEAKER_00] {filler.strip()}\n")
            f.write(f"\n[00:10:00] [SPEAKER_01] МАРКЕР. {filler.strip()}\n")

    def calls(self, log: Path) -> list[str]:
        out = log.read_text().splitlines() if log.exists() else []
        log.unlink(missing_ok=True)
        return out

    def test_resume_after_phase2_failure_skips_done_work(self):
        self.add_long_replicas()
        log = Path(self.tmp.name) / "calls.log"
        self.env["FAKE_CLAUDE_LOG"] = str(log)
        parts = self.out.with_name(self.out.name + ".parts")

        failed = self.run_script(str(self.src), str(self.ctx), "-o", str(self.out),
                                 "--yes", "--workers", "1",
                                 env_extra={"FAKE_CLAUDE_FAIL_MARK": "МАРКЕР"})
        self.assertEqual(failed.returncode, 1)
        self.assertFalse(self.out.exists())
        self.assertIn("перезапуск той же командой", failed.stderr)
        self.assertEqual(sorted(p.name for p in parts.iterdir()),
                         ["clean-0001.json", "map.json"])
        self.assertEqual(self.calls(log), ["map", "clean", "clean", "clean", "clean"])

        resumed = self.run_script(str(self.src), str(self.ctx), "-o", str(self.out),
                                  "--yes", "--workers", "1")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        calls = self.calls(log)
        self.assertNotIn("map", calls)
        self.assertEqual(calls.count("clean"), 1)  # только упавший чанк
        self.assertIn("карта взята из прошлого запуска", resumed.stderr)
        self.assertIn("фаза 2: из прошлых запусков взято чанков: 1 из 2", resumed.stderr)
        self.assertFalse(parts.exists())
        # догадки из сохранённого чанка не потерялись
        self.assertIn("'сентинел-данные' → 'Sentinel-2-данные'", resumed.stderr)

        fresh = Path(self.tmp.name) / "fresh.txt"
        full = self.run_script(str(self.src), str(self.ctx), "-o", str(fresh), "--yes")
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertEqual(self.out.read_text(), fresh.read_text())

    def test_changed_context_file_invalidates_saved_work(self):
        self.add_long_replicas()
        log = Path(self.tmp.name) / "calls.log"
        self.env["FAKE_CLAUDE_LOG"] = str(log)
        failed = self.run_script(str(self.src), str(self.ctx), "-o", str(self.out),
                                 "--yes", "--workers", "1",
                                 env_extra={"FAKE_CLAUDE_FAIL_MARK": "МАРКЕР"})
        self.assertEqual(failed.returncode, 1)
        self.calls(log)

        with self.ctx.open("a", encoding="utf-8") as f:
            f.write("\nНовая строка глоссария.\n")
        rerun = self.run_script(str(self.src), str(self.ctx), "-o", str(self.out),
                                "--yes", "--workers", "1")
        self.assertEqual(rerun.returncode, 0, rerun.stderr)
        calls = self.calls(log)
        self.assertEqual(calls.count("map"), 1)
        self.assertEqual(calls.count("clean"), 2)
        self.assertNotIn("из прошлых запусков", rerun.stderr)

    def test_guess_log_reaches_stderr(self):
        """Догадка мока проходит весь путь: ответ модели → журнал → stderr, и
        применяется к тексту, хотя пары в контекст-файле для неё нет."""
        proc = self.run_script(str(self.src), str(self.ctx),
                               "-o", str(self.out), "--yes")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("догадки без пары в контекст-файле", proc.stderr)
        self.assertIn("'сентинел-данные' → 'Sentinel-2-данные'", proc.stderr)
        self.assertIn("название спутника", proc.stderr)
        text = self.out.read_text()
        self.assertIn("Sentinel-2-данные", text)
        self.assertNotIn("сентинел-данные", text)

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
