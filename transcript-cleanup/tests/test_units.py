"""Юнит-тесты чистых функций cleanup_transcript (без вызовов claude -p)."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cleanup_transcript as ct  # noqa: E402


class ParseTranscriptTests(unittest.TestCase):
    def test_basic_and_unknown_label(self):
        src = (
            "[00:00:06] [SPEAKER_00] Первая реплика.\n"
            "\n"
            "[00:01:12] [UNKNOWN] Вторая, без спикера.\n"
        )
        reps = ct.parse_transcript(src)
        self.assertEqual(len(reps), 2)
        self.assertEqual((reps[0].time, reps[0].label, reps[0].text),
                         ("00:00:06", "SPEAKER_00", "Первая реплика."))
        self.assertEqual(reps[1].label, "UNKNOWN")

    def test_continuation_line_glues_to_previous(self):
        src = "[00:00:06] [SPEAKER_00] начало\nпродолжение той же реплики\n"
        reps = ct.parse_transcript(src)
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0].text, "начало продолжение той же реплики")

    def test_garbage_before_first_replica_raises(self):
        with self.assertRaises(ValueError):
            ct.parse_transcript("просто текст без каркаса\n[00:00:06] [SPEAKER_00] ок\n")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ct.parse_transcript("\n\n")


class SkeletonTests(unittest.TestCase):
    def test_head_cut_at_30_words_with_ellipsis(self):
        r = ct.Replica("00:00:01", "SPEAKER_00", " ".join(f"w{i}" for i in range(50)))
        sk = ct.skeletonize([r])
        self.assertEqual(sk[0]["n"], 1)
        self.assertTrue(sk[0]["head"].endswith("w29…"))
        self.assertEqual(len(sk[0]["head"].split()), 30)

    def test_short_replica_kept_whole(self):
        r = ct.Replica("00:00:01", "SPEAKER_00", "один два три")
        sk = ct.skeletonize([r])
        self.assertEqual(sk[0]["head"], "один два три")
        self.assertFalse(sk[0]["head"].endswith("…"))

    def test_numbers_are_sequential(self):
        reps = [ct.Replica("00:00:01", "SPEAKER_00", "текст") for _ in range(3)]
        sk = ct.skeletonize(reps)
        self.assertEqual([row["n"] for row in sk], [1, 2, 3])


class ChunkTests(unittest.TestCase):
    @staticmethod
    def _reps(lengths):
        return [ct.Replica("00:00:01", "SPEAKER_00", "x" * n) for n in lengths]

    def test_replica_is_atom(self):
        reps = self._reps([10] * 45)
        chunks = ct.chunk_replicas(reps, target=100)
        self.assertEqual(sum(chunks, []), reps)
        self.assertTrue(all(chunks))

    def test_equal_chars_variable_replicas(self):
        # один монолог против россыпи коротких: по символам поровну
        reps = self._reps([1000] + [100] * 10)
        chunks = ct.chunk_replicas(reps, target=1000)
        self.assertEqual([len(c) for c in chunks], [1, 10])

    def test_no_tiny_tail(self):
        # жадный набор дал бы 1000+1000+100; тут три чанка ~700
        reps = self._reps([100] * 21)
        chunks = ct.chunk_replicas(reps, target=1000)
        self.assertEqual([len(c) for c in chunks], [7, 7, 7])

    def test_oversized_replica_alone(self):
        reps = self._reps([100, 5000, 100])
        chunks = ct.chunk_replicas(reps, target=1000)
        self.assertEqual(sum(chunks, []), reps)
        self.assertTrue(all(chunks))
        self.assertIn([reps[1]], chunks)

    def test_small_input_single_chunk(self):
        reps = self._reps([10, 20, 30])
        self.assertEqual(ct.chunk_replicas(reps, target=1000), [reps])

    def test_empty(self):
        self.assertEqual(ct.chunk_replicas([]), [])

    def test_context_lines_full(self):
        reps = [ct.Replica(f"00:00:{i:02d}", "SPEAKER_00", f"r{i}") for i in range(5)]
        ctx = ct.context_lines(reps, upto=4)
        self.assertEqual(ctx.splitlines(), [reps[1].line(), reps[2].line(), reps[3].line()])

    def test_context_lines_at_start(self):
        reps = [ct.Replica("00:00:01", "SPEAKER_00", "r0")]
        self.assertEqual(ct.context_lines(reps, upto=0), "")


class ParagraphTests(unittest.TestCase):
    TEXT = "Привет. Это т. е. тест! «Цитата». 5 штук. и строчная. Конец?"

    def test_split_sentences(self):
        self.assertEqual(
            ct.split_sentences(self.TEXT),
            ["Привет.", "Это т. е. тест!", "«Цитата».", "5 штук. и строчная.", "Конец?"],
        )

    def test_apply_paragraphs_only_inserts_newlines(self):
        out = ct.apply_paragraphs(self.TEXT, [3, 5])
        self.assertEqual(out, "Привет. Это т. е. тест!\n«Цитата». 5 штук. и строчная.\nКонец?")
        self.assertEqual(out.replace("\n", " "), self.TEXT)

    def test_apply_paragraphs_ignores_first_and_out_of_range(self):
        self.assertEqual(ct.apply_paragraphs(self.TEXT, [1, 99]), self.TEXT)
        self.assertEqual(ct.apply_paragraphs(self.TEXT, []), self.TEXT)

    def _chunk(self):
        return [ct.Replica("00:00:01", "A", self.TEXT), ct.Replica("00:00:02", "B", "Раз. Два.")]

    def test_paragraph_chunk_retries_on_bad_index(self):
        answers = [
            {"paragraphs": [{"n": 1, "starts": [9]}, {"n": 2, "starts": []}]},
            {"paragraphs": [{"n": 2, "starts": [2]}, {"n": 1, "starts": [5, 3, 1]}]},
        ]
        payloads = []

        def fake(system_prompt, user_msg, model, schema, timeout_s, budget_usd):
            payloads.append(json.loads(user_msg))
            return answers.pop(0)

        with mock.patch.object(ct, "call_claude", fake), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            starts = ct.paragraph_chunk("p", self._chunk(), "абзацы 1/1")
        self.assertEqual(starts, [[3, 5], [2]])
        self.assertEqual(len(payloads), 2)
        self.assertIn("retry_notice", payloads[1])
        self.assertEqual(payloads[0]["replicas"][1]["sentences"],
                         [{"i": 1, "text": "Раз."}, {"i": 2, "text": "Два."}])

    def test_paragraph_chunk_missing_replica_is_error(self):
        def fake(system_prompt, user_msg, model, schema, timeout_s, budget_usd):
            return {"paragraphs": [{"n": 1, "starts": []}]}

        with mock.patch.object(ct, "call_claude", fake), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(RuntimeError):
                ct.paragraph_chunk("p", self._chunk(), "абзацы 1/1")

    def test_failed_chunk_leaves_replicas_intact(self):
        """Фаза косметическая: исчерпанные ретраи не валят прогон."""
        long_text = "Длинное предложение для проверки абзацев. " * 30
        reps = [ct.Replica("00:00:01", "A", long_text.strip()),
                ct.Replica("00:00:02", "B", "Коротко.")]

        def fake(system_prompt, user_msg, model, schema, timeout_s, budget_usd):
            raise RuntimeError("сбой вызова")

        with mock.patch.object(ct, "call_claude", fake), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            out = ct.paragraph_long_replicas("p", reps, workers=1)
        self.assertEqual(out, reps)
        self.assertIn("остаются без абзацев", err.getvalue())

    def test_short_replicas_skip_model(self):
        reps = [ct.Replica("00:00:01", "A", "Коротко. Совсем.")]

        def fake(*a):
            raise AssertionError("модель не должна вызываться")

        with mock.patch.object(ct, "call_claude", fake), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(ct.paragraph_long_replicas("p", reps), reps)


class AddressFormsTests(unittest.TestCase):
    CTX = """
## Спикеры

Наша сторона:

| Обращение в речи | Имя для транскрипта |
|---|---|
| Аня, Анна | Анна Соколова |
| Игорь | Игорь Левин |
| Стеблов | как есть (фамилия) |

## Глоссарий

| Пара | Куда |
|---|---|
| егерн | ЕГРН |
"""

    def test_extracts_forms_comma_split(self):
        forms = ct.extract_address_forms(self.CTX)
        self.assertEqual(forms, ["Аня", "Анна", "Игорь", "Стеблов"])

    def test_glossary_tables_ignored(self):
        self.assertNotIn("егерн", ct.extract_address_forms(self.CTX))

    def test_empty_context(self):
        self.assertEqual(ct.extract_address_forms(""), [])


class AddressHitTests(unittest.TestCase):
    def test_stem_matches_oblique_forms(self):
        forms = ["Дима", "Игорь"]
        self.assertTrue(ct.has_address_hit("Давай, Димер, начали.", forms))
        self.assertTrue(ct.has_address_hit("спросим Игоря про базу", forms))

    def test_no_false_prefix_hit_without_left_boundary(self):
        # «недимология» не матчится по «Дима»: слева стоит буква
        self.assertFalse(ct.has_address_hit("недимология", ["Дима"]))

    def test_plain_miss(self):
        self.assertFalse(ct.has_address_hit("просто текст без имён", ["Олег"]))


class SelectSkeletonTests(unittest.TestCase):
    def _reps(self):
        texts = [
            "Привет всем, начинаем.",            # 0: head SPEAKER_00
            "Аня, перескажи про базу.",           # 1: head SPEAKER_01 + адресация
            "Да, по базе всё готово.",            # 2: следующая после адресации
            "Ещё вопрос по срокам.",              # 3: head UNKNOWN
            "Сроки в пятницу.",                   # 4: head UNKNOWN (вторая)
            "Тогда всё.",                         # 5: не сигнальная
        ]
        labels = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00",
                  "UNKNOWN", "UNKNOWN", "SPEAKER_00"]
        return [ct.Replica(f"00:00:{i:02d}", labels[i], texts[i])
                for i in range(6)]

    def test_signal_plus_next_plus_heads(self):
        idx = ct.select_skeleton_indices(self._reps(), ["Аня"])
        # 0,1 — головы спикеров, 1 — адресация, 2 — следующая, 3,4 — головы UNKNOWN
        self.assertEqual(idx, [0, 1, 2, 3, 4])

    def test_dedup_and_order(self):
        idx = ct.select_skeleton_indices(self._reps(), ["Аня", "сроки"])
        self.assertEqual(idx, sorted(set(idx)))

    def test_covers_all_speaker_labels(self):
        reps = self._reps()
        idx = ct.select_skeleton_indices(reps, ["Аня"])
        covered = {reps[i].label for i in idx}
        self.assertEqual(covered, {r.label for r in reps})

    def test_skeletonize_takes_global_nums(self):
        reps = self._reps()
        idx = [1, 3]
        sk = ct.skeletonize([reps[i] for i in idx], nums=[i + 1 for i in idx])
        self.assertEqual([row["n"] for row in sk], [2, 4])


class SubstituteTests(unittest.TestCase):
    def test_substitution(self):
        reps = [
            ct.Replica("00:00:01", "SPEAKER_00", "текст"),
            ct.Replica("00:00:02", "UNKNOWN", "текст"),
            ct.Replica("00:00:03", "SPEAKER_01", "текст"),
        ]
        out = ct.substitute_names(reps, {"SPEAKER_00": "Анна Соколова"})
        self.assertEqual([r.label for r in out],
                         ["Анна Соколова", "UNKNOWN", "SPEAKER_01"])

    def test_source_not_mutated(self):
        reps = [ct.Replica("00:00:01", "SPEAKER_00", "текст")]
        ct.substitute_names(reps, {"SPEAKER_00": "Имя"})
        self.assertEqual(reps[0].label, "SPEAKER_00")


class VerifyFinalTests(unittest.TestCase):
    def _src(self):
        return [
            ct.Replica("00:00:01", "SPEAKER_00", "текст один"),
            ct.Replica("00:00:02", "UNKNOWN", "текст два"),
        ]

    def test_ok(self):
        src = self._src()
        out = ct.substitute_names(src, {"SPEAKER_00": "Анна Соколова"})
        out = [ct.Replica(r.time, r.label, r.text + ".") for r in out]
        self.assertEqual(ct.verify_final(src, out, out), [])

    def test_count_mismatch(self):
        src = self._src()
        out = [ct.Replica("00:00:01", "SPEAKER_00", "текст один")]
        self.assertTrue(ct.verify_final(src, out, out))

    def test_timecode_mismatch(self):
        src = self._src()
        out = [ct.Replica("00:00:09", "SPEAKER_00", "текст один"),
               ct.Replica("00:00:02", "UNKNOWN", "текст два")]
        problems = ct.verify_final(src, out, out)
        self.assertTrue(any("таймкод" in p for p in problems))

    def test_label_mismatch(self):
        src = self._src()
        expected = ct.substitute_names(src, {"SPEAKER_00": "Анна Соколова"})
        out = [ct.Replica(r.time, "Чужое Имя", r.text) for r in src]
        problems = ct.verify_final(src, out, expected)
        self.assertTrue(any("метка" in p for p in problems))


class RetryTests(unittest.TestCase):
    def test_clean_chunk_retries_on_wrong_len_then_succeeds(self):
        calls = []

        def fake(system_prompt, user_msg, model, schema, timeout_s, budget_usd):
            calls.append(1)
            if len(calls) == 1:
                return {"cleaned": ["только одна"]}  # wrong length
            return {"cleaned": ["один.", "два."]}

        orig = ct.call_claude
        ct.call_claude = fake
        try:
            chunk = [ct.Replica("00:00:01", "SPEAKER_00", "один"),
                     ct.Replica("00:00:02", "SPEAKER_00", "два")]
            out, guesses = ct.clean_chunk("prompt", chunk, "", "ctx", "label")
        finally:
            ct.call_claude = orig
        self.assertEqual(out, ["один.", "два."])
        self.assertEqual(guesses, [])
        self.assertEqual(len(calls), 2)

    def test_clean_chunk_exhausts_retries(self):
        def fake(system_prompt, user_msg, model, schema, timeout_s, budget_usd):
            return {"cleaned": []}  # always wrong

        orig = ct.call_claude
        ct.call_claude = fake
        try:
            with self.assertRaises(RuntimeError):
                chunk = [ct.Replica("00:00:01", "SPEAKER_00", "один")]
                ct.clean_chunk("prompt", chunk, "", "ctx", "label")
        finally:
            ct.call_claude = orig

    def test_map_speakers_retries_on_empty_map(self):
        calls = []

        def fake(system_prompt, user_msg, model, schema, timeout_s, budget_usd):
            calls.append(1)
            if len(calls) == 1:
                return {"speakers": [{"speaker": "SPEAKER_00", "name": None,
                                      "evidence": []}]}
            return {"speakers": [{"speaker": "SPEAKER_00", "name": "Анна Соколова",
                                  "evidence": [{"time": "00:00:01",
                                                "type": "self_intro"}]}]}

        orig = ct.call_claude
        ct.call_claude = fake
        try:
            skeleton = [{"n": 1, "time": "00:00:01", "speaker": "SPEAKER_00",
                         "head": "я Анна, веду интеграцию"}]
            mapping, evidence = ct.map_speakers("prompt", skeleton, "ctx")
        finally:
            ct.call_claude = orig
        self.assertEqual(mapping, {"SPEAKER_00": "Анна Соколова"})
        # цитату скрипт подтянул из скелета по таймкоду
        self.assertEqual(evidence["SPEAKER_00"][0]["quote"],
                         "я Анна, веду интеграцию")
        self.assertEqual(len(calls), 2)

    def test_map_speakers_covers_all_labels_or_fails(self):
        def fake(system_prompt, user_msg, model, schema, timeout_s, budget_usd):
            # покрытие только одной метки из двух — все попытки, потом ошибка
            return {"speakers": [{"speaker": "SPEAKER_00", "name": "Имя",
                                  "evidence": []}]}

        orig = ct.call_claude
        ct.call_claude = fake
        try:
            skeleton = [
                {"n": 1, "time": "00:00:01", "speaker": "SPEAKER_00", "head": "a"},
                {"n": 2, "time": "00:00:02", "speaker": "SPEAKER_01", "head": "b"},
            ]
            with self.assertRaises(RuntimeError):
                ct.map_speakers("prompt", skeleton, "ctx")
        finally:
            ct.call_claude = orig


class Phase2PoolTests(unittest.TestCase):
    """Фаза 2 пулом: порядок по входу, а не по времени ответа; кап воркеров;
    падение чанка валит фазу."""

    def chunks(self, n: int, size: int = 2) -> list[list[ct.Replica]]:
        return [
            [ct.Replica(f"00:00:{i * size + j:02d}", "SPEAKER_00", f"ч{i}р{j}")
             for j in range(size)]
            for i in range(n)
        ]

    def test_order_follows_input_not_completion(self):
        chunks = self.chunks(4)
        named = [r for c in chunks for r in c]

        def fake(chunk_prompt, chunk, context, context_file, label):
            # поздние чанки отвечают быстрее — порядок завершения обратный
            idx = int(chunk[0].text[1])
            time.sleep(0.05 * (len(chunks) - idx))
            return [f"{r.text}!" for r in chunk], []

        orig = ct.clean_chunk
        ct.clean_chunk = fake
        try:
            out = ct.clean_all_chunks("prompt", named, chunks, "ctx", workers=3)
        finally:
            ct.clean_chunk = orig
        self.assertEqual(out, [[f"{r.text}!" for r in c] for c in chunks])

    def test_first_chunk_runs_alone_then_pool(self):
        chunks = self.chunks(4)
        named = [r for c in chunks for r in c]
        concurrent: list[int] = []
        live = {"n": 0}
        lock = threading.Lock()

        def fake(chunk_prompt, chunk, context, context_file, label):
            with lock:
                live["n"] += 1
                concurrent.append(live["n"])
            time.sleep(0.1)
            with lock:
                live["n"] -= 1
            return [r.text for r in chunk], []

        orig = ct.clean_chunk
        ct.clean_chunk = fake
        try:
            ct.clean_all_chunks("prompt", named, chunks, "ctx", workers=3)
        finally:
            ct.clean_chunk = orig
        self.assertEqual(concurrent[0], 1, "первый чанк должен идти один — прогрев кэша")
        self.assertLessEqual(max(concurrent), 3, "воркеров больше, чем задано")
        self.assertGreater(max(concurrent), 1, "остальные чанки должны идти параллельно")

    def test_single_worker_is_sequential(self):
        chunks = self.chunks(3)
        named = [r for c in chunks for r in c]
        live = {"n": 0}
        peak = {"n": 0}

        def fake(chunk_prompt, chunk, context, context_file, label):
            live["n"] += 1
            peak["n"] = max(peak["n"], live["n"])
            live["n"] -= 1
            return [r.text for r in chunk], []

        orig = ct.clean_chunk
        ct.clean_chunk = fake
        try:
            out = ct.clean_all_chunks("prompt", named, chunks, "ctx", workers=1)
        finally:
            ct.clean_chunk = orig
        self.assertEqual(peak["n"], 1)
        self.assertEqual(len(out), 3)

    def test_chunk_failure_fails_phase(self):
        chunks = self.chunks(4)
        named = [r for c in chunks for r in c]

        def fake(chunk_prompt, chunk, context, context_file, label):
            if chunk[0].text.startswith("ч2"):
                raise RuntimeError("исчерпаны попытки")
            return [r.text for r in chunk], []

        orig = ct.clean_chunk
        ct.clean_chunk = fake
        try:
            with self.assertRaises(RuntimeError):
                ct.clean_all_chunks("prompt", named, chunks, "ctx", workers=3)
        finally:
            ct.clean_chunk = orig

    def test_context_lines_come_from_source_not_previous_output(self):
        """Независимость чанков: контекст края — из исходника. Если бы он брался
        из выхода, параллельность была бы некорректной."""
        chunks = self.chunks(3)
        named = [r for c in chunks for r in c]
        seen: dict[str, str] = {}

        def fake(chunk_prompt, chunk, context, context_file, label):
            seen[chunk[0].text] = context
            return [f"ПЕРЕПИСАНО-{r.text}" for r in chunk], []

        orig = ct.clean_chunk
        ct.clean_chunk = fake
        try:
            ct.clean_all_chunks("prompt", named, chunks, "ctx", workers=1)
        finally:
            ct.clean_chunk = orig
        self.assertNotIn("ПЕРЕПИСАНО", seen["ч1р0"])
        self.assertIn("ч0р1", seen["ч1р0"])


class CostCounterTests(unittest.TestCase):
    def test_call_claude_accumulates_cost_from_envelope(self):
        envelope = json.dumps(
            {"is_error": False, "total_cost_usd": 0.25,
             "structured_output": {"cleaned": ["текст"]}}
        )

        before = ct.spent_usd()
        with mock.patch.object(
            ct.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0, envelope, ""),
        ):
            ct.call_claude("sys", "user", "haiku", ct.SCHEMA_CLEAN, 10, "1.00")
            ct.call_claude("sys", "user", "haiku", ct.SCHEMA_CLEAN, 10, "1.00")
        self.assertAlmostEqual(ct.spent_usd() - before, 0.50, places=6)

    def test_call_log_records_cache_metrics(self):
        envelope = json.dumps({
            "is_error": False, "total_cost_usd": 0.16,
            "usage": {"input_tokens": 6100, "cache_read_input_tokens": 16563,
                      "cache_creation_input_tokens": 17377},
            "structured_output": {"cleaned": ["текст"]},
        })
        before = len(ct.call_log())
        with mock.patch.object(
            ct.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0, envelope, ""),
        ):
            ct.call_claude("sys", "user", "haiku", ct.SCHEMA_CLEAN, 10, "1.00")
        rec = ct.call_log()[-1]
        self.assertEqual(len(ct.call_log()), before + 1)
        self.assertEqual(rec["model"], "haiku")
        self.assertEqual(rec["cache_read"], 16563)
        self.assertEqual(rec["cache_write"], 17377)
        self.assertEqual(rec["fresh"], 6100)

    def test_call_log_tolerates_envelope_without_usage(self):
        """Фейковый claude -p в CLI-тестах usage не отдаёт — нули, не падение."""
        envelope = json.dumps(
            {"is_error": False, "structured_output": {"cleaned": ["текст"]}}
        )
        with mock.patch.object(
            ct.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0, envelope, ""),
        ):
            ct.call_claude("sys", "user", "haiku", ct.SCHEMA_CLEAN, 10, "1.00")
        rec = ct.call_log()[-1]
        self.assertEqual((rec["cache_read"], rec["cache_write"], rec["fresh"]), (0, 0, 0))
        self.assertEqual(rec["usd"], 0.0)

    def test_missing_cost_field_is_tolerated(self):
        """Фейковый claude -p в CLI-тестах total_cost_usd не отдаёт."""
        envelope = json.dumps(
            {"is_error": False, "structured_output": {"cleaned": ["текст"]}}
        )
        before = ct.spent_usd()
        with mock.patch.object(
            ct.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0, envelope, ""),
        ):
            out = ct.call_claude("sys", "user", "haiku", ct.SCHEMA_CLEAN, 10, "1.00")
        self.assertEqual(out, {"cleaned": ["текст"]})
        self.assertEqual(ct.spent_usd(), before)


class GuessLogTests(unittest.TestCase):
    """Журнал догадок: замены без пары в контекст-файле, о которых отчиталась модель."""

    def chunk(self):
        return [
            ct.Replica("00:01:00", "Анна Соколова", "первая"),
            ct.Replica("00:02:00", "Игорь Левин", "вторая"),
        ]

    def test_records_with_timecode_of_replica(self):
        before = len(ct.guess_log())
        ct.record_guesses(
            [{"n": 2, "from": "Dark Haus", "to": "Darkhouse", "basis": "название платформы"}],
            self.chunk(), "чанк 1/1",
        )
        rows = ct.guess_log()
        self.assertEqual(len(rows), before + 1)
        self.assertEqual(rows[-1]["time"], "00:02:00")
        self.assertEqual(rows[-1]["from"], "Dark Haus")
        self.assertEqual(rows[-1]["to"], "Darkhouse")

    def test_skips_number_outside_chunk(self):
        """Догадка с битым n не валит уже очищенный чанк — она отчётность, не выход."""
        before = len(ct.guess_log())
        ct.record_guesses(
            [{"n": 9, "from": "a", "to": "b", "basis": "c"},
             {"n": 0, "from": "a", "to": "b", "basis": "c"}],
            self.chunk(), "чанк 1/1",
        )
        self.assertEqual(len(ct.guess_log()), before)

    def test_ignores_missing_or_malformed_field(self):
        before = len(ct.guess_log())
        ct.record_guesses(None, self.chunk(), "чанк 1/1")
        ct.record_guesses("не массив", self.chunk(), "чанк 1/1")
        ct.record_guesses(["строка вместо объекта"], self.chunk(), "чанк 1/1")
        self.assertEqual(len(ct.guess_log()), before)

    def test_print_groups_by_pair_with_counts(self):
        ct.record_guesses(
            [{"n": 1, "from": "есед", "to": "СЭД", "basis": "аббревиатура со слуха"},
             {"n": 2, "from": "есед", "to": "СЭД", "basis": "аббревиатура со слуха"},
             {"n": 1, "from": "Дарк Хаус", "to": "Darkhouse", "basis": "название платформы"}],
            self.chunk(), "чанк 1/1",
        )
        buf = io.StringIO()
        with mock.patch.object(ct.sys, "stderr", buf):
            ct.print_guess_log()
        out = buf.getvalue()
        self.assertIn("догадки без пары в контекст-файле", out)
        self.assertIn("'есед' → 'СЭД'  ×2", out)
        self.assertIn("00:01:00", out)
        self.assertIn("аббревиатура со слуха", out)

    def test_retry_does_not_duplicate_guesses(self):
        """Догадки отдаются после проверок: догадки неудачной попытки наружу не уходят."""
        chunk = self.chunk()
        guess = [{"n": 1, "from": "a", "to": "b", "basis": "c"}]
        answers = [
            {"cleaned": ["одна строка вместо двух"], "guesses": guess},
            {"cleaned": ["первая", "вторая"], "guesses": guess},
        ]
        before = len(ct.guess_log())
        with mock.patch.object(ct, "call_claude", side_effect=answers):
            out, guesses = ct.clean_chunk("prompt", chunk, "", "ctx", "label")
        self.assertEqual(out, ["первая", "вторая"])
        self.assertEqual(guesses, guess)
        self.assertEqual(len(ct.guess_log()), before)  # журнал пишет вызывающий


class WordLossTests(unittest.TestCase):
    """Потеря слов внутри реплики: каркас цел, дельта символов молчит."""

    def pair(self, src_text, out_text):
        return ([ct.Replica("00:01:00", "Анна Соколова", src_text)],
                [ct.Replica("00:01:00", "Анна Соколова", out_text)])

    def test_reports_dropped_phrase(self):
        src, out = self.pair(
            "нам от того что этот лес это участок с лесом ни жарко ни холодно надо решить",
            "нам надо решить",
        )
        rep = ct.word_losses(src, out, [])
        self.assertEqual(len(rep), 1)
        self.assertEqual(rep[0]["time"], "00:01:00")
        self.assertEqual(rep[0]["net"], 13)
        self.assertIn("жарко", rep[0]["words"])

    def test_replacement_is_not_a_loss(self):
        """Замена слова на слово даёт баланс и в отчёт не идёт."""
        src, out = self.pair("подняли сервис в кувере", "подняли сервис в Kubernetes")
        self.assertEqual(ct.word_losses(src, out, []), [])

    def test_case_and_yo_are_not_losses(self):
        src, out = self.pair("все упало и еще раз", "Всё упало и ещё раз")
        self.assertEqual(ct.word_losses(src, out, []), [])

    def test_declared_collapse_is_not_a_loss(self):
        """Свёрнутое залипание объявлено в guesses — молчим."""
        src, out = self.pair("платформа Dark Haus Dark Dark Dark Dark вроде",
                             "платформа Darkhouse вроде")
        guesses = [{"time": "00:01:00", "from": "Dark Haus Dark Dark Dark Dark",
                    "to": "Darkhouse", "basis": "залипание, 5 повторов"}]
        self.assertEqual(ct.word_losses(src, out, guesses), [])

    def test_undeclared_collapse_is_reported(self):
        """То же сворачивание без записи в guesses — в отчёт."""
        src, out = self.pair("платформа Dark Haus Dark Dark Dark Dark вроде",
                             "платформа Darkhouse вроде")
        rep = ct.word_losses(src, out, [])
        self.assertEqual(len(rep), 1)
        self.assertIn("dark", rep[0]["words"])

    def test_print_lists_timecode_and_words(self):
        buf = io.StringIO()
        with mock.patch.object(ct.sys, "stderr", buf):
            ct.print_word_losses([{"time": "00:02:00", "net": 2, "words": {"смотрите": 1, "вот": 2}}])
        out = buf.getvalue()
        self.assertIn("00:02:00", out)
        self.assertIn("'смотрите'", out)
        self.assertIn("потеря слов", out)

    def test_print_silent_when_nothing_lost(self):
        buf = io.StringIO()
        with mock.patch.object(ct.sys, "stderr", buf):
            ct.print_word_losses([])
        self.assertEqual(buf.getvalue(), "")

    def test_collapse_declared_by_excerpt_covers_whole_repeat(self):
        """Модель цитирует залипание обрывком — хвост повтора не считается потерей."""
        src, out = self.pair("вот " + "Dark " * 20 + "вроде", "вот Darkhouse вроде")
        guesses = [{"time": "00:01:00", "from": "Dark Dark Dark", "to": "Darkhouse",
                    "basis": "залипание, 20+ повторов"}]
        self.assertEqual(ct.word_losses(src, out, guesses), [])

    def test_declared_replacement_still_counts_by_occurrence(self):
        """Обычная догадка вычитается по количеству, а не по словам целиком."""
        src, out = self.pair("есед есед есед поднят", "СЭД поднят")
        guesses = [{"time": "00:01:00", "from": "есед", "to": "СЭД",
                    "basis": "аббревиатура со слуха"}]
        rep = ct.word_losses(src, out, guesses)
        self.assertEqual(len(rep), 1)
        self.assertEqual(rep[0]["words"], {"есед": 2})


if __name__ == "__main__":
    unittest.main()
