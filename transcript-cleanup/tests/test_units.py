"""Юнит-тесты чистых функций cleanup_transcript (без вызовов claude -p)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

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
    def test_replica_is_atom(self):
        reps = [ct.Replica("00:00:01", "SPEAKER_00", "x") for _ in range(45)]
        chunks = ct.chunk_replicas(reps)
        self.assertEqual([len(c) for c in chunks], [40, 5])

    def test_empty(self):
        self.assertEqual(ct.chunk_replicas([]), [])

    def test_context_lines_full(self):
        reps = [ct.Replica(f"00:00:{i:02d}", "SPEAKER_00", f"r{i}") for i in range(5)]
        ctx = ct.context_lines(reps, upto=4)
        self.assertEqual(ctx.splitlines(), [reps[1].line(), reps[2].line(), reps[3].line()])

    def test_context_lines_at_start(self):
        reps = [ct.Replica("00:00:01", "SPEAKER_00", "r0")]
        self.assertEqual(ct.context_lines(reps, upto=0), "")


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
            out = ct.clean_chunk("prompt", chunk, "", "ctx", "label")
        finally:
            ct.call_claude = orig
        self.assertEqual(out, ["один.", "два."])
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


if __name__ == "__main__":
    unittest.main()
