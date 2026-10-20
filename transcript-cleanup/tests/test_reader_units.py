"""Юнит-тесты чистых функций reader_pdf (без вызовов claude -p и без Chrome)."""
from __future__ import annotations

import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reader_pdf as rp  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class ParseChaptersTests(unittest.TestCase):
    def test_three_formats_and_title_cleanup(self):
        text = (
            "Таймкоды:\n"
            "\n"
            "00:00:00 💭 Интро\n"
            "\n"
            "4:12 Почему?\n"
            "01:02:40 — Итог\n"
        )
        chapters = rp.parse_chapters(text)
        self.assertEqual(
            [(c.seconds, c.title) for c in chapters],
            [(0, "Интро"), (252, "Почему?"), (3760, "Итог")],
        )

    def test_hours_without_leading_zero(self):
        chapters = rp.parse_chapters("0:00 А\n1:05:00 Б\n")
        self.assertEqual([c.seconds for c in chapters], [0, 3900])

    def test_trailing_spaces_and_inner_punctuation_kept(self):
        chapters = rp.parse_chapters("00:01:14 📚 Знакомство: «гость» — кто он? \n")
        self.assertEqual(chapters[0].title, "Знакомство: «гость» — кто он?")

    def test_bare_timecode_line_is_not_a_chapter(self):
        """Со страницы выпуска вместе с главами копируются длительности других
        выпусков — голый таймкод без названия главой не считается."""
        chapters = rp.parse_chapters("1:08:09\n00:00:00 Интро\n00:05:00 Дальше\n01:09:18\n")
        self.assertEqual([c.title for c in chapters], ["Интро", "Дальше"])

    def test_timecode_not_at_line_start_ignored(self):
        chapters = rp.parse_chapters("00:00:00 Интро\nсм. 00:03:00 в записи\n")
        self.assertEqual(len(chapters), 1)

    def test_not_ascending_raises(self):
        with self.assertRaises(ValueError):
            rp.parse_chapters("00:05:00 Б\n00:01:00 А\n")

    def test_equal_timecodes_raise(self):
        with self.assertRaises(ValueError):
            rp.parse_chapters("00:01:00 А\n00:01:00 Б\n")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            rp.parse_chapters("Таймкоды:\n\n")


class ParseTranscriptTests(unittest.TestCase):
    def test_paragraphs_kept(self):
        src = (
            "[00:00:05] [Анна] Первый абзац.\n"
            "Второй абзац.\n"
            "\n"
            "[00:00:30] [Игорь] Коротко.\n"
        )
        reps = rp.parse_transcript(src)
        self.assertEqual(len(reps), 2)
        self.assertEqual(reps[0].text, "Первый абзац.\nВторой абзац.")
        self.assertEqual(rp.paragraphs(reps[0]), ["Первый абзац.", "Второй абзац."])
        self.assertEqual(rp.paragraphs(reps[1]), ["Коротко."])

    def test_garbage_before_first_replica_raises(self):
        with self.assertRaises(ValueError):
            rp.parse_transcript("просто текст\n[00:00:05] [Анна] ок\n")

    def test_no_replicas_raises(self):
        with self.assertRaises(ValueError):
            rp.parse_transcript("\n\n")

    def test_fixture(self):
        reps = rp.parse_transcript((FIXTURES / "podcast.clean.txt").read_text())
        self.assertEqual(len(reps), 7)
        self.assertEqual(len(rp.paragraphs(reps[3])), 5)


def _fixture_replicas():
    return rp.parse_transcript((FIXTURES / "podcast.clean.txt").read_text())


class LocateTests(unittest.TestCase):
    def setUp(self):
        self.reps = _fixture_replicas()

    def test_chapter_at_replica_start(self):
        self.assertEqual(rp.locate(self.reps, 20), rp.Spot(2, 0, None))

    def test_before_first_replica_goes_to_first(self):
        self.assertEqual(rp.locate(self.reps, 0), rp.Spot(0, 0, None))

    def test_snaps_to_next_replica_within_threshold(self):
        # 00:04:08 — за 2 секунды до реплики 00:04:10
        self.assertEqual(rp.locate(self.reps, 248), rp.Spot(4, 0, None))

    def test_inside_long_replica_gives_estimate(self):
        # 00:02:40 внутри реплики 00:00:31 … 00:04:10; по доле символов это абзац 2
        spot = rp.locate(self.reps, 160)
        self.assertEqual((spot.replica, spot.paragraph), (3, 1))
        self.assertIsNotNone(spot.candidates)
        self.assertIn(2, spot.candidates)

    def test_inside_replica_without_paragraphs_goes_to_replica(self):
        # реплика 00:04:15 одним абзацем, глава внутри неё — заголовок перед репликой
        self.assertEqual(rp.locate(self.reps, 300), rp.Spot(5, 0, None))

    def test_just_after_replica_start_needs_no_model(self):
        self.assertEqual(rp.locate(self.reps, 33), rp.Spot(3, 0, None))

    def test_candidates_window_clamped(self):
        spot = rp.locate(self.reps, 160)
        self.assertEqual(spot.candidates, [0, 1, 2, 3, 4])


class AnchorChaptersTests(unittest.TestCase):
    def setUp(self):
        self.reps = _fixture_replicas()
        self.chapters = rp.parse_chapters((FIXTURES / "podcast.chapters.txt").read_text())

    def test_model_choice_applied(self):
        with mock.patch.object(rp, "pick_paragraphs", return_value={2: 2}) as pick:
            sections = rp.anchor_chapters("prompt", self.reps, self.chapters)
        pick.assert_called_once()
        self.assertEqual(
            [(s.replica, s.paragraph, s.title) for s in sections],
            [(0, 0, "Интро"), (2, 0, "Подготовка почвы"), (3, 2, "Компост и мульча"),
             (4, 0, "Полив"), (6, 0, "Финал")],
        )
        self.assertEqual(sections[2].time, "00:02:40")

    def test_model_failure_falls_back_to_estimate(self):
        with mock.patch.object(rp, "pick_paragraphs", side_effect=RuntimeError("x")):
            sections = rp.anchor_chapters("prompt", self.reps, self.chapters)
        self.assertEqual((sections[2].replica, sections[2].paragraph), (3, 1))

    def test_no_model_call_when_nothing_ambiguous(self):
        chapters = [rp.Chapter(0, "А"), rp.Chapter(248, "Б")]
        with mock.patch.object(rp, "pick_paragraphs") as pick:
            rp.anchor_chapters("prompt", self.reps, chapters)
        pick.assert_not_called()

    def test_out_of_order_anchor_clamped_to_previous(self):
        chapters = [rp.Chapter(150, "А"), rp.Chapter(160, "Б")]
        with mock.patch.object(rp, "pick_paragraphs", return_value={0: 3, 1: 1}):
            sections = rp.anchor_chapters("prompt", self.reps, chapters)
        self.assertEqual([(s.replica, s.paragraph) for s in sections], [(3, 3), (3, 3)])


def _fake_call(outputs):
    """Подмена ct.call_claude: отдаёт ответы по очереди и считает вызовы."""
    calls = []

    def call(system_prompt, user_msg, model, schema, timeout_s, budget_usd):
        calls.append(user_msg)
        return outputs[len(calls) - 1]

    return call, calls


class PickParagraphsTests(unittest.TestCase):
    def setUp(self):
        self.reps = _fixture_replicas()
        self.items = [(0, "Компост и мульча", rp.locate(self.reps, 160))]

    def test_valid_answer(self):
        call, calls = _fake_call([{"anchors": [{"n": 1, "paragraph": 3}]}])
        with mock.patch.object(rp.ct, "call_claude", call):
            got = rp.pick_paragraphs("p", self.reps, self.items, [])
        self.assertEqual(got, {0: 2})  # номер абзаца в ответе 1-based
        self.assertEqual(len(calls), 1)

    def test_paragraph_outside_candidates_retried(self):
        call, calls = _fake_call([
            {"anchors": [{"n": 1, "paragraph": 9}]},
            {"anchors": [{"n": 1, "paragraph": 3}]},
        ])
        with mock.patch.object(rp.ct, "call_claude", call), \
                mock.patch("sys.stderr"):
            got = rp.pick_paragraphs("p", self.reps, self.items, [])
        self.assertEqual(got, {0: 2})
        self.assertEqual(len(calls), 2)
        self.assertIn("retry_notice", calls[1])

    def test_missing_chapter_retried_then_fails(self):
        call, _ = _fake_call([{"anchors": []}] * 3)
        with mock.patch.object(rp.ct, "call_claude", call), \
                mock.patch("sys.stderr"), self.assertRaises(RuntimeError):
            rp.pick_paragraphs("p", self.reps, self.items, [])


class SectionsByModelTests(unittest.TestCase):
    def setUp(self):
        self.reps = _fixture_replicas()

    def test_units_cover_every_paragraph(self):
        units = rp.units(self.reps)
        self.assertEqual(len(units), 11)  # 6 однострочных реплик + 5 абзацев
        self.assertEqual(units[3], (3, 0))
        self.assertEqual(units[4], (3, 1))

    def test_valid_sections(self):
        call, _ = _fake_call([{"sections": [
            {"start": 1, "title": "Начало"}, {"start": 5, "title": "Почва"},
        ]}])
        with mock.patch.object(rp.ct, "call_claude", call):
            sections = rp.model_sections("p", self.reps)
        self.assertEqual([(s.replica, s.paragraph, s.title) for s in sections],
                         [(0, 0, "Начало"), (3, 1, "Почва")])
        self.assertEqual(sections[0].time, "00:00:05")
        self.assertIsNone(sections[1].time)  # внутри реплики времени нет

    def test_bad_numbers_retried(self):
        for bad in (
            [{"start": 1, "title": "А"}, {"start": 99, "title": "Б"}],  # вне диапазона
            [{"start": 1, "title": "А"}, {"start": 5, "title": "Б"}, {"start": 3, "title": "В"}],
            [{"start": 2, "title": "А"}],  # первый раздел не с начала
            [{"start": 1, "title": "  "}],  # пустое название
            [],
        ):
            with self.subTest(bad=bad):
                call, calls = _fake_call([{"sections": bad},
                                          {"sections": [{"start": 1, "title": "А"}]}])
                with mock.patch.object(rp.ct, "call_claude", call), \
                        mock.patch("sys.stderr"):
                    sections = rp.model_sections("p", self.reps)
                self.assertEqual(len(calls), 2)
                self.assertIn("retry_notice", calls[1])
                self.assertEqual([s.title for s in sections], ["А"])

    def test_exhausted_retries_raise(self):
        call, _ = _fake_call([{"sections": []}] * 3)
        with mock.patch.object(rp.ct, "call_claude", call), \
                mock.patch("sys.stderr"), self.assertRaises(RuntimeError):
            rp.model_sections("p", self.reps)


class _TextCollector(HTMLParser):
    """Текст абзацев реплик: <p> без class. Остальное — разметка шага."""

    def __init__(self):
        super().__init__()
        self.paras: list[str] = []
        self.h2: list[str] = []
        self.toc: list[str] = []
        self._cur: list[str] | None = None
        self._tag = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "p" and "class" not in attrs:
            self._cur, self._tag = [], "p"
        elif tag == "h2" and "id" in attrs:
            self._cur, self._tag = [], "h2"
        elif tag == "a" and attrs.get("href", "").startswith("#s"):
            self._cur, self._tag = [], "a"

    def handle_endtag(self, tag):
        if self._cur is not None and tag == self._tag:
            text = "".join(self._cur)
            {"p": self.paras, "h2": self.h2, "a": self.toc}[tag].append(text)
            self._cur = None

    def handle_data(self, data):
        if self._cur is not None:
            self._cur.append(data)


class RenderHtmlTests(unittest.TestCase):
    def setUp(self):
        self.reps = _fixture_replicas()
        self.sections = [
            rp.Section(0, 0, "Интро", "00:00:00"),
            rp.Section(3, 2, "Компост & <мульча>", "00:02:40"),
        ]
        self.html = rp.render_html("Грядка", self.reps, self.sections, "по главам")
        self.parsed = _TextCollector()
        self.parsed.feed(self.html)

    def test_replica_text_unchanged(self):
        expected = [p for r in self.reps for p in rp.paragraphs(r)]
        self.assertEqual(self.parsed.paras, expected)

    def test_headings_and_toc(self):
        self.assertEqual(self.parsed.toc, ["Интро", "Компост & <мульча>"])
        self.assertEqual(len(self.parsed.h2), 2)
        self.assertIn("Компост & <мульча>", self.parsed.h2[1])

    def test_heading_sits_before_its_paragraph(self):
        h = self.html.index('id="s2"')
        p3 = self.html.index(rp.html.escape(rp.paragraphs(self.reps[3])[2]))
        p2 = self.html.index(rp.html.escape(rp.paragraphs(self.reps[3])[1]))
        self.assertLess(p2, h)
        self.assertLess(h, p3)

    def test_speaker_repeated_after_mid_replica_heading(self):
        h = self.html.index('id="s2"')
        p3 = self.html.index(rp.html.escape(rp.paragraphs(self.reps[3])[2]))
        self.assertIn("Мария Петрова", self.html[h:p3])

    def test_timecodes_present(self):
        for r in self.reps:
            self.assertIn(r.time, self.html)


if __name__ == "__main__":
    unittest.main()
