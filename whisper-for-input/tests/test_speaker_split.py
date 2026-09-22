"""Разрез сегментов по смене говорящего — чистая логика, без whisperx и GPU."""
import pytest

from speaker_split import FORCE_RUN_WORDS, MIN_RUN_WORDS, split_segments

pytestmark = pytest.mark.unit


def _w(word, start, end, speaker="SPEAKER_00"):
    return {"word": word, "start": start, "end": end, "score": 0.9, "speaker": speaker}


def _seg(words, speaker="SPEAKER_00", start=None, end=None):
    return {
        "start": start if start is not None else words[0]["start"],
        "end": end if end is not None else words[-1]["end"],
        "speaker": speaker,
        "text": " ".join(w["word"] for w in words),
        "words": words,
    }


def _run(prefix, n, t0, speaker, step=0.4):
    """n слов одного говорящего подряд, без знаков препинания."""
    return [_w(f"{prefix}{i}", t0 + i * step, t0 + (i + 1) * step - 0.05, speaker) for i in range(n)]


def test_homogeneous_segment_is_untouched():
    seg = _seg(_run("сл", 6, 0.0, "SPEAKER_00"))
    assert split_segments([seg]) == [seg]


def test_segment_without_words_is_untouched():
    seg = {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "без слов"}
    assert split_segments([seg]) == [seg]


def test_short_alien_run_is_not_cut():
    """Дребезг метки на границе интервала диаризации: чужих слов меньше порога —
    режем только уверенные куски, иначе фраза рвётся посреди."""
    words = _run("наш", 5, 0.0, "SPEAKER_00") + _run("чуж", MIN_RUN_WORDS - 1, 2.0, "SPEAKER_01")
    result = split_segments([_seg(words)])
    assert len(result) == 1


def test_long_alien_run_cuts_at_sentence_boundary():
    """Разрез сдвигается к ближайшей точке: pyannote ставит границу с точностью
    до слова-двух, а знак препинания от Whisper — надёжный ориентир."""
    ours = _run("наш", 4, 0.0, "SPEAKER_00")
    ours[-1]["word"] = "конец."
    alien = _run("чуж", MIN_RUN_WORDS + 2, 2.0, "SPEAKER_01")
    # Хвост чужой фразы по ошибке помечен нашим спикером и произнесён вплотную
    # (так это и выглядит в записях) — должен уйти вместе с ней, а не в свой кусок.
    tail = [_w("хвост.", alien[-1]["end"] + 0.05, alien[-1]["end"] + 0.4, "SPEAKER_00")]
    result = split_segments([_seg(ours + alien + tail)])
    assert [s["speaker"] for s in result] == ["SPEAKER_00", "SPEAKER_01"]
    assert result[0]["text"].endswith("конец.")
    assert result[1]["text"].endswith("хвост.")


def test_cut_recomputes_text_and_timings():
    ours = _run("наш", 4, 0.0, "SPEAKER_00")
    ours[-1]["word"] = "конец."
    alien = _run("чуж", MIN_RUN_WORDS + 1, 2.0, "SPEAKER_01")
    seg = _seg(ours + alien, start=0.0, end=9.0)
    first, second = split_segments([seg])
    assert first["text"] == " ".join(w["word"] for w in ours)
    assert second["text"] == " ".join(w["word"] for w in alien)
    # Края сегмента сохраняются: внешние границы задаёт VAD, а не первое слово.
    assert first["start"] == 0.0
    assert second["end"] == 9.0
    assert first["end"] == ours[-1]["end"]
    assert second["start"] == alien[0]["start"]
    assert [w["word"] for w in first["words"]] == [w["word"] for w in ours]


def test_cut_by_pause_when_no_punctuation():
    """Whisper не всегда ставит знаки; заметная пауза — второй ориентир."""
    ours = _run("наш", 4, 0.0, "SPEAKER_00")
    alien = _run("чуж", MIN_RUN_WORDS, 3.0, "SPEAKER_01")  # разрыв 1.4 с после нашего слова
    result = split_segments([_seg(ours + alien)])
    assert [s["speaker"] for s in result] == ["SPEAKER_00", "SPEAKER_01"]


def test_long_run_cuts_without_any_anchor():
    """Ни знака, ни паузы — но чужой речи столько, что оставлять её в чужой
    реплике хуже, чем разрезать фразу."""
    ours = _run("наш", 4, 0.0, "SPEAKER_00")
    alien = _run("чуж", FORCE_RUN_WORDS, 1.6, "SPEAKER_01")
    result = split_segments([_seg(ours + alien)])
    assert [s["speaker"] for s in result] == ["SPEAKER_00", "SPEAKER_01"]


def test_medium_run_without_anchor_is_kept():
    """Между порогами: чужой ран заметный, но опоры нет — не режем, чтобы не
    плодить обрывки фраз."""
    ours = _run("наш", 4, 0.0, "SPEAKER_00")
    alien = _run("чуж", MIN_RUN_WORDS, 1.6, "SPEAKER_01")
    assert len(split_segments([_seg(ours + alien)])) == 1


def test_words_without_speaker_inherit_segment_label():
    """Слово без пересечения с диаризацией не создаёт разрез само по себе."""
    words = _run("наш", 4, 0.0, "SPEAKER_00")
    for w in _run("ничей", MIN_RUN_WORDS + 2, 2.0, "SPEAKER_00"):
        w["speaker"] = None
        words.append(w)
    assert len(split_segments([words and _seg(words)])) == 1


def test_multiple_segments_processed_independently():
    clean = _seg(_run("чист", 5, 0.0, "SPEAKER_00"))
    ours = _run("наш", 4, 10.0, "SPEAKER_00")
    ours[-1]["word"] = "точка."
    mixed = _seg(ours + _run("чуж", MIN_RUN_WORDS + 1, 12.0, "SPEAKER_01"))
    result = split_segments([clean, mixed])
    assert len(result) == 3
    assert result[0] == clean


def test_force_cut_does_not_shave_off_a_word():
    """Хвост чужой фразы, помеченный нашим спикером, не должен стать репликой
    из одного слова: разрез, оставляющий огрызок, отменяется целиком, и фраза
    уходит тому, кто произнёс её основную часть."""
    alien = _run("чуж", FORCE_RUN_WORDS, 0.0, "SPEAKER_01")
    tail = [_w("подключить.", alien[-1]["end"] + 0.05, alien[-1]["end"] + 0.5, "SPEAKER_00")]
    result = split_segments([_seg(alien + tail, speaker="SPEAKER_00")])
    assert len(result) == 1
    assert result[0]["speaker"] == "SPEAKER_01"


def test_short_part_that_is_a_whole_sentence_survives():
    """Короткая часть остаётся, если она сама по себе целая фраза: между двумя
    знаками препинания дребезг не помещается."""
    ours = _run("наш", 4, 0.0, "SPEAKER_00")
    ours[-1]["word"] = "верно?"
    alien = _run("чуж", MIN_RUN_WORDS + 1, 2.0, "SPEAKER_01")
    alien[-1]["word"] = "именно."
    more = _run("ещё", 5, 6.0, "SPEAKER_00")
    result = split_segments([_seg(ours + alien + more)])
    assert [s["speaker"] for s in result] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_segment_label_is_not_touched_on_minor_disagreement():
    """Пара чужих слов на стыке метку сегмента не меняет: это дребезг."""
    words = _run("наш", 8, 0.0, "SPEAKER_00")
    words[-1]["speaker"] = "SPEAKER_01"
    words[-2]["speaker"] = "SPEAKER_01"
    result = split_segments([_seg(words)])
    assert len(result) == 1
    assert result[0]["speaker"] == "SPEAKER_00"
