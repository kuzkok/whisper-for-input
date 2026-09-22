"""Разрез ASR-сегментов по смене говорящего, определённой на уровне слов.

Зачем. whisperx.assign_word_speakers ставит сегменту одну метку — доминирующую
по пересечению с диаризацией (whisperx/diarize.py:229), а каждому слову свою
(там же, :238-256). Если VAD-сегмент накрыл речь двух человек, вся она уходит
доминирующему, и в транскрипте вопрос одного оказывается внутри реплики другого.

Почему не режем по каждой смене метки слова. Замерено на двух реальных записях
(24,7 и 38,9 минут): метки слов на стыках систематически смещены на слово-два
относительно того, что слышно — «мы просто не | викторизуем», «это уже |
нарушение». Наивный разрез чинит 5% реплик и рвёт фразу посреди в 30-60 местах,
где до него таких мест было 1-2. Поэтому режем только уверенные случаи:

- чужой ран короче MIN_RUN_WORDS — дребезг метки на границе интервала, не трогаем;
- точку разреза сдвигаем к ближайшей границе фразы в окне SNAP_WINDOW_WORDS:
  знак препинания от Whisper точнее границы pyannote;
- границы фразы рядом нет — режем по паузе от MIN_PAUSE_S: смена говорящего
  обычно слышна;
- ни знака, ни паузы — режем, только если чужой речи набралось FORCE_RUN_WORDS:
  столько чужого текста в реплике хуже, чем разрезанная фраза;
- разрез, оставляющий кусок короче MIN_PART_WORDS, отменяем целиком: это не
  реплика, а срезанный хвост фразы;
- разрезов нет, но большинство слов чужие — исправляем метку сегмента
  (_relabelled): whisperx считает её по длительности, и она расходится
  с большинством слов, когда чужая фраза целиком попала в сегмент.

Модуль на stdlib и без импорта whisperx: логика сборки тестируется без GPU
и переиспользуется оффлайн в tools/segmentation_metrics.py на сохранённых дампах.
"""

SENTENCE_END = (".", "?", "!", "…")

# Пороги подобраны по метрике на двух записях (см. задачу
# 2026-09-21-diarize-transcript-quality): реплики с чужим раном от 5 слов
# 5→2 и 7→2, слов не под своей меткой −24% и −37%, число реплик +1.
MIN_RUN_WORDS = 5
SNAP_WINDOW_WORDS = 3
MIN_PAUSE_S = 0.4
FORCE_RUN_WORDS = 10
# Кусок короче этого — не реплика, а срезанный хвост чужой фразы («…и |
# подключить.»): такой разрез отменяем целиком. Исключение — кусок, который сам
# является целой фразой: между двумя знаками препинания дребезг не помещается.
MIN_PART_WORDS = 3


def _label(word, fallback):
    """Метка слова; у слов без пересечения с диаризацией её нет (fill_nearest=False)."""
    return word.get("speaker") or fallback


def _runs(words, fallback):
    """Последовательности подряд идущих слов с одинаковой меткой: [(start, end, label)]."""
    runs = []
    i = 0
    while i < len(words):
        j = i
        label = _label(words[i], fallback)
        while j + 1 < len(words) and _label(words[j + 1], fallback) == label:
            j += 1
        runs.append((i, j, label))
        i = j + 1
    return runs


def _snap_to_sentence(words, index, window):
    """Ближайшая к index граница фразы в окне ±window слов, иначе None."""
    for distance in range(window + 1):
        for candidate in (index - distance, index + distance):
            if 1 <= candidate < len(words) and words[candidate - 1]["word"].strip().endswith(SENTENCE_END):
                return candidate
    return None


def _pause_before(words, index):
    if not 1 <= index < len(words):
        return 0.0
    start, end = words[index].get("start"), words[index - 1].get("end")
    if start is None or end is None:
        return 0.0
    return start - end


def _cut_points(words, speaker):
    """Индексы слов, с которых начинается новый сегмент."""
    cuts = set()
    for start, end, label in _runs(words, speaker):
        length = end - start + 1
        if label == speaker or length < MIN_RUN_WORDS:
            continue
        for edge in (start, end + 1):
            if edge in (0, len(words)):
                continue  # край сегмента: резать нечего
            point = _snap_to_sentence(words, edge, SNAP_WINDOW_WORDS)
            if point is None and (_pause_before(words, edge) >= MIN_PAUSE_S or length >= FORCE_RUN_WORDS):
                point = edge
            if point is not None:
                cuts.add(point)
    return _drop_thin_cuts(words, sorted(cuts))


def _is_whole_sentence(words, start, end):
    """Кусок [start, end) начинается после знака препинания и сам им кончается."""
    opens = start == 0 or words[start - 1]["word"].strip().endswith(SENTENCE_END)
    closes = words[end - 1]["word"].strip().endswith(SENTENCE_END)
    return opens and closes


def _drop_thin_cuts(words, cuts):
    """Убирает разрезы, отрезающие огрызок: лучше оставить фразу целой у того,
    кто произнёс её основную часть, чем завести реплику из одного слова."""
    cuts = list(cuts)
    changed = True
    while changed and cuts:
        changed = False
        bounds = [0] + cuts + [len(words)]
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            if end - start >= MIN_PART_WORDS or _is_whole_sentence(words, start, end):
                continue
            # Прижимаем к соседу: слева, если этот разрез наш, иначе справа.
            cuts.remove(start if start in cuts else end)
            changed = True
            break
    return cuts


def _dominant_label(words, fallback):
    counts = {}
    for word in words:
        label = _label(word, fallback)
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


def _part(segment, words, is_first, is_last):
    """Кусок сегмента: тайминги по словам, края — от исходного сегмента.

    Края наследуются, потому что их задаёт VAD: между границей сегмента и первым
    (последним) словом лежит речь без выравнивания, и терять её в таймингах
    не за чем.
    """
    fallback = segment.get("speaker", "UNKNOWN")
    starts = [w["start"] for w in words if w.get("start") is not None]
    ends = [w["end"] for w in words if w.get("end") is not None]
    return {
        **segment,
        "start": float(segment["start"]) if is_first or not starts else float(min(starts)),
        "end": float(segment["end"]) if is_last or not ends else float(max(ends)),
        "speaker": _dominant_label(words, fallback),
        "text": " ".join(w["word"] for w in words),
        "words": words,
    }


def _relabelled(segment, words, speaker):
    """Метка сегмента, если по словам видно, что whisperx ошибся.

    Метку сегменту он ставит по длительности пересечения, и она расходится
    с большинством слов там, где чужая фраза целиком попала в сегмент, а её
    хвост остался под меткой соседа. Переназначаем только при заметном перевесе
    (от MIN_RUN_WORDS чужих слов) — иначе это дребезг на стыке.
    """
    dominant = _dominant_label(words, speaker)
    alien = sum(1 for word in words if _label(word, speaker) != speaker)
    if dominant == speaker or alien < MIN_RUN_WORDS:
        return segment
    return {**segment, "speaker": dominant}


def split_segments(segments):
    """Сегменты после assign_word_speakers → сегменты, однородные по говорящему.

    Сегменты без слов и сегменты без уверенной смены говорящего возвращаются
    как есть, поэтому на записи без перебиваний выход побайтово прежний.
    """
    result = []
    for segment in segments:
        words = segment.get("words")
        if not words:
            result.append(segment)
            continue
        speaker = segment.get("speaker", "UNKNOWN")
        cuts = _cut_points(words, speaker)
        if not cuts:
            result.append(_relabelled(segment, words, speaker))
            continue
        previous = 0
        for cut in cuts + [len(words)]:
            part = words[previous:cut]
            if part:
                result.append(_part(segment, part, is_first=previous == 0, is_last=cut == len(words)))
            previous = cut
    return result
