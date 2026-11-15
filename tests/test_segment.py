from archiver.models import Word
from archiver.segment import MAX_WORDS, build_segments


def make(texts, start=0.0, step=0.3, speaker=None):
    words, t = [], start
    for text in texts:
        words.append(Word(text=text, start=t, end=t + step - 0.05, speaker=speaker))
        t += step
    return words


def test_empty():
    assert build_segments([]) == []


def test_short_sentence_end_does_not_break():
    segs = build_segments(make(["Hi.", "Yes", "ok."]))
    assert len(segs) == 1 and segs[0].text == "Hi. Yes ok."


def test_sentence_end_breaks_after_15_words():
    words = make(["w"] * 14 + ["end."] + ["next", "one"])
    segs = build_segments(words)
    assert [len(s.text.split()) for s in segs] == [15, 2]


def test_sentence_end_breaks_after_6_seconds():
    words = make(["a", "b", "c", "d.", "e"], step=2.0)  # d ends at 7.95 s
    segs = build_segments(words)
    assert [s.text for s in segs] == ["a b c d.", "e"]


def test_ellipsis_and_question_mark():
    words = make(["w"] * 14 + ["really?"] + ["w"] * 14 + ["so…"] + ["x"])
    assert [len(s.text.split()) for s in build_segments(words)] == [15, 15, 1]


def test_pause_breaks():
    a = make(["one", "two"])
    b = make(["three"], start=a[-1].end + 0.8)
    assert [s.text for s in build_segments(a + b)] == ["one two", "three"]


def test_small_pause_does_not_break():
    a = make(["one", "two"])
    b = make(["three"], start=a[-1].end + 0.5)
    assert len(build_segments(a + b)) == 1


def test_speaker_change_breaks_and_keeps_speaker():
    words = make(["hi"], speaker="s0") + make(["yo", "there"], start=0.3, speaker="s1")
    segs = build_segments(words)
    assert [(s.text, s.speaker) for s in segs] == [("hi", "s0"), ("yo there", "s1")]


def test_word_cap():
    segs = build_segments(make(["w"] * (MAX_WORDS + 5), step=0.1))
    assert len(segs[0].text.split()) == MAX_WORDS and len(segs) == 2


def test_time_cap():
    words = make(["w"] * 20, step=2.0)  # no punctuation, 40 s total
    segs = build_segments(words)
    assert segs[0].end - segs[0].start <= 31 and len(segs) == 2


def test_ids_and_no_words_lost_russian():
    texts = ("Привет, друзья. Сегодня мы поговорим о том, как устроены большие языковые модели "
             "и почему они так хорошо работают! А потом посмотрим примеры. Начнём?").split()
    words = make(texts, step=0.5)
    segs = build_segments(words)
    assert [s.id for s in segs] == list(range(len(segs)))
    assert " ".join(s.text for s in segs) == " ".join(texts)
    assert segs[0].start == words[0].start and segs[-1].end == words[-1].end
