"""Group word-level speech-to-text output into sentence-sized segments. Pure code, no model."""

from __future__ import annotations

from archiver.models import Segment, Word

SENTENCE_END = (".", "!", "?", "…")
MIN_SENTENCE_SECONDS = 6.0
MIN_SENTENCE_WORDS = 15
PAUSE_SECONDS = 0.8
MAX_SECONDS = 30.0
MAX_WORDS = 60
_CLOSERS = "\"'»”’)]"


def _ends_sentence(text: str) -> bool:
    return text.rstrip(_CLOSERS).endswith(SENTENCE_END)


def _starts_new_segment(current: list[Word], word: Word) -> bool:
    last = current[-1]
    if word.speaker != current[0].speaker:
        return True
    if word.start - last.end >= PAUSE_SECONDS:
        return True
    duration = last.end - current[0].start
    if len(current) >= MAX_WORDS or word.end - current[0].start > MAX_SECONDS:
        return True
    long_enough = duration >= MIN_SENTENCE_SECONDS or len(current) >= MIN_SENTENCE_WORDS
    return long_enough and _ends_sentence(last.text)


def _make(segment_id: int, words: list[Word]) -> Segment:
    return Segment(
        id=segment_id,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.text for w in words),
        speaker=words[0].speaker,
    )


def build_segments(words: list[Word]) -> list[Segment]:
    segments: list[Segment] = []
    current: list[Word] = []
    for word in words:
        if current and _starts_new_segment(current, word):
            segments.append(_make(len(segments), current))
            current = []
        current.append(word)
    if current:
        segments.append(_make(len(segments), current))
    return segments
