"""Outline stage: TL;DR, key points, topics and tags from the cleaned paragraphs, repaired in code."""

from __future__ import annotations

import re

from pydantic import BaseModel

from archiver.cache import fingerprint
from archiver.config import Settings
from archiver.llm import LLM, load_prompt
from archiver.models import Chapter, Outline, Paragraph, Topic, VideoMeta

MIN_TOPIC_SECONDS = 60.0
MIN_TOPIC_PARAGRAPHS = 2
MAX_TAGS = 5
MIN_CHAPTERS = 3
CHAPTER_MEAN_MIN_S = 2 * 60
CHAPTER_MEAN_MAX_S = 20 * 60
CHAPTER_TOLERANCE = 2.0  # seconds a paragraph may start before its chapter mark
DESCRIPTION_CHARS = 3000
OUTLINE_MAX_TOKENS = 20000  # stays under the SDK's limit for non-streaming requests

LANGUAGE_NAMES = {
    "ru": "Russian",
    "en": "English",
    "uk": "Ukrainian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "be": "Belarusian",
    "kk": "Kazakh",
    "zh": "Chinese",
    "ja": "Japanese",
}

# (minutes, min topics, max topics): about one topic per six minutes, flatter for long videos.
_TARGET_POINTS = [(0, 1, 2), (5, 1, 3), (30, 4, 7), (60, 6, 12), (180, 15, 25)]

_TITLE_JUNK = re.compile(r"[#\[\]|^]")
_TAG_JUNK = re.compile(r"[^\w\-/]", re.UNICODE)
_SPACES = re.compile(r"\s+")


# Model output shapes.
class TopicDraft(BaseModel):
    title: str
    first_paragraph: int
    summary: str


class OutlineDraft(BaseModel):
    tldr: str
    key_points: list[str]
    topics: list[TopicDraft]
    tags: list[str]


class TopicSummary(BaseModel):
    number: int  # 1-based, as listed in the request
    summary: str


class ChapterOutlineDraft(BaseModel):
    tldr: str
    key_points: list[str]
    topic_summaries: list[TopicSummary]
    tags: list[str]


def prompt_fingerprint() -> str:
    return fingerprint(load_prompt("outline"))


# ---------------------------------------------------------------------------
# Small helpers


def _clock(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _video_seconds(paragraphs: list[Paragraph], meta: VideoMeta) -> float:
    last = paragraphs[-1].start if paragraphs else 0.0
    return max(last, meta.duration)


def target_topic_range(minutes: float) -> tuple[int, int]:
    """Topic count range for a video of this length, interpolated between fixed points."""
    pts = _TARGET_POINTS
    if minutes >= pts[-1][0]:
        (m0, lo0, hi0), (m1, lo1, hi1) = pts[-2], pts[-1]
    else:
        i = next(k for k in range(1, len(pts)) if minutes < pts[k][0])
        (m0, lo0, hi0), (m1, lo1, hi1) = pts[i - 1], pts[i]
    t = (minutes - m0) / (m1 - m0)
    lo = round(lo0 + t * (lo1 - lo0))
    hi = round(hi0 + t * (hi1 - hi0))
    return max(1, lo), max(max(1, lo), hi)


def clean_title(title: str) -> str:
    """Remove characters that would break a Markdown heading or an Obsidian link."""
    return _SPACES.sub(" ", _TITLE_JUNK.sub("", title)).strip()


def clean_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        t = _SPACES.sub("-", tag.strip().lstrip("#").strip().casefold())
        t = _TAG_JUNK.sub("", t).strip("-/")
        t = re.sub(r"-{2,}", "-", t)
        # Obsidian ignores purely numeric tags; the note always carries these two already.
        if not t or t.isdigit() or t in {"video", "youtube"} or t in out:
            continue
        out.append(t)
        if len(out) == MAX_TAGS:
            break
    return out


def _fallback_title(paragraph: Paragraph, words: int = 6) -> str:
    return clean_title(" ".join(paragraph.text.split()[:words])) or "…"


# ---------------------------------------------------------------------------
# Repair


def repair_topics(topics: list[Topic], paragraphs: list[Paragraph], meta: VideoMeta) -> list[Topic]:
    """Valid, ordered topics: in range, sorted, unique, first at P0, none too short."""
    n = len(paragraphs)
    if n == 0:
        return [Topic(title=clean_title(meta.title) or "…", first_paragraph=0)]
    kept = sorted(
        (t for t in topics if 0 <= t.first_paragraph < n), key=lambda t: t.first_paragraph
    )
    unique: list[Topic] = []
    for t in kept:
        if not unique or t.first_paragraph != unique[-1].first_paragraph:
            unique.append(t.model_copy())
    if not unique:
        return [
            Topic(
                title=clean_title(meta.title) or _fallback_title(paragraphs[0]), first_paragraph=0
            )
        ]
    unique[0].first_paragraph = 0
    merged = _merge_short(unique, paragraphs, _video_seconds(paragraphs, meta))
    for t in merged:
        t.title = clean_title(t.title) or _fallback_title(paragraphs[t.first_paragraph])
        t.summary = t.summary.strip()
    return merged


def _is_short(topics: list[Topic], i: int, paragraphs: list[Paragraph], end_s: float) -> bool:
    first = topics[i].first_paragraph
    nxt = topics[i + 1].first_paragraph if i + 1 < len(topics) else len(paragraphs)
    end = paragraphs[nxt].start if nxt < len(paragraphs) else end_s
    return nxt - first < MIN_TOPIC_PARAGRAPHS or end - paragraphs[first].start < MIN_TOPIC_SECONDS


def _merge_short(topics: list[Topic], paragraphs: list[Paragraph], end_s: float) -> list[Topic]:
    topics = list(topics)
    i = 1
    while i < len(topics):
        if _is_short(topics, i, paragraphs, end_s):
            del topics[i]  # its paragraphs now belong to the previous topic
        else:
            i += 1
    # The first topic has no previous one: it absorbs the next instead.
    while len(topics) > 1 and _is_short(topics, 0, paragraphs, end_s):
        del topics[1]
    return topics


# ---------------------------------------------------------------------------
# Chapters


def chapters_usable(meta: VideoMeta) -> bool:
    chapters = sorted(meta.chapters, key=lambda c: c.start)
    if len(chapters) < MIN_CHAPTERS:
        return False
    duration = meta.duration or chapters[-1].end
    if duration <= 0:
        return False
    covers = chapters[0].start <= max(30.0, 0.05 * duration) and chapters[-1].end >= 0.9 * duration
    mean = duration / len(chapters)
    return covers and CHAPTER_MEAN_MIN_S <= mean <= CHAPTER_MEAN_MAX_S


def chapter_topics(chapters: list[Chapter], paragraphs: list[Paragraph]) -> list[Topic]:
    """Map each chapter start to the first paragraph that starts at or after it."""
    topics: list[Topic] = []
    for chapter in sorted(chapters, key=lambda c: c.start):
        index = next(
            (i for i, p in enumerate(paragraphs) if p.start >= chapter.start - CHAPTER_TOLERANCE),
            None,
        )
        if index is not None:
            topics.append(Topic(title=chapter.title, first_paragraph=index))
    return topics


def fallback_outline(paragraphs: list[Paragraph], meta: VideoMeta) -> Outline:
    """An outline without any model call: chapters if usable, else one topic."""
    if paragraphs and chapters_usable(meta):
        topics = repair_topics(chapter_topics(meta.chapters, paragraphs), paragraphs, meta)
    else:
        topics = repair_topics([], paragraphs, meta)
    return Outline(topics=topics)


# ---------------------------------------------------------------------------
# Model call


def _language_line(language: str) -> str:
    name = LANGUAGE_NAMES.get(language, language)
    return (
        f"Language of the video: {name} ({language}). Write the TL;DR, key points, topic titles "
        f"and summaries in {name}."
    )


def _meta_lines(meta: VideoMeta, seconds: float) -> str:
    lines = [f"Title: {meta.title}"]
    if meta.channel:
        lines.append(f"Channel: {meta.channel}")
    if meta.upload_date:
        lines.append(f"Published: {meta.upload_date.isoformat()}")
    lines.append(f"Duration: {_clock(seconds)}")
    if meta.description.strip():
        lines.append("Description:\n" + meta.description.strip()[:DESCRIPTION_CHARS])
    return "\n".join(lines)


def _transcript(paragraphs: list[Paragraph]) -> str:
    return "\n\n".join(f"[P{i}] ({_clock(p.start)}) {p.text}" for i, p in enumerate(paragraphs))


def _chapter_request(topics: list[Topic], n: int) -> str:
    lines = [
        (
            "The creator divided the video into the chapters below; they are the topics. Keep them "
            "as they are and write a one-line summary for each, returned in `topic_summaries` with "
            "the chapter's number."
        )
    ]
    for k, t in enumerate(topics):
        last = topics[k + 1].first_paragraph - 1 if k + 1 < len(topics) else n - 1
        lines.append(f"{k + 1}. {t.title} (paragraphs P{t.first_paragraph} to P{last})")
    return "\n".join(lines)


def _free_request(seconds: float, n: int) -> str:
    lo, hi = target_topic_range(seconds / 60)
    hi = min(hi, n)
    lo = min(lo, hi)
    span = f"{lo}" if lo == hi else f"{lo} to {hi}"
    return (
        f"Choose the topics yourself. For a video of this length aim for {span} topics. Return "
        "them in `topics`, each with its title, first_paragraph and a one-line summary."
    )


async def build_outline(
    paragraphs: list[Paragraph], meta: VideoMeta, language: str, llm: LLM, settings: Settings
) -> Outline:
    if not paragraphs:
        return fallback_outline(paragraphs, meta)
    seconds = _video_seconds(paragraphs, meta)
    chapter_mode = chapters_usable(meta)
    given = (
        repair_topics(chapter_topics(meta.chapters, paragraphs), paragraphs, meta)
        if chapter_mode
        else []
    )
    request = (
        _chapter_request(given, len(paragraphs))
        if chapter_mode
        else _free_request(seconds, len(paragraphs))
    )
    user = "\n\n".join(
        [
            _language_line(language),
            _meta_lines(meta, seconds),
            request,
            "Transcript:\n" + _transcript(paragraphs),
        ]
    )
    common = {
        "model": settings.outline_model,
        "effort": settings.outline_effort,
        "system": load_prompt("outline"),
        "user": user,
        "max_tokens": OUTLINE_MAX_TOKENS,
    }
    if chapter_mode:
        draft: ChapterOutlineDraft | OutlineDraft = await llm.parse(
            output_format=ChapterOutlineDraft, **common
        )
        topics = _apply_summaries(given, draft.topic_summaries)
    else:
        draft = await llm.parse(output_format=OutlineDraft, **common)
        proposed = [
            Topic(title=t.title, first_paragraph=t.first_paragraph, summary=t.summary)
            for t in draft.topics
        ]
        topics = repair_topics(proposed, paragraphs, meta)
    return Outline(
        tldr=draft.tldr.strip(),
        key_points=[k.strip() for k in draft.key_points if k.strip()],
        topics=topics,
        tags=clean_tags(draft.tags),
    )


def _apply_summaries(topics: list[Topic], summaries: list[TopicSummary]) -> list[Topic]:
    by_number = {s.number: s.summary.strip() for s in summaries}
    return [
        t.model_copy(update={"summary": by_number.get(k + 1, "")}) for k, t in enumerate(topics)
    ]
