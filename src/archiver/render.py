"""Turn a video's metadata, cleaned paragraphs and outline into an Obsidian note. Pure functions."""

from __future__ import annotations

import re
import unicodedata
from datetime import date

import yaml

from archiver.models import Outline, Paragraph, Topic, VideoMeta

MAX_FILENAME_BYTES = 180
LINK_EVERY_SECONDS = 150
_FORBIDDEN_IN_FILENAME = set('[]#^|\\/:*"<>?')
_ORDERED_LIST = re.compile(r"^(\d+)([.)])(?=\s|$)")

_HEADINGS = {
    "ru": {"key_points": "Ключевые мысли", "contents": "Содержание", "speaker": "Спикер"},
    "en": {"key_points": "Key points", "contents": "Contents", "speaker": "Speaker"},
}


# --- file name ---------------------------------------------------------------------------------


def _clean_title_for_filename(title: str) -> str:
    chars = []
    for ch in title:
        if unicodedata.category(ch) == "Cc":
            chars.append(" ")
        elif ch not in _FORBIDDEN_IN_FILENAME:
            chars.append(ch)
    text = " ".join("".join(chars).split())
    return text.lstrip(".").rstrip(". ")


def note_filename(meta: VideoMeta) -> str:
    prefix = f"{meta.upload_date.isoformat()} " if meta.upload_date else ""
    suffix = f" ({meta.video_id}).md"
    title = _clean_title_for_filename(meta.title) or "Untitled"
    budget = MAX_FILENAME_BYTES - len((prefix + suffix).encode("utf-8"))
    # Cut on a character boundary; `ignore` drops a partial trailing character.
    title = title.encode("utf-8")[: max(budget, 0)].decode("utf-8", errors="ignore")
    title = title.rstrip(". ") or "Untitled"
    return f"{prefix}{title}{suffix}"


# --- helpers -----------------------------------------------------------------------------------


def _timestamp(seconds: float, with_hours: bool) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if with_hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{hours * 60 + minutes:02d}:{secs:02d}"


def _link(meta: VideoMeta, seconds: float, with_hours: bool) -> str:
    return f"[{_timestamp(seconds, with_hours)}](https://youtu.be/{meta.video_id}?t={int(seconds)})"


def _watch_url(meta: VideoMeta) -> str:
    return f"https://www.youtube.com/watch?v={meta.video_id}"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _heading_title(text: str, fallback: str) -> str:
    return _one_line(text).lstrip("#").strip() or fallback


def _escape_leading_marker(text: str) -> str:
    """Stop Markdown from reading a paragraph's first characters as a heading, list or quote."""
    if text[:1] in {"#", ">", "-", "+", "*"}:
        return "\\" + text
    match = _ORDERED_LIST.match(text)
    if match:
        return f"{match.group(1)}\\{match.group(2)}{text[match.end(2):]}"
    return text


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines())


# --- frontmatter -------------------------------------------------------------------------------


class _IndentedDumper(yaml.SafeDumper):
    """safe_dump, but list items are indented under their key (`tags:\n  - video`)."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


def _frontmatter(meta: VideoMeta, outline: Outline, language: str, created: date) -> str:
    data: dict[str, object] = {"title": _one_line(meta.title), "url": _watch_url(meta)}
    data["video_id"] = meta.video_id
    if meta.channel:
        data["channel"] = meta.channel
    if meta.upload_date:
        data["published"] = meta.upload_date
    data["duration_min"] = round(meta.duration / 60)
    data["language"] = language
    data["tags"] = _unique(["video", "youtube", *outline.tags])
    data["created"] = created
    body = yaml.dump(
        data,
        Dumper=_IndentedDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=1_000_000,
    )
    return f"---\n{body}---"


# --- body --------------------------------------------------------------------------------------


def _speaker_labels(paragraphs: list[Paragraph], language: str) -> dict[str, str]:
    """Map raw speaker ids to display labels, or {} when fewer than two speakers occur."""
    order = _unique([p.speaker for p in paragraphs if p.speaker is not None])
    if len(order) < 2:
        return {}
    word = _HEADINGS.get(language, _HEADINGS["en"])["speaker"]
    return {raw: f"{word} {i}" for i, raw in enumerate(order, start=1)}


def _topic_ranges(topics: list[Topic], count: int) -> list[tuple[Topic, int, int]]:
    ranges = []
    for i, topic in enumerate(topics):
        start = min(max(topic.first_paragraph, 0), count)
        end = min(topics[i + 1].first_paragraph, count) if i + 1 < len(topics) else count
        ranges.append((topic, start, max(end, start)))
    return ranges


def _render_paragraphs(
    meta: VideoMeta,
    paragraphs: list[Paragraph],
    labels: dict[str, str],
    with_hours: bool,
    last_link_time: float,
) -> list[str]:
    blocks: list[str] = []
    previous_speaker: str | None = None
    for index, para in enumerate(paragraphs):
        prefix = ""
        if para.start - last_link_time >= LINK_EVERY_SECONDS:
            prefix += _link(meta, para.start, with_hours) + " "
            last_link_time = para.start
        text = para.text.strip()
        if para.speaker is not None and labels and (index == 0 or para.speaker != previous_speaker):
            prefix += f"**{labels[para.speaker]}:** "
        previous_speaker = para.speaker
        blocks.append(prefix + text if prefix else _escape_leading_marker(text))
    return blocks


def _render_section(
    meta: VideoMeta,
    title: str,
    paragraphs: list[Paragraph],
    labels: dict[str, str],
    with_hours: bool,
) -> str:
    head = f"## {title}"
    last_link_time = 0.0
    if paragraphs:
        last_link_time = paragraphs[0].start
        head += "\n" + _link(meta, last_link_time, with_hours)
    blocks = [head, *_render_paragraphs(meta, paragraphs, labels, with_hours, last_link_time)]
    return "\n\n".join(blocks)


def _contents(
    meta: VideoMeta,
    ranges: list[tuple[Topic, int, int]],
    paragraphs: list[Paragraph],
    with_hours: bool,
    heading: str,
) -> str | None:
    if len(ranges) < 2:
        return None
    lines = [f"## {heading}"]
    for topic, start, _end in ranges:
        title = _heading_title(topic.title, "Untitled")
        summary = _one_line(topic.summary)
        label = f"{title}: {summary}" if summary else title
        if start < len(paragraphs):
            lines.append(f"- {_link(meta, paragraphs[start].start, with_hours)} {label}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


def render_note(
    meta: VideoMeta,
    paragraphs: list[Paragraph],
    outline: Outline,
    *,
    language: str,
    warnings: list[str],
    created: date,
) -> str:
    names = _HEADINGS.get(language, _HEADINGS["en"])
    with_hours = meta.duration >= 3600
    labels = _speaker_labels(paragraphs, language)
    ranges = _topic_ranges(outline.topics, len(paragraphs))

    blocks: list[str] = [
        _frontmatter(meta, outline, language, created)
        + f"\n![]({_watch_url(meta)})"
    ]
    if warnings:
        lines = [line for w in warnings for line in w.splitlines() if line.strip()]
        blocks.append("> [!warning]\n" + "\n".join(f"> {line}" for line in lines))
    if outline.tldr.strip():
        blocks.append("> [!summary] TL;DR\n" + _quote(outline.tldr.strip()))
    points = [_one_line(p) for p in outline.key_points if p.strip()]
    if points:
        blocks.append(f"## {names['key_points']}\n" + "\n".join(f"- {p}" for p in points))
    contents = _contents(meta, ranges, paragraphs, with_hours, names["contents"])
    if contents:
        blocks.append(contents)

    if ranges:
        for topic, start, end in ranges:
            title = _heading_title(topic.title, _heading_title(meta.title, "Untitled"))
            blocks.append(_render_section(meta, title, paragraphs[start:end], labels, with_hours))
    elif paragraphs:
        title = _heading_title(meta.title, "Untitled")
        blocks.append(_render_section(meta, title, paragraphs, labels, with_hours))

    return "\n\n".join(blocks) + "\n"
