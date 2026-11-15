from __future__ import annotations

from datetime import date

import yaml

from archiver.models import Outline, Paragraph, Topic, VideoMeta
from archiver.render import note_filename, render_note

CREATED = date(2026, 11, 15)


def make_meta(**kw: object) -> VideoMeta:
    base: dict[str, object] = dict(
        video_id="abc123",
        url="https://www.youtube.com/watch?v=abc123",
        title="Hello",
        channel="Chan",
        upload_date=date(2026, 10, 1),
        duration=2520.0,
    )
    base.update(kw)
    return VideoMeta(**base)  # type: ignore[arg-type]


def para(start: float, text: str = "Text.", speaker: str | None = None) -> Paragraph:
    return Paragraph(start_segment=int(start), start=start, speaker=speaker, text=text)


def render(meta=None, paragraphs=None, outline=None, **kw):
    kw.setdefault("language", "en")
    kw.setdefault("warnings", [])
    kw.setdefault("created", CREATED)
    return render_note(
        meta or make_meta(),
        paragraphs if paragraphs is not None else [para(0)],
        outline or Outline(topics=[Topic(title="T", first_paragraph=0)]),
        **kw,
    )


def frontmatter(note: str) -> dict:
    assert note.startswith("---\n")
    return yaml.safe_load(note.split("---\n")[1])


# --- frontmatter -------------------------------------------------------------------------------


def test_frontmatter_types_and_tags() -> None:
    note = render(outline=Outline(tags=["python", "video", "python", "ai"]))
    fm = frontmatter(note)
    assert fm["published"] == date(2026, 10, 1) and type(fm["published"]) is date
    assert fm["created"] == CREATED and type(fm["created"]) is date
    assert fm["duration_min"] == 42
    assert fm["tags"] == ["video", "youtube", "python", "ai"]
    assert fm["url"] == "https://www.youtube.com/watch?v=abc123"
    assert "published: 2026-10-01\n" in note  # bare, not quoted


def test_frontmatter_omits_missing_fields() -> None:
    fm = frontmatter(render(meta=make_meta(channel="", upload_date=None)))
    assert "channel" not in fm and "published" not in fm


def test_tricky_title_survives() -> None:
    title = "Why: \"quotes\" and 'apostrophes' # not a comment"
    fm = frontmatter(render(meta=make_meta(title=title)))
    assert fm["title"] == title


def test_cyrillic_title_not_escaped() -> None:
    note = render(meta=make_meta(title="Привет, мир"))
    assert "title: Привет, мир" in note


# --- file name ---------------------------------------------------------------------------------


def test_filename_basic_and_no_date() -> None:
    assert note_filename(make_meta()) == "2026-10-01 Hello (abc123).md"
    assert note_filename(make_meta(upload_date=None)) == "Hello (abc123).md"


def test_filename_sanitizing() -> None:
    meta = make_meta(title='  ..[A] #b ^c |d \\e/f: g*h "i" <j> k?\n\tl\x00m.. ')
    assert note_filename(meta) == "2026-10-01 A b c d ef gh i j k l m (abc123).md"


def test_filename_untitled() -> None:
    assert note_filename(make_meta(title="[]:?...")) == "2026-10-01 Untitled (abc123).md"


def test_filename_byte_cap_cyrillic() -> None:
    name = note_filename(make_meta(title="Очень длинное название " * 20))
    assert len(name.encode("utf-8")) <= 180
    assert name.endswith(" (abc123).md") and name.startswith("2026-10-01 Очень")
    name.encode("utf-8").decode("utf-8")  # no split characters
    assert " (abc123).md" in name and not name.split(" (abc123)")[0].endswith(" ")


# --- timestamps, contents, topics --------------------------------------------------------------


def two_topic_note(duration: float) -> str:
    paras = [para(0, "One."), para(312, "Two."), para(400, "Three.")]
    outline = Outline(
        topics=[
            Topic(title="First", first_paragraph=0, summary="intro"),
            Topic(title="# Second\nline", first_paragraph=1),
        ]
    )
    return render(meta=make_meta(duration=duration), paragraphs=paras, outline=outline)


def test_timestamps_under_hour() -> None:
    note = two_topic_note(2520)
    assert "- [00:00](https://youtu.be/abc123?t=0) First: intro" in note
    assert "- [05:12](https://youtu.be/abc123?t=312) Second line" in note
    assert "## Second line\n[05:12](https://youtu.be/abc123?t=312)\n\nTwo." in note


def test_timestamps_over_hour_everywhere() -> None:
    note = two_topic_note(3700)
    assert "[0:00:00](https://youtu.be/abc123?t=0)" in note
    assert "[0:05:12](https://youtu.be/abc123?t=312)" in note
    assert "[05:12]" not in note


def test_minutes_may_exceed_59_never_under_hour_format() -> None:
    note = render(
        meta=make_meta(duration=3700), paragraphs=[para(3661)], outline=Outline()
    )
    assert "[1:01:01](https://youtu.be/abc123?t=3661)" in note


def test_single_topic_has_no_contents() -> None:
    assert "## Contents" not in render()


def test_in_topic_link_every_150_seconds() -> None:
    paras = [para(0, "a"), para(100, "b"), para(149.9, "c"), para(150, "d"), para(290, "e"),
             para(310, "f")]
    note = render(paragraphs=paras)
    body = note.split("## T\n", 1)[1]
    assert "\n\nb\n" in body and "\n\nc\n" in body
    assert "[02:30](https://youtu.be/abc123?t=150) d" in body
    assert "\n\ne\n" in body  # 140 s since the last link
    assert "[05:10](https://youtu.be/abc123?t=310) f" in body


def test_empty_topic_list_uses_video_title() -> None:
    note = render(paragraphs=[para(0, "Hi"), para(5, "There")], outline=Outline())
    assert "## Hello\n[00:00](https://youtu.be/abc123?t=0)\n\nHi\n\nThere\n" in note
    assert "## Contents" not in note


def test_no_paragraphs_no_crash() -> None:
    note = render(paragraphs=[], outline=Outline())
    assert note.endswith("![](https://www.youtube.com/watch?v=abc123)\n")


def test_leading_markers_escaped() -> None:
    paras = [para(0, "# not heading"), para(1, "- item"), para(2, "> quote"),
             para(3, "1. first"), para(4, "2025 was good")]
    note = render(paragraphs=paras)
    for expected in ("\\# not heading", "\\- item", "\\> quote", "1\\. first", "\n2025 was good"):
        assert expected in note


# --- speakers, warnings, empty sections --------------------------------------------------------


def test_speaker_labels() -> None:
    paras = [para(0, "a", "x"), para(10, "b", "x"), para(20, "c", "y"), para(30, "d", "x")]
    note = render(paragraphs=paras)
    assert "**Speaker 1:** a\n\nb\n\n**Speaker 2:** c\n\n**Speaker 1:** d\n" in note


def test_speaker_label_repeats_at_topic_start_and_russian() -> None:
    paras = [para(0, "a", "x"), para(10, "b", "y"), para(20, "c", "y")]
    outline = Outline(topics=[Topic(title="A", first_paragraph=0), Topic(title="B", first_paragraph=2)])
    note = render(paragraphs=paras, outline=outline, language="ru")
    assert "**Спикер 2:** c" in note and "Speaker" not in note
    assert "## Содержание" in note


def test_no_labels_for_single_speaker() -> None:
    note = render(paragraphs=[para(0, "a", "x"), para(5, "b", None)])
    assert "Speaker" not in note


def test_warnings_callout() -> None:
    note = render(warnings=["one", "two"])
    assert "\n\n> [!warning]\n> one\n> two\n\n" in note
    assert "[!warning]" not in render()


def test_empty_tldr_and_key_points_omitted() -> None:
    note = render(outline=Outline(tldr="  ", key_points=[]))
    assert "TL;DR" not in note and "Key points" not in note


def test_russian_headings_keep_tldr_title() -> None:
    note = render(outline=Outline(tldr="кратко", key_points=["раз"]), language="ru")
    assert "> [!summary] TL;DR" in note and "## Ключевые мысли\n- раз" in note


# --- snapshot ----------------------------------------------------------------------------------


def test_snapshot() -> None:
    meta = make_meta(duration=600)
    paras = [para(0, "Hello there."), para(200, "More words."), para(400, "Second part.")]
    outline = Outline(
        tldr="Line one.\n\nLine two.",
        key_points=["Point A", "Point B"],
        topics=[
            Topic(title="Intro", first_paragraph=0, summary="warm up"),
            Topic(title="Main", first_paragraph=2, summary="the thing"),
        ],
        tags=["demo"],
    )
    note = render(meta=meta, paragraphs=paras, outline=outline, warnings=["Careful."])
    assert note == """\
---
title: Hello
url: https://www.youtube.com/watch?v=abc123
video_id: abc123
channel: Chan
published: 2026-10-01
duration_min: 10
language: en
tags:
  - video
  - youtube
  - demo
created: 2026-11-15
---
![](https://www.youtube.com/watch?v=abc123)

> [!warning]
> Careful.

> [!summary] TL;DR
> Line one.
>
> Line two.

## Key points
- Point A
- Point B

## Contents
- [00:00](https://youtu.be/abc123?t=0) Intro: warm up
- [06:40](https://youtu.be/abc123?t=400) Main: the thing

## Intro
[00:00](https://youtu.be/abc123?t=0)

Hello there.

[03:20](https://youtu.be/abc123?t=200) More words.

## Main
[06:40](https://youtu.be/abc123?t=400)

Second part.
"""
