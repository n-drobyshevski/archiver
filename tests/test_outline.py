from collections.abc import Callable
from typing import Any

import pytest

from archiver import outline
from archiver.config import Settings
from archiver.errors import LlmRefused
from archiver.models import Chapter, Paragraph, Topic, VideoMeta
from archiver.outline import ChapterOutlineDraft, OutlineDraft, TopicDraft, TopicSummary


class FakeLLM:
    def __init__(self, handler: Callable[[type, str], Any]) -> None:
        self.handler = handler
        self.calls: list[dict] = []
        self.usage: dict = {}

    async def parse(self, *, model, effort, system, user, output_format, max_tokens=16000):
        self.calls.append(
            {"model": model, "effort": effort, "user": user, "output_format": output_format}
        )
        return self.handler(output_format, user)

    def cost_usd(self):
        return 0.0


def make_paragraphs(n: int, every: float = 30.0) -> list[Paragraph]:
    return [
        Paragraph(
            start_segment=i * 3, start=i * every, text=f"Paragraph number {i} talks about things."
        )
        for i in range(n)
    ]


META = VideoMeta(video_id="v", url="u", title="My [great] #video", duration=40 * 30.0)
SETTINGS = Settings()


def t(first: int, title: str = "T") -> Topic:
    return Topic(title=title, first_paragraph=first)


# ---------------------------------------------------------------------------
# repair


def test_repair_drops_out_of_range_sorts_and_dedupes():
    paras = make_paragraphs(40)
    topics = [t(20, "c"), t(-1, "bad"), t(0, "a"), t(10, "b"), t(10, "dup"), t(99, "bad")]
    fixed = outline.repair_topics(topics, paras, META)
    assert [(x.first_paragraph, x.title) for x in fixed] == [(0, "a"), (10, "b"), (20, "c")]


def test_repair_forces_first_topic_to_zero():
    fixed = outline.repair_topics([t(5, "a"), t(20, "b")], make_paragraphs(40), META)
    assert [x.first_paragraph for x in fixed] == [0, 20]


def test_repair_merges_short_topics_into_previous():
    paras = make_paragraphs(40)  # 30 s per paragraph
    topics = [t(0, "a"), t(10, "b"), t(19, "too-few-paragraphs"), t(20, "c"), t(39, "last-alone")]
    fixed = outline.repair_topics(topics, paras, META)
    assert [x.title for x in fixed] == ["a", "b", "c"]


def test_repair_merges_topics_under_a_minute():
    paras = make_paragraphs(40, every=10.0)  # 3 paragraphs = 30 s
    meta = META.model_copy(update={"duration": 400})
    fixed = outline.repair_topics([t(0, "a"), t(10, "b"), t(13, "c")], paras, meta)
    assert [x.title for x in fixed] == ["a", "c"]


def test_short_first_topic_absorbs_next():
    fixed = outline.repair_topics([t(0, "a"), t(1, "b"), t(20, "c")], make_paragraphs(40), META)
    assert [(x.first_paragraph, x.title) for x in fixed] == [(0, "a"), (20, "c")]


def test_zero_topics_gives_single_topic_from_title():
    fixed = outline.repair_topics([t(50), t(-3)], make_paragraphs(10), META)
    assert len(fixed) == 1
    assert fixed[0].first_paragraph == 0
    assert fixed[0].title == "My great video"


def test_titles_lose_markdown_characters():
    fixed = outline.repair_topics([t(0, "## [Intro] | part ^1")], make_paragraphs(10), META)
    assert fixed[0].title == "Intro part 1"


def test_tags_are_sanitized():
    tags = [
        "#Machine Learning",
        "Python",
        "python",
        "  #  ",
        "2024",
        "C++ tips",
        "youtube",
        "a",
        "b",
        "c",
    ]
    assert outline.clean_tags(tags) == ["machine-learning", "python", "c-tips", "a", "b"]
    assert outline.clean_tags(["Нейронные сети"]) == ["нейронные-сети"]


@pytest.mark.parametrize(
    "minutes,expected",
    [(5, (1, 3)), (30, (4, 7)), (60, (6, 12)), (180, (15, 25))],
)
def test_target_topic_range(minutes, expected):
    assert outline.target_topic_range(minutes) == expected


def test_target_range_grows_with_length():
    ranges = [outline.target_topic_range(m) for m in (1, 10, 45, 120, 240)]
    assert ranges == sorted(ranges)


# ---------------------------------------------------------------------------
# chapters


def chaptered_meta(duration: float = 1200.0) -> VideoMeta:
    return META.model_copy(
        update={
            "duration": duration,
            "chapters": [
                Chapter(start=0, end=300, title="Вступление: зачем это всё"),
                Chapter(start=300, end=700, title="Как мы считали"),
                Chapter(start=700, end=1200, title="Итоги"),
            ],
        }
    )


def test_chapters_usable_rules():
    assert outline.chapters_usable(chaptered_meta())
    two = chaptered_meta().model_copy(update={"chapters": chaptered_meta().chapters[:2]})
    assert not outline.chapters_usable(two)
    not_covering = chaptered_meta(duration=3000)
    assert not outline.chapters_usable(not_covering)
    tiny = META.model_copy(
        update={
            "duration": 90,
            "chapters": [Chapter(start=i * 30, end=i * 30 + 30, title=str(i)) for i in range(3)],
        }
    )
    assert not outline.chapters_usable(tiny)


async def test_chapter_mode_keeps_creator_titles():
    paras = make_paragraphs(40)  # one every 30 s, video 20 min
    meta = chaptered_meta()

    def handler(fmt, user):
        assert fmt is ChapterOutlineDraft
        return ChapterOutlineDraft(
            tldr=" Кратко. ",
            key_points=["Первое", " "],
            topic_summaries=[
                TopicSummary(number=2, summary="Про подсчёт."),
                TopicSummary(number=1, summary="Начало."),
            ],
            tags=["#Финансы"],
        )

    llm = FakeLLM(handler)
    result = await outline.build_outline(paras, meta, "ru", llm, SETTINGS)
    assert [(x.title, x.first_paragraph) for x in result.topics] == [
        ("Вступление: зачем это всё", 0),
        ("Как мы считали", 10),
        ("Итоги", 24),
    ]
    assert [x.summary for x in result.topics] == ["Начало.", "Про подсчёт.", ""]
    assert result.tldr == "Кратко."
    assert result.key_points == ["Первое"]
    assert result.tags == ["финансы"]
    call = llm.calls[0]
    assert call["model"] == SETTINGS.outline_model and call["effort"] == SETTINGS.outline_effort
    assert "Russian" in call["user"]
    assert "1. Вступление: зачем это всё (paragraphs P0 to P9)" in call["user"]


async def test_free_mode_repairs_model_topics_and_gives_target():
    paras = make_paragraphs(40)

    def handler(fmt, user):
        assert fmt is OutlineDraft
        return OutlineDraft(
            tldr="Summary.",
            key_points=["One", "Two"],
            topics=[
                TopicDraft(title="Later", first_paragraph=20, summary="s2"),
                TopicDraft(title="Start", first_paragraph=2, summary="s1"),
                TopicDraft(title="Bogus", first_paragraph=400, summary=""),
            ],
            tags=["AI Safety", "#ml"],
        )

    llm = FakeLLM(handler)
    result = await outline.build_outline(paras, META, "en", llm, SETTINGS)
    assert [(x.title, x.first_paragraph) for x in result.topics] == [("Start", 0), ("Later", 20)]
    assert result.tags == ["ai-safety", "ml"]
    assert "[P0] (00:00)" in llm.calls[0]["user"]
    assert "aim for 3 to 5 topics" in llm.calls[0]["user"]


async def test_refusal_propagates():
    def handler(fmt, user):
        raise LlmRefused("declined")

    with pytest.raises(LlmRefused):
        await outline.build_outline(make_paragraphs(5), META, "en", FakeLLM(handler), SETTINGS)


def test_fallback_outline_uses_chapters_or_title():
    paras = make_paragraphs(40)
    with_chapters = outline.fallback_outline(paras, chaptered_meta())
    assert [x.title for x in with_chapters.topics] == [
        "Вступление: зачем это всё",
        "Как мы считали",
        "Итоги",
    ]
    assert with_chapters.tldr == "" and with_chapters.key_points == []
    plain = outline.fallback_outline(paras, META)
    assert [(x.title, x.first_paragraph) for x in plain.topics] == [("My great video", 0)]


def test_prompt_fingerprint_is_stable():
    assert outline.prompt_fingerprint() == outline.prompt_fingerprint()
    assert len(outline.prompt_fingerprint()) == 16
