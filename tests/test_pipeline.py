"""The whole pipeline with the outside world faked: no network, no paid calls."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest
import yaml

from archiver import pipeline
from archiver.clean import CleanedChunk, CleanedParagraph
from archiver.config import Settings
from archiver.errors import FetchBlocked, LlmRefused, NoteExists, PreflightError, SttFailed
from archiver.models import Chapter, Glossary, RawTranscript, Usage, VideoMeta, Word
from archiver.outline import (
    ChapterOutlineDraft,
    ConceptDraft,
    OutlineDraft,
    TopicDraft,
    TopicSummary,
)

VIDEO_ID = "abcdefghijk"
SENTENCE = "сегодня мы говорим про компиляторы и про то как они превращают текст программы в дерево"


def make_words(n_sentences: int, speakers: int = 1) -> list[Word]:
    words: list[Word] = []
    t = 0.0
    for i in range(n_sentences):
        tokens = f"ну вот {SENTENCE} номер {i}.".split()
        for token in tokens:
            words.append(Word(text=token, start=t, end=t + 0.3, speaker=f"speaker_{i % speakers}"))
            t += 0.35
        t += 1.0
    return words


def make_meta(duration: float, chapters: list[Chapter] | None = None, **extra) -> VideoMeta:
    return VideoMeta(
        video_id=VIDEO_ID,
        url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
        title="Как работают компиляторы: разбор",
        channel="Канал",
        upload_date=date(2026, 10, 1),
        duration=duration,
        chapters=chapters or [],
        **extra,
    )


class FakeTranscriber:
    provider = "fake"
    model = "fake-1"

    def __init__(self, words: list[Word], language: str | None = "ru") -> None:
        self.words = words
        self.language = language
        self.calls = 0
        self.audio_requested = 0

    def transcribe(self, source) -> RawTranscript:
        self.calls += 1
        source.audio_path()
        return RawTranscript(
            provider=self.provider, model=self.model, language=self.language, words=self.words
        )


class FakeLLM:
    """Answers each structured request the way a well-behaved model would."""

    def __init__(self, refuse_outline: bool = False) -> None:
        self.calls: list[type] = []
        self.usage: dict[str, Usage] = {}
        self.refuse_outline = refuse_outline
        self.concepts: list[ConceptDraft] = []

    def cost_usd(self) -> float | None:
        return 0.0

    async def parse(self, *, model, effort, system, user, output_format, max_tokens=16000):
        self.calls.append(output_format)
        entry = self.usage.setdefault(model, Usage(model=model))
        entry.calls += 1
        if output_format is Glossary:
            return Glossary()
        if output_format is CleanedChunk:
            return self._clean(user)
        if self.refuse_outline:
            raise LlmRefused("declined")
        n = len(re.findall(r"^\[P\d+\]", user, flags=re.M))
        if output_format is ChapterOutlineDraft:
            numbers = [int(x) for x in re.findall(r"^\s*(\d+)[.)]", user, flags=re.M)]
            return ChapterOutlineDraft(
                tldr="Коротко о компиляторах.",
                key_points=["Компилятор строит дерево."],
                topic_summaries=[TopicSummary(number=k, summary=f"раздел {k}") for k in numbers],
                tags=["Compilers", "#parsing"],
                concepts=self.concepts,
            )
        assert output_format is OutlineDraft
        return OutlineDraft(
            tldr="Коротко о компиляторах.",
            key_points=["Компилятор строит дерево."],
            topics=[
                TopicDraft(title="Начало", first_paragraph=0, summary="вступление"),
                TopicDraft(title="Дальше", first_paragraph=max(1, n // 2), summary="продолжение"),
            ],
            tags=["Compilers", "#parsing"],
            concepts=self.concepts,
        )

    @staticmethod
    def _clean(user: str) -> CleanedChunk:
        block = user.split("Segments to clean:\n", 1)[1].split("\n\n", 1)[0]
        paragraphs = []
        for line in block.splitlines():
            seg_id, text = re.match(r"\[(\d+)\] (?:\(\w+\) )?(.*)", line).groups()
            text = re.sub(r"^ну вот ", "", text)
            paragraphs.append(
                CleanedParagraph(start_segment=int(seg_id), text=text[0].upper() + text[1:])
            )
        return CleanedChunk(paragraphs=paragraphs)


def make_deps(meta, transcriber, llm, audio_calls: list | None = None) -> pipeline.Deps:
    def download_audio(m, dest: Path, settings) -> Path:
        if audio_calls is not None:
            audio_calls.append(1)
        path = dest / "audio.m4a"
        path.write_bytes(b"x")
        return path

    return pipeline.Deps(
        fetch_meta=lambda url, settings: meta,
        download_audio=download_audio,
        transcriber=transcriber,
        llm=llm,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    vault = tmp_path / "vault"
    vault.mkdir()
    return Settings(vault=vault, cache_dir=tmp_path / "cache", chunk_words=200)


def split_note(text: str) -> tuple[dict, str]:
    _, front, body = text.split("---\n", 2)
    return yaml.safe_load(front), body


async def test_run_writes_a_complete_note(settings):
    words = make_words(60)
    meta = make_meta(duration=words[-1].end)
    llm = FakeLLM()
    events: list[tuple[str, str]] = []
    result = await pipeline.run(
        meta.url,
        settings,
        deps=make_deps(meta, FakeTranscriber(words), llm),
        on_progress=lambda stage, msg: events.append((stage, msg)),
    )

    assert result.path.parent == settings.vault / "Videos"
    assert result.path.name == f"2026-10-01 Как работают компиляторы разбор ({VIDEO_ID}).md"
    assert result.warnings == []
    front, body = split_note(result.path.read_text(encoding="utf-8"))
    assert front["title"] == meta.title
    assert front["video_id"] == VIDEO_ID
    assert front["language"] == "ru"
    assert front["published"] == date(2026, 10, 1)
    assert front["tags"] == ["video", "youtube", "compilers", "parsing"]
    assert "> [!summary] TL;DR" in body
    assert "## Начало" in body and "## Дальше" in body
    assert f"https://youtu.be/{VIDEO_ID}?t=0" in body
    # Every sentence survived cleanup, and the filler was removed.
    assert all(f"номер {i}." in body for i in range(60))
    assert "ну вот" not in body
    assert llm.calls.count(CleanedChunk) > 1  # the transcript really was chunked
    assert [s for s, _ in events][0] == "fetch" and events[-1][0] == "done"


async def test_second_run_makes_no_paid_calls(settings):
    words = make_words(60)
    meta = make_meta(duration=words[-1].end)
    first_stt, first_llm = FakeTranscriber(words), FakeLLM()
    first = await pipeline.run(meta.url, settings, deps=make_deps(meta, first_stt, first_llm))

    stt, llm, audio = FakeTranscriber(words), FakeLLM(), []
    second = await pipeline.run(meta.url, settings, deps=make_deps(meta, stt, llm, audio))
    assert second.path == first.path
    assert stt.calls == 0 and llm.calls == [] and audio == []

    stt, llm = FakeTranscriber(words), FakeLLM()
    await pipeline.run(meta.url, settings, force=True, deps=make_deps(meta, stt, llm))
    assert stt.calls == 1 and len(llm.calls) == len(first_llm.calls)


async def test_creator_chapters_become_topics(settings):
    words = make_words(120)
    end = words[-1].end
    chapters = [
        Chapter(start=0, end=end / 3, title="Лексер"),
        Chapter(start=end / 3, end=2 * end / 3, title="Парсер: дерево"),
        Chapter(start=2 * end / 3, end=end, title="Генерация кода"),
    ]
    meta = make_meta(duration=end, chapters=chapters)
    llm = FakeLLM()
    result = await pipeline.run(meta.url, settings, deps=make_deps(meta, FakeTranscriber(words), llm))
    body = split_note(result.path.read_text(encoding="utf-8"))[1]
    assert ChapterOutlineDraft in llm.calls and OutlineDraft not in llm.calls
    for title in ("## Лексер", "## Парсер: дерево", "## Генерация кода"):
        assert title in body


async def test_two_speakers_get_labels(settings):
    words = make_words(20, speakers=2)
    meta = make_meta(duration=words[-1].end)
    result = await pipeline.run(
        meta.url, settings, deps=make_deps(meta, FakeTranscriber(words), FakeLLM())
    )
    body = split_note(result.path.read_text(encoding="utf-8"))[1]
    assert "**Спикер 1:**" in body and "**Спикер 2:**" in body


async def test_refused_outline_still_writes_the_transcript(settings):
    words = make_words(30)
    meta = make_meta(duration=words[-1].end)
    result = await pipeline.run(
        meta.url, settings, deps=make_deps(meta, FakeTranscriber(words), FakeLLM(refuse_outline=True))
    )
    body = split_note(result.path.read_text(encoding="utf-8"))[1]
    assert len(result.warnings) == 1
    assert "> [!warning]" in body and "TL;DR" not in body
    assert all(f"номер {i}." in body for i in range(30))


async def test_edited_note_is_refused_before_any_paid_call(settings):
    words = make_words(30)
    meta = make_meta(duration=words[-1].end)
    first = await pipeline.run(
        meta.url, settings, deps=make_deps(meta, FakeTranscriber(words), FakeLLM())
    )
    first.path.write_text(first.path.read_text(encoding="utf-8") + "\nmy own notes\n", encoding="utf-8")

    stt, llm = FakeTranscriber(words), FakeLLM()
    with pytest.raises(NoteExists):
        await pipeline.run(meta.url, settings, force=True, deps=make_deps(meta, stt, llm))
    assert stt.calls == 0 and llm.calls == []
    assert "my own notes" in first.path.read_text(encoding="utf-8")

    await pipeline.run(meta.url, settings, overwrite=True, deps=make_deps(meta, stt, llm))
    assert "my own notes" not in first.path.read_text(encoding="utf-8")


async def test_live_stream_too_long_and_silent_videos_stop_early(settings):
    llm = FakeLLM()
    live = make_meta(duration=60, is_live=True)
    with pytest.raises(FetchBlocked):
        await pipeline.run(live.url, settings, deps=make_deps(live, FakeTranscriber([]), llm))

    long = make_meta(duration=5 * 3600)
    stt = FakeTranscriber(make_words(5))
    with pytest.raises(PreflightError):
        await pipeline.run(long.url, settings, deps=make_deps(long, stt, llm))
    assert stt.calls == 0

    silent = make_meta(duration=60)
    with pytest.raises(SttFailed):
        await pipeline.run(silent.url, settings, deps=make_deps(silent, FakeTranscriber([]), llm))
    assert llm.calls == []


def test_preflight_lists_everything_missing(tmp_path, monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ELEVENLABS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    monkeypatch.setattr(pipeline.fetch, "available_js_runtimes", lambda: [])
    with pytest.raises(PreflightError) as exc:
        pipeline.preflight(Settings(vault=tmp_path / "missing", cache_dir=tmp_path))
    message = str(exc.value)
    for expected in ("Vault folder", "ELEVENLABS_API_KEY", "Anthropic", "JavaScript", "ffmpeg"):
        assert expected in message


async def test_note_links_to_existing_vault_notes_and_reuses_tags(settings):
    vault = settings.vault
    (vault / "People").mkdir()
    (vault / "People" / "Ник Ланд.md").write_text(
        "---\naliases:\n  - Nick Land\ntags:\n  - философия\n---\nтекст\n", encoding="utf-8"
    )
    (vault / "Компилятор.md").write_text("Про компиляторы. #Compilers и #языки\n", encoding="utf-8")
    (vault / "Anchor" / "Journal").mkdir(parents=True)
    (vault / "Anchor" / "Journal" / "Дерево.md").write_text("личное\n", encoding="utf-8")

    words = make_words(30)
    meta = make_meta(duration=words[-1].end)
    llm = FakeLLM()
    llm.concepts = [
        ConceptDraft(names=["Компиляторы", "Компилятор", "Compiler"], relation="Главная тема."),
        ConceptDraft(names=["Nick Land"], relation="Упомянут как пример."),
        ConceptDraft(names=["Дерево"], relation="Не должно связаться: личная папка."),
        ConceptDraft(names=["Философия"], relation="Нет такой заметки, но есть такой тег."),
    ]
    result = await pipeline.run(meta.url, settings, deps=make_deps(meta, FakeTranscriber(words), llm))
    front, body = split_note(result.path.read_text(encoding="utf-8"))

    assert "## Связанные заметки" in body
    assert "- [[Компилятор]]: Главная тема." in body
    assert "- [[Ник Ланд]]: Упомянут как пример." in body
    assert "Дерево" not in body.split("## Содержание")[0]
    # Existing tags only, spelled as the vault spells them; the model's unknown "parsing" is dropped.
    assert front["tags"] == ["video", "youtube", "философия", "Compilers"]

    # A second run links the same way and never links the note to itself.
    again = await pipeline.run(meta.url, settings, deps=make_deps(meta, FakeTranscriber(words), llm))
    assert again.path.read_text(encoding="utf-8").count("[[") == 2


async def test_linking_can_be_switched_off(settings):
    (settings.vault / "Компилятор.md").write_text("x\n", encoding="utf-8")
    words = make_words(30)
    meta = make_meta(duration=words[-1].end)
    llm = FakeLLM()
    llm.concepts = [ConceptDraft(names=["Компилятор"], relation="Тема.")]
    off = settings.model_copy(update={"link_notes": False})
    result = await pipeline.run(meta.url, off, deps=make_deps(meta, FakeTranscriber(words), llm))
    front, body = split_note(result.path.read_text(encoding="utf-8"))
    assert "[[" not in body
    assert front["tags"] == ["video", "youtube", "compilers", "parsing"]
