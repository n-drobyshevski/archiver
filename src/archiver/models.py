"""Data shapes passed between pipeline stages. Every stage reads and returns these."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Chapter(BaseModel):
    """A creator-defined chapter from YouTube. Times are seconds from the start."""

    start: float
    end: float
    title: str


class VideoMeta(BaseModel):
    video_id: str
    url: str
    title: str
    channel: str = ""
    upload_date: date | None = None
    duration: float = 0.0  # seconds
    description: str = ""
    chapters: list[Chapter] = Field(default_factory=list)
    is_live: bool = False


class Word(BaseModel):
    """One spoken word from speech-to-text, already stripped of non-speech events."""

    text: str
    start: float
    end: float
    speaker: str | None = None


class RawTranscript(BaseModel):
    """Provider-neutral speech-to-text result."""

    provider: str
    model: str
    language: str | None = None  # ISO 639-1, e.g. "ru", "en"
    words: list[Word]


class Segment(BaseModel):
    """A sentence-sized run of words (roughly 5 to 15 seconds). `id` is its index in the list."""

    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None


class Paragraph(BaseModel):
    """A cleaned paragraph. `start_segment` is the id of the first segment it was made from;
    `start` and `speaker` are copied from that segment by code, never by the model."""

    start_segment: int
    start: float
    speaker: str | None = None
    text: str


class GlossaryEntry(BaseModel):
    canonical: str
    variants: list[str] = Field(default_factory=list)


class Glossary(BaseModel):
    entries: list[GlossaryEntry] = Field(default_factory=list)


class ChunkReport(BaseModel):
    """Fidelity numbers for one cleanup chunk, kept so thresholds can be tuned."""

    index: int
    first_segment: int
    last_segment: int
    recall: float | None = None  # share of source content words that survived
    novelty: float | None = None  # share of cleaned words not found in the source
    longest_gap: int | None = None  # longest run of source content words that vanished
    status: Literal["ok", "retried", "raw_fallback"] = "ok"
    reason: str = ""


class CleanResult(BaseModel):
    paragraphs: list[Paragraph]
    chunks: list[ChunkReport] = Field(default_factory=list)


class Topic(BaseModel):
    title: str
    first_paragraph: int  # index into the paragraph list
    summary: str = ""


class Outline(BaseModel):
    tldr: str = ""
    key_points: list[str] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    """Token use per model, summed over a run."""

    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
