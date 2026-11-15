"""Cleanup stage: glossary pre-pass, chunking, per-chunk cleanup calls with a fidelity guard."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from archiver import fidelity
from archiver.cache import Cache, fingerprint
from archiver.config import Settings
from archiver.errors import LlmRefused, LlmTruncated
from archiver.llm import LLM, load_prompt
from archiver.models import (
    ChunkReport,
    CleanResult,
    Glossary,
    Paragraph,
    Segment,
    VideoMeta,
)

Progress = Callable[[str, str], None]

HARD_CAP_WORDS = 2000
WINDOW = 0.2  # cut points are searched within ±20% of the target size
MIN_LAST_SHARE = 0.4  # a last chunk below this share of the target is merged back
CONTEXT_BEFORE_WORDS = 150
CONTEXT_AFTER_WORDS = 50
PARAGRAPH_WORDS = 120
LONG_PAUSE = 2.0
SHORT_PAUSE = 1.0
GLOSSARY_MAX = 50
DESCRIPTION_CHARS = 4000
CHAPTER_TOLERANCE = 1.0  # seconds; segment starts rarely line up exactly with chapter marks

_SENTENCE_END = re.compile(r"[.!?…][\"'»)\]]*\s*$")


# Model output shapes. Code fills `start` and `speaker`; the model only says where it started.
class CleanedParagraph(BaseModel):
    start_segment: int
    text: str


class CleanedChunk(BaseModel):
    paragraphs: list[CleanedParagraph]


# ---------------------------------------------------------------------------
# Raw fallback


def _word_count(text: str) -> int:
    return len(text.split())


def _paragraph_from(segments: list[Segment]) -> Paragraph:
    first = segments[0]
    text = " ".join(s.text.strip() for s in segments if s.text.strip())
    return Paragraph(start_segment=first.id, start=first.start, speaker=first.speaker, text=text)


def raw_paragraphs(segments: list[Segment]) -> list[Paragraph]:
    """Group raw segments into paragraphs: break on speaker change, a long pause or ~120 words."""
    paragraphs: list[Paragraph] = []
    current: list[Segment] = []
    words = 0
    for seg in segments:
        if current and (
            seg.speaker != current[-1].speaker
            or seg.start - current[-1].end >= LONG_PAUSE
            or words >= PARAGRAPH_WORDS
        ):
            paragraphs.append(_paragraph_from(current))
            current, words = [], 0
        current.append(seg)
        words += _word_count(seg.text)
    if current:
        paragraphs.append(_paragraph_from(current))
    return paragraphs


# ---------------------------------------------------------------------------
# Chunker. Positions are indexes into the segment list; a cut at j splits before segments[j].


def _chapter_positions(segments: list[Segment], meta: VideoMeta) -> set[int]:
    positions: set[int] = set()
    for chapter in meta.chapters:
        if chapter.start <= CHAPTER_TOLERANCE:
            continue
        for j, seg in enumerate(segments):
            if seg.start >= chapter.start - CHAPTER_TOLERANCE:
                if j > 0:
                    positions.add(j)
                break
    return positions


def boundary_scores(segments: list[Segment], meta: VideoMeta) -> list[int]:
    """How good a cut before each segment is: chapter 5, speaker 4, pause ≥2 s 3, ≥1 s 2,
    sentence end 1, otherwise 0. Index 0 is never a cut."""
    chapters = _chapter_positions(segments, meta)
    scores = [0] * len(segments)
    for j in range(1, len(segments)):
        prev, cur = segments[j - 1], segments[j]
        pause = cur.start - prev.end
        if j in chapters:
            scores[j] = 5
        elif prev.speaker != cur.speaker:
            scores[j] = 4
        elif pause >= LONG_PAUSE:
            scores[j] = 3
        elif pause >= SHORT_PAUSE:
            scores[j] = 2
        elif _SENTENCE_END.search(prev.text):
            scores[j] = 1
    return scores


def _prefix_words(segments: list[Segment]) -> list[int]:
    prefix = [0]
    for seg in segments:
        prefix.append(prefix[-1] + _word_count(seg.text))
    return prefix


def _best_cut(start: int, end: int, want: float, prefix: list[int], scores: list[int]) -> int:
    """Best cut strictly inside (start, end) for a first piece of about `want` words."""
    lo = want * (1 - WINDOW)
    hi = min(want * (1 + WINDOW), HARD_CAP_WORDS)
    inside = range(start + 1, end)
    window = [j for j in inside if lo <= prefix[j] - prefix[start] <= hi]
    if window:
        return max(window, key=lambda j: (scores[j], -abs(prefix[j] - prefix[start] - want)))
    # No segment boundary falls in the window (very long segments): stay under the cap.
    under_cap = [j for j in inside if prefix[j] - prefix[start] <= HARD_CAP_WORDS]
    pool = under_cap or [start + 1]
    return min(pool, key=lambda j: abs(prefix[j] - prefix[start] - want))


def plan_chunks(
    segments: list[Segment], meta: VideoMeta, target_words: int
) -> list[tuple[int, int]]:
    """Split segments into [start, end) position ranges of about `target_words` words."""
    if not segments:
        return []
    target = max(1, min(target_words, HARD_CAP_WORDS))
    prefix = _prefix_words(segments)
    scores = boundary_scores(segments, meta)
    n = len(segments)
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < n:
        if prefix[n] - prefix[start] <= min(target * (1 + WINDOW), HARD_CAP_WORDS):
            bounds.append((start, n))
            break
        cut = _best_cut(start, n, target, prefix, scores)
        bounds.append((start, cut))
        start = cut
    return _absorb_tiny_tail(bounds, target, prefix, scores)


def _absorb_tiny_tail(
    bounds: list[tuple[int, int]], target: int, prefix: list[int], scores: list[int]
) -> list[tuple[int, int]]:
    if len(bounds) < 2:
        return bounds
    (a, _), (_, e) = bounds[-2], bounds[-1]
    last_words = prefix[e] - prefix[bounds[-1][0]]
    if last_words >= target * MIN_LAST_SHARE:
        return bounds
    combined = prefix[e] - prefix[a]
    if combined <= HARD_CAP_WORDS:
        return bounds[:-2] + [(a, e)]
    # Too big to merge: split the last two evenly instead.
    if e - a < 2:
        return bounds
    mid = _best_cut(a, e, combined / 2, prefix, scores)
    return bounds[:-2] + [(a, mid), (mid, e)]


# ---------------------------------------------------------------------------
# Glossary


def _raw_text(segments: list[Segment]) -> str:
    return " ".join(s.text.strip() for s in segments)


def _meta_block(meta: VideoMeta) -> str:
    lines = [f"Title: {meta.title}"]
    if meta.channel:
        lines.append(f"Channel: {meta.channel}")
    if meta.chapters:
        lines.append("Chapters: " + "; ".join(c.title for c in meta.chapters))
    if meta.description.strip():
        lines.append("Description:\n" + meta.description.strip()[:DESCRIPTION_CHARS])
    return "\n".join(lines)


def _sanitize_glossary(glossary: Glossary) -> Glossary:
    entries = [e for e in glossary.entries if e.canonical.strip()][:GLOSSARY_MAX]
    return Glossary(entries=entries)


async def build_glossary(
    segments: list[Segment], meta: VideoMeta, llm: LLM, settings: Settings, cache: Cache
) -> Glossary:
    prompt = load_prompt("glossary")
    meta_text = _meta_block(meta)
    transcript = _raw_text(segments)
    fp = fingerprint(prompt, settings.clean_model, settings.clean_effort, meta_text, transcript)
    cached = cache.load("glossary", fp, Glossary)
    if cached is not None:
        return cached
    user = f"{meta_text}\n\nTranscript:\n{transcript}"
    try:
        result = await llm.parse(
            model=settings.clean_model,
            effort=settings.clean_effort,
            system=prompt,
            user=user,
            output_format=Glossary,
            max_tokens=8000,
        )
    except (LlmRefused, LlmTruncated):
        return Glossary()  # cleanup still works without it, just less consistently
    glossary = _sanitize_glossary(result)
    cache.save("glossary", fp, glossary)
    return glossary


def glossary_text(glossary: Glossary) -> str:
    if not glossary.entries:
        return "(none)"
    lines = []
    for entry in glossary.entries:
        variants = [v for v in entry.variants if v.strip() and v != entry.canonical]
        heard = f" (heard as: {', '.join(variants)})" if variants else ""
        lines.append(f"- {entry.canonical}{heard}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation and repair of the model's paragraph ids


def repair_paragraph_ids(
    ids: list[int], segment_ids: list[int], alignment: fidelity.Alignment
) -> list[int]:
    """Make ids valid: first equals the chunk start, all inside the chunk, never decreasing.

    A bad id is replaced by the segment the paragraph's words align to, not by a retry.
    """
    valid = set(segment_ids)
    first = segment_ids[0]
    out: list[int] = []
    prev = first
    for i, pid in enumerate(ids):
        if i == 0:
            pid = first
        elif pid not in valid or pid < prev:
            guess = alignment.source_ref_for(i)
            pid = guess if guess is not None and guess in valid else prev
            pid = max(pid, prev)
        out.append(pid)
        prev = pid
    return out


def build_paragraphs(
    span: list[Segment], output: CleanedChunk
) -> tuple[list[Paragraph], fidelity.FidelityReport]:
    items = [(p.start_segment, p.text.strip()) for p in output.paragraphs if p.text.strip()]
    alignment = fidelity.align(
        [(s.id, s.text) for s in span], [(i, text) for i, (_, text) in enumerate(items)]
    )
    ids = repair_paragraph_ids([pid for pid, _ in items], [s.id for s in span], alignment)
    by_id = {s.id: s for s in span}
    paragraphs = [
        Paragraph(start_segment=pid, start=by_id[pid].start, speaker=by_id[pid].speaker, text=text)
        for pid, (_, text) in zip(ids, items)
    ]
    return paragraphs, alignment.report()


# ---------------------------------------------------------------------------
# Per-chunk cleanup


@dataclass
class _Attempt:
    paragraphs: list[Paragraph] | None  # None when the attempt failed
    report: fidelity.FidelityReport | None = None
    failure: str = ""
    refused: bool = False


class _Cleaner:
    def __init__(
        self,
        segments: list[Segment],
        meta: VideoMeta,
        llm: LLM,
        settings: Settings,
        cache: Cache,
        glossary: Glossary,
    ) -> None:
        self.segments = segments
        self.meta = meta
        self.llm = llm
        self.settings = settings
        self.cache = cache
        self.glossary = glossary_text(glossary)
        self.prompt = load_prompt("clean")
        self.prefix = _prefix_words(segments)
        self.scores = boundary_scores(segments, meta)
        self.label_speakers = len({s.speaker for s in segments if s.speaker}) > 1
        self.semaphore = asyncio.Semaphore(max(1, settings.concurrency))

    # -- input text

    def _lines(self, start: int, end: int) -> str:
        lines = []
        for j in range(start, end):
            seg = self.segments[j]
            changed = j == start or seg.speaker != self.segments[j - 1].speaker
            label = f"({seg.speaker}) " if self.label_speakers and changed and seg.speaker else ""
            lines.append(f"[{seg.id}] {label}{seg.text.strip()}")
        return "\n".join(lines)

    def _context(self, start: int, end: int) -> tuple[str, str]:
        before = _raw_text(self.segments[:start]).split()[-CONTEXT_BEFORE_WORDS:]
        after = _raw_text(self.segments[end:]).split()[:CONTEXT_AFTER_WORDS]
        return " ".join(before), " ".join(after)

    def _user_message(self, start: int, end: int, retry_reason: str) -> str:
        before, after = self._context(start, end)
        parts = [
            f"Video title: {self.meta.title}",
            f"Glossary (preferred spellings, with forms the recognizer produced):\n{self.glossary}",
        ]
        if before:
            parts.append(f"Context before (read-only, do not output it):\n<<<\n{before}\n>>>")
        parts.append(f"Segments to clean:\n{self._lines(start, end)}")
        if after:
            parts.append(f"Context after (read-only, do not output it):\n<<<\n{after}\n>>>")
        if retry_reason:
            parts.append(
                "A previous attempt at a larger piece containing these segments was rejected by "
                f"the source comparison: {retry_reason}. Stay closer to the speaker's words: keep "
                "every passage, do not condense, and do not add wording of your own."
            )
        return "\n\n".join(parts)

    # -- model call, cached by everything that shapes it

    async def _call(self, start: int, end: int, retry_reason: str) -> CleanedChunk:
        user = self._user_message(start, end, retry_reason)
        model, effort = self.settings.clean_model, self.settings.clean_effort
        key = fingerprint(user, self.prompt, model, effort)
        name = f"clean/{key}"
        cached = self.cache.load(name, key, CleanedChunk)
        if cached is not None:
            return cached
        async with self.semaphore:
            result = await self.llm.parse(
                model=model,
                effort=effort,
                system=self.prompt,
                user=user,
                output_format=CleanedChunk,
            )
        self.cache.save(name, key, result)
        return result

    async def _attempt(self, start: int, end: int, retry_reason: str = "") -> _Attempt:
        try:
            output = await self._call(start, end, retry_reason)
        except LlmRefused as exc:
            return _Attempt(None, failure=str(exc), refused=True)
        except LlmTruncated as exc:
            return _Attempt(None, failure=f"output cut off ({exc})")
        span = self.segments[start:end]
        paragraphs, report = build_paragraphs(span, output)
        if not paragraphs:
            return _Attempt(None, failure="empty result")
        if not report.ok:
            return _Attempt(None, report, failure=report.reason)
        return _Attempt(paragraphs, report)

    async def clean_chunk(
        self, index: int, start: int, end: int
    ) -> tuple[list[Paragraph], ChunkReport]:
        span = self.segments[start:end]
        report = ChunkReport(index=index, first_segment=span[0].id, last_segment=span[-1].id)
        first = await self._attempt(start, end)
        if first.paragraphs is not None:
            _set_numbers(report, first.report)
            return first.paragraphs, report

        if first.refused or end - start < 2:
            _set_numbers(report, first.report)
            report.status, report.reason = "raw_fallback", first.failure
            return raw_paragraphs(span), report

        mid = _best_cut(
            start, end, (self.prefix[end] - self.prefix[start]) / 2, self.prefix, self.scores
        )
        halves = await asyncio.gather(
            self._attempt(start, mid, first.failure), self._attempt(mid, end, first.failure)
        )
        paragraphs: list[Paragraph] = []
        failures: list[str] = []
        for (a, b), half in zip(((start, mid), (mid, end)), halves):
            if half.paragraphs is None:
                paragraphs.extend(raw_paragraphs(self.segments[a:b]))
                failures.append(half.failure)
            else:
                paragraphs.extend(half.paragraphs)

        if failures:
            # Keep the numbers that triggered the retry: they are what the thresholds judged.
            _set_numbers(report, first.report)
            report.status = "raw_fallback"
            report.reason = f"first try: {first.failure}; half retry: {'; '.join(failures)}"
        else:
            combined = fidelity.check(
                [(s.id, s.text) for s in span], [(i, p.text) for i, p in enumerate(paragraphs)]
            )
            _set_numbers(report, combined)
            report.status = "retried"
            report.reason = f"first try: {first.failure}"
        return paragraphs, report


def _set_numbers(report: ChunkReport, numbers: fidelity.FidelityReport | None) -> None:
    if numbers is None:
        return
    report.recall = round(numbers.recall, 4)
    report.novelty = round(numbers.novelty, 4)
    report.longest_gap = numbers.longest_gap


# ---------------------------------------------------------------------------
# Entry point


async def clean_transcript(
    segments: list[Segment],
    meta: VideoMeta,
    llm: LLM,
    settings: Settings,
    cache: Cache,
    on_progress: Progress | None = None,
) -> CleanResult:
    if not segments:
        return CleanResult(paragraphs=[])
    bounds = plan_chunks(segments, meta, settings.chunk_words)
    glossary = Glossary()
    if len(bounds) > 1:
        if on_progress:
            on_progress("clean", "building a glossary of names and terms")
        glossary = await build_glossary(segments, meta, llm, settings, cache)

    cleaner = _Cleaner(segments, meta, llm, settings, cache, glossary)
    total = len(bounds)
    done = 0

    async def run(index: int, start: int, end: int) -> tuple[list[Paragraph], ChunkReport]:
        nonlocal done
        result = await cleaner.clean_chunk(index, start, end)
        done += 1
        if on_progress:
            on_progress("clean", f"chunk {done}/{total}")
        return result

    results = await asyncio.gather(*(run(i, s, e) for i, (s, e) in enumerate(bounds)))
    return CleanResult(
        paragraphs=[p for paragraphs, _ in results for p in paragraphs],
        chunks=[report for _, report in results],
    )
