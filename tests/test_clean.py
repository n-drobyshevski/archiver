import itertools
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import ValidationError

from archiver import clean
from archiver.cache import Cache
from archiver.clean import CleanedChunk, CleanedParagraph
from archiver.config import Settings
from archiver.errors import LlmFailed, LlmRefused, LlmTruncated
from archiver.llm import FALLBACK_BETA, LLM
from archiver.models import Chapter, Glossary, GlossaryEntry, Segment, VideoMeta

LINE = re.compile(r"^\[(\d+)\] (?:\([^)]*\) )?(.*)$")


class FakeLLM:
    """Same surface as archiver.llm.LLM. `handler(output_format, user)` returns the result."""

    def __init__(self, handler: Callable[[type, str], Any]) -> None:
        self.handler = handler
        self.calls: list[tuple[type, str]] = []
        self.usage: dict = {}

    async def parse(self, *, model, effort, system, user, output_format, max_tokens=16000):
        self.calls.append((output_format, user))
        return self.handler(output_format, user)

    def cost_usd(self):
        return 0.0


def chunk_lines(user: str) -> list[tuple[int, str]]:
    block = user.split("Segments to clean:\n", 1)[1].split("\n\n", 1)[0]
    return [(int(m[1]), m[2]) for m in map(LINE.match, block.splitlines()) if m]


def faithful(lines: list[tuple[int, str]], per_paragraph: int = 3) -> CleanedChunk:
    paragraphs = []
    for i in range(0, len(lines), per_paragraph):
        group = lines[i : i + per_paragraph]
        text = " ".join(t for _, t in group).capitalize() + "."
        paragraphs.append(CleanedParagraph(start_segment=group[0][0], text=text))
    return CleanedChunk(paragraphs=paragraphs)


def default_handler(output_format: type, user: str) -> Any:
    if output_format is Glossary:
        return Glossary(entries=[GlossaryEntry(canonical="Kubernetes", variants=["cubernetes"])])
    return faithful(chunk_lines(user))


def make_segments(n: int, words: int = 10, *, gap: float = 0.2) -> list[Segment]:
    segments = []
    t = 0.0
    for i in range(n):
        text = " ".join(f"w{i}x{k}" for k in range(words))
        segments.append(Segment(id=i, start=t, end=t + 4.0, text=text, speaker="speaker_0"))
        t += 4.0 + gap
    return segments


META = VideoMeta(video_id="vid", url="https://youtu.be/vid", title="Test video")


@pytest.fixture
def settings(tmp_path):
    return Settings(cache_dir=tmp_path, chunk_words=1350, concurrency=3)


@pytest.fixture
def cache(tmp_path):
    return Cache(tmp_path, "vid")


def words_in(segments, a, b):
    return sum(len(s.text.split()) for s in segments[a:b])


# ---------------------------------------------------------------------------
# raw paragraphs


def test_raw_paragraphs_break_on_speaker_pause_and_length():
    segs = make_segments(30, words=10)
    segs[5] = segs[5].model_copy(update={"speaker": "speaker_1"})
    segs[6] = segs[6].model_copy(update={"speaker": "speaker_1"})
    segs[20] = segs[20].model_copy(update={"start": segs[19].end + 3.0})
    paras = clean.raw_paragraphs(segs)
    starts = [p.start_segment for p in paras]
    assert 5 in starts and 7 in starts and 20 in starts
    assert all(len(p.text.split()) <= 130 for p in paras)
    assert " ".join(p.text for p in paras).split() == " ".join(s.text for s in segs).split()


# ---------------------------------------------------------------------------
# chunker


def test_chunks_respect_window_and_cap():
    segs = make_segments(500, words=10)  # 5000 words
    bounds = clean.plan_chunks(segs, META, 1350)
    assert bounds[0][0] == 0 and bounds[-1][1] == 500
    assert all(a[1] == b[0] for a, b in itertools.pairwise(bounds))
    for a, b in bounds[:-1]:
        assert 1080 <= words_in(segs, a, b) <= 1620
    assert all(words_in(segs, a, b) <= clean.HARD_CAP_WORDS for a, b in bounds)


def test_chunker_prefers_chapter_over_speaker_and_pause():
    segs = make_segments(300, words=10)
    segs[125] = segs[125].model_copy(update={"start": segs[124].end + 5.0})  # long pause
    for j in range(130, 300):
        segs[j] = segs[j].model_copy(update={"speaker": "speaker_1"})  # speaker change at 130
    meta = META.model_copy(
        update={
            "chapters": [
                Chapter(start=0, end=100, title="A"),
                Chapter(start=segs[140].start, end=9999, title="B"),
            ]
        }
    )
    assert clean.plan_chunks(segs, meta, 1350)[0] == (0, 140)
    assert clean.plan_chunks(segs, META, 1350)[0] == (0, 130)


def test_chunker_prefers_long_pause_then_sentence_end():
    segs = make_segments(300, words=10)
    segs[127] = segs[127].model_copy(update={"text": segs[127].text + "."})
    segs[133] = segs[133].model_copy(update={"start": segs[132].end + 2.5})
    assert clean.plan_chunks(segs, META, 1350)[0] == (0, 133)
    segs[133] = segs[133].model_copy(update={"start": segs[132].end + 0.1})
    assert clean.plan_chunks(segs, META, 1350)[0] == (0, 128)


def test_tiny_last_chunk_is_merged():
    segs = make_segments(170, words=10)  # 1700 words: 1350 + 350 would leave a tiny tail
    assert clean.plan_chunks(segs, META, 1350) == [(0, 170)]


def test_tiny_tail_too_big_to_merge_is_rebalanced():
    segs = make_segments(290, words=10)  # 2900 words
    bounds = clean.plan_chunks(segs, META, 1350)
    sizes = [words_in(segs, a, b) for a, b in bounds]
    assert all(s <= clean.HARD_CAP_WORDS for s in sizes)
    assert min(sizes) >= 0.4 * 1350


def test_single_huge_segments_still_respect_cap_where_possible():
    segs = make_segments(10, words=900)
    bounds = clean.plan_chunks(segs, META, 1350)
    assert all(b - a >= 1 for a, b in bounds)
    assert all(words_in(segs, a, b) <= clean.HARD_CAP_WORDS for a, b in bounds)


# ---------------------------------------------------------------------------
# id validation and repair


def test_paragraph_ids_are_repaired_from_alignment():
    segs = [
        Segment(id=10, start=0, end=3, text="мы запустили проект в прошлом году"),
        Segment(id=11, start=3, end=6, text="никто не верил что это взлетит"),
        Segment(id=12, start=6, end=9, text="но за год пришли десять тысяч человек"),
        Segment(id=13, start=9, end=12, text="причём совсем без рекламы"),
    ]
    output = CleanedChunk(
        paragraphs=[
            CleanedParagraph(start_segment=11, text="Мы запустили проект в прошлом году."),
            CleanedParagraph(start_segment=99, text="Никто не верил, что это взлетит."),
            CleanedParagraph(start_segment=5, text="Но за год пришли десять тысяч человек,"),
            CleanedParagraph(start_segment=13, text="причём совсем без рекламы."),
            CleanedParagraph(start_segment=12, text="   "),
        ]
    )
    paragraphs, report = clean.build_paragraphs(segs, output)
    assert [p.start_segment for p in paragraphs] == [10, 11, 12, 13]
    assert [p.start for p in paragraphs] == [0, 3, 6, 9]
    assert report.ok


def test_decreasing_id_is_repaired():
    segs = make_segments(6, words=5)
    lines = [(s.id, s.text) for s in segs]
    output = CleanedChunk(
        paragraphs=[
            CleanedParagraph(start_segment=0, text=" ".join(t for _, t in lines[:2])),
            CleanedParagraph(start_segment=2, text=" ".join(t for _, t in lines[2:4])),
            CleanedParagraph(start_segment=1, text=" ".join(t for _, t in lines[4:])),
        ]
    )
    paragraphs, _ = clean.build_paragraphs(segs, output)
    # The third id goes backwards; its words align to segment 4.
    assert [p.start_segment for p in paragraphs] == [0, 2, 4]


# ---------------------------------------------------------------------------
# full stage with a fake LLM


async def test_clean_transcript_ok_with_glossary_and_progress(settings, cache):
    segs = make_segments(300, words=10)
    llm = FakeLLM(default_handler)
    progress = []
    result = await clean.clean_transcript(
        segs, META, llm, settings, cache, on_progress=lambda s, m: progress.append((s, m))
    )
    n_chunks = len(result.chunks)
    assert n_chunks == 2
    assert llm.calls[0][0] is Glossary
    assert "Kubernetes (heard as: cubernetes)" in llm.calls[1][1]
    assert all(c.status == "ok" and c.recall == 1.0 for c in result.chunks)
    assert result.paragraphs[0].start_segment == 0
    ids = [p.start_segment for p in result.paragraphs]
    assert ids == sorted(ids)
    assert ("clean", f"chunk {n_chunks}/{n_chunks}") in progress
    # Context is given but marked read-only.
    assert (
        "Context before (read-only" in llm.calls[-1][1]
        or "Context before (read-only" in llm.calls[1][1]
    )


async def test_single_chunk_skips_glossary(settings, cache):
    llm = FakeLLM(default_handler)
    await clean.clean_transcript(make_segments(50), META, llm, settings, cache)
    assert [fmt for fmt, _ in llm.calls] == [CleanedChunk]


async def test_glossary_refusal_continues(settings, cache):
    def handler(fmt, user):
        if fmt is Glossary:
            raise LlmRefused("declined")
        return default_handler(fmt, user)

    llm = FakeLLM(handler)
    result = await clean.clean_transcript(make_segments(300), META, llm, settings, cache)
    assert all(c.status == "ok" for c in result.chunks)
    assert (
        "Glossary (preferred spellings, with forms the recognizer produced):\n(none)"
        in llm.calls[1][1]
    )


def summarize(lines):
    return CleanedChunk(
        paragraphs=[CleanedParagraph(start_segment=lines[0][0], text="Short summary.")]
    )


async def test_half_size_retry_succeeds(settings, cache):
    segs = make_segments(100)

    def handler(fmt, user):
        lines = chunk_lines(user)
        if "previous attempt" not in user:
            return summarize(lines)
        return faithful(lines)

    llm = FakeLLM(handler)
    result = await clean.clean_transcript(segs, META, llm, settings, cache)
    (report,) = result.chunks
    assert report.status == "retried"
    assert report.recall == 1.0
    assert "recall" in report.reason
    assert len(llm.calls) == 3
    assert "recall" in llm.calls[1][1]  # the failure reason is passed to the retry
    ids = [p.start_segment for p in result.paragraphs]
    assert ids[0] == 0 and ids == sorted(ids)


async def test_half_fails_again_uses_raw_for_that_half(settings, cache):
    segs = make_segments(100)

    def handler(fmt, user):
        lines = chunk_lines(user)
        if "previous attempt" in user and lines[0][0] == 0:
            return faithful(lines)
        return summarize(lines)

    llm = FakeLLM(handler)
    result = await clean.clean_transcript(segs, META, llm, settings, cache)
    (report,) = result.chunks
    assert report.status == "raw_fallback"
    assert report.recall is not None and report.recall < 0.75
    second_half_start = chunk_lines(llm.calls[2][1])[0][0]
    raw = clean.raw_paragraphs(segs[second_half_start:])
    assert result.paragraphs[-len(raw) :] == raw
    assert result.paragraphs[0].text.startswith("W0x0")  # first half was cleaned


async def test_truncation_triggers_retry(settings, cache):
    def handler(fmt, user):
        if "previous attempt" not in user:
            raise LlmTruncated("hit max_tokens")
        return faithful(chunk_lines(user))

    llm = FakeLLM(handler)
    result = await clean.clean_transcript(make_segments(100), META, llm, settings, cache)
    assert result.chunks[0].status == "retried"
    assert "cut off" in result.chunks[0].reason


async def test_refusal_goes_straight_to_raw(settings, cache):
    segs = make_segments(100)

    def handler(fmt, user):
        raise LlmRefused("declined (category: cyber)")

    llm = FakeLLM(handler)
    result = await clean.clean_transcript(segs, META, llm, settings, cache)
    assert len(llm.calls) == 1
    assert result.chunks[0].status == "raw_fallback"
    assert "cyber" in result.chunks[0].reason
    assert result.paragraphs == clean.raw_paragraphs(segs)


async def test_empty_result_triggers_retry(settings, cache):
    def handler(fmt, user):
        if "previous attempt" not in user:
            return CleanedChunk(paragraphs=[])
        return faithful(chunk_lines(user))

    llm = FakeLLM(handler)
    result = await clean.clean_transcript(make_segments(100), META, llm, settings, cache)
    assert result.chunks[0].status == "retried"
    assert "empty" in result.chunks[0].reason


async def test_second_run_makes_no_calls(settings, tmp_path):
    segs = make_segments(400)
    first_llm = FakeLLM(default_handler)
    first = await clean.clean_transcript(segs, META, first_llm, settings, Cache(tmp_path, "vid"))
    assert first_llm.calls

    second_llm = FakeLLM(default_handler)
    second = await clean.clean_transcript(segs, META, second_llm, settings, Cache(tmp_path, "vid"))
    assert second_llm.calls == []
    assert second == first


async def test_crash_midway_resumes_without_repaying(settings, tmp_path):
    segs = make_segments(400)
    seen: list[int] = []

    def crashing(fmt, user):
        if fmt is Glossary:
            return default_handler(fmt, user)
        lines = chunk_lines(user)
        if lines[0][0] > 0:
            raise RuntimeError("network down")
        seen.append(lines[0][0])
        return faithful(lines)

    settings = settings.model_copy(update={"concurrency": 1})
    with pytest.raises(RuntimeError):
        await clean.clean_transcript(
            segs, META, FakeLLM(crashing), settings, Cache(tmp_path, "vid")
        )

    llm = FakeLLM(default_handler)
    await clean.clean_transcript(segs, META, llm, settings, Cache(tmp_path, "vid"))
    starts = [chunk_lines(u)[0][0] for fmt, u in llm.calls if fmt is CleanedChunk]
    assert 0 not in starts  # chunk 1 came from the cache
    assert Glossary not in [fmt for fmt, _ in llm.calls]


async def test_speaker_labels_only_at_changes(settings, cache):
    segs = make_segments(20, words=5)
    for j in range(10, 20):
        segs[j] = segs[j].model_copy(update={"speaker": "speaker_1"})
    llm = FakeLLM(default_handler)
    await clean.clean_transcript(segs, META, llm, settings, cache)
    user = llm.calls[0][1]
    assert "[0] (speaker_0) " in user and "[10] (speaker_1) " in user
    assert "[1] (speaker_0)" not in user and "[11] (speaker_1)" not in user


# ---------------------------------------------------------------------------
# The LLM wrapper itself, against a stub client (no network)


def response(
    parsed=None, *, stop="end_turn", model="claude-sonnet-5-5", category=None, tokens=(100, 50)
):
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop,
        stop_details=SimpleNamespace(category=category) if category else None,
        model=model,
        usage=SimpleNamespace(
            input_tokens=tokens[0], output_tokens=tokens[1], cache_read_input_tokens=0
        ),
    )


class StubEndpoint:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def stub_client(plain=(), beta=()):
    return SimpleNamespace(
        messages=StubEndpoint(plain), beta=SimpleNamespace(messages=StubEndpoint(beta))
    )


def bad_request(message):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(
        message, response=httpx2.Response(400, request=request), body=None
    )


ARGS = {
    "model": "claude-sonnet-5-5",
    "effort": "low",
    "system": "s",
    "user": "u",
    "output_format": Glossary,
}


async def test_llm_uses_fallback_beta_and_tallies_usage():
    client = stub_client(beta=[response(Glossary()), response(Glossary(), model="claude-opus-5-5")])
    llm = LLM(Settings(refusal_fallbacks=True), client=client)
    assert await llm.parse(**ARGS) == Glossary()
    await llm.parse(**ARGS)
    call = client.beta.messages.calls[0]
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["output_config"] == {"effort": "low"}
    assert "thinking" not in call and "temperature" not in call
    assert set(llm.usage) == {"claude-sonnet-5-5", "claude-opus-5-5"}  # keyed by serving model
    assert llm.cost_usd() == pytest.approx((100 * 2 + 50 * 10 + 100 * 4 + 50 * 20) / 1e6)


async def test_llm_plain_endpoint_without_fallbacks():
    client = stub_client(plain=[response(Glossary())])
    await LLM(Settings(refusal_fallbacks=False), client=client).parse(**ARGS)
    assert "betas" not in client.messages.calls[0]


async def test_llm_drops_fallbacks_after_rejection():
    client = stub_client(
        plain=[response(Glossary()), response(Glossary())],
        beta=[bad_request("fallbacks: unknown beta parameter")],
    )
    llm = LLM(Settings(refusal_fallbacks=True), client=client)
    await llm.parse(**ARGS)
    await llm.parse(**ARGS)
    assert len(client.beta.messages.calls) == 1
    assert len(client.messages.calls) == 2


async def test_llm_other_bad_request_becomes_llm_failed():
    client = stub_client(beta=[bad_request("messages: field required")])
    with pytest.raises(LlmFailed, match="400"):
        await LLM(Settings(), client=client).parse(**ARGS)
    assert len(client.messages.calls) == 0  # not mistaken for a fallback rejection


async def test_llm_refusal_and_truncation():
    client = stub_client(
        beta=[response(stop="refusal", category="cyber"), response(Glossary(), stop="max_tokens")]
    )
    llm = LLM(Settings(), client=client)
    with pytest.raises(LlmRefused, match="cyber"):
        await llm.parse(**ARGS)
    with pytest.raises(LlmTruncated):
        await llm.parse(**ARGS)
    assert llm.usage["claude-sonnet-5-5"].calls == 2


async def test_llm_invalid_json_counts_as_truncated():
    try:
        Glossary.model_validate_json('{"entries": [')
    except ValidationError as exc:
        error = exc
    client = stub_client(beta=[error])
    with pytest.raises(LlmTruncated):
        await LLM(Settings(), client=client).parse(**ARGS)


def test_cost_unknown_model_is_none():
    llm = LLM(Settings(), client=stub_client())
    llm._record_usage(response(model="some-other-model"), "x")
    assert llm.cost_usd() is None
