"""The whole job: URL in, note path out. The CLI calls `run`; a bot can await it the same way."""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from archiver import clean, fetch, links, outline, render, segment, vault
from archiver.cache import Cache, fingerprint
from archiver.config import Settings
from archiver.errors import FetchBlocked, LlmRefused, LlmTruncated, PreflightError, SttFailed
from archiver.llm import LLM
from archiver.models import CleanResult, NoteLink, Outline, RawTranscript, Usage, VideoMeta
from archiver.stt.base import Transcriber, VideoSource

Progress = Callable[[str, str], None]  # (stage, message)


@dataclass
class Deps:
    """The outside world. Tests replace these with fakes."""

    fetch_meta: Callable[[str, Settings], VideoMeta]
    download_audio: Callable[[VideoMeta, Path, Settings], Path]
    transcriber: Transcriber
    llm: LLM


@dataclass
class RunResult:
    path: Path
    meta: VideoMeta
    warnings: list[str] = field(default_factory=list)
    usage: list[Usage] = field(default_factory=list)
    cost_usd: float | None = None


def preflight(settings: Settings) -> None:
    """Fail before any paid call if the machine is not set up."""
    problems: list[str] = []
    if settings.vault is None:
        problems.append("No vault configured. Set ARCHIVER_VAULT or `vault` in the config file.")
    elif not settings.vault.expanduser().is_dir():
        problems.append(f"Vault folder does not exist: {settings.vault}")
    if not os.environ.get("ELEVENLABS_API_KEY"):
        problems.append("ELEVENLABS_API_KEY is not set.")
    has_anthropic = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or (Path.home() / ".config" / "anthropic").is_dir()
    )
    if not has_anthropic:
        problems.append("No Anthropic credentials. Set ANTHROPIC_API_KEY or run `ant auth login`.")
    if not fetch.available_js_runtimes():
        problems.append(
            "No JavaScript runtime found. yt-dlp needs one for YouTube: install Deno (or Node.js)."
        )
    if shutil.which("ffmpeg") is None:
        problems.append("`ffmpeg` is not on PATH (yt-dlp needs it for audio).")
    if problems:
        raise PreflightError("\n".join(problems))


def default_deps(settings: Settings) -> Deps:
    from archiver.stt.scribe import ScribeTranscriber

    return Deps(
        fetch_meta=fetch.fetch_meta,
        download_audio=fetch.download_audio,
        transcriber=ScribeTranscriber(settings),
        llm=LLM(settings),
    )


async def run(
    url: str,
    settings: Settings,
    *,
    force: bool = False,
    overwrite: bool = False,
    on_progress: Progress | None = None,
    deps: Deps | None = None,
) -> RunResult:
    def say(stage: str, message: str) -> None:
        if on_progress:
            on_progress(stage, message)

    if deps is None:
        preflight(settings)
        deps = default_deps(settings)
    notes_dir = settings.notes_dir
    if notes_dir is None:
        raise PreflightError("No vault configured.")

    # fetch
    say("fetch", "reading video metadata")
    meta = await asyncio.to_thread(deps.fetch_meta, url, settings)
    if meta.is_live:
        raise FetchBlocked("This is a live stream. Run it again once the stream has ended.")
    if meta.duration > settings.max_duration_min * 60:
        raise PreflightError(
            f"Video is {meta.duration / 60:.0f} min, over the {settings.max_duration_min} min limit "
            "(max_duration_min)."
        )
    cache = Cache(settings.cache_dir, meta.video_id, force=force)
    # Refuse now, not after paying, if an edited note would be overwritten.
    vault.check_writable(notes_dir, meta.video_id, state_dir=cache.dir, overwrite=overwrite)

    # transcribe
    stt_fp = fingerprint(deps.transcriber.provider, deps.transcriber.model, settings.diarize)
    raw = cache.load("stt_raw", stt_fp, RawTranscript)
    if raw is None:
        say("transcribe", f"transcribing {meta.duration / 60:.0f} min of audio")
        source = VideoSource(
            url=meta.url,
            audio_path=lambda: deps.download_audio(meta, cache.dir, settings),
        )
        raw = await asyncio.to_thread(deps.transcriber.transcribe, source)
        cache.save("stt_raw", stt_fp, raw)
    else:
        say("transcribe", "using cached transcript")
    if not raw.words:
        raise SttFailed("No speech was found in this video.")
    language = raw.language or "en"

    # segment (free, always recomputed)
    segments = segment.build_segments(raw.words)

    # clean (caches per chunk inside)
    say("clean", f"cleaning {len(segments)} segments")
    cleaned: CleanResult = await clean.clean_transcript(
        segments, meta, deps.llm, settings, cache, on_progress=on_progress
    )
    warnings: list[str] = []
    raw_chunks = sum(1 for c in cleaned.chunks if c.status == "raw_fallback")
    if raw_chunks:
        warnings.append(
            f"{raw_chunks} of {len(cleaned.chunks)} parts of the transcript could not be cleaned "
            "and are shown as raw speech-to-text."
        )

    # outline
    outline_fp = fingerprint(
        outline.prompt_fingerprint(),
        settings.outline_model,
        settings.outline_effort,
        language,
        [c.model_dump_json() for c in meta.chapters],
        [p.text for p in cleaned.paragraphs],
    )
    result = cache.load("outline", outline_fp, Outline)
    if result is None:
        say("outline", "finding topics and writing the summary")
        try:
            result = await outline.build_outline(
                cleaned.paragraphs, meta, language, deps.llm, settings
            )
            cache.save("outline", outline_fp, result)
        except (LlmRefused, LlmTruncated) as exc:
            warnings.append(f"The summary could not be written ({exc}).")
            result = outline.fallback_outline(cleaned.paragraphs, meta)
    else:
        say("outline", "using cached outline")

    # link into the existing notes (local and free, so never cached: the vault changes)
    note_links: list[NoteLink] = []
    tags: list[str] | None = None
    if settings.link_notes and settings.vault is not None and result.concepts:
        index = await asyncio.to_thread(
            links.build_index,
            settings.vault,
            exclude=settings.link_exclude,
            skip_video_id=meta.video_id,
        )
        note_links, tags = links.resolve(
            result.concepts, result.tags, index, max_links=settings.link_max
        )
        say("link", f"{len(note_links)} related notes found in the vault")

    # render + write
    content = render.render_note(
        meta,
        cleaned.paragraphs,
        result,
        language=language,
        warnings=warnings,
        created=date.today(),
        links=note_links,
        tags=tags,
    )
    path = vault.write_note(
        notes_dir,
        render.note_filename(meta),
        meta.video_id,
        content,
        overwrite=overwrite,
        state_dir=cache.dir,
    )
    say("done", str(path))
    return RunResult(
        path=path,
        meta=meta,
        warnings=warnings,
        usage=list(deps.llm.usage.values()),
        cost_usd=deps.llm.cost_usd(),
    )
