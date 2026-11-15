"""YouTube access through yt-dlp: metadata first, audio only when speech-to-text needs it."""

from __future__ import annotations

import re
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError, YoutubeDLError

from archiver.config import Settings
from archiver.errors import FetchBlocked
from archiver.models import Chapter, VideoMeta

_ID = r"([A-Za-z0-9_-]{11})"
_URL_PATTERNS = [
    re.compile(r"^(?:https?://)?(?:www\.|m\.|music\.)?youtube\.com/watch\?(?:[^#]*&)?v=" + _ID),
    re.compile(r"^(?:https?://)?youtu\.be/" + _ID),
    re.compile(r"^(?:https?://)?(?:www\.|m\.)?youtube\.com/(?:shorts|live|embed)/" + _ID),
]

_BOT_CHECK = "confirm you’re not a bot", "confirm you're not a bot"


def video_id_from_url(url: str) -> str | None:
    url = url.strip()
    for pattern in _URL_PATTERNS:
        match = pattern.match(url)
        if match:
            return match.group(1)
    return None


# In yt-dlp's own priority order. It enables only Deno unless told otherwise.
_JS_RUNTIMES = ("deno", "node", "quickjs", "bun")


def available_js_runtimes() -> list[str]:
    """JavaScript runtimes on PATH that yt-dlp can use for YouTube."""
    return [name for name in _JS_RUNTIMES if shutil.which(name)]


def _base_opts(settings: Settings) -> dict[str, Any]:
    opts: dict[str, Any] = {"quiet": True, "no_warnings": True, "noplaylist": True}
    runtimes = available_js_runtimes()
    if runtimes:
        opts["js_runtimes"] = {name: {} for name in runtimes}
    if settings.cookies_from_browser:
        opts["cookiesfrombrowser"] = (settings.cookies_from_browser,)
    return opts


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def _chapters(raw: list[dict[str, Any]] | None) -> list[Chapter]:
    return [
        Chapter(
            start=float(c.get("start_time") or 0),
            end=float(c.get("end_time") or 0),
            title=str(c.get("title") or "").strip(),
        )
        for c in raw or []
    ]


def _meta_from_info(info: dict[str, Any]) -> VideoMeta:
    video_id = info["id"]
    return VideoMeta(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        title=info.get("title") or video_id,
        channel=info.get("channel") or info.get("uploader") or "",
        upload_date=_parse_date(info.get("upload_date")),
        duration=float(info.get("duration") or 0),
        description=info.get("description") or "",
        chapters=_chapters(info.get("chapters")),
        is_live=bool(info.get("is_live")) or info.get("live_status") in ("is_live", "is_upcoming"),
    )


def _blocked(exc: Exception) -> FetchBlocked:
    text = str(exc)
    if any(marker in text for marker in _BOT_CHECK):
        return FetchBlocked(
            "YouTube asked for a bot check. Set `cookies_from_browser` (for example \"firefox\") "
            "in the config, or run the tool from a home network."
        )
    return FetchBlocked(
        f"YouTube download failed: {text}\n"
        "YouTube changes often; try `uv lock --upgrade-package yt-dlp && uv sync`."
    )


def fetch_meta(url: str, settings: Settings) -> VideoMeta:
    try:
        with yt_dlp.YoutubeDL(_base_opts(settings)) as ydl:
            info = ydl.extract_info(url, download=False)
    except YoutubeDLError as exc:
        raise _blocked(exc) from exc
    if not info or "id" not in info:
        raise FetchBlocked("YouTube returned no information for this URL.")
    return _meta_from_info(info)


def _existing_audio(dest_dir: Path) -> Path | None:
    for path in sorted(dest_dir.glob("audio.*")):
        # Skip yt-dlp's partial and fragment files.
        if path.suffix not in (".part", ".ytdl", ".temp") and path.stat().st_size > 0:
            return path
    return None


def download_audio(meta: VideoMeta, dest_dir: Path, settings: Settings) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = _existing_audio(dest_dir)
    if cached:
        return cached
    opts = _base_opts(settings) | {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "audio.%(ext)s"),
        # `lang` sorts by language_preference, which the YouTube extractor sets highest for the
        # original track of auto-dubbed videos.
        "format_sort": ["lang"],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([meta.url])
    except YoutubeDLError as exc:
        raise _blocked(exc) from exc
    path = _existing_audio(dest_dir)
    if path is None:
        raise FetchBlocked("Audio download finished but no audio file was found.")
    return path
