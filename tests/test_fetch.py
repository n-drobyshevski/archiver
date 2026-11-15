from datetime import date

import pytest
from yt_dlp.utils import DownloadError

import archiver.fetch as fetch
from archiver.config import Settings
from archiver.errors import FetchBlocked
from archiver.models import VideoMeta

INFO = {
    "id": "abcdefghijk",
    "title": "T",
    "channel": "Chan",
    "uploader": "Up",
    "upload_date": "20240131",
    "duration": 123,
    "description": "d",
    "chapters": [{"start_time": 0, "end_time": 60.5, "title": "Intro"}],
    "is_live": False,
}


def fake_ydl(info=None, error=None, captured=None):
    class FakeYDL:
        def __init__(self, opts):
            if captured is not None:
                captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            if error:
                raise error
            return info

        def download(self, urls):
            if error:
                raise error

    return FakeYDL


def test_mapping(monkeypatch):
    opts = {}
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", fake_ydl(INFO, captured=opts))
    meta = fetch.fetch_meta("https://youtu.be/abcdefghijk", Settings(cookies_from_browser="firefox"))
    assert meta.video_id == "abcdefghijk"
    assert meta.url == "https://www.youtube.com/watch?v=abcdefghijk"
    assert meta.channel == "Chan" and meta.upload_date == date(2024, 1, 31)
    assert meta.duration == 123 and meta.chapters[0].end == 60.5 and not meta.is_live
    assert opts["noplaylist"] and opts["cookiesfrombrowser"] == ("firefox",)


def test_fallbacks(monkeypatch):
    info = {"id": "abcdefghijk", "title": "T", "uploader": "Up", "upload_date": "bad", "chapters": None}
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", fake_ydl(info))
    meta = fetch.fetch_meta("x", Settings())
    assert meta.channel == "Up" and meta.upload_date is None and meta.chapters == []


@pytest.mark.parametrize("extra", [{"is_live": True}, {"live_status": "is_live"}, {"live_status": "is_upcoming"}])
def test_live(monkeypatch, extra):
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", fake_ydl(INFO | extra))
    assert fetch.fetch_meta("x", Settings()).is_live


def test_bot_check(monkeypatch):
    err = DownloadError("ERROR: [youtube] x: Sign in to confirm you’re not a bot")
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", fake_ydl(error=err))
    with pytest.raises(FetchBlocked, match="cookies_from_browser"):
        fetch.fetch_meta("x", Settings())


def test_other_error(monkeypatch):
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", fake_ydl(error=DownloadError("boom")))
    with pytest.raises(FetchBlocked, match="uv lock --upgrade-package yt-dlp"):
        fetch.fetch_meta("x", Settings())


def meta() -> VideoMeta:
    return VideoMeta(video_id="abcdefghijk", url="u", title="T")


def test_download_reuses_existing(tmp_path, monkeypatch):
    (tmp_path / "audio.webm").write_bytes(b"data")
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", fake_ydl(error=DownloadError("must not run")))
    assert fetch.download_audio(meta(), tmp_path, Settings()) == tmp_path / "audio.webm"


def test_download_runs(tmp_path, monkeypatch):
    opts = {}
    cls = fake_ydl(captured=opts)

    def download(self, urls):
        (tmp_path / "audio.m4a").write_bytes(b"data")

    cls.download = download
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", cls)
    assert fetch.download_audio(meta(), tmp_path, Settings()) == tmp_path / "audio.m4a"
    assert opts["format"] == "bestaudio/best"


def test_download_error(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", fake_ydl(error=DownloadError("nope")))
    with pytest.raises(FetchBlocked):
        fetch.download_audio(meta(), tmp_path, Settings())


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abcdefghijk",
    "https://www.youtube.com/watch?list=PL1&v=abcdefghijk&t=5",
    "https://youtu.be/abcdefghijk?t=3",
    "https://www.youtube.com/shorts/abcdefghijk",
    "https://www.youtube.com/live/abcdefghijk?feature=share",
    "https://www.youtube.com/embed/abcdefghijk",
    "https://m.youtube.com/watch?v=abcdefghijk",
])
def test_video_id(url):
    assert fetch.video_id_from_url(url) == "abcdefghijk"


@pytest.mark.parametrize("url", ["https://example.com/watch?v=abcdefghijk", "nonsense", ""])
def test_video_id_none(url):
    assert fetch.video_id_from_url(url) is None
