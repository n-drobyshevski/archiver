from types import SimpleNamespace as NS
from pathlib import Path

import pytest
from elevenlabs.core.api_error import ApiError

from archiver.config import Settings
from archiver.errors import SttFailed
from archiver.stt.base import VideoSource
from archiver.stt.scribe import ScribeTranscriber, map_language


def w(text, start, end, type="word", speaker_id=None):
    return NS(text=text, start=start, end=end, type=type, speaker_id=speaker_id)


class StubClient:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.speech_to_text = self
        self._response, self._error = response, error

    def convert(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


@pytest.fixture
def source(tmp_path: Path) -> VideoSource:
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"x")
    return VideoSource(url="u", audio_path=lambda: audio)


def settings(**kw) -> Settings:
    return Settings(stt_timeout_s=99, **kw)


def test_filters_and_maps(source):
    resp = NS(language_code="rus", text="", words=[
        w("Привет", 0.0, 0.5, speaker_id="speaker_0"),
        w(" ", 0.5, 0.6, type="spacing"),
        w("(смех)", 0.6, 1.0, type="audio_event"),
        w("  ", 1.0, 1.1),
        w("мир", 1.1, 1.5, speaker_id="speaker_1"),
    ])
    client = StubClient(resp)
    raw = ScribeTranscriber(settings(), client=client).transcribe(source)
    assert raw.provider == "elevenlabs" and raw.model == "scribe_v3" and raw.language == "ru"
    assert [(x.text, x.speaker) for x in raw.words] == [("Привет", "speaker_0"), ("мир", "speaker_1")]
    call = client.calls[0]
    assert call["model_id"] == "scribe_v3" and call["tag_audio_events"] is False
    assert call["diarize"] is True
    assert call["request_options"] == {"timeout_in_seconds": 99}


def test_no_words(source):
    resp = NS(language_code="eng", text="", words=[w(" ", 0, 1, type="spacing")])
    with pytest.raises(SttFailed):
        ScribeTranscriber(settings(), client=StubClient(resp)).transcribe(source)


def test_api_error(source):
    err = ApiError(status_code=401, body={"detail": "bad key"})
    with pytest.raises(SttFailed, match="401.*API key"):
        ScribeTranscriber(settings(), client=StubClient(error=err)).transcribe(source)


def test_other_error(source):
    with pytest.raises(SttFailed, match="timed out"):
        ScribeTranscriber(settings(), client=StubClient(error=TimeoutError("timed out"))).transcribe(source)


@pytest.mark.parametrize("code,expected", [
    ("rus", "ru"), ("eng", "en"), ("ukr", "uk"), ("deu", "de"), ("fra", "fr"), ("spa", "es"),
    ("ru", "ru"), ("xyz", None), (None, None), ("", None),
])
def test_language(code, expected):
    assert map_language(code) == expected
