"""ElevenLabs Scribe speech-to-text."""

from __future__ import annotations

import os
from typing import Any

from archiver.config import Settings
from archiver.errors import SttFailed
from archiver.models import RawTranscript, Word
from archiver.stt.base import VideoSource

# Scribe reports ISO 639-3; notes use ISO 639-1.
_LANGUAGES = {
    "rus": "ru", "eng": "en", "ukr": "uk", "deu": "de", "fra": "fr", "spa": "es", "ita": "it",
    "por": "pt", "pol": "pl", "nld": "nl", "tur": "tr", "ces": "cs", "jpn": "ja", "kor": "ko",
    "zho": "zh", "arb": "ar", "ara": "ar", "hin": "hi", "bel": "be", "kaz": "kk", "swe": "sv",
    "fin": "fi", "ell": "el", "heb": "he", "bul": "bg", "ron": "ro", "hun": "hu", "srp": "sr",
}


def map_language(code: str | None) -> str | None:
    if not code:
        return None
    code = code.strip().lower()
    if len(code) == 2:
        return code
    return _LANGUAGES.get(code)


def _words(response: Any) -> list[Word]:
    words: list[Word] = []
    for item in getattr(response, "words", None) or []:
        text = (getattr(item, "text", "") or "").strip()
        if getattr(item, "type", None) != "word" or not text:
            continue
        start = getattr(item, "start", None)
        end = getattr(item, "end", None)
        if start is None or end is None:
            continue
        words.append(Word(text=text, start=start, end=end, speaker=getattr(item, "speaker_id", None)))
    return words


def _api_message(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status is None:
        return f"Speech-to-text failed: {exc}"
    hint = " (the ElevenLabs API key was rejected; check ELEVENLABS_API_KEY)" if status == 401 else ""
    return f"Speech-to-text failed with status {status}{hint}: {getattr(exc, 'body', exc)}"


class ScribeTranscriber:
    provider = "elevenlabs"

    def __init__(self, settings: Settings, client: Any = None) -> None:
        self.settings = settings
        self.model = settings.stt_model
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from elevenlabs import ElevenLabs

            self._client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        return self._client

    def transcribe(self, source: VideoSource) -> RawTranscript:
        from elevenlabs.core.api_error import ApiError

        path = source.audio_path()
        try:
            with open(path, "rb") as audio:
                response = self._get_client().speech_to_text.convert(
                    file=audio,
                    model_id=self.model,
                    diarize=self.settings.diarize,
                    tag_audio_events=False,
                    request_options={"timeout_in_seconds": self.settings.stt_timeout_s},
                )
        except ApiError as exc:
            raise SttFailed(_api_message(exc)) from exc
        except OSError as exc:
            raise SttFailed(f"Could not read the audio file: {exc}") from exc
        except Exception as exc:  # network errors, timeouts, SDK validation
            raise SttFailed(_api_message(exc)) from exc

        words = _words(response)
        if not words:
            raise SttFailed("No speech was found in this video.")
        return RawTranscript(
            provider=self.provider,
            model=self.model,
            language=map_language(getattr(response, "language_code", None)),
            words=words,
        )
