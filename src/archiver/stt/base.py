"""Speech-to-text interface. Providers get the video source, not just a file, so a provider that
can fetch a URL itself never forces an audio download."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from archiver.models import RawTranscript


@dataclass
class VideoSource:
    url: str
    # Downloads the audio on first call and returns the local file. Providers that take a URL
    # never call it.
    audio_path: Callable[[], Path]


class Transcriber(Protocol):
    provider: str
    model: str

    def transcribe(self, source: VideoSource) -> RawTranscript:
        """Blocking call. Raises SttFailed on provider errors or when no speech is found."""
        ...
