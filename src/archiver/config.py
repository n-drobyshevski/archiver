"""Settings: defaults, then ~/.config/archiver/config.toml, then ARCHIVER_* environment variables.

API keys are not settings. The SDKs read ANTHROPIC_API_KEY and ELEVENLABS_API_KEY themselves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

Effort = Literal["low", "medium", "high", "xhigh", "max"]


def config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "archiver" / "config.toml"


def load_env_files() -> list[Path]:
    """Read `.env` from the current folder, then from the config folder, into the environment.

    Variables that are already set win, and the current folder wins over the config folder.
    This is how the API keys can be kept in a file instead of being typed into every terminal.
    """
    from dotenv import load_dotenv

    loaded = []
    for path in (Path.cwd() / ".env", config_path().parent / ".env"):
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded


def default_cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "archiver"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARCHIVER_", extra="ignore")

    # Where notes go
    vault: Path | None = None
    folder: str = "Videos"
    cache_dir: Path = Field(default_factory=default_cache_dir)

    # YouTube
    cookies_from_browser: str | None = None  # e.g. "firefox"; helps with YouTube's bot check
    max_duration_min: int = 240

    # Speech-to-text
    stt_model: str = "scribe_v3"
    diarize: bool = True
    stt_timeout_s: int = 1800

    # Cleanup
    clean_model: str = "claude-sonnet-5-5"
    clean_effort: Effort = "low"
    chunk_words: int = 1350
    concurrency: int = 4

    # Topics and summary
    outline_model: str = "claude-opus-5-5"
    outline_effort: Effort = "high"

    # Links and tags that join a new note to the notes already in the vault (matched locally)
    link_notes: bool = True
    link_exclude: list[str] = Field(default_factory=lambda: ["Anchor"])  # top-level folders
    link_max: int = 12

    # Let the API retry on another model when a request is declined
    refusal_fallbacks: bool = True

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml = TomlConfigSettingsSource(settings_cls, toml_file=config_path())
        return (init_settings, env_settings, toml)

    @property
    def notes_dir(self) -> Path | None:
        if self.vault is None:
            return None
        return self.vault.expanduser() / self.folder
