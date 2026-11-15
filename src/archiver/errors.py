"""Typed errors, so the CLI (and later a bot) can tell the user what went wrong."""


class ArchiverError(Exception):
    """Base class. The message is written for the person running the tool."""


class PreflightError(ArchiverError):
    """Something is missing before any paid call: a key, a binary, the vault folder."""


class FetchBlocked(ArchiverError):
    """YouTube refused the request (bot check, private or unavailable video, live stream)."""


class SttFailed(ArchiverError):
    """Speech-to-text failed or returned no speech."""


class LlmRefused(ArchiverError):
    """The model declined the request, including after any fallback."""


class LlmTruncated(ArchiverError):
    """The model hit max_tokens before finishing."""


class NoteExists(ArchiverError):
    """A note for this video is already in the vault and was edited since it was written."""
