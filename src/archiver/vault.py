"""Safe writes into the vault folder that Obsidian Sync watches."""

from __future__ import annotations

import glob
import hashlib
import os
import tempfile
from pathlib import Path

from archiver.errors import NoteExists

HASH_FILE = "written.sha256"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _existing_notes(notes_dir: Path, video_id: str) -> list[Path]:
    if not notes_dir.is_dir():
        return []
    pattern = f"*({glob.escape(video_id)}).md"
    return sorted(p for p in notes_dir.glob(pattern) if p.is_file() and not p.name.startswith("."))


def _stored_hash(state_dir: Path) -> str | None:
    try:
        return (state_dir / HASH_FILE).read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def check_writable(notes_dir: Path, video_id: str, *, state_dir: Path, overwrite: bool) -> None:
    """Raise NoteExists if a note for this video exists that archiver did not last write as is."""
    if overwrite:
        return
    stored = _stored_hash(state_dir)
    for note in _existing_notes(notes_dir, video_id):
        if stored is None or _sha256(note) != stored:
            raise NoteExists(
                f"{note} was edited since archiver wrote it (or was not written by this cache). "
                "Use --overwrite to replace it."
            )


def write_note(
    notes_dir: Path,
    filename: str,
    video_id: str,
    content: str,
    *,
    overwrite: bool,
    state_dir: Path,
) -> Path:
    check_writable(notes_dir, video_id, state_dir=state_dir, overwrite=overwrite)
    notes_dir.mkdir(parents=True, exist_ok=True)
    target = notes_dir / filename
    data = content.encode("utf-8")  # bytes, so no newline translation on any platform

    # A dot-prefixed temp file in the same folder: same filesystem for an atomic replace,
    # and Obsidian ignores dotfiles so Sync never sees a half-written note.
    fd, tmp_name = tempfile.mkstemp(dir=notes_dir, prefix=".archiver-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        stale = [p for p in _existing_notes(notes_dir, video_id) if p.name != filename]
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / HASH_FILE).write_text(hashlib.sha256(data).hexdigest(), encoding="utf-8")
    for old in stale:  # the title changed upstream; keep a single note per video
        old.unlink(missing_ok=True)
    return target
