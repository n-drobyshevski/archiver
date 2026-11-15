"""Join a new note to the notes already in the vault, by wikilinks and tags.

The model names the concepts a video is about without ever seeing the vault. This module matches
those names against the vault's own note titles, aliases and tags. Everything here runs locally:
nothing read from the vault is sent anywhere.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from archiver.models import Concept, NoteLink

MIN_NAME_CHARS = 3
MAX_LINKS = 12
MAX_TAGS = 7
READ_BYTES = 200_000  # per note; enough for frontmatter and inline tags

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_DASHES = re.compile(r"[-–—_/]+")
_SPACES = re.compile(r"\s+")
_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\s*(?:\r?\n|\Z)", re.DOTALL)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_TAG = re.compile(r"(?<![\w/#&])#([^\W\d][\w/-]*)", re.UNICODE)
_LINK_BREAKERS = set("[]|#^")


def normalize(text: str) -> str:
    """Comparison key for names: case, ё/е, punctuation and spacing do not matter."""
    text = unicodedata.normalize("NFC", text).casefold().replace("ё", "е")
    text = _DASHES.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _tag_key(text: str) -> str:
    return normalize(text.lstrip("#")).replace(" ", "-")


@dataclass(frozen=True)
class NoteRef:
    path: str  # vault-relative, forward slashes, no ".md"
    title: str
    aliases: tuple[str, ...] = ()


@dataclass
class VaultIndex:
    names: dict[str, list[NoteRef]] = field(default_factory=dict)  # normalized title or alias
    titles: Counter = field(default_factory=Counter)  # how many notes share a file name
    tags: dict[str, str] = field(default_factory=dict)  # tag key -> the tag as the vault writes it

    def add_note(self, ref: NoteRef, tags: Iterable[str]) -> None:
        for name in (ref.title, *ref.aliases):
            key = normalize(name)
            if len(key) >= MIN_NAME_CHARS and not key.isdigit():
                bucket = self.names.setdefault(key, [])
                if ref not in bucket:
                    bucket.append(ref)
        for tag in tags:
            tag = tag.strip().lstrip("#").strip("/")
            if tag and not tag.isdigit():
                self.tags.setdefault(_tag_key(tag), tag)
                # A nested tag also answers to its last part: "field/cybernetics" for "cybernetics".
                self.tags.setdefault(_tag_key(tag.rsplit("/", 1)[-1]), tag)

    def find(self, name: str) -> NoteRef | None:
        key = normalize(name)
        if len(key) < MIN_NAME_CHARS:
            return None
        found = self.names.get(key)
        if not found:
            return None
        # A title match beats an alias match; then the note nearest the vault root.
        return min(found, key=lambda r: (normalize(r.title) != key, r.path.count("/"), r.path))


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [part for part in re.split(r"[,\s]+", value) if part]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, int))]
    return []


def _read_note(path: Path) -> tuple[list[str], list[str]]:
    """Aliases and tags of one note. An unreadable or odd note simply has none."""
    try:
        with path.open("rb") as handle:
            text = handle.read(READ_BYTES).decode("utf-8", errors="ignore")
    except OSError:
        return [], []
    aliases: list[str] = []
    tags: list[str] = []
    body = text
    match = _FRONTMATTER.match(text)
    if match:
        body = text[match.end() :]
        try:
            data = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            data = None
        if isinstance(data, dict):
            raw = data.get("aliases")
            aliases = [raw.strip()] if isinstance(raw, str) else _strings(raw)
            tags = _strings(data.get("tags"))
    tags.extend(_INLINE_TAG.findall(_CODE_FENCE.sub("", body)))
    return aliases, tags


def build_index(
    vault: Path, *, exclude: Sequence[str] = (), skip_video_id: str | None = None
) -> VaultIndex:
    """Scan the vault's Markdown notes. Dot-folders are never entered; `exclude` names top-level
    folders whose notes are not link targets (their file names still count for ambiguity)."""
    vault = Path(vault).expanduser()
    excluded = {normalize(name) for name in exclude}
    index = VaultIndex()
    skip_suffix = f"({skip_video_id}).md" if skip_video_id else None
    for root, dirs, files in os.walk(vault):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        rel_root = Path(root).relative_to(vault)
        top = rel_root.parts[0] if rel_root.parts else ""
        for name in sorted(files):
            if not name.endswith(".md") or name.startswith("."):
                continue
            title = name[: -len(".md")]
            index.titles[title.casefold()] += 1
            if normalize(top) in excluded or (skip_suffix and name.endswith(skip_suffix)):
                continue
            if _LINK_BREAKERS & set(title):
                continue
            aliases, tags = _read_note(Path(root) / name)
            rel = (rel_root / title).as_posix()
            index.add_note(NoteRef(path=rel, title=title, aliases=tuple(aliases)), tags)
    return index


def _target(ref: NoteRef, index: VaultIndex) -> str:
    # Obsidian resolves a bare name; spell out the path only when the name is shared.
    return ref.title if index.titles[ref.title.casefold()] <= 1 else ref.path


def resolve(
    concepts: Sequence[Concept],
    model_tags: Sequence[str],
    index: VaultIndex,
    *,
    max_links: int = MAX_LINKS,
) -> tuple[list[NoteLink], list[str]]:
    """Links to existing notes and the tags to use.

    Tags the vault already uses come first: those matching a concept, then those the model
    proposed. The model's new tags are kept only when none of the vault's tags fit, so a note is
    never left untagged and the tag list does not grow with every video.
    """
    links: list[NoteLink] = []
    seen: set[str] = set()
    tags: list[str] = []

    def add_tag(tag: str) -> None:
        if tag not in tags and len(tags) < MAX_TAGS:
            tags.append(tag)

    for concept in concepts:
        ref = next((r for r in (index.find(n) for n in concept.names) if r), None)
        if ref and ref.path not in seen and len(links) < max_links:
            seen.add(ref.path)
            links.append(
                NoteLink(target=_target(ref, index), title=ref.title, relation=concept.relation)
            )
        for name in concept.names:
            existing = index.tags.get(_tag_key(name))
            if existing:
                add_tag(existing)
    for tag in model_tags:
        existing = index.tags.get(_tag_key(tag))
        if existing:
            add_tag(existing)
    if not tags:
        tags = list(model_tags)[:MAX_TAGS]
    return links, tags
