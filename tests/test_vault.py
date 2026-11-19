from __future__ import annotations

from pathlib import Path

import pytest

from archiver.errors import NoteExists
from archiver.vault import check_writable, write_note


def write(tmp_path: Path, content: str = "v1\n", name: str = "A (vid1).md", **kw):
    kw.setdefault("overwrite", False)
    return write_note(
        tmp_path / "notes", name, "vid1", content, state_dir=tmp_path / "state", **kw
    )


def test_fresh_write_creates_folder(tmp_path: Path) -> None:
    path = write(tmp_path)
    assert path == tmp_path / "notes" / "A (vid1).md"
    assert path.read_text(encoding="utf-8") == "v1\n"
    assert (tmp_path / "state" / "written.sha256").exists()


def test_missing_notes_dir_is_writable(tmp_path: Path) -> None:
    check_writable(tmp_path / "nope", "vid1", state_dir=tmp_path / "s", overwrite=False)


def test_rewrite_untouched_note(tmp_path: Path) -> None:
    write(tmp_path)
    path = write(tmp_path, "v2\n")
    assert path.read_text() == "v2\n"
    write(tmp_path, "v3\n")  # the stored hash follows the latest write


def test_edited_note_refuses_then_overwrite(tmp_path: Path) -> None:
    path = write(tmp_path)
    path.write_text("my edits\n")
    with pytest.raises(NoteExists, match=r"A \(vid1\)\.md.*--overwrite"):
        write(tmp_path, "v2\n")
    assert path.read_text() == "my edits\n"
    write(tmp_path, "v2\n", overwrite=True)
    assert path.read_text() == "v2\n"


def test_missing_state_refuses(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "A (vid1).md").write_text("foreign\n")
    with pytest.raises(NoteExists):
        write(tmp_path)
    write(tmp_path, overwrite=True)


def test_renamed_title_removes_old_file(tmp_path: Path) -> None:
    old = write(tmp_path, name="Old (vid1).md")
    other = tmp_path / "notes" / "Other (vid2).md"
    other.write_text("keep")
    new = write(tmp_path, "v2\n", name="New (vid1).md")
    assert not old.exists() and new.exists() and other.exists()


def test_other_video_with_similar_id_untouched(tmp_path: Path) -> None:
    write(tmp_path)
    neighbour = tmp_path / "notes" / "B (xvid1).md"
    neighbour.write_text("x")
    write(tmp_path, "v2\n")
    assert neighbour.exists()


def test_glob_metacharacters_in_id(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "A (a[b]).md").write_text("x")
    (notes / "B (ab).md").write_text("y")
    with pytest.raises(NoteExists, match=r"a\[b\]"):
        check_writable(notes, "a[b]", state_dir=tmp_path / "s", overwrite=False)
    check_writable(notes, "zz", state_dir=tmp_path / "s", overwrite=False)


def test_no_temp_files_left(tmp_path: Path) -> None:
    write(tmp_path)
    write(tmp_path, "v2\n")
    assert sorted(p.name for p in (tmp_path / "notes").iterdir()) == ["A (vid1).md"]


def test_temp_cleaned_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("disk")

    monkeypatch.setattr("archiver.vault.os.replace", boom)
    with pytest.raises(OSError):
        write(tmp_path)
    assert list((tmp_path / "notes").iterdir()) == []


def test_cyrillic_filename_and_utf8(tmp_path: Path) -> None:
    path = write(tmp_path, "Привет\nмир\n", name="2026-10-01 Привет (vid1).md")
    assert path.read_bytes() == "Привет\nмир\n".encode("utf-8")
    write(tmp_path, "ещё\n", name="2026-10-01 Привет (vid1).md")


def test_note_in_subfolder_is_found_and_rewritten_in_place(tmp_path: Path) -> None:
    first = write(tmp_path)
    sub = first.parent / "Series"
    sub.mkdir()
    moved = sub / first.name
    first.rename(moved)
    again = write(tmp_path, "v2\n")
    assert again == moved and moved.read_text() == "v2\n"
    assert not (first.parent / first.name).exists()


def test_renamed_title_in_subfolder_stays_in_subfolder(tmp_path: Path) -> None:
    first = write(tmp_path, name="Old (vid1).md")
    sub = first.parent / "Series"
    sub.mkdir()
    old = first.rename(sub / first.name)
    new = write(tmp_path, "v2\n", name="New (vid1).md")
    assert new.parent == sub and not old.exists() and new.exists()


def test_edited_note_in_subfolder_refuses(tmp_path: Path) -> None:
    first = write(tmp_path)
    sub = first.parent / "Series"
    sub.mkdir()
    moved = first.rename(sub / first.name)
    moved.write_text("edited")
    with pytest.raises(NoteExists):
        write(tmp_path, "v2\n")
