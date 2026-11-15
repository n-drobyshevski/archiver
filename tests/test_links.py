from __future__ import annotations

from pathlib import Path

from archiver.links import build_index, normalize, resolve
from archiver.models import Concept


def write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def concept(*names: str, relation: str = "") -> Concept:
    return Concept(names=list(names), relation=relation)


def test_normalize_ignores_case_yo_punctuation_and_spacing():
    assert normalize("  Тёмный   лес! ") == normalize("темный лес")
    assert normalize("Self-Organisation") == normalize("self organisation")
    assert normalize("W. Ross Ashby") == "w ross ashby"


def test_matches_titles_and_aliases_but_not_short_or_numeric_names(tmp_path):
    write(tmp_path / "Кибернетика.md")
    write(tmp_path / "People" / "Эшби.md", "---\naliases: [W. Ross Ashby, Уильям Росс Эшби]\n---\n")
    write(tmp_path / "ИИ.md")
    write(tmp_path / "2024.md")
    index = build_index(tmp_path)
    links, _ = resolve(
        [
            concept("кибернетика", relation="основа"),
            concept("Ashby", "w ross ashby"),
            concept("ИИ"),
            concept("2024"),
            concept("Нет такой заметки"),
        ],
        [],
        index,
    )
    assert [(link.target, link.relation) for link in links] == [("Кибернетика", "основа"), ("Эшби", "")]


def test_title_beats_alias_and_duplicates_collapse(tmp_path):
    write(tmp_path / "Лес.md")
    write(tmp_path / "Другое.md", "---\naliases:\n  - Лес\n---\n")
    index = build_index(tmp_path)
    links, _ = resolve([concept("лес"), concept("Лес")], [], index)
    assert [link.target for link in links] == ["Лес"]


def test_shared_file_name_links_by_path(tmp_path):
    write(tmp_path / "A" / "Модель.md")
    write(tmp_path / "B" / "Deep" / "Модель.md")
    links, _ = resolve([concept("Модель")], [], build_index(tmp_path))
    assert [(link.target, link.title) for link in links] == [("A/Модель", "Модель")]


def test_excluded_and_dot_folders_and_own_note_are_never_targets(tmp_path):
    write(tmp_path / "Anchor" / "Memory" / "Секрет.md")
    write(tmp_path / ".obsidian" / "Скрытое.md")
    write(tmp_path / ".trash" / "Удалённое.md")
    write(tmp_path / "Videos" / "2026-01-01 Доклад (abcdefghijk).md")
    write(tmp_path / "Videos" / "Доклад.md")
    index = build_index(tmp_path, exclude=["anchor"], skip_video_id="abcdefghijk")
    links, _ = resolve(
        [
            concept("Секрет"),
            concept("Скрытое"),
            concept("Удалённое"),
            concept("2026-01-01 Доклад (abcdefghijk)"),
            concept("Доклад"),
        ],
        [],
        index,
    )
    assert [link.target for link in links] == ["Доклад"]


def test_an_excluded_note_still_makes_a_name_ambiguous(tmp_path):
    write(tmp_path / "Anchor" / "Идея.md")
    write(tmp_path / "Notes" / "Идея.md")
    links, _ = resolve([concept("Идея")], [], build_index(tmp_path, exclude=["Anchor"]))
    assert [link.target for link in links] == ["Notes/Идея"]


def test_titles_that_would_break_a_wikilink_are_skipped(tmp_path):
    write(tmp_path / "C# basics.md")
    write(tmp_path / "A [draft].md")
    links, _ = resolve([concept("C# basics"), concept("A [draft]")], [], build_index(tmp_path))
    assert links == []


def test_link_cap(tmp_path):
    for i in range(20):
        write(tmp_path / f"Тема {i:02d}.md")
    links, _ = resolve([concept(f"Тема {i:02d}") for i in range(20)], [], build_index(tmp_path), max_links=12)
    assert len(links) == 12


def test_tags_reuse_the_vaults_spelling_from_frontmatter_and_inline(tmp_path):
    write(tmp_path / "a.md", "---\ntags: [Кибернетика, field/systems-theory]\n---\n")
    write(tmp_path / "b.md", "Текст с #философия-техники и # не тег, и url http://x/#anchor\n```\n#notatag\n```\n")
    write(tmp_path / "c.md", "---\ntags: one, two\n---\n")
    index = build_index(tmp_path)
    assert "notatag" not in index.tags and "anchor" not in index.tags
    _, tags = resolve(
        [concept("кибернетика"), concept("Systems theory"), concept("Философия техники")],
        ["ONE", "совсем-новый"],
        index,
    )
    assert tags == ["Кибернетика", "field/systems-theory", "философия-техники", "one"]


def test_model_tags_are_kept_only_when_nothing_in_the_vault_fits(tmp_path):
    write(tmp_path / "a.md", "---\ntags: [музыка]\n---\n")
    _, tags = resolve([concept("Компилятор")], ["compilers", "parsing"], build_index(tmp_path))
    assert tags == ["compilers", "parsing"]


def test_odd_notes_do_not_break_the_scan(tmp_path):
    write(tmp_path / "broken.md", "---\ntags: [unclosed\n---\nтекст\n")
    (tmp_path / "binary.md").write_bytes(b"\xff\xfe\x00\x01")
    write(tmp_path / "scalar.md", "---\njust a string\n---\n")
    write(tmp_path / "Хорошая.md", "---\naliases: Одна\n---\n")
    links, _ = resolve([concept("Одна")], [], build_index(tmp_path))
    assert [link.target for link in links] == ["Хорошая"]
