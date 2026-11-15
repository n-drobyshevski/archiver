from __future__ import annotations

from pydantic import BaseModel

from archiver.cache import Cache, fingerprint


class Thing(BaseModel):
    name: str
    items: list[int] = []


def test_round_trip(tmp_path):
    cache = Cache(tmp_path, "vid")
    cache.save("thing", "fp1", Thing(name="ä", items=[1, 2]))
    assert cache.load("thing", "fp1", Thing) == Thing(name="ä", items=[1, 2])


def test_missing_entry_is_a_miss(tmp_path):
    assert Cache(tmp_path, "vid").load("nope", "fp", Thing) is None


def test_different_fingerprint_is_a_miss(tmp_path):
    cache = Cache(tmp_path, "vid")
    cache.save("thing", "fp1", Thing(name="a"))
    assert cache.load("thing", "fp2", Thing) is None


def test_force_misses_but_still_saves(tmp_path):
    Cache(tmp_path, "vid").save("thing", "fp", Thing(name="old"))
    forced = Cache(tmp_path, "vid", force=True)
    assert forced.load("thing", "fp", Thing) is None
    forced.save("thing", "fp", Thing(name="new"))
    assert Cache(tmp_path, "vid").load("thing", "fp", Thing) == Thing(name="new")


def test_corrupted_json_is_a_miss(tmp_path):
    cache = Cache(tmp_path, "vid")
    cache.path("thing").write_text("{not json", encoding="utf-8")
    assert cache.load("thing", "fp", Thing) is None


def test_wrong_shape_is_a_miss(tmp_path):
    cache = Cache(tmp_path, "vid")
    cache.path("thing").write_text('{"fingerprint": "fp", "data": {"items": "x"}}', encoding="utf-8")
    assert cache.load("thing", "fp", Thing) is None


def test_subfolder_names(tmp_path):
    cache = Cache(tmp_path, "vid")
    cache.save("clean/abc", "fp", Thing(name="chunk"))
    assert (tmp_path / "vid" / "clean" / "abc.json").is_file()
    assert cache.load("clean/abc", "fp", Thing) == Thing(name="chunk")


def test_fingerprint_is_stable_and_order_sensitive():
    assert fingerprint("a", 1, "b") == fingerprint("a", 1, "b")
    assert fingerprint("a", "b") != fingerprint("b", "a")
    assert fingerprint("ab", "c") != fingerprint("a", "bc")
