from __future__ import annotations

from pathlib import Path

import pytest

from archiver.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in ("ARCHIVER_VAULT", "ARCHIVER_FOLDER", "ARCHIVER_CONCURRENCY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def write_config(tmp_path: Path, text: str) -> None:
    path = tmp_path / "config" / "archiver" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")


def test_defaults(tmp_path):
    s = Settings()
    assert s.vault is None
    assert s.notes_dir is None
    assert s.folder == "Videos"
    assert s.cache_dir == tmp_path / "cache" / "archiver"
    assert s.clean_model == "claude-sonnet-5-5"
    assert s.outline_model == "claude-opus-5-5"
    assert s.max_duration_min == 240
    assert s.cookies_from_browser is None


def test_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCHIVER_VAULT", str(tmp_path / "v"))
    monkeypatch.setenv("ARCHIVER_CONCURRENCY", "6")
    s = Settings()
    assert s.vault == tmp_path / "v"
    assert s.concurrency == 6


def test_toml_file(tmp_path):
    write_config(tmp_path, f'vault = "{tmp_path / "toml"}"\nfolder = "Talks"\nchunk_words = 900\n')
    s = Settings()
    assert s.vault == tmp_path / "toml"
    assert s.folder == "Talks"
    assert s.chunk_words == 900


def test_env_beats_file(monkeypatch, tmp_path):
    write_config(tmp_path, 'folder = "FromFile"\nchunk_words = 900\n')
    monkeypatch.setenv("ARCHIVER_FOLDER", "FromEnv")
    s = Settings()
    assert s.folder == "FromEnv"
    assert s.chunk_words == 900


def test_init_beats_env_and_file(monkeypatch, tmp_path):
    write_config(tmp_path, 'folder = "FromFile"\n')
    monkeypatch.setenv("ARCHIVER_FOLDER", "FromEnv")
    assert Settings(folder="FromInit").folder == "FromInit"


def test_notes_dir_joins_vault_and_folder(tmp_path):
    assert Settings(vault=tmp_path, folder="Talks").notes_dir == tmp_path / "Talks"


def test_notes_dir_expands_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert Settings(vault=Path("~/vault")).notes_dir == tmp_path / "vault" / "Videos"


def test_env_files_fill_in_missing_variables(tmp_path, monkeypatch):
    import os

    from archiver.config import load_env_files

    names = ("ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY", "ARCHIVER_FOLDER", "ARCHIVER_VAULT")
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        os.environ["ELEVENLABS_API_KEY"] = "from-shell"
        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text(
            'ANTHROPIC_API_KEY="from-project"\nELEVENLABS_API_KEY=from-project\n', encoding="utf-8"
        )
        config_dir = tmp_path / "xdg" / "archiver"
        config_dir.mkdir(parents=True)
        (config_dir / ".env").write_text(
            "ANTHROPIC_API_KEY=from-config\nARCHIVER_FOLDER=Talks\n", encoding="utf-8"
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.chdir(project)

        assert load_env_files() == [project / ".env", config_dir / ".env"]
        assert os.environ["ELEVENLABS_API_KEY"] == "from-shell"  # already set: kept
        assert os.environ["ANTHROPIC_API_KEY"] == "from-project"  # current folder first
        assert Settings().folder == "Talks"  # settings can live there too
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_no_env_files_is_fine(tmp_path, monkeypatch):
    from archiver.config import load_env_files

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    assert load_env_files() == []
