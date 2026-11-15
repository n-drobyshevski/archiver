from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from archiver.errors import FetchBlocked, PreflightError
from archiver.models import Usage

runner = CliRunner()


class FakeResult:
    def __init__(self, path: Path, warnings=None, usage=None, cost_usd=None) -> None:
        self.path = path
        self.warnings = warnings or []
        self.usage = usage or []
        self.cost_usd = cost_usd


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Install a fake archiver.pipeline; `calls` records every run() call."""
    module = types.ModuleType("archiver.pipeline")
    module.calls = []
    module.behaviour = {}  # url -> exception to raise

    async def run(url, settings, *, force=False, overwrite=False, on_progress=None, deps=None):
        module.calls.append(
            {"url": url, "settings": settings, "force": force, "overwrite": overwrite}
        )
        if on_progress:
            on_progress("fetch", "reading video metadata")
            on_progress("clean", "cleaning 10 segments")
            on_progress("clean", "chunk 1/2")
            on_progress("done", "x")
        exc = module.behaviour.get(url)
        if exc is not None:
            raise exc
        return FakeResult(
            tmp_path / f"{url[-3:]}.md",
            warnings=["2 of 5 parts were not cleaned"],
            usage=[Usage(model="claude-sonnet-5-5", calls=3, input_tokens=1000, output_tokens=500)],
            cost_usd=0.42,
        )

    module.run = run
    module.RunResult = FakeResult
    monkeypatch.setitem(sys.modules, "archiver.pipeline", module)
    # `from archiver import pipeline` reads the package attribute once the real module has
    # been imported by another test, so replace that too.
    import archiver

    monkeypatch.setattr(archiver, "pipeline", module, raising=False)
    return module


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("ARCHIVER_VAULT", raising=False)
    monkeypatch.delenv("ARCHIVER_FOLDER", raising=False)


def invoke(*args: str):
    from archiver.cli import app

    return runner.invoke(app, list(args))


def test_success_prints_path_on_stdout(fake_pipeline, tmp_path):
    result = invoke("https://youtu.be/abc")
    assert result.exit_code == 0
    assert result.stdout.strip() == str(tmp_path / "abc.md")
    assert "warning: 2 of 5 parts were not cleaned" in result.stderr
    assert "claude-sonnet-5-5: 3 calls" in result.stderr
    assert "$0.42" in result.stderr
    assert "[fetch]" in result.stderr


def test_cost_line_omitted_when_unknown(fake_pipeline, monkeypatch):
    original = fake_pipeline.run

    async def run(*a, **kw):
        r = await original(*a, **kw)
        r.cost_usd = None
        return r

    monkeypatch.setattr(fake_pipeline, "run", run)
    result = invoke("https://youtu.be/abc")
    assert "approximate cost" not in result.stderr


def test_options_reach_settings_and_run(fake_pipeline, tmp_path):
    vault = tmp_path / "vault"
    result = invoke(
        "https://youtu.be/abc", "--vault", str(vault), "--folder", "Talks", "--force", "--overwrite"
    )
    assert result.exit_code == 0
    call = fake_pipeline.calls[0]
    assert call["force"] is True and call["overwrite"] is True
    assert call["settings"].vault == vault
    assert call["settings"].folder == "Talks"


def test_options_default_to_config(fake_pipeline, monkeypatch, tmp_path):
    monkeypatch.setenv("ARCHIVER_VAULT", str(tmp_path / "envvault"))
    invoke("https://youtu.be/abc")
    call = fake_pipeline.calls[0]
    assert call["force"] is False and call["overwrite"] is False
    assert call["settings"].vault == tmp_path / "envvault"
    assert call["settings"].folder == "Videos"


def test_one_failure_does_not_stop_the_rest(fake_pipeline, tmp_path):
    fake_pipeline.behaviour["https://youtu.be/bad"] = FetchBlocked("Sign in to confirm")
    result = invoke("https://youtu.be/bad", "https://youtu.be/ok1")
    assert result.exit_code == 1
    assert [c["url"] for c in fake_pipeline.calls] == [
        "https://youtu.be/bad",
        "https://youtu.be/ok1",
    ]
    assert "error: Sign in to confirm" in result.stderr
    assert result.stdout.strip() == str(tmp_path / "ok1.md")


def test_preflight_error_aborts_with_code_2(fake_pipeline):
    fake_pipeline.behaviour["https://youtu.be/one"] = PreflightError("ELEVENLABS_API_KEY is not set.")
    result = invoke("https://youtu.be/one", "https://youtu.be/two")
    assert result.exit_code == 2
    assert len(fake_pipeline.calls) == 1
    assert "error: ELEVENLABS_API_KEY is not set." in result.stderr


def test_unexpected_error_hides_traceback_unless_verbose(fake_pipeline):
    fake_pipeline.behaviour["https://youtu.be/bug"] = RuntimeError("boom")
    quiet = invoke("https://youtu.be/bug")
    assert quiet.exit_code == 1
    assert "boom" in quiet.stderr and "Traceback" not in quiet.stderr
    loud = invoke("https://youtu.be/bug", "-v")
    assert loud.exit_code == 1
    assert "Traceback" in loud.stderr


def test_keyboard_interrupt_exits_130_quietly(fake_pipeline):
    fake_pipeline.behaviour["https://youtu.be/ctl"] = KeyboardInterrupt()
    result = invoke("https://youtu.be/ctl")
    assert result.exit_code == 130
    assert "Traceback" not in result.stderr


def test_repeated_clean_messages_hidden_when_piped(fake_pipeline):
    result = invoke("https://youtu.be/abc")
    assert "chunk 1/2" not in result.stderr
    verbose = invoke("https://youtu.be/abc", "-v")
    assert "chunk 1/2" in verbose.stderr


def test_python_dash_m_archiver_starts_the_cli():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "archiver", "--help"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0
    assert "Usage: archiver" in result.stdout and "--vault" in result.stdout
