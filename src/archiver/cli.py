"""Command line front end: `archiver URL [URL ...]`."""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path
from typing import Annotated, Any

import typer

from archiver.errors import ArchiverError, PreflightError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Turn YouTube videos into cleaned, topic-split notes in an Obsidian vault.",
)

EXIT_FAILED = 1
EXIT_PREFLIGHT = 2
EXIT_INTERRUPTED = 130


def err(text: str) -> None:
    typer.echo(text, err=True)


class ProgressPrinter:
    """One line per stage; repeated messages inside a stage update in place on a terminal."""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.tty = sys.stderr.isatty()
        self.stage: str | None = None
        self.inplace = False  # the cursor sits at the end of an unfinished line

    def __call__(self, stage: str, message: str) -> None:
        line = f"[{stage}] {message}"
        if stage != self.stage:
            self.stage = stage
            self.finish_line()
            err(line)
        elif self.tty and stage == "clean":
            sys.stderr.write(f"\r\033[K{line}")
            sys.stderr.flush()
            self.inplace = True
        elif self.verbose:
            err(line)

    def finish_line(self) -> None:
        if self.inplace:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.inplace = False


def report(result: Any) -> None:
    typer.echo(str(result.path))
    for warning in result.warnings:
        err(f"warning: {warning}")
    for u in result.usage:
        err(
            f"usage: {u.model}: {u.calls} calls, "
            f"{u.input_tokens:,} in / {u.output_tokens:,} out tokens"
        )
    if result.cost_usd is not None:
        err(f"approximate cost: ${result.cost_usd:.2f}")


@app.command()
def main(
    urls: Annotated[list[str], typer.Argument(help="One or more YouTube URLs.")],
    vault: Annotated[
        Path | None, typer.Option("--vault", help="Obsidian vault folder (overrides config).")
    ] = None,
    folder: Annotated[
        str | None, typer.Option("--folder", help="Subfolder of the vault for notes.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Ignore the cache and redo every stage.")
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace a note that was edited since it was written."),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show every progress message and tracebacks.")
    ] = False,
) -> None:
    from archiver import pipeline  # imported here so --help works without the heavy deps
    from archiver.config import Settings

    overrides: dict[str, Any] = {}
    if vault is not None:
        overrides["vault"] = vault
    if folder is not None:
        overrides["folder"] = folder
    settings = Settings(**overrides)

    failed = 0
    progress = ProgressPrinter(verbose)
    try:
        for url in urls:
            progress.stage = None
            try:
                result = asyncio.run(
                    pipeline.run(
                        url, settings, force=force, overwrite=overwrite, on_progress=progress
                    )
                )
            except PreflightError as exc:
                progress.finish_line()
                err(f"error: {exc}")
                raise typer.Exit(EXIT_PREFLIGHT) from None
            except ArchiverError as exc:
                progress.finish_line()
                err(f"error: {exc}")
                failed += 1
            except Exception as exc:  # noqa: BLE001 - last resort, keep the batch going
                progress.finish_line()
                err(f"error: unexpected {type(exc).__name__}: {exc}")
                if verbose:
                    err(traceback.format_exc())
                failed += 1
            else:
                progress.finish_line()
                report(result)
    except KeyboardInterrupt:
        progress.finish_line()
        raise typer.Exit(EXIT_INTERRUPTED) from None
    if failed:
        raise typer.Exit(EXIT_FAILED)
