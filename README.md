# archiver

Give it a YouTube link and it writes a note into your Obsidian vault. The note has a short summary, a contents list with timestamp links, and the full transcript split by topic. The transcript comes from ElevenLabs Scribe speech-to-text, so it works whether or not the video has captions. Claude Sonnet 5.5 removes fillers and fixes misheard words. Claude Opus 5.5 finds the topics and writes the summary. Russian and English videos are supported, and the note stays in the video's language.

A note looks like this:

```markdown
---
title: How compilers work
url: https://www.youtube.com/watch?v=ID
video_id: ID
channel: Some Channel
published: 2026-10-01
duration_min: 42
language: en
tags:
  - video
  - youtube
  - compilers
created: 2026-11-15
---
![](https://www.youtube.com/watch?v=ID)

> [!summary] TL;DR
> A walk through the stages of a compiler, from parsing to code generation.

## Key points
- ...

## Contents
- [00:00](https://youtu.be/ID?t=0) Parsing: how source text becomes a tree
- [05:12](https://youtu.be/ID?t=312) Optimization: what the compiler rewrites

## Parsing
[00:00](https://youtu.be/ID?t=0)

A compiler starts by reading the source text and turning it into a tree...
```

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- [Deno](https://deno.com) or Node.js (yt-dlp needs a JavaScript runtime for YouTube; archiver uses whichever is installed)
- ffmpeg
- An Anthropic API key
- An ElevenLabs API key

| | macOS (Homebrew) | Linux (Debian/Ubuntu) |
|---|---|---|
| uv | `brew install uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Deno | `brew install deno` | `curl -fsSL https://deno.land/install.sh \| sh` |
| ffmpeg | `brew install ffmpeg` | `sudo apt install ffmpeg` |

On Windows (PowerShell), then open a new terminal so the programs are on your PATH:

```powershell
winget install astral-sh.uv DenoLand.Deno Gyan.FFmpeg
```

## Install

```sh
git clone https://github.com/n-drobyshevski/archiver
cd archiver
uv sync
uv run archiver <youtube-url>
```

For a global `archiver` command, run `uv tool install .` instead.

## Configuration

The two secrets are read from the environment. The easiest way to supply them is a `.env` file, which archiver loads at start: copy `.env.example` to `.env` in the folder you run archiver from (it is git-ignored), or to `~/.config/archiver/.env` so it works from anywhere, and fill in:

```
ANTHROPIC_API_KEY=...
ELEVENLABS_API_KEY=...
```

A variable already set in the terminal wins over the file. Or set them in the terminal:

```sh
export ANTHROPIC_API_KEY=...
export ELEVENLABS_API_KEY=...
```

In PowerShell, `export` does not exist. For the current window:

```powershell
$env:ANTHROPIC_API_KEY = "..."
$env:ELEVENLABS_API_KEY = "..."
```

To keep them, run `setx ANTHROPIC_API_KEY "..."` and `setx ELEVENLABS_API_KEY "..."` once and open a new window.

Everything else goes in `~/.config/archiver/config.toml` (on Windows, `%USERPROFILE%\.config\archiver\config.toml`; `$XDG_CONFIG_HOME` is honoured). Only `vault` is required. Values shown are the defaults:

```toml
vault = "~/Documents/MyVault"        # no default: your Obsidian vault folder
folder = "Videos"                    # subfolder of the vault for notes
cache_dir = "~/.cache/archiver"

cookies_from_browser = "firefox"     # default: unset. Also "chrome", "safari", ...
max_duration_min = 240               # longer videos are refused

stt_model = "scribe_v3"
diarize = true                       # label speakers when there are two or more
stt_timeout_s = 1800

clean_model = "claude-sonnet-5-5"
clean_effort = "low"                 # low, medium, high, xhigh or max
chunk_words = 1350                   # size of each cleanup request
concurrency = 4                      # cleanup requests in flight at once

outline_model = "claude-opus-5-5"
outline_effort = "high"

refusal_fallbacks = true             # let the API retry elsewhere when a request is declined
```

Every setting also has an environment variable with the `ARCHIVER_` prefix, for example `ARCHIVER_VAULT`, `ARCHIVER_FOLDER`, `ARCHIVER_CLEAN_MODEL`, `ARCHIVER_CONCURRENCY`.

Precedence, highest first: command-line option, environment variable, config file, default.

## Usage

```sh
archiver https://www.youtube.com/watch?v=ID                 # one video
archiver URL1 URL2 URL3                                     # several, one after another
archiver URL --vault ~/Notes --folder Talks                 # override the config for this run
archiver URL --force                                        # ignore the cache and redo every stage
archiver URL --overwrite                                    # replace a note you have edited
```

`uv run python -m archiver ...` is the same command as `uv run archiver ...`; use it where the `archiver` launcher is blocked (see Troubleshooting).

The note's path is printed on stdout, so `archiver URL | pbcopy` or `open "$(archiver URL)"` works. Progress, warnings and token use go to stderr; add `-v` for every progress message and full tracebacks. The exit code is 0 when everything worked, 1 if any video failed (the others still run), and 2 when setup is wrong (missing key, binary or vault).

## How it works

| Stage | What it does | Cached as |
|---|---|---|
| preflight | Checks the vault folder, both API keys, a JavaScript runtime and `ffmpeg` before any paid call. | none |
| fetch | yt-dlp reads title, channel, date, duration, description and chapters, and downloads the audio. Live streams and videos over `max_duration_min` are refused. | audio file (metadata is re-read each run) |
| transcribe | Sends the audio to Scribe v3, with speaker labels. | `stt_raw.json` |
| segment | Splits the words into 5 to 15 second segments in plain code. Free. | none |
| clean | Sonnet cleans the text in chunks of about 1,350 words, using a shared glossary of names and terms. A check compares each result with the source; a chunk that lost or invented too much is retried smaller, then kept raw. | per chunk, `clean/` |
| outline | Opus writes the TL;DR, key points and topics. Chapters set by the creator are used as topic boundaries when they cover the video. | `outline.json` |
| render + write | Builds the Markdown and writes it into the vault through a temp file, so Obsidian Sync never sees a half-written note. | none |

The cache lives in `~/.cache/archiver/<video_id>/`. Each entry records what produced it, so running a video again costs nothing unless a prompt or model changed, and then only the affected stages are redone. A run that failed halfway resumes where it stopped. If you have edited a note since the tool wrote it, the tool refuses to replace it unless you pass `--overwrite`.

## Cost

About $0.20 per hour of audio for speech-to-text, plus roughly $0.30 to $0.60 per hour of video for Claude (Russian uses more tokens than English). After each run the tool prints the actual token use per model and the approximate Claude cost (speech-to-text is billed separately by ElevenLabs).

## Troubleshooting

- **"An Application Control policy has blocked this file" (os error 4551), on Windows.** Smart App Control refuses the `archiver.exe` launcher because `uv` generates it on your machine and it is not signed. Start the tool through Python instead: `uv run python -m archiver <url>`.
- **"Sign in to confirm you're not a bot".** Set `cookies_from_browser = "firefox"` (or `chrome`, `safari`) in the config file, with that browser signed in to YouTube. Running from a home network instead of a server or VPN also helps.
- **Other YouTube download errors.** YouTube changes often. Update yt-dlp first: `uv lock --upgrade-package yt-dlp && uv sync`.
- **A note is refused as edited.** The file in the vault changed after the tool wrote it. Pass `--overwrite` to replace it, or move your edits elsewhere.
- **A warning callout about parts of the transcript shown as raw speech-to-text.** For those parts the cleanup result did not pass the check against the source, even after a retry, so the original transcript text was kept. Everything else in the note is cleaned. Running with `--force` may give a different result.
- **The note does not show up on your phone.** Obsidian Sync only uploads the file while the Obsidian app is running on the machine where it was written.

## Limits and what's next

- YouTube only.
- Always speech-to-text; YouTube captions are not used.
- One speech-to-text provider (ElevenLabs Scribe).
- Planned: a Telegram bot that calls the same pipeline.
