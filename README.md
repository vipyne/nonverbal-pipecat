# non-verbal pipecat

A local, browser-based [Pipecat](https://pipecat.ai) voice bot that also watches
your webcam: the browser detects nonverbal cues (nod, shake, distracted,
confused, amused, yawn) and sends small JSON events to the server, which folds
them into the LLM's context. A silent nod can answer a yes/no question.

Status: voice-only pipeline (milestone 1). The webcam fast lane, cue detector,
and server-side `NonverbalContextGate` land in later milestones — see

## Architecture

```
browser ──mic audio──▶ SmallWebRTC ──▶ Deepgram STT ──▶ Claude ──▶ Cartesia TTS ──▶ bot audio
```

- `server/bot.py` — pipeline and the Pipecat dev-runner entry point.
- `server/services.py` — the single swap spot for STT / LLM / TTS and the
  system prompt.
- `client/src/main.ts` — Pipecat JS client over SmallWebRTC: Connect button,
  mic in, bot audio out, transcript.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed by uv itself)
- Node.js 22+ and npm
- API keys for Deepgram (STT), Anthropic (LLM) and Cartesia (TTS)

## Setup

```bash
cp .env.example .env
```

Fill in `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY` and `CARTESIA_API_KEY`. `.env`
is gitignored and never leaves your machine; the server loads it from the repo
root.

## Run

Two terminals.

Terminal 1 — server (Pipecat dev runner on <http://localhost:7860>):

```bash
cd server && uv run bot.py
```

Terminal 2 — client (Vite on <http://localhost:5173>):

```bash
cd client && npm run dev
```

Open <http://localhost:5173> in Chrome, click **Connect**, grant microphone
access, and talk. Ctrl-C in each terminal stops it.

The client posts its WebRTC offer to `http://localhost:7860/api/offer`. Point it
somewhere else with `VITE_BOT_OFFER_URL` in `client/.env.local`.

## Tests

```bash
cd client && npx vitest run
```

```bash
cd server && uv run pytest
```

## macOS vs Linux

- Everything here is plain Bash and works on both. `uv` and `npm` install all
  dependencies into the repo (`server/.venv`, `client/node_modules`).
- Mic and webcam access require a secure context. `http://localhost` counts as
  one in both Chrome and Firefox, so no TLS setup is needed for local dev.
- macOS: Chrome asks for mic/camera permission per site, and macOS itself asks
  once per app under System Settings → Privacy & Security.
- Linux: if the browser sees no input device, check that PipeWire/PulseAudio is
  running; Chrome under Wayland sometimes needs `--enable-features=WebRTCPipeWireCapturer`
  for camera capture.
- `timeout` (used in a few scripted checks) ships with GNU coreutils on Linux.
  macOS has no `timeout` by default — use a background process plus `kill`, or
  install coreutils and use `gtimeout`.
