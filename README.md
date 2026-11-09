# non-verbal pipecat

A local, browser-based [Pipecat](https://pipecat.ai) voice bot that also watches
your webcam: the browser detects nonverbal cues (nod, shake, distracted,
confused, amused, yawn) and sends small JSON events to the server, which folds
them into the LLM's context. A silent nod can answer a yes/no question.

Status: voice pipeline plus the webcam fast lane's landmarker and debug HUD
(milestones 1-2). The cue detector and server-side `NonverbalContextGate` land
in later milestones.

## Architecture

```
browser ──mic audio──▶ SmallWebRTC ──▶ Deepgram STT ──▶ Claude ──▶ Cartesia TTS ──▶ bot audio
   │
   └─webcam──▶ worker: MediaPipe Face Landmarker ──▶ head pose + blendshapes ──▶ HUD
```

- `server/bot.py` — pipeline and the Pipecat dev-runner entry point.
- `server/services.py` — the single swap spot for STT / LLM / TTS and the
  system prompt.
- `client/src/main.ts` — Pipecat JS client over SmallWebRTC: Connect button,
  mic in, bot audio out, transcript; also starts the camera and HUD.
- `client/src/landmarker/` — Face Landmarker in a Web Worker (`worker.ts`),
  camera pump and worker lifecycle (`index.ts`), head pose from the facial
  transformation matrix (`pose.ts`).
- `client/src/cues/` — baseline calibration (`calibration.ts`) and the
  high-pass used for nod/shake detection (`highpass.ts`).
- `client/src/hud/` — canvas sparklines, cue log, face and calibration state.

Video never leaves the browser: the camera feeds the landmarker in a worker and
only small JSON cue events go to the server.

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

Download the Face Landmarker model (~3.6 MB, gitignored, skipped if present):

```bash
./scripts/fetch-model.sh
```

MediaPipe's wasm runtime is copied from `client/node_modules` into
`client/public/mediapipe/` by the `mediapipe-wasm` vite plugin on every
`npm run dev` / `npm run build`, so nothing is fetched from a CDN at runtime.

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

Open <http://localhost:5173> in Chrome, grant microphone and camera access,
click **Connect**, and talk. Ctrl-C in each terminal stops it.

The HUD shows high-passed pitch (nod axis) and yaw (shake axis) over the last
5 s with ±3° threshold lines, whether a face is detected, and the calibration
state — baselines freeze after about 10 s of face-visible frames.

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
- If the landmarker cannot start in a Web Worker, it falls back to the main
  thread at 15 fps; the status line under the preview says which path is live.
- macOS: Chrome asks for mic/camera permission per site, and macOS itself asks
  once per app under System Settings → Privacy & Security.
- Linux: if the browser sees no input device, check that PipeWire/PulseAudio is
  running; Chrome under Wayland sometimes needs `--enable-features=WebRTCPipeWireCapturer`
  for camera capture.
- `timeout` (used in a few scripted checks) ships with GNU coreutils on Linux.
  macOS has no `timeout` by default — use a background process plus `kill`, or
  install coreutils and use `gtimeout`.
