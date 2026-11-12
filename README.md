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

- `client/src/cues/detector.ts` — the cue detector; `client/src/vlm.ts` — the
  slow-lane flag the client reads.
- `server/nonverbal_gate.py` — folds cues into LLM context; `server/vlm_lane.py`
  — the optional slow lane.

By default video never leaves the browser: the camera feeds the landmarker in a
worker and only small JSON cue events go to the server. The optional slow lane
below is the one exception.

## Optional: VLM slow lane

Off by default (`ENABLE_VLM=0`). With `ENABLE_VLM=1` in `.env` the server also
receives the camera track at 1 fps, hands one frame every 4 s to a local
Moondream VLM, and folds its one-sentence scene description into the same
context note at low confidence. The fast lane stays JSON-only either way.

The client reads the same repo-root `.env` (vite's `envDir` is the repo root and
`ENABLE_` is an env prefix, so only `ENABLE_*` and `VITE_*` names reach the
browser — the API keys beside them do not), and sends its camera to the server
only when the flag is on.

The VLM itself is an extra install (torch + transformers, several GB):

```bash
cd server && uv add "pipecat-ai[moondream]"
```

Without it, `ENABLE_VLM=1` fails when the first client connects, naming that
command.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed by uv itself)
- Node.js 22+ and npm
- API keys for Deepgram (STT), Anthropic (LLM) and Cartesia (TTS)

## Setup

```bash
cp env.example .env
```

Fill in `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY` and `CARTESIA_API_KEY`. `.env`
is gitignored and never leaves your machine; the server loads it from the repo
root.

Download the Face Landmarker model (~3.6 MB, gitignored, skipped if present):

```bash
./scripts/fetch-model.sh
```

MediaPipe's wasm runtime is served from `client/node_modules` under
`/mediapipe` by the `mediapipe-wasm` vite plugin — dev middleware, and emitted
into `dist/mediapipe/` on build — so nothing is fetched from a CDN at runtime.
It is served from outside `public/` on purpose: the landmarker worker loads the
runtime with a dynamic `import()`, and vite refuses to resolve imports of
anything under `public/`. The worker gets the `vision_wasm_module_internal.*`
(ESM) pair; the main-thread fallback gets the classic `vision_wasm_internal.*`
pair.

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
