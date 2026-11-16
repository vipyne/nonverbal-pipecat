# non-verbal pipecat

A local, browser-based [Pipecat](https://pipecat.ai) voice bot that also watches
your webcam: the browser detects nonverbal cues (nod, shake, distracted,
confused, amused, yawn) and sends small JSON events to the server, which folds
them into the LLM's context. A silent nod can answer a yes/no question.

Status: all milestones from `prompt.md` are implemented. Voice pipeline,
webcam fast lane with cue detection, the server-side `NonverbalContextGate`,
and the optional VLM slow lane are all wired and test-covered. See
[What's verified](#whats-verified-vs-what-needs-a-human) for the split
between what ran in CI-style checks and what still needs a human on camera.

## Architecture

```
                    ┌─ mic audio ──▶ Deepgram STT ─┐
browser ──SmallWebRTC┤                             ├─▶ Claude ──▶ Cartesia TTS ──▶ bot audio
                    └─ cue JSON ──▶ NonverbalContextGate (folds cues into context) ─┘
   │
   ├─webcam──▶ worker: MediaPipe Face Landmarker ──▶ head pose + blendshapes
   │              │
   │              ├─▶ cue detector (nod/shake/distracted/confused/amused/yawn) ─▶ HUD + RTVI client messages
   │              └─▶ debug HUD (sparklines, cue log, calibration state)
   │
   └─(ENABLE_VLM=1 only)──▶ 1 fps video track ──▶ server SceneSampler (1 frame / 4s) ──▶ Moondream ──▶ scene note (low confidence)
```

The fast lane (cue detector) never leaves the browser except as small JSON
events. The slow lane (VLM) is the only path that sends video to the server,
and only when `ENABLE_VLM=1`.

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

One command, from the repo root — downloads the model if missing, installs
dependencies if missing, starts server and client together, and stops both on
Ctrl-C:

```bash
make dev
```

Or run the two halves in separate terminals (same commands `make dev` runs):

Terminal 1 — server (Pipecat dev runner on <http://localhost:7860>):

```bash
cd server && uv run bot.py
```

Terminal 2 — client (Vite on <http://localhost:5173>):

```bash
cd client && npm run dev
```

Open <http://localhost:5173> in Chrome, grant microphone and camera access,
click **Connect**, and talk. Ctrl-C stops both.

To try the distraction check-in (off by default), set the flag before
starting the server:

```bash
cd server && ENABLE_DISTRACTION_CHECKIN=1 uv run bot.py
```

The HUD shows high-passed pitch (nod axis) and yaw (shake axis) over the last
5 s, whether a face is detected, and the calibration state — baselines freeze
after about 10 s of face-visible frames. Each lane's dashed threshold lines sit
at that axis's current adaptive turn threshold (1.2°–3°), printed in the lane
label; they tighten on a still head and widen when the axis is noisy.

The client posts its WebRTC offer to `http://localhost:7860/api/offer`. Point it
somewhere else with `VITE_BOT_OFFER_URL` in `client/.env.local`.

## Tuning knobs

Client (`client/src/landmarker/pose.ts`, `client/src/cues/detector.ts`,
`client/src/cues/highpass.ts`):

- `PITCH_SIGN`, `YAW_SIGN` — flip to `-1` if an axis reads backwards on your
  setup (verified empirically at the HUD gate; both are `1` as shipped).
- `HIGHPASS_ALPHA` (0.05 at ~30 Hz) — EMA high-pass coefficient, corner
  ≈0.24 Hz. Raising it rejects slower drift but also dulls real nods.
- `TURN_MIN_DEG` (1.2°) / `TURN_MAX_DEG` (3°) — band for the adaptive turn
  threshold: the minimum high-passed pitch/yaw swing per oscillation reversal
  that counts toward a `nod`/`shake`. The floor is what makes subtle, natural
  nods register; the ceiling is the old static bar, so it never gets less
  sensitive than that.
- `NOISE_K` (4) — threshold = `NOISE_K` × the axis's own noise level, clamped
  to the band above. Lower it for more sensitivity in a noisy signal.
- `NOISE_ALPHA` (0.01 at ~30 Hz, ≈3 s) — EMA coefficient for that noise level.
  Only frames at or below the current threshold feed it, so a gesture can
  never raise its own bar; it tracks from the first frame, through calibration.
- `OSCILLATION_WINDOW_MS` (1200 ms) — window for counting the ≥2 reversals
  that make a `nod`/`shake`.
- `REFRACTORY_MS` (1000 ms) — debounce per cue type; extends while the motion
  continues, so one long nod is one cue, not one per second.
- `FLOORS` (smile 0.25, browDown 0.25, squint 0.2, lookOut 0.4, jawOpen 0.4)
  and `MIN_STD` (0.02) — absolute guards on top of the baseline+2σ rule for
  blendshape-driven cues (`confused`, `amused`, `yawn`, `distracted`'s
  `eyeLookOut*` path), so a dead-still calibration (σ≈0) doesn't fire on any
  twitch.

Server (`server/nonverbal_gate.py`, `server/vlm_lane.py`):

- `ANSWER_WINDOW_S` (2.5 s) — how long after the bot stops speaking a silent
  `nod`/`shake` still answers a yes/no question.
- `DISTRACTION_CHECKIN_S` (4.0 s) — distracted span, opened while the bot was
  speaking, that earns a post-reply check-in. Only applies when
  `ENABLE_DISTRACTION_CHECKIN=1`.
- `MAX_SCENE_NOTES` (2) — most recent VLM scene strings kept per context note.
- `SAMPLE_INTERVAL_S` (4.0 s) — minimum gap between VLM samples; a slow VLM
  query delays the next sample rather than queuing one.

Flags (`.env`, both default off):

- `ENABLE_VLM` — turns on the slow lane (server video sampling + Moondream +
  client camera track). Requires `uv add "pipecat-ai[moondream]"` in
  `server/` first.
- `ENABLE_DISTRACTION_CHECKIN` — turns on the post-reply distraction check-in.
  Off by default because the plan's "pause" would otherwise have to cancel
  TTS, which cues must never do; the check-in waits for the bot's utterance to
  finish instead.

## Known limitations

- Pose sign constants and the worker-vs-main-thread landmarker path are only
  confirmed on the machine(s) a human has actually run the HUD gate on; a
  different camera, OS, or browser could need `PITCH_SIGN`/`YAW_SIGN` flipped
  or fall back to the 15 fps main-thread path.
- `nod`/`shake` only answer a yes/no question when the bot's last sentence
  ends in `?` and starts with a word in `YES_NO_LEADS`
  (`server/nonverbal_gate.py`) — "What do you think?" is not treated as
  answerable, by design.
- The VLM slow lane needs a separate, several-GB install
  (`pipecat-ai[moondream]`, torch + transformers) not included by default.
- Distraction detection and the VLM sampler have no notion of multiple faces
  beyond MediaPipe's single-face mode (`numFaces: 1`); a second person in
  frame is only ever described qualitatively by the VLM, never cue-detected.

## What's verified vs. what needs a human

Machine-verified (every iteration ran these commands and saw them pass):

- All 23 vitest + 40 pytest tests, and `npm run build` / `uv run python -c
  "import bot"` importing cleanly.
- The server starting and listening with dummy keys, and the client dev
  server serving `http://localhost:5173` (including the MediaPipe wasm
  runtime's `?import` URL, which previously 500'd).
- `make dev` bringing up both processes and tearing them down with no leaked
  ports or processes.
- The VLM sampler's timing/in-flight guards and the `ENABLE_VLM`/
  `ENABLE_DISTRACTION_CHECKIN` flags gating their code paths off by default,
  via mutation-checked unit tests.

Needs a human in front of a camera and microphone (see `ralph/HUMAN.md` for
the exact gated steps):

- A real spoken conversation (STT → LLM → TTS round trip with audio).
- That the HUD sparklines track actual head motion and the pitch/yaw signs
  match nodding/shaking (confirmed as shipped: both `+1`).
- That nodding during the bot's reply shows up in the HUD cue log and the
  server's folded context note, and that a silent nod answers a yes/no
  question end to end.
- The `make dev` full "Done means" checklist run in one sitting.

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
