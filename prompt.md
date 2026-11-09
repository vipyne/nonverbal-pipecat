# Task: build a Pipecat voice bot that reads my nonverbal cues from the webcam

Build a local, browser-based voice bot with Pipecat. I talk to it in the browser, and it also watches my webcam to detect nonverbal cues: nodding, head shaking, distraction, confusion, amusement. It uses those cues to shape its replies. Work in the milestones below, run and verify each one before moving on, and commit after each.

## Ground rules
- Pipecat's APIs change between versions. Before writing server code, install the packages and **read the installed source and examples**. That includes the client-message handling on `RTVIProcessor`, the `SmallWebRTCTransport` server setup, and context/aggregator classes. Do not guess event or class names from memory. If the docs and installed source disagree, the source wins.
- Keep cue detection **in the browser**. Send only small JSON events to the server, never video frames, for the fast path.
- Read API keys from `.env` (provide `.env.example`). Default services are Deepgram STT, Anthropic Claude LLM and Cartesia TTS. Keep each one swappable in a single config spot.
- Use `uv` for Python and `npm` + Vite + TypeScript (vanilla, no framework) for the client.
- Shell commands must work in Bash on macOS. Note any Linux differences in the README.

## Architecture

**Fast lane (browser, ~30 Hz):**
- Path: webcam → MediaPipe Face Landmarker (`@mediapipe/tasks-vision`, VIDEO mode, `outputFaceBlendshapes: true`, `outputFacialTransformationMatrixes: true`) → cue detector → RTVI client messages.

**Slow lane (optional, off by default):**
- Path: WebRTC video track → server samples 1 frame every ~4 s → a VLM (Pipecat's Moondream service) → scene-level notes such as "looking at phone" or "another person present".
- Put it behind a `ENABLE_VLM=1` flag.

**Server:**
- Transport and pipeline: `SmallWebRTCTransport` plus a standard STT → LLM → TTS pipeline.
- Add a custom `NonverbalContextGate` FrameProcessor that holds the rolling cue state.

## Client details

**Face landmarker:**
- Run it in a Web Worker, posting `ImageBitmap`s. If tasks-vision fails in a worker, fall back to the main thread throttled to 15 fps and note why.
- Feed it the **unmirrored** video. Mirror only the on-screen preview with CSS, because left/right blendshapes invert if the input is mirrored.
- Download the model to `public/face_landmarker.task` from `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`.

**Head pose:**
- Take it from the column-major 4x4 matrix `m`: `pitch = atan2(m[6], m[10])` and `yaw = asin(-m[2])`, in degrees.
- Verify the signs empirically and expose a sign-flip constant.

**Calibration:**
- Collect a per-user baseline over the first 10 s: mean and std of pitch, yaw and relevant blendshapes.
- Make all thresholds relative to that baseline.

**Cue detector:** a pure TS module with no DOM access, so it's unit-testable. It emits these cues:

| Cue | Rule |
|---|---|
| `nod` | high-pass pitch (subtract EMA, alpha≈0.05); ≥2 velocity sign reversals with \|hp\|>3° within 1.2 s |
| `shake` | same rule on yaw |
| `distracted` | \|yaw - baseline\| > 25°, or `eyeLookOut*` high, or no face, sustained >2 s; emit start/end |
| `confused` | `browDown*` + `eyeSquint*` above baseline + 2σ for >0.7 s |
| `amused` | `mouthSmile*` above baseline + 2σ for >0.5 s |
| `yawn` | `jawOpen` high for >1 s while the mic VAD says I'm not speaking |

Debounce each cue type with a 1 s refractory period.

**Message schema:** `{type:"nonverbal", cue, t_ms, confidence, meta?}`. Send it via the Pipecat JS client's client-message API.

**Debug HUD:** I'm a visual learner, so this matters.
- A live canvas sparkline of high-passed pitch and yaw over the last 5 s, with threshold lines.
- A scrolling cue log.
- A face-detected indicator.
- Current calibration state.

## Server details (`NonverbalContextGate`)
- **Aggregate cues during each bot utterance.** Track bot-started and bot-stopped speaking frames. When the next user turn reaches the LLM, prepend a compact system note such as `[nonverbal during your last reply: nodded x2, confused at 4.1s, gaze steady]`. Add nothing if there were no cues.
- **Nod/shake as an answer.** If the bot's last utterance ended with a yes/no question and the user stays silent but sends `nod` or `shake` within 2.5 s, inject a synthetic user message `(nods)` or `(shakes head)` and trigger a response.
- **Distraction.** If `distracted` lasts more than 4 s while the bot is speaking, the bot pauses and briefly checks in after I return. Keep this behind a flag.
- **Cues never interrupt.** Nonverbal events must never trigger interruption or cancel TTS.
- **System prompt.** Tell the LLM what the nonverbal notes mean and to use them lightly. It should never narrate "I see you nodded" unless that's natural.

## Milestones
1. Scaffold a server and client where voice-only conversation works end to end. Include README run steps.
2. Get the face landmarker plus HUD showing live pose sparklines, with no cues yet.
3. Build the cue detector with unit tests (vitest) that feed synthetic signals: a 2 Hz pitch sinusoid at 8° → nod; a 0.3 Hz drift → no nod; yaw held at 30° for 3 s → distracted; noise → nothing.
4. Wire cue messages to the server and build `NonverbalContextGate`, with pytest tests that drive it with fake frames: aggregation, nod-as-answer, no interruption.
5. Add the optional VLM slow lane behind its flag.
6. Clean up: one command to start everything (a `Makefile` or `scripts/dev.sh`). Run the full test suite. Update the README with architecture, tuning knobs and known limitations.

## Done means
- `make dev` (or equivalent) starts server and client, and I can talk to the bot at `http://localhost:5173`.
- Nodding during the bot's reply visibly shows up in the HUD log and in the server log's injected context note.
- Answering "Want me to continue?" with only a nod makes the bot continue.
- All tests pass. Report what you verified by running it, versus what you couldn't verify without a human in front of the camera.