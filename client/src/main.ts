import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";

import { CUE_MESSAGE_TYPE, CueDetector, type CueEvent } from "./cues/detector";
import { Hud } from "./hud";
import { FaceTracker } from "./landmarker";
import { headPose } from "./landmarker/pose";
import { TranscriptView } from "./transcript";
import { VLM_ENABLED } from "./vlm";

// Pipecat dev runner's WebRTC signalling endpoint.
const OFFER_URL = import.meta.env.VITE_BOT_OFFER_URL ?? "http://localhost:7860/api/offer";

const connectBtn = document.querySelector<HTMLButtonElement>("#connect")!;
const disconnectBtn = document.querySelector<HTMLButtonElement>("#disconnect")!;
const stateEl = document.querySelector<HTMLElement>("#state")!;
const botAudio = document.querySelector<HTMLAudioElement>("#bot-audio")!;
const transcript = new TranscriptView(document.querySelector<HTMLElement>("#transcript")!);

// Mic VAD state; `yawn` only counts while the user is quiet.
let userSpeaking = false;
// Cues only go over the data channel while a session is up.
let connected = false;

const client = new PipecatClient({
  transport: new SmallWebRTCTransport({
    webrtcRequestParams: { endpoint: OFFER_URL },
  }),
  enableMic: true,
  // Camera goes to the server only for the VLM slow lane; the face landmarker
  // below runs off its own getUserMedia stream regardless.
  enableCam: VLM_ENABLED,
  callbacks: {
    onTransportStateChanged: (state) => {
      stateEl.textContent = state;
    },
    onConnected: () => {
      connected = true;
      connectBtn.disabled = true;
      disconnectBtn.disabled = false;
    },
    onDisconnected: () => {
      connected = false;
      connectBtn.disabled = false;
      disconnectBtn.disabled = true;
      transcript.endTurn();
      transcript.append("note", "disconnected");
      transcript.endTurn();
    },
    onBotReady: () => {
      transcript.append("note", "bot ready");
      transcript.endTurn();
    },
    onUserInput: (data) => {
      if (!data.final) return;
      transcript.append("user", data.text);
      transcript.endTurn();
    },
    onBotOutput: (data) => {
      transcript.append("bot", data.text);
    },
    onBotStoppedSpeaking: () => {
      transcript.endTurn();
    },
    onUserStartedSpeaking: () => {
      userSpeaking = true;
    },
    onUserStoppedSpeaking: () => {
      userSpeaking = false;
    },
    onTrackStarted: (track, participant) => {
      if (participant?.local || track.kind !== "audio") return;
      botAudio.srcObject = new MediaStream([track]);
    },
    onError: (message) => {
      transcript.endTurn();
      transcript.append("note", `error: ${JSON.stringify(message.data)}`);
      transcript.endTurn();
    },
  },
});

connectBtn.addEventListener("click", async () => {
  connectBtn.disabled = true;
  try {
    await client.connect();
  } catch (err) {
    connectBtn.disabled = false;
    transcript.append("note", `connect failed: ${String(err)}`);
    transcript.endTurn();
  }
});

disconnectBtn.addEventListener("click", async () => {
  disconnectBtn.disabled = true;
  await client.disconnect();
});

// --- camera / face landmarker + HUD --------------------------------------

const preview = document.querySelector<HTMLVideoElement>("#preview")!;
const trackerStatusEl = document.querySelector<HTMLElement>("#tracker-status")!;

const hud = new Hud({
  canvas: document.querySelector<HTMLCanvasElement>("#hud-canvas")!,
  face: document.querySelector<HTMLElement>("#hud-face")!,
  calibration: document.querySelector<HTMLElement>("#hud-calibration")!,
  log: document.querySelector<HTMLElement>("#hud-log")!,
});

const detector = new CueDetector();
const tracker = new FaceTracker();

tracker
  .start({
    video: preview,
    onFrame: (frame) => {
      const pose = headPose(frame.matrix);
      const cues = detector.push({
        ...pose,
        t_ms: frame.t_ms,
        faceDetected: frame.faceDetected,
        blendshapes: frame.blendshapes,
        userSpeaking,
      });
      hud.setCalibration(
        detector.calibrationState,
        detector.calibrationProgress(frame.t_ms),
        detector.baseline,
      );
      hud.push({
        t_ms: frame.t_ms,
        faceDetected: frame.faceDetected,
        ...detector.highpass,
      });
      for (const cue of cues) {
        hud.logCue(cue.cue, cue.t_ms, cueDetail(cue));
        sendCue(cue);
      }
    },
    onStatus: (status) => {
      trackerStatusEl.textContent = `landmarker: ${status.mode} @ ${status.fps} fps`;
    },
    onError: (message) => {
      transcript.append("note", `camera: ${message}`);
      transcript.endTurn();
    },
  })
  .catch((err) => {
    trackerStatusEl.textContent = `camera unavailable: ${String(err)}`;
  });

window.addEventListener("beforeunload", () => tracker.stop());

/** Fast lane: one small JSON message per cue, fire-and-forget. */
function sendCue(cue: CueEvent): void {
  if (!connected) return;
  try {
    client.sendClientMessage(CUE_MESSAGE_TYPE, cue);
  } catch (err) {
    transcript.append("note", `cue send failed: ${String(err)}`);
    transcript.endTurn();
  }
}

/** HUD suffix: confidence plus the cue's own meta fields. */
function cueDetail(cue: CueEvent): string {
  const meta = Object.entries(cue.meta ?? {})
    .map(([k, v]) => `${k}=${v}`)
    .join(" ");
  return `conf ${cue.confidence.toFixed(2)}${meta ? `  ${meta}` : ""}`;
}
