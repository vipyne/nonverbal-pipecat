import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";

import { Calibrator } from "./cues/calibration";
import { HighPass } from "./cues/highpass";
import { Hud } from "./hud";
import { FaceTracker } from "./landmarker";
import { headPose } from "./landmarker/pose";
import { TranscriptView } from "./transcript";

// Pipecat dev runner's WebRTC signalling endpoint.
const OFFER_URL = import.meta.env.VITE_BOT_OFFER_URL ?? "http://localhost:7860/api/offer";

const connectBtn = document.querySelector<HTMLButtonElement>("#connect")!;
const disconnectBtn = document.querySelector<HTMLButtonElement>("#disconnect")!;
const stateEl = document.querySelector<HTMLElement>("#state")!;
const botAudio = document.querySelector<HTMLAudioElement>("#bot-audio")!;
const transcript = new TranscriptView(document.querySelector<HTMLElement>("#transcript")!);

const client = new PipecatClient({
  transport: new SmallWebRTCTransport({
    webrtcRequestParams: { endpoint: OFFER_URL },
  }),
  enableMic: true,
  enableCam: false,
  callbacks: {
    onTransportStateChanged: (state) => {
      stateEl.textContent = state;
    },
    onConnected: () => {
      connectBtn.disabled = true;
      disconnectBtn.disabled = false;
    },
    onDisconnected: () => {
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

const calibrator = new Calibrator();
const pitchHp = new HighPass();
const yawHp = new HighPass();
const tracker = new FaceTracker();

tracker
  .start({
    video: preview,
    onFrame: (frame) => {
      const pose = headPose(frame.matrix);
      calibrator.push({ ...pose, t_ms: frame.t_ms, faceDetected: frame.faceDetected, blendshapes: frame.blendshapes });
      hud.setCalibration(calibrator.state, calibrator.progress(frame.t_ms), calibrator.baseline);
      hud.push({
        t_ms: frame.t_ms,
        faceDetected: frame.faceDetected,
        pitch: pitchHp.push(pose.pitch),
        yaw: yawHp.push(pose.yaw),
      });
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
