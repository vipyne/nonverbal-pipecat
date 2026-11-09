import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";

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
