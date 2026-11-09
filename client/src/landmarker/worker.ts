/// <reference lib="webworker" />
/** Runs FaceLandmarker off the main thread on ImageBitmaps posted at ~30 Hz. */
import type { FaceLandmarker } from "@mediapipe/tasks-vision";

import { createFaceLandmarker, flattenResult } from "./create";
import type { WorkerRequest, WorkerResponse } from "./types";

let landmarker: FaceLandmarker | null = null;
// detectForVideo rejects non-increasing timestamps (ms).
let lastStamp = 0;

function reply(message: WorkerResponse) {
  self.postMessage(message);
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;

  if (msg.type === "init") {
    try {
      landmarker = await createFaceLandmarker(msg.wasmBase, msg.modelUrl);
      reply({ type: "ready" });
    } catch (err) {
      reply({ type: "error", message: String(err) });
    }
    return;
  }

  if (!landmarker) {
    msg.bitmap.close();
    return;
  }

  try {
    const stamp = Math.max(Math.round(msg.t_ms), lastStamp + 1);
    lastStamp = stamp;
    const result = landmarker.detectForVideo(msg.bitmap, stamp);
    reply({ type: "result", ...flattenResult(result, msg.t_ms) });
  } catch (err) {
    reply({ type: "error", message: String(err) });
  } finally {
    msg.bitmap.close();
  }
};
