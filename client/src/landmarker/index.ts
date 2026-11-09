/** Webcam → landmarker pump. Worker by default, throttled main thread as fallback. */
import type { LandmarkerFrame, WorkerResponse } from "./types";

/** Served by the vite `mediapipe-wasm` plugin. */
const WASM_BASE = "/mediapipe";
const MODEL_URL = "/face_landmarker.task";

const WORKER_FPS = 30;
const FALLBACK_FPS = 15;
/** Worker init is a wasm download; give it room before falling back. */
const WORKER_INIT_TIMEOUT_MS = 15000;

export type FrameHandler = (frame: LandmarkerFrame) => void;

export interface TrackerStatus {
  /** `"worker"` or `"main-thread"` once running. */
  mode: string;
  fps: number;
}

export interface StartOptions {
  /** Preview element; the caller mirrors it with CSS, the pump stays unmirrored. */
  video: HTMLVideoElement;
  onFrame: FrameHandler;
  onStatus?: (status: TrackerStatus) => void;
  onError?: (message: string) => void;
}

interface Engine {
  mode: string;
  fps: number;
  push(bitmap: ImageBitmap, t_ms: number): void;
  close(): void;
}

/** Worker engine; rejects if the worker errors or stays silent past the timeout. */
function startWorkerEngine(onFrame: FrameHandler, onError?: (m: string) => void): Promise<Engine> {
  return new Promise((resolve, reject) => {
    let worker: Worker;
    try {
      worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    } catch (err) {
      reject(err);
      return;
    }

    let ready = false;
    let busy = false;
    const timer = setTimeout(() => {
      if (!ready) {
        worker.terminate();
        reject(new Error("landmarker worker init timed out"));
      }
    }, WORKER_INIT_TIMEOUT_MS);

    worker.onerror = (event) => {
      if (ready) {
        onError?.(event.message);
        return;
      }
      clearTimeout(timer);
      worker.terminate();
      reject(new Error(event.message || "worker failed to load"));
    };

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data;
      if (msg.type === "ready") {
        ready = true;
        clearTimeout(timer);
        resolve({
          mode: "worker",
          fps: WORKER_FPS,
          push(bitmap, t_ms) {
            // Drop frames while inference is in flight; no unbounded queue.
            if (busy) {
              bitmap.close();
              return;
            }
            busy = true;
            worker.postMessage({ type: "frame", bitmap, t_ms }, [bitmap]);
          },
          close() {
            worker.terminate();
          },
        });
        return;
      }
      if (msg.type === "error") {
        busy = false;
        if (!ready) {
          clearTimeout(timer);
          worker.terminate();
          reject(new Error(msg.message));
        } else {
          onError?.(msg.message);
        }
        return;
      }
      busy = false;
      const { type: _type, ...frame } = msg;
      onFrame(frame);
    };

    worker.postMessage({ type: "init", wasmBase: WASM_BASE, modelUrl: MODEL_URL });
  });
}

/** Fallback path; the MediaPipe bundle loads on demand so it stays out of the main chunk. */
async function startMainThreadEngine(onFrame: FrameHandler): Promise<Engine> {
  const { createFaceLandmarker, flattenResult } = await import("./create");
  const landmarker = await createFaceLandmarker(WASM_BASE, MODEL_URL);
  let lastStamp = 0;
  return {
    mode: "main-thread",
    fps: FALLBACK_FPS,
    push(bitmap, t_ms) {
      try {
        const stamp = Math.max(Math.round(t_ms), lastStamp + 1);
        lastStamp = stamp;
        onFrame(flattenResult(landmarker.detectForVideo(bitmap, stamp), t_ms));
      } finally {
        bitmap.close();
      }
    },
    close() {
      landmarker.close();
    },
  };
}

/** Owns the camera stream, the engine, and the capture loop. */
export class FaceTracker {
  private stream: MediaStream | null = null;
  private engine: Engine | null = null;
  private interval: ReturnType<typeof setInterval> | null = null;
  private stopped = false;

  async start(opts: StartOptions): Promise<TrackerStatus> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: WORKER_FPS },
      audio: false,
    });
    opts.video.srcObject = this.stream;
    await opts.video.play();

    try {
      this.engine = await startWorkerEngine(opts.onFrame, opts.onError);
    } catch (err) {
      opts.onError?.(`worker landmarker unavailable, using main thread: ${String(err)}`);
      this.engine = await startMainThreadEngine(opts.onFrame);
    }

    if (this.stopped) {
      this.engine.close();
      throw new Error("tracker stopped during startup");
    }

    const status: TrackerStatus = { mode: this.engine.mode, fps: this.engine.fps };
    opts.onStatus?.(status);
    this.pump(opts.video, status.fps);
    return status;
  }

  private pump(video: HTMLVideoElement, fps: number) {
    this.interval = setInterval(async () => {
      if (!this.engine || video.readyState < 2) return;
      try {
        const bitmap = await createImageBitmap(video);
        this.engine.push(bitmap, performance.now());
      } catch {
        // Transient capture failure (tab hidden, track muted); next tick retries.
      }
    }, Math.round(1000 / fps));
  }

  stop() {
    this.stopped = true;
    if (this.interval !== null) clearInterval(this.interval);
    this.interval = null;
    this.engine?.close();
    this.engine = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
  }
}

export type { LandmarkerFrame, Blendshapes } from "./types";
