/** Shared types between the landmarker worker and the main thread. */

/** Blendshape scores keyed by MediaPipe category name (e.g. `jawOpen`). */
export type Blendshapes = Record<string, number>;

/** One landmarker inference result, stripped to what the cue detector needs. */
export interface LandmarkerFrame {
  /** Frame capture time, `performance.now()` on the main thread. */
  t_ms: number;
  faceDetected: boolean;
  /** Column-major 4x4 facial transformation matrix; empty when no face. */
  matrix: number[];
  blendshapes: Blendshapes;
}

export interface InitMessage {
  type: "init";
  /** Directory serving `vision_wasm_internal.{js,wasm}`. */
  wasmBase: string;
  modelUrl: string;
}

export interface FrameMessage {
  type: "frame";
  bitmap: ImageBitmap;
  t_ms: number;
}

export type WorkerRequest = InitMessage | FrameMessage;

export type WorkerResponse =
  | { type: "ready" }
  | { type: "error"; message: string }
  | ({ type: "result" } & LandmarkerFrame);
