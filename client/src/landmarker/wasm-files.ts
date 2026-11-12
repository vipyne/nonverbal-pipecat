/** MediaPipe wasm runtime served under one unhashed path, dev and build alike. */

/** Directory the `mediapipe-wasm` vite plugin serves. Outside `public/`. */
export const MEDIAPIPE_WASM_BASE = "/mediapipe";

/**
 * Subpaths exported by `@mediapipe/tasks-vision`. Classic pair for the
 * main-thread fallback, `_module_` pair for the ES module worker.
 */
export const MEDIAPIPE_WASM_SUBPATHS = [
  "vision_wasm_internal.js",
  "vision_wasm_internal.wasm",
  "vision_wasm_module_internal.js",
  "vision_wasm_module_internal.wasm",
] as const;
