import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";

import { defineConfig, type Plugin } from "vite";

const require = createRequire(import.meta.url);

/** MediaPipe's FilesetResolver needs these served under one unhashed path. */
const WASM_FILES = ["vision_wasm_internal.js", "vision_wasm_internal.wasm"];

function mediapipeWasm(): Plugin {
  return {
    name: "mediapipe-wasm",
    buildStart() {
      const dest = join(import.meta.dirname, "public", "mediapipe");
      mkdirSync(dest, { recursive: true });
      for (const file of WASM_FILES) {
        copyFileSync(require.resolve(`@mediapipe/tasks-vision/${file}`), join(dest, file));
      }
    },
  };
}

export default defineConfig({
  plugins: [mediapipeWasm()],
  server: {
    port: 5173,
    strictPort: true,
  },
  worker: {
    format: "es",
  },
});
