import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

import { defineConfig, type Plugin } from "vite";

import { MEDIAPIPE_WASM_BASE, MEDIAPIPE_WASM_SUBPATHS } from "./src/landmarker/wasm-files.ts";

const require = createRequire(import.meta.url);

const names: readonly string[] = MEDIAPIPE_WASM_SUBPATHS;
const resolveWasmFile = (subpath: string) => require.resolve(`@mediapipe/tasks-vision/${subpath}`);

/**
 * Serves MediaPipe's wasm runtime from node_modules under `MEDIAPIPE_WASM_BASE`.
 * The landmarker worker loads the runtime with a dynamic `import()`, which vite
 * rewrites to `…js?import` and refuses to resolve for anything under `public/`
 * ("should not be imported from source code"). This middleware is installed
 * ahead of vite's own, so the loader is served as bytes, untransformed.
 */
function mediapipeWasm(): Plugin {
  return {
    name: "mediapipe-wasm",
    configureServer(server) {
      server.middlewares.use(MEDIAPIPE_WASM_BASE, (req, res, next) => {
        const name = (req.url ?? "").split("?")[0].replace(/^\//, "");
        if (!names.includes(name)) {
          next();
          return;
        }
        res.setHeader("Content-Type", name.endsWith(".wasm") ? "application/wasm" : "text/javascript");
        res.setHeader("Cache-Control", "no-cache");
        res.end(readFileSync(resolveWasmFile(name)));
      });
    },
    generateBundle() {
      for (const subpath of MEDIAPIPE_WASM_SUBPATHS) {
        this.emitFile({
          type: "asset",
          fileName: `mediapipe/${subpath}`,
          source: readFileSync(resolveWasmFile(subpath)),
        });
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
