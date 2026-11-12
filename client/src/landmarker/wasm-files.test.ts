import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

import { MEDIAPIPE_WASM_BASE, MEDIAPIPE_WASM_SUBPATHS } from "./wasm-files";

const require = createRequire(import.meta.url);
const resolve = (subpath: string) => require.resolve(`@mediapipe/tasks-vision/${subpath}`);

describe("mediapipe wasm fileset", () => {
  it("serves from a path outside public/", () => {
    expect(MEDIAPIPE_WASM_BASE).toBe("/mediapipe");
  });

  it("lists both the classic and the ES module runtime, js + wasm", () => {
    expect([...MEDIAPIPE_WASM_SUBPATHS]).toEqual([
      "vision_wasm_internal.js",
      "vision_wasm_internal.wasm",
      "vision_wasm_module_internal.js",
      "vision_wasm_module_internal.wasm",
    ]);
  });

  it("resolves every subpath from the installed package", () => {
    for (const subpath of MEDIAPIPE_WASM_SUBPATHS) {
      expect(() => resolve(subpath)).not.toThrow();
    }
  });

  // Why the worker passes useModule=true and the main thread passes false.
  it("only the module variant is importable as ESM", () => {
    const esm = readFileSync(resolve("vision_wasm_module_internal.js"), "utf8");
    const classic = readFileSync(resolve("vision_wasm_internal.js"), "utf8");
    expect(esm).toContain("export default ModuleFactory");
    expect(classic).not.toContain("export default ModuleFactory");
  });
});
