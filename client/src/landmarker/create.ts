/** FaceLandmarker construction + result flattening, shared by worker and main thread. */
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from "@mediapipe/tasks-vision";

import type { Blendshapes, LandmarkerFrame } from "./types";

/** `useModule` picks the ESM wasm runtime — required inside a module worker. */
export async function createFaceLandmarker(
  wasmBase: string,
  modelUrl: string,
  useModule: boolean,
): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(wasmBase, useModule);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: modelUrl, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });
}

/** Flatten a MediaPipe result into the compact frame the detector consumes. */
export function flattenResult(result: FaceLandmarkerResult, t_ms: number): LandmarkerFrame {
  const blendshapes: Blendshapes = {};
  for (const category of result.faceBlendshapes[0]?.categories ?? []) {
    blendshapes[category.categoryName] = category.score;
  }
  return {
    t_ms,
    faceDetected: result.faceLandmarks.length > 0,
    matrix: result.facialTransformationMatrixes[0]?.data ?? [],
    blendshapes,
  };
}
