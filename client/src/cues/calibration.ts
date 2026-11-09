/** Rolling baseline for pose and blendshapes. Pure, no DOM. */

import type { Blendshapes } from "../landmarker/types";

/** Baseline collection window at session start. */
export const CALIBRATION_MS = 10_000;
/** Below this, std is too noisy to threshold against. */
const MIN_SAMPLES = 30;

export interface Stat {
  mean: number;
  std: number;
}

export interface Baseline {
  pitch: Stat;
  yaw: Stat;
  blendshapes: Record<string, Stat>;
  samples: number;
}

export type CalibrationState = "calibrating" | "ready";

export interface CalibrationSample {
  t_ms: number;
  faceDetected: boolean;
  pitch: number;
  yaw: number;
  blendshapes: Blendshapes;
}

class Accumulator {
  private n = 0;
  private sum = 0;
  private sumSq = 0;

  push(v: number) {
    this.n += 1;
    this.sum += v;
    this.sumSq += v * v;
  }

  get count() {
    return this.n;
  }

  stat(): Stat {
    if (this.n === 0) return { mean: 0, std: 0 };
    const mean = this.sum / this.n;
    const variance = Math.max(0, this.sumSq / this.n - mean * mean);
    return { mean, std: Math.sqrt(variance) };
  }
}

/** Collects mean/std over the first `CALIBRATION_MS` of face-visible frames. */
export class Calibrator {
  private start: number | null = null;
  private pitch = new Accumulator();
  private yaw = new Accumulator();
  private shapes = new Map<string, Accumulator>();
  private frozen: Baseline | null = null;

  constructor(private windowMs = CALIBRATION_MS) {}

  push(sample: CalibrationSample) {
    if (this.frozen) return;
    if (this.start === null) this.start = sample.t_ms;

    if (sample.faceDetected) {
      this.pitch.push(sample.pitch);
      this.yaw.push(sample.yaw);
      for (const [name, score] of Object.entries(sample.blendshapes)) {
        let acc = this.shapes.get(name);
        if (!acc) {
          acc = new Accumulator();
          this.shapes.set(name, acc);
        }
        acc.push(score);
      }
    }

    const elapsed = sample.t_ms - this.start;
    if (elapsed >= this.windowMs && this.pitch.count >= MIN_SAMPLES) {
      this.frozen = this.snapshot();
    }
  }

  get state(): CalibrationState {
    return this.frozen ? "ready" : "calibrating";
  }

  /** 0..1 over the collection window. */
  progress(t_ms: number): number {
    if (this.frozen) return 1;
    if (this.start === null) return 0;
    return Math.min(1, (t_ms - this.start) / this.windowMs);
  }

  /** Frozen baseline, or the in-progress estimate while calibrating. */
  get baseline(): Baseline | null {
    return this.frozen;
  }

  snapshot(): Baseline {
    const blendshapes: Record<string, Stat> = {};
    for (const [name, acc] of this.shapes) blendshapes[name] = acc.stat();
    return {
      pitch: this.pitch.stat(),
      yaw: this.yaw.stat(),
      blendshapes,
      samples: this.pitch.count,
    };
  }

  reset() {
    this.start = null;
    this.pitch = new Accumulator();
    this.yaw = new Accumulator();
    this.shapes = new Map();
    this.frozen = null;
  }
}
