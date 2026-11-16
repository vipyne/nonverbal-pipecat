import { describe, expect, it } from "vitest";

import { CueDetector, type CueEvent, type DetectorInput } from "./detector";
import type { Blendshapes } from "../landmarker/types";

const FPS = 30;
const DT = 1000 / FPS;
/** Long enough for the 10 s calibration window to freeze. */
const WARMUP_MS = 11_000;

const NEUTRAL: Blendshapes = {
  jawOpen: 0.05,
  mouthSmileLeft: 0.05,
  mouthSmileRight: 0.05,
  browDownLeft: 0.05,
  browDownRight: 0.05,
  eyeSquintLeft: 0.05,
  eyeSquintRight: 0.05,
  eyeLookOutLeft: 0.05,
  eyeLookOutRight: 0.05,
};

interface Signal {
  pitch?: (t_s: number) => number;
  yaw?: (t_s: number) => number;
  blendshapes?: (t_s: number) => Blendshapes;
  faceDetected?: boolean;
  userSpeaking?: boolean;
}

/** Frames at 30 Hz from `t0` for `durationMs`, sampling `signal`. */
function frames(t0: number, durationMs: number, signal: Signal = {}): DetectorInput[] {
  const out: DetectorInput[] = [];
  for (let t = t0; t < t0 + durationMs; t += DT) {
    const local = (t - t0) / 1000;
    out.push({
      t_ms: t,
      faceDetected: signal.faceDetected ?? true,
      pitch: signal.pitch?.(local) ?? 0,
      yaw: signal.yaw?.(local) ?? 0,
      blendshapes: signal.blendshapes?.(local) ?? NEUTRAL,
      userSpeaking: signal.userSpeaking ?? false,
    });
  }
  return out;
}

function run(input: DetectorInput[]): CueEvent[] {
  const detector = new CueDetector();
  const out: CueEvent[] = [];
  for (const frame of input) out.push(...detector.push(frame));
  return out;
}

/** Deterministic noise in [-1, 1). */
function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return (s / 2147483648) * 2 - 1;
  };
}

const sine = (hz: number, amplitudeDeg: number) => (t_s: number) =>
  amplitudeDeg * Math.sin(2 * Math.PI * hz * t_s);

describe("CueDetector", () => {
  it("emits nod for a 2 Hz, 8° pitch oscillation", () => {
    const cues = run([
      ...frames(0, WARMUP_MS),
      ...frames(WARMUP_MS, 2000, { pitch: sine(2, 8) }),
    ]);

    const nods = cues.filter((c) => c.cue === "nod");
    expect(nods.length).toBeGreaterThanOrEqual(1);
    expect(cues.filter((c) => c.cue === "shake")).toHaveLength(0);
    expect(nods[0]).toMatchObject({ type: "nonverbal", cue: "nod" });
    expect(nods[0].t_ms).toBeGreaterThan(WARMUP_MS);
    expect(nods[0].confidence).toBeGreaterThan(0);
    expect(nods[0].confidence).toBeLessThanOrEqual(1);
  });

  it("ignores a slow 0.3 Hz pitch drift", () => {
    const cues = run([
      ...frames(0, WARMUP_MS),
      ...frames(WARMUP_MS, 6700, { pitch: sine(0.3, 8) }),
    ]);

    expect(cues.filter((c) => c.cue === "nod")).toHaveLength(0);
  });

  it("emits shake for a 2 Hz, 8° yaw oscillation", () => {
    const cues = run([
      ...frames(0, WARMUP_MS),
      ...frames(WARMUP_MS, 2000, { yaw: sine(2, 8) }),
    ]);

    expect(cues.filter((c) => c.cue === "shake").length).toBeGreaterThanOrEqual(1);
    expect(cues.filter((c) => c.cue === "nod")).toHaveLength(0);
  });

  it("emits distracted start and end for yaw held 30° off baseline", () => {
    const cues = run([
      ...frames(0, WARMUP_MS),
      ...frames(WARMUP_MS, 3000, { yaw: () => 30 }),
      ...frames(WARMUP_MS + 3000, 1500),
    ]);

    const distracted = cues.filter((c) => c.cue === "distracted");
    expect(distracted.map((c) => c.meta?.phase)).toEqual(["start", "end"]);
    expect(distracted[0].t_ms).toBeGreaterThanOrEqual(WARMUP_MS + 2000);
    expect(distracted[1].t_ms).toBeGreaterThanOrEqual(WARMUP_MS + 3000);
    expect(distracted[1].meta?.duration_ms).toBeGreaterThanOrEqual(2000);
    expect(cues.filter((c) => c.cue === "shake")).toHaveLength(0);
  });

  it("emits distracted when the face is lost for more than 2 s", () => {
    const cues = run([
      ...frames(0, WARMUP_MS),
      ...frames(WARMUP_MS, 3000, { faceDetected: false }),
      ...frames(WARMUP_MS + 3000, 1000),
    ]);

    expect(cues.filter((c) => c.cue === "distracted").map((c) => c.meta?.phase)).toEqual([
      "start",
      "end",
    ]);
  });

  it("emits no cues for noise within the baseline spread", () => {
    const rand = lcg(7);
    const noisy: Signal = {
      pitch: () => rand() * 0.6,
      yaw: () => rand() * 0.6,
      blendshapes: () => {
        const jitter = () => 0.05 + rand() * 0.01;
        return {
          jawOpen: jitter(),
          mouthSmileLeft: jitter(),
          mouthSmileRight: jitter(),
          browDownLeft: jitter(),
          browDownRight: jitter(),
          eyeSquintLeft: jitter(),
          eyeSquintRight: jitter(),
          eyeLookOutLeft: jitter(),
          eyeLookOutRight: jitter(),
        };
      },
    };

    const cues = run([...frames(0, WARMUP_MS, noisy), ...frames(WARMUP_MS, 8000, noisy)]);

    expect(cues).toEqual([]);
  });

  it("debounces two nod bursts 0.5 s apart into one nod", () => {
    const burst = sine(2, 8);
    const cues = run([
      ...frames(0, WARMUP_MS),
      ...frames(WARMUP_MS, 500, { pitch: burst }),
      ...frames(WARMUP_MS + 500, 500),
      ...frames(WARMUP_MS + 1000, 500, { pitch: burst }),
      ...frames(WARMUP_MS + 1500, 500),
    ]);

    expect(cues.filter((c) => c.cue === "nod")).toHaveLength(1);
  });

  it("emits amused when smile blendshapes stay high", () => {
    const smiling = {
      blendshapes: () => ({ ...NEUTRAL, mouthSmileLeft: 0.6, mouthSmileRight: 0.55 }),
    };
    const cues = run([...frames(0, WARMUP_MS), ...frames(WARMUP_MS, 1500, smiling)]);

    expect(cues.filter((c) => c.cue === "amused").length).toBe(1);
  });

  it("emits confused when brow-down and squint stay high together", () => {
    const confused = {
      blendshapes: () => ({
        ...NEUTRAL,
        browDownLeft: 0.6,
        browDownRight: 0.55,
        eyeSquintLeft: 0.5,
        eyeSquintRight: 0.45,
      }),
    };
    const cues = run([...frames(0, WARMUP_MS), ...frames(WARMUP_MS, 1500, confused)]);

    expect(cues.filter((c) => c.cue === "confused").length).toBe(1);
  });

  it("emits yawn only while the user is not speaking", () => {
    const jawOpen = { blendshapes: () => ({ ...NEUTRAL, jawOpen: 0.75 }) };

    const speaking = run([
      ...frames(0, WARMUP_MS),
      ...frames(WARMUP_MS, 2000, { ...jawOpen, userSpeaking: true }),
    ]);
    expect(speaking.filter((c) => c.cue === "yawn")).toHaveLength(0);

    const silent = run([...frames(0, WARMUP_MS), ...frames(WARMUP_MS, 2000, jawOpen)]);
    expect(silent.filter((c) => c.cue === "yawn").length).toBe(1);
  });

  it("emits nod for a subtle 2 Hz, 2° pitch oscillation after a still calibration", () => {
    const cues = run([
      ...frames(0, WARMUP_MS),
      ...frames(WARMUP_MS, 2000, { pitch: sine(2, 2) }),
    ]);

    expect(cues.filter((c) => c.cue === "nod").length).toBeGreaterThanOrEqual(1);
    expect(cues.filter((c) => c.cue === "shake")).toHaveLength(0);
  });

  it("emits shake for a subtle 2 Hz, 2° yaw oscillation after a still calibration", () => {
    const cues = run([
      ...frames(0, WARMUP_MS),
      ...frames(WARMUP_MS, 2000, { yaw: sine(2, 2) }),
    ]);

    expect(cues.filter((c) => c.cue === "shake").length).toBeGreaterThanOrEqual(1);
    expect(cues.filter((c) => c.cue === "nod")).toHaveLength(0);
  });

  it("raises the turn threshold relative to the signal's own jitter", () => {
    const rand = lcg(3);
    const jitter: Signal = { pitch: () => rand() * 1.5 };

    const cues = run([...frames(0, WARMUP_MS, jitter), ...frames(WARMUP_MS, 8000, jitter)]);

    expect(cues.filter((c) => c.cue === "nod")).toHaveLength(0);
  });

  it("emits nothing while still calibrating", () => {
    const cues = run(frames(0, 5000, { pitch: sine(2, 8) }));
    expect(cues).toEqual([]);
  });
});
