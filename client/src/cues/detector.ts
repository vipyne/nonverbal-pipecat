/** Nonverbal cue detector. Pure TS, no DOM — drives off landmarker frames. */

import type { Blendshapes } from "../landmarker/types";
import { Calibrator, type Baseline, type CalibrationState } from "./calibration";
import { HighPass } from "./highpass";

export type CueType = "nod" | "shake" | "distracted" | "confused" | "amused" | "yawn";

/** Cues detected from an oscillating pose axis. */
type OscillationCue = "nod" | "shake";
/** Cues detected from a condition held over time. */
type SustainCue = "distracted" | "confused" | "amused" | "yawn";

/** RTVI client-message type the server's NonverbalContextGate listens for. */
export const CUE_MESSAGE_TYPE = "nonverbal";

/** Wire format sent to the server. */
export interface CueEvent {
  type: "nonverbal";
  cue: CueType;
  t_ms: number;
  confidence: number;
  meta?: Record<string, unknown>;
}

export interface DetectorInput {
  t_ms: number;
  faceDetected: boolean;
  /** Raw head pose in degrees, from `headPose()`. */
  pitch: number;
  yaw: number;
  blendshapes: Blendshapes;
  /** Mic VAD; suppresses `yawn` while the user talks. */
  userSpeaking?: boolean;
}

/** Oscillation amplitude on the high-passed signal that counts as a turn. */
export const THRESHOLD_DEG = 3;
/** Span the ≥2 direction reversals of a nod/shake must fall inside. */
const OSCILLATION_WINDOW_MS = 1200;
/** Per-cue debounce; renewed motion inside it extends the same gesture. */
const REFRACTORY_MS = 1000;
/** Degrees of yaw off baseline that reads as looking away. */
const DISTRACTED_YAW_DEG = 25;
const DISTRACTED_MS = 2000;
const CONFUSED_MS = 700;
const AMUSED_MS = 500;
const YAWN_MS = 1000;
/** Std floor so a near-constant baseline cannot make a cue hair-trigger. */
const MIN_STD = 0.02;
/** Blendshape scores must clear baseline+2σ AND these absolute floors. */
const FLOORS = { smile: 0.25, browDown: 0.25, squint: 0.2, lookOut: 0.4, jawOpen: 0.4 };

interface AxisState {
  hp: HighPass;
  prev: number | null;
  sign: 0 | 1 | -1;
  /** Reversal amplitudes inside the oscillation window. */
  turns: { t_ms: number; amp: number }[];
}

interface SustainState {
  since: number | null;
  active: boolean;
}

/** Cue detection over calibrated pose + blendshape streams. */
export class CueDetector {
  private calibrator: Calibrator;
  private lastHighpass = { pitch: 0, yaw: 0 };
  private axes: Record<OscillationCue, AxisState> = {
    nod: { hp: new HighPass(), prev: null, sign: 0, turns: [] },
    shake: { hp: new HighPass(), prev: null, sign: 0, turns: [] },
  };
  private sustained: Record<SustainCue, SustainState> = {
    distracted: { since: null, active: false },
    confused: { since: null, active: false },
    amused: { since: null, active: false },
    yawn: { since: null, active: false },
  };
  private lastCue: Partial<Record<CueType, number>> = {};

  constructor(options: { calibrationMs?: number } = {}) {
    this.calibrator = new Calibrator(options.calibrationMs);
  }

  get calibrationState(): CalibrationState {
    return this.calibrator.state;
  }

  get baseline(): Baseline | null {
    return this.calibrator.baseline;
  }

  calibrationProgress(t_ms: number): number {
    return this.calibrator.progress(t_ms);
  }

  /** Feeds one landmarker frame; returns cues that fired on it. */
  push(input: DetectorInput): CueEvent[] {
    this.calibrator.push({
      t_ms: input.t_ms,
      faceDetected: input.faceDetected,
      pitch: input.pitch,
      yaw: input.yaw,
      blendshapes: input.blendshapes,
    });

    const pitchHp = this.axes.nod.hp.push(input.pitch);
    const yawHp = this.axes.shake.hp.push(input.yaw);
    this.lastHighpass = { pitch: pitchHp, yaw: yawHp };

    const baseline = this.calibrator.baseline;
    if (!baseline) return [];

    const out: CueEvent[] = [];
    if (input.faceDetected) {
      this.oscillation("nod", pitchHp, input.t_ms, out);
      this.oscillation("shake", yawHp, input.t_ms, out);
    }
    this.sustainedCues(input, baseline, out);
    return out;
  }

  /** High-passed pitch/yaw of the most recent frame, degrees. */
  get highpass(): { pitch: number; yaw: number } {
    return this.lastHighpass;
  }

  reset() {
    this.calibrator.reset();
    for (const axis of Object.values(this.axes)) {
      axis.hp.reset();
      axis.prev = null;
      axis.sign = 0;
      axis.turns = [];
    }
    for (const state of Object.values(this.sustained)) {
      state.since = null;
      state.active = false;
    }
    this.lastCue = {};
    this.lastHighpass = { pitch: 0, yaw: 0 };
  }

  /** Counts direction reversals of a high-passed axis; emits nod/shake. */
  private oscillation(cue: OscillationCue, hp: number, t_ms: number, out: CueEvent[]) {
    const axis = this.axes[cue];
    const prev = axis.prev;
    if (prev === null) {
      axis.prev = hp;
      return;
    }
    const delta = hp - prev;
    // Flat frame: keep the running extremum so a plateau is not a reversal.
    if (delta === 0) return;

    const sign = delta > 0 ? 1 : -1;
    const reversed = axis.sign !== 0 && sign !== axis.sign;
    axis.sign = sign;
    axis.prev = hp;
    if (!reversed || Math.abs(prev) <= THRESHOLD_DEG) return;

    const last = this.lastCue[cue];
    if (last !== undefined && t_ms - last < REFRACTORY_MS) {
      // Same gesture continuing — extend the debounce, emit nothing.
      this.lastCue[cue] = t_ms;
      axis.turns = [];
      return;
    }

    axis.turns.push({ t_ms, amp: Math.abs(prev) });
    while (axis.turns.length && t_ms - axis.turns[0].t_ms > OSCILLATION_WINDOW_MS) {
      axis.turns.shift();
    }
    if (axis.turns.length < 2) return;

    const amp = axis.turns.reduce((sum, t) => sum + t.amp, 0) / axis.turns.length;
    out.push({
      type: "nonverbal",
      cue,
      t_ms,
      confidence: clamp01(amp / (3 * THRESHOLD_DEG)),
      meta: { amplitude_deg: round1(amp), turns: axis.turns.length },
    });
    this.lastCue[cue] = t_ms;
    axis.turns = [];
  }

  private sustainedCues(input: DetectorInput, baseline: Baseline, out: CueEvent[]) {
    const { blendshapes: shapes, t_ms } = input;
    const high = (name: string, floor: number) => {
      const score = shapes[name] ?? 0;
      const stat = baseline.blendshapes[name];
      const relative = stat ? stat.mean + 2 * Math.max(stat.std, MIN_STD) : floor;
      return score > relative && score > floor;
    };
    const eitherHigh = (left: string, right: string, floor: number) =>
      high(left, floor) || high(right, floor);

    const yawOff = Math.abs(input.yaw - baseline.yaw.mean);
    const distracted =
      !input.faceDetected ||
      yawOff > DISTRACTED_YAW_DEG ||
      eitherHigh("eyeLookOutLeft", "eyeLookOutRight", FLOORS.lookOut);

    this.sustain("distracted", distracted, t_ms, DISTRACTED_MS, out, {
      emitEnd: true,
      meta: () => ({ yaw_off_deg: round1(yawOff), face: input.faceDetected }),
    });

    const confused =
      input.faceDetected &&
      eitherHigh("browDownLeft", "browDownRight", FLOORS.browDown) &&
      eitherHigh("eyeSquintLeft", "eyeSquintRight", FLOORS.squint);
    this.sustain("confused", confused, t_ms, CONFUSED_MS, out, {});

    const amused =
      input.faceDetected && eitherHigh("mouthSmileLeft", "mouthSmileRight", FLOORS.smile);
    this.sustain("amused", amused, t_ms, AMUSED_MS, out, {});

    const yawn = input.faceDetected && !input.userSpeaking && high("jawOpen", FLOORS.jawOpen);
    this.sustain("yawn", yawn, t_ms, YAWN_MS, out, {});
  }

  /** Emits `cue` once a condition holds for `holdMs`; optional end event. */
  private sustain(
    cue: SustainCue,
    condition: boolean,
    t_ms: number,
    holdMs: number,
    out: CueEvent[],
    opts: { emitEnd?: boolean; meta?: () => Record<string, unknown> },
  ) {
    const state = this.sustained[cue];
    if (!condition) {
      if (state.active && opts.emitEnd) {
        out.push({
          type: "nonverbal",
          cue,
          t_ms,
          confidence: 0.6,
          meta: { phase: "end", duration_ms: Math.round(t_ms - (state.since ?? t_ms)) },
        });
      }
      state.since = null;
      state.active = false;
      return;
    }

    if (state.since === null) state.since = t_ms;
    if (state.active) return;

    const held = t_ms - state.since;
    if (held < holdMs) return;

    const last = this.lastCue[cue];
    if (last !== undefined && t_ms - last < REFRACTORY_MS) return;

    state.active = true;
    this.lastCue[cue] = t_ms;
    out.push({
      type: "nonverbal",
      cue,
      t_ms,
      confidence: clamp01(held / (2 * holdMs) + 0.3),
      meta: { ...(opts.emitEnd ? { phase: "start" } : {}), ...(opts.meta?.() ?? {}) },
    });
  }
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function round1(v: number): number {
  return Math.round(v * 10) / 10;
}
