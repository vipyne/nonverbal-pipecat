/** Debug HUD: pitch/yaw sparklines, cue log, face + calibration state. */

import type { Baseline, CalibrationState } from "../cues/calibration";
import { THRESHOLD_DEG } from "../cues/detector";

/** Sparkline window. */
const WINDOW_MS = 5000;
/** Vertical half-range of each sparkline, in degrees. */
const RANGE_DEG = 15;
const MAX_LOG_ROWS = 40;

interface Point {
  t_ms: number;
  pitch: number;
  yaw: number;
}

export interface HudSample {
  t_ms: number;
  faceDetected: boolean;
  /** High-passed pitch/yaw, degrees. */
  pitch: number;
  yaw: number;
}

export interface HudElements {
  canvas: HTMLCanvasElement;
  face: HTMLElement;
  calibration: HTMLElement;
  log: HTMLElement;
}

export class Hud {
  private points: Point[] = [];
  private ctx: CanvasRenderingContext2D;

  constructor(private els: HudElements) {
    const ctx = els.canvas.getContext("2d");
    if (!ctx) throw new Error("HUD canvas 2d context unavailable");
    this.ctx = ctx;
  }

  /** Feed one high-passed frame; redraws the sparklines. */
  push(sample: HudSample) {
    this.points.push({ t_ms: sample.t_ms, pitch: sample.pitch, yaw: sample.yaw });
    const cutoff = sample.t_ms - WINDOW_MS;
    while (this.points.length && this.points[0].t_ms < cutoff) this.points.shift();

    this.els.face.textContent = sample.faceDetected ? "face: detected" : "face: none";
    this.els.face.dataset.state = sample.faceDetected ? "on" : "off";
    this.draw(sample.t_ms);
  }

  setCalibration(state: CalibrationState, progress: number, baseline: Baseline | null) {
    if (state === "calibrating") {
      this.els.calibration.textContent = `calibrating ${Math.round(progress * 100)}%`;
    } else if (baseline) {
      this.els.calibration.textContent =
        `ready — pitch ${baseline.pitch.mean.toFixed(1)}°±${baseline.pitch.std.toFixed(1)}, ` +
        `yaw ${baseline.yaw.mean.toFixed(1)}°±${baseline.yaw.std.toFixed(1)} ` +
        `(${baseline.samples} samples)`;
    } else {
      this.els.calibration.textContent = "ready";
    }
    this.els.calibration.dataset.state = state;
  }

  /** Append a cue to the scrolling log. */
  logCue(cue: string, t_ms: number, detail?: string) {
    const row = document.createElement("div");
    row.className = "cue-row";
    row.textContent = `${(t_ms / 1000).toFixed(1)}s  ${cue}${detail ? `  ${detail}` : ""}`;
    this.els.log.appendChild(row);
    while (this.els.log.childElementCount > MAX_LOG_ROWS) this.els.log.firstElementChild?.remove();
    this.els.log.scrollTop = this.els.log.scrollHeight;
  }

  private draw(now: number) {
    const { ctx } = this;
    const { width, height } = this.els.canvas;
    ctx.clearRect(0, 0, width, height);

    const laneHeight = height / 2;
    this.drawLane(0, laneHeight, "pitch (nod)", (p) => p.pitch, now, "#4ea1ff");
    this.drawLane(laneHeight, laneHeight, "yaw (shake)", (p) => p.yaw, now, "#ffb04e");
  }

  private drawLane(
    top: number,
    laneHeight: number,
    label: string,
    pick: (p: Point) => number,
    now: number,
    color: string,
  ) {
    const { ctx } = this;
    const width = this.els.canvas.width;
    const mid = top + laneHeight / 2;
    const scale = laneHeight / 2 / RANGE_DEG;

    const x = (t_ms: number) => width - ((now - t_ms) / WINDOW_MS) * width;
    const y = (deg: number) => mid - Math.max(-RANGE_DEG, Math.min(RANGE_DEG, deg)) * scale;

    // Zero line.
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();

    // Threshold lines.
    ctx.strokeStyle = "#7a3b3b";
    ctx.setLineDash([4, 4]);
    for (const deg of [THRESHOLD_DEG, -THRESHOLD_DEG]) {
      ctx.beginPath();
      ctx.moveTo(0, y(deg));
      ctx.lineTo(width, y(deg));
      ctx.stroke();
    }
    ctx.setLineDash([]);

    ctx.fillStyle = "#888";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(label, 4, top + 13);

    if (this.points.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    this.points.forEach((p, i) => {
      const px = x(p.t_ms);
      const py = y(pick(p));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
}
