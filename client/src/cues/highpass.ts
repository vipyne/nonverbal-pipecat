/** One-pole high-pass: value minus its EMA. Pure, no DOM. */

/** Slow EMA so only motion faster than ~1 s survives. */
export const HIGHPASS_ALPHA = 0.05;

export class HighPass {
  private ema: number | null = null;

  constructor(private alpha = HIGHPASS_ALPHA) {}

  /** Returns the high-passed value for `v`. */
  push(v: number): number {
    this.ema = this.ema === null ? v : this.ema + this.alpha * (v - this.ema);
    return v - this.ema;
  }

  reset() {
    this.ema = null;
  }
}
