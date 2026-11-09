/** Head pose from MediaPipe's facial transformation matrix. Pure, no DOM. */

/**
 * Sign flips for the exported angles, verified on camera (HUMAN.md §4).
 * `+1` means nodding down raises pitch and turning left raises yaw.
 */
export const PITCH_SIGN = 1;
export const YAW_SIGN = 1;

export interface HeadPose {
  /** Degrees; nod axis. */
  pitch: number;
  /** Degrees; shake axis. */
  yaw: number;
}

const RAD_TO_DEG = 180 / Math.PI;

/** `m` is the column-major 4x4 `facialTransformationMatrixes[0].data`. */
export function headPose(m: number[]): HeadPose {
  if (m.length < 16) return { pitch: 0, yaw: 0 };
  const sinYaw = Math.min(1, Math.max(-1, -m[2]));
  return {
    pitch: PITCH_SIGN * Math.atan2(m[6], m[10]) * RAD_TO_DEG,
    yaw: YAW_SIGN * Math.asin(sinYaw) * RAD_TO_DEG,
  };
}
