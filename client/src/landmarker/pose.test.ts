import { describe, expect, it } from "vitest";

import { PITCH_SIGN, YAW_SIGN, headPose } from "./pose";

const DEG = Math.PI / 180;

/** Column-major 4x4 from a 3x3 row-major rotation. */
function matrix4(r: number[][]): number[] {
  return [
    r[0][0], r[1][0], r[2][0], 0,
    r[0][1], r[1][1], r[2][1], 0,
    r[0][2], r[1][2], r[2][2], 0,
    0, 0, 0, 1,
  ];
}

function rotationX(deg: number): number[] {
  const c = Math.cos(deg * DEG);
  const s = Math.sin(deg * DEG);
  return matrix4([
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ]);
}

function rotationY(deg: number): number[] {
  const c = Math.cos(deg * DEG);
  const s = Math.sin(deg * DEG);
  return matrix4([
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ]);
}

describe("headPose", () => {
  it("reads pure X rotation as pitch, leaving yaw near zero", () => {
    const pose = headPose(rotationX(20));
    expect(pose.pitch).toBeCloseTo(PITCH_SIGN * 20, 4);
    expect(pose.yaw).toBeCloseTo(0, 4);
  });

  it("reads pure Y rotation as yaw, leaving pitch near zero", () => {
    const pose = headPose(rotationY(20));
    expect(pose.yaw).toBeCloseTo(YAW_SIGN * 20, 4);
    expect(pose.pitch).toBeCloseTo(0, 4);
  });

  it("tracks the sign of the rotation on each axis", () => {
    expect(headPose(rotationX(-15)).pitch).toBeCloseTo(PITCH_SIGN * -15, 4);
    expect(headPose(rotationY(-15)).yaw).toBeCloseTo(YAW_SIGN * -15, 4);
  });

  it("returns zeros for an identity matrix", () => {
    const pose = headPose(matrix4([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]));
    expect(pose.pitch).toBeCloseTo(0, 6);
    expect(pose.yaw).toBeCloseTo(0, 6);
  });

  it("returns zeros for a missing or short matrix", () => {
    expect(headPose([])).toEqual({ pitch: 0, yaw: 0 });
    expect(headPose([1, 0, 0])).toEqual({ pitch: 0, yaw: 0 });
  });
});
