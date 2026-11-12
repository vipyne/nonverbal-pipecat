import { describe, expect, it } from "vitest";

import { vlmEnabled } from "./vlm";

describe("vlmEnabled", () => {
  it("is off when the flag is unset", () => {
    expect(vlmEnabled(undefined)).toBe(false);
  });

  it("is off for the values the server reads as off", () => {
    for (const value of ["", "0", "false", "False", " 0 "]) {
      expect(vlmEnabled(value)).toBe(false);
    }
  });

  it("is on for anything else", () => {
    for (const value of ["1", "true", "yes"]) {
      expect(vlmEnabled(value)).toBe(true);
    }
  });
});
