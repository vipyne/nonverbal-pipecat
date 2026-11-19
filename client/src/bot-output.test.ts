import { describe, expect, it } from "vitest";

import { isNewBotOutput } from "./bot-output";

describe("isNewBotOutput", () => {
  it("accepts a segment's first message", () => {
    expect(isNewBotOutput({ text: "Hi.", spoken_status: "new" })).toBe(true);
  });

  it("accepts messages without a spoken status (older servers)", () => {
    expect(isNewBotOutput({ text: "Hi." })).toBe(true);
  });

  it("rejects word-progress repeats", () => {
    expect(isNewBotOutput({ text: "Hi.", spoken_status: "in-progress" })).toBe(false);
  });

  it("rejects the completed repeat", () => {
    expect(isNewBotOutput({ text: "Hi.", spoken_status: "completed" })).toBe(false);
  });
});
