import { speakerColor } from "./color";

test("speaker color mapping is stable for speaker id", () => {
  expect(speakerColor(17)).toBe(speakerColor(17));
  expect(speakerColor(17)).not.toBe(speakerColor(18));
});
