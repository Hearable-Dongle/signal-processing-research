import { render, screen } from "@testing-library/react";

import { WaveformTimeline } from "./WaveformTimeline";

test("renders waveform timeline with playhead and time labels", () => {
  render(<WaveformTimeline bins={[0.1, 0.7, 0.3]} totalDurationMs={5000} playheadMs={1500} />);

  expect(screen.getByLabelText("Waveform timeline")).toBeInTheDocument();
  expect(screen.getByText("0:00")).toBeInTheDocument();
  expect(screen.getByText("0:01 / 0:05")).toBeInTheDocument();
});
