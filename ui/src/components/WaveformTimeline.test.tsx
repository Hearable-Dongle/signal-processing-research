import { render, screen } from "@testing-library/react";

import { WaveformTimeline } from "./WaveformTimeline";

test("renders waveform timeline with playhead and time labels", () => {
  render(
    <WaveformTimeline
      beamformedBins={[0.1, 0.7, 0.3]}
      beamformedDurationMs={5000}
      rawMixedBins={[0.2, 0.3]}
      rawMixedDurationMs={4000}
      playheadMs={1500}
      canPlayBeamformed={true}
      canPauseBeamformed={false}
      isBeamformedPlaying={false}
      onPlayBeamformed={() => undefined}
      onPauseBeamformed={() => undefined}
      canPlayRawMixed={true}
      isRawMixedPlaying={false}
      onToggleRawMixedPlayback={() => undefined}
    />
  );

  expect(screen.getByLabelText("Waveform timeline")).toBeInTheDocument();
  expect(screen.getAllByText("0:00")).toHaveLength(2);
  expect(screen.getByText("0:01 / 0:05")).toBeInTheDocument();
  expect(screen.getAllByText("Raw mixed input").length).toBeGreaterThan(0);
  expect(screen.getByRole("button", { name: "Play Beamformed output" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Play Raw mixed input" })).toBeInTheDocument();
});
