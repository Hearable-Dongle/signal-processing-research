import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SceneLauncher } from "./SceneLauncher";

test("latency controls invoke callback", async () => {
  const user = userEvent.setup();
  const onLatency = vi.fn();
  const onKill = vi.fn();
  const onTogglePlayback = vi.fn();
  const onProcessingModeChange = vi.fn();

  render(
    <SceneLauncher
      status="idle"
      defaultScenePath="x.json"
      onStart={() => undefined}
      onStop={() => undefined}
      onKillRun={onKill}
      canKillRun={true}
      onDownloadWav={() => undefined}
      canDownloadWav={false}
      onTogglePlayback={onTogglePlayback}
      canPlayOutput={true}
      isPlaybackActive={false}
      latencyMs={180}
      onLatencyMsChange={onLatency}
      processingMode="specific_speaker_enhancement"
      onProcessingModeChange={onProcessingModeChange}
    />
  );

  const slider = screen.getByLabelText("Playback latency (ms)");
  await user.clear(screen.getByLabelText("Playback latency number"));
  await user.type(screen.getByLabelText("Playback latency number"), "240");
  await user.click(slider);

  expect(onLatency).toHaveBeenCalled();

  await user.click(screen.getByRole("button", { name: "Kill Current Run" }));
  expect(onKill).toHaveBeenCalledTimes(1);

  await user.click(screen.getByRole("button", { name: "Play Beamformed Output" }));
  expect(onTogglePlayback).toHaveBeenCalledTimes(1);

  await user.selectOptions(screen.getByLabelText("Beamforming mode"), "beamform_from_ground_truth");
  expect(onProcessingModeChange).toHaveBeenCalledWith("beamform_from_ground_truth");
});
