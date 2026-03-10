import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SceneLauncher } from "./SceneLauncher";

test("mode picker gates launcher settings and latency controls invoke callback", async () => {
  const user = userEvent.setup();
  const onLatency = vi.fn();
  const onKill = vi.fn();
  const onProcessingModeChange = vi.fn();
  const onStart = vi.fn();

  render(
    <SceneLauncher
      status="idle"
      defaultScenePath="x.json"
      defaultBackgroundNoisePath="noise.wav"
      defaultBackgroundNoiseGain={0.15}
      onStart={onStart}
      onStop={() => undefined}
      onKillRun={onKill}
      canKillRun={true}
      onStopActiveSession={() => undefined}
      onDownloadWav={() => undefined}
      canDownloadWav={false}
      latencyMs={180}
      onLatencyMsChange={onLatency}
      processingMode="specific_speaker_enhancement"
      onProcessingModeChange={onProcessingModeChange}
      monitorSource="processed"
      onMonitorSourceChange={() => undefined}
    />
  );

  expect(screen.queryByLabelText("Scene config path")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Audio device query")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Kill Current Run" })).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Simulation Scene file plus optional background noise." }));

  const slider = screen.getByLabelText("Playback latency (ms)");
  expect(screen.getByLabelText("Speaker stream mode")).toHaveValue("single_dominant_no_separator");
  await user.clear(screen.getByLabelText("Playback latency number"));
  await user.type(screen.getByLabelText("Playback latency number"), "240");
  await user.click(slider);

  expect(onLatency).toHaveBeenCalled();

  await user.click(screen.getByRole("button", { name: "Kill Current Run" }));
  expect(onKill).toHaveBeenCalledTimes(1);

  await user.selectOptions(screen.getByLabelText("Processing mode"), "beamform_from_ground_truth");
  expect(onProcessingModeChange).toHaveBeenCalledWith("beamform_from_ground_truth");
});

test("localize and beamform locks speaker stream mode to no-separator", async () => {
  const user = userEvent.setup();
  const onStart = vi.fn();

  render(
    <SceneLauncher
      status="idle"
      defaultScenePath="x.json"
      defaultBackgroundNoisePath="noise.wav"
      defaultBackgroundNoiseGain={0.15}
      onStart={onStart}
      onStop={() => undefined}
      onKillRun={() => undefined}
      canKillRun={true}
      onStopActiveSession={() => undefined}
      onDownloadWav={() => undefined}
      canDownloadWav={false}
      latencyMs={180}
      onLatencyMsChange={() => undefined}
      processingMode="localize_and_beamform"
      onProcessingModeChange={() => undefined}
      monitorSource="processed"
      onMonitorSourceChange={() => undefined}
    />
  );

  await user.click(screen.getByRole("button", { name: "Simulation Scene file plus optional background noise." }));
  expect(screen.getByLabelText("Speaker stream mode")).toHaveValue("single_dominant_no_separator");
  expect(screen.getByLabelText("Speaker stream mode")).toBeDisabled();
  expect(screen.getByText(/uses the single-dominant no-separator path/i)).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Start" }));
  expect(onStart).toHaveBeenCalledWith(
    expect.objectContaining({
      separationMode: "single_dominant_no_separator",
    })
  );
});

test("live mode reveals only ReSpeaker-specific settings", async () => {
  const user = userEvent.setup();

  render(
    <SceneLauncher
      status="idle"
      defaultScenePath="x.json"
      defaultBackgroundNoisePath="noise.wav"
      defaultBackgroundNoiseGain={0.15}
      onStart={() => undefined}
      onStop={() => undefined}
      onKillRun={() => undefined}
      canKillRun={true}
      onStopActiveSession={() => undefined}
      onDownloadWav={() => undefined}
      canDownloadWav={false}
      latencyMs={180}
      onLatencyMsChange={() => undefined}
      processingMode="specific_speaker_enhancement"
      onProcessingModeChange={() => undefined}
      monitorSource="processed"
      onMonitorSourceChange={() => undefined}
    />
  );

  await user.click(screen.getByRole("button", { name: "ReSpeaker Live Direct capture from the local USB microphone array." }));

  expect(screen.getByLabelText("Audio device query")).toBeInTheDocument();
  expect(screen.getByLabelText("Channel map (optional)")).toBeInTheDocument();
  expect(screen.getByLabelText("Sample rate (Hz)")).toBeInTheDocument();
  expect(screen.queryByLabelText("Scene config path")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Background noise audio path")).not.toBeInTheDocument();
});
