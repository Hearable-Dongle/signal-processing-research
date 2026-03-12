import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SceneLauncher } from "./SceneLauncher";

test("mode picker gates launcher settings and latency controls invoke callback", async () => {
  const user = userEvent.setup();
  const onLatency = vi.fn();
  const onKill = vi.fn();
  const onStart = vi.fn();

  render(
    <SceneLauncher
      status="idle"
      defaultScenePath="x.json"
      defaultBackgroundNoisePath="noise.wav"
      defaultBackgroundNoiseGain={0.15}
      defaultAlgorithmMode="localization_only"
      onStart={onStart}
      onStop={() => undefined}
      onKillRun={onKill}
      canKillRun={true}
      onStopActiveSession={() => undefined}
      onDownloadWav={() => undefined}
      canDownloadWav={false}
      latencyMs={180}
      onLatencyMsChange={onLatency}
      monitorSource="processed"
      onMonitorSourceChange={() => undefined}
    />
  );

  expect(screen.queryByLabelText("Scene config path")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Audio device query")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Kill Current Run" })).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Simulation Scene file plus optional background noise." }));

  const slider = screen.getByLabelText("Playback latency (ms)");
  expect(screen.getByLabelText("Algorithm mode")).toHaveValue("localization_only");
  expect(screen.getByRole("switch", { name: "Single active speaker" })).toHaveAttribute("aria-checked", "false");
  expect(screen.getByLabelText("Localization hop (ms)")).toHaveValue(95);
  expect(screen.getByLabelText("Localization window (ms)")).toHaveValue(300);
  expect(screen.getByLabelText("Localization overlap")).toHaveValue(0.2);
  expect(screen.getByLabelText("Localization freq low (Hz)")).toHaveValue(1200);
  expect(screen.getByLabelText("Localization freq high (Hz)")).toHaveValue(5400);
  expect(screen.getByLabelText("Speaker centroid history (M)")).toHaveValue(8);
  expect(screen.getByLabelText("Speaker activation observations (N)")).toHaveValue(3);
  expect(screen.getByLabelText("Speaker match window (deg)")).toHaveValue(30);
  await user.clear(screen.getByLabelText("Playback latency number"));
  await user.type(screen.getByLabelText("Playback latency number"), "240");
  await user.click(slider);

  expect(onLatency).toHaveBeenCalled();

  await user.click(screen.getByRole("button", { name: "Kill Current Run" }));
  expect(onKill).toHaveBeenCalledTimes(1);

  await user.click(screen.getByRole("switch", { name: "Single active speaker" }));
  expect(screen.getByLabelText("Algorithm mode")).toHaveValue("speaker_tracking");
  fireEvent.change(screen.getByLabelText("Localization hop (ms)"), { target: { value: "50" } });
  fireEvent.change(screen.getByLabelText("Localization window (ms)"), { target: { value: "200" } });
  expect(screen.getByLabelText("Algorithm mode")).toHaveValue("speaker_tracking");
  expect(screen.getByRole("switch", { name: "Single active speaker" })).toHaveAttribute("aria-checked", "true");
  expect(screen.getByLabelText("Localization hop (ms)")).toHaveValue(50);
  expect(screen.getByLabelText("Localization window (ms)")).toHaveValue(200);
});

test("simulation shows ground-truth toggles and disables oracle speaker sources when irrelevant", async () => {
  const user = userEvent.setup();
  const onStart = vi.fn();

  render(
    <SceneLauncher
      status="idle"
      defaultScenePath="x.json"
      defaultBackgroundNoisePath="noise.wav"
      defaultBackgroundNoiseGain={0.15}
      defaultAlgorithmMode="localization_only"
      onStart={onStart}
      onStop={() => undefined}
      onKillRun={() => undefined}
      canKillRun={true}
      onStopActiveSession={() => undefined}
      onDownloadWav={() => undefined}
      canDownloadWav={false}
      latencyMs={180}
      onLatencyMsChange={() => undefined}
      monitorSource="processed"
      onMonitorSourceChange={() => undefined}
    />
  );

  await user.click(screen.getByRole("button", { name: "Simulation Scene file plus optional background noise." }));
  expect(screen.getByLabelText("Use ground truth location")).toBeInTheDocument();
  expect(screen.getByLabelText("Use ground truth speaker sources")).toBeInTheDocument();
  expect(screen.getByLabelText("Use ground truth speaker sources")).toBeDisabled();
  expect(screen.getByText(/does not use separate speaker-source streams/i)).toBeInTheDocument();
  expect(screen.getByText(/turning this on switches the algorithm to speaker tracking/i)).toBeInTheDocument();

  await user.selectOptions(screen.getByLabelText("Algorithm mode"), "speaker_tracking");
  fireEvent.change(screen.getByLabelText("Localization hop (ms)"), { target: { value: "50" } });
  fireEvent.change(screen.getByLabelText("Localization window (ms)"), { target: { value: "200" } });
  expect(screen.getByLabelText("Use ground truth speaker sources")).not.toBeDisabled();
  await user.click(screen.getByLabelText("Use ground truth location"));
  await user.click(screen.getByLabelText("Use ground truth speaker sources"));

  await user.click(screen.getByRole("button", { name: "Start" }));
  expect(onStart).toHaveBeenCalledWith(
    expect.objectContaining({
      algorithmMode: "speaker_tracking",
      localizationHopMs: 50,
      localizationWindowMs: 200,
      localizationOverlap: 0.2,
      localizationFreqLowHz: 1200,
      localizationFreqHighHz: 5400,
      speakerHistorySize: 8,
      speakerActivationMinPredictions: 3,
      speakerMatchWindowDeg: 30,
      useGroundTruthLocation: true,
      useGroundTruthSpeakerSources: true,
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
      defaultAlgorithmMode="localization_only"
      onStart={() => undefined}
      onStop={() => undefined}
      onKillRun={() => undefined}
      canKillRun={true}
      onStopActiveSession={() => undefined}
      onDownloadWav={() => undefined}
      canDownloadWav={false}
      latencyMs={180}
      onLatencyMsChange={() => undefined}
      monitorSource="processed"
      onMonitorSourceChange={() => undefined}
    />
  );

  await user.click(screen.getByRole("button", { name: "ReSpeaker Live Direct capture from the local USB microphone array." }));

  expect(screen.getByLabelText("Audio device query")).toBeInTheDocument();
  expect(screen.getByLabelText("Channel map (optional)")).toBeInTheDocument();
  expect(screen.getByLabelText("Sample rate (Hz)")).toBeInTheDocument();
  expect(screen.getByLabelText("Algorithm mode")).toBeInTheDocument();
  expect(screen.getByLabelText("Algorithm mode")).toHaveValue("localization_only");
  expect(screen.getByRole("switch", { name: "Single active speaker" })).toHaveAttribute("aria-checked", "false");
  expect(screen.queryByLabelText("Scene config path")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Background noise audio path")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Use ground truth location")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Use ground truth speaker sources")).not.toBeInTheDocument();
});
