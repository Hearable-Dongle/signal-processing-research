import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SceneLauncher } from "./SceneLauncher";

test("mode picker reveals explicit fast and slow path controls", async () => {
  const user = userEvent.setup();
  const onLatency = vi.fn();
  const onKill = vi.fn();

  render(
    <SceneLauncher
      status="idle"
      hasActiveSession={false}
      defaultScenePath="x.json"
      defaultBackgroundNoisePath="noise.wav"
      defaultBackgroundNoiseGain={0.15}
      onStart={() => undefined}
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

  await user.click(screen.getByRole("button", { name: "Simulation Scene file plus optional background noise." }));

  expect(screen.getByRole("switch", { name: "Live playback" })).toHaveAttribute("aria-checked", "true");
  expect(screen.getByRole("button", { name: "Localization" })).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByRole("button", { name: "Beamforming" })).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByRole("button", { name: "Post-proc" })).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByLabelText("Localization backend")).toHaveValue("capon_1src");
  expect(screen.getByLabelText("Beamforming mode")).toHaveValue("mvdr_fd");
  expect(screen.queryByLabelText("Enhancement tier")).not.toBeInTheDocument();
  expect(screen.getByLabelText("Output enhancer mode")).toHaveValue("off");
  expect(screen.getByRole("switch", { name: "Postfilter enabled" })).toHaveAttribute("aria-checked", "true");
  expect(screen.getByRole("switch", { name: "Single-speaker assumption" })).toHaveAttribute("aria-checked", "true");
  expect(screen.getByRole("switch", { name: "Localization VAD" })).toHaveAttribute("aria-checked", "true");
  expect(screen.getByRole("button", { name: "Slow path" })).toHaveAttribute("aria-expanded", "false");
  expect(screen.getByRole("button", { name: "Advanced overrides" })).toHaveAttribute("aria-expanded", "false");
  expect(screen.getByLabelText("Localization hop (ms)")).toHaveValue(95);
  expect(screen.getByLabelText("Localization window (ms)")).toHaveValue(300);

  fireEvent.change(screen.getByLabelText("Playback latency target (ms)"), { target: { value: "240" } });
  expect(onLatency).toHaveBeenCalled();

  await user.click(screen.getByRole("button", { name: "Kill Current Run" }));
  expect(onKill).toHaveBeenCalledTimes(1);
});

test("simulation start emits nested fast_path and slow_path config", async () => {
  const user = userEvent.setup();
  const onStart = vi.fn();

  render(
    <SceneLauncher
      status="idle"
      hasActiveSession={false}
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
      monitorSource="processed"
      onMonitorSourceChange={() => undefined}
    />
  );

  await user.click(screen.getByRole("button", { name: "Simulation Scene file plus optional background noise." }));
  await user.selectOptions(screen.getByLabelText("Localization backend"), "capon_mvdr_refine_1src");
  await user.selectOptions(screen.getByLabelText("Beamforming mode"), "delay_sum");
  await user.selectOptions(screen.getByLabelText("Output enhancer mode"), "wiener");
  await user.click(screen.getByRole("switch", { name: "Postfilter enabled" }));
  await user.click(screen.getByRole("switch", { name: "Live playback" }));
  await user.click(screen.getByRole("button", { name: "Slow path" }));
  await user.click(screen.getByRole("switch", { name: "Enable slow path" }));
  await user.click(screen.getByRole("switch", { name: "Long memory" }));
  fireEvent.change(screen.getByLabelText("Localization hop (ms)"), { target: { value: "50" } });
  fireEvent.change(screen.getByLabelText("Localization window (ms)"), { target: { value: "200" } });
  await user.click(screen.getByRole("button", { name: "Advanced overrides" }));
  fireEvent.change(screen.getByLabelText("Session overrides JSON"), { target: { value: "{\"max_speakers_hint\":2}" } });
  await user.click(screen.getByLabelText("Use ground truth location"));
  await user.click(screen.getByLabelText("Use ground truth speaker sources"));

  await user.click(screen.getByRole("button", { name: "Start" }));
  expect(onStart).toHaveBeenCalledWith(
    expect.objectContaining({
      useGroundTruthLocation: true,
      useGroundTruthSpeakerSources: true,
      sessionOverrides: { max_speakers_hint: 2 },
      livePlaybackEnabled: false,
      fastPath: expect.objectContaining({
        localizationBackend: "capon_mvdr_refine_1src",
        beamformingMode: "delay_sum",
        outputEnhancerMode: "wiener",
        postfilterEnabled: false,
        localizationHopMs: 50,
        localizationWindowMs: 200,
      }),
      slowPath: expect.objectContaining({
        enabled: true,
        longMemoryEnabled: true,
        trackingMode: "doa_centroid_v1",
      }),
    })
  );
});

test("live mode reveals ReSpeaker-specific settings and accepts backend JSON overrides", async () => {
  const user = userEvent.setup();
  const onStart = vi.fn();

  render(
    <SceneLauncher
      status="idle"
      hasActiveSession={false}
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
      monitorSource="processed"
      onMonitorSourceChange={() => undefined}
    />
  );

  await user.click(screen.getByRole("button", { name: "ReSpeaker Live Direct capture from the local USB microphone array." }));

  expect(screen.getByLabelText("Audio device query")).toBeInTheDocument();
  expect(screen.getByLabelText("Mic array profile")).toBeInTheDocument();
  expect(screen.getByLabelText("Sample rate (Hz)")).toBeInTheDocument();
  expect(screen.queryByLabelText("Use ground truth location")).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Advanced overrides" }));
  fireEvent.change(screen.getByLabelText("Fast path overrides JSON"), {
    target: { value: "{\"suppressed_user_voice_doa_deg\":225}" },
  });

  await user.click(screen.getByRole("button", { name: "Start" }));
  expect(onStart).toHaveBeenCalledWith(
    expect.objectContaining({
      fastPathOverrides: { suppressed_user_voice_doa_deg: 225 },
    })
  );
});
