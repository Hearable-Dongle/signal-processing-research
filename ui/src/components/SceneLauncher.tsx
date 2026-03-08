import { useState } from "react";

import type { ProcessingMode } from "../types/contracts";

export type InputSource = "simulation" | "respeaker_live";

export type SessionLaunchConfig = {
  inputSource: InputSource;
  scenePath: string;
  backgroundNoisePath: string;
  backgroundNoiseGain: number;
  audioDeviceQuery: string;
};

type Props = {
  status: string;
  defaultScenePath: string;
  defaultBackgroundNoisePath: string;
  defaultBackgroundNoiseGain: number;
  onStart: (config: SessionLaunchConfig) => void;
  onStop: () => void;
  onKillRun: () => void;
  canKillRun: boolean;
  onDownloadWav: () => void;
  canDownloadWav: boolean;
  latencyMs: number;
  onLatencyMsChange: (latencyMs: number) => void;
  processingMode: ProcessingMode;
  onProcessingModeChange: (mode: ProcessingMode) => void;
};

export function SceneLauncher({
  status,
  defaultScenePath,
  defaultBackgroundNoisePath,
  defaultBackgroundNoiseGain,
  onStart,
  onStop,
  onKillRun,
  canKillRun,
  onDownloadWav,
  canDownloadWav,
  latencyMs,
  onLatencyMsChange,
  processingMode,
  onProcessingModeChange,
}: Props) {
  const [inputSource, setInputSource] = useState<InputSource>("simulation");
  const [scenePath, setScenePath] = useState(defaultScenePath);
  const [backgroundNoisePath, setBackgroundNoisePath] = useState(defaultBackgroundNoisePath);
  const [backgroundNoiseGain, setBackgroundNoiseGain] = useState(defaultBackgroundNoiseGain);
  const [audioDeviceQuery, setAudioDeviceQuery] = useState("ReSpeaker");

  function applyLatency(v: number): void {
    const clamped = Math.max(80, Math.min(2000, Math.round(v)));
    onLatencyMsChange(clamped);
  }

  return (
    <section className="panel">
      <h2>Scene Launcher</h2>
      <label htmlFor="input-source">Input source</label>
      <select
        id="input-source"
        value={inputSource}
        disabled={status === "running" || status === "starting"}
        onChange={(e) => setInputSource(e.target.value as InputSource)}
      >
        <option value="simulation">Simulation</option>
        <option value="respeaker_live">ReSpeaker live</option>
      </select>
      <label htmlFor="scene">Scene config path</label>
      <input
        id="scene"
        value={scenePath}
        onChange={(e) => setScenePath(e.target.value)}
        disabled={inputSource !== "simulation"}
        placeholder="simulation/simulations/configs/library_scene/library_k1_scene00.json"
      />
      <label htmlFor="audio-device-query">Audio device query</label>
      <input
        id="audio-device-query"
        value={audioDeviceQuery}
        onChange={(e) => setAudioDeviceQuery(e.target.value)}
        disabled={inputSource !== "respeaker_live"}
        placeholder="ReSpeaker"
      />
      <label htmlFor="background-noise">Background noise audio path</label>
      <input
        id="background-noise"
        value={backgroundNoisePath}
        onChange={(e) => setBackgroundNoisePath(e.target.value)}
        disabled={inputSource !== "simulation"}
        placeholder="wham_noise/tr/01dc0215_0.22439_01fc0207_-0.22439sp12.wav"
      />
      <label htmlFor="background-noise-gain">Background noise gain</label>
      <input
        id="background-noise-gain"
        type="number"
        min={0}
        max={2}
        step={0.05}
        value={backgroundNoiseGain}
        onChange={(e) => setBackgroundNoiseGain(Math.max(0, Math.min(2, Number(e.target.value))))}
        disabled={inputSource !== "simulation"}
      />

      <label htmlFor="latency-range">Playback latency (ms)</label>
      <input
        id="latency-range"
        type="range"
        min={80}
        max={2000}
        step={10}
        value={latencyMs}
        onChange={(e) => applyLatency(Number(e.target.value))}
      />
      <input
        aria-label="Playback latency number"
        type="number"
        min={80}
        max={2000}
        value={latencyMs}
        onChange={(e) => applyLatency(Number(e.target.value))}
      />

      <label htmlFor="processing-mode">Beamforming mode</label>
      <select
        id="processing-mode"
        aria-label="Beamforming mode"
        value={processingMode}
        disabled={status === "running" || status === "starting"}
        onChange={(e) => onProcessingModeChange(e.target.value as ProcessingMode)}
      >
        <option value="specific_speaker_enhancement">Specific speaker enhancement</option>
        <option value="localize_and_beamform">Localize and beamform</option>
        <option value="beamform_from_ground_truth">Beamform from ground truth</option>
      </select>

      <div className="actions">
        <button
          onClick={() =>
            onStart({
              inputSource,
              scenePath,
              backgroundNoisePath,
              backgroundNoiseGain,
              audioDeviceQuery,
            })
          }
          disabled={status === "running" || status === "starting"}
        >
          Start
        </button>
        <button onClick={onStop} disabled={status !== "running" && status !== "starting"}>
          Stop
        </button>
        <button onClick={onKillRun} disabled={!canKillRun}>
          Kill Current Run
        </button>
        <button onClick={onDownloadWav} disabled={!canDownloadWav}>
          Download WAV
        </button>
      </div>
      <p className="status">Status: {status}</p>
    </section>
  );
}
