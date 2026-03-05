import { useState } from "react";

import type { ProcessingMode } from "../types/contracts";

type Props = {
  status: string;
  defaultScenePath: string;
  onStart: (scenePath: string) => void;
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
  const [scenePath, setScenePath] = useState(defaultScenePath);

  function applyLatency(v: number): void {
    const clamped = Math.max(80, Math.min(2000, Math.round(v)));
    onLatencyMsChange(clamped);
  }

  return (
    <section className="panel">
      <h2>Scene Launcher</h2>
      <label htmlFor="scene">Scene config path</label>
      <input
        id="scene"
        value={scenePath}
        onChange={(e) => setScenePath(e.target.value)}
        placeholder="simulation/simulations/configs/library_scene/library_k1_scene00.json"
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
        <button onClick={() => onStart(scenePath)} disabled={status === "running" || status === "starting"}>
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
