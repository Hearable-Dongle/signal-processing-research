import { useState } from "react";

type Props = {
  status: string;
  defaultScenePath: string;
  onStart: (scenePath: string) => void;
  onStop: () => void;
  onKillRun: () => void;
  canKillRun: boolean;
  onDownloadWav: () => void;
  canDownloadWav: boolean;
  onTogglePlayback: () => void;
  canPlayOutput: boolean;
  isPlaybackActive: boolean;
  latencyMs: number;
  onLatencyMsChange: (latencyMs: number) => void;
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
  onTogglePlayback,
  canPlayOutput,
  isPlaybackActive,
  latencyMs,
  onLatencyMsChange,
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
        <button onClick={onTogglePlayback} disabled={!canPlayOutput}>
          {isPlaybackActive ? "Stop Playback" : "Play Beamformed Output"}
        </button>
      </div>
      <p className="status">Status: {status}</p>
    </section>
  );
}
