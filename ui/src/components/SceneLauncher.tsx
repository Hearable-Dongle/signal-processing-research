import { useState } from "react";

type Props = {
  status: string;
  defaultScenePath: string;
  onStart: (scenePath: string) => void;
  onStop: () => void;
  onDownloadWav: () => void;
  canDownloadWav: boolean;
};

export function SceneLauncher({ status, defaultScenePath, onStart, onStop, onDownloadWav, canDownloadWav }: Props) {
  const [scenePath, setScenePath] = useState(defaultScenePath);

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
      <div className="actions">
        <button onClick={() => onStart(scenePath)} disabled={status === "running" || status === "starting"}>
          Start
        </button>
        <button onClick={onStop} disabled={status !== "running" && status !== "starting"}>
          Stop
        </button>
        <button onClick={onDownloadWav} disabled={!canDownloadWav}>
          Download WAV
        </button>
      </div>
      <p className="status">Status: {status}</p>
    </section>
  );
}
