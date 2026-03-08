import { useState } from "react";

import type { MonitorSource, ProcessingMode } from "../types/contracts";

export type InputSource = "simulation" | "respeaker_live";

export type SessionLaunchConfig = {
  inputSource: InputSource;
  scenePath: string;
  backgroundNoisePath: string;
  backgroundNoiseGain: number;
  audioDeviceQuery: string;
  monitorSource: MonitorSource;
  sampleRateHz: number;
  channelMap: string;
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
  onStopActiveSession: () => void;
  onDownloadWav: () => void;
  canDownloadWav: boolean;
  latencyMs: number;
  onLatencyMsChange: (latencyMs: number) => void;
  processingMode: ProcessingMode;
  onProcessingModeChange: (mode: ProcessingMode) => void;
  monitorSource: MonitorSource;
  onMonitorSourceChange: (source: MonitorSource) => void;
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
  onStopActiveSession,
  onDownloadWav,
  canDownloadWav,
  latencyMs,
  onLatencyMsChange,
  processingMode,
  onProcessingModeChange,
  monitorSource,
  onMonitorSourceChange,
}: Props) {
  const [inputSource, setInputSource] = useState<InputSource | null>(null);
  const [scenePath, setScenePath] = useState(defaultScenePath);
  const [backgroundNoisePath, setBackgroundNoisePath] = useState(defaultBackgroundNoisePath);
  const [backgroundNoiseGain, setBackgroundNoiseGain] = useState(defaultBackgroundNoiseGain);
  const [audioDeviceQuery, setAudioDeviceQuery] = useState("ReSpeaker");
  const [sampleRateHz, setSampleRateHz] = useState(48000);
  const [channelMap, setChannelMap] = useState("0,1,2,3");
  const isBusy = status === "running" || status === "starting";
  const showSimulationSettings = inputSource === "simulation";
  const showLiveSettings = inputSource === "respeaker_live";
  const canStart = inputSource !== null && !isBusy;

  function applyLatency(v: number): void {
    const clamped = Math.max(80, Math.min(2000, Math.round(v)));
    onLatencyMsChange(clamped);
  }

  return (
    <section className="panel">
      <h2>Scene Launcher</h2>
      <div className="mode-picker" role="group" aria-label="Input mode">
        <button
          type="button"
          className={`mode-card ${showSimulationSettings ? "selected" : ""}`}
          aria-pressed={showSimulationSettings}
          disabled={isBusy}
          onClick={() => setInputSource("simulation")}
        >
          <span className="mode-card-title">Simulation</span>
          <span className="mode-card-copy">Scene file plus optional background noise.</span>
        </button>
        <button
          type="button"
          className={`mode-card ${showLiveSettings ? "selected" : ""}`}
          aria-pressed={showLiveSettings}
          disabled={isBusy}
          onClick={() => setInputSource("respeaker_live")}
        >
          <span className="mode-card-title">ReSpeaker Live</span>
          <span className="mode-card-copy">Direct capture from the local USB microphone array.</span>
        </button>
      </div>

      <div className="actions actions-persistent">
        <button onClick={onKillRun} disabled={!canKillRun}>
          Kill Current Run
        </button>
        <button onClick={onStopActiveSession}>Stop Active Session</button>
      </div>

      {!inputSource ? (
        <p className="launcher-hint">Choose a mode to reveal the relevant session settings.</p>
      ) : (
        <div className="launcher-settings">
          {showSimulationSettings && (
            <>
              <label htmlFor="scene">Scene config path</label>
              <input
                id="scene"
                aria-label="Scene config path"
                value={scenePath}
                onChange={(e) => setScenePath(e.target.value)}
                placeholder="simulation/simulations/configs/library_scene/library_k1_scene00.json"
              />
              <label htmlFor="background-noise">Background noise audio path</label>
              <input
                id="background-noise"
                aria-label="Background noise audio path"
                value={backgroundNoisePath}
                onChange={(e) => setBackgroundNoisePath(e.target.value)}
                placeholder="wham_noise/tr/01dc0215_0.22439_01fc0207_-0.22439sp12.wav"
              />
              <label htmlFor="background-noise-gain">Background noise gain</label>
              <input
                id="background-noise-gain"
                aria-label="Background noise gain"
                type="number"
                min={0}
                max={2}
                step={0.05}
                value={backgroundNoiseGain}
                onChange={(e) => setBackgroundNoiseGain(Math.max(0, Math.min(2, Number(e.target.value))))}
              />
            </>
          )}

          {showLiveSettings && (
            <>
              <label htmlFor="audio-device-query">Audio device query</label>
              <input
                id="audio-device-query"
                aria-label="Audio device query"
                value={audioDeviceQuery}
                onChange={(e) => setAudioDeviceQuery(e.target.value)}
                placeholder="ReSpeaker"
              />
              <label htmlFor="channel-map">Channel map (optional)</label>
              <input
                id="channel-map"
                aria-label="Channel map (optional)"
                value={channelMap}
                onChange={(e) => setChannelMap(e.target.value)}
                placeholder="0,1,2,3"
              />
              <label htmlFor="sample-rate-hz">Sample rate (Hz)</label>
              <input
                id="sample-rate-hz"
                aria-label="Sample rate (Hz)"
                type="number"
                min={8000}
                max={96000}
                step={1000}
                value={sampleRateHz}
                onChange={(e) => setSampleRateHz(Math.max(8000, Math.min(96000, Number(e.target.value))))}
              />
            </>
          )}

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
            disabled={isBusy}
            onChange={(e) => onProcessingModeChange(e.target.value as ProcessingMode)}
          >
            <option value="specific_speaker_enhancement">Specific speaker enhancement</option>
            <option value="localize_and_beamform">Localize and beamform</option>
            <option value="beamform_from_ground_truth">Beamform from ground truth</option>
          </select>

          <label htmlFor="monitor-source">Monitor output</label>
          <select
            id="monitor-source"
            aria-label="Monitor output"
            value={monitorSource}
            onChange={(e) => onMonitorSourceChange(e.target.value as MonitorSource)}
          >
            <option value="processed">Processed (UI output)</option>
            <option value="raw_mixed">Raw mixed input</option>
          </select>

          <div className="actions">
            <button
              onClick={() =>
                inputSource &&
                onStart({
                  inputSource,
                  scenePath,
                  backgroundNoisePath,
                  backgroundNoiseGain,
                  audioDeviceQuery,
                  monitorSource,
                  sampleRateHz,
                  channelMap,
                })
              }
              disabled={!canStart}
            >
              Start
            </button>
            <button onClick={onStop} disabled={status !== "running" && status !== "starting"}>
              Stop
            </button>
            <button onClick={onDownloadWav} disabled={!canDownloadWav}>
              Download WAV
            </button>
          </div>
        </div>
      )}
      <p className="status">Status: {status}</p>
    </section>
  );
}
