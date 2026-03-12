import { useState } from "react";

import type { AlgorithmMode, MonitorSource } from "../types/contracts";

export type InputSource = "simulation" | "respeaker_live";

export type SessionLaunchConfig = {
  inputSource: InputSource;
  algorithmMode: AlgorithmMode;
  localizationHopMs: number;
  localizationWindowMs: number;
  localizationOverlap: number;
  localizationFreqLowHz: number;
  localizationFreqHighHz: number;
  speakerHistorySize: number;
  speakerActivationMinPredictions: number;
  speakerMatchWindowDeg: number;
  scenePath: string;
  backgroundNoisePath: string;
  backgroundNoiseGain: number;
  useGroundTruthLocation: boolean;
  useGroundTruthSpeakerSources: boolean;
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
  monitorSource: MonitorSource;
  onMonitorSourceChange: (source: MonitorSource) => void;
};

const ALGORITHM_OPTIONS: Array<{ value: AlgorithmMode; label: string }> = [
  { value: "localization_only", label: "Localization only" },
  { value: "spatial_baseline", label: "Spatial baseline" },
  { value: "speaker_tracking", label: "Speaker tracking" },
  { value: "speaker_tracking_long_memory", label: "Speaker tracking + long memory" },
  { value: "speaker_tracking_single_active", label: "Speaker tracking, single active" },
  { value: "single_dominant_no_separator", label: "Single dominant no-separator" },
];

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
  monitorSource,
  onMonitorSourceChange,
}: Props) {
  const [inputSource, setInputSource] = useState<InputSource | null>(null);
  const [algorithmMode, setAlgorithmMode] = useState<AlgorithmMode>("single_dominant_no_separator");
  const [localizationHopMs, setLocalizationHopMs] = useState(95);
  const [localizationWindowMs, setLocalizationWindowMs] = useState(300);
  const [localizationOverlap, setLocalizationOverlap] = useState(0.9);
  const [localizationFreqLowHz, setLocalizationFreqLowHz] = useState(1200);
  const [localizationFreqHighHz, setLocalizationFreqHighHz] = useState(5400);
  const [speakerHistorySize, setSpeakerHistorySize] = useState(8);
  const [speakerActivationMinPredictions, setSpeakerActivationMinPredictions] = useState(3);
  const [speakerMatchWindowDeg, setSpeakerMatchWindowDeg] = useState(30);
  const [scenePath, setScenePath] = useState(defaultScenePath);
  const [backgroundNoisePath, setBackgroundNoisePath] = useState(defaultBackgroundNoisePath);
  const [backgroundNoiseGain, setBackgroundNoiseGain] = useState(defaultBackgroundNoiseGain);
  const [useGroundTruthLocation, setUseGroundTruthLocation] = useState(false);
  const [useGroundTruthSpeakerSources, setUseGroundTruthSpeakerSources] = useState(false);
  const [audioDeviceQuery, setAudioDeviceQuery] = useState("ReSpeaker");
  const [sampleRateHz, setSampleRateHz] = useState(48000);
  const [channelMap, setChannelMap] = useState("0,1,2,3");
  const isBusy = status === "running" || status === "starting";
  const showSimulationSettings = inputSource === "simulation";
  const showLiveSettings = inputSource === "respeaker_live";
  const canStart = inputSource !== null && !isBusy;
  const supportsGroundTruthSpeakerSources =
    algorithmMode !== "localization_only" && algorithmMode !== "single_dominant_no_separator";

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
          <label htmlFor="algorithm-mode">Algorithm mode</label>
          <select
            id="algorithm-mode"
            aria-label="Algorithm mode"
            value={algorithmMode}
            disabled={isBusy}
            onChange={(e) => setAlgorithmMode(e.target.value as AlgorithmMode)}
          >
            {ALGORITHM_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>

          <label htmlFor="localization-hop-ms">Localization hop (ms)</label>
          <input
            id="localization-hop-ms"
            aria-label="Localization hop (ms)"
            type="number"
            min={10}
            max={500}
            step={10}
            value={localizationHopMs}
            disabled={isBusy}
            onChange={(e) => {
              const nextHop = Math.max(10, Math.min(500, Number(e.target.value) || 10));
              setLocalizationHopMs(nextHop);
              setLocalizationWindowMs((prev) => Math.max(prev, nextHop));
            }}
          />

          <label htmlFor="localization-window-ms">Localization window (ms)</label>
          <input
            id="localization-window-ms"
            aria-label="Localization window (ms)"
            type="number"
            min={20}
            max={2000}
            step={10}
            value={localizationWindowMs}
            disabled={isBusy}
            onChange={(e) => {
              const nextWindow = Math.min(2000, Number(e.target.value) || 160);
              setLocalizationWindowMs(Math.max(Math.max(20, localizationHopMs), nextWindow));
            }}
          />

          <label htmlFor="localization-overlap">Localization overlap</label>
          <input
            id="localization-overlap"
            aria-label="Localization overlap"
            type="number"
            min={0}
            max={0.99}
            step={0.01}
            value={localizationOverlap}
            disabled={isBusy}
            onChange={(e) => {
              const nextOverlap = Number(e.target.value);
              setLocalizationOverlap(Math.max(0, Math.min(0.99, Number.isFinite(nextOverlap) ? nextOverlap : 0)));
            }}
          />

          <label htmlFor="localization-freq-low-hz">Localization freq low (Hz)</label>
          <input
            id="localization-freq-low-hz"
            aria-label="Localization freq low (Hz)"
            type="number"
            min={0}
            max={24000}
            step={50}
            value={localizationFreqLowHz}
            disabled={isBusy}
            onChange={(e) => {
              const nextLow = Math.max(0, Math.min(24000, Number(e.target.value) || 0));
              setLocalizationFreqLowHz(nextLow);
              setLocalizationFreqHighHz((prev) => Math.max(prev, nextLow));
            }}
          />

          <label htmlFor="localization-freq-high-hz">Localization freq high (Hz)</label>
          <input
            id="localization-freq-high-hz"
            aria-label="Localization freq high (Hz)"
            type="number"
            min={0}
            max={24000}
            step={50}
            value={localizationFreqHighHz}
            disabled={isBusy}
            onChange={(e) => {
              const nextHigh = Math.max(0, Math.min(24000, Number(e.target.value) || 0));
              setLocalizationFreqHighHz(Math.max(nextHigh, localizationFreqLowHz));
            }}
          />

          <label htmlFor="speaker-history-size">Speaker centroid history (M)</label>
          <input
            id="speaker-history-size"
            aria-label="Speaker centroid history (M)"
            type="number"
            min={1}
            max={64}
            step={1}
            value={speakerHistorySize}
            disabled={isBusy}
            onChange={(e) => {
              const nextHistory = Math.max(1, Math.min(64, Number(e.target.value) || 1));
              setSpeakerHistorySize(nextHistory);
              setSpeakerActivationMinPredictions((prev) => Math.min(prev, nextHistory));
            }}
          />

          <label htmlFor="speaker-activation-min-predictions">Speaker activation observations (N)</label>
          <input
            id="speaker-activation-min-predictions"
            aria-label="Speaker activation observations (N)"
            type="number"
            min={1}
            max={64}
            step={1}
            value={speakerActivationMinPredictions}
            disabled={isBusy}
            onChange={(e) => {
              const nextCount = Math.max(1, Math.min(64, Number(e.target.value) || 1));
              setSpeakerActivationMinPredictions(Math.min(nextCount, speakerHistorySize));
            }}
          />

          <label htmlFor="speaker-match-window-deg">Speaker match window (deg)</label>
          <input
            id="speaker-match-window-deg"
            aria-label="Speaker match window (deg)"
            type="number"
            min={1}
            max={180}
            step={1}
            value={speakerMatchWindowDeg}
            disabled={isBusy}
            onChange={(e) => setSpeakerMatchWindowDeg(Math.max(1, Math.min(180, Number(e.target.value) || 1)))}
          />

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
              <label className="checkbox-row" htmlFor="gt-location">
                <input
                  id="gt-location"
                  aria-label="Use ground truth location"
                  type="checkbox"
                  checked={useGroundTruthLocation}
                  disabled={isBusy}
                  onChange={(e) => setUseGroundTruthLocation(e.target.checked)}
                />
                <span>Location: use ground truth</span>
              </label>
              <label className="checkbox-row" htmlFor="gt-speaker-sources">
                <input
                  id="gt-speaker-sources"
                  aria-label="Use ground truth speaker sources"
                  type="checkbox"
                  checked={useGroundTruthSpeakerSources}
                  disabled={isBusy || !supportsGroundTruthSpeakerSources}
                  onChange={(e) => setUseGroundTruthSpeakerSources(e.target.checked)}
                />
                <span>Speaker sources: use ground truth</span>
              </label>
              {!supportsGroundTruthSpeakerSources && (
                <p className="launcher-hint">This algorithm does not use separate speaker-source streams.</p>
              )}
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
                  algorithmMode,
                  localizationHopMs,
                  localizationWindowMs,
                  localizationOverlap,
                  localizationFreqLowHz,
                  localizationFreqHighHz,
                  speakerHistorySize,
                  speakerActivationMinPredictions,
                  speakerMatchWindowDeg,
                  scenePath,
                  backgroundNoisePath,
                  backgroundNoiseGain,
                  useGroundTruthLocation: showSimulationSettings ? useGroundTruthLocation : false,
                  useGroundTruthSpeakerSources:
                    showSimulationSettings && supportsGroundTruthSpeakerSources ? useGroundTruthSpeakerSources : false,
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
